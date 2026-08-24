"""Framing and dispatch for the FCI's TCP command channel.

One connected libfranka client at a time: read a length-prefixed
``MessageHeader`` plus payload off the socket, dispatch it to the handler for
its ``Command``, and write the reply back under the same command id.

``_send_tcp`` is the single writer for this socket, so replies cannot interleave
mid-message; :meth:`receive_exact` is the matching reader, returning None rather
than a short read so a half-delivered message is never parsed.
"""

import select
import socket
import struct
import threading
import time
from typing import Optional, Tuple

from franka_sim.franka_protocol import (
    Command,
    ConnectStatus,
    MessageHeader,
)
from franka_sim.server.constants import (
    logger,
)


class TcpSessionMixin:
    """See the module docstring; this mixin carries no state of its own."""

    def _send_tcp(self, client_socket, message: bytes) -> None:
        """Write one complete TCP message, never interleaved with another.

        Every response on this connection goes through here. The FCI's TCP
        stream is a sequence of ``MessageHeader`` + payload frames with no
        resynchronisation marker, so a single ``sendall`` that is interrupted
        halfway by another thread's ``sendall`` does not lose a message, it
        corrupts every message after it: the client reads twelve bytes of one
        frame's tail as the next frame's header.

        That became reachable when reflexes started aborting from more than one
        thread -- a communication-constraints violation is raised on the
        state-publish thread and a motion-limit violation on the UDP receive
        thread, while the TCP thread is answering commands.
        """
        with self._tcp_send_lock:
            client_socket.sendall(message)

    def receive_exact(self, sock: socket.socket, size: int) -> Optional[bytes]:
        """
        Receive exactly 'size' bytes from the socket.

        Args:
            sock: Socket to receive from
            size: Number of bytes to receive

        Returns:
            bytes: Received data, or None if connection closed
        """
        data = bytearray()
        remaining = size

        while remaining > 0:
            try:
                logger.debug(f"Waiting to receive {remaining} bytes...")
                chunk = sock.recv(remaining)
                if not chunk:
                    # Clean close by the client (end of session) -- not an error.
                    logger.debug("Connection closed by client while receiving data")
                    return None
                logger.debug(f"Received chunk of {len(chunk)} bytes")
                data.extend(chunk)
                remaining -= len(chunk)
            except socket.error as e:
                # Reset-by-peer etc. when the client goes away -- expected, not an error.
                logger.debug(f"Socket error while receiving (client disconnected): {e}")
                return None

        logger.debug(f"Successfully received all {size} bytes")
        return bytes(data)

    def receive_message(self, client_socket) -> Tuple[MessageHeader, Optional[bytes]]:
        """
        Receive a complete message following the libfranka protocol.

        Returns:
            Tuple of (MessageHeader, Optional[payload])
        """
        logger.debug("Attempting to receive message header (12 bytes)...")
        header_data = self.receive_exact(client_socket, 12)
        if not header_data:
            raise ConnectionError("Failed to receive message header")

        header = MessageHeader.from_bytes(header_data)
        logger.debug(
            f"Parsed header: command={Command(header.command).name}, "
            f"command_id={header.command_id}, size={header.size}"
        )

        payload_size = header.size - 12
        payload = None
        if payload_size > 0:
            logger.debug(f"Expecting payload of {payload_size} bytes")
            payload = self.receive_exact(client_socket, payload_size)
            if not payload:
                raise ConnectionError("Failed to receive message payload")
            logger.debug(f"Successfully received payload: {payload.hex()}")

        return header, payload

    def send_response(
        self, client_socket, command: int, command_id: int, status: ConnectStatus, version: int
    ):
        """
        Send a Connect response following the libfranka_new (v10) protocol.

        Connect::Response is ``ResponseBase::status`` (uint8) + ``version``
        (uint16) under ``#pragma pack(push, 1)``, i.e. 3 bytes with no padding.
        """
        # Total message size includes header (12 bytes) + response data (3 bytes)
        total_size = 12 + 3  # 3 = 1(status, uint8) + 2(version, uint16)

        # Construct and send header
        header = MessageHeader(command, command_id, total_size)
        header_bytes = header.to_bytes()

        # Construct response data (status: uint8, version: uint16)
        response_data = struct.pack("<BH", status.value, version)

        # Send complete message
        self._send_tcp(client_socket, header_bytes + response_data)
        logger.info(
            f"Sent response: command={Command(command).name}, "
            f"command_id={command_id}, status={status.name}"
        )

    def handle_get_robot_model(self, client_socket, header):
        """Handle GetRobotModel: return the robot URDF for client-side model building.

        In libfranka_new the client builds its own Pinocchio model from this
        URDF. The response payload (a DynamicSizedCommandMessage) is the status
        byte (uint8, 0 = success) followed by the URDF as UTF-8 bytes.
        """
        urdf_bytes = self.urdf_string.encode("utf-8")
        payload = struct.pack("<B", 0) + urdf_bytes  # status kSuccess + URDF

        response_header = MessageHeader(
            Command.kGetRobotModel, header.command_id, 12 + len(payload)
        )
        self._send_tcp(client_socket, response_header.to_bytes() + payload)
        logger.info(f"Sent GetRobotModel response ({len(urdf_bytes)} URDF bytes)")

    def handle_tcp_messages(self, client_socket):
        """Handle TCP messages in a separate thread"""
        logger.info("TCP message handler thread started")
        while self.running:  # Keep the TCP thread running even after client disconnects
            try:
                # Check if socket is still connected
                try:
                    client_socket.getpeername()
                except socket.error as e:
                    logger.info(f"Client socket disconnected: {e}")
                    # Instead of breaking, reset state and continue
                    self.transmitting_state = False
                    self.connection_running = False
                    # connection_running is cleared *first* so the UDP command
                    # thread stops dispatching before the arm is recaptured.
                    self._engage_idle_hold("client socket disconnected")
                    logger.info("Resetting state and waiting for new client...")
                    break  # Break only from the inner loop

                # Try to peek at incoming data
                readable, _, _ = select.select([client_socket], [], [], 0.1)
                if not readable:
                    continue

                logger.debug("Data available on socket, attempting to receive...")
                header, payload = self.receive_message(client_socket)
                logger.info(
                    f"Processing command: {Command(header.command).name} (ID: {header.command_id})"
                )

                if header.command == Command.kMove:
                    logger.debug(f"Move command payload size: {len(payload)} bytes")
                    logger.debug(f"Move command payload hex: {payload.hex()}")
                    self.handle_move_command(client_socket, header, payload)
                elif header.command == Command.kStopMove:
                    logger.info("Handling StopMove command")
                    self.handle_stop_move_command(client_socket, header)
                elif header.command == Command.kSetCollisionBehavior:
                    logger.info("Handling SetCollisionBehavior command")
                    self.handle_set_collision_behavior_command(client_socket, header, payload)
                elif header.command == Command.kSetJointImpedance:
                    logger.info("Handling SetJointImpedance command")
                    self.handle_set_joint_impedance_command(client_socket, header, payload)
                elif header.command == Command.kSetCartesianImpedance:
                    logger.info("Handling SetCartesianImpedance command")
                    self.handle_set_cartesian_impedance_command(client_socket, header, payload)
                elif header.command == Command.kSetGuidingMode:
                    logger.info("Handling SetGuidingMode command")
                    self.handle_set_guiding_mode_command(client_socket, header, payload)
                elif header.command == Command.kSetEEToK:
                    logger.info("Handling SetEEToK command")
                    self.handle_set_ee_to_k_command(client_socket, header, payload)
                elif header.command == Command.kSetNEToEE:
                    logger.info("Handling SetNEToEE command")
                    self.handle_set_ne_to_ee_command(client_socket, header, payload)
                elif header.command == Command.kSetLoad:
                    logger.info("Handling SetLoad command")
                    self.handle_set_load_command(client_socket, header, payload)
                elif header.command == Command.kGetRobotModel:
                    logger.info("Handling GetRobotModel command")
                    self.handle_get_robot_model(client_socket, header)
                elif header.command == Command.kAutomaticErrorRecovery:
                    logger.info("Handling AutomaticErrorRecovery command")
                    self.handle_automatic_error_recovery_command(client_socket, header, payload)
                else:
                    logger.warning(
                        f"Unhandled command in TCP thread: {Command(header.command).name}"
                    )
            except ConnectionError as e:
                logger.info(f"Client disconnected (end of session): {e}")
                # Instead of breaking, reset state and continue
                self.transmitting_state = False
                self.connection_running = False
                self._engage_idle_hold("client disconnected mid-session")
                logger.info("Connection error: Resetting state and waiting for new client...")
                break  # Break only from the inner loop
            except Exception as e:
                logger.error(f"Error in TCP thread: {e}", exc_info=True)
                if not self.running:  # Only break if server is shutting down
                    break
                # For other errors, reset state and continue
                self.transmitting_state = False
                self.connection_running = False
                self._engage_idle_hold("TCP error mid-session")
                logger.info("Error occurred: Resetting state and waiting for new client...")
                break  # Break only from the inner loop

        logger.info("TCP message handler thread ending")

    def handle_client(self, client_socket):
        """
        Handle initial client connection and start message handlers
        """
        try:
            # Reset state for new connection
            self.reset_state()

            self.client_socket = client_socket
            self.connection_running = True
            logger.info("Waiting for initial connect command...")

            # Handle initial connect message
            header, payload = self.receive_message(client_socket)

            if header.command != Command.kConnect:
                logger.error(f"Expected connect command, got {Command(header.command).name}")
                return

            if not payload or len(payload) < 4:
                logger.error("Invalid connect payload: Version or UDP port not found")
                return

            # Log the full payload for debugging
            logger.info(f"Connect payload hex: {payload.hex()}")

            # The payload structure is:
            # - uint16_t version
            # - uint16_t udp_port (from network.udpPort())
            version, network_udp_port = struct.unpack("<HH", payload[:4])
            logger.info(f"Received version: {version}, network UDP port: {network_udp_port}")
            # Send successful connect response
            self.send_response(
                client_socket,
                command=header.command,
                command_id=header.command_id,
                status=ConnectStatus.kSuccess,
                version=self.library_version,
            )
            logger.info("Sent connect response")

            # Start TCP message handler thread
            self.tcp_thread = threading.Thread(
                target=self.handle_tcp_messages, args=(client_socket,)
            )
            self.tcp_thread.daemon = True
            self.tcp_thread.start()
            logger.info("Started TCP message handler thread")

            # Start UDP state transmission
            client_address = client_socket.getpeername()[0]
            logger.info(f"Starting UDP transmission to {client_address}:{network_udp_port}")
            self.start_robot_state_transmission(client_address, network_udp_port)

            # Keep the connection thread alive
            while self.connection_running and self.running:
                time.sleep(0.1)

            # Wait for TCP thread to finish
            if self.tcp_thread and self.tcp_thread.is_alive():
                self.tcp_thread.join(timeout=1.0)

        except Exception as e:
            logger.error(f"Error handling client: {e}", exc_info=True)
        finally:
            logger.info("Closing client connection")
            # Catch-all for every way this connection can end that the TCP
            # thread does not see: the UDP command socket erroring out, the
            # handshake aborting, an exception above. Idempotent, so the common
            # case (the TCP thread already held) costs nothing. Must run before
            # reset_state(), which clears the control mode the hold keys off.
            self._engage_idle_hold("client connection closed")
            if client_socket:
                client_socket.close()
            # Clean up connection state
            self.reset_state()

            # Make sure UDP socket is closed
            if self.udp_socket:
                try:
                    self.udp_socket.close()
                except Exception as e:
                    logger.error(f"Error closing UDP socket: {e}")
                self.udp_socket = None
