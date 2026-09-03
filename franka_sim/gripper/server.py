#!/usr/bin/env python3
"""TCP/UDP server implementing the libfranka gripper wire protocol (port 1338)."""
import errno
import logging
import select
import socket
import threading
import time
from typing import Optional

from franka_sim.gripper.backend import FrankaHandSim, GripperBackend
from franka_sim.gripper.protocol import (
    GRIPPER_COMMAND_PORT,
    GRIPPER_HEADER_SIZE,
    GRIPPER_VERSION,
    ConnectRequest,
    GraspRequest,
    GripperCommand,
    GripperCommandHeader,
    GripperConnectStatus,
    GripperState,
    GripperStatus,
    MoveRequest,
    build_command_response,
    build_connect_response,
)

logger = logging.getLogger(__name__)

# Rate at which GripperState is broadcast over UDP. readOnce() only needs
# reasonably fresh state, so this is far below the arm's 1 kHz.
_BROADCAST_HZ = 60.0


class FrankaGripperServer:
    """TCP/UDP server implementing the libfranka gripper protocol (port 1338).

    Wire-compatible with the hardcoded ``franka::Gripper`` client: a TCP command
    channel (Connect/Homing/Grasp/Move/Stop, one response each) plus a one-way
    UDP ``GripperState`` broadcast. All physical behaviour is delegated to a
    swappable ``GripperBackend``.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = GRIPPER_COMMAND_PORT,
        backend: Optional[GripperBackend] = None,
    ):
        self.host = host
        self.port = port
        self.backend = backend if backend is not None else FrankaHandSim()
        self.server_socket = None
        self.running = False
        self.connection_running = False
        self.udp_socket = None
        self.client_address = None
        self.client_udp_port = None
        self.broadcast_thread = None

    # -- low-level receive --------------------------------------------------
    def receive_exact(self, sock: socket.socket, size: int) -> Optional[bytes]:
        """Block until exactly ``size`` bytes are read from ``sock``, or return None.

        Returns None on a socket error or on a clean close/EOF before
        ``size`` bytes have been received.
        """
        data = bytearray()
        while len(data) < size:
            try:
                chunk = sock.recv(size - len(data))
            except socket.error:
                return None
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def receive_message(self, sock: socket.socket):
        """Read one framed gripper command message: a header plus its payload.

        Returns ``(header, payload)``, or ``(None, None)`` if the header or
        payload could not be fully read (see ``receive_exact``).
        """
        header_data = self.receive_exact(sock, GRIPPER_HEADER_SIZE)
        if header_data is None:
            return None, None
        header = GripperCommandHeader.from_bytes(header_data)
        payload = b""
        payload_size = header.size - GRIPPER_HEADER_SIZE
        if payload_size > 0:
            payload = self.receive_exact(sock, payload_size)
            if payload is None:
                return None, None
        return header, payload

    # -- UDP broadcast ------------------------------------------------------
    def _broadcast_state(self):
        period = 1.0 / _BROADCAST_HZ
        next_deadline = time.perf_counter()
        message_id = 0
        while self.running and self.connection_running:
            state = self.backend.get_state()
            message_id += 1
            packet = GripperState(
                state.width, state.max_width, state.is_grasped, state.temperature
            ).pack(message_id)
            # Snapshot the connection's target together (handler thread writes
            # these on Connect / clears them on disconnect). Reads are atomic
            # under the GIL; the snapshot avoids sending to a half-updated target.
            udp = self.udp_socket
            addr = self.client_address
            port = self.client_udp_port
            if udp is not None and addr is not None:
                try:
                    udp.sendto(packet, (addr, port))
                except socket.error as exc:
                    logger.debug(f"Gripper UDP send failed: {exc}")
            next_deadline += period
            remaining = next_deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_deadline = time.perf_counter()

    # -- command dispatch ---------------------------------------------------
    def _handle_connect(self, sock: socket.socket, header: GripperCommandHeader, payload: bytes):
        req = ConnectRequest.from_bytes(payload)
        self.client_address = sock.getpeername()[0]
        self.client_udp_port = req.udp_port
        status = (
            GripperConnectStatus.kSuccess
            if req.version == GRIPPER_VERSION
            else GripperConnectStatus.kIncompatibleLibraryVersion
        )
        sock.sendall(build_connect_response(header.command_id, status, GRIPPER_VERSION))
        logger.info(f"Gripper Connect from {self.client_address} udp_port={self.client_udp_port}")
        if status == GripperConnectStatus.kSuccess and self.broadcast_thread is None:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.broadcast_thread = threading.Thread(target=self._broadcast_state, daemon=True)
            self.broadcast_thread.start()

    def _dispatch_command(self, sock, header: GripperCommandHeader, payload: bytes):
        """Run one gripper command on the backend and answer it on the wire.

        The status a command that did not succeed gets is not free-form:
        ``franka::Gripper``'s ``executeCommand`` (libfranka ``src/gripper.cpp``)
        maps kSuccess -> True, kUnsuccessful -> False, and both kFail and
        kAborted -> a thrown ``CommandException``. A grasp that ends outside
        the epsilon band is a *failed* grasp on the real hand -- the client is
        supposed to see ``CommandException("libfranka gripper: Command
        failed!")``, not a quiet False -- so it answers kFail.
        """
        cmd = header.command
        # What this command answers when the backend reports no success.
        failure_status = GripperStatus.kUnsuccessful
        try:
            if cmd == GripperCommand.kHoming:
                ok = self.backend.homing()
            elif cmd == GripperCommand.kMove:
                req = MoveRequest.from_bytes(payload)
                ok = self.backend.move(req.width, req.speed)
            elif cmd == GripperCommand.kGrasp:
                failure_status = GripperStatus.kFail
                req = GraspRequest.from_bytes(payload)
                ok = self.backend.grasp(
                    req.width, req.epsilon_inner, req.epsilon_outer, req.speed, req.force
                )
            elif cmd == GripperCommand.kStop:
                ok = self.backend.stop()
            else:
                logger.warning(f"Unknown gripper command: {cmd}")
                sock.sendall(build_command_response(cmd, header.command_id, GripperStatus.kFail))
                return
            status = GripperStatus.kSuccess if ok else failure_status
        except Exception as exc:
            logger.error(f"Gripper command {cmd} failed: {exc}")
            status = GripperStatus.kFail
        sock.sendall(build_command_response(cmd, header.command_id, status))

    # -- connection handling ------------------------------------------------
    def handle_client(self, sock: socket.socket):
        """Serve one client connection: dispatch commands until it disconnects.

        Polls ``sock`` for readability, reads one framed message at a time,
        routes ``kConnect`` to ``_handle_connect`` and everything else to
        ``_dispatch_command``, and on exit tears down the UDP broadcast
        thread/socket and closes ``sock``.
        """
        self.connection_running = True
        try:
            while self.running and self.connection_running:
                readable, _, _ = select.select([sock], [], [], 0.5)
                if not readable:
                    continue
                header, payload = self.receive_message(sock)
                if header is None:
                    break
                if header.command == GripperCommand.kConnect:
                    self._handle_connect(sock, header, payload)
                else:
                    self._dispatch_command(sock, header, payload)
        except Exception as exc:
            logger.error(f"Gripper connection error: {exc}")
        finally:
            self.connection_running = False
            if self.broadcast_thread is not None:
                self.broadcast_thread.join(timeout=1.0)
                self.broadcast_thread = None
            if self.udp_socket is not None:
                try:
                    self.udp_socket.close()
                except socket.error:
                    pass
                self.udp_socket = None
            self.client_address = None
            self.client_udp_port = None
            try:
                sock.close()
            except socket.error:
                pass

    def run_server(self):
        """Bind and listen on ``host``:``port``, accepting one client at a time.

        Blocks until ``stop()`` closes the listening socket; each accepted
        connection is served synchronously via ``handle_client`` before the
        next ``accept()``.
        """
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # No SO_REUSEPORT -- same reasoning as the arm server's
        # _bind_listener(): a silent co-bind on the gripper port would
        # load-balance clients between two servers with no error anywhere.
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.settimeout(1.0)
        try:
            self.server_socket.bind((self.host, self.port))
        except OSError as e:
            self.server_socket.close()
            if e.errno == errno.EADDRINUSE:
                logger.error(
                    "Gripper port %s:%s is already in use -- refusing to co-bind.",
                    self.host,
                    self.port,
                )
            raise
        self.server_socket.listen(1)
        self.running = True
        logger.info(f"Gripper server listening on {self.host}:{self.port}")
        while self.running:
            try:
                client_socket, _ = self.server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.handle_client(client_socket)

    def stop(self):
        """Signal the server loop to exit and close the listening socket."""
        logger.info("Stopping gripper server")
        self.running = False
        self.connection_running = False
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except socket.error:
                pass
            self.server_socket = None
