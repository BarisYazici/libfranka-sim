#!/usr/bin/env python3

import socket
import struct
import logging
import enum
import time
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import select
import argparse

from franka_sim.franka_protocol import (
    Command, ConnectStatus, MoveStatus, MessageHeader, 
    COMMAND_PORT, MoveCommand, ControllerMode, MotionGeneratorMode
)
from franka_sim.robot_state import RobotState
from franka_sim.franka_genesis_sim import FrankaGenesisSim

# Configure detailed logging for debugging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RobotMode(enum.IntEnum):
    """Operating modes of the Franka robot"""
    kOther = 0
    kIdle = 1
    kMove = 2
    kGuiding = 3
    kReflex = 4
    kUserStopped = 5
    kAutomaticErrorRecovery = 6

@dataclass
class MessageHeader:
    """
    Represents the message header structure from libfranka.
    All messages begin with this 12-byte header.
    """
    command: int      # Command type (uint32)
    command_id: int   # Unique command identifier (uint32)
    size: int        # Total message size including header (uint32)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'MessageHeader':
        """Parse header from binary data using little-endian format"""
        command, command_id, size = struct.unpack('<III', data)
        return cls(command, command_id, size)

    def to_bytes(self) -> bytes:
        """Convert header to binary format using little-endian"""
        return struct.pack('<III', self.command, self.command_id, self.size)

class FrankaSimServer:
    """
    A simulation server implementing the Franka robot control interface protocol.
    Handles both TCP command communication and UDP state updates.
    """
    
    def __init__(self, host='0.0.0.0', port=COMMAND_PORT, enable_vis=False):
        """
        Initialize the Franka simulation server.
        
        Args:
            host: IP address to bind to (default: all interfaces)
            port: TCP port for command interface
            enable_vis: Enable visualization of the Genesis simulator
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.transmitting_state = False
        self.library_version = 9  # Current libfranka version
        self.command_socket = None  # UDP socket for receiving commands
        self.current_motion_id = 0
        self.client_socket = None
        self.tcp_thread = None
        self.udp_socket = None
        self.client_address = None
        self.client_udp_port = None
        
        print("INITIALIZING SIMULATION`")
        # Initialize Genesis simulator
        self.genesis_sim = FrankaGenesisSim(enable_vis=enable_vis)
        print("SIMULATION INITIALIZED")
        self.robot_state = RobotState()

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
                    logger.error("Connection closed while receiving data")
                    return None
                logger.debug(f"Received chunk of {len(chunk)} bytes")
                data.extend(chunk)
                remaining -= len(chunk)
            except socket.error as e:
                logger.error(f"Socket error while receiving: {e}")
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
        logger.debug(f"Parsed header: command={Command(header.command).name}, "
                   f"command_id={header.command_id}, size={header.size}")

        payload_size = header.size - 12
        payload = None
        if payload_size > 0:
            logger.debug(f"Expecting payload of {payload_size} bytes")
            payload = self.receive_exact(client_socket, payload_size)
            if not payload:
                raise ConnectionError("Failed to receive message payload")
            logger.debug(f"Successfully received payload: {payload.hex()}")

        return header, payload

    def send_response(self, client_socket, command: int, command_id: int, 
                     status: ConnectStatus, version: int):
        """
        Send a response message following the libfranka protocol.
        """
        # Total message size includes header (12 bytes) + response data (status + version + padding)
        total_size = 12 + 8  # 8 = 2(status) + 2(version) + 4(padding)
        
        # Construct and send header
        header = MessageHeader(command, command_id, total_size)
        header_bytes = header.to_bytes()
        
        # Construct response data (status + version + 4 bytes padding)
        response_data = struct.pack('<HH4x', status.value, version)
        
        # Send complete message
        client_socket.sendall(header_bytes + response_data)
        logger.info(f"Sent response: command={Command(command).name}, "
                   f"command_id={command_id}, status={status.name}")

    def start_command_receiver(self, port: int):
        """Start UDP command receiver on specified port"""
        try:
            self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Set socket options for debugging
            self.command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.command_socket.bind((self.host, port))
            logger.info(f"Started UDP command receiver on {self.host}:{port}")
            
            # Set a timeout to check for packets periodically
            self.command_socket.settimeout(1.0)
            
            # Start command receiver thread
            self.command_thread = threading.Thread(target=self._handle_commands)
            self.command_thread.daemon = True
            self.command_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting command receiver: {e}", exc_info=True)

    def _handle_commands(self):
        """Handle incoming UDP robot commands"""
        logger.info("Command handler thread started")
        packet_count = 0
        last_log_time = time.time()
        
        while self.running:
            try:
                logger.debug("Waiting for UDP command...")
                try:
                    # Buffer size large enough for RobotCommand structure
                    data, addr = self.command_socket.recvfrom(1024)
                    packet_count += 1
                    
                    # Log packet statistics every 5 seconds
                    current_time = time.time()
                    if current_time - last_log_time >= 5.0:
                        logger.info(f"Received {packet_count} UDP packets in last 5 seconds")
                        packet_count = 0
                        last_log_time = current_time
                    
                    logger.debug(f"Received UDP data of length: {len(data)} from {addr}")
                    
                    if len(data) < 72:  # Minimum size for message_id (8) + motion (8) + 7 torques (56)
                        logger.warning(f"Received incomplete UDP data, length: {len(data)}")
                        continue
                    
                    # Unpack the message_id (uint64)
                    message_id = struct.unpack('<Q', data[0:8])[0]
                    
                    # The torque values start after the message_id and motion command
                    # Assuming 7 joints with double (8 bytes) values
                    torque_offset = 8 + 8  # message_id (8 bytes) + motion command offset
                    torque_values = struct.unpack('<7d', data[torque_offset:torque_offset + 56])
                    
                    # Update Genesis simulator with new torques
                    self.genesis_sim.update_torques(torque_values)
                    
                    logger.info(f"Received command from {addr}")
                    logger.info(f"Message ID: {message_id}")
                    logger.info("Received torque values:")
                    for i, torque in enumerate(torque_values):
                        logger.info(f"Joint {i + 1}: {torque}")
                    
                except socket.timeout:
                    # This is expected, just continue
                    continue
                    
            except Exception as e:
                logger.error(f"Error handling command: {e}", exc_info=True)

    def handle_move_command(self, client_socket, header: MessageHeader, payload: bytes) -> None:
        """Handle Move command received over TCP"""
        try:
            # Parse the move command
            move_cmd = MoveCommand.from_bytes(payload)
            logger.info(f"Received Move command: controller_mode={move_cmd.controller_mode.name}, "
                       f"motion_generator_mode={move_cmd.motion_generator_mode.name}")
            
            # Update robot state
            self.robot_state.state['motion_generator_mode'] = move_cmd.motion_generator_mode
            self.robot_state.state['controller_mode'] = move_cmd.controller_mode
            self.robot_state.state['robot_mode'] = RobotMode.kMove
            self.current_motion_id = header.command_id

            # First send motion started response
            self.send_move_response(
                client_socket,
                command_id=header.command_id,
                status=MoveStatus.kMotionStarted
            )
            logger.info(f"Motion started with ID: {self.current_motion_id}")

            # Then immediately send success response to break the waiting loop
            self.send_move_response(
                client_socket,
                command_id=header.command_id,
                status=MoveStatus.kSuccess
            )
            logger.info("Sent success response to break waiting loop")
            
        except Exception as e:
            logger.error(f"Error handling Move command: {e}")
            # Send error response
            self.send_move_response(
                client_socket,
                command_id=header.command_id,
                status=MoveStatus.kAborted
            )

    def send_move_response(self, client_socket, command_id: int, status: MoveStatus):
        """Send response to Move command"""
        try:
            # Total message size includes header (12 bytes) + response data (status + padding)
            total_size = 12 + 4  # 4 = 1(status) + 3(padding)
            
            # Construct and send header
            header = MessageHeader(Command.kMove, command_id, total_size)
            header_bytes = header.to_bytes()
            
            # Construct response data (status + 3 bytes padding)
            response_data = struct.pack('<B3x', status.value)
            
            # Send complete message
            message = header_bytes + response_data
            logger.debug(f"Sending Move response: {message.hex()}")
            client_socket.sendall(message)
            logger.info(f"Sent Move response: command_id={command_id}, status={status.name}")
        except Exception as e:
            logger.error(f"Error sending Move response: {e}")

    def handle_tcp_messages(self, client_socket):
        """Handle TCP messages in a separate thread"""
        logger.info("TCP message handler thread started")
        while self.running:
            try:
                logger.debug("Waiting for next TCP message...")
                # Check if socket is still connected
                try:
                    client_socket.getpeername()
                except socket.error as e:
                    logger.error("Socket disconnected")
                    break

                # Try to peek at incoming data
                readable, _, _ = select.select([client_socket], [], [], 0.1)
                if not readable:
                    continue

                logger.debug("Data available on socket, attempting to receive...")
                header, payload = self.receive_message(client_socket)
                logger.info(f"Processing command: {Command(header.command).name} (ID: {header.command_id})")
                
                if header.command == Command.kMove:
                    logger.debug(f"Move command payload size: {len(payload)} bytes")
                    logger.debug(f"Move command payload hex: {payload.hex()}")
                    self.handle_move_command(client_socket, header, payload)
                elif header.command == Command.kStopMove:
                    logger.info("Handling StopMove command")
                    self.handle_stop_move_command(client_socket, header)
                else:
                    logger.warning(f"Unhandled command in TCP thread: {Command(header.command).name}")
            except ConnectionError as e:
                logger.error(f"Connection error in TCP thread: {e}")
                break
            except Exception as e:
                logger.error(f"Error in TCP thread: {e}", exc_info=True)
                if not self.running:
                    break
        logger.info("TCP message handler thread ending")

    def handle_stop_move_command(self, client_socket, header: MessageHeader):
        """Handle StopMove command received over TCP"""
        try:
            logger.info("Processing StopMove command")
            
            # Send success response
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(Command.kStopMove, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            
            # Status 0 = Success
            response_data = struct.pack('<B3x', 0)  # 1 byte status + 3 bytes padding
            
            client_socket.sendall(header_bytes + response_data)
            logger.info("Sent StopMove success response")
            
            # Send one final state with both modes set to idle
            if hasattr(self, 'udp_socket') and self.udp_socket:
                # Update state to idle modes
                self.robot_state.state['motion_generator_mode'] = 0  # kNone
                self.robot_state.state['controller_mode'] = 3  # kOther
                self.robot_state.state['robot_mode'] = RobotMode.kIdle
                
                # Send state with new message ID
                self.robot_state.update()  # This increments message_id
                final_state = self.robot_state.pack_state()
                self.udp_socket.sendto(final_state, (self.client_address, self.client_udp_port))
                logger.info(f"Sent final robot state with message_id: {self.robot_state.state['message_id']}")
            
            # Stop robot state transmission
            self.transmitting_state = False
            logger.info("Stopped robot state transmission")
            
            # Send Move response to break the waiting loop in the client
            if self.current_motion_id:
                self.send_move_response(
                    client_socket,
                    command_id=self.current_motion_id,
                    status=MoveStatus.kSuccess
                )
                logger.info(f"Sent Move success response for motion ID: {self.current_motion_id}")
                self.current_motion_id = 0
                
        except Exception as e:
            logger.error(f"Error handling StopMove command: {e}")
            # Send error response
            total_size = 12 + 4
            response_header = MessageHeader(Command.kStopMove, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack('<B3x', 5)  # Status 5 = Aborted
            client_socket.sendall(header_bytes + response_data)

    def handle_client(self, client_socket):
        """
        Handle initial client connection and start message handlers
        """
        try:
            self.client_socket = client_socket
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
            version, network_udp_port = struct.unpack('<HH', payload[:4])
            logger.info(f"Received version: {version}, network UDP port: {network_udp_port}")

            # Start command receiver on port 1337
            command_port = 5001
            logger.info(f"Starting command receiver on fixed port {command_port}")
            self.start_command_receiver(command_port)
            logger.info(f"Command receiver started on port {command_port}")

            # Send successful connect response
            self.send_response(
                client_socket,
                command=header.command,
                command_id=header.command_id,
                status=ConnectStatus.kSuccess,
                version=self.library_version
            )
            logger.info("Sent connect response")

            # Start TCP message handler thread
            self.tcp_thread = threading.Thread(target=self.handle_tcp_messages, args=(client_socket,))
            self.tcp_thread.daemon = True
            self.tcp_thread.start()
            logger.info("Started TCP message handler thread")

            # Start UDP state transmission
            client_address = client_socket.getpeername()[0]
            logger.info(f"Starting UDP transmission to {client_address}:{network_udp_port}")
            self.start_robot_state_transmission(client_address, network_udp_port)

            # Keep the main thread alive
            while self.running:
                time.sleep(0.1)

        except Exception as e:
            logger.error(f"Error handling client: {e}", exc_info=True)
        finally:
            logger.info("Closing client connection")
            if client_socket:
                client_socket.close()

    def start_robot_state_transmission(self, client_address: str, client_udp_port: int):
        """
        Start UDP transmission of robot state updates.
        """
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client_address = client_address
        self.client_udp_port = client_udp_port
        
        try:
            logger.info(f"Starting UDP transmission to {client_address}:{client_udp_port}")
            self.transmitting_state = True
            first_state_sent = False
            
            while self.running and self.transmitting_state:
                # Get current state from Genesis simulator
                sim_state = self.genesis_sim.get_robot_state()
                self.robot_state.state.update(sim_state)
                
                # Pack and send current robot state
                state = self.robot_state.pack_state()
                self.udp_socket.sendto(state, (client_address, client_udp_port))
                
                # After first state is sent, send a Move success response
                if not first_state_sent and self.current_motion_id:
                    logger.info(f"Sending Move success response after first state for motion ID: {self.current_motion_id}")
                    self.send_move_response(
                        self.client_socket,
                        command_id=self.current_motion_id,
                        status=MoveStatus.kSuccess
                    )
                    first_state_sent = True
                
                # Update state for next iteration
                self.robot_state.update()
                time.sleep(0.001)  # 1kHz update rate
                
        except Exception as e:
            logger.error(f"UDP transmission error: {e}")
        finally:
            self.udp_socket.close()
            self.udp_socket = None

    def run_server(self):
        """Main server loop that runs in a separate thread when visualization is enabled"""
        try:
            # Start TCP server
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            self.server_socket.settimeout(1.0)
            
            try:
                self.server_socket.bind((self.host, self.port))
            except OSError as e:
                if e.errno == 48:  # Address already in use
                    logger.warning(f"Port is in use, attempting to force close and rebind... port and address: {self.port} and {self.host}")
                    self.server_socket.close()
                    time.sleep(1)
                    self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                    self.server_socket.bind((self.host, self.port))
                else:
                    raise
                    
            self.server_socket.listen(1)
            logger.info(f"Franka simulation server started on {self.host}:{self.port}")
            
            while self.running:
                try:
                    logger.info("Waiting for client connection...")
                    client_socket, address = self.server_socket.accept()
                    client_ip = address[0]
                    client_port = address[1]
                    logger.info(f"New connection from {client_ip}:{client_port}")
                    
                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.handle_client(client_socket)
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Connection handling error: {e}", exc_info=True)
                    if 'client_socket' in locals():
                        client_socket.close()
                        
        except Exception as e:
            logger.error(f"Server start error: {e}", exc_info=True)
        finally:
            self.cleanup()

    def start(self):
        """Start the TCP server and Genesis simulator"""
        try:

            if self.genesis_sim.enable_vis:
                print("Initializing simulation")
                self.genesis_sim.initialize_simulation()
                # If visualization is enabled, run server in a separate thread
                server_thread = threading.Thread(target=self.run_server)
                server_thread.daemon = True
                server_thread.start()
            else:
                # If no visualization, run server in main thread
                self.run_server()

            # Start Genesis simulator
            self.genesis_sim.start()
            self.running = True
            
                
        except Exception as e:
            logger.error(f"Server start error: {e}", exc_info=True)
            self.cleanup()
            raise

    def cleanup(self):
        """Clean up all resources"""
        logger.info("Cleaning up server resources...")
        if hasattr(self, 'client_socket') and self.client_socket:
            try:
                self.client_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass
            self.client_socket.close()
            
        if hasattr(self, 'server_socket') and self.server_socket:
            try:
                self.server_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass
            self.server_socket.close()
            
        if hasattr(self, 'command_socket') and self.command_socket:
            try:
                self.command_socket.shutdown(socket.SHUT_RDWR)
            except:
                pass
            self.command_socket.close()
            
    def stop(self):
        """Stop the server and clean up resources"""
        logger.info("Stopping server...")
        self.running = False
        self.cleanup()
        # Stop Genesis simulator
        self.genesis_sim.stop()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False,
                      help="Enable visualization of the Genesis simulator")
    args = parser.parse_args()

    server = FrankaSimServer(enable_vis=args.vis)
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop()

if __name__ == "__main__":
    main()