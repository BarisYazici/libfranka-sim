#!/usr/bin/env python3

import argparse
import enum
import errno
import importlib
import logging
import select
import socket
import struct
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from franka_sim.control_modes import ControlMode
from franka_sim.franka_protocol import (
    COMMAND_PORT,
    Command,
    ConnectStatus,
    ControllerMode,
    LibfrankaControllerMode,
    LibfrankaMotionGeneratorMode,
    MessageHeader,
    MotionGeneratorMode,
    MoveCommand,
    MoveStatus,
    SetCartesianImpedanceCommand,
    SetCollisionBehaviorCommand,
    SetEEToKCommand,
    SetGuidingModeCommand,
    SetJointImpedanceCommand,
    SetLoadCommand,
    SetNEToEECommand,
    convert_to_libfranka_controller_mode,
    convert_to_libfranka_motion_mode,
)
from franka_sim.gripper_server import FrankaGripperServer
from franka_sim.robot_state import RobotState

# Configure detailed logging for debugging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


#: Single-arm physics backends, mapped to the module and class implementing the
#: simulator contract this server consumes. Imported lazily in
#: :func:`resolve_sim_class` so choosing one backend never pays the other's
#: (multi-second, native) import cost -- and so an install without Genesis can
#: still run the default one.
SINGLE_ARM_PHYSICS = {
    "genesis": ("franka_sim.franka_genesis_sim", "FrankaGenesisSim"),
    "mujoco": ("franka_sim.mujoco_franka_sim", "MujocoFrankaSim"),
}

#: Backend used when the caller does not inject a simulator or name one. MuJoCo
#: holds real time at the 1 ms step the FCI serves; Genesis needs 2.5 ms.
DEFAULT_PHYSICS = "mujoco"


def resolve_sim_class(physics: str = DEFAULT_PHYSICS):
    """Import and return the single-arm simulator class for one physics backend."""
    if physics not in SINGLE_ARM_PHYSICS:
        raise ValueError(
            f"Unknown physics backend {physics!r}; expected one of {sorted(SINGLE_ARM_PHYSICS)}"
        )
    module_name, class_name = SINGLE_ARM_PHYSICS[physics]
    return getattr(importlib.import_module(module_name), class_name)


class RobotMode(enum.IntEnum):
    """Operating modes of the Franka robot"""

    kOther = 0
    kIdle = 1
    kMove = 2
    kGuiding = 3
    kReflex = 4
    kUserStopped = 5
    kAutomaticErrorRecovery = 6


class FrankaSimServer:
    """
    A simulation server implementing the Franka robot control interface protocol.
    Handles both TCP command communication and UDP state updates.
    """

    # Default arm model served via GetRobotModel: the hand-less FR3 URDF that
    # libfranka_new ships as its own test fixture, so a stock client can always
    # build its Pinocchio model from it.
    DEFAULT_ARM_URDF = Path(__file__).resolve().parent / "models" / "fr3.urdf"

    def __init__(
        self,
        host="0.0.0.0",
        port=COMMAND_PORT,
        enable_vis=False,
        genesis_sim=None,
        urdf_path=None,
        enable_gripper=True,
        gripper_backend=None,
        gripper_physics: bool = False,
        mobile_base: bool = False,
        physics: str = DEFAULT_PHYSICS,
    ):
        """
        Initialize the Franka simulation server.

        Args:
            host: IP address to bind to (default: all interfaces)
            port: TCP port for command interface
            enable_vis: Enable visualization of the simulator
            genesis_sim: Optional pre-configured simulator instance (any backend,
                or a fake in tests). Kept under its historical name because
                dozens of callers inject through it.
            physics: Backend built when ``genesis_sim`` is not injected --
                ``mujoco`` (default) or ``genesis``.
            urdf_path: URDF served to the client via GetRobotModel. Defaults to the
                bundled hand-less FR3 arm model.
            mobile_base: Serve a swerve mobile base instead of an arm. The
                simulator must implement ``update_base_twist`` and the client
                drives it with the kCartesianVelocity motion generator.
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.transmitting_state = False
        self.library_version = 10  # Matches research_interface kVersion in libfranka_new
        self.command_socket = None  # UDP socket for receiving commands
        self.current_motion_id = 0
        self.client_socket = None
        self.tcp_thread = None
        self.udp_socket = None
        self.client_address = None
        self.client_udp_port = None
        self.control_mode = ControlMode.NONE
        self.connection_running = False  # New flag for per-connection state
        self.mobile_base = mobile_base
        # Latch so the mobile "motion finished" hold log fires once per
        # transition, not once per datagram: unlike the arm path, the mobile
        # branch's control_mode stays STEERING_DRIVE (never becomes POSITION),
        # so the `if self.control_mode != ControlMode.POSITION` guard around
        # _switch_to_hold_position() never latches on its own. Mirrors
        # SwerveBase._twist_rejected.
        self._mobile_hold_logged = False

        # Build the physics backend unless one was injected.
        if genesis_sim is None:
            logger.info("Initializing simulation (physics backend: %s)", physics)
            self.genesis_sim = resolve_sim_class(physics)(
                enable_vis=enable_vis, enable_hand=(gripper_physics and enable_gripper)
            )
            logger.info("Simulation initialized")
        else:
            self.genesis_sim = genesis_sim

        self.robot_state = RobotState()

        # Robot model (URDF) served to the client via GetRobotModel. In
        # libfranka_new the client builds its own Pinocchio model from this.
        self.urdf_string = self._load_robot_model(urdf_path)

        # Co-located gripper server (libfranka gripper protocol, port 1338).
        # Self-contained: its own backend and sockets, independent of the arm's
        # Genesis loop. Launched from start() as a daemon thread.
        self.gripper_thread = None
        if enable_gripper:
            if gripper_physics:
                from franka_sim.gripper_physics import GenesisFrankaHand

                backend = GenesisFrankaHand(self.genesis_sim)
            else:
                backend = gripper_backend
            self.gripper_server = FrankaGripperServer(host=host, backend=backend)
        else:
            self.gripper_server = None

    def _load_robot_model(self, urdf_path):
        """Read the URDF served via GetRobotModel (defaults to the bundled FR3)."""
        path = Path(urdf_path) if urdf_path is not None else self.DEFAULT_ARM_URDF
        with open(path, "r", encoding="utf-8") as urdf_file:
            return urdf_file.read()

    def reset_state(self):
        """Reset all connection-specific state variables for a new connection"""
        self.transmitting_state = False
        self.current_motion_id = 0
        self.client_socket = None
        self.tcp_thread = None
        self.udp_socket = None
        self.client_address = None
        self.client_udp_port = None
        self.control_mode = ControlMode.NONE
        self.connection_running = False
        self._mobile_hold_logged = False
        self.robot_state = RobotState()  # Create fresh robot state for new connection

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
        client_socket.sendall(header_bytes + response_data)
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
        client_socket.sendall(response_header.to_bytes() + payload)
        logger.info(f"Sent GetRobotModel response ({len(urdf_bytes)} URDF bytes)")

    def start_command_receiver(self):
        """Start UDP command receiver on specified port"""
        try:
            self.command_thread = threading.Thread(target=self._handle_commands)
            self.command_thread.daemon = True
            self.command_thread.start()

        except Exception as e:
            logger.error(f"Error starting command receiver: {e}", exc_info=True)

    def _handle_commands(self):
        """Handle incoming UDP robot commands"""
        logger.info("Command handler thread started")

        try:
            logger.info("Starting UDP command polling")
            # Setup poll object for UDP socket
            poller = select.poll()
            logger.debug(f"Command socket file descriptor: {self.udp_socket.fileno()}")
            poller.register(self.udp_socket.fileno(), select.POLLIN)
            logger.debug(f"Poller: {poller}")
            timeout = 1  # 1ms timeout

            # RobotCommand packet size (matches the client's RobotCommand struct).
            expected_size = 8 + (7 * 8 + 7 * 8 + 16 * 8 + 6 * 8 + 2 * 8 + 1 + 1) + (7 * 8 + 1)

            # Bound this thread to the current connection (connection_running),
            # not the server lifetime, so stale command threads do not pile up
            # across connections and race on self.udp_socket.
            while self.running and self.connection_running:
                udp_socket = self.udp_socket
                if udp_socket is None:
                    break
                events = poller.poll(timeout)
                if not events:
                    continue

                command = None
                for fd, event in events:
                    if not (event & select.POLLIN):
                        # Socket hung up / errored -> the connection is gone.
                        self.connection_running = False
                        break

                    try:
                        data, addr = udp_socket.recvfrom(expected_size)
                        if len(data) != expected_size:
                            logger.warning(
                                f"Got a UDP packet with wrong size! Expected {expected_size} \
                                bytes, got {len(data)} bytes"
                            )
                            continue
                        # Unpack the command data
                        offset = 0

                        # Unpack message_id
                        message_id = struct.unpack("<Q", data[offset : offset + 8])[0]
                        offset += 8

                        # Unpack MotionGeneratorCommand
                        q_c = struct.unpack("<7d", data[offset : offset + 56])
                        offset += 56

                        dq_c = struct.unpack("<7d", data[offset : offset + 56])
                        offset += 56

                        O_T_EE_c = struct.unpack("<16d", data[offset : offset + 128])
                        offset += 128

                        O_dP_EE_c = struct.unpack("<6d", data[offset : offset + 48])
                        offset += 48

                        elbow_c = struct.unpack("<2d", data[offset : offset + 16])
                        offset += 16

                        valid_elbow = bool(data[offset])
                        offset += 1

                        motion_generation_finished = bool(data[offset])
                        offset += 1

                        # Unpack ControllerCommand
                        tau_J_d = struct.unpack("<7d", data[offset : offset + 56])
                        offset += 56

                        torque_command_finished = bool(data[offset])

                        command = {
                            "message_id": message_id,
                            "q_c": q_c,
                            "dq_c": dq_c,
                            "O_T_EE_c": O_T_EE_c,
                            "O_dP_EE_c": O_dP_EE_c,
                            "elbow_c": elbow_c,
                            "valid_elbow": valid_elbow,
                            "motion_generation_finished": motion_generation_finished,
                            "tau_J_d": tau_J_d,
                            "torque_command_finished": torque_command_finished,
                        }

                    except BlockingIOError:
                        break
                    except Exception as e:
                        # logger.error(f"Error receiving message: {e}")
                        break

                # Process newest command if we have one
                if command and command["message_id"] > 0:
                    # End of control: a motion generator signals via
                    # motion_generation_finished, a pure-torque controller
                    # (startTorqueControl) via torque_command_finished. Handle
                    # both -- otherwise the client hangs waiting for the stop to
                    # be acknowledged.
                    if command["motion_generation_finished"] or command["torque_command_finished"]:
                        if self.control_mode != ControlMode.POSITION:
                            self._switch_to_hold_position()

                        # Update state to idle modes
                        self.robot_state.state["motion_generator_mode"] = 0  # kIdle
                        self.robot_state.state["controller_mode"] = 3  # kOther
                        self.robot_state.state["robot_mode"] = RobotMode.kIdle

                        # Send state with new message ID
                        self.robot_state.update()  # This increments message_id
                        final_state = self.robot_state.pack_state()
                        if self.udp_socket is not None:
                            self.udp_socket.sendto(
                                final_state, (self.client_address, self.client_udp_port)
                            )

                        # Send TCP success response for the Move command
                        if self.current_motion_id and self.client_socket is not None:
                            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
                            response_header = MessageHeader(
                                Command.kMove, self.current_motion_id, total_size
                            )
                            header_bytes = response_header.to_bytes()
                            response_data = struct.pack("<B3x", MoveStatus.kSuccess.value)
                            self.client_socket.sendall(header_bytes + response_data)
                            logger.info(f"Sent Move success response for motion ID: \
                                  {self.current_motion_id}")
                            self.current_motion_id = 0  # Reset motion ID after sending response
                        continue

                    # Update Genesis simulator based on control mode
                    if (
                        self.robot_state.state["controller_mode"]
                        == LibfrankaControllerMode.kJointImpedance
                        and self.robot_state.state["motion_generator_mode"]
                        == LibfrankaMotionGeneratorMode.kJointPosition
                    ):
                        if self.control_mode is not ControlMode.POSITION:
                            logger.info("Setting control mode to POSITION")
                            self.genesis_sim.set_control_mode(ControlMode.POSITION)
                            self.control_mode = ControlMode.POSITION
                            # Initialize q_d to current q when first entering position mode
                            self.robot_state.state["q_d"] = self.robot_state.state["q"]
                        # Update q_d with commanded positions
                        self.robot_state.state["q_d"] = list(command["q_c"])
                        self.genesis_sim.update_joint_positions(command["q_c"])
                        self.genesis_sim.update_torques([0.0] * 7)
                    elif (
                        self.robot_state.state["controller_mode"]
                        == LibfrankaControllerMode.kJointImpedance
                        and self.robot_state.state["motion_generator_mode"]
                        == LibfrankaMotionGeneratorMode.kJointVelocity
                    ):
                        if self.control_mode is not ControlMode.VELOCITY:
                            logger.info("Setting control mode to VELOCITY")
                            self.genesis_sim.set_control_mode(ControlMode.VELOCITY)
                            self.control_mode = ControlMode.VELOCITY
                        # Update dq_d with commanded velocities
                        self.robot_state.state["dq_d"] = list(command["dq_c"])
                        self.genesis_sim.update_joint_velocities(command["dq_c"])
                        self.genesis_sim.update_torques([0.0] * 7)
                    elif (
                        self.mobile_base
                        and self.robot_state.state["motion_generator_mode"]
                        == LibfrankaMotionGeneratorMode.kCartesianVelocity
                        and self.robot_state.state["controller_mode"]
                        != LibfrankaControllerMode.kExternalController
                    ):
                        self._handle_cartesian_velocity(command)
                    elif (
                        self.robot_state.state["controller_mode"]
                        == LibfrankaControllerMode.kExternalController
                    ):
                        if self.control_mode is not ControlMode.TORQUE:
                            logger.info("Setting control mode to TORQUE")
                            self.genesis_sim.set_control_mode(ControlMode.TORQUE)
                            self.control_mode = ControlMode.TORQUE
                        # Update tau_J_d with commanded torques
                        self.robot_state.state["tau_J_d"] = list(command["tau_J_d"])
                        self.genesis_sim.update_torques(command["tau_J_d"])

        except Exception as e:
            logger.error(f"Error in read_step: {e}")

    def handle_move_command(self, client_socket, header: MessageHeader, payload: bytes) -> None:
        """Handle Move command received over TCP"""
        try:
            # Parse the move command
            try:
                move_cmd = MoveCommand.from_bytes(payload)
            except ValueError as e:
                logger.error(f"Error handling Move command: {e}")
                self.send_move_response(
                    client_socket,
                    command_id=header.command_id,
                    status=MoveStatus.kInvalidArgumentRejected,
                )
                return

            logger.info(
                f"Received Move command: controller_mode={move_cmd.controller_mode.name}, "
                f"motion_generator_mode={move_cmd.motion_generator_mode.name}"
            )

            # Validate controller mode
            try:
                ControllerMode(move_cmd.controller_mode)
            except ValueError:
                logger.error(f"Error handling Move command:\
                          {move_cmd.controller_mode} is not a valid ControllerMode")
                self.send_move_response(
                    client_socket,
                    command_id=header.command_id,
                    status=MoveStatus.kInvalidArgumentRejected,
                )
                return

            # Update robot state
            self.robot_state.set_motion_generator_mode(
                convert_to_libfranka_motion_mode(move_cmd.motion_generator_mode)
            )
            self.robot_state.set_controller_mode(
                convert_to_libfranka_controller_mode(move_cmd.controller_mode)
            )
            self.robot_state.state["robot_mode"] = RobotMode.kMove
            self.current_motion_id = header.command_id
            # A new Move is a new motion: rearm the mobile hold-log latch so
            # the next time it finishes logs again.
            self._mobile_hold_logged = False

            # Set appropriate control mode in Genesis simulator
            if (
                move_cmd.controller_mode == ControllerMode.kJointImpedance
                and move_cmd.motion_generator_mode == MotionGeneratorMode.kJointPosition
            ):
                logger.info("Setting control mode to POSITION")
                self.genesis_sim.set_control_mode(ControlMode.POSITION)
                self.control_mode = ControlMode.POSITION
            elif (
                move_cmd.controller_mode == ControllerMode.kJointImpedance
                and move_cmd.motion_generator_mode == MotionGeneratorMode.kJointVelocity
            ):
                logger.info("Setting control mode to VELOCITY")
                self.genesis_sim.set_control_mode(ControlMode.VELOCITY)
                self.control_mode = ControlMode.VELOCITY
            elif (
                self.mobile_base
                and move_cmd.motion_generator_mode == MotionGeneratorMode.kCartesianVelocity
                and move_cmd.controller_mode != ControllerMode.kExternalController
            ):
                logger.info("Setting control mode to STEERING_DRIVE")
                self.genesis_sim.set_control_mode(ControlMode.STEERING_DRIVE)
                self.control_mode = ControlMode.STEERING_DRIVE
            elif move_cmd.controller_mode == ControllerMode.kExternalController:
                logger.info("Setting control mode to TORQUE")
                self.genesis_sim.set_control_mode(ControlMode.TORQUE)
                self.control_mode = ControlMode.TORQUE

            # First send motion started response
            logger.info("Sending kMotionStarted response")
            self.send_move_response(
                client_socket, command_id=header.command_id, status=MoveStatus.kMotionStarted
            )
            logger.info(f"Motion started with ID: {self.current_motion_id}")

        except Exception as e:
            logger.error(f"Error handling Move command: {e}")
            # Send error response
            self.send_move_response(
                client_socket, command_id=header.command_id, status=MoveStatus.kAborted
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
            logger.debug(f"Sending Move response with status: {status.name} (value={status.value})")
            # Ensure we're using the enum value, not the enum itself
            status_value = status.value if isinstance(status, MoveStatus) else status
            response_data = struct.pack("<B3x", status_value)

            # Send complete message
            message = header_bytes + response_data
            logger.debug(f"Sending Move response message: {message.hex()}")
            client_socket.sendall(message)
            logger.info(f"Sent Move response: command_id={command_id}, status={status.name}")
        except Exception as e:
            logger.error(f"Error sending Move response: {e}", exc_info=True)

    def _switch_to_hold_position(self):
        """Freeze the simulator when a motion finishes or StopMove arrives.

        A mobile base has no joint-space hold: the correct "stop" is a zero
        body twist. An arm holds its current joint positions with zero torque.
        """
        if self.mobile_base:
            if not self._mobile_hold_logged:
                logger.info("Motion finished: commanding zero base twist")
                self._mobile_hold_logged = True
            self.genesis_sim.update_base_twist([0.0] * 6)
            # Keep the simulator's mode in lockstep with the server's: without
            # this, a client that never re-Moves leaves genesis_sim's own
            # control_mode wherever it was (e.g. still mid-transition), so
            # server and simulator could disagree about mode after a hold.
            self.genesis_sim.set_control_mode(ControlMode.STEERING_DRIVE)
            self.control_mode = ControlMode.STEERING_DRIVE
            return

        logger.info("Motion finished: switching to position control and holding position")
        current_joint_positions = self.genesis_sim.get_robot_state()["q"]
        self.genesis_sim.set_control_mode(ControlMode.POSITION)
        self.control_mode = ControlMode.POSITION
        self.genesis_sim.update_joint_positions(current_joint_positions)
        self.genesis_sim.update_torques([0.0] * 7)

    def _handle_cartesian_velocity(self, command):
        """Route a cartesian-velocity command to the mobile base.

        The TMR master accepts only a body-frame twist; libfranka carries it in
        ``O_dP_EE_c`` = ``[vx, vy, vz, wx, wy, wz]``. Swerve inverse kinematics
        and the wheel targets live in the simulator, mirroring the real robot
        where the master does the IK onboard.

        Only reached on a ``mobile_base`` server: the dispatcher branch below
        carries that guard, so there is no ``mobile_base`` test here -- and no
        log statement either, since this runs once per UDP datagram (~1 kHz).
        """
        twist = list(command["O_dP_EE_c"])
        self.robot_state.state["O_dP_EE_c"] = twist
        self.robot_state.state["O_dP_EE_d"] = twist

        if self.control_mode is not ControlMode.STEERING_DRIVE:
            # Logged only on the transition, never per datagram.
            logger.info("Setting control mode to STEERING_DRIVE")
            self.genesis_sim.set_control_mode(ControlMode.STEERING_DRIVE)
            self.control_mode = ControlMode.STEERING_DRIVE

        self.genesis_sim.update_base_twist(twist)

    def handle_stop_move_command(self, client_socket, header: MessageHeader):
        """Handle StopMove command received over TCP"""
        try:
            logger.info("Processing StopMove command")

            # Send success response for StopMove first
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(Command.kStopMove, header.command_id, total_size)
            header_bytes = response_header.to_bytes()

            # Status 0 = Success
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            client_socket.sendall(header_bytes + response_data)
            logger.info("Sent StopMove success response")

            if self.control_mode != ControlMode.POSITION:
                self._switch_to_hold_position()

            # Send one final state with both modes set to idle
            if hasattr(self, "udp_socket") and self.udp_socket:
                # Update state to idle modes
                self.robot_state.state["motion_generator_mode"] = 0  # kNone
                self.robot_state.state["controller_mode"] = 3  # kOther
                self.robot_state.state["robot_mode"] = RobotMode.kIdle

                # Send state with new message ID
                self.robot_state.update()  # This increments message_id
                final_state = self.robot_state.pack_state()
                self.udp_socket.sendto(final_state, (self.client_address, self.client_udp_port))
                logger.info(f"Sent final robot state with message_id:\
                          {self.robot_state.state['message_id']}")

            # Stop robot state transmission
            self.transmitting_state = False
            logger.info("Stopped robot state transmission")

            # Send Move response to break the waiting loop in the client
            if self.current_motion_id:
                # Create a Move response header
                move_response_header = MessageHeader(Command.kMove, self.current_motion_id, 16)
                move_header_bytes = move_response_header.to_bytes()
                move_response_data = struct.pack("<B3x", MoveStatus.kSuccess.value)
                client_socket.sendall(move_header_bytes + move_response_data)
                logger.info(f"Sent Move success response for motion ID: {self.current_motion_id}")
                self.current_motion_id = 0

            # Set connection_running to False instead of self.running
            self.connection_running = False

        except Exception as e:
            logger.error(f"Error handling StopMove command: {e}")
            # Send error response
            total_size = 12 + 4
            response_header = MessageHeader(Command.kStopMove, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 5)  # Status 5 = Aborted
            client_socket.sendall(header_bytes + response_data)

    def handle_set_collision_behavior_command(
        self, client_socket, header: MessageHeader, payload: bytes
    ):
        """Handle SetCollisionBehavior command received over TCP"""
        try:
            # Parse the command
            cmd = SetCollisionBehaviorCommand.from_bytes(payload)
            logger.info("Received SetCollisionBehavior command with values:")
            logger.debug(f"Lower torque thresholds acc: {cmd.lower_torque_thresholds_acceleration}")
            logger.debug(f"Upper torque thresholds acc: {cmd.upper_torque_thresholds_acceleration}")

            # For now, just acknowledge the command without actually implementing behavior
            # Send success response (status = 0)
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(
                Command.kSetCollisionBehavior, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            client_socket.sendall(header_bytes + response_data)
            logger.info("Sent SetCollisionBehavior success response")

        except Exception as e:
            logger.error(f"Error handling SetCollisionBehavior command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(
                Command.kSetCollisionBehavior, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            client_socket.sendall(header_bytes + response_data)

    def handle_set_joint_impedance_command(
        self, client_socket, header: MessageHeader, payload: bytes
    ):
        """Handle SetJointImpedance command received over TCP"""
        try:
            # Parse the command
            cmd = SetJointImpedanceCommand.from_bytes(payload)
            logger.info("Received SetJointImpedance command with values:")
            logger.debug(f"Joint stiffness values: {cmd.K_theta}")

            # For now, just acknowledge the command without actually implementing behavior
            # Send success response (status = 0)
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(
                Command.kSetJointImpedance, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            client_socket.sendall(header_bytes + response_data)
            logger.info("Sent SetJointImpedance success response")

        except Exception as e:
            logger.error(f"Error handling SetJointImpedance command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(
                Command.kSetJointImpedance, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            client_socket.sendall(header_bytes + response_data)

    def handle_set_cartesian_impedance_command(
        self, client_socket, header: MessageHeader, payload: bytes
    ):
        """Handle SetCartesianImpedance command received over TCP"""
        try:
            # Parse the command
            cmd = SetCartesianImpedanceCommand.from_bytes(payload)
            logger.info("Received SetCartesianImpedance command with values:")
            logger.debug(f"Cartesian stiffness values: {cmd.K_x}")

            # For now, just acknowledge the command without actually implementing behavior
            # Send success response (status = 0)
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(
                Command.kSetCartesianImpedance, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            client_socket.sendall(header_bytes + response_data)
            logger.info("Sent SetCartesianImpedance success response")

        except Exception as e:
            logger.error(f"Error handling SetCartesianImpedance command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(
                Command.kSetCartesianImpedance, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            client_socket.sendall(header_bytes + response_data)

    def handle_set_guiding_mode_command(
        self, client_socket, header: MessageHeader, payload: bytes
    ):
        """Handle SetGuidingMode command received over TCP.

        Without a response, a real libfranka client blocks forever (the
        setGuidingMode() call has no way to complete). No RobotState field
        reflects the guiding mode, so this is ACK-only, mirroring the
        SetCollisionBehavior no-op pattern.
        """
        try:
            # Parse the command
            cmd = SetGuidingModeCommand.from_bytes(payload)
            logger.info("Received SetGuidingMode command with values:")
            logger.debug(f"Guiding mode: {cmd.guiding_mode}, nullspace: {cmd.nullspace}")

            # For now, just acknowledge the command without actually implementing behavior
            # Send success response (status = 0)
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(Command.kSetGuidingMode, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            client_socket.sendall(header_bytes + response_data)
            logger.info("Sent SetGuidingMode success response")

        except Exception as e:
            logger.error(f"Error handling SetGuidingMode command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(Command.kSetGuidingMode, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            client_socket.sendall(header_bytes + response_data)

    def handle_set_ee_to_k_command(self, client_socket, header: MessageHeader, payload: bytes):
        """Handle SetEEToK command received over TCP.

        The real robot reflects EE_T_K (stiffness frame relative to the
        end-effector) back in RobotState, so store it there too.
        """
        try:
            # Parse the command
            cmd = SetEEToKCommand.from_bytes(payload)
            logger.info("Received SetEEToK command with values:")
            logger.debug(f"EE_T_K: {cmd.EE_T_K}")

            # Reflect EE_T_K in subsequent RobotState broadcasts, like the real robot.
            self.robot_state.state["EE_T_K"] = cmd.EE_T_K

            # Send success response (status = 0)
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(Command.kSetEEToK, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            client_socket.sendall(header_bytes + response_data)
            logger.info("Sent SetEEToK success response")

        except Exception as e:
            logger.error(f"Error handling SetEEToK command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(Command.kSetEEToK, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            client_socket.sendall(header_bytes + response_data)

    def handle_set_ne_to_ee_command(self, client_socket, header: MessageHeader, payload: bytes):
        """Handle SetNEToEE command received over TCP.

        The real robot reflects NE_T_EE (end-effector relative to the
        nominal end-effector, i.e. the flange-mount frame) back in
        RobotState, so store it there too.
        """
        try:
            # Parse the command
            cmd = SetNEToEECommand.from_bytes(payload)
            logger.info("Received SetNEToEE command with values:")
            logger.debug(f"NE_T_EE: {cmd.NE_T_EE}")

            # Reflect NE_T_EE in subsequent RobotState broadcasts, like the real robot.
            self.robot_state.state["NE_T_EE"] = cmd.NE_T_EE

            # Send success response (status = 0)
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(Command.kSetNEToEE, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            client_socket.sendall(header_bytes + response_data)
            logger.info("Sent SetNEToEE success response")

        except Exception as e:
            logger.error(f"Error handling SetNEToEE command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(Command.kSetNEToEE, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            client_socket.sendall(header_bytes + response_data)

    def handle_set_load_command(self, client_socket, header: MessageHeader, payload: bytes):
        """Handle SetLoad command received over TCP.

        The real robot reflects the externally-mounted load (mass, center of
        mass, inertia) back in RobotState, so store it there too.
        """
        try:
            # Parse the command
            cmd = SetLoadCommand.from_bytes(payload)
            logger.info("Received SetLoad command with values:")
            logger.debug(
                f"m_load: {cmd.m_load}, F_x_Cload: {cmd.F_x_Cload}, I_load: {cmd.I_load}"
            )

            # Reflect the load in subsequent RobotState broadcasts, like the real robot.
            self.robot_state.state["m_load"] = cmd.m_load
            self.robot_state.state["F_x_Cload"] = cmd.F_x_Cload
            self.robot_state.state["I_load"] = cmd.I_load

            # Send success response (status = 0)
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(Command.kSetLoad, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            client_socket.sendall(header_bytes + response_data)
            logger.info("Sent SetLoad success response")

        except Exception as e:
            logger.error(f"Error handling SetLoad command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(Command.kSetLoad, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            client_socket.sendall(header_bytes + response_data)

    def handle_automatic_error_recovery_command(
        self, client_socket, header: MessageHeader, payload: bytes
    ):
        """Handle AutomaticErrorRecovery command received over TCP.

        The real robot clears any latched reflex/error and returns to Idle.
        Without a response here, libfranka (and franka_hardware, which calls
        automaticErrorRecovery() on activation) blocks forever waiting on the
        TCP reply, stalling the whole control stack.
        """
        try:
            # AutomaticErrorRecovery has an empty request; nothing to parse.
            # Clear any error/reflex state and return the arm to Idle.
            self.robot_state.state["robot_mode"] = RobotMode.kIdle

            # Response is ResponseBase: a single uint8 status (kSuccess = 0).
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(
                Command.kAutomaticErrorRecovery, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            client_socket.sendall(header_bytes + response_data)
            logger.info("Sent AutomaticErrorRecovery success response")

        except Exception as e:
            logger.error(f"Error handling AutomaticErrorRecovery command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(
                Command.kAutomaticErrorRecovery, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            client_socket.sendall(header_bytes + response_data)

    def handle_tcp_messages(self, client_socket):
        """Handle TCP messages in a separate thread"""
        logger.info("TCP message handler thread started")
        while self.running:  # Keep the TCP thread running even after client disconnects
            try:
                # Check if socket is still connected
                try:
                    client_socket.getpeername()
                except socket.error as e:
                    logger.info("Client socket disconnected")
                    # Instead of breaking, reset state and continue
                    self.transmitting_state = False
                    self.connection_running = False
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
                logger.info("Connection error: Resetting state and waiting for new client...")
                break  # Break only from the inner loop
            except Exception as e:
                logger.error(f"Error in TCP thread: {e}", exc_info=True)
                if not self.running:  # Only break if server is shutting down
                    break
                # For other errors, reset state and continue
                self.transmitting_state = False
                self.connection_running = False
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

    def start_robot_state_transmission(self, client_address: str, client_udp_port: int):
        """
        Start UDP transmission of robot state updates.
        """
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # TODO move to somewhere appropriate
            self.start_command_receiver()

            # port of the udp_socket
            udp_port = self.udp_socket.getsockname()[1]
            logger.debug(f"UDP port: {udp_port}")
            self.client_address = client_address
            self.client_udp_port = client_udp_port
            # Initialize timing statistics
            total_cycles = 0
            total_genesis_time = 0
            total_cycle_time = 0
            last_stats_time = time.time()

            logger.info(f"Starting UDP transmission to {client_address}:{client_udp_port}")
            self.transmitting_state = True
            first_state_sent = False

            # Pace the broadcast to a ~1 kHz deadline schedule. State reads are now
            # cheap (the physics thread publishes snapshots), so without a pacer
            # this loop would spin well above 1 kHz and burn a whole CPU core.
            period = 0.001  # 1 kHz target broadcast rate
            next_deadline = time.perf_counter()

            while self.running and self.connection_running and self.transmitting_state:
                try:
                    cycle_start = time.time()

                    genesis_start = time.time()
                    sim_state = self.genesis_sim.get_robot_state()
                    genesis_time = time.time() - genesis_start
                    total_genesis_time += genesis_time

                    # Initialize q_d to current q on first state update if not already set
                    if not first_state_sent:
                        self.robot_state.state["q_d"] = sim_state["q"]

                    self.robot_state.state.update(sim_state)

                    # Pack and send current robot state
                    state = self.robot_state.pack_state()
                    if self.udp_socket and not self.udp_socket._closed:
                        self.udp_socket.sendto(state, (client_address, client_udp_port))

                    # After first state is sent, send a Move success response
                    if not first_state_sent and self.current_motion_id:
                        self.send_move_response(
                            self.client_socket,
                            command_id=self.current_motion_id,
                            status=MoveStatus.kSuccess,
                        )
                        first_state_sent = True

                    # Update state for next iteration
                    self.robot_state.update()
                    # Calculate cycle statistics (work time, before pacing sleep)
                    cycle_time = time.time() - cycle_start
                    total_cycle_time += cycle_time
                    total_cycles += 1

                    # Sleep until the next 1 kHz deadline (soft real-time).
                    next_deadline += period
                    remaining = next_deadline - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)
                    elif remaining < -period:
                        # Fell a full cycle behind; resync to avoid a catch-up burst.
                        next_deadline = time.perf_counter()

                    # Log statistics every second
                    if time.time() - last_stats_time >= 1.0:
                        avg_genesis_time = (
                            total_genesis_time / total_cycles
                        ) * 1000  # Convert to ms
                        avg_cycle_time = (total_cycle_time / total_cycles) * 1000  # Convert to ms
                        freq = total_cycles / (time.time() - last_stats_time)

                        logger.info(
                            f"State Update Stats - Freq: {freq:.1f}Hz, "
                            f"Genesis Time: {avg_genesis_time:.2f}ms, "
                            f"Total Cycle: {avg_cycle_time:.2f}ms"
                        )

                        # Reset statistics
                        total_cycles = 0
                        total_genesis_time = 0
                        total_cycle_time = 0
                        last_stats_time = time.time()

                except Exception as e:
                    logger.error(f"Error in UDP transmission: {e}")
                    if not self.running or not self.connection_running:
                        break

        except Exception as e:
            logger.error(f"Error in robot state transmission: {e}")
        finally:
            self.transmitting_state = False
            if self.udp_socket:
                try:
                    self.udp_socket.close()
                except Exception as e:
                    logger.error(f"Error closing UDP socket: {e}")
                self.udp_socket = None

    def run_server(self):
        """Main server loop that runs in a separate thread when visualization is enabled"""
        try:
            logger.info("Starting TCP server initialization...")
            # Start TCP server
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            logger.info("Created server socket")

            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            self.server_socket.settimeout(1.0)
            logger.info("Set socket options")

            try:
                logger.info(f"Attempting to bind to {self.host}:{self.port}")
                self.server_socket.bind((self.host, self.port))
                logger.info("Successfully bound to address")
            except OSError as e:
                if e.errno == errno.EADDRINUSE:  # Address already in use
                    logger.warning(
                        f"Port {self.port} is in use, attempting to force close and rebind..."
                    )
                    self.server_socket.close()
                    time.sleep(1)
                    self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                    self.server_socket.bind((self.host, self.port))
                    logger.info("Successfully rebound to address after force close")
                else:
                    logger.error(f"Failed to bind: {e}")
                    raise

            self.server_socket.listen(1)
            logger.info(f"Server listening on {self.host}:{self.port}")
            self.running = True

            while self.running:
                try:
                    # Reset state before accepting new connection
                    self.reset_state()
                    logger.info("Server ready for new client connection...")

                    client_socket, address = self.server_socket.accept()
                    client_ip = address[0]
                    client_port = address[1]
                    logger.info(f"New connection from {client_ip}:{client_port}")

                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                    # Handle client - this will block until client disconnects
                    self.handle_client(client_socket)

                    logger.info("Client session ended, ready for next client")

                except socket.timeout:
                    # Just continue waiting for new connections
                    continue
                except Exception as e:
                    logger.error(f"Connection handling error: {e}", exc_info=True)
                    if "client_socket" in locals():
                        try:
                            client_socket.close()
                        except Exception as e:
                            logger.error(f"Error closing client socket: {e}")
                    # Reset state and continue listening for next client
                    self.reset_state()
                    continue

        except Exception as e:
            logger.error(f"Server start error: {e}", exc_info=True)
            self.running = False
        finally:
            self.cleanup()

    def start_gripper_server(self):
        """Launch the co-located gripper server's accept loop in a daemon thread."""
        if self.gripper_server is None:
            return
        self.gripper_thread = threading.Thread(target=self.gripper_server.run_server, daemon=True)
        self.gripper_thread.start()
        logger.info("Gripper server running in background thread")

    def start(self):
        """Start the TCP server and Genesis simulator"""
        try:
            self.running = True
            logger.info("Starting server and simulation")

            # Initialize Genesis simulator first
            self.genesis_sim.initialize_simulation()
            logger.info("Genesis simulation initialized")

            # Bring up the gripper server alongside the arm (port 1338).
            self.start_gripper_server()

            if self.genesis_sim.enable_vis:
                # Run server in a background thread when visualization is enabled
                server_thread = threading.Thread(target=self.run_server)
                server_thread.daemon = True
                server_thread.start()
                logger.info("Server running in background thread")

                # Start Genesis simulator (visualization) in main thread
                logger.info("Starting Genesis simulator with visualization")
                self.genesis_sim.start()
            else:
                # Without visualization, run the TCP/UDP server in a background
                # thread and step the Genesis physics loop in the main thread.
                # (run_server() blocks in its accept loop, so it must not run in
                # the main thread or the simulation would never step.)
                server_thread = threading.Thread(target=self.run_server)
                server_thread.daemon = True
                server_thread.start()
                logger.info("Server running in background thread (headless)")

                logger.info("Starting Genesis simulator (headless)")
                self.genesis_sim.start()

        except Exception as e:
            logger.error(f"Server start error: {e}", exc_info=True)
            self.cleanup()
            raise

    def cleanup(self):
        """Clean up all resources.

        Every socket attribute is cached into a local before use: another
        thread (the per-client connection's teardown, via reset_state()) can
        null these attributes concurrently. Re-reading ``self.<attr>`` between
        the shutdown() and close() calls risks the attribute having gone to
        None in between -- ``None.close()`` raises AttributeError, which is
        not a socket.error/OSError and therefore escapes the except clauses,
        aborting the rest of cleanup() and leaking the SO_REUSEPORT listener.
        Binding the reference once up front makes both calls operate on the
        same object regardless of what happens to the attribute afterwards.
        """
        logger.info("Cleaning up server resources...")

        # Stop all running operations
        self.running = False
        self.transmitting_state = False
        self.connection_running = False

        # Clean up client socket
        sock, self.client_socket = self.client_socket, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

        # Clean up server socket
        sock, self.server_socket = self.server_socket, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

        # Clean up command socket
        sock, self.command_socket = self.command_socket, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

        # Clean up UDP socket. No shutdown(): a connectionless socket has
        # nothing to shut down and shutdown() would raise ENOTCONN.
        sock, self.udp_socket = self.udp_socket, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

        # Wait for any remaining operations to complete
        time.sleep(0.1)

        # Reset all state
        self.reset_state()
        self.running = False

    def stop(self):
        """Stop the server and clean up resources"""
        logger.info("Stopping server...")
        self.running = False
        self.connection_running = False
        self.transmitting_state = False
        self.cleanup()
        if self.gripper_server is not None:
            self.gripper_server.stop()
            if self.gripper_thread is not None:
                self.gripper_thread.join(timeout=2.0)
        # Stop Genesis simulator
        self.genesis_sim.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        default=False,
        help="Enable visualization of the Genesis simulator",
    )
    args = parser.parse_args()

    server = FrankaSimServer(enable_vis=args.vis)
    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop()


if __name__ == "__main__":
    main()
