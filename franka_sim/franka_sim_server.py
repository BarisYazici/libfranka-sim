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

from franka_sim.comm_constraints import (
    COMMUNICATION_CONSTRAINTS_VIOLATION_INDEX,
    CommConstraintTracker,
    enforcement_enabled_by_env,
)
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
from franka_sim.motion_limits import MotionLimitChecker
from franka_sim.motion_limits import (
    enforcement_enabled_by_env as motion_limit_enforcement_enabled_by_env,
)
from franka_sim.robot_state import RobotState

# Configure detailed logging for debugging
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


#: RobotState fields that report what was *commanded*, not what was measured.
#: The FCI server owns them, so a physics backend's snapshot never overwrites
#: them on the publish path: libfranka defines them as "``q_{c,k-1}``,
#: ``dq_{c,k-1}`` and ``ddq_{c,k-1}`` [...] always sent back to the user in the
#: robot state as ``q_d``, ``dq_d`` and ``ddq_d``" (``docs/overview.rst``), and
#: a conforming client feeds them straight back in -- libfranka's default
#: command low-pass filter blends every command with them. A backend that
#: echoed measured values here, or published its own copy a physics step late,
#: put a wobble into the client's own commands that no client could see or
#: prevent.
#:
#: **Arm roles only.** The mobile base's motion generator is Cartesian: the
#: client commands ``O_dP_EE_c`` and libfranka's filter blends it with
#: ``O_dP_EE_d``, both of which this server owns and writes on every datagram.
#: Nothing commands the base bridge's ``q_d``/``dq_d``/``ddq_d`` -- they
#: describe the four swerve steer/drive joints, which the base's own onboard
#: controller servos -- so the backend's reading of them is the only source
#: there is, and filtering it out simply froze ``q_d`` at the first frame and
#: pinned ``dq_d``/``ddq_d`` to zero on the bridge the teleop reads. See
#: :attr:`FrankaSimServer._server_owned_state_fields`.
COMMANDED_STATE_FIELDS = ("q_d", "dq_d", "ddq_d", "tau_J_d")

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
        enforce_comm_constraints: Optional[bool] = None,
        enforce_motion_limits: Optional[bool] = None,
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
            enforce_comm_constraints: Whether a run of lost command cycles
                aborts the motion with ``communication_constraints_violation``.
                ``None`` (the default) takes it from
                ``$FRANKA_SIM_ENFORCE_COMM_CONSTRAINTS``, i.e. off unless that
                is set. Loss accounting and
                ``control_command_success_rate`` run either way.
            enforce_motion_limits: Whether a commanded signal that breaks the
                FCI's position/velocity/acceleration/jerk/torque limits aborts
                the motion with the matching error. ``None`` (the default)
                takes it from ``$FRANKA_SIM_ENFORCE_MOTION_LIMITS``, i.e. off
                unless that is set. Checking and the rate-limited warning run
                either way. Independent of ``enforce_comm_constraints``.
        """
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        self.transmitting_state = False
        self.library_version = 10  # Matches research_interface kVersion
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
        #: Commanded fields the FCI layer owns on this role, i.e. the ones the
        #: physics snapshot may not overwrite. Empty for the mobile base; see
        #: :data:`COMMANDED_STATE_FIELDS`.
        self._server_owned_state_fields = () if mobile_base else COMMANDED_STATE_FIELDS
        # Latch so the mobile "motion finished" hold log fires once per
        # transition, not once per datagram, for callers that drive
        # _switch_to_hold_position() directly rather than through the
        # once-per-session _engage_idle_hold(). Mirrors
        # SwerveBase._twist_rejected.
        self._mobile_hold_logged = False

        # Idle hold (see _engage_idle_hold): True once this session's control
        # has ended and the simulator has been recaptured, cleared by the next
        # Move. Guarded by _hold_lock together with the UDP dispatch, because
        # session teardown runs on the TCP thread (or the accept thread's
        # finally) while the ~1 kHz UDP command thread may still be applying an
        # in-flight datagram: whichever wins must win completely, or a command
        # applied *after* the hold would leave the arm driven by a dead
        # session's last torque.
        self._hold_lock = threading.Lock()
        self._idle_hold = False

        # One writer at a time on the TCP stream; see _send_tcp.
        self._tcp_send_lock = threading.Lock()

        # Motion session (see _motion_generation): every transition that starts,
        # finishes or aborts a motion runs under this lock, because three
        # threads reach for the same two variables. handle_move_command runs on
        # the TCP thread, the motion_generation_finished datagram on the UDP
        # thread, and a reflex abort on the state-publish thread; unserialised,
        # a real violation could be answered kSuccess, or an abort could kill
        # the brand-new motion that had just replaced the one that violated.
        # Re-entrant because _abort_with_error holds it across _engage_idle_hold.
        # Lock order is always _motion_lock -> _hold_lock, never the reverse.
        self._motion_lock = threading.RLock()
        #: Monotonic token for the running motion, handed to the comm tracker
        #: and the limit checker so a late abort or a stale finish can be
        #: recognised as belonging to a motion that is already over. Never
        #: reused, unlike current_motion_id (a client's Move command_id).
        self._motion_generation = 0
        #: ``(motion_id, status)`` owed to the client, flushed by the state loop
        #: *after* the state carrying the error is on the wire; see
        #: :meth:`_abort_with_error`.
        self._pending_move_response: Optional[Tuple[int, MoveStatus, int]] = None
        #: States serialised by the publish loop so far. Stamped into
        #: :attr:`_pending_move_response` at latch time so a deferred
        #: ``kReflexAborted`` can wait for a state that was packed *after* the
        #: error, not merely sent after it. Written only by the publish thread.
        self._states_packed = 0
        #: ``message_id`` that was current when the running motion's ``Move``
        #: was accepted, and whether any control command of it has arrived yet.
        #: Together they identify a ``*_finished`` datagram left over from the
        #: previous motion; see :meth:`_finish_motion`.
        self._motion_epoch_id = 0
        self._motion_has_commands = False

        # Communication-constraints emulation (see franka_sim.comm_constraints).
        # One tracker per connection, rebuilt by reset_state(). A missed cycle
        # holds the last applied command; nothing is extrapolated.
        self.enforce_comm_constraints = (
            enforcement_enabled_by_env()
            if enforce_comm_constraints is None
            else enforce_comm_constraints
        )
        self.comm = CommConstraintTracker(enforce=self.enforce_comm_constraints)

        # Motion-limit emulation (see franka_sim.motion_limits). One checker
        # per connection, rebuilt by reset_state(), differencing every received
        # command against the last *applied* one.
        self.enforce_motion_limits = (
            motion_limit_enforcement_enabled_by_env()
            if enforce_motion_limits is None
            else enforce_motion_limits
        )
        self.motion_limits = MotionLimitChecker(enforce=self.enforce_motion_limits)

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

    def _load_robot_model(self, urdf_path):
        """Read the URDF served via GetRobotModel (defaults to the bundled FR3)."""
        path = Path(urdf_path) if urdf_path is not None else self.DEFAULT_ARM_URDF
        with open(path, "r", encoding="utf-8") as urdf_file:
            return urdf_file.read()

    def reset_state(self):
        """Reset all connection-specific state variables for a new connection"""
        self.transmitting_state = False
        self.current_motion_id = 0
        self._pending_move_response = None
        self._states_packed = 0
        self._motion_epoch_id = 0
        self._motion_has_commands = False
        self.client_socket = None
        self.tcp_thread = None
        self.udp_socket = None
        self.client_address = None
        self.client_udp_port = None
        self.control_mode = ControlMode.NONE
        self.connection_running = False
        self._mobile_hold_logged = False
        # The hold itself is *not* undone here -- the simulator keeps holding
        # the pose it was recaptured at until the next Move commands something
        # else. Only the once-per-session latch is rearmed.
        self._idle_hold = False
        # A new connection is a new communication channel: no window and no
        # latched violation.
        self.comm = CommConstraintTracker(enforce=self.enforce_comm_constraints)
        # ...and no latched motion-limit violation, no command history to
        # difference the new client's first command against.
        self.motion_limits = MotionLimitChecker(enforce=self.enforce_motion_limits)
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
                        self._finish_motion(command)
                        continue

                    # Communication accounting first: this cycle now has an
                    # answer (a stale echo still counts as a loss, see
                    # CommConstraintTracker.command_received). The command is
                    # applied either way -- within one sim cycle a late command
                    # is still the freshest user intent there is, and the abort
                    # after MAX_CONSECUTIVE_LOST_CYCLES is the faithful
                    # consequence of being late.
                    # This motion has been streamed to, so a later finish
                    # datagram is this motion's (see _finish_motion).
                    self._motion_has_commands = True
                    fresh = self.comm.command_received(command["message_id"])

                    # Motion limits second: the packet did arrive (so it counts
                    # for the cycle either way), but a command the robot would
                    # refuse must not reach the simulator, and must not enter
                    # the history the next command is differenced against.
                    #
                    # Nothing gets past this on its way to physics, whether or
                    # not it was fresh: an unchecked path here was a way to walk
                    # a 40 rad/s step into the simulator behind a stale echo.
                    if not self._accept_within_motion_limits(command, fresh=fresh):
                        continue

                    # A command that did not answer its own cycle is applied --
                    # it is still the freshest intent there is -- but it is not
                    # a later sample of the client's trajectory, so it never
                    # becomes the baseline the next one is differenced against.
                    if fresh:
                        self.motion_limits.record(command)
                    self._dispatch_control_command(command)

        except Exception as e:
            logger.error(f"Error in read_step: {e}")

    def _finish_motion(self, command) -> None:
        """Close the motion a ``*_finished`` datagram ends, if it is still this one.

        A client signals the end of control in the datagram itself -- a motion
        generator via ``motion_generation_finished``, a pure-torque controller
        (``startTorqueControl``) via ``torque_command_finished`` -- and the
        server owes it a final idle state plus the ``kSuccess`` answer to its
        pending ``Move``, or the client hangs.

        The datagram can arrive *late*, though: the UDP thread drains one socket
        while the TCP thread accepts the next ``Move`` on another, so a finish
        belonging to the motion that just ended can be handled after its
        successor has already started. Acting on it then ends a motion that has
        barely begun -- accounting off, success rate pinned at 0.0, enforcement
        dead, and the new ``Move`` answered ``kSuccess`` before it ran.

        The datagram's echoed ``message_id`` is what identifies which motion it
        belongs to, and two facts have to hold together:

        * **it echoes a state no newer than the one that was current when this
          motion's ``Move`` was accepted.** The client sent the datagram before
          it sent that ``Move``, so it cannot be answering a state published
          after the ``Move`` was handled -- and a conforming client's first
          command of a new motion always answers a *later* state, because
          ``startMotion`` spins reading states until the modes match before it
          sends anything. Strictly older, not "no newer": a client whose whole
          motion fits inside one publish cycle echoes the epoch id itself, and
          treating that as stale would leave it waiting for a ``kSuccess`` that
          never comes. Equality costs a rare missed stale datagram; the other
          way round costs a hang.
        * **no control command of the running motion has been seen yet.** A
          client streams before it finishes, so a finish arriving before any
          command is not this motion's. A belt to the first condition's braces:
          on its own it would misread a client that finishes on its very first
          command.

        Note what is deliberately *not* required: that this motion preempted an
        unfinished one. It used to be, and that excluded the common case -- a
        motion that finished perfectly normally, whose 1 kHz finish burst is
        still draining out of the socket when the next ``Move`` is accepted. The
        leftover datagram then ended a motion that had barely begun.
        """
        with self._motion_lock:
            if (
                self._motion_generation
                and not self._motion_has_commands
                and command["message_id"] < self._motion_epoch_id
            ):
                logger.info(
                    "Ignoring a motion-finished datagram from a motion that is "
                    "already over (echoed id %s, motion started at %s)",
                    command["message_id"],
                    self._motion_epoch_id,
                )
                return

            motion_id = self._motion_generation
            self._engage_idle_hold("motion finished", motion_id=motion_id)

            # Update state to idle modes -- unless a reflex is latched, in which
            # case the modes already left kMove and ``robot_mode`` is kReflex.
            # Overwriting that with kIdle would tell the client the motion ended
            # cleanly while ``errors`` still says why it did not, and kReflex is
            # the flag that has to survive until AutomaticErrorRecovery.
            if self.robot_state.state["robot_mode"] != RobotMode.kReflex:
                self.robot_state.state["motion_generator_mode"] = 0  # kIdle
                self.robot_state.state["controller_mode"] = 3  # kOther
                self.robot_state.state["robot_mode"] = RobotMode.kIdle

            # Send state with new message ID
            self.robot_state.update()  # This increments message_id
            final_state = self.robot_state.pack_state()
            if self.udp_socket is not None:
                self.udp_socket.sendto(final_state, (self.client_address, self.client_udp_port))

            # Send TCP success response for the Move command
            if self.current_motion_id and self.client_socket is not None:
                total_size = 12 + 4  # Header (12) + status (1) + padding (3)
                response_header = MessageHeader(Command.kMove, self.current_motion_id, total_size)
                header_bytes = response_header.to_bytes()
                response_data = struct.pack("<B3x", MoveStatus.kSuccess.value)
                self._send_tcp(self.client_socket, header_bytes + response_data)
                logger.info(
                    "Sent Move success response for motion ID: %s", self.current_motion_id
                )
                self.current_motion_id = 0  # Reset motion ID after sending response

    def _dispatch_control_command(self, command) -> None:
        """Route one UDP RobotCommand to the simulator, unless the hold is on.

        Split out of :meth:`_handle_commands` so the whole simulator-facing part
        of a control cycle sits inside ``_hold_lock``: a datagram that was
        already in flight when the session ended must not be applied *after*
        :meth:`_engage_idle_hold` recaptured the arm, or the hold would be
        immediately overwritten by a dead session's last torque.
        """
        with self._hold_lock:
            if self._idle_hold:
                return

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
                self._publish_commanded_derivatives("dq_d", "ddq_d")
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
                self._publish_commanded_derivatives("ddq_d")
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

    def _publish_commanded_derivatives(self, *fields: str) -> None:
        """Report the derivatives the applied command implies, in order.

        ``dq_d`` and ``ddq_d`` for a position generator, ``ddq_d`` alone for a
        velocity one -- the fields libfranka documents as carrying
        ``dq_{c,k-1}`` and ``ddq_{c,k-1}``, which is what lets a client compute
        the derivatives the robot will compute "in advance, even in case of
        packet losses" (``docs/overview.rst``).

        The numbers come from the limit checker, which has already differenced
        the command that is being applied. Silent no-op when no motion is armed
        there, so nothing here depends on the checks being switched on.
        """
        derivatives = self.motion_limits.applied_derivatives()
        if derivatives is None:
            return
        for field, values in zip(fields, derivatives):
            self.robot_state.state[field] = values

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

            with self._motion_lock:
                if self._refuse_move_while_latched(client_socket, header):
                    return

                self._preempt_running_motion()

                # A fresh session token for this motion; see _motion_generation.
                self._motion_generation += 1
                generation = self._motion_generation
                self._motion_epoch_id = self.robot_state.state["message_id"]
                self._motion_has_commands = False

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
                # the next time it finishes logs again, and release the idle hold so
                # this motion's UDP commands are applied again. Released before the
                # set_control_mode() calls below, which are what actually override
                # the hold in the simulator.
                self._mobile_hold_logged = False
                with self._hold_lock:
                    self._idle_hold = False
                # A new motion is a new success-rate window.
                self.comm.start_motion(generation)

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

                # Seed the limit checker from the state the client is judged
                # against -- q_d / dq_d / ddq_d / tau_J_d are "always sent back to
                # the user in the robot state" for exactly this purpose (libfranka
                # docs/overview.rst), so the client can predict every derivative the
                # robot will compute. Last, because it needs the control mode the
                # branches above just decided.
                self.motion_limits.start_motion(
                    self.control_mode, self._publish_hold_setpoint(), generation
                )

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

    def _preempt_running_motion(self) -> bool:
        """End a motion that a new ``Move`` is displacing. True if there was one.

        ``Move::Status::kPreempted`` -- "Move command preempted!" in
        ``src/robot_impl.h`` -- is the answer the *displaced* motion gets, so
        that is what the pending ``Move`` is answered with here.

        The point is not the status byte, it is what happens to the history.
        Starting a second motion on top of a running one used to leave the
        limit checker's applied history in place and then rebase it on the new
        motion's first command, which is checked against the start-pose
        tolerance alone -- so every extra ``Move`` bought a free step of up to
        :data:`franka_sim.motion_limits.START_POSE_TOLERANCE` with no
        velocity, acceleration or jerk check on it. Ending the old motion the
        ordinary way first (recapture, zero commanded derivatives) means the
        new one is seeded from a genuine standstill at the pose the robot is
        actually in, and every command after its first is differenced again.

        Called with ``_motion_lock`` held.
        """
        if not self.current_motion_id:
            return False
        logger.warning(
            "Move arrived while motion %s was still running: preempting it",
            self.current_motion_id,
        )
        self._engage_idle_hold("preempted by a new Move")
        if self.client_socket is not None:
            self.send_move_response(
                self.client_socket,
                command_id=self.current_motion_id,
                status=MoveStatus.kPreempted,
            )
        self.current_motion_id = 0
        return True

    def _refuse_move_while_latched(self, client_socket, header: MessageHeader) -> bool:
        """Reject a ``Move`` that arrives while a reflex is still latched.

        The real robot does not start a motion out of ``kReflex``: the latched
        error has to be cleared with ``AutomaticErrorRecovery`` first. libfranka
        spells the refusal out -- ``Move::Status::kCommandNotPossibleRejected``
        becomes "Move command rejected: command not possible in the current mode
        (<mode>)!" (``src/robot_impl.h``, ``handleCommandResponse<Move>`` via
        ``commandNotPossibleMsg``), which is exactly this situation and the only
        Move status that says it.

        Accepting it instead left the state lying: ``robot_mode`` flipped back
        to ``kMove`` while ``errors`` still carried the violation, so a client
        reading the state saw a running motion and a latched fault at once.

        Called with ``_motion_lock`` held. Returns True when the Move was
        refused (and answered).
        """
        latched = (
            self.comm.violated
            or self.motion_limits.violated
            or self.robot_state.state["robot_mode"] == RobotMode.kReflex
        )
        if not latched:
            return False
        logger.warning(
            "Refusing Move %s: a reflex is still latched -- "
            "AutomaticErrorRecovery has to clear it first",
            header.command_id,
        )
        self.send_move_response(
            client_socket,
            command_id=header.command_id,
            status=MoveStatus.kCommandNotPossibleRejected,
        )
        return True

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
            self._send_tcp(client_socket, message)
            logger.info(f"Sent Move response: command_id={command_id}, status={status.name}")
        except Exception as e:
            logger.error(f"Error sending Move response: {e}", exc_info=True)

    def _engage_idle_hold(self, reason: str, motion_id: Optional[int] = None) -> None:
        """Recapture the robot when a control session ends, however it ended.

        The real FR3 hands the joints back to its own internal controller the
        instant external control stops -- normal motion completion, StopMove, a
        client that dies mid-stream, a socket error, error recovery. The sim has
        no such controller, so without this the simulator is simply left in
        whatever mode the dead session set: TORQUE, still applying that
        session's last ``tau_J_d`` (or nothing at all). Gravity is compensated
        and the FR3 model's authored viscous damping is small, so an arm
        carrying residual velocity then swings on like a frictionless pendulum
        in zero-g -- the "robot goes bananas after Ctrl-C" bug.

        Idempotent within a session (the latch is rearmed by the next Move and
        by reset_state) so the 1 kHz finish burst holds once, and exception-safe
        because most callers are teardown paths that must not raise.

        Backend-agnostic on purpose: it only calls the simulator contract every
        backend implements, so single-arm MuJoCo/Genesis and both mobile-duo
        scenes (through SceneView) are covered by this one implementation.
        """
        # However the session ended, no control loop is running any more: stop
        # charging cycles to the client and stop reporting a success rate.
        # Unconditional, ahead of the latches below, because the paths that
        # return early here (already held, or nothing ever started) have still
        # ended the motion.
        #
        # ``motion_id`` names the motion the caller believes is ending; passing
        # one makes both calls no-ops if that motion has already been replaced,
        # so a stale event cannot switch the accounting off underneath a motion
        # that is still running. The teardown paths (disconnect, StopMove,
        # socket error) pass nothing, which means "whatever is running".
        self.comm.end_motion(motion_id)
        # Same reasoning for the limit checker: no motion, nothing to
        # difference. A latched violation survives, as the robot's does --
        # only AutomaticErrorRecovery clears it.
        self.motion_limits.end_motion(motion_id)
        with self._hold_lock:
            if self._idle_hold:
                return
            # No motion ever started on this connection (a bare connect, or a
            # Move whose mode combination this server does not serve): there is
            # nothing to recapture, and holding would stomp on whatever pose the
            # previous session was correctly left in.
            if self.control_mode is ControlMode.NONE:
                return
            self._idle_hold = True
            try:
                self._switch_to_hold_position()
            except Exception:
                logger.exception("Failed to engage the idle hold after %s", reason)
                return
        logger.info("Control session ended (%s): idle hold engaged", reason)

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
            # The base is stopped, so the commanded twist the client reads back
            # has to say so -- and the next motion's limit checker seeds from
            # it, so a stale twist here would make a client that correctly
            # restarts from rest look like it stepped.
            self.robot_state.state["O_dP_EE_c"] = [0.0] * 6
            self.robot_state.state["O_dP_EE_d"] = [0.0] * 6
            # Keep the simulator's mode in lockstep with the server's: without
            # this, a client that never re-Moves leaves genesis_sim's own
            # control_mode wherever it was (e.g. still mid-transition), so
            # server and simulator could disagree about mode after a hold.
            self.genesis_sim.set_control_mode(ControlMode.STEERING_DRIVE)
            self.control_mode = ControlMode.STEERING_DRIVE
            return

        logger.info("Motion finished: switching to position control and holding position")
        # Target first, mode second. The physics thread reads both without a
        # lock, so switching to POSITION before publishing the new target lets
        # a step land in between and servo towards the *previous* position
        # target (the initial pose, or the last q_c of an older motion) at
        # kp=4500 -- a lurch, which is the opposite of a hold.
        current_joint_positions = self.genesis_sim.get_robot_state()["q"]
        self.genesis_sim.update_joint_positions(current_joint_positions)
        self.genesis_sim.update_torques([0.0] * 7)
        self.genesis_sim.set_control_mode(ControlMode.POSITION)
        self.control_mode = ControlMode.POSITION
        self._publish_hold_setpoint(current_joint_positions)

    def _publish_hold_setpoint(self, joint_positions=None) -> Dict[str, Any]:
        """Report the internal controller's own setpoint in ``q_d``/``dq_d``/``tau_J_d``.

        Between motions the robot is held by its internal controller, so the
        commanded values it reports are that hold: the joint positions it is
        holding, zero velocity, zero torque. The sim used to leave the previous
        motion's last command sitting in those fields, which is wrong on the
        wire and worse as a seed -- the next motion's limit checker differences
        the client's first command against them, so a stale ``tau_J_d`` would
        make a client that correctly starts from zero look like it stepped.

        Reads the simulator when no positions are handed in, because a ``Move``
        can arrive before the state loop has published its first frame.

        Returns the setpoint as a state-shaped dict, which is what callers that
        need it as a seed should use. Re-reading ``self.robot_state.state``
        instead works on an arm role -- the publish loop no longer merges the
        backend's ``q_d``/``dq_d``/``ddq_d`` over it -- but not on the mobile
        base, where those fields are the backend's and are rewritten every
        millisecond (see :data:`COMMANDED_STATE_FIELDS`).
        """
        if self.mobile_base:
            return dict(self.robot_state.state)
        if joint_positions is None:
            try:
                joint_positions = self.genesis_sim.get_robot_state()["q"]
            except Exception:  # pragma: no cover - a backend that cannot answer
                logger.exception("Could not read the simulator to publish the hold setpoint")
                return dict(self.robot_state.state)
        setpoint = {
            "q": list(joint_positions),
            "q_d": list(joint_positions),
            "dq_d": [0.0] * 7,
            "ddq_d": [0.0] * 7,
            "tau_J_d": [0.0] * 7,
        }
        self.robot_state.state.update(setpoint)
        return setpoint

    def _account_for_communication_cycle(self) -> None:
        """Run one cycle of the FCI communication-constraints emulation.

        Called from the state-publish thread, once per cycle, just before the
        state goes out:

        1. close the cycle that the previously published state opened,
        2. publish the rolling ``control_command_success_rate``,
        3. tell the limit checker that a cycle went unanswered,
        4. abort the motion once too many cycles are lost in a row.

        What it deliberately does *not* do is command anything. The real FCI
        extrapolates a missed motion-generator cycle under constant
        acceleration; this sim holds the last applied command instead -- nothing
        is dispatched, so the simulator keeps servoing to the last target it was
        given, the last commanded velocity or twist stays applied and the last
        torque is held. That is the FCI's own behaviour for a dropped
        *controller* packet ("FCI will reuse the torques of the last successful
        received packet", ``docs/system_requirements.rst``) applied to every
        signal; the motion-generator extrapolation is a roadmap item, documented
        as a known divergence in ``docs/robot-state.md``.

        Kept out of the transmission loop's body so the loop stays readable and
        this stays testable on its own.

        The few microseconds between this call and the ``sendto`` are charged to
        the client: a command landing in them is late by the tracker's reckoning
        even though the state it was racing is not out yet. That bias is under
        one cycle and always in the conservative direction; see
        :meth:`CommConstraintTracker.command_received` for why crediting it back
        cannot be done without a wall clock.
        """
        outcome = self.comm.tick(self.robot_state.state["message_id"])

        # Zero when no control or motion generator loop is running, which is
        # what the real robot reports: control_command_success_rate "shows a
        # value of zero if no control or motion generator loop is currently
        # running" (libfranka include/franka/robot_state.h).
        self.robot_state.state["control_command_success_rate"] = outcome.success_rate

        if not outcome.active:
            return

        if outcome.violation_triggered:
            self._abort_on_communication_violation(outcome.motion_id)

    def _accept_within_motion_limits(self, command, *, fresh: bool = True) -> bool:
        """Validate one received command; False if enforcement rejected it.

        Always checks and always reports -- a violation logs a rate-limited
        warning naming the joint or axis, the value and the limit, once per
        error class per motion. Only with ``--enforce-motion-limits`` does it
        additionally abort the motion and refuse the command, which is what the
        real robot does with a signal it cannot follow.

        ``fresh`` is False for a datagram that did not answer the cycle it
        arrived in. It is still checked in full -- everything that reaches
        physics is -- but over a single cycle rather than over the interval its
        own echoed id claims, which is the strictest reading available. See
        :meth:`franka_sim.motion_limits.MotionLimitChecker.check`.
        """
        motion_id = self.motion_limits.motion_id
        violation = self.motion_limits.check(command, self.robot_state.state["q"], fresh=fresh)
        if violation is None:
            return True

        self.motion_limits.report(violation, enforced=self.enforce_motion_limits)
        if not self.enforce_motion_limits:
            # Reported, but applied: the sim stays the permissive channel it
            # has always been unless asked to be the robot. A *fatal* violation
            # (a non-finite command) is the exception -- applying NaN is not
            # permissiveness, it poisons the physics state, the backward
            # differences and the wire. It is dropped, but it does not abort:
            # aborting is what the switch is for.
            return not violation.fatal
        if not self.motion_limits.violated:
            # First violation of this motion: latch before aborting, so a burst
            # of bad commands aborts once.
            self.motion_limits.latch()
            self._abort_with_error(violation.error_index, violation.describe(), motion_id)
        return False

    def _abort_on_communication_violation(self, motion_id: int) -> None:
        """Stop the motion with ``communication_constraints_violation``."""
        self._abort_with_error(
            COMMUNICATION_CONSTRAINTS_VIOLATION_INDEX,
            f"communication_constraints_violation: {self.comm.consecutive_lost} "
            "consecutive lost command cycles",
            motion_id,
        )

    def _abort_with_error(self, error_index: int, reason: str, motion_id: int = 0) -> None:
        """Stop the motion with one latched error, the way the robot does.

        Shared by every reflex the sim emulates -- the communication-constraints
        violation and each of the motion-limit violations -- because a client
        observes all of them identically: the error is latched in the state
        (``errors`` -> ``current_errors``, ``reflex_reason`` ->
        ``last_motion_errors``; see ``convertRobotState`` in
        ``src/robot_impl.cpp``), the modes leave ``kMove``, and the pending
        ``Move`` is answered with ``kReflexAborted`` -- the status libfranka
        turns into ``CommandException`` -> ``ControlException`` carrying
        ``last_motion_errors`` (``robot_impl.h``, ``handleCommandResponse``).

        libfranka spots the abort in the state first (``throwOnMotionError``
        keys off ``robot_mode != kMove``) and only then blocks for the TCP
        response, so the state carrying the error has to be on the wire before
        the response is. Latching it into ``robot_state`` is not enough: the
        publish loop latches, *then* packs and sends, so a response sent from
        here would beat the state it belongs to out of the machine by however
        long the packing takes.

        The response is therefore queued in :attr:`_pending_move_response`,
        stamped with :attr:`_states_packed` as it stands at this instant, and
        released only once a state packed *after* that has actually gone out.
        The stamp is what makes the guarantee hold for an abort raised on the
        UDP receive thread as well as one raised on the publish thread: a
        motion-limit violation can latch in the microseconds between the publish
        loop's ``pack_state()`` and its ``sendto()``, in which case the packet
        already on its way was serialised *before* the error existed and cannot
        be the one that carries it. Counting packs rather than sends is
        deliberate -- the send is what the client sees, but the pack is when the
        error either made it into the bytes or did not.

        When no publish loop is running to flush it (teardown, a unit test
        driving this directly) it goes out immediately instead -- late is better
        than never, and there is no state to race.

        ``error_index`` is the 0-based position of the error in the 41-entry
        wire arrays, i.e. its ``research_interface::robot::Error`` enumerator.

        ``motion_id`` is the session token of the motion that violated (see
        :attr:`_motion_generation`). The three threads that start, finish and
        abort motions all reach for the same state, so an abort raised on the
        publish thread can arrive after the TCP thread has already started the
        *next* motion; aborting then would kill an innocent motion and answer
        the wrong ``Move`` with ``kReflexAborted``. A token of 0 means "whatever
        is running", for the callers that have no motion to name.
        """
        with self._motion_lock:
            if motion_id and motion_id != self._motion_generation:
                logger.info(
                    "Not aborting: %s belonged to a motion that is already over", reason
                )
                return

            logger.error("%s -- aborting the motion", reason)
            state = self.robot_state.state
            state["errors"][error_index] = True
            state["reflex_reason"][error_index] = True
            state["robot_mode"] = RobotMode.kReflex
            state["motion_generator_mode"] = LibfrankaMotionGeneratorMode.kIdle.value
            state["controller_mode"] = LibfrankaControllerMode.kOther.value

            # Hands the joints back to the internal controller, exactly as every
            # other way a session can end does.
            self._engage_idle_hold(reason, motion_id=motion_id or None)

            if self.current_motion_id and self.client_socket is not None:
                self._pending_move_response = (
                    self.current_motion_id,
                    MoveStatus.kReflexAborted,
                    self._states_packed,
                )
                self.current_motion_id = 0
        if not self.transmitting_state:
            self._flush_pending_move_response(force=True)

    def _flush_pending_move_response(self, *, force: bool = False) -> None:
        """Send the ``Move`` response an abort deferred, if one is owed and due.

        Called by the publish loop straight after each ``sendto``. The response
        is due once :attr:`_states_packed` has moved past the value stamped when
        the error was latched, i.e. once a state serialised *after* the error
        existed has gone out; until then this is a no-op and the next cycle
        tries again.

        ``force`` skips that wait, for the teardown paths: no further state will
        ever be published, so holding the response back would leave the client
        blocked on a TCP reply for ever.
        """
        with self._motion_lock:
            pending = self._pending_move_response
            if pending is None:
                return
            motion_id, status, latched_after = pending
            if not force and self._states_packed <= latched_after:
                # The packet that just went out was serialised before the error
                # was latched, so it is not the state that carries it.
                return
            if self.client_socket is None:
                # The connection is gone; there is nobody left to answer. Drop
                # it explicitly rather than leaving it queued for a socket that
                # will never come back.
                logger.debug(
                    "Dropping the deferred Move response for motion %s: no client socket",
                    motion_id,
                )
                self._pending_move_response = None
                return
            self._pending_move_response = None
            client_socket = self.client_socket
        self.send_move_response(client_socket, command_id=motion_id, status=status)

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

            self._send_tcp(client_socket, header_bytes + response_data)
            logger.info("Sent StopMove success response")

            self._engage_idle_hold("StopMove")

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
            with self._motion_lock:
                if self.current_motion_id:
                    # Create a Move response header
                    move_response_header = MessageHeader(Command.kMove, self.current_motion_id, 16)
                    move_header_bytes = move_response_header.to_bytes()
                    move_response_data = struct.pack("<B3x", MoveStatus.kSuccess.value)
                    self._send_tcp(client_socket, move_header_bytes + move_response_data)
                    logger.info(
                        "Sent Move success response for motion ID: %s", self.current_motion_id
                    )
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
            self._send_tcp(client_socket, header_bytes + response_data)

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

            self._send_tcp(client_socket, header_bytes + response_data)
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
            self._send_tcp(client_socket, header_bytes + response_data)

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

            self._send_tcp(client_socket, header_bytes + response_data)
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
            self._send_tcp(client_socket, header_bytes + response_data)

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

            self._send_tcp(client_socket, header_bytes + response_data)
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
            self._send_tcp(client_socket, header_bytes + response_data)

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

            self._send_tcp(client_socket, header_bytes + response_data)
            logger.info("Sent SetGuidingMode success response")

        except Exception as e:
            logger.error(f"Error handling SetGuidingMode command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(Command.kSetGuidingMode, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            self._send_tcp(client_socket, header_bytes + response_data)

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

            self._send_tcp(client_socket, header_bytes + response_data)
            logger.info("Sent SetEEToK success response")

        except Exception as e:
            logger.error(f"Error handling SetEEToK command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(Command.kSetEEToK, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            self._send_tcp(client_socket, header_bytes + response_data)

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

            self._send_tcp(client_socket, header_bytes + response_data)
            logger.info("Sent SetNEToEE success response")

        except Exception as e:
            logger.error(f"Error handling SetNEToEE command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(Command.kSetNEToEE, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            self._send_tcp(client_socket, header_bytes + response_data)

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

            self._send_tcp(client_socket, header_bytes + response_data)
            logger.info("Sent SetLoad success response")

        except Exception as e:
            logger.error(f"Error handling SetLoad command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(Command.kSetLoad, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            self._send_tcp(client_socket, header_bytes + response_data)

    def handle_automatic_error_recovery_command(
        self, client_socket, header: MessageHeader, payload: bytes
    ):
        """Handle AutomaticErrorRecovery command received over TCP.

        The real robot clears any latched reflex/error and returns to Idle.
        Without a response here, libfranka (and franka_hardware, which calls
        automaticErrorRecovery() on activation) blocks forever waiting on the
        TCP reply, stalling the whole control stack.

        ``current_errors`` (the wire's ``errors`` array) is cleared;
        ``last_motion_errors`` (the wire's ``reflex_reason``) is not. libfranka
        documents the latter as "the errors that aborted the previous motion",
        so it is a record, not a live condition, and recovery must not erase
        it -- ``createControlException`` reads it *after* the abort.
        """
        try:
            # AutomaticErrorRecovery has an empty request; nothing to parse.
            # Clear any error/reflex state and return the arm to Idle. On the
            # real robot recovery aborts whatever motion latched the reflex and
            # leaves the internal controller holding, so recapture here too --
            # the next Move releases the hold.
            self.robot_state.state["errors"] = [False] * 41
            self.comm.recover()
            self.motion_limits.recover()
            self.robot_state.state["robot_mode"] = RobotMode.kIdle
            self._engage_idle_hold("automatic error recovery")

            # Response is ResponseBase: a single uint8 status (kSuccess = 0).
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(
                Command.kAutomaticErrorRecovery, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            self._send_tcp(client_socket, header_bytes + response_data)
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
            self._send_tcp(client_socket, header_bytes + response_data)

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

                    self.robot_state.state.update(
                        {
                            field: value
                            for field, value in sim_state.items()
                            if field not in self._server_owned_state_fields
                        }
                    )

                    # Close the cycle the previous state opened and open this
                    # one. Done immediately before the send, so "in time for
                    # this cycle" means "arrived before the next state went
                    # out" -- cycle space, not wall clock.
                    self._account_for_communication_cycle()

                    # Pack and send current robot state. The counter advances
                    # first: anything latched into the state from another thread
                    # after this point missed this packet, which is what
                    # _flush_pending_move_response keys off.
                    self._states_packed += 1
                    state = self.robot_state.pack_state()
                    if self.udp_socket and not self.udp_socket._closed:
                        self.udp_socket.sendto(state, (client_address, client_udp_port))
                        # Inside the guard: a state that was never sent cannot
                        # be the one a deferred kReflexAborted is waiting to
                        # follow.
                        self._flush_pending_move_response()

                    # After first state is sent, send a Move success response
                    if not first_state_sent:
                        with self._motion_lock:
                            motion_id = self.current_motion_id
                        if motion_id:
                            self.send_move_response(
                                self.client_socket,
                                command_id=motion_id,
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
            # Nothing will publish another state, so anything an abort deferred
            # has to go out now or the client blocks on a response forever.
            try:
                self._flush_pending_move_response(force=True)
            except Exception:  # pragma: no cover - the socket is already gone
                logger.debug("Could not flush the deferred Move response", exc_info=True)
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
