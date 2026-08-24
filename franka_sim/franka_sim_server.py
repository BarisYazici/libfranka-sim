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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

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
from franka_sim.motion_limits import (
    SINGULAR_POSE_MIN_SINGULAR_VALUE,
    MotionLimitChecker,
)
from franka_sim.motion_limits import (
    enforcement_enabled_by_env as motion_limit_enforcement_enabled_by_env,
)
from franka_sim.motion_limits import (
    smallest_singular_value,
    transform_matrix,
)
from franka_sim.robot_state import RobotState

# A library module must not configure the root logger -- that's the
# embedding application's call (see run_server.main()'s guarded
# basicConfig). A module-level basicConfig() here installs a root handler
# on import, which "wins" the first-call race and silently caps every
# other logger (including run_server's own) at the level given here.
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
#:
#: ``O_T_EE_d``/``O_T_EE_c``, ``O_dP_EE_d``/``O_dP_EE_c`` and
#: ``elbow_d``/``elbow_c`` joined this set once the Cartesian pose, twist and
#: elbow interfaces became FCI-owned fields in their own right (see
#: :meth:`FrankaSimServer._echo_commanded_cartesian`,
#: :meth:`FrankaSimServer._publish_commanded_pose` and
#: :meth:`FrankaSimServer._publish_elbow`), for the identical reason the joint
#: fields are here: none of the physics backends' ``get_robot_state()``
#: snapshots actually carry these keys today, so listing them changes nothing
#: yet, but it keeps this set an honest statement of FCI ownership rather than
#: one that happens to be accidentally correct only because no backend has
#: caught up. (The mobile base takes none of this set -- there the swerve
#: backend really is the only source for ``q_d``/``dq_d``/``ddq_d``, and the
#: base's own twist echo is :meth:`FrankaSimServer._handle_cartesian_velocity`.)
COMMANDED_STATE_FIELDS = (
    "q_d",
    "dq_d",
    "ddq_d",
    "tau_J_d",
    "O_T_EE_d",
    "O_T_EE_c",
    "O_dP_EE_d",
    "O_dP_EE_c",
    "elbow_d",
    "elbow_c",
)

#: Keys a physics backend publishes in its snapshot that are *not* RobotState
#: wire fields: internal readings the FCI layer consumes and never broadcasts.
#: Filtered out of the publish-path state update so ``robot_state.state`` stays
#: exactly the set of fields ``pack_state`` knows about, on every role including
#: the mobile base.
#:
#: ``O_dP_EE`` is the measured end-effector translational velocity the safety
#: controller's Cartesian check reads (see
#: :meth:`FrankaSimServer._run_safety_velocity_check`). libfranka's
#: ``RobotState`` has no measured Cartesian velocity -- only the commanded
#: ``O_dP_EE_d``/``O_dP_EE_c`` -- so there is nowhere on the wire for it to go.
#:
#: ``O_J_EE`` is the 6x7 EE Jacobian a ``Move`` conditions its start pose on
#: (:meth:`FrankaSimServer._refuse_move_at_singular_pose`). libfranka computes
#: its own from the model it downloads and the FCI never sends one, so it has no
#: wire field either -- and unlike the rest of the snapshot it is a 2-D
#: ``ndarray``, which the state dict's consumers (``pack_state``, the
#: state-shaped copies :meth:`FrankaSimServer._publish_hold_setpoint` hands out)
#: have no idea what to do with.
INTERNAL_SIM_STATE_FIELDS = ("O_dP_EE", "O_J_EE")

#: How long :meth:`FrankaSimServer.stop` waits for the gripper server's accept
#: loop to notice its socket was closed. The wait is normally microseconds --
#: closing the listening socket drops accept() out immediately -- so this only
#: bounds the case of a gripper command stuck inside a backend call. Short,
#: because the thread is a daemon and abandoning it costs nothing.
GRIPPER_JOIN_TIMEOUT_S = 1.0

#: How close to zero |measured dq| must be, per joint, for
#: ``FrankaSimServer._wait_for_standstill`` to count a cycle as settled.
#: The real robot's own ``AutomaticErrorRecovery`` inherently completes with
#: the arm at rest -- recovery on hardware clears the reflex and hands the
#: joints back to a holding controller only once the safety layer has stopped
#: reacting to the abort, and by then the arm has stopped. The sim's own
#: reflex handler used to reply the instant the TCP request arrived, so a
#: client that Move'd again straight away started its next motion while the
#: arm was still decelerating from whatever speed the abort caught it at
#: (observed here at ~2.6 rad/s); the new motion's client-side start-pose
#: guard then saw measured q drifting away from the commanded q it had just
#: sent and threw "Performance threshold reached" a few milliseconds in. This
#: is a sim choice -- libfranka publishes no settling tolerance of its own --
#: sized well inside :data:`franka_sim.motion_limits.MEASURED_JOINT_VELOCITY_MARGIN`
#: (0.1 rad/s) so the wait cannot itself look like "still moving" once the
#: idle hold has actually caught the arm.
AUTOMATIC_ERROR_RECOVERY_SETTLE_VELOCITY = 0.005  # rad/s

#: Consecutive settled polls required before the wait is satisfied, at
#: :data:`AUTOMATIC_ERROR_RECOVERY_POLL_PERIOD` apiece -- 50 x 1 ms = 50 ms.
#: "Sustained" rather than "one clean sample" so a single lucky reading
#: between two ring oscillations of a not-yet-caught arm cannot end the wait
#: early.
AUTOMATIC_ERROR_RECOVERY_SETTLE_CYCLES = 50

#: How often ``_wait_for_standstill`` re-reads the latest published ``dq``
#: while waiting. Short on purpose: this runs on the per-connection TCP
#: thread, not the state-publish loop, so a short poll costs that thread
#: nothing else is waiting on, and it is what lets the wait notice
#: :attr:`FrankaSimServer.running` going false promptly during shutdown
#: instead of sleeping through it.
AUTOMATIC_ERROR_RECOVERY_POLL_PERIOD = 0.001  # s

#: Hard ceiling on the wait -- capped well under libfranka's own TCP receive
#: timeout, not the "3 s" figure this constant started at.
#:
#: **Not the sim's choice to make generous.** libfranka's ``Network``
#: defaults ``tcp_timeout`` to ``std::chrono::seconds(1)``
#: (``libfranka/src/network.h``: ``Network(..., std::chrono::milliseconds
#: tcp_timeout = std::chrono::seconds(1), ...)``), and that is the client's
#: own receive timeout on *every* TCP command response, ``AutomaticErrorRecovery``
#: included -- there is no per-command override for it. A 3 s wait was tried
#: against the real gtest smoke suite and reproduced exactly this: the reply
#: arrived correctly, but ~1.000 s after the request the client had already
#: decided the connection was dead (Poco's ``TimeoutException`` surfaces to
#: the caller as "libfranka: TCP connection got interrupted" /
#: "libfranka: UDP receive: Timeout"), and every later test on that connection
#: then failed too as the client kept retrying against a server still mid-wait
#: from the *previous* abort. Going over 1 s here is not "slower, but still
#: correct" -- it silently breaks the wire protocol.
#:
#: 0.7 s leaves ~300 ms of margin under that 1 s ceiling for scheduling
#: jitter and the response's own transmission, while still being comfortably
#: above the 50 ms the settle-cycle count needs for a motion that is actually
#: decelerating under the idle hold. On a run where the arm has not settled
#: by then, the timeout path below fires and replies success anyway -- a
#: partial wait that reduces the residual velocity the next motion starts
#: against, which is strictly better than the pre-fix instant reply, even
#: when it cannot certify full standstill within budget.
AUTOMATIC_ERROR_RECOVERY_TIMEOUT = 0.7  # s

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

#: Closest two consecutive ``RobotState`` datagrams may be published, as a
#: fraction of the 1 ms cycle. The FCI publishes one state per cycle and expects
#: one answer per state; two states microseconds apart give the client no cycle
#: to answer the first in, and the second then carries a ``q_d`` that predates
#: that answer -- which libfranka's own low-pass filter closes around. See the
#: pacing arithmetic in ``FrankaSimServer.start_state_transmission``.
_MIN_STATE_SPACING = 0.8


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
        #: Snapshot keys the publish path must not copy into ``robot_state``:
        #: the commanded fields the FCI layer owns (none on the mobile base; see
        #: :data:`COMMANDED_STATE_FIELDS`) plus the backend readings that are
        #: not wire fields at all (:data:`INTERNAL_SIM_STATE_FIELDS`, on every
        #: role).
        self._server_owned_state_fields = INTERNAL_SIM_STATE_FIELDS + (
            () if mobile_base else COMMANDED_STATE_FIELDS
        )
        # Latch so the mobile "motion finished" hold log fires once per
        # transition, not once per datagram, for callers that drive
        # _switch_to_hold_position() directly rather than through the
        # once-per-session _engage_idle_hold(). Mirrors
        # SwerveBase._twist_rejected.
        self._mobile_hold_logged = False
        # Latch so the "no O_T_EE available to seed a motion" warning in
        # _motion_limit_seed_state fires once per connection, not once per
        # Move -- a client that never lets the publish loop get ahead of it
        # would otherwise log every single motion start.
        self._seed_pose_fallback_logged = False

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
        #: State datagrams this connection's publish loop has actually put on
        #: the wire. Rises monotonically within a connection and is zeroed by
        #: :meth:`reset_state` when the next one starts. Written only by the
        #: publish thread, and an ``int`` rebind is atomic under the GIL, so
        #: readers need no lock. It exists so a caller can wait on the *event*
        #: "this session is broadcasting" instead of inferring it from a side
        #: effect: :attr:`udp_socket` is never explicitly bound (see
        #: :meth:`start_robot_state_transmission`), so its ``getsockname()``
        #: port stays 0 until the first ``sendto`` implicitly binds it, and that
        #: is what waiters used to key off.
        self.states_sent = 0
        #: ``message_id`` that was current when the running motion's ``Move``
        #: was accepted, and whether any control command of it has arrived yet.
        #: Together they identify a ``*_finished`` datagram left over from the
        #: previous motion; see :meth:`_finish_motion`.
        self._motion_epoch_id = 0
        self._motion_has_commands = False

        # Shutdown bookkeeping. stop() can be reached from several directions
        # at once -- the KeyboardInterrupt handler, a second Ctrl+C landing
        # inside the first one's teardown, an embedding application's own exit
        # hook -- and the accept loop's finally clause calls cleanup()
        # concurrently once its listening socket goes away. Both must therefore
        # be idempotent, and both key off these: the first caller does the work
        # and logs it, later ones return early (stop) or stay quiet (cleanup).
        self._stop_lock = threading.Lock()
        self._stopping = False
        self._cleanup_logged = False

        # Communication-constraints emulation (see franka_sim.comm_constraints).
        # One tracker per connection, rebuilt by reset_state(). A missed
        # motion-generator cycle is extrapolated (see
        # _extrapolate_missed_cycle); a missed torque cycle holds, as on
        # hardware.
        self.enforce_comm_constraints = (
            enforcement_enabled_by_env()
            if enforce_comm_constraints is None
            else enforce_comm_constraints
        )
        self.comm = CommConstraintTracker(enforce=self.enforce_comm_constraints)
        #: Readiness poller for :meth:`_drain_gate`, rebuilt whenever the UDP
        #: socket is replaced (a reconnect hands out a new fd).
        self._drain_poller: Optional["select.poll"] = None
        self._drain_poller_fd: int = -1

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

        #: Whether the physics backend can turn a Cartesian command into joint
        #: motion, i.e. whether it implements the differential-IK half of the
        #: simulator contract (:meth:`franka_sim.mujoco_franka_sim.
        #: MujocoFrankaSim.update_cartesian_pose`). Backends that do not --
        #: Genesis, the mobile-duo scene view -- keep the older behaviour on the
        #: two Cartesian interfaces: the commanded stream is checked exactly as
        #: before, but nothing drives the arm from it. Asked once, here, rather
        #: than at 1 kHz on the dispatch path.
        #:
        #: Never set on the ``mobile_base`` role even when the backend would
        #: support it: the base's ``kCartesianVelocity`` commands a *base twist*
        #: for the swerve kinematics, not an end-effector twist, and it has its
        #: own branch (:meth:`_handle_cartesian_velocity`).
        self.cartesian_tracking = not mobile_base and hasattr(
            self.genesis_sim, "update_cartesian_pose"
        )

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
        self.states_sent = 0
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
        self._seed_pose_fallback_logged = False
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
        # The fresh RobotState puts F_T_NE/NE_T_EE back to identity, so the
        # backend has to be told the EE frame moved back to the flange too --
        # otherwise a tool set by the previous connection would keep skewing
        # this one's measured EE velocity.
        self._refresh_ee_transform()

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
            # Capture the socket *now*, at thread-start time, rather than
            # letting the thread re-read self.udp_socket on every loop turn.
            # self.udp_socket is per-server-instance, not per-thread: a fast
            # reconnect swaps it out for a new session while this thread's
            # own poll loop may still be unwinding on the old fd. Passing the
            # socket in as an argument gives the thread a fixed identity to
            # compare against, so it can tell "my socket is dead" apart from
            # "a newer session's socket is dead" -- see _handle_commands.
            sock = self.udp_socket
            if sock is None:
                logger.error("start_command_receiver called with no udp_socket set")
                return
            self.command_thread = threading.Thread(target=self._handle_commands, args=(sock,))
            self.command_thread.daemon = True
            self.command_thread.start()

        except Exception as e:
            logger.error(f"Error starting command receiver: {e}", exc_info=True)

    def _handle_commands(self, udp_socket):
        """Handle incoming UDP robot commands.

        ``udp_socket`` is the specific socket this thread was started for
        (see ``start_command_receiver``), captured once and never re-read
        from ``self.udp_socket``. That distinction matters on a fast
        reconnect: ``self.connection_running`` is a single flag on the
        server instance, shared by whichever session is current, not scoped
        to this thread. If a stale thread -- still polling the *old*,
        closed socket -- cleared ``connection_running`` unconditionally on
        hangup, it would kill the flag out from under the *new* session
        that has since replaced it, and that new session's broadcast loop
        would exit before sending a single state datagram. The client would
        then see "libfranka: UDP receive: Timeout" on a session that never
        did anything wrong. So every place below that would act on
        ``connection_running`` first checks ``udp_socket is self.udp_socket``
        -- i.e. that this thread is still the live session's thread.
        """
        logger.info("Command handler thread started")

        try:
            logger.info("Starting UDP command polling")
            # Setup poll object for UDP socket
            poller = select.poll()
            logger.debug(f"Command socket file descriptor: {udp_socket.fileno()}")
            poller.register(udp_socket.fileno(), select.POLLIN)
            logger.debug(f"Poller: {poller}")
            timeout = 1  # 1ms timeout

            # RobotCommand packet size (matches the client's RobotCommand struct).
            expected_size = 8 + (7 * 8 + 7 * 8 + 16 * 8 + 6 * 8 + 2 * 8 + 1 + 1) + (7 * 8 + 1)

            # Bound this thread to the current connection (connection_running)
            # *and* to its own socket's identity (udp_socket is self.udp_socket):
            # once a reconnect swaps self.udp_socket out, this loop must not
            # keep spinning on the fd that belongs to a session that no longer
            # exists -- see the identity-invariant note in the docstring above.
            while self.running and self.connection_running and udp_socket is self.udp_socket:
                events = poller.poll(timeout)
                if not events:
                    continue

                command = None
                for fd, event in events:
                    if not (event & select.POLLIN):
                        # Socket hung up / errored -> the connection is gone
                        # -- but only clear connection_running if this is
                        # still the live session's socket. A stale thread's
                        # own (now-closed) fd reporting POLLHUP/POLLNVAL says
                        # nothing about whatever session replaced it.
                        if udp_socket is self.udp_socket:
                            self.connection_running = False
                        return

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
                    # Rewind, check and record are **one** operation on the
                    # checker, taken under a single hold of its lock. A datagram
                    # that missed its cycle may still be the real answer to a
                    # cycle the publish loop has already extrapolated; the robot
                    # would have dropped it, this sim applies it, so the guess it
                    # replaces has to be thrown away first or the two are
                    # differenced against each other and report a deceleration
                    # nobody commanded. Doing that in three separate calls left
                    # two windows for the publish thread's own extrapolate() to
                    # land in, and either one re-created the false abort the
                    # rewind exists to prevent. See
                    # MotionLimitChecker.absorb_command -- which also explains
                    # why a *fresh* command is not rewound, and why a replay gets
                    # no rewind: it is applied, because it is still the freshest
                    # intent there is, but it is not a later sample of the
                    # client's trajectory and never becomes the baseline the next
                    # one is differenced against.
                    #
                    # Nothing gets past this on its way to physics, whether or
                    # not it was fresh: an unchecked path here was a way to walk
                    # a 40 rad/s step into the simulator behind a stale echo.
                    if not self._absorb_within_motion_limits(command, fresh=fresh):
                        continue

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

    def _dispatch_control_command(self, command, *, motion_generation=None) -> None:
        """Route one UDP RobotCommand to the simulator, unless the hold is on.

        Split out of :meth:`_handle_commands` so the whole simulator-facing part
        of a control cycle sits inside ``_hold_lock``: a datagram that was
        already in flight when the session ended must not be applied *after*
        :meth:`_engage_idle_hold` recaptured the arm, or the hold would be
        immediately overwritten by a dead session's last torque.

        ``motion_generation`` is the motion this command was built for, for a
        caller that built it a moment ago rather than receiving it: the
        extrapolation path (:meth:`_extrapolate_missed_cycle`). The branches
        below dispatch on ``motion_generator_mode``/``controller_mode``, which a
        ``Move`` accepted in the meantime has already changed, so a substitute
        built under the old motion's generator must not be applied under the new
        one's. Checked here rather than at the call site because here is inside
        the same ``_hold_lock`` the dispatch itself runs in, and the ``Move``
        path takes that lock too. A received datagram passes None: it was
        checked and recorded against whatever motion was running when it
        arrived, and the pre-existing idle-hold gate is what covers it.
        """
        with self._hold_lock:
            if self._idle_hold:
                return
            if motion_generation is not None and motion_generation != self._motion_generation:
                # The motion this substitute belongs to is over; a new one owns
                # the generator fields now.
                return

            # The commanded *Cartesian* fields, before the generator branches
            # below and outside them: a ``kCartesianPosition`` motion can run
            # under either controller (``kJointImpedance`` or
            # ``kExternalController``), so only the motion-generator mode says
            # whether the client is streaming a pose at all -- and none of the
            # branches below is reached for the kJointImpedance case.
            self._echo_commanded_cartesian(command)

            # Update Genesis simulator based on control mode
            if (
                self.robot_state.state["controller_mode"]
                == LibfrankaControllerMode.kJointImpedance
                and self.robot_state.state["motion_generator_mode"]
                == LibfrankaMotionGeneratorMode.kJointPosition
            ):
                # Target first, mode second -- the same rule (and the same
                # reason) as _switch_to_hold_position. The physics thread reads
                # target and mode without a lock, so switching into POSITION
                # before publishing this command's target lets a step land in
                # between and servo towards the *previous* target at kp=4500.
                # It also matters for the velocity feedforward: entering
                # POSITION mode re-seeds its baseline from whatever target is
                # current at that instant (see PositionFeedforward.reset), so
                # doing it before the write seeds it from a dead session's last
                # q_c and turns this command into one huge dq_c on the next
                # step. Publishing first makes the baseline this very command,
                # which is exactly the no-spike invariant the reset is for.
                self.genesis_sim.update_joint_positions(command["q_c"])
                if self.control_mode is not ControlMode.POSITION:
                    logger.info("Setting control mode to POSITION")
                    self.genesis_sim.set_control_mode(ControlMode.POSITION)
                    self.control_mode = ControlMode.POSITION
                    # Initialize q_d to current q when first entering position mode
                    self.robot_state.state["q_d"] = self.robot_state.state["q"]
                # Update q_d with commanded positions
                self.robot_state.state["q_d"] = list(command["q_c"])
                self._publish_commanded_derivatives("dq_d", "ddq_d")
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
                self.cartesian_tracking
                # Any internal controller, not ``kJointImpedance`` alone: the
                # controller mode picks the *stiffness law* the robot holds the
                # generator's output with, and both internal ones are driven by
                # the same generator. Naming only one here while the ``Move``
                # handler put the backend into Cartesian tracking for either
                # left ``kCartesianImpedance`` silently inert -- in a tracking
                # mode with no command ever reaching it. ``kExternalController``
                # is the one that genuinely does not belong: there the client's
                # torques drive the arm and its pose stream is a reference, so
                # it falls through to the TORQUE branch below.
                and self.robot_state.state["controller_mode"]
                != LibfrankaControllerMode.kExternalController
                and self.robot_state.state["motion_generator_mode"]
                in (
                    LibfrankaMotionGeneratorMode.kCartesianPosition,
                    LibfrankaMotionGeneratorMode.kCartesianVelocity,
                )
            ):
                self._drive_cartesian_generator(command)
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

    def _drive_cartesian_generator(self, command) -> None:
        """Hand one accepted Cartesian command to the backend's differential IK.

        The arm's counterpart to the joint branches above, with the two writes
        in the opposite order -- mode first, target second; see the comment on
        the switch below for why the Cartesian interfaces need the mirror image
        of :meth:`_dispatch_control_command`'s POSITION rule.

        Only reached for a command this cycle's checks already accepted, and
        only when the backend can convert one at all
        (:attr:`cartesian_tracking`). Nothing about the checking layer changes:
        this is a second consumer of the accepted stream, not a filter on it.

        ``elbow_c[0]`` is passed through when -- and only when -- the client
        said it commands an elbow (``valid_elbow``); ``elbow_c`` is zero-filled
        otherwise, and steering the redundancy angle to a zero nobody asked for
        would twist the arm on every elbow-less Cartesian motion.
        """
        pose_generator = (
            self.robot_state.state["motion_generator_mode"]
            == LibfrankaMotionGeneratorMode.kCartesianPosition
        )
        mode = ControlMode.CARTESIAN_POSE if pose_generator else ControlMode.CARTESIAN_VELOCITY
        # Mode first, target second -- the *opposite* order to the POSITION
        # branch above, and for the same underlying reason. Entering a Cartesian
        # mode deliberately clears the target (see
        # ``MujocoFrankaSim.set_control_mode``), so publishing this command
        # before the switch would throw it away and leave the arm holding for a
        # cycle. Clearing first and filling immediately after cannot spike: the
        # worst a physics step landing in between sees is "no target", which
        # servos zero velocity.
        if self.control_mode is not mode:
            logger.info("Setting control mode to %s", mode.name)
            self.genesis_sim.set_control_mode(mode)
            self.control_mode = mode
        elbow_angle = command["elbow_c"][0] if command.get("valid_elbow") else None
        if pose_generator:
            self.genesis_sim.update_cartesian_pose(command["O_T_EE_c"], elbow_angle)
        else:
            self.genesis_sim.update_cartesian_velocity(command["O_dP_EE_c"], elbow_angle)
        self.genesis_sim.update_torques([0.0] * 7)

    def _echo_commanded_cartesian(self, command) -> None:
        """Echo a Cartesian generator's commanded pose and elbow into the state.

        The Cartesian half of what :meth:`_dispatch_control_command` already
        does for ``q_d``/``dq_d``: ``O_T_EE_d`` and ``O_T_EE_c`` are the FCI
        layer's fields, not the physics backend's, and on hardware they carry
        the pose the motion generator is tracking. libfranka reads *both* --
        ``O_T_EE_d`` is what a pose motion generator initialises and holds from,
        and ``O_T_EE_c`` is the reference its command low-pass filter blends the
        next command with (``src/control_loop.cpp``, ``ControlLoop<CartesianPose>
        ::convertMotion``) -- so during a ``kCartesianPosition`` motion both
        have to be the commanded stream or the client's own filter drags every
        command towards whatever the sim published instead.

        ``O_T_EE_c`` is the last pose the client commanded; ``O_T_EE_d`` is the
        one the generator is tracking, which in this sim is the same value
        because a commanded pose is applied instantly.
        A *lost* cycle reaches this method too: the publish loop extrapolates the
        pose across it and dispatches the result down this same path
        (:meth:`_extrapolate_missed_cycle`), so both fields keep advancing along
        the commanded trajectory exactly as ``q_d`` does, which is what the robot
        reports -- "the last received c values (after the low pass filter and the
        extrapolation due to packet losses)" (``docs/overview.rst``).

        **The twist generator has the identical requirement, and leaving it out
        silently scaled the whole interface.** ``ControlLoop<CartesianVelocities>
        ::convertMotion`` low-passes every commanded twist toward
        ``robot_state.O_dP_EE_c`` (``src/control_loop.cpp:296-313``), the same
        way the pose loop uses ``O_T_EE_c``. With those fields left at the wire
        struct's permanent zero on an arm role, the filter's reference was zero
        on *every* cycle rather than the client's own previous command, so
        instead of converging on the commanded twist it returned a fixed
        fraction of it: ``gain = dt / (dt + 1 / (2*pi*f_c))`` = 0.3859 at
        libfranka's default 100 Hz cutoff and 1 ms. Every ``kCartesianVelocity``
        client was moving the arm at 39% of the speed it asked for, with no way
        to see why. Echoed here, the reference is the previous command and a
        constant commanded twist converges geometrically to 1.0x.

        The elbow rides along with either Cartesian generator
        (``CartesianPose::hasElbow`` / ``CartesianVelocities::hasElbow``) and is
        echoed only when the client actually sent one -- ``valid_elbow`` is its
        own statement that it did, and ``elbow_c`` is zero-filled otherwise.

        **Arm roles only.** The mobile base commands a *base* twist rather than
        an end-effector one; its ``O_dP_EE_d``/``O_dP_EE_c`` echo and its
        dead-reckoned ``O_T_EE`` are :meth:`_handle_cartesian_velocity`'s
        business and are deliberately untouched here, the same role guard the
        rest of the commanded-field ownership uses
        (:data:`COMMANDED_STATE_FIELDS`).
        """
        if self.mobile_base:
            return
        mode = self.robot_state.state["motion_generator_mode"]
        if mode == LibfrankaMotionGeneratorMode.kCartesianPosition:
            pose = list(command["O_T_EE_c"])
            self.robot_state.state["O_T_EE_c"] = pose
            self.robot_state.state["O_T_EE_d"] = pose
        elif mode == LibfrankaMotionGeneratorMode.kCartesianVelocity:
            twist = list(command["O_dP_EE_c"])
            self.robot_state.state["O_dP_EE_c"] = twist
            self.robot_state.state["O_dP_EE_d"] = twist
        else:
            return
        if command.get("valid_elbow"):
            elbow = list(command["elbow_c"])
            self.robot_state.state["elbow_c"] = elbow
            self.robot_state.state["elbow_d"] = elbow

    def _cartesian_motion_owns_commanded_fields(self) -> bool:
        """Whether a running Cartesian generator owns the commanded-echo fields.

        ``O_T_EE_d``/``O_T_EE_c`` and ``elbow_d``/``elbow_c`` are *commanded*
        fields, and while one of the two Cartesian generators is running the
        only honest source for them is the client's own stream
        (:meth:`_echo_commanded_cartesian`). Where the stream is silent -- a
        pose motion that commands no elbow, the cycle or two before its first
        command lands, a twist motion which commands no pose at all -- they stay
        frozen at the value :meth:`_freeze_commanded_cartesian` stamped when the
        motion started, which *is* the measured value at that instant.

        **Why freezing rather than tracking the measured arm.** libfranka builds
        every Cartesian command out of these fields: they are the reference its
        command low-pass filter blends the next command with, and its rate
        limiter differences against (``ControlLoop<CartesianPose>::
        convertMotion``, ``ControlLoop<CartesianVelocities>::convertMotion``,
        ``src/control_loop.cpp``), and the smoke suite's own generators open
        with ``franka::CartesianPose cmd{state.O_T_EE_d, state.elbow_d}``. Once
        the arm is actually driven from those commands, publishing the *measured*
        pose or elbow there closes a positive feedback loop through the client:
        the command chases the lagging arm, the arm chases the command, and the
        two wind each other up until the (correct) checking layer aborts the
        motion. Freezing breaks the loop at its only closure point without
        inventing a value the robot never commanded.

        Cartesian generators only. Under a joint generator the client references
        ``q_d``/``dq_d``/``ddq_d`` and never these fields, so there is no loop to
        break, and reporting the pose the arm is in stays the closest thing this
        sim has to the hardware's own ``O_T_EE_d = FK(q_d)``. Idle likewise --
        and that is the behaviour the suite's "start from ``state.O_T_EE_d``"
        generators depend on, since they read it on the motion's first cycle.
        """
        return self.robot_state.state["motion_generator_mode"] in (
            LibfrankaMotionGeneratorMode.kCartesianPosition,
            LibfrankaMotionGeneratorMode.kCartesianVelocity,
        )

    def _freeze_commanded_cartesian(self, seed: Dict[str, Any]) -> None:
        """Stamp the commanded-echo fields with the pose/elbow the motion starts in.

        Called once per accepted ``Move`` on an arm role, with the same snapshot
        the limit checker is seeded from. It is what makes "frozen" mean *the
        measured state at motion start* rather than "whatever the publish loop
        happened to leave behind": a ``Move`` can arrive before the publish loop
        has produced its first frame, and the identity the wire struct is
        constructed with is not a pose any generator could legally start from.

        Values only ever *read back* by the client, so a motion whose generator
        is not Cartesian is stamped too -- harmlessly, since the publish loop
        goes straight back to tracking the measured arm for it on the next
        cycle (:meth:`_cartesian_motion_owns_commanded_fields`).
        """
        if self.mobile_base:
            return
        pose = seed.get("O_T_EE")
        if pose is not None:
            pose = pose.tolist() if hasattr(pose, "tolist") else list(pose)
            self.robot_state.state["O_T_EE_d"] = pose
            self.robot_state.state["O_T_EE_c"] = list(pose)
        elbow = self._elbow_from_joints(seed.get("q"))
        if elbow is not None:
            self.robot_state.state["elbow_d"] = list(elbow)
            self.robot_state.state["elbow_c"] = list(elbow)

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

                if self._refuse_move_at_singular_pose(client_socket, header, move_cmd):
                    return

                self._preempt_running_motion()

                # A fresh session token for this motion; see _motion_generation.
                self._motion_generation += 1
                generation = self._motion_generation
                self._motion_epoch_id = self.robot_state.state["message_id"]
                self._motion_has_commands = False

                # The snapshot this motion is judged and reported from, read
                # once and used twice: to stamp the commanded Cartesian fields
                # here, and to seed the limit checker further down (it depends
                # on ``generator_mode``, which the branches below decide).
                #
                # **Fields first, mode second**, the same rule the dispatch path
                # applies to target-then-mode and for the same reason. Publishing
                # the Cartesian generator mode is what makes the publish loop
                # stop writing ``O_T_EE_d/_c`` and ``elbow_d/_c``
                # (:meth:`_cartesian_motion_owns_commanded_fields`), so stamping
                # them afterwards leaves a window for a cycle that reports
                # neither: not the measured arm, because the mode already says
                # "frozen", and not the frozen value, because it has not been
                # written yet. A ``Move`` that arrives before the publish loop's
                # first cycle -- which is exactly what a client that Moves
                # straight after connecting does -- then broadcast the wire
                # struct's zero ``elbow_d``, and libfranka's ``checkElbow``
                # refuses to send a branch flag that is not +-1, so the elbow
                # interface was unreachable for that motion.
                seed = self._motion_limit_seed_state()
                self._freeze_commanded_cartesian(seed)

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

                # The generator this motion actually runs, as opposed to
                # whatever mode the *previous* motion left in
                # ``self.control_mode``. NONE means "a mode this server accepts
                # but does not drive", which is what gates the limit checker
                # below; see the comment there.
                generator_mode = ControlMode.NONE

                # Set appropriate control mode in Genesis simulator
                if (
                    move_cmd.controller_mode == ControllerMode.kJointImpedance
                    and move_cmd.motion_generator_mode == MotionGeneratorMode.kJointPosition
                ):
                    logger.info("Setting control mode to POSITION")
                    self.genesis_sim.set_control_mode(ControlMode.POSITION)
                    self.control_mode = generator_mode = ControlMode.POSITION
                elif (
                    move_cmd.controller_mode == ControllerMode.kJointImpedance
                    and move_cmd.motion_generator_mode == MotionGeneratorMode.kJointVelocity
                ):
                    logger.info("Setting control mode to VELOCITY")
                    self.genesis_sim.set_control_mode(ControlMode.VELOCITY)
                    self.control_mode = generator_mode = ControlMode.VELOCITY
                elif (
                    self.mobile_base
                    and move_cmd.motion_generator_mode == MotionGeneratorMode.kCartesianVelocity
                    and move_cmd.controller_mode != ControllerMode.kExternalController
                ):
                    logger.info("Setting control mode to STEERING_DRIVE")
                    self.genesis_sim.set_control_mode(ControlMode.STEERING_DRIVE)
                    self.control_mode = generator_mode = ControlMode.STEERING_DRIVE
                elif move_cmd.controller_mode == ControllerMode.kExternalController:
                    logger.info("Setting control mode to TORQUE")
                    self.genesis_sim.set_control_mode(ControlMode.TORQUE)
                    self.control_mode = generator_mode = ControlMode.TORQUE
                elif move_cmd.motion_generator_mode in (
                    MotionGeneratorMode.kCartesianPosition,
                    MotionGeneratorMode.kCartesianVelocity,
                ):
                    # An *arm* role on one of the two Cartesian generators (the
                    # mobile base's own kCartesianVelocity was matched further
                    # up, and the kExternalController case one branch above:
                    # there the client's torques drive the arm and the pose
                    # stream is a reference, not a command to the joints).
                    #
                    # Driven when the backend can convert a Cartesian command
                    # into joint motion, checked-only when it cannot; see
                    # :attr:`cartesian_tracking`. Either way the commanded
                    # ``O_T_EE_c``/``O_dP_EE_c``/``elbow_c`` stream is
                    # differentiated and judged exactly as before, so a step in
                    # it aborts with the hardware error rather than being
                    # silently dropped. See
                    # :meth:`franka_sim.motion_limits.MotionLimitChecker._check_cartesian_pose`.
                    generator_mode = (
                        ControlMode.CARTESIAN_POSE
                        if move_cmd.motion_generator_mode
                        == MotionGeneratorMode.kCartesianPosition
                        else ControlMode.CARTESIAN_VELOCITY
                    )
                    if self.cartesian_tracking:
                        logger.info("Setting control mode to %s", generator_mode.name)
                        self.genesis_sim.set_control_mode(generator_mode)
                        self.control_mode = generator_mode
                    else:
                        logger.info(
                            "Move accepted for %s: this backend has no Cartesian "
                            "tracking, so the commanded stream is validated but not "
                            "applied -- the arm will not move",
                            move_cmd.motion_generator_mode.name,
                        )
                else:
                    logger.info(
                        "Move accepted for %s / %s, which this server has no physics "
                        "branch for: nothing will be dispatched and no motion-generator "
                        "signal will be checked",
                        move_cmd.motion_generator_mode.name,
                        move_cmd.controller_mode.name,
                    )

                # Seed the limit checker from the state the client is judged
                # against -- q_d / dq_d / ddq_d / tau_J_d are "always sent back to
                # the user in the robot state" for exactly this purpose (libfranka
                # docs/overview.rst), so the client can predict every derivative the
                # robot will compute. Last, because it needs the control mode the
                # branches above just decided.
                #
                # ``generator_mode``, not ``self.control_mode``: the checker must
                # judge only the signal *this* motion's generator owns. A
                # ``kCartesianPosition`` Move does not touch ``self.control_mode``
                # at all (it has no physics branch), so that field still holds
                # the previous motion's mode -- and handing *that* to the checker
                # made it read the zero-filled ``q_c`` of a Cartesian
                # ``RobotCommand`` as a joint position command and abort live
                # clients with
                # ``joint_motion_generator_position_limits_violation``, because
                # joint 4's range does not contain 0. The safety controller
                # (measured velocity -> joint_velocity_violation) is armed
                # regardless of mode; see MotionLimitChecker.start_motion.
                self.motion_limits.start_motion(generator_mode, seed, generation)

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

    def _refuse_move_at_singular_pose(
        self, client_socket, header: MessageHeader, move_cmd: MoveCommand
    ) -> bool:
        """Reject a Cartesian ``Move`` that would start from a singular configuration.

        A Cartesian motion generator has to be able to realise an arbitrary
        commanded twist, and in a singularity it cannot: some direction of EE
        motion costs unbounded joint speed. The robot refuses to start rather
        than let the client discover that at 1 kHz, and libfranka has a status
        that says exactly this --
        ``Move::Status::kStartAtSingularPoseRejected``, which it turns into
        ``CommandException("libfranka: Move command rejected: cannot start at
        singular pose!")`` (``src/robot_impl.h:423-427``).

        **The rejection is the motion's *terminal* response, not its first, and
        that is forced by libfranka's own control flow.** ``Robot::Impl::
        startMotion`` sends the ``Move`` through ``executeCommand``, which
        blocking-receives the first response and calls ``handleCommandResponse``
        *outside* any try/catch (``src/robot_impl.h:548-553``); a rejection
        delivered there escapes as a bare ``CommandException``. Only the
        responses picked up by the mode-wait loop that follows are converted --
        ``catch (const CommandException& e) { throw ControlException(e.what()); }``
        (``src/robot_impl.cpp:323-332``), and ``throwOnMotionError`` does the
        same (``:96-111``). Franka's ``CartesianMotionInSingularPose``
        (``smoke_test_errors.cpp:231``) catches ``ControlException``, so on
        hardware the singular start is discovered *after* the ``Move`` was
        acknowledged -- Control accepts the command, looks at where the arm is
        standing, and terminates the motion. This method reproduces that shape:
        ``kMotionStarted`` first, the rejection queued as the deferred terminal
        response, and the generator modes deliberately never leaving idle so the
        client stays in the loop that converts it.

        **Cartesian generators only.** The very same test reaches the singular
        configuration by *joint* motion (``moveP2P`` to ``kSingularPose``), so
        joint interfaces plainly start there quite happily -- as they must, since
        a joint generator has no Jacobian to invert and driving *out* of a
        singularity is the only way to leave one.

        The measure is ``sigma_min`` of the EE Jacobian the physics backend
        publishes for the arm's current measured ``q``; see
        :data:`franka_sim.motion_limits.SINGULAR_POSE_MIN_SINGULAR_VALUE` for
        the threshold and the poses it was placed against. A backend that
        publishes no Jacobian -- Genesis, the mobile-base bridge -- cannot be
        asked and the Move is accepted, which is also why the ``mobile_base``
        role (whose ``kCartesianVelocity`` commands a *base twist* and has no
        end effector at all) needs no special case here.

        Called with ``_motion_lock`` held. Returns True when the Move was
        refused (and answered).
        """
        if move_cmd.motion_generator_mode not in (
            MotionGeneratorMode.kCartesianPosition,
            MotionGeneratorMode.kCartesianVelocity,
        ):
            return False
        try:
            sim_state = self.genesis_sim.get_robot_state()
        except Exception:  # pragma: no cover - a backend that cannot answer
            # Same posture as :meth:`_publish_hold_setpoint`: a backend that
            # cannot be read is a backend this check cannot be run against, and
            # refusing a Move on the strength of an exception would be worse
            # than running the motion unconditioned.
            logger.exception("Could not read the simulator to condition the Move's start pose")
            return False
        jacobian = sim_state.get("O_J_EE") if isinstance(sim_state, dict) else None
        sigma_min = smallest_singular_value(jacobian)
        if sigma_min is None:
            return False
        if sigma_min > SINGULAR_POSE_MIN_SINGULAR_VALUE:
            logger.info(
                "Cartesian Move %s start-pose conditioning: sigma_min=%.4f (limit %.4f)",
                header.command_id,
                sigma_min,
                SINGULAR_POSE_MIN_SINGULAR_VALUE,
            )
            return False
        logger.warning(
            "Refusing Move %s: the arm is standing in a singularity "
            "(sigma_min=%.4f <= %.4f), so a Cartesian motion cannot start here",
            header.command_id,
            sigma_min,
            SINGULAR_POSE_MIN_SINGULAR_VALUE,
        )
        # A refused Move still displaces whatever was running, exactly as an
        # accepted one does: the client that sent it has stopped servicing the
        # old motion's control loop, and Control does not keep driving a motion
        # nobody is feeding. This also answers the displaced motion's own
        # ``Move`` (``kPreempted``) and clears ``current_motion_id``, which is
        # what leaves :attr:`_pending_move_response` free for the rejection
        # queued below -- the same sequencing the accepted path gets from
        # calling this immediately before it starts a new motion.
        self._preempt_running_motion()
        # An abort latched a moment ago on another thread can still be sitting
        # in the queue with nothing to release it yet. Flush it first rather
        # than overwrite it: it belongs to a *different* Move, and dropping it
        # leaves that client blocked on a TCP reply for ever. Forced, because
        # the state it was waiting for may never arrive -- this Move starts no
        # motion, so nothing after it will publish one on that response's
        # behalf. Safe to call with ``_motion_lock`` held: it is an ``RLock``.
        self._flush_pending_move_response(force=True)
        self.send_move_response(
            client_socket, command_id=header.command_id, status=MoveStatus.kMotionStarted
        )
        if not self.transmitting_state:
            # No publish loop to release a deferred response, and no state for
            # it to race: answer immediately rather than leave the client
            # blocked on a TCP reply for ever. Same escape hatch as
            # :meth:`_flush_pending_move_response`'s ``force``, taken inline
            # because ``_motion_lock`` is already held here.
            self.send_move_response(
                client_socket,
                command_id=header.command_id,
                status=MoveStatus.kStartAtSingularPoseRejected,
            )
            return True
        # Deferred exactly like a reflex abort: the client has to see at least
        # one state after the acknowledgement before the terminal response, or
        # it can be answered before it has even left ``executeCommand``.
        self._pending_move_response = (
            header.command_id,
            MoveStatus.kStartAtSingularPoseRejected,
            self._states_packed,
        )
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
        # The Cartesian twin of the zero ``dq_d``/``tau_J_d`` that hold setpoint
        # publishes, and the arm-role counterpart of the mobile branch above: an
        # arm held by its internal controller is commanding no end-effector
        # motion, so the twist it reports as commanded is zero. Only a
        # ``kCartesianVelocity`` motion ever writes anything else here
        # (:meth:`_echo_commanded_cartesian`), and every way such a motion can
        # end -- finished, reflex, StopMove, preemption, a client that vanishes
        # -- arrives at this method, so this is the single place the twist has
        # to be given back.
        self.robot_state.state["O_dP_EE_c"] = [0.0] * 6
        self.robot_state.state["O_dP_EE_d"] = [0.0] * 6

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

    def _motion_limit_seed_state(self) -> Dict[str, Any]:
        """The state snapshot a new motion's limit checker is seeded from.

        :meth:`_publish_hold_setpoint` republishes and returns the *commanded*
        fields -- ``q_d``, ``dq_d``, ``ddq_d``, ``tau_J_d`` -- which is what the
        joint and torque generators are differenced against. The Cartesian pose
        generator needs one more thing, and it is a *measured* value rather than
        a commanded one: the flange pose ``O_T_EE`` the robot is actually in, so
        the first ``O_T_EE_c`` of a ``kCartesianPosition`` motion can be judged
        against it (``cartesian_position_motion_generator_start_pose_invalid``).

        ``O_T_EE_d`` is what hardware would be judged against, and this sim now
        publishes it faithfully (:meth:`_publish_commanded_pose`) -- but between
        motions it *is* the measured pose, so the two agree and the measured one
        is used directly. It is also the more honest of the two to seed from:
        the question the check asks is "are you where the robot is", which is
        what the smoke suite's 10 m offset breaks, and answering it from a
        commanded field would let a stale command excuse a stale command.

        Read out of the backend rather than ``self.robot_state.state`` for arm
        roles, for the same reason :meth:`_publish_hold_setpoint` reads the
        backend for ``q``: ``self.robot_state.state["O_T_EE"]`` is still the
        identity the wire struct was constructed with until the publish loop's
        first cycle has run, and a ``Move`` that arrives before then would have
        its correct first pose judged against that identity and false-aborted
        (observed repro: ``Violation`` index 16, 0.57 m from identity). The
        mobile-base branch of ``_publish_hold_setpoint`` already returns the
        whole state dict, ``O_T_EE`` included, so this only has to add it for
        arm roles -- and it adds it to the *returned* snapshot only, leaving
        what gets written back into ``robot_state.state`` exactly as it was.

        When the backend cannot answer either, ``O_T_EE`` is left out of the
        snapshot entirely rather than filled in from identity: ``start_motion``
        already treats a missing ``O_T_EE`` as "skip the start-pose check", and
        :meth:`~franka_sim.motion_limits.MotionLimitChecker._check_elbow_validity`
        skips its start-elbow half the same way, so a motion armed from a
        backend that has not produced a frame yet runs with those two start
        checks off instead of judging them against a guess.
        """
        seed = self._publish_hold_setpoint()
        if "O_T_EE" not in seed:
            seed = dict(seed)
            try:
                backend_pose = self.genesis_sim.get_robot_state().get("O_T_EE")
            except Exception:  # pragma: no cover - a backend that cannot answer
                logger.exception("Could not read the simulator to seed O_T_EE")
                backend_pose = None
            if backend_pose is None:
                if not self._seed_pose_fallback_logged:
                    logger.warning(
                        "No O_T_EE available from the simulator to seed a new "
                        "motion's limit checker; skipping the start-pose/"
                        "start-elbow checks for this motion instead of "
                        "judging them against identity"
                    )
                    self._seed_pose_fallback_logged = True
            else:
                seed["O_T_EE"] = list(backend_pose)
        return seed

    #: Longest the publish loop will hold a state back waiting for the receive
    #: path to catch up; see :meth:`_drain_gate`. Five cycles: long enough to
    #: cover the millisecond-scale hiccups that actually happen (a GIL handover,
    #: a descheduled thread under load -- 2 ms and up, measured), short enough
    #: that a receive path which is genuinely gone -- a dying connection --
    #: costs the loop a bounded slowdown rather than a hang.
    _DRAIN_GATE_TIMEOUT = 0.005
    #: How long each turn of the gate sleeps. Not a busy-wait: the point of the
    #: sleep is to *release* this thread's hold on the CPU (and the GIL) so the
    #: receive thread can run, which is usually all it was waiting for.
    _DRAIN_GATE_YIELD = 0.00005

    def _drain_gate(self) -> float:
        """Hold the state back while a command the client already sent is queued.

        The FCI is one loop: a cycle receives the client's answer, applies it,
        and publishes the state that answer produced. This server splits that
        across two threads -- the publish loop below and the UDP receive thread
        in :meth:`_handle_commands` -- and nothing kept them in step. When the
        receive thread was descheduled for a few milliseconds (CPU contention, a
        GIL handover behind a 5 ms switch interval) the publish loop sailed on,
        emitting states whose ``q_d``/``dq_d``/``ddq_d`` still described the last
        command it had managed to apply, while the client's answers to those very
        cycles sat unread in this process's own socket buffer.

        That is not a cosmetic lag, because **libfranka's control loop is closed
        around those fields**. ``ControlLoop<JointPositions>::convertMotion``
        low-pass filters every waypoint toward ``robot_state.q_d`` with a fixed
        1 ms gain (``src/control_loop.cpp``; the gain at the default 100 Hz
        cutoff is 0.386) and, with rate limiting on, clamps it against
        ``dq_d``/``ddq_d`` as well. Feed a conforming client the *same* stale
        ``q_d`` for several cycles and its commanded stream stops being smooth:
        each waypoint is dragged back toward the frozen reference, and when the
        reference finally moves the filter spends several more cycles catching
        up. The sim then differences that stream and reports the kink it
        manufactured as ``joint_motion_generator_velocity_discontinuity``, at
        hundreds of rad/s^2, against a client that did nothing wrong. That was an
        intermittent abort in the middle of an ordinary approach motion, at a
        ``control_command_success_rate`` of 0.97-0.99.

        So the gate restores the invariant the communication accounting already
        assumes -- "a simulator that stalls delays the state publish and the
        client's answer alike" (:mod:`franka_sim.comm_constraints`) -- for a
        stall that hits only one of the two threads. If a datagram is sitting in
        the socket unread, this cycle's state is not ready to go out yet, so the
        loop sleeps in short turns (which hands the CPU, and the GIL, to the
        receive thread) until the socket is empty or
        :data:`_DRAIN_GATE_TIMEOUT` is up.

        Costs nothing in the ordinary case: the receive path answers a state in
        ~70 us, so by the time the next one is due the socket has long been
        drained and the first ``poll(0)`` returns empty.

        Read-only on the socket. It never calls ``recvfrom`` -- the receive
        thread stays the only consumer -- so polling the same fd from here is
        safe.

        Returns how long it waited, so the caller can push its 1 kHz deadline
        out by the same amount. Without that the pacer treats the wait as lost
        time and fires the next state immediately behind this one, which is the
        opposite of the point: two states microseconds apart give the client no
        cycle to answer the first in.
        """
        udp_socket = self.udp_socket
        if udp_socket is None or udp_socket._closed:
            return 0.0
        try:
            fd = udp_socket.fileno()
        except OSError:
            return 0.0
        if fd < 0:
            return 0.0
        if self._drain_poller is None or self._drain_poller_fd != fd:
            poller = select.poll()
            poller.register(fd, select.POLLIN)
            self._drain_poller = poller
            self._drain_poller_fd = fd
        started = time.perf_counter()
        deadline = started + self._DRAIN_GATE_TIMEOUT
        while True:
            try:
                if not self._drain_poller.poll(0):
                    return time.perf_counter() - started
            except OSError:
                # The socket went away under us (reconnect, shutdown). Nothing
                # to wait for, and certainly nothing to hang on.
                return time.perf_counter() - started
            now = time.perf_counter()
            if now >= deadline:
                logger.debug(
                    "State publish went ahead with a command still unread after "
                    "%.1f ms; the receive path is behind",
                    self._DRAIN_GATE_TIMEOUT * 1e3,
                )
                return now - started
            time.sleep(self._DRAIN_GATE_YIELD)

    def _account_for_communication_cycle(self) -> None:
        """Run one cycle of the FCI communication-constraints emulation.

        Called from the state-publish thread, once per cycle, just before the
        state goes out:

        1. close the cycle that the previously published state opened,
        2. publish the rolling ``control_command_success_rate``,
        3. tell the limit checker which state is going out, so its differencing
           interval is the server's own observation and not the client's echo,
        4. **extrapolate the motion generator across a cycle the client missed**,
           dispatching the result to physics and publishing it back in the
           commanded fields (:meth:`_extrapolate_missed_cycle`),
        5. abort the motion once too many cycles are lost in a row.

        Step 4 is the one that commands something, and it is what the real FCI
        does: Control "takes the previous waypoints and performs a linear
        extrapolation (keep acceleration constant and integrate) for the missed
        time step" (``docs/system_requirements.rst``), and the extrapolated
        value is what comes back in ``q_d``/``dq_d``/``ddq_d``
        (``docs/overview.rst``). A dropped *controller* packet is the exception
        and is left alone: "FCI will reuse the torques of the last successful
        received packet", which is what happens here by doing nothing at all.

        Extrapolation runs whether or not either enforcement flag is set,
        exactly as it does on hardware -- it is not a check, it is what the
        robot's reference *is* while the client is quiet. The flags still decide
        only whether a violation aborts.

        The extrapolation stops at
        :data:`~franka_sim.comm_constraints.MAX_CONSECUTIVE_LOST_CYCLES`
        consecutive misses, which is where the robot stops the motion. Past that
        the reference holds: a client that has genuinely gone away leaves the
        arm standing still rather than flying off along the trajectory it was on
        when it vanished. That bound is enforced here rather than inside the
        checker so the checker never has to know how long the gap has been --
        this is the only place that counts cycles.

        Kept out of the transmission loop's body so the loop stays readable and
        this stays testable on its own.

        The few microseconds between this call and the ``sendto`` are charged to
        the client: a command landing in them is late by the tracker's reckoning
        even though the state it was racing is not out yet. That bias is under
        one cycle and always in the conservative direction; see
        :meth:`CommConstraintTracker.command_received` for why crediting it back
        cannot be done without a wall clock.
        """
        published_id = self.robot_state.state["message_id"]
        outcome = self.comm.tick(published_id)
        # The differencing interval the motion-limit checker uses is bounded by
        # this: how many states the server has actually published since the one
        # the applied command history sits at. See
        # :meth:`MotionLimitChecker.cycles_since_applied`.
        self.motion_limits.note_published(published_id)

        # Zero when no control or motion generator loop is running, which is
        # what the real robot reports: control_command_success_rate "shows a
        # value of zero if no control or motion generator loop is currently
        # running" (libfranka include/franka/robot_state.h).
        self.robot_state.state["control_command_success_rate"] = outcome.success_rate

        if not outcome.active:
            return

        if outcome.lost:
            if outcome.consecutive_lost < self.comm.max_consecutive_lost:
                self._extrapolate_missed_cycle(outcome.closed_id)
            elif outcome.bound_reached:
                # The bound the robot stops at. With enforcement on, the abort
                # below is about to fire and this line is its explanation; with
                # enforcement off it is the whole report, which is why it is not
                # inside the ``violation_triggered`` branch.
                logger.warning(
                    "%s consecutive lost command cycles: no longer extrapolating "
                    "the motion generator, holding the last reference%s",
                    outcome.consecutive_lost,
                    "" if self.enforce_comm_constraints else " (not enforced)",
                )

        if outcome.violation_triggered:
            self._abort_on_communication_violation(outcome.motion_id)

    def _extrapolate_missed_cycle(self, closed_id: int) -> None:
        """Command the waypoint the client did not send, for one missed cycle.

        The motion-generator half of the FCI's packet-loss behaviour. The
        arithmetic is the limit checker's -- it already holds the backward
        differences of the last two real commands, which is the only honest
        source for the acceleration to freeze (see
        :meth:`franka_sim.motion_limits.MotionLimitChecker.extrapolate`) -- and
        what this method owns is the consequences: judging the result, and
        putting it where a real command would have gone.

        Those two are the same plumbing a received command gets, deliberately:

        * the extrapolated waypoint is **checked like a command**, and an
          extrapolation that runs out past the velocity envelope or a joint stop
          latches the same error a client commanding it would have. That is not
          collateral damage, it is the documented hazard -- intermittent drops
          "could trigger `discontinuity` errors even when your source signals
          conform with the interface specification" (``docs/overview.rst``) --
          and a sim that clamped the extrapolation instead would be hiding
          precisely the failure the client came here to find;
        * it is then **dispatched through :meth:`_dispatch_control_command`**,
          the same path a received datagram takes, so the arm keeps moving
          through the gap and ``q_d``/``dq_d``/``ddq_d`` (and
          ``O_T_EE_c``/``O_T_EE_d`` on a pose motion) advance with it. Reusing
          that method rather than writing the fields here is what keeps the
          FCI-layer field ownership in one place; it also means the hold latch
          covers this path for free -- a session that ended between the tick and
          here dispatches nothing.

        ``closed_id`` names the state whose cycle went unanswered. It is stamped
        on the substitute command, so the checker's applied history sits at that
        id and the client's resumed command -- which answers a *later* state --
        is differenced over the standard single cycle.

        Runs on the state-publish thread. Lock order is unchanged from the other
        publish-thread violation path (:meth:`_run_safety_velocity_check`):
        ``_motion_lock`` -> checker lock -> ``_hold_lock``, with the dispatch
        taking ``_hold_lock`` alone and never while another server lock is held.

        **The motion is re-validated inside the dispatch's own lock.** A ``Move``
        handled on the TCP thread between the extrapolation and the dispatch
        changes ``motion_generator_mode`` and ``controller_mode``, and
        :meth:`_dispatch_control_command` branches on exactly those -- so a
        substitute built for the *old* motion's generator would be routed into
        the new one's branch and written to fields the old motion never
        commanded. The re-check is on the server's own motion token, read
        without taking any checker lock so that the innermost-lock rule holds.
        """
        motion_id = self.motion_limits.motion_id
        generation = self._motion_generation
        extrapolated = self.motion_limits.extrapolate(closed_id)
        if extrapolated is None:
            # Nothing to continue: no motion generator armed, no real command
            # recorded yet, or a torque controller -- which holds.
            return
        command, violation = extrapolated

        if violation is not None:
            self.motion_limits.report(violation, enforced=self.enforce_motion_limits)
            if self.enforce_motion_limits:
                self._latch_and_abort(violation, motion_id)
                return
            if violation.fatal:
                return

        self._dispatch_control_command(command, motion_generation=generation)

    def _absorb_within_motion_limits(self, command, *, fresh: bool) -> bool:
        """Rewind, validate and record one received command; False if refused.

        The UDP thread's whole interaction with the limit checker, in one call
        that the checker takes under one hold of its lock -- see
        :meth:`franka_sim.motion_limits.MotionLimitChecker.absorb_command` for
        why that indivisibility is load-bearing and not merely tidy.

        Reporting and aborting stay here, outside the checker's lock, so the
        lock order is exactly what it was: ``_motion_lock`` (taken by
        :meth:`_latch_and_abort`) -> checker lock, never the reverse, and the
        checker's lock is still the innermost one anything in this server takes.

        Always reports -- a violation logs a rate-limited warning naming the
        joint or axis, the value and the limit, once per error class per motion.
        Only with ``--enforce-motion-limits`` does it additionally abort the
        motion and refuse the command, which is what the real robot does with a
        signal it cannot follow. A *fatal* violation (a non-finite command) is
        refused either way: applying NaN is not permissiveness, it poisons the
        physics state, the backward differences and the wire. It does not abort,
        though -- aborting is what the switch is for.
        """
        motion_id = self.motion_limits.motion_id
        outcome = self.motion_limits.absorb_command(
            command,
            self.robot_state.state["q"],
            fresh=fresh,
            enforce=self.enforce_motion_limits,
        )
        if outcome.violation is None:
            return True

        self.motion_limits.report(outcome.violation, enforced=self.enforce_motion_limits)
        if not self.enforce_motion_limits:
            return outcome.accepted
        self._latch_and_abort(outcome.violation, motion_id)
        return False

    def _latch_and_abort(self, violation, motion_id: int) -> None:
        """Latch the first violation of a motion and abort it, as one step.

        The check ("is a violation already latched?") and the act (latch, then
        abort) run from *two* threads -- the UDP receive thread checks each
        command, the state-publish thread runs the safety controller -- so they
        have to be indivisible. Two things go wrong otherwise, and the second
        does not heal:

        * both threads read ``violated`` as False in the same instant and abort
          twice, answering one ``Move`` with two terminal responses;
        * the running motion ends and the next one starts between the caller's
          read of the checker's token and the latch. :meth:`
          franka_sim.motion_limits.MotionLimitChecker.start_motion` clears the
          latch for the new motion, so the stale latch lands on *it* -- while
          :meth:`_abort_with_error`, correctly, refuses to abort a motion whose
          token no longer matches. The new motion then runs with a violation
          latched and no reflex behind it, and because the server refuses a
          ``Move`` while a violation is latched, every later ``Move`` on that
          connection is answered ``kCommandNotPossibleRejected``. Permanently
          un-abortable, until ``AutomaticErrorRecovery``.

        Holding ``_motion_lock`` across the whole sequence closes both. The
        running motion cannot change under us, so re-reading the checker's token
        *inside* the lock decides once and for all whether this violation still
        belongs to a live motion -- and if it does, ``_abort_with_error`` (which
        takes the same re-entrant lock) cannot then refuse it. **The latch and
        the abort now succeed or fail together**, which is the invariant that
        makes an un-abortable motion impossible.

        Lock order here is ``_motion_lock`` -> the checker's own lock (taken and
        released by :attr:`motion_limits.motion_id` and ``latch()``, never held
        across the call into ``_abort_with_error``) -> ``_hold_lock`` (taken
        inside ``_abort_with_error`` -> ``_engage_idle_hold``) -> the simulator
        backend (called from inside ``_hold_lock`` by
        ``_switch_to_hold_position``). ``_motion_lock`` -> ``_tcp_send_lock`` is
        a separate nesting used elsewhere (``_finish_motion``,
        ``handle_stop_move_command``); it does not occur on this path, because
        ``_abort_with_error`` defers its ``Move`` response and sends it, if at
        all, from ``_flush_pending_move_response`` after ``_motion_lock`` has
        already been released. ``_tcp_send_lock`` and ``_hold_lock`` are both
        leaves -- neither is held while acquiring another server lock -- so no
        two locks are ever taken in both orders and there is no cycle.

        ``motion_id`` is the checker's token as the caller read it before
        running its check; 0 means "no motion named", which is what a violation
        raised between motions (a non-finite datagram) carries.
        """
        with self._motion_lock:
            running = self.motion_limits.motion_id
            if motion_id != running:
                logger.info(
                    "Not latching (%s): the motion that violated is already over",
                    violation.describe(),
                )
                return
            if running and running != self._motion_generation:
                # The checker still names a motion the server has moved past:
                # it is on its way out, so latching would strand its successor.
                logger.info(
                    "Not latching (%s): the running motion is being replaced",
                    violation.describe(),
                )
                return
            if self.motion_limits.violated:
                # Already latched: a burst of bad commands aborts once.
                return
            self.motion_limits.latch()
            self._abort_with_error(violation.error_indices, violation.describe(), motion_id)

    def _publish_commanded_pose(self, sim_state) -> None:
        """Report the internal controller's hold pose in ``O_T_EE_d``/``O_T_EE_c``.

        The Cartesian twin of :meth:`_publish_hold_setpoint`. Between motions --
        and during every motion whose generator is not the Cartesian pose one --
        the robot is held by its internal controller, so the pose it reports as
        commanded *is* the pose it is in. These fields were a permanent identity
        stub for as long as nothing read them, and that was not harmless: a
        libfranka Cartesian-pose motion generator initialises and holds from
        ``O_T_EE_d`` (the smoke suite's own helpers open with
        ``std::array<double, 16> cmd = state.O_T_EE_d;``), so an identity there
        made *every* pose motion open ten-ish metres and a full rotation away
        from the robot and trip
        ``cartesian_position_motion_generator_start_pose_invalid`` on cycle 0 --
        a sim artifact that hid five of the suite's real Cartesian checks
        behind it.

        During either Cartesian motion the two fields belong to that motion
        instead -- to the client's stream on the pose generator
        (:meth:`_echo_commanded_cartesian`), and frozen at the motion's start
        pose where the stream carries none
        (:meth:`_cartesian_motion_owns_commanded_fields`, which is also where
        the reason they must not track the measured arm there is written down).
        The moment the motion ends they snap back here, which is what the real
        robot does when the internal controller takes the flange back.

        **Arm roles only.** The mobile-duo base bridge's ``O_T_EE`` is
        dead-reckoned from the commanded twist and its commanded Cartesian
        fields are ``O_dP_EE_d``/``O_dP_EE_c``; neither is this method's
        business. Same role guard as :data:`COMMANDED_STATE_FIELDS`.
        """
        if self.mobile_base or self._cartesian_motion_owns_commanded_fields():
            return
        pose = sim_state.get("O_T_EE")
        if pose is None:
            return
        pose = pose.tolist() if hasattr(pose, "tolist") else list(pose)
        self.robot_state.state["O_T_EE_d"] = pose
        self.robot_state.state["O_T_EE_c"] = pose

    def _publish_elbow(self, sim_state) -> None:
        """Report the arm's elbow configuration in ``elbow`` and ``elbow_d``.

        ``elbow[0]`` is the 7-DOF redundancy angle, which on an FR3 *is* joint 3,
        and ``elbow[1]`` is the branch flag: the sign of joint 4, which libfranka
        insists is exactly +-1 (``isValidElbow``, ``include/franka/control_tools.h``).
        Both are a reading of ``q``, so there is nothing to model and nothing to
        integrate -- which is why these fields were a permanent ``[0.0, 0.0]``
        stub for as long as no generator consumed them.

        Now one does. A ``kCartesianPosition`` client builds its command as
        ``franka::CartesianPose{state.O_T_EE_d, state.elbow_d}``, and libfranka
        refuses to *send* an elbow whose flag is not +-1 -- ``checkElbow`` throws
        ``std::invalid_argument`` client-side -- so a permanently zero
        ``elbow_d`` made the elbow interface unreachable: the client threw before
        a datagram was ever packed. Publishing the real thing is what lets the
        elbow checks in :mod:`franka_sim.motion_limits` be exercised by a real
        client at all.

        Arm roles only. On the mobile-duo base bridge ``q`` is the four swerve
        steer/drive joints, which have no elbow of any kind.
        """
        if self.mobile_base:
            return
        elbow = self._elbow_from_joints(sim_state.get("q"))
        if elbow is None:
            return
        self.robot_state.state["elbow"] = elbow
        # ``elbow_d``/``elbow_c`` are the *commanded* elbow on hardware. While
        # either Cartesian generator is running they belong to that motion --
        # the client's stream when it commands an elbow
        # (:meth:`_echo_commanded_cartesian`), frozen at the motion's start
        # elbow when it does not -- and must not track the arm, for the
        # feedback reason spelled out in
        # :meth:`_cartesian_motion_owns_commanded_fields`. The rest of the time
        # -- idle, between motions, under a joint generator -- the honest value
        # is the one the internal controller is holding, which is the measured
        # elbow. Same reasoning as ``q_d`` between motions; see
        # :meth:`_publish_hold_setpoint`.
        if self._cartesian_motion_owns_commanded_fields():
            return
        self.robot_state.state["elbow_d"] = list(elbow)
        self.robot_state.state["elbow_c"] = list(elbow)

    @staticmethod
    def _elbow_from_joints(joints) -> Optional[List[float]]:
        """``[redundancy angle, branch flag]`` for an arm configuration, or None.

        ``elbow[0]`` is joint 3's angle on an FR3 and ``elbow[1]`` is the sign
        of joint 4, which libfranka insists is exactly +-1 (``isValidElbow``,
        ``include/franka/control_tools.h``). None when the caller has no arm
        configuration to read, so every caller can skip rather than guess.
        """
        if joints is None or len(joints) < 4:
            return None
        return [float(joints[2]), 1.0 if float(joints[3]) >= 0.0 else -1.0]

    def _refresh_ee_transform(self) -> None:
        """Recompute ``F_T_EE`` from ``F_T_NE``/``NE_T_EE`` and tell the backend.

        ``F_T_EE = F_T_NE * NE_T_EE`` -- libfranka's own decomposition
        (``Robot::setEE``, ``include/franka/robot.h``), so the two settable
        halves and the derived whole can never disagree. Both are column-major
        16-element wire poses, and so is the result.

        The backend is told because the EE frame is where it measures Cartesian
        velocity for the safety controller (see
        :meth:`franka_sim.mujoco_franka_sim.MujocoFrankaSim.update_ee_transform`).
        A backend without that setter -- Genesis, the mobile-base bridge -- is
        left alone and simply publishes no ``O_dP_EE``, which switches the
        Cartesian check off rather than feeding it a wrong frame.
        """
        f_t_ee = transform_matrix(self.robot_state.state["F_T_NE"]) @ transform_matrix(
            self.robot_state.state["NE_T_EE"]
        )
        values = [float(value) for value in f_t_ee.T.flatten()]
        self.robot_state.state["F_T_EE"] = values
        # getattr on self too: reset_state() also runs from teardown paths, and
        # a half-built server has no backend yet.
        setter = getattr(getattr(self, "genesis_sim", None), "update_ee_transform", None)
        if setter is None:
            return
        try:
            setter(values)
        except Exception:  # pragma: no cover - a backend that rejects the frame
            logger.warning("Backend rejected the F_T_EE update", exc_info=True)

    def _run_safety_velocity_check(self, sim_state) -> None:
        """Run the safety controller against this cycle's measured velocity.

        The limit checks that judge the *robot* rather than the client, and the
        only ones not tied to a motion generator: the real robot watches
        measured ``dq`` against the position-based velocity envelope and latches
        ``joint_velocity_violation``, and watches the measured end-effector
        speed against the Cartesian translational limit and latches
        ``cartesian_velocity_violation`` -- both in every control mode. Franka's
        hardware smoke suite pins the case no commanded check could ever cover:
        a pure-torque session ramping 3 Nm into joint 6 until the arm folds
        through the envelope, with no commanded velocity anywhere in the session
        (``moveJointVelocityViolation``).

        **The Cartesian check is asked first, and that ordering is the
        hardware's.** ``CartesianVelocityViolationHardware`` ramps joints 2 and
        4 with an EE 0.5 m out along the flange and hardware reports
        ``cartesian_velocity_violation`` *on its own* -- no
        ``joint_velocity_violation`` beside it -- so whichever cycle is the
        first to break both, the Cartesian error is the one that must latch.
        (In this sim the EE passes 3 m/s some 90 ms before either joint check
        objects, so the ordering only matters as a guarantee, not in practice.)

        Called once per cycle from the state-publish loop with the physics
        snapshot it has already read, so it costs one comparison per joint, one
        vector norm, and no extra backend call. Reporting is always on; the
        abort is gated on ``--enforce-motion-limits`` and goes through the same
        latch-then-abort plumbing as every other violation, so the client sees
        the identical thing: the error bit in ``errors``/``reflex_reason``,
        ``kReflex``, and the pending ``Move`` answered ``kReflexAborted``.

        Skipped on a ``mobile_base`` server: the envelope is the FR3's, read out
        of the arm's own URDF, and a swerve base's steering and drive joints are
        not FR3 joints. The duo's *arms* run on ordinary arm servers, which do
        get the check.
        """
        if self.mobile_base or sim_state is None:
            return
        motion_id = self.motion_limits.motion_id
        # Both are asked every cycle, not short-circuited: each also *records*
        # this cycle's reading, and the joint one's ``q`` is the reference
        # configuration the commanded-velocity envelope is built from. Only the
        # verdict is prioritised.
        cartesian = self.motion_limits.check_measured_cartesian_velocity(sim_state.get("O_dP_EE"))
        joint = self.motion_limits.check_measured_velocity(sim_state.get("q"), sim_state.get("dq"))
        violation = cartesian or joint
        if violation is None:
            return
        self.motion_limits.report(violation, enforced=self.enforce_motion_limits)
        if not self.enforce_motion_limits:
            # Nothing to refuse: no command caused this, so "reject the
            # command" is not a remedy. Reported and carried on with.
            return
        self._latch_and_abort(violation, motion_id)

    def _abort_on_communication_violation(self, motion_id: int) -> None:
        """Stop the motion with ``communication_constraints_violation``."""
        self._abort_with_error(
            COMMUNICATION_CONSTRAINTS_VIOLATION_INDEX,
            f"communication_constraints_violation: {self.comm.consecutive_lost} "
            "consecutive lost command cycles",
            motion_id,
        )

    def _abort_with_error(
        self, error_index: Union[int, Sequence[int]], reason: str, motion_id: int = 0
    ) -> None:
        """Stop the motion with one or more latched errors, the way the robot does.

        ``error_index`` is usually a single index, but accepts a sequence too:
        hardware latches *two* bits from a single abort in exactly one case --
        a commanded velocity-envelope violation (13) that trips while the
        safety controller is armed also latches ``joint_velocity_violation``
        (3); see :attr:`franka_sim.motion_limits.Violation.extra_error_index`
        and the citation there. Every index in the sequence lands in the same
        state update and the same deferred ``Move`` response, so the client
        reads them off one abort rather than two.

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
        wire arrays, i.e. its ``research_interface::robot::Error`` enumerator --
        or a sequence of them, all latched in this one abort.

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
            indices = (error_index,) if isinstance(error_index, int) else tuple(error_index)
            for index in indices:
                state["errors"][index] = True
                state["reflex_reason"][index] = True
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
                # Update state to idle modes -- unless a reflex is latched, in
                # which case the modes already left kMove and ``robot_mode``
                # is kReflex. Overwriting that with kIdle would tell the
                # client the motion ended cleanly while ``errors`` still says
                # why it did not, and kReflex is the flag that has to survive
                # until AutomaticErrorRecovery. Mirrors _finish_motion.
                if self.robot_state.state["robot_mode"] != RobotMode.kReflex:
                    self.robot_state.state["motion_generator_mode"] = 0  # kNone
                    self.robot_state.state["controller_mode"] = 3  # kOther
                    self.robot_state.state["robot_mode"] = RobotMode.kIdle

                # Send state with new message ID
                self.robot_state.update()  # This increments message_id
                final_state = self.robot_state.pack_state()
                self.udp_socket.sendto(final_state, (self.client_address, self.client_udp_port))
                logger.info(f"Sent final robot state with message_id:\
                          {self.robot_state.state['message_id']}")

            # StopMove ends the *motion*, not the session: the real FCI keeps
            # streaming RobotState over UDP (now reporting the idle hold set
            # above) from Connect all the way to disconnect, and a client is
            # free to send another Move on this same TCP connection --
            # libfranka's ActiveControl does exactly that on
            # cancelMotion() -> StopMove -> AutomaticErrorRecovery -> Move.
            # ``transmitting_state`` is the flag start_robot_state_transmission's
            # publish loop reads to keep going (see its docstring), and
            # ``connection_running`` gates that same loop *and* handle_client's
            # watchdog that ends the whole connection -- clearing either one
            # here used to tear the session down after the first motion, so a
            # second Move's kMotionStarted would arrive over TCP (the command
            # thread does not check either flag) but never see another UDP
            # datagram, and the client's next receive would hang until it
            # timed out. Neither flag is touched here any more; both are
            # cleared only by an actual disconnect (handle_tcp_messages) or
            # server shutdown (stop()). This mirrors the already-established
            # pattern for a motion that finishes on its own (see the
            # motion-finished handler above, which has never stopped the
            # publish loop either).

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
            # ...and with it F_T_EE, which is *derived*: libfranka documents
            # ``Robot::setEE`` as setting "the transformation NE_T_EE from
            # nominal end effector to end effector frame [...] The transformation
            # from flange to end effector frame is split into two
            # transformations: F_T_EE = F_T_NE * NE_T_EE"
            # (``include/franka/robot.h``). Publishing it is what lets a client
            # -- and this server's own Cartesian safety check -- know where the
            # EE actually is.
            self._refresh_ee_transform()

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

        The response is deferred until the arm is at (near) standstill; see
        :meth:`_wait_for_standstill`. On the real robot recovery inherently
        completes there -- an abort caught mid-motion leaves the arm decelerating
        under the internal controller, and a client that Move's again the instant
        it gets a reply is Move'ing into an arm that has already stopped. This
        sim used to reply immediately, so a client recovering from a fast abort
        could start its next motion while still decelerating from a couple of
        rad/s, and its own start-pose guard would then see measured q drifting
        off the commanded q it had just sent and throw a "Performance threshold
        reached" a few milliseconds later -- on a motion the client did nothing
        wrong on.
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

            # The idle hold above is what does the decelerating -- this only
            # waits for it to have worked. See _wait_for_standstill for why it
            # polls rather than sleeping once, and why it lives here rather
            # than blocking the state-publish loop.
            self._wait_for_standstill()

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

    def _wait_for_standstill(self) -> None:
        """Block the calling thread until the arm has settled, or the timeout passes.

        See :data:`AUTOMATIC_ERROR_RECOVERY_TIMEOUT` for why that ceiling is
        0.7 s rather than the 3 s first tried: libfranka's own TCP receive
        timeout on the response is a hard 1 s, and going over it does not
        make the wait "slower but correct" -- it breaks the connection.

        The 0.7 s budget is wall-clock, but what it is waiting on -- the arm
        ringing down to standstill -- is sim-time. On a scene running below
        roughly 0.7x real-time factor, 0.7 s of wall clock buys less than
        0.7 s of simulated settling, so this method routinely runs out the
        clock before the arm has actually stopped. That is not a bug to chase
        here: the timeout path below degrades to replying anyway, exactly as
        designed, and it is the RTF, not this wait, that wants fixing.

        Called from :meth:`handle_automatic_error_recovery_command`, which runs
        on the per-connection TCP thread (see ``handle_tcp_messages``) -- a
        thread of its own, separate from the state-publish loop
        (:meth:`start_robot_state_transmission`, which runs on the connection's
        accept thread) and from the UDP command receiver
        (:meth:`_handle_commands`). Blocking it here therefore does not stall
        either of those. It does hold up the *next* TCP command on this same
        connection, which is the point: libfranka's ``automaticErrorRecovery()``
        already blocks the client until this reply arrives, so there is no
        "other traffic" on this connection to protect it from -- but it must
        still poll rather than take one long sleep, both so
        :attr:`FrankaSimServer.running` going false during shutdown is noticed
        promptly (see the loop condition below) and so the timeout path can log
        and bail rather than oversleeping a caller that will never settle.

        Queries ``genesis_sim.get_robot_state()`` directly on every poll rather
        than reading the cached ``dq`` off :attr:`RobotState.state`.

        **That distinction is not cosmetic, even though the publish loop now
        keeps running through StopMove.** libfranka always sends ``StopMove``
        before ``AutomaticErrorRecovery`` (confirmed against the real wire:
        ``handle_stop_move_command`` runs first in every observed recovery
        sequence). ``handle_stop_move_command`` used to clear
        ``self.transmitting_state`` -- the flag
        :meth:`start_robot_state_transmission`'s loop reads to keep
        publishing -- which stopped the loop and froze
        ``robot_state.state["dq"]`` at whatever it last was; that stale read
        was the original version of this bug (see the comment in
        :meth:`handle_stop_move_command` for why the flag is not cleared
        there any more). The loop no longer stops for StopMove, but this
        method still
        reads the physics backend directly rather than the publish loop's
        copy: the loop only refreshes ``robot_state.state`` once per 1 kHz
        cycle, so going straight to ``get_robot_state()`` is still the
        freshest read available and costs nothing extra, and it stays correct
        even on any future path that *does* legitimately stop the loop (a
        real disconnect, server shutdown) while a recovery is still in
        flight. The physics backend itself has no such gap either way: it
        keeps stepping and answering ``get_robot_state()`` regardless of
        whether anything is publishing its output.

        Skipped entirely on a ``mobile_base`` server, matching
        :meth:`_run_safety_velocity_check`: ``dq`` there is the swerve base's
        steer/drive joints, not an FR3's, and the base's idle hold is already a
        zero-twist command rather than a decelerating position hold, so there is
        no equivalent "still ringing down" state to wait out.
        """
        if self.mobile_base:
            return
        deadline = time.monotonic() + AUTOMATIC_ERROR_RECOVERY_TIMEOUT
        settled_cycles = 0
        while self.running and time.monotonic() < deadline:
            try:
                dq = self.genesis_sim.get_robot_state().get("dq")
            except Exception:
                # The backend erroring out here is not a reason to hang the
                # client; fall through to "not settled" and let the timeout
                # path below reply anyway.
                logger.exception(
                    "AutomaticErrorRecovery: could not read the simulator "
                    "while waiting for standstill"
                )
                dq = None
            if dq is not None and max(abs(value) for value in dq[:7]) < (
                AUTOMATIC_ERROR_RECOVERY_SETTLE_VELOCITY
            ):
                settled_cycles += 1
                if settled_cycles >= AUTOMATIC_ERROR_RECOVERY_SETTLE_CYCLES:
                    return
            else:
                settled_cycles = 0
            time.sleep(AUTOMATIC_ERROR_RECOVERY_POLL_PERIOD)
        if self.running:
            logger.warning(
                "AutomaticErrorRecovery: arm did not settle below %.3f rad/s "
                "within %.1fs; replying success anyway",
                AUTOMATIC_ERROR_RECOVERY_SETTLE_VELOCITY,
                AUTOMATIC_ERROR_RECOVERY_TIMEOUT,
            )

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
            # Deliberately not bound. This socket only ever *sends*, so the
            # kernel binds it to an ephemeral port on the first ``sendto``
            # below; libfranka reads the source port off the datagrams it
            # receives and answers there, which is what a real FCI does too.
            # The consequence to know about: until that first send,
            # ``getsockname()`` reports port 0. The log line below therefore
            # prints 0 on every session, and anything wanting to know that this
            # session is broadcasting should watch :attr:`states_sent` rather
            # than the port -- binding explicitly here would make the port real
            # *before* any state had gone out, which is a strictly weaker
            # signal than the one the counter gives.
            # TODO move to somewhere appropriate
            self.start_command_receiver()

            # port of the udp_socket (0 until the first sendto; see above)
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
            # When the last state actually went out. The pacer keeps every
            # state at least one cycle behind it; see the deadline arithmetic
            # at the bottom of the loop.
            last_send = next_deadline - period

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

                    self._publish_commanded_pose(sim_state)
                    self._publish_elbow(sim_state)

                    # The safety controller: measured velocity against the
                    # position-based envelope, every cycle, in every control
                    # mode. Judges the arm, not the command, so it lives here
                    # rather than on the UDP receive path.
                    self._run_safety_velocity_check(sim_state)

                    # Everything above is this cycle's work; the pacing sleep
                    # below is not, and the stats line reports work.
                    cycle_time = time.time() - cycle_start
                    total_cycle_time += cycle_time
                    total_cycles += 1

                    # Pace to the ~1 kHz deadline *here*, with the cycle's work
                    # already done, so what the schedule governs is the moment
                    # the state leaves -- which is the only moment the client
                    # can see. Sleeping at the bottom of the loop instead put
                    # this cycle's work between the deadline and the send, and
                    # the minimum-spacing rule below then charged that work to
                    # every cycle and dragged the publish rate down with it.
                    remaining = next_deadline - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)

                    # Do not close this cycle while the client's answer to the
                    # previous one is still sitting unread in our own socket:
                    # the state about to go out would carry a ``q_d`` the client
                    # has already moved past, and libfranka filters its next
                    # waypoint toward exactly that field. Immediately before the
                    # accounting and the send, which is the only moment that
                    # sees every datagram this cycle could have brought. See
                    # _drain_gate. Whatever it waits for is this cycle's own
                    # time, not time to be made up: the deadline moves with it.
                    next_deadline += self._drain_gate()

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
                        last_send = time.perf_counter()
                        # After the send, so a reader that sees this move knows
                        # a datagram really went out (and, with it, that the
                        # socket has been implicitly bound to a real port).
                        self.states_sent += 1
                        # Inside the guard: a state that was never sent cannot
                        # be the one a deferred kReflexAborted is waiting to
                        # follow.
                        self._flush_pending_move_response()

                        # NOTE: the first state datagram does not get a Move
                        # response here. handle_move_command already answered
                        # this motion's Move with kMotionStarted when it was
                        # accepted -- Move gets exactly one reply on the wire,
                        # and the terminal one (kSuccess via StopMove, or
                        # kReflexAborted/etc. via _pending_move_response) is
                        # sent elsewhere. A second kSuccess sent from here
                        # used to sit unread in libfranka's response map,
                        # keyed by command id; when the motion later aborted,
                        # Robot::throwOnMotionError found that stale kSuccess
                        # ahead of the real terminal response and raised
                        # ProtocolException("Unexpected reply to a Move
                        # command") instead of the intended ControlException.
                        #
                        # first_state_sent still has to flip exactly once,
                        # though -- it also gates the q_d seeding above -- so
                        # it is set unconditionally right after the first
                        # successful sendto, decoupled from any response.
                        if not first_state_sent:
                            first_state_sent = True

                    # Update state for next iteration
                    self.robot_state.update()

                    # Schedule the next state (the sleep itself is above, just
                    # before the send).
                    next_deadline += period
                    # ...but never closer than a whole cycle to the state that
                    # just went out. A cycle that ran long used to be made up
                    # by firing the next state immediately behind it -- two
                    # states microseconds apart -- and that is not a schedule
                    # the FCI contract allows. The client cannot answer the
                    # first one in the time the second takes to arrive, so the
                    # second necessarily carries a ``q_d`` that predates the
                    # answer to the first; libfranka filters its next waypoint
                    # toward exactly that field, and the kink that produces is
                    # what the limit checker then reports as
                    # ``joint_motion_generator_velocity_discontinuity`` against
                    # a client that did nothing wrong (see _drain_gate for the
                    # other half of the same story). One cycle per state, even
                    # when the cycle before it overran.
                    #
                    # The floor is a fraction of a cycle short of the full
                    # period on purpose. It has to bite on a real overrun --
                    # the millisecond-plus ones that produced the bursts -- and
                    # not on the few tens of microseconds ``time.sleep`` routinely
                    # overshoots by, or every cycle would be charged that
                    # overshoot and the nominal 1 kHz would drift down to ~900 Hz.
                    # The client's observed turnaround is well under 100 us, so
                    # 0.8 ms is still a whole answering window.
                    next_deadline = max(next_deadline, last_send + _MIN_STATE_SPACING * period)

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
        self._arm_shutdown()
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
                except OSError as e:
                    # stop() closes the listening socket from another thread --
                    # that is what breaks this accept() out of its wait. The
                    # EBADF/EINVAL that follows is the shutdown signal, not a
                    # failure, and logging it with a traceback made every clean
                    # Ctrl+C look like a crash.
                    if not self.running or e.errno in (errno.EBADF, errno.EINVAL):
                        logger.debug("Accept loop stopping: %s", e)
                        break
                    logger.error(f"Connection handling error: {e}", exc_info=True)
                    self.reset_state()
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

    def _arm_shutdown(self):
        """Re-arm the once-per-run shutdown latches, at the start of a run.

        stop() is idempotent *within a run*, not for the lifetime of the
        object: a server that is started again (the accept loop is entered
        afresh, so a new listening socket exists) must be stoppable again, or
        the second stop() would return early and leak that socket.
        """
        with self._stop_lock:
            self._stopping = False
            self._cleanup_logged = False

    def start(self):
        """Start the TCP server and Genesis simulator"""
        self._arm_shutdown()
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

        Idempotent: stop() calls it, and so does the accept loop's finally
        clause once stop() has closed the socket underneath it. Every step is
        None-guarded, so the repeat run is a no-op; only the announcement is
        suppressed, because two "Cleaning up server resources..." lines per
        shutdown read like two shutdowns.
        """
        with self._stop_lock:
            first_cleanup = not self._cleanup_logged
            self._cleanup_logged = True
        if first_cleanup:
            logger.info("Cleaning up server resources...")
        else:
            logger.debug("Cleaning up server resources (already cleaned)...")

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
        """Stop the server and release every resource it owns.

        Idempotent and non-raising. Shutdown is the one path that has to
        survive being called badly: twice (a second Ctrl+C), concurrently
        (accept thread vs. main thread), or with one of its stages already
        broken. A stage that raised used to skip every stage after it, which
        is how a Ctrl+C could leave the listening socket bound or the viewer's
        GL context alive.

        Ordered so nothing is torn down while something else still uses it:
        stop accepting and close the sockets (which is also what breaks the
        accept/receive loops out of their waits), then the gripper server,
        then the simulator -- whose viewer teardown is the slowest step and
        the one the network threads must already be gone for. Every join is
        bounded; every serving thread is a daemon, so a join that does time
        out can never keep the process alive.
        """
        with self._stop_lock:
            if self._stopping:
                logger.debug("stop() already in progress or done")
                return
            self._stopping = True

        logger.info("Stopping server...")
        self.running = False
        self.connection_running = False
        self.transmitting_state = False

        for stage, action in (
            ("socket cleanup", self.cleanup),
            ("gripper server", self._stop_gripper),
            ("simulator", self.genesis_sim.stop),
        ):
            try:
                action()
            except Exception:
                logger.error("Error stopping the %s; continuing shutdown", stage, exc_info=True)

    def _stop_gripper(self):
        """Stop the gripper server and wait (briefly) for its accept loop."""
        if self.gripper_server is None:
            return
        self.gripper_server.stop()
        thread, self.gripper_thread = self.gripper_thread, None
        if thread is not None:
            # stop() has already closed the listening socket, which drops the
            # accept() out of its wait immediately -- this join is a formality
            # that should return in microseconds. It is bounded anyway, and the
            # thread is a daemon, so a gripper client wedged in a backend call
            # delays shutdown by at most GRIPPER_JOIN_TIMEOUT_S instead of
            # holding the process open.
            thread.join(timeout=GRIPPER_JOIN_TIMEOUT_S)
            if thread.is_alive():
                logger.warning(
                    "Gripper server thread did not stop within %.1fs; abandoning it",
                    GRIPPER_JOIN_TIMEOUT_S,
                )


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
    finally:
        # stop() is idempotent and does not raise, so running it on every exit
        # path (not just the interrupt) is safe -- and closing the viewer's window
        # is an exit path too.
        server.stop()


if __name__ == "__main__":
    main()
