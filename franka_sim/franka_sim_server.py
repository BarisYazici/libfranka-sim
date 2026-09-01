#!/usr/bin/env python3
"""The Franka FCI simulation server.

``FrankaSimServer`` is assembled here from the mixins in
:mod:`franka_sim.server`, which hold its methods grouped by responsibility:

* :mod:`franka_sim.server.tcp_session` -- TCP framing and command dispatch.
* :mod:`franka_sim.server.state_stream` -- the UDP command and state loops.
* :mod:`franka_sim.server.motion_session` -- the ``Move`` lifecycle and the
  motion-limit orchestration built around ``self.motion_limits``.
* :mod:`franka_sim.server.set_commands` -- the ``SetX`` family and error
  recovery.
* :mod:`franka_sim.server.lifecycle` -- state reset, accept loop and shutdown.

The mixins share one instance's state (its locks and flags), so they are split
for reading, not for reuse: only ``FrankaSimServer`` composes them. This module
keeps the constructor -- the single place that state is defined -- plus
``main()``. It remains the supported import path.

The import list is not itself an API surface: it carries only what the body
below needs, plus the handful of constants callers do import from this path
(``DEFAULT_PHYSICS``, ``resolve_sim_class`` and the
``AUTOMATIC_ERROR_RECOVERY_*`` timings, all defined in
:mod:`franka_sim.server.constants`). Everything else lives in the module that
defines it and should be imported from there.
"""

import argparse
import select
import threading
from pathlib import Path
from typing import List, Optional, Tuple

from franka_sim.comm_constraints import (
    CommConstraintTracker,
    enforcement_enabled_by_env,
)
from franka_sim.control_modes import ControlMode
from franka_sim.franka_protocol import COMMAND_PORT, MoveStatus
from franka_sim.gripper.server import FrankaGripperServer
from franka_sim.motion_limits import MotionLimitChecker
from franka_sim.motion_limits import (
    enforcement_enabled_by_env as motion_limit_enforcement_enabled_by_env,
)
from franka_sim.robot_state import RobotState
from franka_sim.server.constants import (
    AUTOMATIC_ERROR_RECOVERY_POLL_PERIOD,
    AUTOMATIC_ERROR_RECOVERY_SETTLE_CYCLES,
    AUTOMATIC_ERROR_RECOVERY_TIMEOUT,
    COMMANDED_STATE_FIELDS,
    DEFAULT_PHYSICS,
    INTERNAL_SIM_STATE_FIELDS,
    logger,
    resolve_sim_class,
)
from franka_sim.server.tcp_session import TcpSessionMixin
from franka_sim.server.state_stream import StateStreamMixin
from franka_sim.server.motion_session import MotionSessionMixin
from franka_sim.server.set_commands import SetCommandsMixin
from franka_sim.server.lifecycle import LifecycleMixin


class FrankaSimServer(
    TcpSessionMixin,
    StateStreamMixin,
    MotionSessionMixin,
    SetCommandsMixin,
    LifecycleMixin,
):
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
        physics_sim=None,
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
            physics_sim: Optional pre-configured simulator instance (any backend,
                or a fake in tests).
            physics: Backend built when ``physics_sim`` is not injected --
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
        #: The published ``q_d`` as it stood before the current run of
        #: extrapolated cycles, or None when no run is open. The wire-side
        #: counterpart of the limit checker's ``_gap_snapshot``: when a late
        #: datagram makes the checker throw the guesses away, this puts the
        #: reference back too, so the cycle the datagram answers is integrated
        #: once and not twice. See
        #: :meth:`_extrapolate_missed_cycle` and
        #: :meth:`_absorb_within_motion_limits`.
        self._extrapolated_reference: Optional[List[float]] = None
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
        #: How many received datagrams the UDP receive thread has taken out of
        #: the socket but not yet finished applying. Written only by that
        #: thread (an ``int`` rebind is atomic under the GIL) and read by the
        #: publish thread's :meth:`_drain_gate`, which must not close a cycle
        #: while it is non-zero: the socket empties when a datagram is *read*,
        #: the commanded echo the client filters against is written when it is
        #: *applied*, and everything in between is a window in which a
        #: published state carries a reference the client has already moved
        #: past.
        self._commands_in_flight = 0

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
        if physics_sim is None:
            logger.info("Initializing simulation (physics backend: %s)", physics)
            self.physics_sim = resolve_sim_class(physics)(
                enable_vis=enable_vis, enable_hand=(gripper_physics and enable_gripper)
            )
            logger.info("Simulation initialized")
        else:
            self.physics_sim = physics_sim

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
            self.physics_sim, "update_cartesian_pose"
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
                from franka_sim.gripper.physics import FrankaHandPhysics

                backend = FrankaHandPhysics(self.physics_sim)
            else:
                backend = gripper_backend
            self.gripper_server = FrankaGripperServer(host=host, backend=backend)
        else:
            self.gripper_server = None


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
