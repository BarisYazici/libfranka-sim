"""FCI communication-constraints emulation: per-cycle packet-loss accounting.

The real FCI runs a hard 1 ms budget. Every published ``RobotState`` opens a
cycle, and the client's answer -- a ``RobotCommand`` echoing that state's
``message_id`` -- has to be back before the next state goes out. libfranka's own
documentation is explicit about what happens when it is not:

    "If the **<1 ms constraint** is violated for a cycle, the packet is dropped
    by FCI. After 20 consecutively dropped packets, the robot `will stop` with
    the ``communication_constraints_violation`` error. Communication quality can
    be monitored via the ``RobotState::control_command_success_rate`` field."
    -- libfranka v10, ``docs/network_requirements.rst`` (Time Constraints)

    "If a **motion generator command packet is dropped**, the robot takes the
    previous waypoints and performs a linear extrapolation (keep acceleration
    constant and integrate) for the missed time step. [...] If a **controller
    command packet is dropped**, FCI will reuse the torques of the last
    successful received packet."
    -- libfranka v10, ``docs/system_requirements.rst`` (Packet Handling)

    **The sim does both halves.** A missed *motion generator* cycle is
    extrapolated under the acceleration frozen at the start of the gap, and the
    extrapolated waypoint is dispatched to physics and published back in
    ``q_d``/``dq_d``/``ddq_d`` (and ``O_T_EE_c``/``O_T_EE_d`` on a pose motion)
    exactly as Control reports its own. A missed *controller* cycle holds the
    last torque, which is the quotation's second half. The arithmetic lives in
    :meth:`franka_sim.motion_limits.MotionLimitChecker.extrapolate`; this module
    only says *which* cycles were missed. See ``docs/robot-state.md``.

    Extrapolation stops at :data:`MAX_CONSECUTIVE_LOST_CYCLES`, where the robot
    stops too: past that the reference simply holds while the violation below is
    latched, so a client that has genuinely gone away leaves the arm at a
    standstill rather than flying off along its last trajectory.

    "If ``>=20`` packets are lost in a row the control loop is stopped with the
    ``communication_constraints_violation`` exception."
    -- libfranka v10, ``docs/overview.rst``

    "Percentage of the last 100 control commands that were successfully received
    by the robot. Shows a value of zero if no control or motion generator loop
    is currently running."
    -- libfranka v10, ``include/franka/robot_state.h`` on
    ``RobotState::control_command_success_rate``

This module owns the *accounting* half of that emulation (the wire plumbing
lives in :mod:`franka_sim.franka_sim_server`) so it can be unit-tested without a
socket in sight.

Everything here is counted in **cycle space**, never in wall-clock time: one
cycle is one state-publish tick. A simulator that stalls delays the state
publish and the client's answer alike, so a stall must never register as packet
loss -- which is exactly what a wall-clock deadline would do.

Accounting is always on. The *abort* -- latching
``communication_constraints_violation`` and stopping the motion -- is opt-in;
see :data:`ENFORCE_ENV_VAR`.
"""

import os
import threading
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

#: Consecutive dropped cycles that abort the motion with
#: ``communication_constraints_violation``. Not derivable from the headers --
#: the number is stated in the FCI documentation: "After 20 consecutively
#: dropped packets, the robot will stop with the
#: ``communication_constraints_violation`` error"
#: (libfranka ``docs/network_requirements.rst``), restated as ">=20 packets are
#: lost in a row" in ``docs/overview.rst``. The 20th consecutive loss trips it.
MAX_CONSECUTIVE_LOST_CYCLES = 20

#: Width of the rolling window behind ``control_command_success_rate``. This one
#: *is* real-robot semantics, straight out of ``include/franka/robot_state.h``:
#: "Percentage of the last 100 control commands that were successfully received
#: by the robot."
SUCCESS_RATE_WINDOW = 100

#: How many cycles behind the newest published state a command's echoed
#: ``message_id`` may be and still count for the cycle it arrives in. Zero: the
#: echo has to match the id of the state that opened the cycle.
#:
#: The window is bounded on *both* sides -- ``published_id - horizon <= id <=
#: published_id``. An id the server has never published cannot be an answer to
#: anything: a bit-flip, a replay, or a client whose counter simply does not
#: conform. Counting it as fresh would let such a client suppress packet-loss
#: accounting forever, which is the opposite of what this module is for.
#:
#: That is what libfranka does on the wire. ``Robot::Impl::updateState`` stores
#: every accepted state's ``message_id`` in ``message_id_``
#: (``src/robot_impl.cpp``), and ``Robot::Impl::sendRobotCommand`` stamps the
#: next ``RobotCommand`` with exactly that value -- so a conforming client's
#: command carries the id of the state it is answering, and answering state N
#: inside cycle N means the ids match. Anything older is a client that missed
#: its 1 ms window (or is replaying a frozen id), which is the dropped packet
#: the FCI documentation describes.
#:
#: Kept as a named constant rather than inlined so the tolerance is one edit
#: away, and so the unit tests can exercise a non-zero horizon.
FRESHNESS_HORIZON_CYCLES = 0

#: Index of ``communication_constraints_violation`` in the 41-entry error
#: arrays: ``research_interface::robot::Error::kCommunicationConstraintsViolation``
#: is the 26th enumerator (0-based 25) in ``common/include/research_interface/
#: robot/error.h``, and ``franka::Errors`` binds
#: ``errors_[static_cast<size_t>(Error::kCommunicationConstraintsViolation)]``
#: to that member (``src/errors.cpp``).
COMMUNICATION_CONSTRAINTS_VIOLATION_INDEX = 25

#: Opt-in switch for the violation abort. Loss accounting, the rolling
#: ``control_command_success_rate`` and packet-loss extrapolation are always
#: active; setting this to a truthy value additionally makes a run of
#: :data:`MAX_CONSECUTIVE_LOST_CYCLES` lost cycles latch the error and stop the
#: motion, the way the robot does. Off by default because a sim is routinely
#: driven by clients that are not 1 kHz control loops at all.
ENFORCE_ENV_VAR = "FRANKA_SIM_ENFORCE_COMM_CONSTRAINTS"

#: Spellings of :data:`ENFORCE_ENV_VAR` that turn enforcement on. An allow-list,
#: not a deny-list: "set, therefore on" would read ``=disabled`` and ``=off, really``
#: as *enabled*, which is the wrong way round for a switch that can stop a
#: motion. Any nonzero integer counts too, so ``=2`` behaves like ``=1``.
_TRUTHY = {"1", "true", "t", "yes", "y", "on", "enable", "enabled"}


def _is_truthy(value: str) -> bool:
    """Whether ``value`` spells "on"; see :data:`_TRUTHY`."""
    value = value.strip().lower()
    if value in _TRUTHY:
        return True
    return value.lstrip("+-").isdigit() and int(value) != 0


def enforcement_enabled_by_env(environ: Optional[Dict[str, str]] = None) -> bool:
    """Whether violation aborts are on, per :data:`ENFORCE_ENV_VAR`."""
    env = os.environ if environ is None else environ
    return _is_truthy(env.get(ENFORCE_ENV_VAR, ""))


@dataclass(frozen=True)
class CycleOutcome:
    """What one publish tick concluded about the cycle that just closed.

    ``active`` is False while no motion is being commanded; the other fields
    then carry the idle defaults (nothing lost, rate 0.0) and the caller does
    no accounting at all.
    """

    active: bool
    lost: bool
    success_rate: float
    consecutive_lost: int
    #: True only on the tick that *first* crosses
    #: :data:`MAX_CONSECUTIVE_LOST_CYCLES`, and only when enforcing, so the
    #: caller aborts exactly once.
    violation_triggered: bool
    #: Which motion this outcome belongs to (the caller's session token, 0 when
    #: none is running). The abort is dispatched from the publish thread while
    #: the TCP thread may already have started the *next* motion, so the caller
    #: has to be able to tell whether the motion that violated is still the
    #: current one -- see ``FrankaSimServer._abort_with_error``.
    motion_id: int = 0
    #: ``message_id`` of the state whose cycle this outcome closed -- the cycle
    #: the client either answered or lost. The *previously* published id, not
    #: the one being published now, and the id the caller stamps on the
    #: extrapolated command it substitutes for a lost answer, so the resumed
    #: real command is still exactly one cycle away from the history. Reported
    #: rather than recomputed as ``published_id - 1`` because nothing here
    #: promises the caller's ids advance by exactly one.
    closed_id: int = 0
    #: True on the tick where :attr:`consecutive_lost` first *reaches*
    #: :data:`MAX_CONSECUTIVE_LOST_CYCLES` -- whether or not enforcement is on,
    #: which is what distinguishes it from :attr:`violation_triggered`. It is
    #: the moment the caller must stop extrapolating and hold: the robot stops
    #: there too, and a reference that kept integrating past it would carry the
    #: arm away on the trajectory of a client that is no longer there. Once per
    #: run of losses by construction -- the counter passes the bound exactly
    #: once on its way up, and any answered cycle resets it to zero.
    bound_reached: bool = False


class CommConstraintTracker:
    """Per-cycle packet accounting for one FCI control session.

    Two threads touch this: the UDP receive thread calls
    :meth:`command_received`, the state-publish thread calls :meth:`tick` once
    per cycle. Both go through one lock; every method is short and allocation
    free enough to sit on a 1 kHz path.

    Lifecycle: :meth:`start_motion` on ``Move``, :meth:`end_motion` when the
    motion finishes or the session dies, :meth:`recover` on
    ``AutomaticErrorRecovery``.
    """

    def __init__(
        self,
        *,
        window: int = SUCCESS_RATE_WINDOW,
        max_consecutive_lost: int = MAX_CONSECUTIVE_LOST_CYCLES,
        freshness_horizon: int = FRESHNESS_HORIZON_CYCLES,
        enforce: bool = False,
    ):
        """Build a tracker; see the module constants for the defaults."""
        self._lock = threading.Lock()
        self._window_size = window
        self._max_consecutive_lost = max_consecutive_lost
        self._freshness_horizon = freshness_horizon
        #: Whether a violation may abort the motion. Tracking and reporting run
        #: either way -- that is the point of making this opt-in.
        self.enforce = enforce

        self._cycles: deque = deque(maxlen=window)
        #: Answered cycles inside :attr:`_cycles`, maintained incrementally so
        #: the ~1 kHz tick never sums the window (it used to be O(window)).
        self._received_count = 0
        self._consecutive_lost = 0
        self._received_in_cycle = False
        self._published_id = 0
        #: A Move is running.
        self._motion = False
        #: Caller-supplied token for the running motion, echoed in
        #: :class:`CycleOutcome` so a late abort cannot hit a newer motion.
        self._motion_id = 0
        #: ...and the client has started answering, so cycles count. Without
        #: this the gap between ``kMotionStarted`` and the client's first
        #: command (libfranka spins reading states until the modes match) would
        #: be charged to the client as loss.
        self._armed = False
        self._violated = False

    # -- lifecycle ---------------------------------------------------------

    def start_motion(self, motion_id: int = 0) -> None:
        """Arm a fresh motion: empty window, no losses, nothing armed yet.

        ``motion_id`` is an opaque token identifying this motion. It comes back
        in every :class:`CycleOutcome` and gates :meth:`end_motion`, so a
        ``motion_generation_finished`` datagram from the *previous* motion that
        the UDP thread only gets to after this one started cannot silently
        switch the accounting off (which pinned the success rate at 0.0 and left
        enforcement dead for the rest of the connection).

        The latched violation is re-armed here. Enforcement used to fire at most
        once per *connection* because ``_violated`` was only ever cleared by
        :meth:`recover`; a second run of lost cycles in a later motion went
        unpunished. The server refuses a ``Move`` outright while a violation is
        still latched, so reaching this method at all means the client has
        recovered.
        """
        with self._lock:
            self._cycles.clear()
            self._received_count = 0
            self._consecutive_lost = 0
            self._received_in_cycle = False
            self._motion = True
            self._motion_id = motion_id
            self._armed = False
            self._violated = False

    def end_motion(self, motion_id: Optional[int] = None) -> None:
        """The motion is over (finished, stopped, aborted or the client died).

        ``motion_id`` names the motion the caller believes is ending. A token
        that no longer matches the running motion is a stale event -- most often
        a queued ``motion_generation_finished`` datagram overtaken by the next
        ``Move`` -- and is ignored. ``None`` means "whatever is running", which
        is what the unconditional teardown paths want.
        """
        with self._lock:
            if motion_id is not None and motion_id != self._motion_id:
                return
            self._motion = False
            self._motion_id = 0
            self._armed = False
            self._received_in_cycle = False
            self._consecutive_lost = 0
            self._cycles.clear()
            self._received_count = 0

    def recover(self) -> None:
        """Clear a latched violation, as ``AutomaticErrorRecovery`` does."""
        with self._lock:
            self._violated = False
            self._consecutive_lost = 0
            self._cycles.clear()
            self._received_count = 0
            self._motion = False
            self._motion_id = 0
            self._armed = False

    # -- UDP receive thread ------------------------------------------------

    def command_received(self, command_message_id: int) -> bool:
        """Register one arriving ``RobotCommand``; True if it counts for a cycle.

        A command counts when it echoes the ``message_id`` of the state that
        opened the current cycle (within :data:`FRESHNESS_HORIZON_CYCLES`, which
        is zero by default). The window is closed on both sides: an id *ahead*
        of anything published is not an answer at all and never counts. A stale
        echo is a client that missed its window, and the cycle stays lost -- but
        it still *arms* the accounting, because a client that is answering at
        all, however late, is a client in a control loop.

        The boundary is charged to the client, deliberately. :meth:`tick` closes
        a cycle a few microseconds *before* the state that opens the next one is
        handed to the socket, so a command landing in that gap answers a state
        that is still, for those microseconds, the newest one out there -- and
        is charged as late anyway.

        Crediting it back was tried and removed. Nothing here can tell that
        command apart from one sent by a client running a full cycle behind: the
        one-behind client answers state N during cycle N+1, which in wall-clock
        terms is the same instant, and this module keeps no wall clock on
        purpose (a stall must delay state and answer alike). Crediting the
        boundary therefore credited the one-behind client too, and a client that
        is permanently a cycle late read 1.00 for ever and could never trip the
        violation -- exactly the staleness the two-sided bound exists to catch,
        walking back in through the boundary door.

        What is left is a bias of well under one cycle in the client's
        disfavour: a packet that misses its window by microseconds costs its own
        cycle *and* the one it lands in. That is the conservative direction --
        the sim reports slightly worse communication than the client achieved,
        never better.
        """
        with self._lock:
            if not self._motion:
                return False
            self._armed = True
            if self._is_fresh_locked(command_message_id, self._published_id):
                self._received_in_cycle = True
                return True
            return False

    def _is_fresh_locked(self, command_message_id: int, published_id: int) -> bool:
        """Whether ``command_message_id`` answers ``published_id`` in time."""
        return published_id - self._freshness_horizon <= command_message_id <= published_id

    # -- state publish thread ----------------------------------------------

    def tick(self, published_id: int) -> CycleOutcome:
        """Close the open cycle and open the one that state ``published_id`` starts.

        Call this immediately before publishing the state, so the window that
        follows -- until the next state goes out -- is the cycle the client's
        answer has to land in. The sub-microsecond gap between this call and the
        ``sendto`` is charged to the client; see :meth:`command_received`.
        """
        with self._lock:
            outcome = self._close_cycle_locked(self._published_id)
            self._received_in_cycle = False
            self._published_id = published_id
            return outcome

    def _close_cycle_locked(self, closed_id: int) -> CycleOutcome:
        if not (self._motion and self._armed):
            return CycleOutcome(
                active=False,
                lost=False,
                success_rate=0.0,
                consecutive_lost=self._consecutive_lost,
                violation_triggered=False,
                motion_id=self._motion_id,
                closed_id=closed_id,
            )

        received = self._received_in_cycle
        if len(self._cycles) == self._window_size and self._cycles[0]:
            self._received_count -= 1
        self._cycles.append(received)
        if received:
            self._received_count += 1
            self._consecutive_lost = 0
        else:
            self._consecutive_lost += 1

        triggered = False
        if (
            self.enforce
            and not self._violated
            and self._consecutive_lost >= self._max_consecutive_lost
        ):
            self._violated = True
            triggered = True

        return CycleOutcome(
            active=True,
            lost=not received,
            success_rate=self._success_rate_locked(),
            consecutive_lost=self._consecutive_lost,
            violation_triggered=triggered,
            motion_id=self._motion_id,
            closed_id=closed_id,
            # Equality, not ">=": the counter walks up one at a time, so it
            # stands on the bound for exactly one tick of any run of losses.
            # No latch needed, and a client that recovers and loses another
            # twenty gets told again.
            bound_reached=not received and self._consecutive_lost == self._max_consecutive_lost,
        )

    # -- reporting ---------------------------------------------------------

    def _success_rate_locked(self) -> float:
        if not self._cycles:
            return 0.0
        # Divided by how many cycles the window actually holds, so the first
        # cycles of a motion do not read as 99% packet loss while the 100-deep
        # window fills. Identical to n/100 once it is full. Deliberately
        # different from hardware for the first <100 cycles of a motion, where
        # the robot has no answer either: a rate of "1 of 1" reads as 1.0 here.
        # O(1): the numerator is maintained as the window rolls.
        return self._received_count / len(self._cycles)

    @property
    def success_rate(self) -> float:
        """Fraction of the last :data:`SUCCESS_RATE_WINDOW` cycles answered in time.

        0.0 while no control loop is running, matching the real robot:
        ``control_command_success_rate`` "shows a value of zero if no control or
        motion generator loop is currently running"
        (``include/franka/robot_state.h``).
        """
        with self._lock:
            if not (self._motion and self._armed):
                return 0.0
            return self._success_rate_locked()

    @property
    def violated(self) -> bool:
        """Whether a violation is latched (cleared only by :meth:`recover`)."""
        with self._lock:
            return self._violated

    @property
    def active(self) -> bool:
        """Whether cycles are being counted (a motion is running and answered)."""
        with self._lock:
            return self._motion and self._armed

    @property
    def consecutive_lost(self) -> int:
        """Cycles lost back-to-back, right now."""
        with self._lock:
            return self._consecutive_lost

    @property
    def motion_id(self) -> int:
        """Token of the running motion, 0 when none is."""
        with self._lock:
            return self._motion_id

    @property
    def max_consecutive_lost(self) -> int:
        """Losses in a row this tracker stops at; :data:`MAX_CONSECUTIVE_LOST_CYCLES`.

        Read by the publish loop to decide how long to keep extrapolating a
        silent client's trajectory: exactly as far as the robot does, and not
        one cycle further. Exposed because the constructor takes an override and
        a caller that read the module constant instead would disagree with a
        tracker built with a different bound.
        """
        return self._max_consecutive_lost
