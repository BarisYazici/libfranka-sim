r"""FCI motion-limit and discontinuity checking for commanded signals.

The real FCI does not accept whatever a client sends. Control differentiates
every commanded signal with backward Euler at the 1 ms cycle and compares the
result against published per-joint limits; a signal that steps, ramps too hard
or leaves the joint range stops the motion with a specific error. libfranka's
own documentation spells the arithmetic out:

    "For every motion generator, Control differentiates the signals sent by the
    user with backwards Euler. For instance, if, using a joint position motion
    generator, at time :math:`k` the user sends the command :math:`q_{c,k}`, the
    resulting velocity, acceleration and jerk will be:

    * Velocity :math:`\dot{q}_{c,k} = (q_{c,k} - q_{c,k-1}) / \Delta t`
    * Acceleration :math:`\ddot{q}_{c,k} = (\dot{q}_{c,k} - \dot{q}_{c,k-1}) / \Delta t`
    * Jerk :math:`\dddot{q}_{c,k} = (\ddot{q}_{c,k} - \ddot{q}_{c,k-1}) / \Delta t`

    where :math:`\Delta t = 0.001`."
    -- libfranka v10, ``docs/overview.rst`` (Errors due to velocity limits
    violation and discontinuity errors)

    "Note that :math:`q_{c,k-1}, \dot{q}_{c,k-1}` and :math:`\ddot{q}_{c,k-1}`
    are always sent back to the user in the robot state as :math:`q_{d}`,
    :math:`\dot{q}_{d}` and :math:`\ddot{q}_{d}` so you will be able to compute
    the resulting derivatives in advance, **even in case of packet losses**."
    -- same section

That last clause is why this module differences against the *last applied*
command rather than the last *received* one, and it is also where the sim
knowingly diverges: when a cycle is lost the FCI extrapolates and the
extrapolated value becomes ``q_d``, while this sim holds the last command
through the gap (see :mod:`franka_sim.comm_constraints`). A client resuming
after a gap is therefore differenced against a *held* command rather than a
continued trajectory, so the step it appears to take is the gap's whole worth of
motion. That is why the interval a command is differenced over comes from its
echoed ``message_id`` rather than being assumed to be one cycle -- capped, so
the interval cannot be inflated into an exemption. See
:meth:`MotionLimitChecker.check`.

The error names and the finite-difference formulas are libfranka's; the limits
come from ``include/franka/rate_limiting.h`` and, for the per-joint position,
velocity and torque ranges, from the FR3 URDF this server itself serves over
``GetRobotModel`` (``franka_sim/models/fr3.urdf``) -- the same file libfranka
v10's ``JointVelocityLimitsConfig`` parses. Every constant below carries its
citation.

Validation and logging are always on. The *abort* -- latching the error,
answering the ``Move`` with ``kReflexAborted`` and refusing the offending
command -- is opt-in; see :data:`ENFORCE_ENV_VAR`.
"""

import logging
import math
import os
import sys
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from franka_sim.control_modes import ControlMode

logger = logging.getLogger(__name__)

# -- libfranka constants ------------------------------------------------------

#: Control cycle. ``franka::kDeltaT`` (``include/franka/rate_limiting.h:20``).
DELTA_T = 1e-3

#: ``franka::kLimitEps`` -- "Epsilon value for checking limits"
#: (``include/franka/rate_limiting.h:24``). Every published limit below already
#: has it subtracted, exactly as libfranka's constants do.
LIMIT_EPS = 1e-3

#: ``franka::kNormEps`` = ``std::numeric_limits<double>::epsilon()``
#: (``include/franka/rate_limiting.h:28``). Below this a Cartesian norm is not
#: worth limiting, which is how ``limitRate`` in ``src/rate_limiting.cpp:26``
#: and ``:48`` guards its divisions. Carried for symmetry with libfranka; the
#: comparison it appears in here can never decide anything, since every limit is
#: many orders of magnitude larger.
NORM_EPS = sys.float_info.epsilon

#: ``franka::kTolNumberPacketsLost`` (``include/franka/rate_limiting.h:35``):
#: "For FR3 there are no expected package loses. Therefore this number is set
#: to 0." It widens the velocity limits by the distance a packet-loss
#: extrapolation could cover; zero, so it does not.
TOL_NUMBER_PACKETS_LOST = 0.0

#: ``franka::kMaxJointAcceleration`` (``include/franka/rate_limiting.h:53``):
#: 10 rad/s^2 per joint, less :data:`LIMIT_EPS`.
MAX_JOINT_ACCELERATION = tuple(10.0 - LIMIT_EPS for _ in range(7))

#: ``franka::kMaxJointJerk`` (``include/franka/rate_limiting.h:47``):
#: 5000 rad/s^3 per joint, less :data:`LIMIT_EPS`.
MAX_JOINT_JERK = tuple(5000.0 - LIMIT_EPS for _ in range(7))

#: ``franka::kJointVelocityLimitsTolerance``
#: (``include/franka/rate_limiting.h:62``): "Tolerance value for joint velocity
#: limits to deal with numerical errors and data losses."
JOINT_VELOCITY_LIMITS_TOLERANCE = tuple(
    LIMIT_EPS + TOL_NUMBER_PACKETS_LOST * DELTA_T * MAX_JOINT_ACCELERATION[i] for i in range(7)
)

#: ``franka::kMaxTorqueRate`` (``include/franka/rate_limiting.h:44``):
#: 1000 Nm/s per joint, less :data:`LIMIT_EPS`.
MAX_TORQUE_RATE = tuple(1000.0 - LIMIT_EPS for _ in range(7))

#: ``franka::kMaxTranslationalVelocity`` / ``Acceleration`` / ``Jerk``
#: (``include/franka/rate_limiting.h:70``, ``:74``, ``:78``). The velocity
#: carries the packet-loss term, which is zero for the FR3.
MAX_TRANSLATIONAL_ACCELERATION = 9.0 - LIMIT_EPS
MAX_TRANSLATIONAL_JERK = 4500.0 - LIMIT_EPS
MAX_TRANSLATIONAL_VELOCITY = (
    3.0 - LIMIT_EPS - TOL_NUMBER_PACKETS_LOST * DELTA_T * MAX_TRANSLATIONAL_ACCELERATION
)

#: ``franka::kMaxRotationalVelocity`` / ``Acceleration`` / ``Jerk``
#: (``include/franka/rate_limiting.h:82``, ``:86``, ``:90``).
MAX_ROTATIONAL_ACCELERATION = 17.0 - LIMIT_EPS
MAX_ROTATIONAL_JERK = 8500.0 - LIMIT_EPS
MAX_ROTATIONAL_VELOCITY = (
    2.5 - LIMIT_EPS - TOL_NUMBER_PACKETS_LOST * DELTA_T * MAX_ROTATIONAL_ACCELERATION
)

#: ``franka::kMaxElbowVelocity`` / ``Acceleration`` / ``Jerk``
#: (``include/franka/rate_limiting.h:94``, ``:99``, ``:105``). Carried for
#: completeness -- this server never serves an elbow-carrying motion generator,
#: so nothing checks them yet.
MAX_ELBOW_ACCELERATION = 10.0 - LIMIT_EPS
MAX_ELBOW_JERK = 5000.0 - LIMIT_EPS
MAX_ELBOW_VELOCITY = (
    1.5 - LIMIT_EPS - TOL_NUMBER_PACKETS_LOST * DELTA_T * MAX_ELBOW_ACCELERATION
)

# -- FR3 per-joint ranges -----------------------------------------------------
#
# libfranka publishes no joint position or torque range of its own: those are
# robot-model data, and v10 reads them out of the URDF (``JointVelocityLimitsConfig``
# parses the ``<limit>`` and ``<position_based_velocity_limits>`` elements,
# ``include/franka/joint_velocity_limits.h:105-116``). The numbers below are
# therefore taken from the very URDF this server hands the client over
# ``GetRobotModel`` -- ``franka_sim/models/fr3.urdf`` -- so the limits the sim
# enforces and the model the client builds cannot drift apart.
#
# They are also self-consistent with libfranka: the deprecated
# ``computeUpperLimitsJointVelocity`` / ``computeLowerLimitsJointVelocity``
# (``include/franka/rate_limiting.h:133-190``) hard-code exactly these numbers,
# with ``2 * deceleration_limit`` folded in (6.0 -> 12.0, 2.585 -> 5.17,
# 3.50 -> 7.00, 4.00 -> 8.00, 17.0 -> 34.0, 5.5 -> 11.0, 17.0 -> 34.0) and the
# same ``max_velocity`` caps (2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26).

#: ``(lower, upper)`` per joint, from ``<limit lower= upper=>`` in
#: ``franka_sim/models/fr3.urdf`` (joint1..joint7, lines 89, 131, 173, 215,
#: 257, 285, 313).
JOINT_POSITION_LIMITS = (
    (-2.750100, 2.750100),
    (-1.791800, 1.791800),
    (-2.906500, 2.906500),
    (-3.048100, -0.145800),
    (-2.810100, 2.810100),
    (0.540920, 4.520500),
    (-3.019600, 3.019600),
)

#: ``(max_velocity, velocity_offset, deceleration_limit)`` per joint, from
#: ``<limit velocity=>`` and ``<position_based_velocity_limits
#: velocity_offset= deceleration_limit=>`` in ``franka_sim/models/fr3.urdf``.
POSITION_BASED_VELOCITY_LIMITS = (
    (2.62, 0.30, 6.000),
    (2.62, 0.20, 2.585),
    (2.62, 0.20, 3.500),
    (2.62, 0.30, 4.000),
    (5.26, 0.35, 17.000),
    (4.18, 0.35, 5.500),
    (5.26, 0.35, 17.000),
)

#: Per-joint torque range, from ``<limit effort=>`` in
#: ``franka_sim/models/fr3.urdf``: 87 Nm for joints 1-4, 12 Nm for 5-7.
MAX_TORQUE = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)

# -- error indices ------------------------------------------------------------
#
# Positions in the 41-entry ``errors`` / ``reflex_reason`` wire arrays are the
# 0-based enumerator values of ``research_interface::robot::Error``
# (``common/include/research_interface/robot/error.h:8-50``); ``franka::Errors``
# binds each named member to ``errors_[static_cast<size_t>(Error::k...)]``
# (``src/errors.cpp``). Line numbers below are that enum's declarations.

#: ``kJointPositionMotionGeneratorStartPoseInvalid`` (``error.h:20``) ->
#: ``joint_position_motion_generator_start_pose_invalid``.
JOINT_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX = 11

#: ``kJointMotionGeneratorPositionLimitsViolation`` (``error.h:21``).
JOINT_MOTION_GENERATOR_POSITION_LIMITS_VIOLATION_INDEX = 12

#: ``kJointMotionGeneratorVelocityLimitsViolation`` (``error.h:22``).
JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX = 13

#: ``kJointMotionGeneratorVelocityDiscontinuity`` (``error.h:23``) -- the
#: *acceleration* limit, per ``docs/overview.rst``.
JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX = 14

#: ``kJointMotionGeneratorAccelerationDiscontinuity`` (``error.h:24``) -- the
#: *jerk* limit, per ``docs/overview.rst``.
JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX = 15

#: ``kCartesianPositionMotionGeneratorStartPoseInvalid`` (``error.h:25``).
#: Nothing latches it: this server does not serve ``kCartesianPosition``, so
#: there is no Cartesian *pose* generator to check a start pose for. Kept
#: because :data:`ERROR_NAMES` is the vocabulary the log lines are grepped
#: against, and because the generator is on the roadmap.
CARTESIAN_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX = 16

#: ``kCartesianMotionGeneratorVelocityLimitsViolation`` (``error.h:27``).
CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX = 18

#: ``kCartesianMotionGeneratorVelocityDiscontinuity`` (``error.h:28``) -- the
#: acceleration limit.
CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX = 19

#: ``kCartesianMotionGeneratorAccelerationDiscontinuity`` (``error.h:29``) --
#: the jerk limit.
CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX = 20

#: ``kControllerTorqueDiscontinuity`` (``error.h:41``) -- the torque *rate*
#: limit, per ``docs/overview.rst``: "Control also computes the torque rate with
#: backwards Euler".
CONTROLLER_TORQUE_DISCONTINUITY_INDEX = 32

#: ``kTauJRangeViolation`` (``error.h:43``) -> ``tau_J_range_violation``. There
#: is no ``controller_torque_range_violation`` in the enum; this is the one
#: error the robot has for a commanded torque outside the joint's range ("If
#: the **torque sensor limit** is reached, a ``tau_j_range_violation`` will be
#: triggered", ``docs/overview.rst``).
TAU_J_RANGE_VIOLATION_INDEX = 34

#: Wire name for each index we can latch, as ``getErrorName`` spells it
#: (``error.h:52-138``). Used in the log lines so a message can be grepped
#: straight against libfranka's own vocabulary.
ERROR_NAMES = {
    JOINT_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX: (
        "joint_position_motion_generator_start_pose_invalid"
    ),
    JOINT_MOTION_GENERATOR_POSITION_LIMITS_VIOLATION_INDEX: (
        "joint_motion_generator_position_limits_violation"
    ),
    JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX: (
        "joint_motion_generator_velocity_limits_violation"
    ),
    JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX: (
        "joint_motion_generator_velocity_discontinuity"
    ),
    JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX: (
        "joint_motion_generator_acceleration_discontinuity"
    ),
    CARTESIAN_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX: (
        "cartesian_position_motion_generator_start_pose_invalid"
    ),
    CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX: (
        "cartesian_motion_generator_velocity_limits_violation"
    ),
    CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX: (
        "cartesian_motion_generator_velocity_discontinuity"
    ),
    CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX: (
        "cartesian_motion_generator_acceleration_discontinuity"
    ),
    CONTROLLER_TORQUE_DISCONTINUITY_INDEX: "controller_torque_discontinuity",
    TAU_J_RANGE_VIOLATION_INDEX: "tau_J_range_violation",
}

# -- sim-side choices ---------------------------------------------------------

#: How far the first commanded position of a joint-position motion may sit from
#: the robot's ``q_d`` before it is ``joint_position_motion_generator_start_pose_invalid``.
#:
#: **Not a libfranka constant.** The check lives in Control, not in libfranka,
#: and the tolerance is not published anywhere in the v10 tree or its docs --
#: only the remedy is ("make sure that your control loop starts with the last
#: commanded value observed in the robot state", ``docs/overview.rst``). 0.1 rad
#: is a sim choice: loose enough that the simulator's own position-tracking
#: error can never manufacture the error, tight enough that a client which
#: jumps into a motion from a stale or hard-coded pose is caught. Override it
#: per-instance if your client needs a different contract.
START_POSE_TOLERANCE = 0.1

#: Same idea for a velocity generator's first ``dq_c`` against the reported
#: ``dq_d``, in rad/s. Not a libfranka constant either; the enum has no
#: joint-*velocity* start error, so exceeding this latches
#: ``joint_motion_generator_velocity_discontinuity`` -- which is what a step
#: away from ``dq_d`` physically is.
START_VELOCITY_TOLERANCE = 0.1

#: Opt-in switch for the abort. Validation and the rate-limited warning always
#: run; setting this to a truthy value additionally makes a violation latch the
#: error, stop the motion and refuse the offending command, the way the robot
#: does. Off by default because a sim is routinely driven by scripted clients
#: that step their targets.
ENFORCE_ENV_VAR = "FRANKA_SIM_ENFORCE_MOTION_LIMITS"

#: Spellings of :data:`ENFORCE_ENV_VAR` that turn enforcement on. An allow-list,
#: not a deny-list: "set, therefore on" would read ``=disabled``
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
    """Whether motion-limit aborts are on, per :data:`ENFORCE_ENV_VAR`."""
    env = os.environ if environ is None else environ
    return _is_truthy(env.get(ENFORCE_ENV_VAR, ""))


#: Most 1 ms cycles a single received command may be differenced over.
#:
#: The interval comes from the client's own echoed ``message_id`` (see
#: :meth:`MotionLimitChecker.cycles_since_applied`), and a client is not a
#: trustworthy source for the denominator of the server's own limit check:
#: inflating the echo by a thousand divides every commanded derivative by a
#: thousand, and 50 rad/s steps sail through. Two datagrams landing in one poll
#: is the only reason the sim ever legitimately sees more than one cycle, so
#: three is already generous. Beyond it the extra cycles are the client's
#: claim, not the server's observation, and are not honoured.
MAX_COALESCED_CYCLES = 3


# -- limit helpers ------------------------------------------------------------


def upper_joint_velocity_limits(joint_positions: Sequence[float]) -> List[float]:
    """Per-joint upper velocity limit at ``joint_positions``.

    ``JointVelocityLimitsConfig::getUpperJointVelocityLimits``
    (``src/joint_velocity_limits.cpp:116-130``)::

        min(max_velocity,
            max(0, -velocity_offset
                   + sqrt(max(0, 2 * deceleration_limit * (upper_position_limit - q)))))
        - kJointVelocityLimitsTolerance

    The result is clamped up to zero: right at the joint's position limit the
    raw formula goes slightly *negative* (it is a rate-limiter target, and the
    limiter's job there is to decelerate), and a sim that called a commanded
    zero velocity an error would be wrong about the one command that is always
    safe. The position-limit check catches that configuration anyway.
    """
    limits = []
    for index, position in enumerate(joint_positions[:7]):
        max_velocity, offset, deceleration = POSITION_BASED_VELOCITY_LIMITS[index]
        upper = JOINT_POSITION_LIMITS[index][1]
        ramp = math.sqrt(max(0.0, 2.0 * deceleration * (upper - position)))
        raw = min(max_velocity, max(0.0, -offset + ramp)) - JOINT_VELOCITY_LIMITS_TOLERANCE[index]
        limits.append(max(0.0, raw))
    return limits


def lower_joint_velocity_limits(joint_positions: Sequence[float]) -> List[float]:
    """Per-joint lower velocity limit at ``joint_positions``.

    The mirror of :func:`upper_joint_velocity_limits`
    (``src/joint_velocity_limits.cpp:132-146``), clamped down to zero for the
    same reason.
    """
    limits = []
    for index, position in enumerate(joint_positions[:7]):
        max_velocity, offset, deceleration = POSITION_BASED_VELOCITY_LIMITS[index]
        lower = JOINT_POSITION_LIMITS[index][0]
        ramp = math.sqrt(max(0.0, 2.0 * deceleration * (position - lower)))
        raw = max(-max_velocity, min(0.0, offset - ramp)) + JOINT_VELOCITY_LIMITS_TOLERANCE[index]
        limits.append(min(0.0, raw))
    return limits


# -- violation reporting ------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """One limit that one commanded value broke.

    ``error_index`` is the bit to latch in ``errors``/``reflex_reason``;
    everything else exists so the log line can name the joint or axis, the value
    and the limit, which is what makes an unenforced warning actionable.
    """

    error_index: int
    signal: str
    axis: str
    value: float
    limit: float
    unit: str = ""
    #: Refuse the command even with enforcement off. Only the non-finite class
    #: sets this: "reported but applied" is a defensible choice for a signal
    #: that is merely too fast, and no choice at all for one that is NaN --
    #: applying it puts NaN into the physics state and onto the wire, where it
    #: is not permissive, just broken.
    fatal: bool = False

    @property
    def error_name(self) -> str:
        """The wire name libfranka's ``getErrorName`` would print."""
        return ERROR_NAMES[self.error_index]

    def describe(self) -> str:
        """One line naming the error, the axis, the value and the limit."""
        unit = f" {self.unit}" if self.unit else ""
        return (
            f"{self.error_name}: {self.signal} {self.axis} = {self.value:.6g}{unit}, "
            f"limit {self.limit:.6g}{unit}"
        )


# -- differencing -------------------------------------------------------------


class _Differentiator:
    """Backward-Euler differences of one commanded signal, cycle by cycle.

    Holds the last applied value and its first two derivatives, which is
    everything the three difference formulas in ``docs/overview.rst`` need.
    Used at three depths:

    * joint **position** -- value ``q_c``; ``first``/``second``/``third`` are
      velocity, acceleration and jerk,
    * joint **velocity** -- value ``dq_c``; ``first``/``second`` are
      acceleration and jerk,
    * **torque** -- value ``tau_J_d``; ``first`` is the torque rate.
    """

    def __init__(self, width: int = 7):
        """Start at rest: value, first and second derivative all zero."""
        self.width = width
        self.value = [0.0] * width
        self.first = [0.0] * width
        self.second = [0.0] * width

    def seed(
        self,
        value: Sequence[float],
        first: Optional[Sequence[float]] = None,
        second: Optional[Sequence[float]] = None,
    ) -> None:
        """Set the history a motion starts from (the robot's own ``*_d`` fields)."""
        self.value = [float(item) for item in value[: self.width]]
        self.first = (
            [0.0] * self.width if first is None else [float(x) for x in first[: self.width]]
        )
        self.second = (
            [0.0] * self.width if second is None else [float(x) for x in second[: self.width]]
        )

    def derivatives(
        self, command: Sequence[float], cycles: int = 1
    ) -> Tuple[List[float], List[float], List[float]]:
        """The three backward differences ``command`` implies, without advancing.

        ``cycles`` is how many 1 ms cycles separate ``command`` from the value
        in the history; see :meth:`MotionLimitChecker.cycles_since_applied` for
        why that is not always one in a simulator.
        """
        step = cycles * DELTA_T
        first = [(command[i] - self.value[i]) / step for i in range(self.width)]
        second = [(first[i] - self.first[i]) / step for i in range(self.width)]
        third = [(second[i] - self.second[i]) / step for i in range(self.width)]
        return first, second, third

    def advance(self, command: Sequence[float], cycles: int = 1) -> None:
        """Accept ``command`` as applied: it and its derivatives become the history."""
        first, second, _ = self.derivatives(command, cycles)
        self.value = [float(command[i]) for i in range(self.width)]
        self.first = first
        self.second = second

    def rebase(self, command: Sequence[float]) -> None:
        """Accept ``command`` as a fresh standstill: derivatives reset to zero."""
        self.value = [float(command[i]) for i in range(self.width)]
        self.first = [0.0] * self.width
        self.second = [0.0] * self.width


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


class _CartesianDifferentiator:
    """The same, for a 6-element twist split into translation and rotation.

    ``limitRate`` treats ``O_dP_EE_c`` as two ``Eigen::Vector3d`` and compares
    *norms*, not components (``src/rate_limiting.cpp:184-195`` dispatching into
    the anonymous-namespace overload at ``:13-55``), so this differences the
    whole twist and reports norms per half.
    """

    def __init__(self):
        """Start from a standstill twist with zero acceleration."""
        self.value = [0.0] * 6
        self.first = [0.0] * 6

    def seed(self, value: Sequence[float], first: Optional[Sequence[float]] = None) -> None:
        """Set the twist (and its acceleration) a motion starts from."""
        self.value = [float(item) for item in value[:6]]
        self.first = [0.0] * 6 if first is None else [float(item) for item in first[:6]]

    def derivatives(
        self, command: Sequence[float], cycles: int = 1
    ) -> Tuple[List[float], List[float]]:
        """Acceleration and jerk implied by ``command``, without advancing."""
        step = cycles * DELTA_T
        acceleration = [(command[i] - self.value[i]) / step for i in range(6)]
        jerk = [(acceleration[i] - self.first[i]) / step for i in range(6)]
        return acceleration, jerk

    def advance(self, command: Sequence[float], cycles: int = 1) -> None:
        """Accept ``command`` as applied."""
        acceleration, _ = self.derivatives(command, cycles)
        self.value = [float(command[i]) for i in range(6)]
        self.first = acceleration


# -- the checker --------------------------------------------------------------


class MotionLimitChecker:
    """Validates each received command against the FCI's motion limits.

    One instance per FCI session, mirroring :class:`
    franka_sim.comm_constraints.CommConstraintTracker`: :meth:`start_motion` on
    ``Move``, :meth:`end_motion` when the motion ends however it ends,
    :meth:`recover` on ``AutomaticErrorRecovery``.

    Two threads touch it -- the UDP receive thread checks and records received
    commands, the state-publish thread reports missed cycles -- so every method
    goes through one short leaf lock, the same discipline the comm tracker
    uses.
    """

    def __init__(
        self,
        *,
        enforce: bool = False,
        start_pose_tolerance: float = START_POSE_TOLERANCE,
        start_velocity_tolerance: float = START_VELOCITY_TOLERANCE,
    ):
        """Build a checker; see the module constants for the defaults."""
        self._lock = threading.Lock()
        #: Whether a violation may abort the motion. Checking and logging run
        #: either way -- that is the point of making this opt-in.
        self.enforce = enforce
        self.start_pose_tolerance = start_pose_tolerance
        self.start_velocity_tolerance = start_velocity_tolerance

        self._mode: ControlMode = ControlMode.NONE
        self._joint = _Differentiator(7)
        self._torque = _Differentiator(7)
        self._twist = _CartesianDifferentiator()
        #: Reference configuration for the position-dependent velocity limits.
        self._joint_positions = [0.0] * 7
        self._active = False
        self._first_command = True
        self._violated = False
        #: Caller-supplied token for the running motion; see :meth:`start_motion`.
        self._motion_id = 0
        #: ``message_id`` of the state the applied history currently sits at;
        #: see :meth:`cycles_since_applied`.
        self._applied_id: Optional[int] = None
        #: Error indices already logged in this motion, so a client that steps
        #: every cycle produces one warning, not a thousand.
        self._logged: Set[int] = set()

    # -- lifecycle ---------------------------------------------------------

    def start_motion(
        self, control_mode: ControlMode, state: Dict[str, Any], motion_id: int = 0
    ) -> None:
        """Arm a motion, seeding the history from the robot's own state.

        Exactly what Control differentiates against: ``q_d``, ``dq_d``,
        ``ddq_d`` and ``tau_J_d`` are "always sent back to the user in the robot
        state" for precisely this purpose (``docs/overview.rst``), so seeding
        from them is seeding from what the client was told.

        ``motion_id`` is the caller's session token; it gates :meth:`end_motion`
        so a stale end cannot switch off a motion that has already replaced this
        one, and it lets the caller check that an abort still applies to the
        motion that violated. The latched violation is re-armed here, the same
        way :meth:`franka_sim.comm_constraints.CommConstraintTracker.start_motion`
        re-arms its own: enforcement used to fire at most once per *connection*.
        The server refuses a ``Move`` outright while a violation is latched, so
        reaching this method at all means the client has recovered.
        """
        with self._lock:
            self._mode = control_mode
            self._active = control_mode not in (ControlMode.NONE, None)
            self._motion_id = motion_id
            self._first_command = True
            self._applied_id = None
            self._violated = False
            self._logged = set()
            positions = list(state.get("q_d") or state.get("q") or [0.0] * 7)
            self._joint_positions = [float(value) for value in positions[:7]]
            if control_mode is ControlMode.POSITION:
                self._joint.seed(positions, state.get("dq_d"), state.get("ddq_d"))
            elif control_mode is ControlMode.VELOCITY:
                self._joint.seed(state.get("dq_d") or [0.0] * 7, state.get("ddq_d"))
            self._torque.seed(state.get("tau_J_d") or [0.0] * 7)
            self._twist.seed(state.get("O_dP_EE_d") or [0.0] * 6)

    def end_motion(self, motion_id: Optional[int] = None) -> None:
        """The motion is over. The latched violation survives, as the robot's does.

        ``motion_id`` names the motion the caller believes is ending; a token
        that no longer matches the running one is a stale event and is ignored.
        ``None`` means "whatever is running", which is what the unconditional
        teardown paths want.
        """
        with self._lock:
            if motion_id is not None and motion_id != self._motion_id:
                return
            self._active = False
            self._motion_id = 0
            self._first_command = True

    def recover(self) -> None:
        """Clear a latched violation, as ``AutomaticErrorRecovery`` does."""
        with self._lock:
            self._violated = False
            self._active = False
            self._motion_id = 0
            self._first_command = True
            self._logged = set()

    # -- cycle accounting --------------------------------------------------

    def cycles_since_applied(self, command: Dict[str, Any]) -> int:
        """How many 1 ms cycles separate ``command`` from the applied history.

        On hardware this is always one: Control applies exactly one command per
        cycle and extrapolates the ones that never arrive, so ``q_{c,k-1}`` is
        always precisely one millisecond old. A simulator has two ways to break
        that which the robot does not. The UDP receive loop drains the socket
        and applies only the newest datagram, so when two of a client's commands
        land in the same poll the older one is discarded outright -- neither
        applied nor counted as a lost cycle. And a missed cycle here *holds* the
        last command rather than extrapolating it, so the history does not move
        on while the client is quiet.

        Dividing a two-cycle step by one millisecond would report double the
        velocity and a large acceleration for a *conforming* client, which is a
        sim artifact and not a limit the robot would have flagged. The echoed
        ``message_id`` says how many state cycles the command has travelled, so
        the difference is taken over that interval instead -- capped at
        :data:`MAX_COALESCED_CYCLES`, because past a cycle or two of coalescing
        the interval is no longer explaining a sim artifact, it is diluting a
        limit check with a number the client chose.

        The cap is why a genuine packet gap gets its grace cycle instead (see
        :meth:`check`): over a long gap the honest interval is longer than the
        cap allows, and judging the resumed command over a capped one would
        manufacture the very error the gap caused.
        """
        with self._lock:
            return self._cycles_since_applied_locked(command)

    #: Which command fields each control mode actually reads. Anything else in
    #: the datagram is not this motion's signal and is not the client's problem.
    _CHECKED_FIELDS = {
        ControlMode.POSITION: (("q_c", JOINT_MOTION_GENERATOR_POSITION_LIMITS_VIOLATION_INDEX),),
        ControlMode.VELOCITY: (("dq_c", JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX),),
        ControlMode.TORQUE: (("tau_J_d", TAU_J_RANGE_VIOLATION_INDEX),),
        ControlMode.STEERING_DRIVE: (
            ("O_dP_EE_c", CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX),
        ),
    }

    def _check_finite_locked(self, command: Dict[str, Any]) -> Optional[Violation]:
        """Reject NaN and infinity before any limit is compared.

        Comparisons against NaN are all False, so a NaN sails through every
        ``value > limit`` test in this module, poisons the backward differences
        it is then recorded into (NaN minus anything is NaN, for the rest of the
        motion) and reaches both the physics backend and the wire.

        libfranka refuses to send one at all -- its own command filter throws
        ``std::invalid_argument("lowpass-filter: current or last signal value is
        infinite or NaN.")`` (``src/lowpass_filter.cpp``) -- so a non-finite
        command can only come from a client that is not libfranka. There is no
        ``Error`` enumerator for it, so it is reported as the limits violation
        of the generator it arrived in, which is what a value of infinite
        magnitude is. Marked :attr:`Violation.fatal`, so it is refused whether
        or not enforcement is on.
        """
        # Dispatched on the checker's own ``_mode``, which the ``Move`` handler
        # set, while ``_dispatch_control_command`` reads the mode pair out of
        # ``robot_state``. The two agree except for the sub-millisecond window
        # in which a ``Move`` is being handled, where the checker may still be
        # looking at the previous generator's field. Harmless here: the fields
        # are all present in every datagram, and the fallback below checks the
        # lot when there is no mode at all.
        fields = self._CHECKED_FIELDS.get(self._mode)
        if fields is None:
            # No motion (or one this server does not serve): check everything a
            # dispatcher could plausibly reach for rather than nothing.
            fields = tuple(
                pair for pairs in self._CHECKED_FIELDS.values() for pair in pairs
            )
        for field, error_index in fields:
            values = command.get(field)
            if values is None:
                continue
            for axis, value in enumerate(values):
                if not math.isfinite(value):
                    return Violation(
                        error_index=error_index,
                        signal=field,
                        axis=f"[{axis}]",
                        value=value,
                        limit=math.inf,
                        fatal=True,
                    )
        return None

    def _cycles_since_applied_locked(self, command: Dict[str, Any]) -> int:
        if self._applied_id is None:
            return 1
        cycles = int(command.get("message_id", 0)) - self._applied_id
        # Clamped: the interval is the client's own claim, and an inflated one
        # would divide every derivative down to nothing. See MAX_COALESCED_CYCLES.
        return max(1, min(MAX_COALESCED_CYCLES, cycles))

    # -- checking ----------------------------------------------------------

    def check(
        self,
        command: Dict[str, Any],
        joint_positions: Optional[Sequence[float]] = None,
        *,
        fresh: bool = True,
    ) -> Optional[Violation]:
        """Validate one *received* command; the first violation found, or None.

        ``joint_positions`` is the configuration the position-dependent joint
        velocity limits are evaluated at -- the measured ``q``, which is what
        libfranka's own callers pass ``getUpperJointVelocityLimits``. Omitted,
        the last applied commanded position is used, which is where the robot
        is heading anyway.

        **Every command that will be applied is checked, in full.** There is no
        category of command that reaches the simulator unexamined -- that was a
        hole, and a 40 rad/s step walked through it behind a stale echo.

        ``fresh`` is False for a datagram that did not answer the cycle it
        arrived in -- a duplicate, a replay, a reordered packet. It is still
        applied (see ``FrankaSimServer._handle_commands``), so it is still
        judged; what changes is only the *interval*. Its echoed id says nothing
        trustworthy about how far it has travelled, so it is differenced over a
        single cycle, which is the strictest reading and the one the robot would
        take of a packet arriving in a cycle it does not belong to. It is never
        recorded, so it leaves the history alone.

        There is **no grace cycle**. A command that follows a packet gap is
        differenced over the interval its echoed ``message_id`` reports, capped
        at :data:`MAX_COALESCED_CYCLES` -- so a genuine one-to-three cycle gap,
        which is what the sim's own loss looks like, is measured at the rate the
        client actually commanded and passes. Skipping the differential checks
        for that command instead was a hole of its own: the resume waypoint
        could be anywhere in the joint range, and a full-range teleport reached
        physics with the checker reporting no violation at all.

        The cap is why a *long* gap can still trip a discontinuity. That is the
        honest consequence of this sim holding the last command instead of
        extrapolating it (see :mod:`franka_sim.comm_constraints`): the robot
        would have carried the trajectory forward and the client would resume on
        a signal it was already tracking, while here the history stays where the
        gap began. libfranka warns about the general shape of this -- intermittent
        drops "could trigger `discontinuity` errors even when your source signals
        conform with the interface specification" (``docs/overview.rst``).

        Records nothing: a caller that decides to reject the command leaves no
        trace of it in the history. It does update the reference configuration
        the position-dependent velocity limits are evaluated at, which is the
        one piece of state it owns -- so this is idempotent rather than pure.
        """
        with self._lock:
            non_finite = self._check_finite_locked(command)
            if non_finite is not None:
                return non_finite
            if not self._active:
                return None
            if joint_positions is not None:
                # A non-finite *measured* position is the simulator's problem,
                # not the client's. Substituting zero used to put joint 4
                # outside its range, which drives its velocity limit to 0.0 and
                # aborts the client's next perfectly good command; keeping the
                # previous reference configuration blames nobody.
                candidate = [float(value) for value in joint_positions[:7]]
                if all(math.isfinite(value) for value in candidate):
                    self._joint_positions = candidate
                else:
                    logger.debug(
                        "Ignoring a non-finite measured joint position for the "
                        "velocity-limit reference configuration"
                    )

            cycles = self._cycles_since_applied_locked(command) if fresh else 1
            if self._mode is ControlMode.POSITION:
                return self._check_position(command, cycles)
            if self._mode is ControlMode.VELOCITY:
                return self._check_velocity(command, cycles)
            if self._mode is ControlMode.TORQUE:
                return self._check_torque(command, cycles)
            if self._mode is ControlMode.STEERING_DRIVE:
                return self._check_cartesian_velocity(command, cycles)
            return None

    def record(self, command: Dict[str, Any]) -> None:
        """Accept one received command as applied.

        Only commands the caller accepted, and only *fresh* ones: a stale or
        duplicated echo is not a later sample of the client's trajectory, and
        differencing one produces a velocity and an acceleration nobody
        commanded.

        A command covers however many state cycles its ``message_id`` says it
        does, capped (see :meth:`cycles_since_applied`).
        """
        with self._lock:
            if not self._active:
                return
            first = self._first_command
            self._first_command = False

            cycles = self._cycles_since_applied_locked(command)
            self._applied_id = int(command.get("message_id", 0))

            if self._mode is ControlMode.POSITION:
                # The first waypoint of a motion is a standstill by
                # construction (the start-pose check has just confirmed it sits
                # where the robot is), so it rebases the history rather than
                # being differenced against a pose the client never commanded.
                if first:
                    self._joint.rebase(command["q_c"])
                else:
                    self._joint.advance(command["q_c"], cycles)
            elif self._mode is ControlMode.VELOCITY:
                self._joint.advance(command["dq_c"], cycles)
            elif self._mode is ControlMode.TORQUE:
                self._torque.advance(command["tau_J_d"], cycles)
            elif self._mode is ControlMode.STEERING_DRIVE:
                self._twist.advance(command["O_dP_EE_c"], cycles)

    # -- per-generator checks ----------------------------------------------

    def _check_position(self, command: Dict[str, Any], cycles: int) -> Optional[Violation]:
        q_c = command["q_c"]
        violation = self._check_joint_position_limits(q_c)
        if violation is not None:
            return violation

        if self._first_command:
            # A motion's first command must be where the robot already is;
            # differencing it against ``q_d`` would report the *step into the
            # motion*, which is a start-pose error, not a velocity one.
            return self._check_start_pose(q_c)

        velocity, acceleration, jerk = self._joint.derivatives(q_c, cycles)
        return (
            self._check_joint_velocity_limits(velocity, "q_c")
            or self._check_per_joint(
                acceleration,
                MAX_JOINT_ACCELERATION,
                JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
                "q_c",
                "rad/s^2",
            )
            or self._check_per_joint(
                jerk,
                MAX_JOINT_JERK,
                JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
                "q_c",
                "rad/s^3",
            )
        )

    def _check_velocity(self, command: Dict[str, Any], cycles: int) -> Optional[Violation]:
        # No joint *range* check here, deliberately: a velocity generator
        # commands a rate, and where that rate will put the joint is only
        # knowable by integrating it, which is the physics backend's job, not
        # this module's. The range is not unguarded, though -- the
        # position-dependent velocity limit shrinks to zero as the joint
        # approaches its stop (``upper_joint_velocity_limits``), so a client
        # that keeps commanding into a limit is caught as a *velocity* limits
        # violation, which is also what the real robot reports.
        dq_c = command["dq_c"]
        violation = self._check_joint_velocity_limits(dq_c, "dq_c")
        if violation is not None:
            return violation

        if self._first_command:
            # A velocity motion must continue from the ``dq_d`` the robot last
            # reported. The enum has no joint-velocity start error, so a step
            # away from it is reported as what it is: a velocity discontinuity.
            return self._check_start_velocity(dq_c)

        acceleration, jerk, _ = self._joint.derivatives(dq_c, cycles)
        return self._check_per_joint(
            acceleration,
            MAX_JOINT_ACCELERATION,
            JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
            "dq_c",
            "rad/s^2",
        ) or self._check_per_joint(
            jerk,
            MAX_JOINT_JERK,
            JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
            "dq_c",
            "rad/s^3",
        )

    def _check_torque(self, command: Dict[str, Any], cycles: int) -> Optional[Violation]:
        tau = command["tau_J_d"]
        violation = self._check_per_joint(
            tau, MAX_TORQUE, TAU_J_RANGE_VIOLATION_INDEX, "tau_J_d", "Nm"
        )
        if violation is not None:
            return violation

        rate, _, _ = self._torque.derivatives(tau, cycles)
        return self._check_per_joint(
            rate, MAX_TORQUE_RATE, CONTROLLER_TORQUE_DISCONTINUITY_INDEX, "tau_J_d", "Nm/s"
        )

    def _check_cartesian_velocity(
        self, command: Dict[str, Any], cycles: int
    ) -> Optional[Violation]:
        twist = command["O_dP_EE_c"]
        acceleration, jerk = self._twist.derivatives(twist, cycles)
        checks = (
            (
                twist,
                MAX_TRANSLATIONAL_VELOCITY,
                MAX_ROTATIONAL_VELOCITY,
                CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
                ("m/s", "rad/s"),
            ),
            (
                acceleration,
                MAX_TRANSLATIONAL_ACCELERATION,
                MAX_ROTATIONAL_ACCELERATION,
                CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
                ("m/s^2", "rad/s^2"),
            ),
            (
                jerk,
                MAX_TRANSLATIONAL_JERK,
                MAX_ROTATIONAL_JERK,
                CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
                ("m/s^3", "rad/s^3"),
            ),
        )
        for values, translational, rotational, index, units in checks:
            halves = (
                ("translational", _norm(values[:3]), translational, units[0]),
                ("rotational", _norm(values[3:6]), rotational, units[1]),
            )
            for axis, norm, limit, unit in halves:
                # The kNormEps guard is inert here and kept only to mirror
                # limitRate's shape: every limit is orders of magnitude above
                # machine epsilon, so ``norm > limit`` already implies it. In
                # libfranka it guards a *division* by the norm; nothing here
                # divides.
                if norm > NORM_EPS and norm > limit:
                    return Violation(index, "O_dP_EE_c", f"{axis} norm", norm, limit, unit)
        return None

    # -- primitives --------------------------------------------------------

    def _check_joint_position_limits(self, q_c: Sequence[float]) -> Optional[Violation]:
        for index in range(7):
            lower, upper = JOINT_POSITION_LIMITS[index]
            if q_c[index] > upper or q_c[index] < lower:
                limit = upper if q_c[index] > upper else lower
                return Violation(
                    JOINT_MOTION_GENERATOR_POSITION_LIMITS_VIOLATION_INDEX,
                    "q_c",
                    f"joint {index + 1}",
                    float(q_c[index]),
                    limit,
                    "rad",
                )
        return None

    def _check_joint_velocity_limits(
        self, velocity: Sequence[float], signal: str
    ) -> Optional[Violation]:
        upper = upper_joint_velocity_limits(self._joint_positions)
        lower = lower_joint_velocity_limits(self._joint_positions)
        for index in range(7):
            if velocity[index] > upper[index] or velocity[index] < lower[index]:
                limit = upper[index] if velocity[index] > upper[index] else lower[index]
                return Violation(
                    JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
                    signal,
                    f"joint {index + 1}",
                    float(velocity[index]),
                    limit,
                    "rad/s",
                )
        return None

    def _check_per_joint(
        self,
        values: Sequence[float],
        limits: Sequence[float],
        error_index: int,
        signal: str,
        unit: str,
    ) -> Optional[Violation]:
        for index in range(7):
            if abs(values[index]) > limits[index]:
                return Violation(
                    error_index,
                    signal,
                    f"joint {index + 1}",
                    float(values[index]),
                    limits[index],
                    unit,
                )
        return None

    def _check_start_pose(self, q_c: Sequence[float]) -> Optional[Violation]:
        for index in range(7):
            offset = q_c[index] - self._joint.value[index]
            if abs(offset) > self.start_pose_tolerance:
                return Violation(
                    JOINT_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX,
                    "q_c",
                    f"joint {index + 1}",
                    float(offset),
                    self.start_pose_tolerance,
                    "rad from q_d",
                )
        return None

    def _check_start_velocity(self, dq_c: Sequence[float]) -> Optional[Violation]:
        for index in range(7):
            offset = dq_c[index] - self._joint.value[index]
            if abs(offset) > self.start_velocity_tolerance:
                return Violation(
                    JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
                    "dq_c",
                    f"joint {index + 1}",
                    float(offset),
                    self.start_velocity_tolerance,
                    "rad/s from dq_d",
                )
        return None

    # -- reporting ---------------------------------------------------------

    def applied_derivatives(self) -> Optional[Tuple[List[float], List[float]]]:
        """The first two derivatives of the last applied command, or None.

        These are exactly the fields the FCI publishes back: "``q_{c,k-1}``,
        ``dq_{c,k-1}`` and ``ddq_{c,k-1}`` are always sent back to the user in
        the robot state as ``q_d``, ``dq_d`` and ``ddq_d`` so you will be able
        to compute the resulting derivatives in advance, even in case of packet
        losses" (``docs/overview.rst``). The checker differences every applied
        command anyway, so the numbers already exist here; handing them to the
        server is what lets a client predict what this module will compute.

        For a position generator that is ``(dq_d, ddq_d)``; for a velocity
        generator ``(ddq_d, jerk)``; for the torque controller ``(dtau, ...)``.
        """
        with self._lock:
            if not self._active:
                return None
            if self._mode in (ControlMode.POSITION, ControlMode.VELOCITY):
                return list(self._joint.first), list(self._joint.second)
            if self._mode is ControlMode.TORQUE:
                return list(self._torque.first), list(self._torque.second)
            return None

    def report(self, violation: Violation, *, enforced: bool) -> None:
        """Log ``violation`` once per error class per motion, naming the limit.

        The always-on half of the feature: even with enforcement off, a client
        that commands something the robot would refuse gets told which joint or
        axis, what it sent and what the limit was.
        """
        if not self.should_log(violation):
            return
        logger.warning(
            "motion limit violated: %s%s",
            violation.describe(),
            "" if enforced else " (not enforced)",
        )

    def should_log(self, violation: Violation) -> bool:
        """True the first time this motion sees ``violation``'s error class.

        Rate limiting by error *type* rather than by time: a client that steps
        its target every cycle would otherwise write a thousand identical lines
        a second, and the second one says nothing the first did not.
        """
        with self._lock:
            if violation.error_index in self._logged:
                return False
            self._logged.add(violation.error_index)
            return True

    def latch(self) -> None:
        """Mark a violation as enforced, so only the first one aborts."""
        with self._lock:
            self._violated = True

    @property
    def violated(self) -> bool:
        """Whether a violation is latched (cleared only by :meth:`recover`)."""
        with self._lock:
            return self._violated

    @property
    def active(self) -> bool:
        """Whether a motion this checker validates is running."""
        with self._lock:
            return self._active

    @property
    def motion_id(self) -> int:
        """Token of the running motion, 0 when none is."""
        with self._lock:
            return self._motion_id
