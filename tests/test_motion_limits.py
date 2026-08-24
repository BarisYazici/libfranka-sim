"""FCI motion limits: joint ranges, discontinuities, torque rate, and the abort.

The real FCI differentiates every commanded signal with backward Euler at the
1 ms cycle and stops the motion when the result leaves the published limits --
``joint_motion_generator_velocity_discontinuity`` for an acceleration step,
``..._acceleration_discontinuity`` for a jerk step, ``controller_torque_discontinuity``
for a torque step, and so on (libfranka ``docs/overview.rst``). This file covers
franka-sim's emulation of that, in the three layers a regression would show up
in:

* unit: :class:`franka_sim.motion_limits.MotionLimitChecker` on its own -- the
  difference formulas, every boundary, the start conditions, and the capped
  interval a command is differenced over when the sim's own packet loss delays
  it,
* server over the wire with a mocked simulator: a jump logs but does not abort
  by default; enforced, it aborts with the right bit per violation class, never
  reaches the simulator, and error recovery clears it,
* end to end over real physics: a smooth sine that must complete clean, and a
  deliberate 0.5 rad step that must not.

Enforcement is off by default (see :data:`ENFORCE_ENV_VAR`), so the tests that
want the abort ask for it explicitly.
"""

import logging
import math
import select
import socket
import struct
import threading
import time

import numpy as np
import pytest

from franka_sim.control_modes import ControlMode
from franka_sim.franka_protocol import (
    COMMAND_PORT,
    Command,
    ConnectStatus,
    ControllerMode,
    MessageHeader,
    MotionGeneratorMode,
    MoveStatus,
    RobotMode,
)
from franka_sim.motion_limits import (
    CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
    CARTESIAN_MOTION_GENERATOR_ELBOW_LIMIT_VIOLATION_INDEX,
    CARTESIAN_MOTION_GENERATOR_ELBOW_SIGN_INCONSISTENT_INDEX,
    CARTESIAN_MOTION_GENERATOR_START_ELBOW_INVALID_INDEX,
    CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
    CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
    CARTESIAN_POSITION_MOTION_GENERATOR_INVALID_FRAME_INDEX,
    CARTESIAN_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX,
    CARTESIAN_VELOCITY_VIOLATION_INDEX,
    CONTROLLER_TORQUE_DISCONTINUITY_INDEX,
    DELTA_T,
    ENFORCE_ENV_VAR,
    JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
    JOINT_MOTION_GENERATOR_POSITION_LIMITS_VIOLATION_INDEX,
    JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
    JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
    JOINT_POSITION_LIMITS,
    JOINT_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX,
    JOINT_VELOCITY_VIOLATION_INDEX,
    MAX_COALESCED_CYCLES,
    MAX_ELBOW_ACCELERATION,
    MAX_ELBOW_VELOCITY,
    MAX_JOINT_ACCELERATION,
    MAX_JOINT_JERK,
    MAX_ROTATIONAL_VELOCITY,
    MAX_TORQUE,
    MAX_TORQUE_RATE,
    MAX_TRANSLATIONAL_ACCELERATION,
    MAX_TRANSLATIONAL_JERK,
    MAX_TRANSLATIONAL_VELOCITY,
    MEASURED_CARTESIAN_VELOCITY_LIMIT,
    MEASURED_JOINT_VELOCITY_MARGIN,
    SINGULAR_POSE_MIN_SINGULAR_VALUE,
    TAU_J_RANGE_VIOLATION_INDEX,
    START_CARTESIAN_POSE_ROTATION_TOLERANCE,
    START_CARTESIAN_POSE_TRANSLATION_TOLERANCE,
    START_ELBOW_TOLERANCE,
    MotionLimitChecker,
    Violation,
    _Differentiator,
    enforcement_enabled_by_env,
    is_homogeneous_transformation,
    is_singular_configuration,
    lower_joint_velocity_limits,
    rotation_log,
    smallest_singular_value,
    upper_joint_velocity_limits,
)
from franka_sim.robot_state import _ROBOT_STATE_PACKER, RobotState

#: A configuration comfortably inside every joint's range, so the
#: position-dependent velocity limits sit at their flat caps. Joints 4 and 6 do
#: not straddle zero, hence the non-zero entries.
HOME = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]

#: A measured flange pose that is nothing like identity, column-major as the
#: wire wants it -- the same shape ``MEASURED_POSE`` in
#: test_cartesian_velocity_dispatch.py uses, so "seeded from the robot's own
#: pose" cannot pass by accident of the checker's identity default.
MOCK_O_T_EE = [
    0.0, 0.0, -1.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0,
    0.31, -0.02, 0.48, 1.0,
]


def command(**fields):
    """A RobotCommand dict shaped exactly as the UDP receive path builds one."""
    base = {
        "message_id": 1,
        "q_c": [0.0] * 7,
        "dq_c": [0.0] * 7,
        "O_T_EE_c": [0.0] * 16,
        "O_dP_EE_c": [0.0] * 6,
        "elbow_c": [0.0] * 2,
        "valid_elbow": False,
        "motion_generation_finished": False,
        "tau_J_d": [0.0] * 7,
        "torque_command_finished": False,
    }
    base.update(fields)
    return base


def robot_state_at(q=None, **fields):
    """The robot-state fields a Move seeds the checker from."""
    state = {
        "q": list(HOME if q is None else q),
        "q_d": list(HOME if q is None else q),
        "dq_d": [0.0] * 7,
        "ddq_d": [0.0] * 7,
        "tau_J_d": [0.0] * 7,
        "O_dP_EE_d": [0.0] * 6,
    }
    state.update(fields)
    return state


def position_checker(**state_fields):
    """A checker armed for a joint-position motion from :data:`HOME`."""
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.POSITION, robot_state_at(**state_fields))
    return checker


def drive_position_history(checker, position, velocity, acceleration, start_id=1):
    """Record waypoints until the applied history is exactly what is asked for.

    Solved backwards out of the difference formulas: the waypoint before
    ``position`` is ``position - velocity * dt``, and the velocity before
    ``velocity`` is ``velocity - acceleration * dt``. Four records do it -- the
    first rebases the history to a standstill (that is what a motion's opening
    waypoint does), the last three set position, velocity and acceleration.

    Public API only, on purpose: the history the checker differences against is
    built the same way the server builds it, by recording applied commands.
    """
    previous_velocity = [v - a * DELTA_T for v, a in zip(velocity, acceleration)]
    one_before = [p - v * DELTA_T for p, v in zip(position, velocity)]
    two_before = [p - v * DELTA_T for p, v in zip(one_before, previous_velocity)]
    waypoints = (two_before, two_before, one_before, position)
    for offset, waypoint in enumerate(waypoints):
        checker.record(command(message_id=start_id + offset, q_c=waypoint))


def drive_velocity_history(checker, velocity, acceleration):
    """The same, one derivative up, for a joint-velocity motion."""
    one_before = [v - a * DELTA_T for v, a in zip(velocity, acceleration)]
    for waypoint in (one_before, velocity):
        checker.record(command(dq_c=waypoint))


def spread(value, index=0, other=0.0):
    """A 7-vector holding ``value`` at ``index`` and ``other`` everywhere else."""
    values = [other] * 7
    values[index] = value
    return values


def pose(x=0.0, y=0.0, z=0.0, rotation=None):
    """A homogeneous transform as the 16-element **column-major** wire array.

    The layout matters and is easy to get backwards: ``O_T_EE``/``O_T_EE_c`` are
    column-major, which is why the translation lands at indices 12, 13, 14 --
    and why a client perturbs ``cmd.at(14)`` when it means "move in z".
    """
    matrix = np.eye(4)
    if rotation is not None:
        matrix[:3, :3] = rotation
    matrix[:3, 3] = (x, y, z)
    return [float(value) for value in matrix.T.flatten()]


def mock_pose(dz=0.0):
    """``MOCK_O_T_EE``, optionally shifted ``dz`` further out along z.

    A wire test that streams a ``kCartesianPosition`` Move against ``serve``'s
    default ``mock_arm_sim`` has to open at ``MOCK_O_T_EE`` -- not the identity
    ``pose()`` returns -- or the start-pose check aborts the motion before
    whatever the test actually means to exercise gets a chance to run. Index 14
    is the z-translation, exactly as :func:`pose`'s docstring notes.
    """
    shifted = list(MOCK_O_T_EE)
    shifted[14] += dz
    return shifted


def rotation_about_z(angle):
    """A rotation of ``angle`` about z, as a 3x3 matrix."""
    cos, sin = math.cos(angle), math.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


#: The elbow the mocked arm at :data:`HOME` is actually in: ``(q[2], sign(q[3]))``.
#: ``HOME[3]`` is -1.5, so the branch flag is -1 -- as it is on every real FR3,
#: whose joint 4 range never crosses zero.
HOME_ELBOW = [HOME[2], -1.0]


#: libfranka's default command filter, ``franka::lowpassFilter``
#: (``src/lowpass_filter.cpp``) at ``kDefaultCutoffFrequency`` = 100 Hz and the
#: 1 ms cycle::
#:
#:     gain = dt / (dt + 1 / (2 * pi * cutoff))
#:     y = gain * sample + (1 - gain) * y_last
#:
#: 0.001 / (0.001 + 0.00159155) = 0.385871. This is why a commanded step does
#: not reach the simulator as a step: the FCI never sees the raw
#: ``q_d + 1.0``, it sees a first-order approach to it, whose *first* cycle is
#: still 0.3859 rad -- 386 rad/s of implied velocity.
LOWPASS_GAIN = DELTA_T / (DELTA_T + 1.0 / (2.0 * 3.141592653589793 * 100.0))


def lowpass_step(start, target, cycles):
    """The sequence libfranka actually sends for a step from ``start`` to ``target``.

    Scalar in, list of ``cycles`` scalars out. Used to feed the checker the same
    shape a real client feeds a real robot, rather than an
    idealised one-cycle jump the client could never emit.
    """
    sequence = []
    value = start
    for _ in range(cycles):
        value = LOWPASS_GAIN * target + (1.0 - LOWPASS_GAIN) * value
        sequence.append(value)
    return sequence


# --- layer 1: the checker on its own -----------------------------------------


def test_the_difference_formulas_are_the_documented_backward_euler():
    """Velocity, acceleration and jerk, hand-computed against docs/overview.rst."""
    differentiator = _Differentiator(7)
    # q_{k-1} = 0.1, dq_{k-1} = 2.0, ddq_{k-1} = 3.0
    differentiator.seed([0.1] * 7, [2.0] * 7, [3.0] * 7)

    velocity, acceleration, jerk = differentiator.derivatives([0.1 + 0.004] * 7)

    # dq = (0.104 - 0.1) / 1e-3 = 4.0
    assert velocity == pytest.approx([4.0] * 7)
    # ddq = (4.0 - 2.0) / 1e-3 = 2000
    assert acceleration == pytest.approx([2000.0] * 7)
    # dddq = (2000 - 3.0) / 1e-3 = 1_997_000
    assert jerk == pytest.approx([1997000.0] * 7)


def test_advancing_makes_the_command_and_its_derivatives_the_new_history():
    differentiator = _Differentiator(7)
    differentiator.seed([0.1] * 7, [2.0] * 7, [3.0] * 7)

    differentiator.advance([0.104] * 7)

    assert differentiator.value == pytest.approx([0.104] * 7)
    assert differentiator.first == pytest.approx([4.0] * 7)
    assert differentiator.second == pytest.approx([2000.0] * 7)


def test_an_idle_checker_validates_nothing():
    """No motion, no motion generator: there is nothing to be discontinuous with."""
    assert MotionLimitChecker().check(command(q_c=[100.0] * 7)) is None


def test_a_mode_the_server_does_not_serve_is_not_checked():
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.NONE, robot_state_at())

    assert checker.check(command(q_c=[100.0] * 7)) is None


def test_an_unserved_generator_does_not_judge_the_other_signals_in_the_datagram():
    """The kCartesianPosition bug: zero-filled ``q_c`` read as a joint command.

    A Cartesian ``RobotCommand`` carries ``q_c = 0`` because the client is not
    commanding joints at all -- but all-zeros is not a reachable FR3
    configuration (joint 4 lives in [-3.0481, -0.1458]), so a checker that
    judged it aborted live clients with
    ``joint_motion_generator_position_limits_violation`` on joint 4.
    """
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.NONE, robot_state_at())

    assert checker.check(command(q_c=[0.0] * 7, dq_c=[0.0] * 7, tau_J_d=[0.0] * 7)) is None
    # ...and it is only the *generator* checks that are off: the safety
    # controller is armed for every accepted Move.
    over = spread(upper_joint_velocity_limits(HOME)[5] + 1.0, index=5)
    assert checker.check_measured_velocity(HOME, over).error_index == (
        JOINT_VELOCITY_VIOLATION_INDEX
    )


# -- joint position generator --

def test_a_position_at_the_joint_limit_passes_and_a_hair_past_it_does_not():
    """The FR3 URDF's own <limit lower= upper=>, which is what the client is served."""
    lower, upper = JOINT_POSITION_LIMITS[0]
    at_limit = spread(upper, 0, other=None) and [upper] + HOME[1:]

    checker = position_checker()
    drive_position_history(checker, at_limit, [0.0] * 7, [0.0] * 7)
    assert checker.check(command(q_c=at_limit)) is None

    past = [upper + 1e-6] + HOME[1:]
    violation = checker.check(command(q_c=past))
    assert violation.error_index == JOINT_MOTION_GENERATOR_POSITION_LIMITS_VIOLATION_INDEX
    assert violation.error_name == "joint_motion_generator_position_limits_violation"
    assert violation.axis == "joint 1"
    assert violation.limit == upper

    below = [lower - 1e-6] + HOME[1:]
    assert (
        checker.check(command(q_c=below)).error_index
        == JOINT_MOTION_GENERATOR_POSITION_LIMITS_VIOLATION_INDEX
    )


def test_the_velocity_limit_is_the_urdf_cap_away_from_the_joint_limits():
    """2.62 rad/s on joints 1-4 and 4.18/5.26 on 5-7, less kJointVelocityLimitsTolerance."""
    assert upper_joint_velocity_limits(HOME) == pytest.approx(
        [2.619, 2.619, 2.619, 2.619, 5.259, 4.179, 5.259]
    )
    assert lower_joint_velocity_limits(HOME) == pytest.approx(
        [-2.619, -2.619, -2.619, -2.619, -5.259, -2.897057881, -5.259]
    )


def test_the_velocity_limit_ramps_down_towards_the_joint_limit():
    """The limit is a deceleration ramp near the stop, not a constant."""
    near = [2.7] + HOME[1:]

    # -0.30 + sqrt(2 * 6.0 * (2.7501 - 2.7)) - 0.001
    expected = -0.30 + (12.0 * 0.0501) ** 0.5 - 0.001
    assert upper_joint_velocity_limits(near)[0] == pytest.approx(expected)
    # ...and it never goes negative, so a commanded zero is always legal.
    assert upper_joint_velocity_limits([2.7501] + HOME[1:])[0] == 0.0


#: A velocity ramp that gets a long way past an envelope without ever breaking
#: the acceleration or the jerk limit: the first cycle at half step (so the jerk
#: into the ramp is 4500 rad/s^3, inside kMaxJointJerk), the rest at a constant
#: 9 rad/s^2 (inside kMaxJointAcceleration, and zero jerk). Ten cycles of it
#: move a joint 0.0855 rad/s, which is 3.3% of the tightest cap -- room enough
#: that no plausible future change to LIMIT_EPS can decide these tests.
JOINT_VELOCITY_RAMP = [0.0045] + [0.009] * 10


def test_an_implied_velocity_over_the_cap_is_a_velocity_limits_violation():
    """A signal that is *smoothly* too fast is the envelope error, not a step.

    The overshoot has to be built up over several cycles rather than jumped to:
    a single cycle of legal acceleration only moves a joint 0.009 rad/s, so any
    test that wants a healthy margin over the cap and a legal acceleration on
    the command being judged has to ramp in. ``record()`` does not judge, so the
    crossing itself is silent -- what is on trial is the last command, which is
    over a full 1% past the cap while its acceleration and jerk are both legal,
    so the envelope is the only limit it breaks.
    """
    checker = position_checker()
    cap = upper_joint_velocity_limits(HOME)[0]

    # A steady ramp comfortably inside the cap is legal; the history has to
    # carry that velocity already, or the step into it would be the
    # acceleration violation instead.
    start = cap - 0.05
    drive_position_history(checker, list(HOME), spread(start), [0.0] * 7)
    assert checker.check(command(message_id=5, q_c=[HOME[0] + start * DELTA_T] + HOME[1:])) is None

    position, velocity = HOME[0], start
    for offset, increment in enumerate(JOINT_VELOCITY_RAMP[:-1]):
        velocity += increment
        position += velocity * DELTA_T
        checker.record(command(message_id=5 + offset, q_c=[position] + HOME[1:]))

    velocity += JOINT_VELOCITY_RAMP[-1]
    position += velocity * DELTA_T
    assert velocity > cap * 1.01, "the ramp has to clear the cap by a healthy margin"

    violation = checker.check(
        command(message_id=5 + len(JOINT_VELOCITY_RAMP) - 1, q_c=[position] + HOME[1:])
    )
    assert violation.error_index == JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX
    assert violation.error_name == "joint_motion_generator_velocity_limits_violation"
    assert violation.value == pytest.approx(velocity)
    assert violation.limit == pytest.approx(cap)


def test_acceleration_at_the_limit_passes_and_above_it_is_a_velocity_discontinuity():
    """The kMaxJointAcceleration boundary; note the error name says *velocity*."""
    limit = MAX_JOINT_ACCELERATION[0]

    # Reconstructing a waypoint from an acceleration and differencing it back
    # is not bit-exact, so the boundary is probed a hair either side of the
    # constant rather than exactly on it.
    for acceleration, expected in ((limit - 1e-6, None), (limit + 1e-4, "violation")):
        checker = position_checker()
        # Previous velocity 1.0 rad/s, previous acceleration exactly this one,
        # so the jerk is zero and only the acceleration is on trial.
        drive_position_history(checker, list(HOME), spread(1.0), spread(acceleration))
        velocity = 1.0 + acceleration * DELTA_T
        violation = checker.check(command(q_c=[HOME[0] + velocity * DELTA_T] + HOME[1:]))
        if expected is None:
            assert violation is None
        else:
            assert violation.error_index == JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX
            assert violation.error_name == "joint_motion_generator_velocity_discontinuity"
            assert violation.value == pytest.approx(acceleration, rel=1e-6)


def test_jerk_at_the_limit_passes_and_above_it_is_an_acceleration_discontinuity():
    """The kMaxJointJerk boundary; note the error name says *acceleration*."""
    limit = MAX_JOINT_JERK[0]

    for jerk, expected in ((limit * 0.999, None), (limit * 1.001, "violation")):
        checker = position_checker()
        drive_position_history(checker, list(HOME), spread(1.0), [0.0] * 7)
        acceleration = jerk * DELTA_T
        velocity = 1.0 + acceleration * DELTA_T
        violation = checker.check(command(q_c=[HOME[0] + velocity * DELTA_T] + HOME[1:]))
        if expected is None:
            assert violation is None
        else:
            assert violation.error_index == JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX
            assert violation.error_name == "joint_motion_generator_acceleration_discontinuity"


def test_a_smooth_cosine_ramp_never_violates_anything():
    """The shape libfranka's own examples send, at 1 kHz, start to finish."""
    import math

    checker = position_checker()
    for step in range(5000):
        seconds = step * DELTA_T
        delta = math.pi / 8.0 * (1 - math.cos(math.pi / 2.5 * seconds))
        waypoint = [HOME[0] + delta] + HOME[1:]
        assert checker.check(command(q_c=waypoint)) is None, f"cycle {step}"
        checker.record(command(q_c=waypoint))


# -- joint velocity generator --

def test_a_commanded_velocity_over_the_cap_is_a_velocity_limits_violation():
    cap = upper_joint_velocity_limits(HOME)[0]
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, robot_state_at(dq_d=spread(cap)))

    assert checker.check(command(dq_c=spread(cap))) is None

    violation = checker.check(command(dq_c=spread(cap + 1e-6)))
    assert violation.error_index == JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX
    assert violation.signal == "dq_c"


def test_a_velocity_step_is_an_acceleration_discontinuity():
    """The kMaxJointAcceleration on ``dq_c`` is named *acceleration*, not velocity.

    The interface-relative rule. The same limit on ``q_c`` earns index 14; here
    the commanded channel is already a velocity, so its first difference is an
    acceleration and the error is index 15.
    """
    limit = MAX_JOINT_ACCELERATION[0]
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, robot_state_at())
    drive_velocity_history(checker, spread(0.5), spread(limit))

    assert checker.check(command(dq_c=spread(0.5 + limit * DELTA_T))) is None

    over = limit + 1e-6
    violation = checker.check(command(dq_c=spread(0.5 + over * DELTA_T)))
    assert violation.error_index == JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX
    assert violation.error_name == "joint_motion_generator_acceleration_discontinuity"


def test_a_velocity_generators_jerk_is_an_acceleration_discontinuity():
    limit = MAX_JOINT_JERK[0]
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, robot_state_at())
    drive_velocity_history(checker, spread(0.5), [0.0] * 7)

    assert checker.check(command(dq_c=spread(0.5 + limit * 0.999 * DELTA_T * DELTA_T))) is None

    over = limit * 1.001
    violation = checker.check(command(dq_c=spread(0.5 + over * DELTA_T * DELTA_T)))
    assert violation.error_index == JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX


# -- torque controller --

def test_a_torque_outside_the_joints_range_is_a_tau_j_range_violation():
    """The FR3's <limit effort=>: 87 Nm on joints 1-4, 12 Nm on 5-7."""
    assert MAX_TORQUE == (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.TORQUE, robot_state_at(tau_J_d=spread(87.0)))

    assert checker.check(command(tau_J_d=spread(87.0))) is None

    violation = checker.check(command(tau_J_d=spread(87.0 + 1e-6)))
    assert violation.error_index == TAU_J_RANGE_VIOLATION_INDEX
    assert violation.error_name == "tau_J_range_violation"

    # ...and the wrist joints are held to 12 Nm, not 87.
    assert checker.check(command(tau_J_d=spread(12.5, index=6))).axis == "joint 7"


def test_a_torque_step_is_a_controller_torque_discontinuity():
    """The kMaxTorqueRate boundary: 1000 - kLimitEps Nm/s, i.e. ~1 Nm per cycle."""
    limit = MAX_TORQUE_RATE[0]
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.TORQUE, robot_state_at())

    assert checker.check(command(tau_J_d=spread(limit * DELTA_T))) is None

    violation = checker.check(command(tau_J_d=spread(limit * DELTA_T + 1e-6)))
    assert violation.error_index == CONTROLLER_TORQUE_DISCONTINUITY_INDEX
    assert violation.error_name == "controller_torque_discontinuity"


# -- cartesian velocity generator (the mobile base's twist) --

#: The twist half of :data:`JOINT_VELOCITY_RAMP`, in *per-component* steps on a
#: 45-degree x/y twist: the norm moves sqrt(2) x each step, so 0.003 is an
#: acceleration of 4.24 m/s^2 (and a jerk of 4243 m/s^3, inside
#: kMaxTranslationalJerk) and 0.006 is 8.49 m/s^2, inside
#: kMaxTranslationalAcceleration with the jerk at zero.
TWIST_RAMP = [0.003] + [0.006] * 10


def test_a_cartesian_twist_is_judged_by_norms_not_components():
    """The limiter treats O_dP_EE_c as two Vector3d and compares .norm().

    Two equal components on a 45-degree heading, so the norm is sqrt(2) times
    either one: the norm can be pushed a long way over its cap while every
    single component stays far inside the same number. As with the joint
    envelope, one cycle of legal acceleration is only 0.009 m/s, so the crossing
    is ramped in over ``record()`` calls (which do not judge) and only the last
    command -- 1.3% past the cap, with a legal acceleration and zero jerk -- is
    put on trial. A twist *step* would be an acceleration discontinuity and
    would win instead, which is
    test_a_twist_acceleration_step_is_a_cartesian_acceleration_discontinuity's
    point.
    """
    checker = MotionLimitChecker()
    start = (MAX_TRANSLATIONAL_VELOCITY - 0.05) / 2**0.5
    steady = [start, start, 0.0, 0.0, 0.0, 0.0]
    checker.start_motion(ControlMode.STEERING_DRIVE, robot_state_at(O_dP_EE_d=steady))

    assert checker.check(command(O_dP_EE_c=steady)) is None

    component = start
    for offset, increment in enumerate(TWIST_RAMP[:-1]):
        component += increment
        checker.record(
            command(message_id=1 + offset, O_dP_EE_c=[component, component, 0.0, 0.0, 0.0, 0.0])
        )

    component += TWIST_RAMP[-1]
    over = [component, component, 0.0, 0.0, 0.0, 0.0]
    norm = (2 * component**2) ** 0.5
    assert norm > MAX_TRANSLATIONAL_VELOCITY * 1.01, "the ramp needs a healthy margin"
    assert component < MAX_TRANSLATIONAL_VELOCITY, "no single component may be over the cap"

    violation = checker.check(command(message_id=len(TWIST_RAMP), O_dP_EE_c=over))
    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX
    assert violation.axis == "translational norm"
    assert violation.value == pytest.approx(norm)
    assert violation.limit == pytest.approx(MAX_TRANSLATIONAL_VELOCITY)


def test_the_rotational_half_carries_its_own_limits():
    checker = MotionLimitChecker()
    seed = [0.0, 0.0, 0.0, MAX_ROTATIONAL_VELOCITY, 0.0, 0.0]
    checker.start_motion(ControlMode.STEERING_DRIVE, robot_state_at(O_dP_EE_d=seed))

    assert checker.check(command(O_dP_EE_c=seed)) is None

    over = [0.0, 0.0, 0.0, MAX_ROTATIONAL_VELOCITY + 1e-6, 0.0, 0.0]
    violation = checker.check(command(O_dP_EE_c=over))
    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX
    assert violation.axis == "rotational norm"


def test_a_twist_acceleration_step_is_a_cartesian_acceleration_discontinuity():
    """The joint-velocity mapping, mirrored onto ``O_dP_EE_c``.

    kMaxTranslationalAcceleration broken on the first difference of a commanded
    *twist* is ``cartesian_motion_generator_acceleration_discontinuity`` (20).
    Index 19 belongs to the same limit on a commanded *pose*, which this server
    does not serve.
    """
    limit = MAX_TRANSLATIONAL_ACCELERATION
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.STEERING_DRIVE, robot_state_at())
    # Ramp in at exactly the acceleration limit, which is also the jerk the
    # first cycle carries: 8.999 / 1e-3 = 8999 < 4499999.
    checker.record(command(O_dP_EE_c=[limit * DELTA_T, 0, 0, 0, 0, 0]))

    violation = checker.check(command(O_dP_EE_c=[3 * limit * DELTA_T, 0, 0, 0, 0, 0]))
    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX
    assert violation.error_name == "cartesian_motion_generator_acceleration_discontinuity"
    assert violation.axis == "translational norm"


def test_a_twist_jerk_step_is_a_cartesian_acceleration_discontinuity():
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.STEERING_DRIVE, robot_state_at())

    # An acceleration inside its limit, reached in one cycle from rest: the
    # jerk is acceleration / dt, so anything past kMaxTranslationalJerk * dt^2
    # trips the jerk check first.
    over = (MAX_TRANSLATIONAL_JERK * 1.01) * DELTA_T * DELTA_T
    violation = checker.check(command(O_dP_EE_c=[over, 0, 0, 0, 0, 0]))
    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX


# -- the interface-relative naming rule, against observed hardware behaviour --
#
# Each test below replays one scenario a real FR3 was observed to answer: the
# *commanded* sequence a real client would emit (libfranka low-passes every
# command at 100 Hz, so a "step" reaches the FCI as a first-order approach, not
# a jump) and the one error name hardware answers with, even though the step
# breaks every limit its interface has.


def test_a_position_step_is_a_velocity_discontinuity_not_an_envelope_violation():
    """A mid-motion ``q_c`` step of +1.0 rad -> index 14.

    The step breaks the velocity envelope (13), the acceleration limit (14) and
    the jerk limit (15) on its very first smoothed cycle. Hardware names the
    acceleration one. The sim used to report 13, because the envelope check ran
    first.
    """
    checker = position_checker()
    # Hold the start pose for a few cycles, so what follows is unambiguously a
    # *mid-motion* step and not the start-pose check.
    for cycle in range(1, 4):
        checker.record(command(message_id=cycle, q_c=list(HOME)))

    first_cycle = lowpass_step(HOME[0], HOME[0] + 1.0, 1)[0]
    velocity = (first_cycle - HOME[0]) / DELTA_T
    # The three limits this one cycle breaks, spelled out so the precedence
    # being pinned is visible rather than implied.
    assert velocity == pytest.approx(385.87, abs=0.01)
    assert velocity > upper_joint_velocity_limits(HOME)[0]
    assert velocity / DELTA_T > MAX_JOINT_ACCELERATION[0]
    assert velocity / DELTA_T / DELTA_T > MAX_JOINT_JERK[0]

    violation = checker.check(command(message_id=4, q_c=[first_cycle] + HOME[1:]))
    assert violation.error_index == JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX
    assert violation.error_name == "joint_motion_generator_velocity_discontinuity"


def test_a_velocity_step_is_named_for_the_acceleration_not_the_envelope():
    """A mid-motion ``dq_c`` step of +50 rad/s -> index 15.

    Same shape, one interface up, and hardware answers with a *different* name:
    on ``dq_c`` the first difference is already an acceleration, so it is 15 --
    not the 14 the identical limit earns on ``q_c``, and not the envelope's 13.
    """
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, robot_state_at())
    for cycle in range(1, 4):
        checker.record(command(message_id=cycle, dq_c=[0.0] * 7))

    first_cycle = lowpass_step(0.0, 50.0, 1)[0]
    assert first_cycle == pytest.approx(19.29, abs=0.01)
    assert first_cycle > upper_joint_velocity_limits(HOME)[0]  # breaks 13 too
    assert first_cycle / DELTA_T > MAX_JOINT_ACCELERATION[0]

    violation = checker.check(command(message_id=4, dq_c=spread(first_cycle)))
    assert violation.error_index == JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX
    assert violation.error_name == "joint_motion_generator_acceleration_discontinuity"


def test_a_twist_step_is_a_cartesian_acceleration_discontinuity():
    """A mid-motion ``O_dP_EE_c`` step of 10 m/s -> index 20.

    The joint-velocity mapping mirrored onto the base's twist. Index 19 is the
    same limit on a commanded *pose*, and the pair is what pins the whole
    interface-relative rule -- see
    ``test_a_pose_step_is_a_cartesian_velocity_discontinuity``.
    """
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.STEERING_DRIVE, robot_state_at())
    for cycle in range(1, 4):
        checker.record(command(message_id=cycle, O_dP_EE_c=[0.0] * 6))

    first_cycle = lowpass_step(0.0, 10.0, 1)[0]
    assert first_cycle > MAX_TRANSLATIONAL_VELOCITY  # breaks 18 too
    assert first_cycle / DELTA_T > MAX_TRANSLATIONAL_ACCELERATION

    violation = checker.check(
        command(message_id=4, O_dP_EE_c=[first_cycle, 0.0, 0.0, 0.0, 0.0, 0.0])
    )
    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX
    assert violation.error_name == "cartesian_motion_generator_acceleration_discontinuity"
    # 19 is the pose generator's name for the same limit: a *twist* step must
    # never be called that, whatever the pose generator does.
    assert violation.error_index != CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX


def test_an_arm_role_twist_is_checked_exactly_like_the_base_one():
    """``kCartesianVelocity`` on an arm role runs the same twist checks.

    The generator is the same generator; only the role differs. Before the
    gating was extended, an arm-role Cartesian-velocity Move was checked on
    nothing at all -- the mode did not reach the checker, so a twist step
    produced no error and a client provoking one hung waiting for it.
    """
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.CARTESIAN_VELOCITY, robot_state_at())
    for cycle in range(1, 4):
        checker.record(command(message_id=cycle, O_dP_EE_c=[0.0] * 6))

    first_cycle = lowpass_step(0.0, 10.0, 1)[0]
    violation = checker.check(command(message_id=4, O_dP_EE_c=[first_cycle] * 3 + [0.0] * 3))

    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX


# -- cartesian pose generator (kCartesianPosition: checked, never applied) --
#
# Every test below drives a generator whose commands reach no physics at all.
# That is the deliberate shape of the feature: a client provoking a Cartesian
# error needs the *abort*, not the motion, and an interface that
# silently swallowed its stream left such a client waiting forever.


def pose_checker(q=None, o_t_ee=None, **state_fields):
    """A checker armed for a Cartesian pose motion at the identity pose."""
    checker = MotionLimitChecker()
    checker.start_motion(
        ControlMode.CARTESIAN_POSE,
        robot_state_at(q=q, O_T_EE=pose() if o_t_ee is None else o_t_ee, **state_fields),
    )
    return checker


def test_a_pose_step_is_a_cartesian_velocity_discontinuity():
    """A mid-motion ``O_T_EE_c`` step of 1 m in z -> index 19.

    The interface-relative rule's Cartesian half, and the exact case hardware
    pins.
    The client asks for ``O_T_EE_d`` with ``+1.0`` on z; libfranka's own 100 Hz
    command filter is what the FCI actually receives, so the first cycle is
    0.3859 m -- 386 m/s and 385 871 m/s^2. That breaks the envelope, the
    acceleration limit *and* the jerk limit at once, and hardware still returns
    exactly one name: the acceleration limit, called
    ``cartesian_motion_generator_velocity_discontinuity`` because on a commanded
    *pose* the first difference is a velocity.
    """
    checker = pose_checker()
    for cycle in range(1, 4):
        checker.record(command(message_id=cycle, O_T_EE_c=pose()))

    first_cycle = lowpass_step(0.0, 1.0, 1)[0]
    assert first_cycle == pytest.approx(0.3859, abs=1e-4)
    assert first_cycle / DELTA_T > MAX_TRANSLATIONAL_VELOCITY  # breaks 18 too
    assert first_cycle / DELTA_T**2 > MAX_TRANSLATIONAL_ACCELERATION
    assert first_cycle / DELTA_T**3 > MAX_TRANSLATIONAL_JERK  # and 20

    violation = checker.check(command(message_id=4, O_T_EE_c=pose(z=first_cycle)))

    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX
    assert violation.error_name == "cartesian_motion_generator_velocity_discontinuity"
    assert violation.error_index != CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX


def test_a_rotational_pose_step_is_measured_through_the_log_map():
    """A pure rotation step is judged on the angle of R_prev^T R_curr, not on cells.

    Subtracting two rotation matrices is not an angular velocity, so the check
    composes the *relative* rotation and takes its axis-angle. 0.5 rad inside one
    cycle is 500 rad/s, ~200x kMaxRotationalVelocity, and one cycle up that is an
    acceleration -- so the name is the pose interface's 19, exactly as for the
    translational step.
    """
    checker = pose_checker()
    for cycle in range(1, 4):
        checker.record(command(message_id=cycle, O_T_EE_c=pose()))

    violation = checker.check(
        command(message_id=4, O_T_EE_c=pose(rotation=rotation_about_z(0.5)))
    )

    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX
    assert violation.axis == "rotational norm"


def test_a_smooth_conforming_pose_stream_produces_no_error():
    """A 1 kHz cosine in translation and yaw, inside every limit, runs clean.

    The counterpart to the step tests: the checks must be quiet on a signal a
    real controller would emit. Peak speed here is 0.05 m/s and 0.05 rad/s, and
    the acceleration and jerk that a raised cosine implies at 1 kHz are orders of
    magnitude inside their caps.
    """
    checker = pose_checker()
    amplitude, frequency = 0.05, 0.5  # m and rad, at 0.5 Hz

    for step in range(400):
        phase = 2.0 * math.pi * frequency * step * DELTA_T
        travel = amplitude * (1.0 - math.cos(phase)) / 2.0
        commanded = command(
            message_id=step + 1,
            O_T_EE_c=pose(z=travel, rotation=rotation_about_z(travel)),
        )
        assert checker.check(commanded) is None, f"cycle {step} was blamed for nothing"
        checker.record(commanded)


def test_a_ten_metre_first_pose_is_a_start_pose_error_not_a_discontinuity():
    """A first ``O_T_EE_c`` +10 m away is a start-pose error, from cycle 0.

    Offsetting z by 10 m and holding it there for the whole
    motion makes hardware answer
    ``cartesian_position_motion_generator_start_pose_invalid``. The same
    precedence as the joint start-pose
    check: a first command in the wrong place is a start-pose error, and the
    discontinuity it would also imply never gets computed.
    """
    checker = pose_checker()

    violation = checker.check(command(O_T_EE_c=pose(z=10.0)))

    assert violation.error_index == CARTESIAN_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX
    assert violation.error_name == "cartesian_position_motion_generator_start_pose_invalid"
    assert violation.error_index != CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX


def test_the_start_pose_tolerances_are_the_documented_sim_choices():
    """Just inside passes, just outside is 16 -- for translation and rotation both."""
    inside = pose_checker()
    nudge = pose(z=START_CARTESIAN_POSE_TRANSLATION_TOLERANCE / 2)
    assert inside.check(command(O_T_EE_c=nudge)) is None

    outside = pose_checker()
    violation = outside.check(
        command(O_T_EE_c=pose(z=START_CARTESIAN_POSE_TRANSLATION_TOLERANCE * 2))
    )
    assert violation.error_index == CARTESIAN_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX

    turned = pose_checker()
    violation = turned.check(
        command(
            O_T_EE_c=pose(rotation=rotation_about_z(START_CARTESIAN_POSE_ROTATION_TOLERANCE * 2))
        )
    )
    assert violation.error_index == CARTESIAN_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX
    assert violation.axis == "rotation"


def test_a_state_without_a_measured_pose_skips_the_start_pose_check():
    """No ``O_T_EE`` in the snapshot means nothing to compare against.

    Inventing a reference would fabricate the error rather than find it, so the
    check is skipped -- while every *difference* check still runs.
    """
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.CARTESIAN_POSE, robot_state_at())  # no O_T_EE

    assert checker.check(command(O_T_EE_c=pose(z=10.0))) is None


@pytest.mark.parametrize(
    "matrix, why",
    [
        ([0.0] * 16, "all zeros: the canonical invalid frame"),
        (pose()[:15] + [2.0], "bottom-right is not 1"),
        ([2.0] + pose()[1:], "first column is not a unit vector"),
        (pose(rotation=np.full((3, 3), 0.5)), "rotation block is not orthonormal"),
        ([float("nan")] + pose()[1:], "non-finite"),
    ],
)
def test_a_matrix_that_is_not_a_rigid_transform_is_refused_always(matrix, why):
    """``checkMatrix``'s test, and it is fatal: refused with enforcement off too.

    Same reasoning as a NaN command (:attr:`Violation.fatal`): a garbage matrix
    has no derivatives worth computing, and recording one poisons the history
    every later command in the motion is judged against.
    """
    assert not is_homogeneous_transformation(matrix), why

    checker = pose_checker()
    violation = checker.check(command(O_T_EE_c=matrix))

    assert violation.error_index == CARTESIAN_POSITION_MOTION_GENERATOR_INVALID_FRAME_INDEX
    assert violation.error_name == "cartesian_position_motion_generator_invalid_frame_flag"
    assert violation.fatal is True


def test_the_homogeneity_test_accepts_what_libfranka_accepts():
    """Real transforms pass, including rotated and translated ones."""
    assert is_homogeneous_transformation(pose())
    assert is_homogeneous_transformation(pose(x=1.0, y=-2.0, z=0.3))
    assert is_homogeneous_transformation(pose(z=0.5, rotation=rotation_about_z(1.2)))


def test_the_rotation_log_map_is_robust_at_both_degenerate_angles():
    """Zero and pi are where the 1/sin(theta) form blows up; neither may give NaN."""
    assert float(np.linalg.norm(rotation_log(np.eye(3)))) == pytest.approx(0.0)
    for angle in (1e-12, 1e-6, 0.3, 2.0, math.pi - 1e-6, math.pi):
        magnitude = float(np.linalg.norm(rotation_log(rotation_about_z(angle))))
        assert math.isfinite(magnitude)
        assert magnitude == pytest.approx(angle, abs=1e-6)
    # A trace a hair above 3 -- which an only-1e-5-orthonormal commanded matrix
    # can produce -- must clamp rather than hand acos a value outside [-1, 1].
    assert math.isfinite(float(np.linalg.norm(rotation_log(np.eye(3) * (1.0 + 1e-9)))))


# -- the elbow, on both Cartesian interfaces --


def elbow_command(message_id, angle, sign=None):
    """A pose-generator command holding the identity pose and commanding an elbow."""
    return command(
        message_id=message_id,
        O_T_EE_c=pose(),
        elbow_c=[angle, HOME_ELBOW[1] if sign is None else sign],
        valid_elbow=True,
    )


def drive_elbow_history(checker, angle, velocity, acceleration):
    """Record elbow waypoints until the applied history is exactly what is asked.

    :func:`drive_position_history` one signal over: solved backwards out of the
    same difference formulas, and going through ``record()`` only, so the
    priming waypoints are never put on trial for the derivatives they imply.
    """
    previous_velocity = velocity - acceleration * DELTA_T
    one_before = angle - velocity * DELTA_T
    two_before = one_before - previous_velocity * DELTA_T
    for offset, waypoint in enumerate((two_before, two_before, one_before, angle)):
        checker.record(elbow_command(offset + 1, waypoint))


def test_a_start_elbow_away_from_the_robots_own_is_start_elbow_invalid():
    """A first ``elbow_c[0]`` 0.5 rad away from the robot's own -> 22."""
    checker = pose_checker()

    violation = checker.check(
        command(O_T_EE_c=pose(), elbow_c=[HOME_ELBOW[0] + 0.5, HOME_ELBOW[1]], valid_elbow=True),
        HOME,
    )

    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_START_ELBOW_INVALID_INDEX
    assert violation.error_name == "cartesian_motion_generator_start_elbow_invalid"

    # Just inside the tolerance is not an error; the 0.5 rad hardware is
    # provoked with is five times it.
    inside = pose_checker()
    nudged = HOME_ELBOW[0] + START_ELBOW_TOLERANCE / 2
    assert inside.check(elbow_command(1, nudged), HOME) is None


def test_a_start_elbow_sign_that_disagrees_with_joint_4_is_also_22():
    """The robot's elbow is ``(q[2], sign(q[3]))``, and both halves are checked.

    No hardware observation pins the *name* for a start-time sign mismatch --
    see ``START_ELBOW_SIGN_INCONSISTENT_INDEX`` (24), which by its name would fit
    better and which nothing exercises. This pins the sim's choice so that
    changing it is deliberate.
    """
    checker = pose_checker()

    violation = checker.check(
        command(O_T_EE_c=pose(), elbow_c=[HOME_ELBOW[0], +1.0], valid_elbow=True), HOME
    )

    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_START_ELBOW_INVALID_INDEX


def test_the_robots_own_elbow_opens_a_motion_cleanly():
    """``(q[2], sign(q[3]))`` fed straight back must pass, on both interfaces."""
    for mode in (ControlMode.CARTESIAN_POSE, ControlMode.CARTESIAN_VELOCITY):
        checker = MotionLimitChecker()
        checker.start_motion(mode, robot_state_at(O_T_EE=pose()))
        assert (
            checker.check(
                command(O_T_EE_c=pose(), elbow_c=list(HOME_ELBOW), valid_elbow=True), HOME
            )
            is None
        ), mode


def test_an_elbow_sign_flip_mid_motion_is_elbow_sign_inconsistent():
    """Negating ``elbow_c[1]`` mid-motion -> 21."""
    checker = pose_checker()
    opening = command(O_T_EE_c=pose(), elbow_c=list(HOME_ELBOW), valid_elbow=True)
    assert checker.check(opening, HOME) is None
    checker.record(opening)

    violation = checker.check(
        command(
            message_id=2,
            O_T_EE_c=pose(),
            elbow_c=[HOME_ELBOW[0], -HOME_ELBOW[1]],
            valid_elbow=True,
        ),
        HOME,
    )

    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_ELBOW_SIGN_INCONSISTENT_INDEX
    assert violation.error_name == "cartesian_motion_generator_elbow_sign_inconsistent"


def test_an_elbow_ramp_past_its_velocity_limit_is_an_elbow_limit_violation():
    """A constant elbow acceleration of 0.3 rad/s^2 -> 17.

    Holding the pose and integrating a constant
    ``ddelbow = 0.0003 / 0.001 = 0.3`` rad/s^2 into the elbow makes elbow
    velocity grow linearly and cross ``kMaxElbowVelocity`` (1.499 rad/s) after
    ~5 s while acceleration stays at 0.3 -- two orders of magnitude inside
    ``kMaxElbowAcceleration``. Hardware answers
    ``cartesian_motion_generator_elbow_limit_violation``.

    Five seconds is 5000 cycles, so the history is solved backwards to a point
    late on that same ramp rather than integrated from the start: the
    acceleration is that same 0.3 rad/s^2 and the velocity is set
    1.5 cycles' worth below the cap, which puts the next cycle just inside it and
    the one after just outside. Both are asserted, so this pins the *boundary*
    and not merely that a large enough ramp eventually trips.
    """
    checker = pose_checker()
    acceleration = 0.3  # the ddelbow hardware is provoked with, exactly
    assert acceleration < MAX_ELBOW_ACCELERATION

    velocity = MAX_ELBOW_VELOCITY - 1.5 * acceleration * DELTA_T
    angle = HOME_ELBOW[0]
    drive_elbow_history(checker, angle, velocity, acceleration)

    velocity += acceleration * DELTA_T  # half a cycle's worth inside the cap
    assert velocity < MAX_ELBOW_VELOCITY
    angle += velocity * DELTA_T
    inside = elbow_command(5, angle)
    assert checker.check(inside, HOME) is None
    checker.record(inside)

    velocity += acceleration * DELTA_T  # ...and half a cycle's worth outside it
    assert velocity > MAX_ELBOW_VELOCITY
    angle += velocity * DELTA_T
    violation = checker.check(elbow_command(6, angle), HOME)

    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_ELBOW_LIMIT_VIOLATION_INDEX
    assert violation.error_name == "cartesian_motion_generator_elbow_limit_violation"
    assert violation.value == pytest.approx(velocity, rel=1e-9)
    assert violation.unit == "rad/s"


def test_an_elbow_less_motion_is_never_judged_on_its_zero_filled_elbow():
    """``valid_elbow`` is the gate, and it is what keeps the mobile base out.

    libfranka zero-fills ``elbow_c`` and clears the flag for every motion whose
    generator carries no elbow (``src/control_loop.cpp:284-286``). Judging that
    zero as a commanded elbow would abort the base's twist stream -- and every
    elbow-less arm motion -- with ``start_elbow_invalid`` on the first cycle.
    """
    for mode in (ControlMode.STEERING_DRIVE, ControlMode.CARTESIAN_VELOCITY):
        checker = MotionLimitChecker()
        checker.start_motion(mode, robot_state_at())
        # elbow_c = [0, 0]: angle 0 is 1.5 rad from HOME's q[2]... which is 0,
        # so use a configuration where it is not, to make the point stick.
        assert checker.check(command(O_dP_EE_c=[0.0] * 6), [0.4] * 7) is None, mode

    checker = pose_checker()
    assert checker.check(command(O_T_EE_c=pose()), [0.4] * 7) is None


def test_a_steering_drive_command_with_valid_elbow_latches_nothing():
    """The mobile base has no elbow -- ``STEERING_DRIVE`` skips its checks.

    ``_check_cartesian_velocity`` is shared between the arm's
    ``CARTESIAN_VELOCITY`` and the base's ``STEERING_DRIVE``, but ``record()``
    only ever calls ``_record_elbow_locked`` for the former (see its
    dispatch) -- so ``_elbow_sign`` never leaves its initial ``None`` for a
    base motion. Before the elbow checks were restricted to the arm role, a
    base client that happened to set ``valid_elbow`` therefore re-tripped the
    *start*-elbow check every single cycle, judged against a swerve steering
    angle that was never meant to be an elbow at all.

    ``O_T_EE`` is seeded (unlike the zero-filled-elbow test above) so this
    pins the *mode* gate specifically, not the "no measured pose" skip
    :meth:`~franka_sim.motion_limits.MotionLimitChecker._check_elbow_validity`
    also has.
    """
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.STEERING_DRIVE, robot_state_at(O_T_EE=pose()))

    # A wildly wrong elbow: if the elbow checks ran at all for this mode, this
    # would trip start_elbow_invalid on cycle 0 and every cycle after.
    wrong_elbow = [HOME[2] + 100.0, 1.0]
    for _ in range(5):
        cmd = command(O_dP_EE_c=[0.0] * 6, valid_elbow=True, elbow_c=wrong_elbow)
        assert checker.check(cmd, HOME) is None
        checker.record(cmd)

    assert checker.violated is False


def test_a_torque_that_breaks_both_range_and_rate_reports_the_range():
    """Range before rate -- and this is the one ordering hardware does not pin.

    The observable hardware case steps joint 1 to 10 Nm: a rate
    violation (10 000 Nm/s) comfortably inside the joint's 87 Nm range, so it
    exercises the rate check alone and never covers a command that breaks
    both. No other hardware observation settles it either. This pins what the
    sim does so
    that changing it is a decision rather than an accident; see the comment on
    ``MotionLimitChecker._check_torque`` for why it could go either way.
    """
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.TORQUE, robot_state_at())

    step = 100.0  # > 87 Nm range and > 1000 Nm/s rate, in one cycle from zero
    assert step > MAX_TORQUE[0]
    assert step / DELTA_T > MAX_TORQUE_RATE[0]

    violation = checker.check(command(tau_J_d=spread(step)))
    assert violation.error_index == TAU_J_RANGE_VIOLATION_INDEX
    assert violation.error_name == "tau_J_range_violation"


# -- the safety controller: measured velocity -> joint_velocity_violation --
#
# A different error from 13 and a different *subject*: the robot watching its
# own joints rather than Control judging the client's command. Active in every
# control mode, including pure torque, where no commanded velocity exists.


def test_the_safety_controller_watches_measured_velocity_in_every_mode():
    cap = upper_joint_velocity_limits(HOME)[5]
    for mode, seed in (
        (ControlMode.POSITION, {}),
        (ControlMode.VELOCITY, {}),
        (ControlMode.TORQUE, {}),
    ):
        checker = MotionLimitChecker()
        checker.start_motion(mode, robot_state_at(**seed))

        assert checker.check_measured_velocity(HOME, spread(cap - 0.01, index=5)) is None

        violation = checker.check_measured_velocity(HOME, spread(cap + 1.0, index=5))
        assert violation.error_index == JOINT_VELOCITY_VIOLATION_INDEX, mode
        assert violation.error_name == "joint_velocity_violation"
        assert violation.signal == "dq"
        assert violation.axis == "joint 6"


def test_the_safety_controller_leaves_a_margin_over_the_envelope():
    """Measured signals are noisy; see MEASURED_JOINT_VELOCITY_MARGIN."""
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.TORQUE, robot_state_at())
    cap = upper_joint_velocity_limits(HOME)[5]

    inside = cap + MEASURED_JOINT_VELOCITY_MARGIN * 0.9
    assert checker.check_measured_velocity(HOME, spread(inside, index=5)) is None

    outside = cap + MEASURED_JOINT_VELOCITY_MARGIN * 1.1
    violation = checker.check_measured_velocity(HOME, spread(outside, index=5))
    assert violation.error_index == JOINT_VELOCITY_VIOLATION_INDEX


def test_a_smooth_motion_never_trips_the_safety_controller():
    """A full 0.5 Hz cosine sweep on every joint at once, measured cycle by cycle.

    The margin must not fire during ordinary operation, so this drives a
    trajectory a client would really send and checks not just that nothing
    latched but that the *closest approach* to the envelope stayed well clear of
    it -- otherwise the test would pass for the wrong reason if the sweep were
    accidentally made timid.
    """
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.POSITION, robot_state_at())

    cycles = 2000  # a 2 s period at 1 kHz
    omega = 2.0 * np.pi / (cycles * DELTA_T)
    peak = 1.0  # rad/s, ~38% of the tightest cap on this configuration
    amplitude = peak / omega
    # Joint 4 sits closer to its upper stop than its lower one, so it sweeps the
    # other way; the position-based envelope collapses towards a stop, and a
    # trajectory that drives *into* one is not the "ordinary operation" this is
    # about.
    direction = [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0]
    headroom = np.inf
    for cycle in range(cycles):
        phase = omega * cycle * DELTA_T
        q = [
            HOME[joint] + direction[joint] * amplitude * (1.0 - np.cos(phase))
            for joint in range(7)
        ]
        dq = [direction[joint] * peak * np.sin(phase) for joint in range(7)]
        assert checker.check_measured_velocity(q, dq) is None, f"fired at cycle {cycle}"
        upper = upper_joint_velocity_limits(q)
        lower = lower_joint_velocity_limits(q)
        for joint in range(7):
            headroom = min(headroom, upper[joint] - dq[joint], dq[joint] - lower[joint])

    assert headroom > MEASURED_JOINT_VELOCITY_MARGIN * 5, (
        f"the sweep came within {headroom:.3f} rad/s of the envelope; the margin "
        "is not what kept this test green"
    )


def test_a_velocity_ramp_the_arm_follows_is_the_safety_controllers_error():
    """A commanded velocity ramp past the envelope -> both 13 and 3.

    A 5 rad/s^2 ramp on joint 6 with no cap: no discontinuity anywhere, the
    commanded velocity simply walks out of the envelope and the arm walks out
    with it.

    Hardware reports ``joint_velocity_violation`` in the exception
    message for this scenario. Whether *both* bits are latched is a reported
    behaviour rather than one measured here: hardware raises both errors, with
    ``joint_velocity_violation`` appearing much earlier because the controller
    shapes the envelope down towards the current velocity. Treat it as strong
    but not independently verified. This sim latches both bits
    from the commanded envelope check (13) deterministically rather than picking
    one; see :attr:`franka_sim.motion_limits.Violation.extra_error_index`.

    The ramp is eased in over the first 200 cycles (see :func:`_ease`), not
    seeded with a nonzero ``ddq_d`` the way this test once fast-forwarded past
    the ramp-up: a velocity motion's opening command rebases the checker's
    differencing history to a flat standstill exactly like the position and
    Cartesian-pose generators' does (see the ``VELOCITY`` branch of
    ``_record_locked``), so a seeded acceleration would be discarded anyway --
    and un-eased, jumping straight to a steady 5 rad/s^2 from that flat history
    trips a false jerk discontinuity on the very next command, before the
    envelope violation this test is about.
    """
    accel, start = 5.0, 4.0
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, robot_state_at(dq_d=spread(start, index=5)))
    cap = upper_joint_velocity_limits(HOME)[5]
    assert start < cap, "the ramp has to begin inside the envelope"

    latched = None
    dq = start
    for cycle in range(1, 700):
        dq += accel * _ease(cycle / 200.0) * DELTA_T
        # The arm tracks a velocity generator, so measured follows commanded.
        latched = checker.check_measured_velocity(HOME, spread(dq, index=5))
        if latched is not None:
            break
        outgoing = command(message_id=cycle, dq_c=spread(dq, index=5))
        latched = checker.check(outgoing)
        if latched is not None:
            break
        checker.record(outgoing)

    assert latched is not None, "the ramp never violated anything"
    # The commanded envelope check (13) trips a cycle before the measured
    # safety check's own margin does (see MEASURED_JOINT_VELOCITY_MARGIN), so
    # it is the one that fires here -- but with 3 latched alongside it.
    assert latched.error_index == JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX
    assert latched.extra_error_index == JOINT_VELOCITY_VIOLATION_INDEX
    assert latched.error_indices == (
        JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
        JOINT_VELOCITY_VIOLATION_INDEX,
    )
    assert "joint_velocity_violation" in latched.describe()
    assert latched.axis == "joint 6"


#: Effective inertia (kg m^2) the fake joint 6 below swings under the applied
#: torque. Only the causality matters, not the number: it is picked so the arm
#: is still inside the envelope when the ramp reaches the full 3 Nm,
#: which is what makes "3 Nm folds the arm" the thing actually being measured.
FAKE_JOINT_INERTIA = 0.5


def _torque_ramp_until_the_arm_folds(ramp_rate, cycles=1400):
    """Run a pure-torque session whose measured ``dq`` is *integrated from tau*.

    ``dq[k+1] = dq[k] + tau[k] / I * dt`` -- a plain double integrator, so the
    only thing that can drive the arm out of the envelope is torque this session
    actually applied. Returns ``(violation or None, the torque at that cycle)``.
    """
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.TORQUE, robot_state_at())

    measured, tau = 0.0, 0.0
    for cycle in range(1, cycles):
        tau = min(3.0, ramp_rate * cycle * DELTA_T)  # a 5 Nm/s ramp, capped at 3 Nm
        outgoing = command(message_id=cycle, tau_J_d=spread(tau, index=5))
        assert checker.check(outgoing) is None, "the commanded torque itself is legal"
        checker.record(outgoing)
        measured += tau / FAKE_JOINT_INERTIA * DELTA_T
        latched = checker.check_measured_velocity(HOME, spread(measured, index=5))
        if latched is not None:
            return latched, tau
    return None, tau


def test_a_torque_ramp_that_folds_the_arm_is_the_safety_controllers_error():
    """3 Nm ramped into joint 6, in *pure torque control*.

    There is no commanded velocity anywhere in this session, so no commanded
    check could ever produce this error. Every torque sent is legal -- inside
    the joint's 12 Nm range and far inside kMaxTorqueRate -- and hardware still
    reflexes, because the arm itself leaves the envelope.

    The measured stream is *integrated from the applied torque*, not
    hand-cranked alongside it. It used to be the latter -- a fixed 0.02 rad/s
    per cycle, which folded the arm at 0.68 Nm and would have folded it just the
    same with the torque stream replaced by zeros. The zero-torque control case
    at the end is what pins that: with the arm's motion caused by the torque,
    a session that commands nothing can never leave the envelope.
    """
    cap = upper_joint_velocity_limits(HOME)[5]

    latched, tau_at_fold = _torque_ramp_until_the_arm_folds(5.0)

    assert latched is not None, "the arm never left the envelope"
    assert latched.error_index == JOINT_VELOCITY_VIOLATION_INDEX
    assert latched.value > cap
    assert tau_at_fold == pytest.approx(3.0), (
        f"the arm folded at {tau_at_fold:.2f} Nm, before the ramp reached its "
        "3 Nm cap -- the narrative and the numbers disagree"
    )

    quiet, _ = _torque_ramp_until_the_arm_folds(0.0)
    assert quiet is None, "an arm nothing pushed still left the envelope"


def test_a_commanded_envelope_violation_during_a_motion_latches_both_bits():
    """13 latches 3 alongside it, deterministically -- not only when measured
    dq already happens to have crossed.

    This used to be decided by re-asking the safety controller against
    measured ``dq`` with a zero margin, and reported plain 13 whenever that
    re-ask came back clean. But measured ``dq`` lags commanded ``dq`` by a
    cycle or few, so the re-ask read "clean" on exactly the cycle hardware
    would call ``joint_velocity_violation`` -- nondeterministic in exactly the
    way a `dq_c` ramp past the envelope exposes it. On hardware the paired
    outcome is the expected one rather than exceptional: both errors can
    appear, and ``joint_velocity_violation`` appears much earlier because the
    controller shapes the envelope down towards the current velocity. This
    holds for both the position-limits and the velocity-limits provocation.
    That is a reported behaviour rather than one measured here, so treat it as
    strong but not independently verified. Taking it at its word,
    this now latches both unconditionally, whatever the measured safety check
    has or has not observed yet -- see :meth:`MotionLimitChecker.check`'s
    precedence block.

    A revert of that block back to the margin-0 re-ask makes the first case
    below fail: measured ``dq`` is still (just) inside the envelope at the
    instant of the check, so the re-ask reads clean and only 13 is reported.
    """
    cap = upper_joint_velocity_limits(HOME)[0]
    over = spread(cap + 0.002)
    seed = robot_state_at(dq_d=spread(cap - 0.005))

    # The arm is still comfortably inside the envelope -- nobody has *measured*
    # anything wrong yet -- and the commanded step still latches both bits.
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, seed)
    checker.check_measured_velocity(HOME, spread(cap - 0.5))
    violation = checker.check(command(dq_c=over))
    assert violation.error_index == JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX
    assert violation.extra_error_index == JOINT_VELOCITY_VIOLATION_INDEX
    assert violation.error_indices == (
        JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
        JOINT_VELOCITY_VIOLATION_INDEX,
    )

    # Same command, with the arm already out past the envelope by less than
    # the measured-safety-check's own noise margin -- same result either way.
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, seed)
    assert checker.check_measured_velocity(HOME, spread(cap + 0.001)) is None
    both = checker.check(command(dq_c=over))
    assert both.error_index == JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX
    assert both.extra_error_index == JOINT_VELOCITY_VIOLATION_INDEX


def test_the_pure_safety_path_still_latches_3_alone():
    """Torque mode, no commanded velocity at all: no 13 to pair 3 with."""
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.TORQUE, robot_state_at())
    cap = upper_joint_velocity_limits(HOME)[5]

    violation = checker.check_measured_velocity(HOME, spread(cap + 1.0, index=5))
    assert violation.error_index == JOINT_VELOCITY_VIOLATION_INDEX
    assert violation.extra_error_index is None
    assert violation.error_indices == (JOINT_VELOCITY_VIOLATION_INDEX,)


def test_the_safety_controller_is_disarmed_between_motions():
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.TORQUE, robot_state_at())
    over = spread(upper_joint_velocity_limits(HOME)[5] + 1.0, index=5)
    assert checker.check_measured_velocity(HOME, over) is not None

    checker.end_motion()
    assert checker.check_measured_velocity(HOME, over) is None


def test_a_non_finite_measured_velocity_does_not_abort_the_client():
    """The backend blowing up is not the client's error to be told about."""
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.TORQUE, robot_state_at())

    assert checker.check_measured_velocity(HOME, [float("nan")] * 7) is None
    assert checker.check_measured_velocity(HOME, [float("inf")] * 7) is None


# -- start conditions --

def test_the_first_waypoint_must_be_where_the_robot_already_is():
    """joint_position_motion_generator_start_pose_invalid, seeded from q_d."""
    checker = position_checker()

    ok = [HOME[0] + 0.09] + HOME[1:]
    assert checker.check(command(q_c=ok)) is None

    checker = position_checker()
    off = [HOME[0] + 0.5] + HOME[1:]
    violation = checker.check(command(q_c=off))
    assert violation.error_index == JOINT_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX
    assert violation.error_name == "joint_position_motion_generator_start_pose_invalid"
    assert violation.value == pytest.approx(0.5)


def test_a_step_on_the_very_first_cycle_is_a_start_pose_error_not_a_discontinuity():
    """A first ``q_c`` of ``q_d + 0.2`` rad on cycle 0 is a start-pose error.

    Start-pose beats every discontinuity, whatever the magnitude: the client has
    not commanded a *trajectory* yet, it has commanded a place to begin, and
    beginning somewhere else is the error hardware names.
    """
    checker = position_checker()
    violation = checker.check(command(q_c=[HOME[0] + 0.2] + HOME[1:]))
    assert violation.error_index == JOINT_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX
    assert violation.error_index != JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX

    # ...and a step big enough to break the acceleration and jerk limits five
    # times over is still a start-pose error, not index 14.
    checker = position_checker()
    violation = checker.check(command(q_c=[HOME[0] + 1.0] + HOME[1:]))
    assert violation.error_index == JOINT_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX


def test_the_history_is_seeded_from_q_d_not_from_zero():
    """A motion starting far from the origin is not a 2.75 rad step."""
    away = [2.0, 1.0, -1.0, -2.0, 1.0, 2.0, -1.0]
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.POSITION, robot_state_at(q=away))

    assert checker.check(command(q_c=away)) is None


def test_the_opening_waypoint_rebases_the_history_to_a_standstill():
    """It is a standstill by construction, so cycle two is not a huge acceleration."""
    checker = position_checker()
    opening = [HOME[0] + 0.05] + HOME[1:]
    checker.record(command(q_c=opening))

    assert checker.check(command(q_c=opening)) is None


def test_a_velocity_motion_must_start_from_the_reported_dq_d():
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, robot_state_at(dq_d=spread(1.0)))

    assert checker.check(command(dq_c=spread(1.05))) is None

    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, robot_state_at(dq_d=spread(1.0)))
    violation = checker.check(command(dq_c=spread(2.0)))
    # 15, not 14: the enum has no joint-velocity start error, and a step in
    # ``dq_c`` is an acceleration discontinuity on this interface.
    assert violation.error_index == JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX
    assert violation.signal == "dq_c"


# -- interaction with packet loss (the sim holds, it does not extrapolate) --

def test_a_short_gap_is_judged_at_the_rate_the_client_actually_commanded():
    """A one-to-three cycle gap -- the sim's own loss -- must not read as a step.

    A client streaming a steady 1 rad/s loses two cycles. Its next waypoint is
    three cycles of motion away from the applied history, and differencing that
    over *one* cycle would report 3 rad/s out of nowhere. The echoed
    ``message_id`` says it travelled three cycles, and over three it is the ramp
    it always was.
    """
    velocity = 1.0
    checker = position_checker()
    drive_position_history(checker, list(HOME), spread(velocity), [0.0] * 7)
    last_received = [HOME[0] + velocity * DELTA_T] + HOME[1:]
    assert checker.check(command(message_id=5, q_c=last_received)) is None
    checker.record(command(message_id=5, q_c=last_received))

    resumed = [last_received[0] + 3 * velocity * DELTA_T] + HOME[1:]

    assert checker.check(command(message_id=8, q_c=resumed)) is None


def test_the_command_after_a_gap_is_still_checked():
    """No grace cycle: the resume waypoint may not teleport.

    Skipping the differential checks for the first command after a gap let that
    command go anywhere inside the joint range -- an implied 2700 rad/s that the
    checker reported as no violation at all, and that reached physics.
    """
    checker = position_checker()
    drive_position_history(checker, list(HOME), [0.0] * 7, [0.0] * 7)
    checker.record(command(message_id=5, q_c=list(HOME)))

    # A full-range teleport, arriving after a gap of any length.
    teleport = [2.7, 1.7, 2.9, -0.2, 2.8, 4.5, 3.0]
    violation = checker.check(command(message_id=45, q_c=teleport))

    assert violation is not None
    assert violation.error_index in (
        JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
        JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
    )


def test_a_stale_echo_is_judged_over_a_single_cycle():
    """An echo the client chose cannot buy it a wider interval.

    Sixty steps at 40 rad/s behind an echo five cycles old used to reach physics
    with enforcement on and no error raised at all: the stale command's
    differential checks were skipped outright.
    """
    checker = position_checker()
    drive_position_history(checker, list(HOME), [0.0] * 7, [0.0] * 7)
    checker.record(command(message_id=100, q_c=list(HOME)))

    step = [HOME[0] + 40.0 * DELTA_T] + HOME[1:]  # 40 rad/s for one cycle

    assert checker.check(command(message_id=95, q_c=step), fresh=False) is not None
    # ...and the same command answering its own cycle is caught too.
    assert checker.check(command(message_id=101, q_c=step)) is not None


def test_a_stale_echo_never_enters_the_history():
    """It is checked and applied, but it is not a sample of the trajectory."""
    checker = position_checker()
    drive_position_history(checker, list(HOME), [0.0] * 7, [0.0] * 7)
    checker.record(command(message_id=100, q_c=list(HOME)))

    assert checker.cycles_since_applied(command(message_id=101)) == 1
    # The server never records a non-fresh command; the baseline stays put.
    assert checker.cycles_since_applied(command(message_id=102)) == 2


def test_the_differencing_interval_is_capped():
    """A client-supplied ``message_id`` may not dilute a limit check to nothing."""
    checker = position_checker()
    checker.record(command(message_id=10, q_c=list(HOME)))

    assert checker.cycles_since_applied(command(message_id=10 + MAX_COALESCED_CYCLES)) == (
        MAX_COALESCED_CYCLES
    )
    assert checker.cycles_since_applied(command(message_id=10_000)) == MAX_COALESCED_CYCLES


def test_a_coalesced_command_is_differenced_over_the_cycles_it_travelled():
    """The sim's UDP loop keeps only the newest datagram; the robot never does.

    A client streaming a compliant 0.5 rad/s ramp has one of its commands
    dropped by the receive loop. Judged at 1 ms the survivor looks like 1 rad/s
    reached in a single cycle -- a 500 rad/s^2 acceleration out of nowhere. Its
    echoed ``message_id`` says it travelled two cycles, and over two cycles it
    is exactly the ramp it always was.
    """
    velocity = 0.5
    checker = position_checker()
    drive_position_history(checker, list(HOME), spread(velocity), [0.0] * 7, start_id=100)

    survivor = [HOME[0] + 2 * velocity * DELTA_T] + HOME[1:]
    assert checker.cycles_since_applied(command(message_id=105)) == 2
    assert checker.check(command(message_id=105, q_c=survivor)) is None

    # ...and had it really arrived one cycle later it would be the step it
    # looks like, which is what the robot would have refused.
    assert (
        checker.check(command(message_id=104, q_c=survivor)).error_index
        == JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX
    )


# -- the interval is the server's observation, not the client's claim --

def _cosine_ramp(peak_acceleration, cycles, period=0.2):
    """``q_c`` waypoints for a jerk-limited ramp on joint 1, from :data:`HOME`.

    ``v(t) = V/2 (1 - cos(pi t / T))``, so acceleration peaks at ``V pi / (2T)``
    and jerk is finite everywhere: a client whose own signal conforms with the
    interface specification at every cycle, with margin. Anything the checker
    says about this stream is about the checker.
    """
    peak_velocity = peak_acceleration * 2 * period / math.pi
    waypoints, q = [], list(HOME)
    for cycle in range(cycles):
        t = min(cycle * DELTA_T, period)
        velocity = peak_velocity / 2 * (1 - math.cos(math.pi * t / period))
        q = list(q)
        q[0] += velocity * DELTA_T
        waypoints.append(q)
    return waypoints


def _stream_with_gap(checker, waypoints, gap, gap_at=100, tamper=None):
    """Feed ``waypoints`` to ``checker``, losing ``gap - 1`` cycles at ``gap_at``.

    The server publishes a state every cycle whether or not the client answers
    it -- that is what :meth:`MotionLimitChecker.note_published` reports -- so
    the lost cycles still advance the published id. Returns the first violation,
    or None.
    """
    for cycle, q_c in enumerate(waypoints, start=1):
        checker.note_published(cycle)
        if gap_at < cycle < gap_at + gap:
            continue  # never reached the server
        if tamper is not None and cycle == gap_at + gap:
            q_c = tamper(q_c)
        received = command(message_id=cycle, q_c=q_c)
        violation = checker.check(received)
        if violation is not None:
            return violation
        checker.record(received)
    return None


@pytest.mark.parametrize("gap", [2, 3, 4, 5, 10])
def test_a_conforming_ramp_survives_a_loss_burst_of_any_length(gap):
    """The server saw the gap, so the gap is what the command is judged over.

    The interval used to be capped at :data:`MAX_COALESCED_CYCLES`, and past
    three lost cycles the resumed waypoint was differenced over three
    milliseconds of a journey that took ``gap``. That inflated its velocity by
    ``gap / 3``, and the acceleration that implied aborted the motion --
    ``joint_motion_generator_velocity_discontinuity`` in the middle of an
    ordinary approach, against a client whose signal never left the envelope.

    Not unbounded: the resumed command's velocity is an *average* over the gap,
    and recording it as the history's instantaneous one leaves the next command
    a jerk proportional to the gap. Past ~15 cycles of hold that trips 15 on its
    own, which is the honest consequence of holding instead of extrapolating
    (``docs/robot-state.md``) rather than anything the interval can fix.
    """
    checker = position_checker()
    assert _stream_with_gap(checker, _cosine_ramp(1.0, 400), gap) is None


@pytest.mark.parametrize("gap", [1, 3, 5])
def test_a_real_step_still_aborts_however_long_the_gap(gap):
    """The wider interval is not a grace cycle: the resume waypoint is checked.

    Half a radian in one waypoint is a step whatever the gap around it was, and
    dividing it by even nineteen cycles leaves a velocity the robot refuses.
    """
    checker = position_checker()
    violation = _stream_with_gap(
        checker,
        _cosine_ramp(1.0, 400),
        gap,
        tamper=lambda q_c: [q_c[0] + 0.5] + q_c[1:],
    )
    assert violation is not None
    assert violation.error_index in (
        JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
        JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
    )


def test_an_echo_ahead_of_what_the_server_published_buys_nothing():
    """The bound is the server's own observation; an invented id is not one.

    Inflating the echo divides every commanded derivative by whatever number the
    client chose, and 40 rad/s steps sail through. The server has published
    exactly one state since the applied command, so exactly one cycle is what
    the difference is taken over, whatever the datagram claims.
    """
    checker = position_checker()
    drive_position_history(checker, list(HOME), [0.0] * 7, [0.0] * 7)
    checker.record(command(message_id=100, q_c=list(HOME)))
    checker.note_published(101)

    step = [HOME[0] + 40.0 * DELTA_T] + HOME[1:]  # 40 rad/s for one cycle

    assert checker.cycles_since_applied(command(message_id=10_000)) == 1
    assert checker.check(command(message_id=10_000, q_c=step)) is not None


def test_a_late_command_is_judged_over_the_cycles_it_really_travelled():
    """A datagram the receive path got to late is not a one-cycle step.

    ``fresh`` is False for it -- it did not answer the cycle it arrived in --
    and it used to be differenced over a single cycle for that reason. But the
    state it answers is one the server published, so how far it travelled is a
    fact the server owns, and forcing it to one cycle reported twice the
    velocity the client commanded. That is what aborted conforming clients
    whenever this server's receive thread fell a cycle behind its publish
    thread.
    """
    velocity = 1.0
    checker = position_checker()
    drive_position_history(checker, list(HOME), spread(velocity), [0.0] * 7)
    last = [HOME[0] + velocity * DELTA_T] + HOME[1:]
    checker.record(command(message_id=5, q_c=last))
    checker.note_published(7)

    # Two cycles of the same 1 rad/s ramp, answering state 7 but read during 8.
    resumed = [last[0] + 2 * velocity * DELTA_T] + HOME[1:]

    assert checker.check(command(message_id=7, q_c=resumed), fresh=False) is None
    # A replay -- an id no newer than the history's -- still gets one cycle.
    assert checker.cycles_since_applied(command(message_id=4)) == 1


def test_the_drain_gate_waits_for_a_command_the_receive_path_has_not_read(
    mock_physics_sim,
):
    """A state must not go out while the client's last answer is still queued.

    The publish loop and the UDP receive thread are separate here, and nothing
    kept them in step: a receive thread descheduled for a few milliseconds left
    the publish loop emitting states whose ``q_d`` still described the last
    command it had managed to apply, while the answers to those cycles sat
    unread in this process's own socket. libfranka low-pass filters its next
    waypoint toward exactly that field, so the client's stream acquired a kink
    the checker then reported as a velocity discontinuity.
    """
    from franka_sim.franka_sim_server import FrankaSimServer

    server = FrankaSimServer(physics_sim=mock_physics_sim, enable_gripper=False)
    server.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.udp_socket.bind(("127.0.0.1", 0))
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Nothing queued and nothing in flight: the gate is a poll and a
        # return, and it charges the pacer *nothing* -- the poll and the clock
        # reads are this cycle's own work, and the loop already sleeps before
        # the gate rather than after it. Returning the couple of microseconds
        # they take pushed every deadline out by the gate's own cost and held
        # the publish rate at ~997.5 Hz in every control mode.
        assert server._drain_gate() == 0.0

        # One datagram nobody will read: the gate waits out its whole bound
        # rather than publishing over it, and says so.
        sender.sendto(b"unread", server.udp_socket.getsockname())
        waited = server._drain_gate()
        assert waited >= server._DRAIN_GATE_TIMEOUT

        # ...and once the receive path has taken it, the gate clears again.
        server.udp_socket.recvfrom(64)
        assert server._drain_gate() == 0.0
    finally:
        sender.close()
        server.udp_socket.close()


def test_the_drain_gate_waits_for_a_command_read_but_not_yet_applied(
    mock_physics_sim,
):
    """An empty socket is not the same as an applied command.

    ``_handle_commands`` takes the datagram out of the socket at the *start* of
    its turn and writes the commanded echo -- ``q_d``, ``O_T_EE_c``,
    ``elbow_c`` -- at the *end* of it, with the decoding, the communication
    accounting and the whole limit check in between. A gate that released on an
    empty socket therefore released into the one window where the echo is
    guaranteed stale, and did so exactly when it had engaged at all: the moment
    the receive thread drains the last queued datagram is the moment it still
    has all of its work ahead of it.

    Worse, the communication accounting has already run by then, so the cycle
    counts as answered and the publish loop does not extrapolate over it
    either. The reference simply freezes for one cycle; libfranka's 100 Hz
    command filter blends its next command toward the frozen value and emits a
    ``(1 - gain)`` x one-cycle step; and this server aborts the motion on the
    discontinuity it manufactured. Measured against a Cartesian elbow
    motion, a 179 urad/cycle elbow ramp came back 110 urad short on exactly
    that cycle -- 110 rad/s^2 against a 10 rad/s^2 limit.
    """
    from franka_sim.franka_sim_server import FrankaSimServer

    server = FrankaSimServer(physics_sim=mock_physics_sim, enable_gripper=False)
    server.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.udp_socket.bind(("127.0.0.1", 0))
    try:
        # The socket is empty, but the receive thread is mid-turn on a datagram
        # it has already read: the state that would go out now carries the echo
        # of the command *before* that one.
        server._commands_in_flight = 1
        assert server._drain_gate() >= server._DRAIN_GATE_TIMEOUT

        # The turn ends -- the command is applied, the echo is current -- and
        # the gate clears immediately.
        server._commands_in_flight = 0
        assert server._drain_gate() == 0.0
    finally:
        server.udp_socket.close()


# -- packet-loss extrapolation: the reference keeps moving through a gap --
#
# The FCI does not hold a missed motion-generator cycle, it continues it:
# Control "takes the previous waypoints and performs a linear extrapolation
# (keep acceleration constant and integrate) for the missed time step"
# (``docs/system_requirements.rst``), and what it extrapolated is what comes
# back in ``q_d``/``dq_d``/``ddq_d`` -- "even in case of packet losses"
# (``docs/overview.rst``). These tests pin that this sim does the same thing,
# per interface, and pin the two consequences that follow from it: a client
# resuming on its *own* last waypoint is now differenced against a reference
# that moved on, and an extrapolation is checked rather than exempt.


def _ease(fraction):
    """A 0 -> 1 shaping factor with zero slope at both ends, clamped past 1.

    ``0.5 (1 - cos(pi x))``. Every stream below is fed to a checker that
    enforces jerk limits, so a signal that starts or stops abruptly aborts on
    the way *in* and never reaches the behaviour under test. Easing is how these
    tests stay about extrapolation rather than about their own ramps.
    """
    return 0.5 * (1.0 - math.cos(math.pi * min(1.0, fraction)))


def run_gap(checker, cycles, first_lost_id):
    """Extrapolate ``cycles`` missed cycles, exactly as the publish loop does.

    The loop calls ``note_published(N)`` for the state it is about to send and
    ``extrapolate(N - 1)`` for the cycle that state just closed, so the applied
    history ends up sitting at the last unanswered state's id -- which is what
    puts the client's resumed command one cycle away from it.

    Returns the list of ``(command, violation)`` pairs, one per missed cycle.
    """
    results = []
    for offset in range(cycles):
        lost_id = first_lost_id + offset
        checker.note_published(lost_id + 1)
        results.append(checker.extrapolate(lost_id))
    return results


def stream_position_ramp(checker, cycles=400, peak_acceleration=1.0):
    """Feed a conforming jerk-limited ``q_c`` ramp, answering every cycle.

    Returns ``(waypoints, last_id)``. Asserts as it goes: anything this stream
    trips is the checker's fault, not the client's.
    """
    waypoints = _cosine_ramp(peak_acceleration, cycles)
    for cycle, q_c in enumerate(waypoints, start=1):
        checker.note_published(cycle)
        received = command(message_id=cycle, q_c=q_c)
        assert checker.check(received) is None, f"cycle {cycle}"
        checker.record(received)
    return waypoints, cycles


@pytest.mark.parametrize("gap", [1, 3, 5, 19])
def test_a_gap_in_a_position_stream_is_extrapolated_along_the_trajectory(gap):
    """``q_d`` keeps advancing at the velocity the client had reached.

    The ramp is flat by cycle 200 (the cosine reaches its plateau at 0.2 s), so
    the frozen acceleration is zero and the frozen velocity is the ramp's peak
    -- which makes the expected waypoint an exact arithmetic progression rather
    than something only the implementation can predict.
    """
    checker = position_checker()
    waypoints, last_id = stream_position_ramp(checker)
    velocity = (waypoints[-1][0] - waypoints[-2][0]) / DELTA_T

    extrapolated = run_gap(checker, gap, last_id + 1)

    for step, (substitute, violation) in enumerate(extrapolated, start=1):
        assert violation is None
        assert substitute["q_c"][0] == pytest.approx(
            waypoints[-1][0] + step * velocity * DELTA_T, abs=1e-12
        )
        # Only the commanded channel moves; the other six joints were standing
        # still and a frozen-acceleration extrapolation keeps them there.
        assert substitute["q_c"][1:] == pytest.approx(waypoints[-1][1:])


@pytest.mark.parametrize("gap", [1, 3, 5, 19])
def test_a_client_resuming_its_own_trajectory_after_a_gap_is_differenced_clean(gap):
    """The resumed command is one cycle from the extrapolated reference, and passes.

    This is the whole point of extrapolating: the reference tracked the client's
    trajectory through the silence, so the client's next waypoint -- the one it
    would have sent anyway -- is exactly one cycle of motion away from it. No
    grace, no widened interval, no violation.
    """
    checker = position_checker()
    waypoints, last_id = stream_position_ramp(checker, cycles=400 + gap + 1)
    conforming = waypoints[400 + gap]  # the waypoint for the cycle after the gap

    # Rewind: the client answered through cycle 400 and then went quiet.
    checker = position_checker()
    for cycle, q_c in enumerate(waypoints[:400], start=1):
        checker.note_published(cycle)
        received = command(message_id=cycle, q_c=q_c)
        checker.record(received)
    run_gap(checker, gap, 401)

    resumed = command(message_id=401 + gap, q_c=conforming)
    checker.note_published(401 + gap)

    assert checker.cycles_since_applied(resumed) == 1
    assert checker.check(resumed) is None


@pytest.mark.parametrize("gap", [1, 3, 5, 19])
def test_the_real_robot_resume_trap_now_fires(gap):
    """Resuming from your *own* last sent waypoint is a step, and hardware says so.

    The trap this simulator could not show you until now, and the reason
    libfranka warns that intermittent drops "could trigger `discontinuity`
    errors even when your source signals conform with the interface
    specification" (``docs/overview.rst``). A client that pauses mid-ramp and
    then picks up where *it* left off is commanding a step backwards the size of
    the whole gap, because the robot's reference did not pause with it. Holding
    the reference -- which is what this sim used to do -- made that step
    disappear, so the one bug a sim2real user most needs to find here was
    exactly the one they could not.
    """
    checker = position_checker()
    waypoints, last_id = stream_position_ramp(checker)
    velocity = (waypoints[-1][0] - waypoints[-2][0]) / DELTA_T

    run_gap(checker, gap, last_id + 1)

    # The client's own next waypoint, as if nothing had happened.
    own_next = list(waypoints[-1])
    own_next[0] += velocity * DELTA_T
    violation = checker.check(command(message_id=last_id + 1 + gap, q_c=own_next))

    assert violation is not None
    assert violation.error_index == JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX
    # The step it implies is the gap's worth of motion, undone in one cycle.
    assert violation.value == pytest.approx(-gap * velocity / DELTA_T, rel=1e-6)


@pytest.mark.parametrize("gap", [1, 3, 5])
def test_a_genuine_step_during_a_gap_still_aborts(gap):
    """Extrapolation is not laundering: the resume waypoint is checked in full.

    Half a radian is a step whatever happened around it. The interval it is
    differenced over is one cycle -- the reference is right there -- so there is
    not even an arithmetic route by which a gap could dilute it.
    """
    checker = position_checker()
    waypoints, last_id = stream_position_ramp(checker)
    run_gap(checker, gap, last_id + 1)

    stepped = [waypoints[-1][0] + 0.5] + waypoints[-1][1:]
    violation = checker.check(command(message_id=last_id + 1 + gap, q_c=stepped))

    assert violation is not None
    assert violation.error_index == JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX


def test_a_velocity_gap_extends_dq_at_the_frozen_acceleration():
    """``dq_c += a dt`` per missed cycle, with ``a`` frozen at the gap's start.

    Frozen, not continued. A previous attempt at this feature integrated *jerk*
    through the gap and carried a commanded 0.13 rad/s to 2.41 rad/s across
    twenty milliseconds of silence -- an eighteen-fold runaway dressed up as
    fidelity. The documented law keeps acceleration constant, and this pins that
    the ninth extrapolated sample is exactly nine acceleration-steps out, not
    nine of anything compounding.
    """
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, robot_state_at())

    # A jerk-limited velocity ramp: dq(t) = V/2 (1 - cos(pi t / T)).
    peak_velocity, period = 0.5, 0.2
    samples = []
    for cycle in range(1, 301):
        t = min(cycle * DELTA_T, period)
        dq = peak_velocity / 2 * (1 - math.cos(math.pi * t / period))
        samples.append(spread(dq))
        checker.note_published(cycle)
        received = command(message_id=cycle, dq_c=samples[-1])
        assert checker.check(received) is None, f"cycle {cycle}"
        checker.record(received)

    # On the cosine's plateau the acceleration is zero, so freezing it holds dq.
    frozen = (samples[-1][0] - samples[-2][0]) / DELTA_T
    assert frozen == pytest.approx(0.0, abs=1e-9)

    extrapolated = run_gap(checker, 19, 301)
    for substitute, violation in extrapolated:
        assert violation is None
        assert substitute["dq_c"][0] == pytest.approx(samples[-1][0], abs=1e-12)


def test_a_velocity_gap_mid_ramp_extends_at_the_acceleration_it_froze():
    """...and where the acceleration is *not* zero, the samples are a straight line."""
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, robot_state_at())

    acceleration = 4.0
    dq = 0.0
    # Ease the acceleration in over 200 cycles so the jerk limit is never the
    # thing under test, then hold it constant for a hundred more.
    for cycle in range(1, 301):
        dq += acceleration * _ease(cycle / 200.0) * DELTA_T
        checker.note_published(cycle)
        received = command(message_id=cycle, dq_c=spread(dq))
        assert checker.check(received) is None, f"cycle {cycle}"
        checker.record(received)

    extrapolated = run_gap(checker, 19, 301)

    for step, (substitute, violation) in enumerate(extrapolated, start=1):
        assert violation is None
        assert substitute["dq_c"][0] == pytest.approx(
            dq + step * acceleration * DELTA_T, abs=1e-9
        )
    # ...and ``ddq_d`` -- what the client reads back -- is the frozen value the
    # whole way through, never a re-derivation of it.
    assert checker.applied_derivatives()[0][0] == pytest.approx(acceleration, abs=1e-9)


def test_a_velocity_gap_after_one_flagged_command_borrows_the_clean_acceleration():
    """``_clean_age == 1``: the flagged record sits directly on clean history.

    One flagged step between a clean ramp and the gap is the duplicated- or
    reordered-datagram hazard: :meth:`_freeze_clean_locked` borrows the
    acceleration from the record exactly one cycle older, so the gap keeps
    accelerating at the rate the client was actually commanding, not at
    whatever the flagged step implied.
    """
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, robot_state_at())

    acceleration, dq = 4.0, 0.0
    mid = 0
    for mid in range(1, 201):
        dq += acceleration * DELTA_T
        checker.note_published(mid)
        received = command(message_id=mid, dq_c=spread(dq))
        assert checker.check(received) is None, f"cycle {mid}"
        checker.record(received)
    clean_velocity = dq

    mid += 1
    dq += 0.05  # a one-cycle step: flagged, but still applied (enforcement off)
    checker.note_published(mid)
    flagged = command(message_id=mid, dq_c=spread(dq))
    assert checker.check(flagged) is not None
    checker.record(flagged)
    assert checker._clean_age == 1

    extrapolated = run_gap(checker, 19, mid + 1)
    for step, (substitute, violation) in enumerate(extrapolated, start=1):
        assert violation is None
        assert substitute["dq_c"][0] == pytest.approx(
            dq + step * acceleration * DELTA_T, abs=1e-9
        )
    # The frozen acceleration really is the clean one, not zero.
    assert checker._joint.first[0] == pytest.approx(acceleration, abs=1e-9)
    assert clean_velocity < dq  # sanity: the flagged step really did move dq


def test_a_velocity_gap_after_two_flagged_commands_freezes_flat():
    """``_clean_age >= 2``: no *adjacent* clean record left to borrow from.

    A second flagged command in a row is the "no adjacent clean history"
    fallback, distinct from the single-flag case above: the acceleration goes
    to zero and the gap holds flat at the velocity the second flagged command
    left behind, rather than continuing to accelerate at the rate frozen one
    cycle further back. Regression pin for the bug where
    ``_Differentiator.freeze_flat`` zeroed ``second`` (an unread jerk at this
    depth) instead of ``first`` (the acceleration
    :meth:`_Differentiator.extrapolate_velocity` actually integrates): that
    left the flagged 50 rad/s^2 implied by the step below driving the whole
    gap, dragging ``dq_c`` from 0.9 rad/s to 1.85 rad/s over nineteen cycles
    instead of holding it flat.

    A mutant that widens ``_freeze_clean_locked``'s ``self._clean_age == 1``
    to ``self._clean_age is not None`` would take the freeze-*clean* branch
    here too, and the frozen acceleration would come back as the nonzero
    ``4.0 rad/s^2`` the ramp was commanding -- the ``== 0.0`` assertions below
    catch exactly that.
    """
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, robot_state_at())

    acceleration, dq = 4.0, 0.0
    mid = 0
    for mid in range(1, 201):
        dq += acceleration * DELTA_T
        checker.note_published(mid)
        received = command(message_id=mid, dq_c=spread(dq))
        assert checker.check(received) is None, f"cycle {mid}"
        checker.record(received)

    for _ in range(2):
        mid += 1
        dq += 0.05  # a one-cycle step: flagged, but still applied
        checker.note_published(mid)
        flagged = command(message_id=mid, dq_c=spread(dq))
        assert checker.check(flagged) is not None
        checker.record(flagged)
    assert checker._clean_age == 2
    dq_entering_gap = dq

    extrapolated = run_gap(checker, 19, mid + 1)
    for substitute, violation in extrapolated:
        assert violation is None
        # Flat, not still climbing at the clean 4.0 rad/s^2: the gap must not
        # amplify the flagged step's implied acceleration, and it must not
        # borrow an acceleration from two cycles back either.
        assert substitute["dq_c"][0] == pytest.approx(dq_entering_gap, abs=1e-9)
    assert checker._joint.first[0] == pytest.approx(0.0, abs=1e-9)
    # ``ddq_d`` -- what the client reads back -- is the frozen (zero)
    # acceleration the whole way through.
    assert checker.applied_derivatives()[0][0] == pytest.approx(0.0, abs=1e-9)
    assert checker._joint.value[0] == pytest.approx(dq_entering_gap, abs=1e-9)


def test_extrapolating_into_the_velocity_envelope_fires_the_envelope_violation():
    """An unclamped extrapolation runs out past the limits, and is told so.

    This is the behaviour libfranka documents as the *cost* of packet loss, not
    a bug to be papered over: a client whose own signal conforms can still be
    stopped, because the reference kept accelerating while it was quiet. Clamping
    the extrapolation to the envelope would make this sim quietly kinder than the
    robot in exactly the situation a sim2real user needs it to be honest about.
    """
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, robot_state_at())

    envelope = upper_joint_velocity_limits(HOME)[0]
    acceleration = 9.0
    # Accelerate up to just inside the envelope -- close enough that nineteen
    # cycles of the frozen 9 rad/s^2 (0.171 rad/s) cross it, and no closer, so
    # every *commanded* sample is legal with margin to spare.
    target = envelope - 0.1
    dq, cycle = 0.0, 0
    while dq < target:
        cycle += 1
        dq += acceleration * _ease(cycle / 200.0) * DELTA_T
        checker.note_published(cycle)
        received = command(message_id=cycle, dq_c=spread(dq))
        assert checker.check(received) is None, f"cycle {cycle}: dq={dq}"
        checker.record(received)
    assert dq < envelope, "the commanded stream must never leave the envelope"

    violations = [violation for _, violation in run_gap(checker, 19, cycle + 1)]

    tripped = [violation for violation in violations if violation is not None]
    assert tripped, "nineteen cycles at 9 rad/s^2 must leave the envelope"
    assert tripped[0].error_index == JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX
    # The safety controller is armed for every accepted Move, so hardware's
    # both-bits pairing applies to an extrapolated command exactly as it does to
    # a commanded one.
    assert tripped[0].extra_error_index == JOINT_VELOCITY_VIOLATION_INDEX


def test_a_torque_controller_holds_through_a_gap_and_never_extrapolates():
    """The other half of the quotation, and it is unchanged.

    "If a controller command packet is dropped, FCI will reuse the torques of
    the last successful received packet" (``docs/system_requirements.rst``).
    There is nothing to integrate -- a torque is not a waypoint -- so the
    checker refuses to invent one, and the server's doing nothing *is* the hold.
    """
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.TORQUE, robot_state_at())
    for cycle in range(1, 11):
        checker.note_published(cycle)
        received = command(message_id=cycle, tau_J_d=spread(cycle * 0.05))
        assert checker.check(received) is None
        checker.record(received)

    assert checker.extrapolate(11) is None


def test_an_extrapolation_before_any_command_is_refused():
    """A motion nobody has streamed to has no trajectory to continue."""
    checker = position_checker()

    assert checker.extrapolate(1) is None
    # ...and an idle checker has no motion at all.
    assert MotionLimitChecker().extrapolate(1) is None


def test_extrapolate_refuses_a_repeated_message_id():
    """Calling ``extrapolate`` twice for the same id must not double-integrate.

    ``message_id`` here is the publish loop's own counter, not client data, so
    a caller asking this method to re-derive a cycle it already committed is a
    bug, not a network condition to absorb. Left unguarded, a second call
    integrates a *second* cycle of motion into the history --
    ``commit_position`` advances ``first`` by the frozen ``second`` every time
    it runs -- while ``_applied_id`` (set at the bottom of :meth:`extrapolate`)
    only ever advances by the smaller of one cycle or the gap to
    ``message_id``, so it does not move on the repeat. The next real command
    then differences a doubly-advanced history against an interval its own
    bookkeeping says is one cycle, not two -- a two-cycle jump read back as a
    one-cycle one, which is exactly the shape of the 127 rad/s^2 abort this
    pins.
    """
    checker = position_checker()
    waypoints, last_id = stream_position_ramp(checker, cycles=400)

    checker.note_published(last_id + 2)
    first = checker.extrapolate(last_id + 1)
    assert first is not None

    with pytest.raises(AssertionError):
        checker.extrapolate(last_id + 1)  # the same id again

    # The one legitimate call's effect stands: a strictly later id still
    # works, and the history it left behind is untouched by the refused call.
    checker.note_published(last_id + 3)
    second = checker.extrapolate(last_id + 2)
    assert second is not None
    assert second[1] is None


def test_the_frozen_acceleration_never_re_derives_itself_from_its_own_output():
    """Nineteen cycles of extrapolation leave the acceleration exactly where it was.

    The failure this pins is subtle and was fatal once: advancing the history
    with the ordinary :meth:`record` path re-differences the sample it is given,
    so a chain of extrapolated waypoints re-derives the acceleration from values
    that the acceleration itself produced. One duplicated datagram amplified
    210x that way. Here the acceleration is read once, at the gap's start, and
    the nineteenth waypoint is exactly nineteen steps of *that* number.
    """
    # Cut the ramp off at half its period, where the cosine's acceleration is at
    # its peak rather than back at zero. Every other test here streams to the
    # plateau, and a *zero* frozen acceleration cannot tell freezing apart from
    # re-deriving -- both leave it at zero. This one has to.
    checker = position_checker()
    waypoints, last_id = stream_position_ramp(checker, cycles=100, peak_acceleration=5.0)
    velocity, acceleration = checker.applied_derivatives()
    assert acceleration[0] > 1.0, "the gap must open mid-ramp, not on the plateau"

    substitutes = [substitute for substitute, _ in run_gap(checker, 19, last_id + 1)]

    after_velocity, after_acceleration = checker.applied_derivatives()
    assert after_acceleration == pytest.approx(acceleration, abs=1e-12)
    assert after_velocity[0] == pytest.approx(
        velocity[0] + 19 * acceleration[0] * DELTA_T, abs=1e-12
    )
    # The waypoints themselves are the exact semi-implicit integral: each cycle
    # advances the velocity by the frozen acceleration *first* and then steps
    # the position by the result, so waypoint n has moved by the sum of
    # ``(v + n a dt) dt`` for n = 1..step. See
    # ``_Differentiator.extrapolate_position`` for why that order and not the
    # trapezoidal one -- the half-step the trapezoidal law leaves against the
    # backward differencing accumulates, and the client's own resume pays for it.
    for step, substitute in enumerate(substitutes, start=1):
        expected = waypoints[-1][0] + sum(
            (velocity[0] + n * acceleration[0] * DELTA_T) * DELTA_T
            for n in range(1, step + 1)
        )
        assert substitute["q_c"][0] == pytest.approx(expected, abs=1e-12)


def test_a_cartesian_pose_gap_continues_the_translation_and_the_rotation():
    """``O_T_EE_c`` keeps advancing: translation per axis, rotation by composition.

    The rotation is extended by an axis-angle increment applied on the right,
    which is the exact inverse of the differencing this checker already does
    (``log(R_{k-1}^T R_k)``), so a conforming stream's angular velocity survives
    the gap unchanged rather than being flattened to zero.
    """
    checker = MotionLimitChecker()
    checker.start_motion(
        ControlMode.CARTESIAN_POSE, robot_state_at(O_T_EE=pose(0.3, 0.0, 0.5))
    )

    # Eased in over 200 cycles, then a hundred more at a constant 0.05 m/s and
    # 0.2 rad/s -- both far inside the Cartesian limits, and both with zero
    # acceleration by the end, so the extrapolated samples are an exact
    # arithmetic progression in either half.
    speed, rate, cycles = 0.05, 0.2, 300
    z, angle = 0.5, 0.0
    for cycle in range(1, cycles + 1):
        commanded = pose(0.3, 0.0, z, rotation=rotation_about_z(angle))
        checker.note_published(cycle)
        received = command(message_id=cycle, O_T_EE_c=commanded)
        assert checker.check(received) is None, f"cycle {cycle}"
        checker.record(received)
        eased = _ease(cycle / 200.0)
        z += speed * eased * DELTA_T
        angle += rate * eased * DELTA_T

    last_z, last_angle = z - speed * DELTA_T, angle - rate * DELTA_T

    for step, (substitute, violation) in enumerate(
        run_gap(checker, 19, cycles + 1), start=1
    ):
        assert violation is None
        matrix = np.asarray(substitute["O_T_EE_c"]).reshape(4, 4).T
        assert matrix[2, 3] == pytest.approx(last_z + step * speed * DELTA_T, abs=1e-9)
        angle = float(np.linalg.norm(rotation_log(matrix[:3, :3])))
        assert angle == pytest.approx(last_angle + step * rate * DELTA_T, abs=1e-9)
        assert is_homogeneous_transformation(substitute["O_T_EE_c"])


def test_a_cartesian_velocity_gap_extends_the_twist_and_holds_the_elbow_sign():
    """The twist continues at frozen twist-acceleration; the branch flag does not move.

    ``elbow[1]`` is a configuration flag, not a quantity -- hardware's name for a
    change in it is ``cartesian_motion_generator_elbow_sign_inconsistent`` -- so
    a gap must not invent one. ``elbow[0]`` *is* an angle (joint 3 on an FR3) and
    follows the same 1-D position law the joints do.
    """
    checker = MotionLimitChecker()
    checker.start_motion(
        ControlMode.CARTESIAN_VELOCITY, robot_state_at(O_T_EE=list(MOCK_O_T_EE))
    )

    # The twist ends under a *constant* 2 m/s^2, so the frozen acceleration is
    # what carries it through the gap; the elbow ends at a constant 0.3 rad/s,
    # so its own frozen acceleration is zero. Both easings run over 200 cycles
    # and are then held for a hundred more.
    acceleration, elbow_rate, cycles = 2.0, 0.3, 300
    speed, elbow_angle = 0.0, HOME_ELBOW[0]
    for cycle in range(1, cycles + 1):
        received = command(
            message_id=cycle,
            O_dP_EE_c=[speed, 0.0, 0.0, 0.0, 0.0, 0.0],
            elbow_c=[elbow_angle, HOME_ELBOW[1]],
            valid_elbow=True,
        )
        checker.note_published(cycle)
        assert checker.check(received) is None, f"cycle {cycle}"
        checker.record(received)
        eased = _ease(cycle / 200.0)
        speed += acceleration * eased * DELTA_T
        elbow_angle += elbow_rate * eased * DELTA_T

    speed -= acceleration * DELTA_T
    elbow_angle -= elbow_rate * DELTA_T

    for step, (substitute, violation) in enumerate(run_gap(checker, 19, cycles + 1), start=1):
        assert violation is None
        assert substitute["O_dP_EE_c"][0] == pytest.approx(
            speed + step * acceleration * DELTA_T, abs=1e-9
        )
        assert substitute["elbow_c"][0] == pytest.approx(
            elbow_angle + step * elbow_rate * DELTA_T, abs=1e-9
        )
        assert substitute["elbow_c"][1] == HOME_ELBOW[1]
        assert substitute["valid_elbow"] is True


def test_a_datagram_that_turns_up_late_replaces_the_guess_it_stood_in_for():
    """The one place this sim's packet handling has to differ from the robot's.

    Hardware *drops* a command that missed its 1 ms window; this sim applies it,
    because a simulator is routinely driven by clients that are not realtime
    control loops and dropping their datagrams would leave the arm inert. Once
    missed cycles are extrapolated, those two choices collide: the extrapolation
    for cycle N already took one cycle's worth of motion, and the datagram that
    turns up for cycle N carries the *same* step, measured rather than guessed.
    Differencing the two against each other reports a reference that travelled
    nowhere followed by an enormous deceleration -- and against a stock
    libfranka client it did exactly that, aborting a conforming client's
    approach at a ``control_command_success_rate`` of 0.99.

    So the guess is thrown away when the real answer arrives.
    """
    checker = position_checker()
    waypoints, last_id = stream_position_ramp(checker)
    velocity = (waypoints[-1][0] - waypoints[-2][0]) / DELTA_T
    run_gap(checker, 1, last_id + 1)

    late = list(waypoints[-1])
    late[0] += velocity * DELTA_T
    delayed = command(message_id=last_id + 1, q_c=late)
    checker.note_published(last_id + 2)

    assert checker.rewind_extrapolation(delayed) is True
    assert checker.check(delayed) is None
    checker.record(delayed)
    # ...and the history is on the client's own data afterwards, so the next
    # waypoint continues from it cleanly.
    checker.note_published(last_id + 2)
    following = list(late)
    following[0] += velocity * DELTA_T
    assert checker.check(command(message_id=last_id + 2, q_c=following)) is None


def test_without_the_rewind_the_late_datagram_would_read_as_a_deceleration():
    """The companion to the test above: what the collision actually looks like.

    Identical stream, identical late datagram, the rewind simply not asked for.
    The reported acceleration is the client's whole velocity undone in one
    cycle, which is what was aborting conforming clients.
    """
    checker = position_checker()
    waypoints, last_id = stream_position_ramp(checker)
    velocity = (waypoints[-1][0] - waypoints[-2][0]) / DELTA_T
    run_gap(checker, 1, last_id + 1)

    late = list(waypoints[-1])
    late[0] += velocity * DELTA_T
    checker.note_published(last_id + 2)

    violation = checker.check(command(message_id=last_id + 1, q_c=late))

    assert violation is not None
    assert violation.error_index == JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX
    assert violation.value == pytest.approx(-velocity / DELTA_T, rel=1e-3)


def test_a_replay_is_not_the_answer_to_an_extrapolated_cycle():
    """A rewind for a duplicate would let a stale value overwrite the reference.

    The window is the run of losses itself. An echo at or before the cycle the
    gap started from is not new information -- it is the packet the history is
    already built on -- and the rule that stale echoes never touch the history
    is exactly what stops a replay from rewriting it.
    """
    checker = position_checker()
    waypoints, last_id = stream_position_ramp(checker)
    run_gap(checker, 3, last_id + 1)

    assert checker.rewind_extrapolation(command(message_id=last_id)) is False
    assert checker.rewind_extrapolation(command(message_id=last_id - 40)) is False
    # ...and an id past the run is not one the extrapolation stood in for either.
    assert checker.rewind_extrapolation(command(message_id=last_id + 99)) is False
    # Every cycle actually extrapolated is in the window, though.
    assert checker.rewind_extrapolation(command(message_id=last_id + 2)) is True


def test_a_run_of_losses_stays_rewindable_until_a_fresh_command_closes_it():
    """Not one rewind per run: one per *cycle* of the run, until it is answered.

    A stalled receive thread hands over a whole backlog at once and every
    datagram in it is the real answer to one of the run's cycles. Clearing the
    snapshot on the first of them left the rest applied-but-not-recorded, and the
    resume was then differenced against a reference frozen k cycles in the past
    -- 127 rad/s^2 for a two-cycle stall, growing linearly (see
    :func:`test_a_drained_backlog_of_late_datagrams_resumes_clean`).

    What moves the window's lower bound is *recording*, which is the caller's
    statement that the datagram was accepted; a rewind on its own is idempotent.
    """
    checker = position_checker()
    waypoints, last_id = stream_position_ramp(checker)
    run_gap(checker, 3, last_id + 1)

    # Idempotent: rewinding twice for the same unrecorded datagram is the same
    # restore twice, not a rewind and then a refusal.
    first_late = command(message_id=last_id + 1, q_c=waypoints[-1])
    assert checker.rewind_extrapolation(first_late) is True
    assert checker.rewind_extrapolation(first_late) is True

    # Recording it answers its cycle; the ones after it are still open.
    checker.record(first_late)
    assert checker.rewind_extrapolation(command(message_id=last_id + 1)) is False
    assert checker.rewind_extrapolation(command(message_id=last_id + 2)) is True
    assert checker.rewind_extrapolation(command(message_id=last_id + 3)) is True

    # A command from beyond the run is the client back in step, and closes it.
    checker.record(command(message_id=last_id + 9, q_c=waypoints[-1]))
    assert checker.rewind_extrapolation(command(message_id=last_id + 3)) is False
    # ...and with nothing extrapolated at all there is nothing to rewind.
    assert position_checker().rewind_extrapolation(command(message_id=1)) is False


def test_an_extrapolated_command_can_never_end_the_motion():
    """The substitute is a waypoint, not a decision the client made.

    ``motion_generation_finished`` rides in the same datagram as the command, so
    a naive copy of the last real one would let a gap re-signal an end of motion
    the client already signalled -- or, worse, signal one it never did.
    """
    checker = position_checker()
    waypoints, last_id = stream_position_ramp(checker, cycles=250)
    checker.record(
        command(
            message_id=last_id + 1,
            q_c=waypoints[-1],
            motion_generation_finished=True,
            torque_command_finished=True,
        )
    )

    substitute, _ = run_gap(checker, 1, last_id + 2)[0]

    assert substitute["motion_generation_finished"] is False
    assert substitute["torque_command_finished"] is False
    assert substitute["extrapolated"] is True


def test_a_rejected_command_leaves_no_trace_in_the_history():
    """check() is pure: the caller decides whether the command was applied."""
    checker = position_checker()
    drive_position_history(checker, list(HOME), [0.0] * 7, [0.0] * 7)
    before = checker.check(command(q_c=[HOME[0] + 0.5] + HOME[1:]))
    assert before is not None

    # The same good command still passes: the bad one never entered the history.
    assert checker.check(command(q_c=list(HOME))) is None


# -- non-finite commands --

@pytest.mark.parametrize(
    "mode,field,seed",
    [
        (ControlMode.POSITION, "q_c", "q_d"),
        (ControlMode.VELOCITY, "dq_c", "dq_d"),
        (ControlMode.TORQUE, "tau_J_d", "tau_J_d"),
    ],
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_command_is_a_violation_of_its_generator(mode, field, seed, bad):
    """Every ``value > limit`` comparison against a NaN is False, so nothing caught it.

    A NaN passed every check in the module, poisoned the backward differences it
    was recorded into -- NaN minus anything is NaN, for the rest of the motion --
    and reached both the physics backend and the wire.
    """
    checker = MotionLimitChecker()
    checker.start_motion(mode, robot_state_at(**{seed: list(HOME) if seed == "q_d" else [0.0] * 7}))

    values = [0.0] * 7
    values[3] = bad
    violation = checker.check(command(**{field: values}))

    assert violation is not None
    assert violation.fatal is True
    assert violation.signal == field
    assert violation.axis == "[3]"


def test_a_non_finite_twist_is_caught_on_the_base_too():
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.STEERING_DRIVE, robot_state_at())

    violation = checker.check(command(O_dP_EE_c=[0.1, float("nan"), 0.0, 0.0, 0.0, 0.0]))

    assert violation is not None
    assert violation.fatal is True
    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX


def test_a_non_finite_measured_position_does_not_abort_the_client():
    """The simulator's NaN is not the client's violation.

    ``joint_positions`` is the measured ``q`` the position-dependent velocity
    limits are evaluated at. Substituting zero for a non-finite one put joint 4
    outside its range -- 0.0 is not in [-3.0481, -0.1458] -- which drives that
    joint's velocity limit to 0.0 and turns the client's next perfectly ordinary
    command into a velocity-limits violation. Keeping the previous reference
    configuration blames nobody.
    """
    # Joint 4 is the one that matters: its range is [-3.0481, -0.1458], so a
    # substituted q of 0.0 sits outside it and its velocity limit collapses.
    velocity = [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0]
    checker = position_checker()
    drive_position_history(checker, list(HOME), velocity, [0.0] * 7)

    # A compliant continuation of that ramp, judged while the backend reports NaN.
    creeping = list(HOME)
    creeping[3] += 0.2 * DELTA_T
    violation = checker.check(
        command(message_id=5, q_c=creeping), [float("nan")] * 7
    )

    assert violation is None


def test_an_ordinary_violation_is_not_fatal():
    """Only the non-finite class is refused with enforcement off."""
    checker = position_checker()
    drive_position_history(checker, list(HOME), [0.0] * 7, [0.0] * 7)

    violation = checker.check(command(message_id=5, q_c=[HOME[0] + 0.5] + HOME[1:]))

    assert violation is not None
    assert violation.fatal is False


def test_a_new_motion_re_arms_the_latch():
    """It used to survive start_motion, so a second motion was checked forever.

    With the latch still set, ``_accept_within_motion_limits`` refused every
    command of the next motion without ever aborting it -- a motion in kMove
    whose every command was silently swallowed.
    """
    checker = position_checker()
    checker.latch()
    assert checker.violated is True

    checker.start_motion(ControlMode.POSITION, robot_state_at(q_d=list(HOME)))

    assert checker.violated is False


def test_an_end_for_a_motion_that_is_over_is_ignored():
    """A stale finish must not stop the checking on the motion that replaced it."""
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.POSITION, robot_state_at(q_d=list(HOME)), 1)
    checker.start_motion(ControlMode.POSITION, robot_state_at(q_d=list(HOME)), 2)

    checker.end_motion(motion_id=1)
    assert checker.active is True

    checker.end_motion(motion_id=2)
    assert checker.active is False


# -- reporting and lifecycle --

def test_each_violation_class_is_logged_once_per_motion():
    checker = position_checker()
    discontinuity = JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX
    first = Violation(discontinuity, "q_c", "joint 1", 1, 2)
    again = Violation(discontinuity, "q_c", "joint 4", 3, 2)
    out_of_range = JOINT_MOTION_GENERATOR_POSITION_LIMITS_VIOLATION_INDEX
    other = Violation(out_of_range, "q_c", "joint 1", 1, 2)

    assert checker.should_log(first) is True
    assert checker.should_log(again) is False
    assert checker.should_log(other) is True

    checker.start_motion(ControlMode.POSITION, robot_state_at())
    assert checker.should_log(first) is True


def test_the_description_names_the_error_the_axis_the_value_and_the_limit():
    described = Violation(
        JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
        "q_c",
        "joint 4",
        12.5,
        9.999,
        "rad/s^2",
    ).describe()

    assert "joint_motion_generator_velocity_discontinuity" in described
    assert "joint 4" in described
    assert "12.5 rad/s^2" in described
    assert "9.999 rad/s^2" in described


def test_ending_a_motion_stops_the_checking_but_keeps_the_latch():
    checker = position_checker()
    checker.latch()
    checker.end_motion()

    assert checker.check(command(q_c=[100.0] * 7)) is None
    assert checker.violated is True

    checker.recover()
    assert checker.violated is False


def test_the_env_var_is_the_switch_and_it_is_off_by_default():
    assert ENFORCE_ENV_VAR == "FRANKA_SIM_ENFORCE_MOTION_LIMITS"
    assert enforcement_enabled_by_env({}) is False
    assert enforcement_enabled_by_env({ENFORCE_ENV_VAR: "0"}) is False
    assert enforcement_enabled_by_env({ENFORCE_ENV_VAR: "false"}) is False
    assert enforcement_enabled_by_env({ENFORCE_ENV_VAR: "1"}) is True


# --- shared wire helpers -----------------------------------------------------


def pack_robot_command(
    message_id,
    q_c=None,
    dq_c=None,
    o_t_ee_c=None,
    o_dp_ee_c=None,
    elbow_c=None,
    tau_j_d=None,
    motion_finished=False,
):
    """Pack the UDP RobotCommand exactly as libfranka sends it.

    ``o_t_ee_c`` defaults to all-zeros because that is what libfranka actually
    puts on the wire for a *joint* motion: ``MotionGeneratorCommand{}`` is
    value-initialised and only the field the generator owns is filled in
    (``src/robot_impl.cpp:415-420``). A Cartesian pose test has to pass a real
    transform -- see :func:`pose`.

    ``elbow_c`` doubles as the ``valid_elbow`` flag: ``None`` packs the
    zero-filled, not-valid elbow libfranka sends for every elbow-less motion
    (``src/control_loop.cpp:284-286``), anything else packs the elbow *and* sets
    the flag, which is the only combination libfranka itself emits.
    """
    message = struct.pack("<Q", message_id)
    message += struct.pack("<7d", *(q_c if q_c is not None else [0.0] * 7))
    message += struct.pack("<7d", *(dq_c if dq_c is not None else [0.0] * 7))
    message += struct.pack("<16d", *(o_t_ee_c if o_t_ee_c is not None else [0.0] * 16))
    message += struct.pack("<6d", *(o_dp_ee_c if o_dp_ee_c is not None else [0.0] * 6))
    message += struct.pack("<2d", *(elbow_c if elbow_c is not None else [0.0] * 2))
    message += struct.pack("<B", 0 if elbow_c is None else 1)
    message += struct.pack("<B", 1 if motion_finished else 0)
    message += struct.pack("<7d", *(tau_j_d if tau_j_d is not None else [0.0] * 7))
    message += struct.pack("<B", 0)
    return message


def robot_state_field_slice(field, length):
    """Locate a field inside the packed 1377-byte RobotState, without magic offsets."""
    probe = RobotState()
    sentinel = [float(1000 + index) for index in range(length)]
    probe.state[field] = sentinel
    values = _ROBOT_STATE_PACKER.unpack(probe.pack_state())
    start = values.index(sentinel[0])
    return slice(start, start + length)


MESSAGE_ID_INDEX = 0
Q_SLICE = robot_state_field_slice("q", 7)
Q_D_SLICE = robot_state_field_slice("q_d", 7)
DQ_D_SLICE = robot_state_field_slice("dq_d", 7)
DDQ_D_SLICE = robot_state_field_slice("ddq_d", 7)
ELBOW_D_SLICE = robot_state_field_slice("elbow_d", 2)
ERRORS_SLICE = slice(-84, -43)
REFLEX_REASON_SLICE = slice(-43, -2)
ROBOT_MODE_INDEX = -2


class WireClient:
    """A libfranka-shaped client: every state is answered, echoing its message_id."""

    def __init__(self, host="127.0.0.1", port=COMMAND_PORT):
        self.host = host
        self.port = port
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.settimeout(5.0)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.settimeout(5.0)
        self.udp.bind((host, 0))
        self.udp_port = self.udp.getsockname()[1]
        self.server_udp_address = None
        self.last_message_id = 0

    def connect(self):
        self.tcp.connect((self.host, self.port))
        payload = struct.pack("<HH", 10, self.udp_port)
        header = MessageHeader(command=Command.kConnect, command_id=1, size=12 + len(payload))
        self.tcp.sendall(header.to_bytes() + payload)
        self.tcp.recv(12)
        status, _ = struct.unpack("<BH", self.tcp.recv(3))
        assert status == ConnectStatus.kSuccess

    def move(self, controller_mode, motion_generator_mode, command_id=2):
        payload = struct.pack(
            "<II3d3d",
            controller_mode.value,
            motion_generator_mode.value,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        )
        header = MessageHeader(command=Command.kMove, command_id=command_id, size=12 + len(payload))
        self.tcp.sendall(header.to_bytes() + payload)
        self.tcp.recv(12)
        return struct.unpack("<B3x", self.tcp.recv(4))[0]

    def read_move_response(self, timeout=5.0):
        self.tcp.settimeout(timeout)
        header = MessageHeader.from_bytes(self.tcp.recv(12))
        assert header.command == Command.kMove
        return struct.unpack("<B3x", self.tcp.recv(4))[0]

    def automatic_error_recovery(self, command_id=9):
        header = MessageHeader(Command.kAutomaticErrorRecovery, command_id, 12)
        self.tcp.sendall(header.to_bytes())
        self.tcp.recv(12)
        return struct.unpack("<B3x", self.tcp.recv(4))[0]

    def stop_move(self, command_id=8):
        header = MessageHeader(Command.kStopMove, command_id, 12)
        self.tcp.sendall(header.to_bytes())
        self.tcp.recv(12)
        return struct.unpack("<B3x", self.tcp.recv(4))[0]

    def read_state(self):
        """Return the newest published state, draining any backlog first.

        Real libfranka runs its UDP read on its own background thread, so it is
        always draining state at ~1 kHz no matter what the calling thread is
        blocked on -- including a pending TCP call such as
        ``automaticErrorRecovery()``. This test client is single-threaded: if
        it is off doing something else (blocked in :meth:`automatic_error_recovery`
        while the server's ``AutomaticErrorRecovery`` handler waits for the arm
        to settle, see ``FrankaSimServer._wait_for_standstill``) while the
        publish loop keeps sending, several datagrams pile up unread in the
        socket's receive buffer. UDP is FIFO, so the next plain ``recvfrom``
        would hand back the *oldest* of those -- a state as stale as the wait
        was long, which is not what any real client ever sees. Draining
        whatever is already buffered and keeping only the newest is what makes
        this stand-in behave like the background thread it is standing in for.
        """
        newest = None
        while True:
            readable, _, _ = select.select([self.udp], [], [], 0)
            if not readable:
                break
            newest = self.udp.recvfrom(4096)
        if newest is None:
            newest = self.udp.recvfrom(4096)
        data, address = newest
        self.server_udp_address = address
        assert len(data) == _ROBOT_STATE_PACKER.size
        values = _ROBOT_STATE_PACKER.unpack(data)
        self.last_message_id = values[MESSAGE_ID_INDEX]
        return values

    def answer(self, **command_fields):
        self.udp.sendto(
            pack_robot_command(self.last_message_id, **command_fields),
            self.server_udp_address,
        )

    def answer_stale(self, lag, **command_fields):
        """Answer with an echo ``lag`` cycles behind the newest state."""
        self.udp.sendto(
            pack_robot_command(max(1, self.last_message_id - lag), **command_fields),
            self.server_udp_address,
        )

    def stream(self, cycles, **command_fields):
        """Read ``cycles`` states, answering each one. Returns the last state."""
        state = None
        for _ in range(cycles):
            state = self.read_state()
            self.answer(**command_fields)
        return state

    def ramp(self, waypoints, field="q_c"):
        """Stream a 1 kHz waypoint list, advancing by the cycles that elapsed.

        libfranka's control loop accumulates ``period.toSec()`` -- the
        ``message_id`` delta between accepted states -- so a client that misses
        a state jumps its trajectory forward by exactly that much. Doing the
        same here keeps the stream a true 1 kHz sampling of the trajectory even
        when a datagram is dropped, which is what makes it comparable with the
        FCI's own extrapolation through the gap.
        """
        state = None
        index = 0
        previous_id = None
        for _ in waypoints:
            state = self.read_state()
            if previous_id is not None:
                index += max(1, self.last_message_id - previous_id)
            previous_id = self.last_message_id
            if index >= len(waypoints):
                break
            self.answer(**{field: list(waypoints[index])})
        return state

    def close(self):
        for sock in (self.tcp, self.udp):
            try:
                sock.close()
            except OSError:
                pass


def wait_for_server(port, timeout=5.0):
    """Block until the FCI accept loop answers on ``port``."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            probe = socket.create_connection(("127.0.0.1", port), timeout=1.0)
            probe.close()
            return True
        except OSError:
            time.sleep(0.05)
    return False


# -- packet loss, adversarially: the four ways extrapolation aborted a good client
#
# Every test below started life as a deterministic repro from an adversarial
# review of this feature, and every one of them failed on the code that review
# was given. They are grouped by the property they pin, and each names the
# revert that puts the failure back.


def stream_zero_jerk(checker, acceleration, cycles, start_id=1):
    """Feed ``q(t) = q0 + a t^2 / 2``: acceleration exactly ``a``, jerk exactly 0.

    The only stream on which freezing the acceleration is *exact* -- the client's
    acceleration never changes, so the reference and the client cannot part
    company for any physical reason. Whatever a resume reports on this stream is
    pure arithmetic slack in the extrapolation law, which is what makes it the
    right instrument for measuring one.

    Returns the waypoint function, so a test can ask for the client's own next
    waypoint past the end of the stream.
    """

    def waypoint(cycle):
        t = cycle * DELTA_T
        return [HOME[0] + 0.5 * acceleration * t * t] + list(HOME[1:])

    for cycle in range(1, cycles + 1):
        checker.note_published(cycle)
        received = command(message_id=cycle, q_c=waypoint(cycle - 1))
        violation = checker.check(received)
        assert violation is None, f"the stream itself is not conforming at {cycle}"
        checker.record(received)
    return waypoint


@pytest.mark.parametrize("acceleration", [0.5, 1.0, 2.0, 5.0])
@pytest.mark.parametrize("gap", [1, 3, 5, 10, 19])
def test_a_zero_jerk_stream_is_extrapolated_exactly_and_resumes_clean(acceleration, gap):
    """The extrapolation reproduces a constant-acceleration client waypoint for waypoint.

    The sharpest statement of the extrapolation law there is: on a stream whose
    acceleration genuinely is constant, freezing it and integrating must land on
    the client's *own* next waypoints, to the last bit -- and the resume must
    then report exactly the acceleration the client was commanding all along.

    The trapezoidal law (``q += v dt + a dt^2 / 2``) does not. Its step is
    ``a dt^2 / 2`` short of the semi-implicit one every cycle while the velocity
    it stores runs ``a dt / 2`` ahead of the step it took, and the two errors
    compound into a resume that reports ``a (gap / 2 + 1)``: 10 rad/s^2 for a
    conforming 5 rad/s^2 stream after a **two**-cycle gap, tripping
    ``joint_motion_generator_velocity_discontinuity`` on a client that did
    nothing wrong. Revert ``_Differentiator.extrapolate_position`` to that law
    and every parametrisation here fails.
    """
    checker = position_checker()
    waypoint = stream_zero_jerk(checker, acceleration, 60)
    velocity, frozen = checker.applied_derivatives()
    assert frozen[0] == pytest.approx(acceleration, abs=1e-9)

    substitutes = [substitute for substitute, _ in run_gap(checker, gap, 61)]

    # Waypoint for waypoint, the client's own trajectory.
    for step, substitute in enumerate(substitutes, start=1):
        assert substitute["q_c"][0] == pytest.approx(waypoint(59 + step)[0], abs=1e-12)
    # ...and the frozen acceleration is still exactly the client's.
    after_velocity, after_acceleration = checker.applied_derivatives()
    assert after_acceleration[0] == pytest.approx(acceleration, abs=1e-9)

    resume_id = 61 + gap
    checker.note_published(resume_id)
    resumed = command(message_id=resume_id, q_c=waypoint(resume_id - 1))

    # The acceleration the resume implies, computed the way the checker does:
    # over one cycle, from the last extrapolated waypoint and the velocity
    # stored with it. The number the broken law produced is spelled out.
    reported_velocity = (resumed["q_c"][0] - substitutes[-1]["q_c"][0]) / DELTA_T
    reported = (reported_velocity - after_velocity[0]) / DELTA_T
    assert reported == pytest.approx(acceleration, abs=1e-6)
    assert reported != pytest.approx(acceleration * (gap / 2 + 1), abs=1e-6) or gap == 0

    assert checker.cycles_since_applied(resumed) == 1
    assert checker.check(resumed) is None


@pytest.mark.parametrize("peak_acceleration", [2.0, 5.0])
@pytest.mark.parametrize("gap", [1, 3, 5])
def test_a_jerk_limited_client_resuming_mid_ramp_is_not_aborted(peak_acceleration, gap):
    """The same, on a stream that is still *changing* its acceleration.

    Here the reference and the client genuinely do part company -- freezing an
    acceleration a client is still ramping is an approximation, and libfranka
    documents that as the cost of packet loss. What must not be there on top of
    it is the arithmetic slack: with the trapezoidal law a five-cycle gap at
    5.1 rad/s^2 reported 24.5 rad/s^2 and aborted, against 11.7 for the client's
    own jerk alone. Revert ``_Differentiator.extrapolate_position`` and the
    3- and 5-cycle rows fail.
    """
    waypoints = _cosine_ramp(peak_acceleration, 60 + gap + 2)
    checker = position_checker()
    for cycle in range(1, 61):
        checker.note_published(cycle)
        received = command(message_id=cycle, q_c=waypoints[cycle - 1])
        assert checker.check(received) is None, f"the stream is not conforming at {cycle}"
        checker.record(received)
    # Mid-ramp, where the cosine's acceleration is at its peak rather than back
    # at zero: a gap opening on the plateau cannot tell the two laws apart.
    assert checker.applied_derivatives()[1][0] > 1.0

    run_gap(checker, gap, 61)

    resume_id = 61 + gap
    checker.note_published(resume_id)
    assert checker.check(command(message_id=resume_id, q_c=waypoints[resume_id - 1])) is None


def test_a_pose_gap_with_a_non_zero_rotational_acceleration_resumes_clean():
    """The rotation half of the law, where a *zero* acceleration cannot judge it.

    Every other pose test streams to a plateau, and on a plateau the rotational
    acceleration is zero -- so the increment is ``omega dt`` whichever order the
    velocity update is written in, and a wrong one is invisible. This stream is
    zero-jerk with a genuinely non-zero rotational acceleration, so the
    extrapolated rotation must land exactly on the client's own, and the resume
    must be clean.

    Revert the rotational half of ``_PoseDifferentiator.extrapolate`` to
    ``omega dt + alpha dt^2 / 2`` and the resume aborts.
    """
    linear, angular = 0.4, 1.5  # m/s^2 and rad/s^2, both far inside the limits

    def waypoint(cycle):
        t = cycle * DELTA_T
        return pose(
            0.3, 0.0, 0.5 + 0.5 * linear * t * t,
            rotation=rotation_about_z(0.5 * angular * t * t),
        )

    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.CARTESIAN_POSE, robot_state_at(O_T_EE=waypoint(0)))
    for cycle in range(1, 61):
        checker.note_published(cycle)
        received = command(message_id=cycle, O_T_EE_c=waypoint(cycle - 1))
        assert checker.check(received) is None, f"the stream is not conforming at {cycle}"
        checker.record(received)

    # The stream really is accelerating in *both* halves -- differenced here the
    # way the checker does it, so the premise of the test is on the record and a
    # plateau cannot creep back into it. (``applied_derivatives`` reports the
    # joint generators only, which is why this is spelled out.)
    def rate(cycle):
        earlier = np.asarray(waypoint(cycle - 1)).reshape(4, 4).T
        later = np.asarray(waypoint(cycle)).reshape(4, 4).T
        angular_rate = rotation_log(earlier[:3, :3].T @ later[:3, :3]) / DELTA_T
        return (later[2, 3] - earlier[2, 3]) / DELTA_T, float(angular_rate[2])

    assert (rate(59)[0] - rate(58)[0]) / DELTA_T == pytest.approx(linear, abs=1e-6)
    assert (rate(59)[1] - rate(58)[1]) / DELTA_T == pytest.approx(angular, abs=1e-6)

    substitutes = [substitute for substitute, violation in run_gap(checker, 19, 61)]

    for step, substitute in enumerate(substitutes, start=1):
        expected = np.asarray(waypoint(59 + step)).reshape(4, 4).T
        actual = np.asarray(substitute["O_T_EE_c"]).reshape(4, 4).T
        assert actual[2, 3] == pytest.approx(expected[2, 3], abs=1e-12)
        # The rotation, compared as the angle of the relative rotation so the
        # comparison is on SO(3) rather than on sixteen floats.
        relative = expected[:3, :3].T @ actual[:3, :3]
        assert float(np.linalg.norm(rotation_log(relative))) == pytest.approx(0.0, abs=1e-12)
        assert is_homogeneous_transformation(substitute["O_T_EE_c"])

    resume_id = 61 + 19
    checker.note_published(resume_id)
    resumed = command(message_id=resume_id, O_T_EE_c=waypoint(resume_id - 1))
    assert checker.cycles_since_applied(resumed) == 1
    assert checker.check(resumed) is None


def test_an_elbow_gap_with_a_non_zero_acceleration_resumes_clean():
    """``elbow_c[0]`` rides the same 1-D position law, and pays the same slack.

    A Cartesian-velocity motion whose elbow is accelerating: the twist half is
    held at a standstill so nothing but the elbow can be the thing that aborts.
    """
    acceleration = 3.0

    def angle(cycle):
        t = cycle * DELTA_T
        return HOME_ELBOW[0] + 0.5 * acceleration * t * t

    checker = MotionLimitChecker()
    checker.start_motion(
        ControlMode.CARTESIAN_VELOCITY, robot_state_at(O_T_EE=list(MOCK_O_T_EE))
    )
    for cycle in range(1, 61):
        checker.note_published(cycle)
        received = command(
            message_id=cycle,
            elbow_c=[angle(cycle - 1), HOME_ELBOW[1]],
            valid_elbow=True,
        )
        assert checker.check(received) is None, f"the stream is not conforming at {cycle}"
        checker.record(received)

    substitutes = [substitute for substitute, _ in run_gap(checker, 19, 61)]

    for step, substitute in enumerate(substitutes, start=1):
        assert substitute["elbow_c"][0] == pytest.approx(angle(59 + step), abs=1e-12)

    resume_id = 61 + 19
    checker.note_published(resume_id)
    resumed = command(
        message_id=resume_id,
        elbow_c=[angle(resume_id - 1), HOME_ELBOW[1]],
        valid_elbow=True,
    )
    assert checker.check(resumed) is None


def test_the_velocity_laws_are_unchanged_and_resume_clean():
    """``dq_c`` and ``O_dP_EE_c`` had no slack to begin with, and still have none.

    Their ``first`` *is* the acceleration, so ``value += first dt`` differences
    back to exactly ``first``: the position law's whole problem cannot arise
    here. Pinned so a future tidy-up cannot "make them consistent" with a law
    they were already consistent with.
    """
    acceleration = 4.0

    def sample(cycle):
        return acceleration * cycle * DELTA_T

    for mode, field, build in (
        (ControlMode.VELOCITY, "dq_c", lambda v: spread(v)),
        (ControlMode.CARTESIAN_VELOCITY, "O_dP_EE_c", lambda v: [v, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ):
        checker = MotionLimitChecker()
        checker.start_motion(mode, robot_state_at(O_T_EE=list(MOCK_O_T_EE)))
        for cycle in range(1, 61):
            checker.note_published(cycle)
            received = command(message_id=cycle, **{field: build(sample(cycle - 1))})
            assert checker.check(received) is None, f"{mode} cycle {cycle}"
            checker.record(received)

        for step, (substitute, violation) in enumerate(run_gap(checker, 19, 61), start=1):
            assert violation is None
            assert substitute[field][0] == pytest.approx(sample(59 + step), abs=1e-12)

        resume_id = 61 + 19
        checker.note_published(resume_id)
        resumed = command(message_id=resume_id, **{field: build(sample(resume_id - 1))})
        assert checker.check(resumed) is None, mode


# -- a drained backlog: the run of losses stays rewindable ---------------------


def drain_backlog(checker, waypoints, warmup, stall):
    """Stall for ``stall`` cycles, then hand the whole backlog over at once.

    Exactly what a receive thread that was descheduled produces: the publish
    loop extrapolated every cycle of the stall, and then every one of the
    client's real datagrams for those cycles is sitting in the socket and is
    drained in order, newest last.
    """
    rewound = []
    for offset in range(stall):
        lost = warmup + 1 + offset
        checker.note_published(lost + 1)
        checker.extrapolate(lost)
    for offset in range(stall):
        late = command(message_id=warmup + 1 + offset, q_c=waypoints[warmup + offset])
        outcome = checker.absorb_command(late, fresh=False)
        rewound.append(outcome.rewound)
        assert outcome.violation is None, f"late datagram {late['message_id']}"
    return rewound


@pytest.mark.parametrize("stall", [1, 2, 3, 5, 11])
def test_a_drained_backlog_of_late_datagrams_resumes_clean(stall):
    """A receive-thread stall must not abort the client that was streaming through it.

    The bug: ``rewind_extrapolation`` was one-shot per run of losses, so only the
    *first* datagram of the backlog was rewound and recorded. The rest were
    applied but never entered the history, which was left ``stall - 1`` cycles in
    the past -- while ``extrapolate``'s ``_applied_id = max(...)`` dragged the id
    it claimed to sit at forward to the current cycle. The client's next
    perfectly conforming command was then differenced over *one* cycle against a
    reference that had not moved for several, and reported 127 rad/s^2 for a
    two-cycle stall, 636 for six, 1273 for eleven -- linear in the stall, and
    every one of them an abort under enforcement.

    Two reverts put it back, and either one alone is enough: clearing
    ``_gap_snapshot`` inside ``rewind_extrapolation``, or restoring
    ``_applied_id = max(self._applied_id or 0, int(message_id))`` at the end of
    ``extrapolate``.
    """
    warmup = 400
    waypoints = _cosine_ramp(1.0, warmup + stall + 4)
    checker = position_checker()
    for cycle in range(1, warmup + 1):
        checker.note_published(cycle)
        received = command(message_id=cycle, q_c=waypoints[cycle - 1])
        assert checker.check(received) is None
        checker.record(received)

    rewound = drain_backlog(checker, waypoints, warmup, stall)

    # Every datagram of the backlog is the real answer to a cycle that was
    # guessed, and every one of them gets its guess thrown away.
    assert rewound == [True] * stall

    # The publish loop ticks again -- the client is still not answering *this*
    # cycle -- and only then does the client resume, conforming.
    lost = warmup + 1 + stall
    checker.note_published(lost + 1)
    substitute, violation = checker.extrapolate(lost)
    assert violation is None
    assert substitute["q_c"][0] == pytest.approx(waypoints[lost - 1][0], abs=1e-6)

    resume_id = lost + 1
    checker.note_published(resume_id)
    resumed = command(message_id=resume_id, q_c=waypoints[resume_id - 1])
    assert checker.cycles_since_applied(resumed) == 1
    assert checker.check(resumed) is None


def test_a_partly_drained_backlog_leaves_the_applied_id_where_the_history_is():
    """The id the history claims never runs ahead of the motion it integrated.

    The other half of the same bug, isolated. Some of the backlog arrives and
    some of it never does; the reference is then genuinely behind, and saying so
    is what keeps the client's resume differenced over the interval it really
    covers. ``_applied_id = max(applied, message_id)`` claimed the current cycle
    regardless -- which is how the stale reference above turned into an abort.
    """
    warmup = 400
    waypoints = _cosine_ramp(1.0, warmup + 12)
    checker = position_checker()
    for cycle in range(1, warmup + 1):
        checker.note_published(cycle)
        received = command(message_id=cycle, q_c=waypoints[cycle - 1])
        assert checker.check(received) is None
        checker.record(received)

    run_gap(checker, 5, warmup + 1)  # cycles 401..405 guessed

    # Only the datagram for 401 ever turns up.
    late = command(message_id=warmup + 1, q_c=waypoints[warmup])
    assert checker.absorb_command(late, fresh=False).rewound is True

    # Two more cycles go unanswered. The history has now integrated one real
    # command (401) and two guesses, so it sits at 403 -- not at 407.
    for offset in range(2):
        lost = warmup + 6 + offset
        checker.note_published(lost + 1)
        checker.extrapolate(lost)

    resume_id = warmup + 8
    checker.note_published(resume_id)
    resumed = command(message_id=resume_id, q_c=waypoints[resume_id - 1])
    assert checker.cycles_since_applied(resumed) == 5
    assert checker.check(resumed) is None


# -- the publish thread cannot land in the middle of a late datagram ----------


@pytest.mark.parametrize("gap", [1, 2, 3, 5])
@pytest.mark.parametrize("tick_first", [True, False])
def test_a_publish_tick_around_an_absorbed_late_datagram_cannot_abort(gap, tick_first):
    """Rewind, check and record are one operation, so a tick lands outside it.

    Split into three calls they were three acquisitions of the checker's lock,
    and the publish thread's own ``extrapolate`` could land in either window: the
    rewind restores the pre-gap history, the interleaved extrapolation advances
    it again from there, and the check and record that follow are measured
    against a reference that moved under them. That reported 64 to 191 rad/s^2 on
    an ordinary conforming ramp and aborted it.

    With :meth:`absorb_command` the tick can only be before or after the whole
    operation, and this pins both orderings clean. Revert the atomicity -- make
    ``_absorb_within_motion_limits`` call ``rewind_extrapolation``, ``check`` and
    ``record`` separately again -- and the interleaved order fails; the lock is
    pinned separately by
    :func:`test_absorbing_a_late_datagram_holds_the_checker_lock_throughout`.
    """
    warmup = 400
    waypoints = _cosine_ramp(1.0, warmup + gap + 4)
    checker = position_checker()
    for cycle in range(1, warmup + 1):
        checker.note_published(cycle)
        received = command(message_id=cycle, q_c=waypoints[cycle - 1])
        assert checker.check(received) is None
        checker.record(received)
    run_gap(checker, gap, warmup + 1)

    late_id = warmup + gap  # the last extrapolated cycle
    late = command(message_id=late_id, q_c=waypoints[late_id - 1])
    lost = warmup + gap + 1

    def tick():
        checker.note_published(lost + 1)
        checker.extrapolate(lost)

    if tick_first:
        tick()
    outcome = checker.absorb_command(late, fresh=False)
    assert outcome.rewound is True
    assert outcome.violation is None
    assert outcome.recorded is True
    if not tick_first:
        tick()

    resume_id = lost + 1
    checker.note_published(resume_id)
    assert checker.check(command(message_id=resume_id, q_c=waypoints[resume_id - 1])) is None


def test_absorbing_a_late_datagram_holds_the_checker_lock_throughout():
    """...and the lock really is held for the whole of it, not merely per step.

    Deterministic: a thread that tries to extrapolate while ``absorb_command`` is
    half way through must *block*. If the rewind, the check and the record each
    took the lock on their own it would sail through instead, and the interleaved
    order above would be reachable again.
    """
    warmup = 200
    waypoints = _cosine_ramp(1.0, warmup + 4)
    checker = position_checker()
    for cycle in range(1, warmup + 1):
        checker.note_published(cycle)
        received = command(message_id=cycle, q_c=waypoints[cycle - 1])
        checker.record(received)
    run_gap(checker, 2, warmup + 1)

    finished = threading.Event()
    escaped = []
    original = MotionLimitChecker._check_locked

    def racing_check(self, command, cycles):
        # Disarmed *first*, and before the thread starts: the racing
        # extrapolate() calls _check_locked too, and a patch still in place
        # would send it into this function recursively -- where it would sit out
        # its own 0.5 s and make the test pass for a reason that has nothing to
        # do with the lock.
        MotionLimitChecker._check_locked = original
        # The publish thread, arriving between the rewind and the record.
        # Daemon, so a build of this test that *does* deadlock fails loudly
        # rather than wedging the whole session at interpreter exit.
        thread = threading.Thread(
            target=lambda: (checker.extrapolate(warmup + 3), finished.set()),
            daemon=True,
        )
        thread.start()
        # Judged here, while the lock is still held: once absorb_command returns
        # the thread is free to finish and the observation would be a race.
        escaped.append(finished.wait(timeout=0.5))
        return original(self, command, cycles)

    MotionLimitChecker._check_locked = racing_check
    try:
        late = command(message_id=warmup + 2, q_c=waypoints[warmup + 1])
        checker.absorb_command(late, fresh=False)
    finally:
        MotionLimitChecker._check_locked = original

    assert escaped == [False], "extrapolate() ran inside absorb_command's lock"


@pytest.mark.parametrize("gap", [1, 2, 3, 5])
def test_a_publish_tick_between_a_bare_rewind_and_its_record_cannot_abort(gap):
    """The same interleaving, driven through the *primitives* rather than the op.

    The reviewer's original repro, kept verbatim because it is the one a caller
    outside this server can still produce: rewind, publish tick, check, record.
    Every call takes and releases the checker's lock on its own, so no threads
    are needed to reproduce what two threads would do -- and on the code this
    was found in it reported 63.7, 63.7 and 191.0 rad/s^2 for gaps of one, three
    and five, and aborted a conforming ramp.

    Two things make it clean now, and reverting either one puts a failure back:
    the run of losses stays rewindable (so the record lands where it should),
    and a bare rewind makes the tick that follows it *hold* rather than guess
    the cycle whose real answer is in flight.
    """
    warmup = 400
    waypoints = _cosine_ramp(1.0, warmup + gap + 4)
    checker = position_checker()
    for cycle in range(1, warmup + 1):
        checker.note_published(cycle)
        received = command(message_id=cycle, q_c=waypoints[cycle - 1])
        assert checker.check(received) is None
        checker.record(received)
    run_gap(checker, gap, warmup + 1)

    late_id = warmup + gap
    late = command(message_id=late_id, q_c=waypoints[late_id - 1])
    # UDP thread: the late datagram's guess is thrown away...
    assert checker.rewind_extrapolation(late) is True
    # ...and the publish thread ticks in the window before it is recorded.
    lost = warmup + gap + 1
    checker.note_published(lost + 1)
    checker.extrapolate(lost)
    # UDP thread resumes.
    assert checker.check(late) is None
    checker.record(late)

    resume_id = lost + 1
    checker.note_published(resume_id)
    assert checker.check(command(message_id=resume_id, q_c=waypoints[resume_id - 1])) is None


def test_a_bare_rewind_makes_the_next_extrapolation_hold():
    """The belt to that atomicity, for a caller that takes the rewind on its own.

    ``rewind_extrapolation`` is public and a caller may use it directly; having
    done so it holds a rewound history with the datagram not yet recorded, and a
    publish tick landing there would guess the very cycle whose real answer is in
    flight. One extrapolation holds instead. One-shot, so a caller that then
    throws the datagram away is not wedged.
    """
    checker = position_checker()
    waypoints, last_id = stream_position_ramp(checker, cycles=200)
    run_gap(checker, 2, last_id + 1)

    assert checker.rewind_extrapolation(command(message_id=last_id + 1)) is True
    checker.note_published(last_id + 4)
    assert checker.extrapolate(last_id + 3) is None
    # ...and the next one is served normally.
    assert checker.extrapolate(last_id + 3) is not None


# -- a flagged command does not get to seed nineteen cycles of integration ----


def test_a_duplicate_datagram_cannot_poison_the_frozen_acceleration():
    """One bad sample, nineteen cycles of amplification. Enforcement off.

    The default. A duplicated datagram inside its own cycle is *fresh* by the
    comm tracker's reckoning (its id equals the published one), so it is checked,
    flagged -- ``-127 rad/s^2``, correctly -- and, because enforcement is off,
    still applied and recorded. Its backward differences are zero velocity and
    ``-v/dt`` of acceleration, and freezing *those* through a nineteen-cycle gap
    dispatched a reference running backwards at nineteen times the commanded
    speed: 0.127 rad/s forwards became 2.419 rad/s backwards, straight into
    physics and onto the wire.

    Log-only semantics are unchanged -- the flagged command still governs the
    cycle it commanded -- but it does not get to govern the nineteen after it.
    Revert by dropping the ``_freeze_from_clean`` branch at the top of
    ``extrapolate``.
    """
    checker = position_checker()
    waypoints, last_id = stream_position_ramp(checker)
    commanded_velocity, commanded_acceleration = checker.applied_derivatives()
    assert commanded_velocity[0] > 0.1

    duplicate = command(message_id=last_id, q_c=waypoints[-1])
    assert checker.check(duplicate) is not None  # reported...
    checker.record(duplicate)  # ...and applied anyway: enforcement is off
    poisoned_velocity, poisoned_acceleration = checker.applied_derivatives()
    assert poisoned_acceleration[0] == pytest.approx(-commanded_velocity[0] / DELTA_T)

    substitutes = [substitute for substitute, _ in run_gap(checker, 19, last_id + 1)]

    # The gap is frozen from the last command nobody objected to, so the
    # reference simply carries on at the velocity the client was commanding.
    for step, substitute in enumerate(substitutes, start=1):
        assert substitute["q_c"][0] == pytest.approx(
            waypoints[-1][0] + step * commanded_velocity[0] * DELTA_T, abs=1e-9
        )
    velocity, acceleration = checker.applied_derivatives()
    assert velocity[0] == pytest.approx(commanded_velocity[0], abs=1e-9)
    assert acceleration[0] == pytest.approx(commanded_acceleration[0], abs=1e-9)


def test_a_duplicate_that_enforcement_refuses_never_reaches_the_history_at_all():
    """Enforced, the behaviour is unchanged: the flagged command is not recorded.

    Which is why the clean-history freeze is a fix for the *default* path only.
    Here the duplicate is refused outright, so there is nothing to poison and
    nothing to fall back from.
    """
    checker = MotionLimitChecker(enforce=True)
    checker.start_motion(ControlMode.POSITION, robot_state_at())
    waypoints, last_id = stream_position_ramp(checker)
    before = checker.applied_derivatives()

    duplicate = command(message_id=last_id, q_c=waypoints[-1])
    outcome = checker.absorb_command(duplicate, fresh=True, enforce=True)

    assert outcome.violation is not None
    assert outcome.accepted is False
    assert outcome.recorded is False
    assert checker.applied_derivatives() == before


def test_a_client_permanently_outside_a_limit_still_gets_its_gap_extrapolated():
    """The fallback, when there is no adjacent clean history to borrow.

    A client streaming past a joint stop with enforcement off is a supported way
    to use this sim, and every one of its commands is flagged -- so there is no
    clean record to freeze from. Zeroing the *acceleration* and keeping the
    velocity is the fallback: nothing can be amplified, and the reference still
    tracks the client instead of freezing solid, which is what a wholesale
    borrow-the-last-clean-derivatives rule would have done to it.
    """
    checker = position_checker()
    # Joint 4's range is [-3.0481, -0.1458]; a ramp from zero is outside it from
    # the very first waypoint, and flagged every cycle.
    step = 0.001
    for cycle in range(1, 21):
        checker.note_published(cycle)
        received = command(message_id=cycle, q_c=[cycle * step] * 7)
        assert checker.check(received) is not None
        checker.record(received)

    substitutes = [substitute for substitute, _ in run_gap(checker, 5, 21)]

    for step_index, substitute in enumerate(substitutes, start=1):
        assert substitute["q_c"][0] == pytest.approx(20 * step + step_index * step, abs=1e-9)


# -- a rejected late datagram must not leave the reference rolled back --------


def test_a_late_datagram_enforcement_refuses_leaves_the_extrapolation_standing():
    """The rewind is only final once the command is accepted.

    The rewind used to run *before* the accept gate, so a late datagram carrying
    a genuine step was rewound, then refused, and nothing was recorded in its
    place: the reference stayed rolled back by the whole gap and the next
    extrapolated cycle was dispatched to physics as a backward jump. Folded into
    :meth:`absorb_command`, which restores the extrapolated state on refusal.
    """
    checker = MotionLimitChecker(enforce=True)
    checker.start_motion(ControlMode.POSITION, robot_state_at())
    waypoints, last_id = stream_position_ramp(checker)
    substitutes = [substitute for substitute, _ in run_gap(checker, 5, last_id + 1)]
    after_gap = checker.applied_derivatives()
    reference = substitutes[-1]["q_c"][0]
    velocity = after_gap[0][0]
    assert velocity > 0.1

    stepped = [waypoints[-1][0] + 0.5] + list(waypoints[-1][1:])
    outcome = checker.absorb_command(
        command(message_id=last_id + 2, q_c=stepped), fresh=False, enforce=True
    )

    assert outcome.violation is not None
    assert outcome.accepted is False
    assert outcome.recorded is False
    assert outcome.rewound is False, "a refused datagram must not claim the rewind"
    assert checker.applied_derivatives() == after_gap

    # The reference is where the extrapolation left it, not rolled back: the
    # next missed cycle continues *forward* from it by one cycle of motion.
    # Rewound and not recorded, it went backwards by the whole gap instead --
    # dispatched to physics as a jump of half a millirad the wrong way.
    checker.note_published(last_id + 8)
    following, violation = checker.extrapolate(last_id + 7)
    assert violation is None
    assert following["q_c"][0] == pytest.approx(reference + velocity * DELTA_T, abs=1e-9)

    # ...and the run of losses is still open, so a *good* late datagram for it
    # is still rewindable.
    assert checker.rewind_extrapolation(command(message_id=last_id + 2)) is True


def test_an_extrapolated_command_is_not_dispatched_under_a_newer_motion(mock_physics_sim):
    """A ``Move`` accepted mid-gap must not have the old motion's fields applied.

    ``_dispatch_control_command`` branches on ``motion_generator_mode`` and
    ``controller_mode``, which the TCP thread has already rewritten by the time a
    ``Move`` returns -- so a substitute built for the *previous* motion's
    generator, a few microseconds earlier on the publish thread, would be routed
    into the new motion's branch and written to fields the old motion never
    commanded. The motion token is re-read inside the dispatch's own
    ``_hold_lock``, which is the lock the ``Move`` path takes too.
    """
    from franka_sim.franka_protocol import (
        LibfrankaControllerMode,
        LibfrankaMotionGeneratorMode,
    )
    from franka_sim.franka_sim_server import FrankaSimServer

    server = FrankaSimServer(physics_sim=mock_physics_sim, enable_gripper=False)
    server.robot_state.set_controller_mode(LibfrankaControllerMode.kJointImpedance)
    server.robot_state.set_motion_generator_mode(LibfrankaMotionGeneratorMode.kJointPosition)
    server._motion_generation = 7

    target = [0.11] * 7
    server._dispatch_control_command(command(q_c=target), motion_generation=7)
    assert server.robot_state.state["q_d"] == target

    # A Move lands: a new motion owns the generator fields now.
    server._motion_generation = 8
    server._dispatch_control_command(command(q_c=[0.99] * 7), motion_generation=7)
    assert server.robot_state.state["q_d"] == target, "a dead motion's guess was applied"

    # A received datagram passes no token and is unaffected -- the idle hold is
    # what covers that path, as it always was.
    server._dispatch_control_command(command(q_c=[0.42] * 7))
    assert server.robot_state.state["q_d"] == [0.42] * 7


# --- layer 2: the server over the wire, with a mocked simulator --------------


@pytest.fixture
def mock_arm_sim():
    """A mocked arm reporting a *valid* FR3 configuration.

    The shared ``mock_physics_sim`` reports ``q = 0``, which is not reachable on
    an FR3 at all -- joint 4 lives in [-3.0481, -0.1458] and joint 6 in
    [0.5409, 4.5205] -- so an all-zeros command would trip the position-limit
    check before anything this file is about.

    ``O_T_EE`` is ``MOCK_O_T_EE``, not identity: ``_motion_limit_seed_state``
    reads it straight off this dict (mirroring what ``_publish_hold_setpoint``
    already does for ``q``), and an identity mock would let a bug that fell
    back to ``self.robot_state.state``'s own identity default pass unnoticed --
    the wire tests that seed from this fixture would not exercise
    ``_publish_commanded_pose`` or the seed at all. See
    test_enforced_a_ten_metre_first_pose_aborts_over_the_wire.
    """
    from unittest.mock import Mock

    sim = Mock()
    sim.get_robot_state.return_value = {
        "q": np.array(HOME),
        "dq": np.zeros(7),
        "tau_J": np.zeros(7),
        "O_T_EE": np.array(MOCK_O_T_EE),
    }
    return sim


@pytest.fixture
def serve(mock_arm_sim):
    """Start a FrankaSimServer with a mocked simulator, enforcement as asked."""
    from franka_sim.franka_sim_server import FrankaSimServer

    started = []

    def _serve(enforce=False, sim=None, mobile_base=False):
        server = FrankaSimServer(
            physics_sim=sim if sim is not None else mock_arm_sim,
            enable_gripper=False,
            mobile_base=mobile_base,
            enforce_motion_limits=enforce,
        )
        thread = threading.Thread(target=server.run_server, daemon=True)
        thread.start()
        assert wait_for_server(COMMAND_PORT), "the FCI server never came up"
        started.append((server, thread))
        return server

    yield _serve

    for server, thread in started:
        server.stop()
        thread.join(timeout=3.0)
    time.sleep(0.4)


@pytest.fixture
def client():
    """A wire client, closed however the test ends."""
    clients = []

    def _client(**kwargs):
        made = WireClient(**kwargs)
        clients.append(made)
        return made

    yield _client

    for made in clients:
        made.close()


def _wire(client_factory):
    """A connected client that has *not* sent a Move yet.

    For the tests that need to see how a ``Move`` is answered, rather than to
    get past it; :func:`start_motion` asserts the answer is ``kMotionStarted``.
    """
    wire = client_factory()
    wire.connect()
    return wire


def start_motion(client_factory, controller_mode, motion_mode, command_id=2):
    """Connect and get past the Move handshake, ready to stream commands."""
    wire = client_factory()
    wire.connect()
    assert wire.move(controller_mode, motion_mode, command_id=command_id) == (
        MoveStatus.kMotionStarted
    )
    # No second Move response follows kMotionStarted: Move gets exactly one
    # immediate reply, and the terminal one only arrives once the motion
    # actually ends -- it does not, here.
    return wire


#: (controller mode, motion-generator mode, streamed field, the offending value,
#: the error index it must latch). Each offending value is inside the joint's
#: *range* -- what is wrong with it is how far it is from where the robot was.
VIOLATION_CASES = [
    pytest.param(
        ControllerMode.kJointImpedance,
        MotionGeneratorMode.kJointPosition,
        "q_c",
        [HOME[0] + 0.5] + HOME[1:],
        JOINT_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX,
        id="position-jump",
    ),
    pytest.param(
        ControllerMode.kJointImpedance,
        MotionGeneratorMode.kJointVelocity,
        "dq_c",
        [1.5] + [0.0] * 6,
        # 15, not 14: a step in ``dq_c`` is an *acceleration* discontinuity on
        # the velocity interface. See the interface-relative naming rule.
        JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
        id="velocity-step",
    ),
    pytest.param(
        ControllerMode.kExternalController,
        MotionGeneratorMode.kNone,
        "tau_j_d",
        [5.0] + [0.0] * 6,
        CONTROLLER_TORQUE_DISCONTINUITY_INDEX,
        id="torque-step",
    ),
]


@pytest.mark.parametrize("controller,motion,field,offending,error_index", VIOLATION_CASES)
def test_a_violating_command_is_reported_but_applied_by_default(
    serve, client, controller, motion, field, offending, error_index
):
    """Off by default: the sim stays the permissive channel it has always been."""
    server = serve()
    wire = start_motion(client, controller, motion)

    state = wire.stream(20, **{field: offending})

    assert not any(state[ERRORS_SLICE])
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove
    assert server.motion_limits.violated is False


@pytest.mark.parametrize("controller,motion,field,offending,error_index", VIOLATION_CASES)
def test_enforced_a_violating_command_aborts_with_its_own_error_bit(
    serve, client, controller, motion, field, offending, error_index
):
    """Enforced: the right bit, kReflex, kReflexAborted, and the idle hold."""
    server = serve(enforce=True)
    wire = start_motion(client, controller, motion)

    for _ in range(5):
        wire.read_state()
        wire.answer(**{field: offending})
    status = wire.read_move_response()
    state = wire.read_state()

    assert status == MoveStatus.kReflexAborted
    assert state[ERRORS_SLICE][error_index] == 1
    assert state[REFLEX_REASON_SLICE][error_index] == 1
    assert sum(state[ERRORS_SLICE]) == 1
    assert state[ROBOT_MODE_INDEX] == RobotMode.kReflex
    assert server.motion_limits.violated is True


def test_enforced_a_commanded_envelope_abort_latches_both_bits_on_the_wire(serve, client):
    """13 pairs with 3 in the actual packed ``RobotState``, not just at the
    checker level.

    ``test_a_commanded_envelope_violation_during_a_motion_latches_both_bits``
    (above) pins :attr:`Violation.extra_error_index` on the checker's own
    object; this streams a *smooth* velocity ramp -- constant acceleration, so
    it never trips a discontinuity (14/15) -- through the real server and
    reads the two bits back off the wire, the way a client actually would.

    The ramp has to enter gently: a velocity motion's opening command rebases
    the checker's differencing history to a flat standstill -- the same
    zero-derivative baseline the position and Cartesian-pose generators start
    from -- so it is eased in from ``dq_d`` (0) over the first 200 cycles (see
    :func:`_ease`), exactly as the position-generator ramps in this module do,
    rather than stepping straight to a steady acceleration that would trip a
    jerk discontinuity against that flat baseline.
    """
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kJointImpedance, MotionGeneratorMode.kJointVelocity)

    cap = upper_joint_velocity_limits(HOME)[0]
    accel = 5.0  # rad/s^2: comfortably under MAX_JOINT_ACCELERATION once eased in

    commanded = 0.0
    status = None
    state = None
    for cycle in range(1, 1000):
        wire.read_state()
        commanded += accel * _ease(cycle / 200.0) * DELTA_T
        wire.answer(dq_c=[commanded] + [0.0] * 6)
        if commanded > cap:
            status = wire.read_move_response()
            state = wire.read_state()
            break
    assert state is not None, "the ramp never crossed the envelope"

    assert status == MoveStatus.kReflexAborted
    assert state[ERRORS_SLICE][JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX] == 1
    assert state[ERRORS_SLICE][JOINT_VELOCITY_VIOLATION_INDEX] == 1
    assert state[REFLEX_REASON_SLICE][JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX] == 1
    assert state[REFLEX_REASON_SLICE][JOINT_VELOCITY_VIOLATION_INDEX] == 1
    assert sum(state[ERRORS_SLICE]) == 2, "exactly these two bits, nothing else"
    assert state[ROBOT_MODE_INDEX] == RobotMode.kReflex
    assert server.motion_limits.violated is True


def test_enforced_the_violating_command_never_reaches_the_simulator(serve, client):
    """The real robot rejects it; so must the sim, or the physics saw a jump."""
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)

    for _ in range(5):
        wire.read_state()
        wire.answer(q_c=[HOME[0] + 0.5] + HOME[1:])
    wire.read_move_response()

    commanded = [call.args[0] for call in server.physics_sim.update_joint_positions.call_args_list]
    assert commanded, "the simulator was never commanded at all"
    for target in commanded:
        # Everything the sim ever saw is the hold's own HOME target; the 0.5 rad
        # jump on joint 1 never got past the checker.
        assert np.array(target)[0] == pytest.approx(HOME[0]), f"the jump reached the sim: {target}"


def test_a_violation_from_a_finished_motion_never_latches_onto_the_next(serve, client):
    """The check-then-act race between the UDP and publish threads, pinned.

    Both threads run "is a violation already latched? then latch and abort":
    the UDP thread for each received command, the publish thread for the safety
    controller. Each reads the checker's motion token *before* its check. If the
    motion ends and the next one starts in between, the stale latch lands on the
    new motion -- whose ``start_motion`` has just cleared it -- while
    ``_abort_with_error`` rightly refuses to abort a motion the token no longer
    names. The result is a motion running with a violation latched and no reflex
    behind it, and since the server refuses a ``Move`` while one is latched,
    every later ``Move`` on that connection is answered
    ``kCommandNotPossibleRejected``: permanently un-abortable.

    Winning that interleaving with real threads is not deterministic, so the
    stale token is handed to ``_latch_and_abort`` directly -- which is exactly
    what the losing thread would be holding. The invariant is that the latch and
    the abort succeed or fail *together*.
    """
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)
    wire.stream(3, q_c=list(HOME))
    stale = server.motion_limits.motion_id
    assert stale, "no motion token to go stale"

    # The next motion preempts this one, exactly as the TCP thread would.
    assert (
        wire.move(
            ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition, command_id=21
        )
        == MoveStatus.kPreempted
    )
    assert wire.read_move_response() == MoveStatus.kMotionStarted
    fresh = server.motion_limits.motion_id
    assert fresh != stale

    violation = Violation(
        JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX, "dq", "joint 6", 9.0, 4.18
    )
    server._latch_and_abort(violation, stale)

    assert server.motion_limits.violated is False, "the stale violation latched onto the new motion"
    state = wire.read_state()
    assert not any(state[ERRORS_SLICE])
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove

    # ...and the positive control: the same violation carrying the *running*
    # token still latches and still aborts, so the guard is not just refusing
    # everything.
    server._latch_and_abort(violation, fresh)

    assert server.motion_limits.violated is True
    assert wire.read_move_response() == MoveStatus.kReflexAborted
    aborted = wire.read_state()
    assert aborted[ERRORS_SLICE][JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX] == 1
    assert aborted[ROBOT_MODE_INDEX] == RobotMode.kReflex


def test_enforced_a_smooth_stream_is_never_touched(serve, client):
    """The other half of the contract: a conforming client sees no difference."""
    import math

    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)

    waypoints = []
    for step in range(300):
        delta = math.pi / 8.0 * (1 - math.cos(math.pi / 2.5 * step * DELTA_T))
        waypoints.append([HOME[0] + delta] + HOME[1:])
    wire.ramp(waypoints)
    state = wire.read_state()

    assert not any(state[ERRORS_SLICE])
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove
    assert server.motion_limits.violated is False
    assert state[Q_D_SLICE][0] == pytest.approx(waypoints[-1][0], abs=1e-3)


def test_a_cartesian_pose_motion_latches_no_joint_error(serve, client):
    """The generator gating, over the wire, with enforcement on.

    ``kCartesianPosition`` datagrams carry a zero-filled ``q_c`` because the
    client is not commanding joints at all. Judging that as a joint position
    command aborted real clients with
    ``joint_motion_generator_position_limits_violation``.

    A joint-position motion runs first on purpose: that is what leaves POSITION
    behind in ``server.control_mode``, and handing *that* to the checker instead
    of the mode the new Move actually asked for is the whole bug. On a
    fresh server the stale mode happens to be NONE and nothing goes wrong.

    The pose streamed here is ``mock_pose()``, which is exactly where the mocked
    arm's ``O_T_EE`` is (``MOCK_O_T_EE``), so the pose generator's own checks all
    pass and the motion runs clean.
    """
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)
    wire.stream(5, q_c=list(HOME))
    assert server.control_mode is ControlMode.POSITION

    assert (
        wire.move(
            ControllerMode.kJointImpedance,
            MotionGeneratorMode.kCartesianPosition,
            command_id=15,
        )
        == MoveStatus.kPreempted
    )
    assert wire.read_move_response() == MoveStatus.kMotionStarted

    # q_c = 0, exactly as libfranka packs a Cartesian command.
    state = wire.stream(30, o_t_ee_c=mock_pose())

    assert not any(state[ERRORS_SLICE]), "a Cartesian pose motion latched a joint error"
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove
    assert server.motion_limits.violated is False
    # The pose interface drives the arm now (through differential IK), so the
    # backend is in the Cartesian mode and is being handed the commanded pose --
    # what must *not* happen is the joint branches claiming the datagram and
    # servoing to its zero-filled q_c.
    assert server.control_mode is ControlMode.CARTESIAN_POSE
    assert server.physics_sim.update_cartesian_pose.call_count > 0
    # ...and the joint branch never saw a Cartesian datagram: its zero-filled
    # q_c would show up here as an all-zeros joint target, which is both the
    # original bug and a pose no FR3 can hold.
    assert not any(
        np.allclose(np.asarray(call.args[0], dtype=float), 0.0)
        for call in server.physics_sim.update_joint_positions.call_args_list
    )
    assert server.physics_sim.update_base_twist.call_count == 0


def test_enforced_a_cartesian_pose_step_aborts_over_the_wire(serve, client):
    """A Cartesian velocity discontinuity, over the wire.

    The provocation holds ``O_T_EE_d`` and adds 1.0 m to z at t > 0.5 s.
    libfranka's 100 Hz
    command filter turns that into a 0.3859 m first cycle, which is what
    actually reaches the FCI -- and hardware answers
    ``cartesian_motion_generator_velocity_discontinuity``. Without this abort a
    client provoking it never
    terminates against the sim, which is the whole reason the pose interface is
    checked at all.
    """
    server = serve(enforce=True)
    wire = start_motion(
        client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianPosition
    )

    wire.stream(5, o_t_ee_c=mock_pose())
    for _ in range(5):
        wire.read_state()
        wire.answer(o_t_ee_c=mock_pose(dz=LOWPASS_GAIN * 1.0))

    assert wire.read_move_response() == MoveStatus.kReflexAborted
    latched = server.robot_state.state["errors"]
    assert latched[CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX] is True
    # The discontinuity, not the envelope: 386 m/s breaks both.
    assert latched[CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX] is False


def test_enforced_a_ten_metre_first_pose_aborts_over_the_wire(serve, client):
    """A first ``O_T_EE_c`` +10 m away aborts, over the wire.

    This is the one case that exercises the *seed* plumbing rather than the
    per-command plumbing: the checker has to have been handed the robot's own
    measured ``O_T_EE`` when the ``Move`` was accepted, or there is nothing to
    find the 10 m offset against. The mocked arm reports ``MOCK_O_T_EE``
    (z = 0.48 m), so the client opening at z = 10 m is still about 9.5 m out --
    comfortably over the tolerance either way.
    """
    server = serve(enforce=True)
    wire = start_motion(
        client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianPosition
    )

    for _ in range(5):
        wire.read_state()
        wire.answer(o_t_ee_c=pose(z=10.0))

    assert wire.read_move_response() == MoveStatus.kReflexAborted
    latched = server.robot_state.state["errors"]
    assert latched[CARTESIAN_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX] is True
    # A start-pose error, not the discontinuity a 10 m jump would also imply.
    assert latched[CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX] is False


def test_a_garbage_pose_matrix_is_refused_with_enforcement_off(serve, client, caplog):
    """``checkMatrix``'s test, server-side, and refused like a NaN.

    The provocation zero-fills ``O_T_EE`` mid-motion. On hardware libfranka
    refuses to *send* it
    -- ``checkMatrix`` throws ``std::invalid_argument`` client-side, which is why
    a stock client sees ``std::invalid_argument`` rather than a
    ``ControlException``. A simulator cannot rely
    on the client being libfranka, so the test is done here too, and like every
    other structurally-invalid command it is refused whether or not enforcement
    is on: an all-zeros matrix has no derivatives worth recording.
    """
    server = serve()  # enforcement off on purpose
    wire = start_motion(
        client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianPosition
    )

    wire.stream(5, o_t_ee_c=mock_pose())
    with caplog.at_level(logging.WARNING, logger="franka_sim.motion_limits"):
        state = wire.stream(10, o_t_ee_c=[0.0] * 16)

    # Reported by name, once...
    assert "cartesian_position_motion_generator_invalid_frame_flag" in caplog.text
    # ...but not aborted: the abort is what the switch is for.
    assert not any(state[ERRORS_SLICE])
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove
    assert server.motion_limits.violated is False


def test_enforced_an_arm_role_cartesian_twist_step_aborts_over_the_wire(serve, client):
    """The twist checks reach an *arm* role, not only the mobile base.

    Stepping all six twist components to 10.0 at t > 0.5 s makes hardware answer
    ``cartesian_motion_generator_acceleration_discontinuity``. The base role
    already did this; before the
    gating was extended an arm-role ``kCartesianVelocity`` Move was checked on
    nothing at all.
    """
    server = serve(enforce=True)
    wire = start_motion(
        client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity
    )

    for _ in range(5):
        wire.read_state()
        wire.answer(o_dp_ee_c=[10.0] * 6)

    assert wire.read_move_response() == MoveStatus.kReflexAborted
    latched = server.robot_state.state["errors"]
    assert latched[CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX] is True
    assert latched[CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX] is False


def test_enforced_a_bad_start_elbow_aborts_over_the_wire(serve, client):
    """A first ``elbow_c[0]`` 0.5 rad away aborts, over the wire.

    ``elbow[0] += 0.5`` from cycle 0 with the pose left alone; hardware answers
    ``cartesian_motion_generator_start_elbow_invalid``.
    """
    server = serve(enforce=True)
    wire = start_motion(
        client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianPosition
    )

    for _ in range(5):
        wire.read_state()
        wire.answer(o_t_ee_c=mock_pose(), elbow_c=[HOME_ELBOW[0] + 0.5, HOME_ELBOW[1]])

    assert wire.read_move_response() == MoveStatus.kReflexAborted
    assert server.robot_state.state["errors"][
        CARTESIAN_MOTION_GENERATOR_START_ELBOW_INVALID_INDEX
    ] is True


def test_the_published_elbow_is_the_one_a_client_may_command_back(serve, client):
    """``elbow``/``elbow_d`` report ``(q[2], sign(q[3]))``, not a zero stub.

    A ``kCartesianPosition`` client builds ``CartesianPose{O_T_EE_d, elbow_d}``,
    and libfranka's ``checkElbow`` throws client-side unless the branch flag is
    exactly +-1 -- so a permanently zero ``elbow_d`` made the elbow interface
    unreachable from a real client. Feeding the published elbow straight back
    must also pass the sim's own start-elbow check, which is the round trip this
    asserts.
    """
    server = serve(enforce=True)
    wire = start_motion(
        client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianPosition
    )

    state = wire.read_state()
    elbow = list(state[ELBOW_D_SLICE])
    assert elbow == pytest.approx(HOME_ELBOW)

    wire.answer(o_t_ee_c=mock_pose(), elbow_c=elbow)
    state = wire.stream(10, o_t_ee_c=mock_pose(), elbow_c=elbow)

    assert not any(state[ERRORS_SLICE])
    assert server.motion_limits.violated is False


def test_enforced_the_safety_controller_aborts_on_measured_velocity(
    serve, client, mock_arm_sim
):
    """joint_velocity_violation over the wire, from a torque session.

    No commanded velocity exists anywhere here; the arm is simply reported to be
    moving faster than its position-based envelope allows, which is what the
    a 3 Nm torque ramp provokes on hardware.
    """
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kExternalController, MotionGeneratorMode.kNone)

    over = upper_joint_velocity_limits(HOME)[5] + 1.0
    mock_arm_sim.get_robot_state.return_value = {
        "q": np.array(HOME),
        "dq": np.array(spread(over, index=5)),
        "tau_J": np.zeros(7),
    }

    for _ in range(10):
        wire.read_state()
        wire.answer(tau_j_d=[0.0] * 7)  # every command sent is perfectly legal
    assert wire.read_move_response() == MoveStatus.kReflexAborted

    state = wire.read_state()
    assert state[ERRORS_SLICE][JOINT_VELOCITY_VIOLATION_INDEX] == 1
    assert state[REFLEX_REASON_SLICE][JOINT_VELOCITY_VIOLATION_INDEX] == 1
    assert sum(state[ERRORS_SLICE]) == 1
    assert state[ROBOT_MODE_INDEX] == RobotMode.kReflex
    assert server.motion_limits.violated is True


def test_the_safety_controller_reports_without_enforcement_but_does_not_abort(
    serve, client, mock_arm_sim
):
    server = serve()  # enforcement off
    wire = start_motion(client, ControllerMode.kExternalController, MotionGeneratorMode.kNone)

    over = upper_joint_velocity_limits(HOME)[5] + 1.0
    mock_arm_sim.get_robot_state.return_value = {
        "q": np.array(HOME),
        "dq": np.array(spread(over, index=5)),
        "tau_J": np.zeros(7),
    }

    state = wire.stream(20, tau_j_d=[0.0] * 7)

    assert not any(state[ERRORS_SLICE])
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove
    assert server.motion_limits.violated is False


# -- the safety controller's Cartesian half ----------------------------------
#
# ``cartesian_velocity_violation``: measured end-effector speed against the
# published translational limit, in every control mode. See
# MotionLimitChecker.check_measured_cartesian_velocity and the citation on
# MEASURED_CARTESIAN_VELOCITY_LIMIT.


def test_the_ee_speed_limit_is_the_published_translational_one():
    """The limit is franka::kMaxTranslationalVelocity, not a number of our own."""
    assert MEASURED_CARTESIAN_VELOCITY_LIMIT == MAX_TRANSLATIONAL_VELOCITY
    assert MEASURED_CARTESIAN_VELOCITY_LIMIT == pytest.approx(3.0 - 1e-3)


@pytest.mark.parametrize(
    "speed, expected",
    [
        (0.5 * MEASURED_CARTESIAN_VELOCITY_LIMIT, False),
        (MEASURED_CARTESIAN_VELOCITY_LIMIT - 1e-6, False),
        (MEASURED_CARTESIAN_VELOCITY_LIMIT, False),  # at the limit is still legal
        (MEASURED_CARTESIAN_VELOCITY_LIMIT + 1e-6, True),
        (2.0 * MEASURED_CARTESIAN_VELOCITY_LIMIT, True),
    ],
)
def test_the_ee_speed_check_fires_exactly_above_the_limit(speed, expected):
    """No margin on this one: 3 m/s of EE travel is nowhere near a normal motion."""
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, {"q": list(HOME), "dq": [0.0] * 7})
    # Split across all three axes so the check is provably on the norm, not on
    # any single component.
    axis = np.array([2.0, -1.0, 2.0]) / 3.0
    violation = checker.check_measured_cartesian_velocity(speed * axis)
    assert (violation is not None) is expected
    if expected:
        assert violation.error_index == CARTESIAN_VELOCITY_VIOLATION_INDEX
        assert violation.error_name == "cartesian_velocity_violation"
        assert violation.value == pytest.approx(speed)
        assert violation.unit == "m/s"


def test_the_ee_speed_check_is_silent_without_a_reading():
    """A backend that publishes no EE velocity switches the check off, not on."""
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, {"q": list(HOME), "dq": [0.0] * 7})
    assert checker.check_measured_cartesian_velocity(None) is None
    assert checker.check_measured_cartesian_velocity([float("nan"), 0.0, 0.0]) is None


def test_the_ee_speed_check_is_armed_in_torque_mode_too():
    """Like its joint twin, it judges the arm -- so no motion generator is needed."""
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.TORQUE, {"q": list(HOME), "dq": [0.0] * 7})
    over = MEASURED_CARTESIAN_VELOCITY_LIMIT + 1.0
    assert checker.check_measured_cartesian_velocity([over, 0.0, 0.0]) is not None
    # ...and disarmed once the motion is over, exactly as the joint one is.
    checker.end_motion()
    assert checker.check_measured_cartesian_velocity([over, 0.0, 0.0]) is None


def test_enforced_the_safety_controller_aborts_on_measured_ee_speed(
    serve, client, mock_arm_sim
):
    """cartesian_velocity_violation over the wire, and nothing else with it.

    Hardware records the whole point: with
    an EE 0.5 m out along the flange, the ramp crosses the *Cartesian* limit
    first and hardware reports that error **alone** -- no
    ``joint_velocity_violation`` beside it -- so the two halves of the safety
    controller are ordered, not merely both present.
    """
    serve(enforce=True)
    wire = start_motion(client, ControllerMode.kExternalController, MotionGeneratorMode.kNone)

    mock_arm_sim.get_robot_state.return_value = {
        "q": np.array(HOME),
        "dq": np.zeros(7),  # the joints are inside their envelope
        "tau_J": np.zeros(7),
        "O_dP_EE": np.array([MEASURED_CARTESIAN_VELOCITY_LIMIT + 0.5, 0.0, 0.0]),
    }

    for _ in range(10):
        wire.read_state()
        wire.answer(tau_j_d=[0.0] * 7)
    assert wire.read_move_response() == MoveStatus.kReflexAborted

    state = wire.read_state()
    assert state[ERRORS_SLICE][CARTESIAN_VELOCITY_VIOLATION_INDEX] == 1
    assert state[REFLEX_REASON_SLICE][CARTESIAN_VELOCITY_VIOLATION_INDEX] == 1
    assert sum(state[ERRORS_SLICE]) == 1
    assert state[ROBOT_MODE_INDEX] == RobotMode.kReflex


def test_enforced_the_ee_speed_check_outranks_the_joint_one(serve, client, mock_arm_sim):
    """Both broken in the same cycle: the Cartesian error is the one that latches.

    The precedence hardware records. Without it a motion that leaves both
    envelopes at once would report ``joint_velocity_violation``, which is what
    the sim did before the Cartesian check existed -- and what hardware says is
    the wrong answer once the EE is levered 0.5 m out.
    """
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kExternalController, MotionGeneratorMode.kNone)

    over_joint = upper_joint_velocity_limits(HOME)[5] + 1.0
    mock_arm_sim.get_robot_state.return_value = {
        "q": np.array(HOME),
        "dq": np.array(spread(over_joint, index=5)),
        "tau_J": np.zeros(7),
        "O_dP_EE": np.array([0.0, MEASURED_CARTESIAN_VELOCITY_LIMIT + 0.5, 0.0]),
    }

    for _ in range(10):
        wire.read_state()
        wire.answer(tau_j_d=[0.0] * 7)
    assert wire.read_move_response() == MoveStatus.kReflexAborted

    state = wire.read_state()
    assert state[ERRORS_SLICE][CARTESIAN_VELOCITY_VIOLATION_INDEX] == 1
    assert state[ERRORS_SLICE][JOINT_VELOCITY_VIOLATION_INDEX] == 0
    assert sum(state[ERRORS_SLICE]) == 1
    assert server.motion_limits.violated is True


def test_a_joint_only_excursion_still_reports_the_joint_error(serve, client, mock_arm_sim):
    """The other side of the precedence: a slow EE does not mask a fast joint.

    The two hardware scenarios that must keep answering
    ``joint_velocity_violation`` -- a 3 Nm torque ramp and a 5 rad/s^2 ``dq_c``
    ramp -- both spin joint 6
    while the end effector barely travels; this is that shape in miniature.
    """
    serve(enforce=True)
    wire = start_motion(client, ControllerMode.kExternalController, MotionGeneratorMode.kNone)

    over_joint = upper_joint_velocity_limits(HOME)[5] + 1.0
    mock_arm_sim.get_robot_state.return_value = {
        "q": np.array(HOME),
        "dq": np.array(spread(over_joint, index=5)),
        "tau_J": np.zeros(7),
        "O_dP_EE": np.array([0.3, 0.0, 0.0]),  # a wrist spin moves the EE hardly at all
    }

    for _ in range(10):
        wire.read_state()
        wire.answer(tau_j_d=[0.0] * 7)
    assert wire.read_move_response() == MoveStatus.kReflexAborted

    state = wire.read_state()
    assert state[ERRORS_SLICE][JOINT_VELOCITY_VIOLATION_INDEX] == 1
    assert state[ERRORS_SLICE][CARTESIAN_VELOCITY_VIOLATION_INDEX] == 0


# -- singular start poses -----------------------------------------------------


def test_the_singularity_threshold_separates_singular_from_start_poses():
    """The threshold sits between a singular configuration and ordinary start poses.

    The two numbers it has to keep apart, measured on the sim's own FR3 model
    and quoted in SINGULAR_POSE_MIN_SINGULAR_VALUE: a known singular FR3
    configuration (sigma_min ~ 0.011) and the tightest pose a Cartesian motion
    is expected to open from (~0.139).
    """
    assert 0.011 < SINGULAR_POSE_MIN_SINGULAR_VALUE < 0.139
    # ...and with a comfortable factor on both sides, not by a hair.
    assert SINGULAR_POSE_MIN_SINGULAR_VALUE > 4 * 0.011
    assert SINGULAR_POSE_MIN_SINGULAR_VALUE < 0.139 / 2


def test_is_singular_configuration_at_the_threshold_boundary():
    """At or below the threshold is singular; above it is not."""
    # A diagonal 6x6 Jacobian whose smallest singular value is exactly settable.
    def jacobian(sigma_min):
        return np.hstack((np.diag([1.0, 1.0, 1.0, 1.0, 1.0, sigma_min]), np.zeros((6, 1))))

    threshold = SINGULAR_POSE_MIN_SINGULAR_VALUE
    assert smallest_singular_value(jacobian(threshold)) == pytest.approx(threshold)
    assert is_singular_configuration(jacobian(threshold * 0.5)) is True
    assert is_singular_configuration(jacobian(threshold)) is True
    assert is_singular_configuration(jacobian(threshold * 1.001)) is False
    assert is_singular_configuration(jacobian(threshold * 10)) is False


def test_a_missing_jacobian_is_not_a_singularity():
    """A backend that cannot say is no grounds for refusing the client's Move."""
    assert smallest_singular_value(None) is None
    assert is_singular_configuration(None) is False
    assert is_singular_configuration(np.full((6, 7), np.nan)) is False
    assert is_singular_configuration(np.zeros((0, 0))) is False


def _singular_jacobian():
    """A 6x7 Jacobian whose smallest singular value is well under the threshold."""
    jacobian = np.zeros((6, 7))
    for row in range(5):
        jacobian[row, row] = 1.0
    jacobian[5, 5] = SINGULAR_POSE_MIN_SINGULAR_VALUE / 10.0
    return jacobian


def test_a_cartesian_move_at_a_singular_pose_is_rejected(serve, client, mock_arm_sim):
    """Move::Status::kStartAtSingularPoseRejected, the terminal response.

    libfranka forces the shape: ``executeCommand<Move>`` handles the *first*
    response outside any try/catch, so a rejection delivered there escapes as a
    bare ``CommandException``; only the mode-wait loop that follows converts one
    into the ``ControlException`` a client actually catches. So the
    acknowledgement comes first and the rejection second.
    """
    mock_arm_sim.get_robot_state.return_value = {
        "q": np.array(HOME),
        "dq": np.zeros(7),
        "tau_J": np.zeros(7),
        "O_T_EE": np.array(MOCK_O_T_EE),
        "O_J_EE": _singular_jacobian(),
    }
    server = serve(enforce=True)
    wire = _wire(client)

    assert wire.move(
        ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianPosition
    ) == MoveStatus.kMotionStarted
    # The generator modes never leave idle, which is what keeps the client in
    # the loop that converts the rejection.
    state = wire.read_state()
    assert state[ROBOT_MODE_INDEX] != RobotMode.kMove
    assert wire.read_move_response() == MoveStatus.kStartAtSingularPoseRejected
    # A rejection is not a reflex: nothing is latched, so the next Move needs no
    # error recovery.
    assert not any(state[ERRORS_SLICE])
    assert server.motion_limits.violated is False


def test_a_joint_move_at_a_singular_pose_is_accepted(serve, client, mock_arm_sim):
    """Only the Cartesian generators are refused.

    A singular configuration is *reached* in the first place by a joint
    point-to-point motion, so a joint interface has to start there
    -- and driving out of a singularity is the only way to leave one.
    """
    mock_arm_sim.get_robot_state.return_value = {
        "q": np.array(HOME),
        "dq": np.zeros(7),
        "tau_J": np.zeros(7),
        "O_T_EE": np.array(MOCK_O_T_EE),
        "O_J_EE": _singular_jacobian(),
    }
    serve(enforce=True)
    wire = start_motion(client, ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)
    state = wire.stream(5, q_c=list(HOME))
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove
    assert not any(state[ERRORS_SLICE])


def test_a_cartesian_move_at_a_well_conditioned_pose_is_accepted(
    serve, client, mock_arm_sim
):
    """The other side of the boundary: an ordinary configuration starts normally."""
    jacobian = _singular_jacobian()
    jacobian[5, 5] = SINGULAR_POSE_MIN_SINGULAR_VALUE * 10.0
    mock_arm_sim.get_robot_state.return_value = {
        "q": np.array(HOME),
        "dq": np.zeros(7),
        "tau_J": np.zeros(7),
        "O_T_EE": np.array(MOCK_O_T_EE),
        "O_J_EE": jacobian,
    }
    serve(enforce=True)
    wire = _wire(client)
    assert wire.move(
        ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianPosition
    ) == MoveStatus.kMotionStarted
    state = wire.stream(5, o_t_ee_c=mock_pose())
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove
    assert not any(state[ERRORS_SLICE])


def test_error_recovery_clears_the_violation_and_a_new_move_runs(serve, client):
    """Automatic error recovery is what the abort leaves the client to do."""
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kExternalController, MotionGeneratorMode.kNone)

    for _ in range(5):
        wire.read_state()
        wire.answer(tau_j_d=[5.0] + [0.0] * 6)
    assert wire.read_move_response() == MoveStatus.kReflexAborted

    assert wire.automatic_error_recovery() == 0
    cleared = wire.read_state()
    assert not any(cleared[ERRORS_SLICE])
    assert cleared[ROBOT_MODE_INDEX] == RobotMode.kIdle
    # last_motion_errors is the record of what aborted the previous motion.
    assert cleared[REFLEX_REASON_SLICE][CONTROLLER_TORQUE_DISCONTINUITY_INDEX] == 1
    assert server.motion_limits.violated is False

    assert (
        wire.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone, command_id=11)
        == MoveStatus.kMotionStarted
    )
    # A ramp inside kMaxTorqueRate this time: 0.5 Nm per cycle.
    state = wire.ramp([[0.5 * (step + 1)] + [0.0] * 6 for step in range(8)], field="tau_j_d")
    assert not any(state[ERRORS_SLICE])
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove


def test_stop_move_after_an_abort_leaves_kreflex_latched(serve, client):
    """``StopMove`` must not clobber a latched ``kReflex`` with ``kIdle``.

    Since ``StopMove`` stopped ending the session (see
    ``docs/robot-state.md``), the publish loop keeps streaming ``RobotState``
    after it -- so if ``handle_stop_move_command`` unconditionally wrote
    ``robot_mode`` back to ``kIdle``, a client recovering from an enforced
    abort would see the arm report ``kIdle`` for the whole window between
    ``StopMove`` and ``AutomaticErrorRecovery``, even while ``errors`` still
    says why it aborted. libfranka's own recovery sequence is
    ``cancelMotion()`` -> ``StopMove`` -> ``AutomaticErrorRecovery`` ->
    ``Move`` (see ``ActiveControl``), so this window is not hypothetical --
    it is exactly the one every recovering client passes through. Mirrors the
    guard ``_finish_motion`` already had for the same reason.
    """
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kExternalController, MotionGeneratorMode.kNone)

    for _ in range(5):
        wire.read_state()
        wire.answer(tau_j_d=[5.0] + [0.0] * 6)
    assert wire.read_move_response() == MoveStatus.kReflexAborted

    aborted = wire.read_state()
    assert aborted[ROBOT_MODE_INDEX] == RobotMode.kReflex
    assert aborted[ERRORS_SLICE][CONTROLLER_TORQUE_DISCONTINUITY_INDEX] == 1

    assert wire.stop_move() == 0  # kSuccess

    after_stop = wire.read_state()
    assert after_stop[ROBOT_MODE_INDEX] == RobotMode.kReflex, (
        "StopMove clobbered a latched kReflex with kIdle"
    )
    assert after_stop[ERRORS_SLICE][CONTROLLER_TORQUE_DISCONTINUITY_INDEX] == 1, (
        "StopMove cleared a latched error it should have left alone"
    )
    assert server.motion_limits.violated is True

    assert wire.automatic_error_recovery() == 0  # kSuccess
    cleared = wire.read_state()
    assert cleared[ROBOT_MODE_INDEX] == RobotMode.kIdle
    assert not any(cleared[ERRORS_SLICE])


def test_the_base_twist_is_held_to_the_cartesian_limits(serve, client, mock_base_sim):
    """A body twist is a motion-generator signal, judged by the wire's own constants."""
    server = serve(enforce=True, sim=mock_base_sim, mobile_base=True)
    wire = client()
    wire.connect()
    wire.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)

    # A jerk-limited ramp in is fine...
    ramp = [[0.001 * (step + 1), 0.0, 0.0, 0.0, 0.0, 0.0] for step in range(30)]
    state = wire.ramp(ramp, field="o_dp_ee_c")
    assert not any(state[ERRORS_SLICE])

    # ...a 5 m/s step is not. It breaks the envelope, the acceleration limit and
    # the jerk limit at once, and by the interface-relative rule a *step* on a
    # commanded-velocity interface is named for the acceleration: index 20, not
    # the envelope's 18.
    for _ in range(5):
        wire.read_state()
        wire.answer(o_dp_ee_c=[5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert wire.read_move_response() == MoveStatus.kReflexAborted

    latched = server.robot_state.state["errors"]
    assert latched[CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX] is True
    assert latched[CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX] is False
    assert np.array(mock_base_sim.update_base_twist.call_args.args[0]) == pytest.approx(np.zeros(6))


def test_a_non_finite_command_never_reaches_the_simulator(serve, client):
    """Refused with enforcement *off* too: applying NaN is not permissiveness.

    Every limit comparison against a NaN is false, so it used to sail through the
    checks, into the difference history, into the physics backend and onto the
    wire -- where it stays NaN for the rest of the connection.
    """
    server = serve()  # enforcement off on purpose
    wire = start_motion(client, ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)

    good = list(HOME)
    wire.stream(5, q_c=good)
    poisoned = [float("nan")] + HOME[1:]
    state = wire.stream(20, q_c=poisoned)

    for call in server.physics_sim.update_joint_positions.call_args_list:
        assert np.all(np.isfinite(np.asarray(call.args[0]))), "NaN reached the simulator"
    assert np.all(np.isfinite(np.array(state[Q_D_SLICE]))), "NaN reached the wire"
    # Still permissive about everything else: nothing was latched or aborted.
    assert not any(state[ERRORS_SLICE])
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove


def test_enforced_a_non_finite_command_aborts_like_any_violation(serve, client):
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kExternalController, MotionGeneratorMode.kNone)

    for _ in range(5):
        wire.read_state()
        wire.answer(tau_j_d=[float("inf")] + [0.0] * 6)

    assert wire.read_move_response() == MoveStatus.kReflexAborted
    assert server.robot_state.state["errors"][TAU_J_RANGE_VIOLATION_INDEX] is True


def test_a_second_move_without_recovery_is_refused_not_swallowed(serve, client):
    """The latch survives the motion, so the next Move has to be refused.

    Accepted, it produced a state the robot never produces -- kMove with the
    error bits still set -- and a motion whose every command was silently
    dropped by the still-latched checker, forever.
    """
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kExternalController, MotionGeneratorMode.kNone)
    for _ in range(5):
        wire.read_state()
        wire.answer(tau_j_d=[5.0] + [0.0] * 6)
    assert wire.read_move_response() == MoveStatus.kReflexAborted

    refused = wire.move(
        ControllerMode.kExternalController, MotionGeneratorMode.kNone, command_id=12
    )

    assert refused == MoveStatus.kCommandNotPossibleRejected
    assert server.robot_state.state["robot_mode"] == RobotMode.kReflex
    assert server.motion_limits.violated is True

    # Recovery is the way out, and then a fresh motion really is checked again.
    assert wire.automatic_error_recovery(command_id=13) == 0
    assert (
        wire.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone, command_id=14)
        == MoveStatus.kMotionStarted
    )
    assert server.motion_limits.violated is False
    for _ in range(5):
        wire.read_state()
        wire.answer(tau_j_d=[5.0] + [0.0] * 6)
    assert wire.read_move_response() == MoveStatus.kReflexAborted


def test_a_move_during_a_motion_preempts_instead_of_rebasing(serve, client):
    """Restarting a motion is not a way to buy an unchecked step.

    A Move accepted on top of a running motion left the difference history in
    place and then rebased it on the new motion's first command, which is
    checked against the start-pose tolerance alone -- so every extra Move bought
    a free jump of that size with no velocity, acceleration or jerk check on it.
    """
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)
    wire.stream(5, q_c=list(HOME))

    assert (
        wire.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition, command_id=15)
        == MoveStatus.kPreempted
    )
    assert wire.read_move_response() == MoveStatus.kMotionStarted

    # The new motion is seeded from the recaptured standstill, so a jump away
    # from where the robot actually is is still a start-pose error.
    for _ in range(5):
        wire.read_state()
        wire.answer(q_c=[HOME[0] + 0.5] + HOME[1:])

    assert wire.read_move_response() == MoveStatus.kReflexAborted
    assert server.robot_state.state["errors"][
        JOINT_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX
    ] is True


def test_a_stale_echo_cannot_walk_a_step_past_enforcement(serve, client):
    """Everything that reaches physics is checked, however it was addressed.

    Sixty 40 rad/s steps behind an echo five cycles old used to be applied in
    full with --enforce-motion-limits on, and the checker reported no violation
    at all: a stale command's differential checks were skipped, which in position
    mode took the joint *velocity* limit with them.
    """
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)
    wire.stream(5, q_c=list(HOME))

    target = list(HOME)
    for _ in range(60):
        wire.read_state()
        target = [target[0] + 40.0 * DELTA_T] + HOME[1:]  # 40 rad/s, 15x the limit
        wire.answer_stale(5, q_c=target)

    assert wire.read_move_response() == MoveStatus.kReflexAborted
    assert any(server.robot_state.state["errors"])
    applied = [
        call.args[0][0] for call in server.physics_sim.update_joint_positions.call_args_list
    ]
    assert max(applied) < HOME[0] + 0.1, "the 40 rad/s ramp reached physics"


def test_the_command_after_a_gap_cannot_teleport_into_physics(serve, client):
    """The resume command is checked, and the step that reaches physics is bounded.

    With a grace cycle it was not: the first command after any gap could be
    anywhere in the joint range, and a full-range jump was applied with the
    checker reporting nothing wrong.
    """
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)
    wire.stream(5, q_c=list(HOME))

    # A gap, then a full-range teleport.
    for _ in range(6):
        wire.read_state()
    teleport = [2.7, 1.7, 2.9, -0.2, 2.8, 4.5, 3.0]
    for _ in range(5):
        wire.read_state()
        wire.answer(q_c=teleport)

    assert wire.read_move_response() == MoveStatus.kReflexAborted
    for call in server.physics_sim.update_joint_positions.call_args_list:
        applied = np.asarray(call.args[0])
        assert abs(applied[0] - HOME[0]) < 0.05, "the teleport was applied"


def test_a_gap_of_a_few_cycles_is_not_an_error(serve, client):
    """The sim's own loss is one to three cycles, and must stay invisible."""
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kJointImpedance, MotionGeneratorMode.kJointVelocity)

    # A compliant 4 rad/s^2 ramp with a two-cycle hole punched in it.
    velocity = 0.0
    for step in range(40):
        state = wire.read_state()
        velocity = 4.0 * DELTA_T * (step + 1)
        if step % 12 in (5, 6):
            continue  # the client's answer never arrives
        wire.answer(dq_c=[velocity] + [0.0] * 6)

    assert not any(state[ERRORS_SLICE])
    assert server.motion_limits.violated is False


def test_the_base_role_never_has_its_commanded_torque_written(serve, client, mock_base_sim):
    """The base is a twist generator: tau_J_d is not its to command or report."""
    server = serve(sim=mock_base_sim, mobile_base=True)
    wire = client()
    wire.connect()
    wire.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)

    wire.stream(20, o_dp_ee_c=[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], tau_j_d=[9.0] * 7)

    assert server.robot_state.state["tau_J_d"] == [0.0] * 7
    mock_base_sim.update_torques.assert_not_called()


def test_the_reported_derivatives_are_the_ones_the_checker_computed(serve, client):
    """q_d/dq_d/ddq_d carry dq_{c,k-1} and ddq_{c,k-1}, from the same differencing.

    That is what lets a client predict every derivative the sim will compute
    before it sends the command (libfranka docs/overview.rst).
    """
    server = serve()
    wire = start_motion(client, ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)

    velocity = 0.2  # rad/s, well inside every limit
    state = wire.ramp(
        [[HOME[0] + velocity * DELTA_T * (step + 1)] + HOME[1:] for step in range(30)]
    )

    assert np.array(state[DQ_D_SLICE])[0] == pytest.approx(velocity, rel=0.2)
    assert abs(np.array(state[DDQ_D_SLICE])[0]) < 1.0
    derivatives = server.motion_limits.applied_derivatives()
    assert derivatives is not None
    assert derivatives[0][0] == pytest.approx(np.array(state[DQ_D_SLICE])[0], rel=1e-3)


def test_the_state_carrying_the_error_beats_the_tcp_response(serve, client):
    """A motion-limit abort keeps the same ordering guarantee the comm abort has.

    It is harder to keep, because this abort is raised on the UDP receive thread
    and can latch in the microseconds between the publish loop serialising a
    state and putting it on the socket -- so the packet already on its way was
    built before the error existed and cannot be the one that carries it. The
    response waits for a state packed *after* the latch.
    """
    serve(enforce=True)
    wire = start_motion(client, ControllerMode.kExternalController, MotionGeneratorMode.kNone)

    saw_error_state = False
    status = None
    for _ in range(400):
        if select.select([wire.tcp], [], [], 0)[0]:
            status = wire.read_move_response()
            break
        state = wire.read_state()
        if state[ERRORS_SLICE][CONTROLLER_TORQUE_DISCONTINUITY_INDEX] == 1:
            saw_error_state = True
        wire.answer(tau_j_d=[5.0] + [0.0] * 6)

    assert status == MoveStatus.kReflexAborted
    assert saw_error_state, "kReflexAborted arrived before any state carried the error"


def test_a_deferred_abort_waits_for_a_state_packed_after_the_error(mock_physics_sim):
    """The wire race is microseconds wide, so drive the interleaving directly.

    An abort raised on the UDP thread can latch *between* the publish loop
    serialising a state and putting it on the socket. That packet was built
    before the error existed, so releasing the kReflexAborted when it lands
    would put the response ahead of every state that carries the error --
    exactly the order libfranka's throwOnMotionError cannot tolerate.
    """
    from types import SimpleNamespace

    from franka_sim.franka_sim_server import FrankaSimServer

    server = FrankaSimServer(
        physics_sim=mock_physics_sim, enable_gripper=False, enforce_motion_limits=True
    )
    sent = []
    server.client_socket = SimpleNamespace(sendall=sent.append)
    server.current_motion_id = 5
    server.transmitting_state = True  # a publish loop is running to flush it
    server._states_packed = 7  # ...and has serialised seven states so far

    server._abort_with_error(CONTROLLER_TORQUE_DISCONTINUITY_INDEX, "test violation")

    assert sent == [], "the response went out at latch time"
    assert server.robot_state.state["errors"][CONTROLLER_TORQUE_DISCONTINUITY_INDEX] is True

    # State 7 reaches the socket. It was packed before the error, so it is not
    # the state that carries it and the response must keep waiting.
    server._flush_pending_move_response()
    assert sent == [], "the response followed a state serialised before the error"

    # State 8 is packed -- with the error in it -- and sent.
    server._states_packed += 1
    server._flush_pending_move_response()
    assert len(sent) == 1
    assert struct.unpack("<B3x", sent[0][12:])[0] == MoveStatus.kReflexAborted


def test_any_affirmative_spelling_enables_enforcement():
    """``=2`` and ``=y`` are not typos to be silently ignored."""
    for value in ("1", "2", "7", "true", "T", " yes ", "y", "on", "enabled"):
        assert enforcement_enabled_by_env({ENFORCE_ENV_VAR: value}) is True
    for value in ("0", "", "no", "n", "off", "disabled", "false", "0000"):
        assert enforcement_enabled_by_env({ENFORCE_ENV_VAR: value}) is False


def test_the_two_strictness_switches_are_independent(serve, client):
    """--enforce-motion-limits does not turn --enforce-comm-constraints on."""
    server = serve(enforce=True)

    assert server.enforce_motion_limits is True
    assert server.enforce_comm_constraints is False


# --- layer 3: end to end over real physics -----------------------------------

mujoco = pytest.importorskip("mujoco")

from franka_sim.franka_sim_server import FrankaSimServer  # noqa: E402
from franka_sim.mujoco_franka_sim import MujocoFrankaSim, default_fr3_mjcf  # noqa: E402

try:
    FR3_MJCF = default_fr3_mjcf()
except Exception:  # pragma: no cover - depends on the host's cache/network
    FR3_MJCF = None


@pytest.fixture
def live_server():
    """A real FrankaSimServer over a real MuJoCo arm, enforcement as asked."""
    if FR3_MJCF is None or not FR3_MJCF.exists():
        pytest.skip("the MuJoCo Menagerie FR3 model is neither cached nor downloadable")

    made = []

    def _live(enforce=True):
        sim = MujocoFrankaSim()
        sim.initialize_simulation()
        server = FrankaSimServer(
            physics_sim=sim, enable_gripper=False, enforce_motion_limits=enforce
        )
        accept_thread = threading.Thread(target=server.run_server, daemon=True)
        accept_thread.start()
        sim.running = True
        physics_thread = threading.Thread(target=sim.run_simulation, daemon=True)
        physics_thread.start()
        assert wait_for_server(COMMAND_PORT), "the FCI server never came up on port 1337"
        made.append((server, sim, accept_thread, physics_thread))
        return server

    yield _live

    for server, sim, accept_thread, physics_thread in made:
        server.stop()
        sim.stop()
        physics_thread.join(timeout=3.0)
        accept_thread.join(timeout=3.0)
    time.sleep(0.4)


def test_a_smooth_sine_completes_clean_over_the_real_wire(live_server, client):
    """Enforcement on, real physics, real sockets: a conforming stream is untouched."""
    import math

    server = live_server(enforce=True)
    wire = client()
    wire.connect()
    wire.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)

    start = list(wire.read_state()[Q_SLICE])
    waypoints = []
    for step in range(1500):
        delta = 0.2 * (1 - math.cos(math.pi * step * DELTA_T))
        waypoints.append([start[0] + delta, start[1], start[2], start[3], start[4] + delta]
                         + start[5:7])
    state = wire.ramp(waypoints)

    assert not any(state[ERRORS_SLICE]), "a smooth sine tripped a limit"
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove
    assert server.motion_limits.violated is False


def test_a_half_radian_step_aborts_over_the_real_wire(live_server, client):
    """The counter-example, with the error the client actually reads off the wire."""
    live_server(enforce=True)
    wire = client()
    wire.connect()
    wire.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)

    start = list(wire.read_state()[Q_SLICE])
    # Settle on the start pose first, so what the step breaks is unambiguously a
    # mid-motion discontinuity rather than the start-pose check.
    wire.ramp([start] * 20)
    for _ in range(10):
        wire.read_state()
        wire.answer(q_c=[start[0] + 0.5] + start[1:7])

    assert wire.read_move_response() == MoveStatus.kReflexAborted
    state = wire.read_state()
    # A mid-motion step in ``q_c`` breaks the envelope *and* the acceleration
    # limit; hardware names the acceleration one, so index 14, not 13.
    assert state[ERRORS_SLICE][JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX] == 1
    assert state[ERRORS_SLICE][JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX] == 0
    assert state[ROBOT_MODE_INDEX] == RobotMode.kReflex
