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
    CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
    CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
    CONTROLLER_TORQUE_DISCONTINUITY_INDEX,
    DELTA_T,
    ENFORCE_ENV_VAR,
    JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
    JOINT_MOTION_GENERATOR_POSITION_LIMITS_VIOLATION_INDEX,
    JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
    JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
    JOINT_POSITION_LIMITS,
    JOINT_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX,
    MAX_COALESCED_CYCLES,
    MAX_JOINT_ACCELERATION,
    MAX_JOINT_JERK,
    MAX_ROTATIONAL_VELOCITY,
    MAX_TORQUE,
    MAX_TORQUE_RATE,
    MAX_TRANSLATIONAL_ACCELERATION,
    MAX_TRANSLATIONAL_JERK,
    MAX_TRANSLATIONAL_VELOCITY,
    TAU_J_RANGE_VIOLATION_INDEX,
    MotionLimitChecker,
    Violation,
    _Differentiator,
    enforcement_enabled_by_env,
    lower_joint_velocity_limits,
    upper_joint_velocity_limits,
)
from franka_sim.robot_state import _ROBOT_STATE_PACKER, RobotState

#: A configuration comfortably inside every joint's range, so the
#: position-dependent velocity limits sit at their flat caps. Joints 4 and 6 do
#: not straddle zero, hence the non-zero entries.
HOME = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]


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


def test_an_implied_velocity_over_the_cap_is_a_velocity_limits_violation():
    checker = position_checker()
    cap = upper_joint_velocity_limits(HOME)[0]
    # A steady ramp at the cap is legal; the history has to carry that velocity
    # already, or the step into it would be the acceleration violation instead.
    drive_position_history(checker, list(HOME), spread(cap), [0.0] * 7)
    assert checker.check(command(q_c=[HOME[0] + cap * DELTA_T] + HOME[1:])) is None

    over = cap + 0.01
    violation = checker.check(command(q_c=[HOME[0] + over * DELTA_T] + HOME[1:]))
    assert violation.error_index == JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX
    assert violation.error_name == "joint_motion_generator_velocity_limits_violation"
    assert violation.value == pytest.approx(over)
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


def test_a_velocity_step_is_a_velocity_discontinuity():
    """Acceleration, one derivative up from the position generator's."""
    limit = MAX_JOINT_ACCELERATION[0]
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.VELOCITY, robot_state_at())
    drive_velocity_history(checker, spread(0.5), spread(limit))

    assert checker.check(command(dq_c=spread(0.5 + limit * DELTA_T))) is None

    over = limit + 1e-6
    violation = checker.check(command(dq_c=spread(0.5 + over * DELTA_T)))
    assert violation.error_index == JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX


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

def test_a_cartesian_twist_is_judged_by_norms_not_components():
    """The limiter treats O_dP_EE_c as two Vector3d and compares .norm()."""
    checker = MotionLimitChecker()
    steady = [2.0, 2.0, 0.0, 0.0, 0.0, 0.0]  # norm 2.83 < 2.999, each part < 2.999
    checker.start_motion(ControlMode.STEERING_DRIVE, robot_state_at(O_dP_EE_d=steady))

    assert checker.check(command(O_dP_EE_c=steady)) is None

    # Each component is still under the limit, but the norm is not.
    over = [2.2, 2.2, 0.0, 0.0, 0.0, 0.0]
    violation = checker.check(command(O_dP_EE_c=over))
    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX
    assert violation.axis == "translational norm"
    assert violation.value == pytest.approx((2.2**2 + 2.2**2) ** 0.5)
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


def test_a_twist_acceleration_step_is_a_cartesian_velocity_discontinuity():
    limit = MAX_TRANSLATIONAL_ACCELERATION
    checker = MotionLimitChecker()
    checker.start_motion(ControlMode.STEERING_DRIVE, robot_state_at())
    # Ramp in at exactly the acceleration limit, which is also the jerk the
    # first cycle carries: 8.999 / 1e-3 = 8999 < 4499999.
    checker.record(command(O_dP_EE_c=[limit * DELTA_T, 0, 0, 0, 0, 0]))

    violation = checker.check(command(O_dP_EE_c=[3 * limit * DELTA_T, 0, 0, 0, 0, 0]))
    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX
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
    assert violation.error_index == JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX
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
    message_id, q_c=None, dq_c=None, o_dp_ee_c=None, tau_j_d=None, motion_finished=False
):
    """Pack the UDP RobotCommand exactly as libfranka sends it."""
    message = struct.pack("<Q", message_id)
    message += struct.pack("<7d", *(q_c if q_c is not None else [0.0] * 7))
    message += struct.pack("<7d", *(dq_c if dq_c is not None else [0.0] * 7))
    message += struct.pack("<16d", *([0.0] * 16))
    message += struct.pack("<6d", *(o_dp_ee_c if o_dp_ee_c is not None else [0.0] * 6))
    message += struct.pack("<2d", *([0.0] * 2))
    message += struct.pack("<B", 0)
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

    def read_state(self):
        data, address = self.udp.recvfrom(4096)
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


# --- layer 2: the server over the wire, with a mocked simulator --------------


@pytest.fixture
def mock_arm_sim():
    """A mocked arm reporting a *valid* FR3 configuration.

    The shared ``mock_genesis_sim`` reports ``q = 0``, which is not reachable on
    an FR3 at all -- joint 4 lives in [-3.0481, -0.1458] and joint 6 in
    [0.5409, 4.5205] -- so an all-zeros command would trip the position-limit
    check before anything this file is about.
    """
    from unittest.mock import Mock

    sim = Mock()
    sim.get_robot_state.return_value = {
        "q": np.array(HOME),
        "dq": np.zeros(7),
        "tau_J": np.zeros(7),
    }
    return sim


@pytest.fixture
def serve(mock_arm_sim):
    """Start a FrankaSimServer with a mocked simulator, enforcement as asked."""
    from franka_sim.franka_sim_server import FrankaSimServer

    started = []

    def _serve(enforce=False, sim=None, mobile_base=False):
        server = FrankaSimServer(
            genesis_sim=sim if sim is not None else mock_arm_sim,
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


def start_motion(client_factory, controller_mode, motion_mode, command_id=2):
    """Connect and get past the Move handshake, ready to stream commands."""
    wire = client_factory()
    wire.connect()
    assert wire.move(controller_mode, motion_mode, command_id=command_id) == (
        MoveStatus.kMotionStarted
    )
    wire.read_move_response()  # the kSuccess that follows the first published state
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
        JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
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


def test_enforced_the_violating_command_never_reaches_the_simulator(serve, client):
    """The real robot rejects it; so must the sim, or the physics saw a jump."""
    server = serve(enforce=True)
    wire = start_motion(client, ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)

    for _ in range(5):
        wire.read_state()
        wire.answer(q_c=[HOME[0] + 0.5] + HOME[1:])
    wire.read_move_response()

    commanded = [call.args[0] for call in server.genesis_sim.update_joint_positions.call_args_list]
    assert commanded, "the simulator was never commanded at all"
    for target in commanded:
        # Everything the sim ever saw is the hold's own HOME target; the 0.5 rad
        # jump on joint 1 never got past the checker.
        assert np.array(target)[0] == pytest.approx(HOME[0]), f"the jump reached the sim: {target}"


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


def test_the_base_twist_is_held_to_the_cartesian_limits(serve, client, mock_base_sim):
    """A body twist is a motion-generator signal, judged by the wire's own constants."""
    server = serve(enforce=True, sim=mock_base_sim, mobile_base=True)
    wire = client()
    wire.connect()
    wire.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)
    wire.read_move_response()

    # A jerk-limited ramp in is fine...
    ramp = [[0.001 * (step + 1), 0.0, 0.0, 0.0, 0.0, 0.0] for step in range(30)]
    state = wire.ramp(ramp, field="o_dp_ee_c")
    assert not any(state[ERRORS_SLICE])

    # ...a 5 m/s step is not: 5 m/s exceeds kMaxTranslationalVelocity outright.
    for _ in range(5):
        wire.read_state()
        wire.answer(o_dp_ee_c=[5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert wire.read_move_response() == MoveStatus.kReflexAborted

    latched = server.robot_state.state["errors"]
    assert latched[CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX] is True
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

    for call in server.genesis_sim.update_joint_positions.call_args_list:
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
        call.args[0][0] for call in server.genesis_sim.update_joint_positions.call_args_list
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
    for call in server.genesis_sim.update_joint_positions.call_args_list:
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
    wire.read_move_response()

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


def test_a_deferred_abort_waits_for_a_state_packed_after_the_error(mock_genesis_sim):
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
        genesis_sim=mock_genesis_sim, enable_gripper=False, enforce_motion_limits=True
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
            genesis_sim=sim, enable_gripper=False, enforce_motion_limits=enforce
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
    wire.read_move_response()

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
    wire.read_move_response()

    start = list(wire.read_state()[Q_SLICE])
    # Settle on the start pose first, so what the step breaks is unambiguously
    # the velocity limit rather than the start-pose check.
    wire.ramp([start] * 20)
    for _ in range(10):
        wire.read_state()
        wire.answer(q_c=[start[0] + 0.5] + start[1:7])

    assert wire.read_move_response() == MoveStatus.kReflexAborted
    state = wire.read_state()
    assert state[ERRORS_SLICE][JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX] == 1
    assert state[ROBOT_MODE_INDEX] == RobotMode.kReflex
