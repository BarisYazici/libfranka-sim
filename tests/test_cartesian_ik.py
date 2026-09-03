"""The differential-IK maths behind the two Cartesian interfaces.

:mod:`franka_sim.cartesian_ik` is deliberately model-free -- it takes a
Jacobian, a twist and a configuration -- so it can be tested here on matrices
whose answers are known in closed form, without compiling a robot. The
end-to-end claims (a commanded pose is reached, a commanded twist is realised,
joint limits hold) live in ``test_mujoco_franka_sim.py``, against real physics.
"""

import numpy as np
import pytest

from franka_sim.cartesian_ik import (
    DAMPING,
    ELBOW_GAIN,
    ROTATION_GAIN,
    TRANSLATION_GAIN,
    CartesianFeedforward,
    clamp_twist,
    elbow_null_velocity,
    pose_error,
    resolved_rate,
    tracking_twist,
)
from franka_sim.motion_limits import (
    LIMITER_ROTATIONAL_VELOCITY,
    LIMITER_TRANSLATIONAL_VELOCITY,
    rotation_exp,
)

DT = 1e-3


def transform(translation=(0.0, 0.0, 0.0), rotation_vector=(0.0, 0.0, 0.0)):
    """A 4x4 pose from a translation and an axis-angle rotation."""
    pose = np.eye(4)
    pose[:3, :3] = rotation_exp(np.asarray(rotation_vector, dtype=float))
    pose[:3, 3] = translation
    return pose


# -- pose error ---------------------------------------------------------------


def test_pose_error_of_a_pose_against_itself_is_zero():
    pose = transform((0.3, -0.2, 0.5), (0.4, -0.1, 0.2))
    assert pose_error(pose, pose) == pytest.approx(np.zeros(6), abs=1e-12)


def test_pose_error_translation_is_the_plain_difference():
    current = transform((0.1, 0.2, 0.3))
    desired = transform((0.4, 0.2, -0.1))
    assert pose_error(desired, current)[:3] == pytest.approx([0.3, 0.0, -0.4])


def test_pose_error_rotation_is_the_base_frame_axis_angle():
    """log(R_d R^T), i.e. the residual rotation expressed in the *base* frame.

    Same frame as the Jacobian's rotational rows -- which is what makes the two
    halves composable into one 6-vector.
    """
    rotation_vector = np.array([0.0, 0.0, 0.35])
    current = transform(rotation_vector=(0.2, 0.1, 0.0))
    desired = np.eye(4)
    desired[:3, :3] = rotation_exp(rotation_vector) @ current[:3, :3]
    assert pose_error(desired, current)[3:] == pytest.approx(rotation_vector, abs=1e-9)


# -- twist clamp --------------------------------------------------------------


def test_clamp_twist_leaves_a_legal_twist_untouched():
    twist = np.array([0.1, -0.2, 0.05, 0.3, 0.0, -0.4])
    assert clamp_twist(twist) == pytest.approx(twist)


def test_clamp_twist_scales_each_half_to_its_own_limit_by_norm():
    """Norm-wise per half, as the FR3's limit table states them."""
    twist = np.array([30.0, 40.0, 0.0, 0.0, 0.0, 25.0])
    clamped = clamp_twist(twist)

    assert np.linalg.norm(clamped[:3]) == pytest.approx(LIMITER_TRANSLATIONAL_VELOCITY)
    assert np.linalg.norm(clamped[3:]) == pytest.approx(LIMITER_ROTATIONAL_VELOCITY)
    # Direction preserved: scaled, not truncated per axis.
    assert clamped[:3] / np.linalg.norm(clamped[:3]) == pytest.approx([0.6, 0.8, 0.0])


def test_clamp_twist_bounds_a_teleporting_command_with_checks_off():
    """The guard rail: an unchecked 10 m pose step must not become 400 m/s."""
    desired = transform((10.0, 0.0, 0.0))
    twist = tracking_twist(desired, np.eye(4), np.zeros(6))
    assert np.linalg.norm(twist[:3]) == pytest.approx(LIMITER_TRANSLATIONAL_VELOCITY)


# -- tracking twist -----------------------------------------------------------


def test_tracking_twist_is_feedforward_plus_proportional_error():
    feedforward = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.01])
    current = transform((0.0, 0.0, 0.0))
    desired = transform((0.001, 0.0, 0.0), (0.0, 0.0, 0.002))

    twist = tracking_twist(desired, current, feedforward)

    assert twist[0] == pytest.approx(0.05 + TRANSLATION_GAIN * 0.001)
    assert twist[5] == pytest.approx(0.01 + ROTATION_GAIN * 0.002)


def test_tracking_twist_with_no_error_is_the_feedforward():
    pose = transform((0.2, 0.1, 0.4), (0.1, 0.2, 0.3))
    feedforward = np.array([0.02, -0.01, 0.0, 0.0, 0.03, 0.0])
    assert tracking_twist(pose, pose, feedforward) == pytest.approx(feedforward)


# -- damped least squares -----------------------------------------------------


def test_resolved_rate_realises_the_twist_on_a_well_conditioned_jacobian():
    """Damping costs a little accuracy; on a healthy Jacobian, very little."""
    jacobian = np.hstack((np.eye(6), np.zeros((6, 1))))
    twist = np.array([0.1, -0.2, 0.3, 0.05, 0.0, -0.1])

    joint_velocity = resolved_rate(jacobian, twist)

    assert jacobian @ joint_velocity == pytest.approx(twist, rel=1e-2)
    assert joint_velocity[6] == 0.0  # nothing asked of the redundant column


def test_resolved_rate_bounds_the_joint_velocity_at_a_singularity():
    """The whole point of the damping.

    A direction that has lost rank asks the undamped pseudo-inverse for
    ``1 / sigma`` of joint speed -- unbounded as ``sigma -> 0``. The damped form
    tops out at ``1 / (2 lambda)``.
    """
    twist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    ceiling = 1.0 / (2.0 * DAMPING)

    for sigma in (1e-2, 1e-4, 1e-8, 0.0):
        jacobian = np.hstack((np.eye(6), np.zeros((6, 1))))
        jacobian[5, 5] = sigma
        joint_velocity = resolved_rate(jacobian, twist)
        assert np.all(np.isfinite(joint_velocity))
        assert np.linalg.norm(joint_velocity) <= ceiling + 1e-9

    # ...where an undamped inverse cannot even be formed on the rank-deficient
    # Jacobian the last iteration built.
    with pytest.raises(np.linalg.LinAlgError):
        resolved_rate(jacobian, twist, damping=0.0)


def test_the_null_space_term_never_disturbs_the_end_effector():
    """The redundancy contract: the elbow moves, the EE task does not.

    Exact only in the undamped limit -- the projector is built from the *damped*
    inverse, so at the working :data:`DAMPING` a sliver of the null-space
    velocity does reach the EE. Checked at both: exactly at lambda -> 0, and
    under 1% of the null-space velocity at the real one.
    """
    rng = np.random.default_rng(20260822)
    jacobian = rng.normal(size=(6, 7))
    twist = np.array([0.1, 0.0, -0.05, 0.0, 0.02, 0.0])
    null_velocity = np.array([0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0])

    without = resolved_rate(jacobian, twist, None, damping=1e-9)
    with_null = resolved_rate(jacobian, twist, null_velocity, damping=1e-9)
    assert not np.allclose(with_null, without)  # the joints really did change
    assert jacobian @ with_null == pytest.approx(jacobian @ without, abs=1e-9)

    leak = jacobian @ resolved_rate(jacobian, twist, null_velocity) - jacobian @ resolved_rate(
        jacobian, twist, None
    )
    assert np.linalg.norm(leak) < 1e-2 * np.linalg.norm(null_velocity)


def test_elbow_null_velocity_is_a_proportional_term_on_joint_three_alone():
    joint_positions = np.array([0.1, 0.2, 0.3, -1.5, 0.0, 1.6, 0.7])
    velocity = elbow_null_velocity(0.5, joint_positions)

    assert velocity[2] == pytest.approx(ELBOW_GAIN * (0.5 - 0.3))
    assert np.count_nonzero(velocity) == 1
    assert len(velocity) == 7


# -- feed-forward -------------------------------------------------------------


def test_the_feedforward_differences_a_streamed_pose():
    feedforward = CartesianFeedforward()
    pose = transform((0.0, 0.0, 0.0))
    feedforward.step(pose, DT)  # the first sample only seeds

    for _ in range(5):
        pose = pose.copy()
        pose[0, 3] += 0.1 * DT
        twist = feedforward.step(pose, DT)

    assert twist[:3] == pytest.approx([0.1, 0.0, 0.0], abs=1e-9)
    assert twist[3:] == pytest.approx(np.zeros(3), abs=1e-9)


def test_the_feedforward_differences_a_rotating_pose():
    """log(R_k R_{k-1}^T) / span -- a rotation is not a difference of matrices."""
    feedforward = CartesianFeedforward()
    omega = np.array([0.0, 0.0, 0.4])
    step_rotation = rotation_exp(omega * DT)
    pose = np.eye(4)
    feedforward.step(pose, DT)

    for _ in range(5):
        pose = pose.copy()
        pose[:3, :3] = step_rotation @ pose[:3, :3]
        twist = feedforward.step(pose, DT)

    assert twist[3:] == pytest.approx(omega, abs=1e-9)


def test_the_feedforward_survives_the_arrival_beat():
    """A [0, 2] beat between the network and physics clocks must not alternate.

    The Cartesian twin of PositionFeedforward's reason for existing: the naive
    one-step difference would report 0 and then 2v on alternating steps.
    """
    feedforward = CartesianFeedforward()
    pose = np.eye(4)
    feedforward.step(pose, DT)

    seen = []
    for _ in range(6):
        seen.append(feedforward.step(pose, DT)[0])  # a step with no new command
        pose = pose.copy()
        pose[0, 3] += 0.2 * 2 * DT  # ...then one carrying two cycles' worth
        seen.append(feedforward.step(pose, DT)[0])

    assert seen[2:] == pytest.approx([0.2] * len(seen[2:]), abs=1e-9)


def test_the_feedforward_drops_a_stale_stream_to_zero():
    from franka_sim.sim_common import POSITION_FEEDFORWARD_HOLD_STEPS

    feedforward = CartesianFeedforward()
    pose = np.eye(4)
    feedforward.step(pose, DT)
    pose = pose.copy()
    pose[0, 3] += 0.1 * DT
    assert feedforward.step(pose, DT)[0] == pytest.approx(0.1)

    for _ in range(POSITION_FEEDFORWARD_HOLD_STEPS - 1):
        assert feedforward.step(pose, DT)[0] == pytest.approx(0.1)  # held through the gap
    assert feedforward.step(pose, DT) == pytest.approx(np.zeros(6))


def test_resetting_the_feedforward_forgets_everything():
    """A new motion must not inherit the previous one's velocity or baseline."""
    feedforward = CartesianFeedforward()
    pose = np.eye(4)
    feedforward.step(pose, DT)
    moved = pose.copy()
    moved[0, 3] += 0.1 * DT
    assert feedforward.step(moved, DT)[0] == pytest.approx(0.1)

    feedforward.reset()
    assert feedforward.previous is None
    far = np.eye(4)
    far[0, 3] = 5.0
    # The first sample after a reset seeds; it is never differenced against a
    # baseline from the motion that just ended.
    assert feedforward.step(far, DT) == pytest.approx(np.zeros(6))
