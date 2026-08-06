import math

import pytest
from fakes import FakeEntity

from franka_sim.swerve_base import (
    TMR_DRIVE_JOINTS,
    TMR_JOINT_ORDER,
    TMR_STEER_JOINTS,
    TMR_WHEEL_POSITIONS,
    TMR_WHEEL_RADIUS,
    SwerveBase,
    yaw_to_quat_wxyz,
)


@pytest.fixture
def swerve():
    base = SwerveBase(FakeEntity())
    base.bind()
    return base


def test_tmr_geometry_matches_the_urdf():
    """Module positions and radius come from franka_description tmrv0_2.xacro."""
    assert TMR_WHEEL_POSITIONS == ((0.3, -0.2), (-0.3, 0.2))
    assert TMR_WHEEL_RADIUS == pytest.approx(0.05)
    assert TMR_STEER_JOINTS == ("tmrv0_2_joint_0", "tmrv0_2_joint_2")
    assert TMR_DRIVE_JOINTS == ("tmrv0_2_joint_1", "tmrv0_2_joint_3")
    assert TMR_JOINT_ORDER == (
        "tmrv0_2_joint_0",
        "tmrv0_2_joint_1",
        "tmrv0_2_joint_2",
        "tmrv0_2_joint_3",
    )


def test_bind_resolves_steer_and_drive_dofs(swerve):
    assert swerve.steer_dofs_idx == [0, 2]
    assert swerve.drive_dofs_idx == [1, 3]


def test_yaw_to_quat_wxyz_is_scalar_first():
    root_half = math.sqrt(0.5)
    assert yaw_to_quat_wxyz(0.0) == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert yaw_to_quat_wxyz(math.pi / 2.0) == pytest.approx([root_half, 0.0, 0.0, root_half])


def test_solve_forward_command_points_wheels_forward(swerve):
    swerve.set_twist([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    steer, drive = swerve.solve()
    assert steer == pytest.approx([0.0, 0.0], abs=1e-9)
    assert drive == pytest.approx([0.5 / TMR_WHEEL_RADIUS] * 2, abs=1e-9)


def test_solve_yaw_command_reverses_one_module(swerve):
    """Modules sit at (0.3, -0.2) and (-0.3, 0.2) -- point-symmetric about the
    origin, so a pure yaw needs the same heading on both and opposite speeds.
    The rear module's direct heading is a half turn away from its previous
    angle, so the pi-ambiguity resolution reverses its speed instead.
    """
    swerve.set_twist([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    steer, drive = swerve.solve()

    expected_angle = math.atan2(0.3, 0.2)
    expected_speed = math.hypot(0.2, 0.3) / TMR_WHEEL_RADIUS
    assert steer == pytest.approx([expected_angle, expected_angle], abs=1e-9)
    assert drive == pytest.approx([expected_speed, -expected_speed], abs=1e-9)


def test_set_twist_ignores_non_finite_commands(swerve):
    swerve.set_twist([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    steer_before, drive_before = swerve.solve()
    swerve.set_twist([math.nan, 0.0, 0.0, 0.0, 0.0, 0.0])
    steer_after, drive_after = swerve.solve()
    assert steer_after == pytest.approx(steer_before)
    assert drive_after == pytest.approx(drive_before)


def test_repeated_bad_twists_log_once(swerve, caplog):
    """1 kHz path: the warning must latch, or the log flood adds control jitter."""
    with caplog.at_level("WARNING", logger="franka_sim.swerve_base"):
        for _ in range(50):
            swerve.set_twist([math.inf, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert len(caplog.records) == 1


def test_the_bad_twist_latch_rearms_after_a_good_command(swerve, caplog):
    with caplog.at_level("WARNING", logger="franka_sim.swerve_base"):
        swerve.set_twist([math.inf, 0.0, 0.0, 0.0, 0.0, 0.0])
        swerve.set_twist([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
        swerve.set_twist([math.inf, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert len(caplog.records) == 2


def test_integrate_pose_moves_forward_along_x(swerve):
    swerve.set_twist([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(10):
        swerve.integrate_pose(0.01)
    assert swerve.x == pytest.approx(0.1, abs=1e-9)
    assert swerve.y == pytest.approx(0.0, abs=1e-9)
    assert swerve.theta == pytest.approx(0.0, abs=1e-9)


def test_integrate_pose_accumulates_yaw(swerve):
    swerve.set_twist([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    for _ in range(100):
        swerve.integrate_pose(0.01)
    assert swerve.theta == pytest.approx(1.0, abs=1e-9)


def test_integrate_pose_is_expressed_in_the_body_frame(swerve):
    """After a quarter turn, +vx must move the platform along world +y."""
    swerve.theta = math.pi / 2.0
    swerve.set_twist([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    swerve.integrate_pose(0.1)
    assert swerve.x == pytest.approx(0.0, abs=1e-9)
    assert swerve.y == pytest.approx(0.1, abs=1e-9)


def test_integrate_pose_wraps_theta_to_pi(swerve):
    swerve.set_twist([0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
    for _ in range(100):
        swerve.integrate_pose(0.01)
    assert -math.pi <= swerve.theta <= math.pi


def test_apply_writes_joint_targets_and_base_pose():
    entity = FakeEntity()
    base = SwerveBase(entity, base_height=0.05)
    base.bind()
    base.set_twist([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    base.apply(0.01)

    steer_values, steer_dofs = entity.position_commands[-1]
    drive_values, drive_dofs = entity.velocity_commands[-1]
    assert steer_dofs == [0, 2]
    assert drive_dofs == [1, 3]
    assert steer_values == pytest.approx([0.0, 0.0], abs=1e-9)
    assert drive_values == pytest.approx([1.0 / TMR_WHEEL_RADIUS] * 2, abs=1e-9)
    assert entity.positions[-1] == pytest.approx([0.01, 0.0, 0.05], abs=1e-9)
    assert entity.quaternions[-1] == pytest.approx([1.0, 0.0, 0.0, 0.0], abs=1e-9)


def test_bind_resolves_the_root_dofs():
    entity = FakeEntity(root_dofs=(7, 8, 9))
    base = SwerveBase(entity)
    base.bind()
    assert base.root_dofs_idx == [7, 8, 9]


def test_bind_tolerates_an_entity_without_a_root_joint():
    entity = FakeEntity(root_dofs=())
    base = SwerveBase(entity)
    base.bind()
    assert base.root_dofs_idx == []


def test_apply_zeroes_only_the_root_dofs_not_the_whole_entity():
    """Regression: the base pose write must not stop the arms.

    Genesis' ``set_pos``/``set_quat`` zero the velocity of EVERY DOF of the
    entity unless ``zero_velocity=False``. On the mobile-duo scene that entity
    carries both FR3 arms, and ``apply`` runs once per physics step, so the
    default pinned the arms while the (kinematic) base still moved.
    """
    entity = FakeEntity(root_dofs=(30, 31, 32, 33, 34, 35))
    base = SwerveBase(entity, base_height=0.05)
    base.bind()
    base.set_twist([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    base.apply(0.01)

    assert [zero_velocity for _, _, zero_velocity in entity.pose_writes] == [False, False]
    values, dofs = entity.set_velocity_calls[-1]
    assert dofs == [30, 31, 32, 33, 34, 35]
    assert values == pytest.approx([0.0] * 6)


def test_apply_skips_the_root_zeroing_when_there_is_no_root_joint():
    entity = FakeEntity(root_dofs=())
    base = SwerveBase(entity)
    base.bind()

    base.apply(0.01)

    assert entity.set_velocity_calls == []


def test_wheel_state_is_reported_in_joint_index_order():
    entity = FakeEntity(dof_positions=[0.11, 0.22, 0.33, 0.44], dof_velocities=[1.1, 2.2, 3.3, 4.4])
    base = SwerveBase(entity)
    base.bind()

    positions, velocities = base.wheel_state()

    assert positions == pytest.approx([0.11, 0.22, 0.33, 0.44])
    assert velocities == pytest.approx([1.1, 2.2, 3.3, 4.4])


def test_base_pose_matrix_is_column_major(swerve):
    swerve.x, swerve.y, swerve.theta = 1.0, 2.0, math.pi / 2.0
    flat = swerve.base_pose_matrix()
    assert flat.shape == (16,)
    matrix = flat.reshape(4, 4).T
    assert matrix[:3, 3] == pytest.approx([1.0, 2.0, 0.0], abs=1e-9)
    assert matrix[:2, 0] == pytest.approx([0.0, 1.0], abs=1e-9)
    assert matrix[3, :] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_reset_pose_clears_the_integrated_state(swerve):
    swerve.set_twist([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    swerve.integrate_pose(0.5)
    swerve.reset_pose()
    assert (swerve.x, swerve.y, swerve.theta) == (0.0, 0.0, 0.0)
