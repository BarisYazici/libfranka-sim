import math

import numpy as np
import pytest
from fakes import FakeDuoEntity

from franka_sim.franka_genesis_sim import ControlMode
from franka_sim.mobile_duo_sim import (
    ARM_EE_LINKS,
    ARM_INITIAL_Q,
    ARM_JOINT_NAMES,
    ROLE_BASE,
    ROLE_LEFT,
    ROLE_RIGHT,
    ROLES,
    MobileDuoScene,
    SceneView,
    pose_to_column_major,
)
from franka_sim.swerve_base import TMR_JOINT_ORDER, TMR_WHEEL_RADIUS


@pytest.fixture
def scene(tmp_path):
    """A MobileDuoScene bound to a fake entity, with no Genesis scene built."""
    urdf_path = tmp_path / "duo.urdf"
    urdf_path.write_text('<?xml version="1.0"?><robot name="duo"></robot>')
    duo = MobileDuoScene(urdf_path, enable_vis=False, base_height=0.05)
    duo.robot = FakeDuoEntity()
    duo._bind_entity()
    duo._read_and_publish_state()
    return duo


def test_joint_names_match_the_generated_urdf():
    assert ARM_JOINT_NAMES[ROLE_LEFT] == [f"left_fr3v2_joint{i}" for i in range(1, 8)]
    assert ARM_JOINT_NAMES[ROLE_RIGHT] == [f"right_fr3v2_joint{i}" for i in range(1, 8)]
    assert ARM_EE_LINKS == {
        ROLE_LEFT: "left_fr3v2_link7",
        ROLE_RIGHT: "right_fr3v2_link7",
    }
    assert ROLES == (ROLE_LEFT, ROLE_RIGHT, ROLE_BASE)


def test_arm_initial_pose_matches_upstream_ros2_control_defaults():
    expected = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
    assert ARM_INITIAL_Q == pytest.approx(expected)


def test_pose_to_column_major_is_identity_for_a_unit_quaternion():
    flat = pose_to_column_major([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
    matrix = flat.reshape(4, 4).T
    assert matrix[:3, :3] == pytest.approx(np.eye(3), abs=1e-12)
    assert matrix[:3, 3] == pytest.approx([1.0, 2.0, 3.0])


def test_each_role_reports_seven_element_state(scene):
    for role in ROLES:
        state = scene.get_role_state(role)
        for key in ("q", "dq", "ddq", "q_d", "dq_d", "ddq_d", "tau_J"):
            assert state[key].shape == (7,), (role, key)
        assert state["O_T_EE"].shape == (16,)


def test_arm_state_reads_only_that_arms_joints(scene):
    left_dofs = [scene.robot.joints[name].dof_idx_local for name in ARM_JOINT_NAMES[ROLE_LEFT]]
    expected = [scene.robot.dof_positions[index] for index in left_dofs]
    assert scene.get_role_state(ROLE_LEFT)["q"] == pytest.approx(expected)


def test_base_state_pads_the_wheel_joints(scene):
    state = scene.get_role_state(ROLE_BASE)
    wheel_dofs = [scene.robot.joints[name].dof_idx_local for name in TMR_JOINT_ORDER]
    expected = [scene.robot.dof_positions[index] for index in wheel_dofs]
    assert state["q"][:4] == pytest.approx(expected)
    assert state["q"][4:] == pytest.approx([0.0, 0.0, 0.0])


def test_arm_torque_control_writes_only_that_arms_dofs(scene):
    torques = np.arange(7, dtype=float)
    scene.set_arm_control_mode(ROLE_RIGHT, ControlMode.TORQUE)
    scene.update_arm_torques(ROLE_RIGHT, torques)
    scene._apply_control()

    values, dofs = scene.robot.force_commands[-1]
    right_dofs = [scene.robot.joints[name].dof_idx_local for name in ARM_JOINT_NAMES[ROLE_RIGHT]]
    assert dofs == right_dofs
    assert values == pytest.approx(torques)


def test_arms_keep_independent_control_modes(scene):
    scene.set_arm_control_mode(ROLE_LEFT, ControlMode.TORQUE)
    scene.set_arm_control_mode(ROLE_RIGHT, ControlMode.POSITION)
    assert scene.arm_control_modes[ROLE_LEFT] is ControlMode.TORQUE
    assert scene.arm_control_modes[ROLE_RIGHT] is ControlMode.POSITION


def test_base_twist_moves_the_whole_robot(scene):
    scene.update_base_twist([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    scene._apply_control()

    assert scene.robot.positions[-1] == pytest.approx([scene.dt, 0.0, 0.05], abs=1e-12)
    drive_values, drive_dofs = scene.robot.velocity_commands[-1]
    assert drive_values == pytest.approx([1.0 / TMR_WHEEL_RADIUS] * 2, abs=1e-9)
    assert drive_dofs == scene.swerve.drive_dofs_idx


def test_spine_joint_is_bound(scene):
    assert scene.spine_dof_idx == scene.robot.joints["franka_spine_vertical_joint"].dof_idx_local


def test_set_spine_position_writes_the_prismatic_joint(scene):
    scene.set_spine_position(0.42)

    values, dofs = scene.robot.set_position_calls[-1]
    assert dofs == [scene.spine_dof_idx]
    assert values == pytest.approx([0.42])


def test_set_spine_position_clamps_to_the_urdf_limits(scene):
    scene.set_spine_position(-1.0)
    assert scene.robot.set_position_calls[-1][0] == pytest.approx([0.0])
    scene.set_spine_position(9.0)
    assert scene.robot.set_position_calls[-1][0] == pytest.approx([0.85])


def test_spine_model_drives_the_joint_every_step(scene):
    class StubSpine:
        def __init__(self):
            self.value = 0.0

        def position_m(self):
            return self.value

    spine = StubSpine()
    scene.spine_model = spine

    spine.value = 0.3
    scene._apply_control()
    assert scene.robot.set_position_calls[-1][0] == pytest.approx([0.3])

    spine.value = 0.6
    scene._apply_control()
    assert scene.robot.set_position_calls[-1][0] == pytest.approx([0.6])


def test_no_spine_model_leaves_the_joint_alone(scene):
    assert scene.spine_model is None
    before = len(scene.robot.set_position_calls)
    scene._apply_control()
    assert len(scene.robot.set_position_calls) == before


def test_view_lifecycle_methods_are_no_ops(scene):
    view = scene.view(ROLE_LEFT)
    view.initialize_simulation()
    view.start()
    view.stop()
    assert scene.running is False
    assert scene.robot is not None


def test_view_delegates_the_arm_contract(scene):
    view = scene.view(ROLE_LEFT)
    torques = np.ones(7)
    view.set_control_mode(ControlMode.TORQUE)
    view.update_torques(torques)

    assert scene.arm_control_modes[ROLE_LEFT] is ControlMode.TORQUE
    assert scene.arm_torques[ROLE_LEFT] == pytest.approx(torques)
    assert view.get_robot_state()["q"].shape == (7,)


def test_base_view_delegates_the_twist(scene):
    scene.view(ROLE_BASE).update_base_twist([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    steer, drive = scene.swerve.solve()
    assert drive == pytest.approx([0.5 / TMR_WHEEL_RADIUS] * 2, abs=1e-9)


def test_arm_view_ignores_a_twist(scene):
    scene.view(ROLE_LEFT).update_base_twist([9.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _, drive = scene.swerve.solve()
    assert drive == pytest.approx([0.0, 0.0], abs=1e-12)


def test_view_rejects_an_unknown_role(scene):
    with pytest.raises(ValueError):
        scene.view("middle")


def test_view_reports_the_scene_visualisation_flag(scene):
    assert SceneView(scene, ROLE_LEFT).enable_vis is False
