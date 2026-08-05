import numpy as np
import pytest
from fakes import FakeEntity

from franka_sim.franka_genesis_sim import ControlMode
from franka_sim.swerve_base import TMR_WHEEL_RADIUS
from franka_sim.tmr_genesis_sim import TMRGenesisSim


@pytest.fixture
def tmr_sim(tmp_path):
    """A TMRGenesisSim wired to a fake entity, with no Genesis scene built."""
    urdf_path = tmp_path / "tmr.urdf"
    urdf_path.write_text('<?xml version="1.0"?><robot name="tmr"></robot>')
    sim = TMRGenesisSim(urdf_path, enable_vis=False, base_height=0.05)
    sim.tmr = FakeEntity()
    sim._bind_entity()
    sim._read_and_publish_state()
    return sim


def test_robot_state_arrays_are_seven_elements(tmr_sim):
    state = tmr_sim.get_robot_state()
    for key in ("q", "dq", "ddq", "q_d", "dq_d", "ddq_d", "tau_J"):
        assert state[key].shape == (7,), key
    assert state["O_T_EE"].shape == (16,)


def test_wheel_state_is_padded_into_the_first_four_joints(tmr_sim):
    tmr_sim.tmr.dof_positions = [0.11, 0.22, 0.33, 0.44]
    tmr_sim.tmr.dof_velocities = [1.1, 2.2, 3.3, 4.4]
    tmr_sim._read_and_publish_state()

    state = tmr_sim.get_robot_state()
    assert state["q"][:4] == pytest.approx([0.11, 0.22, 0.33, 0.44])
    assert state["q"][4:] == pytest.approx([0.0, 0.0, 0.0])
    assert state["dq"][:4] == pytest.approx([1.1, 2.2, 3.3, 4.4])
    assert state["dq"][4:] == pytest.approx([0.0, 0.0, 0.0])


def test_update_base_twist_reaches_the_swerve_solver(tmr_sim):
    tmr_sim.update_base_twist([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    steer, drive = tmr_sim.swerve.solve()
    assert steer == pytest.approx([0.0, 0.0], abs=1e-9)
    assert drive == pytest.approx([1.0 / TMR_WHEEL_RADIUS] * 2, abs=1e-9)


def test_step_once_drives_the_wheels_and_moves_the_base(tmr_sim):
    tmr_sim.set_control_mode(ControlMode.STEERING_DRIVE)
    tmr_sim.update_base_twist([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    tmr_sim._apply_control()

    assert tmr_sim.tmr.velocity_commands[-1][1] == [1, 3]
    assert tmr_sim.tmr.positions[-1] == pytest.approx([tmr_sim.dt, 0.0, 0.05], abs=1e-12)


def test_o_t_ee_tracks_the_integrated_base_pose(tmr_sim):
    tmr_sim.update_base_twist([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(10):
        tmr_sim._apply_control()
    tmr_sim._read_and_publish_state()

    matrix = tmr_sim.get_robot_state()["O_T_EE"].reshape(4, 4).T
    assert matrix[0, 3] == pytest.approx(10 * tmr_sim.dt, abs=1e-12)
    assert matrix[2, 3] == pytest.approx(0.05)


def test_zero_twist_holds_the_base_still(tmr_sim):
    tmr_sim.update_base_twist([0.0] * 6)
    for _ in range(10):
        tmr_sim._apply_control()
    assert tmr_sim.tmr.positions[-1] == pytest.approx([0.0, 0.0, 0.05])


def test_joint_space_updates_are_accepted_and_ignored(tmr_sim):
    """These are called by FrankaSimServer on the arm path; they must never raise."""
    tmr_sim.update_joint_positions(np.zeros(7))
    tmr_sim.update_joint_velocities(np.zeros(7))
    tmr_sim.update_torques(np.zeros(7))
    assert tmr_sim.get_robot_state()["tau_J"] == pytest.approx(np.zeros(7))


def test_set_control_mode_rejects_non_enum(tmr_sim):
    with pytest.raises(ValueError):
        tmr_sim.set_control_mode("steering_drive")
