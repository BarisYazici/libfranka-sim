"""Physics-level tests for the MuJoCo single-arm backend (the default engine).

Nothing here is faked: MuJoCo compiles the Menagerie FR3 in a fraction of a
second and steps it at ~40x real time, so every assertion is made against the
engine that ships. The whole module is skipped when neither ``$FR3_MJCF`` nor
the ``robot_descriptions`` cache can produce the model.
"""

import threading
import time

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from franka_sim.control_modes import ControlMode  # noqa: E402
from franka_sim.mujoco_franka_sim import (  # noqa: E402
    ARM_INITIAL_Q,
    DEFAULT_DT,
    FLANGE_OFFSET_Z,
    MAX_FINGER_TRAVEL,
    MujocoFrankaSim,
    default_fr3_mjcf,
)
from franka_sim.sim_common import FR3_FORCE_LIMITS  # noqa: E402

try:
    FR3_MJCF = default_fr3_mjcf()
except Exception:  # pragma: no cover - depends on the host's cache/network
    FR3_MJCF = None

pytestmark = pytest.mark.skipif(
    FR3_MJCF is None or not FR3_MJCF.exists(),
    reason="the MuJoCo Menagerie FR3 model is neither cached nor downloadable",
)

#: Simulated seconds the servo tests settle for; long enough for the slowest
#: joint's PD response, short enough to stay well under a second of wall clock.
SETTLE_S = 2.0
SETTLE_STEPS = int(SETTLE_S / DEFAULT_DT)


@pytest.fixture
def sim():
    """A built, hand-less 7-DOF arm holding its initial pose."""
    simulator = MujocoFrankaSim()
    simulator.initialize_simulation()
    yield simulator
    simulator.stop()


@pytest.fixture
def hand_sim():
    """A built 9-DOF arm + Franka Hand, fingers open."""
    simulator = MujocoFrankaSim(enable_hand=True)
    simulator.initialize_simulation()
    yield simulator
    simulator.stop()


def _body_pose(sim, name):
    """Position and 3x3 rotation of one body in the world frame."""
    body_id = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert body_id >= 0, f"body {name!r} missing"
    return sim.data.xpos[body_id].copy(), sim.data.xmat[body_id].reshape(3, 3).copy()


# --- model -------------------------------------------------------------------


def test_model_loads_with_seven_arm_dofs(sim):
    assert sim.model.nq == 7
    assert sim.model.nv == 7
    assert sim.arm_qpos_adr.shape == (7,)
    assert sim.arm_dofs_idx.shape == (7,)
    assert sim.dt == DEFAULT_DT
    assert sim.model.opt.timestep == pytest.approx(DEFAULT_DT)


def test_gravity_compensation_is_compiled_in(sim):
    """Gravity compensation only runs when ngravcomp is non-zero at compile time."""
    assert sim.model.ngravcomp > 0
    assert sim.data.qfrc_gravcomp == pytest.approx(sim.data.qfrc_bias, abs=1e-9)


def test_contacts_stay_enabled_for_the_single_arm(sim):
    """Grasping needs contacts; only the mobile-duo URDF has to disable them."""
    assert not sim.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT


def test_the_arm_holds_its_initial_pose_after_the_build_settle(sim):
    assert sim.get_robot_state()["q"] == pytest.approx(ARM_INITIAL_Q, abs=1e-3)


def test_joint_damping_override_is_honored(monkeypatch):
    monkeypatch.setenv("FR3_JOINT_DAMPING", "3.5")
    simulator = MujocoFrankaSim()
    simulator.initialize_simulation()
    try:
        assert simulator.model.dof_damping[simulator.arm_dofs_idx] == pytest.approx([3.5] * 7)
    finally:
        simulator.stop()


def test_menagerie_joint_defaults_survive_without_the_override(sim):
    """Unset, the Menagerie's own calibrated damping/armature/friction stay put."""
    dofs = sim.arm_dofs_idx
    assert sim.model.dof_damping[dofs] == pytest.approx([0.003] * 7)
    assert sim.model.dof_armature[dofs] == pytest.approx([0.195] * 7)
    assert sim.model.dof_frictionloss[dofs] == pytest.approx([0.2] * 7)


# --- control modes -----------------------------------------------------------


def test_set_control_mode_rejects_a_non_control_mode(sim):
    with pytest.raises(ValueError):
        sim.set_control_mode("position")


def test_position_servo_converges_on_the_commanded_target(sim):
    target = ARM_INITIAL_Q + np.array([0.2, -0.1, 0.15, 0.1, -0.2, 0.1, 0.3])
    sim.set_control_mode(ControlMode.POSITION)
    sim.update_joint_positions(target)
    sim.step(SETTLE_STEPS)

    state = sim.get_robot_state()
    assert state["q"] == pytest.approx(target, abs=1e-3)
    assert state["dq"] == pytest.approx(np.zeros(7), abs=1e-3)
    assert state["q_d"] == pytest.approx(target)


def test_velocity_mode_tracks_the_commanded_velocity(sim):
    commanded = np.array([0.3, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3])
    sim.set_control_mode(ControlMode.VELOCITY)
    sim.update_joint_velocities(commanded)
    sim.step(500)

    dq = sim.get_robot_state()["dq"]
    assert dq[0] == pytest.approx(0.3, rel=0.1)
    assert dq[6] == pytest.approx(-0.3, rel=0.1)
    assert np.sign(dq[0]) == 1 and np.sign(dq[6]) == -1


def test_torque_mode_applies_the_commanded_torque(sim):
    sim.set_control_mode(ControlMode.TORQUE)
    sim.update_torques(np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    start = sim.get_robot_state()["q"][0]
    sim.step(200)

    state = sim.get_robot_state()
    assert state["q"][0] > start
    assert state["dq"][0] > 0.1
    assert state["tau_J"][0] == pytest.approx(5.0)


def test_zero_torque_leaves_the_gravity_compensated_arm_where_it_was(sim):
    sim.set_control_mode(ControlMode.TORQUE)
    sim.update_torques(np.zeros(7))
    sim.step(500)
    assert sim.get_robot_state()["q"] == pytest.approx(ARM_INITIAL_Q, abs=1e-3)


def test_servo_torque_is_clamped_to_the_fr3_actuator_limits(sim):
    sim.set_control_mode(ControlMode.POSITION)
    sim.update_joint_positions(ARM_INITIAL_Q + 10.0)
    assert sim.arm_control_torque() == pytest.approx(FR3_FORCE_LIMITS)

    sim.update_joint_positions(ARM_INITIAL_Q - 10.0)
    assert sim.arm_control_torque() == pytest.approx(-FR3_FORCE_LIMITS)


# --- reported state ----------------------------------------------------------


def test_get_robot_state_has_the_full_key_set_and_shapes(sim):
    state = sim.get_robot_state()
    for key in ("q", "dq", "ddq", "q_d", "dq_d", "ddq_d", "tau_J"):
        assert state[key].shape == (7,), key
    assert state["O_T_EE"].shape == (16,)


def test_o_t_ee_is_a_valid_column_major_se3_at_the_flange(sim):
    sim.step(10)
    state = sim.get_robot_state()
    # Column-major 16-vector -> the 4x4 it packs.
    transform = state["O_T_EE"].reshape(4, 4).T
    rotation = transform[:3, :3]

    assert transform[3, :] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0)

    # Mirrors FrankaGenesisSim's frame choice: link7, the flange the labs'
    # F_T_EE=identity assumption is measured from.
    position, orientation = _body_pose(sim, f"{sim.prefix}link7")
    assert transform[:3, 3] == pytest.approx(position)
    assert rotation == pytest.approx(orientation, abs=1e-9)


def test_state_snapshot_is_swapped_not_mutated(sim):
    """The network threads keep a reference; stepping must not change it."""
    before = sim.get_robot_state()
    q_before = before["q"].copy()
    sim.set_control_mode(ControlMode.VELOCITY)
    sim.update_joint_velocities(np.full(7, 0.2))
    sim.step(100)

    assert before["q"] == pytest.approx(q_before)
    assert sim.get_robot_state() is not before


# --- hand --------------------------------------------------------------------


def test_hand_attach_gives_nine_dofs(hand_sim):
    assert hand_sim.model.nq == 9
    assert hand_sim.model.nv == 9
    assert hand_sim.finger_qpos_adr.shape == (2,)
    assert hand_sim.finger_dofs_idx.shape == (2,)
    for name in ("fr3v2_finger_joint1", "fr3v2_finger_joint2"):
        assert mujoco.mj_name2id(hand_sim.model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0


def test_hand_sits_on_the_flange_at_the_franka_wrist_transform(hand_sim):
    """0.107 m along link7's z, rotated -45 deg about it (the real wrist mount)."""
    link7_pos, link7_rot = _body_pose(hand_sim, "fr3v2_link7")
    hand_pos, hand_rot = _body_pose(hand_sim, "fr3v2_hand")

    assert link7_rot.T @ (hand_pos - link7_pos) == pytest.approx(
        [0.0, 0.0, FLANGE_OFFSET_Z], abs=1e-9
    )
    half = np.sqrt(0.5)
    expected = np.array([[half, half, 0.0], [-half, half, 0.0], [0.0, 0.0, 1.0]])
    assert link7_rot.T @ hand_rot == pytest.approx(expected, abs=1e-6)


def test_update_finger_positions_moves_the_fingers(hand_sim):
    assert hand_sim.get_finger_state()["q"] == pytest.approx([MAX_FINGER_TRAVEL] * 2)

    hand_sim.update_finger_positions([0.0, 0.0])
    hand_sim.step(500)
    assert hand_sim.get_finger_state()["q"] == pytest.approx([0.0, 0.0], abs=1e-4)

    hand_sim.update_finger_positions([0.02, 0.02])
    hand_sim.step(500)
    assert hand_sim.get_finger_state()["q"] == pytest.approx([0.02, 0.02], abs=1e-4)


def test_get_finger_state_shapes(hand_sim):
    hand_sim.step(1)
    state = hand_sim.get_finger_state()
    assert state["q"].shape == (2,)
    assert state["dq"].shape == (2,)


def test_the_arm_is_undisturbed_by_the_grafted_hand(hand_sim):
    hand_sim.update_finger_positions([0.0, 0.0])
    hand_sim.step(500)
    assert hand_sim.get_robot_state()["q"] == pytest.approx(ARM_INITIAL_Q, abs=1e-3)


# --- the physics gripper backend on top of it --------------------------------


def _wait_until(pred, timeout=10.0, dt=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(dt)
    return False


@pytest.fixture
def running_hand_sim(hand_sim):
    """The 9-DOF sim stepping in its own thread, as the server runs it."""
    hand_sim.running = True
    thread = threading.Thread(target=hand_sim.run_simulation, daemon=True)
    thread.start()
    yield hand_sim
    hand_sim.stop()
    thread.join(timeout=2.0)


def test_genesis_franka_hand_drives_the_mujoco_fingers(running_hand_sim):
    """The GenesisFrankaHand backend is sim-agnostic: it works unchanged here."""
    from franka_sim.gripper_physics import GenesisFrankaHand

    backend = GenesisFrankaHand(running_hand_sim)

    assert backend.homing() is True
    assert backend.get_state().width == pytest.approx(0.08, abs=2e-3)

    assert backend.move(0.02, 0.1) is True
    assert backend.get_state().width == pytest.approx(0.02, abs=2e-3)
    assert backend.is_stuck is False


def test_wire_client_move_drives_the_mujoco_fingers(running_hand_sim):
    """A libfranka-style wire client opens/closes the fingers and sees it over UDP."""
    from examples.gripper.gripper_wire_client import GripperWireClient
    from franka_sim.gripper_physics import GenesisFrankaHand
    from franka_sim.gripper_server import FrankaGripperServer

    server = FrankaGripperServer(
        host="127.0.0.1", port=0, backend=GenesisFrankaHand(running_hand_sim)
    )
    thread = threading.Thread(target=server.run_server, daemon=True)
    thread.start()
    assert _wait_until(lambda: server.running and server.server_socket is not None)

    client = GripperWireClient(port=server.server_socket.getsockname()[1])
    client.connect()
    try:
        client.homing()
        assert client.move(0.08, 0.1).name == "kSuccess"
        assert _wait_until(lambda: client.read_state() and client.read_state().width > 0.06)
        assert client.move(0.0, 0.1).name == "kSuccess"
        assert _wait_until(lambda: client.read_state() and client.read_state().width < 0.02)
    finally:
        client.close()
        server.stop()
        thread.join(timeout=2.0)


# --- server wiring -----------------------------------------------------------


def test_the_server_builds_this_backend_by_default():
    from franka_sim.franka_sim_server import DEFAULT_PHYSICS, resolve_sim_class

    assert DEFAULT_PHYSICS == "mujoco"
    assert resolve_sim_class() is MujocoFrankaSim
    assert resolve_sim_class("mujoco") is MujocoFrankaSim


def test_the_genesis_backend_is_still_selectable():
    """--physics genesis must keep working -- it is optional, not removed."""
    from franka_sim.franka_sim_server import resolve_sim_class

    assert resolve_sim_class("genesis").__name__ == "FrankaGenesisSim"


def test_resolve_sim_class_rejects_an_unknown_backend():
    from franka_sim.franka_sim_server import resolve_sim_class

    with pytest.raises(ValueError, match="bullet"):
        resolve_sim_class("bullet")
