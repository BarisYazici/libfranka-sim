import time

import genesis as gs
import numpy as np
import pytest

from franka_sim.franka_genesis_sim import ControlMode, FrankaGenesisSim


@pytest.fixture(scope="session", autouse=True)
def genesis_init():
    """Initialize Genesis once for all tests"""
    try:
        gs.init(backend=gs.gpu, logging_level=None)
    except gs.GenesisException as e:
        if "Genesis already initialized" not in str(e):
            raise e
    yield
    # No cleanup needed as Genesis doesn't have a cleanup method


@pytest.fixture
def genesis_sim():
    """Fixture providing a Genesis simulator instance"""
    sim = FrankaGenesisSim(enable_vis=False)
    try:
        sim.initialize_simulation()
    except gs.GenesisException as e:
        if "Genesis already initialized" in str(e):
            # If Genesis is already initialized, set up the rest of the simulator
            sim.scene = gs.Scene(
                sim_options=gs.options.SimOptions(dt=sim.dt),
                show_viewer=sim.enable_vis,
                show_FPS=False,
            )
            # Add plane and continue with the rest of initialization
            sim.scene.add_entity(gs.morphs.Plane())
            sim.franka = sim.scene.add_entity(
                gs.morphs.MJCF(file=str(sim.xml_path)),
                material=gs.materials.Rigid(gravity_compensation=1.0),
            )
            sim.scene.build()
            sim.model, sim.data = sim.load_panda_model()
            sim.jnt_names = [
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
                "joint6",
                "joint7",
                "finger_joint1",
                "finger_joint2",
            ]
            sim.dofs_idx = [sim.franka.get_joint(name).dof_idx_local for name in sim.jnt_names]
            initial_q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
            with sim.joint_position_lock:
                sim.latest_joint_positions = initial_q.copy()
            for _ in range(100):
                sim.franka.set_dofs_position(
                    np.concatenate([initial_q, [0.04, 0.04]]), sim.dofs_idx
                )
                sim.scene.step()
        else:
            raise e
    yield sim
    sim.stop()


def test_simulator_initialization(genesis_sim):
    """Test proper initialization of Genesis simulator"""
    # Check if simulator components are initialized
    assert genesis_sim.scene is not None
    assert genesis_sim.franka is not None
    assert genesis_sim.model is not None
    assert genesis_sim.data is not None

    # Check initial joint positions
    q = genesis_sim.franka.get_dofs_position(genesis_sim.dofs_idx).cpu().numpy()
    assert len(q) == 9  # 7 joints + 2 fingers
    assert np.allclose(q[-2:], [0.04, 0.04])  # Check finger positions


def test_position_control(genesis_sim):
    """Test position control mode"""
    # Set position control mode
    genesis_sim.set_control_mode(ControlMode.POSITION)

    # Set target joint positions
    target_positions = np.array([0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4])
    genesis_sim.update_joint_positions(target_positions)

    # Start simulation
    genesis_sim.running = True

    # Run for a few steps
    for _ in range(100):
        genesis_sim.run_simulation()

    # Get final positions
    state = genesis_sim.get_robot_state()
    assert np.allclose(state["q"], target_positions, atol=0.1)


def test_velocity_control(genesis_sim):
    """Test velocity control mode"""
    # Set velocity control mode
    genesis_sim.set_control_mode(ControlMode.VELOCITY)

    # Set target joint velocities
    target_velocities = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1])
    genesis_sim.update_joint_velocities(target_velocities)

    # Start simulation
    genesis_sim.running = True

    # Get initial positions
    initial_state = genesis_sim.get_robot_state()
    initial_positions = initial_state["q"]

    # Run for a short time
    for _ in range(10):
        genesis_sim.run_simulation()

    # Get final positions
    final_state = genesis_sim.get_robot_state()

    # Check that joints have moved in the correct direction
    position_changes = final_state["q"] - initial_positions
    assert np.all(np.sign(position_changes) == np.sign(target_velocities))


def test_torque_control(genesis_sim):
    """Test torque control mode"""
    # Set torque control mode
    genesis_sim.set_control_mode(ControlMode.TORQUE)

    # Set target torques
    target_torques = np.array([1.0, -1.0, 1.0, -1.0, 0.5, -0.5, 0.5])
    genesis_sim.update_torques(target_torques)

    # Start simulation
    genesis_sim.running = True

    # Run for a few steps
    for _ in range(10):
        genesis_sim.run_simulation()

    # Get state and verify torques
    state = genesis_sim.get_robot_state()
    assert np.allclose(state["tau_J"], target_torques, atol=0.1)


def test_control_mode_switching(genesis_sim):
    """Test switching between different control modes"""
    # Start with position control
    genesis_sim.set_control_mode(ControlMode.POSITION)
    assert genesis_sim.control_mode == ControlMode.POSITION

    # Switch to velocity control
    genesis_sim.set_control_mode(ControlMode.VELOCITY)
    assert genesis_sim.control_mode == ControlMode.VELOCITY

    # Switch to torque control
    genesis_sim.set_control_mode(ControlMode.TORQUE)
    assert genesis_sim.control_mode == ControlMode.TORQUE

    # Test invalid mode
    with pytest.raises(ValueError):
        genesis_sim.set_control_mode("invalid_mode")


def test_robot_state_consistency(genesis_sim):
    """Test consistency of robot state updates"""
    # Start simulation
    genesis_sim.running = True

    # Get initial state
    initial_state = genesis_sim.get_robot_state()

    # Run simulation for a few steps
    for _ in range(10):
        genesis_sim.run_simulation()

    # Get updated state
    updated_state = genesis_sim.get_robot_state()

    # Check state structure consistency
    assert set(initial_state.keys()) == set(updated_state.keys())
    assert len(initial_state["q"]) == len(updated_state["q"]) == 7
    assert len(initial_state["dq"]) == len(updated_state["dq"]) == 7
    assert len(initial_state["tau_J"]) == len(updated_state["tau_J"]) == 7
