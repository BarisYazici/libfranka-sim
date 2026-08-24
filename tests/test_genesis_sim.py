import genesis as gs
import numpy as np
import pytest

from franka_sim.franka_genesis_sim import ControlMode, FrankaGenesisSim

# These tests exercise the real Genesis physics engine. When Genesis is not
# installed it is stubbed with a mock (see tests/conftest.py), which has no real
# module file -- in that case skip the whole module rather than erroring.
if not isinstance(getattr(gs, "__file__", None), str):
    pytest.skip("Genesis is not installed; skipping simulator tests.", allow_module_level=True)


@pytest.fixture(scope="session", autouse=True)
def genesis_init():
    """Initialize Genesis once for all tests"""
    try:
        # Always use CPU backend for tests
        print("Using CPU backend for Genesis tests")
        gs.init(backend=gs.cpu, logging_level=None)
    except gs.GenesisException as e:
        if "Genesis already initialized" not in str(e):
            # Only raise if not already initialized
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
            # FR3: 7-DOF, hand-less, joints fr3_joint1..7, EE link fr3_link7.
            sim.jnt_names = [f"fr3_joint{i}" for i in range(1, 8)]
            sim.dofs_idx = [sim.franka.get_joint(name).dof_idx_local for name in sim.jnt_names]
            sim.hand_link = sim.franka.get_link("fr3_link7")
            initial_q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
            sim.latest_joint_positions = initial_q.copy()
            for _ in range(100):
                sim.franka.set_dofs_position(initial_q, sim.dofs_idx)
                sim.scene.step()
            sim._read_and_publish_state()
        else:
            raise e
    yield sim
    sim.stop()


def test_simulator_initialization(genesis_sim):
    """Test proper initialization of Genesis simulator"""
    # Check if simulator components are initialized
    assert genesis_sim.scene is not None
    assert genesis_sim.franka is not None

    # Check initial joint positions (FR3: 7-DOF, hand-less -- no finger joints).
    q = genesis_sim.franka.get_dofs_position(genesis_sim.dofs_idx).cpu().numpy()
    assert len(q) == 7
    assert np.allclose(q, [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785], atol=0.05)


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
