from franka_sim.gripper.backend import FRANKA_HAND_MAX_WIDTH, FrankaHandSim


def test_homing_opens_to_max_width():
    gripper = FrankaHandSim()
    assert gripper.homing() is True
    state = gripper.get_state()
    assert state.width == FRANKA_HAND_MAX_WIDTH
    assert state.max_width == FRANKA_HAND_MAX_WIDTH
    assert state.is_grasped is False


def test_move_sets_and_clamps_width():
    gripper = FrankaHandSim()
    assert gripper.move(0.03, 0.1) is True
    assert gripper.get_state().width == 0.03
    gripper.move(0.5, 0.1)  # above max -> clamped
    assert gripper.get_state().width == FRANKA_HAND_MAX_WIDTH
    gripper.move(-0.1, 0.1)  # below zero -> clamped
    assert gripper.get_state().width == 0.0


def test_move_clears_is_grasped():
    gripper = FrankaHandSim(object_width=0.03)
    gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0)
    assert gripper.get_state().is_grasped is True
    gripper.move(0.05, 0.1)
    assert gripper.get_state().is_grasped is False


def test_grasp_succeeds_when_object_within_epsilon():
    gripper = FrankaHandSim()
    gripper.set_object_width(0.03)
    assert gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0) is True
    state = gripper.get_state()
    assert state.is_grasped is True
    assert state.width == 0.03


def test_grasp_fails_without_object():
    gripper = FrankaHandSim()
    assert gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0) is False
    assert gripper.get_state().is_grasped is False


def test_grasp_fails_when_object_outside_epsilon():
    gripper = FrankaHandSim()
    gripper.set_object_width(0.05)
    assert gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0) is False
    assert gripper.get_state().is_grasped is False


def test_stop_returns_true():
    assert FrankaHandSim().stop() is True


def test_stop_releases_grasp_and_opens():
    gripper = FrankaHandSim(object_width=0.03)
    assert gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0) is True
    assert gripper.stop() is True
    state = gripper.get_state()
    assert state.is_grasped is False
    assert state.width == FRANKA_HAND_MAX_WIDTH
