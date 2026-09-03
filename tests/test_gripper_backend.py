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
    # The object stalls the fingers at 0.03, inside the band around the
    # commanded 0.03 -> grasped.
    assert gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0) is True
    state = gripper.get_state()
    assert state.is_grasped is True
    assert state.width == 0.03


def test_grasp_in_free_space_succeeds_at_the_commanded_width():
    """Nothing between the fingers is not a failure -- the robot succeeds here.

    ``franka::Gripper::grasp`` (``include/franka/gripper.h``) defines success
    purely as the final finger distance landing in ``(width - epsilon_inner,
    width + epsilon_outer)``. With nothing in the way the fingers reach the
    commanded width exactly, which is the dead centre of that band.
    """
    gripper = FrankaHandSim()
    assert gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0) is True
    state = gripper.get_state()
    assert state.is_grasped is True
    assert state.width == 0.03


def test_grasp_fails_when_the_commanded_width_is_beyond_the_stroke():
    """A width the hand cannot reach fails: the band is around the *request*.

    The fingers stop at the 0.08 m stroke limit, which is outside the band
    (0.085, 0.095) around the commanded 0.09 -- so the grasp did not succeed,
    even though the fingers went exactly as far as they could.
    """
    gripper = FrankaHandSim()
    assert gripper.grasp(0.09, 0.005, 0.005, 0.1, 60.0) is False
    state = gripper.get_state()
    assert state.is_grasped is False
    assert state.width == FRANKA_HAND_MAX_WIDTH


def test_grasp_fails_when_object_outside_epsilon():
    gripper = FrankaHandSim()
    gripper.set_object_width(0.05)
    # The 0.05 object stalls the fingers well outside the band around 0.03.
    assert gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0) is False
    state = gripper.get_state()
    assert state.is_grasped is False
    assert state.width == 0.05


def test_stop_returns_true():
    assert FrankaHandSim().stop() is True


def test_stop_releases_grasp_and_opens():
    gripper = FrankaHandSim(object_width=0.03)
    assert gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0) is True
    assert gripper.stop() is True
    state = gripper.get_state()
    assert state.is_grasped is False
    assert state.width == FRANKA_HAND_MAX_WIDTH
