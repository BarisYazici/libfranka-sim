import pytest

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
    # The closing fingers stall on the object at 0.03, inside the band around
    # the commanded 0.03 -> grasped.
    assert gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0) is True
    state = gripper.get_state()
    assert state.is_grasped is True
    assert state.width == 0.03


def test_grasp_in_free_space_closes_fully_and_fails():
    """Nothing between the fingers: they close to ~0 and the grasp returns False.

    The real hand closes under force until it stalls -- on nothing, that is the
    fingers meeting each other -- and only then applies the band. 0 is nowhere
    near the commanded 0.04, so ``franka::Gripper::grasp`` returns false ("True
    if an object has been grasped, false otherwise"), which is exactly what
    libfranka's ``examples/grasp_object.cpp`` branches on.
    """
    gripper = FrankaHandSim()
    assert gripper.grasp(0.04, 0.005, 0.005, 0.1, 60.0) is False
    state = gripper.get_state()
    assert state.is_grasped is False
    assert state.width == 0.0


def test_grasp_of_an_object_at_the_commanded_width_succeeds():
    """The band is inclusive of its centre: object width == commanded width."""
    gripper = FrankaHandSim(object_width=0.04)
    assert gripper.grasp(0.04, 0.005, 0.005, 0.1, 60.0) is True
    assert gripper.get_state().width == 0.04


def test_grasp_of_an_object_narrower_than_epsilon_inner_fails():
    """A 0.03 object under a 0.04 command is 0.01 too small for eps_inner 0.005."""
    gripper = FrankaHandSim(object_width=0.03)
    assert gripper.grasp(0.04, 0.005, 0.005, 0.1, 60.0) is False
    assert gripper.get_state().is_grasped is False
    # The same object inside a wider inner tolerance is a grasp.
    gripper = FrankaHandSim(object_width=0.03)
    assert gripper.grasp(0.04, 0.02, 0.005, 0.1, 60.0) is True
    assert gripper.get_state().width == 0.03


def test_grasp_beyond_the_stroke_raises():
    """An unreachable width is an error, not a missed grasp -> kFail on the wire."""
    gripper = FrankaHandSim()
    with pytest.raises(ValueError):
        gripper.grasp(0.09, 0.005, 0.005, 0.1, 60.0)
    with pytest.raises(ValueError):
        gripper.grasp(-0.01, 0.005, 0.005, 0.1, 60.0)
    # The rejected command left the fingers where they were.
    assert gripper.get_state().width == FRANKA_HAND_MAX_WIDTH


def test_grasp_fails_when_object_outside_epsilon():
    gripper = FrankaHandSim()
    gripper.set_object_width(0.05)
    # The 0.05 object stalls the fingers well outside the band around 0.03.
    assert gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0) is False
    state = gripper.get_state()
    assert state.is_grasped is False
    assert state.width == 0.05


def test_grasp_does_not_stall_on_an_object_wider_than_the_current_opening():
    """A stale ``width`` must not fake a grasp of an object the fingers are inside.

    With the fingers already at 0.02 a 0.05-wide object cannot be between them,
    so the grasp closes them the rest of the way instead of reporting a hold at
    a width narrower than the object.
    """
    gripper = FrankaHandSim(object_width=0.05)
    gripper.move(0.02, 0.1)
    assert gripper.grasp(0.02, 0.005, 0.005, 0.1, 60.0) is False
    assert gripper.get_state().width == 0.0


def test_stop_returns_true():
    assert FrankaHandSim().stop() is True


def test_stop_releases_grasp_and_opens():
    gripper = FrankaHandSim(object_width=0.03)
    assert gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0) is True
    assert gripper.stop() is True
    state = gripper.get_state()
    assert state.is_grasped is False
    assert state.width == FRANKA_HAND_MAX_WIDTH
