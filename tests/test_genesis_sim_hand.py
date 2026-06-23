import numpy as np

from franka_sim.franka_genesis_sim import FrankaGenesisSim


def test_default_is_hand_less_7dof():
    sim = FrankaGenesisSim()
    assert sim.enable_hand is False
    assert sim.finger_dofs_idx is None
    fs = sim.get_finger_state()
    assert np.array_equal(fs["q"], np.zeros(2))


def test_update_finger_positions_is_lock_free_swap():
    sim = FrankaGenesisSim(enable_hand=True)
    sim.update_finger_positions([0.01, 0.01])
    assert np.array_equal(sim.latest_finger_positions, np.array([0.01, 0.01]))
    # New array bound, not mutated in place.
    before = sim.latest_finger_positions
    sim.update_finger_positions([0.02, 0.02])
    assert before is not sim.latest_finger_positions


def test_finger_snapshot_shape_when_hand_enabled():
    sim = FrankaGenesisSim(enable_hand=True)
    fs = sim.get_finger_state()
    assert fs["q"].shape == (2,)
    assert fs["dq"].shape == (2,)
