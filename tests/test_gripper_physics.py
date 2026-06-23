import numpy as np

from franka_sim.gripper_backend import GripperStateData
from franka_sim.gripper_physics import GenesisFrankaHand


class FakeSim:
    """Minimal finger dynamics: each poll moves q toward the per-finger target
    by `step`, but never closes past `object_half` (an object between fingers).
    """

    def __init__(self, object_half=None, step=0.01):
        self.max_finger_width = 0.08
        self.q = np.array([0.04, 0.04])
        self.target = np.array([0.04, 0.04])
        self.object_half = object_half
        self.step = step

    def update_finger_positions(self, positions):
        self.target = np.array(positions)

    def _advance(self):
        nxt = self.q + np.clip(self.target - self.q, -self.step, self.step)
        if self.object_half is not None:
            nxt = np.maximum(nxt, self.object_half)  # blocked by the object
        self.dq = nxt - self.q
        self.q = nxt

    def get_finger_state(self):
        self._advance()
        return {"q": self.q.copy(), "dq": getattr(self, "dq", np.zeros(2))}


class StaleFirstPollSim(FakeSim):
    def __init__(self, object_half=None, step=0.01):
        super().__init__(object_half=object_half, step=step)
        self._stale_polls = 0

    def update_finger_positions(self, positions):
        super().update_finger_positions(positions)
        self._stale_polls = 1

    def get_finger_state(self):
        if self._stale_polls:
            self._stale_polls -= 1
            return {"q": self.q.copy(), "dq": np.zeros(2)}
        return super().get_finger_state()


def _hand(sim):
    return GenesisFrankaHand(sim, settle_timeout=2.0, settle_velocity=1e-3, poll_dt=0.0)


def test_move_opens_to_commanded_width():
    sim = FakeSim()
    hand = _hand(sim)
    assert hand.move(0.06, 0.1) is True
    st = hand.get_state()
    assert abs(st.width - 0.06) < 5e-3
    assert st.is_grasped is False


def test_grasp_on_object_reports_grasped():
    # Real Franka idiom: command a width slightly UNDER the object so the fingers
    # squeeze. Object is 0.03 wide; commanding 0.02 with eps_outer 0.02 gives the
    # band [0.015, 0.04], which contains the 0.03 stall width -> grasped.
    sim = FakeSim(object_half=0.015)  # object 0.03 wide
    hand = _hand(sim)
    ok = hand.grasp(0.02, epsilon_inner=0.005, epsilon_outer=0.02, speed=0.1, force=40)
    assert ok is True
    assert hand.get_state().is_grasped is True
    assert abs(hand.get_state().width - 0.03) < 5e-3


def test_grasp_ignores_stale_settled_snapshot_after_recommand():
    sim = StaleFirstPollSim(object_half=0.015)
    hand = _hand(sim)
    ok = hand.grasp(0.02, epsilon_inner=0.005, epsilon_outer=0.02, speed=0.1, force=40)
    assert ok is True
    assert hand.get_state().is_grasped is True
    assert abs(hand.get_state().width - 0.03) < 5e-3


def test_grasp_without_object_is_unsuccessful():
    # No object: fingers reach the commanded 0.02 without stalling above it -> no grasp.
    sim = FakeSim(object_half=None)
    hand = _hand(sim)
    ok = hand.grasp(0.02, epsilon_inner=0.005, epsilon_outer=0.02, speed=0.1, force=40)
    assert ok is False
    assert hand.get_state().is_grasped is False


def test_grasp_object_outside_epsilon_is_unsuccessful():
    # Object 0.06 wide stalls the fingers at 0.06, outside the band [0.015, 0.04]
    # around the commanded 0.02 (object too big) -> unsuccessful.
    sim = FakeSim(object_half=0.03)  # object 0.06 wide, above commanded 0.02 + eps_outer
    hand = _hand(sim)
    ok = hand.grasp(0.02, epsilon_inner=0.005, epsilon_outer=0.02, speed=0.1, force=40)
    assert ok is False


def test_get_state_returns_gripper_state_data():
    hand = _hand(FakeSim())
    assert isinstance(hand.get_state(), GripperStateData)


def test_homing_opens_fully():
    sim = FakeSim()
    sim.q = np.array([0.0, 0.0])
    hand = _hand(sim)
    assert hand.homing() is True
    assert abs(hand.get_state().width - 0.08) < 5e-3


def test_stop_releases_grasp_and_opens_fully():
    sim = FakeSim(object_half=0.015)
    hand = _hand(sim)
    assert hand.grasp(0.02, epsilon_inner=0.005, epsilon_outer=0.02, speed=0.1, force=40)
    assert hand.stop() is True
    state = hand.get_state()
    assert state.is_grasped is False
    assert abs(state.width - 0.08) < 5e-3
