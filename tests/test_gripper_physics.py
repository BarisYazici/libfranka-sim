import time

import numpy as np
import pytest

from franka_sim.gripper.backend import FrankaHandSim, GripperStateData
from franka_sim.gripper.physics import FrankaHandPhysics


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


class ScriptedFingerSim:
    """Replays a recorded finger trace: one (width, speed) sample per poll.

    Closing fingers do not decelerate monotonically -- contact settling in
    MuJoCo puts near-zero-velocity samples in the middle of the travel -- so a
    trace is the only way to pin down which sample a settle detector picks.
    The last sample repeats forever, which is the rest the fingers come to.
    """

    def __init__(self, trace):
        self.trace = list(trace)
        self.index = 0
        self.commanded = None

    def update_finger_positions(self, positions):
        self.commanded = float(positions[0] + positions[1])

    def get_finger_state(self):
        width, speed = self.trace[min(self.index, len(self.trace) - 1)]
        self.index += 1
        return {"q": np.array([width / 2, width / 2]), "dq": np.array([speed, speed])}


def _hand(sim, **kwargs):
    kwargs.setdefault("settle_timeout", 2.0)
    kwargs.setdefault("settle_velocity", 1e-3)
    kwargs.setdefault("poll_dt", 0.0)
    return FrankaHandPhysics(sim, **kwargs)


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


def test_grasp_in_free_space_closes_fully_and_fails():
    """Nothing in the way: the fingers close on each other and nothing is grasped.

    The real hand closes under force until it stalls and only then applies
    ``franka::Gripper::grasp``'s band ``(width - epsilon_inner, width +
    epsilon_outer)``. With nothing to stall on, the stall width is ~0, far
    outside the band around the commanded 0.04 -> false, per gripper.h's "True
    if an object has been grasped, false otherwise".
    """
    sim = FakeSim(object_half=None)
    hand = _hand(sim)
    ok = hand.grasp(0.04, epsilon_inner=0.005, epsilon_outer=0.005, speed=0.1, force=40)
    assert ok is False
    assert hand.get_state().is_grasped is False
    assert hand.get_state().width < 5e-3


def test_grasp_of_an_object_at_the_commanded_width_succeeds():
    """Object exactly as wide as the command: the centre of the band, a grasp.

    The pre-existing stall heuristic required the stall to be at least 1e-4
    *wider* than the commanded width, so this -- the most natural way to
    command a grasp -- failed. It must not.
    """
    sim = FakeSim(object_half=0.02)  # object 0.04 wide
    hand = _hand(sim)
    ok = hand.grasp(0.04, epsilon_inner=0.005, epsilon_outer=0.005, speed=0.1, force=40)
    assert ok is True
    assert hand.get_state().is_grasped is True
    assert abs(hand.get_state().width - 0.04) < 5e-3


def test_grasp_of_an_object_narrower_than_epsilon_inner_fails():
    """A 0.03 object under a 0.04 command is 0.01 too small for eps_inner 0.005."""
    hand = _hand(FakeSim(object_half=0.015))  # object 0.03 wide
    assert hand.grasp(0.04, epsilon_inner=0.005, epsilon_outer=0.005, speed=0.1, force=40) is False
    # The same object inside a wider inner tolerance is a grasp.
    hand = _hand(FakeSim(object_half=0.015))
    assert hand.grasp(0.04, epsilon_inner=0.02, epsilon_outer=0.005, speed=0.1, force=40) is True


def test_grasp_object_outside_epsilon_is_unsuccessful():
    # Object 0.06 wide stalls the fingers at 0.06, outside the band (0.015, 0.04)
    # around the commanded 0.02 (object too big) -> unsuccessful.
    sim = FakeSim(object_half=0.03)  # object 0.06 wide, above commanded 0.02 + eps_outer
    hand = _hand(sim)
    ok = hand.grasp(0.02, epsilon_inner=0.005, epsilon_outer=0.02, speed=0.1, force=40)
    assert ok is False
    assert hand.get_state().is_grasped is False


def test_grasp_beyond_the_stroke_raises():
    """An unreachable width is an error the server answers with kFail.

    libfranka maps kUnsuccessful -> a quiet false and kFail -> CommandException
    (``src/gripper.cpp``); a command outside the 0..0.08 m stroke belongs in
    the second bucket. This is the franky-suite case
    (``test_gripper_grasp_failure``).
    """
    hand = _hand(FakeSim(object_half=None))
    with pytest.raises(ValueError):
        hand.grasp(0.09, epsilon_inner=0.005, epsilon_outer=0.005, speed=0.1, force=40)


def test_a_second_grasp_on_a_held_object_settles_without_waiting_out_the_timeout():
    """Fingers already stalled on the object neither move nor reach the target."""
    sim = FakeSim(object_half=0.02)  # object 0.04 wide
    hand = _hand(sim)
    assert hand.grasp(0.04, epsilon_inner=0.005, epsilon_outer=0.005, speed=0.1, force=40) is True
    started = time.monotonic()
    assert hand.grasp(0.04, epsilon_inner=0.005, epsilon_outer=0.005, speed=0.1, force=40) is True
    assert time.monotonic() - started < hand.settle_timeout / 2
    assert hand.is_stuck is False


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


def test_grasp_waits_out_a_mid_travel_quiet_poll():
    """One near-zero-velocity sample is not a stall; ``stall_polls`` of them are.

    The settle test used to accept the first quiet poll once the fingers had
    "moved", which any sample past the start satisfies. Fingers closing onto an
    object pass through momentary near-zero velocities before they rest, so the
    grasp read a width the fingers were still travelling through: this trace
    comes to rest at 0.028 m but goes quiet once at 0.050 m on the way, and the
    band around 0.028 m rejected that sample.
    """
    trace = [
        (0.080, 0.0),  # stale first snapshot, from the previous settled target
        (0.062, 0.9),
        (0.050, 0.0),  # momentary contact-settling quiet -- not the stall
        (0.041, 0.7),
        (0.030, 0.3),
        (0.028, 0.0),  # the real rest
    ]
    hand = _hand(ScriptedFingerSim(trace))
    assert hand.grasp(0.028, epsilon_inner=0.002, epsilon_outer=0.002, speed=0.1, force=40) is True
    assert hand.get_state().width == pytest.approx(0.028)
    assert hand.is_grasped is True


def test_object_width_stalls_the_fingers_where_a_body_would():
    """``--gripper-object-width`` is a rigid obstacle for the physics backend too.

    The ``--gripper-physics`` scene holds nothing graspable, so without this
    every grasp closes to ~0 and answers false. The flag clamps the drive
    target, which is what a body between the fingers does to them.
    """
    sim = FakeSim(object_half=None)  # nothing in the scene
    empty = _hand(sim)
    assert empty.grasp(0.03, 0.005, 0.005, 0.1, 40.0) is False

    hand = _hand(FakeSim(object_half=None), object_width=0.03)
    assert hand.grasp(0.03, 0.005, 0.005, 0.1, 40.0) is True
    assert hand.get_state().width == pytest.approx(0.03, abs=5e-3)
    assert hand.is_grasped is True
    # A body stops an inward move as well, and never an opening one.
    assert hand.move(0.01, 0.1) is True
    assert hand.get_state().width == pytest.approx(0.03, abs=5e-3)
    assert hand.move(0.08, 0.1) is True
    assert hand.get_state().width == pytest.approx(0.08, abs=5e-3)


def test_move_beyond_the_stroke_raises():
    """Same kFail contract as ``grasp``; see :func:`validate_width`."""
    hand = _hand(FakeSim())
    with pytest.raises(ValueError):
        hand.move(0.09, 0.1)
    with pytest.raises(ValueError):
        hand.move(-0.01, 0.1)


@pytest.mark.parametrize(
    "object_width, width, eps_in, eps_out, expected",
    [
        (None, 0.04, 0.005, 0.005, False),  # free space: fingers close on each other
        (0.04, 0.04, 0.005, 0.005, True),  # object exactly at the commanded width
        (0.03, 0.04, 0.005, 0.005, False),  # 0.01 too small for eps_inner 0.005
        (0.03, 0.04, 0.02, 0.005, True),  # ... but inside eps_inner 0.02
        (0.06, 0.02, 0.005, 0.02, False),  # object too big for eps_outer
        # An object exactly as wide as the current (fully open) stroke: the
        # fingers rest on it, they are not inside it, so it stalls them.
        (0.08, 0.08, 0.005, 0.005, True),
    ],
)
def test_stub_and_physics_backends_agree(object_width, width, eps_in, eps_out, expected):
    """Same scenario, same answer: the kinematic stub is a faithful stand-in.

    CI runs the stub; the viewer runs the physics backend. A client must not
    have to know which one it is talking to.
    """
    stub = FrankaHandSim(object_width=object_width)
    physics = _hand(FakeSim(object_half=None if object_width is None else object_width / 2))
    assert stub.grasp(width, eps_in, eps_out, 0.1, 40.0) is expected
    assert physics.grasp(width, eps_in, eps_out, 0.1, 40.0) is expected
    assert stub.get_state().is_grasped is physics.get_state().is_grasped
    assert abs(stub.get_state().width - physics.get_state().width) < 5e-3


def test_stub_and_physics_backends_agree_on_a_repeated_grasp():
    """Grasping twice on a held 0.04 object: True both times, still 0.04 wide.

    The stub compared the object with ``<`` and so released it on the second
    grasp while the physics backend, whose fingers are blocked, kept holding
    it. The virtual object of ``--gripper-object-width`` has to behave the same
    way on both, since that is the flag CI and the viewer share.
    """
    stub = FrankaHandSim(object_width=0.04)
    physics = _hand(FakeSim(object_half=0.02))
    virtual = _hand(FakeSim(object_half=None), object_width=0.04)
    for _ in range(2):
        for backend in (stub, physics, virtual):
            assert backend.grasp(0.04, 0.005, 0.005, 0.1, 40.0) is True
            assert backend.get_state().width == pytest.approx(0.04, abs=5e-3)
            assert backend.get_state().is_grasped is True
