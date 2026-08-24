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
    ARM_POSITION_KD,
    ARM_POSITION_KP,
    DEFAULT_DT,
    FLANGE_OFFSET_Z,
    MAX_FINGER_TRAVEL,
    MujocoFrankaSim,
    default_fr3_mjcf,
)
from franka_sim.sim_common import (  # noqa: E402
    FR3_FORCE_LIMITS,
    POSITION_FEEDFORWARD_HOLD_STEPS,
    PositionFeedforward,
)

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
def sim_factory():
    """Build independent hand-less arms; all of them torn down after the test."""
    built = []

    def _build(**kwargs):
        simulator = MujocoFrankaSim(**kwargs)
        simulator.initialize_simulation()
        built.append(simulator)
        return simulator

    yield _build

    for simulator in built:
        simulator.stop()


@pytest.fixture
def sim(sim_factory):
    """A built, hand-less 7-DOF arm holding its initial pose."""
    return sim_factory()


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


def test_idle_hold_shaped_constant_target_matches_the_old_law(sim):
    """The idle hold sets q_d = current q, then switches to POSITION mode --
    the target never differs from where the arm already is, so the new law's
    velocity feedforward is 0 on every step and its output must match the old
    ``KP*error - KD*dq`` law bit for bit.
    """
    current_q = sim.get_robot_state()["q"].copy()
    sim.update_joint_positions(current_q)  # target first, mode second (idle-hold order)
    sim.set_control_mode(ControlMode.POSITION)

    for _ in range(50):
        q = sim.data.qpos[sim.arm_qpos_adr].copy()
        dq = sim.data.qvel[sim.arm_dofs_idx].copy()
        old_law = np.clip(
            ARM_POSITION_KP * (current_q - q) - ARM_POSITION_KD * dq,
            -FR3_FORCE_LIMITS,
            FR3_FORCE_LIMITS,
        )
        new_law = sim.arm_control_torque()
        assert new_law == pytest.approx(old_law)
        sim.step(1)


def test_switching_into_position_mode_resets_the_feedforward_baseline(sim):
    """A mode switch must not let a stale target produce a first-step spike.

    Runs POSITION mode with a drifting target long enough that the stored
    feedforward baseline has moved on, switches to TORQUE for a few steps,
    then switches back into POSITION the way the idle hold does: the target
    is set to the arm's actual current position *before* the mode switch.
    The switch must snap the baseline to that target, so the very first
    torque computed afterwards has zero velocity feedforward -- it must equal
    the old, undamped-feedforward law exactly.
    """
    sim.set_control_mode(ControlMode.POSITION)
    drifting_target = ARM_INITIAL_Q.copy()
    for _ in range(50):
        drifting_target = drifting_target + 0.001
        sim.update_joint_positions(drifting_target)
        sim.step(1)
    assert not np.allclose(sim.prev_position_target, ARM_INITIAL_Q)

    sim.set_control_mode(ControlMode.TORQUE)
    sim.update_torques(np.zeros(7))
    sim.step(5)

    current_q = sim.get_robot_state()["q"].copy()
    sim.update_joint_positions(current_q)
    sim.set_control_mode(ControlMode.POSITION)
    assert sim.prev_position_target == pytest.approx(current_q)

    q = sim.data.qpos[sim.arm_qpos_adr].copy()
    dq = sim.data.qvel[sim.arm_dofs_idx].copy()
    expected_no_spike = np.clip(
        ARM_POSITION_KP * (current_q - q) - ARM_POSITION_KD * dq,
        -FR3_FORCE_LIMITS,
        FR3_FORCE_LIMITS,
    )
    assert sim.arm_control_torque() == pytest.approx(expected_no_spike)


def test_constant_velocity_ramp_stays_within_the_tracking_guard(sim):
    """A smooth streamed trajectory must not fail libfranka's tracking guard.

    libfranka's own smoke-test rejects a motion whose measured joint
    positions stray from the *previous cycle's* commanded positions by more
    than 6e-3 rad RMSE at 1 kHz. The undamped-feedforward law fails this on an
    ordinary ramp because its damping term fights the commanded motion,
    adding a (KD/KP)*v lag; the fix (damping against the commanded velocity)
    must keep this comfortably inside the guard.
    """
    joint_idx = 3  # joint4 (1-indexed), well inside its travel limits
    velocity = 0.5  # rad/s
    dt = DEFAULT_DT
    steps = 300

    sim.set_control_mode(ControlMode.POSITION)
    q_c = ARM_INITIAL_Q.copy()
    sim.update_joint_positions(q_c)

    max_error = 0.0
    for _ in range(steps):
        prev_q_c = q_c[joint_idx]
        q_c = q_c.copy()
        q_c[joint_idx] += velocity * dt
        sim.update_joint_positions(q_c)
        sim.step(1)
        q_measured = sim.get_robot_state()["q"][joint_idx]
        max_error = max(max_error, abs(q_measured - prev_q_c))

    assert max_error < 0.003


# --- the feedforward under a UDP/physics clock beat --------------------------

#: The ramp the beat tests stream: joint 4 accelerating to 0.5 rad/s and holding
#: it, 300 cycles in all.
BEAT_JOINT = 3
BEAT_VELOCITY = 0.5
BEAT_STEPS = 300
#: Cycles the commanded velocity takes to reach BEAT_VELOCITY from rest -- 5
#: rad/s^2, half of kMaxJointAcceleration. A *step* to 0.5 rad/s would be 500
#: rad/s^2, which the motion-limit checker refuses outright and which pins the
#: actuator rail for its first two cycles in every delivery pattern; that spike
#: is the client's, not the clock beat's, and measuring it here would tell us
#: nothing about either.
BEAT_RAMP_CYCLES = 100


def _ramp_waypoints(cycles):
    """The client's 1 kHz waypoint stream for the beat tests."""
    waypoints, q = [], ARM_INITIAL_Q.copy()
    for cycle in range(cycles):
        velocity = BEAT_VELOCITY * min(1.0, (cycle + 1) / BEAT_RAMP_CYCLES)
        q = q.copy()
        q[BEAT_JOINT] += velocity * DEFAULT_DT
        waypoints.append(q)
    return waypoints


def _stream_position_ramp(simulator, arrivals):
    """Stream :func:`_ramp_waypoints` on a given per-step arrival pattern.

    The client always sends exactly one waypoint per 1 ms cycle. ``arrivals[k]``
    is how many of those waypoints the *physics* thread finds already published
    when step ``k`` runs -- a conforming client on a lockstep box gives
    ``[1, 1, 1, ...]``, two independent ~1 kHz clocks beating against each other
    give runs of ``[0, 2]``. Either way the same waypoints are applied, in the
    same order, over the same number of steps: only the arrival bookkeeping
    differs.

    Returns ``(torques, errors)``: the arm torque applied on each step, and each
    step's tracking error against the waypoint that was current when it ran.
    """
    waypoints = _ramp_waypoints(sum(arrivals))
    simulator.set_control_mode(ControlMode.POSITION)
    q_c = ARM_INITIAL_Q.copy()
    simulator.update_joint_positions(q_c)

    torques, errors = [], []
    sent = 0
    for count in arrivals:
        if count:
            sent += count
            q_c = waypoints[sent - 1]
            simulator.update_joint_positions(q_c)
        commanded = q_c[BEAT_JOINT]
        simulator.step(1)
        torques.append(simulator.data.qfrc_applied[simulator.arm_dofs_idx].copy())
        errors.append(simulator.get_robot_state()["q"][BEAT_JOINT] - commanded)
    return np.array(torques), np.array(errors)


def _saturated_fraction(torques):
    """Share of steps where any joint's torque sat on the actuator rail."""
    on_the_rail = np.isclose(np.abs(torques), FR3_FORCE_LIMITS, rtol=0, atol=1e-9)
    return float(on_the_rail.any(axis=1).mean())


def test_a_udp_physics_clock_beat_does_not_saturate_the_position_servo(sim_factory):
    """The revert-and-fail pin for the one-step-difference feedforward.

    Nothing synchronises the UDP receive thread with the physics thread, so a
    step can see no new target or two of them. Differencing over one step
    regardless makes ``dq_c`` alternate 0 / 2v, which at KD=450 and v=0.5 rad/s
    is a +/-225 Nm square wave -- clipped on the +/-87 Nm rail on essentially
    every step. With the interval counted properly, the identical stream
    delivered on a ``[0, 2]`` beat produces the same steady ``dq_c = v`` the
    lockstep delivery does, so neither run touches the rail and the tracking is
    the same to within a hair.
    """
    lockstep_tau, lockstep_err = _stream_position_ramp(sim_factory(), [1] * BEAT_STEPS)
    beat_tau, beat_err = _stream_position_ramp(sim_factory(), [0, 2] * (BEAT_STEPS // 2))

    assert _saturated_fraction(lockstep_tau) == 0.0, "the lockstep baseline itself saturates"
    assert _saturated_fraction(beat_tau) == 0.0, (
        f"{_saturated_fraction(beat_tau):.1%} of the beat run's steps hit the "
        "actuator rail; the feedforward is being differenced over the wrong interval"
    )

    lockstep_rms = float(np.sqrt(np.mean(lockstep_err**2)))
    beat_rms = float(np.sqrt(np.mean(beat_err**2)))
    assert beat_rms < 1.5 * lockstep_rms + 1e-4, (
        f"beat tracking rms {beat_rms:.2e} rad is far worse than lockstep's "
        f"{lockstep_rms:.2e} rad"
    )
    # Both are inside libfranka's own 6e-3 rad tracking guard, with room.
    assert max(beat_rms, lockstep_rms) < 3e-3


def test_the_feedforward_reads_the_same_velocity_lockstep_or_beat():
    """The three behaviours the beat fix is defined by, on the class itself.

    Pure arithmetic, no engine: a lockstep stream, the same stream on a
    ``[0, 2]`` beat, and a target that stops changing.
    """
    dt = DEFAULT_DT
    velocity = np.full(7, 0.5)
    step_delta = velocity * dt

    lockstep = PositionFeedforward(np.zeros(7))
    target = np.zeros(7)
    for _ in range(10):
        target = target + step_delta
        assert lockstep.step(target, dt) == pytest.approx(velocity)

    beat = PositionFeedforward(np.zeros(7))
    target = np.zeros(7)
    seen = []
    for _ in range(10):
        seen.append(beat.step(target, dt).copy())  # nothing arrived this step
        target = target + 2 * step_delta  # ...so two waypoints land before the next
        seen.append(beat.step(target, dt).copy())
    # The very first step of the run has no stream behind it yet, so it holds
    # the reset's zero; from the first arrival onwards it reads the true v.
    assert seen[0] == pytest.approx(np.zeros(7))
    for dq_c in seen[1:]:
        assert dq_c == pytest.approx(velocity)

    # The stream stops: the held feedforward is gone within three steps and
    # never comes back, so the servo settles onto the plain KP*e - KD*dq law.
    for index in range(20):
        dq_c = beat.step(target, dt)
        if index >= POSITION_FEEDFORWARD_HOLD_STEPS - 1:
            assert dq_c == pytest.approx(np.zeros(7)), f"still coasting at step {index}"


def test_the_feedforward_spans_a_gap_longer_than_the_hold_window():
    """A gap longer than ``POSITION_FEEDFORWARD_HOLD_STEPS`` must still be
    differenced over its true span, not over 1 step.

    Regression pin for the stale-drop branch: it must zero ``dq_c`` without
    also resetting ``_unchanged_steps``, because ``self.previous`` already
    equals ``target`` at that point -- resetting the counter would throw away
    how many physics steps the gap actually spanned. Deriving the change
    over the wrong (1-step) span would read the resumed target's velocity as
    ``n`` times too fast for an ``n``-step gap: 3x at a 3-step gap, 6x here.
    """
    dt = DEFAULT_DT
    velocity = np.full(7, 0.5)
    gap_steps = 6  # longer than POSITION_FEEDFORWARD_HOLD_STEPS (3)
    assert gap_steps > POSITION_FEEDFORWARD_HOLD_STEPS

    ff = PositionFeedforward(np.zeros(7))
    target = np.zeros(7)

    # The target holds steady for gap_steps - 1 physics steps: the held
    # feedforward must decay to zero by POSITION_FEEDFORWARD_HOLD_STEPS in,
    # same as the shorter beat-recovery case above.
    for index in range(gap_steps - 1):
        dq_c = ff.step(target, dt)
        if index >= POSITION_FEEDFORWARD_HOLD_STEPS - 1:
            assert dq_c == pytest.approx(np.zeros(7)), f"still coasting at step {index}"

    # A single arrival now carries the whole gap's worth of motion -- the
    # mean velocity over the gap, not gap_steps times the true velocity.
    target = target + gap_steps * velocity * dt
    dq_c = ff.step(target, dt)
    assert dq_c == pytest.approx(velocity), (
        f"expected the gap-spanning velocity {velocity}, got {dq_c} -- the "
        "stale-drop branch is resetting the unchanged-step counter"
    )


def test_the_duo_backend_shares_this_backends_feedforward(sim_factory):
    """The two MuJoCo backends must not drift apart on the beat fix.

    Structural pin: both drive the *same* PositionFeedforward class, so the
    behaviour measured above is one implementation, not two. The duo runs its
    own physics-level beat test (``test_mobile_duo_mujoco_sim.py``).
    """
    from franka_sim import mobile_duo_mujoco_sim

    assert mobile_duo_mujoco_sim.PositionFeedforward is PositionFeedforward
    assert isinstance(sim_factory()._position_feedforward, PositionFeedforward)


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
