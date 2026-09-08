"""The joint side of a Cartesian command: errors 29 and 30.

A real Franka does not accept a Cartesian pose as a pose. Every ``O_T_EE_c``
goes through the controller's inverse kinematics and the *joint* trajectory
that comes out is held to per-joint limits, which is how a +x ramp at
2.5 m/s^2 -- a fifth of libfranka's Cartesian acceleration limit -- is refused
on a real FER with ``cartesian_motion_generator_joint_velocity_discontinuity``.
This file pins the sim's emulation of that judgement
(:meth:`franka_sim.limits.checker.MotionLimitChecker._check_cartesian_joint_continuity`)
against the hardware calibration recorded next to
:data:`franka_sim.limits.tables.JOINT_VELOCITY_DISCONTINUITY_LIMIT`:

* the calibration itself, cycle for cycle: the 2.5 m/s^2 / 500 m/s^3 ramp from
  the calibration pose trips 29 on joint 2 the moment the per-cycle velocity
  increment reaches 2.5 mm/s, the 1.5 m/s^2 / 200 m/s^3 ramp never does;
* the cases that must *not* trip -- a motion's opening command, a resume after
  a held reference, lost cycles, libfranka's own Cartesian pose example;
* the plumbing: the scale knob (env var, CLI flag), the no-IK backend, and the
  reflex over the real wire with real physics, including recovery.

Everything here needs the MuJoCo FR3 model for its kinematics; without it the
module is skipped rather than failed, like the other MuJoCo-backed tests.
"""

import logging
import math
import threading
import time
from unittest.mock import MagicMock, patch

import mujoco
import numpy as np
import pytest
from test_motion_limits import (
    ERRORS_SLICE,
    REFLEX_REASON_SLICE,
    ROBOT_MODE_INDEX,
    WireClient,
    wait_for_server,
)

import franka_sim
from franka_sim.cartesian_ik import CartesianJointSolver
from franka_sim.control_modes import ControlMode
from franka_sim.franka_protocol import (
    COMMAND_PORT,
    ControllerMode,
    MotionGeneratorMode,
    MoveStatus,
    RobotMode,
)
from franka_sim.franka_sim_server import FrankaSimServer
from franka_sim.motion_limits import (
    CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
    CARTESIAN_MOTION_GENERATOR_JOINT_ACCELERATION_DISCONTINUITY_INDEX,
    CARTESIAN_MOTION_GENERATOR_JOINT_VELOCITY_DISCONTINUITY_INDEX,
    DELTA_T,
    JOINT_DISCONTINUITY_SCALE_ENV_VAR,
    JOINT_VELOCITY_DISCONTINUITY_LIMIT,
    MotionLimitChecker,
    joint_discontinuity_scale_from_env,
)
from franka_sim.mujoco_franka_sim import MujocoFrankaSim, default_fr3_mjcf
from franka_sim.run_server import build_parser, run_single_arm, validate_args

try:
    FR3_MJCF = default_fr3_mjcf()
except Exception:  # pragma: no cover - depends on the host's cache/network
    FR3_MJCF = None

pytestmark = pytest.mark.skipif(
    FR3_MJCF is None or not FR3_MJCF.exists(),
    reason="the MuJoCo Menagerie FR3 model is neither cached nor downloadable",
)

#: The calibration pose: the left FER's ``q`` when the ramps below were run
#: (2026-09-08), essentially libfranka's ready pose.
CALIBRATION_Q = [-0.046, -0.873, -0.104, -2.437, -0.051, 1.617, 0.748]

#: The Franka Hand's ``F_T_EE``: 45 deg about the flange z, 0.1034 m along it.
#: Row-major.
FRANKA_HAND_F_T_EE = np.array(
    [
        [math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0, 0.0],
        [math.sin(math.pi / 4), math.cos(math.pi / 4), 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.1034],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

#: The errors' indices, by the short names the assertions below read best with.
JOINT_VELOCITY = CARTESIAN_MOTION_GENERATOR_JOINT_VELOCITY_DISCONTINUITY_INDEX
JOINT_ACCELERATION = CARTESIAN_MOTION_GENERATOR_JOINT_ACCELERATION_DISCONTINUITY_INDEX


@pytest.fixture(scope="module")
def arm():
    """One compiled FR3, parked at the calibration pose with the hand's ``F_T_EE``.

    Module-scoped because compiling the Menagerie model is the slow part and
    nothing here steps the physics: the tests only ask it for kinematics.
    """
    sim = MujocoFrankaSim()
    sim.initialize_simulation()
    sim.data.qpos[sim.arm_qpos_adr] = CALIBRATION_Q
    sim.data.qvel[sim.arm_dofs_idx] = 0.0
    # ``ee_pose()`` reads ``xpos``/``xmat``, which only a forward pass refreshes.
    mujoco.mj_forward(sim.model, sim.data)
    sim.update_ee_transform(FRANKA_HAND_F_T_EE.T.flatten())
    yield sim
    sim.stop()


def wire_pose(matrix):
    """A row-major 4x4 as the wire's column-major 16-element list."""
    return [float(value) for value in np.asarray(matrix).T.flatten()]


def command(message_id, **fields):
    """A RobotCommand dict shaped as the UDP receive path builds one."""
    base = {
        "message_id": message_id,
        "q_c": [0.0] * 7,
        "dq_c": [0.0] * 7,
        "O_T_EE_c": [0.0] * 16,
        "O_dP_EE_c": [0.0] * 6,
        "elbow_c": [0.0] * 2,
        "valid_elbow": False,
        "motion_generation_finished": False,
        "tau_J_d": [0.0] * 7,
        "torque_command_finished": False,
    }
    base.update(fields)
    return base


def limit_rate_ramp(velocity, acceleration, jerk, cycles, axis=0):
    """Per-cycle EE offsets of a libfranka-style ``limitRate`` ramp from rest.

    What the calibration client sent: a jerk-limited, then acceleration-limited,
    then velocity-capped ramp of one translation axis. Returns ``cycles``
    offsets (m) along ``axis``; the first is one cycle in.
    """
    offsets = []
    a = v = x = 0.0
    for _ in range(cycles):
        a = min(a + jerk * DELTA_T, acceleration)
        v = min(v + a * DELTA_T, velocity)
        x += v * DELTA_T
        offsets.append(x)
    return offsets


def pose_stream(start, offsets, axis=0):
    """Wire poses: ``start`` translated by each offset along ``axis``."""
    poses = []
    for offset in offsets:
        pose = np.array(start, dtype=float)
        pose[axis, 3] += offset
        poses.append(wire_pose(pose))
    return poses


def seed_state(sim):
    """The state a Cartesian ``Move`` seeds the checker from, off this arm."""
    q = list(sim.data.qpos[sim.arm_qpos_adr])
    return {
        "q": q,
        "q_d": q,
        "dq_d": [0.0] * 7,
        "ddq_d": [0.0] * 7,
        "tau_J_d": [0.0] * 7,
        "O_dP_EE_d": [0.0] * 6,
        "O_T_EE": wire_pose(sim.ee_pose()),
    }


def checker_for(sim, mode=ControlMode.CARTESIAN_POSE, scale=1.0, solver=True, **kwargs):
    """A checker armed for a Cartesian motion on ``sim``'s kinematics."""
    checker = MotionLimitChecker(
        joint_kinematics=(lambda: sim.joint_space_solver()) if solver else None,
        joint_discontinuity_scale=scale,
        **kwargs,
    )
    checker.start_motion(mode, seed_state(sim))
    return checker


def stream(checker, signals, field="O_T_EE_c", start_id=1, rest=3):
    """Feed ``signals`` after ``rest`` standstill commands; the first verdict.

    Returns ``(ramp_cycle, violation)`` -- ``ramp_cycle`` counting from 1 at
    the first non-rest signal, so it reads like the calibration notes -- or
    ``(None, None)`` when every command passed. Every command that passes is
    recorded, as the server records it.
    """
    opening = wire_pose(checker_start_pose(checker)) if field == "O_T_EE_c" else [0.0] * 6
    sequence = [opening] * rest + list(signals)
    for index, value in enumerate(sequence):
        message_id = start_id + index
        checker.note_published(message_id)
        received = command(message_id, **{field: value})
        violation = checker.check(received)
        if violation is not None:
            return index - rest + 1, violation
        checker.record(received)
    return None, None


def checker_start_pose(checker):
    """The measured pose the checker was armed at, as a row-major 4x4."""
    return np.asarray(checker._start_pose, dtype=float).reshape(4, 4).T


# -- the calibration, cycle for cycle ----------------------------------------


def test_the_sim_puts_the_calibration_pose_where_the_robot_reported_it(arm):
    """Frame check before anything else: the sim's kinematics are the robot's.

    The robot reported its EE at [0.2895, -0.0550, 0.4769] m with the hand
    mounted. The sim's ``O_T_EE`` is measured at ``link7``, 0.107 m short of the
    true flange (see docs/compatibility.md), so composing the hand transform
    onto the flange has to reproduce that position -- and does, to 0.2 mm.
    Without this the calibration below would be numerology.
    """
    solver = arm.joint_space_solver()
    flange = np.eye(4)
    flange[2, 3] = 0.107
    arm.update_ee_transform((flange @ FRANKA_HAND_F_T_EE).T.flatten())
    try:
        pose, _ = solver._forward(np.array(CALIBRATION_Q))
    finally:
        arm.update_ee_transform(FRANKA_HAND_F_T_EE.T.flatten())
    assert pose[:3, 3] == pytest.approx([0.2895, -0.0550, 0.4769], abs=5e-4)
    # EE pointing down, as it was.
    assert pose[:3, 2] == pytest.approx([0.0, 0.0, -1.0], abs=0.06)


def test_the_refused_ramp_trips_joint_2_when_the_velocity_increment_reaches_2_5_mm_s(arm):
    """The calibration's refused case: 0.5 m/s, 2.5 m/s^2, 500 m/s^3 in +x.

    Hardware: refused on the sixth ramp cycle with 29 alone, when the per-cycle
    EE velocity increment reached 2.5 mm/s (steps 0.5, 1.5, 3, 5, 7.5, 10 um).
    Here that increment is the fifth ramp command, and the sim's IK puts
    8.0 rad/s^2 on joint 2 for it against the 7.5 rad/s^2 of its table --
    1.07x, which is the tighter side of the calibration bracket.
    """
    checker = checker_for(arm)
    offsets = limit_rate_ramp(0.5, 2.5, 500.0, 60)
    cycle, violation = stream(checker, pose_stream(checker_start_pose(checker), offsets))

    assert violation is not None, "the ramp the robot refused ran clean"
    assert violation.error_index == JOINT_VELOCITY
    assert violation.error_indices == (JOINT_VELOCITY,), "29 alone, no jerk error"
    assert violation.axis == "joint 2"
    assert violation.unit == "rad/s^2"
    assert violation.limit == pytest.approx(JOINT_VELOCITY_DISCONTINUITY_LIMIT[1])
    assert abs(violation.value) == pytest.approx(8.0, abs=0.1)
    # The increments run 0.5, 1.0, 1.5, 2.0, 2.5 mm/s: the fifth command is the
    # first at 2.5 mm/s, and hardware counted the abort on the sixth cycle.
    assert 5 <= cycle <= 6, f"tripped on ramp cycle {cycle}, hardware: the sixth"
    assert checker.violated is False, "check() judges; the server latches"


def test_the_velocity_cap_is_irrelevant_to_the_refusal(arm):
    """The same ramp capped at 0.3 m/s: identical outcome on hardware, and here."""
    checker = checker_for(arm)
    cycle, violation = stream(
        checker,
        pose_stream(checker_start_pose(checker), limit_rate_ramp(0.3, 2.5, 500.0, 60)),
    )
    assert violation is not None and violation.error_index == JOINT_VELOCITY
    assert 5 <= cycle <= 6


def test_the_accepted_ramp_runs_clean_to_the_velocity_it_reached_on_hardware(arm):
    """The calibration's accepted case: 0.3 m/s, 1.5 m/s^2, 200 m/s^3.

    Hardware ran it to 0.22 m/s before an unrelated contact reflex. Peak joint-2
    acceleration here is 4.8 rad/s^2, 0.64x its limit -- the looser side of
    the bracket. 200 cycles takes the ramp through its whole jerk phase and
    to 0.29 m/s, past the speed hardware reached and short of the 0.3 m/s cap
    (which the robot never hit; a ``limitRate`` cap is an acceleration step,
    and that step is a jerk the calibration says nothing about).
    """
    checker = checker_for(arm)
    offsets = limit_rate_ramp(0.3, 1.5, 200.0, 200)
    assert offsets[-1] - offsets[-2] > 0.22 * DELTA_T, "the ramp never reached hardware's speed"
    cycle, violation = stream(checker, pose_stream(checker_start_pose(checker), offsets))
    assert violation is None, f"refused on ramp cycle {cycle}: {violation.describe()}"

    _, velocity, acceleration = checker.cartesian_joint_history()
    assert abs(velocity[1]) > 0.5, "joint 2 is the joint this ramp loads"
    assert max(abs(a) for a in acceleration) < JOINT_VELOCITY_DISCONTINUITY_LIMIT[1]


@pytest.mark.parametrize(
    "velocity, acceleration, jerk, cycles",
    [(0.4, 1.0, 100.0, 450), (0.3, 0.5, 20.0, 600)],
)
def test_the_gentler_ramps_hardware_accepted_run_clean(arm, velocity, acceleration, jerk, cycles):
    """0.4 m/s / 1 m/s^2 / 100 m/s^3 and 0.3 / 0.5 / 20: clean for 19 s on hardware.

    Those 19 s were the commander's script -- +-5 cm steps inside a 12 cm box,
    each ramp ending where its target was -- so the ramps are run here for the
    displacement the robot actually covered (under 12 cm), not driven on at
    speed towards the workspace boundary, where the IK's joint accelerations
    grow without bound at constant Cartesian velocity and hardware would refuse
    too.
    """
    checker = checker_for(arm)
    offsets = limit_rate_ramp(velocity, acceleration, jerk, cycles)
    assert 0.05 < offsets[-1] < 0.12, "run the ramp over the commander's own box"
    cycle, violation = stream(checker, pose_stream(checker_start_pose(checker), offsets))
    assert violation is None, f"refused on ramp cycle {cycle}: {violation.describe()}"


def test_a_ramp_at_libfrankas_own_cartesian_limits_latches_both_joint_errors(arm):
    """Hardware refused 13 m/s^2 / 6500 m/s^3 with 29 **and** 30 from one abort.

    This sim judges the Cartesian side by the FR3's tables (9 m/s^2, 4500
    m/s^3), under which that exact ramp is already a Cartesian jerk violation
    (20) before the joint side is reached -- see the test below. The joint-side
    pairing is pinned with a ramp *inside* the FR3 Cartesian limits: 8 m/s^2 /
    4000 m/s^3 puts 12.8 rad/s^2 and 12 800 rad/s^3 on joint 2 in its first
    cycle, both over, and both bits come back from the one verdict, as the
    13/3 pairing does for the joint-velocity envelope.
    """
    checker = checker_for(arm)
    cycle, violation = stream(
        checker,
        pose_stream(checker_start_pose(checker), limit_rate_ramp(1.0, 8.0, 4000.0, 20)),
    )
    assert cycle == 1
    assert violation.error_indices == (JOINT_VELOCITY, JOINT_ACCELERATION)
    assert violation.axis == "joint 2"
    assert "+" in violation.describe()


def test_the_cartesian_side_still_comes_first(arm):
    """13 m/s^2 / 6500 m/s^3 on an FR3-limited sim is 20, the Cartesian name.

    Precedence between the Cartesian and the joint side of one command is not
    pinned by hardware (the FER's Cartesian limits are 13 / 6500, so that ramp
    exercised only the joint side there); the sim keeps the Cartesian checks
    first, as they were.
    """
    checker = checker_for(arm)
    cycle, violation = stream(
        checker,
        pose_stream(checker_start_pose(checker), limit_rate_ramp(13.0, 13.0, 6500.0, 20)),
    )
    assert cycle == 1
    assert violation.error_index == CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX


# -- the knob -----------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.6, 0.4, 0.2])
def test_tightening_the_scale_refuses_the_ramp_hardware_accepted(arm, scale):
    """The accepted 1.5 m/s^2 ramp peaks at 0.64x joint 2's limit.

    Any scale under 0.64 has to refuse it, with the same error, on joint 2 --
    and the tighter the scale, the earlier in the ramp.
    """
    checker = checker_for(arm, scale=scale)
    cycle, violation = stream(
        checker,
        pose_stream(checker_start_pose(checker), limit_rate_ramp(0.3, 1.5, 200.0, 400)),
    )
    assert violation is not None and violation.error_index == JOINT_VELOCITY
    assert violation.axis == "joint 2"
    assert violation.limit == pytest.approx(JOINT_VELOCITY_DISCONTINUITY_LIMIT[1] * scale)
    assert cycle <= 8 * scale / 0.6 + 1


def test_the_calibration_bracket_scaled(arm):
    """0.65 keeps the accepted ramp clean; 1.1 lets the refused one through.

    The two ends of the bracket the hardware data leaves for the factor
    (0.641, 1.069): the calibrated 1.0 is inside, and this is what moving it
    to either edge does.
    """
    start = checker_start_pose(checker_for(arm))
    _, violation = stream(
        checker_for(arm, scale=0.65), pose_stream(start, limit_rate_ramp(0.3, 1.5, 200.0, 200))
    )
    assert violation is None
    _, violation = stream(
        checker_for(arm, scale=1.1), pose_stream(start, limit_rate_ramp(0.5, 2.5, 500.0, 200))
    )
    assert violation is None


def test_the_scale_comes_from_the_environment_when_not_given(arm, monkeypatch):
    monkeypatch.setenv(JOINT_DISCONTINUITY_SCALE_ENV_VAR, "0.5")
    checker = MotionLimitChecker()
    assert checker.joint_discontinuity_scale == 0.5
    assert MotionLimitChecker(joint_discontinuity_scale=2.0).joint_discontinuity_scale == 2.0
    monkeypatch.delenv(JOINT_DISCONTINUITY_SCALE_ENV_VAR)
    assert MotionLimitChecker().joint_discontinuity_scale == 1.0


@pytest.mark.parametrize("raw", ["", "  ", "abc", "0", "-1", "nan", "inf"])
def test_an_unusable_scale_in_the_environment_is_reported_and_ignored(raw, caplog):
    """0 would refuse every Cartesian command and -1 none; neither is a setting."""
    with caplog.at_level(logging.WARNING, logger="franka_sim.motion_limits"):
        assert joint_discontinuity_scale_from_env({JOINT_DISCONTINUITY_SCALE_ENV_VAR: raw}) == 1.0
    if raw.strip():
        assert "Ignoring" in caplog.text
    assert joint_discontinuity_scale_from_env({JOINT_DISCONTINUITY_SCALE_ENV_VAR: "0.75"}) == 0.75
    assert joint_discontinuity_scale_from_env({}) == 1.0


def test_the_cli_flag_reaches_the_server_and_wins_over_the_environment(monkeypatch):
    def constructed(argv):
        args = build_parser().parse_args(argv)
        validate_args(args)
        with patch.object(franka_sim, "FrankaSimServer", return_value=MagicMock()) as ctor:
            run_single_arm(args)
        return ctor.call_args.kwargs

    assert constructed(["--no-gripper"])["joint_discontinuity_scale"] is None
    assert (
        constructed(["--no-gripper", "--joint-discontinuity-scale", "0.7"])[
            "joint_discontinuity_scale"
        ]
        == 0.7
    )
    with pytest.raises(ValueError):
        validate_args(build_parser().parse_args(["--joint-discontinuity-scale", "0"]))
    with pytest.raises(ValueError):
        validate_args(build_parser().parse_args(["--joint-discontinuity-scale", "-2"]))
    # None reaches the checker, which then reads the environment.
    monkeypatch.setenv(JOINT_DISCONTINUITY_SCALE_ENV_VAR, "0.9")
    server = FrankaSimServer(physics_sim=MagicMock(), enable_gripper=False)
    assert server.motion_limits.joint_discontinuity_scale == 0.9
    server = FrankaSimServer(
        physics_sim=MagicMock(), enable_gripper=False, joint_discontinuity_scale=1.2
    )
    assert server.motion_limits.joint_discontinuity_scale == 1.2


# -- what must not trip ---------------------------------------------------------


def test_libfrankas_cartesian_pose_example_runs_clean(arm):
    """``generate_cartesian_pose_motion``: the circle the same robot accepts.

    ``angle = pi/4 (1 - cos(pi/5 t))``, ``dx = 0.3 sin(angle)``, ``dz = 0.3
    (cos(angle) - 1)``, ten seconds. Its initial acceleration is ~0.09 m/s^2 and
    it never gets anywhere near the joint tables: the whole example at scale
    1.0, every cycle judged, with no violation.
    """
    checker = checker_for(arm)
    start = checker_start_pose(checker)
    poses = []
    for cycle in range(1, 10001):
        t = cycle * DELTA_T
        angle = math.pi / 4 * (1 - math.cos(math.pi / 5 * t))
        pose = start.copy()
        pose[0, 3] += 0.3 * math.sin(angle)
        pose[2, 3] += 0.3 * (math.cos(angle) - 1)
        poses.append(wire_pose(pose))
    # The first half, out to the far side of the circle...
    cycle, violation = stream(checker, poses[:5000])
    assert violation is None, f"the circle was refused at cycle {cycle}: {violation.describe()}"
    # ...did move the arm through the IK, so the check was live...
    q, _, _ = checker.cartesian_joint_history()
    assert max(abs(a - b) for a, b in zip(q, CALIBRATION_Q)) > 0.3
    # ...and the way back closes the circle, still clean, at the start pose.
    for index, pose in enumerate(poses[5000:], start=5004):
        checker.note_published(index)
        received = command(index, O_T_EE_c=pose)
        violation = checker.check(received)
        assert violation is None, f"cycle {index}: {violation.describe()}"
        checker.record(received)
    q, _, _ = checker.cartesian_joint_history()
    assert q == pytest.approx(CALIBRATION_Q, abs=1e-3)


def test_a_first_command_inside_the_start_tolerance_is_not_a_joint_step(arm):
    """The opening command rebases the joint history; nothing is differenced.

    2 cm away from the measured pose is inside the 5 cm start-pose tolerance,
    so the command is accepted -- and the 2 cm the IK has to travel from the
    seeded ``q`` to reach it must not read as a velocity, an acceleration or a
    jerk on the commands that follow. Held there, and then ramped gently, the
    motion is clean.
    """
    checker = checker_for(arm)
    start = checker_start_pose(checker)
    opening = start.copy()
    opening[0, 3] += 0.02
    poses = pose_stream(opening, [0.0] * 5 + limit_rate_ramp(0.3, 1.0, 100.0, 200))
    for index, pose in enumerate(poses):
        checker.note_published(index + 1)
        received = command(index + 1, O_T_EE_c=pose)
        violation = checker.check(received)
        assert violation is None, f"cycle {index + 1}: {violation.describe()}"
        checker.record(received)
    q, _, _ = checker.cartesian_joint_history()
    solver = arm.joint_space_solver()
    reached, converged = solver.solve_pose(poses[-1], q)
    assert converged and np.allclose(reached, q, atol=1e-6), "the history is the IK's own"


def test_a_resume_after_a_held_reference_is_not_charged_the_gap(arm):
    """``note_hold``: the re-seed window covers the joint side too.

    The reference stood still for the whole hold; a client that comes back at
    the 0.15 m/s it was commanding would, differenced against the standstill,
    read as 150 m/s^2 in Cartesian terms and ~480 rad/s^2 on joint 2. Both
    histories honour the same two-command window, so the resume is judged on
    its velocity envelope only and the joint side is silent until the window
    closes -- after which the constant-velocity stream is honestly clean.
    """
    checker = checker_for(arm)
    start = checker_start_pose(checker)
    offsets = limit_rate_ramp(0.15, 1.5, 200.0, 150)
    cycle, violation = stream(checker, pose_stream(start, offsets))
    assert violation is None
    held = offsets[-1]
    checker.note_published(153 + 160)
    checker.note_hold()

    for index in range(1, 8):
        message_id = 153 + 160 + index
        checker.note_published(message_id)
        pose = start.copy()
        pose[0, 3] += held + 0.15 * index * DELTA_T
        received = command(message_id, O_T_EE_c=wire_pose(pose))
        violation = checker.check(received)
        assert violation is None, f"resume command {index}: {violation.describe()}"
        checker.record(received)


def test_lost_cycles_are_extrapolated_along_the_joint_trajectory_too(arm):
    """Three missed cycles in the constant-acceleration phase of the accepted ramp.

    The extrapolated poses (frozen Cartesian acceleration, which is exact for a
    ramp at constant 1.5 m/s^2) go through the same IK and advance the joint
    history, so the client's resume -- its own waypoint for the cycle after
    the gap -- is one honest cycle from the history on both sides.
    """
    checker = checker_for(arm)
    start = checker_start_pose(checker)
    offsets = limit_rate_ramp(0.3, 1.5, 200.0, 200)
    poses = pose_stream(start, offsets)
    rest = 3
    # 100 cycles in, the jerk phase (7.5 cycles) is long over.
    cycle, violation = stream(checker, poses[:100], rest=rest)
    assert violation is None
    applied = rest + 100  # message id of the last recorded command
    for lost in range(1, 4):
        message_id = applied + lost
        checker.note_published(message_id)
        extrapolated = checker.extrapolate(message_id)
        assert extrapolated is not None
        _, violation = extrapolated
        assert violation is None, f"guess {lost}: {violation.describe()}"
    for index in range(103, 160):
        message_id = rest + index + 1
        checker.note_published(message_id)
        received = command(message_id, O_T_EE_c=poses[index])
        violation = checker.check(received)
        assert violation is None, f"resume at {index}: {violation.describe()}"
        checker.record(received)


def test_a_backend_without_inverse_kinematics_keeps_the_joint_side_off(arm):
    """No solver, no joint trajectory to judge: the refused ramp passes.

    This is the Genesis / mobile-duo case, and the mocked-backend wire tests.
    The Cartesian side is untouched by it -- a genuine pose step is still 19.
    """
    checker = checker_for(arm, solver=False)
    assert checker.cartesian_joint_history() is None
    start = checker_start_pose(checker)
    _, violation = stream(checker, pose_stream(start, limit_rate_ramp(0.5, 2.5, 500.0, 60)))
    assert violation is None
    _, violation = stream(
        checker_for(arm, solver=False), pose_stream(start, [0.0] * 3 + [0.01] * 3)
    )
    assert violation is not None and violation.error_index == 19


def test_a_solver_that_does_not_converge_is_skipped_not_latched(arm, caplog):
    """A pose the IK cannot reach is not a discontinuity the client commanded."""

    class Stubborn(CartesianJointSolver):
        def solve_pose(self, pose, seed, elbow_angle=None):
            q, _ = super().solve_pose(pose, seed, elbow_angle)
            return q, False

    real = arm.joint_space_solver()
    checker = MotionLimitChecker(joint_kinematics=lambda: Stubborn(real._forward))
    checker.start_motion(ControlMode.CARTESIAN_POSE, seed_state(arm))
    start = checker_start_pose(checker)
    with caplog.at_level(logging.WARNING, logger="franka_sim.motion_limits"):
        _, violation = stream(checker, pose_stream(start, limit_rate_ramp(0.5, 2.5, 500.0, 30)))
    assert violation is None
    assert caplog.text.count("did not converge") == 1, "logged once per motion"


# -- the Cartesian velocity interface --------------------------------------------


def twist_ramp(velocity, acceleration, jerk, cycles, axis=0):
    """The twists of a ``limitRate`` ramp: one 6-vector per cycle."""
    twists = []
    a = v = 0.0
    for _ in range(cycles):
        a = min(a + jerk * DELTA_T, acceleration)
        v = min(v + a * DELTA_T, velocity)
        twist = [0.0] * 6
        twist[axis] = v
        twists.append(twist)
    return twists


def test_on_the_velocity_interface_the_joint_side_lands_on_30(arm):
    """The same +x ramp as a twist: the joint acceleration is 30, not 29.

    Interface-relative naming, as for ``dq_c``: on a commanded velocity the
    first difference is already an acceleration. No hardware observation pins
    this half; it follows the documented rule for the joint generators.
    """
    checker = checker_for(arm, mode=ControlMode.CARTESIAN_VELOCITY)
    cycle, violation = stream(checker, twist_ramp(0.5, 2.5, 500.0, 60), field="O_dP_EE_c")
    assert violation is not None
    assert violation.error_indices == (JOINT_ACCELERATION,)
    assert violation.axis == "joint 2" and violation.unit == "rad/s^2"
    assert 5 <= cycle <= 6

    checker = checker_for(arm, mode=ControlMode.CARTESIAN_VELOCITY)
    cycle, violation = stream(checker, twist_ramp(0.3, 1.5, 200.0, 400), field="O_dP_EE_c")
    assert violation is None, f"refused at {cycle}: {violation.describe()}"
    q, velocity, _ = checker.cartesian_joint_history()
    assert velocity[1] == pytest.approx(0.3 * 3.2, rel=0.1)
    assert abs(q[1] - CALIBRATION_Q[1]) > 0.1, "the twist was integrated into the joint history"


# -- the reflex over the real wire ------------------------------------------------


@pytest.fixture
def live_server():
    """A real FrankaSimServer over a real MuJoCo arm parked at the calibration pose."""
    made = []

    def _live(enforce=True, scale=None):
        sim = MujocoFrankaSim()
        sim.initialize_simulation()
        sim.data.qpos[sim.arm_qpos_adr] = CALIBRATION_Q
        sim.data.qvel[sim.arm_dofs_idx] = 0.0
        sim.update_joint_positions(np.array(CALIBRATION_Q))
        sim.set_control_mode(ControlMode.POSITION)
        sim.step(200)
        server = FrankaSimServer(
            physics_sim=sim,
            enable_gripper=False,
            enforce_motion_limits=enforce,
            joint_discontinuity_scale=scale,
        )
        accept_thread = threading.Thread(target=server.run_server, daemon=True)
        accept_thread.start()
        sim.running = True
        physics_thread = threading.Thread(target=sim.run_simulation, daemon=True)
        physics_thread.start()
        assert wait_for_server(server, COMMAND_PORT), (
            f"this test's FCI server never took port {COMMAND_PORT} -- something "
            "else is listening there"
        )
        made.append((server, sim, accept_thread, physics_thread))
        return server

    yield _live

    for server, sim, accept_thread, physics_thread in made:
        server.stop()
        sim.stop()
        physics_thread.join(timeout=3.0)
        accept_thread.join(timeout=3.0)
    time.sleep(0.4)


@pytest.fixture
def client():
    clients = []

    def _client(**kwargs):
        made = WireClient(**kwargs)
        clients.append(made)
        return made

    yield _client
    for made in clients:
        made.close()


def test_the_refused_ramp_is_a_reflex_over_the_wire_and_recovery_clears_it(live_server, client):
    """End to end: kReflexAborted, kReflex with bit 29, recovery, a clean re-Move.

    Real physics, real sockets, enforcement on. The client is libfranka-shaped:
    it opens the pose motion on the robot's own ``O_T_EE`` and ramps it exactly
    as the calibration client did. The abort has to reach it as the same reflex
    every other motion-limit violation does, and ``AutomaticErrorRecovery`` has
    to leave the arm ready for a motion the robot accepts -- the 1 m/s^2 /
    100 m/s^3 ramp, run from wherever the reflex parked the arm.
    """
    server = live_server(enforce=True)
    wire = client()
    wire.connect()
    assert (
        wire.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianPosition)
        == MoveStatus.kMotionStarted
    )
    start = np.asarray(server.physics_sim.get_robot_state()["O_T_EE"]).reshape(4, 4).T
    waypoints = pose_stream(start, [0.0] * 3 + limit_rate_ramp(0.5, 2.5, 500.0, 40))
    wire.ramp(waypoints, field="o_t_ee_c")

    assert wire.read_move_response() == MoveStatus.kReflexAborted
    aborted = wire.read_state()
    assert aborted[ROBOT_MODE_INDEX] == RobotMode.kReflex
    assert aborted[ERRORS_SLICE][JOINT_VELOCITY] == 1
    assert aborted[ERRORS_SLICE][JOINT_ACCELERATION] == 0
    assert sum(aborted[ERRORS_SLICE]) == 1
    assert server.motion_limits.violated is True

    # A Move while latched is refused, as on the robot.
    assert (
        wire.move(
            ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianPosition, command_id=5
        )
        == MoveStatus.kCommandNotPossibleRejected
    )

    assert wire.automatic_error_recovery() == 0
    cleared = wire.read_state()
    assert not any(cleared[ERRORS_SLICE])
    assert cleared[REFLEX_REASON_SLICE][JOINT_VELOCITY] == 1
    assert cleared[ROBOT_MODE_INDEX] == RobotMode.kIdle
    assert server.motion_limits.violated is False

    assert (
        wire.move(
            ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianPosition, command_id=11
        )
        == MoveStatus.kMotionStarted
    )
    start = np.asarray(server.physics_sim.get_robot_state()["O_T_EE"]).reshape(4, 4).T
    state = wire.ramp(
        pose_stream(start, [0.0] * 3 + limit_rate_ramp(0.4, 1.0, 100.0, 400)), field="o_t_ee_c"
    )
    assert not any(state[ERRORS_SLICE]), "the accepted ramp tripped a limit over the wire"
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove
    assert server.motion_limits.violated is False
