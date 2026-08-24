"""FCI communication constraints: lost cycles, the held command, and the abort.

The real FCI drops a command packet that misses its 1 ms cycle, reports the
damage in ``control_command_success_rate``, and stops the motion with
``communication_constraints_violation`` after 20 consecutive drops (libfranka
``docs/network_requirements.rst``). This file covers franka-sim's emulation of
that, in the three layers a regression would show up in:

* unit: the tracker on its own -- window math, freshness in both directions,
  the loss run, latching, re-arming and recovery,
* server over the wire with a mocked simulator: the success rate on the wire,
  the last command *held* through a gap, the abort's error bits and TCP status,
  a Move refused while a violation is latched, error recovery,
* end to end over real physics: a torque loop that drops a measured number of
  cycles, and one that drops enough to be stopped.

The sim does **not** extrapolate a missed motion-generator cycle the way the
robot does; it holds the last applied command. That divergence is deliberate
(see ``docs/robot-state.md``) and is pinned here.

Enforcement is off by default (see :data:`ENFORCE_ENV_VAR`), so the tests that
want the abort ask for it explicitly.
"""

import logging
import select
import socket
import struct
import threading
import time

import numpy as np
import pytest

from franka_sim.comm_constraints import (
    COMMUNICATION_CONSTRAINTS_VIOLATION_INDEX,
    MAX_CONSECUTIVE_LOST_CYCLES,
    SUCCESS_RATE_WINDOW,
    CommConstraintTracker,
    enforcement_enabled_by_env,
)
from franka_sim.control_modes import ControlMode
from franka_sim.franka_protocol import (
    COMMAND_PORT,
    Command,
    ConnectStatus,
    ControllerMode,
    LibfrankaControllerMode,
    LibfrankaMotionGeneratorMode,
    MessageHeader,
    MotionGeneratorMode,
    MoveStatus,
    RobotMode,
)
from franka_sim.robot_state import _ROBOT_STATE_PACKER, RobotState

VIOLATION = COMMUNICATION_CONSTRAINTS_VIOLATION_INDEX


# --- layer 1: the tracker on its own -----------------------------------------


def stream(tracker, cycles, *, start_id=1, answered=True):
    """Run ``cycles`` publish ticks, answering each one (or none of them).

    Returns the last :class:`CycleOutcome`. Ids advance by one per cycle, as
    the server's monotonic ``message_id`` counter does.
    """
    outcome = None
    for offset in range(cycles):
        message_id = start_id + offset
        outcome = tracker.tick(message_id)
        if answered:
            tracker.command_received(message_id)
    return outcome


def test_no_motion_means_no_accounting_and_a_zero_rate():
    """Idle is not "100% success": the robot reports zero with no loop running."""
    tracker = CommConstraintTracker()

    outcome = tracker.tick(1)

    assert outcome.active is False
    assert outcome.success_rate == 0.0
    assert tracker.success_rate == 0.0


def test_cycles_are_not_counted_until_the_client_answers():
    """The gap between kMotionStarted and the first command is not the client's fault."""
    tracker = CommConstraintTracker()
    tracker.start_motion()

    for message_id in range(1, 50):
        assert tracker.tick(message_id).active is False

    tracker.command_received(49)
    assert tracker.tick(50).active is True


def test_a_command_echoing_the_published_id_counts_for_that_cycle():
    tracker = CommConstraintTracker()
    tracker.start_motion()
    tracker.tick(7)

    assert tracker.command_received(7) is True
    assert tracker.tick(8).lost is False


def test_a_stale_echo_is_a_lost_cycle_but_still_arms_the_accounting():
    """A client answering an old state missed its window, exactly as on hardware."""
    tracker = CommConstraintTracker()
    tracker.start_motion()
    tracker.tick(7)

    assert tracker.command_received(6) is False

    outcome = tracker.tick(8)
    assert outcome.active is True
    assert outcome.lost is True
    assert outcome.success_rate == 0.0


def test_the_freshness_horizon_is_configurable():
    tracker = CommConstraintTracker(freshness_horizon=2)
    tracker.start_motion()
    tracker.tick(10)

    assert tracker.command_received(8) is True
    tracker.tick(11)
    assert tracker.command_received(8) is False


def test_the_success_rate_is_the_last_hundred_cycles():
    """Rolling window of SUCCESS_RATE_WINDOW cycles, per include/franka/robot_state.h."""
    tracker = CommConstraintTracker()
    tracker.start_motion()
    stream(tracker, SUCCESS_RATE_WINDOW + 20)
    assert tracker.success_rate == pytest.approx(1.0)

    # Ten cycles with no answer at all.
    lost = 10
    for offset in range(lost):
        tracker.tick(1000 + offset)
    outcome = tracker.tick(2000)

    assert outcome.success_rate == pytest.approx((SUCCESS_RATE_WINDOW - lost) / SUCCESS_RATE_WINDOW)

    # ...and it climbs back as the window refills. Two cycles more than the
    # window, because the tick that resumes the stream is still closing the
    # last lost cycle.
    stream(tracker, SUCCESS_RATE_WINDOW + 2, start_id=3000)
    assert tracker.success_rate == pytest.approx(1.0)


def test_a_partly_filled_window_divides_by_what_it_holds():
    """Three answered cycles are 100%, not 3% -- the window is still filling."""
    tracker = CommConstraintTracker()
    tracker.start_motion()
    outcome = stream(tracker, 4)

    assert outcome.success_rate == pytest.approx(1.0)


def test_consecutive_losses_reset_on_any_answered_cycle():
    tracker = CommConstraintTracker()
    tracker.start_motion()
    stream(tracker, 3)

    # Five ticks close four unanswered cycles: the first of them is still
    # closing the last cycle the stream above answered.
    for offset in range(5):
        tracker.tick(100 + offset)
    assert tracker.consecutive_lost == 4

    tracker.command_received(200)
    tracker.tick(200)
    tracker.command_received(200)
    assert tracker.tick(201).consecutive_lost == 0


def test_the_bound_is_the_documented_literal_twenty():
    """Twenty. Not "whatever the constant says".

    Every other test in this file spells the bound as
    ``MAX_CONSECUTIVE_LOST_CYCLES``, which is right for reading but means a
    mutation of the constant itself changes what all of them assert and none of
    them notices. The number is not derivable from anything in this repo -- it is
    stated in the FCI documentation, "After 20 consecutively dropped packets, the
    robot ``will stop`` with the ``communication_constraints_violation`` error"
    (libfranka ``docs/network_requirements.rst``) -- so it is pinned as a
    literal, once, here.
    """
    assert MAX_CONSECUTIVE_LOST_CYCLES == 20

    tracker = CommConstraintTracker(enforce=True)
    tracker.start_motion()
    stream(tracker, 3)
    triggered_at = None
    for cycle in range(1, 30):
        outcome = tracker.tick(100 + cycle)
        if outcome.violation_triggered:
            triggered_at = outcome.consecutive_lost
            break
    assert triggered_at == 20


def test_the_violation_trips_on_the_twentieth_consecutive_loss():
    """Twenty or more packets lost in a row (libfranka docs/overview.rst)."""
    tracker = CommConstraintTracker(enforce=True)
    tracker.start_motion()
    stream(tracker, 3)

    triggered_at = None
    for cycle in range(1, MAX_CONSECUTIVE_LOST_CYCLES + 5):
        outcome = tracker.tick(100 + cycle)
        if outcome.violation_triggered:
            triggered_at = outcome.consecutive_lost
            break

    assert triggered_at == MAX_CONSECUTIVE_LOST_CYCLES
    assert tracker.violated is True


def test_the_outcome_names_the_cycle_it_closed():
    """The extrapolation needs an id, and it is the *previous* published state's.

    A tick closes the cycle the last state opened and opens the one this state
    starts, so the cycle that was just lost belongs to the id before this one.
    The publish loop stamps that id on the waypoint it substitutes, which is
    what leaves the client's resumed command exactly one cycle from the applied
    history (see ``MotionLimitChecker.extrapolate``). Reported rather than
    inferred as ``published_id - 1``: nothing here promises the caller's ids
    advance by one, and the *first* tick of a session has no predecessor at all.
    """
    tracker = CommConstraintTracker()
    tracker.start_motion()

    assert tracker.tick(41).closed_id == 0  # nothing published before this one
    tracker.command_received(41)
    assert tracker.tick(42).closed_id == 41
    # ...and ids that jump are reported as they are, not as 42 + 1.
    assert tracker.tick(90).closed_id == 42


def test_the_extrapolation_bound_is_reported_whether_or_not_enforcement_is_on():
    """Where the robot stops extrapolating, and where a *reporting* sim must too.

    ``violation_triggered`` only fires when enforcement is on, because it is an
    instruction to abort. The bound is not: it is the point past which a
    reference that kept integrating would carry the arm away on the trajectory
    of a client that is no longer there, and that has to stop happening whether
    or not the sim is also going to stop the motion.
    """
    for enforce in (False, True):
        tracker = CommConstraintTracker(enforce=enforce)
        tracker.start_motion()
        stream(tracker, 3)

        reached = [
            outcome.consecutive_lost
            for outcome in (
                tracker.tick(100 + cycle)
                for cycle in range(3 * MAX_CONSECUTIVE_LOST_CYCLES)
            )
            if outcome.bound_reached
        ]

        assert reached == [MAX_CONSECUTIVE_LOST_CYCLES], f"enforce={enforce}"


def test_the_bound_is_reported_again_for_a_second_run_of_losses():
    """Once per run, not once per motion: a client that recovers can lose it again.

    Unlike the violation latch, which is a terminal state the client has to
    recover from, this is a statement about one gap. The counter walks up one at
    a time and any answered cycle sends it back to zero, so it stands on the
    bound exactly once per run with no latch needed.
    """
    tracker = CommConstraintTracker(enforce=False)
    tracker.start_motion()
    stream(tracker, 3)

    runs = 0
    for _ in range(2):
        for cycle in range(MAX_CONSECUTIVE_LOST_CYCLES + 3):
            runs += 1 if tracker.tick(200 + runs + cycle).bound_reached else 0
        stream(tracker, 5, start_id=900 + runs)

    assert runs == 2


def test_the_violation_triggers_once_and_stays_latched():
    tracker = CommConstraintTracker(enforce=True)
    tracker.start_motion()
    stream(tracker, 3)

    triggers = sum(
        1
        for cycle in range(3 * MAX_CONSECUTIVE_LOST_CYCLES)
        if tracker.tick(cycle).violation_triggered
    )

    assert triggers == 1
    assert tracker.violated is True


def test_without_enforcement_the_losses_are_reported_but_never_abort():
    tracker = CommConstraintTracker(enforce=False)
    tracker.start_motion()
    stream(tracker, SUCCESS_RATE_WINDOW)

    # One tick more than the window, so every cycle it holds is a lost one.
    for cycle in range(SUCCESS_RATE_WINDOW + 1):
        assert tracker.tick(1000 + cycle).violation_triggered is False

    assert tracker.violated is False
    assert tracker.consecutive_lost == SUCCESS_RATE_WINDOW
    assert tracker.success_rate == pytest.approx(0.0)


def test_recovery_clears_the_latch_and_the_window():
    tracker = CommConstraintTracker(enforce=True)
    tracker.start_motion()
    stream(tracker, 3)
    for cycle in range(MAX_CONSECUTIVE_LOST_CYCLES + 1):
        tracker.tick(100 + cycle)
    assert tracker.violated is True

    tracker.recover()

    assert tracker.violated is False
    assert tracker.consecutive_lost == 0
    assert tracker.success_rate == 0.0

    # A fresh motion can trip it again.
    tracker.start_motion()
    stream(tracker, 3, start_id=500)
    triggered = any(
        tracker.tick(600 + cycle).violation_triggered
        for cycle in range(MAX_CONSECUTIVE_LOST_CYCLES + 1)
    )
    assert triggered is True


def test_ending_a_motion_stops_the_accounting():
    tracker = CommConstraintTracker(enforce=True)
    tracker.start_motion()
    stream(tracker, 5)

    tracker.end_motion()

    for cycle in range(5 * MAX_CONSECUTIVE_LOST_CYCLES):
        outcome = tracker.tick(1000 + cycle)
        assert outcome.active is False
        assert outcome.violation_triggered is False
    assert tracker.success_rate == 0.0


def test_a_new_motion_starts_from_an_empty_window():
    tracker = CommConstraintTracker()
    tracker.start_motion()
    stream(tracker, 10)
    for cycle in range(10):
        tracker.tick(100 + cycle)
    assert tracker.success_rate < 1.0

    tracker.start_motion()
    stream(tracker, 5, start_id=500)

    assert tracker.success_rate == pytest.approx(1.0)


def test_a_future_id_is_not_an_answer_to_anything():
    """The freshness window is closed at the top as well as the bottom.

    An id the server has never published cannot be answering a published state:
    a bit-flip, a replay, a client whose counter is simply its own. Counted as
    fresh, such a client suppresses the loss accounting forever.
    """
    tracker = CommConstraintTracker()
    tracker.start_motion()
    tracker.tick(7)

    assert tracker.command_received(8) is False
    assert tracker.command_received(10_000) is False

    outcome = tracker.tick(8)
    assert outcome.lost is True


def test_a_horizon_widens_the_window_downwards_only():
    tracker = CommConstraintTracker(freshness_horizon=2)
    tracker.start_motion()
    tracker.tick(10)

    assert tracker.command_received(9) is True
    assert tracker.command_received(11) is False


def test_a_second_violation_in_the_same_session_still_fires():
    """The latch is re-armed by the next motion, not only by recovery.

    It used to survive :meth:`start_motion`, so enforcement fired at most once
    per *connection*: a second run of lost cycles in a later motion went
    unpunished for the rest of the client's life.
    """
    tracker = CommConstraintTracker(enforce=True)

    fired = []
    for motion in range(2):
        tracker.start_motion(motion_id=motion + 1)
        stream(tracker, 3, start_id=1000 * (motion + 1))
        fired.append(
            any(
                tracker.tick(1000 * (motion + 1) + 100 + cycle).violation_triggered
                for cycle in range(MAX_CONSECUTIVE_LOST_CYCLES + 1)
            )
        )

    assert fired == [True, True]


def test_an_end_for_a_motion_that_is_over_is_ignored():
    """A stale ``motion_generation_finished`` must not disarm a fresh motion.

    The UDP thread can reach a finish datagram after the TCP thread has already
    started the next motion. Ending *that* motion pinned the success rate at 0.0
    and left enforcement dead for the rest of the connection.
    """
    tracker = CommConstraintTracker(enforce=True)
    tracker.start_motion(motion_id=1)
    stream(tracker, 5)

    tracker.start_motion(motion_id=2)
    stream(tracker, 5, start_id=500)

    tracker.end_motion(motion_id=1)  # the stale one

    assert tracker.active is True
    assert tracker.success_rate == pytest.approx(1.0)

    tracker.end_motion(motion_id=2)
    assert tracker.active is False


def test_an_unnamed_end_still_ends_whatever_is_running():
    """The teardown paths (disconnect, StopMove, socket error) name no motion."""
    tracker = CommConstraintTracker()
    tracker.start_motion(motion_id=7)
    stream(tracker, 5)

    tracker.end_motion()

    assert tracker.active is False


def test_the_outcome_names_the_motion_it_belongs_to():
    """So a late abort can tell whether the motion that violated is still on."""
    tracker = CommConstraintTracker(enforce=True)
    tracker.start_motion(motion_id=42)
    stream(tracker, 3)

    outcome = tracker.tick(100)

    assert outcome.motion_id == 42


def test_an_answer_that_arrives_after_its_cycle_closed_is_late():
    """The tick/sendto boundary is charged to the client, and stays charged.

    Crediting it back was tried and removed: nothing in cycle space can tell a
    packet that missed its window by microseconds from one sent by a client
    running a whole cycle behind, and crediting the boundary credited both. The
    residue is a sub-cycle bias in the conservative direction -- the sim reports
    slightly worse communication than the client achieved, never better.
    """
    tracker = CommConstraintTracker()
    tracker.start_motion()
    tracker.tick(6)
    tracker.command_received(6)

    # Cycle 6 answered; cycle 7 opens and goes unanswered.
    assert tracker.tick(7).lost is False
    outcome = tracker.tick(8)
    assert outcome.lost is True

    # The answer to state 7 turns up now. Too late: cycle 7 is closed and cycle
    # 8 is not the cycle it answers.
    assert tracker.command_received(7) is False
    assert tracker.tick(9).lost is True


def test_a_run_broken_by_an_answered_cycle_never_reaches_the_violation():
    """Only *consecutive* losses count, so nine at a time can never add up.

    The counter has to be reset by an answered cycle, not nudged: decrementing
    it let a client that never loses more than a handful in a row accumulate its
    way to the 20-cycle violation anyway.
    """
    tracker = CommConstraintTracker(enforce=True)
    tracker.start_motion()
    stream(tracker, 3, start_id=1)

    triggered = False
    runs = []
    for cycle in range(4, 80):
        outcome = tracker.tick(cycle)
        triggered = triggered or outcome.violation_triggered
        runs.append(outcome.consecutive_lost)
        if cycle % 10 == 0:
            tracker.command_received(cycle)

    assert triggered is False, "a run of at most nine losses tripped the violation"
    assert max(runs) <= 10, "the run kept growing through answered cycles"


def test_a_client_permanently_one_cycle_behind_is_charged_every_cycle():
    """One cycle late is late, however narrowly it misses.

    Such a client answers state N during cycle N+1, and it must not be able to
    read 1.00 for ever and sit out of reach of the violation -- that is exactly
    the staleness the two-sided freshness bound exists to catch.
    """
    tracker = CommConstraintTracker(enforce=True)
    tracker.start_motion()
    tracker.tick(1)
    tracker.command_received(0)

    triggered = False
    for cycle in range(2, 40):
        outcome = tracker.tick(cycle)
        triggered = triggered or outcome.violation_triggered
        tracker.command_received(cycle - 1)

    assert tracker.success_rate == pytest.approx(0.0)
    assert triggered is True


def test_a_duplicate_answer_credits_nothing():
    """A second answer to the same state is not an answer to the next one."""
    tracker = CommConstraintTracker()
    tracker.start_motion()
    tracker.tick(6)
    tracker.command_received(6)
    tracker.tick(7)  # cycle 6 closes, answered

    # The client answers state 6 a second time. Cycle 7 is still unanswered.
    assert tracker.command_received(6) is False

    assert tracker.tick(8).lost is True


def test_the_success_rate_counter_tracks_the_window_it_summarises():
    """The rate is kept incrementally now; it must still equal the long way."""
    tracker = CommConstraintTracker(window=8)
    tracker.start_motion()
    tracker.command_received(0)

    # The cycle a tick closes is the one answered *before* it, so the expected
    # window lags the loop by one.
    pending = True
    closed = []
    for cycle in range(1, 40):
        tracker.tick(cycle)
        closed.append(pending)
        pending = bool(cycle % 3)
        if pending:
            tracker.command_received(cycle)
        window = closed[-8:]
        assert tracker.success_rate == pytest.approx(sum(window) / len(window))


def test_the_env_var_is_the_switch_and_it_is_off_by_default():
    assert enforcement_enabled_by_env({}) is False
    assert enforcement_enabled_by_env({"FRANKA_SIM_ENFORCE_COMM_CONSTRAINTS": "1"}) is True
    assert enforcement_enabled_by_env({"FRANKA_SIM_ENFORCE_COMM_CONSTRAINTS": "0"}) is False
    assert enforcement_enabled_by_env({"FRANKA_SIM_ENFORCE_COMM_CONSTRAINTS": "off"}) is False


def test_only_an_affirmative_value_enables_enforcement():
    """An allow-list, not a deny-list: ``=disabled`` used to mean *enabled*."""
    for value in ("disabled", "no", "false", "off, really", "nope", " "):
        assert enforcement_enabled_by_env({"FRANKA_SIM_ENFORCE_COMM_CONSTRAINTS": value}) is False
    for value in ("1", "true", "TRUE", " yes ", "on", "enabled"):
        assert enforcement_enabled_by_env({"FRANKA_SIM_ENFORCE_COMM_CONSTRAINTS": value}) is True


def test_the_cli_flags_are_tri_state_and_can_force_enforcement_off():
    """``--no-enforce-...`` is the only way out of an exported environment variable.

    Without it the flag could only ever turn enforcement *on*, so a run inside a
    shell, launch file or container that exports the variable had no override.
    """
    from franka_sim.run_server import build_parser, comm_constraints_setting, motion_limits_setting

    parser = build_parser()

    assert comm_constraints_setting(parser.parse_args([])) is None
    assert comm_constraints_setting(parser.parse_args(["--enforce-comm-constraints"])) is True
    assert comm_constraints_setting(parser.parse_args(["--no-enforce-comm-constraints"])) is False

    assert motion_limits_setting(parser.parse_args([])) is None
    assert motion_limits_setting(parser.parse_args(["--enforce-motion-limits"])) is True
    assert motion_limits_setting(parser.parse_args(["--no-enforce-motion-limits"])) is False


def test_an_explicit_off_beats_the_environment(monkeypatch, mock_physics_sim):
    from franka_sim.franka_sim_server import FrankaSimServer

    monkeypatch.setenv("FRANKA_SIM_ENFORCE_COMM_CONSTRAINTS", "1")

    assert FrankaSimServer(
        physics_sim=mock_physics_sim, enable_gripper=False
    ).enforce_comm_constraints is True
    assert FrankaSimServer(
        physics_sim=mock_physics_sim, enable_gripper=False, enforce_comm_constraints=False
    ).enforce_comm_constraints is False


def test_message_ids_start_at_one():
    """A state carrying id 0 cannot be answered: the server drops such a command.

    ``_handle_commands`` treats ``message_id == 0`` as unparsable, so the first
    published state used to be unanswerable by construction -- and the client
    was charged a lost cycle for it.
    """
    state = RobotState()

    assert state.state["message_id"] == 1
    state.update()
    assert state.state["message_id"] == 2


def test_the_tracker_survives_two_threads_hammering_it():
    """The publish thread ticks while the receive thread answers, for real.

    Invariants that must hold at every observation, whatever the interleaving:
    the success rate is a probability, the consecutive-lost counter never
    exceeds the number of cycles that have closed, and the window never grows
    past its bound.
    """
    tracker = CommConstraintTracker(window=32, enforce=False)
    tracker.start_motion(motion_id=1)
    stop = threading.Event()
    failures = []
    published = [0]

    def publish():
        try:
            cycles = 0
            while not stop.is_set():
                cycles += 1
                published[0] = cycles
                outcome = tracker.tick(cycles)
                assert 0.0 <= outcome.success_rate <= 1.0
                assert outcome.consecutive_lost <= cycles
        except Exception as error:  # pragma: no cover - reported below
            failures.append(error)

    def answer():
        try:
            while not stop.is_set():
                tracker.command_received(published[0])
                assert 0.0 <= tracker.success_rate <= 1.0
                assert tracker.consecutive_lost >= 0
        except Exception as error:  # pragma: no cover - reported below
            failures.append(error)

    threads = [threading.Thread(target=publish), threading.Thread(target=answer)]
    for thread in threads:
        thread.start()
    time.sleep(1.0)
    stop.set()
    for thread in threads:
        thread.join(timeout=5.0)
        assert not thread.is_alive()

    assert not failures, failures
    assert published[0] > 100, "the publish thread barely ran; the test proves nothing"
    assert 0.0 <= tracker.success_rate <= 1.0


# --- shared wire helpers -----------------------------------------------------


def pack_robot_command(
    message_id, q_c=None, dq_c=None, o_dp_ee_c=None, tau_j_d=None, motion_finished=False
):
    """Pack the UDP RobotCommand exactly as libfranka sends it."""
    message = struct.pack("<Q", message_id)
    message += struct.pack("<7d", *(q_c if q_c is not None else [0.0] * 7))
    message += struct.pack("<7d", *(dq_c if dq_c is not None else [0.0] * 7))
    message += struct.pack("<16d", *([0.0] * 16))
    message += struct.pack("<6d", *(o_dp_ee_c if o_dp_ee_c is not None else [0.0] * 6))
    message += struct.pack("<2d", *([0.0] * 2))
    message += struct.pack("<B", 0)
    message += struct.pack("<B", 1 if motion_finished else 0)
    message += struct.pack("<7d", *(tau_j_d if tau_j_d is not None else [0.0] * 7))
    message += struct.pack("<B", 0)
    return message


def robot_state_field_slice(field, length):
    """Locate a field inside the packed 1377-byte RobotState, without magic offsets."""
    probe = RobotState()
    sentinel = [float(1000 + index) for index in range(length)]
    probe.state[field] = sentinel
    values = _ROBOT_STATE_PACKER.unpack(probe.pack_state())
    start = values.index(sentinel[0])
    return slice(start, start + length)


MESSAGE_ID_INDEX = 0
Q_D_SLICE = robot_state_field_slice("q_d", 7)
DQ_SLICE = robot_state_field_slice("dq", 7)
DQ_D_SLICE = robot_state_field_slice("dq_d", 7)
TAU_J_D_SLICE = robot_state_field_slice("tau_J_d", 7)
ERRORS_SLICE = slice(-84, -43)
REFLEX_REASON_SLICE = slice(-43, -2)
ROBOT_MODE_INDEX = -2
SUCCESS_RATE_INDEX = -1

#: Command id every motion in this file is started with, and therefore the id
#: its terminal ``Move`` response comes back on.
MOTION_COMMAND_ID = 2

#: Total wire size of a plain status response: the 12-byte header plus
#: ``<B3x>``, one status byte and three of padding. Both ``kSetGuidingMode``
#: (``server/set_commands.py``) and the terminal ``kMove``
#: (``server/motion_session.py``) answer in that shape.
STATUS_RESPONSE_SIZE = 12 + 4


def recv_exactly(sock, count):
    """Read exactly ``count`` bytes off ``sock``, or fail saying how short it ran.

    ``recv`` may legally return fewer bytes than asked for, and a reader that
    treats a short read as a whole frame manufactures a desynchronisation of
    its own -- which is the exact thing the tests here are trying to detect.
    """
    chunks = bytearray()
    while len(chunks) < count:
        piece = sock.recv(count - len(chunks))
        assert piece, f"the TCP stream ended after {len(chunks)} of {count} bytes"
        chunks += piece
    return bytes(chunks)


class WireClient:
    """A libfranka-shaped client: every state is answered, echoing its message_id.

    The echo is the whole point of the freshness rule under test --
    ``Robot::Impl::sendRobotCommand`` stamps each command with the id of the
    last accepted state -- so this client streams from its own receive loop
    rather than on a wall clock, exactly as libfranka's control loop does.
    """

    def __init__(self, host="127.0.0.1", port=COMMAND_PORT):
        self.host = host
        self.port = port
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.settimeout(5.0)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.settimeout(5.0)
        self.udp.bind((host, 0))
        self.udp_port = self.udp.getsockname()[1]
        self.server_udp_address = None
        self.last_message_id = 0
        #: Every ``message_id`` this client has echoed. The server publishes on
        #: its own 1 kHz thread, so "one state per :meth:`read_state`" is only
        #: true while this single-threaded client keeps up -- on a loaded
        #: machine :meth:`read_state`'s drain skips whole cycles. The ids it did
        #: answer are the only exact record of the loss it injected; see
        #: :func:`injected_loss`.
        self.answered_ids = set()

    # -- TCP ---------------------------------------------------------------

    def connect(self):
        self.tcp.connect((self.host, self.port))
        payload = struct.pack("<HH", 10, self.udp_port)
        header = MessageHeader(command=Command.kConnect, command_id=1, size=12 + len(payload))
        self.tcp.sendall(header.to_bytes() + payload)
        self.tcp.recv(12)
        status, _ = struct.unpack("<BH", self.tcp.recv(3))
        assert status == ConnectStatus.kSuccess

    def move(self, controller_mode, motion_generator_mode, command_id=MOTION_COMMAND_ID):
        """Send a Move and consume the kMotionStarted response."""
        payload = struct.pack(
            "<II3d3d",
            controller_mode.value,
            motion_generator_mode.value,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
        )
        header = MessageHeader(command=Command.kMove, command_id=command_id, size=12 + len(payload))
        self.tcp.sendall(header.to_bytes() + payload)
        self.tcp.recv(12)
        return struct.unpack("<B3x", self.tcp.recv(4))[0]

    def read_move_response(self, timeout=5.0):
        """Read the next Move response status byte off the TCP stream."""
        self.tcp.settimeout(timeout)
        header = MessageHeader.from_bytes(self.tcp.recv(12))
        assert header.command == Command.kMove
        return struct.unpack("<B3x", self.tcp.recv(4))[0]

    def automatic_error_recovery(self, command_id=9):
        header = MessageHeader(Command.kAutomaticErrorRecovery, command_id, 12)
        self.tcp.sendall(header.to_bytes())
        self.tcp.recv(12)
        return struct.unpack("<B3x", self.tcp.recv(4))[0]

    # -- UDP ---------------------------------------------------------------

    def read_state(self):
        """Return the newest published state, draining any backlog first.

        Real libfranka reads UDP state on its own background thread, always
        draining at ~1 kHz regardless of what the calling thread is blocked on
        -- including a pending TCP call such as ``automaticErrorRecovery()``.
        This test client is single-threaded, so a wait on the server side (see
        ``FrankaSimServer._wait_for_standstill``) lets several state datagrams
        pile up unread while this client is blocked in
        :meth:`automatic_error_recovery`. UDP is FIFO, so a plain ``recvfrom``
        would return the *oldest* of those instead of the current state.
        Draining whatever is already buffered and keeping the newest is what
        makes this stand-in behave like the background thread it replaces.
        """
        newest = None
        while True:
            readable, _, _ = select.select([self.udp], [], [], 0)
            if not readable:
                break
            newest = self.udp.recvfrom(4096)
        if newest is None:
            newest = self.udp.recvfrom(4096)
        data, address = newest
        self.server_udp_address = address
        assert len(data) == _ROBOT_STATE_PACKER.size
        values = _ROBOT_STATE_PACKER.unpack(data)
        self.last_message_id = values[MESSAGE_ID_INDEX]
        return values

    def answer(self, **command_fields):
        """Send one command echoing the last state's id, as libfranka does."""
        self.answered_ids.add(self.last_message_id)
        self.udp.sendto(
            pack_robot_command(self.last_message_id, **command_fields),
            self.server_udp_address,
        )

    def stream(self, cycles, **command_fields):
        """Read ``cycles`` states, answering each one. Returns the last state."""
        state = None
        for _ in range(cycles):
            state = self.read_state()
            self.answer(**command_fields)
        return state

    def drop(self, cycles):
        """Keep reading states but send nothing: exactly ``cycles`` lost cycles."""
        state = None
        for _ in range(cycles):
            state = self.read_state()
        return state

    def kill(self):
        self.tcp.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        self.tcp.close()

    def close(self):
        for sock in (self.tcp, self.udp):
            try:
                sock.close()
            except OSError:
                pass


def wait_for_server(port, timeout=5.0):
    """Block until the FCI accept loop answers on ``port``."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            probe = socket.create_connection(("127.0.0.1", port), timeout=1.0)
            probe.close()
            return True
        except OSError:
            time.sleep(0.05)
    return False


# --- layer 2: the server over the wire, with a mocked simulator --------------


@pytest.fixture
def serve(mock_physics_sim):
    """Start a FrankaSimServer with a mocked simulator, enforcement as asked."""
    from franka_sim.franka_sim_server import FrankaSimServer

    started = []

    def _serve(enforce=False, sim=None, mobile_base=False):
        server = FrankaSimServer(
            physics_sim=sim if sim is not None else mock_physics_sim,
            enable_gripper=False,
            mobile_base=mobile_base,
            enforce_comm_constraints=enforce,
        )
        thread = threading.Thread(target=server.run_server, daemon=True)
        thread.start()
        assert wait_for_server(COMMAND_PORT), "the FCI server never came up"
        started.append((server, thread))
        return server

    yield _serve

    for server, thread in started:
        server.stop()
        thread.join(timeout=3.0)
    time.sleep(0.4)


@pytest.fixture
def client():
    """A wire client, closed however the test ends."""
    clients = []

    def _client(**kwargs):
        made = WireClient(**kwargs)
        clients.append(made)
        return made

    yield _client

    for made in clients:
        made.close()


TORQUES = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.25]


def start_torque_motion(client_factory):
    """Connect, start an external-controller motion, and stream it into shape."""
    wire = client_factory()
    wire.connect()
    assert wire.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone) == (
        MoveStatus.kMotionStarted
    )
    # No second Move response follows kMotionStarted: Move gets exactly one
    # immediate reply, and the terminal one (kSuccess/abort) only arrives
    # once the motion actually ends -- it does not, here.
    wire.stream(SUCCESS_RATE_WINDOW, tau_j_d=TORQUES)
    return wire


#: How many of the cycles a client *did* answer may still be charged to it as
#: lost before the accounting is considered broken. ``command_received``
#: documents the bias deliberately: an answer that lands in the microseconds
#: between :meth:`CommConstraintTracker.tick` and the ``sendto`` is late by the
#: tracker's reckoning, and costs its own cycle as well as the one it lands in.
#: That race is a photo finish on an idle machine and a real gap on a loaded
#: one (CI runs these on two shared cores), so the *lower* bound on the
#: reported rate has to leave room for a handful of them. It stays far away
#: from the failure this bounds: an answered cycle counted as lost anyway would
#: charge every one of the ~90 answered cycles in the window, not ten.
LATE_ANSWER_SLACK_CYCLES = 10

#: Half a cycle's worth of float32: ``control_command_success_rate`` crosses the
#: wire as a 4-byte float, so 0.89 comes back as 0.8899999857 and reading the
#: loss back out of it lands a rounding error away from a whole number.
CYCLE_EPSILON = 0.01


def injected_loss(wire, state):
    """Cycles ``wire`` really failed to answer inside the window ``state`` reports on.

    ``CommConstraintTracker.tick`` closes the cycle the *previous* state opened,
    so the rate carried by the state with ``message_id`` F is the verdict on
    cycles ``F - SUCCESS_RATE_WINDOW .. F - 1`` -- one cycle per published id.
    The client answered exactly the ids in :attr:`WireClient.answered_ids`, so
    counting the rest is the client's own arithmetic about its own packet loss,
    which is what the reported rate has to agree with.

    Deliberately not "the number of ``drop`` calls": that only equals the loss
    while the client keeps up with the server's 1 kHz publish thread. On a
    loaded machine it falls behind, genuinely loses extra cycles, and the sim is
    right to charge for them -- so the expected value has to be measured, not
    assumed.
    """
    newest = state[MESSAGE_ID_INDEX]
    window = range(newest - SUCCESS_RATE_WINDOW, newest)
    # Every cycle in the window has to be one the tracker was counting, or the
    # denominator below is not SUCCESS_RATE_WINDOW (the window is divided by
    # what it actually holds while it fills).
    assert window.start >= min(wire.answered_ids), "the window reaches back past the motion"
    return sum(1 for message_id in window if message_id not in wire.answered_ids)


def charged_loss(state):
    """Cycles the server is reporting as lost, read back out of the success rate."""
    return (1.0 - state[SUCCESS_RATE_INDEX]) * SUCCESS_RATE_WINDOW


def assert_loss_is_accounted_once(wire, state, at_least):
    """The reported rate is the client's own packet-loss arithmetic, and nothing else.

    Two bounds, and they fail for different reasons:

    * ``charged >= injected`` -- exact, no slack. Every cycle the client did not
      answer is charged for. This is the one that catches a substituted
      command being mistaken for an answer: the sim would report a client's
      packet loss back to it as perfect communication, and ``charged`` would
      collapse towards zero while ``injected`` stayed where it was.
    * ``charged <= injected + slack`` -- the other direction, a lost cycle
      counted more than once (or an answered one counted as lost), with
      :data:`LATE_ANSWER_SLACK_CYCLES` of room for the documented boundary bias.
    """
    injected = injected_loss(wire, state)
    assert injected >= at_least, "the client did not inject the loss the test meant to"
    charged = charged_loss(state)
    assert charged >= injected - CYCLE_EPSILON, (
        f"the sim charged {charged:.2f} of the {injected} cycles the client never answered: "
        "an unanswered cycle was counted as a successful command"
    )
    assert charged <= injected + LATE_ANSWER_SLACK_CYCLES, (
        f"the sim charged {charged:.2f} cycles for {injected} lost ones: loss counted twice"
    )


def test_a_streaming_client_reports_a_perfect_success_rate(serve, client):
    """The baseline: answer every state in its own cycle and nothing is lost."""
    serve()
    wire = start_torque_motion(client)

    state = wire.stream(SUCCESS_RATE_WINDOW, tau_j_d=TORQUES)

    # Same one-cycle slack every other over-the-wire success-rate assertion
    # carries: the publisher is soft real time, so a state can go out between
    # the client's recv and its send and cost the window one cycle. The default
    # 1e-6 relative tolerance was strict enough to flake on exactly that.
    assert state[SUCCESS_RATE_INDEX] == pytest.approx(1.0, abs=0.02)


def test_the_success_rate_falls_on_a_gap_and_climbs_back(serve, client):
    """The wire signal a controller is told to watch actually moves."""
    serve()
    wire = start_torque_motion(client)

    lost = 30
    wire.drop(lost)
    after_gap = wire.stream(1, tau_j_d=TORQUES)

    # One cycle of slack either way: the publisher is soft real time, so a
    # state can go out between the client's recv and its send.
    expected = (SUCCESS_RATE_WINDOW - lost) / SUCCESS_RATE_WINDOW
    assert after_gap[SUCCESS_RATE_INDEX] == pytest.approx(expected, abs=0.03)

    recovered = wire.stream(SUCCESS_RATE_WINDOW + 5, tau_j_d=TORQUES)
    assert recovered[SUCCESS_RATE_INDEX] == pytest.approx(1.0, abs=0.02)


def test_a_gap_never_aborts_the_motion_by_default(serve, client):
    """Enforcement is opt-in: the losses are reported, the motion runs on."""
    server = serve(enforce=False)
    wire = start_torque_motion(client)

    wire.drop(5 * MAX_CONSECUTIVE_LOST_CYCLES)
    state = wire.stream(1, tau_j_d=TORQUES)

    assert state[SUCCESS_RATE_INDEX] < 0.5
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove
    assert not any(state[ERRORS_SLICE])
    assert server.comm.violated is False


def test_a_lost_torque_cycle_reuses_the_last_torque(serve, client):
    """FCI reuses the torques of the last successfully received packet.

    The controller half of the packet-handling rule, and the half that does
    *not* extrapolate: a torque is not a waypoint, so there is nothing to
    integrate and the last one simply stays applied. Pinned by count as well as
    by value -- the simulator must not be commanded at all on a lost cycle, or a
    future change to the motion-generator extrapolation could quietly start
    driving torque through gaps too.
    """
    server = serve()
    wire = start_torque_motion(client)
    # Two silent cycles first, so the last streamed answer is certainly through
    # the receive path before the count is taken -- otherwise this asserts
    # against a datagram still in flight rather than against the gap.
    wire.drop(2)
    commands_before = server.physics_sim.update_torques.call_count

    state = wire.drop(3 * MAX_CONSECUTIVE_LOST_CYCLES)

    assert np.array(state[TAU_J_D_SLICE]) == pytest.approx(TORQUES, abs=1e-5)
    assert np.array(server.physics_sim.update_torques.call_args.args[0]) == pytest.approx(TORQUES)
    # The idle hold writes zero torque once when the enforced abort fires; this
    # server is unenforced, so nothing at all should have been written.
    assert server.physics_sim.update_torques.call_count == commands_before


def test_a_missed_cycle_extrapolates_the_commanded_position(serve, client):
    """A gap continues the waypoint stream, and the wire says so.

    The real FCI "takes the previous waypoints and performs a linear
    extrapolation (keep acceleration constant and integrate) for the missed time
    step" (``docs/system_requirements.rst``), and what it extrapolated is what
    comes back in ``q_d`` -- "even in case of packet losses"
    (``docs/overview.rst``). This sim held instead, for a long time, and the
    divergence is now gone.

    The stream is a clean 1 rad/s ramp (one millirad per cycle) whose commanded
    acceleration is zero, so the frozen-acceleration extrapolation is a plain
    arithmetic progression at exactly the same millirad per cycle -- and it runs
    for :data:`MAX_CONSECUTIVE_LOST_CYCLES` - 1 cycles before the bound stops it
    and the reference holds. Both halves are asserted: a runaway would overshoot
    the plateau, and the old hold behaviour never leaves the last commanded
    value at all.
    """
    server = serve()
    wire = client()
    wire.connect()
    wire.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)

    step = 0.001
    for waypoint in range(1, 21):
        wire.read_state()
        wire.answer(q_c=[waypoint * step] * 7)
    last_commanded = 20 * step
    # One cycle short of the bound is the last one extrapolated: the bound
    # itself is where the robot stops, and so does this.
    plateau = last_commanded + (MAX_CONSECUTIVE_LOST_CYCLES - 1) * step

    # Long enough to cover the bound several times over, so a runaway of any
    # size would show up in the plateau.
    cycles = 3 * MAX_CONSECUTIVE_LOST_CYCLES
    published = [np.array(wire.drop(1)[Q_D_SLICE]).mean() for _ in range(cycles)]

    assert published == sorted(published), "the reference must never go backwards"
    assert published[0] < published[-1], "a gap must not freeze the reference"
    # A cycle of slack: the publisher is soft real time, so the client's last
    # answer can land a cycle either side of the state it was racing, which
    # moves the whole progression by at most one step.
    assert published[-1] == pytest.approx(plateau, abs=1.5 * step)
    assert max(published) == pytest.approx(plateau, abs=1.5 * step)
    # ...and the arm was driven through the gap, not left behind at the last
    # real waypoint. Physics receives the extrapolated targets, exactly as it
    # receives the commanded ones.
    applied = np.asarray(server.physics_sim.update_joint_positions.call_args.args[0])
    assert applied == pytest.approx([published[-1]] * 7, abs=1e-9)


def test_the_extrapolation_stops_at_the_bound_and_the_violation_latches(serve, client):
    """Twenty missed cycles: the reference holds, the reflex fires, the arm is recaptured.

    The previous attempt at this feature left the state *unsafe* past the bound
    -- it kept integrating, and a client that had simply gone away took the arm
    with it. Termination is therefore pinned explicitly here: extrapolation
    stops, the last reference holds, and the existing communication-constraints
    latch and abort machinery runs exactly as it did before, unchanged.
    """
    server = serve(enforce=True)
    wire = client()
    wire.connect()
    wire.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)

    step = 0.001
    for waypoint in range(1, 21):
        wire.read_state()
        wire.answer(q_c=[waypoint * step] * 7)

    assert wire.read_move_response() == MoveStatus.kReflexAborted
    state = wire.read_state()

    assert state[ERRORS_SLICE][VIOLATION] == 1
    assert state[REFLEX_REASON_SLICE][VIOLATION] == 1
    assert state[ROBOT_MODE_INDEX] == RobotMode.kReflex
    assert server.comm.violated is True
    # The reference stopped where the bound left it, and the arm is back under
    # the internal controller's hold rather than still flying along the ramp.
    published = np.array(state[Q_D_SLICE]).mean()
    assert published <= 20 * step + MAX_CONSECUTIVE_LOST_CYCLES * step + 1e-9
    assert server.control_mode is ControlMode.POSITION
    assert server.physics_sim.update_torques.call_args.args[0] == [0.0] * 7

    # ...and it stays put: nothing publishes another waypoint after the abort.
    settled = [np.array(wire.drop(1)[Q_D_SLICE]).mean() for _ in range(10)]
    assert settled == pytest.approx([settled[0]] * len(settled), abs=1e-9)


def test_an_unenforced_bound_holds_the_reference_without_aborting(serve, client, caplog):
    """Off by default, the sim reports the bound and holds -- it does not stop the motion.

    The same two-flag contract every other emulated reflex has: the *behaviour*
    (extrapolate, then hold) is always on because it is what the robot's
    reference does, while the *abort* is opt-in. What a client running without
    ``--enforce-comm-constraints`` gets instead is the log line.
    """
    server = serve(enforce=False)
    wire = client()
    wire.connect()
    wire.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)

    step = 0.001
    for waypoint in range(1, 21):
        wire.read_state()
        wire.answer(q_c=[waypoint * step] * 7)

    with caplog.at_level(logging.WARNING, logger="franka_sim.franka_sim_server"):
        settled = [
            np.array(wire.drop(1)[Q_D_SLICE]).mean()
            for _ in range(3 * MAX_CONSECUTIVE_LOST_CYCLES)
        ]

    assert server.comm.violated is False
    assert not any(wire.read_state()[ERRORS_SLICE])
    # The tail is flat: the bound stopped the extrapolation even with nothing
    # enforcing anything.
    assert settled[-10:] == pytest.approx([settled[-1]] * 10, abs=1e-9)
    assert any("no longer extrapolating" in record.getMessage() for record in caplog.records)


def test_extrapolation_does_not_double_count_the_communication_accounting(serve, client):
    """A substituted waypoint is not an answer, and the success rate must not think so.

    The one wire-level way this feature could have gone quietly wrong: the
    extrapolated command flows through the same dispatch path a received
    datagram does, and if it also reached the tracker the sim would report a
    client's packet loss back to it as perfect communication. The rate here is
    computed from the client's own arithmetic -- it dropped ``lost`` of the last
    hundred cycles -- and has to match whether or not anything was extrapolated
    into those cycles.
    """
    serve()
    wire = client()
    wire.connect()
    wire.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)

    step = 0.001
    for waypoint in range(1, SUCCESS_RATE_WINDOW + 1):
        wire.read_state()
        wire.answer(q_c=[waypoint * step] * 7)
    before = wire.read_state()

    lost = 10
    state = wire.drop(lost + 1)

    assert np.array(state[Q_D_SLICE]).mean() > np.array(before[Q_D_SLICE]).mean()
    assert_loss_is_accounted_once(wire, state, at_least=lost)


def test_a_missed_cycle_still_costs_the_success_rate(serve, client):
    """Holding the command is not the same as pretending the packet arrived."""
    serve()
    wire = client()
    wire.connect()
    wire.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kJointPosition)

    for waypoint in range(1, SUCCESS_RATE_WINDOW + 1):
        wire.read_state()
        wire.answer(q_c=[waypoint * 0.001] * 7)

    dropped = 10
    state = wire.drop(dropped + 1)

    assert_loss_is_accounted_once(wire, state, at_least=dropped)


def test_the_violation_aborts_the_motion_with_a_reflex(serve, client):
    """Enforced: error bits, kReflex, the kReflexAborted Move status, idle hold."""
    server = serve(enforce=True)
    wire = start_torque_motion(client)

    wire.drop(MAX_CONSECUTIVE_LOST_CYCLES + 5)
    status = wire.read_move_response()
    state = wire.read_state()

    assert status == MoveStatus.kReflexAborted
    # errors -> current_errors, reflex_reason -> last_motion_errors
    # (convertRobotState in libfranka's src/robot_impl.cpp).
    assert state[ERRORS_SLICE][VIOLATION] == 1
    assert state[REFLEX_REASON_SLICE][VIOLATION] == 1
    assert sum(state[ERRORS_SLICE]) == 1
    assert state[ROBOT_MODE_INDEX] == RobotMode.kReflex
    assert state[MESSAGE_ID_INDEX] > 0
    # Modes leave kMove, which is how libfranka's throwOnMotionError notices.
    assert server.robot_state.state["motion_generator_mode"] == LibfrankaMotionGeneratorMode.kIdle
    assert server.robot_state.state["controller_mode"] == LibfrankaControllerMode.kOther
    # ...and the arm is back under the internal controller's hold.
    assert server.control_mode is ControlMode.POSITION
    assert server.physics_sim.update_torques.call_args.args[0] == [0.0] * 7


def test_the_success_rate_reads_zero_once_the_motion_is_aborted(serve, client):
    """No loop is running any more, and the robot reports zero then."""
    serve(enforce=True)
    wire = start_torque_motion(client)

    wire.drop(MAX_CONSECUTIVE_LOST_CYCLES + 5)
    wire.read_move_response()

    assert wire.read_state()[SUCCESS_RATE_INDEX] == pytest.approx(0.0)


def test_error_recovery_clears_the_violation_and_a_new_move_runs(serve, client):
    """Automatic error recovery is what the abort leaves the client to do."""
    server = serve(enforce=True)
    wire = start_torque_motion(client)
    wire.drop(MAX_CONSECUTIVE_LOST_CYCLES + 5)
    assert wire.read_move_response() == MoveStatus.kReflexAborted

    assert wire.automatic_error_recovery() == 0

    cleared = wire.read_state()
    assert not any(cleared[ERRORS_SLICE])
    assert cleared[ROBOT_MODE_INDEX] == RobotMode.kIdle
    # last_motion_errors is a record of what aborted the previous motion, so
    # recovery must not erase it (libfranka include/franka/robot_state.h).
    assert cleared[REFLEX_REASON_SLICE][VIOLATION] == 1
    assert server.comm.violated is False

    assert (
        wire.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone, command_id=11)
        == MoveStatus.kMotionStarted
    )
    state = wire.stream(SUCCESS_RATE_WINDOW, tau_j_d=TORQUES)
    assert state[SUCCESS_RATE_INDEX] == pytest.approx(1.0, abs=0.02)
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove


def test_the_base_role_is_held_to_the_same_constraints(serve, client, mock_base_sim):
    """A base twist is a motion command: the base is accounted for like an arm."""
    server = serve(enforce=True, sim=mock_base_sim, mobile_base=True)
    twist = [0.25, -0.1, 0.0, 0.0, 0.0, 0.4]
    wire = client()
    wire.connect()
    wire.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)
    wire.stream(30, o_dp_ee_c=twist)

    # A held twist keeps the base rolling through a short gap...
    wire.drop(5)
    assert np.array(mock_base_sim.update_base_twist.call_args.args[0]) == pytest.approx(twist)

    # ...until the violation, whose stop for a base is a zero twist.
    wire.drop(MAX_CONSECUTIVE_LOST_CYCLES + 5)
    assert wire.read_move_response() == MoveStatus.kReflexAborted
    assert np.array(mock_base_sim.update_base_twist.call_args.args[0]) == pytest.approx(np.zeros(6))
    assert server.robot_state.state["errors"][VIOLATION] is True
    mock_base_sim.update_joint_positions.assert_not_called()


def test_the_base_role_reports_the_swerve_joints_it_is_driving(serve, client, mock_base_sim):
    """The base bridge's q_d / dq_d must track the wheels, not freeze at frame one.

    The commanded-field filter that keeps a backend from echoing measured values
    into an arm's q_d/dq_d has no business on the base: nothing commands those
    fields there (the client commands a twist), so filtering the backend out left
    q_d stuck at the first frame and dq_d/ddq_d permanently zero on the very
    bridge the teleop reads.
    """
    turning = iter(range(1, 10_000))

    def rolling_state():
        step = next(turning) * 0.01
        wheels = [step, -step, step, -step, 0.0, 0.0, 0.0]
        speeds = [1.0, -1.0, 1.0, -1.0, 0.0, 0.0, 0.0]
        return {
            "q": wheels,
            "dq": speeds,
            "ddq": [0.0] * 7,
            "q_d": wheels,
            "dq_d": speeds,
            "ddq_d": [0.0] * 7,
            "tau_J": [0.0] * 7,
            "O_T_EE": [0.0] * 16,
        }

    mock_base_sim.get_robot_state.side_effect = rolling_state
    serve(sim=mock_base_sim, mobile_base=True)
    wire = client()
    wire.connect()
    wire.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)

    twist = [0.25, -0.1, 0.0, 0.0, 0.0, 0.4]
    first = wire.stream(5, o_dp_ee_c=twist)
    later = wire.stream(20, o_dp_ee_c=twist)

    first_q_d = np.array(first[Q_D_SLICE])
    later_q_d = np.array(later[Q_D_SLICE])
    assert not np.allclose(first_q_d, later_q_d), "q_d froze on the base bridge"
    assert np.array(later[DQ_D_SLICE])[:4] == pytest.approx([1.0, -1.0, 1.0, -1.0])


def test_a_move_is_refused_while_a_violation_is_latched(serve, client):
    """The robot does not start a motion out of kReflex; nor may the sim.

    Accepted, the Move left the state lying: robot_mode back to kMove with the
    violation still latched in errors -- a combination the robot never produces.
    """
    server = serve(enforce=True)
    wire = start_torque_motion(client)
    wire.drop(MAX_CONSECUTIVE_LOST_CYCLES + 5)
    assert wire.read_move_response() == MoveStatus.kReflexAborted

    refused = wire.move(
        ControllerMode.kExternalController, MotionGeneratorMode.kNone, command_id=21
    )

    assert refused == MoveStatus.kCommandNotPossibleRejected
    assert server.robot_state.state["robot_mode"] == RobotMode.kReflex
    assert server.robot_state.state["errors"][VIOLATION] is True

    # ...and after the recovery the robot has to accept one.
    assert wire.automatic_error_recovery(command_id=22) == 0
    assert (
        wire.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone, command_id=23)
        == MoveStatus.kMotionStarted
    )


def test_two_violations_in_one_connection_both_abort(serve, client):
    """The latch is re-armed by the next accepted Move, not only by the connection.

    It used to survive start_motion, so the second silence in a session was never
    punished: the client got no kReflexAborted and no error bits at all.
    """
    serve(enforce=True)
    wire = start_torque_motion(client)

    aborts = []
    for attempt in range(2):
        wire.drop(MAX_CONSECUTIVE_LOST_CYCLES + 5)
        aborts.append(wire.read_move_response())
        assert wire.automatic_error_recovery(command_id=30 + attempt) == 0
        assert (
            wire.move(
                ControllerMode.kExternalController,
                MotionGeneratorMode.kNone,
                command_id=40 + attempt,
            )
            == MoveStatus.kMotionStarted
        )
        wire.stream(5, tau_j_d=TORQUES)

    assert aborts == [MoveStatus.kReflexAborted, MoveStatus.kReflexAborted]


def test_the_state_carrying_the_error_beats_the_tcp_response(serve, client):
    """The abort is visible in a state before kReflexAborted goes out.

    That is the order libfranka needs: ``throwOnMotionError`` keys off
    ``robot_mode != kMove`` in the state and only afterwards blocks for the TCP
    response. Latching the error into the struct is not enough -- the publish
    loop latches, *then* packs, *then* sends, so a response written from the
    abort itself left the machine first.
    """
    serve(enforce=True)
    wire = start_torque_motion(client)

    wire.udp.settimeout(1.0)
    saw_error_state = False
    status = None
    for _ in range(6 * MAX_CONSECUTIVE_LOST_CYCLES):
        if select.select([wire.tcp], [], [], 0)[0]:
            status = wire.read_move_response()
            break
        state = wire.read_state()
        if state[ERRORS_SLICE][VIOLATION] == 1:
            saw_error_state = True

    assert status == MoveStatus.kReflexAborted
    assert saw_error_state, "kReflexAborted arrived before any state carried the error"


def test_a_move_while_a_motion_runs_preempts_it(serve, client):
    """Move::Status::kPreempted is what the displaced motion is answered with.

    And the displacement is a real one: the robot is recaptured first, so the new
    motion is seeded from a standstill instead of inheriting the old motion's
    difference history.
    """
    server = serve()
    wire = start_torque_motion(client)

    assert (
        wire.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone, command_id=50)
        == MoveStatus.kPreempted
    )
    assert wire.read_move_response() == MoveStatus.kMotionStarted
    assert server.robot_state.state["robot_mode"] == RobotMode.kMove


def test_a_stale_finish_datagram_cannot_disarm_the_next_motion(serve, client):
    """The UDP thread can reach a finish after the TCP thread started the next Move.

    Acting on it ended a motion that had barely begun: accounting off, success
    rate pinned at 0.0, enforcement dead, and the new Move answered kSuccess
    before it ran.
    """
    server = serve(enforce=True)
    wire = start_torque_motion(client)
    stale_id = wire.last_message_id
    wire.drop(3)  # let the publish loop move past the id the leftover echoes

    # Preempt without letting the finish through first, then deliver it late.
    assert (
        wire.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone, command_id=60)
        == MoveStatus.kPreempted
    )
    assert wire.read_move_response() == MoveStatus.kMotionStarted
    wire.udp.sendto(
        pack_robot_command(stale_id, tau_j_d=TORQUES, motion_finished=True),
        wire.server_udp_address,
    )

    state = wire.stream(SUCCESS_RATE_WINDOW, tau_j_d=TORQUES)

    assert server.comm.active is True, "the stale finish switched the accounting off"
    assert state[SUCCESS_RATE_INDEX] > 0.5
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove


def test_a_leftover_finish_from_a_normal_completion_cannot_disarm_the_next_motion(
    serve, client
):
    """The common case, and the one the guard used to miss entirely.

    Motion 1 finishes perfectly normally. Its 1 kHz finish burst is still
    draining out of the socket when motion 2's Move is accepted, and the
    leftover datagram then ended a motion that had barely begun -- answered
    kSuccess before it ran, its commands swallowed by the idle hold, the tracker
    switched off and the success rate pinned at 0.0. Requiring the running
    motion to have *preempted* an unfinished one excluded exactly this path.
    """
    server = serve(enforce=True)
    wire = start_torque_motion(client)
    leftover_id = wire.last_message_id

    # Motion 1 finishes cleanly, the way a client actually ends a motion.
    wire.answer(tau_j_d=TORQUES, motion_finished=True)
    assert wire.read_move_response() == MoveStatus.kSuccess
    wire.drop(3)

    # Motion 2 starts, and only then does the burst's leftover datagram drain.
    assert (
        wire.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone, command_id=70)
        == MoveStatus.kMotionStarted
    )
    wire.udp.sendto(
        pack_robot_command(leftover_id, tau_j_d=TORQUES, motion_finished=True),
        wire.server_udp_address,
    )

    state = wire.stream(SUCCESS_RATE_WINDOW, tau_j_d=TORQUES)

    assert server.comm.active is True, "the leftover finish switched the accounting off"
    assert state[SUCCESS_RATE_INDEX] > 0.5
    assert state[ROBOT_MODE_INDEX] == RobotMode.kMove
    assert np.array(server.physics_sim.update_torques.call_args.args[0]) == pytest.approx(TORQUES)


def test_a_motion_that_finishes_without_streaming_is_still_answered(serve, client):
    """The stale-finish guard must not strand a client that finishes at once.

    A client whose whole motion fits inside one publish cycle echoes the id the
    Move was accepted at and has sent no control command. Both halves of the
    guard's pattern -- and it must still get its kSuccess, or it blocks on the
    TCP reply for ever.
    """
    serve()
    wire = client()
    wire.connect()
    assert wire.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone) == (
        MoveStatus.kMotionStarted
    )

    wire.read_state()
    wire.answer(tau_j_d=TORQUES, motion_finished=True)

    assert wire.read_move_response() == MoveStatus.kSuccess


def test_the_abort_never_hits_a_motion_that_already_replaced_it(serve, client):
    """Three threads, one motion session: the abort must name the motion it aborts.

    The abort is raised on the publish thread, the finish on the UDP thread and
    the next Move on the TCP thread, all reaching for current_motion_id. Looped,
    because the window is microseconds wide.
    """
    server = serve(enforce=True)
    wire = start_torque_motion(client)

    for attempt in range(25):
        # Drive right up to the abort, then start a new motion in the same breath.
        wire.drop(MAX_CONSECUTIVE_LOST_CYCLES - 1)
        wire.udp.sendto(
            pack_robot_command(wire.last_message_id, tau_j_d=TORQUES, motion_finished=True),
            wire.server_udp_address,
        )
        wire.move(
            ControllerMode.kExternalController, MotionGeneratorMode.kNone, command_id=100 + attempt
        )
        wire.stream(5, tau_j_d=TORQUES)

        # Whatever happened, the invariant holds: a latched violation and a
        # running motion are never both true.
        latched = server.comm.violated or server.robot_state.state["errors"][VIOLATION]
        running = server.robot_state.state["robot_mode"] == RobotMode.kMove
        assert not (latched and running), f"attempt {attempt}: kMove with a latched reflex"

        if latched:
            assert wire.automatic_error_recovery(command_id=200 + attempt) == 0
            assert (
                wire.move(
                    ControllerMode.kExternalController,
                    MotionGeneratorMode.kNone,
                    command_id=300 + attempt,
                )
                in (MoveStatus.kMotionStarted, MoveStatus.kPreempted)
            )
            wire.stream(5, tau_j_d=TORQUES)


def test_concurrent_tcp_responses_never_interleave(serve, client):
    """23 sendall sites, three threads: one send lock or a desynchronised stream.

    The FCI's TCP stream has no resynchronisation marker, so a sendall
    interrupted halfway does not lose a message, it corrupts every frame after
    it. Hammered with TCP commands while the UDP and publish threads are both
    live, every frame must still parse.

    The pump replays one frozen ``message_id``, so every cycle of this motion is
    lost and the violation trips about twenty publish cycles in. That is the
    point rather than an accident: the abort's terminal ``Move`` response is
    sent from the *publish* thread while the TCP thread is answering
    ``SetGuidingMode``, and two threads writing one socket is exactly what the
    send lock exists for. The hammering therefore runs until that response has
    been read, so the second sender is guaranteed to have been in the stream
    rather than merely likely to be.

    What is pinned is **byte-level** non-interleaving. A whole, well-formed
    response to a command this client still has outstanding does not violate
    that -- it is the second sender doing its job, and it is what a loaded CI
    runner surfaced by letting the abort land mid-loop. So the reader consumes
    it, by command *and* command id, taking its payload length from its own
    header. Anything that is not a complete response to a known outstanding
    command still fails the test, and now nothing except real stream corruption
    can produce one.
    """
    serve(enforce=True)
    wire = start_torque_motion(client)

    stop = threading.Event()

    def keep_streaming():
        while not stop.is_set():
            try:
                wire.udp.sendto(
                    pack_robot_command(wire.last_message_id, tau_j_d=TORQUES),
                    wire.server_udp_address,
                )
            except OSError:  # pragma: no cover - socket closed by teardown
                return
            # Five times the control rate: enough that the server's UDP thread
            # is genuinely contending, without one unthrottled Python spinner
            # holding the GIL away from the three threads this test needs live
            # (CI runs it on two shared cores).
            time.sleep(0.0002)

    def read_response():
        """One complete response off the stream: header, then its own payload."""
        head = recv_exactly(wire.tcp, 12)
        try:
            reply = MessageHeader.from_bytes(head)
        except ValueError as error:  # pragma: no cover - only on real corruption
            raise AssertionError(f"the TCP stream desynchronised: {head.hex()} ({error})")
        assert reply.size == STATUS_RESPONSE_SIZE, f"the TCP stream desynchronised: {reply}"
        return reply, recv_exactly(wire.tcp, reply.size - 12)

    pump = threading.Thread(target=keep_streaming, daemon=True)
    pump.start()
    try:
        wire.tcp.settimeout(5.0)
        aborted = None
        command_id = 400
        deadline = time.time() + 30.0
        while command_id < 440 or aborted is None:
            assert time.time() < deadline, "the starved motion never sent its Move response"
            header = MessageHeader(Command.kSetGuidingMode, command_id, 12 + 7)
            wire.tcp.sendall(header.to_bytes() + struct.pack("<6?B", *([False] * 6), False))
            while True:
                reply, payload = read_response()
                if reply.command == Command.kSetGuidingMode:
                    assert reply.command_id == command_id, f"desynchronised: {reply}"
                    break
                # The only other response outstanding on this stream, and only
                # ever once: the starved motion's terminal Move.
                assert (reply.command, reply.command_id) == (
                    Command.kMove,
                    MOTION_COMMAND_ID,
                ), f"the TCP stream desynchronised: unsolicited {reply}"
                assert aborted is None, "one motion answered twice"
                aborted = struct.unpack("<B3x", payload)[0]
            command_id += 1
    finally:
        stop.set()
        pump.join(timeout=2.0)

    assert aborted == MoveStatus.kReflexAborted


# --- layer 3: end to end over real physics -----------------------------------

mujoco = pytest.importorskip("mujoco")

from franka_sim.franka_sim_server import FrankaSimServer  # noqa: E402
from franka_sim.mujoco_franka_sim import MujocoFrankaSim, default_fr3_mjcf  # noqa: E402

try:
    FR3_MJCF = default_fr3_mjcf()
except Exception:  # pragma: no cover - depends on the host's cache/network
    FR3_MJCF = None


@pytest.fixture
def live_server():
    """A real FrankaSimServer over a real MuJoCo arm, enforcement as asked."""
    if FR3_MJCF is None or not FR3_MJCF.exists():
        pytest.skip("the MuJoCo Menagerie FR3 model is neither cached nor downloadable")

    made = []

    def _live(enforce=False):
        sim = MujocoFrankaSim()
        sim.initialize_simulation()
        server = FrankaSimServer(
            physics_sim=sim, enable_gripper=False, enforce_comm_constraints=enforce
        )
        accept_thread = threading.Thread(target=server.run_server, daemon=True)
        accept_thread.start()
        sim.running = True
        physics_thread = threading.Thread(target=sim.run_simulation, daemon=True)
        physics_thread.start()
        assert wait_for_server(COMMAND_PORT), "the FCI server never came up on port 1337"
        made.append((server, sim, accept_thread, physics_thread))
        return server

    yield _live

    for server, sim, accept_thread, physics_thread in made:
        server.stop()
        sim.stop()
        physics_thread.join(timeout=3.0)
        accept_thread.join(timeout=3.0)
    time.sleep(0.4)


def test_dropping_n_cycles_shows_up_as_n_percent_of_loss(live_server, client):
    """The reported rate is real accounting, over real physics and a real socket."""
    live_server()
    wire = client()
    wire.connect()
    wire.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone)

    torques = [0.0] * 7
    full = wire.stream(SUCCESS_RATE_WINDOW + 20, tau_j_d=torques)
    assert full[SUCCESS_RATE_INDEX] == pytest.approx(1.0, abs=0.03)

    lost = 25
    wire.drop(lost)
    state = wire.stream(1, tau_j_d=torques)

    expected = (SUCCESS_RATE_WINDOW - lost) / SUCCESS_RATE_WINDOW
    assert state[SUCCESS_RATE_INDEX] == pytest.approx(expected, abs=0.04)


def test_a_long_silence_stops_the_arm_and_recovery_brings_it_back(live_server, client):
    """The whole cycle: violation, hold, AutomaticErrorRecovery, a fresh motion."""
    server = live_server(enforce=True)
    wire = client()
    wire.connect()
    wire.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone)

    # Get the arm genuinely moving, then go silent.
    spin_up = [8.0, -8.0, 8.0, -8.0, 4.0, -4.0, 2.0]
    moving = wire.stream(400, tau_j_d=spin_up)
    assert np.abs(np.array(moving[DQ_SLICE])).max() > 0.2, "the arm never got moving"

    wire.drop(MAX_CONSECUTIVE_LOST_CYCLES + 10)

    assert wire.read_move_response() == MoveStatus.kReflexAborted
    state = wire.read_state()
    assert state[ERRORS_SLICE][VIOLATION] == 1
    assert state[ROBOT_MODE_INDEX] == RobotMode.kReflex

    # The idle hold catches the arm, exactly as any other session end does.
    deadline = time.time() + 3.0
    while time.time() < deadline:
        state = wire.read_state()
        if np.abs(np.array(state[DQ_SLICE])).max() < 0.05:
            break
    assert np.abs(np.array(state[DQ_SLICE])).max() < 0.05, "the arm is still moving after the abort"

    assert wire.automatic_error_recovery() == 0
    # Move gets exactly one immediate reply, kMotionStarted; the terminal
    # response only arrives once this motion actually ends.
    assert (
        wire.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone, command_id=12)
        == MoveStatus.kMotionStarted
    )
    resumed = wire.stream(SUCCESS_RATE_WINDOW + 10, tau_j_d=[0.0] * 7)

    assert not any(resumed[ERRORS_SLICE])
    assert resumed[ROBOT_MODE_INDEX] == RobotMode.kMove
    assert resumed[SUCCESS_RATE_INDEX] == pytest.approx(1.0, abs=0.03)
    assert server.comm.violated is False
