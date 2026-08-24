import socket
import struct
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from franka_sim.franka_genesis_sim import ControlMode
from franka_sim.franka_protocol import (
    COMMAND_PORT,
    Command,
    ConnectStatus,
    ControllerMode,
    LibfrankaMotionGeneratorMode,
    MessageHeader,
    MotionGeneratorMode,
    MoveStatus,
)

BASE_TWIST = [0.25, -0.1, 0.0, 0.0, 0.0, 0.4]

#: Wall-clock budget for one TCP exchange with the server, drain included. A
#: test that outlives it has hit a bug, not a slow box.
TCP_DEADLINE_S = 5.0


def recv_exactly(sock, size, deadline):
    """Read exactly ``size`` bytes before ``deadline``, or fail the test.

    A bare ``sock.recv(n)`` is allowed to return fewer bytes than asked for, so
    reading a 12-byte header with one call is a latent desync: everything after
    a short read is parsed off the wrong offset. And with no deadline a drain
    loop waiting for a reply that will never come hangs the whole run instead of
    reporting a failure.
    """
    original = sock.gettimeout()
    chunks, remaining = [], size
    try:
        while remaining:
            budget = deadline - time.monotonic()
            if budget <= 0:
                pytest.fail(f"timed out reading {size} bytes from the FCI TCP socket")
            sock.settimeout(budget)
            try:
                chunk = sock.recv(remaining)
            except (socket.timeout, TimeoutError):
                pytest.fail(f"timed out reading {size} bytes from the FCI TCP socket")
            if not chunk:
                pytest.fail("the FCI TCP socket closed mid-message")
            chunks.append(chunk)
            remaining -= len(chunk)
    finally:
        sock.settimeout(original)
    return b"".join(chunks)


def perform_handshake(tcp_client):
    """Connect handshake (libfranka v10): version + UDP port in, status + version out."""
    tcp_client.connect(("localhost", COMMAND_PORT))
    payload = struct.pack("<HH", 10, 1338)
    header = MessageHeader(command=Command.kConnect, command_id=1, size=12 + len(payload))
    tcp_client.sendall(header.to_bytes() + payload)
    deadline = time.monotonic() + TCP_DEADLINE_S
    recv_exactly(tcp_client, 12, deadline)
    status, _ = struct.unpack("<BH", recv_exactly(tcp_client, 3, deadline))
    return status == ConnectStatus.kSuccess


def send_move(tcp_client, controller_mode, motion_generator_mode, command_id=2):
    """Send a Move command and drain TCP through this command's own reply.

    Move gets exactly one reply per command id: kMotionStarted, sent the
    moment the Move is accepted. The *terminal* response (kSuccess via
    StopMove or a motion-finished datagram, or an abort status) only arrives
    once the motion actually ends -- so it is not read here. A previous
    motion's terminal response can still be unread on the socket when this
    fires; it is drained and discarded since it does not match `command_id`.

    The drain has a deadline: a reply for `command_id` that never arrives is a
    bug to report, not a reason to hang the suite.
    """
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
    tcp_client.sendall(header.to_bytes() + payload)
    deadline = time.monotonic() + TCP_DEADLINE_S
    while True:
        response_header = MessageHeader.from_bytes(recv_exactly(tcp_client, 12, deadline))
        recv_exactly(tcp_client, 4, deadline)  # status (1 byte) + padding (3 bytes)
        if response_header.command_id == command_id:
            break


def pack_robot_command(message_id=1, o_dp_ee_c=None, tau_j_d=None, motion_finished=False):
    """Pack the UDP RobotCommand exactly as libfranka sends it."""
    o_dp_ee_c = o_dp_ee_c if o_dp_ee_c is not None else [0.0] * 6
    tau_j_d = tau_j_d if tau_j_d is not None else [0.0] * 7
    message = struct.pack("<Q", message_id)
    message += struct.pack("<7d", *([0.0] * 7))  # q_c
    message += struct.pack("<7d", *([0.0] * 7))  # dq_c
    message += struct.pack("<16d", *([0.0] * 16))  # O_T_EE_c
    message += struct.pack("<6d", *o_dp_ee_c)  # O_dP_EE_c
    message += struct.pack("<2d", *([0.0] * 2))  # elbow_c
    message += struct.pack("<B", 0)  # valid_elbow
    message += struct.pack("<B", 1 if motion_finished else 0)  # motion_generation_finished
    message += struct.pack("<7d", *tau_j_d)  # tau_J_d
    message += struct.pack("<B", 0)  # torque_command_finished
    return message


def wait_for_udp_socket(server, timeout=2.0, poll_interval=0.01):
    """Block until ``server.udp_socket`` exists *and* is bound to a real port.

    It is created on the broadcast thread, and -- since nothing in this
    codebase calls ``bind()`` on it, it only ever sends -- stays at port 0
    until the publish loop's first ``sendto()`` implicitly binds it.
    Addressing a datagram to port 0 raises ``OSError: [Errno 22]``.

    Both conditions are waited out by watching ``states_sent``, the count of
    state datagrams the publish loop has actually put on the wire: it can only
    have moved if that loop reached its ``sendto()``, which means the socket
    exists and is bound. Watching the port instead used the bind as a proxy for
    the send -- the weaker of the two signals, and one an explicit ``bind()``
    in the server would have quietly broken.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.states_sent > 0 and server.udp_socket is not None:
            return
        time.sleep(poll_interval)
    raise AssertionError("server's UDP socket never became ready (created and bound)")


def send_robot_command(udp_client, server, **kwargs):
    wait_for_udp_socket(server)
    udp_client.sendto(
        pack_robot_command(**kwargs), ("localhost", server.udp_socket.getsockname()[1])
    )
    time.sleep(0.2)


def test_move_with_cartesian_velocity_selects_steering_drive(
    tcp_client, base_sim_server, mock_base_sim
):
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)

    mock_base_sim.set_control_mode.assert_called_with(ControlMode.STEERING_DRIVE)
    assert base_sim_server.control_mode is ControlMode.STEERING_DRIVE


def test_cartesian_velocity_command_reaches_the_base(
    tcp_client, udp_client, base_sim_server, mock_base_sim
):
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)
    send_robot_command(udp_client, base_sim_server, o_dp_ee_c=BASE_TWIST)

    # assert_any_call, not assert_called_with: this client sends one datagram
    # and then goes quiet, so twenty cycles later the communication-constraints
    # emulation ramps the commanded twist to a safe stop and the *last* call is
    # a decelerating one. What this test is about is that the twist arrived.
    mock_base_sim.update_base_twist.assert_any_call(BASE_TWIST)


def test_cartesian_velocity_is_echoed_in_the_reported_state(
    tcp_client, udp_client, base_sim_server, mock_base_sim
):
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)

    # Snapshot the reported command at the moment it is handed to the base.
    # Reading the state afterwards would read the safe stop instead (see above),
    # and this test is about the echo, not about the stop.
    echoed = []
    state = base_sim_server.robot_state.state
    mock_base_sim.update_base_twist.side_effect = lambda twist: echoed.append(
        (list(state["O_dP_EE_c"]), list(state["O_dP_EE_d"]))
    )

    send_robot_command(udp_client, base_sim_server, o_dp_ee_c=BASE_TWIST)

    assert echoed, "the twist never reached the base"
    assert echoed[0][0] == pytest.approx(BASE_TWIST)
    assert echoed[0][1] == pytest.approx(BASE_TWIST)
    assert state["motion_generator_mode"] == LibfrankaMotionGeneratorMode.kCartesianVelocity.value


def test_motion_finished_zeroes_the_base_twist(
    tcp_client, udp_client, base_sim_server, mock_base_sim
):
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)
    send_robot_command(udp_client, base_sim_server, o_dp_ee_c=BASE_TWIST)
    send_robot_command(udp_client, base_sim_server, message_id=2, motion_finished=True)

    mock_base_sim.update_base_twist.assert_called_with([0.0] * 6)
    mock_base_sim.update_joint_positions.assert_not_called()


def test_cartesian_velocity_is_ignored_by_an_arm_server(
    tcp_client, udp_client, sim_server, mock_genesis_sim
):
    """A non-mobile server must drop the twist, not crash and not move the arm."""
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)
    send_robot_command(udp_client, sim_server, o_dp_ee_c=BASE_TWIST)

    mock_genesis_sim.update_base_twist.assert_not_called()
    assert sim_server.control_mode is not ControlMode.STEERING_DRIVE


def test_torque_dispatch_is_unchanged_on_an_arm_server(
    tcp_client, udp_client, sim_server, mock_genesis_sim
):
    """Regression guard: the existing external-controller path still wins."""
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kExternalController, MotionGeneratorMode.kNone)
    torques = [1.0, -1.0, 1.0, -1.0, 0.5, -0.5, 0.5]
    send_robot_command(udp_client, sim_server, tau_j_d=torques)

    assert sim_server.control_mode is ControlMode.TORQUE
    assert np.allclose(sim_server.robot_state.state["tau_J_d"], torques)


def test_arm_server_ignores_a_cartesian_velocity_move(tcp_client, sim_server, mock_genesis_sim):
    """A cartesian-velocity Move must not switch an arm server's control mode.

    FrankaGenesisSim's physics loop has no STEERING_DRIVE branch -- it would
    silently stop actuating the arm.
    """
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)

    assert sim_server.control_mode is not ControlMode.STEERING_DRIVE
    for call in mock_genesis_sim.set_control_mode.call_args_list:
        assert call.args[0] is not ControlMode.STEERING_DRIVE


def test_external_controller_keeps_torque_even_with_a_cartesian_motion_generator(
    tcp_client, udp_client, base_sim_server, mock_base_sim
):
    """The cartesian branch must not shadow the external-controller torque path."""
    assert perform_handshake(tcp_client)
    send_move(
        tcp_client, ControllerMode.kExternalController, MotionGeneratorMode.kCartesianVelocity
    )
    torques = [2.0, -2.0, 2.0, -2.0, 1.0, -1.0, 1.0]
    send_robot_command(udp_client, base_sim_server, o_dp_ee_c=BASE_TWIST, tau_j_d=torques)

    assert base_sim_server.control_mode is ControlMode.TORQUE
    assert np.allclose(base_sim_server.robot_state.state["tau_J_d"], torques)
    mock_base_sim.update_base_twist.assert_not_called()


# --- motion-finished hold-log latch (mobile path) ---------------------------


def _move_request(command_id, controller_mode=ControllerMode.kJointImpedance):
    """A ready-to-use (header, payload) pair for handle_move_command()."""
    payload = struct.pack(
        "<II3d3d",
        controller_mode.value,
        MotionGeneratorMode.kCartesianVelocity.value,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    )
    header = MessageHeader(command=Command.kMove, command_id=command_id, size=12 + len(payload))
    return header, payload


def test_repeated_motion_finished_holds_log_the_hold_once(base_sim_server, mock_base_sim, caplog):
    """1 kHz hot path: libfranka's finishMotion burst calls this once per
    datagram, but the "commanding zero base twist" log must latch.
    """
    with caplog.at_level("INFO", logger="franka_sim.franka_sim_server"):
        for _ in range(5):
            base_sim_server._switch_to_hold_position()

    hold_logs = [r for r in caplog.records if "commanding zero base twist" in r.message]
    assert len(hold_logs) == 1
    # The actual hold command still goes out on every call -- only the log latches.
    assert mock_base_sim.update_base_twist.call_count == 5


def test_a_new_move_rearms_the_hold_log_latch(base_sim_server, mock_base_sim, caplog):
    header, payload = _move_request(command_id=42)
    client_socket = MagicMock()

    with caplog.at_level("INFO", logger="franka_sim.franka_sim_server"):
        base_sim_server._switch_to_hold_position()
        base_sim_server._switch_to_hold_position()
        base_sim_server.handle_move_command(client_socket, header, payload)
        base_sim_server._switch_to_hold_position()

    hold_logs = [r for r in caplog.records if "commanding zero base twist" in r.message]
    assert len(hold_logs) == 2


def test_hold_position_keeps_the_simulator_mode_in_lockstep(base_sim_server, mock_base_sim):
    """Server and simulator must not disagree about control mode after a hold."""
    base_sim_server._switch_to_hold_position()
    mock_base_sim.set_control_mode.assert_called_with(ControlMode.STEERING_DRIVE)
    assert base_sim_server.control_mode is ControlMode.STEERING_DRIVE


# --- commanded Cartesian fields: O_T_EE_d / O_T_EE_c and the elbow ----------
#
# These four are the FCI layer's, not the physics backend's, exactly like
# q_d/dq_d/ddq_d/tau_J_d (see COMMANDED_STATE_FIELDS). They were a permanent
# identity / zero stub, and that was not harmless: a libfranka Cartesian-pose
# motion generator initialises and holds from ``O_T_EE_d`` -- Franka's own smoke
# helpers open every pose motion with ``std::array<double, 16> cmd =
# state.O_T_EE_d;`` -- so an identity there put the commanded stream ~10 m and a
# full rotation away from the robot and tripped
# ``cartesian_position_motion_generator_start_pose_invalid`` on cycle 0 of every
# pose motion, hiding five of the suite's real Cartesian checks behind it.

#: A measured flange pose that is nothing like identity, column-major as the
#: wire wants it, so "tracks the measured pose" cannot pass by accident.
MEASURED_POSE = [
    0.0, 0.0, -1.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0,
    0.31, -0.02, 0.48, 1.0,
]

#: A different one, standing in for what a client commands.
COMMANDED_POSE = [
    0.0, 0.0, -1.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    1.0, 0.0, 0.0, 0.0,
    0.33, -0.02, 0.50, 1.0,
]

#: Joint vector behind MEASURED_POSE. Only q[2] and q[3] matter here: the elbow
#: *is* (q[2], sign(q[3])) on an FR3.
MEASURED_Q = [0.0, -0.785, 0.21, -2.356, 0.0, 1.571, 0.785]


def _arm_server_with_pose():
    """An unstarted arm server whose backend reports MEASURED_POSE/MEASURED_Q.

    No sockets: everything below drives the publish-loop and UDP-dispatch
    helpers directly, which is the whole of the behaviour under test and keeps
    the pins deterministic.
    """
    from unittest.mock import Mock

    from franka_sim.franka_sim_server import FrankaSimServer

    sim = Mock()
    sim.get_robot_state.return_value = {
        "q": list(MEASURED_Q),
        "dq": [0.0] * 7,
        "tau_J": [0.0] * 7,
        "O_T_EE": list(MEASURED_POSE),
    }
    return FrankaSimServer(enable_vis=False, genesis_sim=sim, enable_gripper=False), sim


def _publish_one_cycle(server, sim):
    """The two publish-loop steps that own the commanded Cartesian fields."""
    sim_state = sim.get_robot_state()
    server._publish_commanded_pose(sim_state)
    server._publish_elbow(sim_state)


def _pose_command(pose, elbow=None, message_id=1):
    """A decoded UDP RobotCommand carrying a pose (and optionally an elbow)."""
    return {
        "message_id": message_id,
        "q_c": tuple([0.0] * 7),
        "dq_c": tuple([0.0] * 7),
        "O_T_EE_c": tuple(pose),
        "O_dP_EE_c": tuple([0.0] * 6),
        "elbow_c": tuple(elbow if elbow is not None else [0.0, 0.0]),
        "valid_elbow": elbow is not None,
        "motion_generation_finished": False,
        "tau_J_d": tuple([0.0] * 7),
        "torque_command_finished": False,
    }


def _arm_pose_motion(server):
    """Put the server in the state a ``kCartesianPosition`` Move leaves behind."""
    server.robot_state.state["motion_generator_mode"] = (
        LibfrankaMotionGeneratorMode.kCartesianPosition.value
    )
    server.robot_state.state["controller_mode"] = 0  # kJointImpedance


def test_the_commanded_pose_reports_the_measured_flange_when_nothing_commands_one():
    """Idle, between motions, under a joint generator: O_T_EE_d/c *are* O_T_EE.

    The internal controller is holding the flange where it is, so the pose it
    reports as commanded is the pose the robot is in -- the Cartesian twin of
    ``q_d`` reporting the hold setpoint.
    """
    server, sim = _arm_server_with_pose()

    _publish_one_cycle(server, sim)

    assert list(server.robot_state.state["O_T_EE_d"]) == pytest.approx(MEASURED_POSE)
    assert list(server.robot_state.state["O_T_EE_c"]) == pytest.approx(MEASURED_POSE)
    # The old stub was a permanent identity, which is what made every pose
    # motion open ~10 m from the robot.
    assert list(server.robot_state.state["O_T_EE_d"]) != pytest.approx(
        [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]
    )


def test_the_commanded_elbow_reports_the_measured_elbow_when_nothing_commands_one():
    """``elbow_c`` used to be a permanent [0, 0]; a zero branch flag is one
    libfranka refuses to even send (``checkElbow``).
    """
    server, sim = _arm_server_with_pose()

    _publish_one_cycle(server, sim)

    expected = [MEASURED_Q[2], -1.0]  # (q[2], sign(q[3])); joint 4 is always negative
    assert list(server.robot_state.state["elbow"]) == pytest.approx(expected)
    assert list(server.robot_state.state["elbow_d"]) == pytest.approx(expected)
    assert list(server.robot_state.state["elbow_c"]) == pytest.approx(expected)


def test_a_pose_motion_echoes_the_commanded_pose_and_elbow():
    """During kCartesianPosition the fields belong to the client's stream.

    Both of them: libfranka reads ``O_T_EE_d`` to build the next command and
    ``O_T_EE_c`` as the reference its command low-pass blends that command with
    (``ControlLoop<CartesianPose>::convertMotion``), so publishing anything else
    drags every command the client sends towards whatever the sim invented.
    """
    server, sim = _arm_server_with_pose()
    _publish_one_cycle(server, sim)
    _arm_pose_motion(server)

    server._dispatch_control_command(_pose_command(COMMANDED_POSE, elbow=[0.42, -1.0]))

    assert list(server.robot_state.state["O_T_EE_c"]) == pytest.approx(COMMANDED_POSE)
    assert list(server.robot_state.state["O_T_EE_d"]) == pytest.approx(COMMANDED_POSE)
    assert list(server.robot_state.state["elbow_c"]) == pytest.approx([0.42, -1.0])
    assert list(server.robot_state.state["elbow_d"]) == pytest.approx([0.42, -1.0])

    # ...and the publish loop must not stomp on the echo the next millisecond.
    _publish_one_cycle(server, sim)
    assert list(server.robot_state.state["O_T_EE_d"]) == pytest.approx(COMMANDED_POSE)
    assert list(server.robot_state.state["elbow_d"]) == pytest.approx([0.42, -1.0])
    # ``elbow`` is *measured* and keeps reporting the arm either way.
    assert list(server.robot_state.state["elbow"]) == pytest.approx([MEASURED_Q[2], -1.0])


def test_a_lost_cycle_holds_the_commanded_pose_rather_than_reverting_it():
    """No command applied this cycle -> the fields hold, exactly as ``q_d`` does."""
    server, sim = _arm_server_with_pose()
    _arm_pose_motion(server)
    server._dispatch_control_command(_pose_command(COMMANDED_POSE))

    for _ in range(5):  # five publish cycles with no command in between
        _publish_one_cycle(server, sim)

    assert list(server.robot_state.state["O_T_EE_d"]) == pytest.approx(COMMANDED_POSE)


def test_the_commanded_pose_snaps_back_to_the_measured_flange_when_the_motion_ends():
    """Idle-hold semantics: the internal controller takes the flange back.

    Every way a motion ends -- a motion-finished datagram, a reflex abort,
    StopMove, a client that simply vanishes -- puts the motion generator back to
    kIdle, which is the single condition this keys off.
    """
    server, sim = _arm_server_with_pose()
    _arm_pose_motion(server)
    server._dispatch_control_command(_pose_command(COMMANDED_POSE, elbow=[0.42, -1.0]))
    assert list(server.robot_state.state["O_T_EE_d"]) == pytest.approx(COMMANDED_POSE)

    server.robot_state.state["motion_generator_mode"] = (
        LibfrankaMotionGeneratorMode.kIdle.value
    )
    _publish_one_cycle(server, sim)

    assert list(server.robot_state.state["O_T_EE_d"]) == pytest.approx(MEASURED_POSE)
    assert list(server.robot_state.state["O_T_EE_c"]) == pytest.approx(MEASURED_POSE)
    assert list(server.robot_state.state["elbow_d"]) == pytest.approx([MEASURED_Q[2], -1.0])
    assert list(server.robot_state.state["elbow_c"]) == pytest.approx([MEASURED_Q[2], -1.0])


def test_a_new_move_stops_echoing_the_previous_motions_pose():
    """A fresh Move commands nothing yet, so the hold pose is what is reported."""
    from unittest.mock import MagicMock

    server, sim = _arm_server_with_pose()
    _arm_pose_motion(server)
    server._dispatch_control_command(_pose_command(COMMANDED_POSE))
    assert server._commanded_pose_echoed is True

    header, payload = _move_request(command_id=7)
    server.handle_move_command(MagicMock(), header, payload)

    assert server._commanded_pose_echoed is False
    _publish_one_cycle(server, sim)
    assert list(server.robot_state.state["O_T_EE_d"]) == pytest.approx(MEASURED_POSE)


def test_the_base_role_publishes_no_commanded_pose_or_elbow(base_sim_server, mock_base_sim):
    """The mobile-duo base bridge is untouched by all of the above.

    Its commanded Cartesian fields are ``O_dP_EE_d``/``O_dP_EE_c`` (a twist),
    its ``O_T_EE`` is dead-reckoned from that twist, and its ``q`` is four
    swerve steer/drive joints with no elbow of any kind. Same role guard the
    rest of the commanded-field ownership uses.
    """
    identity = [1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 1.0]
    state = base_sim_server.robot_state.state
    state["O_T_EE_d"] = list(identity)
    state["O_T_EE_c"] = list(identity)
    state["elbow_d"] = [0.0, 0.0]
    state["elbow_c"] = [0.0, 0.0]

    sim_state = {"q": list(MEASURED_Q), "O_T_EE": list(MEASURED_POSE)}
    base_sim_server._publish_commanded_pose(sim_state)
    base_sim_server._publish_elbow(sim_state)
    base_sim_server._echo_commanded_cartesian(_pose_command(COMMANDED_POSE, elbow=[0.42, -1.0]))

    assert list(state["O_T_EE_d"]) == pytest.approx(identity)
    assert list(state["O_T_EE_c"]) == pytest.approx(identity)
    assert list(state["elbow_d"]) == pytest.approx([0.0, 0.0])
    assert list(state["elbow_c"]) == pytest.approx([0.0, 0.0])
