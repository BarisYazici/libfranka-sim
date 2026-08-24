import logging
import socket
import struct
import threading
import time

import numpy as np
import pytest

from franka_sim.franka_protocol import (
    COMMAND_PORT,
    Command,
    ConnectStatus,
    ControllerMode,
    MessageHeader,
    MotionGeneratorMode,
    MoveCommand,
    MoveStatus,
    RobotMode,
    convert_to_libfranka_controller_mode,
    convert_to_libfranka_motion_mode,
)
from franka_sim.franka_sim_server import (
    AUTOMATIC_ERROR_RECOVERY_POLL_PERIOD,
    AUTOMATIC_ERROR_RECOVERY_SETTLE_CYCLES,
    AUTOMATIC_ERROR_RECOVERY_TIMEOUT,
    FrankaSimServer,
)
from franka_sim.robot_state import _ROBOT_STATE_PACKER, RobotState

logger = logging.getLogger(__name__)


def perform_handshake(tcp_client):
    """Helper function to perform initial handshake (libfranka_new v10)."""
    tcp_client.connect(("localhost", COMMAND_PORT))

    # Send connect message
    version = 10
    udp_port = 1338
    payload = struct.pack("<HH", version, udp_port)

    header = MessageHeader(command=Command.kConnect, command_id=1, size=12 + len(payload))

    tcp_client.sendall(header.to_bytes() + payload)

    # Receive and verify response: header(12) + status(uint8) + version(uint16)
    tcp_client.recv(12)
    response_data = tcp_client.recv(3)
    status, _ = struct.unpack("<BH", response_data)

    return status == ConnectStatus.kSuccess


#: Deadline for every "wait until the server got there" poll in this file.
#:
#: Deliberately far larger than any of these transitions takes: they are all
#: sub-millisecond on the server's own threads, and the wait returns the instant
#: the observable appears, so a generous bound costs a fast machine nothing. It
#: is only ever paid in full by a genuine hang -- which is a real failure worth
#: fifteen seconds -- whereas a tight bound is paid by a *loaded* machine that
#: was going to get there a moment later, which is a false failure worth twenty
#: minutes of CI round-trip. Two shared cores under a parallel job is the case
#: that has to work, not the case that has to be fast.
STATE_TRANSITION_TIMEOUT = 15.0


def wait_until(predicate, timeout=STATE_TRANSITION_TIMEOUT, poll_interval=0.005):
    """Poll ``predicate`` until it holds; True if it did, False at the deadline.

    The counterpart of :func:`wait_for_state_update` for observables that are
    not fields of ``sim_server.robot_state.state`` -- what the simulator mock
    was handed, say. Both exist so that no test in this file has to guess *how
    long* the server needs: a fixed ``time.sleep`` before an assertion encodes
    the speed of the machine it was written on, passes on a quiet laptop and
    fails on a loaded two-core runner that had simply not got there yet.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_interval)
    return predicate()


def wait_for_udp_socket(sim_server, timeout=STATE_TRANSITION_TIMEOUT, poll_interval=0.01):
    """Block until the broadcast thread's ``sim_server.udp_socket`` is usable.

    Two separate races, both on the broadcast thread:

    * the socket object itself is created there, so a test that reads
      ``sim_server.udp_socket`` too early can observe it as still ``None``;
    * that socket is never ``bind()``-ed -- it only sends, so the kernel binds
      it on the first ``sendto()``. Until then ``getsockname()`` reports port
      0, and addressing a datagram to port 0 raises ``OSError: [Errno 22]
      Invalid argument``.

    Both are waited out by watching ``states_sent``, the count of state
    datagrams the publish loop has actually put on the wire: it can only have
    moved if the loop reached its ``sendto()``, which means the socket exists
    *and* has been bound. Watching the port instead read the bind as a proxy
    for the send, which is the weaker signal of the two and the one an explicit
    ``bind()`` in the server would have quietly broken.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if sim_server.states_sent > 0 and sim_server.udp_socket is not None:
            return
        time.sleep(poll_interval)
    raise AssertionError("server's UDP socket never became ready (created and bound)")


def test_move_command(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test sending and handling of Move command with mocked simulator"""
    assert perform_handshake(tcp_client)

    # Create Move command
    move_cmd = MoveCommand(
        controller_mode=ControllerMode.kJointImpedance,
        motion_generator_mode=MotionGeneratorMode.kJointPosition,
        maximum_path_deviation=(0.1, 0.1, 0.1),
        maximum_goal_pose_deviation=(0.1, 0.1, 0.1),
    )

    # Pack command
    payload = struct.pack(
        "<II3d3d",
        move_cmd.controller_mode.value,
        move_cmd.motion_generator_mode.value,
        *move_cmd.maximum_path_deviation,
        *move_cmd.maximum_goal_pose_deviation,
    )

    header = MessageHeader(command=Command.kMove, command_id=2, size=12 + len(payload))

    # Send command
    tcp_client.sendall(header.to_bytes() + payload)

    # Receive motion started response
    response_header_data = tcp_client.recv(12)
    response_header = MessageHeader.from_bytes(response_header_data)
    assert response_header.command == Command.kMove
    assert response_header.command_id == 2

    response_data = tcp_client.recv(4)  # Status (1) + padding (3)
    status = struct.unpack("<B3x", response_data)[0]
    logger.debug(
        f"Received Move response status: {status} (expected {MoveStatus.kMotionStarted.value})"
    )
    assert status == MoveStatus.kMotionStarted.value

    # Wait for state update with proper verification using Libfranka modes
    expected_libfranka_motion_mode = convert_to_libfranka_motion_mode(
        move_cmd.motion_generator_mode
    )
    expected_libfranka_controller_mode = convert_to_libfranka_controller_mode(
        move_cmd.controller_mode
    )

    assert wait_for_state_update(
        sim_server,
        lambda state: (
            state["motion_generator_mode"] == expected_libfranka_motion_mode.value
            and state["controller_mode"] == expected_libfranka_controller_mode.value
        ),
    ), "Failed to receive expected state update"

    # No second Move response follows kMotionStarted: Move gets exactly one
    # reply per libfranka's response-map contract, and the terminal one
    # (kSuccess via StopMove, or an abort status) only comes once the motion
    # actually ends -- this motion never does in this test. A short,
    # non-blocking read confirms nothing unsolicited showed up on the wire.
    tcp_client.settimeout(0.2)
    with pytest.raises(socket.timeout):
        tcp_client.recv(16)

    # Verify simulator interactions
    mock_physics_sim.set_control_mode.assert_called()


def test_no_unsolicited_move_reply_arrives_before_stop_move(
    tcp_client, udp_client, sim_server, mock_physics_sim
):
    """Regression: the broadcast loop must not answer Move a second time.

    Before the fix, the first UDP state datagram of a session was followed
    by an extra ``kSuccess`` Move response sent from the publish loop --
    even though ``handle_move_command`` had already answered the same Move
    with ``kMotionStarted`` over TCP. libfranka never expected that second
    reply: it sat unread in the client's response map keyed by command id,
    and when the motion later aborted, ``Robot::throwOnMotionError`` found
    the stale ``kSuccess`` ahead of the real terminal response and raised
    ``ProtocolException("Unexpected reply to a Move command")`` instead of
    the intended ``ControlException``.

    This drives several UDP state datagrams through the socket (more than
    enough for the old bug to have fired on the first one) and asserts the
    TCP socket stays silent until ``StopMove`` is actually sent -- then
    checks StopMove still gets exactly one terminal Move response.
    """
    # perform_handshake() advertises a fixed, never-bound UDP port (every
    # other test in this file only inspects sim_server's state directly, so
    # it never mattered); this test actually reads the broadcast state
    # datagrams, so udp_client is bound first and its real port advertised
    # instead.
    udp_client.bind(("0.0.0.0", 0))
    udp_port = udp_client.getsockname()[1]

    tcp_client.connect(("localhost", COMMAND_PORT))
    connect_payload = struct.pack("<HH", 10, udp_port)
    connect_header = MessageHeader(
        command=Command.kConnect, command_id=1, size=12 + len(connect_payload)
    )
    tcp_client.sendall(connect_header.to_bytes() + connect_payload)
    tcp_client.recv(12)
    connect_status, _ = struct.unpack("<BH", tcp_client.recv(3))
    assert connect_status == ConnectStatus.kSuccess

    move_cmd = MoveCommand(
        controller_mode=ControllerMode.kExternalController,
        motion_generator_mode=MotionGeneratorMode.kNone,
        maximum_path_deviation=(0.1, 0.1, 0.1),
        maximum_goal_pose_deviation=(0.1, 0.1, 0.1),
    )
    payload = struct.pack(
        "<II3d3d",
        move_cmd.controller_mode.value,
        move_cmd.motion_generator_mode.value,
        *move_cmd.maximum_path_deviation,
        *move_cmd.maximum_goal_pose_deviation,
    )
    header = MessageHeader(command=Command.kMove, command_id=2, size=12 + len(payload))
    tcp_client.sendall(header.to_bytes() + payload)

    # The one response Move is owed immediately.
    response_header = MessageHeader.from_bytes(tcp_client.recv(12))
    assert response_header.command == Command.kMove
    assert response_header.command_id == 2
    status = struct.unpack("<B3x", tcp_client.recv(4))[0]
    assert status == MoveStatus.kMotionStarted.value

    # Read several state datagrams -- N states, well past the first one the
    # old bug piggybacked its bogus reply onto.
    udp_client.settimeout(2.0)
    for _ in range(10):
        udp_client.recvfrom(4096)

    # Nothing unsolicited should have shown up on the TCP socket meanwhile.
    tcp_client.settimeout(0.2)
    with pytest.raises(socket.timeout):
        tcp_client.recv(16)

    # StopMove now gets exactly one terminal Move response, matching this
    # motion's command id.
    stop_header = MessageHeader(command=Command.kStopMove, command_id=3, size=12)
    tcp_client.sendall(stop_header.to_bytes())
    tcp_client.settimeout(5.0)

    stop_response_header = MessageHeader.from_bytes(tcp_client.recv(12))
    assert stop_response_header.command == Command.kStopMove
    tcp_client.recv(4)

    move_response_header = MessageHeader.from_bytes(tcp_client.recv(12))
    assert move_response_header.command == Command.kMove
    assert move_response_header.command_id == 2
    move_status = struct.unpack("<B3x", tcp_client.recv(4))[0]
    assert move_status == MoveStatus.kSuccess.value

    # And nothing further follows that terminal response either.
    tcp_client.settimeout(0.2)
    with pytest.raises(socket.timeout):
        tcp_client.recv(16)


def test_stop_move_command(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test sending and handling of StopMove command with mocked simulator"""
    assert perform_handshake(tcp_client)

    # First send a Move command
    move_cmd = MoveCommand(
        controller_mode=ControllerMode.kJointImpedance,
        motion_generator_mode=MotionGeneratorMode.kJointPosition,
        maximum_path_deviation=(0.1, 0.1, 0.1),
        maximum_goal_pose_deviation=(0.1, 0.1, 0.1),
    )

    payload = struct.pack(
        "<II3d3d",
        move_cmd.controller_mode.value,
        move_cmd.motion_generator_mode.value,
        *move_cmd.maximum_path_deviation,
        *move_cmd.maximum_goal_pose_deviation,
    )

    header = MessageHeader(command=Command.kMove, command_id=2, size=12 + len(payload))

    tcp_client.sendall(header.to_bytes() + payload)

    # Skip the Move command's kMotionStarted response
    tcp_client.recv(16)  # Header (12) + status (1) + padding (3)

    # Wait for move command to be processed
    assert wait_for_state_update(
        sim_server, lambda state: state["robot_mode"] == RobotMode.kMove.value
    ), "Failed to enter move mode"

    # Send StopMove command
    stop_header = MessageHeader(command=Command.kStopMove, command_id=3, size=12)

    tcp_client.sendall(stop_header.to_bytes())

    # Receive both responses
    # First response should be StopMove
    response_header_data = tcp_client.recv(12)
    response_header = MessageHeader.from_bytes(response_header_data)
    assert response_header.command == Command.kStopMove
    assert response_header.command_id == 3

    response_data = tcp_client.recv(4)  # Status (1) + padding (3)
    status = struct.unpack("<B3x", response_data)[0]
    assert status == 0  # Success

    # Wait for robot to enter idle mode
    assert wait_for_state_update(
        sim_server, lambda state: state["robot_mode"] == RobotMode.kIdle.value
    ), "Failed to enter idle mode after stop"

    # Second response should be Move (to break the waiting loop)
    move_response_header_data = tcp_client.recv(12)
    move_response_header = MessageHeader.from_bytes(move_response_header_data)
    assert move_response_header.command == Command.kMove
    assert move_response_header.command_id == 2


def test_second_move_after_stop_move_gets_live_state_on_same_connection(
    tcp_client, udp_client, sim_server, mock_physics_sim, field_indices
):
    """A StopMove ends the motion, not the session -- pinned end to end.

    Regression for ``handle_stop_move_command`` clearing ``transmitting_state``
    (and ``connection_running``): both used to permanently stop the 1 kHz
    publish loop the first time a connection saw a ``StopMove``, and nothing
    but a fresh ``Connect`` ever started it again. That was invisible to
    every other test here because none of them do a second ``Move`` on the
    same connection -- but libfranka's ``ActiveControl`` does exactly that on
    ``cancelMotion()`` -> ``StopMove`` -> ``AutomaticErrorRecovery`` ->
    ``Move``, and the regression made the second ``Move`` receive
    ``kMotionStarted`` over TCP (the command thread does not check either
    flag) and then nothing at all over UDP -- a receive timeout on hardware
    that had, from the client's point of view, just accepted the motion.

    This drives that exact sequence over the real wire: a first motion, a
    ``StopMove``, proof the broadcast keeps going with fresh ``message_id``s
    while idle, a second ``Move`` on the *same* TCP connection, and proof its
    states are live too -- not backlog left over from before ``StopMove``.
    """
    udp_client.bind(("0.0.0.0", 0))
    udp_port = udp_client.getsockname()[1]

    tcp_client.connect(("localhost", COMMAND_PORT))
    connect_payload = struct.pack("<HH", 10, udp_port)
    connect_header = MessageHeader(
        command=Command.kConnect, command_id=1, size=12 + len(connect_payload)
    )
    tcp_client.sendall(connect_header.to_bytes() + connect_payload)
    tcp_client.recv(12)
    connect_status, _ = struct.unpack("<BH", tcp_client.recv(3))
    assert connect_status == ConnectStatus.kSuccess

    send_move = _pack_move_command
    header, payload = send_move(command_id=2)
    tcp_client.sendall(header.to_bytes() + payload)
    status = _recv_move_response_status(tcp_client)
    assert status == MoveStatus.kMotionStarted.value

    udp_client.settimeout(2.0)
    # Drain to the first live datagram, then fully empty the socket's
    # backlog: with the fix, the publish loop never stops, so without this a
    # later read could be satisfied by a state queued before StopMove was
    # even sent -- exactly the trap that made an earlier hand check of this
    # look like it worked when it did not.
    udp_client.recvfrom(4096)
    udp_client.setblocking(False)
    try:
        while True:
            udp_client.recvfrom(4096)
    except BlockingIOError:
        pass
    udp_client.setblocking(True)
    udp_client.settimeout(2.0)

    stop_header = MessageHeader(command=Command.kStopMove, command_id=3, size=12)
    tcp_client.sendall(stop_header.to_bytes())
    stop_response_header = MessageHeader.from_bytes(_recv_exact(tcp_client, 12))
    assert stop_response_header.command == Command.kStopMove
    _recv_exact(tcp_client, 4)
    move_response_header = MessageHeader.from_bytes(_recv_exact(tcp_client, 12))
    assert move_response_header.command == Command.kMove
    assert move_response_header.command_id == 2
    _recv_exact(tcp_client, 4)

    # The stream must keep flowing with idle-mode states -- not stop, and not
    # replay backlog: a batch of fresh datagrams whose message_id genuinely
    # advances, the last one reporting the idle hold. Non-decreasing rather
    # than strictly increasing pairwise -- the manual "final state" send in
    # handle_stop_move_command and the publish loop's own next cycle race to
    # pack/send a state each without a shared lock (the same pattern the
    # motion-finished-on-its-own path already uses alongside a live loop), so
    # two consecutive datagrams can rarely carry the same message_id. What
    # must never happen is the id going nowhere at all, which is what "the
    # loop died" or "this is stale backlog" would look like.
    last_values = _assert_stream_makes_progress(udp_client, field_indices)
    assert last_values[field_indices["motion_generator_mode"]] == 0  # kIdle / kNone
    assert last_values[field_indices["controller_mode"]] == 3  # kOther

    # A second Move on the SAME TCP connection: kMotionStarted over TCP...
    header, payload = send_move(command_id=4)
    tcp_client.sendall(header.to_bytes() + payload)
    tcp_client.settimeout(3.0)
    status = _recv_move_response_status(tcp_client)
    assert status == MoveStatus.kMotionStarted.value

    # ...and it must actually see live UDP state, not a receive timeout.
    _assert_stream_makes_progress(udp_client, field_indices)


def _assert_stream_makes_progress(udp_client, field_indices, count=8):
    """Read ``count`` datagrams and prove the stream is actually live.

    Returns the last datagram's fields. See the caller for why this checks
    overall progress across the batch rather than strict pairwise increase.
    """
    first_values, first_msg_id = _read_state_datagram(udp_client, field_indices)
    last_values = first_values
    last_msg_id = first_msg_id
    for _ in range(count - 1):
        values, msg_id = _read_state_datagram(udp_client, field_indices)
        assert msg_id >= last_msg_id, "message_id went backwards -- not just stalled"
        last_values, last_msg_id = values, msg_id
    assert last_msg_id > first_msg_id, (
        f"message_id never advanced across {count} datagrams "
        f"({first_msg_id} throughout) -- the publish loop looks dead, or this "
        "is stale backlog rather than a live stream"
    )
    return last_values


def _recv_exact(sock, size):
    """Read exactly ``size`` bytes off a TCP stream socket.

    A single ``recv(n)`` is not guaranteed to return all ``n`` bytes even
    when the peer sent them in one ``sendall`` -- TCP has no message
    boundaries -- so this loops the way ``FrankaSimServer.receive_exact``
    itself does on the server side.
    """
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Socket closed before the expected bytes arrived")
        chunks += chunk
    return bytes(chunks)


def _recv_move_response_status(tcp_client):
    """Read one Move response (header + status) and return the status byte."""
    header = MessageHeader.from_bytes(_recv_exact(tcp_client, 12))
    assert header.command == Command.kMove
    return struct.unpack("<B3x", _recv_exact(tcp_client, 4))[0]


def _pack_move_command(command_id):
    move_cmd = MoveCommand(
        controller_mode=ControllerMode.kJointImpedance,
        motion_generator_mode=MotionGeneratorMode.kJointPosition,
        maximum_path_deviation=(0.1, 0.1, 0.1),
        maximum_goal_pose_deviation=(0.1, 0.1, 0.1),
    )
    payload = struct.pack(
        "<II3d3d",
        move_cmd.controller_mode.value,
        move_cmd.motion_generator_mode.value,
        *move_cmd.maximum_path_deviation,
        *move_cmd.maximum_goal_pose_deviation,
    )
    header = MessageHeader(command=Command.kMove, command_id=command_id, size=12 + len(payload))
    return header, payload


def _read_state_datagram(udp_client, field_indices):
    """Read one RobotState datagram, returning (all fields, message_id)."""
    data, _ = udp_client.recvfrom(4096)
    values = _ROBOT_STATE_PACKER.unpack(data)
    return values, values[field_indices["message_id"]]


# Field positions in the flat tuple ``_ROBOT_STATE_PACKER.unpack()`` returns,
# located the same way ``robot_state_field_slice`` does in test_idle_hold.py:
# probe a fresh ``RobotState`` with a distinctive sentinel and find where it
# landed, so this does not hardcode the wire layout by hand. Verified unique
# (not just present) so a sentinel that happens to collide with some other
# field's default value -- 0.0, 1.0 from the identity transforms -- fails
# loudly here instead of silently reading the wrong field in the test above.
def _locate_scalar_field(field, sentinel):
    probe = RobotState()
    probe.state[field] = sentinel
    values = _ROBOT_STATE_PACKER.unpack(probe.pack_state())
    matches = [index for index, value in enumerate(values) if value == sentinel]
    assert len(matches) == 1, (
        f"sentinel for {field!r} was not unique in the packed state: {matches}"
    )
    return matches[0]


@pytest.fixture(scope="session")
def field_indices():
    """Wire-layout field positions this file needs, resolved once per session.

    These used to be resolved by three ``_locate_scalar_field()`` calls sitting
    at module level, which ran at *import* time -- i.e. during pytest
    collection, not inside any test. A wire-layout regression there raised
    during collection, which pytest reports as a collection error rather than
    a normal test failure: the whole module (every test in this file, not just
    the ones that touch this layout) disappears from the run with no
    per-test failure to point at. Resolving them here instead means the same
    regression still fails loudly, on the first test that requests this
    fixture, but as an ordinary assertion failure that does not take the rest
    of the file down with it.
    """
    return {
        "message_id": _locate_scalar_field("message_id", 20260821130000),
        "motion_generator_mode": _locate_scalar_field("motion_generator_mode", 201),
        "controller_mode": _locate_scalar_field("controller_mode", 202),
    }


def test_invalid_move_parameters(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test Move command with invalid parameters using mocked simulator"""
    assert perform_handshake(tcp_client)

    # Create Move command with invalid controller mode
    payload = struct.pack(
        "<II3d3d",  # Changed format to match the actual data
        99,  # Invalid controller mode
        MotionGeneratorMode.kJointPosition.value,
        0.1,
        0.1,
        0.1,  # maximum_path_deviation
        0.1,
        0.1,
        0.1,  # maximum_goal_pose_deviation
    )

    header = MessageHeader(command=Command.kMove, command_id=2, size=12 + len(payload))

    # Send command
    tcp_client.sendall(header.to_bytes() + payload)

    # Receive error response
    response_header_data = tcp_client.recv(12)
    MessageHeader.from_bytes(response_header_data)  # must parse as a header
    response_data = tcp_client.recv(4)
    status = struct.unpack("<B3x", response_data)[0]
    assert status == MoveStatus.kInvalidArgumentRejected


def test_robot_state_updates(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test that robot state updates are correctly transmitted over UDP"""
    assert perform_handshake(tcp_client)

    # Set up mock robot state
    test_state = {
        "q": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        "dq": np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]),
        "tau_J": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
    }
    mock_physics_sim.get_robot_state.return_value = test_state

    # Wait for state update with verification
    assert wait_for_state_update(
        sim_server,
        lambda state: np.allclose(state["q"], test_state["q"])
        and np.allclose(state["dq"], test_state["dq"])
        and np.allclose(state["tau_J"], test_state["tau_J"]),
    ), "Failed to receive expected robot state"

    # Verify that the simulator was called to get state
    mock_physics_sim.get_robot_state.assert_called()


def test_position_control_desired_states(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test that desired joint positions (q_d) are correctly tracked in position control mode"""
    assert perform_handshake(tcp_client)

    # Set up initial robot state
    initial_state = {
        "q": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        "dq": np.zeros(7),
        "tau_J": np.zeros(7),
    }
    mock_physics_sim.get_robot_state.return_value = initial_state

    # Send Move command for position control
    move_cmd = MoveCommand(
        controller_mode=ControllerMode.kJointImpedance,
        motion_generator_mode=MotionGeneratorMode.kJointPosition,
        maximum_path_deviation=(0.1, 0.1, 0.1),
        maximum_goal_pose_deviation=(0.1, 0.1, 0.1),
    )

    payload = struct.pack(
        "<II3d3d",
        move_cmd.controller_mode.value,
        move_cmd.motion_generator_mode.value,
        *move_cmd.maximum_path_deviation,
        *move_cmd.maximum_goal_pose_deviation,
    )

    header = MessageHeader(command=Command.kMove, command_id=2, size=12 + len(payload))

    tcp_client.sendall(header.to_bytes() + payload)

    # Skip the Move command's kMotionStarted response
    tcp_client.recv(16)  # Header (12) + status (1) + padding (3)
    wait_for_udp_socket(sim_server)

    # Send a motion command with desired positions
    desired_positions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    command_msg = struct.pack("<Q", 1)  # message_id
    command_msg += struct.pack("<7d", *desired_positions)  # q_c
    command_msg += struct.pack("<7d", *([0.0] * 7))  # dq_c
    command_msg += struct.pack("<16d", *([0.0] * 16))  # O_T_EE_c
    command_msg += struct.pack("<6d", *([0.0] * 6))  # O_dP_EE_c
    command_msg += struct.pack("<2d", *([0.0] * 2))  # elbow_c
    command_msg += struct.pack("<B", 0)  # valid_elbow
    command_msg += struct.pack("<B", 0)  # motion_generation_finished
    command_msg += struct.pack("<7d", *([0.0] * 7))  # tau_J_d
    command_msg += struct.pack("<B", 0)  # torque_command_finished

    udp_client.sendto(command_msg, ("localhost", sim_server.udp_socket.getsockname()[1]))

    # Verify that q_d was updated to match commanded positions
    assert wait_until(
        lambda: np.allclose(sim_server.robot_state.state["q_d"], desired_positions)
    ), "q_d never echoed the commanded joint positions"


def test_velocity_control_desired_states(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test that desired joint velocities (dq_d) are correctly tracked in velocity control mode"""
    assert perform_handshake(tcp_client)

    # Send Move command for velocity control
    move_cmd = MoveCommand(
        controller_mode=ControllerMode.kJointImpedance,
        motion_generator_mode=MotionGeneratorMode.kJointVelocity,
        maximum_path_deviation=(0.1, 0.1, 0.1),
        maximum_goal_pose_deviation=(0.1, 0.1, 0.1),
    )

    payload = struct.pack(
        "<II3d3d",
        move_cmd.controller_mode.value,
        move_cmd.motion_generator_mode.value,
        *move_cmd.maximum_path_deviation,
        *move_cmd.maximum_goal_pose_deviation,
    )

    header = MessageHeader(command=Command.kMove, command_id=2, size=12 + len(payload))

    tcp_client.sendall(header.to_bytes() + payload)

    # Skip the Move command's kMotionStarted response
    tcp_client.recv(16)
    wait_for_udp_socket(sim_server)

    # Send a motion command with desired velocities
    desired_velocities = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4]
    command_msg = struct.pack("<Q", 1)  # message_id
    command_msg += struct.pack("<7d", *([0.0] * 7))  # q_c
    command_msg += struct.pack("<7d", *desired_velocities)  # dq_c
    command_msg += struct.pack("<16d", *([0.0] * 16))  # O_T_EE_c
    command_msg += struct.pack("<6d", *([0.0] * 6))  # O_dP_EE_c
    command_msg += struct.pack("<2d", *([0.0] * 2))  # elbow_c
    command_msg += struct.pack("<B", 0)  # valid_elbow
    command_msg += struct.pack("<B", 0)  # motion_generation_finished
    command_msg += struct.pack("<7d", *([0.0] * 7))  # tau_J_d
    command_msg += struct.pack("<B", 0)  # torque_command_finished

    # Snapshot dq_d at the moment the command is handed to the simulator. This
    # client sends one datagram and then goes quiet, so twenty cycles later the
    # communication-constraints emulation ramps the commanded velocity to a safe
    # stop; reading the state afterwards would read that ramp, not the echo.
    reported = []
    mock_physics_sim.update_joint_velocities.side_effect = lambda dq: reported.append(
        list(sim_server.robot_state.state["dq_d"])
    )

    udp_client.sendto(command_msg, ("localhost", sim_server.udp_socket.getsockname()[1]))

    # Verify that dq_d was updated to match commanded velocities
    assert wait_until(lambda: bool(reported)), "the velocity command never reached the simulator"
    assert np.allclose(reported[0], desired_velocities)


def test_torque_control_desired_states(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test that desired joint torques (tau_J_d) are correctly tracked in torque control mode"""
    assert perform_handshake(tcp_client)

    # Send Move command for external control (torque mode)
    move_cmd = MoveCommand(
        controller_mode=ControllerMode.kExternalController,
        motion_generator_mode=MotionGeneratorMode.kJointPosition,
        maximum_path_deviation=(0.1, 0.1, 0.1),
        maximum_goal_pose_deviation=(0.1, 0.1, 0.1),
    )

    payload = struct.pack(
        "<II3d3d",
        move_cmd.controller_mode.value,
        move_cmd.motion_generator_mode.value,
        *move_cmd.maximum_path_deviation,
        *move_cmd.maximum_goal_pose_deviation,
    )

    header = MessageHeader(command=Command.kMove, command_id=2, size=12 + len(payload))

    tcp_client.sendall(header.to_bytes() + payload)

    # Skip the Move command's kMotionStarted response
    tcp_client.recv(16)
    wait_for_udp_socket(sim_server)

    # Send a command with desired torques
    desired_torques = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0]
    command_msg = struct.pack("<Q", 1)  # message_id
    command_msg += struct.pack("<7d", *([0.0] * 7))  # q_c
    command_msg += struct.pack("<7d", *([0.0] * 7))  # dq_c
    command_msg += struct.pack("<16d", *([0.0] * 16))  # O_T_EE_c
    command_msg += struct.pack("<6d", *([0.0] * 6))  # O_dP_EE_c
    command_msg += struct.pack("<2d", *([0.0] * 2))  # elbow_c
    command_msg += struct.pack("<B", 0)  # valid_elbow
    command_msg += struct.pack("<B", 0)  # motion_generation_finished
    command_msg += struct.pack("<7d", *desired_torques)  # tau_J_d
    command_msg += struct.pack("<B", 0)  # torque_command_finished

    udp_client.sendto(command_msg, ("localhost", sim_server.udp_socket.getsockname()[1]))

    # Verify that tau_J_d was updated to match commanded torques
    assert wait_until(
        lambda: np.allclose(sim_server.robot_state.state["tau_J_d"], desired_torques)
    ), "tau_J_d never echoed the commanded torques"


def test_initial_desired_states(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test that desired states are correctly initialized

    The arm's position has to be in place *before* the handshake. The broadcast
    loop seeds ``q_d`` from ``sim_state["q"]`` on its very first iteration and
    never again (``first_state_sent`` in ``server/state_stream.py``), and that
    loop is started by the connect handshake -- so a test that connects first
    and sets the simulator's position afterwards is racing the publish thread
    for the seed. Losing that race does not merely delay the assertion, it
    decides it: ``q_d`` is seeded from the *old* value and stays there for the
    rest of the session, so no amount of extra waiting can rescue it. That is
    what CI's two-core runner hit -- ``q_d`` reported as all zeros against a
    ``q`` of 0.1..0.7 -- and it is a property of the ordering, not of the clock,
    so the ordering is what is fixed here.
    """
    # Set up initial robot state -- before connecting; see the docstring.
    initial_state = {
        "q": np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        "dq": np.zeros(7),
        "tau_J": np.zeros(7),
    }
    mock_physics_sim.get_robot_state.return_value = initial_state

    assert perform_handshake(tcp_client)

    # Wait for the first state to actually go out: the seeding happens on the
    # iteration that sends it, so states_sent > 0 is exactly "q_d has been
    # initialised", with no guess about how long that took.
    wait_for_udp_socket(sim_server)

    # Verify that q_d was initialized to match current positions
    assert np.allclose(sim_server.robot_state.state["q_d"], initial_state["q"])

    # Verify that dq_d and tau_J_d start at zero
    assert np.allclose(sim_server.robot_state.state["dq_d"], np.zeros(7))
    assert np.allclose(sim_server.robot_state.state["tau_J_d"], np.zeros(7))


def test_set_collision_behavior(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test SetCollisionBehavior command handling"""
    assert perform_handshake(tcp_client)

    # Test data based on examples_common.cpp setDefaultBehavior
    lower_torque_acc = [20.0] * 7
    upper_torque_acc = [20.0] * 7
    lower_torque_nom = [10.0] * 7
    upper_torque_nom = [10.0] * 7
    lower_force_acc = [20.0] * 6
    upper_force_acc = [20.0] * 6
    lower_force_nom = [10.0] * 6
    upper_force_nom = [10.0] * 6

    # Create command payload
    payload = bytearray()
    payload.extend(struct.pack("<7d", *lower_torque_acc))
    payload.extend(struct.pack("<7d", *upper_torque_acc))
    payload.extend(struct.pack("<7d", *lower_torque_nom))
    payload.extend(struct.pack("<7d", *upper_torque_nom))
    payload.extend(struct.pack("<6d", *lower_force_acc))
    payload.extend(struct.pack("<6d", *upper_force_acc))
    payload.extend(struct.pack("<6d", *lower_force_nom))
    payload.extend(struct.pack("<6d", *upper_force_nom))

    # Create and send command message
    command_id = 1
    header = MessageHeader(Command.kSetCollisionBehavior, command_id, 12 + len(payload))
    message = header.to_bytes() + payload
    tcp_client.sendall(message)

    # Receive and verify response
    response_header_data = tcp_client.recv(12)
    response_header = MessageHeader.from_bytes(response_header_data)
    assert response_header.command == Command.kSetCollisionBehavior
    assert response_header.command_id == command_id
    assert response_header.size == 16  # Header (12) + status (1) + padding (3)

    # Get response status
    response_data = tcp_client.recv(4)  # status (1 byte) + padding (3 bytes)
    status = struct.unpack("<B3x", response_data)[0]
    assert status == 0  # Success


def test_automatic_error_recovery(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test AutomaticErrorRecovery command handling.

    Regression: without a response, libfranka (and franka_hardware, which calls
    automaticErrorRecovery() on activation) blocks forever, stalling the whole
    control stack. The command has an empty request and a single uint8 status.
    """
    assert perform_handshake(tcp_client)

    # AutomaticErrorRecovery has an empty request payload.
    command_id = 1
    header = MessageHeader(Command.kAutomaticErrorRecovery, command_id, 12)
    tcp_client.sendall(header.to_bytes())

    # Receive and verify response. A timeout makes a regression (no reply ->
    # libfranka blocks forever) surface as a fast failure, not a hung suite.
    tcp_client.settimeout(2.0)
    response_header = MessageHeader.from_bytes(tcp_client.recv(12))
    assert response_header.command == Command.kAutomaticErrorRecovery
    assert response_header.command_id == command_id
    assert response_header.size == 16  # Header (12) + status (1) + padding (3)

    # Get response status
    status = struct.unpack("<B3x", tcp_client.recv(4))[0]
    assert status == 0  # Success


def _bare_server(mock_physics_sim):
    """A ``FrankaSimServer`` with ``running`` set, but no socket ever opened.

    ``_wait_for_standstill`` only touches ``self.running``, ``self.mobile_base``
    and ``physics_sim.get_robot_state()`` -- no networking -- so exercising it
    does not need :func:`run_server` or any of the ``sim_server`` fixture's
    real sockets. Going through those anyway (as an earlier version of these
    tests did) hit a spurious ``accept()`` failure in this environment
    part-way through the wait that flips ``running`` False out from under the
    test for reasons unrelated to settling, which is exactly the kind of
    incidental coupling a unit test of one internal method should not have.
    """
    server = FrankaSimServer(enable_vis=False, physics_sim=mock_physics_sim, enable_gripper=False)
    server.running = True
    return server


def test_wait_for_standstill_blocks_until_dq_settles(mock_physics_sim):
    """``_wait_for_standstill`` genuinely waits: it does not return before the
    arm has actually settled, only once it has.

    Regression for the ``AutomaticErrorRecovery`` handler replying instantly
    while the arm was still decelerating from a fast abort (~2.6 rad/s
    observed), which let the client start its next motion mid-deceleration
    and trip a "Performance threshold reached" a few milliseconds later.

    Drives ``mock_physics_sim.get_robot_state()`` -- the live physics query
    the wait actually polls -- rather than ``robot_state.state["dq"]``. That
    distinction is the point of this test's shape: the publish loop (the only
    thing that ever copies the backend's ``dq`` into ``robot_state.state``)
    is not running here at all (no client connected, so it was never
    started), which stands in for the moment ``AutomaticErrorRecovery``
    always finds itself in a fraction of a millisecond after ``StopMove``:
    ``robot_state.state["dq"]`` reflects whatever the loop last copied in,
    not the arm's velocity right now. A version of this method that read the
    cached state would see it lag by up to one publish cycle and either
    return early on stale "settled" data or spin longer than necessary on
    stale "still moving" data -- this test would not catch that bug if it
    drove the same field the fix must not read.
    """
    server = _bare_server(mock_physics_sim)
    state = {
        "q": np.zeros(7),
        "dq": np.array([2.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        "tau_J": np.zeros(7),
    }
    mock_physics_sim.get_robot_state.side_effect = lambda: state

    settle_delay = 0.08

    def settle_after_delay():
        time.sleep(settle_delay)
        state["dq"] = np.zeros(7)

    settler = threading.Thread(target=settle_after_delay, daemon=True)
    settler.start()
    start = time.monotonic()
    server._wait_for_standstill()
    elapsed = time.monotonic() - start
    settler.join(timeout=1.0)

    settle_cycles_time = (
        AUTOMATIC_ERROR_RECOVERY_SETTLE_CYCLES * AUTOMATIC_ERROR_RECOVERY_POLL_PERIOD
    )
    # Must not have returned before the arm was actually driven to rest, plus
    # (most of) the sustained-settle window that confirms it stayed there.
    assert elapsed >= settle_delay + 0.5 * settle_cycles_time
    assert elapsed < AUTOMATIC_ERROR_RECOVERY_TIMEOUT


def test_wait_for_standstill_times_out_rather_than_hanging(mock_physics_sim):
    """A caller that never settles gets the wait back at the timeout, not never.

    A sim can inject state (a mocked backend, a stalled physics thread) that
    never converges; ``_wait_for_standstill`` must reply rather than hang the
    client on this forever, which is why the timeout replies success (see the
    warning it logs) rather than raising.
    """
    server = _bare_server(mock_physics_sim)
    mock_physics_sim.get_robot_state.return_value = {
        "q": np.zeros(7),
        "dq": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # never settles
        "tau_J": np.zeros(7),
    }

    start = time.monotonic()
    server._wait_for_standstill()
    elapsed = time.monotonic() - start

    assert AUTOMATIC_ERROR_RECOVERY_TIMEOUT * 0.9 <= elapsed < 3.5


def test_automatic_error_recovery_reply_waits_for_the_arm_to_settle(
    tcp_client, udp_client, sim_server, mock_physics_sim
):
    """End to end over the wire: the TCP reply itself is what is deferred.

    A fast-moving arm (as an abort at speed leaves it) only reports settled
    ``dq`` after a short simulated delay; the ``AutomaticErrorRecovery``
    response must not arrive before that.
    """
    assert perform_handshake(tcp_client)

    moving = {"q": np.zeros(7), "dq": np.array([2.6, 0.0, 0, 0, 0, 0, 0]), "tau_J": np.zeros(7)}
    settled = {"q": np.zeros(7), "dq": np.zeros(7), "tau_J": np.zeros(7)}
    settle_at = None  # armed below, once the warm-up is out of the way

    def arm_state():
        return settled if settle_at is not None and time.monotonic() >= settle_at else moving

    mock_physics_sim.get_robot_state.side_effect = arm_state

    # Give the publish loop a moment to have observed the "still moving" state
    # at least once before recovery is requested.
    time.sleep(0.02)

    # Arm the settle clock *after* that warm-up rather than before it. Started
    # before, the 0.1 s the assertion below relies on is measured from a moment
    # that includes however long the sleep and the scheduler actually took: on a
    # loaded runner the arm can already have "settled" by the time the request
    # goes out, and a reply that was never deferred at all still passes. Armed
    # here, the arm is guaranteed to still be moving when recovery is asked for,
    # which is the situation the test exists to describe.
    settle_at = time.monotonic() + 0.1

    command_id = 1
    header = MessageHeader(Command.kAutomaticErrorRecovery, command_id, 12)
    start = time.monotonic()
    tcp_client.sendall(header.to_bytes())
    tcp_client.settimeout(4.0)
    tcp_client.recv(12)
    status = struct.unpack("<B3x", tcp_client.recv(4))[0]
    elapsed = time.monotonic() - start

    assert status == 0
    assert elapsed >= 0.1, "the reply arrived before the arm ever reported settled"
    assert elapsed < 2.5, "the reply waited far longer than settling should have taken"


def test_set_joint_impedance(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test SetJointImpedance command handling"""
    assert perform_handshake(tcp_client)

    # Test data - 7 joint stiffness values
    joint_stiffness = [3000.0, 3000.0, 2500.0, 2500.0, 2000.0, 2000.0, 1500.0]

    # Create command payload
    payload = struct.pack("<7d", *joint_stiffness)

    # Create and send command message
    command_id = 1
    header = MessageHeader(Command.kSetJointImpedance, command_id, 12 + len(payload))
    message = header.to_bytes() + payload
    tcp_client.sendall(message)

    # Receive and verify response
    response_header_data = tcp_client.recv(12)
    response_header = MessageHeader.from_bytes(response_header_data)
    assert response_header.command == Command.kSetJointImpedance
    assert response_header.command_id == command_id
    assert response_header.size == 16  # Header (12) + status (1) + padding (3)

    # Get response status
    response_data = tcp_client.recv(4)  # status (1 byte) + padding (3 bytes)
    status = struct.unpack("<B3x", response_data)[0]
    assert status == 0  # Success


def test_set_cartesian_impedance(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test SetCartesianImpedance command handling"""
    assert perform_handshake(tcp_client)

    # Test data - 6 cartesian stiffness values (x, y, z, roll, pitch, yaw)
    cartesian_stiffness = [3000.0, 3000.0, 3000.0, 300.0, 300.0, 300.0]

    # Create command payload
    payload = struct.pack("<6d", *cartesian_stiffness)

    # Create and send command message
    command_id = 1
    header = MessageHeader(Command.kSetCartesianImpedance, command_id, 12 + len(payload))
    message = header.to_bytes() + payload
    tcp_client.sendall(message)

    # Receive and verify response
    response_header_data = tcp_client.recv(12)
    response_header = MessageHeader.from_bytes(response_header_data)
    assert response_header.command == Command.kSetCartesianImpedance
    assert response_header.command_id == command_id
    assert response_header.size == 16  # Header (12) + status (1) + padding (3)

    # Get response status
    response_data = tcp_client.recv(4)  # status (1 byte) + padding (3 bytes)
    status = struct.unpack("<B3x", response_data)[0]
    assert status == 0  # Success


def test_set_guiding_mode(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test SetGuidingMode command handling.

    Regression: without a response, a real libfranka client's setGuidingMode()
    call blocks forever. The request is ``std::array<bool, 6> guiding_mode`` +
    ``bool nullspace`` (7 bytes, no padding under pack(1)); no RobotState field
    reflects guiding mode, so the handler is ACK-only.
    """
    assert perform_handshake(tcp_client)

    guiding_mode = [True, False, True, False, True, False]
    nullspace = True

    payload = struct.pack("<7?", *guiding_mode, nullspace)
    assert len(payload) == 7

    command_id = 1
    header = MessageHeader(Command.kSetGuidingMode, command_id, 12 + len(payload))
    tcp_client.sendall(header.to_bytes() + payload)

    tcp_client.settimeout(2.0)
    response_header = MessageHeader.from_bytes(tcp_client.recv(12))
    assert response_header.command == Command.kSetGuidingMode
    assert response_header.command_id == command_id
    assert response_header.size == 16  # Header (12) + status (1) + padding (3)

    status = struct.unpack("<B3x", tcp_client.recv(4))[0]
    assert status == 0  # Success


def test_set_ee_to_k(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test SetEEToK command handling.

    Request is a single ``std::array<double, 16> EE_T_K`` (128 bytes). The
    real robot reflects EE_T_K back in RobotState, so verify the sim does too.
    """
    assert perform_handshake(tcp_client)

    # Identity rotation with a translation in the last column (column-major 4x4).
    ee_t_k = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.01, 0.02, 0.03, 1.0,
    ]

    payload = struct.pack("<16d", *ee_t_k)
    assert len(payload) == 128

    command_id = 1
    header = MessageHeader(Command.kSetEEToK, command_id, 12 + len(payload))
    tcp_client.sendall(header.to_bytes() + payload)

    tcp_client.settimeout(2.0)
    response_header = MessageHeader.from_bytes(tcp_client.recv(12))
    assert response_header.command == Command.kSetEEToK
    assert response_header.command_id == command_id
    assert response_header.size == 16

    status = struct.unpack("<B3x", tcp_client.recv(4))[0]
    assert status == 0  # Success

    assert np.allclose(sim_server.robot_state.state["EE_T_K"], ee_t_k)


def test_set_ne_to_ee(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test SetNEToEE command handling.

    Request is a single ``std::array<double, 16> NE_T_EE`` (128 bytes). The
    real robot reflects NE_T_EE back in RobotState, so verify the sim does too.
    """
    assert perform_handshake(tcp_client)

    ne_t_ee = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.1, 1.0,
    ]

    payload = struct.pack("<16d", *ne_t_ee)
    assert len(payload) == 128

    command_id = 1
    header = MessageHeader(Command.kSetNEToEE, command_id, 12 + len(payload))
    tcp_client.sendall(header.to_bytes() + payload)

    tcp_client.settimeout(2.0)
    response_header = MessageHeader.from_bytes(tcp_client.recv(12))
    assert response_header.command == Command.kSetNEToEE
    assert response_header.command_id == command_id
    assert response_header.size == 16

    status = struct.unpack("<B3x", tcp_client.recv(4))[0]
    assert status == 0  # Success

    assert np.allclose(sim_server.robot_state.state["NE_T_EE"], ne_t_ee)


def test_set_load(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test SetLoad command handling.

    Request is ``double m_load`` + ``std::array<double, 3> F_x_Cload`` +
    ``std::array<double, 9> I_load`` (104 bytes, no padding). The real robot
    reflects the mounted load back in RobotState, so verify the sim does too.
    """
    assert perform_handshake(tcp_client)

    m_load = 1.5
    f_x_cload = [0.01, 0.02, 0.03]
    i_load = [0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.002]

    payload = struct.pack("<d3d9d", m_load, *f_x_cload, *i_load)
    assert len(payload) == 104

    command_id = 1
    header = MessageHeader(Command.kSetLoad, command_id, 12 + len(payload))
    tcp_client.sendall(header.to_bytes() + payload)

    tcp_client.settimeout(2.0)
    response_header = MessageHeader.from_bytes(tcp_client.recv(12))
    assert response_header.command == Command.kSetLoad
    assert response_header.command_id == command_id
    assert response_header.size == 16

    status = struct.unpack("<B3x", tcp_client.recv(4))[0]
    assert status == 0  # Success

    assert sim_server.robot_state.state["m_load"] == pytest.approx(m_load)
    assert np.allclose(sim_server.robot_state.state["F_x_Cload"], f_x_cload)
    assert np.allclose(sim_server.robot_state.state["I_load"], i_load)


# Payloads used by test_no_command_hangs below: one well-formed request per
# guarded command, sized exactly per its libfranka_new v10 Request struct
# (research_interface/robot/service_types.h, #pragma pack(push, 1)).
_GUARD_COMMAND_PAYLOADS = {
    Command.kStopMove: b"",
    Command.kSetCollisionBehavior: struct.pack("<52d", *([10.0] * 52)),
    Command.kSetJointImpedance: struct.pack("<7d", *([3000.0] * 7)),
    Command.kSetCartesianImpedance: struct.pack("<6d", *([3000.0] * 6)),
    Command.kSetGuidingMode: struct.pack("<7?", *([False] * 7)),
    Command.kSetEEToK: struct.pack("<16d", *([0.0] * 16)),
    Command.kSetNEToEE: struct.pack("<16d", *([0.0] * 16)),
    Command.kSetLoad: struct.pack("<13d", *([0.0] * 13)),
    Command.kAutomaticErrorRecovery: b"",
    Command.kGetRobotModel: b"",
}

# kConnect is the handshake itself (exercised by perform_handshake() in every
# test above) and kMove answers immediately (kMotionStarted) but only reaches
# its *terminal* response once the motion actually ends -- via StopMove or a
# motion-finished datagram, neither of which this generic guard drives. Both
# are documented exclusions rather than a weaker guard, per
# test_move_command/test_stop_move_command above.
_GUARD_EXCLUDED_COMMANDS = {Command.kConnect, Command.kMove}


@pytest.mark.parametrize("command", [c for c in Command if c not in _GUARD_EXCLUDED_COMMANDS])
def test_no_command_hangs(command, tcp_client, udp_client, sim_server, mock_physics_sim):
    """Regression guard: every non-interactive v10 command gets a real TCP reply.

    Before this fix, SetGuidingMode/SetEEToK/SetNEToEE/SetLoad had no handler
    at all: the server read the header, matched no branch, and never replied --
    a real libfranka client blocks forever on the TCP recv. This iterates every
    command the v10 enum defines (except the documented interactive
    exclusions) and asserts a well-formed response arrives within a bounded
    timeout, so a silent server surfaces as a fast test failure/timeout, not a
    hang.
    """
    assert perform_handshake(tcp_client)

    payload = _GUARD_COMMAND_PAYLOADS[command]
    command_id = 42
    header = MessageHeader(command=command, command_id=command_id, size=12 + len(payload))
    tcp_client.sendall(header.to_bytes() + payload)

    tcp_client.settimeout(2.0)
    response_header = MessageHeader.from_bytes(_recv_exact(tcp_client, 12))
    assert response_header.command == command
    assert response_header.command_id == command_id
    assert response_header.size >= 12

    # Drain and sanity-check the rest of the reply so a truncated/garbage body
    # (not just a header) also fails the guard.
    remaining = response_header.size - 12
    if remaining > 0:
        body = _recv_exact(tcp_client, remaining)
        assert len(body) == remaining


def test_motion_generation_finished(tcp_client, udp_client, sim_server, mock_physics_sim):
    """Test handling of motion_generation_finished flag

    The finish datagram echoes the newest published ``message_id``, the way a
    conforming client does (``Robot::Impl::sendRobotCommand`` stamps every
    command with the id of the last state it accepted). It used to be hardcoded
    to 1, which quietly depended on the ``Move`` being accepted while state 1
    was still the current one: ``_finish_motion`` discards a finish that echoes
    a state *older* than the one its motion started at (``_motion_epoch_id``),
    which is the guard that stops a previous motion's trailing finish burst from
    ending its successor. One extra millisecond between the handshake and the
    ``Move`` -- routine on a loaded two-core runner -- put the epoch at 2, and
    the server correctly ignored the datagram ("Ignoring a motion-finished
    datagram from a motion that is already over (echoed id 1, motion started at
    2)"), leaving the motion running and this test waiting for an idle that was
    never coming. The freshness rule itself is pinned in test_comm_constraints;
    what this test is about is what a *valid* finish does.
    """
    assert perform_handshake(tcp_client)

    # First send a Move command
    move_cmd = MoveCommand(
        controller_mode=ControllerMode.kJointImpedance,
        motion_generator_mode=MotionGeneratorMode.kJointPosition,
        maximum_path_deviation=(0.1, 0.1, 0.1),
        maximum_goal_pose_deviation=(0.1, 0.1, 0.1),
    )

    payload = struct.pack(
        "<II3d3d",
        move_cmd.controller_mode.value,
        move_cmd.motion_generator_mode.value,
        *move_cmd.maximum_path_deviation,
        *move_cmd.maximum_goal_pose_deviation,
    )

    header = MessageHeader(command=Command.kMove, command_id=2, size=12 + len(payload))
    tcp_client.sendall(header.to_bytes() + payload)

    # Skip the Move command's kMotionStarted response
    tcp_client.recv(16)  # Motion started response

    # Wait for move command to be processed
    assert wait_for_state_update(
        sim_server, lambda state: state["robot_mode"] == RobotMode.kMove.value
    ), "Failed to enter move mode"
    wait_for_udp_socket(sim_server)

    # Send a command with motion_generation_finished=True, echoing the newest
    # published state -- read after the kMove wait above, so it cannot predate
    # the motion's epoch however slowly the Move was accepted. See the
    # docstring.
    command_msg = struct.pack("<Q", sim_server.robot_state.state["message_id"])  # message_id
    command_msg += struct.pack("<7d", *([0.0] * 7))  # q_c
    command_msg += struct.pack("<7d", *([0.0] * 7))  # dq_c
    command_msg += struct.pack("<16d", *([0.0] * 16))  # O_T_EE_c
    command_msg += struct.pack("<6d", *([0.0] * 6))  # O_dP_EE_c
    command_msg += struct.pack("<2d", *([0.0] * 2))  # elbow_c
    command_msg += struct.pack("<B", 0)  # valid_elbow
    command_msg += struct.pack("<B", 1)  # motion_generation_finished = True
    command_msg += struct.pack("<7d", *([0.0] * 7))  # tau_J_d
    command_msg += struct.pack("<B", 0)  # torque_command_finished

    udp_client.sendto(command_msg, ("localhost", sim_server.udp_socket.getsockname()[1]))

    # Wait for robot to enter idle mode
    assert wait_for_state_update(
        sim_server, lambda state: state["robot_mode"] == RobotMode.kIdle.value
    ), "Failed to enter idle mode after motion finished"

    # Verify we receive the final Move success response
    response_header_data = tcp_client.recv(12)
    response_header = MessageHeader.from_bytes(response_header_data)
    assert response_header.command == Command.kMove
    assert response_header.command_id == 2  # Should match our original move command ID

    response_data = tcp_client.recv(4)  # Status (1) + padding (3)
    status = struct.unpack("<B3x", response_data)[0]
    assert status == MoveStatus.kSuccess.value


def wait_for_state_update(
    sim_server, condition_fn, timeout=STATE_TRANSITION_TIMEOUT, poll_interval=0.005
):
    """Helper function to wait for a specific state condition with timeout

    Args:
        sim_server: The FrankaSimServer instance
        condition_fn: Function that takes robot_state and returns True when condition is met
        timeout: Maximum time to wait in seconds; see STATE_TRANSITION_TIMEOUT for
            why the default is generous rather than tight
        poll_interval: Time between checks in seconds

    Returns:
        True if condition was met within timeout, False otherwise
    """
    return wait_until(
        lambda: condition_fn(sim_server.robot_state.state),
        timeout=timeout,
        poll_interval=poll_interval,
    )
