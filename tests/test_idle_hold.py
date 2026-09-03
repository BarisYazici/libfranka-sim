"""The idle hold: whatever ends a control session, the arm must be recaptured.

The real FR3 hands its joints back to its own internal controller the moment
external control stops. The sim has no such controller, so before
``_engage_idle_hold`` a session that ended without a clean StopMove left the
simulator in TORQUE mode still applying the dead session's last ``tau_J_d``
(or in a mode with no servo at all). With ``body_gravcomp`` on and the
Menagerie FR3's small authored viscous damping, an arm carrying residual
velocity then swings on like a frictionless pendulum in zero-g -- the arm
"going bananas" in the viewer after a Ctrl-C'd torque example.

Three layers, in the order a regression would show up:

* mock-server: every stop path reaches the simulator contract (and the mobile
  base's does not get a bogus joint hold),
* physics: a real MuJoCo arm swinging fast is actually brought to rest,
* end-to-end: the original scenario over the v10 wire, connection killed
  mid-stream.
"""

import socket
import struct
import threading
import time

import numpy as np
import pytest

from franka_sim.control_modes import ControlMode
from franka_sim.franka_protocol import (
    COMMAND_PORT,
    Command,
    ConnectStatus,
    ControllerMode,
    MessageHeader,
    MotionGeneratorMode,
)
from franka_sim.robot_state import _ROBOT_STATE_PACKER, RobotState

#: A pose no fixture defaults to, so "held at the current q" cannot pass by
#: accidentally matching zeros or the initial pose.
HELD_Q = np.array([0.11, -0.22, 0.33, -1.44, 0.55, 1.66, 0.77])

TORQUES = [3.0, -3.0, 3.0, -3.0, 1.5, -1.5, 1.5]
BASE_TWIST = [0.25, -0.1, 0.0, 0.0, 0.0, 0.4]


# --- shared wire helpers -----------------------------------------------------


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


def perform_handshake(tcp_client, udp_port=1338, host="localhost"):
    """Connect handshake (libfranka v10): version + UDP port in, status + version out."""
    tcp_client.connect((host, COMMAND_PORT))
    payload = struct.pack("<HH", 10, udp_port)
    header = MessageHeader(command=Command.kConnect, command_id=1, size=12 + len(payload))
    tcp_client.sendall(header.to_bytes() + payload)
    deadline = time.monotonic() + TCP_DEADLINE_S
    recv_exactly(tcp_client, 12, deadline)
    status, _ = struct.unpack("<BH", recv_exactly(tcp_client, 3, deadline))
    return status == ConnectStatus.kSuccess


def send_move(tcp_client, controller_mode, motion_generator_mode, command_id=2):
    """Send a Move and drain TCP through this command's own reply.

    Move gets exactly one reply per command id: kMotionStarted, sent the
    moment the Move is accepted. The *terminal* response (kSuccess via a
    motion-finished datagram, kPreempted via StopMove, or an abort status)
    only arrives once the motion actually ends -- so it is not read here.

    A caller starting a second motion on a connection where the previous one
    already finished (e.g. via a `motion_finished` UDP datagram) can still
    have that first motion's terminal response sitting unread on the socket
    when this fires; that response is not for `command_id`, so it is drained
    and discarded here rather than left for the caller to trip over.

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


def wait_until(predicate, timeout=3.0, interval=0.02):
    """Poll ``predicate`` until it is true or the timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def held_position(mock_sim):
    """The joint target of the last update_joint_positions call, as an array."""
    assert mock_sim.update_joint_positions.called, "the simulator was never told to hold"
    return np.asarray(mock_sim.update_joint_positions.call_args.args[0], dtype=float)


def assert_holds_at(mock_sim, q):
    """The simulator was switched to POSITION and parked at ``q``."""
    assert wait_until(
        lambda: mock_sim.set_control_mode.call_args is not None
        and mock_sim.set_control_mode.call_args.args[0] is ControlMode.POSITION
    ), f"never switched to POSITION (last: {mock_sim.set_control_mode.call_args})"
    assert held_position(mock_sim) == pytest.approx(q)


# --- layer 1: every stop path reaches the simulator --------------------------


@pytest.fixture
def holding_sim(mock_physics_sim):
    """The mocked arm simulator, reporting a distinctive current pose."""
    mock_physics_sim.get_robot_state.return_value = {
        "q": HELD_Q,
        "dq": np.zeros(7),
        "tau_J": np.zeros(7),
    }
    return mock_physics_sim


def _start_torque_motion(tcp_client, udp_client, sim_server):
    """Handshake, start an external-controller motion and stream one torque."""
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kExternalController, MotionGeneratorMode.kNone)
    send_robot_command(udp_client, sim_server, tau_j_d=TORQUES)
    assert sim_server.control_mode is ControlMode.TORQUE


def test_client_disconnect_mid_torque_motion_holds_position(
    tcp_client, udp_client, sim_server, holding_sim
):
    """The Ctrl-C case: the client vanishes without StopMove or a finish flag."""
    _start_torque_motion(tcp_client, udp_client, sim_server)

    # Abrupt, not graceful: SO_LINGER 0 makes close() send an RST, which is what
    # a SIGINT-killed libfranka control loop leaves behind.
    tcp_client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    tcp_client.close()

    assert_holds_at(holding_sim, HELD_Q)


def test_stop_move_holds_position(tcp_client, udp_client, sim_server, holding_sim):
    """A StopMove mid-torque must recapture, not just stop commanding."""
    _start_torque_motion(tcp_client, udp_client, sim_server)

    stop_header = MessageHeader(command=Command.kStopMove, command_id=3, size=12)
    tcp_client.sendall(stop_header.to_bytes())

    assert_holds_at(holding_sim, HELD_Q)


def test_motion_completion_holds_position(tcp_client, udp_client, sim_server, holding_sim):
    """Clean finish: torque_command_finished / motion_generation_finished."""
    _start_torque_motion(tcp_client, udp_client, sim_server)

    send_robot_command(udp_client, sim_server, message_id=2, motion_finished=True)

    assert_holds_at(holding_sim, HELD_Q)


def test_a_late_datagram_cannot_undo_the_hold(tcp_client, udp_client, sim_server, holding_sim):
    """A torque still in flight when the session ended must be dropped.

    Otherwise the hold is engaged and then immediately overwritten by the dead
    session's last command -- the bug, one datagram later.
    """
    _start_torque_motion(tcp_client, udp_client, sim_server)
    send_robot_command(udp_client, sim_server, message_id=2, motion_finished=True)
    assert_holds_at(holding_sim, HELD_Q)

    holding_sim.update_torques.reset_mock()
    holding_sim.set_control_mode.reset_mock()
    send_robot_command(udp_client, sim_server, message_id=3, tau_j_d=TORQUES)

    for call in holding_sim.update_torques.call_args_list:
        assert np.asarray(call.args[0]) == pytest.approx(np.zeros(7))
    for call in holding_sim.set_control_mode.call_args_list:
        assert call.args[0] is not ControlMode.TORQUE


def test_error_recovery_mid_motion_holds_position(tcp_client, udp_client, sim_server, holding_sim):
    """Recovery aborts the motion on the real robot, so it must recapture here."""
    _start_torque_motion(tcp_client, udp_client, sim_server)

    header = MessageHeader(Command.kAutomaticErrorRecovery, 5, 12)
    tcp_client.sendall(header.to_bytes())
    tcp_client.settimeout(2.0)
    tcp_client.recv(12)
    assert struct.unpack("<B3x", tcp_client.recv(4))[0] == 0

    assert_holds_at(holding_sim, HELD_Q)


def test_a_new_move_overrides_the_hold(tcp_client, udp_client, sim_server, holding_sim):
    """Session start must release the hold, or the next motion never moves."""
    _start_torque_motion(tcp_client, udp_client, sim_server)
    send_robot_command(udp_client, sim_server, message_id=2, motion_finished=True)
    assert_holds_at(holding_sim, HELD_Q)

    send_move(
        tcp_client, ControllerMode.kExternalController, MotionGeneratorMode.kNone, command_id=4
    )
    holding_sim.update_torques.reset_mock()
    send_robot_command(udp_client, sim_server, message_id=3, tau_j_d=TORQUES)

    assert sim_server.control_mode is ControlMode.TORQUE
    assert holding_sim.update_torques.call_args.args[0] == pytest.approx(TORQUES)


def test_the_base_role_is_never_given_a_joint_hold(tcp_client, udp_client, base_sim_server,
                                                   mock_base_sim):
    """The base has no joint-space hold: stopping it means a zero body twist."""
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)
    send_robot_command(udp_client, base_sim_server, o_dp_ee_c=BASE_TWIST)

    tcp_client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
    tcp_client.close()

    assert wait_until(
        lambda: mock_base_sim.update_base_twist.call_args is not None
        and np.asarray(mock_base_sim.update_base_twist.call_args.args[0])
        == pytest.approx(np.zeros(6))
    ), "the base twist was never zeroed on disconnect"
    mock_base_sim.update_joint_positions.assert_not_called()
    for call in mock_base_sim.set_control_mode.call_args_list:
        assert call.args[0] is not ControlMode.POSITION


# --- layer 2: a real MuJoCo arm is actually brought to rest -------------------

mujoco = pytest.importorskip("mujoco")

from franka_sim.franka_sim_server import FrankaSimServer  # noqa: E402
from franka_sim.mujoco_franka_sim import (  # noqa: E402
    MujocoFrankaSim,
    default_fr3_mjcf,
)

try:
    FR3_MJCF = default_fr3_mjcf()
except Exception:  # pragma: no cover - depends on the host's cache/network
    FR3_MJCF = None

#: Steps of open-loop torque used to get the arm genuinely moving, and the
#: settling budget afterwards (2.5 simulated seconds, ~0.06 s of wall clock).
SPIN_UP_STEPS = 200
SETTLE_STEPS = 1500

#: Torque (Nm) applied to every joint during spin-up. Well inside
#: FR3_FORCE_LIMITS, but with gravity compensated it accelerates the arm hard.
SPIN_UP_TORQUE = 20.0


@pytest.mark.skipif(
    FR3_MJCF is None or not FR3_MJCF.exists(),
    reason="the MuJoCo Menagerie FR3 model is neither cached nor downloadable",
)
def test_the_hold_recaptures_a_fast_moving_mujoco_arm():
    """Physics, not mocks: the swinging arm is caught and stays caught."""
    sim = MujocoFrankaSim()
    sim.initialize_simulation()
    server = FrankaSimServer(physics_sim=sim, enable_gripper=False)
    try:
        sim.set_control_mode(ControlMode.TORQUE)
        sim.update_torques([SPIN_UP_TORQUE] * 7)
        server.control_mode = ControlMode.TORQUE
        sim.step(SPIN_UP_STEPS)

        moving_dq = np.abs(sim.get_robot_state()["dq"])
        assert moving_dq.max() > 1.0, f"the arm never got moving: |dq|max={moving_dq.max()}"

        # Exactly what a session teardown runs.
        server._engage_idle_hold("test")
        q_at_hold = np.array(sim.get_robot_state()["q"])

        sim.step(SETTLE_STEPS)
        state = sim.get_robot_state()
    finally:
        sim.stop()

    assert np.abs(state["dq"]).max() < 0.05, f"still swinging: dq={state['dq']}"
    # It coasts a little into the servo, but nothing like the radians the
    # unheld arm travelled: it was recaptured, not left to swing.
    assert np.abs(state["q"] - q_at_hold).max() < 0.2, (
        f"drifted away from the hold pose: {state['q'] - q_at_hold}"
    )


# --- layer 3: end to end over the v10 wire ----------------------------------


class WireClient:
    """A minimal v10 client: TCP commands plus a UDP command/state channel."""

    def __init__(self, host="127.0.0.1"):
        self.host = host
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.settimeout(5.0)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.settimeout(5.0)
        self.udp.bind(("127.0.0.1", 0))
        self.udp_port = self.udp.getsockname()[1]
        self.server_udp_address = None

    def connect(self):
        assert perform_handshake(self.tcp, udp_port=self.udp_port, host=self.host)

    def move(self, controller_mode, motion_generator_mode):
        send_move(self.tcp, controller_mode, motion_generator_mode)

    def read_state(self):
        data, address = self.udp.recvfrom(4096)
        self.server_udp_address = address
        assert len(data) == _ROBOT_STATE_PACKER.size
        return _ROBOT_STATE_PACKER.unpack(data)

    def send_command(self, message_id, tau_j_d=None):
        self.udp.sendto(pack_robot_command(message_id, tau_j_d=tau_j_d), self.server_udp_address)

    def kill(self):
        """Drop the connection the way a SIGINT'd control loop does: RST, no StopMove."""
        self.tcp.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        self.tcp.close()

    def close(self):
        for sock in (self.tcp, self.udp):
            try:
                sock.close()
            except OSError:
                pass


def robot_state_field_slice(field, length):
    """Locate a field inside the packed 1377-byte RobotState, without magic offsets."""
    probe = RobotState()
    sentinel = [float(1000 + index) for index in range(length)]
    probe.state[field] = sentinel
    values = _ROBOT_STATE_PACKER.unpack(probe.pack_state())
    start = values.index(sentinel[0])
    return slice(start, start + length)


Q_SLICE = robot_state_field_slice("q", 7)
DQ_SLICE = robot_state_field_slice("dq", 7)


@pytest.fixture
def live_mujoco_server():
    """A real FrankaSimServer on port 1337 over a real MuJoCo arm."""
    if FR3_MJCF is None or not FR3_MJCF.exists():
        pytest.skip("the MuJoCo Menagerie FR3 model is neither cached nor downloadable")

    sim = MujocoFrankaSim()
    sim.initialize_simulation()
    server = FrankaSimServer(physics_sim=sim, enable_gripper=False)

    accept_thread = threading.Thread(target=server.run_server, daemon=True)
    accept_thread.start()
    sim.running = True
    physics_thread = threading.Thread(target=sim.run_simulation, daemon=True)
    physics_thread.start()

    deadline = time.time() + 5.0
    while time.time() < deadline:
        try:
            probe = socket.create_connection(("127.0.0.1", COMMAND_PORT), timeout=1.0)
            probe.close()
            break
        except OSError:
            time.sleep(0.1)
    else:  # pragma: no cover - only on a wedged host
        server.stop()
        raise AssertionError("the FCI server never came up on port 1337")

    yield server

    server.stop()
    sim.stop()
    physics_thread.join(timeout=3.0)
    accept_thread.join(timeout=3.0)
    time.sleep(0.3)


def test_a_killed_torque_session_leaves_the_arm_at_rest(live_mujoco_server):
    """The reported scenario: torque control, connection killed mid-stream."""
    client = WireClient()
    try:
        client.connect()
        client.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone)
        client.read_state()

        for message_id in range(1, 400):
            client.send_command(message_id, tau_j_d=[SPIN_UP_TORQUE] * 7)
            state = client.read_state()

        dq_before_kill = np.abs(np.array(state[DQ_SLICE]))
        assert dq_before_kill.max() > 0.5, f"the arm never got moving: {dq_before_kill}"
        q_at_kill = np.array(state[Q_SLICE])

        client.kill()
    finally:
        client.close()

    time.sleep(2.0)

    # Reconnect and read what the arm is doing now.
    observer = WireClient()
    try:
        observer.connect()
        for _ in range(20):
            state = observer.read_state()
    finally:
        observer.close()

    dq_after = np.abs(np.array(state[DQ_SLICE]))
    q_after = np.array(state[Q_SLICE])
    assert dq_after.max() < 0.05, f"the arm is still swinging after the kill: dq={dq_after}"
    # Generous, because how far it coasts into the servo depends on how many
    # physics steps the wire loop got through -- but nothing like the radians
    # the unheld arm travelled before slamming into its joint limits.
    assert np.abs(q_after - q_at_kill).max() < 0.5, (
        f"the arm drifted away from where it was killed: {q_after - q_at_kill}"
    )
