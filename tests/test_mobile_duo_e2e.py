"""End-to-end: three FCI bridges on one scene, plus the spine stub.

Layer 1 (always on) drives the bridges with raw sockets speaking the v10 wire
format, against a fake-but-integrating Genesis entity, so the whole protocol
path runs without the physics engine. Layer 2 repeats the same commands with a
real libfranka client when the prebuilt library is available.
"""

import http.client
import json
import os
import shutil
import socket
import ssl
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeDuoEntity

from franka_sim.franka_protocol import (
    COMMAND_PORT,
    Command,
    ConnectStatus,
    ControllerMode,
    MessageHeader,
    MotionGeneratorMode,
)
from franka_sim.control_modes import ControlMode
from franka_sim.mobile_duo_runner import MobileDuoRunner
from franka_sim.mobile_duo_sim import (
    ARM_INITIAL_Q as DUO_ARM_INITIAL_Q,
    ROLE_BASE,
    ROLE_LEFT,
    ROLE_RIGHT,
    MobileDuoScene,
)
from franka_sim.robot_state import _ROBOT_STATE_PACKER, RobotState
from franka_sim.spine_stub import (
    SPINE_DEFAULT_HOST,
    SPINE_DEFAULT_PORT,
    SpineStubServer,
    make_self_signed_cert,
)

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="127.0.0.0/8 loopback aliases are Linux-specific",
)

BINDS = {ROLE_LEFT: "127.0.0.11", ROLE_RIGHT: "127.0.0.12", ROLE_BASE: "127.0.0.10"}
SPINE_HOST = SPINE_DEFAULT_HOST  # 127.0.0.13
# SpineApiClient hardcodes 443, so that is the default. Task 9 Step 10 Option C
# (iptables REDIRECT 443 -> 8443) binds a different port instead; exporting
# FRANKA_SIM_SPINE_PORT points these tests at whichever port is really bound.
SPINE_PORT = int(os.environ.get("FRANKA_SIM_SPINE_PORT", SPINE_DEFAULT_PORT))
BASE_TWIST = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBFRANKA = REPO_ROOT / "libfranka_new"
LIB_BUILD = LIBFRANKA / "build"
EIGEN_INCLUDE = Path("/usr/include/eigen3")
PROBE_SRC = Path(__file__).parent / "_mobile_duo_probe.cpp"


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


class IntegratingDuoEntity(FakeDuoEntity):
    """A fake entity whose joints actually integrate their control targets.

    A Genesis control target *persists*: the solver re-applies it on every
    ``scene.step`` until something overwrites it, which is exactly why the
    physics loop writes a target only when its value changes -- at 400 Hz the
    redundant kernel launches, not the physics, are what costs. So the
    integration has to hang off the step, not off the ``control_dofs_*`` call:
    a fake that moved a joint once per call reports a robot that stops dead
    while the client is still holding a perfectly good constant command.

    :meth:`step` therefore stands in for ``scene.step`` and must be driven by
    whatever loop drives ``_apply_control``.
    """

    def __init__(self, dt):
        super().__init__()
        self.dt = dt
        self.dof_positions = np.zeros(self.n_dofs)
        self.dof_velocities = np.zeros(self.n_dofs)
        #: dof index -> (kind, value) for the target the solver still holds.
        #: One entry per DOF: a later target of either kind replaces the
        #: earlier one, which is what switching a bridge's control mode does.
        self.targets = {}

    def control_dofs_position(self, values, dofs_idx_local):
        super().control_dofs_position(values, dofs_idx_local)
        self._hold("position", values, dofs_idx_local)

    def control_dofs_velocity(self, values, dofs_idx_local):
        super().control_dofs_velocity(values, dofs_idx_local)
        self._hold("velocity", values, dofs_idx_local)

    def control_dofs_force(self, values, dofs_idx_local):
        super().control_dofs_force(values, dofs_idx_local)

    def _hold(self, kind, values, dofs_idx_local):
        for value, index in zip(np.asarray(values, dtype=float), dofs_idx_local):
            self.targets[int(index)] = (kind, float(value))

    def step(self):
        """Advance every held target by one ``dt``, as ``scene.step`` would."""
        for index, (kind, value) in self.targets.items():
            if kind == "position":
                self.dof_positions[index] = value
                self.dof_velocities[index] = 0.0
            else:
                self.dof_velocities[index] = value
                self.dof_positions[index] += value * self.dt


@pytest.fixture
def duo_stack(tmp_path):
    """Three live bridges on one stepping scene. Yields (runner, scene)."""
    urdf_path = tmp_path / "duo.urdf"
    urdf_path.write_text('<?xml version="1.0"?><robot name="duo"></robot>')

    scene = MobileDuoScene(urdf_path, enable_vis=False, base_height=0.05)
    scene.robot = IntegratingDuoEntity(scene.dt)
    scene._bind_entity()
    scene._read_and_publish_state()

    stop_event = threading.Event()

    def step_loop():
        # read / apply / step, in run_simulation's order: the entity's step is
        # what advances the held control targets, so it stands where
        # scene.step() stands in the real loop.
        while not stop_event.is_set():
            scene._read_and_publish_state()
            scene._apply_control()
            scene.robot.step()
            time.sleep(scene.dt)

    stepper = threading.Thread(target=step_loop, name="duo-stepper", daemon=True)
    stepper.start()

    runner = MobileDuoRunner(scene, BINDS)
    runner.start_servers()
    wait_for_bridges()

    yield runner, scene

    runner.stop()
    stop_event.set()
    stepper.join(timeout=3.0)
    time.sleep(0.3)


def wait_for_bridges(timeout=5.0):
    """Block until all three bridges accept TCP connections."""
    deadline = time.time() + timeout
    for role, host in BINDS.items():
        while True:
            try:
                probe = socket.create_connection((host, COMMAND_PORT), timeout=1.0)
                probe.close()
                break
            except OSError:
                if time.time() > deadline:
                    raise AssertionError(f"{role} bridge never came up on {host}")
                time.sleep(0.1)


class WireClient:
    """A minimal v10 client: TCP commands plus a UDP command/state channel."""

    def __init__(self, host):
        self.host = host
        self.tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.tcp.settimeout(5.0)
        self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp.settimeout(5.0)
        self.udp.bind(("0.0.0.0", 0))
        self.udp_port = self.udp.getsockname()[1]
        self.server_udp_address = None

    def connect(self):
        self.tcp.connect((self.host, COMMAND_PORT))
        payload = struct.pack("<HH", 10, self.udp_port)
        header = MessageHeader(command=Command.kConnect, command_id=1, size=12 + len(payload))
        self.tcp.sendall(header.to_bytes() + payload)
        self.tcp.recv(12)
        status, _ = struct.unpack("<BH", self.tcp.recv(3))
        assert status == ConnectStatus.kSuccess

    def move(self, controller_mode, motion_generator_mode):
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
        header = MessageHeader(command=Command.kMove, command_id=2, size=12 + len(payload))
        self.tcp.sendall(header.to_bytes() + payload)
        # Move gets exactly one immediate reply (kMotionStarted); the
        # terminal one (kSuccess/abort) only arrives once the motion ends.
        self.tcp.recv(16)

    def read_state(self):
        """Read one RobotState datagram and return the unpacked tuple."""
        data, address = self.udp.recvfrom(4096)
        self.server_udp_address = address
        assert len(data) == _ROBOT_STATE_PACKER.size
        return _ROBOT_STATE_PACKER.unpack(data)

    def send_command(self, message_id, o_dp_ee_c=None, tau_j_d=None, motion_finished=False):
        o_dp_ee_c = o_dp_ee_c if o_dp_ee_c is not None else [0.0] * 6
        tau_j_d = tau_j_d if tau_j_d is not None else [0.0] * 7
        message = struct.pack("<Q", message_id)
        message += struct.pack("<7d", *([0.0] * 7))
        message += struct.pack("<7d", *([0.0] * 7))
        message += struct.pack("<16d", *([0.0] * 16))
        message += struct.pack("<6d", *o_dp_ee_c)
        message += struct.pack("<2d", *([0.0] * 2))
        message += struct.pack("<B", 0)
        message += struct.pack("<B", 1 if motion_finished else 0)
        message += struct.pack("<7d", *tau_j_d)
        message += struct.pack("<B", 0)
        self.udp.sendto(message, self.server_udp_address)

    def close(self):
        for sock in (self.tcp, self.udp):
            try:
                sock.close()
            except OSError:
                pass


def test_base_bridge_moves_the_platform_and_reports_wheel_state(duo_stack):
    runner, scene = duo_stack
    client = WireClient(BINDS[ROLE_BASE])
    try:
        client.connect()
        client.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)

        first = client.read_state()
        q_before = np.array(first[Q_SLICE])
        x_before = scene.swerve.x

        for message_id in range(1, 60):
            client.send_command(message_id, o_dp_ee_c=BASE_TWIST)
            client.read_state()
        time.sleep(0.3)

        last = client.read_state()
        q_after = np.array(last[Q_SLICE])
        dq_after = np.array(last[DQ_SLICE])
    finally:
        client.close()

    # Wheel state is reported over the wire, padded into the 7-element arrays.
    assert q_after[4:] == pytest.approx([0.0, 0.0, 0.0])
    # Both drive joints (indices 1 and 3) spun forward.
    assert q_after[1] - q_before[1] > 0.1
    assert q_after[3] - q_before[3] > 0.1
    assert dq_after[1] > 0.0
    # Steering stayed forward for a pure +vx command.
    assert q_after[0] == pytest.approx(0.0, abs=1e-6)
    # And the platform itself advanced along +x.
    assert scene.swerve.x > x_before + 0.01


def test_base_bridge_stops_on_motion_finished(duo_stack):
    runner, scene = duo_stack
    client = WireClient(BINDS[ROLE_BASE])
    try:
        client.connect()
        client.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)
        client.read_state()
        for message_id in range(1, 30):
            client.send_command(message_id, o_dp_ee_c=BASE_TWIST)
            client.read_state()

        client.send_command(100, motion_finished=True)
        time.sleep(0.3)
        x_at_stop = scene.swerve.x
        time.sleep(0.3)
    finally:
        client.close()

    assert scene.swerve.x == pytest.approx(x_at_stop, abs=1e-6)


@pytest.mark.parametrize("role", [ROLE_LEFT, ROLE_RIGHT])
def test_arm_bridges_accept_torque_control_independently(duo_stack, role):
    runner, scene = duo_stack
    torques = [1.0, -1.0, 1.0, -1.0, 0.5, -0.5, 0.5]
    client = WireClient(BINDS[role])
    try:
        client.connect()
        client.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone)
        client.read_state()
        for message_id in range(1, 20):
            client.send_command(message_id, tau_j_d=torques)
            client.read_state()
        time.sleep(0.2)
        # Sampled while the session is still live: closing the socket ends the
        # session, and the bridge then recaptures the arm into a position hold
        # with zero torque (see test_idle_hold.py), which would erase this.
        commanded = np.array(scene.arm_torques[role])
        other = ROLE_RIGHT if role == ROLE_LEFT else ROLE_LEFT
        idle = np.array(scene.arm_torques[other])
    finally:
        client.close()

    assert commanded == pytest.approx(torques)
    assert idle == pytest.approx(np.zeros(7))


@pytest.mark.parametrize("role", [ROLE_LEFT, ROLE_RIGHT])
def test_an_arm_bridge_holds_its_pose_when_the_client_dies(duo_stack, role):
    """The idle hold reaches the duo arms through SceneView, unchanged.

    Same recapture the single-arm server does, proving the fix is in the server
    layer and not in one backend: kill a torque session and the arm must be
    parked in POSITION at the joints it was killed at, not left driven by the
    dead session's torque.
    """
    runner, scene = duo_stack
    client = WireClient(BINDS[role])
    try:
        client.connect()
        client.move(ControllerMode.kExternalController, MotionGeneratorMode.kNone)
        client.read_state()
        for message_id in range(1, 20):
            client.send_command(message_id, tau_j_d=[1.0, -1.0, 1.0, -1.0, 0.5, -0.5, 0.5])
            client.read_state()
        time.sleep(0.2)
        q_at_kill = np.array(scene.get_role_state(role)["q"])
        client.tcp.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.tcp.close()
        time.sleep(0.8)
    finally:
        client.close()

    assert scene.arm_control_modes[role] == ControlMode.POSITION
    assert scene.arm_joint_positions[role] == pytest.approx(q_at_kill)
    assert scene.arm_torques[role] == pytest.approx(np.zeros(7))


def test_the_base_bridge_stops_the_platform_when_the_client_dies(duo_stack):
    """The base's stop is a zero twist -- never a joint hold on the wheels."""
    runner, scene = duo_stack
    client = WireClient(BINDS[ROLE_BASE])
    try:
        client.connect()
        client.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)
        client.read_state()
        for message_id in range(1, 30):
            client.send_command(message_id, o_dp_ee_c=BASE_TWIST)
            client.read_state()

        client.tcp.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        client.tcp.close()
        time.sleep(0.8)
        x_at_stop = scene.swerve.x
        time.sleep(0.3)
    finally:
        client.close()

    assert scene.swerve.x == pytest.approx(x_at_stop, abs=1e-6)
    # The wheels are not arm joints: neither arm may be touched by a base stop.
    for arm in (ROLE_LEFT, ROLE_RIGHT):
        assert scene.arm_control_modes[arm] == ControlMode.POSITION
        assert scene.arm_joint_positions[arm] == pytest.approx(DUO_ARM_INITIAL_Q)


def test_all_three_bridges_serve_the_robot_model(duo_stack):
    for host in BINDS.values():
        client = WireClient(host)
        try:
            client.connect()
            header = MessageHeader(command=Command.kGetRobotModel, command_id=3, size=12)
            client.tcp.sendall(header.to_bytes())
            response_header = MessageHeader.from_bytes(client.tcp.recv(12))
            payload = b""
            while len(payload) < response_header.size - 12:
                payload += client.tcp.recv(response_header.size - 12 - len(payload))
        finally:
            client.close()

        assert payload[0] == 0  # kSuccess
        assert b"joint1" in payload


@pytest.fixture
def spine_stack(tmp_path):
    """The same three bridges, plus the spine stub the runner owns.

    The stub is launched **by the runner**, exactly as ``--spine`` does in
    production: `SpineStubServer` bound to `SPINE_HOST:SPINE_PORT`
    (`127.0.0.13:443` by default) over TLS with the self-signed certificate from
    `make_self_signed_cert`, its `SpineModel` shared into `scene.spine_model`.
    Port 443 is privileged; if the interpreter has not been granted
    `cap_net_bind_service` (Task 9 Step 10 Option A) the bind fails and this
    test skips with that instruction. Under Option C, export
    `FRANKA_SIM_SPINE_PORT=8443` and the fixture binds the redirected port
    instead.
    """
    if shutil.which("openssl") is None:
        pytest.skip("openssl CLI is required to generate the stub certificate")

    certfile, keyfile = make_self_signed_cert(tmp_path)
    try:
        spine_server = SpineStubServer(
            host=SPINE_HOST, port=SPINE_PORT, certfile=certfile, keyfile=keyfile
        )
    except PermissionError:
        pytest.skip(
            f"binding {SPINE_HOST}:{SPINE_PORT} is not permitted -- see Task 9 Step 10: "
            "grant cap_net_bind_service (Option A), re-run under sudo -E (Option B), or "
            "set up the REDIRECT and export FRANKA_SIM_SPINE_PORT=8443 (Option C)"
        )

    urdf_path = tmp_path / "duo.urdf"
    urdf_path.write_text('<?xml version="1.0"?><robot name="duo"></robot>')
    scene = MobileDuoScene(urdf_path, enable_vis=False, base_height=0.05)
    scene.robot = IntegratingDuoEntity(scene.dt)
    scene._bind_entity()
    scene._read_and_publish_state()

    stop_event = threading.Event()

    def step_loop():
        while not stop_event.is_set():
            scene._read_and_publish_state()
            scene._apply_control()
            scene.robot.step()
            time.sleep(scene.dt)

    stepper = threading.Thread(target=step_loop, name="spine-stepper", daemon=True)
    stepper.start()

    runner = MobileDuoRunner(scene, BINDS, spine_server=spine_server)
    runner.start_servers()
    wait_for_bridges()

    yield runner, scene

    runner.stop()
    stop_event.set()
    stepper.join(timeout=3.0)
    time.sleep(0.3)


def spine_call(method, endpoint, body=None):
    """Call the stub over TLS the way SpineApiClient does (verify disabled)."""
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    connection = http.client.HTTPSConnection(SPINE_HOST, SPINE_PORT, context=context, timeout=5)
    connection.request(
        method,
        f"/spine/api/{endpoint}",
        body=json.dumps(body) if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    payload = json.loads(connection.getresponse().read())
    connection.close()
    return payload


def test_spine_stub_serves_a_lift_move_alongside_the_bridges(spine_stack):
    """The spine is a separate REST device; prove it runs next to the bridges."""
    assert spine_call("POST", "spine:switch-on") == "SwitchedOn"
    assert spine_call("POST", "motion-mm:start", {"position": 500, "velocity": 100})["StopBy"]
    first = spine_call("GET", "position-mm")["position"]
    time.sleep(0.4)
    second = spine_call("GET", "position-mm")["position"]
    assert second > first
    assert spine_call("GET", "state") == "SwitchedOn"


def test_a_rest_lift_command_moves_the_spine_joint_in_the_scene(spine_stack):
    """The point of --spine: a REST move drives franka_spine_vertical_joint."""
    runner, scene = spine_stack
    assert scene.spine_model is runner.spine_server.model

    spine_call("POST", "spine:switch-on")
    spine_call("POST", "motion-mm:start", {"position": 600, "velocity": 200})
    time.sleep(0.5)

    values, dofs = scene.robot.set_position_calls[-1]
    assert dofs == [scene.spine_dof_idx]
    assert 0.05 < float(values[0]) <= 0.6


def test_driving_the_base_and_the_lift_together(spine_stack):
    """Phase-3 shape: base twist and lift move at the same time."""
    runner, scene = spine_stack

    spine_call("POST", "spine:switch-on")
    spine_call("POST", "motion-mm:start", {"position": 400, "velocity": 200})

    client = WireClient(BINDS[ROLE_BASE])
    try:
        client.connect()
        client.move(ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)
        client.read_state()
        for message_id in range(1, 60):
            client.send_command(message_id, o_dp_ee_c=BASE_TWIST)
            client.read_state()
        time.sleep(0.3)
    finally:
        client.close()

    assert scene.swerve.x > 0.01
    assert float(scene.robot.set_position_calls[-1][0][0]) > 0.05


# --- real libfranka client -------------------------------------------------


def _probe_prereqs_available():
    return (
        shutil.which("g++") is not None
        and (LIB_BUILD / "libfranka.so").exists()
        and (LIBFRANKA / "include" / "franka" / "robot.h").exists()
        and EIGEN_INCLUDE.exists()
    )


@pytest.fixture(scope="module")
def probe_binary(tmp_path_factory):
    """Compile the mobile-duo probe against the prebuilt libfranka_new."""
    if not _probe_prereqs_available():
        pytest.skip("prebuilt libfranka_new + g++ + eigen3 are required")
    out = tmp_path_factory.mktemp("duo_probe") / "duo_probe"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            f"-I{LIBFRANKA / 'include'}",
            f"-I{LIBFRANKA / 'common' / 'include'}",
            f"-I{EIGEN_INCLUDE}",
            str(PROBE_SRC),
            f"-L{LIB_BUILD}",
            "-lfranka",
            "-o",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return out


def test_real_libfranka_drives_all_three_bridges(duo_stack, probe_binary):
    """A stock libfranka client commands the base by twist and both arms by torque."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{LIB_BUILD}:" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(
        [str(probe_binary), BINDS[ROLE_BASE], BINDS[ROLE_LEFT], BINDS[ROLE_RIGHT]],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )

    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    assert "BASE_OK" in result.stdout
    assert f"ARM_OK {BINDS[ROLE_LEFT]}" in result.stdout
    assert f"ARM_OK {BINDS[ROLE_RIGHT]}" in result.stdout

    drive_delta = float(
        [token for token in result.stdout.split() if token.startswith("drive_delta=")][0].split(
            "="
        )[1]
    )
    assert drive_delta > 0.1, result.stdout
