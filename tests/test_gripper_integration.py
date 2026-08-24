import socket
import struct
import time

from franka_sim.franka_sim_server import FrankaSimServer
from franka_sim.gripper.protocol import (
    GRIPPER_COMMAND_PORT,
    GRIPPER_HEADER_SIZE,
    GRIPPER_VERSION,
    GripperCommand,
    GripperCommandHeader,
    GripperConnectStatus,
)


def _wait_for_port(port, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(1.0)
        try:
            probe.connect(("localhost", port))
            return True
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(0.1)
        finally:
            probe.close()
    return False


def test_gripper_server_constructed_when_enabled(mock_physics_sim):
    server = FrankaSimServer(enable_gripper=True, physics_sim=mock_physics_sim)
    assert server.gripper_server is not None
    assert server.gripper_server.port == GRIPPER_COMMAND_PORT


def test_no_gripper_server_when_disabled(mock_physics_sim):
    server = FrankaSimServer(enable_gripper=False, physics_sim=mock_physics_sim)
    assert server.gripper_server is None


def test_start_gripper_server_listens_and_handshakes(mock_physics_sim):
    server = FrankaSimServer(enable_gripper=True, physics_sim=mock_physics_sim)
    server.start_gripper_server()
    try:
        assert _wait_for_port(GRIPPER_COMMAND_PORT), "gripper server did not start"
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(("localhost", GRIPPER_COMMAND_PORT))
        try:
            payload = struct.pack("<HH", GRIPPER_VERSION, 0)
            header = GripperCommandHeader(
                GripperCommand.kConnect, 1, GRIPPER_HEADER_SIZE + len(payload)
            )
            sock.sendall(header.to_bytes() + payload)
            resp_header = GripperCommandHeader.from_bytes(sock.recv(GRIPPER_HEADER_SIZE))
            assert resp_header.command == GripperCommand.kConnect
            status, version = struct.unpack("<HH", sock.recv(4))
            assert status == GripperConnectStatus.kSuccess
            assert version == GRIPPER_VERSION
        finally:
            sock.close()
    finally:
        server.gripper_server.stop()
        if server.gripper_thread is not None:
            server.gripper_thread.join(timeout=2.0)
