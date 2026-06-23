import struct

from franka_sim.gripper_protocol import (
    GRIPPER_COMMAND_PORT,
    GRIPPER_HEADER_SIZE,
    GRIPPER_STATE_SIZE,
    GRIPPER_VERSION,
    ConnectRequest,
    GraspRequest,
    GripperCommand,
    GripperCommandHeader,
    GripperConnectStatus,
    GripperState,
    GripperStatus,
    MoveRequest,
    build_command_response,
    build_connect_response,
)


def test_constants():
    assert GRIPPER_COMMAND_PORT == 1338
    assert GRIPPER_VERSION == 3


def test_header_roundtrip():
    header = GripperCommandHeader(GripperCommand.kMove, 7, 26)
    data = header.to_bytes()
    assert len(data) == GRIPPER_HEADER_SIZE == 10
    parsed = GripperCommandHeader.from_bytes(data)
    assert parsed.command == GripperCommand.kMove
    assert parsed.command_id == 7
    assert parsed.size == 26


def test_connect_request_parse():
    req = ConnectRequest.from_bytes(struct.pack("<HH", 3, 50000))
    assert req.version == 3
    assert req.udp_port == 50000


def test_move_request_parse():
    req = MoveRequest.from_bytes(struct.pack("<dd", 0.05, 0.1))
    assert req.width == 0.05
    assert req.speed == 0.1


def test_grasp_request_parse():
    req = GraspRequest.from_bytes(struct.pack("<ddddd", 0.03, 0.005, 0.004, 0.1, 60.0))
    assert req.width == 0.03
    assert req.epsilon_inner == 0.005
    assert req.epsilon_outer == 0.004
    assert req.speed == 0.1
    assert req.force == 60.0


def test_gripper_state_pack_roundtrip():
    state = GripperState(width=0.08, max_width=0.08, is_grasped=True, temperature=30)
    data = state.pack(message_id=42)
    assert len(data) == GRIPPER_STATE_SIZE == 23
    message_id, parsed = GripperState.unpack(data)
    assert message_id == 42
    assert parsed.width == 0.08
    assert parsed.max_width == 0.08
    assert parsed.is_grasped is True
    assert parsed.temperature == 30


def test_build_connect_response():
    msg = build_connect_response(1, GripperConnectStatus.kSuccess, GRIPPER_VERSION)
    assert len(msg) == 14
    header = GripperCommandHeader.from_bytes(msg[:10])
    assert header.command == GripperCommand.kConnect
    assert header.command_id == 1
    assert header.size == 14
    status, version = struct.unpack("<HH", msg[10:14])
    assert status == GripperConnectStatus.kSuccess
    assert version == GRIPPER_VERSION


def test_build_command_response():
    msg = build_command_response(GripperCommand.kHoming, 5, GripperStatus.kSuccess)
    assert len(msg) == 12
    header = GripperCommandHeader.from_bytes(msg[:10])
    assert header.command == GripperCommand.kHoming
    assert header.command_id == 5
    assert header.size == 12
    (status,) = struct.unpack("<H", msg[10:12])
    assert status == GripperStatus.kSuccess
