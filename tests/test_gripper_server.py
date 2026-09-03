import socket
import struct

import pytest

from franka_sim.gripper.protocol import (
    GRIPPER_COMMAND_PORT,
    GRIPPER_HEADER_SIZE,
    GRIPPER_VERSION,
    GripperCommand,
    GripperCommandHeader,
    GripperConnectStatus,
    GripperState,
    GripperStatus,
)


def _recv_response(sock):
    header_data = sock.recv(GRIPPER_HEADER_SIZE)
    header = GripperCommandHeader.from_bytes(header_data)
    payload = b""
    remaining = header.size - GRIPPER_HEADER_SIZE
    while len(payload) < remaining:
        payload += sock.recv(remaining - len(payload))
    return header, payload


def _open_client():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect(("localhost", GRIPPER_COMMAND_PORT))
    return sock


def _open_udp():
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    udp.settimeout(5.0)
    return udp, udp.getsockname()[1]


def _connect(sock, udp_port, command_id=1):
    payload = struct.pack("<HH", GRIPPER_VERSION, udp_port)
    header = GripperCommandHeader(
        GripperCommand.kConnect, command_id, GRIPPER_HEADER_SIZE + len(payload)
    )
    sock.sendall(header.to_bytes() + payload)
    return _recv_response(sock)


def _send_header_only(sock, command, command_id):
    header = GripperCommandHeader(command, command_id, GRIPPER_HEADER_SIZE)
    sock.sendall(header.to_bytes())


def _send_with_payload(sock, command, command_id, payload):
    header = GripperCommandHeader(command, command_id, GRIPPER_HEADER_SIZE + len(payload))
    sock.sendall(header.to_bytes() + payload)


def test_connect_handshake(gripper_server):
    sock = _open_client()
    udp, udp_port = _open_udp()
    try:
        header, payload = _connect(sock, udp_port)
        assert header.command == GripperCommand.kConnect
        assert header.size == 14
        status, version = struct.unpack("<HH", payload)
        assert status == GripperConnectStatus.kSuccess
        assert version == GRIPPER_VERSION
    finally:
        sock.close()
        udp.close()


def test_homing_acks_and_broadcasts_state(gripper_server):
    sock = _open_client()
    udp, udp_port = _open_udp()
    try:
        _connect(sock, udp_port)
        _send_header_only(sock, GripperCommand.kHoming, 2)
        header, payload = _recv_response(sock)
        assert header.command == GripperCommand.kHoming
        assert header.size == 12
        (status,) = struct.unpack("<H", payload)
        assert status == GripperStatus.kSuccess
        _, state = GripperState.unpack(udp.recv(64))
        assert state.max_width == pytest.approx(0.08)
        assert state.width == pytest.approx(0.08)
    finally:
        sock.close()
        udp.close()


def test_move_updates_broadcast_width(gripper_server):
    sock = _open_client()
    udp, udp_port = _open_udp()
    try:
        _connect(sock, udp_port)
        _send_with_payload(sock, GripperCommand.kMove, 3, struct.pack("<dd", 0.03, 0.1))
        header, payload = _recv_response(sock)
        assert header.command == GripperCommand.kMove
        (status,) = struct.unpack("<H", payload)
        assert status == GripperStatus.kSuccess
        # The broadcaster runs at ~60 Hz; read a few packets until width settles.
        width = None
        for _ in range(10):
            _, state = GripperState.unpack(udp.recv(64))
            width = state.width
            if width == pytest.approx(0.03):
                break
        assert width == pytest.approx(0.03)
    finally:
        sock.close()
        udp.close()


def test_grasp_success_with_object(gripper_server, gripper_backend):
    gripper_backend.set_object_width(0.03)
    sock = _open_client()
    udp, udp_port = _open_udp()
    try:
        _connect(sock, udp_port)
        payload = struct.pack("<ddddd", 0.03, 0.005, 0.005, 0.1, 60.0)
        _send_with_payload(sock, GripperCommand.kGrasp, 4, payload)
        header, resp = _recv_response(sock)
        (status,) = struct.unpack("<H", resp)
        assert status == GripperStatus.kSuccess
        grasped = False
        for _ in range(10):
            _, state = GripperState.unpack(udp.recv(64))
            grasped = state.is_grasped
            if grasped:
                break
        assert grasped is True
    finally:
        sock.close()
        udp.close()


def test_grasp_in_free_space_is_a_success(gripper_server):
    """No object is not a failure: the fingers reach the commanded width.

    ``franka::Gripper::grasp`` (``include/franka/gripper.h``) calls a grasp
    successful when the final finger distance is inside ``(width -
    epsilon_inner, width + epsilon_outer)``, which the commanded width itself
    always is.
    """
    sock = _open_client()
    udp, udp_port = _open_udp()
    try:
        _connect(sock, udp_port)
        payload = struct.pack("<ddddd", 0.03, 0.005, 0.005, 0.1, 60.0)
        _send_with_payload(sock, GripperCommand.kGrasp, 5, payload)
        header, resp = _recv_response(sock)
        (status,) = struct.unpack("<H", resp)
        assert status == GripperStatus.kSuccess
    finally:
        sock.close()
        udp.close()


def test_grasp_outside_epsilon_answers_kfail(gripper_server):
    """A grasp that ends outside the band is kFail, not kUnsuccessful.

    ``franka::Gripper``'s ``executeCommand`` (libfranka ``src/gripper.cpp``)
    turns kUnsuccessful into a quiet ``false`` and kFail into
    ``CommandException("libfranka gripper: Command failed!")`` -- the latter
    is what a client is supposed to see from a grasp that did not take hold.
    Commanded 0.09 m is past the 0.08 m stroke, so the fingers stop outside
    the band (0.085, 0.095).
    """
    sock = _open_client()
    udp, udp_port = _open_udp()
    try:
        _connect(sock, udp_port)
        payload = struct.pack("<ddddd", 0.09, 0.005, 0.005, 0.1, 60.0)
        _send_with_payload(sock, GripperCommand.kGrasp, 5, payload)
        header, resp = _recv_response(sock)
        assert header.command == GripperCommand.kGrasp
        (status,) = struct.unpack("<H", resp)
        assert status == GripperStatus.kFail
    finally:
        sock.close()
        udp.close()


def test_stop_acked(gripper_server):
    sock = _open_client()
    udp, udp_port = _open_udp()
    try:
        _connect(sock, udp_port)
        _send_header_only(sock, GripperCommand.kStop, 6)
        header, payload = _recv_response(sock)
        assert header.command == GripperCommand.kStop
        (status,) = struct.unpack("<H", payload)
        assert status == GripperStatus.kSuccess
    finally:
        sock.close()
        udp.close()


def test_unknown_command_replies_fail_and_keeps_connection(gripper_server):
    sock = _open_client()
    udp, udp_port = _open_udp()
    try:
        _connect(sock, udp_port)
        # Unknown command value 99, header-only (size == GRIPPER_HEADER_SIZE).
        sock.sendall(struct.pack("<HII", 99, 7, GRIPPER_HEADER_SIZE))
        header, payload = _recv_response(sock)
        assert header.command_id == 7
        (status,) = struct.unpack("<H", payload)
        assert status == GripperStatus.kFail
        # Connection still alive: a valid Homing still gets a success ack.
        _send_header_only(sock, GripperCommand.kHoming, 8)
        h2, p2 = _recv_response(sock)
        assert h2.command == GripperCommand.kHoming
        (status2,) = struct.unpack("<H", p2)
        assert status2 == GripperStatus.kSuccess
    finally:
        sock.close()
        udp.close()
