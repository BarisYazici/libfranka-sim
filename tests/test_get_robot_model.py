"""Wire-compatibility tests for libfranka_new (protocol v10).

These verify the two things a stock libfranka_new client needs during
``franka::Robot`` construction:

1. The Connect handshake reports protocol version 10 with the v10 response
   layout (``status:uint8 + version:uint16`` == 3 bytes), and
2. ``GetRobotModel`` returns a valid arm URDF the client can build a Pinocchio
   model from.

The wire format is taken verbatim from
``libfranka_new/common/include/research_interface/robot/service_types.h``
(all structs under ``#pragma pack(push, 1)``).
"""

import struct

from franka_sim.franka_protocol import COMMAND_PORT, Command, ConnectStatus, MessageHeader

PROTOCOL_VERSION = 10


def _recv_exact(sock, size):
    """Receive exactly ``size`` bytes from ``sock``."""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed before receiving expected bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(sock):
    """Receive one ``MessageHeader`` + payload, returning (header, payload)."""
    header = MessageHeader.from_bytes(_recv_exact(sock, 12))
    payload = _recv_exact(sock, header.size - 12)
    return header, payload


def _connect(sock, command_id=1, version=PROTOCOL_VERSION, udp_port=1338):
    """Perform the Connect handshake and return (header, payload)."""
    sock.connect(("localhost", COMMAND_PORT))
    request = struct.pack("<HH", version, udp_port)
    header = MessageHeader(command=Command.kConnect, command_id=command_id, size=12 + len(request))
    sock.sendall(header.to_bytes() + request)
    return _recv_message(sock)


def test_connect_reports_protocol_v10(tcp_client, sim_server, mock_physics_sim):
    """Connect response must use the v10 layout: status(uint8) + version(uint16)."""
    header, payload = _connect(tcp_client)

    assert header.command == Command.kConnect
    # v10 ResponseBase::status is uint8, Connect adds a uint16 version -> 3 bytes.
    assert header.size == 12 + 3, "Connect response must be header(12) + status(1) + version(2)"

    status, server_version = struct.unpack("<BH", payload)
    assert status == ConnectStatus.kSuccess
    assert server_version == PROTOCOL_VERSION


def test_get_robot_model_returns_handless_fr3_urdf(tcp_client, sim_server, mock_physics_sim):
    """Verify GetRobotModel returns a buildable, hand-less FR3 arm URDF."""
    _connect(tcp_client, command_id=1)

    # GetRobotModel request carries an empty body (header only).
    request = MessageHeader(command=Command.kGetRobotModel, command_id=2, size=12)
    tcp_client.sendall(request.to_bytes())

    header, payload = _recv_message(tcp_client)
    assert header.command == Command.kGetRobotModel

    # v10 response: status(uint8) + URDF (UTF-8 string), via DynamicSizedCommandMessage.
    status = payload[0]
    urdf = payload[1:].decode("utf-8")
    assert status == 0  # kSuccess

    # It must be the 7-DOF arm (joint1..joint7) ...
    for i in range(1, 8):
        assert f'"joint{i}"' in urdf, f"URDF missing joint{i}"
    # ... and hand-less (no gripper finger joints) -- the hand is a later feature.
    assert "finger" not in urdf.lower(), "URDF unexpectedly contains a gripper/hand"
