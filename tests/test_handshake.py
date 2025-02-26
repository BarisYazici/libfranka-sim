import socket
import struct
import time

import pytest

from franka_sim.franka_protocol import COMMAND_PORT, Command, ConnectStatus, MessageHeader


def test_successful_handshake(tcp_client, sim_server, mock_genesis_sim):
    """Test successful handshake between client and server"""
    # Connect to server
    tcp_client.connect(("localhost", COMMAND_PORT))

    # Create connect message
    version = 9  # Current libfranka version
    udp_port = 1338  # Test UDP port
    payload = struct.pack("<HH", version, udp_port)

    # Create message header
    header = MessageHeader(
        command=Command.kConnect,
        command_id=1,
        size=12 + len(payload),  # Header size (12) + payload size
    )

    # Send message
    tcp_client.sendall(header.to_bytes() + payload)

    # Receive response
    response_header_data = tcp_client.recv(12)
    assert len(response_header_data) == 12

    response_header = MessageHeader.from_bytes(response_header_data)
    assert response_header.command == Command.kConnect
    assert response_header.command_id == 1

    response_data = tcp_client.recv(8)  # Status (2) + version (2) + padding (4)
    status, server_version = struct.unpack("<HH4x", response_data)

    assert status == ConnectStatus.kSuccess
    assert server_version == version

    # Wait a bit for UDP transmission to start
    time.sleep(0.1)

    # Verify mock simulator was used
    mock_genesis_sim.get_robot_state.assert_called()


@pytest.mark.skip(reason="Version check feature disabled")
def test_incompatible_version(tcp_client, sim_server, mock_genesis_sim):
    """Test handshake with incompatible version"""
    # Connect to server
    tcp_client.connect(("localhost", COMMAND_PORT))

    # Create connect message with incompatible version
    version = 1  # Old version
    udp_port = 1338
    payload = struct.pack("<HH", version, udp_port)

    header = MessageHeader(command=Command.kConnect, command_id=1, size=12 + len(payload))

    # Send message
    tcp_client.sendall(header.to_bytes() + payload)

    # Receive response
    response_header_data = tcp_client.recv(12)
    response_header = MessageHeader.from_bytes(response_header_data)

    response_data = tcp_client.recv(8)
    status, server_version = struct.unpack("<HH4x", response_data)

    assert status == ConnectStatus.kIncompatibleLibraryVersion

    # Connection should be closed after incompatible version
    with pytest.raises((ConnectionError, socket.error)):
        tcp_client.recv(1)


# TODO: Fix this test
@pytest.mark.skip(reason="Not raising error currently")
def test_invalid_command(tcp_client, sim_server, mock_genesis_sim):
    """Test sending invalid command during handshake"""
    # Connect to server
    tcp_client.connect(("localhost", COMMAND_PORT))

    # Create message with invalid command
    header = MessageHeader(
        command=Command.kMove, command_id=1, size=12  # Should be kConnect for handshake
    )

    # Send message
    tcp_client.sendall(header.to_bytes())

    # Server should close connection
    time.sleep(0.1)  # Give server time to process and close
    with pytest.raises((ConnectionError, socket.error)):
        tcp_client.recv(1)


# TODO: Fix this test
@pytest.mark.skip(reason="Not raising error currently")
def test_malformed_payload(tcp_client, sim_server, mock_genesis_sim):
    """Test handshake with malformed payload"""
    # Connect to server
    tcp_client.connect(("localhost", COMMAND_PORT))

    # Create connect message with incomplete payload
    payload = struct.pack("<H", 9)  # Only version, missing UDP port

    header = MessageHeader(command=Command.kConnect, command_id=1, size=12 + len(payload))

    # Send message
    tcp_client.sendall(header.to_bytes() + payload)

    # Server should close connection
    time.sleep(0.1)  # Give server time to process and close
    with pytest.raises((ConnectionError, socket.error)):
        tcp_client.recv(1)
