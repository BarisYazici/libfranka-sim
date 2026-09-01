"""Tests for franka_sim.health_check — the CI/container readiness probe.

The probe speaks the real wire protocol: a v10 Connect handshake over TCP,
then one RobotState datagram over UDP. Success proves the server is not just
listening but actually serving the FCI.
"""

import socket
import struct
import threading

import pytest

from franka_sim.franka_protocol import COMMAND_PORT
from franka_sim.health_check import HealthCheckError, check_server, main


def _free_tcp_port() -> int:
    """Reserve a port nothing is listening on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fake_tcp_server(response: bytes) -> int:
    """One-shot TCP server that answers any connection with ``response``."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]

    def serve():
        with listener:
            conn, _ = listener.accept()
            with conn:
                conn.recv(4096)
                conn.sendall(response)

    threading.Thread(target=serve, daemon=True).start()
    return port


def test_check_server_succeeds_against_live_server(sim_server, mock_physics_sim):
    report = check_server("127.0.0.1", COMMAND_PORT, timeout=10.0)
    assert report.server_version == 10
    # One real RobotState datagram arrived — the server is streaming, not
    # merely accepting TCP connections.
    assert report.state_bytes > 0


def test_check_server_raises_when_nothing_listens():
    port = _free_tcp_port()
    with pytest.raises(HealthCheckError):
        check_server("127.0.0.1", port, timeout=2.0)


def test_garbage_response_is_a_health_error_not_a_crash():
    # An HTTP server (or anything else) on the probed port must surface as
    # HealthCheckError, never as a raw ValueError from the enum parse.
    port = _fake_tcp_server(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
    with pytest.raises(HealthCheckError, match="malformed"):
        check_server("127.0.0.1", port, timeout=3.0)


def test_short_size_field_is_a_health_error_not_a_struct_error():
    # Well-formed header claiming a body too small to hold status+version.
    response = struct.pack("<III", 0, 1, 13) + b"\x00"
    port = _fake_tcp_server(response)
    with pytest.raises(HealthCheckError, match="claims 13 bytes"):
        check_server("127.0.0.1", port, timeout=3.0)


def test_main_returns_zero_on_success(sim_server, mock_physics_sim, capsys):
    assert main(["--host", "127.0.0.1", "--port", str(COMMAND_PORT), "--timeout", "10"]) == 0
    out = capsys.readouterr().out
    assert "ok" in out.lower()


def test_main_returns_nonzero_on_failure(capsys):
    port = _free_tcp_port()
    assert main(["--host", "127.0.0.1", "--port", str(port), "--timeout", "2"]) != 0
    err = capsys.readouterr().err
    assert err.strip()  # says why, on stderr
