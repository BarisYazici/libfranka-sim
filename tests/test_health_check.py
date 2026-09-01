"""Tests for franka_sim.health_check — the CI/container readiness probe.

The probe speaks the real wire protocol: a v10 Connect handshake over TCP,
then one RobotState datagram over UDP. Success proves the server is not just
listening but actually serving the FCI.
"""

import socket

import pytest

from franka_sim.franka_protocol import COMMAND_PORT
from franka_sim.health_check import HealthCheckError, check_server, main


def _free_tcp_port() -> int:
    """Reserve a port nothing is listening on."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


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


def test_main_returns_zero_on_success(sim_server, mock_physics_sim, capsys):
    assert main(["--host", "127.0.0.1", "--port", str(COMMAND_PORT), "--timeout", "10"]) == 0
    out = capsys.readouterr().out
    assert "ok" in out.lower()


def test_main_returns_nonzero_on_failure(capsys):
    port = _free_tcp_port()
    assert main(["--host", "127.0.0.1", "--port", str(port), "--timeout", "2"]) != 0
    err = capsys.readouterr().err
    assert err.strip()  # says why, on stderr
