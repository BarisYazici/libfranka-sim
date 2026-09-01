"""Pytest plugin: boot a real franka-sim server for a test session.

Installed with the package and registered under the ``pytest11`` entry point,
so any project that has ``franka-sim`` in its test environment gets the
``franka_sim_server`` fixture with no conftest wiring::

    def test_my_controller(franka_sim_server):
        robot = my_stack.connect(franka_sim_server.host, franka_sim_server.port)
        ...

The fixture starts ``franka_sim.run_server`` as a subprocess on a free TCP
port (so parallel CI jobs on one machine never collide), waits until the
server passes the real-handshake readiness probe (`franka_sim.health_check`),
and tears the process down at the end of the session. It is session-scoped:
one simulated robot serves the whole suite, exactly as one real robot would.

The co-located gripper server is disabled by default because its port (1338)
is fixed by the libfranka gripper protocol and would collide across parallel
sessions. To test gripper code, override the args fixture in your conftest —
dropping ``--no-gripper`` re-enables it on port 1338::

    @pytest.fixture(scope="session")
    def franka_sim_server_args():
        return []  # gripper on; don't run two such sessions on one machine
"""

import os
import signal
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import pytest

from franka_sim.health_check import HealthCheckError, check_server

#: Readiness budget for the subprocess. Generous because the very first run on
#: a fresh machine downloads the FR3 model through robot_descriptions before
#: the server can listen.
DEFAULT_STARTUP_TIMEOUT_S = 180.0

#: How long teardown waits after SIGINT before escalating to SIGKILL. The
#: server's own shutdown watchdog forces it out at 15 s, so 20 s means a
#: SIGKILL only ever reaps a process wedged beyond its last-resort exit.
DEFAULT_STOP_GRACE_S = 20.0


class FrankaSimStartupError(RuntimeError):
    """The franka-sim subprocess never became ready; carries its output tail."""


@dataclass
class FrankaSimProcess:
    """Handle to the running server subprocess handed to tests."""

    host: str
    port: int
    process: subprocess.Popen = field(repr=False)

    @property
    def address(self) -> str:
        """``host:port`` as most client configs want it."""
        return f"{self.host}:{self.port}"


def free_tcp_port() -> int:
    """Pick a TCP port that is free right now.

    The classic bind-close-reuse race is possible but vanishingly rare in
    practice, and the readiness probe turns a lost race into a clear startup
    failure rather than a hang.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start_server(
    extra_args: Sequence[str] = ("--no-gripper",),
    port: Optional[int] = None,
    timeout: float = DEFAULT_STARTUP_TIMEOUT_S,
    _command: Optional[List[str]] = None,
) -> FrankaSimProcess:
    """Launch a franka-sim server subprocess and wait until it serves the FCI.

    Raises FrankaSimStartupError (with the process output tail) if the server
    exits early or never passes the readiness probe. ``_command`` replaces the
    whole command line and exists for this module's own tests.
    """
    if port is None:
        port = free_tcp_port()
    command = _command or [
        sys.executable,
        "-m",
        "franka_sim.run_server",
        "--port",
        str(port),
        *extra_args,
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,  # its own process group: signals never leak to pytest
    )
    server = FrankaSimProcess(host="127.0.0.1", port=port, process=process)
    try:
        check_server(server.host, server.port, timeout=timeout)
    except HealthCheckError as exc:
        stop_server(server, grace=5.0)
        tail = _drain_output(process)
        raise FrankaSimStartupError(
            f"franka-sim server on port {port} did not become ready: {exc}\n"
            f"--- server output ---\n{tail}"
        ) from exc
    return server


def stop_server(server: FrankaSimProcess, grace: float = DEFAULT_STOP_GRACE_S) -> None:
    """Stop the subprocess: SIGINT (clean server shutdown), then SIGKILL."""
    process = server.process
    if process.poll() is not None:
        return
    _signal_group(process, signal.SIGINT)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        _signal_group(process, signal.SIGKILL)
        process.wait(timeout=5.0)


def _signal_group(process: subprocess.Popen, sig: int) -> None:
    """Signal the child's whole process group, tolerating an already-dead child."""
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            pass


def _drain_output(process: subprocess.Popen, limit: int = 8000) -> str:
    """Return the tail of the (now dead or dying) process' combined output."""
    if process.stdout is None:
        return "<output not captured>"
    try:
        output = process.stdout.read() or ""
    except Exception:
        return "<output unavailable>"
    return output[-limit:]


@pytest.fixture(scope="session")
def franka_sim_server_args() -> List[str]:
    """Extra CLI args for the session server; override in conftest to customise."""
    return ["--no-gripper"]


@pytest.fixture(scope="session")
def franka_sim_server(franka_sim_server_args) -> FrankaSimProcess:
    """A real franka-sim server (MuJoCo physics) shared by the whole session."""
    server = start_server(extra_args=franka_sim_server_args)
    yield server
    stop_server(server)
