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
The server's log is written to a temp file (``franka_sim_server.log_path``)
rather than a pipe, so a chatty server can never block on an unread pipe.

The co-located gripper server is disabled by default because its port (1338)
is fixed by the libfranka gripper protocol and would collide across parallel
sessions. To test gripper code, override the args fixture in your conftest —
dropping ``--no-gripper`` re-enables it on port 1338::

    @pytest.fixture(scope="session")
    def franka_sim_server_args():
        return []  # gripper on; don't run two such sessions on one machine

Known limit: the subprocess runs in its own session (signals aimed at pytest
must not hit it), so if pytest itself is SIGKILLed the server outlives it and
keeps its port until killed by hand.
"""

import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
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

#: Upper bound on one readiness attempt inside the startup loop; short enough
#: that a child that dies mid-attempt is noticed promptly.
_PROBE_SLICE_S = 5.0


class FrankaSimStartupError(RuntimeError):
    """The franka-sim subprocess never became ready; carries its output tail."""


@dataclass
class FrankaSimProcess:
    """Handle to the running server subprocess handed to tests."""

    host: str
    port: int
    process: subprocess.Popen = field(repr=False)
    log_path: Optional[str] = None

    @property
    def address(self) -> str:
        """``host:port`` as most client configs want it."""
        return f"{self.host}:{self.port}"


def free_tcp_port() -> int:
    """Pick a TCP port that is free right now.

    Binds the wildcard address because that is what the server itself binds.
    The classic bind-close-reuse race is possible but vanishingly rare in
    practice, and the readiness probe turns a lost race into a clear startup
    failure rather than a hang.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def start_server(
    extra_args: Sequence[str] = ("--no-gripper",),
    port: Optional[int] = None,
    timeout: float = DEFAULT_STARTUP_TIMEOUT_S,
    _command: Optional[List[str]] = None,
) -> FrankaSimProcess:
    """Launch a franka-sim server subprocess and wait until it serves the FCI.

    Raises FrankaSimStartupError (with the process output tail) if the server
    exits early or never passes the readiness probe; the subprocess is never
    left running on failure. ``_command`` replaces the whole command line and
    exists for this module's own tests.
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
    # A file, not a pipe: nothing reads the server's output while tests run,
    # and a filled pipe would block the server's logging — including the 1 kHz
    # state-broadcast thread — wedging the whole simulation mid-session.
    log_file = tempfile.NamedTemporaryFile(
        mode="w", prefix="franka-sim-", suffix=".log", delete=False
    )
    with log_file:
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,  # its own process group: signals never leak to pytest
        )
    server = FrankaSimProcess(host="127.0.0.1", port=port, process=process, log_path=log_file.name)

    deadline = time.monotonic() + timeout
    last_probe_error: Optional[Exception] = None
    while True:
        if process.poll() is not None:
            # Fail fast: a dead child will not come up however long we wait.
            raise FrankaSimStartupError(
                f"franka-sim server on port {port} exited with code "
                f"{process.returncode} before becoming ready\n"
                f"--- server output ({server.log_path}) ---\n{_log_tail(server.log_path)}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            stop_server(server, grace=5.0)
            raise FrankaSimStartupError(
                f"franka-sim server on port {port} did not become ready within "
                f"{timeout:.0f}s (last probe error: {last_probe_error})\n"
                f"--- server output ({server.log_path}) ---\n{_log_tail(server.log_path)}"
            )
        try:
            check_server(server.host, server.port, timeout=min(_PROBE_SLICE_S, remaining))
            return server
        except HealthCheckError as exc:
            last_probe_error = exc
        except Exception as exc:
            # A probe bug or a garbled response must still not leak the child.
            stop_server(server, grace=5.0)
            raise FrankaSimStartupError(
                f"readiness probe failed unexpectedly for the franka-sim server on "
                f"port {port}: {exc!r}\n"
                f"--- server output ({server.log_path}) ---\n{_log_tail(server.log_path)}"
            ) from exc


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
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            # SIGKILL was delivered; reaping is now the OS's problem. Do not
            # let teardown die over it.
            pass


def _signal_group(process: subprocess.Popen, sig: int) -> None:
    """Signal the child's whole process group, tolerating an already-dead child."""
    try:
        os.killpg(process.pid, sig)
    except (ProcessLookupError, PermissionError):
        try:
            process.send_signal(sig)
        except ProcessLookupError:
            pass


def _log_tail(log_path: Optional[str], limit: int = 8000) -> str:
    """Return the tail of the server's log file."""
    if not log_path:
        return "<output not captured>"
    try:
        with open(log_path, "r", errors="replace") as handle:
            return handle.read()[-limit:]
    except OSError:
        return f"<could not read {log_path}>"


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
