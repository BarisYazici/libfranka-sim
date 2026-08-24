"""Regression tests for FrankaSimServer's teardown: cleanup()'s check-then-act
socket race, and the shutdown contract stop() has to honour.

A concurrent connection teardown (handle_client's reset_state(), running on
the per-client thread) can null out ``self.client_socket`` between
cleanup()'s shutdown() and close() calls on that attribute. Before the fix,
re-reading the attribute for the second call hit None and raised
AttributeError, which ``except socket.error`` does not catch -- aborting the
rest of cleanup() and leaking the SO_REUSEPORT listener socket (still bound,
so it silently keeps receiving traffic for a scene the caller believes is
dead).

The second group of tests covers stop() itself, after a Ctrl+C could leave
``python -m franka_sim.run_server`` running forever: shutdown must be
bounded, idempotent (a second Ctrl+C lands inside the first one's teardown),
survive a stage that raises, leave no non-daemon thread behind, and give the
listening port back. And a plain client disconnect must not trigger any of
it -- the server serves the next client.
"""

import errno
import logging
import socket
import threading
import time

import pytest

from franka_sim.franka_sim_server import FrankaSimServer


def _free_port() -> int:
    """Pick a port nothing is listening on, for a server this test owns."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


@pytest.fixture
def listening_server(mock_genesis_sim):
    """A FrankaSimServer with its accept loop running on a private port.

    Yields ``(server, thread, port)``. The accept thread is deliberately
    non-daemon: these tests assert that stop() actually ends it, and a daemon
    thread would let a broken stop() pass by dying with the interpreter.
    """
    port = _free_port()
    server = FrankaSimServer(
        host="127.0.0.1", port=port, genesis_sim=mock_genesis_sim, enable_gripper=False
    )
    thread = threading.Thread(target=server.run_server, name="test-accept-loop")
    thread.start()

    deadline = time.time() + 5.0
    while time.time() < deadline and not server.running:
        time.sleep(0.01)
    assert server.running, "server never started listening"

    try:
        yield server, thread, port
    finally:
        server.stop()
        thread.join(timeout=5.0)


class RacyCloseSocket:
    """A socket double that reproduces the teardown race.

    Its shutdown() nulls the named attribute on ``owner`` as a side effect --
    standing in for another thread's reset_state() running concurrently with
    cleanup(). If cleanup() re-reads the attribute for the close() call
    instead of using a locally cached reference, this raises AttributeError.
    """

    def __init__(self, owner, attr_name):
        self._owner = owner
        self._attr_name = attr_name
        self.shutdown_called = False
        self.close_called = False

    def shutdown(self, how):
        self.shutdown_called = True
        setattr(self._owner, self._attr_name, None)

    def close(self):
        self.close_called = True


class RecordingSocket:
    """A plain socket double that just records shutdown()/close() calls."""

    def __init__(self):
        self.shutdown_called = False
        self.close_called = False

    def shutdown(self, how):
        self.shutdown_called = True

    def close(self):
        self.close_called = True


def test_cleanup_survives_client_socket_nulled_between_shutdown_and_close(mock_genesis_sim):
    """The exact race from the review: client_socket races cleanup(); server_socket
    (a stand-in for the still-listening SO_REUSEPORT socket) must still get closed.
    """
    server = FrankaSimServer(genesis_sim=mock_genesis_sim, enable_gripper=False)
    client_sock = RacyCloseSocket(server, "client_socket")
    server_sock = RecordingSocket()
    server.client_socket = client_sock
    server.server_socket = server_sock

    server.cleanup()  # must not raise

    assert client_sock.shutdown_called is True
    # close() ran on the *cached* reference even though the attribute itself
    # was nulled out mid-cleanup by the racing shutdown() side effect.
    assert client_sock.close_called is True
    assert server.client_socket is None

    assert server_sock.shutdown_called is True
    assert server_sock.close_called is True
    assert server.server_socket is None


def test_cleanup_survives_server_socket_nulled_between_shutdown_and_close(mock_genesis_sim):
    """The same race, but on server_socket itself (e.g. a concurrent restart)."""
    server = FrankaSimServer(genesis_sim=mock_genesis_sim, enable_gripper=False)
    server_sock = RacyCloseSocket(server, "server_socket")
    server.server_socket = server_sock

    server.cleanup()  # must not raise

    assert server_sock.shutdown_called is True
    assert server_sock.close_called is True
    assert server.server_socket is None


def test_cleanup_survives_command_socket_nulled_between_shutdown_and_close(mock_genesis_sim):
    server = FrankaSimServer(genesis_sim=mock_genesis_sim, enable_gripper=False)
    command_sock = RacyCloseSocket(server, "command_socket")
    server.command_socket = command_sock

    server.cleanup()  # must not raise

    assert command_sock.shutdown_called is True
    assert command_sock.close_called is True
    assert server.command_socket is None


def test_cleanup_survives_udp_socket_nulled_between_check_and_close(mock_genesis_sim):
    """udp_socket has no shutdown() call, but the same check-then-act race applies
    between the truthiness check and close() -- cache-before-use covers it too.
    """
    server = FrankaSimServer(genesis_sim=mock_genesis_sim, enable_gripper=False)

    class RacyUdpSocket:
        def __init__(self, owner):
            self._owner = owner
            self.close_called = False

        def close(self):
            # Simulate the attribute being nulled by another thread just
            # before this close() call actually runs.
            self._owner.udp_socket = None
            self.close_called = True

    udp_sock = RacyUdpSocket(server)
    server.udp_socket = udp_sock

    server.cleanup()  # must not raise

    assert udp_sock.close_called is True
    assert server.udp_socket is None


def test_stale_command_thread_does_not_kill_the_next_sessions_connection_running(
    mock_genesis_sim,
):
    """A dead session's UDP thread must not clear the NEW session's flag.

    Regression: ``_handle_commands`` used to read ``self.udp_socket`` at
    runtime instead of the socket it was actually started for. After a fast
    reconnect, ``self.udp_socket`` points at the new session's socket while
    a stale thread from the old session is still unwinding on its own
    (about to be closed) fd. ``self.connection_running`` is one flag on the
    server instance, not scoped per-thread, so when that stale thread's poll
    loop saw its own fd hang up and cleared the flag unconditionally, it
    killed the NEW session's flag out from under it -- the new session's
    broadcast loop then exited before sending a single state datagram, and
    the new client saw "libfranka: UDP receive: Timeout" on a connection
    that never did anything wrong.

    This reproduces the race directly, without a live TCP handshake: start
    the command receiver on an "old" socket, then simulate a reconnect by
    pointing ``self.udp_socket`` at a "new" one before the old socket's
    thread notices its fd died. The flag must survive, and the stale thread
    must still exit on its own (it is not allowed to spin on a dead fd).
    """
    server = FrankaSimServer(genesis_sim=mock_genesis_sim, enable_gripper=False)
    server.running = True

    old_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    old_socket.bind(("127.0.0.1", 0))
    server.udp_socket = old_socket
    server.connection_running = True

    server.start_command_receiver()
    thread = server.command_thread
    assert thread is not None

    # Give the thread a moment to actually start polling old_socket's fd
    # before the "reconnect" swaps self.udp_socket out from under it.
    time.sleep(0.05)

    new_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    new_socket.bind(("127.0.0.1", 0))
    try:
        # Simulate the fast reconnect: a NEW session takes over
        # self.udp_socket and connection_running while the OLD thread is
        # still alive, polling the socket it was actually started with.
        server.udp_socket = new_socket
        server.connection_running = True

        # Kill the old thread's own socket. Its poll() will see the fd hang
        # up / error out on the next iteration.
        old_socket.close()

        thread.join(timeout=2.0)
        assert not thread.is_alive(), "the stale thread never noticed its own socket died"

        # The critical assertion: the NEW session's flag must be untouched.
        assert server.connection_running is True
        assert server.udp_socket is new_socket
    finally:
        server.connection_running = False
        new_socket.close()


# --------------------------------------------------------------------------
# stop(): the shutdown contract behind Ctrl+C
# --------------------------------------------------------------------------


def test_stop_is_idempotent(mock_genesis_sim):
    """A second stop() (i.e. a second Ctrl+C) must be a quiet no-op.

    It used to re-run the whole teardown, including a second gripper-thread
    join -- which is exactly where the user's second Ctrl+C landed.
    """
    server = FrankaSimServer(genesis_sim=mock_genesis_sim, enable_gripper=False)

    server.stop()
    server.stop()
    server.stop()

    assert mock_genesis_sim.stop.call_count == 1


def test_stop_does_not_raise_when_a_stage_fails(mock_genesis_sim, caplog):
    """One broken stage must not abort the rest of the shutdown.

    cleanup() runs first, so a simulator that raises used to be harmless --
    but the reverse (a cleanup that raises) skipped the simulator, leaving
    the viewer's GL context alive, which is what wedges the process on exit.
    Every stage is isolated now; the failure is logged, not propagated.
    """
    server = FrankaSimServer(genesis_sim=mock_genesis_sim, enable_gripper=False)
    mock_genesis_sim.stop.side_effect = RuntimeError("viewer already gone")

    with caplog.at_level(logging.ERROR):
        server.stop()  # must not raise

    assert any("simulator" in record.message for record in caplog.records)


def test_cleanup_announces_itself_once(mock_genesis_sim, caplog):
    """The accept loop's finally clause calls cleanup() after stop() already did.

    Both runs are harmless (every step is None-guarded), but two "Cleaning up
    server resources..." lines per shutdown read like two shutdowns -- which
    is what made the reported log look as though a client disconnect had
    triggered one.
    """
    server = FrankaSimServer(genesis_sim=mock_genesis_sim, enable_gripper=False)

    with caplog.at_level(logging.INFO, logger="franka_sim.franka_sim_server"):
        server.cleanup()
        server.cleanup()

    announcements = [r for r in caplog.records if r.message.startswith("Cleaning up server")]
    assert len(announcements) == 1


def test_a_second_server_cannot_co_bind_the_command_port():
    """The listener must refuse to share a port that is already taken.

    `SO_REUSEPORT` used to let two servers co-bind with no error on either
    side, after which the kernel load-balanced incoming clients between
    them -- silent cross-talk on both. The occupier here opts INTO
    SO_REUSEPORT on purpose: the bind can only fail because our listener
    abstains, which is exactly the property this test pins.
    """
    port = _free_port()
    occupier = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupier.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    occupier.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    occupier.bind(("127.0.0.1", port))
    occupier.listen(1)
    try:
        server = FrankaSimServer.__new__(FrankaSimServer)
        server.host = "127.0.0.1"
        server.port = port
        server.server_socket = None
        with pytest.raises(OSError) as excinfo:
            server._bind_listener()
        assert excinfo.value.errno == errno.EADDRINUSE
        # The failed attempt must not leave a half-open socket behind.
        assert server.server_socket is None
    finally:
        occupier.close()


def test_stop_ends_the_accept_loop_and_frees_the_port(listening_server):
    """stop() must break accept() out of its wait and give the port back."""
    server, thread, port = listening_server

    started = time.monotonic()
    server.stop()
    elapsed = time.monotonic() - started

    assert elapsed < 3.0, f"stop() took {elapsed:.2f}s"
    thread.join(timeout=3.0)
    assert not thread.is_alive(), "the accept loop outlived stop()"

    # The listener really is gone: a fresh bind on the same port succeeds.
    # (SO_REUSEPORT means a leaked listener would *not* show up as EADDRINUSE,
    # so bind alone proves nothing -- connect must be refused as well.)
    rebound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        rebound.bind(("127.0.0.1", port))
    finally:
        rebound.close()

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    with pytest.raises(OSError):
        probe.connect(("127.0.0.1", port))
    probe.close()


def test_stop_returns_promptly_with_a_client_connected(listening_server):
    """The reported hang happened with a session open, so that is the case that counts."""
    server, thread, port = listening_server

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", port))
    # Let the accept loop hand the connection to handle_client, which then
    # blocks reading the Connect message this client never sends.
    deadline = time.time() + 3.0
    while time.time() < deadline and not server.connection_running:
        time.sleep(0.01)
    assert server.connection_running, "server never picked the connection up"

    try:
        started = time.monotonic()
        server.stop()
        elapsed = time.monotonic() - started
    finally:
        client.close()

    assert elapsed < 3.0, f"stop() took {elapsed:.2f}s with a client connected"
    thread.join(timeout=3.0)
    assert not thread.is_alive(), "the accept loop outlived stop()"


def test_stop_leaves_no_non_daemon_threads(mock_genesis_sim):
    """Nothing the server starts may be able to keep the process alive.

    Serving threads are daemons on purpose: every join in stop() is bounded,
    and a join that times out must degrade to an abandoned thread, never to a
    process that will not exit.
    """
    before = set(threading.enumerate())

    server = FrankaSimServer(
        host="127.0.0.1", port=_free_port(), genesis_sim=mock_genesis_sim, enable_gripper=True
    )
    server.start_gripper_server()
    thread = threading.Thread(target=server.run_server, name="test-accept-loop", daemon=True)
    thread.start()
    deadline = time.time() + 5.0
    while time.time() < deadline and not server.running:
        time.sleep(0.01)

    try:
        server.stop()
    finally:
        thread.join(timeout=3.0)

    survivors = [
        t for t in threading.enumerate() if t not in before and t is not thread and t.is_alive()
    ]
    assert [t.name for t in survivors if not t.daemon] == []


def test_client_disconnect_does_not_stop_the_server(listening_server):
    """A client hanging up ends its session, not the server.

    The reported log showed "Client session ended, ready for next client"
    immediately followed by "Cleaning up server resources...", which reads
    like a disconnect tearing the whole server down. It does not: the accept
    loop only leaves (and only then cleans up) when ``running`` goes false,
    and nothing on the disconnect path touches it. Pinned here so it stays
    that way.
    """
    server, thread, port = listening_server

    first = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    first.connect(("127.0.0.1", port))
    deadline = time.time() + 3.0
    while time.time() < deadline and not server.connection_running:
        time.sleep(0.01)
    first.close()

    # The next client is served, which can only happen if the accept loop and
    # its listening socket survived the disconnect.
    deadline = time.time() + 3.0
    while time.time() < deadline and server.connection_running:
        time.sleep(0.01)

    second = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    second.settimeout(2.0)
    try:
        second.connect(("127.0.0.1", port))
    finally:
        second.close()

    assert server.running is True
    assert server.server_socket is not None
    assert server._stopping is False
    assert server._cleanup_logged is False, "cleanup() ran on a plain client disconnect"


# --------------------------------------------------------------------------
# The CLI's shutdown wrapper: what a second Ctrl+C does
# --------------------------------------------------------------------------


class _ExitCalled(Exception):
    """Stands in for os._exit, which a test process cannot afford to run."""

    def __init__(self, code):
        super().__init__(code)
        self.code = code


@pytest.fixture
def no_watchdog(monkeypatch):
    """Disarm the shutdown watchdog and neuter os._exit for the CLI tests.

    Both would take the *pytest* process down: the watchdog thread outlives
    the call that armed it by design, because the wedge it exists to break is
    in the interpreter's own exit path.
    """
    from franka_sim import run_server

    monkeypatch.setattr(run_server, "_arm_exit_watchdog", lambda *a, **k: None)

    def fake_exit(code):
        raise _ExitCalled(code)

    monkeypatch.setattr(run_server.os, "_exit", fake_exit)
    return run_server


def test_shutdown_turns_a_second_interrupt_into_an_immediate_exit(no_watchdog):
    """A Ctrl+C during stop() must not become a traceback on top of the shutdown log."""

    class Interrupted:
        def stop(self):
            raise KeyboardInterrupt()

    with pytest.raises(_ExitCalled) as exit_info:
        no_watchdog._shutdown(Interrupted())

    assert exit_info.value.code == 130


def test_shutdown_swallows_a_failing_stop(no_watchdog):
    """A stop() that raises must not stop the process from exiting normally."""

    class Broken:
        def stop(self):
            raise RuntimeError("teardown exploded")

    no_watchdog._shutdown(Broken())  # must not raise


def test_exit_watchdog_forces_the_process_out_and_is_a_daemon(monkeypatch):
    """The last-resort watchdog: fires after its timeout, never holds the process open.

    The real ``_arm_exit_watchdog`` is used here (not the no_watchdog stub) --
    it is the thing under test -- with only its exit call intercepted.
    """
    from franka_sim import run_server

    exited = []
    monkeypatch.setattr(run_server, "_force_exit", lambda message, code=130: exited.append(code))

    before = set(threading.enumerate())
    run_server._arm_exit_watchdog(0.05)
    watchdog = next(t for t in threading.enumerate() if t not in before)
    assert watchdog.daemon is True, "a non-daemon watchdog would itself delay every exit"

    watchdog.join(timeout=3.0)
    assert exited == [130]
