"""Regression tests for FrankaSimServer.cleanup(): the check-then-act socket
teardown race (Critical, final review of feat/mobile-duo-sim).

A concurrent connection teardown (handle_client's reset_state(), running on
the per-client thread) can null out ``self.client_socket`` between
cleanup()'s shutdown() and close() calls on that attribute. Before the fix,
re-reading the attribute for the second call hit None and raised
AttributeError, which ``except socket.error`` does not catch -- aborting the
rest of cleanup() and leaking the SO_REUSEPORT listener socket (still bound,
so it silently keeps receiving traffic for a scene the caller believes is
dead).
"""

from franka_sim.franka_sim_server import FrankaSimServer


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
