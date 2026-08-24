"""Server and session lifecycle: state reset, accept loop, shutdown.

Building the per-session state a control session starts from, running the
accept loop, and taking the whole thing down again.

``cleanup``/``stop`` are idempotent and must not raise: shutdown runs on every
exit path (including the viewer window closing), and a stage that fails must
still let the remaining stages run.
"""

import errno
import socket
import threading
import time
from pathlib import Path

from franka_sim.comm_constraints import CommConstraintTracker
from franka_sim.control_modes import ControlMode
from franka_sim.motion_limits import MotionLimitChecker
from franka_sim.robot_state import RobotState
from franka_sim.server.constants import (
    GRIPPER_JOIN_TIMEOUT_S,
    logger,
)


class LifecycleMixin:
    """See the module docstring; this mixin carries no state of its own."""

    def _load_robot_model(self, urdf_path):
        """Read the URDF served via GetRobotModel (defaults to the bundled FR3)."""
        path = Path(urdf_path) if urdf_path is not None else self.DEFAULT_ARM_URDF
        with open(path, "r", encoding="utf-8") as urdf_file:
            return urdf_file.read()

    def reset_state(self):
        """Reset all connection-specific state variables for a new connection"""
        self.transmitting_state = False
        self.current_motion_id = 0
        self._pending_move_response = None
        self._states_packed = 0
        self.states_sent = 0
        self._motion_epoch_id = 0
        self._motion_has_commands = False
        self.client_socket = None
        self.tcp_thread = None
        self.udp_socket = None
        self.client_address = None
        self.client_udp_port = None
        self.control_mode = ControlMode.NONE
        self.connection_running = False
        self._mobile_hold_logged = False
        self._seed_pose_fallback_logged = False
        # The hold itself is *not* undone here -- the simulator keeps holding
        # the pose it was recaptured at until the next Move commands something
        # else. Only the once-per-session latch is rearmed.
        self._idle_hold = False
        # A new connection is a new communication channel: no window and no
        # latched violation.
        self.comm = CommConstraintTracker(enforce=self.enforce_comm_constraints)
        # ...and no latched motion-limit violation, no command history to
        # difference the new client's first command against.
        self.motion_limits = MotionLimitChecker(enforce=self.enforce_motion_limits)
        self.robot_state = RobotState()  # Create fresh robot state for new connection
        # The fresh RobotState puts F_T_NE/NE_T_EE back to identity, so the
        # backend has to be told the EE frame moved back to the flange too --
        # otherwise a tool set by the previous connection would keep skewing
        # this one's measured EE velocity.
        self._refresh_ee_transform()

    def _bind_listener(self):
        """Create, bind and listen the TCP command socket -- loudly.

        Deliberately no ``SO_REUSEPORT``: with it set, two servers co-bind
        the FCI port without an error on either side and the kernel
        load-balances incoming clients between them, silently corrupting
        both (observed as spurious connection timeouts and cross-talk in
        long-running client sessions). A second server must die here with
        ``EADDRINUSE`` instead. ``SO_REUSEADDR`` stays so a restart can
        rebind through a lingering TIME_WAIT.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(1.0)
        try:
            sock.bind((self.host, self.port))
        except OSError as e:
            sock.close()
            if e.errno == errno.EADDRINUSE:
                logger.error(
                    "Port %s:%s is already in use -- another franka-sim server "
                    "(or another process) is listening there. Stop it or pass "
                    "a different port; refusing to co-bind.",
                    self.host,
                    self.port,
                )
            else:
                logger.error(f"Failed to bind: {e}")
            raise
        sock.listen(1)
        self.server_socket = sock
        logger.info(f"Server listening on {self.host}:{self.port}")

    def run_server(self):
        """Main server loop that runs in a separate thread when visualization is enabled"""
        self._arm_shutdown()
        try:
            # start() usually binds the listener up front (fail-fast on a
            # busy port, before physics init); bind here only when
            # run_server() is driven directly, e.g. by tests.
            if self.server_socket is None:
                self._bind_listener()
            self.running = True

            while self.running:
                try:
                    # Reset state before accepting new connection
                    self.reset_state()
                    logger.info("Server ready for new client connection...")

                    client_socket, address = self.server_socket.accept()
                    client_ip = address[0]
                    client_port = address[1]
                    logger.info(f"New connection from {client_ip}:{client_port}")

                    client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                    # Handle client - this will block until client disconnects
                    self.handle_client(client_socket)

                    logger.info("Client session ended, ready for next client")

                except socket.timeout:
                    # Just continue waiting for new connections
                    continue
                except OSError as e:
                    # stop() closes the listening socket from another thread --
                    # that is what breaks this accept() out of its wait. The
                    # EBADF/EINVAL that follows is the shutdown signal, not a
                    # failure, and logging it with a traceback made every clean
                    # Ctrl+C look like a crash.
                    if not self.running or e.errno in (errno.EBADF, errno.EINVAL):
                        logger.debug("Accept loop stopping: %s", e)
                        break
                    logger.error(f"Connection handling error: {e}", exc_info=True)
                    self.reset_state()
                    continue
                except Exception as e:
                    logger.error(f"Connection handling error: {e}", exc_info=True)
                    if "client_socket" in locals():
                        try:
                            client_socket.close()
                        except Exception as e:
                            logger.error(f"Error closing client socket: {e}")
                    # Reset state and continue listening for next client
                    self.reset_state()
                    continue

        except Exception as e:
            logger.error(f"Server start error: {e}", exc_info=True)
            self.running = False
        finally:
            self.cleanup()

    def start_gripper_server(self):
        """Launch the co-located gripper server's accept loop in a daemon thread."""
        if self.gripper_server is None:
            return
        self.gripper_thread = threading.Thread(target=self.gripper_server.run_server, daemon=True)
        self.gripper_thread.start()
        logger.info("Gripper server running in background thread")

    def _arm_shutdown(self):
        """Re-arm the once-per-run shutdown latches, at the start of a run.

        stop() is idempotent *within a run*, not for the lifetime of the
        object: a server that is started again (the accept loop is entered
        afresh, so a new listening socket exists) must be stoppable again, or
        the second stop() would return early and leak that socket.
        """
        with self._stop_lock:
            self._stopping = False
            self._cleanup_logged = False

    def start(self):
        """Start the TCP server and Genesis simulator"""
        self._arm_shutdown()
        try:
            self.running = True
            logger.info("Starting server and simulation")

            # Claim the FCI port before the (slow) physics init so a busy
            # port kills the process immediately with EADDRINUSE.
            self._bind_listener()

            # Initialize Genesis simulator first
            self.physics_sim.initialize_simulation()
            logger.info("Genesis simulation initialized")

            # Bring up the gripper server alongside the arm (port 1338).
            self.start_gripper_server()

            if self.physics_sim.enable_vis:
                # Run server in a background thread when visualization is enabled
                server_thread = threading.Thread(target=self.run_server)
                server_thread.daemon = True
                server_thread.start()
                logger.info("Server running in background thread")

                # Start Genesis simulator (visualization) in main thread
                logger.info("Starting Genesis simulator with visualization")
                self.physics_sim.start()
            else:
                # Without visualization, run the TCP/UDP server in a background
                # thread and step the Genesis physics loop in the main thread.
                # (run_server() blocks in its accept loop, so it must not run in
                # the main thread or the simulation would never step.)
                server_thread = threading.Thread(target=self.run_server)
                server_thread.daemon = True
                server_thread.start()
                logger.info("Server running in background thread (headless)")

                logger.info("Starting Genesis simulator (headless)")
                self.physics_sim.start()

        except Exception as e:
            logger.error(f"Server start error: {e}", exc_info=True)
            self.cleanup()
            raise

    def cleanup(self):
        """Clean up all resources.

        Every socket attribute is cached into a local before use: another
        thread (the per-client connection's teardown, via reset_state()) can
        null these attributes concurrently. Re-reading ``self.<attr>`` between
        the shutdown() and close() calls risks the attribute having gone to
        None in between -- ``None.close()`` raises AttributeError, which is
        not a socket.error/OSError and therefore escapes the except clauses,
        aborting the rest of cleanup() and leaking the SO_REUSEPORT listener.
        Binding the reference once up front makes both calls operate on the
        same object regardless of what happens to the attribute afterwards.

        Idempotent: stop() calls it, and so does the accept loop's finally
        clause once stop() has closed the socket underneath it. Every step is
        None-guarded, so the repeat run is a no-op; only the announcement is
        suppressed, because two "Cleaning up server resources..." lines per
        shutdown read like two shutdowns.
        """
        with self._stop_lock:
            first_cleanup = not self._cleanup_logged
            self._cleanup_logged = True
        if first_cleanup:
            logger.info("Cleaning up server resources...")
        else:
            logger.debug("Cleaning up server resources (already cleaned)...")

        # Stop all running operations
        self.running = False
        self.transmitting_state = False
        self.connection_running = False

        # Clean up client socket
        sock, self.client_socket = self.client_socket, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

        # Clean up server socket
        sock, self.server_socket = self.server_socket, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

        # Clean up command socket
        sock, self.command_socket = self.command_socket, None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass

        # Clean up UDP socket. No shutdown(): a connectionless socket has
        # nothing to shut down and shutdown() would raise ENOTCONN.
        sock, self.udp_socket = self.udp_socket, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

        # Wait for any remaining operations to complete
        time.sleep(0.1)

        # Reset all state
        self.reset_state()
        self.running = False

    def stop(self):
        """Stop the server and release every resource it owns.

        Idempotent and non-raising. Shutdown is the one path that has to
        survive being called badly: twice (a second Ctrl+C), concurrently
        (accept thread vs. main thread), or with one of its stages already
        broken. A stage that raised used to skip every stage after it, which
        is how a Ctrl+C could leave the listening socket bound or the viewer's
        GL context alive.

        Ordered so nothing is torn down while something else still uses it:
        stop accepting and close the sockets (which is also what breaks the
        accept/receive loops out of their waits), then the gripper server,
        then the simulator -- whose viewer teardown is the slowest step and
        the one the network threads must already be gone for. Every join is
        bounded; every serving thread is a daemon, so a join that does time
        out can never keep the process alive.
        """
        with self._stop_lock:
            if self._stopping:
                logger.debug("stop() already in progress or done")
                return
            self._stopping = True

        logger.info("Stopping server...")
        self.running = False
        self.connection_running = False
        self.transmitting_state = False

        for stage, action in (
            ("socket cleanup", self.cleanup),
            ("gripper server", self._stop_gripper),
            ("simulator", self.physics_sim.stop),
        ):
            try:
                action()
            except Exception:
                logger.error("Error stopping the %s; continuing shutdown", stage, exc_info=True)

    def _stop_gripper(self):
        """Stop the gripper server and wait (briefly) for its accept loop."""
        if self.gripper_server is None:
            return
        self.gripper_server.stop()
        thread, self.gripper_thread = self.gripper_thread, None
        if thread is not None:
            # stop() has already closed the listening socket, which drops the
            # accept() out of its wait immediately -- this join is a formality
            # that should return in microseconds. It is bounded anyway, and the
            # thread is a daemon, so a gripper client wedged in a backend call
            # delays shutdown by at most GRIPPER_JOIN_TIMEOUT_S instead of
            # holding the process open.
            thread.join(timeout=GRIPPER_JOIN_TIMEOUT_S)
            if thread.is_alive():
                logger.warning(
                    "Gripper server thread did not stop within %.1fs; abandoning it",
                    GRIPPER_JOIN_TIMEOUT_S,
                )
