"""The UDP halves of a control session: commands in, ``RobotState`` out.

The FCI publishes one ``RobotState`` per 1 ms cycle and expects one command
datagram back per state. This module owns both directions -- the receive loop
that decodes commands and hands them to the motion session, and the publish loop
that paces states at the cycle and stamps each one with the commanded fields.

The two loops run on separate threads against shared session state, so the
ordering they must preserve (a state must not overtake the answer to the state
before it) is kept by the drain gate in
:mod:`franka_sim.server.motion_session`.
"""

import select
import socket
import struct
import threading
import time

from franka_sim.server.constants import (
    _MIN_STATE_SPACING,
    logger,
)


class StateStreamMixin:
    """See the module docstring; this mixin carries no state of its own."""

    def start_command_receiver(self):
        """Start UDP command receiver on specified port"""
        try:
            # Capture the socket *now*, at thread-start time, rather than
            # letting the thread re-read self.udp_socket on every loop turn.
            # self.udp_socket is per-server-instance, not per-thread: a fast
            # reconnect swaps it out for a new session while this thread's
            # own poll loop may still be unwinding on the old fd. Passing the
            # socket in as an argument gives the thread a fixed identity to
            # compare against, so it can tell "my socket is dead" apart from
            # "a newer session's socket is dead" -- see _handle_commands.
            sock = self.udp_socket
            if sock is None:
                logger.error("start_command_receiver called with no udp_socket set")
                return
            self.command_thread = threading.Thread(target=self._handle_commands, args=(sock,))
            self.command_thread.daemon = True
            self.command_thread.start()

        except Exception as e:
            logger.error(f"Error starting command receiver: {e}", exc_info=True)

    def _handle_commands(self, udp_socket):
        """Handle incoming UDP robot commands.

        ``udp_socket`` is the specific socket this thread was started for
        (see ``start_command_receiver``), captured once and never re-read
        from ``self.udp_socket``. That distinction matters on a fast
        reconnect: ``self.connection_running`` is a single flag on the
        server instance, shared by whichever session is current, not scoped
        to this thread. If a stale thread -- still polling the *old*,
        closed socket -- cleared ``connection_running`` unconditionally on
        hangup, it would kill the flag out from under the *new* session
        that has since replaced it, and that new session's broadcast loop
        would exit before sending a single state datagram. The client would
        then see "libfranka: UDP receive: Timeout" on a session that never
        did anything wrong. So every place below that would act on
        ``connection_running`` first checks ``udp_socket is self.udp_socket``
        -- i.e. that this thread is still the live session's thread.
        """
        logger.info("Command handler thread started")

        try:
            logger.info("Starting UDP command polling")
            # Setup poll object for UDP socket
            poller = select.poll()
            logger.debug(f"Command socket file descriptor: {udp_socket.fileno()}")
            poller.register(udp_socket.fileno(), select.POLLIN)
            logger.debug(f"Poller: {poller}")
            timeout = 1  # 1ms timeout

            # RobotCommand packet size (matches the client's RobotCommand struct).
            expected_size = 8 + (7 * 8 + 7 * 8 + 16 * 8 + 6 * 8 + 2 * 8 + 1 + 1) + (7 * 8 + 1)

            # Bound this thread to the current connection (connection_running)
            # *and* to its own socket's identity (udp_socket is self.udp_socket):
            # once a reconnect swaps self.udp_socket out, this loop must not
            # keep spinning on the fd that belongs to a session that no longer
            # exists -- see the identity-invariant note in the docstring above.
            while self.running and self.connection_running and udp_socket is self.udp_socket:
                events = poller.poll(timeout)
                if not events:
                    continue

                # From here to the end of this turn is one datagram's whole
                # journey: out of the socket, through the checks, into the
                # simulator and into the commanded state fields. The publish
                # loop must not close a cycle anywhere inside it -- see
                # _drain_gate, which reads this counter -- because the socket
                # goes empty at the *start* of that journey while the echo the
                # client filters against is only written at the end.
                self._commands_in_flight += 1
                try:
                    command = None
                    for fd, event in events:
                        if not (event & select.POLLIN):
                            # Socket hung up / errored -> the connection is gone
                            # -- but only clear connection_running if this is
                            # still the live session's socket. A stale thread's
                            # own (now-closed) fd reporting POLLHUP/POLLNVAL says
                            # nothing about whatever session replaced it.
                            if udp_socket is self.udp_socket:
                                self.connection_running = False
                            return

                        try:
                            data, addr = udp_socket.recvfrom(expected_size)
                            if len(data) != expected_size:
                                logger.warning(
                                    f"Got a UDP packet with wrong size! Expected {expected_size} \
                                    bytes, got {len(data)} bytes"
                                )
                                continue
                            # Unpack the command data
                            offset = 0

                            # Unpack message_id
                            message_id = struct.unpack("<Q", data[offset : offset + 8])[0]
                            offset += 8

                            # Unpack MotionGeneratorCommand
                            q_c = struct.unpack("<7d", data[offset : offset + 56])
                            offset += 56

                            dq_c = struct.unpack("<7d", data[offset : offset + 56])
                            offset += 56

                            O_T_EE_c = struct.unpack("<16d", data[offset : offset + 128])
                            offset += 128

                            O_dP_EE_c = struct.unpack("<6d", data[offset : offset + 48])
                            offset += 48

                            elbow_c = struct.unpack("<2d", data[offset : offset + 16])
                            offset += 16

                            valid_elbow = bool(data[offset])
                            offset += 1

                            motion_generation_finished = bool(data[offset])
                            offset += 1

                            # Unpack ControllerCommand
                            tau_J_d = struct.unpack("<7d", data[offset : offset + 56])
                            offset += 56

                            torque_command_finished = bool(data[offset])

                            command = {
                                "message_id": message_id,
                                "q_c": q_c,
                                "dq_c": dq_c,
                                "O_T_EE_c": O_T_EE_c,
                                "O_dP_EE_c": O_dP_EE_c,
                                "elbow_c": elbow_c,
                                "valid_elbow": valid_elbow,
                                "motion_generation_finished": motion_generation_finished,
                                "tau_J_d": tau_J_d,
                                "torque_command_finished": torque_command_finished,
                            }

                        except BlockingIOError:
                            break
                        except Exception as e:
                            logger.error(f"Error receiving message: {e}")
                            break

                    # Process newest command if we have one
                    if command and command["message_id"] > 0:
                        # End of control: a motion generator signals via
                        # motion_generation_finished, a pure-torque controller
                        # (startTorqueControl) via torque_command_finished. Handle
                        # both -- otherwise the client hangs waiting for the stop to
                        # be acknowledged.
                        if (
                            command["motion_generation_finished"]
                            or command["torque_command_finished"]
                        ):
                            self._finish_motion(command)
                            continue

                        # Communication accounting first: this cycle now has an
                        # answer (a stale echo still counts as a loss, see
                        # CommConstraintTracker.command_received). The command is
                        # applied either way -- within one sim cycle a late command
                        # is still the freshest user intent there is, and the abort
                        # after MAX_CONSECUTIVE_LOST_CYCLES is the faithful
                        # consequence of being late.
                        # This motion has been streamed to, so a later finish
                        # datagram is this motion's (see _finish_motion).
                        self._motion_has_commands = True
                        fresh = self.comm.command_received(command["message_id"])

                        # Motion limits second: the packet did arrive (so it counts
                        # for the cycle either way), but a command the robot would
                        # refuse must not reach the simulator, and must not enter
                        # the history the next command is differenced against.
                        #
                        # Rewind, check and record are **one** operation on the
                        # checker, taken under a single hold of its lock. A datagram
                        # that missed its cycle may still be the real answer to a
                        # cycle the publish loop has already extrapolated; the robot
                        # would have dropped it, this sim applies it, so the guess it
                        # replaces has to be thrown away first or the two are
                        # differenced against each other and report a deceleration
                        # nobody commanded. Doing that in three separate calls left
                        # two windows for the publish thread's own extrapolate() to
                        # land in, and either one re-created the false abort the
                        # rewind exists to prevent. See
                        # MotionLimitChecker.absorb_command -- which also explains
                        # why a *fresh* command is not rewound, and why a replay gets
                        # no rewind: it is applied, because it is still the freshest
                        # intent there is, but it is not a later sample of the
                        # client's trajectory and never becomes the baseline the next
                        # one is differenced against.
                        #
                        # Nothing gets past this on its way to physics, whether or
                        # not it was fresh: an unchecked path here was a way to walk
                        # a 40 rad/s step into the simulator behind a stale echo.
                        if not self._absorb_within_motion_limits(command, fresh=fresh):
                            continue

                        self._dispatch_control_command(command)
                finally:
                    self._commands_in_flight -= 1

        except Exception as e:
            logger.error(f"Error in read_step: {e}")

    def start_robot_state_transmission(self, client_address: str, client_udp_port: int):
        """
        Start UDP transmission of robot state updates.
        """
        try:
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            # Deliberately not bound. This socket only ever *sends*, so the
            # kernel binds it to an ephemeral port on the first ``sendto``
            # below; libfranka reads the source port off the datagrams it
            # receives and answers there, which is what a real FCI does too.
            # The consequence to know about: until that first send,
            # ``getsockname()`` reports port 0. The log line below therefore
            # prints 0 on every session, and anything wanting to know that this
            # session is broadcasting should watch :attr:`states_sent` rather
            # than the port -- binding explicitly here would make the port real
            # *before* any state had gone out, which is a strictly weaker
            # signal than the one the counter gives.
            # Start the command receiver on this socket before the first
            # state goes out, so the client's very first reply already has
            # a reader attached.
            self.start_command_receiver()

            # port of the udp_socket (0 until the first sendto; see above)
            udp_port = self.udp_socket.getsockname()[1]
            logger.debug(f"UDP port: {udp_port}")
            self.client_address = client_address
            self.client_udp_port = client_udp_port
            # Initialize timing statistics
            total_cycles = 0
            total_genesis_time = 0
            total_cycle_time = 0
            last_stats_time = time.time()

            logger.info(f"Starting UDP transmission to {client_address}:{client_udp_port}")
            self.transmitting_state = True
            first_state_sent = False

            # Pace the broadcast to a ~1 kHz deadline schedule. State reads are now
            # cheap (the physics thread publishes snapshots), so without a pacer
            # this loop would spin well above 1 kHz and burn a whole CPU core.
            period = 0.001  # 1 kHz target broadcast rate
            next_deadline = time.perf_counter()
            # When the last state actually went out. The pacer keeps every
            # state at least one cycle behind it; see the deadline arithmetic
            # at the bottom of the loop.
            last_send = next_deadline - period

            while self.running and self.connection_running and self.transmitting_state:
                try:
                    cycle_start = time.time()

                    genesis_start = time.time()
                    sim_state = self.physics_sim.get_robot_state()
                    genesis_time = time.time() - genesis_start
                    total_genesis_time += genesis_time

                    # Initialize q_d to current q on first state update if not already set
                    if not first_state_sent:
                        self.robot_state.state["q_d"] = sim_state["q"]

                    self.robot_state.state.update(
                        {
                            field: value
                            for field, value in sim_state.items()
                            if field not in self._server_owned_state_fields
                        }
                    )

                    self._publish_commanded_pose(sim_state)
                    self._publish_elbow(sim_state)

                    # The safety controller: measured velocity against the
                    # position-based envelope, every cycle, in every control
                    # mode. Judges the arm, not the command, so it lives here
                    # rather than on the UDP receive path.
                    self._run_safety_velocity_check(sim_state)

                    # Everything above is this cycle's work; the pacing sleep
                    # below is not, and the stats line reports work.
                    cycle_time = time.time() - cycle_start
                    total_cycle_time += cycle_time
                    total_cycles += 1

                    # Pace to the ~1 kHz deadline *here*, with the cycle's work
                    # already done, so what the schedule governs is the moment
                    # the state leaves -- which is the only moment the client
                    # can see. Sleeping at the bottom of the loop instead put
                    # this cycle's work between the deadline and the send, and
                    # the minimum-spacing rule below then charged that work to
                    # every cycle and dragged the publish rate down with it.
                    remaining = next_deadline - time.perf_counter()
                    if remaining > 0:
                        time.sleep(remaining)

                    # Do not close this cycle while the client's answer to the
                    # previous one is still sitting unread in our own socket:
                    # the state about to go out would carry a ``q_d`` the client
                    # has already moved past, and libfranka filters its next
                    # waypoint toward exactly that field. Immediately before the
                    # accounting and the send, which is the only moment that
                    # sees every datagram this cycle could have brought. See
                    # _drain_gate. Whatever it waits for is this cycle's own
                    # time, not time to be made up: the deadline moves with it.
                    next_deadline += self._drain_gate()

                    # Close the cycle the previous state opened and open this
                    # one. Done immediately before the send, so "in time for
                    # this cycle" means "arrived before the next state went
                    # out" -- cycle space, not wall clock.
                    self._account_for_communication_cycle()

                    # Pack and send current robot state. The counter advances
                    # first: anything latched into the state from another thread
                    # after this point missed this packet, which is what
                    # _flush_pending_move_response keys off.
                    self._states_packed += 1
                    state = self.robot_state.pack_state()
                    if self.udp_socket and not self.udp_socket._closed:
                        self.udp_socket.sendto(state, (client_address, client_udp_port))
                        last_send = time.perf_counter()
                        # After the send, so a reader that sees this move knows
                        # a datagram really went out (and, with it, that the
                        # socket has been implicitly bound to a real port).
                        self.states_sent += 1
                        # Inside the guard: a state that was never sent cannot
                        # be the one a deferred kReflexAborted is waiting to
                        # follow.
                        self._flush_pending_move_response()

                        # NOTE: the first state datagram does not get a Move
                        # response here. handle_move_command already answered
                        # this motion's Move with kMotionStarted when it was
                        # accepted -- Move gets exactly one reply on the wire,
                        # and the terminal one (kSuccess via StopMove, or
                        # kReflexAborted/etc. via _pending_move_response) is
                        # sent elsewhere. A second kSuccess sent from here
                        # used to sit unread in libfranka's response map,
                        # keyed by command id; when the motion later aborted,
                        # Robot::throwOnMotionError found that stale kSuccess
                        # ahead of the real terminal response and raised
                        # ProtocolException("Unexpected reply to a Move
                        # command") instead of the intended ControlException.
                        #
                        # first_state_sent still has to flip exactly once,
                        # though -- it also gates the q_d seeding above -- so
                        # it is set unconditionally right after the first
                        # successful sendto, decoupled from any response.
                        if not first_state_sent:
                            first_state_sent = True

                    # Update state for next iteration
                    self.robot_state.update()

                    # Schedule the next state (the sleep itself is above, just
                    # before the send).
                    next_deadline += period
                    # ...but never closer than a whole cycle to the state that
                    # just went out. A cycle that ran long used to be made up
                    # by firing the next state immediately behind it -- two
                    # states microseconds apart -- and that is not a schedule
                    # the FCI contract allows. The client cannot answer the
                    # first one in the time the second takes to arrive, so the
                    # second necessarily carries a ``q_d`` that predates the
                    # answer to the first; libfranka filters its next waypoint
                    # toward exactly that field, and the kink that produces is
                    # what the limit checker then reports as
                    # ``joint_motion_generator_velocity_discontinuity`` against
                    # a client that did nothing wrong (see _drain_gate for the
                    # other half of the same story). One cycle per state, even
                    # when the cycle before it overran.
                    #
                    # The floor is a fraction of a cycle short of the full
                    # period on purpose. It has to bite on a real overrun --
                    # the millisecond-plus ones that produced the bursts -- and
                    # not on the few tens of microseconds ``time.sleep`` routinely
                    # overshoots by, or every cycle would be charged that
                    # overshoot and the nominal 1 kHz would drift down to ~900 Hz.
                    # The client's observed turnaround is well under 100 us, so
                    # 0.8 ms is still a whole answering window.
                    next_deadline = max(next_deadline, last_send + _MIN_STATE_SPACING * period)

                    # Log statistics every second
                    if time.time() - last_stats_time >= 1.0:
                        avg_genesis_time = (
                            total_genesis_time / total_cycles
                        ) * 1000  # Convert to ms
                        avg_cycle_time = (total_cycle_time / total_cycles) * 1000  # Convert to ms
                        freq = total_cycles / (time.time() - last_stats_time)

                        logger.info(
                            f"State Update Stats - Freq: {freq:.1f}Hz, "
                            f"Genesis Time: {avg_genesis_time:.2f}ms, "
                            f"Total Cycle: {avg_cycle_time:.2f}ms"
                        )

                        # Reset statistics
                        total_cycles = 0
                        total_genesis_time = 0
                        total_cycle_time = 0
                        last_stats_time = time.time()

                except Exception as e:
                    logger.error(f"Error in UDP transmission: {e}")
                    if not self.running or not self.connection_running:
                        break

        except Exception as e:
            logger.error(f"Error in robot state transmission: {e}")
        finally:
            self.transmitting_state = False
            # Nothing will publish another state, so anything an abort deferred
            # has to go out now or the client blocks on a response forever.
            try:
                self._flush_pending_move_response(force=True)
            except Exception:  # pragma: no cover - the socket is already gone
                logger.debug("Could not flush the deferred Move response", exc_info=True)
            if self.udp_socket:
                try:
                    self.udp_socket.close()
                except Exception as e:
                    logger.error(f"Error closing UDP socket: {e}")
                self.udp_socket = None
