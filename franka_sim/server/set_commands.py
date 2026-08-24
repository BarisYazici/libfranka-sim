"""The parameter-setting TCP commands, and error recovery.

The ``SetX`` family (collision behaviour, joint and Cartesian impedance, guiding
mode, ``EE_T_K``, ``NE_T_EE``, load) plus ``AutomaticErrorRecovery``. Each reads
a fixed-size payload, records the parameter on the session and answers success.

``AutomaticErrorRecovery`` is the exception: it clears the latched error only
once the arm has actually come to a standstill, which is what
:meth:`_wait_for_standstill` waits for (and gives up on, on a budget).
"""

import struct
import time

from franka_sim.franka_protocol import (
    Command,
    MessageHeader,
    SetCartesianImpedanceCommand,
    SetCollisionBehaviorCommand,
    SetEEToKCommand,
    SetGuidingModeCommand,
    SetJointImpedanceCommand,
    SetLoadCommand,
    SetNEToEECommand,
)
from franka_sim.server.constants import (
    AUTOMATIC_ERROR_RECOVERY_POLL_PERIOD,
    AUTOMATIC_ERROR_RECOVERY_SETTLE_CYCLES,
    AUTOMATIC_ERROR_RECOVERY_SETTLE_VELOCITY,
    AUTOMATIC_ERROR_RECOVERY_TIMEOUT,
    RobotMode,
    logger,
)


class SetCommandsMixin:
    """See the module docstring; this mixin carries no state of its own."""

    def handle_set_collision_behavior_command(
        self, client_socket, header: MessageHeader, payload: bytes
    ):
        """Handle SetCollisionBehavior command received over TCP"""
        try:
            # Parse the command
            cmd = SetCollisionBehaviorCommand.from_bytes(payload)
            logger.info("Received SetCollisionBehavior command with values:")
            logger.debug(f"Lower torque thresholds acc: {cmd.lower_torque_thresholds_acceleration}")
            logger.debug(f"Upper torque thresholds acc: {cmd.upper_torque_thresholds_acceleration}")

            # For now, just acknowledge the command without actually implementing behavior
            # Send success response (status = 0)
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(
                Command.kSetCollisionBehavior, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            self._send_tcp(client_socket, header_bytes + response_data)
            logger.info("Sent SetCollisionBehavior success response")

        except Exception as e:
            logger.error(f"Error handling SetCollisionBehavior command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(
                Command.kSetCollisionBehavior, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            self._send_tcp(client_socket, header_bytes + response_data)

    def handle_set_joint_impedance_command(
        self, client_socket, header: MessageHeader, payload: bytes
    ):
        """Handle SetJointImpedance command received over TCP"""
        try:
            # Parse the command
            cmd = SetJointImpedanceCommand.from_bytes(payload)
            logger.info("Received SetJointImpedance command with values:")
            logger.debug(f"Joint stiffness values: {cmd.K_theta}")

            # For now, just acknowledge the command without actually implementing behavior
            # Send success response (status = 0)
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(
                Command.kSetJointImpedance, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            self._send_tcp(client_socket, header_bytes + response_data)
            logger.info("Sent SetJointImpedance success response")

        except Exception as e:
            logger.error(f"Error handling SetJointImpedance command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(
                Command.kSetJointImpedance, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            self._send_tcp(client_socket, header_bytes + response_data)

    def handle_set_cartesian_impedance_command(
        self, client_socket, header: MessageHeader, payload: bytes
    ):
        """Handle SetCartesianImpedance command received over TCP"""
        try:
            # Parse the command
            cmd = SetCartesianImpedanceCommand.from_bytes(payload)
            logger.info("Received SetCartesianImpedance command with values:")
            logger.debug(f"Cartesian stiffness values: {cmd.K_x}")

            # For now, just acknowledge the command without actually implementing behavior
            # Send success response (status = 0)
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(
                Command.kSetCartesianImpedance, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            self._send_tcp(client_socket, header_bytes + response_data)
            logger.info("Sent SetCartesianImpedance success response")

        except Exception as e:
            logger.error(f"Error handling SetCartesianImpedance command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(
                Command.kSetCartesianImpedance, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            self._send_tcp(client_socket, header_bytes + response_data)

    def handle_set_guiding_mode_command(
        self, client_socket, header: MessageHeader, payload: bytes
    ):
        """Handle SetGuidingMode command received over TCP.

        Without a response, a real libfranka client blocks forever (the
        setGuidingMode() call has no way to complete). No RobotState field
        reflects the guiding mode, so this is ACK-only, mirroring the
        SetCollisionBehavior no-op pattern.
        """
        try:
            # Parse the command
            cmd = SetGuidingModeCommand.from_bytes(payload)
            logger.info("Received SetGuidingMode command with values:")
            logger.debug(f"Guiding mode: {cmd.guiding_mode}, nullspace: {cmd.nullspace}")

            # For now, just acknowledge the command without actually implementing behavior
            # Send success response (status = 0)
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(Command.kSetGuidingMode, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            self._send_tcp(client_socket, header_bytes + response_data)
            logger.info("Sent SetGuidingMode success response")

        except Exception as e:
            logger.error(f"Error handling SetGuidingMode command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(Command.kSetGuidingMode, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            self._send_tcp(client_socket, header_bytes + response_data)

    def handle_set_ee_to_k_command(self, client_socket, header: MessageHeader, payload: bytes):
        """Handle SetEEToK command received over TCP.

        The real robot reflects EE_T_K (stiffness frame relative to the
        end-effector) back in RobotState, so store it there too.
        """
        try:
            # Parse the command
            cmd = SetEEToKCommand.from_bytes(payload)
            logger.info("Received SetEEToK command with values:")
            logger.debug(f"EE_T_K: {cmd.EE_T_K}")

            # Reflect EE_T_K in subsequent RobotState broadcasts, like the real robot.
            self.robot_state.state["EE_T_K"] = cmd.EE_T_K

            # Send success response (status = 0)
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(Command.kSetEEToK, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            self._send_tcp(client_socket, header_bytes + response_data)
            logger.info("Sent SetEEToK success response")

        except Exception as e:
            logger.error(f"Error handling SetEEToK command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(Command.kSetEEToK, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            self._send_tcp(client_socket, header_bytes + response_data)

    def handle_set_ne_to_ee_command(self, client_socket, header: MessageHeader, payload: bytes):
        """Handle SetNEToEE command received over TCP.

        The real robot reflects NE_T_EE (end-effector relative to the
        nominal end-effector, i.e. the flange-mount frame) back in
        RobotState, so store it there too.
        """
        try:
            # Parse the command
            cmd = SetNEToEECommand.from_bytes(payload)
            logger.info("Received SetNEToEE command with values:")
            logger.debug(f"NE_T_EE: {cmd.NE_T_EE}")

            # Reflect NE_T_EE in subsequent RobotState broadcasts, like the real robot.
            self.robot_state.state["NE_T_EE"] = cmd.NE_T_EE
            # ...and with it F_T_EE, which is *derived*: libfranka documents
            # ``Robot::setEE`` as setting "the transformation NE_T_EE from
            # nominal end effector to end effector frame [...] The transformation
            # from flange to end effector frame is split into two
            # transformations: F_T_EE = F_T_NE * NE_T_EE"
            # (``include/franka/robot.h``). Publishing it is what lets a client
            # -- and this server's own Cartesian safety check -- know where the
            # EE actually is.
            self._refresh_ee_transform()

            # Send success response (status = 0)
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(Command.kSetNEToEE, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            self._send_tcp(client_socket, header_bytes + response_data)
            logger.info("Sent SetNEToEE success response")

        except Exception as e:
            logger.error(f"Error handling SetNEToEE command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(Command.kSetNEToEE, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            self._send_tcp(client_socket, header_bytes + response_data)

    def handle_set_load_command(self, client_socket, header: MessageHeader, payload: bytes):
        """Handle SetLoad command received over TCP.

        The real robot reflects the externally-mounted load (mass, center of
        mass, inertia) back in RobotState, so store it there too.
        """
        try:
            # Parse the command
            cmd = SetLoadCommand.from_bytes(payload)
            logger.info("Received SetLoad command with values:")
            logger.debug(
                f"m_load: {cmd.m_load}, F_x_Cload: {cmd.F_x_Cload}, I_load: {cmd.I_load}"
            )

            # Reflect the load in subsequent RobotState broadcasts, like the real robot.
            self.robot_state.state["m_load"] = cmd.m_load
            self.robot_state.state["F_x_Cload"] = cmd.F_x_Cload
            self.robot_state.state["I_load"] = cmd.I_load

            # Send success response (status = 0)
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(Command.kSetLoad, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            self._send_tcp(client_socket, header_bytes + response_data)
            logger.info("Sent SetLoad success response")

        except Exception as e:
            logger.error(f"Error handling SetLoad command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(Command.kSetLoad, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            self._send_tcp(client_socket, header_bytes + response_data)

    def handle_automatic_error_recovery_command(
        self, client_socket, header: MessageHeader, payload: bytes
    ):
        """Handle AutomaticErrorRecovery command received over TCP.

        The real robot clears any latched reflex/error and returns to Idle.
        Without a response here, libfranka (and franka_hardware, which calls
        automaticErrorRecovery() on activation) blocks forever waiting on the
        TCP reply, stalling the whole control stack.

        ``current_errors`` (the wire's ``errors`` array) is cleared;
        ``last_motion_errors`` (the wire's ``reflex_reason``) is not. libfranka
        documents the latter as "the errors that aborted the previous motion",
        so it is a record, not a live condition, and recovery must not erase
        it -- ``createControlException`` reads it *after* the abort.

        The response is deferred until the arm is at (near) standstill; see
        :meth:`_wait_for_standstill`. On the real robot recovery inherently
        completes there -- an abort caught mid-motion leaves the arm decelerating
        under the internal controller, and a client that Move's again the instant
        it gets a reply is Move'ing into an arm that has already stopped. This
        sim used to reply immediately, so a client recovering from a fast abort
        could start its next motion while still decelerating from a couple of
        rad/s, and its own start-pose guard would then see measured q drifting
        off the commanded q it had just sent and throw a "Performance threshold
        reached" a few milliseconds later -- on a motion the client did nothing
        wrong on.
        """
        try:
            # AutomaticErrorRecovery has an empty request; nothing to parse.
            # Clear any error/reflex state and return the arm to Idle. On the
            # real robot recovery aborts whatever motion latched the reflex and
            # leaves the internal controller holding, so recapture here too --
            # the next Move releases the hold.
            self.robot_state.state["errors"] = [False] * 41
            self.comm.recover()
            self.motion_limits.recover()
            self.robot_state.state["robot_mode"] = RobotMode.kIdle
            self._engage_idle_hold("automatic error recovery")

            # The idle hold above is what does the decelerating -- this only
            # waits for it to have worked. See _wait_for_standstill for why it
            # polls rather than sleeping once, and why it lives here rather
            # than blocking the state-publish loop.
            self._wait_for_standstill()

            # Response is ResponseBase: a single uint8 status (kSuccess = 0).
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(
                Command.kAutomaticErrorRecovery, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            self._send_tcp(client_socket, header_bytes + response_data)
            logger.info("Sent AutomaticErrorRecovery success response")

        except Exception as e:
            logger.error(f"Error handling AutomaticErrorRecovery command: {e}")
            # Send error response (status = 1)
            total_size = 12 + 4
            response_header = MessageHeader(
                Command.kAutomaticErrorRecovery, header.command_id, total_size
            )
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 1)  # Status 1 = Error
            self._send_tcp(client_socket, header_bytes + response_data)

    def _wait_for_standstill(self) -> None:
        """Block the calling thread until the arm has settled, or the timeout passes.

        See :data:`AUTOMATIC_ERROR_RECOVERY_TIMEOUT` for why that ceiling is
        0.7 s rather than the 3 s first tried: libfranka's own TCP receive
        timeout on the response is a hard 1 s, and going over it does not
        make the wait "slower but correct" -- it breaks the connection.

        The 0.7 s budget is wall-clock, but what it is waiting on -- the arm
        ringing down to standstill -- is sim-time. On a scene running below
        roughly 0.7x real-time factor, 0.7 s of wall clock buys less than
        0.7 s of simulated settling, so this method routinely runs out the
        clock before the arm has actually stopped. That is not a bug to chase
        here: the timeout path below degrades to replying anyway, exactly as
        designed, and it is the RTF, not this wait, that wants fixing.

        Called from :meth:`handle_automatic_error_recovery_command`, which runs
        on the per-connection TCP thread (see ``handle_tcp_messages``) -- a
        thread of its own, separate from the state-publish loop
        (:meth:`start_robot_state_transmission`, which runs on the connection's
        accept thread) and from the UDP command receiver
        (:meth:`_handle_commands`). Blocking it here therefore does not stall
        either of those. It does hold up the *next* TCP command on this same
        connection, which is the point: libfranka's ``automaticErrorRecovery()``
        already blocks the client until this reply arrives, so there is no
        "other traffic" on this connection to protect it from -- but it must
        still poll rather than take one long sleep, both so
        :attr:`FrankaSimServer.running` going false during shutdown is noticed
        promptly (see the loop condition below) and so the timeout path can log
        and bail rather than oversleeping a caller that will never settle.

        Queries ``genesis_sim.get_robot_state()`` directly on every poll rather
        than reading the cached ``dq`` off :attr:`RobotState.state`.

        **That distinction is not cosmetic, even though the publish loop now
        keeps running through StopMove.** libfranka always sends ``StopMove``
        before ``AutomaticErrorRecovery`` (confirmed against the real wire:
        ``handle_stop_move_command`` runs first in every observed recovery
        sequence). ``handle_stop_move_command`` used to clear
        ``self.transmitting_state`` -- the flag
        :meth:`start_robot_state_transmission`'s loop reads to keep
        publishing -- which stopped the loop and froze
        ``robot_state.state["dq"]`` at whatever it last was; that stale read
        was the original version of this bug (see the comment in
        :meth:`handle_stop_move_command` for why the flag is not cleared
        there any more). The loop no longer stops for StopMove, but this
        method still
        reads the physics backend directly rather than the publish loop's
        copy: the loop only refreshes ``robot_state.state`` once per 1 kHz
        cycle, so going straight to ``get_robot_state()`` is still the
        freshest read available and costs nothing extra, and it stays correct
        even on any future path that *does* legitimately stop the loop (a
        real disconnect, server shutdown) while a recovery is still in
        flight. The physics backend itself has no such gap either way: it
        keeps stepping and answering ``get_robot_state()`` regardless of
        whether anything is publishing its output.

        Skipped entirely on a ``mobile_base`` server, matching
        :meth:`_run_safety_velocity_check`: ``dq`` there is the swerve base's
        steer/drive joints, not an FR3's, and the base's idle hold is already a
        zero-twist command rather than a decelerating position hold, so there is
        no equivalent "still ringing down" state to wait out.
        """
        if self.mobile_base:
            return
        deadline = time.monotonic() + AUTOMATIC_ERROR_RECOVERY_TIMEOUT
        settled_cycles = 0
        while self.running and time.monotonic() < deadline:
            try:
                dq = self.genesis_sim.get_robot_state().get("dq")
            except Exception:
                # The backend erroring out here is not a reason to hang the
                # client; fall through to "not settled" and let the timeout
                # path below reply anyway.
                logger.exception(
                    "AutomaticErrorRecovery: could not read the simulator "
                    "while waiting for standstill"
                )
                dq = None
            if dq is not None and max(abs(value) for value in dq[:7]) < (
                AUTOMATIC_ERROR_RECOVERY_SETTLE_VELOCITY
            ):
                settled_cycles += 1
                if settled_cycles >= AUTOMATIC_ERROR_RECOVERY_SETTLE_CYCLES:
                    return
            else:
                settled_cycles = 0
            time.sleep(AUTOMATIC_ERROR_RECOVERY_POLL_PERIOD)
        if self.running:
            logger.warning(
                "AutomaticErrorRecovery: arm did not settle below %.3f rad/s "
                "within %.1fs; replying success anyway",
                AUTOMATIC_ERROR_RECOVERY_SETTLE_VELOCITY,
                AUTOMATIC_ERROR_RECOVERY_TIMEOUT,
            )
