"""The ``Move`` lifecycle and the motion-limit orchestration around it.

Everything between ``Move`` and the motion ending, however it ends: starting and
preempting a motion, refusing one that would start at a singular pose or while
an error is latched, dispatching each accepted command to the simulator,
echoing the commanded pose/elbow back into the published state, accounting for
the communication cycle, extrapolating a cycle whose command never arrived, and
latching a reflex when a limit or a communication constraint is broken.

This is the half built around ``self.motion_limits``
(:class:`franka_sim.motion_limits.MotionLimitChecker`): validation and logging
are always on, while the abort -- latching the error, answering the ``Move``
with ``kReflexAborted`` and refusing the offending command -- is opt-in.

A command that is refused must not enter the differencing history, so accepting
a command and recording it are one decision, taken in
:meth:`_absorb_within_motion_limits`.
"""

import select
import struct
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from franka_sim.comm_constraints import COMMUNICATION_CONSTRAINTS_VIOLATION_INDEX
from franka_sim.control_modes import ControlMode
from franka_sim.franka_protocol import (
    Command,
    ControllerMode,
    LibfrankaControllerMode,
    LibfrankaMotionGeneratorMode,
    MessageHeader,
    MotionGeneratorMode,
    MoveCommand,
    MoveStatus,
    convert_to_libfranka_controller_mode,
    convert_to_libfranka_motion_mode,
)
from franka_sim.motion_limits import (
    DELTA_T,
    SINGULAR_POSE_MIN_SINGULAR_VALUE,
    smallest_singular_value,
    transform_matrix,
)
from franka_sim.server.constants import (
    RobotMode,
    logger,
)


class MotionSessionMixin:
    """See the module docstring; this mixin carries no state of its own."""

    def _finish_motion(self, command) -> None:
        """Close the motion a ``*_finished`` datagram ends, if it is still this one.

        A client signals the end of control in the datagram itself -- a motion
        generator via ``motion_generation_finished``, a pure-torque controller
        (``startTorqueControl``) via ``torque_command_finished`` -- and the
        server owes it a final idle state plus the ``kSuccess`` answer to its
        pending ``Move``, or the client hangs.

        The datagram can arrive *late*, though: the UDP thread drains one socket
        while the TCP thread accepts the next ``Move`` on another, so a finish
        belonging to the motion that just ended can be handled after its
        successor has already started. Acting on it then ends a motion that has
        barely begun -- accounting off, success rate pinned at 0.0, enforcement
        dead, and the new ``Move`` answered ``kSuccess`` before it ran.

        The datagram's echoed ``message_id`` is what identifies which motion it
        belongs to, and two facts have to hold together:

        * **it echoes a state no newer than the one that was current when this
          motion's ``Move`` was accepted.** The client sent the datagram before
          it sent that ``Move``, so it cannot be answering a state published
          after the ``Move`` was handled -- and a conforming client's first
          command of a new motion always answers a *later* state, because
          ``startMotion`` spins reading states until the modes match before it
          sends anything. Strictly older, not "no newer": a client whose whole
          motion fits inside one publish cycle echoes the epoch id itself, and
          treating that as stale would leave it waiting for a ``kSuccess`` that
          never comes. Equality costs a rare missed stale datagram; the other
          way round costs a hang. This is also why the epoch is seeded from
          ``self._last_published_message_id`` rather than
          ``self.robot_state.state["message_id"]``: the latter is bumped by
          the publish loop's own ``update()`` *before* the datagram for that
          new id has gone out (see ``start_robot_state_transmission`` in
          ``state_stream.py``), so reading it here could seed an epoch one
          id ahead of anything the client could possibly hold yet -- turning
          the client's own first, legitimate answer into exactly the "other
          way round" this bullet is about.
        * **no control command of the running motion has been seen yet.** A
          client streams before it finishes, so a finish arriving before any
          command is not this motion's. A belt to the first condition's braces:
          on its own it would misread a client that finishes on its very first
          command.

        Note what is deliberately *not* required: that this motion preempted an
        unfinished one. It used to be, and that excluded the common case -- a
        motion that finished perfectly normally, whose 1 kHz finish burst is
        still draining out of the socket when the next ``Move`` is accepted. The
        leftover datagram then ended a motion that had barely begun.
        """
        with self._motion_lock:
            if (
                self._motion_generation
                and not self._motion_has_commands
                and command["message_id"] < self._motion_epoch_id
            ):
                logger.info(
                    "Ignoring a motion-finished datagram from a motion that is "
                    "already over (echoed id %s, motion started at %s)",
                    command["message_id"],
                    self._motion_epoch_id,
                )
                return

            motion_id = self._motion_generation
            self._engage_idle_hold("motion finished", motion_id=motion_id)

            # Update state to idle modes -- unless a reflex is latched, in which
            # case the modes already left kMove and ``robot_mode`` is kReflex.
            # Overwriting that with kIdle would tell the client the motion ended
            # cleanly while ``errors`` still says why it did not, and kReflex is
            # the flag that has to survive until AutomaticErrorRecovery.
            if self.robot_state.state["robot_mode"] != RobotMode.kReflex:
                self.robot_state.state["motion_generator_mode"] = 0  # kIdle
                self.robot_state.state["controller_mode"] = 3  # kOther
                self.robot_state.state["robot_mode"] = RobotMode.kIdle

            # Send state with new message ID
            self.robot_state.update()  # This increments message_id
            final_state = self.robot_state.pack_state()
            if self.udp_socket is not None:
                self.udp_socket.sendto(final_state, (self.client_address, self.client_udp_port))

            # Send TCP success response for the Move command
            if self.current_motion_id and self.client_socket is not None:
                total_size = 12 + 4  # Header (12) + status (1) + padding (3)
                response_header = MessageHeader(Command.kMove, self.current_motion_id, total_size)
                header_bytes = response_header.to_bytes()
                response_data = struct.pack("<B3x", MoveStatus.kSuccess.value)
                self._send_tcp(self.client_socket, header_bytes + response_data)
                logger.info(
                    "Sent Move success response for motion ID: %s", self.current_motion_id
                )
                self.current_motion_id = 0  # Reset motion ID after sending response

    def _dispatch_control_command(self, command, *, motion_generation=None) -> None:
        """Route one UDP RobotCommand to the simulator, unless the hold is on.

        Split out of :meth:`_handle_commands` so the whole simulator-facing part
        of a control cycle sits inside ``_hold_lock``: a datagram that was
        already in flight when the session ended must not be applied *after*
        :meth:`_engage_idle_hold` recaptured the arm, or the hold would be
        immediately overwritten by a dead session's last torque.

        ``motion_generation`` is the motion this command was built for, for a
        caller that built it a moment ago rather than receiving it: the
        extrapolation path (:meth:`_extrapolate_missed_cycle`). The branches
        below dispatch on ``motion_generator_mode``/``controller_mode``, which a
        ``Move`` accepted in the meantime has already changed, so a substitute
        built under the old motion's generator must not be applied under the new
        one's. Checked here rather than at the call site because here is inside
        the same ``_hold_lock`` the dispatch itself runs in, and the ``Move``
        path takes that lock too. A received datagram passes None: it was
        checked and recorded against whatever motion was running when it
        arrived, and the pre-existing idle-hold gate is what covers it.
        """
        with self._hold_lock:
            if self._idle_hold:
                return
            if motion_generation is not None and motion_generation != self._motion_generation:
                # The motion this substitute belongs to is over; a new one owns
                # the generator fields now.
                return

            # The commanded *Cartesian* fields, before the generator branches
            # below and outside them: a ``kCartesianPosition`` motion can run
            # under either controller (``kJointImpedance`` or
            # ``kExternalController``), so only the motion-generator mode says
            # whether the client is streaming a pose at all -- and none of the
            # branches below is reached for the kJointImpedance case.
            self._echo_commanded_cartesian(command)

            # Update Genesis simulator based on control mode
            if (
                self.robot_state.state["controller_mode"]
                == LibfrankaControllerMode.kJointImpedance
                and self.robot_state.state["motion_generator_mode"]
                == LibfrankaMotionGeneratorMode.kJointPosition
            ):
                # Target first, mode second -- the same rule (and the same
                # reason) as _switch_to_hold_position. The physics thread reads
                # target and mode without a lock, so switching into POSITION
                # before publishing this command's target lets a step land in
                # between and servo towards the *previous* target at kp=4500.
                # It also matters for the velocity feedforward: entering
                # POSITION mode re-seeds its baseline from whatever target is
                # current at that instant (see PositionFeedforward.reset), so
                # doing it before the write seeds it from a dead session's last
                # q_c and turns this command into one huge dq_c on the next
                # step. Publishing first makes the baseline this very command,
                # which is exactly the no-spike invariant the reset is for.
                self.physics_sim.update_joint_positions(command["q_c"])
                if self.control_mode is not ControlMode.POSITION:
                    logger.info("Setting control mode to POSITION")
                    self.physics_sim.set_control_mode(ControlMode.POSITION)
                    self.control_mode = ControlMode.POSITION
                    # Initialize q_d to current q when first entering position mode
                    self.robot_state.state["q_d"] = self.robot_state.state["q"]
                # Update q_d with commanded positions
                self.robot_state.state["q_d"] = list(command["q_c"])
                self._publish_commanded_derivatives("dq_d", "ddq_d")
                self.physics_sim.update_torques([0.0] * 7)
            elif (
                self.robot_state.state["controller_mode"]
                == LibfrankaControllerMode.kJointImpedance
                and self.robot_state.state["motion_generator_mode"]
                == LibfrankaMotionGeneratorMode.kJointVelocity
            ):
                if self.control_mode is not ControlMode.VELOCITY:
                    logger.info("Setting control mode to VELOCITY")
                    self.physics_sim.set_control_mode(ControlMode.VELOCITY)
                    self.control_mode = ControlMode.VELOCITY
                # Update dq_d with commanded velocities
                self.robot_state.state["dq_d"] = list(command["dq_c"])
                self._publish_commanded_derivatives("ddq_d")
                self._integrate_commanded_position(command["dq_c"])
                self.physics_sim.update_joint_velocities(command["dq_c"])
                self.physics_sim.update_torques([0.0] * 7)
            elif (
                self.mobile_base
                and self.robot_state.state["motion_generator_mode"]
                == LibfrankaMotionGeneratorMode.kCartesianVelocity
                and self.robot_state.state["controller_mode"]
                != LibfrankaControllerMode.kExternalController
            ):
                self._handle_cartesian_velocity(command)
            elif (
                self.cartesian_tracking
                # Any internal controller, not ``kJointImpedance`` alone: the
                # controller mode picks the *stiffness law* the robot holds the
                # generator's output with, and both internal ones are driven by
                # the same generator. Naming only one here while the ``Move``
                # handler put the backend into Cartesian tracking for either
                # left ``kCartesianImpedance`` silently inert -- in a tracking
                # mode with no command ever reaching it. ``kExternalController``
                # is the one that genuinely does not belong: there the client's
                # torques drive the arm and its pose stream is a reference, so
                # it falls through to the TORQUE branch below.
                and self.robot_state.state["controller_mode"]
                != LibfrankaControllerMode.kExternalController
                and self.robot_state.state["motion_generator_mode"]
                in (
                    LibfrankaMotionGeneratorMode.kCartesianPosition,
                    LibfrankaMotionGeneratorMode.kCartesianVelocity,
                )
            ):
                self._drive_cartesian_generator(command)
            elif (
                self.robot_state.state["controller_mode"]
                == LibfrankaControllerMode.kExternalController
            ):
                if self.control_mode is not ControlMode.TORQUE:
                    logger.info("Setting control mode to TORQUE")
                    self.physics_sim.set_control_mode(ControlMode.TORQUE)
                    self.control_mode = ControlMode.TORQUE
                # Update tau_J_d with commanded torques
                self.robot_state.state["tau_J_d"] = list(command["tau_J_d"])
                self.physics_sim.update_torques(command["tau_J_d"])
                self._echo_external_controller_generator(command)

    def _integrate_commanded_position(self, dq_c) -> None:
        """Advance ``q_d`` by one control cycle of the commanded velocity.

        A joint-*velocity* generator commands no position, but the robot still
        reports one: ``q_d`` is the reference the internal generator integrates,
        and libfranka documents it as always present in the robot state so a
        client can predict what the robot will do "even in case of packet
        losses" (``docs/overview.rst``). The sim used to leave it frozen for the
        whole motion, which is wrong on the wire and breaks two real clients:
        a torque controller servoing against ``q_d`` (see
        :meth:`_echo_external_controller_generator`) sees a zero error for ever,
        and libfranka's own ``MotionGenerator`` -- which seeds ``q_start`` from
        ``q_d`` -- plans a zero-length trajectory when the next motion asks it
        to move away from where a velocity motion actually left the arm.

        One cycle per applied command, which is exact rather than approximate:
        every cycle the client misses gets its own extrapolated substitute
        dispatched through :meth:`_dispatch_control_command` (see
        :meth:`_extrapolate_missed_command`), so one dispatch really is one
        millisecond of reference.

        The reference is integrated, not slaved to the measured ``q``: it is
        what the client *commanded*, and a torque controller's whole job is to
        close the gap between the two. Copying the measurement here would make
        that gap identically zero and the controller a no-op -- which is exactly
        the failure this method exists to fix.
        """
        state = self.robot_state.state
        q_d = state["q_d"]
        state["q_d"] = [float(q_d[i]) + float(dq_c[i]) * DELTA_T for i in range(7)]

    def _echo_external_controller_generator(self, command) -> None:
        """Report the motion generator's reference during an external-controller session.

        ``kExternalController`` replaces the **controller**, not the generator.
        The robot still runs the joint motion generator the ``Move`` asked for
        and still reports its output -- that is what makes libfranka's
        two-callback ``robot.control(torque_callback, motion_callback)`` pattern
        work at all, since the torque callback has nothing else to servo
        against. The sim
        published only ``tau_J_d`` here and left the generator fields frozen, so
        every such client read ``q_d == q`` and ``dq_d == 0``, computed a zero
        torque, and watched an arm that never moved. Measured on the reference
        provocation: eight seconds of ``dq_c`` ramped to -0.1 rad/s with
        ``q_d[3]`` pinned at -1.5700, ``dq_d[3]`` at 0.0 and the client's
        ``200 * (q_d - q)`` at exactly 0.0000 Nm.
        (The Cartesian generators' own echo is
        :meth:`_echo_commanded_cartesian`, which already runs for every command
        whatever the controller.)

        **Which fields, per generator.** A ``kJointPosition`` session writes
        ``q_d`` alone -- the position the client commanded, which is the field
        its torque law servos against and the one that was frozen. A
        ``kJointVelocity`` session writes all three: ``dq_d`` is the command,
        ``ddq_d`` its backward difference, and ``q_d`` the integral the
        generator would have produced
        (:meth:`_integrate_commanded_position`).

        The physics is untouched, and deliberately: in this mode the client's
        torques drive the arm and the generator's output is a reference only,
        which is why the branch above puts the backend in ``TORQUE`` mode. This
        method is reporting, nothing else.

        ``ddq_d`` is differenced here rather than taken from the limit checker.
        The checker is armed in ``TORQUE`` mode for this session -- it is
        judging ``tau_J_d``, the only signal this server checks in an
        external-controller session -- so
        :meth:`_publish_commanded_derivatives` would hand back the torque rate
        and write it into ``ddq_d``. One backward difference over the same
        single cycle :meth:`_integrate_commanded_position` uses is the honest
        substitute.
        """
        generator = self.robot_state.state["motion_generator_mode"]
        if generator == LibfrankaMotionGeneratorMode.kJointPosition:
            self.robot_state.state["q_d"] = list(command["q_c"])
        elif generator == LibfrankaMotionGeneratorMode.kJointVelocity:
            previous = [float(value) for value in self.robot_state.state["dq_d"]]
            commanded = [float(value) for value in command["dq_c"]]
            self.robot_state.state["dq_d"] = commanded
            self.robot_state.state["ddq_d"] = [
                (commanded[i] - previous[i]) / DELTA_T for i in range(7)
            ]
            self._integrate_commanded_position(commanded)

    def _drive_cartesian_generator(self, command) -> None:
        """Hand one accepted Cartesian command to the backend's differential IK.

        The arm's counterpart to the joint branches above, with the two writes
        in the opposite order -- mode first, target second; see the comment on
        the switch below for why the Cartesian interfaces need the mirror image
        of :meth:`_dispatch_control_command`'s POSITION rule.

        Only reached for a command this cycle's checks already accepted, and
        only when the backend can convert one at all
        (:attr:`cartesian_tracking`). Nothing about the checking layer changes:
        this is a second consumer of the accepted stream, not a filter on it.

        ``elbow_c[0]`` is passed through when -- and only when -- the client
        said it commands an elbow (``valid_elbow``); ``elbow_c`` is zero-filled
        otherwise, and steering the redundancy angle to a zero nobody asked for
        would twist the arm on every elbow-less Cartesian motion.
        """
        pose_generator = (
            self.robot_state.state["motion_generator_mode"]
            == LibfrankaMotionGeneratorMode.kCartesianPosition
        )
        mode = ControlMode.CARTESIAN_POSE if pose_generator else ControlMode.CARTESIAN_VELOCITY
        # Mode first, target second -- the *opposite* order to the POSITION
        # branch above, and for the same underlying reason. Entering a Cartesian
        # mode deliberately clears the target (see
        # ``MujocoFrankaSim.set_control_mode``), so publishing this command
        # before the switch would throw it away and leave the arm holding for a
        # cycle. Clearing first and filling immediately after cannot spike: the
        # worst a physics step landing in between sees is "no target", which
        # servos zero velocity.
        if self.control_mode is not mode:
            logger.info("Setting control mode to %s", mode.name)
            self.physics_sim.set_control_mode(mode)
            self.control_mode = mode
        elbow_angle = command["elbow_c"][0] if command.get("valid_elbow") else None
        if pose_generator:
            self.physics_sim.update_cartesian_pose(command["O_T_EE_c"], elbow_angle)
        else:
            self.physics_sim.update_cartesian_velocity(command["O_dP_EE_c"], elbow_angle)
        self.physics_sim.update_torques([0.0] * 7)

    def _echo_commanded_cartesian(self, command) -> None:
        """Echo a Cartesian generator's commanded pose and elbow into the state.

        The Cartesian half of what :meth:`_dispatch_control_command` already
        does for ``q_d``/``dq_d``: ``O_T_EE_d`` and ``O_T_EE_c`` are the FCI
        layer's fields, not the physics backend's, and on hardware they carry
        the pose the motion generator is tracking. libfranka reads *both* --
        ``O_T_EE_d`` is what a pose motion generator initialises and holds from,
        and ``O_T_EE_c`` is the reference its command low-pass filter blends the
        next command with (``src/control_loop.cpp``, ``ControlLoop<CartesianPose>
        ::convertMotion``) -- so during a ``kCartesianPosition`` motion both
        have to be the commanded stream or the client's own filter drags every
        command towards whatever the sim published instead.

        ``O_T_EE_c`` is the last pose the client commanded; ``O_T_EE_d`` is the
        one the generator is tracking, which in this sim is the same value
        because a commanded pose is applied instantly.
        A *lost* cycle reaches this method too: the publish loop extrapolates the
        pose across it and dispatches the result down this same path
        (:meth:`_extrapolate_missed_cycle`), so both fields keep advancing along
        the commanded trajectory exactly as ``q_d`` does, which is what the robot
        reports -- "the last received c values (after the low pass filter and the
        extrapolation due to packet losses)" (``docs/overview.rst``).

        **The twist generator has the identical requirement, and leaving it out
        silently scaled the whole interface.** ``ControlLoop<CartesianVelocities>
        ::convertMotion`` low-passes every commanded twist toward
        ``robot_state.O_dP_EE_c`` (``src/control_loop.cpp:296-313``), the same
        way the pose loop uses ``O_T_EE_c``. With those fields left at the wire
        struct's permanent zero on an arm role, the filter's reference was zero
        on *every* cycle rather than the client's own previous command, so
        instead of converging on the commanded twist it returned a fixed
        fraction of it: ``gain = dt / (dt + 1 / (2*pi*f_c))`` = 0.3859 at
        libfranka's default 100 Hz cutoff and 1 ms. Every ``kCartesianVelocity``
        client was moving the arm at 39% of the speed it asked for, with no way
        to see why. Echoed here, the reference is the previous command and a
        constant commanded twist converges geometrically to 1.0x.

        The elbow rides along with either Cartesian generator
        (``CartesianPose::hasElbow`` / ``CartesianVelocities::hasElbow``) and is
        echoed only when the client actually sent one -- ``valid_elbow`` is its
        own statement that it did, and ``elbow_c`` is zero-filled otherwise.

        **Arm roles only.** The mobile base commands a *base* twist rather than
        an end-effector one; its ``O_dP_EE_d``/``O_dP_EE_c`` echo and its
        dead-reckoned ``O_T_EE`` are :meth:`_handle_cartesian_velocity`'s
        business and are deliberately untouched here, the same role guard the
        rest of the commanded-field ownership uses
        (:data:`COMMANDED_STATE_FIELDS`).
        """
        if self.mobile_base:
            return
        mode = self.robot_state.state["motion_generator_mode"]
        if mode == LibfrankaMotionGeneratorMode.kCartesianPosition:
            pose = list(command["O_T_EE_c"])
            self.robot_state.state["O_T_EE_c"] = pose
            self.robot_state.state["O_T_EE_d"] = pose
        elif mode == LibfrankaMotionGeneratorMode.kCartesianVelocity:
            twist = list(command["O_dP_EE_c"])
            self.robot_state.state["O_dP_EE_c"] = twist
            self.robot_state.state["O_dP_EE_d"] = twist
        else:
            return
        if command.get("valid_elbow"):
            elbow = list(command["elbow_c"])
            self.robot_state.state["elbow_c"] = elbow
            self.robot_state.state["elbow_d"] = elbow

    def _cartesian_motion_owns_commanded_fields(self) -> bool:
        """Whether a running Cartesian generator owns the commanded-echo fields.

        ``O_T_EE_d``/``O_T_EE_c`` and ``elbow_d``/``elbow_c`` are *commanded*
        fields, and while one of the two Cartesian generators is running the
        only honest source for them is the client's own stream
        (:meth:`_echo_commanded_cartesian`). Where the stream is silent -- a
        pose motion that commands no elbow, the cycle or two before its first
        command lands, a twist motion which commands no pose at all -- they stay
        frozen at the value :meth:`_freeze_commanded_cartesian` stamped when the
        motion started, which *is* the measured value at that instant.

        **Why freezing rather than tracking the measured arm.** libfranka builds
        every Cartesian command out of these fields: they are the reference its
        command low-pass filter blends the next command with, and its rate
        limiter differences against (``ControlLoop<CartesianPose>::
        convertMotion``, ``ControlLoop<CartesianVelocities>::convertMotion``,
        ``src/control_loop.cpp``), and the conventional way to open a Cartesian
        motion is
        ``franka::CartesianPose cmd{state.O_T_EE_d, state.elbow_d}``. Once
        the arm is actually driven from those commands, publishing the *measured*
        pose or elbow there closes a positive feedback loop through the client:
        the command chases the lagging arm, the arm chases the command, and the
        two wind each other up until the (correct) checking layer aborts the
        motion. Freezing breaks the loop at its only closure point without
        inventing a value the robot never commanded.

        Cartesian generators only. Under a joint generator the client references
        ``q_d``/``dq_d``/``ddq_d`` and never these fields, so there is no loop to
        break, and reporting the pose the arm is in stays the closest thing this
        sim has to the hardware's own ``O_T_EE_d = FK(q_d)``. Idle likewise --
        and that is the behaviour any "start from ``state.O_T_EE_d``"
        generator depends on, since it reads the field on the motion's first
        cycle.
        """
        return self.robot_state.state["motion_generator_mode"] in (
            LibfrankaMotionGeneratorMode.kCartesianPosition,
            LibfrankaMotionGeneratorMode.kCartesianVelocity,
        )

    def _freeze_commanded_cartesian(self, seed: Dict[str, Any]) -> None:
        """Stamp the commanded-echo fields with the pose/elbow the motion starts in.

        Called once per accepted ``Move`` on an arm role, with the same snapshot
        the limit checker is seeded from. It is what makes "frozen" mean *the
        measured state at motion start* rather than "whatever the publish loop
        happened to leave behind": a ``Move`` can arrive before the publish loop
        has produced its first frame, and the identity the wire struct is
        constructed with is not a pose any generator could legally start from.

        Values only ever *read back* by the client, so a motion whose generator
        is not Cartesian is stamped too -- harmlessly, since the publish loop
        goes straight back to tracking the measured arm for it on the next
        cycle (:meth:`_cartesian_motion_owns_commanded_fields`).
        """
        if self.mobile_base:
            return
        pose = seed.get("O_T_EE")
        if pose is not None:
            pose = pose.tolist() if hasattr(pose, "tolist") else list(pose)
            self.robot_state.state["O_T_EE_d"] = pose
            self.robot_state.state["O_T_EE_c"] = list(pose)
        elbow = self._elbow_from_joints(seed.get("q"))
        if elbow is not None:
            self.robot_state.state["elbow_d"] = list(elbow)
            self.robot_state.state["elbow_c"] = list(elbow)

    def _publish_commanded_derivatives(self, *fields: str) -> None:
        """Report the derivatives the applied command implies, in order.

        ``dq_d`` and ``ddq_d`` for a position generator, ``ddq_d`` alone for a
        velocity one -- the fields libfranka documents as carrying
        ``dq_{c,k-1}`` and ``ddq_{c,k-1}``, which is what lets a client compute
        the derivatives the robot will compute "in advance, even in case of
        packet losses" (``docs/overview.rst``).

        The numbers come from the limit checker, which has already differenced
        the command that is being applied. Silent no-op when no motion is armed
        there, so nothing here depends on the checks being switched on.
        """
        derivatives = self.motion_limits.applied_derivatives()
        if derivatives is None:
            return
        for field, values in zip(fields, derivatives):
            self.robot_state.state[field] = values

    def handle_move_command(self, client_socket, header: MessageHeader, payload: bytes) -> None:
        """Handle Move command received over TCP"""
        try:
            # Parse the move command
            try:
                move_cmd = MoveCommand.from_bytes(payload)
            except ValueError as e:
                logger.error(f"Error handling Move command: {e}")
                self.send_move_response(
                    client_socket,
                    command_id=header.command_id,
                    status=MoveStatus.kInvalidArgumentRejected,
                )
                return

            logger.info(
                f"Received Move command: controller_mode={move_cmd.controller_mode.name}, "
                f"motion_generator_mode={move_cmd.motion_generator_mode.name}"
            )

            # Validate controller mode
            try:
                ControllerMode(move_cmd.controller_mode)
            except ValueError:
                logger.error(f"Error handling Move command:\
                          {move_cmd.controller_mode} is not a valid ControllerMode")
                self.send_move_response(
                    client_socket,
                    command_id=header.command_id,
                    status=MoveStatus.kInvalidArgumentRejected,
                )
                return

            with self._motion_lock:
                if self._refuse_move_while_latched(client_socket, header):
                    return

                if self._refuse_move_at_singular_pose(client_socket, header, move_cmd):
                    return

                self._preempt_running_motion()

                # A fresh session token for this motion; see _motion_generation.
                self._motion_generation += 1
                generation = self._motion_generation
                # NOT self.robot_state.state["message_id"]: that counter is
                # bumped by the publish loop's own update() *before* the
                # datagram carrying the next id has actually gone out (send,
                # then drain-gate/accounting, then update() -- see
                # start_robot_state_transmission), so reading it here can
                # observe an id one ahead of anything the client could
                # possibly have received yet. self._last_published_message_id
                # is written *after* the sendto() that id belongs to, so it
                # never promises a datagram that is not already on the wire.
                # See its docstring for the failure this produced.
                self._motion_epoch_id = self._last_published_message_id
                self._motion_has_commands = False
                # No run of losses is open on a motion that has not started;
                # the checker drops its own snapshot in start_motion below.
                self._extrapolated_reference = None

                # The snapshot this motion is judged and reported from, read
                # once and used twice: to stamp the commanded Cartesian fields
                # here, and to seed the limit checker further down (it depends
                # on ``generator_mode``, which the branches below decide).
                #
                # **Fields first, mode second**, the same rule the dispatch path
                # applies to target-then-mode and for the same reason. Publishing
                # the Cartesian generator mode is what makes the publish loop
                # stop writing ``O_T_EE_d/_c`` and ``elbow_d/_c``
                # (:meth:`_cartesian_motion_owns_commanded_fields`), so stamping
                # them afterwards leaves a window for a cycle that reports
                # neither: not the measured arm, because the mode already says
                # "frozen", and not the frozen value, because it has not been
                # written yet. A ``Move`` that arrives before the publish loop's
                # first cycle -- which is exactly what a client that Moves
                # straight after connecting does -- then broadcast the wire
                # struct's zero ``elbow_d``, and libfranka's ``checkElbow``
                # refuses to send a branch flag that is not +-1, so the elbow
                # interface was unreachable for that motion.
                seed = self._motion_limit_seed_state()
                self._freeze_commanded_cartesian(seed)

                # Update robot state
                self.robot_state.set_motion_generator_mode(
                    convert_to_libfranka_motion_mode(move_cmd.motion_generator_mode)
                )
                self.robot_state.set_controller_mode(
                    convert_to_libfranka_controller_mode(move_cmd.controller_mode)
                )
                self.robot_state.state["robot_mode"] = RobotMode.kMove
                self.current_motion_id = header.command_id
                # A new Move is a new motion: rearm the mobile hold-log latch so
                # the next time it finishes logs again, and release the idle hold so
                # this motion's UDP commands are applied again. Released before the
                # set_control_mode() calls below, which are what actually override
                # the hold in the simulator.
                self._mobile_hold_logged = False
                with self._hold_lock:
                    self._idle_hold = False
                # A new motion is a new success-rate window.
                self.comm.start_motion(generation)

                # The generator this motion actually runs, as opposed to
                # whatever mode the *previous* motion left in
                # ``self.control_mode``. NONE means "a mode this server accepts
                # but does not drive", which is what gates the limit checker
                # below; see the comment there.
                generator_mode = ControlMode.NONE

                # Set appropriate control mode in Genesis simulator
                if (
                    move_cmd.controller_mode == ControllerMode.kJointImpedance
                    and move_cmd.motion_generator_mode == MotionGeneratorMode.kJointPosition
                ):
                    logger.info("Setting control mode to POSITION")
                    self.physics_sim.set_control_mode(ControlMode.POSITION)
                    self.control_mode = generator_mode = ControlMode.POSITION
                elif (
                    move_cmd.controller_mode == ControllerMode.kJointImpedance
                    and move_cmd.motion_generator_mode == MotionGeneratorMode.kJointVelocity
                ):
                    logger.info("Setting control mode to VELOCITY")
                    self.physics_sim.set_control_mode(ControlMode.VELOCITY)
                    self.control_mode = generator_mode = ControlMode.VELOCITY
                elif (
                    self.mobile_base
                    and move_cmd.motion_generator_mode == MotionGeneratorMode.kCartesianVelocity
                    and move_cmd.controller_mode != ControllerMode.kExternalController
                ):
                    logger.info("Setting control mode to STEERING_DRIVE")
                    self.physics_sim.set_control_mode(ControlMode.STEERING_DRIVE)
                    self.control_mode = generator_mode = ControlMode.STEERING_DRIVE
                elif move_cmd.controller_mode == ControllerMode.kExternalController:
                    logger.info("Setting control mode to TORQUE")
                    self.physics_sim.set_control_mode(ControlMode.TORQUE)
                    self.control_mode = generator_mode = ControlMode.TORQUE
                elif move_cmd.motion_generator_mode in (
                    MotionGeneratorMode.kCartesianPosition,
                    MotionGeneratorMode.kCartesianVelocity,
                ):
                    # An *arm* role on one of the two Cartesian generators (the
                    # mobile base's own kCartesianVelocity was matched further
                    # up, and the kExternalController case one branch above:
                    # there the client's torques drive the arm and the pose
                    # stream is a reference, not a command to the joints).
                    #
                    # Driven when the backend can convert a Cartesian command
                    # into joint motion, checked-only when it cannot; see
                    # :attr:`cartesian_tracking`. Either way the commanded
                    # ``O_T_EE_c``/``O_dP_EE_c``/``elbow_c`` stream is
                    # differentiated and judged exactly as before, so a step in
                    # it aborts with the hardware error rather than being
                    # silently dropped. See
                    # :meth:`franka_sim.motion_limits.MotionLimitChecker._check_cartesian_pose`.
                    generator_mode = (
                        ControlMode.CARTESIAN_POSE
                        if move_cmd.motion_generator_mode
                        == MotionGeneratorMode.kCartesianPosition
                        else ControlMode.CARTESIAN_VELOCITY
                    )
                    if self.cartesian_tracking:
                        logger.info("Setting control mode to %s", generator_mode.name)
                        self.physics_sim.set_control_mode(generator_mode)
                        self.control_mode = generator_mode
                    else:
                        logger.info(
                            "Move accepted for %s: this backend has no Cartesian "
                            "tracking, so the commanded stream is validated but not "
                            "applied -- the arm will not move",
                            move_cmd.motion_generator_mode.name,
                        )
                else:
                    logger.info(
                        "Move accepted for %s / %s, which this server has no physics "
                        "branch for: nothing will be dispatched and no motion-generator "
                        "signal will be checked",
                        move_cmd.motion_generator_mode.name,
                        move_cmd.controller_mode.name,
                    )

                # Seed the limit checker from the state the client is judged
                # against -- q_d / dq_d / ddq_d / tau_J_d are "always sent back to
                # the user in the robot state" for exactly this purpose (libfranka
                # docs/overview.rst), so the client can predict every derivative the
                # robot will compute. Last, because it needs the control mode the
                # branches above just decided.
                #
                # ``generator_mode``, not ``self.control_mode``: the checker must
                # judge only the signal *this* motion's generator owns. A
                # ``kCartesianPosition`` Move does not touch ``self.control_mode``
                # at all (it has no physics branch), so that field still holds
                # the previous motion's mode -- and handing *that* to the checker
                # made it read the zero-filled ``q_c`` of a Cartesian
                # ``RobotCommand`` as a joint position command and abort live
                # clients with
                # ``joint_motion_generator_position_limits_violation``, because
                # joint 4's range does not contain 0. The safety controller
                # (measured velocity -> joint_velocity_violation) is armed
                # regardless of mode; see MotionLimitChecker.start_motion.
                self.motion_limits.start_motion(generator_mode, seed, generation)

                # First send motion started response
                logger.info("Sending kMotionStarted response")
                self.send_move_response(
                    client_socket, command_id=header.command_id, status=MoveStatus.kMotionStarted
                )
                logger.info(f"Motion started with ID: {self.current_motion_id}")

        except Exception as e:
            logger.error(f"Error handling Move command: {e}")
            # Send error response
            self.send_move_response(
                client_socket, command_id=header.command_id, status=MoveStatus.kAborted
            )

    def _preempt_running_motion(self) -> bool:
        """End a motion that a new ``Move`` is displacing. True if there was one.

        ``Move::Status::kPreempted`` -- "Move command preempted!" in
        ``src/robot_impl.h`` -- is the answer the *displaced* motion gets, so
        that is what the pending ``Move`` is answered with here.

        The point is not the status byte, it is what happens to the history.
        Starting a second motion on top of a running one used to leave the
        limit checker's applied history in place and then rebase it on the new
        motion's first command, which is checked against the start-pose
        tolerance alone -- so every extra ``Move`` bought a free step of up to
        :data:`franka_sim.motion_limits.START_POSE_TOLERANCE` with no
        velocity, acceleration or jerk check on it. Ending the old motion the
        ordinary way first (recapture, zero commanded derivatives) means the
        new one is seeded from a genuine standstill at the pose the robot is
        actually in, and every command after its first is differenced again.

        Called with ``_motion_lock`` held.
        """
        if not self.current_motion_id:
            return False
        logger.warning(
            "Move arrived while motion %s was still running: preempting it",
            self.current_motion_id,
        )
        self._engage_idle_hold("preempted by a new Move")
        if self.client_socket is not None:
            self.send_move_response(
                self.client_socket,
                command_id=self.current_motion_id,
                status=MoveStatus.kPreempted,
            )
        self.current_motion_id = 0
        return True

    def _refuse_move_at_singular_pose(
        self, client_socket, header: MessageHeader, move_cmd: MoveCommand
    ) -> bool:
        """Reject a Cartesian ``Move`` that would start from a singular configuration.

        A Cartesian motion generator has to be able to realise an arbitrary
        commanded twist, and in a singularity it cannot: some direction of EE
        motion costs unbounded joint speed. The robot refuses to start rather
        than let the client discover that at 1 kHz, and libfranka has a status
        that says exactly this --
        ``Move::Status::kStartAtSingularPoseRejected``, which it turns into
        ``CommandException("libfranka: Move command rejected: cannot start at
        singular pose!")`` (``src/robot_impl.h:423-427``).

        **The rejection is the motion's *terminal* response, not its first, and
        that is forced by libfranka's own control flow.** ``Robot::Impl::
        startMotion`` sends the ``Move`` through ``executeCommand``, which
        blocking-receives the first response and calls ``handleCommandResponse``
        *outside* any try/catch (``src/robot_impl.h:548-553``); a rejection
        delivered there escapes as a bare ``CommandException``. Only the
        responses picked up by the mode-wait loop that follows are converted --
        ``catch (const CommandException& e) { throw ControlException(e.what()); }``
        (``src/robot_impl.cpp:323-332``), and ``throwOnMotionError`` does the
        same (``:96-111``). A client starting a Cartesian motion from a singular
        configuration sees a ``ControlException``, so on
        hardware the singular start is discovered *after* the ``Move`` was
        acknowledged -- Control accepts the command, looks at where the arm is
        standing, and terminates the motion. This method reproduces that shape:
        ``kMotionStarted`` first, the rejection queued as the deferred terminal
        response, and the generator modes deliberately never leaving idle so the
        client stays in the loop that converts it.

        **Cartesian generators only.** Reaching that singular configuration in
        the first place is done by a *joint* point-to-point motion, so
        joint interfaces plainly start there quite happily -- as they must, since
        a joint generator has no Jacobian to invert and driving *out* of a
        singularity is the only way to leave one.

        The measure is ``sigma_min`` of the EE Jacobian the physics backend
        publishes for the arm's current measured ``q``; see
        :data:`franka_sim.motion_limits.SINGULAR_POSE_MIN_SINGULAR_VALUE` for
        the threshold and the poses it was placed against. A backend that
        publishes no Jacobian -- Genesis, the mobile-base bridge -- cannot be
        asked and the Move is accepted, which is also why the ``mobile_base``
        role (whose ``kCartesianVelocity`` commands a *base twist* and has no
        end effector at all) needs no special case here.

        Called with ``_motion_lock`` held. Returns True when the Move was
        refused (and answered).
        """
        if move_cmd.motion_generator_mode not in (
            MotionGeneratorMode.kCartesianPosition,
            MotionGeneratorMode.kCartesianVelocity,
        ):
            return False
        try:
            sim_state = self.physics_sim.get_robot_state()
        except Exception:  # pragma: no cover - a backend that cannot answer
            # Same posture as :meth:`_publish_hold_setpoint`: a backend that
            # cannot be read is a backend this check cannot be run against, and
            # refusing a Move on the strength of an exception would be worse
            # than running the motion unconditioned.
            logger.exception("Could not read the simulator to condition the Move's start pose")
            return False
        jacobian = sim_state.get("O_J_EE") if isinstance(sim_state, dict) else None
        sigma_min = smallest_singular_value(jacobian)
        if sigma_min is None:
            return False
        if sigma_min > SINGULAR_POSE_MIN_SINGULAR_VALUE:
            logger.info(
                "Cartesian Move %s start-pose conditioning: sigma_min=%.4f (limit %.4f)",
                header.command_id,
                sigma_min,
                SINGULAR_POSE_MIN_SINGULAR_VALUE,
            )
            return False
        logger.warning(
            "Refusing Move %s: the arm is standing in a singularity "
            "(sigma_min=%.4f <= %.4f), so a Cartesian motion cannot start here",
            header.command_id,
            sigma_min,
            SINGULAR_POSE_MIN_SINGULAR_VALUE,
        )
        # A refused Move still displaces whatever was running, exactly as an
        # accepted one does: the client that sent it has stopped servicing the
        # old motion's control loop, and Control does not keep driving a motion
        # nobody is feeding. This also answers the displaced motion's own
        # ``Move`` (``kPreempted``) and clears ``current_motion_id``, which is
        # what leaves :attr:`_pending_move_response` free for the rejection
        # queued below -- the same sequencing the accepted path gets from
        # calling this immediately before it starts a new motion.
        self._preempt_running_motion()
        # An abort latched a moment ago on another thread can still be sitting
        # in the queue with nothing to release it yet. Flush it first rather
        # than overwrite it: it belongs to a *different* Move, and dropping it
        # leaves that client blocked on a TCP reply for ever. Forced, because
        # the state it was waiting for may never arrive -- this Move starts no
        # motion, so nothing after it will publish one on that response's
        # behalf. Safe to call with ``_motion_lock`` held: it is an ``RLock``.
        self._flush_pending_move_response(force=True)
        self.send_move_response(
            client_socket, command_id=header.command_id, status=MoveStatus.kMotionStarted
        )
        if not self.transmitting_state:
            # No publish loop to release a deferred response, and no state for
            # it to race: answer immediately rather than leave the client
            # blocked on a TCP reply for ever. Same escape hatch as
            # :meth:`_flush_pending_move_response`'s ``force``, taken inline
            # because ``_motion_lock`` is already held here.
            self.send_move_response(
                client_socket,
                command_id=header.command_id,
                status=MoveStatus.kStartAtSingularPoseRejected,
            )
            return True
        # Deferred exactly like a reflex abort: the client has to see at least
        # one state after the acknowledgement before the terminal response, or
        # it can be answered before it has even left ``executeCommand``.
        self._pending_move_response = (
            header.command_id,
            MoveStatus.kStartAtSingularPoseRejected,
            self._states_packed,
        )
        return True

    def _refuse_move_while_latched(self, client_socket, header: MessageHeader) -> bool:
        """Reject a ``Move`` that arrives while a reflex is still latched.

        The real robot does not start a motion out of ``kReflex``: the latched
        error has to be cleared with ``AutomaticErrorRecovery`` first. libfranka
        spells the refusal out -- ``Move::Status::kCommandNotPossibleRejected``
        becomes "Move command rejected: command not possible in the current mode
        (<mode>)!" (``src/robot_impl.h``, ``handleCommandResponse<Move>`` via
        ``commandNotPossibleMsg``), which is exactly this situation and the only
        Move status that says it.

        Accepting it instead left the state lying: ``robot_mode`` flipped back
        to ``kMove`` while ``errors`` still carried the violation, so a client
        reading the state saw a running motion and a latched fault at once.

        Called with ``_motion_lock`` held. Returns True when the Move was
        refused (and answered).
        """
        latched = (
            self.comm.violated
            or self.motion_limits.violated
            or self.robot_state.state["robot_mode"] == RobotMode.kReflex
        )
        if not latched:
            return False
        logger.warning(
            "Refusing Move %s: a reflex is still latched -- "
            "AutomaticErrorRecovery has to clear it first",
            header.command_id,
        )
        self.send_move_response(
            client_socket,
            command_id=header.command_id,
            status=MoveStatus.kCommandNotPossibleRejected,
        )
        return True

    def send_move_response(self, client_socket, command_id: int, status: MoveStatus):
        """Send response to Move command"""
        try:
            # Total message size includes header (12 bytes) + response data (status + padding)
            total_size = 12 + 4  # 4 = 1(status) + 3(padding)

            # Construct and send header
            header = MessageHeader(Command.kMove, command_id, total_size)
            header_bytes = header.to_bytes()

            # Construct response data (status + 3 bytes padding)
            logger.debug(f"Sending Move response with status: {status.name} (value={status.value})")
            # Ensure we're using the enum value, not the enum itself
            status_value = status.value if isinstance(status, MoveStatus) else status
            response_data = struct.pack("<B3x", status_value)

            # Send complete message
            message = header_bytes + response_data
            logger.debug(f"Sending Move response message: {message.hex()}")
            self._send_tcp(client_socket, message)
            logger.info(f"Sent Move response: command_id={command_id}, status={status.name}")
        except Exception as e:
            logger.error(f"Error sending Move response: {e}", exc_info=True)

    def _engage_idle_hold(self, reason: str, motion_id: Optional[int] = None) -> None:
        """Recapture the robot when a control session ends, however it ended.

        The real FR3 hands the joints back to its own internal controller the
        instant external control stops -- normal motion completion, StopMove, a
        client that dies mid-stream, a socket error, error recovery. The sim has
        no such controller, so without this the simulator is simply left in
        whatever mode the dead session set: TORQUE, still applying that
        session's last ``tau_J_d`` (or nothing at all). Gravity is compensated
        and the FR3 model's authored viscous damping is small, so an arm
        carrying residual velocity then swings on like a frictionless pendulum
        in zero-g -- the "robot goes bananas after Ctrl-C" bug.

        Idempotent within a session (the latch is rearmed by the next Move and
        by reset_state) so the 1 kHz finish burst holds once, and exception-safe
        because most callers are teardown paths that must not raise.

        Backend-agnostic on purpose: it only calls the simulator contract every
        backend implements, so single-arm MuJoCo/Genesis and both mobile-duo
        scenes (through SceneView) are covered by this one implementation.
        """
        # However the session ended, no control loop is running any more: stop
        # charging cycles to the client and stop reporting a success rate.
        # Unconditional, ahead of the latches below, because the paths that
        # return early here (already held, or nothing ever started) have still
        # ended the motion.
        #
        # ``motion_id`` names the motion the caller believes is ending; passing
        # one makes both calls no-ops if that motion has already been replaced,
        # so a stale event cannot switch the accounting off underneath a motion
        # that is still running. The teardown paths (disconnect, StopMove,
        # socket error) pass nothing, which means "whatever is running".
        self.comm.end_motion(motion_id)
        # Same reasoning for the limit checker: no motion, nothing to
        # difference. A latched violation survives, as the robot's does --
        # only AutomaticErrorRecovery clears it.
        self.motion_limits.end_motion(motion_id)
        with self._hold_lock:
            if self._idle_hold:
                return
            # No motion ever started on this connection (a bare connect, or a
            # Move whose mode combination this server does not serve): there is
            # nothing to recapture, and holding would stomp on whatever pose the
            # previous session was correctly left in.
            if self.control_mode is ControlMode.NONE:
                return
            self._idle_hold = True
            try:
                self._switch_to_hold_position()
            except Exception:
                logger.exception("Failed to engage the idle hold after %s", reason)
                return
        logger.info("Control session ended (%s): idle hold engaged", reason)

    def _switch_to_hold_position(self):
        """Freeze the simulator when a motion finishes or StopMove arrives.

        A mobile base has no joint-space hold: the correct "stop" is a zero
        body twist. An arm holds its current joint positions with zero torque.
        """
        if self.mobile_base:
            if not self._mobile_hold_logged:
                logger.info("Motion finished: commanding zero base twist")
                self._mobile_hold_logged = True
            self.physics_sim.update_base_twist([0.0] * 6)
            # The base is stopped, so the commanded twist the client reads back
            # has to say so -- and the next motion's limit checker seeds from
            # it, so a stale twist here would make a client that correctly
            # restarts from rest look like it stepped.
            self.robot_state.state["O_dP_EE_c"] = [0.0] * 6
            self.robot_state.state["O_dP_EE_d"] = [0.0] * 6
            # Keep the simulator's mode in lockstep with the server's: without
            # this, a client that never re-Moves leaves physics_sim's own
            # control_mode wherever it was (e.g. still mid-transition), so
            # server and simulator could disagree about mode after a hold.
            self.physics_sim.set_control_mode(ControlMode.STEERING_DRIVE)
            self.control_mode = ControlMode.STEERING_DRIVE
            return

        logger.info("Motion finished: switching to position control and holding position")
        # Target first, mode second. The physics thread reads both without a
        # lock, so switching to POSITION before publishing the new target lets
        # a step land in between and servo towards the *previous* position
        # target (the initial pose, or the last q_c of an older motion) at
        # kp=4500 -- a lurch, which is the opposite of a hold.
        current_joint_positions = self.physics_sim.get_robot_state()["q"]
        self.physics_sim.update_joint_positions(current_joint_positions)
        self.physics_sim.update_torques([0.0] * 7)
        self.physics_sim.set_control_mode(ControlMode.POSITION)
        self.control_mode = ControlMode.POSITION
        self._publish_hold_setpoint(current_joint_positions)
        # The Cartesian twin of the zero ``dq_d``/``tau_J_d`` that hold setpoint
        # publishes, and the arm-role counterpart of the mobile branch above: an
        # arm held by its internal controller is commanding no end-effector
        # motion, so the twist it reports as commanded is zero. Only a
        # ``kCartesianVelocity`` motion ever writes anything else here
        # (:meth:`_echo_commanded_cartesian`), and every way such a motion can
        # end -- finished, reflex, StopMove, preemption, a client that vanishes
        # -- arrives at this method, so this is the single place the twist has
        # to be given back.
        self.robot_state.state["O_dP_EE_c"] = [0.0] * 6
        self.robot_state.state["O_dP_EE_d"] = [0.0] * 6

    def _publish_hold_setpoint(self, joint_positions=None) -> Dict[str, Any]:
        """Report the internal controller's own setpoint in ``q_d``/``dq_d``/``tau_J_d``.

        Between motions the robot is held by its internal controller, so the
        commanded values it reports are that hold: the joint positions it is
        holding, zero velocity, zero torque. The sim used to leave the previous
        motion's last command sitting in those fields, which is wrong on the
        wire and worse as a seed -- the next motion's limit checker differences
        the client's first command against them, so a stale ``tau_J_d`` would
        make a client that correctly starts from zero look like it stepped.

        Reads the simulator when no positions are handed in, because a ``Move``
        can arrive before the state loop has published its first frame.

        Returns the setpoint as a state-shaped dict, which is what callers that
        need it as a seed should use. Re-reading ``self.robot_state.state``
        instead works on an arm role -- the publish loop no longer merges the
        backend's ``q_d``/``dq_d``/``ddq_d`` over it -- but not on the mobile
        base, where those fields are the backend's and are rewritten every
        millisecond (see :data:`COMMANDED_STATE_FIELDS`).
        """
        if self.mobile_base:
            return dict(self.robot_state.state)
        if joint_positions is None:
            try:
                joint_positions = self.physics_sim.get_robot_state()["q"]
            except Exception:  # pragma: no cover - a backend that cannot answer
                logger.exception("Could not read the simulator to publish the hold setpoint")
                return dict(self.robot_state.state)
        setpoint = {
            "q": list(joint_positions),
            "q_d": list(joint_positions),
            "dq_d": [0.0] * 7,
            "ddq_d": [0.0] * 7,
            "tau_J_d": [0.0] * 7,
        }
        self.robot_state.state.update(setpoint)
        return setpoint

    def _motion_limit_seed_state(self) -> Dict[str, Any]:
        """The state snapshot a new motion's limit checker is seeded from.

        :meth:`_publish_hold_setpoint` republishes and returns the *commanded*
        fields -- ``q_d``, ``dq_d``, ``ddq_d``, ``tau_J_d`` -- which is what the
        joint and torque generators are differenced against. The Cartesian pose
        generator needs one more thing, and it is a *measured* value rather than
        a commanded one: the flange pose ``O_T_EE`` the robot is actually in, so
        the first ``O_T_EE_c`` of a ``kCartesianPosition`` motion can be judged
        against it (``cartesian_position_motion_generator_start_pose_invalid``).

        ``O_T_EE_d`` is what hardware would be judged against, and this sim now
        publishes it faithfully (:meth:`_publish_commanded_pose`) -- but between
        motions it *is* the measured pose, so the two agree and the measured one
        is used directly. It is also the more honest of the two to seed from:
        the question the check asks is "are you where the robot is", which is
        what a 10 m start-pose offset breaks, and answering it from a
        commanded field would let a stale command excuse a stale command.

        Read out of the backend rather than ``self.robot_state.state`` for arm
        roles, for the same reason :meth:`_publish_hold_setpoint` reads the
        backend for ``q``: ``self.robot_state.state["O_T_EE"]`` is still the
        identity the wire struct was constructed with until the publish loop's
        first cycle has run, and a ``Move`` that arrives before then would have
        its correct first pose judged against that identity and false-aborted
        (observed repro: ``Violation`` index 16, 0.57 m from identity). The
        mobile-base branch of ``_publish_hold_setpoint`` already returns the
        whole state dict, ``O_T_EE`` included, so this only has to add it for
        arm roles -- and it adds it to the *returned* snapshot only, leaving
        what gets written back into ``robot_state.state`` exactly as it was.

        When the backend cannot answer either, ``O_T_EE`` is left out of the
        snapshot entirely rather than filled in from identity: ``start_motion``
        already treats a missing ``O_T_EE`` as "skip the start-pose check", and
        :meth:`~franka_sim.motion_limits.MotionLimitChecker._check_elbow_validity`
        skips its start-elbow half the same way, so a motion armed from a
        backend that has not produced a frame yet runs with those two start
        checks off instead of judging them against a guess.
        """
        seed = self._publish_hold_setpoint()
        if "O_T_EE" not in seed:
            seed = dict(seed)
            try:
                backend_pose = self.physics_sim.get_robot_state().get("O_T_EE")
            except Exception:  # pragma: no cover - a backend that cannot answer
                logger.exception("Could not read the simulator to seed O_T_EE")
                backend_pose = None
            if backend_pose is None:
                if not self._seed_pose_fallback_logged:
                    logger.warning(
                        "No O_T_EE available from the simulator to seed a new "
                        "motion's limit checker; skipping the start-pose/"
                        "start-elbow checks for this motion instead of "
                        "judging them against identity"
                    )
                    self._seed_pose_fallback_logged = True
            else:
                seed["O_T_EE"] = list(backend_pose)
        return seed

    #: Longest the publish loop will hold a state back waiting for the receive
    #: path to catch up; see :meth:`_drain_gate`. Five cycles: long enough to
    #: cover the millisecond-scale hiccups that actually happen (a GIL handover,
    #: a descheduled thread under load -- 2 ms and up, measured), short enough
    #: that a receive path which is genuinely gone -- a dying connection --
    #: costs the loop a bounded slowdown rather than a hang.
    _DRAIN_GATE_TIMEOUT = 0.005
    #: How long each turn of the gate sleeps. Not a busy-wait: the point of the
    #: sleep is to *release* this thread's hold on the CPU (and the GIL) so the
    #: receive thread can run, which is usually all it was waiting for.
    _DRAIN_GATE_YIELD = 0.00005

    def _drain_gate(self) -> float:
        """Hold the state back while a command the client already sent is queued.

        The FCI is one loop: a cycle receives the client's answer, applies it,
        and publishes the state that answer produced. This server splits that
        across two threads -- the publish loop below and the UDP receive thread
        in :meth:`_handle_commands` -- and nothing kept them in step. When the
        receive thread was descheduled for a few milliseconds (CPU contention, a
        GIL handover behind a 5 ms switch interval) the publish loop sailed on,
        emitting states whose ``q_d``/``dq_d``/``ddq_d`` still described the last
        command it had managed to apply, while the client's answers to those very
        cycles sat unread in this process's own socket buffer.

        That is not a cosmetic lag, because **libfranka's control loop is closed
        around those fields**. ``ControlLoop<JointPositions>::convertMotion``
        low-pass filters every waypoint toward ``robot_state.q_d`` with a fixed
        1 ms gain (``src/control_loop.cpp``; the gain at the default 100 Hz
        cutoff is 0.386) and, with rate limiting on, clamps it against
        ``dq_d``/``ddq_d`` as well. Feed a conforming client the *same* stale
        ``q_d`` for several cycles and its commanded stream stops being smooth:
        each waypoint is dragged back toward the frozen reference, and when the
        reference finally moves the filter spends several more cycles catching
        up. The sim then differences that stream and reports the kink it
        manufactured as ``joint_motion_generator_velocity_discontinuity``, at
        hundreds of rad/s^2, against a client that did nothing wrong. That was an
        intermittent abort in the middle of an ordinary approach motion, at a
        ``control_command_success_rate`` of 0.97-0.99.

        So the gate restores the invariant the communication accounting already
        assumes -- "a simulator that stalls delays the state publish and the
        client's answer alike" (:mod:`franka_sim.comm_constraints`) -- for a
        stall that hits only one of the two threads. If a datagram is sitting in
        the socket unread, this cycle's state is not ready to go out yet, so the
        loop sleeps in short turns (which hands the CPU, and the GIL, to the
        receive thread) until the socket is empty or
        :data:`_DRAIN_GATE_TIMEOUT` is up.

        Costs nothing in the ordinary case: the receive path answers a state in
        ~70 us, so by the time the next one is due the socket has long been
        drained and the first ``poll(0)`` returns empty.

        Read-only on the socket. It never calls ``recvfrom`` -- the receive
        thread stays the only consumer -- so polling the same fd from here is
        safe.

        Returns how long it waited, so the caller can push its 1 kHz deadline
        out by the same amount. Without that the pacer treats the wait as lost
        time and fires the next state immediately behind this one, which is the
        opposite of the point: two states microseconds apart give the client no
        cycle to answer the first in.

        **Exactly zero when it did not wait**, rather than the couple of
        microseconds the poll and the two clock reads take. Those microseconds
        are this cycle's own work, and the pacer already accounts for the
        cycle's work by sleeping *before* this call rather than after it --
        charging them to the deadline as well pushed every cycle out by the
        gate's own cost, compounding into a systematic ~2.5 us per cycle and a
        publish rate that sat at ~997.5 Hz in every control mode, idle included,
        for no reason connected to anything the gate is for.
        """
        udp_socket = self.udp_socket
        if udp_socket is None or udp_socket._closed:
            return 0.0
        try:
            fd = udp_socket.fileno()
        except OSError:
            return 0.0
        if fd < 0:
            return 0.0
        if self._drain_poller is None or self._drain_poller_fd != fd:
            poller = select.poll()
            poller.register(fd, select.POLLIN)
            self._drain_poller = poller
            self._drain_poller_fd = fd
        started = time.perf_counter()
        deadline = started + self._DRAIN_GATE_TIMEOUT
        waited = False
        while True:
            try:
                queued = bool(self._drain_poller.poll(0))
            except OSError:
                # The socket went away under us (reconnect, shutdown). Nothing
                # to wait for, and certainly nothing to hang on.
                return (time.perf_counter() - started) if waited else 0.0
            # The socket being empty is not the same as the answer being in.
            # ``_handle_commands`` takes the datagram out of the socket at the
            # *start* of its turn and writes the commanded echo -- ``q_d``,
            # ``O_T_EE_c``, ``elbow_c`` -- at the *end* of it, with the
            # decoding, the communication accounting and the whole limit check
            # in between. Releasing on an empty socket therefore released
            # straight into the one window where the state is guaranteed
            # stale, and did so precisely when the gate had engaged at all: the
            # moment the receive thread drains the last queued datagram is the
            # moment it still has all of its work ahead of it. Worse, the
            # communication accounting has already run by then, so the cycle
            # counts as *answered* and the publish loop does not extrapolate
            # over it either -- the reference simply freezes for one cycle, the
            # client's own 100 Hz command filter blends toward the frozen value
            # and emits a (1 - gain) x one-cycle step, and this server aborts
            # the motion on the discontinuity it manufactured. Measured on a
            # Cartesian elbow motion: a 179 urad/cycle elbow ramp came back
            # 110 urad short on exactly that cycle, i.e. 110 rad/s^2 against a
            # 10 rad/s^2 limit. So the gate waits for the datagram to be
            # *applied*, not merely read.
            if not queued and self._commands_in_flight == 0:
                return (time.perf_counter() - started) if waited else 0.0
            now = time.perf_counter()
            if now >= deadline:
                logger.debug(
                    "State publish went ahead with a command still unapplied "
                    "after %.1f ms; the receive path is behind",
                    self._DRAIN_GATE_TIMEOUT * 1e3,
                )
                return now - started
            time.sleep(self._DRAIN_GATE_YIELD)
            waited = True

    def _account_for_communication_cycle(self) -> None:
        """Run one cycle of the FCI communication-constraints emulation.

        Called from the state-publish thread, once per cycle, just before the
        state goes out:

        1. close the cycle that the previously published state opened,
        2. publish the rolling ``control_command_success_rate``,
        3. tell the limit checker which state is going out, so its differencing
           interval is the server's own observation and not the client's echo,
        4. **extrapolate the motion generator across a cycle the client missed**,
           dispatching the result to physics and publishing it back in the
           commanded fields (:meth:`_extrapolate_missed_cycle`),
        5. abort the motion once too many cycles are lost in a row.

        Step 4 is the one that commands something, and it is what the real FCI
        does: Control "takes the previous waypoints and performs a linear
        extrapolation (keep acceleration constant and integrate) for the missed
        time step" (``docs/system_requirements.rst``), and the extrapolated
        value is what comes back in ``q_d``/``dq_d``/``ddq_d``
        (``docs/overview.rst``). A dropped *controller* packet is the exception
        and is left alone: "FCI will reuse the torques of the last successful
        received packet", which is what happens here by doing nothing at all.

        Extrapolation runs whether or not either enforcement flag is set,
        exactly as it does on hardware -- it is not a check, it is what the
        robot's reference *is* while the client is quiet. The flags still decide
        only whether a violation aborts.

        The extrapolation stops at
        :data:`~franka_sim.comm_constraints.MAX_CONSECUTIVE_LOST_CYCLES`
        consecutive misses, which is where the robot stops the motion. Past that
        the reference holds: a client that has genuinely gone away leaves the
        arm standing still rather than flying off along the trajectory it was on
        when it vanished. That bound is enforced here rather than inside the
        checker so the checker never has to know how long the gap has been --
        this is the only place that counts cycles.

        Kept out of the transmission loop's body so the loop stays readable and
        this stays testable on its own.

        The few microseconds between this call and the ``sendto`` are charged to
        the client: a command landing in them is late by the tracker's reckoning
        even though the state it was racing is not out yet. That bias is under
        one cycle and always in the conservative direction; see
        :meth:`CommConstraintTracker.command_received` for why crediting it back
        cannot be done without a wall clock.
        """
        published_id = self.robot_state.state["message_id"]
        outcome = self.comm.tick(published_id)
        # The differencing interval the motion-limit checker uses is bounded by
        # this: how many states the server has actually published since the one
        # the applied command history sits at. See
        # :meth:`MotionLimitChecker.cycles_since_applied`.
        self.motion_limits.note_published(published_id)

        # Zero when no control or motion generator loop is running, which is
        # what the real robot reports: control_command_success_rate "shows a
        # value of zero if no control or motion generator loop is currently
        # running" (libfranka include/franka/robot_state.h).
        self.robot_state.state["control_command_success_rate"] = outcome.success_rate

        if not outcome.active:
            return

        if outcome.lost:
            if outcome.consecutive_lost < self.comm.max_consecutive_lost:
                self._extrapolate_missed_cycle(outcome.closed_id)
            elif outcome.bound_reached:
                # Past this point the reference is *held*, so the derivatives
                # the checker would judge a resumed command against describe a
                # client that is no longer there. It re-seeds them from the
                # client's own first two commands instead; see
                # :meth:`MotionLimitChecker.note_hold`.
                self.motion_limits.note_hold()
                # The bound the robot stops at. With enforcement on, the abort
                # below is about to fire and this line is its explanation; with
                # enforcement off it is the whole report, which is why it is not
                # inside the ``violation_triggered`` branch.
                logger.warning(
                    "%s consecutive lost command cycles: no longer extrapolating "
                    "the motion generator, holding the last reference%s",
                    outcome.consecutive_lost,
                    "" if self.enforce_comm_constraints else " (not enforced)",
                )

        if outcome.violation_triggered:
            self._abort_on_communication_violation(outcome.motion_id)

    def _extrapolate_missed_cycle(self, closed_id: int) -> None:
        """Command the waypoint the client did not send, for one missed cycle.

        The motion-generator half of the FCI's packet-loss behaviour. The
        arithmetic is the limit checker's -- it already holds the backward
        differences of the last two real commands, which is the only honest
        source for the acceleration to freeze (see
        :meth:`franka_sim.motion_limits.MotionLimitChecker.extrapolate`) -- and
        what this method owns is the consequences: judging the result, and
        putting it where a real command would have gone.

        Those two are the same plumbing a received command gets, deliberately:

        * the extrapolated waypoint is **checked like a command**, and an
          extrapolation that runs out past the velocity envelope or a joint stop
          latches the same error a client commanding it would have. That is not
          collateral damage, it is the documented hazard -- intermittent drops
          "could trigger `discontinuity` errors even when your source signals
          conform with the interface specification" (``docs/overview.rst``) --
          and a sim that clamped the extrapolation instead would be hiding
          precisely the failure the client came here to find;
        * it is then **dispatched through :meth:`_dispatch_control_command`**,
          the same path a received datagram takes, so the arm keeps moving
          through the gap and ``q_d``/``dq_d``/``ddq_d`` (and
          ``O_T_EE_c``/``O_T_EE_d`` on a pose motion) advance with it. Reusing
          that method rather than writing the fields here is what keeps the
          FCI-layer field ownership in one place; it also means the hold latch
          covers this path for free -- a session that ended between the tick and
          here dispatches nothing.

        ``closed_id`` names the state whose cycle went unanswered. It is stamped
        on the substitute command, so the checker's applied history sits at that
        id and the client's resumed command -- which answers a *later* state --
        is differenced over the standard single cycle.

        Runs on the state-publish thread. Lock order is unchanged from the other
        publish-thread violation path (:meth:`_run_safety_velocity_check`):
        ``_motion_lock`` -> checker lock -> ``_hold_lock``, with the dispatch
        taking ``_hold_lock`` alone and never while another server lock is held.

        **The motion is re-validated inside the dispatch's own lock.** A ``Move``
        handled on the TCP thread between the extrapolation and the dispatch
        changes ``motion_generator_mode`` and ``controller_mode``, and
        :meth:`_dispatch_control_command` branches on exactly those -- so a
        substitute built for the *old* motion's generator would be routed into
        the new one's branch and written to fields the old motion never
        commanded. The re-check is on the server's own motion token, read
        without taking any checker lock so that the innermost-lock rule holds.
        """
        motion_id = self.motion_limits.motion_id
        generation = self._motion_generation
        extrapolated = self.motion_limits.extrapolate(closed_id)
        if extrapolated is None:
            # Nothing to continue: no motion generator armed, no real command
            # recorded yet, or a torque controller -- which holds.
            return
        command, violation = extrapolated

        if violation is not None:
            self.motion_limits.report(violation, enforced=self.enforce_motion_limits)
            if self.enforce_motion_limits:
                self._latch_and_abort(violation, motion_id)
                return
            if violation.fatal:
                return

        # The wire-side half of the checker's ``_gap_snapshot``, taken on the
        # first guess of a run and put back by
        # :meth:`_absorb_within_motion_limits` when a late datagram makes the
        # checker throw that run's guesses away. Without it the rewind is only
        # half done: the history goes back but the published ``q_d`` keeps the
        # extrapolated ``dq_ext * dt``, and then integrates the real command on
        # top of it -- one extra millisecond of reference per rewound packet,
        # measured as -2.35630 where two cycles of -0.1 rad/s put it at
        # -2.35620. Only the *first* guess snapshots: the ones after it are
        # part of the same run and rewind to the same place.
        if self._extrapolated_reference is None:
            self._extrapolated_reference = list(self.robot_state.state["q_d"])

        self._dispatch_control_command(command, motion_generation=generation)

    def _absorb_within_motion_limits(self, command, *, fresh: bool) -> bool:
        """Rewind, validate and record one received command; False if refused.

        The UDP thread's whole interaction with the limit checker, in one call
        that the checker takes under one hold of its lock -- see
        :meth:`franka_sim.motion_limits.MotionLimitChecker.absorb_command` for
        why that indivisibility is load-bearing and not merely tidy.

        Reporting and aborting stay here, outside the checker's lock, so the
        lock order is exactly what it was: ``_motion_lock`` (taken by
        :meth:`_latch_and_abort`) -> checker lock, never the reverse, and the
        checker's lock is still the innermost one anything in this server takes.

        Always reports -- a violation logs a rate-limited warning naming the
        joint or axis, the value and the limit, once per error class per motion.
        Only with ``--enforce-motion-limits`` does it additionally abort the
        motion and refuse the command, which is what the real robot does with a
        signal it cannot follow. A *fatal* violation (a non-finite command) is
        refused either way: applying NaN is not permissiveness, it poisons the
        physics state, the backward differences and the wire. It does not abort,
        though -- aborting is what the switch is for.

        **A rewind takes the published reference back with it.** The checker
        undoes the guesses this datagram turns out to be the real answer for
        (:meth:`franka_sim.motion_limits.MotionLimitChecker.rewind_extrapolation`),
        and the reference those guesses were dispatched into has to follow, or
        the caller's dispatch integrates the same millisecond twice: once as
        ``dq_ext * dt`` when the cycle was guessed and again as ``dq_c * dt``
        now. See :attr:`
        franka_sim.franka_sim_server.FrankaSimServer._extrapolated_reference`.
        """
        motion_id = self.motion_limits.motion_id
        outcome = self.motion_limits.absorb_command(
            command,
            self.robot_state.state["q"],
            fresh=fresh,
            enforce=self.enforce_motion_limits,
        )
        if outcome.rewound:
            reference = self._extrapolated_reference
            self._extrapolated_reference = None
            if reference is not None:
                self.robot_state.state["q_d"] = list(reference)
        elif outcome.recorded:
            # A fresh command closes the run of losses (the checker drops its
            # snapshot in the same step), so the guesses it was judged against
            # stand and there is nothing left to put back.
            self._extrapolated_reference = None
        if outcome.recorded and self.control_mode is ControlMode.TORQUE:
            # **The published reference must never lag the one this command was
            # just recorded into.** ``tau_J_d`` is written for real by
            # :meth:`_dispatch_control_command`, three statements and a physics
            # call later -- and the state-publish thread runs concurrently. Its
            # drain gate normally holds it off (``_commands_in_flight``), but the
            # gate gives up after :data:`_DRAIN_GATE_TIMEOUT`, which is exactly
            # what a starved 2-core CI runner makes it do. A state that goes out
            # in that window carries the *previous* command's ``tau_J_d`` while
            # the checker's torque reference has already advanced to this one.
            #
            # ``franka_hardware`` rate-limits torques against the robot's own
            # reported reference -- ``franka::limitRate(kMaxTorqueRate, tau,
            # current_state_.tau_J_d)`` (``franka_hardware/src/robot.cpp:171``)
            # -- and libfranka's ``receiveRobotState`` hands it the newest state
            # on the socket, so a client cannot route around a reference the
            # server published one command behind. It then saturates its limiter
            # from that stale value and the next command differences to
            # ``999.999 + |the client's own previous torque rate|`` over one
            # cycle: a conforming torque controller aborted with
            # ``controller_torque_discontinuity: tau_J_d joint 4 = 1002.81 Nm/s``
            # off a 2.81 Nm/s ramp. Hardware has no such window -- the state for
            # cycle k carries the torque Control applied at cycle k, by
            # construction -- so publishing the echo here, before anything else
            # can be observed, is what makes the sim's reference honest.
            self.robot_state.state["tau_J_d"] = list(command["tau_J_d"])

        if outcome.violation is None:
            return True

        self.motion_limits.report(outcome.violation, enforced=self.enforce_motion_limits)
        if not self.enforce_motion_limits:
            return outcome.accepted
        self._latch_and_abort(outcome.violation, motion_id)
        return False

    def _latch_and_abort(self, violation, motion_id: int) -> None:
        """Latch the first violation of a motion and abort it, as one step.

        The check ("is a violation already latched?") and the act (latch, then
        abort) run from *two* threads -- the UDP receive thread checks each
        command, the state-publish thread runs the safety controller -- so they
        have to be indivisible. Two things go wrong otherwise, and the second
        does not heal:

        * both threads read ``violated`` as False in the same instant and abort
          twice, answering one ``Move`` with two terminal responses;
        * the running motion ends and the next one starts between the caller's
          read of the checker's token and the latch. :meth:`
          franka_sim.motion_limits.MotionLimitChecker.start_motion` clears the
          latch for the new motion, so the stale latch lands on *it* -- while
          :meth:`_abort_with_error`, correctly, refuses to abort a motion whose
          token no longer matches. The new motion then runs with a violation
          latched and no reflex behind it, and because the server refuses a
          ``Move`` while a violation is latched, every later ``Move`` on that
          connection is answered ``kCommandNotPossibleRejected``. Permanently
          un-abortable, until ``AutomaticErrorRecovery``.

        Holding ``_motion_lock`` across the whole sequence closes both. The
        running motion cannot change under us, so re-reading the checker's token
        *inside* the lock decides once and for all whether this violation still
        belongs to a live motion -- and if it does, ``_abort_with_error`` (which
        takes the same re-entrant lock) cannot then refuse it. **The latch and
        the abort now succeed or fail together**, which is the invariant that
        makes an un-abortable motion impossible.

        Lock order here is ``_motion_lock`` -> the checker's own lock (taken and
        released by :attr:`motion_limits.motion_id` and ``latch()``, never held
        across the call into ``_abort_with_error``) -> ``_hold_lock`` (taken
        inside ``_abort_with_error`` -> ``_engage_idle_hold``) -> the simulator
        backend (called from inside ``_hold_lock`` by
        ``_switch_to_hold_position``). ``_motion_lock`` -> ``_tcp_send_lock`` is
        a separate nesting used elsewhere (``_finish_motion``,
        ``handle_stop_move_command``); it does not occur on this path, because
        ``_abort_with_error`` defers its ``Move`` response and sends it, if at
        all, from ``_flush_pending_move_response`` after ``_motion_lock`` has
        already been released. ``_tcp_send_lock`` and ``_hold_lock`` are both
        leaves -- neither is held while acquiring another server lock -- so no
        two locks are ever taken in both orders and there is no cycle.

        ``motion_id`` is the checker's token as the caller read it before
        running its check; 0 means "no motion named", which is what a violation
        raised between motions (a non-finite datagram) carries.
        """
        with self._motion_lock:
            running = self.motion_limits.motion_id
            if motion_id != running:
                logger.info(
                    "Not latching (%s): the motion that violated is already over",
                    violation.describe(),
                )
                return
            if running and running != self._motion_generation:
                # The checker still names a motion the server has moved past:
                # it is on its way out, so latching would strand its successor.
                logger.info(
                    "Not latching (%s): the running motion is being replaced",
                    violation.describe(),
                )
                return
            if self.motion_limits.violated:
                # Already latched: a burst of bad commands aborts once.
                return
            self.motion_limits.latch()
            self._abort_with_error(violation.error_indices, violation.describe(), motion_id)

    def _publish_commanded_pose(self, sim_state) -> None:
        """Report the internal controller's hold pose in ``O_T_EE_d``/``O_T_EE_c``.

        The Cartesian twin of :meth:`_publish_hold_setpoint`. Between motions --
        and during every motion whose generator is not the Cartesian pose one --
        the robot is held by its internal controller, so the pose it reports as
        commanded *is* the pose it is in. These fields were a permanent identity
        stub for as long as nothing read them, and that was not harmless: a
        libfranka Cartesian-pose motion generator initialises and holds from
        ``O_T_EE_d`` (the conventional opening is
        ``std::array<double, 16> cmd = state.O_T_EE_d;``), so an identity there
        made *every* pose motion open ten-ish metres and a full rotation away
        from the robot and trip
        ``cartesian_position_motion_generator_start_pose_invalid`` on cycle 0 --
        a sim artifact that masked every other Cartesian error behind it.

        During either Cartesian motion the two fields belong to that motion
        instead -- to the client's stream on the pose generator
        (:meth:`_echo_commanded_cartesian`), and frozen at the motion's start
        pose where the stream carries none
        (:meth:`_cartesian_motion_owns_commanded_fields`, which is also where
        the reason they must not track the measured arm there is written down).
        The moment the motion ends they snap back here, which is what the real
        robot does when the internal controller takes the flange back.

        **Arm roles only.** The mobile-duo base bridge's ``O_T_EE`` is
        dead-reckoned from the commanded twist and its commanded Cartesian
        fields are ``O_dP_EE_d``/``O_dP_EE_c``; neither is this method's
        business. Same role guard as :data:`COMMANDED_STATE_FIELDS`.
        """
        if self.mobile_base or self._cartesian_motion_owns_commanded_fields():
            return
        pose = sim_state.get("O_T_EE")
        if pose is None:
            return
        pose = pose.tolist() if hasattr(pose, "tolist") else list(pose)
        self.robot_state.state["O_T_EE_d"] = pose
        self.robot_state.state["O_T_EE_c"] = pose

    def _publish_elbow(self, sim_state) -> None:
        """Report the arm's elbow configuration in ``elbow`` and ``elbow_d``.

        ``elbow[0]`` is the 7-DOF redundancy angle, which on an FR3 *is* joint 3,
        and ``elbow[1]`` is the branch flag: the sign of joint 4, which libfranka
        insists is exactly +-1 (``isValidElbow``, ``include/franka/control_tools.h``).
        Both are a reading of ``q``, so there is nothing to model and nothing to
        integrate -- which is why these fields were a permanent ``[0.0, 0.0]``
        stub for as long as no generator consumed them.

        Now one does. A ``kCartesianPosition`` client builds its command as
        ``franka::CartesianPose{state.O_T_EE_d, state.elbow_d}``, and libfranka
        refuses to *send* an elbow whose flag is not +-1 -- ``checkElbow`` throws
        ``std::invalid_argument`` client-side -- so a permanently zero
        ``elbow_d`` made the elbow interface unreachable: the client threw before
        a datagram was ever packed. Publishing the real thing is what lets the
        elbow checks in :mod:`franka_sim.motion_limits` be exercised by a real
        client at all.

        Arm roles only. On the mobile-duo base bridge ``q`` is the four swerve
        steer/drive joints, which have no elbow of any kind.
        """
        if self.mobile_base:
            return
        elbow = self._elbow_from_joints(sim_state.get("q"))
        if elbow is None:
            return
        self.robot_state.state["elbow"] = elbow
        # ``elbow_d``/``elbow_c`` are the *commanded* elbow on hardware. While
        # either Cartesian generator is running they belong to that motion --
        # the client's stream when it commands an elbow
        # (:meth:`_echo_commanded_cartesian`), frozen at the motion's start
        # elbow when it does not -- and must not track the arm, for the
        # feedback reason spelled out in
        # :meth:`_cartesian_motion_owns_commanded_fields`. The rest of the time
        # -- idle, between motions, under a joint generator -- the honest value
        # is the one the internal controller is holding, which is the measured
        # elbow. Same reasoning as ``q_d`` between motions; see
        # :meth:`_publish_hold_setpoint`.
        if self._cartesian_motion_owns_commanded_fields():
            return
        self.robot_state.state["elbow_d"] = list(elbow)
        self.robot_state.state["elbow_c"] = list(elbow)

    @staticmethod
    def _elbow_from_joints(joints) -> Optional[List[float]]:
        """``[redundancy angle, branch flag]`` for an arm configuration, or None.

        ``elbow[0]`` is joint 3's angle on an FR3 and ``elbow[1]`` is the sign
        of joint 4, which libfranka insists is exactly +-1 (``isValidElbow``,
        ``include/franka/control_tools.h``). None when the caller has no arm
        configuration to read, so every caller can skip rather than guess.
        """
        if joints is None or len(joints) < 4:
            return None
        return [float(joints[2]), 1.0 if float(joints[3]) >= 0.0 else -1.0]

    def _refresh_ee_transform(self) -> None:
        """Recompute ``F_T_EE`` from ``F_T_NE``/``NE_T_EE`` and tell the backend.

        ``F_T_EE = F_T_NE * NE_T_EE`` -- libfranka's own decomposition
        (``Robot::setEE``, ``include/franka/robot.h``), so the two settable
        halves and the derived whole can never disagree. Both are column-major
        16-element wire poses, and so is the result.

        The backend is told because the EE frame is where it measures Cartesian
        velocity for the safety controller (see
        :meth:`franka_sim.mujoco_franka_sim.MujocoFrankaSim.update_ee_transform`).
        A backend without that setter -- Genesis, the mobile-base bridge -- is
        left alone and simply publishes no ``O_dP_EE``, which switches the
        Cartesian check off rather than feeding it a wrong frame.
        """
        f_t_ee = transform_matrix(self.robot_state.state["F_T_NE"]) @ transform_matrix(
            self.robot_state.state["NE_T_EE"]
        )
        values = [float(value) for value in f_t_ee.T.flatten()]
        self.robot_state.state["F_T_EE"] = values
        # getattr on self too: reset_state() also runs from teardown paths, and
        # a half-built server has no backend yet.
        setter = getattr(getattr(self, "physics_sim", None), "update_ee_transform", None)
        if setter is None:
            return
        try:
            setter(values)
        except Exception:  # pragma: no cover - a backend that rejects the frame
            logger.warning("Backend rejected the F_T_EE update", exc_info=True)

    def _run_safety_velocity_check(self, sim_state) -> None:
        """Run the safety controller against this cycle's measured velocity.

        The limit checks that judge the *robot* rather than the client, and the
        only ones not tied to a motion generator: the real robot watches
        measured ``dq`` against the position-based velocity envelope and latches
        ``joint_velocity_violation``, and watches the measured end-effector
        speed against the Cartesian translational limit and latches
        ``cartesian_velocity_violation`` -- both in every control mode. Hardware
        pins the case no commanded check could ever cover:
        a pure-torque session ramping 3 Nm into joint 6 until the arm folds
        through the envelope, with no commanded velocity anywhere in the
        session.

        **The Cartesian check is asked first, and that ordering is the
        hardware's.** Ramping joints 2 and
        4 with an EE 0.5 m out along the flange, hardware reports
        ``cartesian_velocity_violation`` *on its own* -- no
        ``joint_velocity_violation`` beside it -- so whichever cycle is the
        first to break both, the Cartesian error is the one that must latch.
        (In this sim the EE passes 3 m/s some 90 ms before either joint check
        objects, so the ordering only matters as a guarantee, not in practice.)

        Called once per cycle from the state-publish loop with the physics
        snapshot it has already read, so it costs one comparison per joint, one
        vector norm, and no extra backend call. Reporting is always on; the
        abort is gated on ``--enforce-motion-limits`` and goes through the same
        latch-then-abort plumbing as every other violation, so the client sees
        the identical thing: the error bit in ``errors``/``reflex_reason``,
        ``kReflex``, and the pending ``Move`` answered ``kReflexAborted``.

        Skipped on a ``mobile_base`` server: the envelope is the FR3's, read out
        of the arm's own URDF, and a swerve base's steering and drive joints are
        not FR3 joints. The duo's *arms* run on ordinary arm servers, which do
        get the check.
        """
        if self.mobile_base or sim_state is None:
            return
        motion_id = self.motion_limits.motion_id
        # Both are asked every cycle, not short-circuited: each also *records*
        # this cycle's reading, and the joint one's ``q`` is the reference
        # configuration the commanded-velocity envelope is built from. Only the
        # verdict is prioritised.
        cartesian = self.motion_limits.check_measured_cartesian_velocity(sim_state.get("O_dP_EE"))
        joint = self.motion_limits.check_measured_velocity(sim_state.get("q"), sim_state.get("dq"))
        violation = cartesian or joint
        if violation is None:
            return
        self.motion_limits.report(violation, enforced=self.enforce_motion_limits)
        if not self.enforce_motion_limits:
            # Nothing to refuse: no command caused this, so "reject the
            # command" is not a remedy. Reported and carried on with.
            return
        self._latch_and_abort(violation, motion_id)

    def _run_self_collision_check(self, sim_state) -> None:
        """Run the safety controller's geometric third against this cycle's contacts.

        The third measured-side check, and the same shape as
        :meth:`_run_safety_velocity_check`: it judges where the arm *is*, not
        what the client asked for, so it is armed in every control mode -- which
        is what the reference provocation's torque-control variant needs, since
        there the client drives joint 4 from its own controller and there is no
        commanded signal for any other check to object to.

        The backend does the geometry (``MujocoFrankaSim.self_collision``) and
        publishes either a link pair inside the margin or None. **A snapshot
        with no ``self_collision`` key at all switches the check off**, which is
        deliberately not the same as a key holding None: the Genesis backend,
        the mobile base's swerve backend and every mocked simulator in the tests
        publish no such key, and reading their silence as "the arm is clear"
        would be inventing a measurement none of them took.
        Also skipped on a ``mobile_base`` server, for the reason
        :meth:`_run_safety_velocity_check` is: the monitored pairs are the FR3's
        own links.
        (The duo's *arms* run on ordinary arm servers and do get the check.)
        Run *before* the velocity checks in the publish loop, because on the
        reference provocation the fold ends with joint 4 parked against its
        position limit, where the position-based envelope collapses; the sim's
        own margin buys 833 cycles of lead over that, and the ordering makes the
        geometric error win even if it did not.

        Reporting is always on; the abort is gated on
        ``--enforce-motion-limits`` and goes through the same latch-then-abort
        plumbing as every other violation, so the client sees the identical
        thing: the error bit in ``errors``/``reflex_reason``, ``kReflex``, and
        the pending ``Move`` answered ``kReflexAborted``.
        """
        if self.mobile_base or not isinstance(sim_state, dict):
            return
        if "self_collision" not in sim_state:
            return
        motion_id = self.motion_limits.motion_id
        violation = self.motion_limits.check_self_collision(sim_state["self_collision"])
        if violation is None:
            return
        self.motion_limits.report(violation, enforced=self.enforce_motion_limits)
        if not self.enforce_motion_limits:
            # Nothing to refuse: no command caused this -- the arm is simply
            # where it is -- so "reject the command" is not a remedy. Reported
            # once per approach (see MotionLimitChecker.check_self_collision)
            # and carried on with.
            return
        self._latch_and_abort(violation, motion_id)

    def _abort_on_communication_violation(self, motion_id: int) -> None:
        """Stop the motion with ``communication_constraints_violation``."""
        self._abort_with_error(
            COMMUNICATION_CONSTRAINTS_VIOLATION_INDEX,
            f"communication_constraints_violation: {self.comm.consecutive_lost} "
            "consecutive lost command cycles",
            motion_id,
        )

    def _abort_with_error(
        self, error_index: Union[int, Sequence[int]], reason: str, motion_id: int = 0
    ) -> None:
        """Stop the motion with one or more latched errors, the way the robot does.

        ``error_index`` is usually a single index, but accepts a sequence too:
        hardware latches *two* bits from a single abort in exactly one case --
        a commanded velocity-envelope violation (13) that trips while the
        safety controller is armed also latches ``joint_velocity_violation``
        (3); see :attr:`franka_sim.motion_limits.Violation.extra_error_index`
        and the citation there. Every index in the sequence lands in the same
        state update and the same deferred ``Move`` response, so the client
        reads them off one abort rather than two.

        Shared by every reflex the sim emulates -- the communication-constraints
        violation and each of the motion-limit violations -- because a client
        observes all of them identically: the error is latched in the state
        (``errors`` -> ``current_errors``, ``reflex_reason`` ->
        ``last_motion_errors``; see ``convertRobotState`` in
        ``src/robot_impl.cpp``), the modes leave ``kMove``, and the pending
        ``Move`` is answered with ``kReflexAborted`` -- the status libfranka
        turns into ``CommandException`` -> ``ControlException`` carrying
        ``last_motion_errors`` (``robot_impl.h``, ``handleCommandResponse``).

        libfranka spots the abort in the state first (``throwOnMotionError``
        keys off ``robot_mode != kMove``) and only then blocks for the TCP
        response, so the state carrying the error has to be on the wire before
        the response is. Latching it into ``robot_state`` is not enough: the
        publish loop latches, *then* packs and sends, so a response sent from
        here would beat the state it belongs to out of the machine by however
        long the packing takes.

        The response is therefore queued in :attr:`_pending_move_response`,
        stamped with :attr:`_states_packed` as it stands at this instant, and
        released only once a state packed *after* that has actually gone out.
        The stamp is what makes the guarantee hold for an abort raised on the
        UDP receive thread as well as one raised on the publish thread: a
        motion-limit violation can latch in the microseconds between the publish
        loop's ``pack_state()`` and its ``sendto()``, in which case the packet
        already on its way was serialised *before* the error existed and cannot
        be the one that carries it. Counting packs rather than sends is
        deliberate -- the send is what the client sees, but the pack is when the
        error either made it into the bytes or did not.

        When no publish loop is running to flush it (teardown, a unit test
        driving this directly) it goes out immediately instead -- late is better
        than never, and there is no state to race.

        ``error_index`` is the 0-based position of the error in the 41-entry
        wire arrays, i.e. its ``research_interface::robot::Error`` enumerator --
        or a sequence of them, all latched in this one abort.

        ``motion_id`` is the session token of the motion that violated (see
        :attr:`_motion_generation`). The three threads that start, finish and
        abort motions all reach for the same state, so an abort raised on the
        publish thread can arrive after the TCP thread has already started the
        *next* motion; aborting then would kill an innocent motion and answer
        the wrong ``Move`` with ``kReflexAborted``. A token of 0 means "whatever
        is running", for the callers that have no motion to name.
        """
        with self._motion_lock:
            if motion_id and motion_id != self._motion_generation:
                logger.info(
                    "Not aborting: %s belonged to a motion that is already over", reason
                )
                return

            logger.error("%s -- aborting the motion", reason)
            state = self.robot_state.state
            indices = (error_index,) if isinstance(error_index, int) else tuple(error_index)
            for index in indices:
                state["errors"][index] = True
                state["reflex_reason"][index] = True
            state["robot_mode"] = RobotMode.kReflex
            state["motion_generator_mode"] = LibfrankaMotionGeneratorMode.kIdle.value
            state["controller_mode"] = LibfrankaControllerMode.kOther.value

            # Hands the joints back to the internal controller, exactly as every
            # other way a session can end does.
            self._engage_idle_hold(reason, motion_id=motion_id or None)

            if self.current_motion_id and self.client_socket is not None:
                self._pending_move_response = (
                    self.current_motion_id,
                    MoveStatus.kReflexAborted,
                    self._states_packed,
                )
                self.current_motion_id = 0
        if not self.transmitting_state:
            self._flush_pending_move_response(force=True)

    def _flush_pending_move_response(self, *, force: bool = False) -> None:
        """Send the ``Move`` response an abort deferred, if one is owed and due.

        Called by the publish loop straight after each ``sendto``. The response
        is due once :attr:`_states_packed` has moved past the value stamped when
        the error was latched, i.e. once a state serialised *after* the error
        existed has gone out; until then this is a no-op and the next cycle
        tries again.

        ``force`` skips that wait, for the teardown paths: no further state will
        ever be published, so holding the response back would leave the client
        blocked on a TCP reply for ever.
        """
        with self._motion_lock:
            pending = self._pending_move_response
            if pending is None:
                return
            motion_id, status, latched_after = pending
            if not force and self._states_packed <= latched_after:
                # The packet that just went out was serialised before the error
                # was latched, so it is not the state that carries it.
                return
            if self.client_socket is None:
                # The connection is gone; there is nobody left to answer. Drop
                # it explicitly rather than leaving it queued for a socket that
                # will never come back.
                logger.debug(
                    "Dropping the deferred Move response for motion %s: no client socket",
                    motion_id,
                )
                self._pending_move_response = None
                return
            self._pending_move_response = None
            client_socket = self.client_socket
        self.send_move_response(client_socket, command_id=motion_id, status=status)

    def _handle_cartesian_velocity(self, command):
        """Route a cartesian-velocity command to the mobile base.

        The TMR master accepts only a body-frame twist; libfranka carries it in
        ``O_dP_EE_c`` = ``[vx, vy, vz, wx, wy, wz]``. Swerve inverse kinematics
        and the wheel targets live in the simulator, mirroring the real robot
        where the master does the IK onboard.

        Only reached on a ``mobile_base`` server: the dispatcher branch below
        carries that guard, so there is no ``mobile_base`` test here -- and no
        log statement either, since this runs once per UDP datagram (~1 kHz).
        """
        twist = list(command["O_dP_EE_c"])
        self.robot_state.state["O_dP_EE_c"] = twist
        self.robot_state.state["O_dP_EE_d"] = twist

        if self.control_mode is not ControlMode.STEERING_DRIVE:
            # Logged only on the transition, never per datagram.
            logger.info("Setting control mode to STEERING_DRIVE")
            self.physics_sim.set_control_mode(ControlMode.STEERING_DRIVE)
            self.control_mode = ControlMode.STEERING_DRIVE

        self.physics_sim.update_base_twist(twist)

    def handle_stop_move_command(self, client_socket, header: MessageHeader):
        """Handle StopMove command received over TCP"""
        responded = False
        try:
            logger.info("Processing StopMove command")

            # Send success response for StopMove first
            total_size = 12 + 4  # Header (12) + status (1) + padding (3)
            response_header = MessageHeader(Command.kStopMove, header.command_id, total_size)
            header_bytes = response_header.to_bytes()

            # Status 0 = Success
            response_data = struct.pack("<B3x", 0)  # 1 byte status + 3 bytes padding

            self._send_tcp(client_socket, header_bytes + response_data)
            logger.info("Sent StopMove success response")
            responded = True

            self._engage_idle_hold("StopMove")

            # Put the published state back to idle modes -- unless a reflex is
            # latched, in which case the modes already left kMove and
            # ``robot_mode`` is kReflex. Overwriting that with kIdle would tell
            # the client the motion ended cleanly while ``errors`` still says
            # why it did not, and kReflex is the flag that has to survive until
            # AutomaticErrorRecovery. Mirrors _finish_motion.
            #
            # This is unconditional: it is the modes libfranka watches to learn
            # the motion is over (``throwOnMotionError``/``cancelMotion`` in
            # ``src/robot_impl.cpp`` both key off them), and the publish loop
            # carries them out on its own next cycle. It used to sit inside the
            # "do we know the peer?" guard below, so a StopMove that arrived
            # before the state thread had recorded the peer left the stream
            # reporting kMove for the rest of the session.
            if self.robot_state.state["robot_mode"] != RobotMode.kReflex:
                self.robot_state.state["motion_generator_mode"] = 0  # kNone
                self.robot_state.state["controller_mode"] = 3  # kOther
                self.robot_state.state["robot_mode"] = RobotMode.kIdle

            # Then hand the client one state carrying them straight away rather
            # than waiting for the loop's next cycle. Snapshot the peer at the
            # point of use: it is set by the state thread when transmission
            # starts and cleared by reset_state on disconnect, and this handler
            # runs on neither of those threads -- so reading it once, here,
            # avoids both a half-updated target and a value read before the
            # state thread had one.
            peer = (self.client_address, self.client_udp_port)
            if hasattr(self, "udp_socket") and self.udp_socket and None not in peer:
                # Send state with new message ID
                self.robot_state.update()  # This increments message_id
                final_state = self.robot_state.pack_state()
                self.udp_socket.sendto(final_state, peer)
                logger.info(f"Sent final robot state with message_id:\
                          {self.robot_state.state['message_id']}")

            # StopMove ends the *motion*, not the session: the real FCI keeps
            # streaming RobotState over UDP (now reporting the idle hold set
            # above) from Connect all the way to disconnect, and a client is
            # free to send another Move on this same TCP connection --
            # libfranka's ActiveControl does exactly that on
            # cancelMotion() -> StopMove -> AutomaticErrorRecovery -> Move.
            # ``transmitting_state`` is the flag start_robot_state_transmission's
            # publish loop reads to keep going (see its docstring), and
            # ``connection_running`` gates that same loop *and* handle_client's
            # watchdog that ends the whole connection -- clearing either one
            # here used to tear the session down after the first motion, so a
            # second Move's kMotionStarted would arrive over TCP (the command
            # thread does not check either flag) but never see another UDP
            # datagram, and the client's next receive would hang until it
            # timed out. Neither flag is touched here any more; both are
            # cleared only by an actual disconnect (handle_tcp_messages) or
            # server shutdown (stop()). This mirrors the already-established
            # pattern for a motion that finishes on its own (see the
            # motion-finished handler above, which has never stopped the
            # publish loop either).

            # Terminate the interrupted motion's Move with ``kPreempted``, the
            # status the robot itself sends when a StopMove cuts a Move short.
            # libfranka's own model of the robot pins this: its cancelMotion
            # tests (``test/robot_impl_tests.cpp``, CanCancelMotion and
            # CancelMotionErrorThrowsControlException) answer StopMove by
            # sending ``Move::Response(Move::Status::kPreempted)`` for the
            # running motion id before the StopMove response itself.
            #
            # The status is not cosmetic. ``Robot::stop()`` is called from a
            # different thread than the running control loop, so it races that
            # loop: the loop's ``throwOnMotionError`` (``src/robot_impl.cpp``)
            # sees the idle modes in the state below, reads *this* response and
            # runs it through ``handleCommandResponse<Move>``
            # (``src/robot_impl.h``). kPreempted throws CommandException there,
            # which becomes the ControlException ("Move command preempted!")
            # that a stopped motion is supposed to raise out of the control
            # thread. kSuccess throws nothing, so the code right after it --
            # ``throw ProtocolException("Unexpected reply to a Move command")``
            # -- fires instead, which is what clients used to see here.
            #
            # A conforming client answers that ControlException by calling
            # ``cancelMotion`` (``ControlLoop::loop``'s catch-all), i.e. a
            # *second* StopMove on the same connection. It arrives with no
            # motion running, gets its own kSuccess below, and no Move response
            # (``current_motion_id`` is already 0) -- and its
            # ``do { receiveRobotState(); } while (...)`` needs the publish
            # loop to still be running, which is why nothing here stops it.
            with self._motion_lock:
                if self.current_motion_id:
                    # Create a Move response header
                    move_response_header = MessageHeader(Command.kMove, self.current_motion_id, 16)
                    move_header_bytes = move_response_header.to_bytes()
                    move_response_data = struct.pack("<B3x", MoveStatus.kPreempted.value)
                    self._send_tcp(client_socket, move_header_bytes + move_response_data)
                    logger.info(
                        "Sent Move kPreempted response for motion ID: %s", self.current_motion_id
                    )
                    self.current_motion_id = 0

        except Exception as e:
            logger.error(f"Error handling StopMove command: {e}")
            if responded:
                # The command already has its response on the wire; a second
                # one would be read by the client as the answer to whatever
                # it sends next (the Move response it is waiting for).
                return
            # Send error response
            total_size = 12 + 4
            response_header = MessageHeader(Command.kStopMove, header.command_id, total_size)
            header_bytes = response_header.to_bytes()
            response_data = struct.pack("<B3x", 5)  # Status 5 = Aborted
            self._send_tcp(client_socket, header_bytes + response_data)
