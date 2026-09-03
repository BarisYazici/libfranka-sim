"""Physics-backed gripper backend that drives a scene's shared finger DOFs."""

import logging
import time
from typing import Optional

import numpy as np

from franka_sim.gripper.backend import (
    DEFAULT_TEMPERATURE,
    FRANKA_HAND_MAX_WIDTH,
    GripperBackend,
    GripperStateData,
    validate_width,
)

logger = logging.getLogger(__name__)


class FrankaHandPhysics(GripperBackend):
    """Physics-backed Franka Hand: drives the finger DOFs of a loaded sim scene.

    Engine-agnostic: it only calls ``update_finger_positions``/``get_finger_state``
    on the sim handed to it, so it works unchanged against
    :class:`~franka_sim.mujoco_franka_sim.MujocoFrankaSim` and
    :class:`~franka_sim.franka_genesis_sim.FrankaGenesisSim` alike.

    Command methods run on the gripper server's TCP handler thread and BLOCK,
    polling the sim's finger snapshot until the fingers settle (velocity ~0) or a
    timeout elapses -- matching the real franka::Gripper blocking semantics.
    ``get_state`` only reads the lock-free snapshot, so it is safe to call from
    the UDP broadcaster thread concurrently. ``is_grasped``/``is_stuck`` come from
    a finger-position stall -- a grasp closes the fingers all the way and sees
    where they were stopped -- so no engine contact-force API is needed.

    ``object_width`` (``--gripper-object-width``) puts a rigid virtual object
    of that width between the fingers, for scenes that contain nothing
    graspable; see :meth:`_blocked_target`.
    """

    def __init__(
        self,
        physics_sim,
        max_width: float = FRANKA_HAND_MAX_WIDTH,
        temperature: int = DEFAULT_TEMPERATURE,
        settle_timeout: float = 4.0,
        settle_velocity: float = 1e-3,
        poll_dt: float = 0.01,
        stall_polls: int = 3,
        object_width: Optional[float] = None,
    ):
        self.sim = physics_sim
        self.max_width = max_width
        self.temperature = temperature
        self.settle_timeout = settle_timeout
        self.settle_velocity = settle_velocity
        self.poll_dt = poll_dt
        self.stall_polls = stall_polls
        self.object_width = object_width
        self.is_grasped = False
        self.is_stuck = False

    # -- helpers ------------------------------------------------------------
    def _clamp_width(self, width: float) -> float:
        return max(0.0, min(self.max_width, width))

    def _command_width(self, width: float) -> None:
        half = self._clamp_width(width) / 2.0
        self.sim.update_finger_positions([half, half])

    def _current_width(self) -> float:
        q = self.sim.get_finger_state()["q"]
        return float(q[0] + q[1])

    def _blocked_target(self, target: float) -> float:
        """Where a virtual object stops the fingers short of ``target``.

        ``object_width`` (``--gripper-object-width``) is a rigid obstacle the
        scene does not contain: the sim's default scene has nothing graspable
        in it, so without this no grasp could ever succeed and every
        ``franka_gripper`` Grasp action would answer false. Clamping the drive
        target here stalls the fingers at ``object_width`` exactly as a body
        between them would, which is what :meth:`_drive_and_settle` then reads
        back -- no contact-force API, and no scene edit, needed.

        An object wider than the current opening is not between the fingers
        (they are already inside it) and does not block them; the same guard,
        and the same ``<=``, as :meth:`FrankaHandSim.grasp
        <franka_sim.gripper.backend.FrankaHandSim.grasp>`, so the two backends
        answer a repeated grasp on a held object identically.
        """
        if self.object_width is None:
            return target
        obstacle = self._clamp_width(self.object_width)
        if obstacle <= self._current_width():
            return max(target, obstacle)
        return target

    def _drive_and_settle(self, width: float):
        """Command width, block until fingers settle or timeout. Returns
        (final_width, settled_before_timeout).

        The first snapshot after a new command may still be from the previous
        settled target (dq ~= 0). Treat low velocity as settled only after the
        fingers either reached the target or have visibly moved from the initial
        width, which covers both free motion and object-stall grasping.

        Settled means ``stall_polls`` consecutive near-zero-velocity polls,
        always -- never a single quiet one, however plausible it looks. The
        first snapshot after a new command may still be from the previous
        settled target (dq ~= 0), and closing fingers pass through momentary
        near-zero velocities of their own on the way in (contact settling in
        MuJoCo does exactly that). A grasp accepted the first of those as the
        stall and judged the band against a width the fingers were still
        travelling through: a trace resting at 0.028 m answered a
        ``grasp(0.028)`` False from a mid-motion sample. Requiring the run
        unconditionally costs ``(stall_polls - 1) * poll_dt`` (~20 ms) on every
        command and removes the whole class of early reads; fingers already
        blocked when the command arrives -- a second grasp on an object still
        held from the first -- settle by the same run rather than blocking for
        the whole ``settle_timeout`` and reporting themselves stuck.

        The target is the one a virtual ``object_width`` leaves reachable
        (:meth:`_blocked_target`), so the fingers stall on it exactly as they
        would on a body in the scene.
        """
        target = self._blocked_target(self._clamp_width(width))
        self._command_width(target)
        if self.poll_dt:
            time.sleep(self.poll_dt)
        deadline = time.monotonic() + self.settle_timeout
        still = 0
        while time.monotonic() < deadline:
            fs = self.sim.get_finger_state()
            current = float(fs["q"][0] + fs["q"][1])
            speed = float(np.max(np.abs(fs["dq"])))
            still = still + 1 if speed < self.settle_velocity else 0
            if still >= self.stall_polls:
                return current, True
            if self.poll_dt:
                time.sleep(self.poll_dt)
        return self._current_width(), False

    # -- GripperBackend -----------------------------------------------------
    def homing(self) -> bool:
        """Drive the fingers fully open and block until they settle.

        Always reports success (real homing calibration has no sim equivalent
        here) and clears any prior grasp/stuck state.
        """
        self._drive_and_settle(self.max_width)
        self.is_grasped = False
        self.is_stuck = False
        return True

    def move(self, width: float, speed: float) -> bool:
        """Drive the fingers to ``width`` and block until they settle or time out.

        ``speed`` is accepted for protocol compatibility but not used to pace
        the motion. Returns True; a width outside the stroke raises
        ``ValueError`` -> kFail (see :meth:`GripperBackend.move`). ``is_stuck``
        is set if the fingers never settled within the timeout.
        """
        validate_width(width, self.max_width, "move")
        final, settled = self._drive_and_settle(width)
        self.is_grasped = False
        self.is_stuck = not settled
        return True

    def grasp(
        self,
        width: float,
        epsilon_inner: float,
        epsilon_outer: float,
        speed: float,
        force: float,
    ) -> bool:
        """Close the fingers until they stall, then apply the epsilon band.

        The real hand does not *move* to ``width``: it closes under force until
        the fingers stall, on an object or on each other, and then judges the
        stall width against the band ``franka::Gripper::grasp`` documents
        (``include/franka/gripper.h``): ``width - epsilon_inner < d < width +
        epsilon_outer``, around the width the client asked for. So this drives
        the fingers fully closed and reads where the object stopped them --
        free space stops them at ~0, which is outside the band of any
        meaningful commanded width and correctly returns False.

        Position-stall stands in for force sensing: ``speed``/``force`` are
        accepted for protocol compatibility but not used (no contact-force API
        here). Driving to 0 saturates the sim's finger servo at
        :data:`~franka_sim.mujoco_franka_sim.FINGER_FORCE_LIMIT` whatever
        ``force`` the client asked for, so the width read back is the
        *compressed* one -- where a servo pushing at full force stopped -- and
        not the width that force would have settled at. A ``width`` outside the
        stroke raises ``ValueError`` -> kFail.
        """
        validate_width(width, self.max_width)
        final, settled = self._drive_and_settle(0.0)
        grasped = width - epsilon_inner < final < width + epsilon_outer
        self.is_grasped = bool(grasped)
        # Stuck = caught nothing and never settled (still pushing at the
        # timeout) rather than having come to rest somewhere wrong.
        self.is_stuck = (not grasped) and (not settled)
        if not grasped:
            logger.info(
                f"grasp: fingers stalled at {final:.4f} m, outside the band "
                f"({width - epsilon_inner:.4f}, {width + epsilon_outer:.4f}) m"
            )
        return self.is_grasped

    def stop(self) -> bool:
        """Stop any in-progress motion by driving the fingers fully open.

        Always returns True and clears any prior grasp/stuck state.
        """
        self._drive_and_settle(self.max_width)
        self.is_grasped = False
        self.is_stuck = False
        return True

    def get_state(self) -> GripperStateData:
        """Return the current width, max width, grasp flag, and temperature.

        Reads only the lock-free finger snapshot, so this is safe to call
        concurrently with the sim's physics thread (e.g. from the UDP
        broadcaster thread).
        """
        return GripperStateData(
            self._current_width(), self.max_width, self.is_grasped, self.temperature
        )
