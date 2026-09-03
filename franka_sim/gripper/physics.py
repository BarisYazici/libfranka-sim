"""Physics-backed gripper backend that drives a scene's shared finger DOFs."""

import logging
import time

import numpy as np

from franka_sim.gripper.backend import (
    DEFAULT_TEMPERATURE,
    FRANKA_HAND_MAX_WIDTH,
    GripperBackend,
    GripperStateData,
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
    the UDP broadcaster thread concurrently. ``is_grasped``/``is_stuck`` are read
    off where the fingers came to rest -- the epsilon band of
    ``franka::Gripper::grasp`` -- so no engine contact-force API is needed.
    """

    def __init__(
        self,
        physics_sim,
        max_width: float = FRANKA_HAND_MAX_WIDTH,
        temperature: int = DEFAULT_TEMPERATURE,
        settle_timeout: float = 4.0,
        settle_velocity: float = 1e-3,
        poll_dt: float = 0.01,
    ):
        self.sim = physics_sim
        self.max_width = max_width
        self.temperature = temperature
        self.settle_timeout = settle_timeout
        self.settle_velocity = settle_velocity
        self.poll_dt = poll_dt
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

    def _drive_and_settle(self, width: float):
        """Command width, block until fingers settle or timeout. Returns
        (final_width, settled_before_timeout).

        The first snapshot after a new command may still be from the previous
        settled target (dq ~= 0). Treat low velocity as settled only after the
        fingers either reached the target or have visibly moved from the initial
        width, which covers both free motion and object-stall grasping.
        """
        target = self._clamp_width(width)
        initial = self._current_width()
        moved = abs(initial - target) < 1e-4
        self._command_width(width)
        if self.poll_dt:
            time.sleep(self.poll_dt)
        deadline = time.monotonic() + self.settle_timeout
        while time.monotonic() < deadline:
            fs = self.sim.get_finger_state()
            current = float(fs["q"][0] + fs["q"][1])
            speed = float(np.max(np.abs(fs["dq"])))
            at_target = abs(current - target) < 1e-4
            moved = moved or abs(current - initial) > 1e-4
            if speed < self.settle_velocity and (at_target or moved):
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
        the motion. Always returns True; ``is_stuck`` is set if the fingers
        never settled within the timeout.
        """
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
        """Close the fingers toward ``width`` and report the epsilon check.

        Success is where the fingers came to rest, exactly as
        ``franka::Gripper::grasp`` documents it (``include/franka/gripper.h``):
        the final distance ``d`` must satisfy ``width - epsilon_inner < d <
        width + epsilon_outer``. That covers both outcomes the real hand has:
        an object stalls the fingers wider than commanded (in band if it is
        close enough to the commanded width), and free space lets them reach
        the commanded width itself -- which is a success, not a failure.

        The band is around the width the client asked for, *not* the
        stroke-clamped one, so a grasp commanded wider than the hand can open
        fails instead of trivially matching its own clamp.
        ``speed``/``force`` are accepted for protocol compatibility but not
        used (no contact-force API here).
        """
        final, settled = self._drive_and_settle(width)
        grasped = width - epsilon_inner < final < width + epsilon_outer
        self.is_grasped = bool(grasped)
        # Stuck = ended outside the band and never settled (still pushing at
        # the timeout) rather than having come to rest somewhere wrong.
        self.is_stuck = (not grasped) and (not settled)
        if not grasped:
            logger.info(
                f"grasp: fingers settled at {final:.4f} m, outside the band "
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
