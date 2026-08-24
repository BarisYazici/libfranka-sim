"""Engine-agnostic pieces shared by the mobile-duo Genesis and MuJoCo scenes.

Everything here is pure Python/NumPy plus :mod:`franka_sim.control_modes` and
:mod:`franka_sim.sim_common`: no ``genesis`` and no ``mujoco`` import, so the
MuJoCo mobile-duo backend (and anything that only needs to talk to a scene
through :class:`SceneView`, like the runner) can import this module without
paying Genesis' multi-second native import cost -- or requiring it installed
at all.

:mod:`franka_sim.mobile.duo_sim` re-exports the names it used to define, so
every existing ``from franka_sim.mobile.duo_sim import X`` keeps working.
"""

import logging
import math
from typing import Dict, List, Sequence

import numpy as np

from franka_sim.control_modes import ControlMode

logger = logging.getLogger(__name__)

ROLE_LEFT = "left"
ROLE_RIGHT = "right"
ROLE_BASE = "base"
ARM_ROLES = (ROLE_LEFT, ROLE_RIGHT)
ROLES = (ROLE_LEFT, ROLE_RIGHT, ROLE_BASE)

#: Arm joints in the combined URDF, generated with
#: ``robot_types:="['tmrv0_2','fr3v2','fr3v2']"`` and arm prefixes left/right.
ARM_JOINT_NAMES: Dict[str, List[str]] = {
    ROLE_LEFT: [f"left_fr3v2_joint{i}" for i in range(1, 8)],
    ROLE_RIGHT: [f"right_fr3v2_joint{i}" for i in range(1, 8)],
}

#: Flange link per arm (hand:=false, so link7 is the attachment frame).
ARM_EE_LINKS: Dict[str, str] = {
    ROLE_LEFT: "left_fr3v2_link7",
    ROLE_RIGHT: "right_fr3v2_link7",
}

#: Initial arm pose, matching the initial_value parameters in upstream's
#: mobile_fr3_duo.ros2_control.xacro.
ARM_INITIAL_Q = np.array([0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4])

#: Prismatic lift joint carrying the mount, the head and both arms.
SPINE_JOINT_NAME = "franka_spine_vertical_joint"

#: Travel limits of that joint in the combined URDF (metres).
SPINE_LIMITS_M = (0.0, 0.85)


class SceneView:
    """One role's view of a shared mobile-duo scene, shaped like a simulator.

    Implements the interface ``FrankaSimServer`` calls on ``physics_sim``.
    ``initialize_simulation``/``start``/``stop`` are no-ops because the runner
    owns the shared scene's lifecycle. Engine-agnostic: it only calls methods
    on ``scene`` (either :class:`~franka_sim.mobile.duo_sim.MobileDuoScene` or
    :class:`~franka_sim.mobile.duo_mujoco_sim.MobileDuoMujocoScene`), so it
    lives here rather than in either engine-specific module.
    """

    def __init__(self, scene, role: str):
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}; expected one of {ROLES}")
        self.scene = scene
        self.role = role
        # Latch: update_base_twist can be reached from the ~1 kHz UDP thread, so
        # the misroute warning must fire once, not once per datagram.
        self._twist_misroute_logged = False

    @property
    def enable_vis(self) -> bool:
        """Mirror the shared scene's visualisation flag."""
        return self.scene.enable_vis

    def initialize_simulation(self) -> None:
        """No-op: the runner builds the shared scene exactly once."""

    def start(self) -> None:
        """No-op: the runner owns the single physics loop."""

    def stop(self) -> None:
        """No-op: the runner stops the shared scene."""

    def set_control_mode(self, mode: ControlMode) -> None:
        """Set this arm's control mode; ignored for the base (twist-driven)."""
        if self.role == ROLE_BASE:
            if not isinstance(mode, ControlMode):
                raise ValueError(f"Mode must be a ControlMode enum, got {type(mode)}")
            return
        self.scene.set_arm_control_mode(self.role, mode)

    def update_joint_positions(self, positions) -> None:
        """Publish this arm's joint positions; ignored for the base."""
        if self.role == ROLE_BASE:
            return
        self.scene.update_arm_joint_positions(self.role, positions)

    def update_joint_velocities(self, velocities) -> None:
        """Publish this arm's joint velocities; ignored for the base."""
        if self.role == ROLE_BASE:
            return
        self.scene.update_arm_joint_velocities(self.role, velocities)

    def update_torques(self, torques) -> None:
        """Publish this arm's torques; ignored for the base."""
        if self.role == ROLE_BASE:
            return
        self.scene.update_arm_torques(self.role, torques)

    def update_base_twist(self, twist: Sequence[float]) -> None:
        """Publish the base twist. Dropped (warned once) on an arm view."""
        if self.role != ROLE_BASE:
            if not self._twist_misroute_logged:
                logger.warning("Base twist ignored on arm view %r", self.role)
                self._twist_misroute_logged = True
            return
        self.scene.update_base_twist(twist)

    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """This role's latest state snapshot."""
        return self.scene.get_role_state(self.role)
