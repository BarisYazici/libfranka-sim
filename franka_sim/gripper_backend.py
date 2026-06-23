import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Franka Hand maximum stroke (m) between the two fingers.
FRANKA_HAND_MAX_WIDTH = 0.08
# Constant temperature (deg C) reported in GripperState.
DEFAULT_TEMPERATURE = 30


@dataclass
class GripperStateData:
    width: float
    max_width: float
    is_grasped: bool
    temperature: int


class GripperBackend(ABC):
    """Swappable gripper backend.

    The hardcoded libfranka wire client (``franka::Gripper``) only ever talks
    to ``FrankaGripperServer``, which translates its fixed protocol into these
    five calls. A different gripper (e.g. Robotiq) implements the same contract
    and is free to additionally expose its own richer, non-libfranka transport
    -- the server-facing surface stays identical.
    """

    @abstractmethod
    def homing(self) -> bool:
        """Home the gripper; return True on success."""

    @abstractmethod
    def move(self, width: float, speed: float) -> bool:
        """Move fingers to ``width`` (m) at ``speed`` (m/s); return True on
        success.
        """

    @abstractmethod
    def grasp(
        self,
        width: float,
        epsilon_inner: float,
        epsilon_outer: float,
        speed: float,
        force: float,
    ) -> bool:
        """Grasp at ``width`` (m); return True only if an object was
        grasped.
        """

    @abstractmethod
    def stop(self) -> bool:
        """Abort the current motion; return True on success."""

    @abstractmethod
    def get_state(self) -> GripperStateData:
        """Return the current gripper state. May be called from the server's UDP
        broadcaster thread concurrently with command methods, so implementations
        must be thread-safe.
        """


class FrankaHandSim(GripperBackend):
    """Kinematic Franka Hand model (no physics, no threads, no sleeps).

    Width updates are instant: a command sets the final width and returns,
    which is enough to be wire-compatible with ``franka::Gripper`` and fully
    unit-testable. ``grasp`` succeeds only if a configured ``object_width``
    lies within the commanded width's epsilon band -- the same notion of "did
    the fingers catch something" the real hand reports. Timed/physical motion
    can replace the internals later without changing the interface.
    """

    def __init__(
        self,
        max_width: float = FRANKA_HAND_MAX_WIDTH,
        temperature: int = DEFAULT_TEMPERATURE,
        object_width: Optional[float] = None,
    ):
        self.max_width = max_width
        self.temperature = temperature
        self.object_width = object_width
        self.width = max_width  # start fully open
        self.is_grasped = False

    def set_object_width(self, width: Optional[float]) -> None:
        """Configure a graspable object width (m), or None for free space."""
        self.object_width = width

    def _clamp(self, width: float) -> float:
        return max(0.0, min(self.max_width, width))

    def homing(self) -> bool:
        self.width = self.max_width
        self.is_grasped = False
        return True

    def move(self, width: float, speed: float) -> bool:
        self.width = self._clamp(width)
        self.is_grasped = False
        return True

    def grasp(
        self,
        width: float,
        epsilon_inner: float,
        epsilon_outer: float,
        speed: float,
        force: float,
    ) -> bool:
        if self.object_width is not None and (
            width - epsilon_inner <= self.object_width <= width + epsilon_outer
        ):
            self.width = self.object_width
            self.is_grasped = True
            return True
        self.width = self._clamp(width)
        self.is_grasped = False
        return False

    def stop(self) -> bool:
        self.width = self.max_width
        self.is_grasped = False
        return True

    def get_state(self) -> GripperStateData:
        return GripperStateData(self.width, self.max_width, self.is_grasped, self.temperature)
