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


def validate_grasp_width(width: float, max_width: float) -> None:
    """Raise ``ValueError`` if ``width`` is not a width the hand can be asked for.

    A grasp width outside the 0..``max_width`` stroke is not a grasp that
    missed, it is a command the hand cannot execute. libfranka distinguishes
    the two: ``executeCommand`` (``src/gripper.cpp``) turns kUnsuccessful into
    a quiet ``false`` and kFail into ``CommandException``. The server turns the
    exception raised here into kFail; a grasp that ran and simply found nothing
    returns ``False`` instead.
    """
    if not 0.0 <= width <= max_width:
        raise ValueError(f"grasp width {width} m is outside the 0..{max_width} m stroke")


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
        """Grasp at ``width`` (m); return True only if an object was grasped.

        A grasp is not a move to ``width``: the real Franka Hand closes the
        fingers *under force* until they stall -- on an object, or on each
        other at ~0 m -- and only then judges the result. The judgement is the
        one ``franka::Gripper::grasp`` documents (``include/franka/gripper.h``):
        "An object is considered grasped if the distance d between the gripper
        fingers satisfies (width - epsilon_inner) < d < (width +
        epsilon_outer)", the band being centred on the width the *client* asked
        for.

        So with nothing between the fingers they close to ~0, which is outside
        the band of any non-zero commanded width, and the call returns False --
        "True if an object has been grasped, false otherwise", which is what
        libfranka's own ``examples/grasp_object.cpp`` branches on.

        Implementations raise ``ValueError`` for a width outside the hand's
        stroke (see :func:`validate_grasp_width`); the server answers that with
        kFail rather than the kUnsuccessful of a missed grasp.
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
    unit-testable. An optional ``object_width`` stands in for something
    between the fingers: a grasp closes toward 0 and stops there instead,
    which is what the physics backend gets from an actual stall. Whether the
    resulting grasp *succeeded* is then the same epsilon check the real hand
    applies (see :meth:`GripperBackend.grasp`). Timed/physical motion can
    replace the internals later without changing the interface.
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
        """Close the fingers until they stall, then apply the epsilon band.

        Modelled on the real hand (see :meth:`GripperBackend.grasp`): the
        fingers close from where they are toward 0 and stop at the first thing
        they meet -- a configured ``object_width`` narrower than the current
        opening, or each other. ``speed`` is not simulated (the stub has no
        clock), and ``force`` is not applied as a limit.

        An object at least as wide as the current opening cannot be between
        the fingers -- they are already inside it -- so it does not stall
        them; without that guard a stale ``self.width`` would report a grasp
        with the fingers narrower than the object they supposedly hold.
        """
        validate_grasp_width(width, self.max_width)
        if self.object_width is not None and self.object_width < self.width:
            final = max(self.object_width, 0.0)
        else:
            final = 0.0
        self.width = final
        self.is_grasped = width - epsilon_inner < final < width + epsilon_outer
        return self.is_grasped

    def stop(self) -> bool:
        self.width = self.max_width
        self.is_grasped = False
        return True

    def get_state(self) -> GripperStateData:
        return GripperStateData(self.width, self.max_width, self.is_grasped, self.temperature)
