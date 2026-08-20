"""The control mode every simulator backend and the FCI server agree on.

Kept in its own module so importing it costs nothing: ``franka_sim_server`` and
the MuJoCo backends need the enum, and pulling it from ``franka_genesis_sim``
used to drag the (multi-second, native) ``genesis`` import in with it. Genesis
is optional now -- only ``--physics genesis`` imports it -- so the enum lives
here and ``franka_genesis_sim`` re-exports it for existing callers.
"""

from enum import Enum


class ControlMode(Enum):
    """Motion-generator mode the sim's physics loop is currently servoing."""

    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    STEERING_DRIVE = "steering_drive"  # mobile base: steering=position, drive=velocity
    NONE = "none"
