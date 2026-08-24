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
    #: ``kCartesianPosition`` on an arm role. **Checking-only.** No physics
    #: backend is ever put into this mode -- the arm stays inert on the pose
    #: interface -- but the commanded ``O_T_EE_c``/``elbow_c`` stream is
    #: differentiated and judged exactly as the robot judges it, so a client
    #: that steps its pose gets the hardware error instead of silence. See
    #: :meth:`franka_sim.motion_limits.MotionLimitChecker._check_cartesian_pose`.
    CARTESIAN_POSE = "cartesian_pose"
    #: ``kCartesianVelocity`` on an *arm* role. Checking-only for the same
    #: reason; the mobile base's own twist generator is
    #: :attr:`STEERING_DRIVE`, which is driven for real.
    CARTESIAN_VELOCITY = "cartesian_velocity"
    NONE = "none"
