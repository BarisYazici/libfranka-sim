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
    #: ``kCartesianPosition`` on an arm role: the client streams ``O_T_EE_c``
    #: (and optionally ``elbow_c``) and the backend converts it to joint motion
    #: with differential IK (:mod:`franka_sim.cartesian_ik`), feeding the result
    #: into the same velocity servo :attr:`VELOCITY` drives.
    #:
    #: A backend that implements no IK is simply never put into this mode: the
    #: server asks once whether it can (``FrankaSimServer.cartesian_tracking``)
    #: and leaves the arm inert if not. Either way the commanded stream is
    #: differentiated and judged exactly as the robot judges it, so a client
    #: that steps its pose gets the hardware error instead of silence. See
    #: :meth:`franka_sim.motion_limits.MotionLimitChecker._check_cartesian_pose`.
    CARTESIAN_POSE = "cartesian_pose"
    #: ``kCartesianVelocity`` on an *arm* role: the client streams the EE twist
    #: ``O_dP_EE_c``, resolved to joint velocity the same way. Not to be confused
    #: with the mobile base's twist generator, which commands a *base* velocity
    #: for the swerve kinematics and is :attr:`STEERING_DRIVE`.
    CARTESIAN_VELOCITY = "cartesian_velocity"
    NONE = "none"
