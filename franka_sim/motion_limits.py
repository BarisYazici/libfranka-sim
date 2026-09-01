r"""FCI motion-limit and discontinuity checking for commanded signals.

The implementation now lives in :mod:`franka_sim.limits`:

* :mod:`franka_sim.limits.tables` -- the libfranka limit tables, the error
  indices they latch and the sim-side tolerances.
* :mod:`franka_sim.limits.differencing` -- the backward-Euler differentiators
  and the pose math the Cartesian ones need.
* :mod:`franka_sim.limits.checker` -- :class:`MotionLimitChecker` and the
  :class:`Violation` / :class:`AbsorbedCommand` it reports through.

This module stays the supported import path for the names callers actually take
from it, re-exported below. It is not a mirror of the packages: a name that is
only used inside :mod:`franka_sim.limits` should be imported from the module
that defines it. See the package modules for the full documentation of each
constant and the invariants it participates in.
"""

import logging

# Pinned to this module's own name so that callers capturing
# "franka_sim.motion_limits" keep seeing the records the split modules emit.
logger = logging.getLogger("franka_sim.motion_limits")

from franka_sim.limits.tables import (  # noqa: E402
    CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
    CARTESIAN_MOTION_GENERATOR_ELBOW_LIMIT_VIOLATION_INDEX,
    CARTESIAN_MOTION_GENERATOR_ELBOW_SIGN_INCONSISTENT_INDEX,
    CARTESIAN_MOTION_GENERATOR_START_ELBOW_INVALID_INDEX,
    CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
    CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
    CARTESIAN_POSITION_MOTION_GENERATOR_INVALID_FRAME_INDEX,
    CARTESIAN_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX,
    CARTESIAN_VELOCITY_VIOLATION_INDEX,
    CONTROLLER_TORQUE_DISCONTINUITY_INDEX,
    DELTA_T,
    ELBOW_POSITION_LIMITS,
    ENFORCE_ENV_VAR,
    ERROR_NAMES,
    JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
    JOINT_MOTION_GENERATOR_POSITION_LIMITS_VIOLATION_INDEX,
    JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
    JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
    JOINT_POSITION_LIMITS,
    JOINT_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX,
    JOINT_VELOCITY_VIOLATION_INDEX,
    MAX_COALESCED_CYCLES,
    MAX_ELBOW_ACCELERATION,
    MAX_ELBOW_VELOCITY,
    MAX_JOINT_ACCELERATION,
    MAX_JOINT_JERK,
    MAX_ROTATIONAL_VELOCITY,
    MAX_TORQUE,
    MAX_TORQUE_RATE,
    MAX_TRANSLATIONAL_ACCELERATION,
    MAX_TRANSLATIONAL_JERK,
    MAX_TRANSLATIONAL_VELOCITY,
    MEASURED_CARTESIAN_VELOCITY_LIMIT,
    MEASURED_JOINT_VELOCITY_MARGIN,
    SELF_COLLISION_AVOIDANCE_VIOLATION_INDEX,
    SELF_COLLISION_CLOSING_DISTANCE,
    SINGULAR_POSE_MIN_SINGULAR_VALUE,
    START_CARTESIAN_POSE_ROTATION_TOLERANCE,
    START_CARTESIAN_POSE_TRANSLATION_TOLERANCE,
    START_ELBOW_TOLERANCE,
    TAU_J_RANGE_VIOLATION_INDEX,
    enforcement_enabled_by_env,
    is_singular_configuration,
    lower_joint_velocity_limits,
    smallest_singular_value,
    upper_joint_velocity_limits,
)
from franka_sim.limits.differencing import (  # noqa: E402
    _Differentiator,
    is_homogeneous_transformation,
    rotation_exp,
    rotation_log,
    transform_matrix,
)
from franka_sim.limits.checker import (  # noqa: E402
    MotionLimitChecker,
    Violation,
)
