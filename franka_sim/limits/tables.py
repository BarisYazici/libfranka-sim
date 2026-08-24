"""FCI limit tables, error indices and the sim-side tolerances built on them.

Every number here is a constant lifted from libfranka's client-side
implementation or derived from observed FR3 hardware behaviour, together with
the small helpers that read straight off those tables
(the position-based joint-velocity envelope, the singular-configuration test
and the enforcement environment switch).

Nothing in this module holds per-session state; the invariants these values
participate in are documented on each constant and enforced by
:class:`franka_sim.limits.checker.MotionLimitChecker`.
"""

import logging
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# The logger name is pinned to the pre-split module so that callers (and tests)
# that capture ``franka_sim.motion_limits`` keep seeing these records.
logger = logging.getLogger("franka_sim.motion_limits")

# -- libfranka constants ------------------------------------------------------

#: Control cycle. ``franka::kDeltaT`` (``include/franka/rate_limiting.h:20``).
DELTA_T = 1e-3

#: ``franka::kLimitEps`` -- "Epsilon value for checking limits"
#: (``include/franka/rate_limiting.h:24``). Every published limit below already
#: has it subtracted, exactly as libfranka's constants do.
LIMIT_EPS = 1e-3

#: ``franka::kNormEps`` = ``std::numeric_limits<double>::epsilon()``
#: (``include/franka/rate_limiting.h:28``). Below this a Cartesian norm is not
#: worth limiting, which is how ``limitRate`` in ``src/rate_limiting.cpp:26``
#: and ``:48`` guards its divisions. Carried for symmetry with libfranka; the
#: comparison it appears in here can never decide anything, since every limit is
#: many orders of magnitude larger.
NORM_EPS = sys.float_info.epsilon

#: ``franka::kTolNumberPacketsLost`` (``include/franka/rate_limiting.h:35``):
#: "For FR3 there are no expected package loses. Therefore this number is set
#: to 0." It widens the velocity limits by the distance a packet-loss
#: extrapolation could cover; zero, so it does not.
TOL_NUMBER_PACKETS_LOST = 0.0

#: ``franka::kMaxJointAcceleration`` (``include/franka/rate_limiting.h:53``):
#: 10 rad/s^2 per joint, less :data:`LIMIT_EPS`.
MAX_JOINT_ACCELERATION = tuple(10.0 - LIMIT_EPS for _ in range(7))

#: ``franka::kMaxJointJerk`` (``include/franka/rate_limiting.h:47``):
#: 5000 rad/s^3 per joint, less :data:`LIMIT_EPS`.
MAX_JOINT_JERK = tuple(5000.0 - LIMIT_EPS for _ in range(7))

#: ``franka::kJointVelocityLimitsTolerance``
#: (``include/franka/rate_limiting.h:62``): "Tolerance value for joint velocity
#: limits to deal with numerical errors and data losses."
JOINT_VELOCITY_LIMITS_TOLERANCE = tuple(
    LIMIT_EPS + TOL_NUMBER_PACKETS_LOST * DELTA_T * MAX_JOINT_ACCELERATION[i] for i in range(7)
)

#: ``franka::kMaxTorqueRate`` (``include/franka/rate_limiting.h:44``):
#: 1000 Nm/s per joint, less :data:`LIMIT_EPS`.
MAX_TORQUE_RATE = tuple(1000.0 - LIMIT_EPS for _ in range(7))

#: ``franka::kMaxTranslationalVelocity`` / ``Acceleration`` / ``Jerk``
#: (``include/franka/rate_limiting.h:70``, ``:74``, ``:78``). The velocity
#: carries the packet-loss term, which is zero for the FR3.
MAX_TRANSLATIONAL_ACCELERATION = 9.0 - LIMIT_EPS
MAX_TRANSLATIONAL_JERK = 4500.0 - LIMIT_EPS
MAX_TRANSLATIONAL_VELOCITY = (
    3.0 - LIMIT_EPS - TOL_NUMBER_PACKETS_LOST * DELTA_T * MAX_TRANSLATIONAL_ACCELERATION
)

#: ``franka::kMaxRotationalVelocity`` / ``Acceleration`` / ``Jerk``
#: (``include/franka/rate_limiting.h:82``, ``:86``, ``:90``).
MAX_ROTATIONAL_ACCELERATION = 17.0 - LIMIT_EPS
MAX_ROTATIONAL_JERK = 8500.0 - LIMIT_EPS
MAX_ROTATIONAL_VELOCITY = (
    2.5 - LIMIT_EPS - TOL_NUMBER_PACKETS_LOST * DELTA_T * MAX_ROTATIONAL_ACCELERATION
)

#: ``franka::kMaxElbowVelocity`` / ``Acceleration`` / ``Jerk``
#: (``include/franka/rate_limiting.h:94``, ``:99``, ``:105``). ``elbow[0]`` is a
#: *position* (the redundancy angle, joint 3 on an FR3) on both Cartesian
#: interfaces, so its first difference is a velocity on either -- there is no
#: interface-relative shift for the elbow, and no elbow *discontinuity* name in
#: the enum: every one of the three lands on
#: :data:`CARTESIAN_MOTION_GENERATOR_ELBOW_LIMIT_VIOLATION_INDEX`.
MAX_ELBOW_ACCELERATION = 10.0 - LIMIT_EPS
MAX_ELBOW_JERK = 5000.0 - LIMIT_EPS
MAX_ELBOW_VELOCITY = (
    1.5 - LIMIT_EPS - TOL_NUMBER_PACKETS_LOST * DELTA_T * MAX_ELBOW_ACCELERATION
)

#: ``franka::kFactorCartesianRotationPoseInterface`` = 0.99
#: (``include/franka/rate_limiting.h:39``): "Factor for the definition of
#: rotational limits using the Cartesian Pose interface". libfranka applies it
#: *client-side*, inside its own optional rate limiter, and only to the three
#: rotational limits of the ``CartesianPose`` overload
#: (``src/rate_limiting.cpp:235-237``) -- it shrinks what the client will emit,
#: so the client stays a hair inside whatever Control checks.
#:
#: **Deliberately not applied here.** This module is the server side: it judges
#: what arrived, and nothing published says Control's own rotational bound on
#: the pose interface is 0.99x rather than the plain ``kMaxRotational*``.
#: Multiplying here would make the sim refuse a rotation libfranka itself
#: considers legal at the boundary. Recorded so the 1% is a decision rather
#: than an oversight.
FACTOR_CARTESIAN_ROTATION_POSE_INTERFACE = 0.99

# -- FR3 per-joint ranges -----------------------------------------------------
#
# libfranka publishes no joint position or torque range of its own: those are
# robot-model data, and v10 reads them out of the URDF (``JointVelocityLimitsConfig``
# parses the ``<limit>`` and ``<position_based_velocity_limits>`` elements,
# ``include/franka/joint_velocity_limits.h:105-116``). The numbers below are
# therefore taken from the very URDF this server hands the client over
# ``GetRobotModel`` -- ``franka_sim/models/fr3.urdf`` -- so the limits the sim
# enforces and the model the client builds cannot drift apart.
#
# They are also self-consistent with libfranka: the deprecated
# ``computeUpperLimitsJointVelocity`` / ``computeLowerLimitsJointVelocity``
# (``include/franka/rate_limiting.h:133-190``) hard-code exactly these numbers,
# with ``2 * deceleration_limit`` folded in (6.0 -> 12.0, 2.585 -> 5.17,
# 3.50 -> 7.00, 4.00 -> 8.00, 17.0 -> 34.0, 5.5 -> 11.0, 17.0 -> 34.0) and the
# same ``max_velocity`` caps (2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26).

#: ``(lower, upper)`` per joint, from ``<limit lower= upper=>`` in
#: ``franka_sim/models/fr3.urdf`` (joint1..joint7, lines 89, 131, 173, 215,
#: 257, 285, 313).
JOINT_POSITION_LIMITS = (
    (-2.750100, 2.750100),
    (-1.791800, 1.791800),
    (-2.906500, 2.906500),
    (-3.048100, -0.145800),
    (-2.810100, 2.810100),
    (0.540920, 4.520500),
    (-3.019600, 3.019600),
)

#: ``(max_velocity, velocity_offset, deceleration_limit)`` per joint, from
#: ``<limit velocity=>`` and ``<position_based_velocity_limits
#: velocity_offset= deceleration_limit=>`` in ``franka_sim/models/fr3.urdf``.
POSITION_BASED_VELOCITY_LIMITS = (
    (2.62, 0.30, 6.000),
    (2.62, 0.20, 2.585),
    (2.62, 0.20, 3.500),
    (2.62, 0.30, 4.000),
    (5.26, 0.35, 17.000),
    (4.18, 0.35, 5.500),
    (5.26, 0.35, 17.000),
)

#: Per-joint torque range, from ``<limit effort=>`` in
#: ``franka_sim/models/fr3.urdf``: 87 Nm for joints 1-4, 12 Nm for 5-7.
MAX_TORQUE = (87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0)

# -- error indices ------------------------------------------------------------
#
# Positions in the 41-entry ``errors`` / ``reflex_reason`` wire arrays are the
# 0-based enumerator values of ``research_interface::robot::Error``
# (``common/include/research_interface/robot/error.h:8-50``); ``franka::Errors``
# binds each named member to ``errors_[static_cast<size_t>(Error::k...)]``
# (``src/errors.cpp``). Line numbers below are that enum's declarations.

#: ``kJointVelocityViolation`` (``error.h:12``) ->
#: ``joint_velocity_violation`` (``error.h:108-109``; ``franka::Errors`` binds
#: it at ``src/errors.cpp:32``).
#:
#: **The safety controller's error, not the motion generator's.** Everything
#: else in this block is Control judging the signal the *client* sent; this one
#: is the robot watching what the arm is actually doing. It is the error real
#: hardware reports whenever *measured* joint velocity
#: leaves the position-based envelope, whatever the client was commanding --
#: including in pure torque control, where there is no commanded velocity to
#: judge at all (observed with a 3 Nm torque ramp on joint 6, which raises
#: ``joint_velocity_violation`` even though no velocity was ever commanded).
#: It also *outranks* the motion generator's own envelope check: ramping
#: ``dq_c`` past the envelope on real hardware answers
#: ``joint_velocity_violation`` rather than
#: ``joint_motion_generator_velocity_limits_violation`` -- the safety controller
#: sees the arm cross the envelope before Control finishes objecting to the
#: command that put it there.
JOINT_VELOCITY_VIOLATION_INDEX = 3

#: ``kCartesianVelocityViolation`` (``error.h:13``) ->
#: ``cartesian_velocity_violation``.
#:
#: The Cartesian half of the safety controller, and the same kind of check as
#: :data:`JOINT_VELOCITY_VIOLATION_INDEX`: it watches how fast the *end
#: effector* is actually travelling, not what the client asked for, so it is
#: armed in every control mode. See
#: :meth:`MotionLimitChecker.check_measured_cartesian_velocity`.
#:
#: Hardware makes the EE frame the deciding factor. Take a run that first calls
#: ``robot->setEE`` with an EE 0.5 m out along the flange z, *then* commands a
#: joint-velocity motion ramping joints 2 and 4
#: at +-3 rad/s^2 towards 4 rad/s. Without the 0.5 m lever the same ramp
#: reaches the joint envelope first and hardware answers
#: ``joint_velocity_violation``; with it the EE crosses the translational limit
#: first and hardware answers ``cartesian_velocity_violation`` *alone*.
CARTESIAN_VELOCITY_VIOLATION_INDEX = 4

#: ``kJointPositionMotionGeneratorStartPoseInvalid`` (``error.h:20``) ->
#: ``joint_position_motion_generator_start_pose_invalid``.
JOINT_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX = 11

#: ``kJointMotionGeneratorPositionLimitsViolation`` (``error.h:21``).
JOINT_MOTION_GENERATOR_POSITION_LIMITS_VIOLATION_INDEX = 12

#: ``kJointMotionGeneratorVelocityLimitsViolation`` (``error.h:22``).
JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX = 13

#: ``kJointMotionGeneratorVelocityDiscontinuity`` (``error.h:23``) -- the
#: acceleration limit **of a joint-position generator**. See
#: :ref:`the interface-relative naming rule <interface-relative>`: only ``q_c``
#: latches this, because only there is velocity the first derivative of the
#: commanded channel.
JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX = 14

#: ``kJointMotionGeneratorAccelerationDiscontinuity`` (``error.h:24``) -- the
#: jerk limit of a joint-*position* generator, and the *acceleration* limit of a
#: joint-*velocity* one. Again the interface-relative rule: on ``dq_c`` the
#: first derivative is already an acceleration, so breaking
#: :data:`MAX_JOINT_ACCELERATION` there is named an acceleration discontinuity.
JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX = 15

#: ``kCartesianPositionMotionGeneratorStartPoseInvalid`` (``error.h:25``) --
#: the first ``O_T_EE_c`` of a ``kCartesianPosition`` motion does not sit where
#: the robot actually is. Hardware reports it for a +10 m z offset
#: held from cycle 0, and, exactly like the joint start-pose check,
#: it **outranks** the discontinuity checks: a first command that is 10 m away
#: is a start-pose error, not a velocity discontinuity.
CARTESIAN_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX = 16

#: ``kCartesianMotionGeneratorElbowLimitViolation`` (``error.h:26``) -- the
#: elbow's velocity, acceleration and jerk limits, all three. The velocity one
#: is observable on hardware: hold the pose and ramp the elbow at a constant
#: ``ddelbow = 0.0003 / 0.001 = 0.3`` rad/s^2, so elbow velocity grows linearly
#: and crosses :data:`MAX_ELBOW_VELOCITY` (1.499 rad/s) after ~5 s while
#: acceleration stays at 0.3, far inside :data:`MAX_ELBOW_ACCELERATION`.
CARTESIAN_MOTION_GENERATOR_ELBOW_LIMIT_VIOLATION_INDEX = 17

#: ``kCartesianMotionGeneratorVelocityLimitsViolation`` (``error.h:27``).
CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX = 18

#: ``kCartesianMotionGeneratorVelocityDiscontinuity`` (``error.h:28``) -- the
#: acceleration limit **of a Cartesian pose generator** (``O_T_EE_c``).
#: Hardware is explicit about the pairing: a step in ``O_T_EE_c``
#: gives ``cartesian_motion_generator_velocity_discontinuity``, while a step in
#: ``O_dP_EE_c`` gives ``..._acceleration_discontinuity``.
CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX = 19

#: ``kCartesianMotionGeneratorAccelerationDiscontinuity`` (``error.h:29``) --
#: the jerk limit of a Cartesian *pose* generator, and the *acceleration* limit
#: of a Cartesian *velocity* one, which is what the mobile base runs.
CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX = 20

#: ``kCartesianMotionGeneratorElbowSignInconsistent`` (``error.h:30``) -- the
#: ``elbow[1]`` branch flag (the sign of joint 4) flipped *mid-motion*.
#: Reproduced on hardware by holding ``O_T_EE_d`` and negating ``elbow[1]``
#: at t > 0.5 s.
CARTESIAN_MOTION_GENERATOR_ELBOW_SIGN_INCONSISTENT_INDEX = 21

#: ``kCartesianMotionGeneratorStartElbowInvalid`` (``error.h:31``) -- the first
#: ``elbow_c`` of a Cartesian motion does not describe the elbow the robot is
#: actually in. Reproduced on hardware by adding 0.5 rad to ``elbow[0]`` from
#: cycle 0 with the pose left alone.
CARTESIAN_MOTION_GENERATOR_START_ELBOW_INVALID_INDEX = 22

#: ``kStartElbowSignInconsistent`` (``error.h:33``) ->
#: ``start_elbow_sign_inconsistent``.
#:
#: **Not latched by this module, and that is a judgement call worth seeing.**
#: By its name this is the start-time twin of
#: :data:`CARTESIAN_MOTION_GENERATOR_ELBOW_SIGN_INCONSISTENT_INDEX`, so a first
#: ``elbow_c[1]`` disagreeing with ``sign(q[3])`` is arguably *this* error
#: rather than :data:`CARTESIAN_MOTION_GENERATOR_START_ELBOW_INVALID_INDEX`.
#: Nothing pins it: no hardware observation is available for a motion opened
#: with a wrong elbow *sign* (the observable start-elbow case perturbs
#: ``elbow[0]`` and leaves the sign alone), and libfranka refuses to *send* an
#: elbow whose flag is not exactly +-1 (``checkElbow`` -> ``isValidElbow``,
#: ``control_tools.h``), so no libfranka client can produce the ambiguous case
#: at all. With no evidence, both halves of the start-elbow check report 22, the
#: one index hardware is confirmed to report for "the elbow you opened with is
#: not the elbow you are in". Kept in
#: :data:`ERROR_NAMES` so the vocabulary stays complete.
START_ELBOW_SIGN_INCONSISTENT_INDEX = 24

#: ``kCartesianPositionMotionGeneratorInvalidFrame`` (``error.h:40``) ->
#: ``cartesian_position_motion_generator_invalid_frame_flag``. A commanded
#: ``O_T_EE_c`` that is not a homogeneous transformation at all.
#:
#: On hardware this is a *client-side* refusal: libfranka's ``checkMatrix``
#: (``include/franka/control_tools.h``) throws ``std::invalid_argument`` before
#: the datagram is ever packed, which is why a stock libfranka client sees
#: ``std::invalid_argument`` and not a ``ControlException`` for an invalid
#: frame. A simulator cannot rely on that -- the
#: client on the other end need not be libfranka -- so the same test is done
#: server-side and the enum's own name for the condition is what gets reported.
CARTESIAN_POSITION_MOTION_GENERATOR_INVALID_FRAME_INDEX = 31

#: ``kControllerTorqueDiscontinuity`` (``error.h:41``) -- the torque *rate*
#: limit, per ``docs/overview.rst``: "Control also computes the torque rate with
#: backwards Euler".
CONTROLLER_TORQUE_DISCONTINUITY_INDEX = 32

#: ``kTauJRangeViolation`` (``error.h:43``) -> ``tau_J_range_violation``. There
#: is no ``controller_torque_range_violation`` in the enum; this is the one
#: error the robot has for a commanded torque outside the joint's range ("If
#: the **torque sensor limit** is reached, a ``tau_j_range_violation`` will be
#: triggered", ``docs/overview.rst``).
TAU_J_RANGE_VIOLATION_INDEX = 34

#: Wire name for each index we can latch, as ``getErrorName`` spells it
#: (``error.h:52-138``). Used in the log lines so a message can be grepped
#: straight against libfranka's own vocabulary.
ERROR_NAMES = {
    JOINT_VELOCITY_VIOLATION_INDEX: "joint_velocity_violation",
    CARTESIAN_VELOCITY_VIOLATION_INDEX: "cartesian_velocity_violation",
    JOINT_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX: (
        "joint_position_motion_generator_start_pose_invalid"
    ),
    JOINT_MOTION_GENERATOR_POSITION_LIMITS_VIOLATION_INDEX: (
        "joint_motion_generator_position_limits_violation"
    ),
    JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX: (
        "joint_motion_generator_velocity_limits_violation"
    ),
    JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX: (
        "joint_motion_generator_velocity_discontinuity"
    ),
    JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX: (
        "joint_motion_generator_acceleration_discontinuity"
    ),
    CARTESIAN_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX: (
        "cartesian_position_motion_generator_start_pose_invalid"
    ),
    CARTESIAN_MOTION_GENERATOR_ELBOW_LIMIT_VIOLATION_INDEX: (
        "cartesian_motion_generator_elbow_limit_violation"
    ),
    CARTESIAN_MOTION_GENERATOR_ELBOW_SIGN_INCONSISTENT_INDEX: (
        "cartesian_motion_generator_elbow_sign_inconsistent"
    ),
    CARTESIAN_MOTION_GENERATOR_START_ELBOW_INVALID_INDEX: (
        "cartesian_motion_generator_start_elbow_invalid"
    ),
    START_ELBOW_SIGN_INCONSISTENT_INDEX: "start_elbow_sign_inconsistent",
    CARTESIAN_POSITION_MOTION_GENERATOR_INVALID_FRAME_INDEX: (
        "cartesian_position_motion_generator_invalid_frame_flag"
    ),
    CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX: (
        "cartesian_motion_generator_velocity_limits_violation"
    ),
    CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX: (
        "cartesian_motion_generator_velocity_discontinuity"
    ),
    CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX: (
        "cartesian_motion_generator_acceleration_discontinuity"
    ),
    CONTROLLER_TORQUE_DISCONTINUITY_INDEX: "controller_torque_discontinuity",
    TAU_J_RANGE_VIOLATION_INDEX: "tau_J_range_violation",
}

# -- the interface-relative naming rule ---------------------------------------
#
# .. _interface-relative:
#
# The enum has *two* discontinuity names per generator family, and which one a
# violation gets does **not** depend on which derivative broke its limit. It
# depends on which derivative the *client commands*: hardware names a
# discontinuity one derivative above the commanded channel.
#
#   commanded channel   first difference   second difference
#   -----------------   ----------------   -----------------
#   q_c                 velocity -> 14     acceleration -> 15
#   dq_c                acceleration -> 15 jerk -> 15
#   O_T_EE_c            velocity -> 19     acceleration -> 20
#   O_dP_EE_c           acceleration -> 20 jerk -> 20
#   tau_J_d             rate -> 32         --
#
# Hardware observation is what pins this, as a *pair* of runs per family that
# differ only in the interface they command:
#
# * stepping ``q_c`` by 1.0 rad mid-motion, and hardware answers
#   ``joint_motion_generator_velocity_discontinuity`` (14).
# * stepping ``dq_c`` by 50 rad/s mid-motion, and hardware answers
#   ``joint_motion_generator_acceleration_discontinuity`` (15).
# * the same pair exists for Cartesian pose (19) and Cartesian velocity (20).
#
# Both joint steps are enormous: after libfranka's 100 Hz command low-pass
# (gain 0.3859 on the first cycle) the 1.0 rad position step still implies
# ~385 rad/s, and the 50 rad/s velocity step still implies ~19,300 rad/s^2 --
# 19.3 rad/s of change inside one 1 ms cycle, ~1930x kMaxJointAcceleration. Each
# therefore breaks *every* limit its interface has -- envelope, acceleration and
# jerk together -- and hardware still returns exactly one name. So the second
# half of the rule is a precedence: **the discontinuity check beats the envelope
# check.** A joint-position step is 14, never 13; a joint-velocity step is 15,
# never 13; a twist step is 20, never 18.
#
# The order the per-generator checks below run in *is* that precedence, which is
# why they are ordered the way they are and not by derivative.

# -- sim-side choices ---------------------------------------------------------

#: How far the first commanded position of a joint-position motion may sit from
#: the robot's ``q_d`` before it is ``joint_position_motion_generator_start_pose_invalid``.
#:
#: **Not a libfranka constant.** The check lives in Control, not in libfranka,
#: and the tolerance is not published anywhere in the v10 tree or its docs --
#: only the remedy is ("make sure that your control loop starts with the last
#: commanded value observed in the robot state", ``docs/overview.rst``). 0.1 rad
#: is a sim choice: loose enough that the simulator's own position-tracking
#: error can never manufacture the error, tight enough that a client which
#: jumps into a motion from a stale or hard-coded pose is caught. Override it
#: per-instance if your client needs a different contract.
#:
#: **Checked against observed hardware behaviour.** The reference provocation is
#: a motion that opens with **+0.2 rad on joint 1**, zero elsewhere, added to
#: ``state.q_d`` from cycle 0. libfranka does
#: not attenuate that opening step the way it attenuates a mid-motion one: on the
#: first cycle ``initialized_filter_`` is false, so the command low-pass takes
#: its reference from the command itself and is an identity
#: (``ControlLoop<JointPositions>::convertMotion``). The sim therefore sees the
#: full 0.2 rad, and 0.1 rad catches it with 2x margin. Nothing observed
#: constrains the tolerance from below -- every conforming motion opens from
#: ``q_d`` exactly, at offset 0 -- so this is the tightest value that
#: is evidenced rather than guessed.
START_POSE_TOLERANCE = 0.1

#: How far the first commanded pose of a ``kCartesianPosition`` motion may sit
#: from the robot's *actual* ``O_T_EE`` before it is
#: ``cartesian_position_motion_generator_start_pose_invalid`` -- translation in
#: metres, rotation as the angle of the relative rotation in radians.
#:
#: **Not libfranka constants**, exactly like :data:`START_POSE_TOLERANCE`, and
#: chosen on the same reasoning: the check lives in Control, the tolerance is
#: published nowhere in the v10 tree, and only the remedy is ("make sure that
#: your control loop starts with the last commanded value observed in the robot
#: state", ``docs/overview.rst``). 0.1 rad is the joint-space precedent, so the
#: rotational half simply reuses it; 0.05 m is its translational counterpart --
#: loose enough that the simulator's own flange-pose reporting can never
#: manufacture the error, tight enough to catch a client jumping in from a stale
#: or hard-coded pose. The reference hardware provocation offsets by **10 m**,
#: three orders of magnitude clear of either
#: number, so nothing about the pinned behaviour is sensitive to the choice.
#: Override per-instance if your client needs a different contract.
START_CARTESIAN_POSE_TRANSLATION_TOLERANCE = 0.05
START_CARTESIAN_POSE_ROTATION_TOLERANCE = 0.1

#: How far the first commanded ``elbow_c[0]`` may sit from the robot's actual
#: elbow angle before it is ``cartesian_motion_generator_start_elbow_invalid``,
#: in radians. Another sim choice, sized to the joint-space precedent
#: (:data:`START_POSE_TOLERANCE`, 0.1 rad) because ``elbow[0]`` *is* a joint
#: angle -- joint 3 of an FR3. The reference hardware provocation offsets by
#: 0.5 rad, five times this.
START_ELBOW_TOLERANCE = 0.1

#: ``kOrthonormalThreshold`` from ``franka::isHomogeneousTransformation``
#: (``include/franka/control_tools.h``): how far a row or column norm of the
#: rotation block may sit from 1 before the matrix is not a rigid transform.
ORTHONORMAL_THRESHOLD = 1e-5

#: Below this angle the rotation log-map takes its small-angle branch; see
#: :func:`rotation_log`. Well under a microradian, i.e. under a nanoradian per
#: millisecond of commanded rotation, so nothing a client can command lands here
#: except a genuinely unrotating stream.
ROTATION_LOG_SMALL_ANGLE = 1e-8

#: How close to pi the rotation log-map switches to its symmetric-part branch;
#: see :func:`rotation_log`.
#:
#: Not a taste call -- it is where the ordinary ``theta / (2 sin theta)`` form
#: stops being accurate. ``theta`` comes out of ``acos``, which is
#: ill-conditioned at both ends: an absolute error of one ulp in the trace
#: becomes ``sqrt(2 * ulp)`` ~ 2e-8 in ``theta``. For a *small* angle that does
#: not matter, because ``theta / sin(theta) -> 1`` and the error cancels out of
#: the result. Near pi it does not cancel: at ``theta = pi - 1e-6`` the
#: recomputed ``sin(theta)`` is 4.4e-5 relative off, and the reported magnitude
#: lands 1.4e-4 rad away from the true angle. The symmetric branch never
#: recomputes a sine, so it stays accurate to ``theta`` itself (~1e-11 there).
#: 1e-4 is where the ordinary branch's error is still ~1e-8 rad.
ROTATION_LOG_NEAR_PI = 1e-4

#: Same idea for a velocity generator's first ``dq_c`` against the reported
#: ``dq_d``, in rad/s. Not a libfranka constant either; the enum has no
#: joint-*velocity* start error, so exceeding this latches
#: ``joint_motion_generator_velocity_discontinuity`` -- which is what a step
#: away from ``dq_d`` physically is.
START_VELOCITY_TOLERANCE = 0.1

#: How far *measured* joint velocity may sit outside the position-based
#: envelope before the safety controller's ``joint_velocity_violation`` fires,
#: in rad/s.
#:
#: **Not a libfranka constant.** The safety controller lives in Control, not in
#: libfranka, and its margin is not published anywhere in the v10 tree; only the
#: error name is. This is a sim choice, and it exists because the two signals
#: being compared are not the same kind of number: the envelope is analytic and
#: exact, while ``dq`` comes out of a physics integrator and carries settling
#: ring, contact spikes and finite-difference noise. Judging a noisy signal
#: against an exact bound with no margin turns every millisecond of numerical
#: jitter at the envelope into a reflex.
#:
#: 0.1 rad/s is ~4% of the *tightest* envelope value the FR3 has in free space
#: (2.62 rad/s, joints 1-4), so it cannot mask a violation anyone cares about:
#: the excursions that provoke it on hardware overshoot by whole rad/s (a 5
#: rad/s^2 ramp crosses the envelope and keeps accelerating, and a 3 Nm torque
#: ramp folds the arm through it). It is also far below the headroom a
#: *conforming* motion keeps -- ``test_a_smooth_motion_never_trips_the_safety_``
#: ``controller`` sweeps all seven joints through a full cosine and never comes
#: within five times this margin of the envelope.
MEASURED_JOINT_VELOCITY_MARGIN = 0.1

#: Bound on *measured* end-effector translational speed, in m/s, above which the
#: safety controller latches ``cartesian_velocity_violation``. Franka's own
#: number: :data:`MAX_TRANSLATIONAL_VELOCITY`, i.e. ``franka::kMaxTranslational``
#: ``Velocity`` = 3.0 m/s less ``kLimitEps`` (``include/franka/rate_limiting.h:82``),
#: which is the FR3 specification's Cartesian translational velocity limit
#: (``p_dot_max = 3.0 m/s`` in Franka's "Robot and interface specifications").
#: Nothing separate is published for the safety controller's copy of it, so the
#: one published Cartesian translational bound is used for both.
#:
#: **Translation only, and that is measured from hardware rather than assumed.**
#: The obvious companion bound would be :data:`MAX_ROTATIONAL_VELOCITY`
#: (2.5 rad/s) on the EE angular velocity, but hardware behaviour rules it
#: out. Two provocations spin *joint 6* -- whose axis is (near enough) the EE's
#: own -- straight through the joint envelope: a 3 Nm torque ramp, and a
#: 5 rad/s^2 ``dq_c`` ramp. Joint 6's envelope
#: is 4.18 rad/s, so in both the EE angular speed passes 2.5 rad/s well before
#: the joint limit -- and hardware still answers ``joint_velocity_violation``,
#: never ``cartesian_velocity_violation``. A rotational term here would rename
#: both of those errors and break them. Elbow speed is excluded for the same
#: reason plus one more: the enum's only elbow-speed error is the motion
#: generator's (:data:`CARTESIAN_MOTION_GENERATOR_ELBOW_LIMIT_VIOLATION_INDEX`),
#: and it is a *commanded* check.
#:
#: No tolerance is added on top, unlike :data:`MEASURED_JOINT_VELOCITY_MARGIN`.
#: That margin exists because ordinary motions ride right up against the joint
#: envelope, where integrator noise alone can cross it; 3 m/s of end-effector
#: travel is an order of magnitude beyond anything a conforming motion reaches
#: (the fastest point-to-point moves observed peak near 1 m/s), so there is no
#: jitter at the boundary to absorb.
MEASURED_CARTESIAN_VELOCITY_LIMIT = MAX_TRANSLATIONAL_VELOCITY

#: Smallest singular value of the end-effector Jacobian at or below which a
#: configuration counts as *singular*, so that a ``Move`` asking for a Cartesian
#: motion generator from there is rejected with
#: ``Move::Status::kStartAtSingularPoseRejected`` -- the status libfranka turns
#: into "Move command rejected: cannot start at singular pose!"
#: (``src/robot_impl.h:427``).
#:
#: **Not a libfranka constant.** Control's own singularity test is not published
#: anywhere; only the rejection status and its message are. The measure chosen
#: here is the standard one -- ``sigma_min`` of the 6x7 geometric Jacobian,
#: which is the reciprocal of the joint speed needed per unit of EE speed in the
#: worst-conditioned direction, and goes to zero exactly when a Cartesian
#: direction becomes unreachable.
#:
#: 0.05 is placed empirically, against the two things it has to separate on the
#: sim's own FR3 model (``robot_descriptions`` ``fr3_v2``):
#:
#: * A known singular configuration of the FR3,
#:   ``q = {0, 1.28, 0, -0.5415, 0, 2.74, 0}``, where
#:   ``sigma_min`` = 0.0108 -- a factor 4.6 below the threshold. Other
#:   near-singular configurations of the arm sit under it too (0.092 for a
#:   second one measured on this model).
#: * Every start pose from which a Cartesian motion is expected to open
#:   *successfully*. The reference start configurations span 0.151..0.205, with
#:   two further ones at 0.227 and 0.139 -- the tightest of them still a factor
#:   2.8 above the threshold.
#:
#: The gap between 0.011 and 0.139 is two orders of magnitude wide in the ratio
#: sense; 0.05 is the geometric middle of it.
SINGULAR_POSE_MIN_SINGULAR_VALUE = 0.05

#: Opt-in switch for the abort. Validation and the rate-limited warning always
#: run; setting this to a truthy value additionally makes a violation latch the
#: error, stop the motion and refuse the offending command, the way the robot
#: does. Off by default because a sim is routinely driven by scripted clients
#: that step their targets.
ENFORCE_ENV_VAR = "FRANKA_SIM_ENFORCE_MOTION_LIMITS"

#: Spellings of :data:`ENFORCE_ENV_VAR` that turn enforcement on. An allow-list,
#: not a deny-list: "set, therefore on" would read ``=disabled``
#: as *enabled*, which is the wrong way round for a switch that can stop a
#: motion. Any nonzero integer counts too, so ``=2`` behaves like ``=1``.
_TRUTHY = {"1", "true", "t", "yes", "y", "on", "enable", "enabled"}


def _is_truthy(value: str) -> bool:
    """Whether ``value`` spells "on"; see :data:`_TRUTHY`."""
    value = value.strip().lower()
    if value in _TRUTHY:
        return True
    return value.lstrip("+-").isdigit() and int(value) != 0


def enforcement_enabled_by_env(environ: Optional[Dict[str, str]] = None) -> bool:
    """Whether motion-limit aborts are on, per :data:`ENFORCE_ENV_VAR`."""
    env = os.environ if environ is None else environ
    return _is_truthy(env.get(ENFORCE_ENV_VAR, ""))


#: Fallback cap on the cycles a single received command may be differenced over,
#: used only when the server has not told the checker what it published (see
#: :meth:`MotionLimitChecker.note_published`).
#:
#: Without that, the interval can only come from the client's own echoed
#: ``message_id``, and a client is not a trustworthy source for the denominator
#: of the server's own limit check: inflating the echo by a thousand divides
#: every commanded derivative by a thousand, and 50 rad/s steps sail through.
#: Two datagrams landing in one poll is the only reason a sim legitimately sees
#: more than one cycle without observing the gap itself, so three is already
#: generous. (A packet *gap* is not one of those reasons: the missed cycles are
#: extrapolated and recorded, so the history is never more than a cycle behind
#: the newest published state -- see :meth:`MotionLimitChecker.extrapolate`.)
#:
#: The live server does not use this. It calls :meth:`note_published` every
#: cycle, and the interval is then bounded by the server's *own* observation --
#: how many states it has published since the command the history sits at --
#: which is both honest about a real gap and immune to an inflated echo. See
#: :meth:`cycles_since_applied`.
MAX_COALESCED_CYCLES = 3


# -- limit helpers ------------------------------------------------------------


def smallest_singular_value(jacobian: Optional[Any]) -> Optional[float]:
    """``sigma_min`` of a Jacobian, or None when there is nothing to measure.

    None rather than an exception for a missing, misshapen, unconvertible or
    non-finite Jacobian: the caller's remedy in every one of those cases is the
    same -- skip the singularity test rather than guess -- and the input comes
    from a physics backend, not from the client. The conversion is guarded for
    the same reason the shape is: a backend free to publish whatever it likes
    under ``O_J_EE`` can publish something ``np.asarray(..., dtype=float)``
    raises on (a ragged nested sequence, a string, an object array), and a
    ``Move`` must not fail with a ``ValueError`` from a diagnostic.
    """
    if jacobian is None:
        return None
    try:
        matrix = np.asarray(jacobian, dtype=float)
    except (TypeError, ValueError):
        return None
    if matrix.ndim != 2 or matrix.size == 0 or not np.all(np.isfinite(matrix)):
        return None
    try:
        return float(np.linalg.svd(matrix, compute_uv=False)[-1])
    except np.linalg.LinAlgError:  # pragma: no cover - SVD not converging
        return None


def is_singular_configuration(
    jacobian: Optional[Any], threshold: float = SINGULAR_POSE_MIN_SINGULAR_VALUE
) -> bool:
    """Whether ``jacobian`` describes a singular configuration.

    False when no Jacobian is available at all, which is the safe answer: a
    backend that cannot say is not grounds for refusing the client's ``Move``.
    See :data:`SINGULAR_POSE_MIN_SINGULAR_VALUE` for where the threshold comes
    from.
    """
    sigma_min = smallest_singular_value(jacobian)
    return sigma_min is not None and sigma_min <= threshold


def upper_joint_velocity_limits(joint_positions: Sequence[float]) -> List[float]:
    """Per-joint upper velocity limit at ``joint_positions``.

    ``JointVelocityLimitsConfig::getUpperJointVelocityLimits``
    (``src/joint_velocity_limits.cpp:116-130``)::

        min(max_velocity,
            max(0, -velocity_offset
                   + sqrt(max(0, 2 * deceleration_limit * (upper_position_limit - q)))))
        - kJointVelocityLimitsTolerance

    The result is clamped up to zero: right at the joint's position limit the
    raw formula goes slightly *negative* (it is a rate-limiter target, and the
    limiter's job there is to decelerate), and a sim that called a commanded
    zero velocity an error would be wrong about the one command that is always
    safe. The position-limit check catches that configuration anyway.
    """
    limits = []
    for index, position in enumerate(joint_positions[:7]):
        max_velocity, offset, deceleration = POSITION_BASED_VELOCITY_LIMITS[index]
        upper = JOINT_POSITION_LIMITS[index][1]
        ramp = math.sqrt(max(0.0, 2.0 * deceleration * (upper - position)))
        raw = min(max_velocity, max(0.0, -offset + ramp)) - JOINT_VELOCITY_LIMITS_TOLERANCE[index]
        limits.append(max(0.0, raw))
    return limits


def lower_joint_velocity_limits(joint_positions: Sequence[float]) -> List[float]:
    """Per-joint lower velocity limit at ``joint_positions``.

    The mirror of :func:`upper_joint_velocity_limits`
    (``src/joint_velocity_limits.cpp:132-146``), clamped down to zero for the
    same reason.
    """
    limits = []
    for index, position in enumerate(joint_positions[:7]):
        max_velocity, offset, deceleration = POSITION_BASED_VELOCITY_LIMITS[index]
        lower = JOINT_POSITION_LIMITS[index][0]
        ramp = math.sqrt(max(0.0, 2.0 * deceleration * (position - lower)))
        raw = max(-max_velocity, min(0.0, offset - ramp)) + JOINT_VELOCITY_LIMITS_TOLERANCE[index]
        limits.append(min(0.0, raw))
    return limits
