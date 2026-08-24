r"""FCI motion-limit and discontinuity checking for commanded signals.

The real FCI does not accept whatever a client sends. Control differentiates
every commanded signal with backward Euler at the 1 ms cycle and compares the
result against published per-joint limits; a signal that steps, ramps too hard
or leaves the joint range stops the motion with a specific error. libfranka's
own documentation spells the arithmetic out:

    "For every motion generator, Control differentiates the signals sent by the
    user with backwards Euler. For instance, if, using a joint position motion
    generator, at time :math:`k` the user sends the command :math:`q_{c,k}`, the
    resulting velocity, acceleration and jerk will be:

    * Velocity :math:`\dot{q}_{c,k} = (q_{c,k} - q_{c,k-1}) / \Delta t`
    * Acceleration :math:`\ddot{q}_{c,k} = (\dot{q}_{c,k} - \dot{q}_{c,k-1}) / \Delta t`
    * Jerk :math:`\dddot{q}_{c,k} = (\ddot{q}_{c,k} - \ddot{q}_{c,k-1}) / \Delta t`

    where :math:`\Delta t = 0.001`."
    -- libfranka v10, ``docs/overview.rst`` (Errors due to velocity limits
    violation and discontinuity errors)

    "Note that :math:`q_{c,k-1}, \dot{q}_{c,k-1}` and :math:`\ddot{q}_{c,k-1}`
    are always sent back to the user in the robot state as :math:`q_{d}`,
    :math:`\dot{q}_{d}` and :math:`\ddot{q}_{d}` so you will be able to compute
    the resulting derivatives in advance, **even in case of packet losses**."
    -- same section

That last clause is why this module differences against the *last applied*
command rather than the last *received* one -- and "applied" includes the
commands nobody sent. When a cycle is lost the FCI extrapolates, the
extrapolated value becomes ``q_d``, and the client's next real command is
differenced against *that*. This module does the same:
:meth:`MotionLimitChecker.extrapolate` produces one waypoint per missed cycle
under the acceleration frozen at the start of the gap, records it as applied,
and checks it exactly as it checks a received command. A client resuming after
a gap is therefore differenced against a reference that kept moving, over the
standard single millisecond -- which is both what hardware does and why
resuming from the client's *own* last waypoint is the trap libfranka warns
about (``docs/overview.rst``: intermittent drops "could trigger `discontinuity`
errors even when your source signals conform with the interface
specification").

The error names and the finite-difference formulas are libfranka's; the limits
come from ``include/franka/rate_limiting.h`` and, for the per-joint position,
velocity and torque ranges, from the FR3 URDF this server itself serves over
``GetRobotModel`` (``franka_sim/models/fr3.urdf``) -- the same file libfranka
v10's ``JointVelocityLimitsConfig`` parses. Every constant below carries its
citation.

Two things about the *names* are easy to get wrong, and both are pinned here
against Franka's own hardware smoke suite (``arm_smoke_tests``, whose
expectations are recorded from real robots):

* **Which discontinuity a violation is called depends on the interface, not on
  the derivative that broke.** Hardware names a discontinuity one derivative
  above the channel the client commands, and a discontinuity outranks the
  velocity-envelope check when a single step breaks both. See the
  interface-relative naming table below.
* **Not every velocity error is a motion-generator error.** The robot's safety
  controller separately watches *measured* joint velocity against the same
  position-based envelope and latches ``joint_velocity_violation`` -- a
  different error, active in every control mode including pure torque, where no
  commanded velocity exists at all. See
  :meth:`MotionLimitChecker.check_measured_velocity`.

Validation and logging are always on. The *abort* -- latching the error,
answering the ``Move`` with ``kReflexAborted`` and refusing the offending
command -- is opt-in; see :data:`ENFORCE_ENV_VAR`.
"""

import logging
import math
import os
import sys
import threading
from dataclasses import dataclass, replace
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np

from franka_sim.control_modes import ControlMode

logger = logging.getLogger(__name__)

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
#: is the robot watching what the arm is actually doing. It is the error
#: Franka's own hardware smoke suite observes whenever *measured* joint velocity
#: leaves the position-based envelope, whatever the client was commanding --
#: including in pure torque control, where there is no commanded velocity to
#: judge at all (``smoke_errors.cpp:773`` ``moveJointVelocityViolation``: a
#: 3 Nm ramp on joint 6, expectation pinned in ``smoke_test_errors.cpp:173``).
#: It also *outranks* the motion generator's own envelope check: the suite's
#: ``JointMotionGeneratorVelocityLimitsViolationHardware``
#: (``smoke_test_errors.cpp:135``) ramps ``dq_c`` past the envelope and records
#: that hardware answers ``joint_velocity_violation`` rather than
#: ``joint_motion_generator_velocity_limits_violation`` -- the safety controller
#: sees the arm cross the envelope before Control finishes objecting to the
#: command that put it there.
JOINT_VELOCITY_VIOLATION_INDEX = 3

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
#: the robot actually is. Franka's smoke suite pins it with a +10 m z offset
#: from cycle 0 (``smoke_errors.cpp:242``
#: ``moveCartesianPositionMotionGeneratorStartPoseInvalid``, expectation at
#: ``smoke_test_errors.cpp:204``), and, exactly like the joint start-pose check,
#: it **outranks** the discontinuity checks: a first command that is 10 m away
#: is a start-pose error, not a velocity discontinuity.
CARTESIAN_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX = 16

#: ``kCartesianMotionGeneratorElbowLimitViolation`` (``error.h:26``) -- the
#: elbow's velocity, acceleration and jerk limits, all three. Franka's smoke
#: suite pins the velocity one: ``moveCartesianMotionGeneratorElbowLimitViolation``
#: (``smoke_errors.cpp:509``) holds the pose and ramps the elbow at a constant
#: ``ddelbow = 0.0003 / 0.001 = 0.3`` rad/s^2, so elbow velocity grows linearly
#: and crosses :data:`MAX_ELBOW_VELOCITY` (1.499 rad/s) after ~5 s while
#: acceleration stays at 0.3, far inside :data:`MAX_ELBOW_ACCELERATION`. The
#: expectation is recorded at ``smoke_test_errors.cpp:268``.
CARTESIAN_MOTION_GENERATOR_ELBOW_LIMIT_VIOLATION_INDEX = 17

#: ``kCartesianMotionGeneratorVelocityLimitsViolation`` (``error.h:27``).
CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX = 18

#: ``kCartesianMotionGeneratorVelocityDiscontinuity`` (``error.h:28``) -- the
#: acceleration limit **of a Cartesian pose generator** (``O_T_EE_c``).
#: Franka's smoke suite is explicit about the pairing: a step in ``O_T_EE_c``
#: gives ``cartesian_motion_generator_velocity_discontinuity``
#: (``smoke_errors.cpp:88`` + ``smoke_test_errors.cpp:48``) while a step in
#: ``O_dP_EE_c`` gives ``..._acceleration_discontinuity``
#: (``smoke_errors.cpp:124`` + ``smoke_test_errors.cpp:64``).
CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX = 19

#: ``kCartesianMotionGeneratorAccelerationDiscontinuity`` (``error.h:29``) --
#: the jerk limit of a Cartesian *pose* generator, and the *acceleration* limit
#: of a Cartesian *velocity* one, which is what the mobile base runs.
CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX = 20

#: ``kCartesianMotionGeneratorElbowSignInconsistent`` (``error.h:30``) -- the
#: ``elbow[1]`` branch flag (the sign of joint 4) flipped *mid-motion*. Franka's
#: smoke suite pins it: ``moveCartesianMotionGeneratorElbowSignInconsistent``
#: (``smoke_errors.cpp:308``) holds ``O_T_EE_d`` and negates ``elbow[1]`` at
#: t > 0.5 s; expectation at ``smoke_test_errors.cpp:237``.
CARTESIAN_MOTION_GENERATOR_ELBOW_SIGN_INCONSISTENT_INDEX = 21

#: ``kCartesianMotionGeneratorStartElbowInvalid`` (``error.h:31``) -- the first
#: ``elbow_c`` of a Cartesian motion does not describe the elbow the robot is
#: actually in. ``moveCartesianMotionGeneratorStartElbowInvalid``
#: (``smoke_errors.cpp:342``) adds 0.5 rad to ``elbow[0]`` from cycle 0 with the
#: pose left alone; expectation at ``smoke_test_errors.cpp:252``.
CARTESIAN_MOTION_GENERATOR_START_ELBOW_INVALID_INDEX = 22

#: ``kStartElbowSignInconsistent`` (``error.h:33``) ->
#: ``start_elbow_sign_inconsistent``.
#:
#: **Not latched by this module, and that is a judgement call worth seeing.**
#: By its name this is the start-time twin of
#: :data:`CARTESIAN_MOTION_GENERATOR_ELBOW_SIGN_INCONSISTENT_INDEX`, so a first
#: ``elbow_c[1]`` disagreeing with ``sign(q[3])`` is arguably *this* error
#: rather than :data:`CARTESIAN_MOTION_GENERATOR_START_ELBOW_INVALID_INDEX`.
#: Nothing pins it: Franka's smoke suite has no test that opens a motion with a
#: wrong elbow sign (its start-elbow test perturbs ``elbow[0]`` and leaves the
#: sign alone), and libfranka refuses to *send* an elbow whose flag is not
#: exactly +-1 (``checkElbow`` -> ``isValidElbow``, ``control_tools.h``), so no
#: libfranka client can produce the ambiguous case at all. With no evidence,
#: both halves of the start-elbow check report 22, the one index the suite does
#: pin for "the elbow you opened with is not the elbow you are in". Kept in
#: :data:`ERROR_NAMES` so the vocabulary stays complete.
START_ELBOW_SIGN_INCONSISTENT_INDEX = 24

#: ``kCartesianPositionMotionGeneratorInvalidFrame`` (``error.h:40``) ->
#: ``cartesian_position_motion_generator_invalid_frame_flag``. A commanded
#: ``O_T_EE_c`` that is not a homogeneous transformation at all.
#:
#: On hardware this is a *client-side* refusal: libfranka's ``checkMatrix``
#: (``include/franka/control_tools.h``) throws ``std::invalid_argument`` before
#: the datagram is ever packed, which is why the smoke suite's
#: ``CartesianPositionMotionGeneratorInvalidFrame`` expects
#: ``std::invalid_argument`` and not a ``ControlException``
#: (``smoke_test_errors.cpp:221``). A simulator cannot rely on that -- the
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
# Franka's hardware smoke suite is what pins this, and it pins it as a *pair*
# of tests per family that differ only in the interface they command:
#
# * ``moveJointMotionGeneratorVelocityDiscontinuity`` steps ``q_c`` by 1.0 rad
#   mid-motion and hardware answers ``joint_motion_generator_velocity_``
#   ``discontinuity`` (14) -- ``smoke_errors.cpp:15`` / ``smoke_test_errors.cpp:16``.
# * ``moveJointMotionGeneratorAccelerationDiscontinuity`` steps ``dq_c`` by
#   50 rad/s mid-motion and hardware answers ``joint_motion_generator_``
#   ``acceleration_discontinuity`` (15) -- ``smoke_errors.cpp:51`` /
#   ``smoke_test_errors.cpp:32``.
# * the same pair exists for Cartesian pose (19) and Cartesian velocity (20) --
#   ``smoke_errors.cpp:88`` and ``:124``.
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
#: **Checked against both revisions of Franka's smoke suite, which agree.**
#: ``moveJointPositionMotionGeneratorStartPoseInvalid`` opens with
#: ``std::array<double, kNrJoints> discontinuity{{0.2}}`` -- +0.2 rad on joint 1,
#: zero elsewhere -- added to ``state.q_d`` from cycle 0, and the code is
#: byte-identical in the old ``libfranka/test/smoke/src/smoke_errors.cpp:194``
#: and the current ``arm_smoke_tests/src/smoke_errors.cpp:211``. libfranka does
#: not attenuate that opening step the way it attenuates a mid-motion one: on the
#: first cycle ``initialized_filter_`` is false, so the command low-pass takes
#: its reference from the command itself and is an identity
#: (``ControlLoop<JointPositions>::convertMotion``). The sim therefore sees the
#: full 0.2 rad in both suites, and 0.1 rad catches it with 2x margin. Neither
#: suite constrains the tolerance from below -- every conforming motion in them
#: opens from ``q_d`` exactly, at offset 0 -- so this is the tightest value that
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
#: or hard-coded pose. Franka's own smoke test offsets by **10 m**
#: (``smoke_errors.cpp:247``), three orders of magnitude clear of either
#: number, so nothing about the pinned behaviour is sensitive to the choice.
#: Override per-instance if your client needs a different contract.
START_CARTESIAN_POSE_TRANSLATION_TOLERANCE = 0.05
START_CARTESIAN_POSE_ROTATION_TOLERANCE = 0.1

#: How far the first commanded ``elbow_c[0]`` may sit from the robot's actual
#: elbow angle before it is ``cartesian_motion_generator_start_elbow_invalid``,
#: in radians. Another sim choice, sized to the joint-space precedent
#: (:data:`START_POSE_TOLERANCE`, 0.1 rad) because ``elbow[0]`` *is* a joint
#: angle -- joint 3 of an FR3. The smoke suite offsets by 0.5 rad
#: (``smoke_errors.cpp:350``), five times this.
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
#: the excursions the hardware suite provokes overshoot by whole rad/s (a 5
#: rad/s^2 ramp crosses the envelope and keeps accelerating, and a 3 Nm torque
#: ramp folds the arm through it). It is also far below the headroom a
#: *conforming* motion keeps -- ``test_a_smooth_motion_never_trips_the_safety_``
#: ``controller`` sweeps all seven joints through a full cosine and never comes
#: within five times this margin of the envelope.
MEASURED_JOINT_VELOCITY_MARGIN = 0.1

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


# -- violation reporting ------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    """One limit that one commanded value broke.

    ``error_index`` is the bit to latch in ``errors``/``reflex_reason``;
    everything else exists so the log line can name the joint or axis, the value
    and the limit, which is what makes an unenforced warning actionable.
    """

    error_index: int
    signal: str
    axis: str
    value: float
    limit: float
    unit: str = ""
    #: Refuse the command even with enforcement off. Only the non-finite class
    #: sets this: "reported but applied" is a defensible choice for a signal
    #: that is merely too fast, and no choice at all for one that is NaN --
    #: applying it puts NaN into the physics state and onto the wire, where it
    #: is not permissive, just broken.
    fatal: bool = False
    #: A second error bit to latch alongside :attr:`error_index`, in the same
    #: abort. Set only by :meth:`MotionLimitChecker.check`'s precedence block,
    #: where a commanded velocity-envelope violation (13) trips while the
    #: safety controller is armed: hardware's own dev comment records "Both
    #: error[s] can appear. However with the current RCU joint_velocity_violation
    #: appear[s] much earl[ier] as we shape the limit" for exactly this pair
    #: (``smoke_test_errors.cpp:110-127``, both the position- and
    #: velocity-limits-violation-hardware tests). None everywhere else,
    #: including the pure safety-controller path
    #: (:meth:`MotionLimitChecker.check_measured_velocity`), which latches 3
    #: alone.
    extra_error_index: Optional[int] = None

    @property
    def error_indices(self) -> Tuple[int, ...]:
        """The bit(s) to latch: one, or two when :attr:`extra_error_index` is set."""
        if self.extra_error_index is None:
            return (self.error_index,)
        return (self.error_index, self.extra_error_index)

    @property
    def error_name(self) -> str:
        """The wire name libfranka's ``getErrorName`` would print for the primary index."""
        return ERROR_NAMES[self.error_index]

    def describe(self) -> str:
        """One line naming the error(s), the axis, the value and the limit."""
        names = "+".join(ERROR_NAMES[index] for index in self.error_indices)
        unit = f" {self.unit}" if self.unit else ""
        return (
            f"{names}: {self.signal} {self.axis} = {self.value:.6g}{unit}, "
            f"limit {self.limit:.6g}{unit}"
        )


# -- differencing -------------------------------------------------------------


class _Differentiator:
    """Backward-Euler differences of one commanded signal, cycle by cycle.

    Holds the last applied value and its first two derivatives, which is
    everything the three difference formulas in ``docs/overview.rst`` need.
    Used at three depths:

    * joint **position** -- value ``q_c``; ``first``/``second``/``third`` are
      velocity, acceleration and jerk,
    * joint **velocity** -- value ``dq_c``; ``first``/``second`` are
      acceleration and jerk,
    * **torque** -- value ``tau_J_d``; ``first`` is the torque rate.
    """

    def __init__(self, width: int = 7):
        """Start at rest: value, first and second derivative all zero."""
        self.width = width
        self.value = [0.0] * width
        self.first = [0.0] * width
        self.second = [0.0] * width
        #: The derivatives as of the last command the checker did **not** flag.
        #: What a gap freezes; see :meth:`MotionLimitChecker.extrapolate`. Zero
        #: until the first clean command, which is the fallback the docstring
        #: there promises.
        self.clean_first = [0.0] * width
        self.clean_second = [0.0] * width

    def mark_clean(self) -> None:
        """Remember the current derivatives as the last unflagged ones."""
        self.clean_first = list(self.first)
        self.clean_second = list(self.second)

    def freeze_clean(self) -> None:
        """Adopt the last unflagged derivatives, keeping the applied value.

        The value stays: the reference really is where the last command put it,
        flagged or not. Only the *rates* a gap integrates are taken from data
        nobody objected to.
        """
        self.first = list(self.clean_first)
        self.second = list(self.clean_second)

    def freeze_flat_position(self) -> None:
        """Zero the acceleration, keeping value and velocity.

        For a **position**-like depth (``q_c``, ``elbow_c[0]``): ``first`` is
        velocity, ``second`` is acceleration, and :meth:`extrapolate_position`
        integrates ``first`` at frozen ``second``. Zeroing ``second`` is the
        fallback when there is no *adjacent* clean history to borrow rates
        from. It cannot run away -- there is nothing left to integrate -- and it
        does not invent a standstill the client never commanded either.

        Also the right fallback for **torque** (``tau_J_d``): torque is never
        extrapolated (:meth:`MotionLimitChecker.extrapolate` returns early for
        ``ControlMode.TORQUE``), so which field this zeroes is moot there, but
        ``second`` is the one nothing downstream ever reads either way.
        """
        self.second = [0.0] * self.width

    def freeze_flat_velocity(self) -> None:
        """Zero the acceleration, keeping value.

        For a **velocity**-like depth (``dq_c``): ``value`` is already a
        velocity and ``first`` -- not ``second`` -- is the acceleration
        :meth:`extrapolate_velocity` integrates; ``second`` there is a jerk
        nothing reads. Calling :meth:`freeze_flat_position` on this depth
        would zero the unread jerk and leave a flagged acceleration driving
        the whole gap, which is exactly the bug this method exists to not be.
        """
        self.first = [0.0] * self.width

    def seed(
        self,
        value: Sequence[float],
        first: Optional[Sequence[float]] = None,
        second: Optional[Sequence[float]] = None,
    ) -> None:
        """Set the history a motion starts from (the robot's own ``*_d`` fields)."""
        self.value = [float(item) for item in value[: self.width]]
        self.first = (
            [0.0] * self.width if first is None else [float(x) for x in first[: self.width]]
        )
        self.second = (
            [0.0] * self.width if second is None else [float(x) for x in second[: self.width]]
        )

    def derivatives(
        self, command: Sequence[float], cycles: int = 1
    ) -> Tuple[List[float], List[float], List[float]]:
        """The three backward differences ``command`` implies, without advancing.

        ``cycles`` is how many 1 ms cycles separate ``command`` from the value
        in the history; see :meth:`MotionLimitChecker.cycles_since_applied` for
        why that is not always one in a simulator.
        """
        step = cycles * DELTA_T
        first = [(command[i] - self.value[i]) / step for i in range(self.width)]
        second = [(first[i] - self.first[i]) / step for i in range(self.width)]
        third = [(second[i] - self.second[i]) / step for i in range(self.width)]
        return first, second, third

    def advance(self, command: Sequence[float], cycles: int = 1) -> None:
        """Accept ``command`` as applied: it and its derivatives become the history."""
        first, second, _ = self.derivatives(command, cycles)
        self.value = [float(command[i]) for i in range(self.width)]
        self.first = first
        self.second = second

    def rebase(self, command: Sequence[float]) -> None:
        """Accept ``command`` as a fresh standstill: derivatives reset to zero."""
        self.value = [float(command[i]) for i in range(self.width)]
        self.first = [0.0] * self.width
        self.second = [0.0] * self.width

    # -- packet-loss extrapolation ----------------------------------------
    #
    # Two laws, because this class is used at two depths. On a *position*-like
    # signal (``q_c``, ``elbow_c[0]``) ``first`` is a velocity and ``second`` an
    # acceleration, so the missed cycle is a constant-acceleration integration.
    # On a *velocity*-like signal (``dq_c``) ``first`` is already the
    # acceleration and there is nothing above it to integrate -- extending at
    # constant acceleration *is* ``value += first * dt``.
    #
    # Both freeze the highest derivative rather than continuing it. That is a
    # decision, and the reason is in the quotation this module opens with:
    # Control keeps *acceleration* constant. Integrating jerk instead turns a
    # gap into a divergence -- a previous attempt at this feature carried a
    # commanded 0.13 rad/s to 2.41 rad/s across twenty milliseconds of silence,
    # which is not an extrapolation of the client's trajectory, it is a runaway.

    def extrapolate_position(self) -> List[float]:
        """The next waypoint of a position signal, at frozen acceleration.

        Semi-implicit (symplectic) Euler, in this order: ``v_k = v_{k-1} + a
        dt`` first, then ``q_k = q_{k-1} + v_k dt``. Per axis, acceleration
        frozen.

        **Why not the trapezoidal ``q += v dt + a dt^2 / 2``.** Because this
        history is differenced with *backward* Euler, and it has to be able to
        difference its own output. The trapezoidal step advances the value by
        ``(v_{k-1} + a dt / 2) dt`` while :meth:`commit_position` stores
        ``v_{k-1} + a dt`` as the new velocity -- so the stored first difference
        is a half-step ahead of the step actually taken, every cycle. That slack
        does **not** wash out: the client resuming on its own conforming
        waypoint is differenced against a reference that is short by
        ``a dt^2 / 2`` per extrapolated cycle *and* against a velocity that is
        ``a dt / 2`` too high, and the acceleration the resume reports comes out
        at exactly ``a (gap / 2 + 1)``. A conforming 5 rad/s^2 stream aborted
        after a two-cycle gap, which is the opposite of what extrapolating is
        for.

        Ordering the update the other way makes the law self-consistent: the
        backward difference of the returned waypoint is exactly the ``v_k``
        :meth:`commit_position` stores, and its second difference is exactly the
        frozen ``a``. Nothing accumulates, and a zero-jerk stream resumes to
        precisely the acceleration it was commanding. It is still "keep
        acceleration constant and integrate" -- the same two integrals, composed
        in the order the differencing reads them back.
        """
        return [
            self.value[i] + (self.first[i] + self.second[i] * DELTA_T) * DELTA_T
            for i in range(self.width)
        ]

    def commit_position(self, value: Sequence[float]) -> None:
        """Accept an extrapolated position waypoint, advancing ``first`` by the frozen ``second``.

        Deliberately **not** :meth:`advance`: advance re-derives both
        derivatives from the sample it is given, so a chain of extrapolated
        samples would re-derive the acceleration from values that were
        themselves produced by it. That compounds -- and it is exactly how a
        previous attempt at this feature turned one duplicated datagram into a
        210x amplification. The acceleration set here is the one differenced
        from the last two *real* commands, and it stays that way for the whole
        gap.
        """
        self.value = [float(item) for item in value[: self.width]]
        self.first = [self.first[i] + self.second[i] * DELTA_T for i in range(self.width)]
        # ``second`` untouched: that is the freeze.

    def extrapolate_velocity(self) -> List[float]:
        """The next sample of a velocity signal, at frozen acceleration.

        ``dq_k = dq_{k-1} + a dt``, whose backward difference is exactly ``a``
        -- so unlike :meth:`extrapolate_position` this law has no slack against
        the differencing at all.
        """
        return [self.value[i] + self.first[i] * DELTA_T for i in range(self.width)]

    def commit_velocity(self, value: Sequence[float]) -> None:
        """Accept an extrapolated velocity sample; ``first`` is the frozen acceleration.

        ``second`` -- the jerk of a velocity generator -- goes to zero, which is
        what a constant acceleration implies and what :meth:`advance` would have
        computed. It decides nothing here (``_check_velocity`` reads only the
        first two derivatives), but leaving the pre-gap jerk in place would make
        the history describe a signal that is not the one being commanded.
        """
        self.value = [float(item) for item in value[: self.width]]
        self.second = [0.0] * self.width


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


# -- homogeneous transforms ---------------------------------------------------


def transform_matrix(values: Sequence[float]) -> "np.ndarray":
    """The column-major 16-element wire pose as a 4x4 matrix.

    ``O_T_EE``/``O_T_EE_c`` are column-major on the wire -- element ``j * 4 + i``
    is row ``i`` of column ``j``, which is why libfranka's own
    ``isHomogeneousTransformation`` indexes them that way and why its error
    message ends "Has to be column major!". ``reshape(4, 4)`` reads the flat
    array as *rows*, so the transpose is what puts it back.
    """
    return np.asarray(values, dtype=float).reshape(4, 4).T


def homogeneous_transformation_residual(values: Sequence[float]) -> float:
    """How far ``values`` is from being a rigid transform, in one number.

    The largest of the six orthonormality deviations and the four bottom-row
    deviations that :func:`is_homogeneous_transformation` tests, so a log line
    can say *how* wrong the matrix was rather than only that it was. Infinite
    for a matrix that is the wrong length or carries a non-finite entry.
    """
    transform = [float(value) for value in values]
    if len(transform) != 16 or not all(math.isfinite(value) for value in transform):
        return math.inf
    deviations = [
        abs(transform[3]),
        abs(transform[7]),
        abs(transform[11]),
        abs(transform[15] - 1.0),
    ]
    for column in range(3):
        deviations.append(abs(_norm(transform[column * 4 : column * 4 + 3]) - 1.0))
    for row in range(3):
        entries = [transform[0 * 4 + row], transform[1 * 4 + row], transform[2 * 4 + row]]
        deviations.append(abs(_norm(entries) - 1.0))
    return max(deviations)


def is_homogeneous_transformation(values: Sequence[float]) -> bool:
    """Whether ``values`` is a rigid transform, by libfranka's own test.

    A transcription of ``franka::isHomogeneousTransformation``
    (``include/franka/control_tools.h``), operating on the flat column-major
    array exactly as it does: the bottom row must be ``[0, 0, 0, 1]``, and every
    column *and* every row of the 3x3 block must have unit norm to within
    :data:`ORTHONORMAL_THRESHOLD`.

    The bottom-row comparison is libfranka's exact equality, not a tolerance,
    and is kept that way on purpose: this is a transcription, and a client whose
    matrix would fail on hardware should fail here for the same reason. A
    non-finite entry fails it too -- every comparison against NaN is false, so
    the ``!= 0.0`` and ``> threshold`` tests both reject it, which matches
    ``checkMatrix`` running ``checkFinite`` first.
    """
    transform = [float(value) for value in values]
    if len(transform) != 16:
        return False
    if not all(math.isfinite(value) for value in transform):
        return False
    if (
        transform[3] != 0.0
        or transform[7] != 0.0
        or transform[11] != 0.0
        or transform[15] != 1.0
    ):
        return False
    for column in range(3):  # column norms of the rotation block
        norm = _norm(transform[column * 4 : column * 4 + 3])
        if abs(norm - 1.0) > ORTHONORMAL_THRESHOLD:
            return False
    for row in range(3):  # row norms of the rotation block
        norm = _norm([transform[0 * 4 + row], transform[1 * 4 + row], transform[2 * 4 + row]])
        if abs(norm - 1.0) > ORTHONORMAL_THRESHOLD:
            return False
    return True


def rotation_log(rotation: "np.ndarray") -> "np.ndarray":
    """The axis-angle (rotation vector) of a 3x3 rotation matrix.

    ``SO(3)``'s log map, written out with numpy alone rather than pulled from
    scipy: this module is imported on the UDP receive path of every session and
    is deliberately dependency-light.

    Three branches, and the guards on them are the whole point of writing it by
    hand:

    * ``theta`` near zero -- the ``1 / sin(theta)`` form is 0/0, so the
      skew-symmetric part is used directly (``vee(R - R^T) / 2``, which *is*
      ``theta * axis`` to first order). This is the branch an identical pair of
      commanded rotations lands in -- a stream that is not rotating at all, and
      the one where the division would actually be undefined. A stream that *is*
      rotating lands in the ordinary branch below however slowly: one cycle at
      1 mrad/s is still 1e-6 rad, a hundred times
      :data:`ROTATION_LOG_SMALL_ANGLE`.
    * the ordinary case -- ``theta * vee(R - R^T) / (2 sin(theta))``.
    * ``theta`` within :data:`ROTATION_LOG_NEAR_PI` of pi -- ``sin(theta)``
      collapses again and the skew part vanishes with it, so the axis comes out
      of the symmetric part instead: at ``theta = pi``, ``(R + I) / 2 = a a^T``,
      so the axis is the square root of its diagonal, signs resolved off the
      column with the largest entry. This branch never recomputes a sine, which
      is why it is entered well before ``sin(theta)`` actually underflows --
      see :data:`ROTATION_LOG_NEAR_PI`.

    The trace is clamped into ``[-1, 3]`` before ``acos`` because a commanded
    matrix is only orthonormal to :data:`ORTHONORMAL_THRESHOLD`, and an
    unclamped ``acos(1 + 1e-12)`` is NaN -- which would then sail through every
    limit comparison in this module, the precise failure mode
    :meth:`MotionLimitChecker._check_finite_locked` exists to prevent.

    **The near-pi branch is conservative by construction.** ``R`` and its
    negative-axis twin are the same matrix at exactly ``theta = pi``, so the
    *sign* of the returned axis is genuinely ambiguous there and this picks one.
    That costs nothing in practice: the magnitude, which is what every limit
    comparison in this module uses, is ``pi`` either way -- and a rotation of
    ``pi`` inside one 1 ms cycle is ~3140 rad/s, a thousand times
    :data:`MAX_ROTATIONAL_VELOCITY`, so the violation is reported whichever
    branch of the sign is taken.
    """
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.clip(matrix[0, 0] + matrix[1, 1] + matrix[2, 2], -1.0, 3.0))
    theta = math.acos(float(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    skew = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
    if theta < ROTATION_LOG_SMALL_ANGLE:
        return 0.5 * skew
    if theta < math.pi - ROTATION_LOG_NEAR_PI:
        return (theta / (2.0 * math.sin(theta))) * skew
    # theta ~ pi: (R + I) / 2 == a a^T, so the diagonal holds a_i^2.
    symmetric = 0.5 * (matrix + np.eye(3))
    squares = np.clip(np.diag(symmetric), 0.0, None)
    dominant = int(np.argmax(squares))
    scale = math.sqrt(float(squares[dominant]))
    if scale <= 0.0:  # pragma: no cover - only for a matrix that is not a rotation
        return np.zeros(3)
    axis = np.asarray(symmetric[:, dominant], dtype=float) / scale
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:  # pragma: no cover - ditto
        return np.zeros(3)
    return theta * (axis / norm)


def rotation_exp(vector: Sequence[float]) -> "np.ndarray":
    """The rotation matrix of an axis-angle (rotation vector), Rodrigues' formula.

    :func:`rotation_log`'s inverse, and the only new piece of geometry
    packet-loss extrapolation needs: continuing a commanded *pose* through a gap
    means composing a rotation increment onto the last one, and an increment is
    an axis-angle vector.

    ``R = I + sin(theta) K + (1 - cos(theta)) K^2`` for ``K = skew(axis)``. That
    is the branch every *rotating* extrapolation lands in, and it is exact: one
    millisecond at the FR3's ``kMaxRotationalVelocity`` is 2.5 mrad, five orders
    of magnitude above :data:`ROTATION_LOG_SMALL_ANGLE`.

    The small-angle branch returns ``I + skew(vector)``, the first-order
    truncation, for the same reason :func:`rotation_log` has one: below
    :data:`ROTATION_LOG_SMALL_ANGLE` the division by ``theta`` that recovers the
    axis is 0/0. It is **not** dead code and must not be deleted: the increment
    is exactly zero for every gap in a stream that commands no rotation at all
    -- a pure-translation pose motion, or any pose stream at a standstill, which
    is the common case -- and ``skew / 0.0`` there is a matrix of NaN that would
    poison the pose history for the rest of the motion. What it is *not* is a
    branch a conforming rotating stream ever reaches; the deviation from a true
    rotation it carries (``||R^T R - I|| ~ theta^2`` <= 1e-16) is therefore
    bounded by that, far below :data:`ORTHONORMAL_THRESHOLD`.

    A mutation test that turns this branch's ``np.eye(3) + skew`` into a bare
    ``np.eye(3)`` -- dropping the linear term entirely -- is an *equivalent*
    mutant here, not a surviving one worth chasing: below
    :data:`ROTATION_LOG_SMALL_ANGLE` (``1e-8``), ``skew``'s largest entry is
    itself below ``1e-8``, so the two return values differ by less than a
    double's precision can distinguish at ``1.0`` and no test built on
    :data:`ORTHONORMAL_THRESHOLD` or wire-level assertions can tell them apart.
    Do not add a test aimed at killing it.
    """
    values = np.asarray(vector, dtype=float)
    theta = float(np.linalg.norm(values))
    skew = np.array(
        [
            [0.0, -values[2], values[1]],
            [values[2], 0.0, -values[0]],
            [-values[1], values[0], 0.0],
        ]
    )
    if theta < ROTATION_LOG_SMALL_ANGLE:
        return np.eye(3) + skew
    axis_skew = skew / theta
    return (
        np.eye(3)
        + math.sin(theta) * axis_skew
        + (1.0 - math.cos(theta)) * (axis_skew @ axis_skew)
    )


class _CartesianDifferentiator:
    """The same, for a 6-element twist split into translation and rotation.

    ``limitRate`` treats ``O_dP_EE_c`` as two ``Eigen::Vector3d`` and compares
    *norms*, not components (``src/rate_limiting.cpp:184-195`` dispatching into
    the anonymous-namespace overload at ``:13-55``), so this differences the
    whole twist and reports norms per half.
    """

    def __init__(self):
        """Start from a standstill twist with zero acceleration."""
        self.value = [0.0] * 6
        self.first = [0.0] * 6
        #: See :attr:`_Differentiator.clean_first`.
        self.clean_first = [0.0] * 6

    def mark_clean(self) -> None:
        """Remember the current acceleration as the last unflagged one."""
        self.clean_first = list(self.first)

    def freeze_clean(self) -> None:
        """Adopt the last unflagged acceleration, keeping the applied twist."""
        self.first = list(self.clean_first)

    def freeze_flat(self) -> None:
        """Zero the twist acceleration; see :meth:`_Differentiator.freeze_flat`."""
        self.first = [0.0] * 6

    def seed(self, value: Sequence[float], first: Optional[Sequence[float]] = None) -> None:
        """Set the twist (and its acceleration) a motion starts from."""
        self.value = [float(item) for item in value[:6]]
        self.first = [0.0] * 6 if first is None else [float(item) for item in first[:6]]

    def derivatives(
        self, command: Sequence[float], cycles: int = 1
    ) -> Tuple[List[float], List[float]]:
        """Acceleration and jerk implied by ``command``, without advancing."""
        step = cycles * DELTA_T
        acceleration = [(command[i] - self.value[i]) / step for i in range(6)]
        jerk = [(acceleration[i] - self.first[i]) / step for i in range(6)]
        return acceleration, jerk

    def advance(self, command: Sequence[float], cycles: int = 1) -> None:
        """Accept ``command`` as applied."""
        acceleration, _ = self.derivatives(command, cycles)
        self.value = [float(command[i]) for i in range(6)]
        self.first = acceleration

    def extrapolate(self) -> List[float]:
        """The next twist of a missed cycle, at frozen twist-acceleration.

        Component-wise ``V_k = V_{k-1} + A dt``, the twist twin of
        :meth:`_Differentiator.extrapolate_velocity`. Per component and not per
        norm: the limits are compared as norms (see
        :meth:`MotionLimitChecker._check_cartesian_halves`) but the *signal* is
        six independent numbers, and rescaling a norm would rotate the commanded
        direction.
        """
        return [self.value[i] + self.first[i] * DELTA_T for i in range(6)]

    def commit(self, command: Sequence[float]) -> None:
        """Accept an extrapolated twist; the acceleration stays frozen.

        Not :meth:`advance`, for the reason spelled out in
        :meth:`_Differentiator.commit_position`: re-deriving the acceleration
        from extrapolated samples compounds.
        """
        self.value = [float(command[i]) for i in range(6)]


class _PoseDifferentiator:
    """Backward-Euler differences of a commanded *pose*, ``O_T_EE_c``.

    The same three formulas as :class:`_Differentiator`, but the value being
    differenced is a rigid transform rather than a number, so the first
    difference is taken in two pieces:

    * **translation** -- the plain backward difference of the position column,
      which is a linear velocity in m/s;
    * **rotation** -- the axis-angle of ``R_{k-1}^T R_k`` divided by the
      interval, which is an angular velocity in rad/s. Composing the *relative*
      rotation and taking its log is the only difference that means anything for
      a rotation: subtracting two rotation matrices, or two of their log-maps,
      is not an angular velocity at all once the rotations are large.

    From there acceleration and jerk are ordinary differences of the resulting
    6-vector, exactly as :class:`_CartesianDifferentiator` does for a commanded
    twist -- and the halves are compared as *norms* against
    ``kMaxTranslational*`` / ``kMaxRotational*`` for the same reason: that is how
    ``limitRate`` treats them (``src/rate_limiting.cpp:13-55``).

    The angular velocity is expressed in the *previous* commanded frame. Nothing
    here depends on that choice: every comparison is on the norm, which the
    frame does not change.
    """

    def __init__(self):
        """Start at the identity pose, at rest."""
        self.rotation = np.eye(3)
        self.translation = np.zeros(3)
        self.first = [0.0] * 6
        self.second = [0.0] * 6
        #: See :attr:`_Differentiator.clean_first`.
        self.clean_first = [0.0] * 6
        self.clean_second = [0.0] * 6

    def mark_clean(self) -> None:
        """Remember the current twist derivatives as the last unflagged ones."""
        self.clean_first = list(self.first)
        self.clean_second = list(self.second)

    def freeze_clean(self) -> None:
        """Adopt the last unflagged derivatives, keeping the applied pose."""
        self.first = list(self.clean_first)
        self.second = list(self.clean_second)

    def freeze_flat(self) -> None:
        """Zero the twist acceleration; see :meth:`_Differentiator.freeze_flat`."""
        self.second = [0.0] * 6

    def seed(self, pose: Sequence[float]) -> None:
        """Set the pose a motion starts from; derivatives start at zero."""
        matrix = transform_matrix(pose)
        # Explicit copies: transform_matrix's np.asarray does not copy an
        # ndarray input, so these slices would otherwise be views into
        # whatever array the caller passed (e.g. a backend's own state
        # buffer), silently changing this history if that buffer is mutated
        # in place later.
        self.rotation = np.array(matrix[:3, :3])
        self.translation = np.array(matrix[:3, 3])
        self.first = [0.0] * 6
        self.second = [0.0] * 6

    def derivatives(
        self, pose: Sequence[float], cycles: int = 1
    ) -> Tuple[List[float], List[float], List[float]]:
        """Velocity, acceleration and jerk implied by ``pose``, without advancing."""
        step = cycles * DELTA_T
        matrix = transform_matrix(pose)
        linear = (matrix[:3, 3] - self.translation) / step
        angular = rotation_log(self.rotation.T @ matrix[:3, :3]) / step
        velocity = [float(value) for value in (*linear, *angular)]
        acceleration = [(velocity[i] - self.first[i]) / step for i in range(6)]
        jerk = [(acceleration[i] - self.second[i]) / step for i in range(6)]
        return velocity, acceleration, jerk

    def advance(self, pose: Sequence[float], cycles: int = 1) -> None:
        """Accept ``pose`` as applied: it and its derivatives become the history."""
        velocity, acceleration, _ = self.derivatives(pose, cycles)
        matrix = transform_matrix(pose)
        # Explicit copies -- see the comment in seed().
        self.rotation = np.array(matrix[:3, :3])
        self.translation = np.array(matrix[:3, 3])
        self.first = velocity
        self.second = acceleration

    def rebase(self, pose: Sequence[float]) -> None:
        """Accept ``pose`` as a fresh standstill: derivatives reset to zero."""
        self.seed(pose)

    def extrapolate(self) -> List[float]:
        """The next commanded pose of a missed cycle, at frozen acceleration.

        Translation follows the same semi-implicit law every other position-like
        signal here does, per axis: ``v_k = v + a dt`` first, then
        ``p_k = p_{k-1} + v_k dt``. See
        :meth:`_Differentiator.extrapolate_position` for why the order is what
        keeps a gap from accumulating slack against the backward differencing.

        **The rotation composition, which is a choice.** The increment is the
        axis-angle vector ``theta = (omega + alpha dt) dt`` -- the rotational
        half of the same semi-implicit integral -- applied on the *right*:
        ``R_k = R_{k-1} exp(skew(theta))``. Right-multiplication is not
        arbitrary: :meth:`derivatives` recovers an angular velocity as
        ``log(R_{k-1}^T R_k) / dt``, i.e. in the previous commanded frame, so
        composing on the right is precisely the inverse of the differencing this
        class already does. Feed the result straight back into
        :meth:`derivatives` and the angular velocity that comes out is exactly
        the ``omega + alpha dt`` :meth:`commit` stores, matching the
        translational half -- so a pose gap resumes clean with a non-zero
        rotational acceleration, not only with a zero one.

        Integrating an axis-angle increment per cycle is a first-order
        approximation of the exact rigid motion whenever ``omega`` and ``alpha``
        are not parallel, since ``SO(3)`` does not commute. The error is
        ``O(|omega| |alpha| dt^3)``: at the FR3's own rotational limits that is
        ~4e-8 rad per cycle, ~8e-7 rad across the whole 19-cycle extrapolation
        window. Below the orthonormality threshold this module tests matrices
        against, so it cannot even be observed on the wire.
        """
        velocity = np.asarray(self.first[:3], dtype=float)
        omega = np.asarray(self.first[3:6], dtype=float)
        acceleration = np.asarray(self.second[:3], dtype=float)
        alpha = np.asarray(self.second[3:6], dtype=float)
        # The velocity half-step first -- this is the semi-implicit order, and
        # it is the same 6-vector ``commit`` is about to store.
        linear = velocity + acceleration * DELTA_T
        angular = omega + alpha * DELTA_T
        matrix = np.eye(4)
        matrix[:3, :3] = self.rotation @ rotation_exp(angular * DELTA_T)
        matrix[:3, 3] = self.translation + linear * DELTA_T
        return [float(value) for value in matrix.T.flatten()]

    def commit(self, pose: Sequence[float]) -> None:
        """Accept an extrapolated pose; the 6-vector acceleration stays frozen.

        Not :meth:`advance`, for the reason in
        :meth:`_Differentiator.commit_position`. The velocity advances by the
        frozen acceleration -- linear and angular alike, the angular one being
        the body-frame rate the next increment is built from.
        """
        matrix = transform_matrix(pose)
        self.rotation = np.array(matrix[:3, :3])
        self.translation = np.array(matrix[:3, 3])
        self.first = [self.first[i] + self.second[i] * DELTA_T for i in range(6)]


# -- the checker --------------------------------------------------------------


class AbsorbedCommand(NamedTuple):
    """What :meth:`MotionLimitChecker.absorb_command` did with one datagram.

    * ``violation`` -- what the command broke, or None. Always the checker's
      full verdict, whether or not enforcement acted on it.
    * ``accepted`` -- whether the caller should dispatch it to the simulator.
      False only for a violation enforcement refuses (or a non-finite one,
      which is refused either way).
    * ``recorded`` -- whether it entered the differencing history, i.e. whether
      it is the baseline the next command is measured against.
    * ``rewound`` -- whether it was the real answer to a cycle the publish loop
      had already extrapolated, and that guess was thrown away for it.
    """

    violation: Optional[Violation]
    accepted: bool
    recorded: bool
    rewound: bool


class MotionLimitChecker:
    """Validates each received command against the FCI's motion limits.

    One instance per FCI session, mirroring :class:`
    franka_sim.comm_constraints.CommConstraintTracker`: :meth:`start_motion` on
    ``Move``, :meth:`end_motion` when the motion ends however it ends,
    :meth:`recover` on ``AutomaticErrorRecovery``.

    Two threads touch it -- the UDP receive thread checks and records received
    commands, the state-publish thread reports missed cycles -- so every method
    goes through one short leaf lock, the same discipline the comm tracker
    uses.
    """

    def __init__(
        self,
        *,
        enforce: bool = False,
        start_pose_tolerance: float = START_POSE_TOLERANCE,
        start_velocity_tolerance: float = START_VELOCITY_TOLERANCE,
        measured_velocity_margin: float = MEASURED_JOINT_VELOCITY_MARGIN,
        start_cartesian_translation_tolerance: float = (
            START_CARTESIAN_POSE_TRANSLATION_TOLERANCE
        ),
        start_cartesian_rotation_tolerance: float = START_CARTESIAN_POSE_ROTATION_TOLERANCE,
        start_elbow_tolerance: float = START_ELBOW_TOLERANCE,
    ):
        """Build a checker; see the module constants for the defaults."""
        self._lock = threading.Lock()
        #: Whether a violation may abort the motion. Checking and logging run
        #: either way -- that is the point of making this opt-in.
        self.enforce = enforce
        self.start_pose_tolerance = start_pose_tolerance
        self.start_velocity_tolerance = start_velocity_tolerance
        self.measured_velocity_margin = measured_velocity_margin
        self.start_cartesian_translation_tolerance = start_cartesian_translation_tolerance
        self.start_cartesian_rotation_tolerance = start_cartesian_rotation_tolerance
        self.start_elbow_tolerance = start_elbow_tolerance

        self._mode: ControlMode = ControlMode.NONE
        self._joint = _Differentiator(7)
        self._torque = _Differentiator(7)
        self._twist = _CartesianDifferentiator()
        self._pose = _PoseDifferentiator()
        #: ``elbow_c[0]`` -- the redundancy angle -- differenced as a position,
        #: so ``first``/``second`` are its velocity and acceleration.
        self._elbow = _Differentiator(1)
        #: ``elbow_c[1]`` as last commanded: the +-1 branch flag whose
        #: mid-motion flip is ``cartesian_motion_generator_elbow_sign_inconsistent``.
        #: None until this motion's first elbow-carrying command.
        self._elbow_sign: Optional[float] = None
        #: The robot's own ``O_T_EE`` when the motion was armed, which is what a
        #: Cartesian pose motion's first command is judged against. None when
        #: the caller's state snapshot carried no pose at all, in which case the
        #: start-pose check is skipped rather than guessed at.
        self._start_pose: Optional[List[float]] = None
        #: Reference configuration for the position-dependent velocity limits.
        self._joint_positions = [0.0] * 7
        #: Last *measured* joint velocity the server handed over, or None when
        #: none has been seen yet in this motion. The safety controller's input;
        #: see :meth:`check_measured_velocity`.
        self._measured_velocity: Optional[List[float]] = None
        self._active = False
        #: Whether *a motion* is running, as opposed to whether a motion
        #: generator this module differences is running (:attr:`_active`). The
        #: safety controller is not a motion-generator check -- it watches the
        #: arm, not the command -- so it stays armed for generator modes this
        #: server accepts but does not check, and in pure torque control where
        #: there is no motion generator at all.
        self._safety_active = False
        self._first_command = True
        self._violated = False
        #: Caller-supplied token for the running motion; see :meth:`start_motion`.
        self._motion_id = 0
        #: ``message_id`` of the state the applied history currently sits at;
        #: see :meth:`cycles_since_applied`.
        self._applied_id: Optional[int] = None
        #: Newest ``message_id`` the server has published, as reported by
        #: :meth:`note_published`. None until the server reports one, which is
        #: what a caller driving the checker directly looks like. It is *not*
        #: cleared between motions: it is a fact about the wire, not about the
        #: motion, and the publish loop keeps counting across a Move.
        self._published_id: Optional[int] = None
        #: Error indices already logged in this motion, so a client that steps
        #: every cycle produces one warning, not a thousand.
        self._logged: Set[int] = set()
        #: The last command :meth:`record` accepted, kept whole. Packet-loss
        #: extrapolation substitutes a command for one the client never sent,
        #: and everything in a ``RobotCommand`` that is *not* the extrapolated
        #: generator channel has to keep its last real value -- ``valid_elbow``
        #: above all, which is the client's own statement that this motion
        #: commands an elbow at all. See :meth:`extrapolate`.
        self._last_command: Optional[Dict[str, Any]] = None
        #: Latch for the "could not extrapolate" warning, which sits on a 1 kHz
        #: path and describes a condition that cannot heal within a motion.
        self._extrapolation_refused = False
        #: History as it stood the instant before the still-unanswered part of
        #: the current run of missed cycles was extrapolated, with the ids that
        #: part covers. None while no gap is open. Re-taken every time a late
        #: datagram is absorbed, because a run of losses stays rewindable for as
        #: long as datagrams for it can still turn up: it is closed by a *fresh*
        #: command, not by the first late one. See :meth:`rewind_extrapolation`.
        self._gap_snapshot: Optional[Dict[str, Any]] = None
        self._gap_first_id: Optional[int] = None
        self._gap_last_id: Optional[int] = None
        #: ``(message_id, violated)`` of the last command :meth:`check` judged,
        #: so :meth:`record` knows whether what it is about to accept was
        #: flagged. Only consulted when the caller does not say outright; see
        #: ``clean`` there.
        self._last_verdict: Optional[Tuple[int, bool]] = None
        #: True when the last recorded command was one the checker flagged. The
        #: next extrapolation freezes its derivatives from the last *clean*
        #: history instead of from that command's; see :meth:`extrapolate`.
        self._freeze_from_clean = False
        #: How many records ago the last clean one was: 0 when the last record
        #: was clean, 1 when exactly one flagged record sits on top of it, and
        #: None when this motion has recorded no clean command at all. Decides
        #: which fallback :meth:`_freeze_clean_locked` takes.
        self._clean_age: Optional[int] = None
        #: Set by a bare :meth:`rewind_extrapolation`, cleared by the
        #: :meth:`record` that follows it -- or by the one :meth:`extrapolate`
        #: it makes hold. See :meth:`rewind_extrapolation`.
        self._rewind_in_flight = False

    # -- lifecycle ---------------------------------------------------------

    def start_motion(
        self, control_mode: ControlMode, state: Dict[str, Any], motion_id: int = 0
    ) -> None:
        """Arm a motion, seeding the history from the robot's own state.

        Exactly what Control differentiates against: ``q_d``, ``dq_d``,
        ``ddq_d`` and ``tau_J_d`` are "always sent back to the user in the robot
        state" for precisely this purpose (``docs/overview.rst``), so seeding
        from them is seeding from what the client was told.

        ``motion_id`` is the caller's session token; it gates :meth:`end_motion`
        so a stale end cannot switch off a motion that has already replaced this
        one, and it lets the caller check that an abort still applies to the
        motion that violated. The latched violation is re-armed here, the same
        way :meth:`franka_sim.comm_constraints.CommConstraintTracker.start_motion`
        re-arms its own: enforcement used to fire at most once per *connection*.
        The server refuses a ``Move`` outright while a violation is latched, so
        reaching this method at all means the client has recovered.
        """
        with self._lock:
            self._mode = control_mode
            self._active = control_mode not in (ControlMode.NONE, None)
            # Armed for every accepted Move, including the generator modes this
            # server does not check: the safety controller watches the arm.
            self._safety_active = True
            self._measured_velocity = None
            self._motion_id = motion_id
            self._first_command = True
            self._applied_id = None
            self._violated = False
            self._logged = set()
            self._last_command = None
            self._extrapolation_refused = False
            self._gap_snapshot = None
            self._gap_first_id = None
            self._gap_last_id = None
            self._last_verdict = None
            self._freeze_from_clean = False
            self._clean_age = None
            self._rewind_in_flight = False
            positions = list(state.get("q_d") or state.get("q") or [0.0] * 7)
            self._joint_positions = [float(value) for value in positions[:7]]
            if control_mode is ControlMode.POSITION:
                self._joint.seed(positions, state.get("dq_d"), state.get("ddq_d"))
            elif control_mode is ControlMode.VELOCITY:
                self._joint.seed(state.get("dq_d") or [0.0] * 7, state.get("ddq_d"))
            self._torque.seed(state.get("tau_J_d") or [0.0] * 7)
            self._twist.seed(state.get("O_dP_EE_d") or [0.0] * 6)
            # The *measured* flange pose, not ``O_T_EE_d``. The server does now
            # publish the commanded-pose fields faithfully (see
            # docs/robot-state.md), and between motions they are exactly this
            # value -- but the check asks "are you where the robot is", which is
            # what the smoke suite's 10 m offset breaks, and judging a command
            # against another command would let a stale one excuse itself. A
            # snapshot without a pose -- a bare hold setpoint, a mocked backend
            # -- leaves it None and the check is skipped rather than run against
            # a guess.
            start_pose = state.get("O_T_EE")
            self._start_pose = (
                None if start_pose is None else [float(value) for value in start_pose]
            )
            if self._start_pose is not None:
                self._pose.seed(self._start_pose)
            else:
                self._pose = _PoseDifferentiator()
            # ``elbow[0]`` is joint 3's angle on an FR3, so the robot's own
            # elbow is (q[2], sign(q[3])) and there is nothing to look up.
            self._elbow.seed([self._joint_positions[2]])
            self._elbow_sign = None
            # The robot's own reported derivatives are clean by construction:
            # nothing has been commanded yet for the checker to object to.
            self._mark_clean_locked()

    def end_motion(self, motion_id: Optional[int] = None) -> None:
        """The motion is over. The latched violation survives, as the robot's does.

        ``motion_id`` names the motion the caller believes is ending; a token
        that no longer matches the running one is a stale event and is ignored.
        ``None`` means "whatever is running", which is what the unconditional
        teardown paths want.
        """
        with self._lock:
            if motion_id is not None and motion_id != self._motion_id:
                return
            self._active = False
            self._safety_active = False
            self._measured_velocity = None
            self._motion_id = 0
            self._first_command = True

    def recover(self) -> None:
        """Clear a latched violation, as ``AutomaticErrorRecovery`` does."""
        with self._lock:
            self._violated = False
            self._active = False
            self._safety_active = False
            self._measured_velocity = None
            self._motion_id = 0
            self._first_command = True
            self._logged = set()

    # -- cycle accounting --------------------------------------------------

    def note_published(self, published_id: int) -> None:
        """Tell the checker which state the server has just published.

        Called once per control cycle from the state-publish thread, which is
        what turns the differencing interval from *the client's claim* into
        *the server's own observation*. See :meth:`cycles_since_applied`.
        """
        with self._lock:
            self._published_id = int(published_id)

    def cycles_since_applied(self, command: Dict[str, Any]) -> int:
        """How many 1 ms cycles separate ``command`` from the applied history.

        On hardware this is always one: Control applies exactly one command per
        cycle and extrapolates the ones that never arrive, so ``q_{c,k-1}`` is
        always precisely one millisecond old. A simulator breaks that in a way
        the robot does not. A command that did not answer the cycle it arrived
        in -- because the receive path was behind, or because the client was --
        is still applied, but it is not recorded (see :meth:`check`), so the
        next command the history *does* record sits several state cycles away
        from it.

        A *missed* cycle is not one of those cases any more:
        :meth:`extrapolate` records a substitute waypoint for it, so the history
        keeps moving through a gap and the resumed command is one cycle from it
        exactly as on hardware.

        Dividing a two-cycle step by one millisecond would report double the
        velocity and a large acceleration for a *conforming* client, which is a
        sim artifact and not a limit the robot would have flagged. So the
        difference is taken over the real interval instead.

        **The interval is the server's own, not the client's.** Both ends of it
        are ids the server itself published: the ``message_id`` the applied
        history sits at, and that of the state this command answers. The
        client's echo is honoured only up to the newest state the server has
        actually published (:meth:`note_published`) -- an id ahead of that
        answers nothing, and honouring it would let a client inflate the
        denominator of the server's own check until 50 rad/s steps sailed
        through. Never larger than the server observed, in other words.

        Without a :meth:`note_published` -- a mocked backend, a unit test, a
        caller that drives the checker directly -- there is no observation to
        bound the echo with, and the interval falls back to the client's claim
        capped at :data:`MAX_COALESCED_CYCLES`.
        """
        with self._lock:
            return self._cycles_since_applied_locked(command)

    #: Which command fields each control mode actually reads. Anything else in
    #: the datagram is not this motion's signal and is not the client's problem.
    _CHECKED_FIELDS = {
        ControlMode.POSITION: (("q_c", JOINT_MOTION_GENERATOR_POSITION_LIMITS_VIOLATION_INDEX),),
        ControlMode.VELOCITY: (("dq_c", JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX),),
        ControlMode.TORQUE: (("tau_J_d", TAU_J_RANGE_VIOLATION_INDEX),),
        ControlMode.STEERING_DRIVE: (
            ("O_dP_EE_c", CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX),
        ),
        ControlMode.CARTESIAN_VELOCITY: (
            ("O_dP_EE_c", CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX),
            ("elbow_c", CARTESIAN_MOTION_GENERATOR_ELBOW_LIMIT_VIOLATION_INDEX),
        ),
        # A non-finite pose is the *invalid frame* condition, not a limits
        # violation: libfranka's ``checkMatrix`` runs ``checkFinite`` and the
        # homogeneity test together and throws one exception for both
        # (``include/franka/control_tools.h``), so they get one name here too.
        ControlMode.CARTESIAN_POSE: (
            ("O_T_EE_c", CARTESIAN_POSITION_MOTION_GENERATOR_INVALID_FRAME_INDEX),
            ("elbow_c", CARTESIAN_MOTION_GENERATOR_ELBOW_LIMIT_VIOLATION_INDEX),
        ),
    }

    def _check_finite_locked(self, command: Dict[str, Any]) -> Optional[Violation]:
        """Reject NaN and infinity before any limit is compared.

        Comparisons against NaN are all False, so a NaN sails through every
        ``value > limit`` test in this module, poisons the backward differences
        it is then recorded into (NaN minus anything is NaN, for the rest of the
        motion) and reaches both the physics backend and the wire.

        libfranka refuses to send one at all -- its own command filter throws
        ``std::invalid_argument("lowpass-filter: current or last signal value is
        infinite or NaN.")`` (``src/lowpass_filter.cpp``) -- so a non-finite
        command can only come from a client that is not libfranka. There is no
        ``Error`` enumerator for it, so it is reported as the limits violation
        of the generator it arrived in, which is what a value of infinite
        magnitude is. Marked :attr:`Violation.fatal`, so it is refused whether
        or not enforcement is on.
        """
        # Dispatched on the checker's own ``_mode``, which the ``Move`` handler
        # set, while ``_dispatch_control_command`` reads the mode pair out of
        # ``robot_state``. The two agree except for the sub-millisecond window
        # in which a ``Move`` is being handled, where the checker may still be
        # looking at the previous generator's field. Harmless here: the fields
        # are all present in every datagram, and the fallback below checks the
        # lot when there is no mode at all.
        fields = self._CHECKED_FIELDS.get(self._mode)
        if fields is None:
            # No motion (or one this server does not serve): check everything a
            # dispatcher could plausibly reach for rather than nothing.
            fields = tuple(
                pair for pairs in self._CHECKED_FIELDS.values() for pair in pairs
            )
        for field, error_index in fields:
            values = command.get(field)
            if values is None:
                continue
            for axis, value in enumerate(values):
                if not math.isfinite(value):
                    return Violation(
                        error_index=error_index,
                        signal=field,
                        axis=f"[{axis}]",
                        value=value,
                        limit=math.inf,
                        fatal=True,
                    )
        return None

    def _cycles_since_applied_locked(self, command: Dict[str, Any]) -> int:
        if self._applied_id is None:
            return 1
        claimed = int(command.get("message_id", 0)) - self._applied_id
        if self._published_id is None:
            # Nobody told us what was published, so the client's echo is the
            # only interval on offer and it is not a trustworthy one.
            return max(1, min(MAX_COALESCED_CYCLES, claimed))
        # The server's own observation: how many states it has published since
        # the one the applied history sits at. The echo may name any state up
        # to that -- it cannot name a later one, because no later one exists.
        observed = self._published_id - self._applied_id
        return max(1, min(claimed, observed))

    # -- checking ----------------------------------------------------------

    def check(
        self,
        command: Dict[str, Any],
        joint_positions: Optional[Sequence[float]] = None,
        *,
        fresh: bool = True,
    ) -> Optional[Violation]:
        """Validate one *received* command; the first violation found, or None.

        ``joint_positions`` is the configuration the position-dependent joint
        velocity limits are evaluated at -- the measured ``q``, which is what
        libfranka's own callers pass ``getUpperJointVelocityLimits``. Omitted,
        the last applied commanded position is used, which is where the robot
        is heading anyway.

        **Every command that will be applied is checked, in full.** There is no
        category of command that reaches the simulator unexamined -- that was a
        hole, and a 40 rad/s step walked through it behind a stale echo.

        ``fresh`` is False for a datagram that did not answer the cycle it
        arrived in -- a duplicate, a replay, a reordered packet, or one the
        receive path only got to a cycle late. It is still applied (see
        ``FrankaSimServer._handle_commands``), so it is still judged, and it is
        never *recorded*, so it leaves the history alone. What it is differenced
        over is the same server-observed interval every other command gets:
        the state it answers is one the server published, and the distance from
        there back to the applied history is a fact the server owns. Forcing it
        to a single cycle was over-strict in exactly the case that matters --
        a command two cycles ahead of the history read as twice its own
        velocity, and the acceleration that implied aborted conforming clients.
        A replay or a reordered packet still gets one cycle, because its id is
        no *newer* than the history's and the interval floors at one. So the
        flag no longer changes what this method computes; it is kept because
        the caller has to make the same distinction for :meth:`record`, and
        because saying "this datagram did not answer its cycle" at the call
        site is worth more than the one line it costs.

        There is **no grace cycle**. A command that follows a packet gap is
        differenced over the interval the server observed between them (see
        :meth:`cycles_since_applied`) -- so a genuine gap, however long, is
        measured at the rate the client actually commanded and passes. Skipping
        the differential checks for that command instead was a hole of its own:
        the resume waypoint could be anywhere in the joint range, and a
        full-range teleport reached physics with the checker reporting no
        violation at all.

        A gap does not stop the history moving, either. Every missed cycle is
        extrapolated and recorded (:meth:`extrapolate`), so the interval back to
        the applied history is one cycle again the moment the client resumes --
        and what the resumed command is differenced against is where the
        *reference* got to, not where the client last spoke. A client that
        resumes from its own last waypoint therefore commands a step backwards
        the size of the whole gap, and is told so. That is not a sim artifact;
        it is the hazard libfranka warns about in so many words -- intermittent
        drops "could trigger `discontinuity` errors even when your source
        signals conform with the interface specification"
        (``docs/overview.rst``).

        Records nothing: a caller that decides to reject the command leaves no
        trace of it in the history. It does update the reference configuration
        the position-dependent velocity limits are evaluated at, which is the
        one piece of state it owns -- so this is idempotent rather than pure.
        """
        with self._lock:
            return self._check_public_locked(command, joint_positions)

    def _check_public_locked(
        self,
        command: Dict[str, Any],
        joint_positions: Optional[Sequence[float]] = None,
    ) -> Optional[Violation]:
        """The body of :meth:`check`; lock held.

        Split out so :meth:`absorb_command` can rewind, check and record a
        datagram without ever releasing the lock in between.
        """
        non_finite = self._check_finite_locked(command)
        if non_finite is not None:
            self._note_verdict_locked(command, non_finite)
            return non_finite
        if not self._active:
            return None
        if joint_positions is not None:
            # A non-finite *measured* position is the simulator's problem,
            # not the client's. Substituting zero used to put joint 4
            # outside its range, which drives its velocity limit to 0.0 and
            # aborts the client's next perfectly good command; keeping the
            # previous reference configuration blames nobody.
            candidate = [float(value) for value in joint_positions[:7]]
            if all(math.isfinite(value) for value in candidate):
                self._joint_positions = candidate
            else:
                logger.debug(
                    "Ignoring a non-finite measured joint position for the "
                    "velocity-limit reference configuration"
                )

        cycles = self._cycles_since_applied_locked(command)
        violation = self._check_locked(command, cycles)
        self._note_verdict_locked(command, violation)
        return violation

    def _note_verdict_locked(
        self, command: Dict[str, Any], violation: Optional[Violation]
    ) -> None:
        """Remember what this command was judged, for :meth:`record`; lock held."""
        self._last_verdict = (int(command.get("message_id", 0)), violation is not None)

    def _mark_clean_locked(self) -> None:
        """Take the current derivatives as the last unflagged ones; lock held.

        All five histories, not only the running generator's: the mode can
        change between motions and a shadow that is stale for the *other*
        generator is a trap for the next one.
        """
        self._joint.mark_clean()
        self._torque.mark_clean()
        self._twist.mark_clean()
        self._pose.mark_clean()
        self._elbow.mark_clean()

    def _freeze_clean_locked(self) -> None:
        """Seed the coming gap from derivatives nobody objected to; lock held.

        Two cases, and the difference is how far back the clean history is.

        * The record before the flagged one was clean -- the duplicated or
          reordered datagram, which is the whole hazard: one bad sample sits
          between the gap and a perfectly good history exactly one cycle older.
          Borrow that history's velocity *and* acceleration. It is one cycle
          stale, which is no staler than any frozen derivative already is.
        * Anything else (no clean record in this motion at all, or several
          flagged ones in a row -- a client running outside a limit on purpose
          with enforcement off, which is a supported way to use this sim).
          Borrowing from a history dozens of cycles old would be its own
          divergence, so the velocity the client actually commanded is kept and
          only the acceleration is zeroed. Nothing is amplified either way, and
          that is the property that matters here.
        """
        if self._clean_age == 1:
            self._joint.freeze_clean()
            self._torque.freeze_clean()
            self._twist.freeze_clean()
            self._pose.freeze_clean()
            self._elbow.freeze_clean()
            return
        # ``_joint`` is shared between two depths (see the class docstring):
        # a joint-position motion keeps its acceleration in ``second``, a
        # joint-velocity motion keeps it in ``first``. Freezing the wrong one
        # is a no-op that leaves a flagged acceleration driving the gap --
        # see :meth:`_Differentiator.freeze_flat_velocity`.
        if self._mode is ControlMode.VELOCITY:
            self._joint.freeze_flat_velocity()
        else:
            self._joint.freeze_flat_position()
        self._torque.freeze_flat_position()
        self._twist.freeze_flat()
        self._pose.freeze_flat()
        self._elbow.freeze_flat_position()

    def _check_locked(self, command: Dict[str, Any], cycles: int) -> Optional[Violation]:
        """Dispatch one command to its generator's checks; lock held, ``_active`` true.

        Shared by :meth:`check`, which judges what a client sent, and
        :meth:`extrapolate`, which judges what the FCI substituted for what a
        client did *not* send. Both have to reach the identical verdict, and the
        surest way to guarantee that is for there to be one implementation --
        the extrapolation is not a privileged signal, and libfranka is explicit
        that it is not: intermittent drops "could trigger `discontinuity` errors
        even when your source signals conform with the interface specification"
        (``docs/overview.rst``), which can only be true if the extrapolated
        waypoints are checked like any other.
        """
        # Exhaustive by construction: every caller returns early when
        # ``_active`` is False, which it is for NONE (and for None), so whatever
        # reaches here is one of the
        # six modes with a generator behind it. **Each branch judges only the
        # signal its own generator owns**, and the gate that guarantees that
        # is on the server side: ``start_motion`` is handed the accepted
        # ``Move``'s own generator mode rather than whatever ``control_mode``
        # the previous motion left behind. Judging the zero-filled ``q_c`` of
        # a Cartesian ``RobotCommand`` as a joint position command used to
        # abort real clients with
        # ``joint_motion_generator_position_limits_violation``, because joint
        # 4's range does not contain 0 -- and, symmetrically, judging the
        # zero-filled ``O_T_EE_c`` of a *joint* command would abort them with
        # ``cartesian_position_motion_generator_invalid_frame_flag``, because
        # an all-zeros matrix is not a transform. See
        # :meth:`FrankaSimServer.handle_move_command`.
        #
        # ``CARTESIAN_POSE`` and ``CARTESIAN_VELOCITY`` are *checking-only*
        # modes: no physics backend is ever put into them, so the arm stays
        # inert for the whole motion while its commanded stream is judged in
        # full. See :meth:`_check_cartesian_pose`.
        if self._mode is ControlMode.POSITION:
            violation = self._check_position(command, cycles)
        elif self._mode is ControlMode.VELOCITY:
            violation = self._check_velocity(command, cycles)
        elif self._mode is ControlMode.TORQUE:
            violation = self._check_torque(command, cycles)
        elif self._mode is ControlMode.CARTESIAN_POSE:
            violation = self._check_cartesian_pose(command, cycles)
        else:  # ControlMode.STEERING_DRIVE / ControlMode.CARTESIAN_VELOCITY
            violation = self._check_cartesian_velocity(command, cycles)

        if (
            violation is not None
            and violation.error_index
            == JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX
            and self._safety_active
        ):
            # Hardware latches *both* bits from a single command that
            # crosses the commanded envelope while a motion is running.
            # This used to re-ask the safety controller against measured
            # dq with a zero margin and report 13 alone when that re-ask
            # came back clean -- but measured dq lags commanded dq by a
            # cycle or few, so the re-ask was nondeterministic: it read
            # "clean" on exactly the cycle hardware would have called
            # joint_velocity_violation. Franka's own hardware smoke suite
            # documents that outcome as the expected one, not an
            # exception: "Both error[s] can appear. However with the
            # current RCU joint_velocity_violation appear[s] much
            # earl[ier] as we shape the limit. From 50% of max allowed
            # speed we already start bringing the limit towards the
            # current velocity" (dev comment on both
            # ``JointMotionGeneratorPositionLimitsViolationHardware`` and
            # ``JointMotionGeneratorVelocityLimitsViolationHardware``,
            # ``smoke_test_errors.cpp:110-127``). So a 13 during an armed
            # motion latches 3 alongside it unconditionally, rather than
            # asking whether measured dq happens to have crossed yet.
            #
            # The pure safety-controller path
            # (:meth:`check_measured_velocity`, e.g. torque mode with no
            # commanded velocity at all) is untouched: it returns a
            # plain 3, with no 13 to pair it with.
            violation = replace(violation, extra_error_index=JOINT_VELOCITY_VIOLATION_INDEX)
        return violation

    def check_measured_velocity(
        self,
        joint_positions: Optional[Sequence[float]] = None,
        joint_velocities: Optional[Sequence[float]] = None,
    ) -> Optional[Violation]:
        """Run the safety controller against the arm's *measured* velocity.

        Called once per control cycle from the state-publish loop, with the
        physics snapshot that loop already reads. Unlike every other check in
        this module this one judges the robot, not the client: it compares the
        measured ``dq`` against the same position-based envelope
        (:func:`upper_joint_velocity_limits` /
        :func:`lower_joint_velocity_limits`) that the commanded check uses, and
        latches ``joint_velocity_violation``.

        That distinction is the whole point. Franka's hardware smoke suite has
        one test for it that no commanded check could ever satisfy:
        ``moveJointVelocityViolation`` (``smoke_errors.cpp:773``) runs *pure
        torque control* -- there is no commanded velocity anywhere in the
        session -- and ramps 3 Nm into joint 6 until the arm folds through the
        envelope. Hardware answers ``joint_velocity_violation``. So this check
        is active in **all** control modes, including torque, and it is the one
        check in this module that does not care what the client sent.

        A margin of :attr:`measured_velocity_margin` is allowed above the
        envelope; see :data:`MEASURED_JOINT_VELOCITY_MARGIN` for why a measured
        signal needs one and why 0.1 rad/s is the size chosen.

        Returns the violation, or None. Records the snapshot either way, so the
        precedence rule in :meth:`check` has something to consult.
        """
        with self._lock:
            if joint_positions is not None:
                candidate = [float(value) for value in joint_positions[:7]]
                if all(math.isfinite(value) for value in candidate):
                    self._joint_positions = candidate
            if joint_velocities is not None:
                measured = [float(value) for value in joint_velocities[:7]]
                # A non-finite *measured* velocity is the simulator's problem,
                # not the client's -- exactly as for a measured position in
                # :meth:`check`. Blaming the client for the backend blowing up
                # would be the wrong error on the wire.
                self._measured_velocity = (
                    measured if all(math.isfinite(value) for value in measured) else None
                )
            if not self._safety_active:
                return None
            return self._check_measured_velocity_locked()

    def _check_measured_velocity_locked(self) -> Optional[Violation]:
        """The comparison behind :meth:`check_measured_velocity`; lock held.

        Compares against :attr:`measured_velocity_margin`, the sole allowance
        above the envelope; see :data:`MEASURED_JOINT_VELOCITY_MARGIN` for why
        a measured signal needs one and why 0.1 rad/s is the size chosen.
        """
        measured = self._measured_velocity
        if measured is None or len(measured) < 7:
            return None
        upper = upper_joint_velocity_limits(self._joint_positions)
        lower = lower_joint_velocity_limits(self._joint_positions)
        margin = self.measured_velocity_margin
        for index in range(7):
            value = measured[index]
            if value > upper[index] + margin or value < lower[index] - margin:
                limit = upper[index] if value > upper[index] else lower[index]
                return Violation(
                    JOINT_VELOCITY_VIOLATION_INDEX,
                    "dq",
                    f"joint {index + 1}",
                    value,
                    limit,
                    "rad/s",
                )
        return None

    def record(self, command: Dict[str, Any], *, clean: Optional[bool] = None) -> None:
        """Accept one *received* command as applied.

        The extrapolated substitute for a command that never arrived is
        recorded by :meth:`extrapolate` instead, which does its own advancing
        under the frozen-acceleration rule rather than re-deriving derivatives
        from a sample it produced itself.

        Only commands the caller accepted, and only *fresh* ones -- or ones a
        rewind has established are the real answer to an extrapolated cycle
        (:meth:`rewind_extrapolation`). A stale or duplicated echo is not a later
        sample of the client's trajectory, and differencing one produces a
        velocity and an acceleration nobody commanded.

        A command covers however many state cycles its ``message_id`` says it
        does, capped (see :meth:`cycles_since_applied`).

        ``clean`` is whether the checker passed this command. It decides nothing
        about what is recorded -- with enforcement off a flagged command is
        still applied, and the history has to describe what was applied -- but a
        flagged command's *derivatives* are not allowed to seed the next gap's
        frozen acceleration. One duplicated datagram sets the commanded velocity
        to zero and the acceleration to ``-v/dt``; integrating that for
        nineteen extrapolated cycles dispatched a reference running backwards at
        nineteen times the commanded speed. So a flagged record arms
        :meth:`extrapolate` to freeze from the last clean history instead.
        ``None`` (the default) means "whatever :meth:`check` said about this same
        ``message_id``", which is what a caller that checks and records in the
        obvious order gets for free; a caller that never checked gets the benefit
        of the doubt.
        """
        with self._lock:
            self._record_locked(command, clean=clean)

    def _record_locked(self, command: Dict[str, Any], *, clean: Optional[bool] = None) -> None:
        """The body of :meth:`record`; lock held."""
        # Whatever a bare rewind armed is answered by this record.
        self._rewind_in_flight = False
        if not self._active:
            return
        first = self._first_command
        self._first_command = False

        if clean is None:
            verdict = self._last_verdict
            clean = not (
                verdict is not None
                and verdict[0] == int(command.get("message_id", 0))
                and verdict[1]
            )
        self._freeze_from_clean = not clean

        cycles = self._cycles_since_applied_locked(command)
        message_id = int(command.get("message_id", 0))
        self._applied_id = message_id
        # A copy, not the caller's dict: the UDP path builds a fresh one per
        # datagram today, but :meth:`extrapolate` reads this again cycles
        # later and must not be looking at something that has been mutated
        # under it since.
        self._last_command = dict(command)
        # Whether this closes the open run of losses or only eats into it.
        # A datagram from *inside* the run is a late answer to one of its
        # cycles: the cycles after it can still be answered too, so the run
        # stays rewindable and only its lower bound moves up (the snapshot
        # is re-taken below, once the history has advanced onto this
        # command). A command from beyond the run is the client back in
        # step, and closes it. See :meth:`rewind_extrapolation`.
        absorbing_late = (
            self._gap_snapshot is not None
            and self._gap_first_id is not None
            and self._gap_last_id is not None
            and self._gap_first_id <= message_id <= self._gap_last_id
        )
        if not absorbing_late:
            self._gap_snapshot = None
            self._gap_first_id = None
            self._gap_last_id = None

        if self._mode is ControlMode.POSITION:
            # The first waypoint of a motion is a standstill by
            # construction (the start-pose check has just confirmed it sits
            # where the robot is), so it rebases the history rather than
            # being differenced against a pose the client never commanded.
            if first:
                self._joint.rebase(command["q_c"])
            else:
                self._joint.advance(command["q_c"], cycles)
        elif self._mode is ControlMode.VELOCITY:
            # Same reasoning as the joint-position generator's ``first``
            # branch just above: the opening command has only just been
            # excused by ``_check_start_velocity``'s tolerance around
            # ``dq_d``, not confirmed equal to it, so differencing it against
            # the seeded history would freeze whatever offset the tolerance
            # let through as an implied acceleration -- up to
            # ``START_VELOCITY_TOLERANCE / DELTA_T`` = 100 rad/s^2 -- and a
            # gap would integrate that offset for the rest of the motion. It
            # rebases instead, exactly like ``advance`` would have if the
            # opening command had actually equalled ``dq_d``.
            if first:
                self._joint.rebase(command["dq_c"])
            else:
                self._joint.advance(command["dq_c"], cycles)
        elif self._mode is ControlMode.TORQUE:
            self._torque.advance(command["tau_J_d"], cycles)
        elif self._mode is ControlMode.CARTESIAN_POSE:
            # Same reasoning as the joint-position generator's ``first``
            # branch: the opening pose has just been confirmed to sit where
            # the robot is, so it is a standstill, not a step from the
            # seeded pose.
            if first:
                self._pose.rebase(command["O_T_EE_c"])
            else:
                self._pose.advance(command["O_T_EE_c"], cycles)
            self._record_elbow_locked(command, cycles)
        elif self._mode in (ControlMode.STEERING_DRIVE, ControlMode.CARTESIAN_VELOCITY):
            self._twist.advance(command["O_dP_EE_c"], cycles)
            if self._mode is ControlMode.CARTESIAN_VELOCITY:
                self._record_elbow_locked(command, cycles)

        if clean:
            self._mark_clean_locked()
            self._clean_age = 0
        elif self._clean_age is not None:
            self._clean_age += 1
        if absorbing_late:
            # The run is now answered up to and including this id, and the
            # history sits on the client's own data for it. That is the
            # state a *later* late datagram has to be rewound to.
            self._gap_snapshot = self._snapshot_locked()
            self._gap_first_id = message_id + 1
            if self._gap_first_id > (self._gap_last_id or 0):
                self._gap_snapshot = None
                self._gap_first_id = None
                self._gap_last_id = None

    def _record_elbow_locked(self, command: Dict[str, Any], cycles: int) -> None:
        """Accept this command's elbow, if it carries one; lock held.

        ``valid_elbow`` is the client's own statement that it commanded an
        elbow at all: libfranka sets it only when the motion object was
        constructed with one and zero-fills ``elbow_c`` otherwise
        (``src/control_loop.cpp:270-286``, ``src/robot_impl.cpp:429-433``). A
        motion that never sends an elbow therefore leaves this history alone --
        which is what keeps the mobile base's elbow-less twist stream out of it.

        "First" here is the first *elbow-carrying* command rather than the
        motion's first command, which is the same thing for any libfranka client
        (the motion object either has an elbow or it does not, for its whole
        life) and the honest reading for one that is not libfranka.
        """
        if not command.get("valid_elbow"):
            return
        elbow = command["elbow_c"]
        if self._elbow_sign is None:
            self._elbow.rebase([elbow[0]])
        else:
            self._elbow.advance([elbow[0]], cycles)
        self._elbow_sign = float(elbow[1])

    # -- packet-loss extrapolation -----------------------------------------

    def extrapolate(
        self, message_id: int
    ) -> Optional[Tuple[Dict[str, Any], Optional[Violation]]]:
        """The command the FCI substitutes for one missed cycle: built, judged, recorded.

        Returns ``(command, violation)`` -- a ``RobotCommand``-shaped dict the
        caller dispatches to physics and publishes back in the commanded state
        fields, and whatever that command breaks. ``None`` when there is nothing
        to extrapolate: no motion, no real command recorded yet, or a torque
        controller, which *holds* ("FCI will reuse the torques of the last
        successful received packet", ``docs/system_requirements.rst``).

        **What is extrapolated, per interface.** Every law freezes the highest
        derivative the last two *real* commands implied and integrates below it,
        which is the documented behaviour -- Control "takes the previous
        waypoints and performs a linear extrapolation (keep acceleration
        constant and integrate) for the missed time step":

        * joint position -- ``v += a dt`` then ``q += v dt``, ``a`` frozen (that
          order, and :meth:`_Differentiator.extrapolate_position` explains why
          the other one accumulates slack against the differencing);
        * joint velocity -- ``dq += a dt``, ``a`` frozen (there is no jerk term:
          integrating one turns twenty milliseconds of silence into a runaway);
        * Cartesian pose -- the same law per translation axis, and the rotation
          extended by an axis-angle increment built from the same advanced
          angular velocity (see :meth:`_PoseDifferentiator.extrapolate`);
        * Cartesian velocity -- the twist extended at frozen twist-acceleration;
        * the elbow -- ``elbow[0]`` on the 1-D position law, ``elbow[1]`` held,
          because a branch flag has no derivative and hardware calls a change in
          one an inconsistency rather than a rate.

        **The result is checked, not trusted.** It goes through the same
        :meth:`_check_locked` a received command does, and it is deliberately
        *not* clamped to any limit on the way. An extrapolation running out past
        the velocity envelope or a joint stop is not a bug to be papered over --
        it is precisely the mechanism behind libfranka's warning that
        intermittent drops "could trigger `discontinuity` errors even when your
        source signals conform with the interface specification"
        (``docs/overview.rst``). Clamping it would make the sim quietly kinder
        than the robot in the one situation the client most needs warning about.

        **It is recorded whether or not it violates.** Control's reference
        advances through the gap regardless; the violation stops the *motion*,
        it does not rewind the reference. Recording it is also what keeps the
        client's resumed command exactly one cycle from the history, which is
        the whole point (``docs/overview.rst``: ``q_{c,k-1}`` is "always sent
        back ... even in case of packet losses").

        **What the freeze reads is the last command nobody objected to.** With
        enforcement off a flagged command is still applied and still recorded --
        the sim stays the permissive channel it has always been -- but its
        derivatives are not allowed to seed a gap. A single duplicated datagram
        differences to zero velocity and ``-v/dt`` of acceleration; frozen and
        integrated for nineteen cycles that dispatched a reference running
        backwards at nineteen times the commanded speed, which is not a
        consequence of packet loss but of one bad sample being amplified by it.
        So when :meth:`record` was told the command it accepted was flagged, the
        derivatives are taken from the last *clean* history instead (zero, if
        this motion has not had one yet). The flagged command still governs the
        cycle it commanded; it just does not govern the nineteen after it.

        The caller stops calling this at
        :data:`franka_sim.comm_constraints.MAX_CONSECUTIVE_LOST_CYCLES`
        consecutive misses, where the robot stops too.

        Everything runs under one acquisition of the checker's lock -- build,
        check, record -- because the UDP receive thread may be recording a real
        command at the same moment and a torn sequence would leave the history
        describing a trajectory neither of them commanded.
        """
        with self._lock:
            if not self._active or self._first_command or self._last_command is None:
                return None
            if self._mode is ControlMode.TORQUE:
                return None
            # ``message_id`` is the publish loop's own strictly-increasing
            # counter, not client-controlled data, so a call that repeats or
            # goes backwards relative to ``_applied_id`` is a caller bug, not
            # something to tolerate quietly. Below, ``commit_position`` et al.
            # integrate one cycle of motion into the history every call --
            # calling this twice for the same id integrates *two* cycles
            # while ``_applied_id`` (updated at the bottom of this method)
            # advances by at most one, so a later resume differences the
            # doubly-integrated history against an interval its own
            # bookkeeping says is one cycle, not two. That is exactly how one
            # repeated ``extrapolate(401)`` call turned into a
            # 127 rad/s^2 abort at the next real command.
            assert self._applied_id is None or int(message_id) > self._applied_id, (
                f"extrapolate({message_id}) called at applied_id={self._applied_id}: "
                "message_id must strictly advance past the history"
            )
            if self._rewind_in_flight:
                # A caller has rewound this run for a datagram it has not
                # recorded yet (see :meth:`rewind_extrapolation`). Guessing the
                # cycle that datagram is about to answer is what the rewind was
                # for; hold this one cycle instead. One-shot, so nothing can be
                # wedged by a caller that then throws the datagram away.
                self._rewind_in_flight = False
                return None

            command = dict(self._last_command)
            command["message_id"] = int(message_id)
            # This is a substitute for a command that was never sent, so it
            # cannot be the one that ends the motion, whatever the last real
            # datagram happened to carry.
            command["motion_generation_finished"] = False
            command["torque_command_finished"] = False
            # Not part of the wire format -- a marker for the server's logging
            # and for a reader of a captured command. Ignored everywhere else.
            command["extrapolated"] = True

            # The pre-gap state, before anything below can touch it -- including
            # the clean-derivative substitution, which is part of what a rewind
            # has to be able to undo. Installed as the run's snapshot only once
            # the substitute survives the finite check; a refused extrapolation
            # leaves the history exactly as it found it.
            opening = self._gap_snapshot is None
            snapshot = self._snapshot_locked() if opening else None
            if self._freeze_from_clean:
                # The last recorded command was flagged. Freeze the gap from the
                # last history nobody objected to instead of from it.
                self._freeze_clean_locked()
                self._freeze_from_clean = False

            elbow = None
            if self._mode is ControlMode.POSITION:
                command["q_c"] = self._joint.extrapolate_position()
            elif self._mode is ControlMode.VELOCITY:
                command["dq_c"] = self._joint.extrapolate_velocity()
            elif self._mode is ControlMode.CARTESIAN_POSE:
                command["O_T_EE_c"] = self._pose.extrapolate()
                elbow = self._extrapolate_elbow_locked(command)
            else:  # ControlMode.STEERING_DRIVE / ControlMode.CARTESIAN_VELOCITY
                command["O_dP_EE_c"] = self._twist.extrapolate()
                if self._mode is ControlMode.CARTESIAN_VELOCITY:
                    elbow = self._extrapolate_elbow_locked(command)

            if self._check_finite_locked(command) is not None:
                # Unreachable from a finite history -- every recorded command
                # passed the same test -- so this is a guard against the
                # arithmetic above, not against the client. Hold rather than
                # record: a NaN in the history poisons every later difference,
                # and refusing to record keeps that out of the history at all.
                #
                # Latched, because it cannot heal: a history that has gone
                # non-finite stays that way for the motion, and this sits on a
                # 1 kHz path.
                if not self._extrapolation_refused:
                    self._extrapolation_refused = True
                    logger.warning(
                        "Extrapolated command was not finite; holding the last "
                        "applied command through this gap instead"
                    )
                if snapshot is not None:
                    self._restore_locked(snapshot)
                return None

            violation = self._check_locked(command, 1)

            # The run of losses this cycle belongs to, and the state it has to
            # be rewindable to. Kept for the whole run -- and past the first late
            # datagram of it, because the cycles *after* that one can still be
            # answered too; see :meth:`rewind_extrapolation`.
            if opening:
                self._gap_snapshot = snapshot
                self._gap_first_id = int(message_id)
            self._gap_last_id = int(message_id)

            if self._mode is ControlMode.POSITION:
                self._joint.commit_position(command["q_c"])
            elif self._mode is ControlMode.VELOCITY:
                self._joint.commit_velocity(command["dq_c"])
            elif self._mode is ControlMode.CARTESIAN_POSE:
                self._pose.commit(command["O_T_EE_c"])
            else:
                self._twist.commit(command["O_dP_EE_c"])
            if elbow is not None:
                self._elbow.commit_position([elbow[0]])

            # One cycle of motion was integrated, so the id the history sits at
            # advances by exactly one. In the ordinary case that *is*
            # ``message_id`` -- the history was at ``message_id - 1`` and the
            # client's next real command, which answers a later state, is one
            # cycle away from it, exactly as on hardware.
            #
            # It is not ``max(applied, message_id)``. After a late datagram has
            # been absorbed the history sits back on real data from the middle
            # of the run, several cycles behind the cycle now being
            # extrapolated; claiming the current id there would say the
            # reference had travelled cycles it never integrated, and the
            # client's resume would then be differenced over an interval far
            # shorter than the motion it actually contains. That was a 636
            # rad/s^2 abort out of a two-cycle receive-thread stall. The id never
            # runs past ``message_id``, and never backwards.
            advanced = (self._applied_id if self._applied_id is not None else 0) + 1
            self._applied_id = max(
                self._applied_id or 0, min(advanced, int(message_id))
            )
            return command, violation

    def rewind_extrapolation(self, command: Dict[str, Any]) -> bool:
        """Undo the guesses ``command`` turns out to be the real answer for.

        The one place this simulator's packet handling has to differ from the
        robot's, and it exists because the two disagree about late datagrams.

        The FCI **drops** a command that misses its 1 ms window; the cycle is
        lost, the extrapolation stands, and the packet never existed. This sim
        deliberately *applies* it instead -- "within one sim cycle a late
        command is still the freshest user intent there is", and dropping it
        would leave every non-realtime scripted client unable to move the arm at
        all, which is most of what a simulator is driven by.

        Those two choices collide the moment extrapolation exists. A datagram
        answering state ``N`` that arrives just after state ``N+1`` went out has
        already been stood in for: the publish loop extrapolated cycle ``N`` and
        the reference moved by one cycle's worth of motion. The datagram carries
        one cycle's worth of motion too -- it is the *same* step, measured
        rather than guessed -- so differencing it against the extrapolation
        reports a reference that travelled nowhere and then a huge deceleration.
        That is not the client's doing and not a limit the robot would flag; on
        an ordinary ramp at 0.5 rad/s it comes out as ~500 rad/s^2 and abort
        conforming clients. It was measured doing exactly that against Franka's
        own smoke suite, at a ``control_command_success_rate`` of 0.99.

        So: **when the real packet turns up, the guess it replaced is thrown
        away.** The history is restored to where it stood before this run of
        losses was extrapolated, and the caller then checks and records the
        datagram like any other -- over the honest interval, which
        :meth:`cycles_since_applied` now measures back to the last *real*
        command rather than to a substitute.

        Returns True when a rewind happened, which is the caller's signal that
        this datagram is a genuine later sample of the client's trajectory and
        should be recorded.

        **A run of losses stays rewindable until a fresh command closes it.**
        Not until the first late datagram of it: a stalled receive thread hands
        over a whole *backlog* at once, and every datagram in it is the real
        answer to one of the run's cycles. Rewinding only for the first left the
        rest applied-but-not-recorded -- the history frozen k cycles in the past
        while :meth:`extrapolate` claimed the current id for it -- and the
        client's next perfectly conforming command was then differenced over one
        cycle against a reference k cycles stale. A two-cycle stall came out as
        127 rad/s^2, a six-cycle one as 636, growing without bound. So the
        snapshot is *retained*, and :meth:`record` re-takes it against the
        history the absorbed datagram just established, moving the window's
        lower bound past the cycle it answered. The run closes when the client
        answers a cycle beyond it (or when every cycle in it has been answered).

        **Only for a datagram inside the still-unanswered part of the run.** A
        replay, a duplicate or a reordered packet echoes an id at or before the
        window's lower bound, is not within it, and gets no rewind -- the
        history is not rewound for a datagram that is not new information, which
        is the whole reason stale echoes are never recorded.

        **And only for a datagram that missed its cycle.** A *fresh* command --
        the client resuming after a real pause -- is judged against the
        extrapolated reference, unrewound, which is the hardware behaviour and
        the reason the resume trap fires at all. The caller makes that
        distinction; this method is not called for a fresh command.

        Idempotent: rewinding twice for the same datagram restores the same
        state twice. What moves the window is *recording*, which is the caller's
        statement that the datagram was accepted.

        Prefer :meth:`absorb_command`, which does this, the check and the record
        without releasing the lock in between. This method is the primitive
        underneath it, kept public because the rewind is a distinct decision
        worth being able to test and reason about on its own -- but a caller
        that takes it on its own has released the lock with the history rewound
        and the datagram not yet recorded, and a publish tick landing in that
        window would guess the very cycle whose real answer is in flight. So a
        bare rewind arms a one-shot refusal: the next :meth:`extrapolate` holds
        instead of guessing. Self-healing (one cycle at most, and only for a
        caller that does not use :meth:`absorb_command`) and belt to that
        method's braces.

        **This primitive does not restore on reject.** It only ever moves the
        history one way -- backward, to the pre-gap snapshot -- and has no
        memory of what it discarded to put back. :meth:`absorb_command` is
        what pairs a rewind with a restore: it snapshots the *extrapolated*
        state before calling this method, and if the datagram then turns out
        to be one the caller rejects (enforcement, or a fatal violation), it
        puts that snapshot back rather than leaving the reference rewound with
        nothing recorded to replace it. A caller that calls this method
        directly, rewinds, and then does not go on to record the datagram --
        because it decided to reject it, or for any other reason -- leaves the
        reference having jumped backward by the gap with nothing to show for
        it: a corrupt history, not merely a stale one. Use
        :meth:`absorb_command` unless the caller is prepared to snapshot and
        restore this itself.
        """
        with self._lock:
            rewound = self._rewind_extrapolation_locked(command)
            if rewound:
                self._rewind_in_flight = True
            return rewound

    def _rewind_extrapolation_locked(self, command: Dict[str, Any]) -> bool:
        """The body of :meth:`rewind_extrapolation`; lock held."""
        if self._gap_snapshot is None:
            return False
        message_id = int(command.get("message_id", 0))
        if not (self._gap_first_id <= message_id <= self._gap_last_id):
            return False
        self._restore_locked(self._gap_snapshot)
        return True

    def absorb_command(
        self,
        command: Dict[str, Any],
        joint_positions: Optional[Sequence[float]] = None,
        *,
        fresh: bool,
        enforce: bool = False,
    ) -> "AbsorbedCommand":
        """Rewind, check and record one received command as **one** operation.

        The server's whole per-datagram interaction with this checker, taken
        under a single acquisition of its lock. Which matters because the other
        half of the interaction -- :meth:`extrapolate` -- runs on the
        state-publish thread at 1 kHz and mutates the same histories.

        Split into ``rewind_extrapolation`` / ``check`` / ``record`` it was three
        acquisitions with two windows between them, and a publish tick landing in
        either one re-created exactly the false abort the rewind exists to
        prevent: the rewind restores the pre-gap history, the interleaved
        extrapolation advances it again from there, and the ``check`` and
        ``record`` that follow are then measured against a reference that moved
        under them. Measured at 63 to 191 rad/s^2 on an ordinary conforming ramp,
        on gaps of one, three and five cycles.

        Lock ordering is unchanged and this is the reason the method exists in
        *this* class rather than as a sequence in the server: the checker's lock
        stays the innermost one. Nothing here calls back into the server, takes
        ``_motion_lock`` or ``_hold_lock``, or logs a violation -- the verdict is
        returned and the caller reports and aborts on it, outside the lock,
        exactly as before.

        ``fresh`` is the comm tracker's verdict on whether this datagram
        answered the cycle it arrived in. A fresh one is judged against the
        reference as it stands (the hardware behaviour, and what makes the resume
        trap fire); a late one first gets :meth:`rewind_extrapolation`.

        ``enforce`` is the server's ``--enforce-motion-limits``. It decides
        acceptance, not checking: a violation is always found and always
        returned. A rejected command is recorded nowhere **and rewound
        nowhere** -- if the rewind already happened, the extrapolated reference
        is put back before returning, because a command the server refuses must
        not be able to roll the reference back and leave nothing in its place.
        That ordering bug reached physics as a backward jump the size of the gap.
        """
        with self._lock:
            rewound = False
            extrapolated_state = None
            if not fresh:
                # Keep the state the rewind is about to discard: if the command
                # turns out to be one the server refuses, this is what has to go
                # back. (M2: rewind is only final once the command is accepted.)
                extrapolated_state = self._snapshot_locked()
                rewound = self._rewind_extrapolation_locked(command)
                if not rewound:
                    extrapolated_state = None

            violation = self._check_public_locked(command, joint_positions)

            rejected = violation is not None and (enforce or violation.fatal)
            if rejected:
                if extrapolated_state is not None:
                    self._restore_locked(extrapolated_state)
                return AbsorbedCommand(
                    violation=violation, accepted=False, recorded=False, rewound=False
                )

            recorded = fresh or rewound
            if recorded:
                self._record_locked(command, clean=violation is None)
            return AbsorbedCommand(
                violation=violation, accepted=True, recorded=recorded, rewound=rewound
            )

    def _snapshot_locked(self) -> Dict[str, Any]:
        """Copy every differencing history, so a gap can be un-guessed; lock held.

        All of them rather than only the active generator's: the elbow rides
        along with either Cartesian mode, and a snapshot that is complete cannot
        be wrong about which mode was running when it was taken. It is a few
        dozen floats, taken once per run of losses.
        """
        return {
            "applied_id": self._applied_id,
            "joint": (
                list(self._joint.value),
                list(self._joint.first),
                list(self._joint.second),
            ),
            "twist": (list(self._twist.value), list(self._twist.first)),
            "pose": (
                np.array(self._pose.rotation),
                np.array(self._pose.translation),
                list(self._pose.first),
                list(self._pose.second),
            ),
            "elbow": (
                list(self._elbow.value),
                list(self._elbow.first),
                list(self._elbow.second),
            ),
        }

    def _restore_locked(self, snapshot: Dict[str, Any]) -> None:
        """Put every differencing history back to a :meth:`_snapshot_locked`; lock held."""
        self._applied_id = snapshot["applied_id"]
        self._joint.value, self._joint.first, self._joint.second = (
            list(values) for values in snapshot["joint"]
        )
        self._twist.value, self._twist.first = (list(values) for values in snapshot["twist"])
        rotation, translation, first, second = snapshot["pose"]
        self._pose.rotation = np.array(rotation)
        self._pose.translation = np.array(translation)
        self._pose.first = list(first)
        self._pose.second = list(second)
        self._elbow.value, self._elbow.first, self._elbow.second = (
            list(values) for values in snapshot["elbow"]
        )

    def _extrapolate_elbow_locked(self, command: Dict[str, Any]) -> Optional[List[float]]:
        """Extend this motion's elbow through a missed cycle; lock held.

        Only for a motion that actually commands one: ``valid_elbow`` is the
        client's own statement that it does, and ``_elbow_sign`` is None until
        the first elbow-carrying command has been recorded. A motion without an
        elbow leaves ``elbow_c`` exactly as the last real datagram had it
        (zero-filled), which is what libfranka sends.

        ``elbow[1]`` is *held*, never extrapolated. It is a branch flag, not a
        quantity: hardware's name for a change in it is
        ``cartesian_motion_generator_elbow_sign_inconsistent``, an inconsistency
        rather than a rate, so there is nothing to integrate and a gap must not
        invent a configuration change.
        """
        if not command.get("valid_elbow") or self._elbow_sign is None:
            return None
        elbow = [self._elbow.extrapolate_position()[0], float(self._elbow_sign)]
        command["elbow_c"] = elbow
        return elbow

    # -- per-generator checks ----------------------------------------------

    def _check_position(self, command: Dict[str, Any], cycles: int) -> Optional[Violation]:
        q_c = command["q_c"]
        violation = self._check_joint_position_limits(q_c)
        if violation is not None:
            return violation

        if self._first_command:
            # A motion's first command must be where the robot already is;
            # differencing it against ``q_d`` would report the *step into the
            # motion*, which is a start-pose error, not a velocity one.
            return self._check_start_pose(q_c)

        velocity, acceleration, jerk = self._joint.derivatives(q_c, cycles)
        # Acceleration first, and that ordering is the point: a step in ``q_c``
        # breaks the envelope, the acceleration limit and the jerk limit all at
        # once, and hardware calls it exactly one of them --
        # ``joint_motion_generator_velocity_discontinuity``. See the
        # interface-relative naming rule at the top of this module. Checking the
        # envelope first made the sim report 13 for the smoke suite's 1.0 rad
        # step where hardware reports 14.
        return (
            self._check_per_joint(
                acceleration,
                MAX_JOINT_ACCELERATION,
                JOINT_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
                "q_c",
                "rad/s^2",
            )
            or self._check_joint_velocity_limits(velocity, "q_c")
            or self._check_per_joint(
                jerk,
                MAX_JOINT_JERK,
                JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
                "q_c",
                "rad/s^3",
            )
        )

    def _check_velocity(self, command: Dict[str, Any], cycles: int) -> Optional[Violation]:
        # No joint *range* check here, deliberately: a velocity generator
        # commands a rate, and where that rate will put the joint is only
        # knowable by integrating it, which is the physics backend's job, not
        # this module's. The range is not unguarded, though -- the
        # position-dependent velocity limit shrinks to zero as the joint
        # approaches its stop (``upper_joint_velocity_limits``), so a client
        # that keeps commanding into a limit is caught as a *velocity* limits
        # violation, which is also what the real robot reports.
        dq_c = command["dq_c"]
        if self._first_command:
            # A velocity motion must continue from the ``dq_d`` the robot last
            # reported. The enum has no joint-velocity start error, so a step
            # away from it is reported as what a step in ``dq_c`` is on this
            # interface: an *acceleration* discontinuity. The envelope check
            # still runs behind it, so an opening command that continues from
            # ``dq_d`` but is itself too fast is not waved through.
            return self._check_start_velocity(dq_c) or self._check_joint_velocity_limits(
                dq_c, "dq_c"
            )

        acceleration, jerk, _ = self._joint.derivatives(dq_c, cycles)
        # Interface-relative naming, one derivative up from ``dq_c``: breaking
        # kMaxJointAcceleration on the first difference of a *velocity* command
        # is ``joint_motion_generator_acceleration_discontinuity`` (15), not the
        # 14 the same limit earns on a position command. The jerk check lands on
        # 15 as well -- the enum has nothing above it for this family -- so the
        # two are folded together, and both come before the envelope check,
        # which a step also breaks and which hardware never names for one.
        return (
            self._check_per_joint(
                acceleration,
                MAX_JOINT_ACCELERATION,
                JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
                "dq_c",
                "rad/s^2",
            )
            or self._check_per_joint(
                jerk,
                MAX_JOINT_JERK,
                JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
                "dq_c",
                "rad/s^3",
            )
            or self._check_joint_velocity_limits(dq_c, "dq_c")
        )

    def _check_torque(self, command: Dict[str, Any], cycles: int) -> Optional[Violation]:
        # Range first, rate second -- and unlike every other ordering in this
        # module that is *not* pinned to hardware. The smoke suite's
        # ``moveControllerTorqueDiscontinuity`` (``smoke_errors.cpp:165``) steps
        # joint 1 to 10 Nm, which is a rate violation (10 000 Nm/s) and well
        # inside the joint's 87 Nm range, so it exercises the rate check alone
        # and says nothing about which name a command that breaks *both* should
        # get. Nothing else in the suite does either.
        #
        # The interface-relative rule would argue for the rate (discontinuity
        # beats envelope everywhere else); ``tau_J_range_violation`` arguably
        # sits in a different category -- libfranka documents it as the *torque
        # sensor* limit ("If the torque sensor limit is reached, a
        # tau_j_range_violation will be triggered", ``docs/overview.rst``),
        # i.e. something measured rather than a bound on the command. With no
        # evidence either way the order is left as it has always been, and
        # ``test_a_torque_that_breaks_both_range_and_rate_reports_the_range``
        # pins it so a future change to it is deliberate rather than accidental.
        tau = command["tau_J_d"]
        violation = self._check_per_joint(
            tau, MAX_TORQUE, TAU_J_RANGE_VIOLATION_INDEX, "tau_J_d", "Nm"
        )
        if violation is not None:
            return violation

        rate, _, _ = self._torque.derivatives(tau, cycles)
        return self._check_per_joint(
            rate, MAX_TORQUE_RATE, CONTROLLER_TORQUE_DISCONTINUITY_INDEX, "tau_J_d", "Nm/s"
        )

    def _check_cartesian_pose(self, command: Dict[str, Any], cycles: int) -> Optional[Violation]:
        """The ``kCartesianPosition`` generator: ``O_T_EE_c`` and its elbow.

        **Checked, never applied.** No physics branch drives the arm from a
        commanded pose (see ``docs/compatibility.md``), so the arm stays where it
        is for the whole motion -- but every hardware error the pose interface
        can raise is raised here, which is what makes Franka's own smoke tests
        for Cartesian errors terminate against this sim instead of hanging
        forever waiting for an abort that never came.

        The order below *is* the precedence, and every step of it is pinned:

        1. **Is it a transform at all?** ``checkMatrix``'s test, server-side.
           Fatal: refused whether or not enforcement is on, for the same reason
           a NaN is -- a garbage matrix has no derivatives worth computing and
           poisons the history that the rest of the motion is judged against.
        2. **The first command's start checks**, which outrank every
           discontinuity below exactly as the joint-position generator's start
           check does. Pose first, then elbow.
        3. **The elbow's branch flag**, whose mid-motion flip is a validity
           error rather than a limit -- there is nothing to differentiate about
           a sign.
        4. **The pose's own derivatives**, interface-relative: on a commanded
           *pose* the first difference is a velocity, so breaking the
           **acceleration** limit is named
           ``cartesian_motion_generator_velocity_discontinuity`` (19), and it
           comes before the velocity-envelope check (18) that the same step also
           breaks. Jerk is one further up, at 20. This is the exact mirror of
           :meth:`_check_position`'s 14 / 13 / 15, and it is what makes the smoke
           suite's 1.0 m z step -- 0.3859 m after libfranka's own 100 Hz command
           filter, i.e. 386 m/s and 385 871 m/s^2 in a single cycle -- come back
           as 19 rather than 18.
        5. **The elbow's derivatives**, all three of which land on 17.

        Steps 4 and 5 never compete in Franka's suite: every Cartesian test that
        moves the elbow holds the pose still, and every one that moves the pose
        sends no elbow at all. Pose-before-elbow is therefore this sim's
        ordering, not a hardware pin.
        """
        pose = command["O_T_EE_c"]
        if not is_homogeneous_transformation(pose):
            return Violation(
                CARTESIAN_POSITION_MOTION_GENERATOR_INVALID_FRAME_INDEX,
                "O_T_EE_c",
                "orthonormality residual",
                homogeneous_transformation_residual(pose),
                ORTHONORMAL_THRESHOLD,
                fatal=True,
            )

        if self._first_command:
            violation = self._check_start_cartesian_pose(pose)
            if violation is not None:
                return violation

        elbow_violation = self._check_elbow_validity(command)
        if elbow_violation is not None:
            return elbow_violation

        if self._first_command:
            # Nothing to difference yet; the start checks above are the whole
            # judgement of a motion's opening command.
            return None

        velocity, acceleration, jerk = self._pose.derivatives(pose, cycles)
        return (
            self._check_cartesian_halves(
                acceleration,
                MAX_TRANSLATIONAL_ACCELERATION,
                MAX_ROTATIONAL_ACCELERATION,
                CARTESIAN_MOTION_GENERATOR_VELOCITY_DISCONTINUITY_INDEX,
                "O_T_EE_c",
                ("m/s^2", "rad/s^2"),
            )
            or self._check_cartesian_halves(
                velocity,
                MAX_TRANSLATIONAL_VELOCITY,
                MAX_ROTATIONAL_VELOCITY,
                CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
                "O_T_EE_c",
                ("m/s", "rad/s"),
            )
            or self._check_cartesian_halves(
                jerk,
                MAX_TRANSLATIONAL_JERK,
                MAX_ROTATIONAL_JERK,
                CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
                "O_T_EE_c",
                ("m/s^3", "rad/s^3"),
            )
            or self._check_elbow_limits(command, cycles)
        )

    def _check_cartesian_velocity(
        self, command: Dict[str, Any], cycles: int
    ) -> Optional[Violation]:
        # Elbow checks are the arm role's alone. This generator is shared with
        # STEERING_DRIVE (the mobile base), which has no elbow of any kind, and
        # ``record()`` never calls ``_record_elbow_locked`` for that mode
        # either (see its dispatch) -- so ``_elbow_sign`` never leaves its
        # initial ``None``, and running the start-elbow half of
        # ``_check_elbow_validity`` here for STEERING_DRIVE would re-fire
        # ``start_elbow_invalid`` every single cycle a base client happens to
        # set ``valid_elbow``, judged against a swerve steering angle that was
        # never meant to be one.
        if self._mode is ControlMode.CARTESIAN_VELOCITY:
            elbow_violation = self._check_elbow_validity(command)
            if elbow_violation is not None:
                return elbow_violation
        twist = command["O_dP_EE_c"]
        acceleration, jerk = self._twist.derivatives(twist, cycles)
        # The joint-velocity mapping, mirrored: ``O_dP_EE_c`` is a commanded
        # *velocity*, so its first difference is an acceleration and breaking
        # kMaxTranslational/RotationalAcceleration on it is
        # ``cartesian_motion_generator_acceleration_discontinuity`` (20) -- the
        # 19 that the same limit earns on a commanded *pose*. The jerk check
        # lands on 20 too, and both precede the envelope check (18), which a
        # twist step also breaks and which hardware never names for one. See the
        # interface-relative naming rule at the top of this module.
        checks = (
            (
                acceleration,
                MAX_TRANSLATIONAL_ACCELERATION,
                MAX_ROTATIONAL_ACCELERATION,
                CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
                ("m/s^2", "rad/s^2"),
            ),
            (
                jerk,
                MAX_TRANSLATIONAL_JERK,
                MAX_ROTATIONAL_JERK,
                CARTESIAN_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
                ("m/s^3", "rad/s^3"),
            ),
            (
                twist,
                MAX_TRANSLATIONAL_VELOCITY,
                MAX_ROTATIONAL_VELOCITY,
                CARTESIAN_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
                ("m/s", "rad/s"),
            ),
        )
        for values, translational, rotational, index, units in checks:
            violation = self._check_cartesian_halves(
                values, translational, rotational, index, "O_dP_EE_c", units
            )
            if violation is not None:
                return violation
        # An elbow rides along with a Cartesian *velocity* motion exactly as it
        # does with a pose one (``CartesianVelocities::hasElbow``,
        # ``src/control_loop.cpp:308-323``), so the same limits apply -- to the
        # arm role only, for the reason above.
        if self._mode is ControlMode.CARTESIAN_VELOCITY:
            return self._check_elbow_limits(command, cycles)
        return None

    # -- primitives --------------------------------------------------------

    def _check_cartesian_halves(
        self,
        values: Sequence[float],
        translational_limit: float,
        rotational_limit: float,
        error_index: int,
        signal: str,
        units: Tuple[str, str],
    ) -> Optional[Violation]:
        """Compare a 6-vector's translational and rotational halves, as norms.

        ``limitRate`` treats a Cartesian signal as two ``Eigen::Vector3d`` and
        compares *norms*, not components (``src/rate_limiting.cpp:184-195``
        dispatching into the anonymous-namespace overload at ``:13-55``), so
        both the twist generator and the pose generator judge them that way.
        """
        halves = (
            ("translational", _norm(values[:3]), translational_limit, units[0]),
            ("rotational", _norm(values[3:6]), rotational_limit, units[1]),
        )
        for axis, norm, limit, unit in halves:
            # The kNormEps guard is inert here and kept only to mirror
            # limitRate's shape: every limit is orders of magnitude above
            # machine epsilon, so ``norm > limit`` already implies it. In
            # libfranka it guards a *division* by the norm; nothing here
            # divides.
            if norm > NORM_EPS and norm > limit:
                return Violation(error_index, signal, f"{axis} norm", norm, limit, unit)
        return None

    def _check_joint_position_limits(self, q_c: Sequence[float]) -> Optional[Violation]:
        for index in range(7):
            lower, upper = JOINT_POSITION_LIMITS[index]
            if q_c[index] > upper or q_c[index] < lower:
                limit = upper if q_c[index] > upper else lower
                return Violation(
                    JOINT_MOTION_GENERATOR_POSITION_LIMITS_VIOLATION_INDEX,
                    "q_c",
                    f"joint {index + 1}",
                    float(q_c[index]),
                    limit,
                    "rad",
                )
        return None

    def _check_joint_velocity_limits(
        self, velocity: Sequence[float], signal: str
    ) -> Optional[Violation]:
        upper = upper_joint_velocity_limits(self._joint_positions)
        lower = lower_joint_velocity_limits(self._joint_positions)
        for index in range(7):
            if velocity[index] > upper[index] or velocity[index] < lower[index]:
                limit = upper[index] if velocity[index] > upper[index] else lower[index]
                return Violation(
                    JOINT_MOTION_GENERATOR_VELOCITY_LIMITS_VIOLATION_INDEX,
                    signal,
                    f"joint {index + 1}",
                    float(velocity[index]),
                    limit,
                    "rad/s",
                )
        return None

    def _check_per_joint(
        self,
        values: Sequence[float],
        limits: Sequence[float],
        error_index: int,
        signal: str,
        unit: str,
    ) -> Optional[Violation]:
        for index in range(7):
            if abs(values[index]) > limits[index]:
                return Violation(
                    error_index,
                    signal,
                    f"joint {index + 1}",
                    float(values[index]),
                    limits[index],
                    unit,
                )
        return None

    def _check_start_pose(self, q_c: Sequence[float]) -> Optional[Violation]:
        for index in range(7):
            offset = q_c[index] - self._joint.value[index]
            if abs(offset) > self.start_pose_tolerance:
                return Violation(
                    JOINT_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX,
                    "q_c",
                    f"joint {index + 1}",
                    float(offset),
                    self.start_pose_tolerance,
                    "rad from q_d",
                )
        return None

    def _check_start_cartesian_pose(self, pose: Sequence[float]) -> Optional[Violation]:
        """The first ``O_T_EE_c`` against the pose the robot is actually in.

        Franka's smoke suite pins this with a +10 m z offset held from cycle 0
        (``moveCartesianPositionMotionGeneratorStartPoseInvalid``,
        ``smoke_errors.cpp:242``), and pins that it is a *start-pose* error
        rather than a discontinuity -- which is why this runs before any
        differencing, exactly as :meth:`_check_start_pose` does for ``q_c``.

        Skipped when the caller's state snapshot carried no ``O_T_EE``: with no
        measured pose there is nothing to compare against, and inventing one
        would fabricate the error rather than find it.
        """
        if self._start_pose is None:
            return None
        actual = transform_matrix(self._start_pose)
        commanded = transform_matrix(pose)
        offset = float(np.linalg.norm(commanded[:3, 3] - actual[:3, 3]))
        if offset > self.start_cartesian_translation_tolerance:
            return Violation(
                CARTESIAN_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX,
                "O_T_EE_c",
                "translation",
                offset,
                self.start_cartesian_translation_tolerance,
                "m from O_T_EE",
            )
        angle = float(np.linalg.norm(rotation_log(actual[:3, :3].T @ commanded[:3, :3])))
        if angle > self.start_cartesian_rotation_tolerance:
            return Violation(
                CARTESIAN_POSITION_MOTION_GENERATOR_START_POSE_INVALID_INDEX,
                "O_T_EE_c",
                "rotation",
                angle,
                self.start_cartesian_rotation_tolerance,
                "rad from O_T_EE",
            )
        return None

    def _robot_elbow(self) -> Tuple[float, float]:
        """The elbow the robot is actually in: ``(q[2], sign(q[3]))``.

        ``elbow[0]`` is the redundancy angle, which on a 7-DOF FR3 *is* joint 3;
        ``elbow[1]`` is the sign of joint 4, the branch flag libfranka insists is
        exactly +-1 (``isValidElbow``, ``include/franka/control_tools.h``). So
        there is no elbow to look up anywhere -- it is a reading of the joint
        vector this checker already tracks.

        Joint 4's URDF range is [-3.0481, -0.1458], strictly negative, so a
        real FR3's flag is always -1; the ``>= 0`` branch exists only so a
        degenerate ``q[3] == 0`` from a mocked backend yields +-1 rather than 0.
        """
        return self._joint_positions[2], (1.0 if self._joint_positions[3] >= 0.0 else -1.0)

    def _check_elbow_validity(self, command: Dict[str, Any]) -> Optional[Violation]:
        """Start-elbow and mid-motion sign checks for an elbow-carrying command.

        Two distinct hardware errors, pinned by two smoke tests that differ only
        in *when* they perturb the elbow:

        * the motion's first elbow must be the elbow the robot is in --
          ``elbow[0] += 0.5`` from cycle 0 gives
          ``cartesian_motion_generator_start_elbow_invalid`` (22)
          (``smoke_errors.cpp:342`` / ``smoke_test_errors.cpp:252``);
        * the branch flag must not change once the motion is running --
          negating ``elbow[1]`` at t > 0.5 s gives
          ``cartesian_motion_generator_elbow_sign_inconsistent`` (21)
          (``smoke_errors.cpp:308`` / ``smoke_test_errors.cpp:237``).

        A *start-time* sign mismatch is reported as 22 as well; see
        :data:`START_ELBOW_SIGN_INCONSISTENT_INDEX` for why that index, which by
        its name would fit better, is not used.

        Commands without ``valid_elbow`` are not judged at all: the client did
        not command an elbow, and ``elbow_c`` is zero-filled precisely because
        of that.

        The start-elbow half is skipped when the caller's state snapshot
        carried no ``O_T_EE`` -- same guard, same reasoning as
        :meth:`_check_start_cartesian_pose`: ``elbow[0]`` on an FR3 is joint
        3's angle, and a seed with no measured pose is a seed from a backend
        that has not produced a trustworthy frame yet, so ``q`` read from it
        is no more trustworthy than the pose would have been. The mid-motion
        sign-inconsistency check below is unaffected -- it compares against
        the elbow this motion itself already committed to, not the robot's
        state at start.
        """
        if not command.get("valid_elbow"):
            return None
        elbow = command["elbow_c"]
        actual_angle, actual_sign = self._robot_elbow()
        if self._elbow_sign is None:
            if self._start_pose is None:
                return None
            offset = float(elbow[0]) - actual_angle
            if abs(offset) > self.start_elbow_tolerance:
                return Violation(
                    CARTESIAN_MOTION_GENERATOR_START_ELBOW_INVALID_INDEX,
                    "elbow_c",
                    "angle",
                    offset,
                    self.start_elbow_tolerance,
                    "rad from q[2]",
                )
            if float(elbow[1]) != actual_sign:
                return Violation(
                    CARTESIAN_MOTION_GENERATOR_START_ELBOW_INVALID_INDEX,
                    "elbow_c",
                    "sign",
                    float(elbow[1]),
                    actual_sign,
                    "sign(q[3])",
                )
            return None
        if float(elbow[1]) != self._elbow_sign:
            return Violation(
                CARTESIAN_MOTION_GENERATOR_ELBOW_SIGN_INCONSISTENT_INDEX,
                "elbow_c",
                "sign",
                float(elbow[1]),
                self._elbow_sign,
                "sign",
            )
        return None

    def _check_elbow_limits(self, command: Dict[str, Any], cycles: int) -> Optional[Violation]:
        """The elbow's velocity, acceleration and jerk, all three named 17.

        ``elbow[0]`` is a commanded *position* on both Cartesian interfaces, so
        its first difference is a velocity on either -- there is no
        interface-relative shift to apply, and the enum has no elbow
        discontinuity name to shift onto: ``kMaxElbowVelocity``,
        ``kMaxElbowAcceleration`` and ``kMaxElbowJerk`` all land on
        ``cartesian_motion_generator_elbow_limit_violation``.

        Franka's suite pins the velocity limit (a 0.3 rad/s^2 ramp that crosses
        1.499 rad/s after ~5 s while acceleration stays at 0.3); nothing pins an
        order between the three, so they are checked lowest-derivative first,
        which is the order in which a ramp reaches them.
        """
        if not command.get("valid_elbow") or self._elbow_sign is None:
            return None
        elbow = command["elbow_c"]
        velocity, acceleration, jerk = self._elbow.derivatives([elbow[0]], cycles)
        checks = (
            (velocity, MAX_ELBOW_VELOCITY, "rad/s"),
            (acceleration, MAX_ELBOW_ACCELERATION, "rad/s^2"),
            (jerk, MAX_ELBOW_JERK, "rad/s^3"),
        )
        for values, limit, unit in checks:
            if abs(values[0]) > limit:
                return Violation(
                    CARTESIAN_MOTION_GENERATOR_ELBOW_LIMIT_VIOLATION_INDEX,
                    "elbow_c",
                    "angle",
                    float(values[0]),
                    limit,
                    unit,
                )
        return None

    def _check_start_velocity(self, dq_c: Sequence[float]) -> Optional[Violation]:
        for index in range(7):
            offset = dq_c[index] - self._joint.value[index]
            if abs(offset) > self.start_velocity_tolerance:
                return Violation(
                    # 15, not 14: on the velocity interface a step in the
                    # commanded signal is an *acceleration* discontinuity. See
                    # the interface-relative naming rule.
                    JOINT_MOTION_GENERATOR_ACCELERATION_DISCONTINUITY_INDEX,
                    "dq_c",
                    f"joint {index + 1}",
                    float(offset),
                    self.start_velocity_tolerance,
                    "rad/s from dq_d",
                )
        return None

    # -- reporting ---------------------------------------------------------

    def applied_derivatives(self) -> Optional[Tuple[List[float], List[float]]]:
        """The first two derivatives of the last applied command, or None.

        These are exactly the fields the FCI publishes back: "``q_{c,k-1}``,
        ``dq_{c,k-1}`` and ``ddq_{c,k-1}`` are always sent back to the user in
        the robot state as ``q_d``, ``dq_d`` and ``ddq_d`` so you will be able
        to compute the resulting derivatives in advance, even in case of packet
        losses" (``docs/overview.rst``). The checker differences every applied
        command anyway, so the numbers already exist here; handing them to the
        server is what lets a client predict what this module will compute.

        For a position generator that is ``(dq_d, ddq_d)``; for a velocity
        generator ``(ddq_d, jerk)``; for the torque controller ``(dtau, ...)``.
        """
        with self._lock:
            if not self._active:
                return None
            if self._mode in (ControlMode.POSITION, ControlMode.VELOCITY):
                return list(self._joint.first), list(self._joint.second)
            if self._mode is ControlMode.TORQUE:
                return list(self._torque.first), list(self._torque.second)
            return None

    def report(self, violation: Violation, *, enforced: bool) -> None:
        """Log ``violation`` once per error class per motion, naming the limit.

        The always-on half of the feature: even with enforcement off, a client
        that commands something the robot would refuse gets told which joint or
        axis, what it sent and what the limit was.
        """
        if not self.should_log(violation):
            return
        logger.warning(
            "motion limit violated: %s%s",
            violation.describe(),
            "" if enforced else " (not enforced)",
        )

    def should_log(self, violation: Violation) -> bool:
        """True the first time this motion sees ``violation``'s error class.

        Rate limiting by error *type* rather than by time: a client that steps
        its target every cycle would otherwise write a thousand identical lines
        a second, and the second one says nothing the first did not.
        """
        with self._lock:
            if violation.error_index in self._logged:
                return False
            self._logged.add(violation.error_index)
            return True

    def latch(self) -> None:
        """Mark a violation as enforced, so only the first one aborts."""
        with self._lock:
            self._violated = True

    @property
    def violated(self) -> bool:
        """Whether a violation is latched (cleared only by :meth:`recover`)."""
        with self._lock:
            return self._violated

    @property
    def active(self) -> bool:
        """Whether a motion this checker validates is running."""
        with self._lock:
            return self._active

    @property
    def motion_id(self) -> int:
        """Token of the running motion, 0 when none is."""
        with self._lock:
            return self._motion_id
