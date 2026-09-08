r"""Differential inverse kinematics for the FCI's two Cartesian interfaces.

The FCI lets a client command the *end effector*, not the joints: a
``kCartesianPosition`` motion streams ``O_T_EE_c`` (a 4x4 pose per millisecond)
and a ``kCartesianVelocity`` motion streams ``O_dP_EE_c`` (a twist). The robot
turns that into joint motion; a simulator has to do the same, or the arm simply
never moves on either interface.

This module is that conversion, and nothing else. It is deliberately free of
MuJoCo, of the wire protocol and of any state: it takes a Jacobian, a desired
twist and a configuration, and returns a joint velocity. The physics backend
owns the model (:meth:`franka_sim.mujoco_franka_sim.MujocoFrankaSim.ee_jacobian`)
and the servo; the FCI server owns the command stream and every limit check.

**Where this sits in the pipeline, and what it is not.** The commanded stream is
checked exactly as it was before -- differentiated, compared against the
Cartesian velocity/acceleration/jerk limits, the elbow limits and the start-pose
rules in :mod:`franka_sim.motion_limits`, and refused when it breaks them. That
layer is hardware-verified and is untouched: this module is a *second consumer*
of the same accepted stream, not a replacement for the first. Everything
downstream is likewise unchanged -- the joint velocity produced here goes into
the same velocity servo a ``kJointVelocity`` motion drives, so the measured-side
safety controller (joint velocity envelope, EE speed) judges a Cartesian motion
with the identical code that judges a joint one.

Method: **damped least squares** (Levenberg-Marquardt / "singularity-robust"
pseudo-inverse, Nakamura & Hanafusa 1986; Wampler 1986)

.. math::

    \dot q = J^{T}\,(J J^{T} + \lambda^{2} I)^{-1}\, v
             + (I - J^{+} J)\, \dot q_{0}

with :math:`v` the desired EE twist and :math:`\dot q_0` an optional null-space
velocity. The damping is what keeps the arm sane near a singularity: a plain
pseudo-inverse asks for unbounded joint speed in the direction that is losing
rank, while the damped form trades a little tracking error for a bounded
:math:`\dot q`. The null-space term is how the FCI's *elbow* command is honoured
-- ``elbow_c[0]`` is the redundancy angle (joint 3 on an FR3), and steering it
inside the null space moves the elbow without moving the end effector, which is
exactly what the interface promises.

A closed-loop term is added to :math:`v` on the pose interface
(:func:`tracking_twist`): the feed-forward alone integrates open-loop and would
accumulate whatever the servo fails to deliver, so the commanded pose is also
compared against the measured one and the error is fed back. Standard CLIK; see
:data:`TRANSLATION_GAIN`.
"""

import threading
from typing import Callable, Optional, Tuple

import numpy as np

from franka_sim.motion_limits import (
    LIMITER_ROTATIONAL_VELOCITY,
    LIMITER_TRANSLATIONAL_VELOCITY,
    rotation_log,
    transform_matrix,
)

#: Damping factor :math:`\lambda` of the damped-least-squares inverse, in the
#: units of the Jacobian's singular values (m/rad mixed with 1, as usual for a
#: geometric Jacobian).
#:
#: **Not a libfranka constant** -- the robot's own IK is not published. 0.05 is
#: chosen against the same measurement that placed
#: :data:`franka_sim.motion_limits.SINGULAR_POSE_MIN_SINGULAR_VALUE`: the FR3's
#: ``sigma_min`` sits around 0.15-0.23 across the whole working region a
#: Cartesian motion ordinarily drives through, so at 0.05 the damping term is
#: ~5% of the
#: smallest singular value there and costs essentially nothing in tracking
#: (measured: sub-millimetre and sub-milliradian on every Cartesian motion
#: exercised), while still bounding :math:`\dot q` at the 0.011 of a genuine
#: singularity -- which a Cartesian ``Move`` is refused from anyway, but which a
#: motion can still wander towards after it has started.
DAMPING = 0.05

#: Proportional gain on the *translational* pose error, in 1/s -- a 25 ms
#: closed-loop time constant. Fast enough that the correction never accumulates
#: (the measured error stays at ~1e-4 m through ordinary Cartesian motions),
#: slow enough to sit two orders of magnitude below the 1 kHz command rate, so
#: it cannot interact with the arrival beat the way a near-deadbeat gain would.
TRANSLATION_GAIN = 40.0

#: The same for the rotational error, in 1/s. Equal to
#: :data:`TRANSLATION_GAIN` on purpose: the two halves of the twist are driven
#: through one shared damped inverse, so giving them different bandwidths would
#: only make the slower one lag behind the faster one's coupling.
ROTATION_GAIN = 40.0

#: Gain on the elbow's null-space correction, in 1/s. An order of magnitude
#: below :data:`TRANSLATION_GAIN`: the elbow is a *redundancy* command, and the
#: null-space term must never be the loudest thing in :math:`\dot q` -- the EE
#: task owns the arm, the elbow only owns what is left over. The FCI's own elbow
#: limits (``kMaxElbowVelocity`` = 1.5 rad/s) bound the commanded stream this
#: chases, so a proportional gain of 5 reaches them from any error the client
#: can legally build up.
ELBOW_GAIN = 5.0


def pose_error(desired: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Twist that carries ``current`` onto ``desired`` in one second.

    Both are 4x4 homogeneous transforms (row-major ``numpy``, *not* the wire's
    column-major layout). The translational half is the plain difference; the
    rotational half is the matrix logarithm of ``R_d R^T``, i.e. the axis-angle
    vector of the residual rotation in the *base* frame -- the same frame the
    Jacobian's rotational rows are expressed in, which is what makes the two
    halves composable into one 6-vector.
    """
    error = np.zeros(6)
    error[:3] = desired[:3, 3] - current[:3, 3]
    error[3:] = rotation_log(desired[:3, :3] @ current[:3, :3].T)
    return error


def clamp_twist(
    twist: np.ndarray,
    max_translation: float = LIMITER_TRANSLATIONAL_VELOCITY,
    max_rotation: float = LIMITER_ROTATIONAL_VELOCITY,
) -> np.ndarray:
    """Scale each half of ``twist`` back to the FR3's Cartesian velocity limits.

    Norm-wise per half, matching how ``limitRate`` bounds a Cartesian velocity
    (``src/rate_limiting.cpp``) and how the FR3's specification states the
    limits -- one bound on the translation vector, one on the rotation vector,
    not six per-axis bounds.

    The bounds are libfranka's own published constants
    (:data:`franka_sim.motion_limits.LIMITER_TRANSLATIONAL_VELOCITY`), i.e. the
    robot's thresholds less ``kLimitEps`` -- the client-side value, because this
    is client-side code: it *emits* a twist. Saturating at the threshold itself
    would hand :data:`franka_sim.motion_limits.MEASURED_CARTESIAN_VELOCITY_LIMIT`
    a speed it then has to decide by rounding, which is the false positive
    documented on :data:`franka_sim.limits.tables.LIMIT_EPS`.

    This is a guard rail, not a feature. On the accepted command stream it never
    binds: a twist that reaches these values has already been refused by
    :meth:`franka_sim.motion_limits.MotionLimitChecker._check_cartesian_velocity`.
    It exists for the case where those checks are switched *off*
    (``--enforce-motion-limits`` absent, the default): the tracking error of a
    client that teleports its commanded pose 10 m would otherwise become a
    400 m/s twist and a joint velocity with no physical meaning at all. Bounding
    the twist keeps the arm's response to an unchecked command merely wrong
    rather than explosive.
    """
    bounded = np.array(twist, dtype=float)
    translation = np.linalg.norm(bounded[:3])
    if translation > max_translation:
        bounded[:3] *= max_translation / translation
    rotation = np.linalg.norm(bounded[3:])
    if rotation > max_rotation:
        bounded[3:] *= max_rotation / rotation
    return bounded


def tracking_twist(
    desired: np.ndarray,
    current: np.ndarray,
    feedforward: np.ndarray,
    translation_gain: float = TRANSLATION_GAIN,
    rotation_gain: float = ROTATION_GAIN,
) -> np.ndarray:
    """Feed-forward plus proportional error: the twist a pose motion should realise.

    ``feedforward`` is the commanded pose's own velocity (see
    :class:`CartesianFeedforward`) and carries the motion; the error term only
    cleans up what the servo did not deliver. Without the feed-forward the arm
    would lag the command by ``v / gain``; without the error term the
    integration would be open-loop and the lag would never be recovered.
    """
    error = pose_error(desired, current)
    return clamp_twist(
        feedforward + np.hstack((translation_gain * error[:3], rotation_gain * error[3:]))
    )


def resolved_rate(
    jacobian: np.ndarray,
    twist: np.ndarray,
    null_velocity: Optional[np.ndarray] = None,
    damping: float = DAMPING,
) -> np.ndarray:
    """Joint velocity that realises ``twist``, damped-least-squares.

    ``jacobian`` is the 6xN geometric Jacobian of the EE frame in the base
    frame, ``twist`` the desired ``[v; omega]``, and ``null_velocity`` an
    optional N-vector projected into the Jacobian's null space -- what is left
    of it after the EE task has taken what it needs.

    The result is deliberately **not** clamped to the joint velocity envelope.
    Driving the arm through that envelope is a real thing a Cartesian command
    can ask for -- a Cartesian motion generator driven towards the joint
    position limits does exactly that, and hardware answers
    ``joint_velocity_violation`` -- so clamping
    here would hide the error the robot is supposed to raise. The bound that
    does apply is physical: the servo's torque is clipped to
    :data:`franka_sim.sim_common.FR3_FORCE_LIMITS`, as it is in every other
    control mode.
    """
    jacobian = np.asarray(jacobian, dtype=float)
    twist = np.asarray(twist, dtype=float)
    square = jacobian @ jacobian.T + (damping**2) * np.eye(jacobian.shape[0])
    if null_velocity is None:
        return jacobian.T @ np.linalg.solve(square, twist)
    # ``(I - J^+ J) n`` written as ``n - J^+ (J n)``: algebraically the same
    # vector, but it never forms the NxN projector or the explicit inverse the
    # pseudo-inverse would need, and the two right-hand sides go through one
    # factorisation instead of two. This runs once per physics step inside a
    # 1 ms real-time loop, where the difference is a third of the control law's
    # cost.
    null_velocity = np.asarray(null_velocity, dtype=float)
    solution = np.linalg.solve(square, np.column_stack((twist, jacobian @ null_velocity)))
    return jacobian.T @ (solution[:, 0] - solution[:, 1]) + null_velocity


def elbow_null_velocity(
    elbow_angle: float, joint_positions: np.ndarray, gain: float = ELBOW_GAIN
) -> np.ndarray:
    """Null-space velocity that steers the elbow towards ``elbow_angle``.

    ``elbow[0]`` is the 7-DOF redundancy angle, which on an FR3 *is* joint 3
    (the same identity :meth:`franka_sim.franka_sim_server.FrankaSimServer.
    _publish_elbow` reports it by), so the correction is a proportional term on
    that one joint and zero on the rest. ``elbow[1]``, the branch flag, is not
    steered: it is the *sign* of joint 4, a discrete choice, and changing it
    means passing through the elbow-straight singularity -- which is why the FCI
    treats a mid-motion flip as an error
    (``cartesian_motion_generator_elbow_sign_inconsistent``) rather than a
    command to follow.
    """
    velocity = np.zeros(len(joint_positions))
    velocity[2] = gain * (float(elbow_angle) - float(joint_positions[2]))
    return velocity


class CartesianFeedforward:
    """The commanded EE twist a Cartesian pose motion is tracking.

    The Cartesian twin of :class:`franka_sim.sim_common.PositionFeedforward`,
    and it exists for the same reason: the network thread writes the commanded
    pose and the physics thread reads it, on two independent ~1 kHz clocks, so a
    naive one-step backward difference alternates between zero and twice the
    true velocity whenever the two beat against each other. The difference is
    therefore taken over the number of physics steps since the target last
    *changed*, and held through the steps where nothing arrived.

    What it cannot share with the joint version is the arithmetic: a rotation is
    not a vector, so the rotational half is ``log(R_k R_{k-1}^T) / span``, the
    axis-angle of the residual rotation over the span, rather than a difference
    of matrices.

    Locked, exactly like the joint version and for the identical reason:
    :meth:`reset` is called from the network threads -- ``set_control_mode`` runs
    on whichever thread accepted the ``Move`` or the UDP command that changed
    modes -- while the physics thread is inside :meth:`step`. Each writes
    ``previous``, ``twist`` and ``_unchanged_steps`` as three separate attribute
    assignments, so an unlucky interleaving could leave a mix of the two (a
    ``reset`` landing between ``step``'s span computation and its ``previous``
    update loses the re-seed's no-spike guarantee). Both bodies are microseconds
    of pure NumPy, so the lock costs nothing measurable.
    """

    def __init__(self, baseline: Optional[np.ndarray] = None):
        """Seed the baseline at ``baseline`` (a 4x4 pose) with a zero twist."""
        self._lock = threading.Lock()
        self.reset(baseline)

    def reset(self, baseline: Optional[np.ndarray] = None) -> None:
        """Re-seed on a discontinuous entry, with the feed-forward back to zero."""
        with self._lock:
            self.previous = None if baseline is None else np.array(baseline, dtype=float)
            self.twist = np.zeros(6)
            self._unchanged_steps = 0

    def step(self, target: np.ndarray, dt: float) -> np.ndarray:
        """Feed-forward twist for this physics step; advances the step count.

        Call exactly once per physics step, with the pose that step is about to
        track. The hold window is
        :data:`franka_sim.sim_common.POSITION_FEEDFORWARD_HOLD_STEPS`, shared
        with the joint feed-forward so both interfaces treat "the client stopped
        streaming" identically.
        """
        from franka_sim.sim_common import POSITION_FEEDFORWARD_HOLD_STEPS

        target = np.asarray(target, dtype=float)
        with self._lock:
            if self.previous is None:
                self.previous = np.array(target, dtype=float)
                return self.twist
            if np.array_equal(target, self.previous):
                self._unchanged_steps += 1
                if self._unchanged_steps >= POSITION_FEEDFORWARD_HOLD_STEPS:
                    self.twist = np.zeros(6)
                return self.twist
            span = (self._unchanged_steps + 1) * dt
            self.twist = np.hstack(
                (
                    (target[:3, 3] - self.previous[:3, 3]) / span,
                    rotation_log(target[:3, :3] @ self.previous[:3, :3].T) / span,
                )
            )
            self.previous = np.array(target, dtype=float)
            self._unchanged_steps = 0
            return self.twist


#: Damping of the damped-least-squares step :class:`CartesianJointSolver` takes
#: while *solving* a pose, as opposed to :data:`DAMPING`, which the servo uses
#: while *tracking* one. Much smaller: the solver is asked where the joints
#: have to be for a pose that moved by microns since the last solution, and
#: every unit of damping slows the convergence of exactly that small step (the
#: contraction per iteration is ``lambda^2 / (sigma_min^2 + lambda^2)``, ~2e-4
#: at 0.002 against the ~0.15 ``sigma_min`` of the working region -- one
#: Newton step lands inside :data:`IK_TOLERANCE`, where 0.01 needed two). It
#: still bounds the step at a genuine singularity, which is all the damping is
#: for here -- a Cartesian ``Move`` is refused from one anyway, at
#: :data:`franka_sim.motion_limits.SINGULAR_POSE_MIN_SINGULAR_VALUE` = 0.05,
#: where this damping is still 1/25 of ``sigma_min``; tracking error and
#: null-space behaviour are the servo's concern.
IK_DAMPING = 0.002

#: Iteration cap of :meth:`CartesianJointSolver.solve_pose`. Seeded from the
#: previous cycle's solution a conforming stream converges in two or three;
#: the cap only matters for the first command of a motion, which may sit up to
#: the start-pose tolerance away from the seed.
IK_MAX_ITERATIONS = 12

#: Residual (m and rad, per half) at which :meth:`CartesianJointSolver.solve_pose`
#: stops iterating. Tight on purpose: the solution is *differenced three times*
#: by the joint-side continuity check, and at 1 ms a residual of 1e-10 rad is
#: 1e-4 rad/s^2 of acceleration noise and 0.1 rad/s^3 of jerk noise -- three
#: orders of magnitude under the tightest threshold the check applies.
IK_TOLERANCE = 1e-10

#: Residual above which the solution is reported as *not converged*; see
#: :meth:`CartesianJointSolver.solve_pose`.
IK_CONVERGED_TOLERANCE = 1e-6


class CartesianJointSolver:
    """The joint-space image of a Cartesian command: where the joints have to be.

    The real controller does not accept a Cartesian pose as a pose. It runs
    inverse kinematics on every ``O_T_EE_c`` it is sent and judges the *joint*
    trajectory that comes out -- which is where
    ``cartesian_motion_generator_joint_velocity_discontinuity`` and
    ``..._joint_acceleration_discontinuity`` come from. This class is the sim's
    version of that step and nothing more: given a forward-kinematics callable
    it turns a commanded pose into the joint configuration that reaches it, and
    a commanded twist into the joint velocity that realises it. The checker
    differences what comes out (see
    :meth:`franka_sim.limits.checker.MotionLimitChecker._check_cartesian_joint_continuity`).

    ``forward(q) -> (pose, jacobian)`` is the only thing it needs from a
    backend: the EE pose (row-major 4x4, in the base frame, ``F_T_EE`` applied)
    and the 6x7 geometric Jacobian of the same frame, both at ``q``. It is
    model-free for the same reason the rest of this module is -- the backend
    owns the model -- and :meth:`franka_sim.mujoco_franka_sim.MujocoFrankaSim.
    joint_space_solver` is the MuJoCo binding of it.

    :meth:`solve_pose` is damped Gauss-Newton on the pose error: from the seed,
    repeat ``q += J^+_damped * pose_error(target, fk(q))`` until the residual is
    under :data:`IK_TOLERANCE`. Seeded from the previous cycle's own solution,
    which is what makes it both fast (the pose moved by microns) and
    *continuous*: the redundancy is resolved the way a resolved-rate
    controller resolves it, by the minimum-norm step from where the joints
    already are, so the joint trajectory of a smooth pose stream is itself
    smooth and its differences mean something. An elbow, when the client
    commands one, is followed in the null space exactly as the servo follows
    it (:func:`elbow_null_velocity`).

    Not thread-safe by itself: ``forward`` typically binds a private
    ``mjData`` and the caller (the checker, under its own lock) serialises
    every call.
    """

    def __init__(
        self,
        forward: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
        damping: float = IK_DAMPING,
        max_iterations: int = IK_MAX_ITERATIONS,
        tolerance: float = IK_TOLERANCE,
    ):
        """Bind ``forward``; see the class docstring for its contract."""
        self._forward = forward
        self.damping = float(damping)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

    def solve_pose(
        self, pose, seed, elbow_angle: Optional[float] = None
    ) -> Tuple[np.ndarray, bool]:
        """Joint configuration reaching ``pose``, iterated from ``seed``.

        ``pose`` is either the wire's 16-element column-major array or a
        row-major 4x4. Returns ``(q, converged)``: ``converged`` is False when
        the residual after :attr:`max_iterations` is still above
        :data:`IK_CONVERGED_TOLERANCE` -- a pose out of reach or at a
        singularity -- and the caller decides what to do with a solution that
        is only approximate. The solution is returned either way, because the
        history it feeds has to move on to *something*, and the least-wrong
        something is the closest configuration found.
        """
        target = np.asarray(pose, dtype=float)
        if target.size == 16 and target.ndim == 1:
            target = transform_matrix(target)
        q = np.array(seed, dtype=float)
        for _ in range(self.max_iterations):
            current, jacobian = self._forward(q)
            error = pose_error(target, current)
            if self._within(error, self.tolerance):
                return q, True
            q = q + resolved_rate(
                jacobian, error, self._null_step(elbow_angle, q, 1.0), damping=self.damping
            )
        current, _ = self._forward(q)
        error = pose_error(target, current)
        return q, self._within(error, IK_CONVERGED_TOLERANCE)

    def joint_velocity(self, q, twist, elbow_angle: Optional[float] = None) -> np.ndarray:
        """Joint velocity realising ``twist`` at ``q``, damped-least-squares."""
        _, jacobian = self._forward(np.asarray(q, dtype=float))
        return resolved_rate(
            jacobian,
            np.asarray(twist, dtype=float)[:6],
            self._null_step(elbow_angle, q, ELBOW_GAIN),
            damping=self.damping,
        )

    @staticmethod
    def _within(error: np.ndarray, tolerance: float) -> bool:
        return bool(
            np.linalg.norm(error[:3]) <= tolerance and np.linalg.norm(error[3:]) <= tolerance
        )

    @staticmethod
    def _null_step(elbow_angle: Optional[float], q, gain: float) -> Optional[np.ndarray]:
        if elbow_angle is None:
            return None
        return elbow_null_velocity(elbow_angle, np.asarray(q, dtype=float), gain=gain)
