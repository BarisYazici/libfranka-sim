"""Backward-Euler differencing of commanded signals, and the pose math it needs.

Control differentiates every commanded signal at the 1 ms cycle and compares
the result against the tables in :mod:`franka_sim.limits.tables`; these are the
differentiators that hold the per-signal history, plus the homogeneous-transform
and rotation log/exp helpers the Cartesian ones are built on.

The differentiators carry the history a violation is judged against, so a
command that is refused must not be recorded here -- that sequencing is
:class:`franka_sim.limits.checker.MotionLimitChecker`'s to keep.
"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

from franka_sim.limits.tables import (
    DELTA_T,
    ORTHONORMAL_THRESHOLD,
    ROTATION_LOG_NEAR_PI,
    ROTATION_LOG_SMALL_ANGLE,
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
