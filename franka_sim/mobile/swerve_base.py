"""Drive a Genesis TMR platform from a body-frame twist.

The platform pose is advanced *kinematically*: the commanded twist is
integrated into a planar pose and written to the entity every physics step,
while the wheel joints are driven so they look and report correctly. On real
hardware the TMR master does the swerve IK onboard and ``tmrv0_2_joint_0..3``
are state-report-only, so this reproduces the same contract without depending on
tyre friction converging in the physics engine.
"""

import logging
import math
from typing import Sequence, Tuple

import numpy as np

from franka_sim.mobile.swerve_kinematics import SwerveKinematics

logger = logging.getLogger(__name__)

#: Steering joints, module 0 (front) then module 1 (rear).
TMR_STEER_JOINTS = ("tmrv0_2_joint_0", "tmrv0_2_joint_2")
#: Drive joints, module 0 (front) then module 1 (rear).
TMR_DRIVE_JOINTS = ("tmrv0_2_joint_1", "tmrv0_2_joint_3")
#: Wire order of the four wheel joints in the reported RobotState arrays.
TMR_JOINT_ORDER = (
    "tmrv0_2_joint_0",
    "tmrv0_2_joint_1",
    "tmrv0_2_joint_2",
    "tmrv0_2_joint_3",
)

#: Nominal module positions in the base frame (x forward, y left), metres.
#: Derived from franka_description robots/tmrv0_2/tmrv0_2.xacro:
#:   front: argo_drive_front is fixed to base_link at (0.3, -0.2, 0.05) --
#:          rigid, so this position is exact.
#:   rear:  argo_drive_rear hangs off rocker_arm_link at (0.0, 0.2, -0.0345),
#:          and rocker_arm_link is attached to base_link at (-0.3, 0.0, 0.0845)
#:          through rocker_arm_joint, which is REVOLUTE (travel -0.16..0.18 rad).
#:          (-0.3, 0.2) is therefore the nominal position with the rocker at
#:          zero; the real lever swings the rear module by a few centimetres as
#:          the suspension articulates.
#: Upstream's SwerveKinematics is likewise parameterised with fixed positions,
#: so this matches what the real master computes -- the suspension travel is an
#: unmodelled error term in both, not a difference between sim and robot.
TMR_WHEEL_POSITIONS = ((0.3, -0.2), (-0.3, 0.2))

#: Wheel radius, metres (argo_drive.xacro: ``radius:='0.05'``).
TMR_WHEEL_RADIUS = 0.05

#: The drive joint axis is +Y and the wheel contacts the ground below its
#: centre, so a positive joint velocity omega rolls the module forward at
#: +omega*r along its heading -- the sign convention SwerveKinematics assumes.
WHEEL_DRIVE_SIGN = 1.0


def yaw_to_quat_wxyz(theta: float) -> np.ndarray:
    """Quaternion for a rotation of ``theta`` about +Z, scalar-first (Genesis)."""
    return np.array([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)])


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle into ``[-pi, pi]``."""
    return math.atan2(math.sin(angle), math.cos(angle))


class SwerveBase:
    """Swerve motion for one Genesis entity whose root is the TMR chassis."""

    def __init__(
        self,
        entity,
        wheel_positions: Sequence[Sequence[float]] = TMR_WHEEL_POSITIONS,
        wheel_radius: float = TMR_WHEEL_RADIUS,
        steer_joints: Sequence[str] = TMR_STEER_JOINTS,
        drive_joints: Sequence[str] = TMR_DRIVE_JOINTS,
        base_height: float = 0.0,
    ):
        self.entity = entity
        self.kinematics = SwerveKinematics(wheel_positions, wheel_radius)
        self.steer_joints = tuple(steer_joints)
        self.drive_joints = tuple(drive_joints)
        self.base_height = float(base_height)

        self.steer_dofs_idx = None
        self.drive_dofs_idx = None
        #: DOFs of the entity's own root (floating) joint; see ``bind``/``apply``.
        self.root_dofs_idx: list = []
        #: qpos entries of that same root joint; empty when the base is fixed.
        self.root_qs_idx: list = []

        self._twist = np.zeros(6)
        self._steer_targets = np.zeros(2)
        self._drive_targets = np.zeros(2)

        # Resolved by bind(); every one of them is a per-step fast path that
        # needs the built entity's indices, so they only exist as placeholders
        # until then.
        self._steer_dofs_np = None
        self._drive_dofs_np = None
        self._last_steer_written = None
        self._last_drive_written = None
        self._root_qpos = None
        self._root_zero_vel = np.zeros(0)

        # Latches so the two rejection paths log once per transition instead of
        # once per call. set_twist runs at ~1 kHz (UDP thread) and solve at
        # ~400 Hz (physics thread); an unlatched warning there floods the log
        # and adds enough jitter to trip communication_constraints_violation on
        # the ROS 2 side.
        self._twist_rejected = False
        self._ik_rejected = False

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def bind(self) -> None:
        """Resolve the wheel and root DOF indices. Call once, after ``scene.build()``."""
        self.steer_dofs_idx = [
            self.entity.get_joint(name).dof_idx_local for name in self.steer_joints
        ]
        self.drive_dofs_idx = [
            self.entity.get_joint(name).dof_idx_local for name in self.drive_joints
        ]
        self.root_dofs_idx = self._resolve_root_dofs()
        self.root_qs_idx = self._resolve_root_qs()
        self._steer_dofs_np = np.asarray(self.steer_dofs_idx, dtype=np.intp)
        self._drive_dofs_np = np.asarray(self.drive_dofs_idx, dtype=np.intp)
        self._last_steer_written = None
        self._last_drive_written = None

        # Scratch buffers for the per-step root write, allocated once: at ~400 Hz
        # the allocations and the float64->float32 conversion Genesis does on
        # every setter argument are not free. z and the two zero quaternion
        # components never change, so only x, y and the yaw pair are rewritten.
        self._root_qpos = np.zeros(7) if len(self.root_qs_idx) == 7 else None
        if self._root_qpos is not None:
            self._root_qpos[2] = self.base_height
        self._root_zero_vel = np.zeros(len(self.root_dofs_idx))

    def _resolve_root_dofs(self) -> list:
        """DOF indices of the entity's root joint (empty when the base is fixed).

        ``apply`` writes the base pose with ``zero_velocity=False`` because
        Genesis' ``set_pos``/``set_quat`` otherwise zero **every** DOF of the
        entity -- on the mobile-duo scene that includes both FR3 arms, which
        pins them (see ``MobileDuoScene.set_spine_position`` for the same trap).
        Only the root's own DOFs still need zeroing after the teleport, so they
        are resolved once here.
        """
        base_joint = getattr(self.entity, "base_joint", None)
        if base_joint is None:
            return []
        dof_idx = getattr(base_joint, "dof_idx_local", None)
        if dof_idx is None:
            return []
        return [int(idx) for idx in np.atleast_1d(dof_idx)]

    def _resolve_root_qs(self) -> list:
        """Generalised-coordinate indices of the root joint (empty when fixed).

        A *free* root joint owns seven qpos entries, laid out
        ``[x, y, z, qw, qx, qy, qz]`` -- that is what Genesis' own
        ``set_links_pos``/``set_links_quat`` kernels write for a non-fixed base
        link. ``apply`` writes all seven in one ``set_qpos`` when they are
        present, because every Genesis setter runs a full forward-kinematics and
        geometry pass after its write: the per-step cost is counted in setter
        *calls*, not in values written, so folding the pose into one call halves
        it. A fixed base has no qpos of its own (its pose lives directly in
        ``links_state``) and falls back to ``set_pos``/``set_quat``.
        """
        base_joint = getattr(self.entity, "base_joint", None)
        if base_joint is None:
            return []
        q_idx = getattr(base_joint, "q_idx_local", None)
        if q_idx is None:
            return []
        return [int(idx) for idx in np.atleast_1d(q_idx)]

    def reset_pose(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        """Reset the integrated planar pose."""
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)

    def set_twist(self, twist: Sequence[float]) -> None:
        """Publish the latest body-frame twist ``[vx, vy, vz, wx, wy, wz]``.

        Lock-free like the arm command arrays: a fresh array is bound in one
        bytecode step (atomic under the GIL) and the physics thread is the only
        reader. Non-finite commands are dropped so a broken client cannot poison
        the integrated pose.

        Runs at ~1 kHz, so the rejection is logged only on the transition into
        the rejecting state (see ``self._twist_rejected``).
        """
        values = np.asarray(twist, dtype=float).reshape(6)
        if not np.all(np.isfinite(values)):
            if not self._twist_rejected:
                logger.warning("Ignoring non-finite base twist: %s", values)
                self._twist_rejected = True
            return
        self._twist_rejected = False
        self._twist = values

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Swerve IK on the latest twist -> ``(steer_targets, drive_targets)``.

        Runs once per physics step (~400 Hz), so the rejection is logged only on
        the transition into the rejecting state (see ``self._ik_rejected``).
        """
        vx = float(self._twist[0])
        vy = float(self._twist[1])
        wz = float(self._twist[5])

        solution = self.kinematics.inverse_kinematics(vx, vy, wz)
        if solution is None:
            # Unreachable by construction today: set_twist() already drops any
            # non-finite command before it reaches self._twist, and
            # SwerveKinematics.inverse_kinematics only returns None for
            # non-finite vx/vy/wz. Kept as defence-in-depth in case either
            # invariant changes.
            if not self._ik_rejected:
                logger.warning("Swerve IK rejected twist (%s, %s, %s); holding", vx, vy, wz)
                self._ik_rejected = True
            return self._steer_targets, self._drive_targets

        self._ik_rejected = False
        steering_angles, wheel_speeds = solution
        self._steer_targets = np.asarray(steering_angles, dtype=float)
        self._drive_targets = np.asarray(wheel_speeds, dtype=float) * WHEEL_DRIVE_SIGN
        return self._steer_targets, self._drive_targets

    def integrate_pose(self, dt: float) -> Tuple[float, float, float]:
        """Advance the planar pose by one ``dt`` of the commanded body twist."""
        vx = float(self._twist[0])
        vy = float(self._twist[1])
        wz = float(self._twist[5])

        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        self.x += (vx * cos_theta - vy * sin_theta) * dt
        self.y += (vx * sin_theta + vy * cos_theta) * dt
        self.theta = wrap_to_pi(self.theta + wz * dt)
        return self.x, self.y, self.theta

    def apply(self, dt: float) -> None:
        """One physics step: write wheel targets and the new base pose.

        Genesis control targets persist until overwritten, so the wheel
        targets are rewritten only when the IK solution actually changes:
        at ~400 Hz the kernel-launch overhead of redundant writes is what
        dominates the physics loop, not the physics itself. The kinematic
        root teleport below still runs every step so contact/reaction
        forces can never drift the base between (sparse) twist updates.
        """
        steer_targets, drive_targets = self.solve()
        if not (
            np.array_equal(steer_targets, self._last_steer_written)
            and np.array_equal(drive_targets, self._last_drive_written)
        ):
            self.entity.control_dofs_position(steer_targets, self.steer_dofs_idx)
            self.entity.control_dofs_velocity(drive_targets, self.drive_dofs_idx)
            self._last_steer_written = steer_targets.copy()
            self._last_drive_written = drive_targets.copy()

        x, y, theta = self.integrate_pose(dt)
        self._write_root_pose(x, y, theta)

    def _write_root_pose(self, x: float, y: float, theta: float) -> None:
        """Teleport the chassis root to the integrated planar pose.

        ``zero_velocity=False`` is load-bearing: Genesis' pose setters default to
        zeroing EVERY DOF of the entity, so on the mobile-duo scene these
        kinematic writes would zero both arms' joint velocities every physics
        step -- an effectively infinite damper that pins the arms while the
        (kinematically teleported) base still looks fine. Only the root's own
        DOFs are zeroed, which is what the teleport actually invalidates.

        A free root's whole pose goes in one ``set_qpos`` rather than a
        ``set_pos``/``set_quat`` pair; see :meth:`_resolve_root_qs` for the qpos
        layout that relies on and why the call count is what costs.
        """
        if self._root_qpos is not None:
            self._root_qpos[0] = x
            self._root_qpos[1] = y
            self._root_qpos[3] = math.cos(theta / 2.0)
            self._root_qpos[6] = math.sin(theta / 2.0)
            self.entity.set_qpos(self._root_qpos, self.root_qs_idx, zero_velocity=False)
        else:
            self.entity.set_pos(np.array([x, y, self.base_height]), zero_velocity=False)
            self.entity.set_quat(yaw_to_quat_wxyz(theta), zero_velocity=False)
        if self.root_dofs_idx:
            self.entity.set_dofs_velocity(self._root_zero_vel, self.root_dofs_idx)

    def wheel_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Wheel ``(q, dq)`` as 4-element arrays in ``TMR_JOINT_ORDER``."""
        steer_q = self.entity.get_dofs_position(self.steer_dofs_idx).cpu().numpy()
        steer_dq = self.entity.get_dofs_velocity(self.steer_dofs_idx).cpu().numpy()
        drive_q = self.entity.get_dofs_position(self.drive_dofs_idx).cpu().numpy()
        drive_dq = self.entity.get_dofs_velocity(self.drive_dofs_idx).cpu().numpy()

        positions = np.array([steer_q[0], drive_q[0], steer_q[1], drive_q[1]])
        velocities = np.array([steer_dq[0], drive_dq[0], steer_dq[1], drive_dq[1]])
        return positions, velocities

    def wheel_state_from(
        self, q_all: np.ndarray, dq_all: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Wheel ``(q, dq)`` sliced from whole-entity DOF arrays.

        Same ``TMR_JOINT_ORDER`` composition as :meth:`wheel_state`, but with
        no Genesis reads of its own -- the mobile-duo loop already reads the
        entity's full DOF state once per step and passes it in.
        """
        steer_q = q_all[self._steer_dofs_np]
        steer_dq = dq_all[self._steer_dofs_np]
        drive_q = q_all[self._drive_dofs_np]
        drive_dq = dq_all[self._drive_dofs_np]

        positions = np.array([steer_q[0], drive_q[0], steer_q[1], drive_q[1]])
        velocities = np.array([steer_dq[0], drive_dq[0], steer_dq[1], drive_dq[1]])
        return positions, velocities

    def base_pose_matrix(self) -> np.ndarray:
        """Base pose as a column-major 16-element transform (``O_T_EE`` layout)."""
        cos_theta = math.cos(self.theta)
        sin_theta = math.sin(self.theta)
        matrix = np.eye(4)
        matrix[:3, :3] = np.array(
            [
                [cos_theta, -sin_theta, 0.0],
                [sin_theta, cos_theta, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        matrix[:3, 3] = [self.x, self.y, self.base_height]
        return matrix.T.flatten()
