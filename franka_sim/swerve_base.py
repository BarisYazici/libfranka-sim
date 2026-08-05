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

from franka_sim.swerve_kinematics import SwerveKinematics

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

        self._twist = np.zeros(6)
        self._steer_targets = np.zeros(2)
        self._drive_targets = np.zeros(2)

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
        """Resolve the wheel DOF indices. Call once, after ``scene.build()``."""
        self.steer_dofs_idx = [
            self.entity.get_joint(name).dof_idx_local for name in self.steer_joints
        ]
        self.drive_dofs_idx = [
            self.entity.get_joint(name).dof_idx_local for name in self.drive_joints
        ]

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
        """One physics step: write wheel targets and the new base pose."""
        steer_targets, drive_targets = self.solve()
        self.entity.control_dofs_position(steer_targets, self.steer_dofs_idx)
        self.entity.control_dofs_velocity(drive_targets, self.drive_dofs_idx)

        x, y, theta = self.integrate_pose(dt)
        self.entity.set_pos(np.array([x, y, self.base_height]))
        self.entity.set_quat(yaw_to_quat_wxyz(theta))

    def wheel_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Wheel ``(q, dq)`` as 4-element arrays in ``TMR_JOINT_ORDER``."""
        steer_q = self.entity.get_dofs_position(self.steer_dofs_idx).cpu().numpy()
        steer_dq = self.entity.get_dofs_velocity(self.steer_dofs_idx).cpu().numpy()
        drive_q = self.entity.get_dofs_position(self.drive_dofs_idx).cpu().numpy()
        drive_dq = self.entity.get_dofs_velocity(self.drive_dofs_idx).cpu().numpy()

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
