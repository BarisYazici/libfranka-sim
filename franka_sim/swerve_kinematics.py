"""Two-module swerve-drive kinematics.

Python port of ``franka_mobile::SwerveKinematics`` from
`frankarobotics/franka_ros2 <https://github.com/frankarobotics/franka_ros2>`_,
branch ``jazzy``:

* ``franka_mobile/src/swerve_kinematics.cpp``
* ``franka_mobile/include/franka_mobile/swerve_kinematics.hpp``

Copyright (c) 2026 Franka Robotics GmbH

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Deliberate differences from the C++ original:

* Results are returned instead of written through reference out-parameters.
* The C++ ``bool`` return becomes ``None`` for "no valid solution"; a successful
  call always returns a value.
"""

import math
from typing import List, Optional, Sequence, Tuple

import numpy as np

#: Below this wheel speed (rad/s) the previous steering angle is kept, so a
#: near-zero command does not make the modules chase numerical noise.
SPEED_EPSILON = 1e-3


class SwerveKinematics:
    """Forward/inverse kinematics for a two-module swerve (holonomic) platform.

    Wheel positions are given in the robot base frame with X forward and Y left.
    Velocities are expressed in the robot body frame.

    Not thread-safe: :meth:`inverse_kinematics` stores the last commanded
    steering angles to resolve the heading ambiguity (a module pointing at
    ``theta`` is equivalent to one pointing at ``theta + pi`` with reversed
    speed), so a single instance must be driven from one thread.
    """

    def __init__(self, wheel_positions: Sequence[Sequence[float]], wheel_radius: float):
        if not math.isfinite(wheel_radius) or abs(wheel_radius) < 1e-3 or wheel_radius < 0:
            raise ValueError("Wheel radius must be positive")

        positions = [np.asarray(position, dtype=float).reshape(2) for position in wheel_positions]
        if len(positions) != 2:
            raise ValueError("Exactly two wheel positions are required")
        for position in positions:
            if np.all(np.abs(position) <= 1e-3):
                raise ValueError("Wheel position cannot be zero")

        self._wheel_positions = positions
        self._wheel_radius = float(wheel_radius)
        self._steering_angles = [0.0, 0.0]
        self._wheel_speeds = [0.0, 0.0]

    @property
    def wheel_radius(self) -> float:
        """Wheel radius in metres."""
        return self._wheel_radius

    def forward_kinematics(
        self, steering_angles: Sequence[float], wheel_speeds: Sequence[float]
    ) -> Optional[Tuple[float, float, float]]:
        """Body-frame ``(vx, vy, wz)`` from wheel states (closed form)."""

        def wheel_velocity(index: int) -> np.ndarray:
            return (
                wheel_speeds[index]
                * self._wheel_radius
                * np.array([math.cos(steering_angles[index]), math.sin(steering_angles[index])])
            )

        velocity_0 = wheel_velocity(0)
        velocity_1 = wheel_velocity(1)

        mean_velocity = (velocity_0 + velocity_1) / 2.0

        def cross_2d(velocity: np.ndarray, radius: np.ndarray) -> float:
            return float(velocity[1] * radius[0] - velocity[0] * radius[1])

        norm_sq = float(
            self._wheel_positions[0].dot(self._wheel_positions[0])
            + self._wheel_positions[1].dot(self._wheel_positions[1])
        )
        wz = (
            cross_2d(velocity_0, self._wheel_positions[0])
            + cross_2d(velocity_1, self._wheel_positions[1])
        ) / norm_sq
        return float(mean_velocity[0]), float(mean_velocity[1]), float(wz)

    def forward_kinematics_qr(
        self, steering_angles: Sequence[float], wheel_speeds: Sequence[float]
    ) -> Optional[Tuple[float, float, float]]:
        """Body-frame ``(vx, vy, wz)`` from wheel states (least squares).

        Per wheel the rigid-body constraint is ``v*cos(theta) = vx - ry*wz`` and
        ``v*sin(theta) = vy + rx*wz``; the resulting 4x3 system is solved in a
        least-squares sense.
        """
        matrix = np.zeros((4, 3))
        rhs = np.zeros(4)

        for index in range(2):
            speed = wheel_speeds[index] * self._wheel_radius
            position = self._wheel_positions[index]

            matrix[2 * index] = [1.0, 0.0, -float(position[1])]
            matrix[2 * index + 1] = [0.0, 1.0, float(position[0])]

            rhs[2 * index] = speed * math.cos(steering_angles[index])
            rhs[2 * index + 1] = speed * math.sin(steering_angles[index])

        solution, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
        if not np.all(np.isfinite(solution)):
            return None
        return float(solution[0]), float(solution[1]), float(solution[2])

    def inverse_kinematics(
        self, vx: float, vy: float, wz: float
    ) -> Optional[Tuple[List[float], List[float]]]:
        """Wheel commands for a desired body-frame twist.

        Each module heading is chosen to minimise steering travel from the
        previously commanded angle, resolving the pi-ambiguity by reversing the
        wheel speed when the direct heading would need more than a quarter turn.
        """
        if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(wz)):
            return None

        steering_angles = [0.0, 0.0]
        wheel_speeds = [0.0, 0.0]

        for index in range(2):
            rx = float(self._wheel_positions[index][0])
            ry = float(self._wheel_positions[index][1])

            # Wheel contact velocity in the base frame.
            vx_wheel = vx - wz * ry
            vy_wheel = vy + wz * rx

            speed = math.hypot(vx_wheel, vy_wheel) / self._wheel_radius
            if speed > SPEED_EPSILON:
                angle = math.atan2(vy_wheel, vx_wheel)
            else:
                angle = self._steering_angles[index]

            if abs(angle - self._steering_angles[index]) > math.pi / 2.0:
                if speed > SPEED_EPSILON:
                    angle = math.atan2(-vy_wheel, -vx_wheel)
                else:
                    angle = self._steering_angles[index]
                speed = -speed

            self._steering_angles[index] = angle
            self._wheel_speeds[index] = speed
            steering_angles[index] = angle
            wheel_speeds[index] = speed

        return steering_angles, wheel_speeds
