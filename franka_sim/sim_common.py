"""Engine-agnostic pieces shared by the Genesis and MuJoCo simulator backends.

Everything here is pure Python/NumPy: the FR3 actuator limits, the joint-damping
override, the column-major pose packing the FCI wire format wants and the
real-time-factor monitor every paced stepping loop reports through. It lives in
its own module so the MuJoCo backends -- now the default -- can reuse the
Genesis backends' calibration and pacing without importing ``genesis`` at all.

``franka_genesis_sim`` and ``mobile_duo_sim`` re-export the names they used to
define, so every existing import path keeps working.
"""

import os

import numpy as np

#: FR3 actuator limits (Nm): joints 1-4 +/-87, joints 5-7 +/-12.
FR3_FORCE_LIMITS = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

#: Default joint viscous damping (Nm*s/rad), applied to all 7 joints unless
#: overridden via $FR3_JOINT_DAMPING. Calibrated against a logged real-robot
#: joint_impedance run so the sim's joint excursions match the real FR3 to within
#: ~5% (the bare MJCF value of 0.21 is far too small and the sim over-travels).
DEFAULT_FR3_DAMPING = 5.0

#: Wall-clock window (s) the WARNING check averages the real-time factor
#: over. Short enough to surface overload promptly, long enough that one
#: slow step (a GC pause, a render frame) doesn't false-trigger it.
RTF_WARN_INTERVAL_S = 5.0

#: Wall-clock window (s) the informational real-time-factor log uses,
#: regardless of value.
RTF_INFO_INTERVAL_S = 60.0

#: Below this real-time factor over an RTF_WARN_INTERVAL_S window, physics
#: is falling behind wall-clock real time and a WARNING is logged.
RTF_WARN_THRESHOLD = 0.95


def resolve_fr3_joint_damping(default: float = DEFAULT_FR3_DAMPING) -> np.ndarray:
    """Resolve per-joint FR3 viscous damping (Nm*s/rad) from $FR3_JOINT_DAMPING.

    A scalar broadcasts to all 7 joints; 7 comma-separated values set each
    joint individually; unset (or empty) falls back to ``default``. Shared by
    the single-arm sims and the mobile-duo scenes so every FR3 arm in the
    process honors the same override with identical parsing/validation.
    """
    damping_env = os.environ.get("FR3_JOINT_DAMPING")
    if damping_env:
        vals = [float(x) for x in damping_env.split(",")]
        if len(vals) not in (1, 7):
            raise ValueError(
                f"FR3_JOINT_DAMPING must be 1 (scalar) or 7 comma-separated "
                f"values, got {len(vals)}"
            )
        return np.array(vals * 7 if len(vals) == 1 else vals, dtype=float)
    return np.full(7, default, dtype=float)


def pose_to_column_major(position, quat_wxyz) -> np.ndarray:
    """Build a column-major 4x4 transform from a position and a (w, x, y, z) quat."""
    w, x, y, z = (float(value) for value in quat_wxyz)
    rotation = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix.T.flatten()


class RealtimeFactorMonitor:
    """Tracks the achieved real-time factor (RTF) of the paced stepping loop.

    ``run_simulation`` calls :meth:`update` once per iteration with the
    current wall-clock time (``time.perf_counter()``, taken *after* that
    iteration's pacing sleep so it reflects the loop's true wall-clock rate)
    and the amount of simulated time the step just advanced (``dt``).

    ``start_wall`` is the loop's own wall-clock origin (``run_simulation``
    passes its ``next_step`` starting value) so the first window's elapsed
    wall time is measured from when the loop actually started, not from an
    arbitrary first ``update()`` call -- seeding the marks from the first
    call instead would silently shift every later window boundary by one
    step's worth of wall time.

    Two independent windows are tracked purely by wall-clock elapsed time:
    every ``RTF_WARN_INTERVAL_S`` the windowed RTF (simulated-time delta /
    wall-time delta) is checked and a WARNING is logged if physics fell
    behind (RTF < ``RTF_WARN_THRESHOLD``, i.e. the loop could not keep pace
    with real time despite the pacing sleep); every ``RTF_INFO_INTERVAL_S``
    the windowed RTF is logged at INFO regardless of value. Nothing is
    logged per-step.
    """

    def __init__(
        self,
        logger_,
        start_wall: float,
        warn_interval_s: float = RTF_WARN_INTERVAL_S,
        info_interval_s: float = RTF_INFO_INTERVAL_S,
        warn_threshold: float = RTF_WARN_THRESHOLD,
    ):
        self._logger = logger_
        self._warn_interval_s = warn_interval_s
        self._info_interval_s = info_interval_s
        self._warn_threshold = warn_threshold

        self._sim_time = 0.0
        self._warn_wall_mark = start_wall
        self._warn_sim_mark = 0.0
        self._info_wall_mark = start_wall
        self._info_sim_mark = 0.0

    def update(self, now: float, dt: float) -> None:
        """Record one step (``dt`` of sim time at wall-clock time ``now``)."""
        self._sim_time += dt

        if now - self._warn_wall_mark >= self._warn_interval_s:
            rtf = self._window_rtf(now, self._warn_wall_mark, self._warn_sim_mark)
            if rtf < self._warn_threshold:
                self._logger.warning(
                    "simulation running at %.2fx real time (physics overloaded)", rtf
                )
            self._warn_wall_mark = now
            self._warn_sim_mark = self._sim_time

        if now - self._info_wall_mark >= self._info_interval_s:
            rtf = self._window_rtf(now, self._info_wall_mark, self._info_sim_mark)
            self._logger.info("simulation running at %.2fx real time", rtf)
            self._info_wall_mark = now
            self._info_sim_mark = self._sim_time

    def _window_rtf(self, now: float, wall_mark: float, sim_mark: float) -> float:
        wall_elapsed = now - wall_mark
        sim_elapsed = self._sim_time - sim_mark
        if wall_elapsed <= 0:
            return 1.0
        return sim_elapsed / wall_elapsed
