"""Engine-agnostic pieces shared by the Genesis and MuJoCo simulator backends.

Everything here is pure Python/NumPy: the FR3 actuator limits, the joint-damping
override, the column-major pose packing the FCI wire format wants and the
real-time-factor monitor every paced stepping loop reports through. It lives in
its own module so the MuJoCo backends -- now the default -- can reuse the
Genesis backends' calibration and pacing without importing ``genesis`` at all.

``franka_genesis_sim`` and ``mobile.duo_sim`` re-export the names they used to
define, so every existing import path keeps working.
"""

import os
import threading

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


#: How many consecutive physics steps a POSITION-mode target may stay unchanged
#: before its velocity feedforward is treated as stale and dropped to zero.
#: Three: the worst beat two independent 1 kHz clocks can produce is a couple of
#: steps with no arrival followed by one carrying the whole backlog, and holding
#: through that is the entire point of :class:`PositionFeedforward`. Longer than
#: that is not a beat, it is a client that stopped streaming (or one holding a
#: pose on purpose), and carrying a stale ``dq_c`` into that would keep pushing
#: an arm nobody is driving any more.
POSITION_FEEDFORWARD_HOLD_STEPS = 3


class PositionFeedforward:
    """The commanded joint velocity a POSITION-mode servo damps against.

    One instance per controlled arm, owned and advanced by the physics thread.

    ``dq_c`` is a backward difference of the *commanded* target, and the naive
    one-step form ``(target - previous) / dt`` is only right when exactly one
    UDP command lands per physics step. Nothing enforces that: the network
    thread writes the target and the physics thread reads it, on two independent
    ~1 kHz clocks that beat against each other. A ``[0, 2]`` beat -- one step
    that sees no new target, the next that sees two of them -- makes the naive
    form alternate between ``0`` and ``2v``, and at ``ARM_POSITION_KD = 450`` a
    0.5 rad/s stream turns that into a +/-225 Nm square wave that spends nearly
    every step clipped on the +/-87 Nm actuator rail. Measured on an idle box it
    is ~0.7% of steps; under load it was 89.6%.

    So the difference is taken over the interval it actually covers: the number
    of physics steps since the target last *changed*. When it changes after
    ``n`` steps, ``dq_c = (target - previous) / (n * dt)`` -- the mean commanded
    velocity over exactly the span the change happened in -- and that value is
    then *held* through the following no-change steps instead of collapsing to
    zero and back. Lockstep is untouched (``n`` is always 1, so ``dq_c = v``
    every step, bit for bit what it was before); the same stream arriving in a
    ``[0, 2]`` beat now also yields ``v`` every step.

    Counting steps rather than filtering is deliberate. A low-pass on ``dq_c``
    would smear every genuine target change across several steps, lag every real
    acceleration, and *still* pass a share of the beat's amplitude through. The
    beat is not noise in the signal, it is bookkeeping error about *when* a
    sample arrived -- counting the steps removes it exactly, and removes nothing
    else.

    After :data:`POSITION_FEEDFORWARD_HOLD_STEPS` unchanged steps the stream is
    taken to have ended: the held ``dq_c`` drops to zero and the baseline
    re-bases onto the current target, so a constant target settles to exactly
    the old ``KP*error - KD*dq`` law and stays there.

    ``reset`` is called from the TCP/UDP threads (a mode switch, an idle-hold
    engage) while ``step`` is called from the physics thread on every tick.
    Each writes ``previous``, ``dq_c`` and ``_unchanged_steps`` as three
    separate attribute assignments, so an unlucky interleaving between the
    two could observe (or leave behind) a mix of old and new values -- e.g. a
    ``reset`` landing between ``step``'s span computation and its
    ``self.previous`` update would lose the re-seed's no-spike guarantee. A
    lock around both bodies closes that window; both methods are microseconds
    of pure NumPy, so there is no meaningful contention cost.
    """

    def __init__(self, baseline):
        """Seed the baseline at ``baseline`` with a zero feedforward."""
        self._lock = threading.Lock()
        self.reset(baseline)

    def reset(self, baseline) -> None:
        """Re-seed on a discontinuous entry (mode switch, idle-hold engage).

        Both halves have to go: the baseline snaps to whatever target is current
        *now*, and the held feedforward goes to zero. The first physics step
        after a switch therefore sees ``dq_c = 0`` -- no spike -- whichever half
        of the state was stale.
        """
        with self._lock:
            self.previous = np.array(baseline, dtype=float)
            self.dq_c = np.zeros_like(self.previous)
            self._unchanged_steps = 0

    def step(self, target, dt: float) -> np.ndarray:
        """Feedforward velocity for this physics step; advances the step count.

        Call exactly once per physics step per arm, with the target that step is
        about to servo to.
        """
        target = np.asarray(target, dtype=float)
        with self._lock:
            if np.array_equal(target, self.previous):
                self._unchanged_steps += 1
                if self._unchanged_steps >= POSITION_FEEDFORWARD_HOLD_STEPS:
                    # Stale: nobody is streaming. Drop the feedforward -- but
                    # leave _unchanged_steps counting. self.previous already
                    # equals target here, so the next real change still
                    # differences against the sample where the arm was
                    # actually left; only resetting the counter would forget
                    # how many steps the gap spanned and difference the
                    # resumed target over 1 step instead of n, reading dq_c
                    # as n times the true velocity.
                    self.dq_c = np.zeros_like(self.previous)
                return self.dq_c

            span = (self._unchanged_steps + 1) * dt
            self.dq_c = (target - self.previous) / span
            self.previous = np.array(target, dtype=float)
            self._unchanged_steps = 0
            return self.dq_c


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


#: How long :func:`close_passive_viewer` waits for the viewer's own render
#: thread to finish destroying its GL context. Teardown is a handful of X
#: round-trips; a second is already generous, and waiting is strictly better
#: than the alternative (see that function's docstring), so this only bounds
#: the pathological case.
VIEWER_CLOSE_TIMEOUT_S = 2.0


def launch_passive_viewer(model, data):
    """Open a passive MuJoCo viewer and return ``(handle, render_thread)``.

    ``mujoco.viewer.launch_passive`` starts a private daemon thread that calls
    ``glfwInit()``, creates the window and owns the GL context for its whole
    life, but it only hands back the ``Handle`` -- not the thread. Shutdown
    needs the thread (see :func:`close_passive_viewer`), so it is recovered
    here by matching the thread's target against the launcher's internal entry
    point. ``None`` if it cannot be identified (macOS runs the viewer on the
    process' main thread under ``mjpython``, so there is no such thread).
    """
    import mujoco.viewer

    handle = mujoco.viewer.launch_passive(model, data)
    thread = next(
        (
            t
            for t in threading.enumerate()
            if getattr(t, "_target", None) is mujoco.viewer._launch_internal
        ),
        None,
    )
    return handle, thread


def close_passive_viewer(handle, thread, logger_, timeout: float = VIEWER_CLOSE_TIMEOUT_S) -> None:
    """Shut a passive viewer down without wedging the process on exit.

    The passive viewer's GL/GLFW state belongs to the render thread that
    created it: that thread calls ``glfwInit()``, ``glfwCreateWindow()`` and,
    once ``Handle.close()`` sets the exit request, ``Simulate.destroy()`` ->
    ``glfwDestroyWindow()`` -> ``glXDestroyContext()``. Two things then race
    the main thread against it:

    1. ``mujoco.viewer._launch_internal`` does ``atexit.register(glfw.terminate)``,
       and atexit handlers run on the **main** thread. ``glfwTerminate()``
       destroys any window still alive, so while the render thread is inside
       its own ``glfwDestroyWindow()`` the same GLX context gets destroyed
       twice -- ``GLXBadContext``, which Xlib's default error handler answers
       by calling ``exit(1)`` from whichever thread hit it. Re-entering
       ``exit()`` from inside an atexit handler is undefined behaviour, and in
       practice both threads plus the driver's own worker deadlock in the GL
       teardown path. The process then sits in C forever: no Python bytecode
       ever runs again, so the pending SIGINT is never handled and Ctrl+C does
       nothing at all.
    2. Even without the atexit handler, returning from ``stop()`` while the
       render thread is mid-teardown lets the interpreter start finalising
       under it.

    So: drop the atexit handler first (the render thread does the same work,
    on the thread that is allowed to do it), then ask the viewer to exit, then
    wait for that thread to actually finish. Bounded by ``timeout`` -- a
    render thread wedged in the driver must not make ``stop()`` hang, which is
    the bug this whole function exists to avoid.
    """
    _disarm_glfw_atexit()

    if handle is not None:
        try:
            handle.close()
        except Exception:  # pragma: no cover - viewer already gone
            logger_.debug("Error closing the MuJoCo viewer", exc_info=True)

    if thread is not None and thread.is_alive():
        thread.join(timeout=timeout)
        if thread.is_alive():
            logger_.warning(
                "MuJoCo viewer thread did not finish its GL teardown within %.1fs; "
                "abandoning it (it is a daemon thread)",
                timeout,
            )


def _disarm_glfw_atexit() -> None:
    """Unregister the ``glfw.terminate`` atexit hook the viewer installs.

    See :func:`close_passive_viewer`: running it on the main thread races the
    render thread's own teardown. Skipping it leaks nothing that matters --
    the process is exiting, and the render thread already destroyed the
    window and context.
    """
    try:
        import atexit

        import glfw
    except ImportError:  # pragma: no cover - glfw ships with the viewer
        return
    try:
        atexit.unregister(glfw.terminate)
    except Exception:  # pragma: no cover - defensive
        pass
