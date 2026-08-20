"""One Genesis scene serving three FCI bridges: left arm, right arm, TMR base.

The combined ``mobile_fr3_duo`` URDF is loaded as a single entity, so base
motion physically carries both arms. Each protocol server attaches through a
:class:`~franka_sim.mobile_duo_common.SceneView`, a thin adapter that
implements the simulator contract ``FrankaSimServer`` expects but scopes every
read and write to one role.

Only the runner may build the scene or step physics: with three servers on one
scene, three physics loops would corrupt it, so the view lifecycle methods are
no-ops.

The roles, joint/link names, initial pose and spine travel -- everything the
MuJoCo backend must agree on too -- and :class:`SceneView` itself live in
:mod:`franka_sim.mobile_duo_common`, a genesis-free module, and are re-imported
below so every existing ``from franka_sim.mobile_duo_sim import X`` keeps
working. This module imports ``genesis`` at module level, so it is *not* safe
for a genesis-free install -- only the MuJoCo backend and the runner need to
avoid that, and they import the shared names from ``mobile_duo_common``
directly instead of from here.
"""

import logging
import platform
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import genesis as gs
import numpy as np

from franka_sim.franka_genesis_sim import (
    ControlMode,
    resolve_fr3_joint_damping,
    resolve_gs_backend,
)
from franka_sim.mobile_duo_common import (
    ARM_EE_LINKS,
    ARM_INITIAL_Q,
    ARM_JOINT_NAMES,
    ARM_ROLES,
    ROLE_BASE,
    ROLE_LEFT,
    ROLE_RIGHT,
    ROLES,
    SPINE_JOINT_NAME,
    SPINE_LIMITS_M,
    SceneView,
)
from franka_sim.sim_common import (
    FR3_FORCE_LIMITS,
    RealtimeFactorMonitor,
    pose_to_column_major,
)
from franka_sim.swerve_base import SwerveBase
from franka_sim.urdf_assets import resolve_urdf_meshes

logger = logging.getLogger(__name__)

#: Re-exported so every existing ``from franka_sim.mobile_duo_sim import X``
#: keeps working now that the engine-agnostic names live in
#: ``mobile_duo_common`` (see the module docstring).
__all__ = [
    "ROLE_LEFT",
    "ROLE_RIGHT",
    "ROLE_BASE",
    "ARM_ROLES",
    "ROLES",
    "ARM_JOINT_NAMES",
    "ARM_EE_LINKS",
    "ARM_INITIAL_Q",
    "SPINE_JOINT_NAME",
    "SPINE_LIMITS_M",
    "SceneView",
    "MobileDuoScene",
]

#: Position-hold gains and force ceiling for the lift DOF. The URDF gives the
#: joint no actuation of its own, so once the teleport stops repeating it
#: free-runs under the arms' reaction forces -- a sub-millimetre sawtooth that
#: rides straight into both flanges' O_T_EE. A stiff PD target persists in
#: Genesis at zero per-step cost and holds it instead. The gains are not a model
#: of the real drive (the real spine is a separate closed-loop device the sim
#: only mirrors), so the URDF's ``effort="100"`` -- which describes that drive --
#: does not bound them; the ceiling only has to beat the reaction forces.
SPINE_HOLD_KP = 1.0e5
SPINE_HOLD_KV = 1.0e4
SPINE_HOLD_FORCE_N = 2000.0

#: Visualizer refresh rate when the scene is started with a viewer. Genesis'
#: viewer window repaints on its own thread, but the scene-graph update that
#: feeds it runs inside ``scene.step`` on the physics thread, and its whole cost
#: lands on the single step that happens to carry it -- pacing cannot claw that
#: back, so it comes straight off the real-time factor (measured ~2% of the
#: budget at this rate). 30 Hz is the point where the viewer still reads as
#: smooth motion; drop it if a scene ever needs the headroom back.
RENDER_FPS = 30.0

#: Physics steps between two whole-entity link-pose reads. The joint state
#: (q/dq) is re-read every step because the UDP RobotState carries it at 1 kHz,
#: but the two link-pose reads only feed the arms' ``O_T_EE``, whose consumers
#: are the 30 Hz ROS 2 state publishers and the teleop stack's FrankaRobotState
#: at 50 Hz. At dt=2.5 ms this still refreshes the flange pose at 100 Hz --
#: twice the fastest consumer -- for a quarter of the kernel launches. Raising
#: it further would start to alias those consumers; lowering it to 1 restores
#: per-step reads.
LINK_POSE_READ_EVERY = 4


class MobileDuoScene:
    """The single Genesis scene holding the TMR base, the spine and both arms."""

    def __init__(
        self,
        urdf_path,
        mesh_root=None,
        enable_vis: bool = False,
        dt: float = 0.0025,
        base_height: float = 0.05,
    ):
        self.urdf_path = Path(urdf_path)
        self.mesh_root = mesh_root
        self.enable_vis = enable_vis
        self.dt = dt
        self.base_height = base_height

        self.scene = None
        self.robot = None
        self.running = False
        self.swerve: Optional[SwerveBase] = None
        self.arm_dofs_idx: Dict[str, List[int]] = {}
        self.arm_links = {}
        self.spine_dof_idx: Optional[int] = None

        # Optional lift source: any object with ``position_m() -> float``. The
        # runner sets this to the spine stub's SpineModel when started with
        # --spine, which is what makes the lift move in the viewer. Left None,
        # this module has no dependency on spine_stub at all.
        self.spine_model = None

        self.arm_control_modes = {role: ControlMode.POSITION for role in ARM_ROLES}
        self.arm_torques = {role: np.zeros(7) for role in ARM_ROLES}
        self.arm_joint_positions = {role: ARM_INITIAL_Q.copy() for role in ARM_ROLES}
        self.arm_joint_velocities = {role: np.zeros(7) for role in ARM_ROLES}

        self._prev_dq = {role: np.zeros(7) for role in ROLES}
        self._ddq_filtered = {role: np.zeros(7) for role in ROLES}
        self._alpha_acc = 0.95
        self._resolved_urdf: Optional[Path] = None

        # Per-step fast-path caches. The index maps need the built entity, so
        # _bind_entity fills them in; the rest are the "what did we last write /
        # last read" state the write-on-change and decimated-read paths compare
        # against, and start deliberately empty so the first pass through the
        # loop writes and reads everything.
        self._arm_dofs_np: Dict[str, np.ndarray] = {}
        self._arm_link_idx: Dict[str, int] = {}
        self._last_arm_cmd = {role: (None, None) for role in ARM_ROLES}
        self._spine_last_written: Optional[float] = None
        self._link_read_countdown = 0
        self._arm_ee_pose = {role: np.eye(4).T.flatten() for role in ARM_ROLES}

        empty = {
            "q": np.zeros(7),
            "dq": np.zeros(7),
            "ddq": np.zeros(7),
            "q_d": np.zeros(7),
            "dq_d": np.zeros(7),
            "ddq_d": np.zeros(7),
            "tau_J": np.zeros(7),
            "O_T_EE": np.eye(4).T.flatten(),
        }
        self._state_snapshots = {role: dict(empty) for role in ROLES}

    # -- construction ------------------------------------------------------

    def initialize_simulation(self) -> None:
        """Build the scene from the combined URDF and bind every joint group."""
        if not getattr(gs, "_initialized", False):
            try:
                gs.init(backend=resolve_gs_backend(gs), logging_level=None)
            except Exception as exc:
                if "already initialized" not in str(exc).lower():
                    raise

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.8),
                camera_fov=40,
                res=(640, 480),
                # Must stay >= RENDER_FPS: this caps a rate limiter inside
                # Genesis' viewer.update(), which sleeps on whatever thread
                # called it -- here the physics thread. run_simulation does its
                # own visualizer pacing at the slower RENDER_FPS, so the limiter
                # never fires; set it below that and it would stall physics.
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(dt=self.dt),
            show_viewer=self.enable_vis,
            show_FPS=False,
        )
        self.scene.add_entity(gs.morphs.Plane())

        self._resolved_urdf = resolve_urdf_meshes(self.urdf_path, mesh_root=self.mesh_root)
        # If anything from here on raises, the resolved-URDF temp file (and
        # its dae->obj conversion cache) would otherwise leak: stop() is never
        # reached, so its unlink never runs. Mirror that cleanup here on the
        # failure path, then re-raise.
        try:
            self.robot = self.scene.add_entity(
                gs.morphs.URDF(
                    file=str(self._resolved_urdf),
                    pos=(0.0, 0.0, self.base_height),
                    fixed=False,
                    merge_fixed_links=False,
                ),
                material=gs.materials.Rigid(gravity_compensation=1.0),
            )
            self.scene.build()

            self._bind_entity()

            # Same viscous damping as the single-arm sim, applied identically to
            # both arms; override both via $FR3_JOINT_DAMPING (see
            # franka_genesis_sim.resolve_fr3_joint_damping for the default and
            # parsing rules).
            damping = resolve_fr3_joint_damping()
            logger.info("Arm joint damping (left + right): %s", damping)
            for role in ARM_ROLES:
                dofs = self.arm_dofs_idx[role]
                self.robot.set_dofs_force_range(
                    lower=-FR3_FORCE_LIMITS, upper=FR3_FORCE_LIMITS, dofs_idx_local=dofs
                )
                self.robot.set_dofs_damping(damping, dofs)
                self.robot.set_dofs_position(ARM_INITIAL_Q, dofs)

            # Arm the lift's position hold once; set_spine_position then only
            # has to move the target. See SPINE_HOLD_KP.
            spine_dofs = [self.spine_dof_idx]
            self.robot.set_dofs_force_range(
                lower=np.array([-SPINE_HOLD_FORCE_N]),
                upper=np.array([SPINE_HOLD_FORCE_N]),
                dofs_idx_local=spine_dofs,
            )
            self.robot.set_dofs_kp(np.array([SPINE_HOLD_KP]), spine_dofs)
            self.robot.set_dofs_kv(np.array([SPINE_HOLD_KV]), spine_dofs)
            self.set_spine_position(0.0)

            for _ in range(100):
                self.scene.step()
            self._read_and_publish_state()
        except Exception:
            Path(self._resolved_urdf).unlink(missing_ok=True)
            self._resolved_urdf = None
            raise

    def _bind_entity(self) -> None:
        """Resolve joint/link handles once the entity exists."""
        self.arm_dofs_idx = {
            role: [self.robot.get_joint(name).dof_idx_local for name in ARM_JOINT_NAMES[role]]
            for role in ARM_ROLES
        }
        self.arm_links = {role: self.robot.get_link(ARM_EE_LINKS[role]) for role in ARM_ROLES}
        self.spine_dof_idx = self.robot.get_joint(SPINE_JOINT_NAME).dof_idx_local
        self.swerve = SwerveBase(self.robot, base_height=self.base_height)
        self.swerve.bind()
        self.swerve.reset_pose()

        # Precomputed slices for the batched per-step reads/writes: the indexed
        # Genesis getters re-validate their dof-index argument and launch one
        # kernel per call, which at ~400 Hz dominates the loop (measured 88% of
        # loop time before batching). See _read_and_publish_state.
        self._arm_dofs_np = {
            role: np.asarray(self.arm_dofs_idx[role], dtype=np.intp) for role in ARM_ROLES
        }
        self._arm_link_idx = {role: self.arm_links[role].idx_local for role in ARM_ROLES}

    def view(self, role: str) -> "SceneView":
        """Return the simulator adapter for one bridge."""
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}; expected one of {ROLES}")
        return SceneView(self, role)

    # -- command interface -------------------------------------------------

    def set_arm_control_mode(self, role: str, mode: ControlMode) -> None:
        """Set one arm's control mode (lock-free atomic reference swap)."""
        if not isinstance(mode, ControlMode):
            raise ValueError(f"Mode must be a ControlMode enum, got {type(mode)}")
        logger.info("Arm %s control mode -> %s", role, mode.value)
        self.arm_control_modes[role] = mode

    # The three command setters below run on the FCI bridge threads and are
    # lock-free by atomic reference swap: each binds a *fresh* array, which the
    # physics thread then owns for as long as it reads it. ``np.array`` (a copy)
    # rather than ``np.asarray`` so a caller that recycles one decode buffer
    # cannot mutate a command the physics thread is mid-way through publishing.

    def update_arm_torques(self, role: str, torques) -> None:
        """Publish one arm's commanded torques."""
        self.arm_torques[role] = np.array(torques, dtype=float)

    def update_arm_joint_positions(self, role: str, positions) -> None:
        """Publish one arm's commanded joint positions."""
        self.arm_joint_positions[role] = np.array(positions, dtype=float)

    def update_arm_joint_velocities(self, role: str, velocities) -> None:
        """Publish one arm's commanded joint velocities."""
        self.arm_joint_velocities[role] = np.array(velocities, dtype=float)

    def update_base_twist(self, twist: Sequence[float]) -> None:
        """Publish the base body-frame twist ``[vx, vy, vz, wx, wy, wz]``."""
        self.swerve.set_twist(twist)

    def set_spine_position(self, position_m: float) -> None:
        """Place the lift carriage, clamped to the URDF's travel limits.

        Kinematic, like the base pose: the real spine is a separate REST device
        with its own closed-loop controller, so the sim only mirrors where it
        says it is. Everything above the joint -- mount, head, both arms --
        follows because they are children of it in the one entity.

        ``zero_velocity=False`` is load-bearing: Genesis' ``set_dofs_position``
        zeroes the velocity of EVERY DOF of the entity, not just the ones named
        in ``dofs_idx_local``. Called once per physics step, the default would
        wipe both arms' joint velocities each step -- an effectively infinite
        damper that pins them while the teleported lift still moves. Only the
        spine's own DOF is zeroed, which is what the teleport invalidates.

        Write-on-change, and *only* on change: the carriage sits still almost
        all the time, and three redundant kernel launches per step (each of
        which drags a scene-wide forward-kinematics pass behind it) are pure
        overhead at 400 Hz. What holds the joint between changes is the PD
        target set alongside the teleport -- Genesis keeps control targets until
        they are overwritten, so the hold costs nothing per step. Re-teleporting
        periodically instead would leave the joint free-running in between,
        which shows up as a sawtooth on both flanges' O_T_EE (see
        SPINE_HOLD_KP).
        """
        lower, upper = SPINE_LIMITS_M
        clamped = min(max(float(position_m), lower), upper)
        if clamped == self._spine_last_written:
            return
        self._spine_last_written = clamped
        spine_dofs = [self.spine_dof_idx]
        target = np.array([clamped])
        self.robot.set_dofs_position(target, spine_dofs, zero_velocity=False)
        self.robot.set_dofs_velocity(np.zeros(1), spine_dofs)
        self.robot.control_dofs_position(target, spine_dofs)

    def get_role_state(self, role: str) -> Dict[str, np.ndarray]:
        """Latest snapshot for one role (lock-free read)."""
        return self._state_snapshots[role]

    # -- physics loop ------------------------------------------------------

    def _apply_control(self) -> None:
        """Apply one control step for the base, the lift and both arms."""
        self.swerve.apply(self.dt)

        if self.spine_model is not None:
            self.set_spine_position(self.spine_model.position_m())

        # Genesis control targets persist across steps, so each arm's target is
        # rewritten only when its VALUE changed since the last write. Comparing
        # values rather than array identity matters both ways: a client holding
        # a constant target (franka_ros2's joint-impedance controller streaming
        # the same q_d at 1 kHz) keeps hitting the skip, and a caller that
        # recycles one buffer can never be mistaken for "nothing new". Comparing
        # 7 floats costs a few microseconds against tens per kernel launch.
        for role in ARM_ROLES:
            mode = self.arm_control_modes[role]
            if mode == ControlMode.POSITION:
                target = self.arm_joint_positions[role]
            elif mode == ControlMode.VELOCITY:
                target = self.arm_joint_velocities[role]
            elif mode == ControlMode.TORQUE:
                target = self.arm_torques[role]
            else:
                continue
            last_mode, last_target = self._last_arm_cmd[role]
            if last_mode == mode and np.array_equal(last_target, target):
                continue
            dofs = self._arm_dofs_np[role]
            if mode == ControlMode.POSITION:
                self.robot.control_dofs_position(target, dofs)
            elif mode == ControlMode.VELOCITY:
                self.robot.control_dofs_velocity(target, dofs)
            else:
                self.robot.control_dofs_force(target, dofs)
            self._last_arm_cmd[role] = (mode, np.array(target, dtype=float))

    def _filtered_acceleration(self, role: str, dq: np.ndarray) -> np.ndarray:
        """Low-passed numerical joint acceleration for one role."""
        raw = (dq - self._prev_dq[role]) / self.dt
        self._ddq_filtered[role] = (
            self._alpha_acc * self._ddq_filtered[role] + (1 - self._alpha_acc) * raw
        )
        self._prev_dq[role] = dq.copy()
        return self._ddq_filtered[role]

    def _read_and_publish_state(self) -> None:
        """Read the entity once and publish one snapshot per role.

        Whole-entity reads (position, velocity, link positions, link
        quaternions) instead of per-role indexed getters: the indexed reads
        cost one kernel launch plus dof-index validation EACH, which at
        ~400 Hz measured as 54% of the whole loop. Slicing the full arrays
        in numpy is effectively free by comparison.

        The two link-pose reads are decimated on top of that
        (``LINK_POSE_READ_EVERY``); the joint-state pair is not, because the
        UDP RobotState reports q/dq at 1 kHz.
        """
        q_all = self.robot.get_dofs_position().cpu().numpy()
        dq_all = self.robot.get_dofs_velocity().cpu().numpy()

        self._link_read_countdown -= 1
        if self._link_read_countdown <= 0:
            self._link_read_countdown = LINK_POSE_READ_EVERY
            links_pos = self.robot.get_links_pos().cpu().numpy()
            links_quat = self.robot.get_links_quat().cpu().numpy()
            self._arm_ee_pose = {
                role: pose_to_column_major(
                    links_pos[self._arm_link_idx[role]], links_quat[self._arm_link_idx[role]]
                )
                for role in ARM_ROLES
            }

        snapshots = {}

        for role in ARM_ROLES:
            idx = self._arm_dofs_np[role]
            q = q_all[idx]
            dq = dq_all[idx]
            ddq = self._filtered_acceleration(role, dq)
            snapshots[role] = {
                "q": q,
                "dq": dq,
                "ddq": ddq,
                "q_d": self.arm_joint_positions[role],
                "dq_d": dq,
                "ddq_d": ddq,
                "tau_J": self.arm_torques[role],
                "O_T_EE": self._arm_ee_pose[role],
            }

        wheel_q, wheel_dq = self.swerve.wheel_state_from(q_all, dq_all)
        q = np.zeros(7)
        dq = np.zeros(7)
        q[:4] = wheel_q
        dq[:4] = wheel_dq
        ddq = self._filtered_acceleration(ROLE_BASE, dq)
        snapshots[ROLE_BASE] = {
            "q": q,
            "dq": dq,
            "ddq": ddq,
            "q_d": q,
            "dq_d": dq,
            "ddq_d": ddq,
            "tau_J": np.zeros(7),
            "O_T_EE": self.swerve.base_pose_matrix(),
        }

        self._state_snapshots = snapshots

    def run_simulation(self) -> None:
        """Physics loop, paced to wall-clock realtime (see FrankaGenesisSim)."""
        logger.info("Starting mobile-duo Genesis simulation loop")

        next_step = time.perf_counter()
        next_render = next_step
        render_period = 1.0 / RENDER_FPS
        rtf_monitor = RealtimeFactorMonitor(logger, next_step)

        while self.running:
            self._read_and_publish_state()
            self._apply_control()

            now = time.perf_counter()
            do_render = self.enable_vis and now >= next_render
            self.scene.step(update_visualizer=do_render)
            if do_render:
                next_render += render_period
                if next_render < now:
                    next_render = now + render_period

            next_step += self.dt
            slack = next_step - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            elif slack < -self.dt:
                next_step = time.perf_counter()
                next_render = max(next_render, next_step)

            # Wall-clock RTF measured after pacing: when physics keeps up,
            # the sleep above pads each iteration back to ~dt (RTF ~= 1); an
            # overloaded step skips the sleep and the iteration runs long
            # (RTF < 1). See RealtimeFactorMonitor.
            rtf_monitor.update(time.perf_counter(), self.dt)

        if self.enable_vis:
            self.scene.viewer.stop()

    def start(self) -> None:
        """Build (if needed) and run the physics loop in the calling thread."""
        if not self.scene:
            self.initialize_simulation()

        self.running = True
        if self.enable_vis and platform.system() == "Darwin" and platform.machine() == "arm64":
            gs.tools.run_in_another_thread(fn=self.run_simulation, args=())
            self.scene.viewer.start()
        else:
            self.run_simulation()

    def stop(self) -> None:
        """Stop the loop and remove the generated URDF copy."""
        self.running = False
        if self.enable_vis and self.scene is not None:
            self.scene.viewer.stop()
        if self._resolved_urdf is not None:
            Path(self._resolved_urdf).unlink(missing_ok=True)
            self._resolved_urdf = None
