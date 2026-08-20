"""One Genesis scene serving three FCI bridges: left arm, right arm, TMR base.

The combined ``mobile_fr3_duo`` URDF is loaded as a single entity, so base
motion physically carries both arms. Each protocol server attaches through a
:class:`SceneView`, a thin adapter that implements the simulator contract
``FrankaSimServer`` expects but scopes every read and write to one role.

Only the runner may build the scene or step physics: with three servers on one
scene, three physics loops would corrupt it, so the view lifecycle methods are
no-ops.
"""

import logging
import math
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
from franka_sim.sim_common import (
    FR3_FORCE_LIMITS,
    RealtimeFactorMonitor,
    pose_to_column_major,
)
from franka_sim.swerve_base import SwerveBase
from franka_sim.urdf_assets import resolve_urdf_meshes

logger = logging.getLogger(__name__)

ROLE_LEFT = "left"
ROLE_RIGHT = "right"
ROLE_BASE = "base"
ARM_ROLES = (ROLE_LEFT, ROLE_RIGHT)
ROLES = (ROLE_LEFT, ROLE_RIGHT, ROLE_BASE)

#: Arm joints in the combined URDF, generated with
#: ``robot_types:="['tmrv0_2','fr3v2','fr3v2']"`` and arm prefixes left/right.
ARM_JOINT_NAMES: Dict[str, List[str]] = {
    ROLE_LEFT: [f"left_fr3v2_joint{i}" for i in range(1, 8)],
    ROLE_RIGHT: [f"right_fr3v2_joint{i}" for i in range(1, 8)],
}

#: Flange link per arm (hand:=false, so link7 is the attachment frame).
ARM_EE_LINKS: Dict[str, str] = {
    ROLE_LEFT: "left_fr3v2_link7",
    ROLE_RIGHT: "right_fr3v2_link7",
}

#: Initial arm pose, matching the initial_value parameters in upstream's
#: mobile_fr3_duo.ros2_control.xacro.
ARM_INITIAL_Q = np.array([0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4])

#: Prismatic lift joint carrying the mount, the head and both arms.
SPINE_JOINT_NAME = "franka_spine_vertical_joint"

#: Travel limits of that joint in the combined URDF (metres).
SPINE_LIMITS_M = (0.0, 0.85)


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

    def update_arm_torques(self, role: str, torques) -> None:
        """Publish one arm's commanded torques."""
        self.arm_torques[role] = np.asarray(torques, dtype=float)

    def update_arm_joint_positions(self, role: str, positions) -> None:
        """Publish one arm's commanded joint positions."""
        self.arm_joint_positions[role] = np.asarray(positions, dtype=float)

    def update_arm_joint_velocities(self, role: str, velocities) -> None:
        """Publish one arm's commanded joint velocities."""
        self.arm_joint_velocities[role] = np.asarray(velocities, dtype=float)

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
        """
        lower, upper = SPINE_LIMITS_M
        clamped = min(max(float(position_m), lower), upper)
        self.robot.set_dofs_position(np.array([clamped]), [self.spine_dof_idx], zero_velocity=False)
        self.robot.set_dofs_velocity(np.zeros(1), [self.spine_dof_idx])

    def get_role_state(self, role: str) -> Dict[str, np.ndarray]:
        """Latest snapshot for one role (lock-free read)."""
        return self._state_snapshots[role]

    # -- physics loop ------------------------------------------------------

    def _apply_control(self) -> None:
        """Apply one control step for the base, the lift and both arms."""
        self.swerve.apply(self.dt)

        if self.spine_model is not None:
            self.set_spine_position(self.spine_model.position_m())

        for role in ARM_ROLES:
            dofs = self.arm_dofs_idx[role]
            mode = self.arm_control_modes[role]
            if mode == ControlMode.POSITION:
                self.robot.control_dofs_position(self.arm_joint_positions[role], dofs)
            elif mode == ControlMode.VELOCITY:
                self.robot.control_dofs_velocity(self.arm_joint_velocities[role], dofs)
            elif mode == ControlMode.TORQUE:
                self.robot.control_dofs_force(self.arm_torques[role], dofs)

    def _filtered_acceleration(self, role: str, dq: np.ndarray) -> np.ndarray:
        """Low-passed numerical joint acceleration for one role."""
        raw = (dq - self._prev_dq[role]) / self.dt
        self._ddq_filtered[role] = (
            self._alpha_acc * self._ddq_filtered[role] + (1 - self._alpha_acc) * raw
        )
        self._prev_dq[role] = dq.copy()
        return self._ddq_filtered[role]

    def _read_and_publish_state(self) -> None:
        """Read the entity once and publish one snapshot per role."""
        snapshots = {}

        for role in ARM_ROLES:
            dofs = self.arm_dofs_idx[role]
            q = self.robot.get_dofs_position(dofs).cpu().numpy()
            dq = self.robot.get_dofs_velocity(dofs).cpu().numpy()
            ddq = self._filtered_acceleration(role, dq)
            link = self.arm_links[role]
            snapshots[role] = {
                "q": q,
                "dq": dq,
                "ddq": ddq,
                "q_d": self.arm_joint_positions[role],
                "dq_d": dq,
                "ddq_d": ddq,
                "tau_J": self.arm_torques[role],
                "O_T_EE": pose_to_column_major(
                    link.get_pos().cpu().numpy(), link.get_quat().cpu().numpy()
                ),
            }

        wheel_q, wheel_dq = self.swerve.wheel_state()
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
        render_period = 1.0 / 30.0
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


class SceneView:
    """One role's view of a :class:`MobileDuoScene`, shaped like a simulator.

    Implements the interface ``FrankaSimServer`` calls on ``genesis_sim``.
    ``initialize_simulation``/``start``/``stop`` are no-ops because the runner
    owns the shared scene's lifecycle.
    """

    def __init__(self, scene: MobileDuoScene, role: str):
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}; expected one of {ROLES}")
        self.scene = scene
        self.role = role
        # Latch: update_base_twist can be reached from the ~1 kHz UDP thread, so
        # the misroute warning must fire once, not once per datagram.
        self._twist_misroute_logged = False

    @property
    def enable_vis(self) -> bool:
        """Mirror the shared scene's visualisation flag."""
        return self.scene.enable_vis

    def initialize_simulation(self) -> None:
        """No-op: the runner builds the shared scene exactly once."""

    def start(self) -> None:
        """No-op: the runner owns the single physics loop."""

    def stop(self) -> None:
        """No-op: the runner stops the shared scene."""

    def set_control_mode(self, mode: ControlMode) -> None:
        """Set this arm's control mode; ignored for the base (twist-driven)."""
        if self.role == ROLE_BASE:
            if not isinstance(mode, ControlMode):
                raise ValueError(f"Mode must be a ControlMode enum, got {type(mode)}")
            return
        self.scene.set_arm_control_mode(self.role, mode)

    def update_joint_positions(self, positions) -> None:
        """Publish this arm's joint positions; ignored for the base."""
        if self.role == ROLE_BASE:
            return
        self.scene.update_arm_joint_positions(self.role, positions)

    def update_joint_velocities(self, velocities) -> None:
        """Publish this arm's joint velocities; ignored for the base."""
        if self.role == ROLE_BASE:
            return
        self.scene.update_arm_joint_velocities(self.role, velocities)

    def update_torques(self, torques) -> None:
        """Publish this arm's torques; ignored for the base."""
        if self.role == ROLE_BASE:
            return
        self.scene.update_arm_torques(self.role, torques)

    def update_base_twist(self, twist: Sequence[float]) -> None:
        """Publish the base twist. Dropped (warned once) on an arm view."""
        if self.role != ROLE_BASE:
            if not self._twist_misroute_logged:
                logger.warning("Base twist ignored on arm view %r", self.role)
                self._twist_misroute_logged = True
            return
        self.scene.update_base_twist(twist)

    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """This role's latest state snapshot."""
        return self.scene.get_role_state(self.role)
