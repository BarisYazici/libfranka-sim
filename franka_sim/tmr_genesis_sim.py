"""Genesis simulation of the TMR mobile platform (base only).

Ported from the parked ``feat/simulate-tmr`` branch onto the current arm and
gripper code, with the lock-free snapshot pattern and realtime pacing used by
``FrankaGenesisSim``, and with the base commanded by a body-frame twist
(``update_base_twist``) instead of steering angles smuggled through ``dq_c``.

The four wheel joints ``tmrv0_2_joint_0..3`` are reported in the first four
elements of the 7-element ``RobotState`` arrays, which is what the real TMR
master reports; indices 4..6 are zero.
"""

import logging
import platform
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import genesis as gs
import numpy as np

from franka_sim.franka_genesis_sim import ControlMode
from franka_sim.swerve_base import SwerveBase
from franka_sim.urdf_assets import resolve_urdf_meshes

logger = logging.getLogger(__name__)

#: Effort limit of the wheel joints (argo_drive.xacro: ``effort="500"``).
WHEEL_FORCE_LIMIT = 500.0


class TMRGenesisSim:
    """Simulator backend for a TMR-only scene, speaking the FrankaSimServer contract.

    Has no CLI entry point today: it is a programmatic/manual bring-up target
    (e.g. for standalone base testing), not something ``run_server.py`` wires
    up. :class:`~franka_sim.mobile_duo_runner.MobileDuoRunner` (via
    :class:`~franka_sim.mobile_duo_sim.MobileDuoScene`) is the shipped path.
    """

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
        self.tmr = None
        self.running = False
        self.control_mode = ControlMode.STEERING_DRIVE
        self.swerve: Optional[SwerveBase] = None

        self._resolved_urdf: Optional[Path] = None
        self._prev_dq = np.zeros(4)
        self._ddq_filtered = np.zeros(4)
        self._alpha_acc = 0.95

        self._state_snapshot: Dict[str, np.ndarray] = {
            "q": np.zeros(7),
            "dq": np.zeros(7),
            "ddq": np.zeros(7),
            "q_d": np.zeros(7),
            "dq_d": np.zeros(7),
            "ddq_d": np.zeros(7),
            "tau_J": np.zeros(7),
            "O_T_EE": np.eye(4).T.flatten(),
        }

    # -- construction ------------------------------------------------------

    def initialize_simulation(self) -> None:
        """Build the Genesis scene and bind the wheel joints."""
        if not getattr(gs, "_initialized", False):
            try:
                gs.init(backend=gs.cpu, logging_level=None)
            except Exception as exc:  # already initialised by another scene
                if "already initialized" not in str(exc).lower():
                    raise

        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
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
            self.tmr = self.scene.add_entity(
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
            self.tmr.set_dofs_force_range(
                lower=np.full(4, -WHEEL_FORCE_LIMIT),
                upper=np.full(4, WHEEL_FORCE_LIMIT),
                dofs_idx_local=self.swerve.steer_dofs_idx + self.swerve.drive_dofs_idx,
            )

            for _ in range(100):
                self.scene.step()
            self._read_and_publish_state()
        except Exception:
            Path(self._resolved_urdf).unlink(missing_ok=True)
            self._resolved_urdf = None
            raise

    def _bind_entity(self) -> None:
        """Attach the swerve controller to the built entity."""
        self.swerve = SwerveBase(self.tmr, base_height=self.base_height)
        self.swerve.bind()
        self.swerve.reset_pose()

    # -- command interface (FrankaSimServer contract) ----------------------

    def set_control_mode(self, mode: ControlMode) -> None:
        """Set the control mode (lock-free atomic reference swap)."""
        if not isinstance(mode, ControlMode):
            raise ValueError(f"Mode must be a ControlMode enum, got {type(mode)}")
        logger.info("Switching control mode to: %s", mode.value)
        self.control_mode = mode

    def update_base_twist(self, twist: Sequence[float]) -> None:
        """Publish the latest body-frame twist ``[vx, vy, vz, wx, wy, wz]``."""
        self.swerve.set_twist(twist)

    def update_joint_positions(self, positions) -> None:
        """Accepted for contract compatibility; the TMR has no joint interface."""

    def update_joint_velocities(self, velocities) -> None:
        """Accepted for contract compatibility; the TMR has no joint interface."""

    def update_torques(self, torques) -> None:
        """Accepted for contract compatibility; the TMR has no torque interface."""

    def get_robot_state(self) -> Dict[str, np.ndarray]:
        """Latest snapshot published by the physics thread (lock-free read)."""
        return self._state_snapshot

    # -- physics loop ------------------------------------------------------

    def _apply_control(self) -> None:
        """Apply one control step.

        The TMR has exactly one actuation path -- the swerve base -- so every
        control mode routes here. ``control_mode`` is tracked only because the
        protocol server reports it back to the client; a joint-space mode simply
        leaves the last twist in force (zero after a motion finishes).
        """
        self.swerve.apply(self.dt)

    def _read_and_publish_state(self) -> None:
        """Read wheel state once and publish a padded 7-element snapshot."""
        wheel_q, wheel_dq = self.swerve.wheel_state()

        ddq_raw = (wheel_dq - self._prev_dq) / self.dt
        self._ddq_filtered = self._alpha_acc * self._ddq_filtered + (1 - self._alpha_acc) * ddq_raw
        self._prev_dq = wheel_dq.copy()

        q = np.zeros(7)
        dq = np.zeros(7)
        ddq = np.zeros(7)
        q[:4] = wheel_q
        dq[:4] = wheel_dq
        ddq[:4] = self._ddq_filtered

        self._state_snapshot = {
            "q": q,
            "dq": dq,
            "ddq": ddq,
            "q_d": q,
            "dq_d": dq,
            "ddq_d": ddq,
            "tau_J": np.zeros(7),
            "O_T_EE": self.swerve.base_pose_matrix(),
        }

    def run_simulation(self) -> None:
        """Physics loop, paced to wall-clock realtime (see FrankaGenesisSim)."""
        logger.info("Starting TMR Genesis simulation loop")

        next_step = time.perf_counter()
        next_render = next_step
        render_period = 1.0 / 30.0

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
