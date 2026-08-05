import argparse
import logging
import os
import platform
import sys
import time
from enum import Enum
from pathlib import Path

import genesis as gs
import numpy as np

# import pinocchio as pin

logger = logging.getLogger(__name__)


def default_fr3_mjcf():
    """Resolve the FR3 MJCF physics model (MuJoCo Menagerie ``franka_fr3``).

    The real robot is an FR3 (not a Panda); this model carries the FR3
    kinematics/inertia and built-in joint damping/armature. Resolution order:
    the ``$FR3_MJCF`` override, otherwise ``robot_descriptions``, which downloads
    and caches the model on first use -- so a ``pip install`` of this package
    works out of the box (no vendored ~57 MB of meshes, just a small dependency).
    """
    override = os.environ.get("FR3_MJCF")
    if override:
        return Path(override)
    try:
        from robot_descriptions import fr3_mj_description

        return Path(fr3_mj_description.MJCF_PATH)
    except Exception as exc:  # offline / proxy / fetch failure on first use
        raise RuntimeError(
            f"Could not obtain the FR3 model via robot_descriptions ({type(exc).__name__}: "
            f"{exc}). It is downloaded and cached on first use, so the first run needs "
            "network access; set $FR3_MJCF to a local MJCF path to run fully offline."
        ) from exc


# Default joint viscous damping (Nm*s/rad), applied to all 7 joints unless
# overridden via $FR3_JOINT_DAMPING. Calibrated against a logged real-robot
# joint_impedance run so the sim's joint excursions match the real FR3 to within
# ~5% (the bare MJCF value of 0.21 is far too small and the sim over-travels).
DEFAULT_FR3_DAMPING = 5.0


class ControlMode(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    STEERING_DRIVE = "steering_drive"  # mobile base: steering=position, drive=velocity
    NONE = "none"


class FrankaGenesisSim:
    def __init__(self, enable_vis=False, model_path=None, enable_hand=False, extra_morphs=None):
        self.enable_vis = enable_vis
        self.scene = None
        self.franka = None
        self.model = None
        self.data = None
        self.running = False
        # Latest commands from the network threads, published lock-free: a writer
        # always assigns a brand-new array (an atomic reference swap under the
        # GIL) and the physics thread is the sole reader, so no mutex is needed
        # (see the update_* methods). The array is never mutated in place.
        self.latest_torques = np.zeros(7)
        self.latest_joint_positions = np.zeros(7)
        self.latest_joint_velocities = np.zeros(7)
        self.control_mode = ControlMode.POSITION  # Default to position control
        # Physics timestep. run_simulation paces the loop to wall-clock realtime
        # so a controller tuned for the real 1 kHz FCI sees the same wall-time
        # dynamics (not physics that free-runs several times faster). dt is
        # bounded below by how fast one step + state read runs on this host: at
        # 1 ms the loop only sustained ~0.6x realtime (slow-motion), so ~2.5 ms
        # is the finest step that still holds true 1x here. Lower it once the
        # per-step Genesis read is cheaper (throttle the snapshot below the step
        # rate) or calibrate it against logged real-robot trajectories.
        self.dt = 0.0025  # ~400 Hz physics, paced to realtime (see run_simulation)
        self.sim_thread = None
        self.hand_link = None  # cached end-effector link handle (set on build)

        # Optional physics gripper (9-DOF: 7 arm + 2 fingers). When enabled, the
        # combined FR3+hand model is generated at init and the fingers are driven
        # from latest_finger_positions (lock-free, like the arm command arrays).
        self.enable_hand = enable_hand
        # Optional scene entities added before scene.build() (e.g. graspable objects).
        # Default empty → zero behavior change for existing callers.
        self.extra_morphs = extra_morphs or []
        self.finger_dofs_idx = None
        self.max_finger_width = 0.08  # total opening; 0.04 per finger
        self._hand_model_tmp = None  # temp MJCF path to clean up in stop()
        # Per-finger targets (m), each 0..0.04. Start fully open.
        self.latest_finger_positions = np.array([0.04, 0.04])
        self._finger_snapshot = {"q": np.zeros(2), "dq": np.zeros(2)}

        # Numerical-differentiation state for joint acceleration (physics thread).
        # 7 DOF: the FR3 model is hand-less (no finger joints).
        self.prev_dq_full = np.zeros(7)
        self.ddq_filtered = np.zeros(7)
        self._alpha_acc = 0.95  # acceleration low-pass factor

        # Latest robot-state snapshot published by the physics thread and read by
        # the UDP broadcast thread. The dict reference is swapped in atomically
        # each step, so get_robot_state() needs no lock and does no Genesis
        # tensor reads on the network hot path.
        self._state_snapshot = {
            "q": np.zeros(7),
            "dq": np.zeros(7),
            "ddq": np.zeros(7),
            "q_d": np.zeros(7),
            "dq_d": np.zeros(7),
            "ddq_d": np.zeros(7),
            "tau_J": np.zeros(7),
            "O_T_EE": np.eye(4).T.flatten(),
        }

        # FR3 physics model (MJCF). The real robot is an FR3; this model matches
        # its kinematics/inertia and brings the joint damping/friction the old
        # frictionless Panda model lacked.
        if model_path is not None:
            self.xml_path = Path(model_path)
        elif enable_hand:
            from franka_sim.assets import build_fr3_with_hand_mjcf

            self._hand_model_tmp = build_fr3_with_hand_mjcf()
            self.xml_path = self._hand_model_tmp
        else:
            self.xml_path = default_fr3_mjcf()
        logger.info(f"Using Genesis FR3 MJCF: {self.xml_path}")

    def load_panda_model(self):
        pass
        # TODO: load pinocchio model
        # model = pin.buildModelFromUrdf(str(self.urdf_path))
        # data = model.createData()
        # return model, data

    def initialize_simulation(self):
        # Initialize Genesis with CPU backend (idempotent: ignore "already initialized").
        if not getattr(gs, "_initialized", False):
            try:
                gs.init(backend=gs.cpu, logging_level=None)
            except Exception as exc:
                if "already initialized" not in str(exc).lower():
                    raise

        # Create scene
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                # Lower render resolution so drawing each viewer frame is cheap
                # enough to leave the single (Linux) main thread time to keep the
                # physics loop at realtime (rendering and physics share it).
                res=(640, 480),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=self.dt,
            ),
            show_viewer=self.enable_vis,
            show_FPS=False,
        )

        # Add entities
        self.scene.add_entity(gs.morphs.Plane())
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(self.xml_path),
            ),
            material=gs.materials.Rigid(gravity_compensation=1.0),
        )

        # Optional extra entities (e.g. graspable boxes) injected before build.
        for morph in self.extra_morphs:
            self.scene.add_entity(morph)

        # Build scene
        self.scene.build()

        # Load Pinocchio model
        # TODO: load pinocchio model
        # self.model, self.data = self.load_panda_model()

        # Joint names and indices (FR3, 7-DOF, hand-less -> no finger joints).
        self.jnt_names = [f"fr3_joint{i}" for i in range(1, 8)]
        self.dofs_idx = [self.franka.get_joint(name).dof_idx_local for name in self.jnt_names]

        # Set force range for safety (FR3 actuator limits: J1-4 +/-87, J5-7 +/-12).
        self.franka.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12]),
            dofs_idx_local=self.dofs_idx,
        )

        # Joint viscous damping, applied by default so the sim matches the real
        # FR3 out of the box (no env var needed). Genesis only supports viscous
        # damping (no Coulomb frictionloss), and the MJCF's 0.21 is far too small
        # next to the impedance controller's own damping, so the frictionless
        # model over-travels vs the real arm; DEFAULT_FR3_DAMPING was fit to a
        # logged real-robot joint_impedance run (~5% match). Override per-joint
        # via $FR3_JOINT_DAMPING (a scalar or 7 comma-separated values).
        damping_env = os.environ.get("FR3_JOINT_DAMPING")
        if damping_env:
            vals = [float(x) for x in damping_env.split(",")]
            if len(vals) not in (1, 7):
                raise ValueError(
                    f"FR3_JOINT_DAMPING must be 1 (scalar) or 7 comma-separated "
                    f"values, got {len(vals)}"
                )
            damping = np.array(vals * 7 if len(vals) == 1 else vals, dtype=float)
        else:
            damping = np.full(7, DEFAULT_FR3_DAMPING)
        self.franka.set_dofs_damping(damping, self.dofs_idx)
        logger.info(f"Joint damping: {damping}")

        # Cache the end-effector link handle so the hot path never resolves it by
        # name. FR3 is hand-less; link7 is the flange (attachment frame).
        self.hand_link = self.franka.get_link("fr3_link7")

        if self.enable_hand:
            self.finger_dofs_idx = [
                self.franka.get_joint(n).dof_idx_local
                for n in ("fr3_finger_joint1", "fr3_finger_joint2")
            ]
            # Finger actuator force range (Franka Hand ~70 N grip; cap for safety).
            self.franka.set_dofs_force_range(
                lower=np.array([-100, -100]),
                upper=np.array([100, 100]),
                dofs_idx_local=self.finger_dofs_idx,
            )
            self.franka.set_dofs_position(self.latest_finger_positions, self.finger_dofs_idx)

        # Initialize to default position
        initial_q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
        # Set the initial position as the target position for the controller
        self.latest_joint_positions = initial_q.copy()

        for _ in range(100):
            self.franka.set_dofs_position(initial_q, self.dofs_idx)
            self.scene.step()

        # Seed the published snapshot so get_robot_state() is valid before the
        # physics loop starts.
        self._read_and_publish_state()

    def set_control_mode(self, mode: ControlMode):
        """Set the control mode for the robot (lock-free atomic reference swap)."""
        if not isinstance(mode, ControlMode):
            raise ValueError(f"Mode must be a ControlMode enum, got {type(mode)}")
        logger.info(f"Switching control mode to: {mode.value}")
        self.control_mode = mode

    def update_torques(self, torques):
        """Publish the latest commanded torques for the physics thread.

        Lock-free: a fresh array is bound in a single bytecode step (atomic under
        the GIL) and the physics thread is the only reader, so no mutex is needed.
        The array must never be mutated in place after assignment.
        """
        self.latest_torques = np.array(torques)

    def update_joint_positions(self, positions):
        """Publish the latest commanded joint positions (lock-free; see update_torques)."""
        self.latest_joint_positions = np.array(positions)

    def update_joint_velocities(self, velocities):
        """Publish the latest commanded joint velocities (lock-free; see update_torques)."""
        self.latest_joint_velocities = np.array(velocities)

    def update_finger_positions(self, positions):
        """Publish per-finger position targets (length-2, each 0..0.04 m).

        Lock-free (see update_torques): a fresh array is bound atomically and the
        physics thread is the only reader.
        """
        self.latest_finger_positions = np.array(positions)

    def get_finger_state(self):
        """Return the latest finger snapshot {'q': (2,), 'dq': (2,)} (atomic read)."""
        return self._finger_snapshot

    def _pose_to_column_major(self, ee_pos, ee_quat):
        """Build a column-major 4x4 O_T_EE from an EE position and [x, y, z, w] quat."""
        x, y, z, w = ee_quat
        R = np.array(
            [
                [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
                [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
            ]
        )
        O_T_EE = np.eye(4)
        O_T_EE[:3, :3] = R
        O_T_EE[:3, 3] = ee_pos
        return O_T_EE.T.flatten()  # column-major 16-element array

    def _read_and_publish_state(self):
        """Read joint + EE state from Genesis once and publish a state snapshot.

        Runs only on the physics thread (single producer). The whole snapshot is
        swapped in as a new dict reference that consumers read atomically under
        the GIL, so get_robot_state() does no Genesis tensor reads and needs no
        lock. Doing the (~0.4 ms) Genesis reads here, once per step, also removes
        the previous double-read where the network thread fetched the same
        tensors independently.
        """
        q_full = self.franka.get_dofs_position(self.dofs_idx).cpu().numpy()
        dq_full = self.franka.get_dofs_velocity(self.dofs_idx).cpu().numpy()

        # Filtered numerical acceleration.
        ddq_raw = (dq_full - self.prev_dq_full) / self.dt
        self.ddq_filtered = self._alpha_acc * self.ddq_filtered + (1 - self._alpha_acc) * ddq_raw
        self.prev_dq_full = dq_full.copy()

        # End-effector pose via the cached link handle.
        ee_pos = self.hand_link.get_pos().cpu().numpy()
        ee_quat = self.hand_link.get_quat().cpu().numpy()  # [x, y, z, w]
        O_T_EE = self._pose_to_column_major(ee_pos, ee_quat)

        if self.enable_hand and self.finger_dofs_idx is not None:
            fq = self.franka.get_dofs_position(self.finger_dofs_idx).cpu().numpy()
            fdq = self.franka.get_dofs_velocity(self.finger_dofs_idx).cpu().numpy()
            self._finger_snapshot = {"q": fq, "dq": fdq}

        # q_d / tau_J mirror the latest network commands (atomic reads).
        self._state_snapshot = {
            "q": q_full[:7],
            "dq": dq_full[:7],
            "ddq": self.ddq_filtered[:7],
            "q_d": self.latest_joint_positions,
            "dq_d": dq_full[:7],
            "ddq_d": self.ddq_filtered[:7],
            "tau_J": self.latest_torques,
            "O_T_EE": O_T_EE,
        }

    def run_simulation(self):
        """Main simulation loop (physics thread): read state, apply control, step.

        Paced to wall-clock realtime (dt of sim-time per dt of wall-time) so the
        physics advances in lockstep with the 1 kHz control loop rather than
        free-running several times faster than realtime.
        """
        logger.info("Starting Genesis simulation loop")

        next_step = time.perf_counter()
        next_render = next_step
        # Viewer refresh rate, decoupled from the physics rate. Each pyrender
        # frame costs ~15-20 ms on the shared (Linux) main thread regardless of
        # resolution, so 60 FPS alone would exceed a full core and starve the
        # physics loop; 30 FPS leaves enough budget for realtime physics.
        render_period = 1.0 / 30.0
        while self.running:
            # Read Genesis once and publish the snapshot for the network thread.
            self._read_and_publish_state()

            # Apply control for the current mode (atomic reads, no lock).
            current_mode = self.control_mode
            if current_mode == ControlMode.POSITION:
                self.franka.control_dofs_position(self.latest_joint_positions, self.dofs_idx)
            elif current_mode == ControlMode.VELOCITY:
                self.franka.control_dofs_velocity(self.latest_joint_velocities, self.dofs_idx)
            elif current_mode == ControlMode.TORQUE:
                self.franka.control_dofs_force(self.latest_torques, self.dofs_idx)

            if self.enable_hand and self.finger_dofs_idx is not None:
                self.franka.control_dofs_position(
                    self.latest_finger_positions, self.finger_dofs_idx
                )

            # Step physics. scene.step(update_visualizer=True) refreshes the
            # viewer AND sleeps to cap it at max_FPS, which would otherwise
            # throttle the whole realtime physics loop down to ~60 Hz
            # (slow-motion / under-driven arm). So refresh the viewer only on a
            # ~60 FPS schedule and step physics without it the rest of the time.
            now = time.perf_counter()
            do_render = self.enable_vis and now >= next_render
            self.scene.step(update_visualizer=do_render)
            if do_render:
                next_render += render_period
                if next_render < now:
                    next_render = now + render_period

            # Pace to realtime: advance one dt of sim-time per dt of wall-time.
            # If a step takes longer than dt, run flat out (no catch-up burst).
            next_step += self.dt
            slack = next_step - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            elif slack < -self.dt:
                # Fell a full step behind; resync both schedules so we don't burst
                # a run of catch-up steps or renders.
                next_step = time.perf_counter()
                next_render = max(next_render, next_step)

        if self.enable_vis:
            self.scene.viewer.stop()

    def start(self):
        """Start the simulation"""
        if not self.scene:
            self.initialize_simulation()

        self.running = True

        if self.enable_vis:
            # Run simulation in a separate thread when visualization is enabled
            # if macos, run in a separate thread
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                gs.tools.run_in_another_thread(fn=self.run_simulation, args=())
            else:
                self.run_simulation()
            # Run viewer in main thread
            self.scene.viewer.start()
        else:
            # Without visualization, just run simulation in current thread
            self.run_simulation()

    def stop(self):
        """Stop the simulation"""
        self.running = False
        if self.enable_vis:
            self.scene.viewer.stop()
        if self.sim_thread:
            self.sim_thread.join(timeout=1.0)  # Wait for simulation thread to finish
        if self._hand_model_tmp is not None:
            try:
                Path(self._hand_model_tmp).unlink(missing_ok=True)
            except OSError:
                pass
            self._hand_model_tmp = None

    def get_robot_state(self):
        """Return the latest state snapshot published by the physics thread.

        Lock-free: reads the current snapshot reference (atomic under the GIL).
        No Genesis tensor reads happen here, so the UDP broadcast thread is not
        bottlenecked by the ~0.5 ms cost of fetching state from the simulator.
        The snapshot holds only the first 7 joints (fingers excluded).
        """
        return self._state_snapshot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    sim = FrankaGenesisSim(enable_vis=args.vis)
    sim.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
