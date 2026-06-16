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


class ControlMode(Enum):
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    NONE = "none"


class FrankaGenesisSim:
    def __init__(self, enable_vis=False):
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
        self.dt = 0.01  # Simulation timestep
        self.sim_thread = None
        self.hand_link = None  # cached end-effector link handle (set on build)

        # Numerical-differentiation state for joint acceleration (physics thread).
        self.prev_dq_full = np.zeros(9)
        self.ddq_filtered = np.zeros(9)
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

        # Get the Genesis assets path instead of our own
        import genesis

        genesis_path = Path(genesis.__file__).parent
        self.xml_path = genesis_path / "assets/xml/franka_emika_panda/panda.xml"

        # Keep URDF path for future use if needed (for Pinocchio)
        # This is currently unused, but kept for reference
        current_dir = Path(__file__).parent
        assets_dir = current_dir.parent / "assets"
        self.urdf_path = assets_dir / "urdf/panda_bullet/panda.urdf"

        logger.info(f"Using Genesis XML path: {self.xml_path}")

    def load_panda_model(self):
        pass
        # TODO: load pinocchio model
        # model = pin.buildModelFromUrdf(str(self.urdf_path))
        # data = model.createData()
        # return model, data

    def initialize_simulation(self):
        # Initialize Genesis with CPU backend
        gs.init(backend=gs.cpu, logging_level=None)

        # Create scene
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(1280, 800),
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

        # Build scene
        self.scene.build()

        # Load Pinocchio model
        # TODO: load pinocchio model
        # self.model, self.data = self.load_panda_model()

        # Joint names and indices
        self.jnt_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
            "finger_joint1",
            "finger_joint2",
        ]
        self.dofs_idx = [self.franka.get_joint(name).dof_idx_local for name in self.jnt_names]

        # Set force range for safety
        self.franka.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
            dofs_idx_local=self.dofs_idx,
        )

        # Cache the end-effector link handle so the hot path never resolves it
        # by name (get_link("hand") costs ~6.5 us per call otherwise).
        self.hand_link = self.franka.get_link("hand")

        # Initialize to default position
        initial_q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
        # Set the initial position as the target position for the controller
        self.latest_joint_positions = initial_q.copy()

        for _ in range(100):
            self.franka.set_dofs_position(np.concatenate([initial_q, [0.04, 0.04]]), self.dofs_idx)
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
        """Main simulation loop (physics thread): read state, apply control, step."""
        logger.info("Starting Genesis simulation loop")

        while self.running:
            # Read Genesis once and publish the snapshot for the network thread.
            self._read_and_publish_state()

            # Apply control for the current mode (atomic reads, no lock).
            current_mode = self.control_mode
            if current_mode == ControlMode.POSITION:
                q_cmd = np.concatenate([self.latest_joint_positions, [0.04, 0.04]])
                self.franka.control_dofs_position(q_cmd, self.dofs_idx)
            elif current_mode == ControlMode.VELOCITY:
                dq_cmd = np.concatenate([self.latest_joint_velocities, [0.0, 0.0]])
                self.franka.control_dofs_velocity(dq_cmd, self.dofs_idx)
            elif current_mode == ControlMode.TORQUE:
                tau_cmd = np.concatenate([self.latest_torques, [0.0, 0.0]])
                self.franka.control_dofs_force(tau_cmd, self.dofs_idx)

            # Step simulation.
            self.scene.step()

            # Yield a slice so the loop does not busy-spin a core.
            time.sleep(0.001)

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
