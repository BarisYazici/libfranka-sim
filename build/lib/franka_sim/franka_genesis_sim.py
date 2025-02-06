import argparse
import numpy as np
import genesis as gs
import pinocchio as pin
from pathlib import Path
import threading
import time
import logging

logger = logging.getLogger(__name__)

class FrankaGenesisSim:
    def __init__(self, enable_vis=False):
        self.enable_vis = enable_vis
        self.scene = None
        self.franka = None
        self.model = None
        self.data = None
        self.running = False
        self.latest_torques = np.zeros(7)
        self.torque_lock = threading.Lock()
        self.dt = 0.01  # Simulation timestep
        self.sim_thread = None
        
        # Set up URDF path
        current_dir = Path(__file__).parent
        root_dir = current_dir.parent.parent  # Go up two levels to reach libfranka root
        self.urdf_path = root_dir / "assets/urdf/panda_bullet/panda.urdf"
        self.xml_path = root_dir / "assets/xml/franka_emika_panda/panda.xml"

        # self.initialize_simulation()

    def load_panda_model(self):
        model = pin.buildModelFromUrdf(str(self.urdf_path))
        data = model.createData()
        return model, data

    def initialize_simulation(self):
        # Initialize Genesis
        gs.init(backend=gs.gpu, logging_level=None)

        # Create scene
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0, -3.5, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=30,
                res=(960, 640),
                max_FPS=60,
            ),
            sim_options=gs.options.SimOptions(
                dt=self.dt,
            ),
            show_viewer=self.enable_vis,
            show_FPS=False
        )

        # Add entities
        self.scene.add_entity(gs.morphs.Plane())
        # self.franka = self.scene.add_entity(
        #     gs.morphs.URDF(
        #         file=str(self.urdf_path),
        #         merge_fixed_links=False
        #     ),
        # )
        self.franka = self.scene.add_entity(
            gs.morphs.MJCF(
                file=str(self.xml_path),
            ),
        )

        # Build scene
        self.scene.build()

        # Load Pinocchio model
        self.model, self.data = self.load_panda_model()

        # Joint names and indices
        # self.jnt_names = [
        #     'panda_joint1', 'panda_joint2', 'panda_joint3',
        #     'panda_joint4', 'panda_joint5', 'panda_joint6',
        #     'panda_joint7', 'panda_finger_joint1', 'panda_finger_joint2',
        # ]
        self.jnt_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'joint7',
            'finger_joint1',
            'finger_joint2',
        ]
        self.dofs_idx = [self.franka.get_joint(name).dof_idx_local for name in self.jnt_names]

        # Set force range for safety
        self.franka.set_dofs_force_range(
            lower=np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
            upper=np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
            dofs_idx_local=self.dofs_idx,
        )

        # Initialize to default position
        initial_q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])
        for _ in range(100):
            self.franka.set_dofs_position(
                np.concatenate([initial_q, [0.04, 0.04]]),
                self.dofs_idx
            )
            self.scene.step()

    def update_torques(self, torques):
        """Update the latest torques to be applied in simulation"""
        with self.torque_lock:
            self.latest_torques = np.array(torques)

    def run_simulation(self):
        """Main simulation loop"""
        logger.info("Starting Genesis simulation loop")
        
        # For numerical differentiation
        prev_dq_full = np.zeros(9)
        ddq_filtered = np.zeros(9)
        alpha_acc = 0.95

        while self.running:
            # Get current joint states
            q_full = self.franka.get_dofs_position(self.dofs_idx).cpu().numpy()
            dq_full = self.franka.get_dofs_velocity(self.dofs_idx).cpu().numpy()

            # Calculate acceleration
            ddq_raw = (dq_full - prev_dq_full) / self.dt
            ddq_filtered = alpha_acc * ddq_filtered + (1 - alpha_acc) * ddq_raw
            prev_dq_full = dq_full.copy()

            # Get the latest torques to apply
            with self.torque_lock:
                tau_d = self.latest_torques.copy()

            # Apply control forces (including zero forces for fingers)
            self.franka.control_dofs_force(
                np.concatenate([tau_d, [0.0, 0.0]]),
                self.dofs_idx
            )

            # Step simulation
            self.scene.step()

            # Optional: Add small sleep to prevent too high CPU usage
            time.sleep(0.001)

        if self.enable_vis:
            self.scene.viewer.stop()

    def start(self):
        """Start the simulation in a separate thread"""
        self.running = True
        
        # Run simulation in a separate thread
        self.sim_thread = threading.Thread(target=self.run_simulation)
        self.sim_thread.daemon = True
        self.sim_thread.start()
        
        # If visualization is enabled, run it in the main thread
        if self.enable_vis:
            self.scene.viewer.start()
        
    def stop(self):
        """Stop the simulation"""
        self.running = False
        if self.enable_vis:
            self.scene.viewer.stop()
        if self.sim_thread:
            self.sim_thread.join(timeout=1.0)  # Wait for simulation thread to finish

    def get_robot_state(self):
        """Get current robot state for network transmission"""
        q_full = self.franka.get_dofs_position(self.dofs_idx).cpu().numpy()
        dq_full = self.franka.get_dofs_velocity(self.dofs_idx).cpu().numpy()
        
        # Return only the first 7 joints (excluding fingers)
        return {
            'q': q_full[:7],
            'dq': dq_full[:7],
            'tau_J': self.latest_torques  # Current commanded torques
        }

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