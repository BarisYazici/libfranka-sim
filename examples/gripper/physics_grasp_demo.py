"""Client-driven visual demo: a libfranka-style gripper client opening/closing
(and grasping a box with) the simulated Franka Hand, shown in the Genesis viewer.

Run (env active):  python -m examples.gripper.physics_grasp_demo            # viewer
                   python -m examples.gripper.physics_grasp_demo --headless # no window
Every finger motion is commanded over the gripper wire protocol (port 1338),
NOT by poking the sim directly.
"""

import argparse
import threading
import time

from examples.gripper.gripper_wire_client import GripperWireClient
from franka_sim.franka_genesis_sim import FrankaGenesisSim
from franka_sim.gripper.physics import FrankaHandPhysics
from franka_sim.gripper.server import FrankaGripperServer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    import genesis as gs

    gs.init(backend=gs.cpu, logging_level=None)

    # A graspable box placed between the fingers (fingers slide along the y axis).
    # Default config: fingers at x~0.554, z~0.566, y=±0.04 when fully open.
    # Box is 0.035 m wide in y (> 0.02 m grasp target) so fingers stall and
    # report is_grasped=True (finger-position stall detection in FrankaHandPhysics).
    # fixed=True keeps the box in place so it doesn't fall before the fingers close.
    box = gs.morphs.Box(size=(0.06, 0.035, 0.08), pos=(0.554, 0.0, 0.566), fixed=True)
    sim = FrankaGenesisSim(enable_vis=not args.headless, enable_hand=True, extra_morphs=[box])
    sim.initialize_simulation()
    sim.running = True
    sim_thread = threading.Thread(target=sim.run_simulation, daemon=True)
    sim.sim_thread = sim_thread
    sim_thread.start()

    server = FrankaGripperServer(backend=FrankaHandPhysics(sim))
    server_thread = threading.Thread(target=server.run_server, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    client = GripperWireClient()
    try:
        client.connect()
        print("homing...")
        client.homing()
        time.sleep(1.0)
        print("opening...")
        client.move(0.08, 0.1)
        time.sleep(1.0)
        print("grasping (command 0.01, just under the 0.035 box)...")
        status = client.grasp(0.01, 0.01, 0.04, 0.05, 40)
        time.sleep(0.5)
        st = client.read_state()
        print(
            f"grasp status={status.name}  is_grasped={st.is_grasped if st else '?'} "
            f"width={st.width if st else '?'}"
        )
        time.sleep(5.0 if not args.headless else 0.0)
    finally:
        client.close()
        server.stop()
        sim.stop()
        server_thread.join(timeout=2.0)


if __name__ == "__main__":
    main()
