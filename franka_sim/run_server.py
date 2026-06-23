#!/usr/bin/env python3
import argparse
import logging

from franka_sim import FrankaSimServer

# Configure logging to silence Numba debug output
logging.getLogger("numba").setLevel(logging.WARNING)


def main():
    """Run the Franka simulation server."""
    # get command line arguments visualization
    parser = argparse.ArgumentParser(description="Run a Franka simulation server")
    parser.add_argument(
        "-v",
        "--vis",
        action="store_true",
        default=False,
        help="Enable visualization of the Genesis simulator",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        default=None,
        help="URDF served to the client via GetRobotModel "
        "(defaults to the bundled hand-less FR3 arm model)",
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        default=False,
        help="Disable the co-located gripper server (port 1338)",
    )
    parser.add_argument(
        "--gripper-physics",
        action="store_true",
        help="Use the Genesis physics gripper (9-DOF, fingers move in the viewer)",
    )
    args = parser.parse_args()

    print(f"Starting Franka Simulation Server {'with' if args.vis else 'without'} visualization")
    print("Connect to the server using 'localhost' or '127.0.0.1' as the robot IP address")
    print("Press Ctrl+C to stop the server")

    server = FrankaSimServer(
        enable_vis=args.vis,
        urdf_path=args.urdf,
        enable_gripper=not args.no_gripper,
        gripper_physics=args.gripper_physics,
    )
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.stop()


if __name__ == "__main__":
    main()
