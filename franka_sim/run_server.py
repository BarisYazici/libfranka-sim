#!/usr/bin/env python3
import argparse
import importlib
import logging
import sys

from franka_sim.franka_protocol import COMMAND_PORT
from franka_sim.spine_stub import SPINE_DEFAULT_HOST, SPINE_DEFAULT_PORT

# Configure logging to silence Numba debug output
logging.getLogger("numba").setLevel(logging.WARNING)

#: Mobile-duo physics backends, mapped to the module and class implementing the
#: scene contract MobileDuoRunner consumes. Imported lazily in
#: :func:`resolve_scene_class` so choosing one backend never pays the other's
#: (multi-second, native) import cost.
MOBILE_DUO_PHYSICS = {
    "genesis": ("franka_sim.mobile_duo_sim", "MobileDuoScene"),
    "mujoco": ("franka_sim.mobile_duo_mujoco_sim", "MobileDuoMujocoScene"),
}


def resolve_scene_class(physics: str):
    """Import and return the mobile-duo scene class for one physics backend."""
    module_name, class_name = MOBILE_DUO_PHYSICS[physics]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the simulation server."""
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
    parser.add_argument(
        "--mobile-duo",
        action="store_true",
        default=False,
        help="Serve the mobile FR3 duo: one Genesis scene, three FCI bridges "
        "(left arm, right arm, TMR base)",
    )
    parser.add_argument(
        "--scene-urdf",
        type=str,
        default=None,
        help="Combined mobile_fr3_duo URDF loaded into Genesis (required with --mobile-duo)",
    )
    parser.add_argument(
        "--physics",
        choices=sorted(MOBILE_DUO_PHYSICS),
        default="genesis",
        help="Physics backend for the mobile-duo scene (requires --mobile-duo): "
        "genesis (default) or mujoco, which holds real time at a 1 ms step",
    )
    parser.add_argument(
        "--mesh-root",
        type=str,
        default=None,
        help="Package root used to resolve package:// mesh URIs "
        "(a franka_description checkout; defaults to the URDF's directory)",
    )
    parser.add_argument(
        "--bind",
        action="append",
        default=[],
        metavar="ROLE=HOST",
        help="Bind one bridge to a host address; repeat for left, right and base",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=COMMAND_PORT,
        help=f"TCP command port for every bridge (default: {COMMAND_PORT})",
    )
    parser.add_argument(
        "--spine",
        action="store_true",
        default=False,
        help="Also run the fake spine REST device in-process and drive the "
        "franka_spine_vertical_joint from it (requires --mobile-duo)",
    )
    parser.add_argument(
        "--spine-host",
        default=SPINE_DEFAULT_HOST,
        help=f"Address the spine stub binds (default: {SPINE_DEFAULT_HOST})",
    )
    parser.add_argument(
        "--spine-port",
        type=int,
        default=SPINE_DEFAULT_PORT,
        help="Port the spine stub binds; SpineApiClient hardcodes "
        f"{SPINE_DEFAULT_PORT} (default: {SPINE_DEFAULT_PORT})",
    )
    parser.add_argument("--spine-cert", default=None, help="TLS certificate for the spine stub")
    parser.add_argument("--spine-key", default=None, help="TLS private key for the spine stub")
    return parser


def validate_args(args) -> None:
    """Reject incoherent CLI combinations. Raises ValueError."""
    if not args.mobile_duo:
        if args.scene_urdf or args.bind:
            raise ValueError("--scene-urdf and --bind require --mobile-duo")
        if args.spine or args.spine_cert or args.spine_key:
            raise ValueError("--spine and the --spine-* options require --mobile-duo")
        if args.physics != "genesis":
            raise ValueError("--physics requires --mobile-duo")
        return

    if not args.scene_urdf:
        raise ValueError("--scene-urdf is required with --mobile-duo")

    from franka_sim.mobile_duo_runner import parse_bind_specs

    parse_bind_specs(args.bind)


def run_mobile_duo(args) -> None:
    """Bring up the three-bridge mobile duo simulation (blocks)."""
    from franka_sim.mobile_duo_runner import MobileDuoRunner, parse_bind_specs

    binds = parse_bind_specs(args.bind)
    scene = resolve_scene_class(args.physics)(
        args.scene_urdf,
        mesh_root=args.mesh_root,
        enable_vis=args.vis,
    )
    spine_server = None
    if args.spine:
        from franka_sim.spine_stub import (
            SPINE_CERT_DIR,
            SpineStubServer,
            make_self_signed_cert,
        )

        certfile = args.spine_cert
        keyfile = args.spine_key
        if certfile is None:
            certfile, keyfile = make_self_signed_cert(SPINE_CERT_DIR)
        spine_server = SpineStubServer(
            host=args.spine_host,
            port=args.spine_port,
            certfile=certfile,
            keyfile=keyfile,
        )

    runner = MobileDuoRunner(
        scene, binds, port=args.port, arm_urdf=args.urdf, spine_server=spine_server
    )

    print(f"  physics backend -> {args.physics} (dt={scene.dt}s)")
    for role, host in binds.items():
        print(f"  {role:>5} bridge -> {host}:{args.port}")
    if spine_server is not None:
        print(f"  spine device -> https://{args.spine_host}:{args.spine_port}/spine/api")
    print("Press Ctrl+C to stop the server")

    try:
        runner.run_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        runner.stop()


def run_single_arm(args) -> None:
    """Bring up the classic single-arm simulation server (blocks)."""
    from franka_sim import FrankaSimServer

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


def main():
    """Run the Franka simulation server."""
    args = build_parser().parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)

    if args.mobile_duo:
        run_mobile_duo(args)
    else:
        run_single_arm(args)


if __name__ == "__main__":
    main()
