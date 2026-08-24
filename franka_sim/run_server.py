#!/usr/bin/env python3
import argparse
import importlib
import logging
import os
import sys
import threading
import time
from typing import Optional

from franka_sim.franka_protocol import COMMAND_PORT
from franka_sim.mobile.spine_stub import SPINE_DEFAULT_HOST, SPINE_DEFAULT_PORT

# Configure logging to silence Numba debug output
logging.getLogger("numba").setLevel(logging.WARNING)

#: How long the whole shutdown -- stop() plus the interpreter's own exit, atexit
#: hooks and all -- may take before the process is forced out. Every stage of
#: stop() is individually bounded well below this; the margin covers the part of
#: the process' life that is not ours to bound. See :func:`_arm_exit_watchdog`.
#:
#: Sized for the mobile-duo worst case, which is the slowest shutdown this
#: module drives: ``MobileDuoRunner.stop()`` (``mobile/runner.py``) stops
#: its three FCI bridges (arms + base) one at a time, then the shared scene.
#: Each bridge's own ``FrankaSimServer.stop()``/``cleanup()`` sums to roughly
#: 2.2 s in its own internal, individually-bounded waits -- ``cleanup()``'s
#: 0.1 s settle sleep, ``_stop_gripper``'s ``GRIPPER_JOIN_TIMEOUT_S`` (1.0 s),
#: and ``handle_client``'s ``tcp_thread.join(timeout=1.0)`` -- underneath the
#: 3.0 s outer join ``MobileDuoRunner.stop()`` itself applies per bridge, so
#: three bridges cost up to ``3 * 3.0 = 9.0`` s in the worst case where every
#: internal bound is actually hit. The shared scene's viewer teardown adds
#: ``VIEWER_CLOSE_TIMEOUT_S`` (``sim_common.py``, 2.0 s) on top of that. 5.0 s
#: was already tight for a *single*-role server and comfortably too short for
#: three bridges plus a viewer (up to ``9.0 + 2.0 = 11.0`` s); 15.0 s leaves a
#: margin above even that worst case.
SHUTDOWN_WATCHDOG_S = 15.0

#: Mobile-duo physics backends, mapped to the module and class implementing the
#: scene contract MobileDuoRunner consumes. Imported lazily in
#: :func:`resolve_scene_class` so choosing one backend never pays the other's
#: (multi-second, native) import cost.
MOBILE_DUO_PHYSICS = {
    "genesis": ("franka_sim.mobile.duo_sim", "MobileDuoScene"),
    "mujoco": ("franka_sim.mobile.duo_mujoco_sim", "MobileDuoMujocoScene"),
}

#: Physics backend used unless ``--physics`` says otherwise, for the single arm
#: and the mobile duo alike. MuJoCo holds real time at the 1 ms step the FCI
#: serves, where Genesis needs 2.5 ms and still falls behind in the duo scene.
DEFAULT_PHYSICS = "mujoco"


def _arm_exit_watchdog(timeout: float = SHUTDOWN_WATCHDOG_S) -> None:
    """Last resort: force the process out if shutdown wedges past ``timeout``.

    Not the shutdown mechanism -- everything in ``stop()`` is ordered, bounded
    and joined -- but the tail of the process' life is not ours to bound. The
    MuJoCo viewer leaves GL/driver ``atexit`` hooks behind, and a deadlock in
    those runs entirely in C: no Python bytecode executes again, so the pending
    SIGINT is never delivered to a handler and further Ctrl+C presses do
    nothing at all. Only an independent thread can get the process out of that,
    and only ``os._exit`` (no atexit, no finalisation) can do it without
    re-entering the very code that is stuck.

    A daemon thread, so it never delays a shutdown that does complete: if the
    process exits first, this thread simply dies with it and nothing is printed.
    """

    def _bail():
        time.sleep(timeout)
        _force_exit(
            f"Shutdown did not finish within {timeout:.0f}s (stuck in native "
            "teardown); forcing exit."
        )

    threading.Thread(target=_bail, name="shutdown-watchdog", daemon=True).start()


def _force_exit(message: str, code: int = 130) -> None:
    """Print ``message`` and leave immediately, skipping interpreter shutdown.

    ``os._exit`` runs no atexit hook and flushes no buffer, so anything still
    sitting in stdout's buffer (``print`` is block-buffered when stdout is a
    pipe rather than a terminal) would be lost -- including the "Shutting down
    server..." line the user is looking at. Flush first, then go.
    """
    try:
        print(message, file=sys.stderr, flush=True)
        sys.stdout.flush()
    except Exception:  # pragma: no cover - the streams may already be gone
        pass
    os._exit(code)


def _shutdown(stoppable) -> None:
    """Run ``stoppable.stop()`` under the shutdown watchdog, absorbing a re-interrupt.

    A second Ctrl+C almost always lands here, because this is where the
    shutdown time is spent. Left alone it raises KeyboardInterrupt out of the
    middle of teardown -- a traceback on top of the shutdown log, and whatever
    had not been released yet (the listening socket, the viewer's GL context)
    left dangling. Catching it turns the second press into what the user meant
    by it: leave now.
    """
    _arm_exit_watchdog()
    try:
        stoppable.stop()
    except KeyboardInterrupt:
        _force_exit("\nInterrupted during shutdown; exiting now.")
    except Exception:
        logging.getLogger(__name__).exception("Error during shutdown")


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
        help="Enable visualization of the simulator",
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
        help="Use the physics gripper (9-DOF, fingers move in the viewer)",
    )
    parser.add_argument(
        "--enforce-comm-constraints",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Abort a motion with communication_constraints_violation after 20 "
        "consecutively lost command cycles, as the real FCI does. Off by "
        "default; packet loss is tracked and reported in "
        "control_command_success_rate either way, and a missed motion-generator "
        "cycle is extrapolated (a missed torque cycle is held). Same as "
        "setting FRANKA_SIM_ENFORCE_COMM_CONSTRAINTS=1; pass "
        "--no-enforce-comm-constraints to force it off even when that is set",
    )
    parser.add_argument(
        "--enforce-motion-limits",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Abort a motion when a commanded signal breaks the FCI's joint "
        "position, velocity, acceleration, jerk or torque-rate limits, as the "
        "real robot does. Off by default; violations are always checked and "
        "logged either way. Independent of --enforce-comm-constraints. Same as "
        "setting FRANKA_SIM_ENFORCE_MOTION_LIMITS=1; pass "
        "--no-enforce-motion-limits to force it off even when that is set",
    )
    parser.add_argument(
        "--mobile-duo",
        action="store_true",
        default=False,
        help="Serve the mobile FR3 duo: one physics scene, three FCI bridges "
        "(left arm, right arm, TMR base)",
    )
    parser.add_argument(
        "--scene-urdf",
        type=str,
        default=None,
        help="Combined mobile_fr3_duo URDF loaded into the scene (required with --mobile-duo)",
    )
    parser.add_argument(
        "--physics",
        choices=sorted(MOBILE_DUO_PHYSICS),
        default=DEFAULT_PHYSICS,
        help="Physics backend for the single arm and the mobile-duo scene alike: "
        "mujoco, which holds real time at a 1 ms step, or genesis "
        f"(default: {DEFAULT_PHYSICS})",
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
        return

    if not args.scene_urdf:
        raise ValueError("--scene-urdf is required with --mobile-duo")

    from franka_sim.mobile.runner import parse_bind_specs

    parse_bind_specs(args.bind)


def comm_constraints_setting(args) -> Optional[bool]:
    """Translate ``--enforce-comm-constraints`` into the server's tri-state flag.

    Three states, not two. ``None`` -- neither flag given -- means "no opinion"
    and leaves the decision to ``$FRANKA_SIM_ENFORCE_COMM_CONSTRAINTS``. True
    and False are explicit and *do* override the environment, which is the
    point of ``--no-enforce-comm-constraints``: without it there was no way to
    switch enforcement off for one run of a server started from a shell (or a
    launch file, or a container) that exports the variable.
    """
    return args.enforce_comm_constraints


def motion_limits_setting(args) -> Optional[bool]:
    """Translate ``--enforce-motion-limits`` into the server's tri-state flag.

    Same tri-state as :func:`comm_constraints_setting`, with
    ``--no-enforce-motion-limits`` as the explicit off switch that overrides
    ``$FRANKA_SIM_ENFORCE_MOTION_LIMITS``.
    """
    return args.enforce_motion_limits


def run_mobile_duo(args) -> None:
    """Bring up the three-bridge mobile duo simulation (blocks)."""
    from franka_sim.mobile.runner import MobileDuoRunner, parse_bind_specs

    binds = parse_bind_specs(args.bind)
    scene = resolve_scene_class(args.physics)(
        args.scene_urdf,
        mesh_root=args.mesh_root,
        enable_vis=args.vis,
    )
    spine_server = None
    if args.spine:
        from franka_sim.mobile.spine_stub import (
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
        scene,
        binds,
        port=args.port,
        arm_urdf=args.urdf,
        spine_server=spine_server,
        enforce_comm_constraints=comm_constraints_setting(args),
        enforce_motion_limits=motion_limits_setting(args),
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
        print("\nShutting down server...", flush=True)
    finally:
        # finally, not just the KeyboardInterrupt path: the run loop also ends
        # when the viewer window is closed, and that exit has exactly the same
        # resources to release.
        _shutdown(runner)


def run_single_arm(args) -> None:
    """Bring up the classic single-arm simulation server (blocks)."""
    from franka_sim import FrankaSimServer

    print(f"Starting Franka Simulation Server {'with' if args.vis else 'without'} visualization")
    print(f"  physics backend -> {args.physics}")
    print("Connect to the server using 'localhost' or '127.0.0.1' as the robot IP address")
    print("Press Ctrl+C to stop the server")

    server = FrankaSimServer(
        enable_vis=args.vis,
        urdf_path=args.urdf,
        enable_gripper=not args.no_gripper,
        gripper_physics=args.gripper_physics,
        physics=args.physics,
        enforce_comm_constraints=comm_constraints_setting(args),
        enforce_motion_limits=motion_limits_setting(args),
    )
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down server...", flush=True)
    finally:
        # finally, not just the KeyboardInterrupt path: start() also returns
        # normally when the viewer window is closed, and that exit leaves the
        # same sockets bound and the same GL context alive.
        _shutdown(server)


def main():
    """Run the Franka simulation server."""
    # Make the sim's own logging visible by default: without a handler,
    # operationally important lines (idle hold engaged, RTF overload,
    # motion-limit violations) are silently dropped. Respect any handler
    # the embedding application configured first.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    args = build_parser().parse_args()
    try:
        validate_args(args)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(2)

    try:
        if args.mobile_duo:
            run_mobile_duo(args)
        else:
            run_single_arm(args)
    except KeyboardInterrupt:
        # A Ctrl+C that landed in the narrow gaps the shutdown path cannot
        # catch itself -- inside the handler that announces the shutdown, or
        # just after it finished. The teardown has already run (it is in a
        # finally block); all that is left is to exit with the conventional
        # interrupted status instead of dumping a traceback the user can do
        # nothing about.
        sys.exit(130)


if __name__ == "__main__":
    main()
