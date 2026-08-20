"""Run three FCI bridges against one shared mobile-duo scene (Genesis or MuJoCo).

Each bridge is an ordinary :class:`~franka_sim.franka_sim_server.FrankaSimServer`
attached to a :class:`~franka_sim.mobile_duo_common.SceneView`, so the arm
protocol path is exactly the one the single-arm simulator already ships.
libfranka clients cannot override port 1337, so the bridges are separated by
host IP: the runner binds one address per role.

The runner is engine-agnostic -- ``scene`` can be either
:class:`~franka_sim.mobile_duo_sim.MobileDuoScene` (Genesis) or
:class:`~franka_sim.mobile_duo_mujoco_sim.MobileDuoMujocoScene` (MuJoCo) -- so
this module imports only :mod:`franka_sim.mobile_duo_common` (genesis-free) at
runtime; the Genesis scene type is imported under ``TYPE_CHECKING`` only, so
``--physics mujoco`` never pays Genesis' import cost.
"""

import logging
import threading
from typing import TYPE_CHECKING, Dict, Optional, Sequence

from franka_sim.franka_protocol import COMMAND_PORT
from franka_sim.franka_sim_server import FrankaSimServer
from franka_sim.mobile_duo_common import ROLE_BASE, ROLES

if TYPE_CHECKING:
    from franka_sim.mobile_duo_sim import MobileDuoScene

logger = logging.getLogger(__name__)


def parse_bind_specs(values: Sequence[str]) -> Dict[str, str]:
    """Parse repeated ``ROLE=HOST`` CLI values into a role -> host mapping.

    Every role in :data:`~franka_sim.mobile_duo_common.ROLES` must be present
    exactly once.
    """
    binds: Dict[str, str] = {}
    for value in values:
        role, separator, host = value.partition("=")
        if not separator or not role or not host:
            raise ValueError(f"--bind expects ROLE=HOST, got {value!r}")
        if role not in ROLES:
            raise ValueError(f"unknown --bind role {role!r}; expected one of {ROLES}")
        if role in binds:
            raise ValueError(f"duplicate --bind role {role!r}")
        binds[role] = host

    missing = [role for role in ROLES if role not in binds]
    if missing:
        raise ValueError(f"missing --bind for role(s): {', '.join(missing)}")
    return binds


class MobileDuoRunner:
    """Own the lifecycle of three bridges and the one scene they share."""

    def __init__(
        self,
        scene: "MobileDuoScene",
        binds: Dict[str, str],
        port: int = COMMAND_PORT,
        arm_urdf: Optional[str] = None,
        spine_server=None,
    ):
        self.scene = scene
        self.binds = dict(binds)
        self.port = port
        self.spine_server = spine_server
        self.threads: Dict[str, threading.Thread] = {}

        # The scene reads the stub's SpineModel every physics step, so the lift
        # (and the arms it carries) moves in the viewer. Sharing the object
        # in-process avoids any IPC or clock skew between the two.
        if spine_server is not None:
            scene.spine_model = spine_server.model

        # enable_gripper=False: Robotiq is not emulated in this milestone, so no
        # FCI gripper server (port 1338) is served on any of the three bridges.
        self.servers: Dict[str, FrankaSimServer] = {
            role: FrankaSimServer(
                host=self.binds[role],
                port=port,
                genesis_sim=scene.view(role),
                urdf_path=arm_urdf,
                enable_gripper=False,
                mobile_base=(role == ROLE_BASE),
            )
            for role in ROLES
        }

    def start_servers(self) -> None:
        """Start the spine stub (if any) and each bridge's accept loop."""
        if self.spine_server is not None:
            self.spine_server.start()
            logger.info("Spine stub listening on port %s", self.spine_server.port)

        for role, server in self.servers.items():
            thread = threading.Thread(
                target=server.run_server, name=f"fci-bridge-{role}", daemon=True
            )
            thread.start()
            self.threads[role] = thread
            logger.info("Bridge %s listening on %s:%s", role, self.binds[role], self.port)

    def run_forever(self) -> None:
        """Build the scene, start the bridges and step physics in this thread."""
        self.scene.initialize_simulation()
        self.start_servers()
        self.scene.start()

    def stop(self) -> None:
        """Stop every bridge and then the shared scene.

        Each bridge's stop() is isolated with its own try/except: a failure
        tearing one bridge down must not prevent the others from being asked
        to stop, and must never skip stopping the shared scene (and the spine
        stub) -- otherwise a single crashed bridge would leak the whole
        scene's Genesis process.
        """
        for role, server in self.servers.items():
            try:
                server.stop()
            except Exception:
                logger.exception("Error stopping bridge %s", role)
            thread = self.threads.get(role)
            if thread is not None:
                thread.join(timeout=3.0)
        self.threads.clear()
        if self.spine_server is not None:
            try:
                self.spine_server.stop()
            except Exception:
                logger.exception("Error stopping spine stub")
            self.scene.spine_model = None
        self.scene.stop()
