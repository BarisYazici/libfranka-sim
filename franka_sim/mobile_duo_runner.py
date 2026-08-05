"""Run three FCI bridges against one shared mobile-duo Genesis scene.

Each bridge is an ordinary :class:`~franka_sim.franka_sim_server.FrankaSimServer`
attached to a :class:`~franka_sim.mobile_duo_sim.SceneView`, so the arm protocol
path is exactly the one the single-arm simulator already ships. libfranka
clients cannot override port 1337, so the bridges are separated by host IP: the
runner binds one address per role.
"""

import logging
import threading
from typing import Dict, Optional, Sequence

from franka_sim.franka_protocol import COMMAND_PORT
from franka_sim.franka_sim_server import FrankaSimServer
from franka_sim.mobile_duo_sim import ROLE_BASE, ROLES, MobileDuoScene

logger = logging.getLogger(__name__)


def parse_bind_specs(values: Sequence[str]) -> Dict[str, str]:
    """Parse repeated ``ROLE=HOST`` CLI values into a role -> host mapping.

    Every role in :data:`~franka_sim.mobile_duo_sim.ROLES` must be present
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
        scene: MobileDuoScene,
        binds: Dict[str, str],
        port: int = COMMAND_PORT,
        arm_urdf: Optional[str] = None,
    ):
        self.scene = scene
        self.binds = dict(binds)
        self.port = port
        self.threads: Dict[str, threading.Thread] = {}

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
        """Start each bridge's accept loop in a daemon thread."""
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
        """Stop every bridge and then the shared scene."""
        for role, server in self.servers.items():
            server.stop()
            thread = self.threads.get(role)
            if thread is not None:
                thread.join(timeout=3.0)
        self.threads.clear()
        self.scene.stop()
