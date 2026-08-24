"""franka-sim: a headless, interface-identical Franka FCI simulator.

Genesis is optional. The default physics backend is MuJoCo, so every name below
that would pull ``genesis`` in is resolved lazily through the module-level
``__getattr__`` (PEP 562) instead of being imported eagerly: ``import
franka_sim`` on a machine without Genesis installed must keep working, and does.
Attribute access is unchanged for callers -- ``franka_sim.MobileDuoScene`` still
works, it just imports Genesis at that moment.
"""

import importlib

from franka_sim.control_modes import ControlMode
from franka_sim.franka_protocol import Command, ConnectStatus, MessageHeader, RobotMode
from franka_sim.franka_sim_server import FrankaSimServer
from franka_sim.gripper_backend import FrankaHandSim, GripperBackend
from franka_sim.gripper_physics import GenesisFrankaHand
from franka_sim.gripper_server import FrankaGripperServer
from franka_sim.mobile.spine_stub import SpineModel, SpineStubServer
from franka_sim.mobile.swerve_base import SwerveBase
from franka_sim.mobile.swerve_kinematics import SwerveKinematics
from franka_sim.robot_state import RobotState
from franka_sim.run_server import main as run_server_main

#: Names whose defining module imports a physics engine (``genesis`` or
#: ``mujoco``), resolved on first access instead of at package import.
_LAZY_EXPORTS = {
    "FrankaGenesisSim": "franka_sim.franka_genesis_sim",
    "MobileDuoRunner": "franka_sim.mobile.runner",
    "MobileDuoScene": "franka_sim.mobile.duo_sim",
    "MujocoFrankaSim": "franka_sim.mujoco_franka_sim",
    "SceneView": "franka_sim.mobile.duo_sim",
    "TMRGenesisSim": "franka_sim.mobile.tmr_genesis_sim",
    "parse_bind_specs": "franka_sim.mobile.runner",
}


def __getattr__(name):
    """Import an engine-backed export on first access (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(module_name), name)


def __dir__():
    """List the eager and lazy exports together, so tab-completion sees both."""
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "Command",
    "ConnectStatus",
    "ControlMode",
    "RobotMode",
    "MessageHeader",
    "RobotState",
    "FrankaSimServer",
    "FrankaGenesisSim",
    "MujocoFrankaSim",
    "run_server_main",
    "GripperBackend",
    "FrankaHandSim",
    "GenesisFrankaHand",
    "FrankaGripperServer",
    "SwerveBase",
    "SwerveKinematics",
    "TMRGenesisSim",
    "MobileDuoScene",
    "SceneView",
    "MobileDuoRunner",
    "parse_bind_specs",
    "SpineModel",
    "SpineStubServer",
]
