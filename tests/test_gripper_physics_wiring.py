from unittest.mock import Mock

from franka_sim.franka_sim_server import FrankaSimServer
from franka_sim.gripper_backend import FrankaHandSim
from franka_sim.gripper_physics import GenesisFrankaHand


def test_physics_flag_selects_genesis_backend_bound_to_sim():
    sim = Mock()
    server = FrankaSimServer(
        enable_vis=False, genesis_sim=sim, enable_gripper=True, gripper_physics=True
    )
    backend = server.gripper_server.backend
    assert isinstance(backend, GenesisFrankaHand)
    assert backend.sim is sim


def test_default_keeps_kinematic_backend():
    sim = Mock()
    server = FrankaSimServer(enable_vis=False, genesis_sim=sim, enable_gripper=True)
    assert isinstance(server.gripper_server.backend, FrankaHandSim)


def test_self_constructed_sim_enables_hand_under_physics(monkeypatch):
    captured = {}

    # Spy on FrankaGenesisSim construction.
    import franka_sim.franka_sim_server as mod

    class SpySim(Mock):
        def __init__(self, *a, **kw):
            super().__init__()
            captured.update(kw)

    monkeypatch.setattr(mod, "FrankaGenesisSim", SpySim)
    FrankaSimServer(enable_vis=False, enable_gripper=True, gripper_physics=True)
    assert captured.get("enable_hand") is True


def test_physics_without_gripper_does_not_build_hand(monkeypatch):
    """gripper_physics=True + enable_gripper=False must not build a 9-DOF sim."""
    captured = {}

    import franka_sim.franka_sim_server as mod

    class SpySim(Mock):
        def __init__(self, *a, **kw):
            super().__init__()
            captured.update(kw)

    monkeypatch.setattr(mod, "FrankaGenesisSim", SpySim)
    server = FrankaSimServer(enable_vis=False, enable_gripper=False, gripper_physics=True)
    # (a) sim must NOT be built with the hand
    assert captured.get("enable_hand") is False
    # (b) no gripper server when gripper is disabled
    assert server.gripper_server is None
