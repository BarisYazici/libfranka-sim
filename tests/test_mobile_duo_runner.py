import socket
import sys
import threading
import time

import pytest
from fakes import FakeDuoEntity

from franka_sim import run_server
from franka_sim.franka_protocol import COMMAND_PORT
from franka_sim.mobile_duo_runner import MobileDuoRunner, parse_bind_specs
from franka_sim.mobile_duo_sim import ROLE_BASE, ROLE_LEFT, ROLE_RIGHT, MobileDuoScene

LOOPBACK = {ROLE_LEFT: "127.0.0.11", ROLE_RIGHT: "127.0.0.12", ROLE_BASE: "127.0.0.10"}


@pytest.fixture
def bound_scene(tmp_path):
    urdf_path = tmp_path / "duo.urdf"
    urdf_path.write_text('<?xml version="1.0"?><robot name="duo"></robot>')
    scene = MobileDuoScene(urdf_path, enable_vis=False, base_height=0.05)
    scene.robot = FakeDuoEntity()
    scene._bind_entity()
    scene._read_and_publish_state()
    return scene


# --- bind spec parsing -----------------------------------------------------


def test_parse_bind_specs_returns_all_three_roles():
    binds = parse_bind_specs(["left=10.0.0.1", "right=10.0.0.2", "base=10.0.0.3"])
    assert binds == {"left": "10.0.0.1", "right": "10.0.0.2", "base": "10.0.0.3"}


@pytest.mark.parametrize(
    "values",
    [
        ["left10.0.0.1", "right=10.0.0.2", "base=10.0.0.3"],
        ["=10.0.0.1", "right=10.0.0.2", "base=10.0.0.3"],
        ["left=", "right=10.0.0.2", "base=10.0.0.3"],
    ],
    ids=["no_separator", "no_role", "no_host"],
)
def test_parse_bind_specs_rejects_malformed_specs(values):
    with pytest.raises(ValueError, match="ROLE=HOST"):
        parse_bind_specs(values)


def test_parse_bind_specs_rejects_an_unknown_role():
    with pytest.raises(ValueError, match="unknown"):
        parse_bind_specs(["middle=10.0.0.1", "right=10.0.0.2", "base=10.0.0.3"])


def test_parse_bind_specs_rejects_a_duplicate_role():
    with pytest.raises(ValueError, match="duplicate"):
        parse_bind_specs(["left=10.0.0.1", "left=10.0.0.9", "base=10.0.0.3"])


def test_parse_bind_specs_rejects_missing_roles():
    with pytest.raises(ValueError, match="missing"):
        parse_bind_specs(["left=10.0.0.1"])


# --- runner wiring ---------------------------------------------------------


def test_runner_builds_one_server_per_role(bound_scene):
    runner = MobileDuoRunner(bound_scene, LOOPBACK)
    assert set(runner.servers) == {ROLE_LEFT, ROLE_RIGHT, ROLE_BASE}
    for role, server in runner.servers.items():
        assert server.host == LOOPBACK[role]
        assert server.port == COMMAND_PORT


def test_runner_marks_only_the_base_server_as_mobile(bound_scene):
    runner = MobileDuoRunner(bound_scene, LOOPBACK)
    assert runner.servers[ROLE_BASE].mobile_base is True
    assert runner.servers[ROLE_LEFT].mobile_base is False
    assert runner.servers[ROLE_RIGHT].mobile_base is False


def test_runner_disables_the_fci_gripper_server(bound_scene):
    """Robotiq is not emulated in milestone 1, and port 1338 is not needed."""
    runner = MobileDuoRunner(bound_scene, LOOPBACK)
    for server in runner.servers.values():
        assert server.gripper_server is None


def test_runner_attaches_each_server_to_its_own_view(bound_scene):
    runner = MobileDuoRunner(bound_scene, LOOPBACK)
    for role, server in runner.servers.items():
        assert server.genesis_sim.role == role
        assert server.genesis_sim.scene is bound_scene


def test_runner_honours_a_custom_port(bound_scene):
    runner = MobileDuoRunner(bound_scene, LOOPBACK, port=13370)
    assert all(server.port == 13370 for server in runner.servers.values())


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="127.0.0.0/8 loopback aliases are Linux-specific",
)
def test_all_three_bridges_accept_connections_on_port_1337(bound_scene):
    runner = MobileDuoRunner(bound_scene, LOOPBACK)
    runner.start_servers()
    try:
        deadline = time.time() + 5.0
        for role, host in LOOPBACK.items():
            while True:
                try:
                    probe = socket.create_connection((host, COMMAND_PORT), timeout=1.0)
                    probe.close()
                    break
                except OSError:
                    if time.time() > deadline:
                        raise AssertionError(f"{role} bridge never accepted on {host}")
                    time.sleep(0.1)
    finally:
        runner.stop()


# --- CLI -------------------------------------------------------------------


def test_cli_parses_the_mobile_duo_invocation():
    args = run_server.build_parser().parse_args(
        [
            "--mobile-duo",
            "--scene-urdf",
            "/tmp/duo.urdf",
            "--mesh-root",
            "/tmp/franka_description",
            "--bind",
            "left=127.0.0.11",
            "--bind",
            "right=127.0.0.12",
            "--bind",
            "base=127.0.0.10",
        ]
    )
    assert args.mobile_duo is True
    assert args.scene_urdf == "/tmp/duo.urdf"
    assert args.mesh_root == "/tmp/franka_description"
    assert args.bind == ["left=127.0.0.11", "right=127.0.0.12", "base=127.0.0.10"]
    assert args.port == COMMAND_PORT
    run_server.validate_args(args)


def test_cli_requires_a_scene_urdf_for_mobile_duo():
    args = run_server.build_parser().parse_args(
        ["--mobile-duo", "--bind", "left=a", "--bind", "right=b", "--bind", "base=c"]
    )
    with pytest.raises(ValueError, match="--scene-urdf"):
        run_server.validate_args(args)


def test_cli_requires_all_three_binds_for_mobile_duo():
    args = run_server.build_parser().parse_args(
        ["--mobile-duo", "--scene-urdf", "/tmp/duo.urdf", "--bind", "left=127.0.0.11"]
    )
    with pytest.raises(ValueError, match="missing"):
        run_server.validate_args(args)


def test_cli_defaults_leave_the_single_arm_path_untouched():
    args = run_server.build_parser().parse_args([])
    assert args.mobile_duo is False
    assert args.scene_urdf is None
    assert args.bind == []
    assert args.urdf is None
    assert args.no_gripper is False
    assert args.gripper_physics is False
    run_server.validate_args(args)


class StubSpineServer:
    """Stands in for SpineStubServer: exposes .model, .start() and .stop()."""

    def __init__(self, position_m=0.0):
        from franka_sim.spine_stub import SpineModel

        self.model = SpineModel(position_m=position_m)
        self.port = 4430
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def test_runner_shares_the_spine_model_with_the_scene(bound_scene):
    spine = StubSpineServer()
    runner = MobileDuoRunner(bound_scene, LOOPBACK, spine_server=spine)
    assert bound_scene.spine_model is spine.model


def test_runner_leaves_the_spine_unset_without_a_stub(bound_scene):
    MobileDuoRunner(bound_scene, LOOPBACK)
    assert bound_scene.spine_model is None


def test_runner_starts_and_stops_the_spine_stub(bound_scene):
    spine = StubSpineServer()
    runner = MobileDuoRunner(bound_scene, LOOPBACK, spine_server=spine)
    runner.start_servers()
    try:
        assert spine.started is True
    finally:
        runner.stop()
    assert spine.stopped is True
    assert bound_scene.spine_model is None


def test_a_rest_move_raises_the_spine_joint_in_the_scene(bound_scene):
    """The whole point of --spine: a REST command moves the lift in the viewer."""
    spine = StubSpineServer()
    MobileDuoRunner(bound_scene, LOOPBACK, spine_server=spine)

    spine.model.switch_on()
    spine.model.start_motion(0.5, 1.0)
    time.sleep(0.3)
    bound_scene._apply_control()

    values, dofs = bound_scene.robot.set_position_calls[-1]
    assert dofs == [bound_scene.spine_dof_idx]
    assert 0.0 < float(values[0]) <= 0.5


def test_cli_parses_the_spine_flags():
    args = run_server.build_parser().parse_args(
        [
            "--mobile-duo",
            "--scene-urdf",
            "/tmp/duo.urdf",
            "--bind",
            "left=127.0.0.11",
            "--bind",
            "right=127.0.0.12",
            "--bind",
            "base=127.0.0.10",
            "--spine",
        ]
    )
    assert args.spine is True
    assert args.spine_host == "127.0.0.13"
    assert args.spine_port == 443
    assert args.spine_cert is None
    assert args.spine_key is None
    run_server.validate_args(args)


def test_cli_rejects_spine_without_mobile_duo():
    args = run_server.build_parser().parse_args(["--spine"])
    with pytest.raises(ValueError, match="--mobile-duo"):
        run_server.validate_args(args)
