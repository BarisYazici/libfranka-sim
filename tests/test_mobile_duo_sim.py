import math
import types
from unittest.mock import Mock

import numpy as np
import pytest
from fakes import FakeDuoEntity

from franka_sim.franka_genesis_sim import (
    DEFAULT_FR3_DAMPING,
    ControlMode,
    resolve_fr3_joint_damping,
    resolve_gs_backend,
)
from franka_sim.mobile_duo_sim import (
    ARM_EE_LINKS,
    ARM_INITIAL_Q,
    ARM_JOINT_NAMES,
    LINK_POSE_READ_EVERY,
    ROLE_BASE,
    ROLE_LEFT,
    ROLE_RIGHT,
    ROLES,
    SPINE_HOLD_KP,
    SPINE_HOLD_KV,
    MobileDuoScene,
    RealtimeFactorMonitor,
    SceneView,
    pose_to_column_major,
)
from franka_sim.swerve_base import TMR_JOINT_ORDER, TMR_WHEEL_RADIUS


@pytest.fixture
def scene(tmp_path):
    """A MobileDuoScene bound to a fake entity, with no Genesis scene built."""
    urdf_path = tmp_path / "duo.urdf"
    urdf_path.write_text('<?xml version="1.0"?><robot name="duo"></robot>')
    duo = MobileDuoScene(urdf_path, enable_vis=False, base_height=0.05)
    duo.robot = FakeDuoEntity()
    duo._bind_entity()
    duo._read_and_publish_state()
    return duo


def test_joint_names_match_the_generated_urdf():
    assert ARM_JOINT_NAMES[ROLE_LEFT] == [f"left_fr3v2_joint{i}" for i in range(1, 8)]
    assert ARM_JOINT_NAMES[ROLE_RIGHT] == [f"right_fr3v2_joint{i}" for i in range(1, 8)]
    assert ARM_EE_LINKS == {
        ROLE_LEFT: "left_fr3v2_link7",
        ROLE_RIGHT: "right_fr3v2_link7",
    }
    assert ROLES == (ROLE_LEFT, ROLE_RIGHT, ROLE_BASE)


def test_arm_initial_pose_matches_upstream_ros2_control_defaults():
    expected = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]
    assert ARM_INITIAL_Q == pytest.approx(expected)


def test_pose_to_column_major_is_identity_for_a_unit_quaternion():
    flat = pose_to_column_major([1.0, 2.0, 3.0], [1.0, 0.0, 0.0, 0.0])
    matrix = flat.reshape(4, 4).T
    assert matrix[:3, :3] == pytest.approx(np.eye(3), abs=1e-12)
    assert matrix[:3, 3] == pytest.approx([1.0, 2.0, 3.0])


def test_each_role_reports_seven_element_state(scene):
    for role in ROLES:
        state = scene.get_role_state(role)
        for key in ("q", "dq", "ddq", "q_d", "dq_d", "ddq_d", "tau_J"):
            assert state[key].shape == (7,), (role, key)
        assert state["O_T_EE"].shape == (16,)


def test_arm_state_reads_only_that_arms_joints(scene):
    left_dofs = [scene.robot.joints[name].dof_idx_local for name in ARM_JOINT_NAMES[ROLE_LEFT]]
    expected = [scene.robot.dof_positions[index] for index in left_dofs]
    assert scene.get_role_state(ROLE_LEFT)["q"] == pytest.approx(expected)


def test_each_arm_reports_its_own_flange_pose(scene):
    """The whole-entity link read is sliced by local link index, so a swapped or
    off-by-one index would hand one arm the other's flange (or a filler link's).
    """
    for role in (ROLE_LEFT, ROLE_RIGHT):
        link = scene.robot.links[ARM_EE_LINKS[role]]
        matrix = scene.get_role_state(role)["O_T_EE"].reshape(4, 4).T
        assert matrix[:3, 3] == pytest.approx(link._position), role
        expected = pose_to_column_major(link._position, link._quaternion)
        assert scene.get_role_state(role)["O_T_EE"] == pytest.approx(expected), role


def test_link_poses_are_re_read_on_the_decimation_period(scene):
    """q/dq come off the entity every step; the flange poses do not.

    Their consumers top out at 50 Hz, so LINK_POSE_READ_EVERY trades three of
    every four whole-entity link reads away. The pose must still refresh on the
    period, and must be the stale one in between.
    """
    left = scene.robot.links[ARM_EE_LINKS[ROLE_LEFT]]
    stale = left._position.copy()
    reads_before = scene.robot.links_read_count
    left._position = np.array([9.0, 9.0, 9.0])

    for _ in range(LINK_POSE_READ_EVERY - 1):
        scene._read_and_publish_state()
        assert scene.robot.links_read_count == reads_before
        assert scene.get_role_state(ROLE_LEFT)["O_T_EE"][12:15] == pytest.approx(stale)

    scene._read_and_publish_state()
    assert scene.robot.links_read_count == reads_before + 1
    assert scene.get_role_state(ROLE_LEFT)["O_T_EE"][12:15] == pytest.approx([9.0, 9.0, 9.0])


def test_base_state_pads_the_wheel_joints(scene):
    state = scene.get_role_state(ROLE_BASE)
    wheel_dofs = [scene.robot.joints[name].dof_idx_local for name in TMR_JOINT_ORDER]
    expected = [scene.robot.dof_positions[index] for index in wheel_dofs]
    assert state["q"][:4] == pytest.approx(expected)
    assert state["q"][4:] == pytest.approx([0.0, 0.0, 0.0])


def test_arm_torque_control_writes_only_that_arms_dofs(scene):
    torques = np.arange(7, dtype=float)
    scene.set_arm_control_mode(ROLE_RIGHT, ControlMode.TORQUE)
    scene.update_arm_torques(ROLE_RIGHT, torques)
    scene._apply_control()

    values, dofs = scene.robot.force_commands[-1]
    right_dofs = [scene.robot.joints[name].dof_idx_local for name in ARM_JOINT_NAMES[ROLE_RIGHT]]
    assert dofs == right_dofs
    assert values == pytest.approx(torques)


def test_left_arm_torque_control_writes_only_that_arms_dofs(scene):
    """Mirrors test_arm_torque_control_writes_only_that_arms_dofs above,
    pinned to the left arm instead, so both arms' write-routing is checked.
    Uses TORQUE (the right arm stays at its default POSITION mode and never
    touches force_commands), so the write is unambiguously the left arm's.
    """
    torques = np.arange(7, dtype=float) * -1.0
    scene.set_arm_control_mode(ROLE_LEFT, ControlMode.TORQUE)
    scene.update_arm_torques(ROLE_LEFT, torques)
    scene._apply_control()

    values, dofs = scene.robot.force_commands[-1]
    left_dofs = [scene.robot.joints[name].dof_idx_local for name in ARM_JOINT_NAMES[ROLE_LEFT]]
    right_dofs = [scene.robot.joints[name].dof_idx_local for name in ARM_JOINT_NAMES[ROLE_RIGHT]]
    assert dofs == left_dofs
    assert dofs != right_dofs
    assert values == pytest.approx(torques)


def test_arms_keep_independent_control_modes(scene):
    scene.set_arm_control_mode(ROLE_LEFT, ControlMode.TORQUE)
    scene.set_arm_control_mode(ROLE_RIGHT, ControlMode.POSITION)
    assert scene.arm_control_modes[ROLE_LEFT] is ControlMode.TORQUE
    assert scene.arm_control_modes[ROLE_RIGHT] is ControlMode.POSITION


def test_base_twist_moves_the_whole_robot(scene):
    scene.update_base_twist([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    scene._apply_control()

    assert scene.robot.positions[-1] == pytest.approx([scene.dt, 0.0, 0.05], abs=1e-12)
    drive_values, drive_dofs = scene.robot.velocity_commands[-1]
    assert drive_values == pytest.approx([1.0 / TMR_WHEEL_RADIUS] * 2, abs=1e-9)
    assert drive_dofs == scene.swerve.drive_dofs_idx


def test_spine_joint_is_bound(scene):
    assert scene.spine_dof_idx == scene.robot.joints["franka_spine_vertical_joint"].dof_idx_local


def test_set_spine_position_writes_the_prismatic_joint(scene):
    scene.set_spine_position(0.42)

    values, dofs = scene.robot.set_position_calls[-1]
    assert dofs == [scene.spine_dof_idx]
    assert values == pytest.approx([0.42])


def test_set_spine_position_never_zeroes_the_other_dofs(scene):
    """Regression: the lift write must not stop the arms.

    Genesis' ``set_dofs_position`` zeroes the velocity of EVERY DOF of the
    entity unless ``zero_velocity=False``. This runs once per physics step, so
    the default pinned both arms while the (kinematic) lift still moved. Only
    the spine's own DOF may be zeroed.
    """
    scene.set_spine_position(0.42)

    values, dofs, zero_velocity = scene.robot.set_position_writes[-1]
    assert dofs == [scene.spine_dof_idx]
    assert zero_velocity is False
    zeroed_values, zeroed_dofs = scene.robot.set_velocity_calls[-1]
    assert zeroed_dofs == [scene.spine_dof_idx]
    assert zeroed_values == pytest.approx([0.0])


def test_set_spine_position_clamps_to_the_urdf_limits(scene):
    scene.set_spine_position(-1.0)
    assert scene.robot.set_position_calls[-1][0] == pytest.approx([0.0])
    scene.set_spine_position(9.0)
    assert scene.robot.set_position_calls[-1][0] == pytest.approx([0.85])


def test_set_spine_position_holds_with_a_persisting_control_target(scene):
    """The teleport alone leaves an unactuated DOF free-running between writes.

    Genesis keeps a control target until it is overwritten, so pairing the
    teleport with one ``control_dofs_position`` holds the carriage for free --
    which is what lets the write-on-change skip below be unconditional.
    """
    scene.set_spine_position(0.42)

    values, dofs = scene.robot.position_commands[-1]
    assert dofs == [scene.spine_dof_idx]
    assert values == pytest.approx([0.42])


def test_set_spine_position_writes_nothing_while_the_target_is_unchanged(scene):
    scene.set_spine_position(0.42)
    writes = len(scene.robot.set_position_calls)
    commands = len(scene.robot.position_commands)

    for _ in range(50):
        scene.set_spine_position(0.42)

    assert len(scene.robot.set_position_calls) == writes
    assert len(scene.robot.position_commands) == commands


def test_initialize_simulation_arms_the_spine_hold(tmp_path, monkeypatch):
    """The lift gains have to be set once, before any position is commanded."""
    urdf_path = tmp_path / "duo.urdf"
    urdf_path.write_text('<?xml version="1.0"?><robot name="duo"></robot>')
    resolved = tmp_path / "resolved.urdf"
    resolved.write_text("<robot/>")

    duo_entity = FakeDuoEntity()
    fake_gs, _ = _fake_gs(robot_entity=duo_entity)

    duo = MobileDuoScene(urdf_path, enable_vis=False)
    monkeypatch.setattr("franka_sim.mobile_duo_sim.gs", fake_gs)
    monkeypatch.setattr("franka_sim.mobile_duo_sim.resolve_urdf_meshes", lambda *a, **kw: resolved)

    duo.initialize_simulation()

    spine_dofs = [duo.spine_dof_idx]
    assert [dofs for _, dofs in duo_entity.kp_calls] == [spine_dofs]
    assert [dofs for _, dofs in duo_entity.kv_calls] == [spine_dofs]
    assert duo_entity.kp_calls[0][0] == pytest.approx([SPINE_HOLD_KP])
    assert duo_entity.kv_calls[0][0] == pytest.approx([SPINE_HOLD_KV])
    assert duo_entity.position_commands[-1][1] == spine_dofs
    assert duo_entity.position_commands[-1][0] == pytest.approx([0.0])


def test_spine_model_drives_the_joint_every_step(scene):
    class StubSpine:
        def __init__(self):
            self.value = 0.0

        def position_m(self):
            return self.value

    spine = StubSpine()
    scene.spine_model = spine

    spine.value = 0.3
    scene._apply_control()
    assert scene.robot.set_position_calls[-1][0] == pytest.approx([0.3])

    spine.value = 0.6
    scene._apply_control()
    assert scene.robot.set_position_calls[-1][0] == pytest.approx([0.6])


def test_no_spine_model_leaves_the_joint_alone(scene):
    assert scene.spine_model is None
    before = len(scene.robot.set_position_calls)
    scene._apply_control()
    assert len(scene.robot.set_position_calls) == before


def test_view_lifecycle_methods_are_no_ops(scene):
    view = scene.view(ROLE_LEFT)
    view.initialize_simulation()
    view.start()
    view.stop()
    assert scene.running is False
    assert scene.robot is not None


def test_view_delegates_the_arm_contract(scene):
    view = scene.view(ROLE_LEFT)
    torques = np.ones(7)
    view.set_control_mode(ControlMode.TORQUE)
    view.update_torques(torques)

    assert scene.arm_control_modes[ROLE_LEFT] is ControlMode.TORQUE
    assert scene.arm_torques[ROLE_LEFT] == pytest.approx(torques)
    assert view.get_robot_state()["q"].shape == (7,)


def test_base_view_delegates_the_twist(scene):
    scene.view(ROLE_BASE).update_base_twist([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    steer, drive = scene.swerve.solve()
    assert drive == pytest.approx([0.5 / TMR_WHEEL_RADIUS] * 2, abs=1e-9)


def test_arm_view_ignores_a_twist(scene):
    scene.view(ROLE_LEFT).update_base_twist([9.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _, drive = scene.swerve.solve()
    assert drive == pytest.approx([0.0, 0.0], abs=1e-12)


def test_repeated_misrouted_twists_log_once(scene, caplog):
    """update_base_twist can be reached from the ~1 kHz UDP thread on a
    misconfigured bridge, so the misroute warning must latch, not flood.
    """
    view = scene.view(ROLE_LEFT)
    # SceneView (and the logger it warns through) now lives in
    # mobile_duo_common, re-exported into mobile_duo_sim for backward compat.
    with caplog.at_level("WARNING", logger="franka_sim.mobile_duo_common"):
        for _ in range(50):
            view.update_base_twist([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert len(caplog.records) == 1


def test_view_rejects_an_unknown_role(scene):
    with pytest.raises(ValueError):
        scene.view("middle")


def test_view_reports_the_scene_visualisation_flag(scene):
    assert SceneView(scene, ROLE_LEFT).enable_vis is False


# --- resolved-URDF cleanup on a failed build --------------------------------


def _fake_gs(robot_entity=None):
    """A minimal genesis stand-in whose Scene.build() the caller can break.

    ``scene.add_entity`` is called twice by ``initialize_simulation``: once for
    the ground plane, once for the combined URDF robot. When ``robot_entity``
    is given, the second call returns it (so post-build calls -- force range,
    damping, initial pose -- land on an inspectable fake); otherwise both
    calls return a bare ``object()``, matching the previous behavior.
    """
    calls = {"count": 0}

    def add_entity(*args, **kwargs):
        calls["count"] += 1
        if robot_entity is not None and calls["count"] == 2:
            return robot_entity
        return object()

    fake_scene = types.SimpleNamespace(add_entity=add_entity, build=lambda: None, step=lambda: None)
    return (
        types.SimpleNamespace(
            _initialized=True,
            Scene=lambda **kw: fake_scene,
            morphs=types.SimpleNamespace(Plane=lambda: object(), URDF=lambda **kw: object()),
            materials=types.SimpleNamespace(Rigid=lambda **kw: object()),
            options=types.SimpleNamespace(
                ViewerOptions=lambda **kw: object(), SimOptions=lambda **kw: object()
            ),
        ),
        fake_scene,
    )


def test_initialize_simulation_unlinks_the_resolved_urdf_when_build_raises(tmp_path, monkeypatch):
    urdf_path = tmp_path / "duo.urdf"
    urdf_path.write_text('<?xml version="1.0"?><robot name="duo"></robot>')
    resolved = tmp_path / "resolved.urdf"
    resolved.write_text("<robot/>")

    fake_gs, fake_scene = _fake_gs()
    fake_scene.build = lambda: (_ for _ in ()).throw(RuntimeError("boom"))

    duo = MobileDuoScene(urdf_path, enable_vis=False)
    monkeypatch.setattr("franka_sim.mobile_duo_sim.gs", fake_gs)
    monkeypatch.setattr("franka_sim.mobile_duo_sim.resolve_urdf_meshes", lambda *a, **kw: resolved)

    with pytest.raises(RuntimeError, match="boom"):
        duo.initialize_simulation()

    assert not resolved.exists()
    assert duo._resolved_urdf is None


# --- arm joint damping (FR3_JOINT_DAMPING) ----------------------------------
#
# mobile_duo_sim shares franka_genesis_sim.resolve_fr3_joint_damping with the
# single-arm sim, so both the pure-function contract and its wiring into both
# arms of the duo scene are covered here.


def test_resolve_fr3_joint_damping_defaults_when_unset(monkeypatch):
    monkeypatch.delenv("FR3_JOINT_DAMPING", raising=False)
    damping = resolve_fr3_joint_damping()
    assert damping == pytest.approx([DEFAULT_FR3_DAMPING] * 7)


def test_resolve_fr3_joint_damping_scalar_override_broadcasts(monkeypatch):
    monkeypatch.setenv("FR3_JOINT_DAMPING", "1.5")
    damping = resolve_fr3_joint_damping()
    assert damping == pytest.approx([1.5] * 7)


def test_resolve_fr3_joint_damping_seven_value_override_is_per_joint(monkeypatch):
    monkeypatch.setenv("FR3_JOINT_DAMPING", "1,2,3,4,5,6,7")
    damping = resolve_fr3_joint_damping()
    assert damping == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])


def test_resolve_fr3_joint_damping_rejects_wrong_count(monkeypatch):
    monkeypatch.setenv("FR3_JOINT_DAMPING", "1,2,3")
    with pytest.raises(ValueError, match="1 \\(scalar\\) or 7 comma-separated"):
        resolve_fr3_joint_damping()


def test_resolve_fr3_joint_damping_rejects_non_numeric_values(monkeypatch):
    monkeypatch.setenv("FR3_JOINT_DAMPING", "not-a-number")
    with pytest.raises(ValueError):
        resolve_fr3_joint_damping()


def test_resolve_fr3_joint_damping_treats_empty_string_as_unset(monkeypatch):
    """An exported-but-empty $FR3_JOINT_DAMPING (e.g. ``export FOO=`` in a
    sourced script) must not be parsed as zero values; it falls back to the
    default like an unset var, matching the single-arm path's truthiness
    check (``if damping_env:``).
    """
    monkeypatch.setenv("FR3_JOINT_DAMPING", "")
    damping = resolve_fr3_joint_damping()
    assert damping == pytest.approx([DEFAULT_FR3_DAMPING] * 7)


def test_initialize_simulation_applies_default_damping_to_both_arms(tmp_path, monkeypatch):
    """Both arms get DEFAULT_FR3_DAMPING when $FR3_JOINT_DAMPING is unset --
    the pre-fix mobile-duo behavior, now routed through the shared resolver.
    """
    urdf_path = tmp_path / "duo.urdf"
    urdf_path.write_text('<?xml version="1.0"?><robot name="duo"></robot>')
    resolved = tmp_path / "resolved.urdf"
    resolved.write_text("<robot/>")

    duo_entity = FakeDuoEntity()
    fake_gs, _ = _fake_gs(robot_entity=duo_entity)

    duo = MobileDuoScene(urdf_path, enable_vis=False)
    monkeypatch.setattr("franka_sim.mobile_duo_sim.gs", fake_gs)
    monkeypatch.setattr("franka_sim.mobile_duo_sim.resolve_urdf_meshes", lambda *a, **kw: resolved)
    monkeypatch.delenv("FR3_JOINT_DAMPING", raising=False)

    duo.initialize_simulation()

    assert len(duo_entity.damping_calls) == 2
    left_dofs = [duo_entity.joints[name].dof_idx_local for name in ARM_JOINT_NAMES[ROLE_LEFT]]
    right_dofs = [duo_entity.joints[name].dof_idx_local for name in ARM_JOINT_NAMES[ROLE_RIGHT]]
    (left_damping, left_call_dofs), (right_damping, right_call_dofs) = duo_entity.damping_calls
    assert left_call_dofs == left_dofs
    assert right_call_dofs == right_dofs
    assert left_damping == pytest.approx([DEFAULT_FR3_DAMPING] * 7)
    assert right_damping == pytest.approx([DEFAULT_FR3_DAMPING] * 7)


def test_initialize_simulation_applies_env_override_identically_to_both_arms(tmp_path, monkeypatch):
    urdf_path = tmp_path / "duo.urdf"
    urdf_path.write_text('<?xml version="1.0"?><robot name="duo"></robot>')
    resolved = tmp_path / "resolved.urdf"
    resolved.write_text("<robot/>")

    duo_entity = FakeDuoEntity()
    fake_gs, _ = _fake_gs(robot_entity=duo_entity)

    duo = MobileDuoScene(urdf_path, enable_vis=False)
    monkeypatch.setattr("franka_sim.mobile_duo_sim.gs", fake_gs)
    monkeypatch.setattr("franka_sim.mobile_duo_sim.resolve_urdf_meshes", lambda *a, **kw: resolved)
    monkeypatch.setenv("FR3_JOINT_DAMPING", "1.5")

    duo.initialize_simulation()

    assert len(duo_entity.damping_calls) == 2
    for damping, _dofs in duo_entity.damping_calls:
        assert damping == pytest.approx([1.5] * 7)


def test_initialize_simulation_propagates_malformed_damping_env(tmp_path, monkeypatch):
    urdf_path = tmp_path / "duo.urdf"
    urdf_path.write_text('<?xml version="1.0"?><robot name="duo"></robot>')
    resolved = tmp_path / "resolved.urdf"
    resolved.write_text("<robot/>")

    duo_entity = FakeDuoEntity()
    fake_gs, _ = _fake_gs(robot_entity=duo_entity)

    duo = MobileDuoScene(urdf_path, enable_vis=False)
    monkeypatch.setattr("franka_sim.mobile_duo_sim.gs", fake_gs)
    monkeypatch.setattr("franka_sim.mobile_duo_sim.resolve_urdf_meshes", lambda *a, **kw: resolved)
    monkeypatch.setenv("FR3_JOINT_DAMPING", "1,2,3")

    with pytest.raises(ValueError, match="1 \\(scalar\\) or 7 comma-separated"):
        duo.initialize_simulation()

    # Same cleanup-on-raise contract as the resolved-URDF-build-failure case.
    assert not resolved.exists()
    assert duo._resolved_urdf is None


# --- Genesis backend selection (FRANKA_SIM_BACKEND) -------------------------
#
# resolve_gs_backend lives in franka_genesis_sim (next to
# resolve_fr3_joint_damping) but is shared by every gs.init call site --
# franka_genesis_sim.FrankaGenesisSim, tmr_genesis_sim.TMRGenesisSim, and
# MobileDuoScene below -- so the whole process agrees on one backend. Its
# pure-function contract is covered here, alongside the mobile-duo wiring;
# resolve_fr3_joint_damping's tests above use the same colocation.
#
# Each call site passes resolve_gs_backend its *own* bound ``gs``
# (``resolve_gs_backend(gs)``) rather than relying on
# franka_genesis_sim's -- test_mobile_duo_physics.py's real-Genesis fixture
# rebinds only ``mobile_duo_sim.gs`` (conftest.py otherwise stubs ``gs``
# process-wide with a MagicMock), and resolving the backend attribute off
# a *different* module's ``gs`` mixed a stub sentinel into a real
# gs.init() call. The regression test below pins that down.


def _fake_backend_gs():
    """A minimal gs stand-in with distinguishable cpu/gpu sentinel backends."""
    return types.SimpleNamespace(cpu="BACKEND_CPU", gpu="BACKEND_GPU")


def test_resolve_gs_backend_defaults_to_cpu_when_unset(monkeypatch):
    monkeypatch.delenv("FRANKA_SIM_BACKEND", raising=False)
    assert resolve_gs_backend(_fake_backend_gs()) == "BACKEND_CPU"


def test_resolve_gs_backend_treats_empty_string_as_unset(monkeypatch):
    """Mirrors FR3_JOINT_DAMPING's empty-string-is-unset convention."""
    monkeypatch.setenv("FRANKA_SIM_BACKEND", "")
    assert resolve_gs_backend(_fake_backend_gs()) == "BACKEND_CPU"


def test_resolve_gs_backend_gpu_selects_gs_gpu(monkeypatch):
    monkeypatch.setenv("FRANKA_SIM_BACKEND", "gpu")
    assert resolve_gs_backend(_fake_backend_gs()) == "BACKEND_GPU"


def test_resolve_gs_backend_is_case_insensitive_and_strips_whitespace(monkeypatch):
    monkeypatch.setenv("FRANKA_SIM_BACKEND", "  GPU\n")
    assert resolve_gs_backend(_fake_backend_gs()) == "BACKEND_GPU"


def test_resolve_gs_backend_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("FRANKA_SIM_BACKEND", "tpu")
    with pytest.raises(ValueError, match="FRANKA_SIM_BACKEND must be one of"):
        resolve_gs_backend(_fake_backend_gs())


def test_resolve_gs_backend_defaults_to_its_own_modules_gs(monkeypatch):
    """No explicit gs_module: falls back to franka_genesis_sim's own ``gs``,
    resolved at call time (so monkeypatching it still works), for direct
    callers that have no gs of their own to pass.
    """
    monkeypatch.delenv("FRANKA_SIM_BACKEND", raising=False)
    monkeypatch.setattr("franka_sim.franka_genesis_sim.gs", _fake_backend_gs())
    assert resolve_gs_backend() == "BACKEND_CPU"


def test_resolve_gs_backend_uses_the_passed_module_not_franka_genesis_sims(monkeypatch):
    """Regression: resolving off the wrong module's ``gs`` mixes a stub
    sentinel into a real ``gs.init()`` call (see test_mobile_duo_physics.py's
    real-Genesis fixture, which rebinds only one module's ``gs``).
    """
    monkeypatch.setenv("FRANKA_SIM_BACKEND", "gpu")
    # franka_genesis_sim's own gs is left as whatever conftest/collection
    # bound it (a MagicMock or the real package) -- irrelevant here, because
    # an explicit gs_module is passed and must win.
    own_gs = _fake_backend_gs()
    assert resolve_gs_backend(own_gs) is own_gs.gpu


def test_initialize_simulation_initializes_genesis_with_the_resolved_backend(tmp_path, monkeypatch):
    """The scene's gs.init call site routes through resolve_gs_backend(gs),
    same as the single-arm and TMR-only sims (grepped for other gs.init
    call sites when adding backend selection), passing its own gs so the
    resolved backend and the module gs.init() runs on always match.
    """
    urdf_path = tmp_path / "duo.urdf"
    urdf_path.write_text('<?xml version="1.0"?><robot name="duo"></robot>')
    resolved = tmp_path / "resolved.urdf"
    resolved.write_text("<robot/>")

    duo_entity = FakeDuoEntity()
    fake_gs, _ = _fake_gs(robot_entity=duo_entity)
    fake_gs._initialized = False
    fake_gs.cpu = "BACKEND_CPU"
    fake_gs.gpu = "BACKEND_GPU"
    init_calls = []

    def fake_init(**kwargs):
        init_calls.append(kwargs)
        fake_gs._initialized = True

    fake_gs.init = fake_init

    duo = MobileDuoScene(urdf_path, enable_vis=False)
    monkeypatch.setattr("franka_sim.mobile_duo_sim.gs", fake_gs)
    monkeypatch.setattr("franka_sim.mobile_duo_sim.resolve_urdf_meshes", lambda *a, **kw: resolved)
    monkeypatch.setenv("FRANKA_SIM_BACKEND", "gpu")

    duo.initialize_simulation()

    assert init_calls == [{"backend": "BACKEND_GPU", "logging_level": None}]


# --- real-time-factor monitoring (RealtimeFactorMonitor) --------------------
#
# run_simulation's pacing logic (next_step += dt / sleep-slack) is unchanged;
# RealtimeFactorMonitor only observes it. Fed synthetic (now, dt) pairs here
# so the warning/info cadence and threshold logic are exercised without a
# real wall clock or Genesis scene.


def test_rtf_monitor_no_warning_and_one_info_log_when_realtime():
    log = Mock()
    monitor = RealtimeFactorMonitor(log, 0.0, warn_interval_s=5.0, info_interval_s=60.0)

    now = 0.0
    for _ in range(60):
        now += 1.0
        monitor.update(now, dt=1.0)  # sim time keeps pace with wall time: RTF == 1

    log.warning.assert_not_called()
    log.info.assert_called_once()
    message, rtf = log.info.call_args[0]
    assert "real time" in message
    assert rtf == pytest.approx(1.0)


def test_rtf_monitor_warns_once_per_window_when_overloaded():
    log = Mock()
    monitor = RealtimeFactorMonitor(log, 0.0, warn_interval_s=5.0, info_interval_s=60.0)

    now = 0.0
    for _ in range(5):
        now += 1.0
        monitor.update(now, dt=0.5)  # sim advances at half the wall-clock rate: RTF == 0.5

    log.warning.assert_called_once()
    message, rtf = log.warning.call_args[0]
    assert "physics overloaded" in message
    assert rtf == pytest.approx(0.5)


def test_rtf_monitor_no_warning_just_above_the_threshold():
    log = Mock()
    monitor = RealtimeFactorMonitor(
        log, 0.0, warn_interval_s=5.0, info_interval_s=60.0, warn_threshold=0.95
    )

    now = 0.0
    for _ in range(5):
        now += 1.0
        monitor.update(now, dt=0.96)  # RTF 0.96 > the 0.95 threshold

    log.warning.assert_not_called()


def test_rtf_monitor_warns_at_the_threshold_boundary():
    """RTF exactly at the threshold does not warn: the check is strict '<'.

    A single window only (not several accumulated ones): summing ``0.95``
    many times over multiple resets drifts a couple ULPs below the exact
    value in floating point, which would make this a flaky float-precision
    test rather than a check of the '<' boundary itself.
    """
    log = Mock()
    monitor = RealtimeFactorMonitor(
        log, 0.0, warn_interval_s=5.0, info_interval_s=60.0, warn_threshold=0.95
    )

    now = 0.0
    for _ in range(5):
        now += 1.0
        monitor.update(now, dt=0.95)  # RTF exactly 0.95

    log.warning.assert_not_called()


def test_rtf_monitor_info_log_fires_once_per_minute_regardless_of_overload():
    log = Mock()
    monitor = RealtimeFactorMonitor(log, 0.0, warn_interval_s=5.0, info_interval_s=60.0)

    now = 0.0
    for _ in range(60):
        now += 1.0
        monitor.update(now, dt=0.5)  # overloaded for the whole minute

    # One INFO log at the 60s mark, and one WARNING per 5s window (12 of them),
    # never more than that -- i.e. no per-step logging.
    assert log.info.call_count == 1
    assert log.warning.call_count == 12
    _, info_rtf = log.info.call_args[0]
    assert info_rtf == pytest.approx(0.5)


def test_rtf_monitor_logs_nothing_before_the_first_window_elapses():
    log = Mock()
    monitor = RealtimeFactorMonitor(log, 0.0, warn_interval_s=5.0, info_interval_s=60.0)

    now = 0.0
    for _ in range(4):
        now += 1.0
        monitor.update(now, dt=0.1)  # badly overloaded, but only 4s of wall time so far

    log.warning.assert_not_called()
    log.info.assert_not_called()
