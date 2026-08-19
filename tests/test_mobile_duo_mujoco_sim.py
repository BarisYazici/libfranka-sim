"""Physics-level tests for the MuJoCo mobile-duo backend.

These run the real engine on the real combined URDF -- MuJoCo compiles it in
about a second and steps it at ~10x real time, so nothing here is faked. They
are skipped when the generated ``mobile_fr3_duo.urdf`` or the
``franka_description`` mesh checkout it references is not present.
"""

import math
import os
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from franka_sim.franka_genesis_sim import ControlMode  # noqa: E402
from franka_sim.mobile_duo_mujoco_sim import (  # noqa: E402
    DEFAULT_DT,
    MobileDuoMujocoScene,
    patch_urdf_for_mujoco,
)
from franka_sim.mobile_duo_sim import (  # noqa: E402
    ARM_INITIAL_Q,
    ARM_ROLES,
    FR3_FORCE_LIMITS,
    ROLE_BASE,
    ROLE_LEFT,
    ROLE_RIGHT,
    ROLES,
    SPINE_LIMITS_M,
    SceneView,
)
from franka_sim.swerve_base import TMR_WHEEL_RADIUS  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_URDF = REPO_ROOT / "assets" / "mobile_duo" / "mobile_fr3_duo.urdf"
MESH_ROOT = Path(os.environ.get("MOBILE_DUO_MESH_ROOT", Path.home() / "franka_description-jazzy"))

pytestmark = [
    pytest.mark.skipif(
        not SCENE_URDF.exists(),
        reason=f"generated scene URDF not present at {SCENE_URDF}",
    ),
    pytest.mark.skipif(
        not MESH_ROOT.exists(),
        reason=f"franka_description meshes not present at {MESH_ROOT} "
        "(set $MOBILE_DUO_MESH_ROOT)",
    ),
]

#: Simulated seconds most tests settle for; 1 s at DEFAULT_DT.
SETTLE_S = 1.0


@pytest.fixture
def scene():
    """A freshly built MuJoCo mobile-duo scene, torn down after the test."""
    built = MobileDuoMujocoScene(SCENE_URDF, mesh_root=MESH_ROOT)
    built.initialize_simulation()
    yield built
    built.stop()


def step(built, seconds=SETTLE_S):
    """Run the physics loop body for ``seconds`` of simulated time."""
    for _ in range(int(round(seconds / built.dt))):
        built._read_and_publish_state()
        built._apply_control()
        mujoco.mj_step(built.model, built.data)
    built._read_and_publish_state()


class FixedSpine:
    """Minimal stand-in for the spine stub's SpineModel."""

    def __init__(self, position_m):
        self.value = position_m

    def position_m(self):
        """Commanded carriage height (m)."""
        return self.value


# -- construction ---------------------------------------------------------


def test_patch_urdf_for_mujoco_is_idempotent(tmp_path):
    urdf = tmp_path / "robot.urdf"
    urdf.write_text('<?xml version="1.0"?><robot name="r"></robot>')

    patch_urdf_for_mujoco(urdf)
    patch_urdf_for_mujoco(urdf)

    import xml.etree.ElementTree as ET

    root = ET.parse(str(urdf)).getroot()
    assert len(root.findall("mujoco")) == 1
    assert root.find("mujoco/compiler").get("strippath") == "false"


def test_scene_builds_from_the_real_urdf(scene):
    assert scene.model is not None
    assert scene.dt == DEFAULT_DT
    # Freejoint (7) + every URDF joint. The chassis freejoint must be joint 0,
    # which is what MujocoSwerveBase.apply writes.
    assert scene.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert scene.model.nq == scene.model.nv + 1
    for role in ARM_ROLES:
        assert len(scene.arm_qpos_adr[role]) == 7
        assert len(scene.arm_dofs_idx[role]) == 7


def test_gravity_is_compensated_on_every_body(scene):
    assert np.all(scene.model.body_gravcomp == 1.0)


def test_view_returns_a_scene_view_per_role(scene):
    for role in ROLES:
        view = scene.view(role)
        assert isinstance(view, SceneView)
        assert view.role == role
    with pytest.raises(ValueError):
        scene.view("middle")


# -- arm control ----------------------------------------------------------


def test_arms_hold_the_initial_pose_under_gravity_compensation(scene):
    step(scene)

    for role in ARM_ROLES:
        state = scene.get_role_state(role)
        assert state["q"] == pytest.approx(ARM_INITIAL_Q, abs=1e-3)
        assert np.abs(state["dq"]).max() < 1e-3


def test_position_mode_tracks_a_step_target(scene):
    target = ARM_INITIAL_Q.copy()
    target[0] += 0.3
    target[3] += 0.2
    scene.update_arm_joint_positions(ROLE_LEFT, target)

    step(scene)

    assert scene.get_role_state(ROLE_LEFT)["q"] == pytest.approx(target, abs=5e-3)
    # The other arm is unaffected: the two arms share one model but no DOFs.
    assert scene.get_role_state(ROLE_RIGHT)["q"] == pytest.approx(ARM_INITIAL_Q, abs=1e-3)


def test_velocity_mode_tracks_a_commanded_joint_velocity(scene):
    scene.set_arm_control_mode(ROLE_RIGHT, ControlMode.VELOCITY)
    scene.update_arm_joint_velocities(ROLE_RIGHT, np.full(7, 0.2))

    step(scene, seconds=0.5)

    assert scene.get_role_state(ROLE_RIGHT)["dq"] == pytest.approx(np.full(7, 0.2), abs=0.02)


def test_torque_mode_with_zero_torque_holds_against_gravity(scene):
    scene.set_arm_control_mode(ROLE_LEFT, ControlMode.TORQUE)
    scene.update_arm_torques(ROLE_LEFT, np.zeros(7))

    step(scene)

    # Gravity is compensated, so a zero torque command leaves the arm where it
    # is; only numerical drift remains.
    assert scene.get_role_state(ROLE_LEFT)["q"] == pytest.approx(ARM_INITIAL_Q, abs=1e-2)


def test_commanded_torque_is_clamped_to_the_fr3_actuator_limits(scene):
    scene.set_arm_control_mode(ROLE_LEFT, ControlMode.TORQUE)
    scene.update_arm_torques(ROLE_LEFT, np.full(7, 1000.0))

    assert scene.arm_control_torque(ROLE_LEFT) == pytest.approx(FR3_FORCE_LIMITS)

    scene.update_arm_torques(ROLE_LEFT, np.full(7, -1000.0))
    assert scene.arm_control_torque(ROLE_LEFT) == pytest.approx(-FR3_FORCE_LIMITS)


def test_set_arm_control_mode_rejects_a_non_enum(scene):
    with pytest.raises(ValueError):
        scene.set_arm_control_mode(ROLE_LEFT, "position")


# -- base and spine -------------------------------------------------------


def test_base_twist_integrates_the_planar_pose(scene):
    scene.update_base_twist([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    step(scene)

    assert scene.swerve.x == pytest.approx(0.1, abs=1e-3)
    assert scene.swerve.y == pytest.approx(0.0, abs=1e-6)
    assert scene.swerve.theta == pytest.approx(0.0, abs=1e-6)
    # The freejoint carries the integrated pose, so the whole robot moved.
    adr = scene.swerve.root_qpos_adr
    assert scene.data.qpos[adr : adr + 3] == pytest.approx([0.1, 0.0, scene.base_height], abs=1e-3)


def test_base_drive_wheels_spin_at_the_swerve_ik_speed(scene):
    scene.update_base_twist([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])

    step(scene)

    state = scene.get_role_state(ROLE_BASE)
    expected = 0.1 / TMR_WHEEL_RADIUS
    # TMR_JOINT_ORDER is (front steer, front drive, rear steer, rear drive).
    assert state["dq"][1] == pytest.approx(expected, rel=1e-3)
    assert state["dq"][3] == pytest.approx(expected, rel=1e-3)
    assert state["q"][0] == pytest.approx(0.0, abs=1e-3)
    assert state["q"][2] == pytest.approx(0.0, abs=1e-3)


def test_base_steering_follows_a_lateral_twist(scene):
    scene.update_base_twist([0.0, 0.1, 0.0, 0.0, 0.0, 0.0])

    step(scene)

    state = scene.get_role_state(ROLE_BASE)
    assert state["q"][0] == pytest.approx(math.pi / 2, abs=1e-2)
    assert state["q"][2] == pytest.approx(math.pi / 2, abs=1e-2)
    assert scene.swerve.y == pytest.approx(0.1, abs=1e-3)


def test_spine_follows_the_spine_model_and_clamps_to_its_travel(scene):
    scene.spine_model = FixedSpine(0.4)
    step(scene, seconds=0.05)
    assert scene.data.qpos[scene.spine_qpos_adr] == pytest.approx(0.4, abs=1e-3)

    scene.spine_model = FixedSpine(5.0)
    step(scene, seconds=0.05)
    assert scene.data.qpos[scene.spine_qpos_adr] == pytest.approx(SPINE_LIMITS_M[1])

    scene.spine_model = FixedSpine(-1.0)
    step(scene, seconds=0.05)
    assert scene.data.qpos[scene.spine_qpos_adr] == pytest.approx(SPINE_LIMITS_M[0])


def test_moving_the_spine_does_not_pin_the_arms(scene):
    """The lift teleport must not zero the arm velocities (the Genesis trap)."""
    scene.spine_model = FixedSpine(0.3)
    target = ARM_INITIAL_Q.copy()
    target[0] += 0.5
    scene.update_arm_joint_positions(ROLE_LEFT, target)

    step(scene, seconds=0.1)

    assert np.abs(scene.get_role_state(ROLE_LEFT)["dq"]).max() > 1e-3


# -- state snapshots ------------------------------------------------------


def test_every_role_snapshot_has_the_expected_keys_and_shapes(scene):
    step(scene, seconds=0.05)

    for role in ROLES:
        state = scene.get_role_state(role)
        assert set(state) == {
            "q",
            "dq",
            "ddq",
            "q_d",
            "dq_d",
            "ddq_d",
            "tau_J",
            "O_T_EE",
        }
        for key in ("q", "dq", "ddq", "q_d", "dq_d", "ddq_d", "tau_J"):
            assert np.asarray(state[key]).shape == (7,), f"{role}/{key}"
        assert np.asarray(state["O_T_EE"]).shape == (16,)
        # dq_d/ddq_d echo the measured values, as the Genesis scene does.
        assert state["dq_d"] is state["dq"]
        assert state["ddq_d"] is state["ddq"]


def test_base_snapshot_pads_the_four_wheel_joints_into_seven(scene):
    scene.update_base_twist([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    step(scene, seconds=0.2)

    state = scene.get_role_state(ROLE_BASE)
    assert state["q"][4:] == pytest.approx([0.0, 0.0, 0.0])
    assert state["dq"][4:] == pytest.approx([0.0, 0.0, 0.0])
    assert state["tau_J"] == pytest.approx(np.zeros(7))


@pytest.mark.parametrize("role", ROLES)
def test_o_t_ee_is_a_valid_rigid_transform(scene, role):
    step(scene, seconds=0.05)

    matrix = np.asarray(scene.get_role_state(role)["O_T_EE"]).reshape(4, 4).T
    rotation = matrix[:3, :3]
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-9)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-9)
    assert matrix[3] == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_arm_ee_poses_are_above_the_platform_and_mirrored(scene):
    step(scene, seconds=0.05)

    left = np.asarray(scene.get_role_state(ROLE_LEFT)["O_T_EE"]).reshape(4, 4).T[:3, 3]
    right = np.asarray(scene.get_role_state(ROLE_RIGHT)["O_T_EE"]).reshape(4, 4).T[:3, 3]
    assert left[2] > scene.base_height
    assert left[1] > 0.0 > right[1]
    assert left[0] == pytest.approx(right[0], abs=1e-6)
    assert left[2] == pytest.approx(right[2], abs=1e-6)


def test_snapshots_are_copies_the_next_step_cannot_mutate(scene):
    step(scene, seconds=0.05)
    before = scene.get_role_state(ROLE_LEFT)
    q_before = np.array(before["q"])

    scene.update_arm_joint_positions(ROLE_LEFT, ARM_INITIAL_Q + 0.4)
    step(scene, seconds=0.2)

    assert before["q"] == pytest.approx(q_before)
    assert scene.get_role_state(ROLE_LEFT)["q"] != pytest.approx(q_before, abs=1e-3)
