"""Physics-level tests for the MuJoCo mobile-duo backend.

These run the real engine on the real combined URDF -- MuJoCo compiles it in
about a second and steps it at ~10x real time, so nothing here is faked. They
are skipped when the generated ``mobile_fr3_duo.urdf`` or the
``franka_description`` mesh checkout it references is not present.
"""

import math
import os
import threading
import time
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from franka_sim.franka_genesis_sim import ControlMode  # noqa: E402
from franka_sim.mobile_duo_mujoco_sim import (  # noqa: E402
    COLLISION_GEOM_GROUP,
    DEFAULT_DT,
    MAX_CATCHUP_LAG_S,
    MobileDuoMujocoScene,
    log_gl_renderer,
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
from franka_sim.mujoco_visuals import (  # noqa: E402
    ASSET_PREFIX,
    FR3V2_VISUAL_LINKS,
    apply_dae_material_visuals,
    resolve_fr3v2_mjcf,
)
from franka_sim.swerve_base import TMR_WHEEL_RADIUS  # noqa: E402
from franka_sim.urdf_assets import resolve_urdf_meshes  # noqa: E402

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

#: Menagerie ``fr3v2.xml``, or None when it is neither cached nor downloadable.
#: The scene falls back to the converted-URDF visuals in that case, so only the
#: tests that assert on the Menagerie visuals themselves are skipped.
try:
    MENAGERIE_MJCF = resolve_fr3v2_mjcf()
except Exception:  # pragma: no cover - depends on the host's cache/network
    MENAGERIE_MJCF = None

requires_menagerie = pytest.mark.skipif(
    MENAGERIE_MJCF is None,
    reason="the MuJoCo Menagerie franka_fr3_v2 model is not available",
)

#: Visual geoms the Menagerie's ``fr3v2_link0..7`` carry, one per material the
#: obj2mjcf split produced. Hard-coded rather than read back from the model so a
#: silently truncated transplant fails the test instead of agreeing with itself.
MENAGERIE_VISUALS_PER_LINK = (6, 1, 1, 1, 1, 1, 10, 7)

#: What each arm link had before the swap: the one merged ``.obj`` that
#: ``resolve_urdf_meshes`` writes per ``<visual>`` element.
URDF_VISUALS_PER_LINK = (1,) * len(FR3V2_VISUAL_LINKS)

#: Visual geoms each non-arm link carries once its COLLADA is split by
#: material: one per material in the ``.dae`` files its ``<visual>`` elements
#: reference. Counted from the sources rather than read back from the model, so
#: a split that silently collapses to one merged geom fails the test.
#: ``mount_link`` has two ``<visual>`` elements (mount + cover), one material
#: each; ``base_link``'s TMR body is the only genuinely multi-material one.
CHASSIS_VISUALS_PER_LINK = {
    "base_link": 5,
    "franka_spine": 3,
    "mount_link": 2,
    "head_link": 1,
}

#: What those links had before the split: one merged ``.obj`` per ``<visual>``.
MERGED_VISUALS_PER_LINK = {"base_link": 1, "franka_spine": 1, "mount_link": 2, "head_link": 1}


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


def test_gravity_compensation_actually_runs(scene):
    """``body_gravcomp`` alone is not enough: MuJoCo only runs the gravcomp
    pass when ``ngravcomp`` is non-zero at compile time (see
    ``_compile_model``/``_configure_model``). This is the regression check for
    the bug where ``_configure_model`` set ``body_gravcomp`` on the already-
    compiled model, which is a silent no-op -- ``ngravcomp`` stayed 0 and
    ``qfrc_gravcomp`` stayed all-zero no matter what.
    """
    assert scene.model.ngravcomp > 0
    mujoco.mj_forward(scene.model, scene.data)
    assert np.any(scene.data.qfrc_gravcomp != 0.0)
    # At qvel == 0 (the held pose the fixture settles to), qfrc_bias is pure
    # gravity (no Coriolis/centrifugal term), so gravcomp -- which cancels
    # weight only -- reproduces it exactly.
    assert scene.data.qvel == pytest.approx(np.zeros(scene.model.nv), abs=1e-6)
    assert scene.data.qfrc_gravcomp == pytest.approx(scene.data.qfrc_bias, abs=1e-9)


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


def test_the_lift_holds_its_height_without_a_spine_device(scene):
    """Gravity is compensated, so an unwritten lift would drift up the tower."""
    assert scene.spine_model is None
    step(scene, seconds=2.0)
    assert scene.data.qpos[scene.spine_qpos_adr] == pytest.approx(SPINE_LIMITS_M[0], abs=1e-9)

    scene.spine_model = FixedSpine(0.5)
    step(scene, seconds=0.05)
    scene.spine_model = None
    step(scene, seconds=2.0)
    # The last commanded height is held, not released.
    assert scene.data.qpos[scene.spine_qpos_adr] == pytest.approx(0.5, abs=1e-9)


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
        # q_d/dq_d/ddq_d are published for shape compatibility only: on an arm
        # role the FCI layer owns those fields and the publish loop drops the
        # backend's copy (see COMMANDED_STATE_FIELDS), so what the snapshot puts
        # in them never reaches a client. Asserting that dq_d *is* dq here used
        # to read as a promise about the wire; it is not one.


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


# -- rendering ------------------------------------------------------------
#
# The viewer renders geom groups 0-2 by default, so what lands in which group
# decides what the user actually sees. These lock that down: the physics is
# read through data.xpos/qpos and the picture through data.geom_xpos, and both
# come off the same forward kinematics, so a wrong picture means a wrong group
# or a geom bound to the wrong body -- not stale data.

DEFAULT_VIEWER_GROUPS = (0, 1, 2)


def geoms_of(built, body_name):
    """Geom ids attached to one body."""
    body_id = mujoco.mj_name2id(built.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return [i for i in range(built.model.ngeom) if built.model.geom_bodyid[i] == body_id]


def visible_geoms_of(built, body_name):
    """Geom ids on one body that the default viewer draws."""
    return [
        i for i in geoms_of(built, body_name) if built.model.geom_group[i] in DEFAULT_VIEWER_GROUPS
    ]


def test_collision_geoms_are_hidden_from_the_default_viewer(scene):
    colliding = (scene.model.geom_contype != 0) | (scene.model.geom_conaffinity != 0)
    assert colliding.any(), "the URDF should still carry collision geoms"
    assert np.all(scene.model.geom_group[colliding] == COLLISION_GEOM_GROUP)
    assert COLLISION_GEOM_GROUP not in DEFAULT_VIEWER_GROUPS

    # ...and the visual geoms are the ones left visible.
    assert np.all(np.isin(scene.model.geom_group[~colliding], DEFAULT_VIEWER_GROUPS))


def test_the_ground_plane_stays_visible(scene):
    ground = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM, "ground_plane")
    assert ground >= 0
    assert scene.model.geom_group[ground] in DEFAULT_VIEWER_GROUPS


def expected_visuals_per_link(built):
    """Per-link visual-geom counts for whichever visual set the scene got."""
    swapped = any(
        built.model.geom(geom).name.startswith("left_fr3v2_link0_visual")
        for geom in geoms_of(built, "left_fr3v2_link0")
    )
    return MENAGERIE_VISUALS_PER_LINK if swapped else URDF_VISUALS_PER_LINK


def arm_visual_geoms(built, prefix):
    """Every visible geom of one arm, flattened across its eight links."""
    return [
        geom
        for index in FR3V2_VISUAL_LINKS
        for geom in visible_geoms_of(built, f"{prefix}_link{index}")
    ]


@pytest.mark.parametrize("prefix", ["left_fr3v2", "right_fr3v2"])
def test_every_arm_link_draws_its_visual_set_exactly_once(scene, prefix):
    """Both arms render as arms: their visual set once per link, no hull on top."""
    expected = expected_visuals_per_link(scene)
    for index in FR3V2_VISUAL_LINKS:
        visible = visible_geoms_of(scene, f"{prefix}_link{index}")
        assert len(visible) == expected[index], f"{prefix}_link{index} draws {len(visible)} geoms"
        for geom in visible:
            assert scene.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH


@requires_menagerie
@pytest.mark.parametrize("prefix", ["left_fr3v2", "right_fr3v2"])
def test_arm_links_wear_the_menagerie_material_palette(scene, prefix):
    """The point of the swap: the arms are painted, not one flat default grey.

    The merged ``.obj`` the URDF conversion produces carries no material at all,
    so every arm geom used to fall back to MuJoCo's default grey ``geom_rgba``.
    """
    geoms = arm_visual_geoms(scene, prefix)
    assert len(geoms) == sum(MENAGERIE_VISUALS_PER_LINK)

    matids = scene.model.geom_matid[geoms]
    assert np.all(matids >= 0), "an arm visual geom has no material"

    colours = {tuple(np.round(scene.model.mat_rgba[matid], 6)) for matid in matids}
    assert len(colours) > 1, f"{prefix} renders in a single colour: {colours}"
    # White body panels and near-black joint rings both appear, so the palette
    # is the FR3's and not a shade of the old uniform grey.
    luminance = sorted(sum(colour[:3]) / 3.0 for colour in colours)
    assert luminance[0] < 0.3 and luminance[-1] > 0.9


@requires_menagerie
def test_both_arms_get_the_same_menagerie_visual_set(scene):
    """The swap is applied per arm prefix; neither may be skipped."""
    left = arm_visual_geoms(scene, "left_fr3v2")
    right = arm_visual_geoms(scene, "right_fr3v2")
    assert len(left) == len(right) == sum(MENAGERIE_VISUALS_PER_LINK)

    def signature(geoms):
        return [
            (
                scene.model.geom(geom).name.split("_", 1)[1],
                int(scene.model.geom_dataid[geom]),
                int(scene.model.geom_matid[geom]),
            )
            for geom in geoms
        ]

    assert signature(left) == signature(right)


@requires_menagerie
def test_menagerie_mesh_assets_are_shared_between_the_two_arms(scene):
    """One copy of each mesh serves both arms; duplicating them would double RAM."""
    geoms = arm_visual_geoms(scene, "left_fr3v2") + arm_visual_geoms(scene, "right_fr3v2")
    meshes = {int(scene.model.geom_dataid[geom]) for geom in geoms}
    assert len(geoms) == 2 * sum(MENAGERIE_VISUALS_PER_LINK)
    assert len(meshes) == sum(MENAGERIE_VISUALS_PER_LINK)

    names = {scene.model.mesh(mesh).name for mesh in meshes}
    assert all(name.startswith(ASSET_PREFIX) for name in names), names


def test_scene_falls_back_to_the_urdf_visuals_without_the_menagerie(monkeypatch, scene):
    """A missing (or un-downloadable) Menagerie must not stop the scene building.

    The fallback is also the reference for "visual only": the two models differ
    in their visual geoms and in nothing the physics reads.
    """

    def unavailable():
        raise RuntimeError("no menagerie for you")

    monkeypatch.setattr(
        "franka_sim.mujoco_visuals.resolve_fr3v2_mjcf",
        unavailable,
    )
    fallback = MobileDuoMujocoScene(SCENE_URDF, mesh_root=MESH_ROOT)
    fallback.initialize_simulation()
    try:
        for index in FR3V2_VISUAL_LINKS:
            for prefix in ("left_fr3v2", "right_fr3v2"):
                visible = visible_geoms_of(fallback, f"{prefix}_link{index}")
                assert len(visible) == URDF_VISUALS_PER_LINK[index]

        assert (fallback.model.nq, fallback.model.nv) == (scene.model.nq, scene.model.nv)
        assert fallback.model.body_mass == pytest.approx(scene.model.body_mass)
        assert fallback.model.body_inertia == pytest.approx(scene.model.body_inertia)
        assert fallback.model.body_ipos == pytest.approx(scene.model.body_ipos)
        assert fallback.model.dof_damping == pytest.approx(scene.model.dof_damping)
        assert fallback.model.dof_armature == pytest.approx(scene.model.dof_armature)
        # Same collision set too: only non-colliding geoms were touched.
        colliding = fallback.model.geom_contype != 0
        assert int(colliding.sum()) == int((scene.model.geom_contype != 0).sum())
    finally:
        fallback.stop()


def test_visual_geoms_are_rigidly_attached_to_their_link_frame(scene):
    """Each visible geom sits at its own body's frame, at every joint angle."""
    q = ARM_INITIAL_Q + np.array([0.3, 0.2, -0.4, 0.3, 0.5, -0.2, 0.1])
    for role in ARM_ROLES:
        scene.data.qpos[scene.arm_qpos_adr[role]] = q
    mujoco.mj_forward(scene.model, scene.data)

    for index in range(8):
        for prefix in ("left_fr3v2", "right_fr3v2"):
            body = f"{prefix}_link{index}"
            body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, body)
            for geom in visible_geoms_of(scene, body):
                expected = (
                    scene.data.xpos[body_id]
                    + scene.data.xmat[body_id].reshape(3, 3) @ scene.model.geom_pos[geom]
                )
                assert scene.data.geom_xpos[geom] == pytest.approx(expected, abs=1e-9), body


def test_arm_visuals_move_with_the_commanded_joint_angles(scene):
    link7 = visible_geoms_of(scene, "left_fr3v2_link7")[0]
    before = np.array(scene.data.geom_xpos[link7])

    target = ARM_INITIAL_Q.copy()
    target[0] += 0.5
    scene.update_arm_joint_positions(ROLE_LEFT, target)
    step(scene)

    after = np.array(scene.data.geom_xpos[link7])
    assert np.linalg.norm(after - before) > 0.1
    # The drawn geom and the reported O_T_EE agree: one forward kinematics.
    reported = np.asarray(scene.get_role_state(ROLE_LEFT)["O_T_EE"]).reshape(4, 4).T[:3, 3]
    assert np.linalg.norm(after - reported) < 0.2


def test_base_and_spine_visuals_follow_the_kinematic_writes(scene):
    """The chassis and lift teleports move the drawn geometry, not just qpos."""
    platform = visible_geoms_of(scene, "base_link")[0]
    carriage = visible_geoms_of(scene, "left_fr3v2_link0")[0]
    platform_before = np.array(scene.data.geom_xpos[platform])
    carriage_before = np.array(scene.data.geom_xpos[carriage])

    scene.spine_model = FixedSpine(0.6)
    scene.update_base_twist([0.15, 0.0, 0.0, 0.0, 0.0, 0.0])
    step(scene)

    platform_after = np.array(scene.data.geom_xpos[platform])
    carriage_after = np.array(scene.data.geom_xpos[carriage])
    # The platform advanced along +x with the integrated pose.
    assert platform_after[0] - platform_before[0] == pytest.approx(0.15, abs=5e-3)
    # The lift carried the arm mount up the tower.
    assert carriage_after[2] - carriage_before[2] == pytest.approx(0.6, abs=1e-2)
    # ...and the arm went along for the ride in x too.
    assert carriage_after[0] - carriage_before[0] == pytest.approx(0.15, abs=5e-3)


# -- chassis and lift colours ---------------------------------------------
#
# The TMR platform and the lift have no Menagerie counterpart, so their colours
# come out of their own COLLADA: one geom per material, each wearing that
# material's diffuse rgba. Before this the merged .obj gave each link a single
# default-grey geom -- the lift read as a featureless dark box.


@pytest.fixture
def flat_scene(monkeypatch):
    """A scene built with the COLLADA split disabled: merged grey visuals.

    This is both the "before" picture and the reference for "visual only": the
    two models must differ in their visual geoms and in nothing else.
    """

    def unavailable(dae_path, cache_dir=None):
        raise RuntimeError("no trimesh for you")

    monkeypatch.setattr("franka_sim.mujoco_visuals.split_dae_by_material", unavailable)
    built = MobileDuoMujocoScene(SCENE_URDF, mesh_root=MESH_ROOT)
    built.initialize_simulation()
    yield built
    built.stop()


def visual_colours_of(built, body_name):
    """Distinct rgba the visible geoms of one body are drawn in."""
    colours = set()
    for geom in visible_geoms_of(built, body_name):
        matid = int(built.model.geom_matid[geom])
        rgba = built.model.mat_rgba[matid] if matid >= 0 else built.model.geom_rgba[geom]
        colours.add(tuple(np.round(rgba, 6)))
    return colours


@pytest.mark.parametrize("link,expected", sorted(CHASSIS_VISUALS_PER_LINK.items()))
def test_each_non_arm_link_draws_one_geom_per_collada_material(scene, link, expected):
    visible = visible_geoms_of(scene, link)
    assert len(visible) == expected, f"{link} draws {len(visible)} geoms"
    for geom in visible:
        assert scene.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
        assert scene.model.geom_matid[geom] >= 0, f"{link} geom {geom} has no material"


def test_the_tmr_platform_wears_its_own_livery(scene):
    """Five materials in the chassis COLLADA, five colours on the platform."""
    colours = visual_colours_of(scene, "base_link")
    assert len(colours) == CHASSIS_VISUALS_PER_LINK["base_link"]

    # The TMR is red-bodied with black wheels/rubber and white panels: both
    # extremes of the luminance range are present, which no shade of the old
    # uniform grey could produce.
    luminance = sorted(sum(colour[:3]) / 3.0 for colour in colours)
    assert luminance[0] < 0.1 and luminance[-1] > 0.9
    assert any(colour[0] - colour[1] > 0.4 and colour[0] - colour[2] > 0.4 for colour in colours)


def test_the_lift_column_is_not_one_flat_colour(scene):
    """The regression this fixes: the spine used to render as dark grey boxes."""
    colours = visual_colours_of(scene, "franka_spine")
    assert len(colours) == CHASSIS_VISUALS_PER_LINK["franka_spine"]
    luminance = sorted(sum(colour[:3]) / 3.0 for colour in colours)
    assert luminance[-1] > 0.9, f"the lift has no light panel: {colours}"


def dominant_visual_material(built, body_name):
    """The rgba of one body's visible geom with the most mesh triangles.

    Triangle count is what :data:`~franka_sim.mujoco_visuals.LINK_COLOR_OVERRIDES`
    was picked by: it is the geom that actually reads as "the surface" of a
    link in a render, as opposed to a thin trim panel that happens to also
    carry a ``<visual>`` element of its own.
    """
    geoms = visible_geoms_of(built, body_name)
    dominant = max(geoms, key=lambda g: built.model.mesh_facenum[built.model.geom_dataid[g]])
    matid = int(built.model.geom_matid[dominant])
    rgba = built.model.mat_rgba[matid] if matid >= 0 else built.model.geom_rgba[dominant]
    return tuple(np.round(rgba, 6))


@pytest.mark.parametrize("link", ["franka_spine", "mount_link"])
def test_the_lift_columns_dominant_material_reads_as_franka_white(scene, link):
    """The user-visible fix: each lift link's biggest submesh is near-white.

    ``mount_link`` is the surprising one: its dominant submesh by triangle
    count (222,326 of 254,792 triangles) is ``fr3_duo_mount.dae``'s own
    mid-grey (0.439), not the small white ``fr3_duo_cover.dae`` sitting on top
    of it -- without the override this assertion fails on ``mount_link`` even
    though ``base_link``'s and the arms' materials are all correct as authored.
    """
    rgba = dominant_visual_material(scene, link)
    luminance = sum(rgba[:3]) / 3.0
    assert luminance > 0.9, f"{link}'s dominant visual material is not white-ish: {rgba}"


def test_the_lift_colour_override_leaves_the_chassis_and_arms_alone(scene):
    """The brightening is scoped to the lift, not a blanket repaint.

    ``base_link`` reuses ``franka_spine``'s exact white MJCF material for its
    rim (both come from the same 0.98 COLLADA colour, deduplicated by the
    palette), so a naive in-place material edit would have brightened the TMR
    chassis too. It must not: the override gives the lift links private
    material copies instead.

    Rounded to 3 decimals rather than reusing ``visual_colours_of``'s 6: the
    rgba values come back as float32, and float32 rounded to 6 decimals does
    not reliably equal the float64 literals below (it does at 3).
    """
    colours = {
        tuple(round(float(channel), 3) for channel in rgba)
        for rgba in (
            scene.model.mat_rgba[int(scene.model.geom_matid[g])]
            for g in visible_geoms_of(scene, "base_link")
        )
    }
    assert colours == {
        (0.0, 0.0, 0.0, 1.0),
        (0.745, 0.196, 0.196, 1.0),
        (0.745, 0.745, 0.745, 1.0),
        (0.937, 0.937, 0.153, 1.0),
        (0.98, 0.98, 0.98, 1.0),
    }


def test_a_single_material_dae_becomes_a_single_coloured_geom(scene):
    """One material is the right answer for the head, not a failed split."""
    (geom,) = visible_geoms_of(scene, "head_link")
    matid = int(scene.model.geom_matid[geom])
    assert matid >= 0
    assert tuple(np.round(scene.model.mat_rgba[matid][:3], 3)) != (0.5, 0.5, 0.5)


def test_the_split_keeps_each_visual_elements_origin():
    """``mount_link``'s cover sits 68 mm up its mount; the split must keep that.

    Asserted on the spec rather than the compiled model because MuJoCo re-centres
    every mesh's vertices at compile time and folds the offset into ``geom_pos``,
    which hides the ``<origin>`` the replacement geoms inherited.
    """
    resolved = resolve_urdf_meshes(SCENE_URDF, mesh_root=MESH_ROOT)
    try:
        spec = mujoco.MjSpec.from_file(str(patch_urdf_for_mujoco(resolved)))
        apply_dae_material_visuals(spec, SCENE_URDF, mesh_root=MESH_ROOT)
        mount = spec.find_body("mount_link")
        offsets = sorted(
            round(float(geom.pos[2]), 4)
            for geom in mount.geoms
            if geom.contype == 0 and geom.conaffinity == 0
        )
        assert offsets == [0.0, 0.068]
    finally:
        Path(resolved).unlink(missing_ok=True)


def test_non_arm_visuals_are_rigidly_attached_to_their_link_frame(scene):
    """Every split geom draws at its own body's frame, not scattered at the origin."""
    scene.spine_model = FixedSpine(0.35)
    scene.update_base_twist([0.1, 0.05, 0.0, 0.0, 0.0, 0.2])
    step(scene, seconds=0.5)

    for link in CHASSIS_VISUALS_PER_LINK:
        body_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, link)
        for geom in visible_geoms_of(scene, link):
            expected = (
                scene.data.xpos[body_id]
                + scene.data.xmat[body_id].reshape(3, 3) @ scene.model.geom_pos[geom]
            )
            assert scene.data.geom_xpos[geom] == pytest.approx(expected, abs=1e-9), link


def test_the_split_geoms_cover_the_merged_mesh_they_replaced(scene, flat_scene):
    """Same bounding box as the merged visual: the node transforms were applied.

    Sub-meshes of a COLLADA Scene are stored in their own local frames, so a
    split that forgets the scene-graph transform scatters them around the link
    origin -- visible here as a bounding box that no longer matches the merged
    mesh's.
    """
    for link in CHASSIS_VISUALS_PER_LINK:
        split = _visual_bounds(scene, link)
        merged = _visual_bounds(flat_scene, link)
        assert split[0] == pytest.approx(merged[0], abs=1e-6), f"{link} lower bound"
        assert split[1] == pytest.approx(merged[1], abs=1e-6), f"{link} upper bound"


def _visual_bounds(built, link):
    """World-frame AABB over the drawn vertices of one link's visible geoms."""
    lower = []
    upper = []
    for geom in visible_geoms_of(built, link):
        mesh = int(built.model.geom_dataid[geom])
        start = int(built.model.mesh_vertadr[mesh])
        vertices = built.model.mesh_vert[start : start + int(built.model.mesh_vertnum[mesh])]
        placed = vertices @ built.data.geom_xmat[geom].reshape(3, 3).T + built.data.geom_xpos[geom]
        lower.append(placed.min(axis=0))
        upper.append(placed.max(axis=0))
    return np.min(lower, axis=0), np.max(upper, axis=0)


def test_the_collada_split_changes_nothing_the_physics_reads(scene, flat_scene):
    """Visual-only, held to the same bar as the Menagerie arm swap."""
    assert (flat_scene.model.nq, flat_scene.model.nv) == (scene.model.nq, scene.model.nv)
    assert flat_scene.model.body_mass == pytest.approx(scene.model.body_mass)
    assert flat_scene.model.body_inertia == pytest.approx(scene.model.body_inertia)
    assert flat_scene.model.body_ipos == pytest.approx(scene.model.body_ipos)
    assert flat_scene.model.body_iquat == pytest.approx(scene.model.body_iquat)
    assert flat_scene.model.dof_damping == pytest.approx(scene.model.dof_damping)
    assert flat_scene.model.dof_armature == pytest.approx(scene.model.dof_armature)
    assert flat_scene.model.dof_frictionloss == pytest.approx(scene.model.dof_frictionloss)
    assert flat_scene.model.jnt_range == pytest.approx(scene.model.jnt_range)

    # The collision set is untouched: same geoms, same sizes, same positions.
    def collision_geoms(built):
        colliding = (built.model.geom_contype != 0) | (built.model.geom_conaffinity != 0)
        return (
            built.model.geom_type[colliding],
            built.model.geom_size[colliding],
            built.model.geom_pos[colliding],
            built.model.geom_bodyid[colliding],
        )

    for split, merged in zip(collision_geoms(scene), collision_geoms(flat_scene)):
        assert split == pytest.approx(merged)


def test_the_scene_still_builds_when_the_collada_cannot_be_split(flat_scene):
    """The COLLADA split is optional; without trimesh the merged visuals stay."""
    for link, expected in MERGED_VISUALS_PER_LINK.items():
        assert len(visible_geoms_of(flat_scene, link)) == expected, link
    # ...and the arms are unaffected, because the two upgrades are independent.
    assert (
        len(visible_geoms_of(flat_scene, "left_fr3v2_link0"))
        == (MENAGERIE_VISUALS_PER_LINK if MENAGERIE_MJCF else URDF_VISUALS_PER_LINK)[0]
    )


def test_the_arms_are_left_to_the_menagerie_swap(scene):
    """The COLLADA split must not fight the arm transplant over the same geoms."""
    expected = expected_visuals_per_link(scene)
    for index in FR3V2_VISUAL_LINKS:
        for prefix in ("left_fr3v2", "right_fr3v2"):
            visible = visible_geoms_of(scene, f"{prefix}_link{index}")
            assert len(visible) == expected[index], f"{prefix}_link{index}"


# -- pacing ---------------------------------------------------------------


class SlowSyncViewer:
    """Stub passive-viewer handle whose sync() blocks like the real one.

    ``mujoco.viewer``'s ``sync()`` waits on the mutex the render thread holds
    while drawing, which measures ~12 ms per call on the reference host.
    """

    def __init__(self, sync_delay_s=0.012):
        self.sync_delay_s = sync_delay_s
        self.syncs = 0

    def is_running(self):
        """The stub viewer window is always open."""
        return True

    def sync(self):
        """Block for one render-thread mutex hold."""
        self.syncs += 1
        time.sleep(self.sync_delay_s)


def test_max_catchup_lag_absorbs_a_whole_render_frame():
    """A 30 FPS frame's worth of stall must be caught up, never discarded."""
    assert MAX_CATCHUP_LAG_S > 1.0 / 30.0


def test_paced_loop_holds_real_time_through_slow_viewer_syncs(scene):
    """Regression: viewer syncs used to reset the deadline and drop sim time."""
    scene.viewer = SlowSyncViewer()
    scene.running = True

    start_time = scene.data.time
    wall_start = time.perf_counter()
    thread = threading.Thread(target=scene.run_simulation, daemon=True)
    thread.start()
    time.sleep(3.0)
    scene.running = False
    thread.join(timeout=5.0)
    wall_elapsed = time.perf_counter() - wall_start
    scene.viewer = None

    assert not thread.is_alive()
    rtf = (scene.data.time - start_time) / wall_elapsed
    assert rtf > 0.95, f"real-time factor {rtf:.2f} with a slow viewer sync"


def test_log_gl_renderer_never_raises():
    """Diagnostics only: a headless or GL-less host must not break startup."""
    result = log_gl_renderer()
    assert result is None or isinstance(result, str)


def test_snapshots_are_copies_the_next_step_cannot_mutate(scene):
    step(scene, seconds=0.05)
    before = scene.get_role_state(ROLE_LEFT)
    q_before = np.array(before["q"])

    scene.update_arm_joint_positions(ROLE_LEFT, ARM_INITIAL_Q + 0.4)
    step(scene, seconds=0.2)

    assert before["q"] == pytest.approx(q_before)
    assert scene.get_role_state(ROLE_LEFT)["q"] != pytest.approx(q_before, abs=1e-3)
