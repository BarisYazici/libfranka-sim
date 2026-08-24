"""Physics-level regression tests for the mobile-duo arms.

These build a real Genesis scene from the combined ``mobile_fr3_duo`` URDF --
no bridges, no sockets -- and assert that the arm DOFs actually respond to the
control commands the FCI bridges issue.

They exist because the arms were once completely inert on this scene while
every layer above them looked healthy: the base and the lift are advanced
*kinematically* (``set_pos``/``set_quat``/``set_dofs_position``), and Genesis'
defaults zero the velocity of **every** DOF of the entity on each of those
calls -- once per physics step, that pinned both arms. Nothing in the
fake-entity unit tests could see it, so the check has to run against real
physics.

The scene needs the franka_description meshes, which are not vendored, so the
module skips itself unless they can be found (``$MOBILE_DUO_MESH_ROOT``, else a
``franka_description-jazzy`` checkout next to $HOME).
"""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

import franka_sim.mobile.duo_sim as mobile_duo_sim
from franka_sim.franka_genesis_sim import ControlMode
from franka_sim.mobile.duo_sim import ARM_INITIAL_Q, ROLE_LEFT, MobileDuoScene

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_URDF = REPO_ROOT / "assets" / "mobile_duo" / "mobile_fr3_duo.urdf"

#: dt of the scene; 2400 steps is 6 s of simulated time.
STEPS = 2400

#: Constant torque applied to left joint 2 (Nm). Small on purpose: the bug
#: showed up as "small torques do nothing", and a large torque partly masked it.
TEST_TORQUE_NM = 2.0

#: A moving base and a moving lift are what trigger the kinematic pose writes
#: whose velocity-zeroing pinned the arms, so the regression test drives both.
TEST_TWIST = [0.2, 0.0, 0.0, 0.0, 0.0, 0.1]


def _mesh_root():
    from_env = os.environ.get("MOBILE_DUO_MESH_ROOT")
    if from_env:
        return Path(from_env)
    default = Path.home() / "franka_description-jazzy"
    return default if default.is_dir() else None


def _pop_genesis_modules():
    saved = {}
    for name in list(sys.modules):
        if name == "genesis" or name.startswith("genesis."):
            saved[name] = sys.modules.pop(name)
    return saved


def _restore_genesis_modules(saved):
    for name in list(sys.modules):
        if name == "genesis" or name.startswith("genesis."):
            sys.modules.pop(name)
    sys.modules.update(saved)


def _genesis_available():
    """True only when the *real* Genesis package is installed.

    ``conftest`` stubs ``genesis`` with a MagicMock so the protocol tests can
    import without the native dependency, so the cached module cannot be
    trusted; probe the file system instead (same idiom as
    ``test_fr3_hand_model``).
    """
    saved = _pop_genesis_modules()
    try:
        spec = importlib.util.find_spec("genesis")
        return spec is not None and isinstance(getattr(spec, "origin", None), str)
    except Exception:
        return False
    finally:
        _restore_genesis_modules(saved)


MESH_ROOT = _mesh_root()

pytestmark = [
    pytest.mark.skipif(not _genesis_available(), reason="Genesis is not installed"),
    pytest.mark.skipif(not SCENE_URDF.is_file(), reason=f"missing scene URDF {SCENE_URDF}"),
    pytest.mark.skipif(
        MESH_ROOT is None,
        reason="franka_description meshes not found; set $MOBILE_DUO_MESH_ROOT",
    ),
]


class _StubSpine:
    """Stands in for the spine stub's SpineModel: a fixed lift height."""

    def __init__(self, height_m=0.4):
        self.height_m = height_m

    def position_m(self):
        return self.height_m


@pytest.fixture(scope="module")
def duo_scene():
    """One built Genesis scene shared by this module (building it costs ~15 s).

    ``conftest`` leaves a MagicMock bound as ``mobile_duo_sim.gs``; swap the
    real package in for the duration of this module (rather than reloading
    ``franka_sim``, which would give the tests a second, non-identical
    ``ControlMode`` enum) and put the mock back afterwards.
    """
    saved_modules = _pop_genesis_modules()
    saved_gs = mobile_duo_sim.gs
    try:
        mobile_duo_sim.gs = importlib.import_module("genesis")
    except Exception:  # pragma: no cover - guarded by the module-level skips
        _restore_genesis_modules(saved_modules)
        pytest.skip("the real genesis package could not be imported")

    scene = MobileDuoScene(SCENE_URDF, mesh_root=MESH_ROOT, enable_vis=False)
    try:
        scene.initialize_simulation()
        scene.spine_model = _StubSpine()
        yield scene
    finally:
        scene.stop()
        mobile_duo_sim.gs = saved_gs
        _restore_genesis_modules(saved_modules)


@pytest.fixture
def arm(duo_scene):
    """Reset the left arm to its initial pose and hand back a small helper."""

    class _Arm:
        role = ROLE_LEFT
        dofs = duo_scene.arm_dofs_idx[ROLE_LEFT]

        def q(self):
            return duo_scene.robot.get_dofs_position(self.dofs).cpu().numpy().copy()

        def run(self, steps=STEPS):
            for _ in range(steps):
                duo_scene._apply_control()
                duo_scene.scene.step()

    duo_scene.robot.set_dofs_position(ARM_INITIAL_Q, duo_scene.arm_dofs_idx[ROLE_LEFT])
    duo_scene.update_arm_torques(ROLE_LEFT, np.zeros(7))
    duo_scene.update_arm_joint_positions(ROLE_LEFT, ARM_INITIAL_Q.copy())
    duo_scene.set_arm_control_mode(ROLE_LEFT, ControlMode.POSITION)
    duo_scene.update_base_twist([0.0] * 6)
    return _Arm()


def test_torque_control_actually_moves_an_arm_joint(duo_scene, arm):
    """A constant torque must move the joint, base and lift motion notwithstanding.

    This is the regression guard for the inert-arm bug: with the kinematic base
    and lift writes zeroing every DOF each step, this same command travelled
    ~0.03 rad in 6 s instead of the ~2 rad it does when the arm is free.
    """
    q0 = arm.q()

    duo_scene.set_arm_control_mode(ROLE_LEFT, ControlMode.TORQUE)
    torques = np.zeros(7)
    torques[1] = TEST_TORQUE_NM
    duo_scene.update_arm_torques(ROLE_LEFT, torques)
    duo_scene.update_base_twist(TEST_TWIST)

    arm.run()

    travel = abs(arm.q()[1] - q0[1])
    assert (
        travel > 0.05
    ), f"joint 2 is pinned: travelled only {travel:.5f} rad under {TEST_TORQUE_NM} Nm"


def test_position_control_still_tracks_a_target(duo_scene, arm):
    """Position mode must still converge on its target (no regression)."""
    target = ARM_INITIAL_Q.copy()
    target[1] += 0.3

    duo_scene.set_arm_control_mode(ROLE_LEFT, ControlMode.POSITION)
    duo_scene.update_arm_joint_positions(ROLE_LEFT, target)
    duo_scene.update_base_twist(TEST_TWIST)

    arm.run()

    error = np.abs(arm.q() - target)
    assert error.max() < 0.05, f"position tracking error too large: {np.round(error, 4)}"


def test_the_base_link_lands_where_the_integrated_pose_says(duo_scene, arm):
    """Guards the qpos layout the one-call base-pose write relies on.

    ``SwerveBase`` writes a free root joint's whole pose as a single ``set_qpos``
    over ``[x, y, z, qw, qx, qy, qz]``. Nothing in the fake-entity tests can tell
    that layout from a wrong one -- only real Genesis forward kinematics can, so
    drive the base and compare the base link against the pose the swerve
    integrator thinks it commanded.
    """
    duo_scene.swerve.reset_pose()
    duo_scene.update_base_twist(TEST_TWIST)

    arm.run(steps=400)

    pos = duo_scene.robot.get_pos().cpu().numpy()
    quat = duo_scene.robot.get_quat().cpu().numpy()
    yaw = 2.0 * np.arctan2(quat[3], quat[0])

    assert pos[0] == pytest.approx(duo_scene.swerve.x, abs=1e-4)
    assert pos[1] == pytest.approx(duo_scene.swerve.y, abs=1e-4)
    assert pos[2] == pytest.approx(duo_scene.base_height, abs=1e-4)
    assert yaw == pytest.approx(duo_scene.swerve.theta, abs=1e-4)


def test_the_lift_holds_its_height_against_the_arms(duo_scene, arm):
    """The spine is written once per change, so its PD hold is what pins it.

    Before the hold existed the joint free-ran between periodic re-teleports and
    the carriage (and with it both flanges) jittered under arm reaction forces.
    """
    duo_scene.set_arm_control_mode(ROLE_LEFT, ControlMode.TORQUE)
    torques = np.zeros(7)
    torques[1] = TEST_TORQUE_NM
    duo_scene.update_arm_torques(ROLE_LEFT, torques)
    duo_scene.update_base_twist(TEST_TWIST)

    arm.run(steps=800)

    height = duo_scene.robot.get_dofs_position([duo_scene.spine_dof_idx]).cpu().numpy()[0]
    assert height == pytest.approx(_StubSpine().position_m(), abs=1e-3)


def test_the_other_arm_stays_put_while_one_is_torque_driven(duo_scene, arm):
    """Freeing the arm DOFs must not make the idle arm drift."""
    from franka_sim.mobile.duo_sim import ROLE_RIGHT

    right_dofs = duo_scene.arm_dofs_idx[ROLE_RIGHT]
    duo_scene.robot.set_dofs_position(ARM_INITIAL_Q, right_dofs)
    duo_scene.set_arm_control_mode(ROLE_RIGHT, ControlMode.POSITION)
    duo_scene.update_arm_joint_positions(ROLE_RIGHT, ARM_INITIAL_Q.copy())

    q0_right = duo_scene.robot.get_dofs_position(right_dofs).cpu().numpy().copy()

    duo_scene.set_arm_control_mode(ROLE_LEFT, ControlMode.TORQUE)
    torques = np.zeros(7)
    torques[1] = TEST_TORQUE_NM
    duo_scene.update_arm_torques(ROLE_LEFT, torques)
    duo_scene.update_base_twist(TEST_TWIST)

    arm.run()

    drift = np.abs(duo_scene.robot.get_dofs_position(right_dofs).cpu().numpy() - q0_right)
    assert drift.max() < 0.1, f"idle arm drifted: {np.round(drift, 4)}"
