"""MuJoCo backend for the mobile-duo scene, behind the same adapter contract.

Drop-in alternative to :class:`~franka_sim.mobile.duo_sim.MobileDuoScene`: the
runner, the three FCI bridges and :class:`~franka_sim.mobile.duo_sim.SceneView`
are unchanged, only the physics engine differs. Genesis' per-call kernel-launch
overhead caps the shared scene at ~0.4x real time at dt=2.5 ms; MuJoCo's
zero-copy ``qpos``/``qvel`` views hold real time at dt=1 ms, the rate the FCI
bridges actually serve.

Everything the two backends must agree on -- joint and link names, the initial
arm pose, the actuator limits, the spine travel, the state-snapshot layout and
the real-time-factor monitor -- is imported from
:mod:`franka_sim.mobile.common` (the engine-agnostic constants and
``SceneView``) and ``sim_common`` (the pure-numpy helpers) rather than
restated here, so the two backends cannot drift apart.
Neither of those modules imports ``genesis``, and this module must not either:
it is the ``--physics mujoco`` path, which has to work on a genesis-free
install.
"""

import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Sequence

import mujoco
import numpy as np

from franka_sim.control_modes import ControlMode
from franka_sim.mobile.common import (
    ARM_EE_LINKS,
    ARM_INITIAL_Q,
    ARM_JOINT_NAMES,
    ARM_ROLES,
    ROLE_BASE,
    ROLES,
    SPINE_JOINT_NAME,
    SPINE_LIMITS_M,
    SceneView,
)
from franka_sim.mobile.swerve_base import SwerveBase, yaw_to_quat_wxyz
from franka_sim.mujoco_visuals import (
    apply_dae_material_visuals,
    apply_fr3v2_visuals,
    apply_lift_color_overrides,
)
from franka_sim.sim_common import (
    FR3_FORCE_LIMITS,
    PositionFeedforward,
    RealtimeFactorMonitor,
    close_passive_viewer,
    launch_passive_viewer,
    pose_to_column_major,
    resolve_fr3_joint_damping,
)
from franka_sim.urdf_assets import resolve_urdf_meshes

logger = logging.getLogger(__name__)

#: Physics step (s). The FCI bridges serve 1 kHz, so a 1 ms step makes one
#: physics step per commanded control cycle -- what the real robot does.
DEFAULT_DT = 0.001

#: Joint-space PD gains for POSITION mode (Nm/rad and Nm*s/rad), taken from the
#: MuJoCo Menagerie ``franka_fr3`` position actuators. They reproduce the
#: tracking stiffness Genesis' ``control_dofs_position`` defaults give on the
#: same arm while staying stable at DEFAULT_DT with an explicit servo.
ARM_POSITION_KP = np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0])
ARM_POSITION_KD = np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0])

#: VELOCITY-mode servo gain (Nm per rad/s of velocity error). Same magnitude as
#: the position servo's damping term, so both modes saturate FR3_FORCE_LIMITS at
#: comparable errors.
ARM_VELOCITY_KV = ARM_POSITION_KD.copy()

#: Rotor inertia reflected through the gearbox (kg*m^2), per arm DOF. The ROS
#: URDF carries no ``armature``, which leaves the distal joints light enough for
#: the explicit ARM_POSITION_KP servo to ring at DEFAULT_DT; the Menagerie FR3
#: value damps that without measurably changing the arm's response.
ARM_ARMATURE = 0.1

#: Wheel servo gains. The wheels are report-only (the platform pose is
#: integrated kinematically, see MujocoSwerveBase), so these only need to make
#: ``q``/``dq`` follow the swerve IK targets over the wire.
WHEEL_STEER_KP = 200.0
WHEEL_STEER_KD = 10.0
WHEEL_DRIVE_KV = 20.0

#: Wheel actuator saturation (Nm), so a servo transient cannot inject a large
#: impulse into the rest of the tree.
WHEEL_FORCE_LIMIT = 50.0

#: Rotor inertia (kg*m^2) given to the four TMR wheel DOFs. The URDF's drive
#: wheel is only ~2e-3 kg*m^2 about its axle, which makes WHEEL_DRIVE_KV's
#: explicit servo diverge at DEFAULT_DT (gain*dt/inertia >> 2). A geared wheel
#: motor's reflected rotor inertia is of this order, so this both models the
#: real drive and keeps the servo stable.
WHEEL_ARMATURE = 0.05

#: Settling steps run at build time, matching the Genesis scene's warm-up.
SETTLE_STEPS = 100

#: Geom group the URDF's ``<collision>`` geoms are moved into. MuJoCo's URDF
#: importer leaves them in group 0 next to the ``<visual>`` geoms in group 1,
#: and every MuJoCo viewer renders groups 0-2 by default -- so each link draws
#: its detailed visual mesh *and* its coarse collision hull on top of it, which
#: reads as one tangled clump of interpenetrating meshes rather than two arms.
#: Group 3 is MuJoCo's convention for "collision only, hidden by default".
COLLISION_GEOM_GROUP = 3

#: MuJoCo compiler settings injected into the URDF as a ``<mujoco>`` extension
#: element. ``strippath=false`` keeps the absolute mesh paths that
#: ``resolve_urdf_meshes`` wrote (MuJoCo defaults to basename-only for URDF);
#: ``balanceinertia`` repairs the franka_description links whose inertia tensor
#: violates the triangle inequality; ``fusestatic=false`` keeps every named link
#: as its own body so link7 stays addressable for ``O_T_EE``.
URDF_COMPILER_ATTRS = {
    "strippath": "false",
    "balanceinertia": "true",
    "discardvisual": "false",
    "fusestatic": "false",
}


def patch_urdf_for_mujoco(urdf_path) -> Path:
    """Append the ``<mujoco><compiler .../></mujoco>`` extension to a URDF in place.

    ``urdf_path`` must be a copy the caller owns (the temp file
    ``resolve_urdf_meshes`` writes), never a checked-in asset.
    """
    urdf_path = Path(urdf_path)
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    for existing in root.findall("mujoco"):
        root.remove(existing)
    extension = ET.SubElement(root, "mujoco")
    ET.SubElement(extension, "compiler", URDF_COMPILER_ATTRS)
    tree.write(str(urdf_path), xml_declaration=True)
    return urdf_path


class MujocoSwerveBase(SwerveBase):
    """:class:`SwerveBase` writing MuJoCo ``qpos``/``qfrc_applied`` directly.

    The twist handling, swerve IK and pose integration are inherited unchanged,
    so both backends drive the platform from identical math. Only the writes are
    re-implemented: Genesis' ``set_pos``/``control_dofs_*`` entity calls have no
    MuJoCo equivalent.

    The chassis carries a freejoint purely so the pose *can* be written; it is
    overwritten from the integrated pose every step (and its velocity zeroed),
    exactly as in the Genesis scene, so tyre friction never propels the robot.
    """

    def __init__(self, model, data, base_height: float = 0.0):
        super().__init__(entity=None, base_height=base_height)
        self.model = model
        self.data = data
        self.root_qpos_adr: Optional[int] = None
        self.steer_qpos_adr: Optional[np.ndarray] = None
        self.drive_qpos_adr: Optional[np.ndarray] = None

    def bind(self) -> None:
        """Resolve the wheel and freejoint addresses. Call once, after compile.

        Addresses are kept as ``intp`` arrays, not lists: they index ``qpos`` and
        ``qvel`` several times per physics step and a list costs an array
        conversion on every one of them.
        """
        self.steer_dofs_idx = _address_array(self.model, self.steer_joints, _joint_dof_adr)
        self.drive_dofs_idx = _address_array(self.model, self.drive_joints, _joint_dof_adr)
        self.steer_qpos_adr = _address_array(self.model, self.steer_joints, _joint_qpos_adr)
        self.drive_qpos_adr = _address_array(self.model, self.drive_joints, _joint_qpos_adr)

        root_joint = 0
        if self.model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("Expected the chassis freejoint to be joint 0 of the model")
        self.root_qpos_adr = int(self.model.jnt_qposadr[root_joint])
        self.root_dofs_idx = list(range(int(self.model.jnt_dofadr[root_joint]), 6))

    def apply(self, dt: float) -> None:
        """One physics step: servo the wheels, then teleport the chassis."""
        steer_targets, drive_targets = self.solve()

        steer_q = self.data.qpos[self.steer_qpos_adr]
        steer_dq = self.data.qvel[self.steer_dofs_idx]
        steer_tau = WHEEL_STEER_KP * (steer_targets - steer_q) - WHEEL_STEER_KD * steer_dq
        self.data.qfrc_applied[self.steer_dofs_idx] = np.clip(
            steer_tau, -WHEEL_FORCE_LIMIT, WHEEL_FORCE_LIMIT
        )

        drive_dq = self.data.qvel[self.drive_dofs_idx]
        drive_tau = WHEEL_DRIVE_KV * (drive_targets - drive_dq)
        self.data.qfrc_applied[self.drive_dofs_idx] = np.clip(
            drive_tau, -WHEEL_FORCE_LIMIT, WHEEL_FORCE_LIMIT
        )

        x, y, theta = self.integrate_pose(dt)
        adr = self.root_qpos_adr
        self.data.qpos[adr : adr + 3] = (x, y, self.base_height)
        self.data.qpos[adr + 3 : adr + 7] = yaw_to_quat_wxyz(theta)
        self.data.qvel[:6] = 0.0

    def wheel_state(self):
        """Wheel ``(q, dq)`` as 4-element arrays in ``TMR_JOINT_ORDER``."""
        steer_q = self.data.qpos[self.steer_qpos_adr]
        drive_q = self.data.qpos[self.drive_qpos_adr]
        steer_dq = self.data.qvel[self.steer_dofs_idx]
        drive_dq = self.data.qvel[self.drive_dofs_idx]
        positions = np.array([steer_q[0], drive_q[0], steer_q[1], drive_q[1]])
        velocities = np.array([steer_dq[0], drive_dq[0], steer_dq[1], drive_dq[1]])
        return positions, velocities


def _joint_id(model, name: str) -> int:
    """Resolve a joint name to its id, raising a named error when it is absent."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise KeyError(f"Joint {name!r} not found in the compiled model")
    return joint_id


def _joint_qpos_adr(model, name: str) -> int:
    """``qpos`` index of a scalar (hinge/slide) joint."""
    return int(model.jnt_qposadr[_joint_id(model, name)])


def _joint_dof_adr(model, name: str) -> int:
    """``qvel``/``qfrc`` index of a scalar (hinge/slide) joint."""
    return int(model.jnt_dofadr[_joint_id(model, name)])


def _address_array(model, names: Sequence[str], resolve) -> np.ndarray:
    """Resolve several joint names to an ``intp`` index array."""
    return np.array([resolve(model, name) for name in names], dtype=np.intp)


def log_gl_renderer() -> Optional[str]:
    """Log the OpenGL renderer the viewer will get, and return it.

    The passive viewer draws through GLFW, so a throwaway hidden GLFW window
    reports the same driver the viewer thread will bind. Worth one line in the
    log: a software renderer (llvmpipe/swrast, which a conda-shipped Mesa can
    shadow the system driver with) is slow enough to stall the paced loop, and
    it is otherwise invisible from inside the process. Never fatal -- this is
    diagnostics, not a dependency.
    """
    try:
        import glfw
        from OpenGL import GL

        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(1, 1, "gl-probe", None, None)
        if window is None:
            raise RuntimeError("GLFW could not create a probe window")
        try:
            glfw.make_context_current(window)
            renderer = GL.glGetString(GL.GL_RENDERER).decode()
            vendor = GL.glGetString(GL.GL_VENDOR).decode()
        finally:
            glfw.destroy_window(window)
    except Exception as exc:
        logger.info("Could not determine the OpenGL renderer (%s: %s)", type(exc).__name__, exc)
        return None

    logger.info("Viewer OpenGL renderer: %s (%s)", renderer, vendor)
    if any(marker in renderer.lower() for marker in ("llvmpipe", "softpipe", "swrast")):
        logger.warning(
            "OpenGL is running on a software rasteriser (%s); the viewer will be slow. "
            "Check that no Mesa libGL from the Python environment shadows the system driver.",
            renderer,
        )
    return renderer


class MobileDuoMujocoScene:
    """The single MuJoCo model holding the TMR base, the spine and both arms."""

    def __init__(
        self,
        urdf_path,
        mesh_root=None,
        enable_vis: bool = False,
        dt: float = DEFAULT_DT,
        base_height: float = 0.05,
    ):
        self.urdf_path = Path(urdf_path)
        self.mesh_root = mesh_root
        self.enable_vis = enable_vis
        self.dt = dt
        self.base_height = base_height

        self.model = None
        self.data = None
        self.viewer = None
        #: The viewer's own render thread; see sim_common.close_passive_viewer.
        self._viewer_thread = None
        self.running = False
        self.swerve: Optional[MujocoSwerveBase] = None
        self.arm_qpos_adr: Dict[str, np.ndarray] = {}
        self.arm_dofs_idx: Dict[str, np.ndarray] = {}
        self.arm_body_ids: Dict[str, int] = {}
        self.spine_qpos_adr: Optional[int] = None
        self.spine_dof_idx: Optional[int] = None

        # Optional lift source: any object with ``position_m() -> float``, set by
        # the runner from the spine stub. See MobileDuoScene for the contract.
        self.spine_model = None
        #: Last height the lift was placed at, held when no spine device is
        #: attached. See ``_apply_control``.
        self._spine_position_m = SPINE_LIMITS_M[0]

        self.arm_control_modes = {role: ControlMode.POSITION for role in ARM_ROLES}
        self.arm_torques = {role: np.zeros(7) for role in ARM_ROLES}
        self.arm_joint_positions = {role: ARM_INITIAL_Q.copy() for role in ARM_ROLES}
        self.arm_joint_velocities = {role: np.zeros(7) for role in ARM_ROLES}
        # POSITION mode's velocity feedforward per role: the commanded velocity
        # the damping term servos against, differenced over the number of
        # physics steps the target actually took to change (see
        # PositionFeedforward for why that is not the same as "one step").
        # Seeded to the initial pose so the settle loop's constant target starts
        # at dq_c = 0, and re-seeded by set_arm_control_mode() on every switch
        # into POSITION mode for that role, so a stale baseline never produces a
        # first-step spike. Identical law to the single-arm backend, from the
        # same class -- the two must not drift apart.
        self._arm_position_feedforward = {
            role: PositionFeedforward(ARM_INITIAL_Q) for role in ARM_ROLES
        }

        self._prev_dq = {role: np.zeros(7) for role in ROLES}
        self._ddq_filtered = {role: np.zeros(7) for role in ROLES}
        self._alpha_acc = 0.95
        self._resolved_urdf: Optional[Path] = None

        empty = {
            "q": np.zeros(7),
            "dq": np.zeros(7),
            "ddq": np.zeros(7),
            "q_d": np.zeros(7),
            "dq_d": np.zeros(7),
            "ddq_d": np.zeros(7),
            "tau_J": np.zeros(7),
            "O_T_EE": np.eye(4).T.flatten(),
        }
        self._state_snapshots = {role: dict(empty) for role in ROLES}

    # -- construction ------------------------------------------------------

    def initialize_simulation(self) -> None:
        """Compile the combined URDF and bind every joint group."""
        self._resolved_urdf = resolve_urdf_meshes(self.urdf_path, mesh_root=self.mesh_root)
        # Mirror MobileDuoScene: if anything below raises, stop() is never
        # reached, so the resolved-URDF temp file would leak. Clean it up on the
        # failure path and re-raise.
        try:
            self.model = self._compile_model(patch_urdf_for_mujoco(self._resolved_urdf))
            self.data = mujoco.MjData(self.model)

            self._bind_model()
            self._configure_model()

            for role in ARM_ROLES:
                self.data.qpos[self.arm_qpos_adr[role]] = ARM_INITIAL_Q
            mujoco.mj_forward(self.model, self.data)

            for _ in range(SETTLE_STEPS):
                self._apply_control()
                mujoco.mj_step(self.model, self.data)
            self._read_and_publish_state()
        except Exception:
            Path(self._resolved_urdf).unlink(missing_ok=True)
            self._resolved_urdf = None
            raise

    def _compile_model(self, urdf_path: Path):
        """Compile the patched URDF, giving the chassis a freejoint and a ground.

        The URDF root link is welded to the world by MuJoCo's URDF importer.
        ``add_freejoint`` on it is what makes the kinematic pose write in
        :meth:`MujocoSwerveBase.apply` possible at all, and it lands at joint
        index 0 (``qpos[0:7]``, ``qvel[0:6]``) because it is the first joint of
        the first body.

        Contacts are disabled model-wide. The chassis' URDF collision meshes
        interpenetrate each other by up to 16 cm as authored (they are sized for
        ROS collision *checking*, where neighbouring links are filtered out, not
        for a contact solver), so enabling them makes the solver push the whole
        tree apart and shove both arms off their commanded pose by ~0.8 rad. The
        base pose is integrated kinematically and the arms are servo-driven, so
        no contact in this scene is load-bearing for what the FCI bridges
        report; the teleop stack does its own collision avoidance.

        The spec is also where every link trades its flat merged-COLLADA visual
        for per-material ones; see :meth:`_upgrade_visuals`.

        Gravity compensation is switched on here, before compiling, and not on
        the compiled ``mjModel``: MuJoCo skips the gravcomp pass entirely
        unless ``mjModel.ngravcomp`` is non-zero, and that count is fixed at
        compile time, so assigning ``model.body_gravcomp`` afterwards is
        silently a no-op (see :meth:`_configure_model`). This is the same trap
        and the same fix as the single-arm backend's ``build_model``.
        """
        spec = mujoco.MjSpec.from_file(str(urdf_path))
        spec.option.timestep = self.dt
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

        # The real FR3 reports tau_J with gravity already removed, and the
        # Genesis scene loads the whole entity with gravity_compensation=1.0.
        # MuJoCo's per-body gravcomp is the exact analogue: it cancels weight
        # only, leaving Coriolis and the joint damping intact.
        for body in spec.bodies:
            body.gravcomp = 1.0

        self._upgrade_visuals(spec)

        chassis = spec.worldbody.bodies[0]
        chassis.add_freejoint()

        ground = spec.worldbody.add_geom()
        ground.name = "ground_plane"
        ground.type = mujoco.mjtGeom.mjGEOM_PLANE
        ground.size = [0.0, 0.0, 0.05]
        ground.rgba = [0.55, 0.55, 0.58, 1.0]
        ground.contype = 0
        ground.conaffinity = 0

        light = spec.worldbody.add_light()
        light.directional = True
        light.pos = [0.0, 0.0, 4.0]
        light.dir = [0.0, 0.0, -1.0]

        return spec.compile()

    def _upgrade_visuals(self, spec) -> bool:
        """Repaint the scene: Menagerie arms, per-material chassis and lift.

        Both halves are purely cosmetic and optional by design.
        ``robot_descriptions`` fetches the Menagerie over the network on first
        use, so a fresh or offline host can fail the arm swap; the COLLADA split
        needs the optional ``trimesh`` dependency. Neither may stop the scene
        from building -- the converted-URDF visuals fallen back to are the same
        geometry, just flat grey. See :mod:`franka_sim.mujoco_visuals`.

        The chassis and lift are repainted first: their per-material split maps
        one ``<visual>`` element to one geom positionally, which only holds while
        the spec still looks like the imported URDF. The arm swap runs second and
        replaces the arm geoms wholesale, so ordering matters.
        """
        # Passed the source URDF, not the resolved copy: the split reads the
        # original .dae files, which the copy has already replaced with merged
        # .obj paths.
        repainted = self._apply_visual_upgrade(
            "chassis and lift COLLADA",
            lambda: apply_dae_material_visuals(spec, self.urdf_path, mesh_root=self.mesh_root),
        )
        if repainted:
            # Runs only once the COLLADA split above has actually given
            # franka_spine/mount_link their per-material palette -- without
            # it there is nothing for the colour map to match against.
            self._apply_visual_upgrade(
                "lift Franka-white brightening", lambda: apply_lift_color_overrides(spec)
            )
        swapped = self._apply_visual_upgrade(
            "Menagerie FR3 v2 arm", lambda: apply_fr3v2_visuals(spec)
        )
        return repainted and swapped

    @staticmethod
    def _apply_visual_upgrade(what: str, upgrade) -> bool:
        """Run one visual upgrade, downgrading any failure to a log line."""
        try:
            upgrade()
        except Exception as exc:
            logger.info(
                "Keeping the converted-URDF visuals; the %s visuals are unavailable (%s: %s)",
                what,
                type(exc).__name__,
                exc,
            )
            return False
        return True

    def _bind_model(self) -> None:
        """Resolve joint/body addresses once the model is compiled."""
        self.arm_qpos_adr = {
            role: _address_array(self.model, ARM_JOINT_NAMES[role], _joint_qpos_adr)
            for role in ARM_ROLES
        }
        self.arm_dofs_idx = {
            role: _address_array(self.model, ARM_JOINT_NAMES[role], _joint_dof_adr)
            for role in ARM_ROLES
        }
        self.arm_body_ids = {}
        for role in ARM_ROLES:
            name = ARM_EE_LINKS[role]
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise KeyError(f"Link {name!r} not found in the compiled model")
            self.arm_body_ids[role] = body_id

        self.spine_qpos_adr = _joint_qpos_adr(self.model, SPINE_JOINT_NAME)
        self.spine_dof_idx = _joint_dof_adr(self.model, SPINE_JOINT_NAME)

        self.swerve = MujocoSwerveBase(self.model, self.data, base_height=self.base_height)
        self.swerve.bind()
        self.swerve.reset_pose()

    def _configure_model(self) -> None:
        """Apply the optional joint-damping override, and arm/wheel armature.

        Gravity compensation is not set here: it has to be on before compiling
        (see :meth:`_compile_model`) because ``mjModel.ngravcomp`` is fixed at
        compile time -- assigning ``model.body_gravcomp`` on the compiled model
        is silently a no-op, which is what this scene did before this check
        existed.
        """
        if self.model.ngravcomp == 0:
            raise RuntimeError("Model was compiled without gravity compensation")

        damping = resolve_fr3_joint_damping()
        logger.info("Arm joint damping (left + right): %s", damping)
        for role in ARM_ROLES:
            dofs = self.arm_dofs_idx[role]
            self.model.dof_damping[dofs] = damping
            self.model.dof_armature[dofs] = ARM_ARMATURE

        # The wheel joints are state-report-only on the real TMR (the master
        # closes their loop onboard). The URDF's viscous damping (5/20 Nm*s/rad)
        # and Coulomb friction (1/5 Nm) would only make the proportional servos
        # below under-report the commanded wheel speed and steering angle by a
        # constant offset, so both are dropped and the DOFs get WHEEL_ARMATURE
        # instead. The arms keep their URDF friction: that one is real.
        wheel_dofs = np.concatenate([self.swerve.steer_dofs_idx, self.swerve.drive_dofs_idx])
        self.model.dof_damping[wheel_dofs] = 0.0
        self.model.dof_frictionloss[wheel_dofs] = 0.0
        self.model.dof_armature[wheel_dofs] = WHEEL_ARMATURE

        self._hide_collision_geoms()

    def _hide_collision_geoms(self) -> None:
        """Move the URDF's collision geoms into COLLISION_GEOM_GROUP.

        MuJoCo's URDF importer is the discriminator: it gives ``<visual>`` geoms
        ``contype``/``conaffinity`` 0 and ``<collision>`` geoms 1. The ground
        plane this module adds is non-colliding too (contacts are disabled), so
        it keeps its own group and stays visible.
        """
        collision = (self.model.geom_contype != 0) | (self.model.geom_conaffinity != 0)
        self.model.geom_group[collision] = COLLISION_GEOM_GROUP
        logger.info(
            "Hid %d collision geoms from the viewer (group %d); %d visual geoms remain",
            int(collision.sum()),
            COLLISION_GEOM_GROUP,
            int((~collision).sum()),
        )

    def view(self, role: str) -> SceneView:
        """Return the simulator adapter for one bridge."""
        if role not in ROLES:
            raise ValueError(f"Unknown role {role!r}; expected one of {ROLES}")
        return SceneView(self, role)

    # -- command interface -------------------------------------------------

    def set_arm_control_mode(self, role: str, mode: ControlMode) -> None:
        """Set one arm's control mode (lock-free atomic reference swap)."""
        if not isinstance(mode, ControlMode):
            raise ValueError(f"Mode must be a ControlMode enum, got {type(mode)}")
        logger.info("Arm %s control mode -> %s", role, mode.value)
        self.arm_control_modes[role] = mode
        if mode == ControlMode.POSITION:
            # Discontinuous-target entry point for this role (fresh Move into
            # POSITION mode, or the idle hold, which sets the target to the
            # current q before calling this): snap the feedforward baseline to
            # whatever target is current right now -- and zero the held dq_c
            # with it -- so the next physics step sees dq_c = 0 instead of
            # differencing against, or coasting on, a baseline left over from an
            # older streaming session or another mode.
            self._arm_position_feedforward[role].reset(self.arm_joint_positions[role])

    def update_arm_torques(self, role: str, torques) -> None:
        """Publish one arm's commanded torques."""
        self.arm_torques[role] = np.asarray(torques, dtype=float)

    def update_arm_joint_positions(self, role: str, positions) -> None:
        """Publish one arm's commanded joint positions."""
        self.arm_joint_positions[role] = np.asarray(positions, dtype=float)

    def update_arm_joint_velocities(self, role: str, velocities) -> None:
        """Publish one arm's commanded joint velocities."""
        self.arm_joint_velocities[role] = np.asarray(velocities, dtype=float)

    def update_base_twist(self, twist: Sequence[float]) -> None:
        """Publish the base body-frame twist ``[vx, vy, vz, wx, wy, wz]``."""
        self.swerve.set_twist(twist)

    def set_spine_position(self, position_m: float) -> None:
        """Place the lift carriage, clamped to the URDF's travel limits.

        Kinematic, like the base pose: the real spine is a separate REST device
        with its own closed-loop controller, so the sim only mirrors where it
        says it is. Writing ``qpos``/``qvel`` for that one address touches no
        other DOF, so unlike Genesis' entity-wide ``set_dofs_position`` it
        cannot pin the arms.
        """
        lower, upper = SPINE_LIMITS_M
        self._spine_position_m = min(max(float(position_m), lower), upper)
        self.data.qpos[self.spine_qpos_adr] = self._spine_position_m
        self.data.qvel[self.spine_dof_idx] = 0.0

    def get_role_state(self, role: str) -> Dict[str, np.ndarray]:
        """Latest snapshot for one role (lock-free read)."""
        return self._state_snapshots[role]

    # -- physics loop ------------------------------------------------------

    def arm_control_torque(self, role: str) -> np.ndarray:
        """Actuator torque for one arm this step, clamped to FR3_FORCE_LIMITS.

        Gravity is not part of this: ``body_gravcomp`` already cancels it, so
        TORQUE mode passes the client's command through unchanged -- the same
        contract the real FCI offers.

        POSITION mode's damping term is not plain ``-KD*dq``: undamped, it
        damps against zero velocity, so it fights any commanded motion and
        adds a ``(KD/KP)*qdot`` lag behind the target -- enough on its own to
        fail libfranka's tracking guard (RMSE(q, q_c) < 6e-3 rad) on an
        ordinary point-to-point motion. Instead it damps against the commanded
        velocity ``dq_c``, produced once per physics step per role by
        :class:`~franka_sim.sim_common.PositionFeedforward`: a backward
        difference of the target taken over the number of steps it took to
        change, held across the steps where nothing arrived, and dropped to
        zero once the stream stops. A held target -> dq_c = 0 -> identical to
        the old law. A target that jumps (teleport) produces one step of large
        dq_c; the FR3_FORCE_LIMITS clip below is what bounds that, same as it
        already bounds every other mode.
        """
        mode = self.arm_control_modes[role]
        if mode == ControlMode.TORQUE:
            tau = self.arm_torques[role]
        else:
            qpos_adr = self.arm_qpos_adr[role]
            dofs = self.arm_dofs_idx[role]
            dq = self.data.qvel[dofs]
            if mode == ControlMode.POSITION:
                target = self.arm_joint_positions[role]
                dq_c = self._arm_position_feedforward[role].step(target, self.dt)
                error = target - self.data.qpos[qpos_adr]
                tau = ARM_POSITION_KP * error + ARM_POSITION_KD * (dq_c - dq)
            else:
                tau = ARM_VELOCITY_KV * (self.arm_joint_velocities[role] - dq)
        return np.clip(tau, -FR3_FORCE_LIMITS, FR3_FORCE_LIMITS)

    def _apply_control(self) -> None:
        """Apply one control step for the base, the lift and both arms."""
        # qfrc_applied persists across steps, so every controlled DOF is
        # rewritten each step and the uncontrolled ones (casters, rocker arm)
        # are held at zero.
        self.data.qfrc_applied[:] = 0.0

        self.swerve.apply(self.dt)

        # The lift is placed every step, not only when a spine device is
        # attached: gravity is compensated, so an unwritten prismatic joint is
        # weightless and free, and the carriage (with both arms on it) creeps up
        # the tower on numerical noise -- ~0.15 m over four minutes when run
        # without --spine. Holding the last commanded height matches the real
        # device, which is closed-loop and stays where it was put.
        self.set_spine_position(
            self.spine_model.position_m()
            if self.spine_model is not None
            else self._spine_position_m
        )

        for role in ARM_ROLES:
            self.data.qfrc_applied[self.arm_dofs_idx[role]] = self.arm_control_torque(role)

    def _filtered_acceleration(self, role: str, dq: np.ndarray) -> np.ndarray:
        """Low-passed numerical joint acceleration for one role."""
        raw = (dq - self._prev_dq[role]) / self.dt
        self._ddq_filtered[role] = (
            self._alpha_acc * self._ddq_filtered[role] + (1 - self._alpha_acc) * raw
        )
        self._prev_dq[role] = dq.copy()
        return self._ddq_filtered[role]

    def _read_and_publish_state(self) -> None:
        """Read the model once and publish one snapshot per role."""
        snapshots = {}

        for role in ARM_ROLES:
            # Copies, not views: the snapshot is handed to the bridge threads
            # and must not change under them when the next step runs.
            q = np.array(self.data.qpos[self.arm_qpos_adr[role]])
            dq = np.array(self.data.qvel[self.arm_dofs_idx[role]])
            ddq = self._filtered_acceleration(role, dq)
            body_id = self.arm_body_ids[role]
            snapshots[role] = {
                "q": q,
                "dq": dq,
                "ddq": ddq,
                "q_d": self.arm_joint_positions[role],
                "dq_d": dq,
                "ddq_d": ddq,
                "tau_J": self.arm_torques[role],
                "O_T_EE": pose_to_column_major(self.data.xpos[body_id], self.data.xquat[body_id]),
            }

        wheel_q, wheel_dq = self.swerve.wheel_state()
        q = np.zeros(7)
        dq = np.zeros(7)
        q[:4] = wheel_q
        dq[:4] = wheel_dq
        ddq = self._filtered_acceleration(ROLE_BASE, dq)
        snapshots[ROLE_BASE] = {
            "q": q,
            "dq": dq,
            "ddq": ddq,
            "q_d": q,
            "dq_d": dq,
            "ddq_d": ddq,
            "tau_J": np.zeros(7),
            "O_T_EE": self.swerve.base_pose_matrix(),
        }

        self._state_snapshots = snapshots

    def run_simulation(self) -> None:
        """Physics loop, paced to wall-clock realtime (see MobileDuoScene)."""
        logger.info("Starting mobile-duo MuJoCo simulation loop (dt=%.4fs)", self.dt)

        next_step = time.perf_counter()
        next_render = next_step
        render_period = 1.0 / 30.0
        rtf_monitor = RealtimeFactorMonitor(logger, next_step)

        while self.running:
            self._read_and_publish_state()
            self._apply_control()
            mujoco.mj_step(self.model, self.data)

            # One local reference per iteration: stop() clears self.viewer, and
            # it can run on another thread while this loop is between the None
            # check and the sync.
            viewer = self.viewer
            now = time.perf_counter()
            if viewer is not None and now >= next_render:
                if not viewer.is_running():
                    self.running = False
                    break
                viewer.sync()
                next_render += render_period
                if next_render < now:
                    next_render = now + render_period

            # Pace to realtime: advance one dt of sim-time per dt of wall-time.
            # If a step takes longer than dt, run flat out (no catch-up burst).
            next_step += self.dt
            now = time.perf_counter()
            if next_step > now:
                time.sleep(next_step - now)
            elif now - next_step > self.dt:
                # Fell a full step behind (e.g. a ~12 ms viewer.sync() under
                # load): resync both schedules instead of bursting mj_step to
                # make up the lost wall time. A 1 kHz client reads one
                # RobotState per 1 ms of simulated time; measured under
                # viewer + CPU load, bursting let 58 physics steps land
                # between two published states 16.9 ms apart wall-clock, so
                # the client's PD servo saw a joint delta it read as
                # instantaneous and answered with a torque spike that tripped
                # controller_torque_discontinuity. Dropping the backlog here
                # (RTF < 1, surfaced by the monitor below) keeps that
                # one-state-per-ms invariant instead of hiding the stall.
                next_step = now
                next_render = max(next_render, next_step)

            # Wall-clock RTF measured after pacing: when physics keeps up, the
            # sleep above pads each iteration back to ~dt (RTF ~= 1); an
            # overloaded step skips the sleep and the iteration runs long
            # (RTF < 1). See RealtimeFactorMonitor.
            rtf_monitor.update(time.perf_counter(), self.dt)

    def start(self) -> None:
        """Build (if needed) and run the physics loop in the calling thread."""
        if self.model is None:
            self.initialize_simulation()

        if self.enable_vis and self.viewer is None:
            log_gl_renderer()
            self.viewer, self._viewer_thread = launch_passive_viewer(self.model, self.data)

        self.running = True
        self.run_simulation()

    def stop(self) -> None:
        """Stop the loop, close the viewer and remove the generated URDF copy.

        Idempotent: a second call is a no-op.
        """
        self.running = False
        viewer, self.viewer = self.viewer, None
        thread, self._viewer_thread = self._viewer_thread, None
        if viewer is not None:
            close_passive_viewer(viewer, thread, logger)
        if self._resolved_urdf is not None:
            Path(self._resolved_urdf).unlink(missing_ok=True)
            self._resolved_urdf = None
