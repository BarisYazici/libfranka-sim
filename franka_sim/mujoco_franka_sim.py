"""MuJoCo backend for the single-arm FCI simulator (the default physics engine).

Drop-in replacement for :class:`~franka_sim.franka_genesis_sim.FrankaGenesisSim`:
``FrankaSimServer`` and :class:`~franka_sim.gripper.physics.FrankaHandPhysics`
call exactly the same methods on it, so nothing above the simulator changes when
the engine does. Only the physics differs -- Genesis' per-call kernel-launch
overhead forced the arm to a 2.5 ms step to hold real time, while MuJoCo's
zero-copy ``qpos``/``qvel`` views hold it at the 1 ms step the FCI actually
serves, so one physics step happens per commanded control cycle.

The model is the MuJoCo Menagerie ``franka_fr3_v2`` arm, unmodified: its
kinematics, inertias, joint damping, armature and Coulomb friction are the ones
the Menagerie calibrated against the real FR3. With ``enable_hand`` the
Menagerie Franka Hand is grafted onto the flange through MjSpec attach at the
same transform :mod:`franka_sim.assets.build_fr3_with_hand` uses for the Genesis
model, so the two backends agree on where the fingers are.
"""

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from franka_sim.cartesian_ik import (
    CartesianFeedforward,
    elbow_null_velocity,
    resolved_rate,
    tracking_twist,
)
from franka_sim.control_modes import ControlMode
from franka_sim.sim_common import (
    FR3_FORCE_LIMITS,
    PositionFeedforward,
    RealtimeFactorMonitor,
    SelfCollisionContact,
    close_passive_viewer,
    launch_passive_viewer,
    resolve_fr3_joint_damping,
)

logger = logging.getLogger(__name__)

#: Physics step (s). The FCI serves 1 kHz, so a 1 ms step makes one physics step
#: per commanded control cycle -- what the real robot does.
DEFAULT_DT = 0.001

#: Joint-space PD gains for POSITION mode (Nm/rad and Nm*s/rad), taken from the
#: Menagerie ``fr3v2`` position actuators (``kp``/``kv``). Driving them as an
#: explicit servo on ``qfrc_applied`` rather than through those actuators keeps
#: all three control modes on one code path: TORQUE mode must be able to hand
#: the client's command straight to the joint, which a position actuator
#: sitting on the same DOF would fight.
ARM_POSITION_KP = np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0])
ARM_POSITION_KD = np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0])

#: VELOCITY-mode servo gain (Nm per rad/s of velocity error). Same magnitude as
#: the position servo's damping term, so both modes saturate FR3_FORCE_LIMITS at
#: comparable errors.
ARM_VELOCITY_KV = ARM_POSITION_KD.copy()

#: Per-finger PD gains (N/m and N*s/m) and force cap (N). The finger slide DOF
#: carries 0.015 kg of finger plus the hand model's 0.1 armature, so these are
#: ~critically damped (2*sqrt(kp*m) ~= 21) and settle a full 40 mm stroke in
#: ~50 ms without overshoot. The cap matches the Franka Hand's ~70 N grip
#: rounded to the Genesis model's finger force range.
FINGER_KP = 1000.0
FINGER_KD = 20.0
FINGER_FORCE_LIMIT = 100.0

#: Travel of one finger (m); the hand's total opening is twice this.
MAX_FINGER_TRAVEL = 0.04

#: Initial arm pose, identical to the Genesis single-arm sim's.
ARM_INITIAL_Q = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.785])

#: Flange -> hand rotation of the real Franka wrist: -45 deg about the flange z.
#: The Menagerie ``hand.xml`` body already carries 180 deg about z of its own, so
#: the frame the hand is attached through must supply the remaining +135 deg.
#: Verified to reproduce, bit for bit, the hand pose
#: :mod:`franka_sim.assets.build_fr3_with_hand` grafts into the Genesis model.
FLANGE_TO_HAND_QUAT = (0.3826834, 0.0, 0.0, 0.9238795)

#: Flange offset (m) along link7's z, used only when the arm model has no
#: ``link8`` body to attach the hand to (the Menagerie ``franka_fr3`` v1 model
#: marks the same point with an ``attachment_site`` instead).
FLANGE_OFFSET_Z = 0.107

#: Settling steps run at build time, matching the Genesis sim's warm-up.
SETTLE_STEPS = 100

#: How close two monitored arm links may come (m) before the sim calls it a
#: self-collision and the FCI layer raises ``self_collision_avoidance_violation``.
#:
#: **Not a libfranka constant**, like every other tolerance in
#: :mod:`franka_sim.limits.tables`: Control's self-collision model is not
#: published. What *is* known about it is its shape -- the robot does not test
#: the visual meshes but a set of simplified volumes inflated around each link,
#: the same simplification ``franka_description`` ships as its collision meshes
#: -- so the reflex fires with an offset still standing, before the links
#: physically touch. This margin is that inflation, applied here as MuJoCo's own
#: contact ``margin`` rather than by fattening geometry (see
#: :meth:`MujocoFrankaSim._bind_self_collision`).
#:
#: 50 mm, placed between the two things it has to separate on the sim's own FR3
#: model (``robot_descriptions`` ``fr3_v2``), both measured over the *monitored*
#: pairs only (:data:`SELF_COLLISION_MIN_LINK_SEPARATION`):
#:
#: * **The provocation.** Folding joint 4 at 0.1 rad/s from
#:   ``kInitPoseSelfCollision`` while twisting joint 5 at 0.2 rad/s brings link5
#:   (the forearm) down onto link1 (the shoulder). Its closest approach is
#:   24 mm -- the convex hulls never touch at all -- so a contact-based detector
#:   would wait for ever and any margin below ~25 mm never fires. Run closed
#:   loop through this backend's velocity servo, 50 mm is reached at t = 10.80 s.
#: * **The other error that scenario can raise.** The fold ends with joint 4
#:   parked against its position limit, where the position-based velocity
#:   envelope collapses towards zero and the commanded -0.1 rad/s becomes
#:   ``joint_motion_generator_velocity_limits_violation``. Measured on the same
#:   closed-loop run, that happens at t = 11.63 s -- 833 control cycles after
#:   this margin fires, which is the lead the geometric error needs to be the
#:   one the client sees.
#:
#: The other bound is ordinary operation, and it is not tight: across the home
#: pose, ``kInitPoseSelfCollision``, ``kSingularPose`` and ``kIkBugPose`` no
#: monitored pair comes closer than 136 mm, and three of those four have no
#: monitored pair within 200 mm at all. So this sits ~2.7x inside the tightest
#: ordinary clearance and ~2x outside the provocation's closest approach.
SELF_COLLISION_MARGIN = 0.05

#: How far apart in the kinematic chain two links must be for the pair to be
#: monitored. Three, i.e. link ``i`` is checked against ``i+3`` and beyond.
#:
#: Adjacent links (``i``, ``i+1``) are the obvious exclusion -- they touch by
#: construction, and on this model MuJoCo does not even filter link0/link1 for
#: us (link0 carries no joint, so it welds to the world and the parent-child
#: filter, which skips pairs involving the world weld, lets it through); that
#: pair is taken out of the contact model altogether in
#: :func:`_exclude_base_shoulder_contact`, because its hulls overlap. The
#: once-removed pairs are excluded on measurement rather than principle:
#: link5/link7 sit **10-22 mm** apart in every configuration -- their relative
#: pose depends only on joints 6 and 7 -- which is *inside* this margin and
#: closer than the provocation ever gets. They are held apart by the wrist's
#: joint limits, not by a reflex, and monitoring them would report a
#: self-collision on a freshly built arm. link2/link4 and link3/link5 behave the
#: same way at ~70 mm. Two links apart is still mechanically a neighbour; three
#: is the arm reaching back on itself.
SELF_COLLISION_MIN_LINK_SEPARATION = 3

#: Which links carry a monitored collision volume: the arm's own, ``link0``
#: through ``link7``, and nothing else.
#:
#: **The grafted hand is deliberately out.** It is an end effector, not an arm
#: link, and on this model it sits 26-68 mm off link5 in ordinary poses (37 mm
#: at ``kSingularPose``) -- inside :data:`SELF_COLLISION_MARGIN`, so including
#: it would report a self-collision on a freshly built ``--gripper-physics``
#: arm and make the reflex differ between the two builds. It is also what the
#: reference provocation legislates for: it twists joint 5 specifically to get
#: the gripper "out of the way ... so self-collision between links can be
#: detected".
SELF_COLLISION_LINKS = tuple(range(8))


def default_fr3_mjcf() -> Path:
    """Resolve the FR3 MJCF physics model (MuJoCo Menagerie ``franka_fr3_v2``).

    The real robot is an FR3 (not a Panda); this model carries the FR3
    kinematics/inertia and built-in joint damping, armature and Coulomb
    friction. Resolution order: the ``$FR3_MJCF`` override, otherwise
    ``robot_descriptions``, which downloads and caches the model on first use --
    so a ``pip install`` of this package works out of the box (no vendored
    ~57 MB of meshes, just a small dependency).
    """
    override = os.environ.get("FR3_MJCF")
    if override:
        return Path(override)
    try:
        from robot_descriptions import fr3_v2_mj_description

        return Path(fr3_v2_mj_description.MJCF_PATH)
    except Exception as exc:  # offline / proxy / fetch failure on first use
        raise RuntimeError(
            f"Could not obtain the FR3 model via robot_descriptions ({type(exc).__name__}: "
            f"{exc}). It is downloaded and cached on first use, so the first run needs "
            "network access; set $FR3_MJCF to a local MJCF path to run fully offline."
        ) from exc


def default_hand_mjcf() -> Path:
    """Resolve the Franka Hand MJCF (Menagerie ``franka_emika_panda/hand.xml``).

    The Menagerie ships the hand as a standalone model next to the Panda, which
    is what makes attaching it to the FR3 a two-line MjSpec graft instead of the
    XML surgery the Genesis model needs. Overridable via ``$FRANKA_HAND_MJCF``
    for offline runs, mirroring ``$FR3_MJCF``.
    """
    override = os.environ.get("FRANKA_HAND_MJCF")
    if override:
        return Path(override)
    try:
        from robot_descriptions import panda_mj_description

        return Path(panda_mj_description.PACKAGE_PATH) / "hand.xml"
    except Exception as exc:  # offline / proxy / fetch failure on first use
        raise RuntimeError(
            f"Could not obtain the Franka Hand model via robot_descriptions "
            f"({type(exc).__name__}: {exc}); set $FRANKA_HAND_MJCF to a local MJCF path "
            "to run fully offline."
        ) from exc


def arm_prefix(spec) -> str:
    """Infer the arm's joint/body name prefix from an MjSpec's first joint.

    The Menagerie names everything ``fr3v2_*``; the older ``franka_fr3`` model
    (reachable through ``$FR3_MJCF``) uses ``fr3_*``. Deriving the prefix once
    means the joint names, the flange body and the grafted hand all agree
    whichever model was loaded, instead of hard-coding one family.
    """
    for joint in spec.joints:
        if joint.name.endswith("joint1"):
            return joint.name[: -len("joint1")]
    raise RuntimeError("Arm MJCF has no '*joint1'; cannot infer its joint-name prefix")


def build_model(enable_hand: bool = False, dt: float = DEFAULT_DT, model_path=None):
    """Compile the arm model (optionally with the hand) and return (model, prefix).

    Actuation is disabled model-wide: every control mode drives ``qfrc_applied``
    directly, and the model's own position actuators would otherwise pull each
    joint towards ``ctrl`` (0 rad) on top of it. Contacts stay enabled -- unlike
    the mobile-duo URDF, the Menagerie meshes are authored for a contact solver,
    and grasping needs them.

    Gravity compensation is switched on here, before compiling, and not on the
    compiled ``mjModel``: MuJoCo skips the gravcomp pass entirely unless
    ``mjModel.ngravcomp`` is non-zero, and that count is fixed at compile time,
    so assigning ``model.body_gravcomp`` afterwards is silently a no-op.
    """
    spec = mujoco.MjSpec.from_file(str(model_path or default_fr3_mjcf()))
    prefix = arm_prefix(spec)

    # Held until compile(): the attached child spec must outlive the graft.
    hand_spec = None
    if enable_hand:
        hand_spec = mujoco.MjSpec.from_file(str(default_hand_mjcf()))
        _attach_hand(spec, hand_spec, prefix)

    spec.option.timestep = dt
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION

    _exclude_base_shoulder_contact(spec, prefix)

    # The real FR3 reports tau_J with gravity already removed, and the Genesis
    # sim loads the arm with gravity_compensation=1.0. MuJoCo's per-body
    # gravcomp is the exact analogue: it cancels weight only, leaving Coriolis,
    # the joint damping and the Coulomb friction intact. Set after the graft so
    # the hand and fingers are covered too.
    for body in spec.bodies:
        body.gravcomp = 1.0

    ground = spec.worldbody.add_geom()
    ground.name = "ground_plane"
    ground.type = mujoco.mjtGeom.mjGEOM_PLANE
    ground.size = [0.0, 0.0, 0.05]
    ground.rgba = [0.55, 0.55, 0.58, 1.0]

    light = spec.worldbody.add_light()
    light.directional = True
    light.pos = [0.0, 0.0, 4.0]
    light.dir = [0.0, 0.0, -1.0]

    model = spec.compile()
    del hand_spec
    return model, prefix


def _exclude_base_shoulder_contact(spec, prefix: str) -> None:
    """Turn off collision between ``link0`` (the base) and ``link1`` (the shoulder).

    They are adjacent links joined by joint 1, and adjacent links never collide
    on the real robot: Control's self-collision model does not test them, and
    neither does MuJoCo for any *other* neighbouring pair -- its parent-child
    filter drops those automatically. link0/link1 is the one pair that filter
    lets through: link0 carries no joint, so it is welded to the world, and the
    filter deliberately keeps world-vs-child contacts (a body must be able to
    rest on the floor it is attached to). So on the unmodified Menagerie
    ``fr3v2`` model the two convex collision hulls *are* collided, and they sit
    0.1-1.2 mm apart, interpenetrating by ~0.1 mm over a narrow band of joint-1
    angles (measured with libccd: ``dist = -0.096 mm`` at ``q1 = 0.226 rad``,
    negative over 0.2144-0.2272 rad).

    That overlap is a physical brake the real arm does not have. Passing
    through it at speed costs one step of ~75 Nm of ``qfrc_constraint`` on
    joint 1; arriving in it slowly, under a torque controller, is worse -- the
    contact's dry friction (mesh ``friction`` 1.0) holds joint 1 against
    everything a compliant controller can bring: measured, 5 Nm of commanded
    torque on joint 1 answered by -4.99 Nm of constraint torque and the joint
    parked at ``q1 = 0.223``. franky's Cartesian-impedance null-space posture
    task (20 Nm/rad) stalled there with its target 0.27 rad away, which is what
    this was found from. The position servo (4.5 kNm/rad) never noticed.

    A pair exclusion is the exact MuJoCo counterpart of what the parent-child
    filter does for every other neighbour -- and it is what Menagerie itself
    does: both ``franka_fr3/fr3.xml`` and ``franka_emika_panda/panda.xml`` ship
    a ``<contact><exclude body1="link0" body2="link1"/></contact>``. Only the
    ``fr3v2`` model this sim loads is missing it, so this adds it back. It
    changes nothing else: the self-collision reflex only monitors links three
    or more apart (:data:`SELF_COLLISION_MIN_LINK_SEPARATION`), and the pair
    was already skipped there by name. Skipped silently if either body is
    missing, so a caller pointing ``$FR3_MJCF`` at a model without them still
    compiles.
    """
    first, second = f"{prefix}link0", f"{prefix}link1"
    if spec.find_body(first) is None or spec.find_body(second) is None:
        return
    exclude = spec.add_exclude()
    exclude.name = f"{first}_{second}"
    exclude.bodyname1 = first
    exclude.bodyname2 = second


def _attach_hand(spec, hand_spec, prefix: str) -> None:
    """Graft the Franka Hand onto the arm flange, prefixing its names like the arm's.

    The hand goes on ``<prefix>link8`` (the flange body of the ``fr3v2`` model)
    when it exists, otherwise on ``<prefix>link7`` shifted by FLANGE_OFFSET_Z --
    the same point, which is where the older model puts its ``attachment_site``.
    Names come out as ``<prefix>hand`` / ``<prefix>finger_joint1|2``, so they can
    never collide with the arm's.
    """
    flange = spec.find_body(f"{prefix}link8")
    offset = [0.0, 0.0, 0.0]
    if flange is None:
        flange = spec.find_body(f"{prefix}link7")
        offset = [0.0, 0.0, FLANGE_OFFSET_Z]
    if flange is None:
        raise RuntimeError(f"Arm MJCF has neither a {prefix}link8 nor a {prefix}link7 body")

    frame = flange.add_frame()
    frame.pos = offset
    frame.quat = list(FLANGE_TO_HAND_QUAT)
    frame.attach_body(hand_spec.worldbody.first_body(), prefix, "")


def _joint_id(model, name: str) -> int:
    """Resolve a joint name to its id, raising a named error when it is absent."""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise KeyError(f"Joint {name!r} not found in the compiled model")
    return joint_id


def _qpos_addresses(model, names) -> np.ndarray:
    """``qpos`` indices of several scalar (hinge/slide) joints, as an intp array."""
    return np.array([model.jnt_qposadr[_joint_id(model, n)] for n in names], dtype=np.intp)


def _dof_addresses(model, names) -> np.ndarray:
    """``qvel``/``qfrc`` indices of several scalar joints, as an intp array."""
    return np.array([model.jnt_dofadr[_joint_id(model, n)] for n in names], dtype=np.intp)


class MujocoFrankaSim:
    """The MuJoCo FR3 (optionally FR3 + Franka Hand) behind the sim contract.

    Threading mirrors :class:`FrankaGenesisSim` exactly, because the server uses
    it the same way: the physics loop in :meth:`run_simulation` is the only
    writer of ``qpos``/``qvel`` and the only producer of the state snapshot,
    while the TCP/UDP threads only ever rebind a whole command array or read the
    current snapshot reference. Both are single bytecode operations, atomic
    under the GIL, so no mutex is needed -- provided no published array is ever
    mutated in place.
    """

    def __init__(self, enable_vis: bool = False, model_path=None, enable_hand: bool = False):
        self.enable_vis = enable_vis
        self.model_path = model_path
        self.enable_hand = enable_hand
        self.dt = DEFAULT_DT

        self.model = None
        self.data = None
        self.viewer = None
        #: The viewer's own render thread, so stop() can wait for it to finish
        #: its GL teardown; see sim_common.close_passive_viewer.
        self._viewer_thread = None
        self.running = False
        self.prefix = ""

        self.arm_qpos_adr: Optional[np.ndarray] = None
        self.arm_dofs_idx: Optional[np.ndarray] = None
        self.finger_qpos_adr: Optional[np.ndarray] = None
        self.finger_dofs_idx: Optional[np.ndarray] = None
        self.ee_body_id: Optional[int] = None

        # Self-collision monitoring, resolved once at build time (see
        # _bind_self_collision): ``{(lo_geom_id, hi_geom_id): ("link1", "link5")}``
        # for every pair worth watching. Empty when the model carries no
        # ``*_collision`` geoms at all, which switches the whole check off
        # rather than guessing.
        self._self_collision_pairs = {}

        # Flange -> EE transform (row-major 4x4), i.e. the FCI's ``F_T_EE``.
        # Identity until a client sends a SetEE-family command; see
        # update_ee_transform(). Only the translation is read today (the EE's
        # *speed* does not depend on how the EE frame is rotated), but the whole
        # transform is kept so a rotational or pose consumer needs no new API.
        self._f_t_ee = np.eye(4)
        # Scratch for mj_jac, allocated once: it runs at 1 kHz. One 6xnv block
        # with the translational and rotational halves as *views* into it, so
        # the stacked Jacobian every consumer wants is already assembled and
        # ee_jacobian() is a single column-slice rather than two index copies
        # and a vstack. Three allocations a step become one, which matters here
        # for the garbage collector as much as for the arithmetic: this loop has
        # a 1 ms deadline and a gen-2 pass is milliseconds.
        self._jacobian_scratch: Optional[np.ndarray] = None
        self._jacp: Optional[np.ndarray] = None
        self._jacr: Optional[np.ndarray] = None
        # The EE Jacobian and pose for one model state, memoised on ``data.time``
        # so the two consumers that both want them on the same physics step --
        # the state snapshot and the Cartesian control law -- pay for one
        # evaluation between them. See _step_ee_readings().
        self._readings_time: Optional[float] = None
        self._readings_jacobian: Optional[np.ndarray] = None
        self._readings_pose: Optional[np.ndarray] = None

        # Latest commands from the network threads, published lock-free (see the
        # class docstring and the update_* methods).
        self.control_mode = ControlMode.POSITION  # Default to position control
        self.latest_torques = np.zeros(7)
        self.latest_joint_positions = ARM_INITIAL_Q.copy()
        self.latest_joint_velocities = np.zeros(7)
        # POSITION mode's velocity feedforward: the commanded velocity the
        # damping term servos against, differenced over the number of physics
        # steps the target actually took to change (see PositionFeedforward for
        # why that is not the same as "one step"). Seeded to the initial pose so
        # the settle loop's constant target starts at dq_c = 0, and re-seeded by
        # set_control_mode() on every switch into POSITION mode so a stale
        # baseline from an old streaming session or a different control mode
        # never produces a first-step spike.
        self._position_feedforward = PositionFeedforward(ARM_INITIAL_Q)

        # Cartesian generators. ``latest_cartesian_pose`` is the commanded
        # ``O_T_EE_c`` as a row-major 4x4 (None until a pose motion streams
        # one); ``latest_cartesian_twist`` is the commanded ``O_dP_EE_c``;
        # ``latest_elbow_angle`` is ``elbow_c[0]``, the redundancy angle, or
        # None when the client commands no elbow. All rebound whole, never
        # mutated, exactly like the joint commands above.
        self.latest_cartesian_pose: Optional[np.ndarray] = None
        self.latest_cartesian_twist = np.zeros(6)
        self.latest_elbow_angle: Optional[float] = None
        self._cartesian_feedforward = CartesianFeedforward()

        # Per-finger targets (m), each 0..MAX_FINGER_TRAVEL. Start fully open.
        self.max_finger_width = 2 * MAX_FINGER_TRAVEL  # total opening
        self.latest_finger_positions = np.array([MAX_FINGER_TRAVEL, MAX_FINGER_TRAVEL])
        self._finger_snapshot = {"q": np.zeros(2), "dq": np.zeros(2)}

        # Numerical-differentiation state for joint acceleration (physics thread).
        self.prev_dq = np.zeros(7)
        self.ddq_filtered = np.zeros(7)
        self._alpha_acc = 0.95  # acceleration low-pass factor

        self._state_snapshot = {
            "q": np.zeros(7),
            "dq": np.zeros(7),
            "ddq": np.zeros(7),
            "q_d": np.zeros(7),
            "dq_d": np.zeros(7),
            "ddq_d": np.zeros(7),
            "tau_J": np.zeros(7),
            "O_T_EE": np.eye(4).T.flatten(),
            "O_dP_EE": np.zeros(3),
            "O_J_EE": np.zeros((6, 7)),
            "self_collision": None,
        }

    # -- construction ------------------------------------------------------

    def initialize_simulation(self) -> None:
        """Compile the model, bind every joint group and settle the initial pose."""
        self.model, self.prefix = build_model(
            enable_hand=self.enable_hand, dt=self.dt, model_path=self.model_path
        )
        self.data = mujoco.MjData(self.model)
        logger.info(
            "MuJoCo FR3 model compiled: %d dofs (%s), hand=%s",
            self.model.nv,
            self.model_path or default_fr3_mjcf(),
            self.enable_hand,
        )

        self._bind_model()
        self._configure_model()

        self.data.qpos[self.arm_qpos_adr] = ARM_INITIAL_Q
        if self.enable_hand:
            self.data.qpos[self.finger_qpos_adr] = self.latest_finger_positions
        mujoco.mj_forward(self.model, self.data)

        for _ in range(SETTLE_STEPS):
            self._apply_control()
            mujoco.mj_step(self.model, self.data)
        self._read_and_publish_state()

    def _bind_model(self) -> None:
        """Resolve joint and body addresses once the model is compiled."""
        joint_names = [f"{self.prefix}joint{i}" for i in range(1, 8)]
        self.arm_qpos_adr = _qpos_addresses(self.model, joint_names)
        self.arm_dofs_idx = _dof_addresses(self.model, joint_names)

        ee_body = f"{self.prefix}link7"
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        if self.ee_body_id < 0:
            raise KeyError(f"Body {ee_body!r} not found in the compiled model")

        self._jacobian_scratch = np.zeros((6, self.model.nv))
        self._jacp = self._jacobian_scratch[:3]
        self._jacr = self._jacobian_scratch[3:]

        if self.enable_hand:
            finger_names = [f"{self.prefix}finger_joint{i}" for i in (1, 2)]
            self.finger_qpos_adr = _qpos_addresses(self.model, finger_names)
            self.finger_dofs_idx = _dof_addresses(self.model, finger_names)

        self._bind_self_collision()

    def _bind_self_collision(self) -> None:
        """Arm the self-collision reading: widen the margin, list the pairs.

        Two halves, both done once here so the 1 kHz path is a lookup.

        **The margin.** MuJoCo puts a contact into ``mjData.contact`` as soon as
        the two geoms are closer than the pair's ``margin`` (the larger of the
        two geoms'), and hands it to the constraint solver only once it is
        closer than ``margin - gap`` -- the field MuJoCo calls
        ``includemargin``. Setting ``margin == gap ==``
        :data:`SELF_COLLISION_MARGIN` therefore makes that solver threshold
        exactly 0, which is the value it has on the untouched model: the
        *reported* set grows to everything within 50 mm while the *simulated*
        set does not grow at all. **No pair that was not already penetrating
        enters the solver**, which is the property the reflex is added under,
        and it is why this is not done by inflating the geoms.

        What that does *not* claim is that the two models are bit-identical
        once links do interpenetrate. ``margin`` is an input to the narrowphase
        as well as a threshold on its output -- libccd is run with it -- so for
        mesh hulls that already overlap, the *depth* it converges on differs
        between the two models. Measured on a fold with link1/link5 through
        each other: -2.799 mm reported plain against -5.771 mm with the margin
        set, and a closed position loop settling at 48.9 Nm of
        ``qfrc_constraint`` on joint 4 against 26.6 Nm (12 of 300 random
        in-joint-range configurations differ at all). That state is one the FCI
        layer never reaches under control: the reflex fires while the
        separation is still positive (see
        :meth:`franka_sim.motion_limits.MotionLimitChecker.check_self_collision`)
        and the arm is stopped at the margin, tens of millimetres before any
        hull intersects. Reaching it needs a one-cycle teleport straight into
        an overlap, i.e. ``update_joint_positions``, not a motion.

        It also has to be MuJoCo's own detection rather than a Python distance
        sweep: the engine's broadphase culls the far pairs in C and hands back
        the few that are close, at a measured 14-23 us per 1 ms step (61 ->
        75-84 us for the whole step, control law and state publish included,
        the wide end being the folded pose with an extra contact live), of which
        :meth:`self_collision`'s own scan is 3-4 us. A per-step
        ``mj_geomDistance`` loop over the 15 monitored pairs is both an order of
        magnitude dearer and, on this model, *wrong*: it reports 42 mm of
        penetration between link1 and link5 in configurations where the two
        hulls are provably 12 mm apart.

        **The pairs.** Everything at least
        :data:`SELF_COLLISION_MIN_LINK_SEPARATION` links apart among
        :data:`SELF_COLLISION_LINKS`, which on the FR3 is 15 pairs. The margin
        goes on all eight link geoms even so -- a geom's margin is a property of
        the geom, not of a pair, so an unmonitored pair (link5/link7, or an arm
        link against the ground) simply shows up in ``mjData.contact`` and is
        skipped by name in :meth:`self_collision`. (link0/link1 does not show
        up at all: :func:`_exclude_base_shoulder_contact` removes that pair
        from the contact model before compile.)

        A model without ``<prefix>linkN_collision`` geoms leaves both halves
        empty and the check silently off: no margin is touched, and
        :meth:`self_collision` answers None for ever. That is the honest answer
        for a backend that cannot see its own links, and it keeps a caller
        pointing ``$FR3_MJCF`` at a stripped-down model working.
        """
        geoms = {}
        for index in SELF_COLLISION_LINKS:
            geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{self.prefix}link{index}_collision"
            )
            if geom_id >= 0:
                geoms[index] = geom_id

        self._self_collision_pairs = {
            (min(first_id, second_id), max(first_id, second_id)): (
                f"link{first}",
                f"link{second}",
            )
            for first, first_id in geoms.items()
            for second, second_id in geoms.items()
            if second - first >= SELF_COLLISION_MIN_LINK_SEPARATION
        }
        if not self._self_collision_pairs:
            logger.warning(
                "No %slinkN_collision geoms in the model: self-collision detection is off",
                self.prefix,
            )
            return

        margins = np.array(sorted(geoms.values()), dtype=np.intp)
        self.model.geom_margin[margins] = SELF_COLLISION_MARGIN
        self.model.geom_gap[margins] = SELF_COLLISION_MARGIN

    def self_collision(self) -> Optional[SelfCollisionContact]:
        """The closest monitored link pair inside the margin, or None.

        Reads ``mjData.contact`` as the last forward pass left it -- the
        collisions are already computed, so this adds no geometry work to the
        step, only the scan. ``ncon`` is 1-3 on this model in practice (the
        unmonitored link5/link7 near-touch, the ground, and whatever is genuinely
        close), so the scan is a handful of dictionary lookups.

        *Closest*, not first: which pair is reported ends up in the log line and
        in the error's ``describe()``, and "the two links that are 24 mm apart"
        is a more useful answer than "the first pair the solver happened to
        list". The distance is MuJoCo's own ``contact.dist``, i.e. surface
        separation -- positive while the safety offset stands.
        """
        contacts = self.data.ncon
        if not contacts or not self._self_collision_pairs:
            return None
        contact_list = self.data.contact
        geoms = contact_list.geom[:contacts]
        distances = contact_list.dist[:contacts]
        closest = None
        for index in range(contacts):
            first_id, second_id = int(geoms[index, 0]), int(geoms[index, 1])
            pair = self._self_collision_pairs.get(
                (min(first_id, second_id), max(first_id, second_id))
            )
            if pair is None:
                continue
            distance = float(distances[index])
            if closest is None or distance < closest.distance:
                closest = SelfCollisionContact(pair[0], pair[1], distance, SELF_COLLISION_MARGIN)
        return closest

    def _configure_model(self) -> None:
        """Apply the optional joint-damping override to the compiled model.

        Gravity compensation is not set here: it has to be on before compiling
        (see :func:`build_model`).
        """
        if self.model.ngravcomp == 0:
            raise RuntimeError("Model was compiled without gravity compensation")

        # The Menagerie fr3v2 damping/armature/frictionloss are calibrated
        # against the real arm, so they are kept as authored; $FR3_JOINT_DAMPING
        # still overrides the viscous term for calibration experiments, with the
        # same parsing every other FR3 in this package uses.
        if os.environ.get("FR3_JOINT_DAMPING"):
            damping = resolve_fr3_joint_damping()
            self.model.dof_damping[self.arm_dofs_idx] = damping
            logger.info("Joint damping overridden from $FR3_JOINT_DAMPING: %s", damping)

    # -- command interface -------------------------------------------------

    @property
    def prev_position_target(self) -> np.ndarray:
        """The POSITION target the feedforward is currently differencing from."""
        return self._position_feedforward.previous

    def set_control_mode(self, mode: ControlMode) -> None:
        """Set the control mode for the robot (lock-free atomic reference swap)."""
        if not isinstance(mode, ControlMode):
            raise ValueError(f"Mode must be a ControlMode enum, got {type(mode)}")
        logger.info("Switching control mode to: %s", mode.value)
        if mode in (ControlMode.CARTESIAN_POSE, ControlMode.CARTESIAN_VELOCITY):
            # Same discontinuous-entry rule as POSITION below, in Cartesian
            # terms: forget the previous motion's target entirely rather than
            # carry it into this one. Cleared to *nothing*, not to the current
            # pose: this runs on a network thread while the physics thread
            # steps, so reading the model here would be a torn read of
            # ``xpos``/``xmat``, and the arm has no need of a seed -- with no
            # target the servo simply holds zero velocity until this motion's
            # first command lands, which is one millisecond later.
            #
            # Cleared *before* ``self.control_mode`` is published, and that
            # order is load-bearing: the physics thread branches on the mode and
            # then reads the targets, both without a lock, so publishing the
            # mode first opens a window where a step sees "Cartesian" and the
            # *previous* motion's pose or twist still standing -- one full step
            # of tracking a target this motion never asked for (measured: an
            # 11 rad/s joint-4 target on entry). Clearing first can only cost
            # the opposite, harmless thing: a step that still sees the old mode
            # with its own target already gone, which every mode below reads as
            # "hold".
            self.latest_cartesian_pose = None
            self.latest_cartesian_twist = np.zeros(6)
            self.latest_elbow_angle = None
            self._cartesian_feedforward.reset()
        self.control_mode = mode
        if mode == ControlMode.POSITION:
            # Discontinuous-target entry point: a fresh Move into POSITION mode
            # and the idle hold (which sets the target to the current q before
            # calling this) both land here. Snap the feedforward baseline to
            # whatever target is current right now -- and zero the held dq_c
            # with it -- so the very next physics step sees dq_c = 0 rather than
            # differencing against, or coasting on, a baseline left over from an
            # older streaming session or another mode.
            self._position_feedforward.reset(self.latest_joint_positions)

    def update_torques(self, torques) -> None:
        """Publish the latest commanded torques for the physics thread.

        Lock-free: a fresh array is bound in a single bytecode step (atomic under
        the GIL) and the physics thread is the only reader, so no mutex is
        needed. The array must never be mutated in place after assignment.
        """
        self.latest_torques = np.asarray(torques, dtype=float)

    def update_joint_positions(self, positions) -> None:
        """Publish the latest commanded joint positions (lock-free; see update_torques)."""
        self.latest_joint_positions = np.asarray(positions, dtype=float)

    def update_joint_velocities(self, velocities) -> None:
        """Publish the latest commanded joint velocities (lock-free; see update_torques)."""
        self.latest_joint_velocities = np.asarray(velocities, dtype=float)

    def update_cartesian_pose(self, o_t_ee_c, elbow_angle=None) -> None:
        """Publish the commanded EE pose for a ``kCartesianPosition`` motion.

        ``o_t_ee_c`` is the wire's 16-element **column-major** 4x4;
        ``elbow_angle`` is ``elbow_c[0]`` when the client commands an elbow and
        None otherwise. Lock-free (see :meth:`update_torques`): both are rebound
        whole and only the physics thread reads them.

        Publishing a target is all this does. The conversion to joint motion
        happens on the physics thread, once per step, in
        :meth:`cartesian_joint_velocity` -- so a client streaming at 1 kHz and a
        physics loop stepping at 1 kHz stay decoupled, and a cycle that brings
        no command simply tracks the pose that is still standing.
        """
        self.latest_cartesian_pose = np.asarray(o_t_ee_c, dtype=float).reshape(4, 4).T
        self.latest_elbow_angle = None if elbow_angle is None else float(elbow_angle)

    def update_cartesian_velocity(self, o_dp_ee_c, elbow_angle=None) -> None:
        """Publish the commanded EE twist for a ``kCartesianVelocity`` motion.

        ``o_dp_ee_c`` is ``[vx, vy, vz, wx, wy, wz]`` in the base frame, the
        layout libfranka packs ``CartesianVelocities`` in. Lock-free; see
        :meth:`update_cartesian_pose`.
        """
        self.latest_cartesian_twist = np.asarray(o_dp_ee_c, dtype=float)[:6]
        self.latest_elbow_angle = None if elbow_angle is None else float(elbow_angle)

    def update_ee_transform(self, f_t_ee) -> None:
        """Publish the flange -> EE transform the FCI calls ``F_T_EE``.

        ``f_t_ee`` is a 16-element **column-major** 4x4, the layout every
        transform on the FCI wire uses (``std::array<double, 16>``), so it can
        be handed straight through from a ``SetEE``/``SetNEToEE`` request.

        What it changes is where :meth:`_read_and_publish_state` measures the
        arm's Cartesian velocity. The default identity measures at the flange;
        a client that mounts a tool 0.5 m out is measured 0.5 m out, and the
        lever arm is real -- which is exactly the difference that separates
        ``cartesian_velocity_violation`` from ``joint_velocity_violation`` on
        hardware (see
        :meth:`franka_sim.motion_limits.MotionLimitChecker.check_measured_cartesian_velocity`).

        Lock-free (see :meth:`update_torques`): a fresh array is bound in one
        bytecode step and the physics thread only ever reads it.
        """
        matrix = np.asarray(f_t_ee, dtype=float)
        if matrix.size != 16:
            raise ValueError(f"F_T_EE must have 16 elements, got {matrix.size}")
        self._f_t_ee = matrix.reshape(4, 4).T

    def update_finger_positions(self, positions) -> None:
        """Publish per-finger position targets (length-2, each 0..0.04 m).

        Lock-free (see update_torques). The two fingers are mechanically coupled
        by the hand's tendon equality, exactly as on the real single-motor hand,
        so asymmetric targets are met halfway rather than independently.
        """
        self.latest_finger_positions = np.asarray(positions, dtype=float)

    def get_finger_state(self):
        """Return the latest finger snapshot {'q': (2,), 'dq': (2,)} (atomic read)."""
        return self._finger_snapshot

    def get_robot_state(self):
        """Return the latest state snapshot published by the physics thread.

        Lock-free: reads the current snapshot reference (atomic under the GIL).
        No model reads happen here, so the UDP broadcast thread never contends
        with the physics loop. The snapshot holds only the 7 arm joints (the
        fingers have their own, see :meth:`get_finger_state`).
        """
        return self._state_snapshot

    # -- physics loop ------------------------------------------------------

    def arm_control_torque(self) -> np.ndarray:
        """Actuator torque for the arm this step, clamped to FR3_FORCE_LIMITS.

        Gravity is not part of this: ``body_gravcomp`` already cancels it, so
        TORQUE mode passes the client's command through unchanged -- the same
        contract the real FCI offers. The clamp is MuJoCo's equivalent of the
        Genesis sim's ``set_dofs_force_range``.

        POSITION mode's damping term is not plain ``-KD*dq``: undamped, it
        damps against zero velocity, so it fights any commanded motion and
        adds a ``(KD/KP)*qdot`` lag behind the target -- at these gains ~0.1s
        of velocity's worth of position error, enough on its own to fail
        libfranka's tracking guard (RMSE(q, q_c) < 6e-3 rad) on an ordinary
        point-to-point motion. Instead it damps against the commanded
        velocity ``dq_c``, produced once per physics step by
        :class:`~franka_sim.sim_common.PositionFeedforward`: a backward
        difference of the target taken over the number of steps it took to
        change, held across the steps where nothing arrived, and dropped to
        zero once the stream stops. A held target -> dq_c = 0 -> identical to
        the old law. A target that jumps (teleport with enforcement off)
        produces one step of large dq_c; the FR3_FORCE_LIMITS clip below is
        what bounds that, same as it already bounds every other mode.
        """
        mode = self.control_mode
        if mode == ControlMode.TORQUE:
            tau = self.latest_torques
        else:
            dq = self.data.qvel[self.arm_dofs_idx]
            if mode in (ControlMode.CARTESIAN_POSE, ControlMode.CARTESIAN_VELOCITY):
                # A Cartesian command becomes a joint velocity and then goes
                # through the *same* velocity servo a kJointVelocity motion
                # drives -- no separate Cartesian control law, so every
                # measured-side check downstream (the joint velocity envelope,
                # the EE speed limit) judges a Cartesian motion with the code
                # that judges a joint one. See cartesian_joint_velocity().
                jacobian, ee_pose = self._step_ee_readings()
                tau = ARM_VELOCITY_KV * (self.cartesian_joint_velocity(jacobian, ee_pose) - dq)
            elif mode == ControlMode.VELOCITY:
                tau = ARM_VELOCITY_KV * (self.latest_joint_velocities - dq)
            else:
                target = self.latest_joint_positions
                dq_c = self._position_feedforward.step(target, self.dt)
                error = target - self.data.qpos[self.arm_qpos_adr]
                tau = ARM_POSITION_KP * error + ARM_POSITION_KD * (dq_c - dq)
        return np.clip(tau, -FR3_FORCE_LIMITS, FR3_FORCE_LIMITS)

    def finger_control_force(self) -> np.ndarray:
        """Per-finger servo force this step, clamped to FINGER_FORCE_LIMIT."""
        targets = np.clip(self.latest_finger_positions, 0.0, MAX_FINGER_TRAVEL)
        error = targets - self.data.qpos[self.finger_qpos_adr]
        force = FINGER_KP * error - FINGER_KD * self.data.qvel[self.finger_dofs_idx]
        return np.clip(force, -FINGER_FORCE_LIMIT, FINGER_FORCE_LIMIT)

    def _apply_control(self) -> None:
        """Apply one control step to the arm and (when present) the fingers."""
        # qfrc_applied persists across steps, so every controlled DOF is
        # rewritten each step and any other DOF is held at zero.
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.arm_dofs_idx] = self.arm_control_torque()
        if self.enable_hand:
            self.data.qfrc_applied[self.finger_dofs_idx] = self.finger_control_force()

    def ee_pose(self) -> np.ndarray:
        """Measured EE pose as a row-major 4x4 in the base frame.

        The flange body's pose composed with ``F_T_EE``
        (:meth:`update_ee_transform`) -- the same frame :meth:`ee_jacobian`,
        :meth:`measured_ee_velocity` and the published ``O_T_EE`` describe;
        ``_read_and_publish_state`` transposes this very matrix onto the wire.
        Row-major, i.e. ordinary matrix layout: the wire's column-major form is
        one ``.T.flatten()`` away, and mixing the two up is the classic way to
        get a transposed rotation that looks almost right.
        """
        pose = np.eye(4)
        pose[:3, :3] = self.data.xmat[self.ee_body_id].reshape(3, 3)
        pose[:3, 3] = self.data.xpos[self.ee_body_id]
        return pose @ self._f_t_ee

    def cartesian_joint_velocity(
        self, jacobian: np.ndarray, ee_pose: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Joint velocity that realises this step's Cartesian command.

        The bridge between the FCI's two Cartesian interfaces and the joint
        servo: differential IK on ``jacobian``, with the desired twist taken
        from whichever generator is running. ``ee_pose`` is the measured pose
        the same step's ``jacobian`` was taken at (see :meth:`_step_ee_readings`);
        it is read back off the model when a caller does not have one.

        * ``kCartesianPosition``: the commanded pose's own velocity (the
          :class:`~franka_sim.cartesian_ik.CartesianFeedforward` backward
          difference) plus a proportional term on the pose error, so the arm
          both follows the trajectory and closes on it.
        * ``kCartesianVelocity``: the commanded twist as it stands. There is no
          pose to track and therefore nothing to correct towards -- a twist
          generator's *only* statement is a velocity, and integrating one into a
          pose to servo against would invent a reference the client never sent
          and then fight the client's next twist with it.

        The elbow rides in the Jacobian's null space when the client commands
        one; see :func:`franka_sim.cartesian_ik.elbow_null_velocity`.
        """
        if self.control_mode == ControlMode.CARTESIAN_POSE:
            desired = self.latest_cartesian_pose
            if desired is None:
                return np.zeros(7)
            feedforward = self._cartesian_feedforward.step(desired, self.dt)
            if ee_pose is None:
                ee_pose = self.ee_pose()
            twist = tracking_twist(desired, ee_pose, feedforward)
        else:
            twist = self.latest_cartesian_twist
        elbow_angle = self.latest_elbow_angle
        null_velocity = (
            None
            if elbow_angle is None
            else elbow_null_velocity(elbow_angle, self.data.qpos[self.arm_qpos_adr])
        )
        return resolved_rate(jacobian, twist, null_velocity)

    def ee_jacobian(self) -> np.ndarray:
        """Geometric Jacobian of the EE frame, base-oriented, 6x7 over the arm joints.

        Rows 0-2 map ``dq`` to the EE's translational velocity, rows 3-5 to its
        angular velocity. The point is the flange origin shifted by ``F_T_EE``'s
        translation (:meth:`update_ee_transform`) -- and the "flange" is
        whichever body this backend already reports as ``O_T_EE`` (``link7``),
        so the Jacobian, the pose and the velocity the client is told about all
        describe one and the same frame.

        Only the arm's seven columns: the finger DOFs are not ancestors of the
        EE body, so their columns are structurally zero and carrying them would
        only make every consumer slice them off again.

        Two consumers, both in the FCI layer above: the measured Cartesian
        velocity the safety controller watches (:meth:`measured_ee_velocity`),
        and the smallest singular value a ``Move`` uses to decide whether the
        arm is standing in a singularity. Computed once per physics step and
        published in the snapshot so neither of them has to touch ``mjData``
        from another thread.
        """
        rotation = self.data.xmat[self.ee_body_id].reshape(3, 3)
        point = self.data.xpos[self.ee_body_id] + rotation @ self._f_t_ee[:3, 3]
        mujoco.mj_jac(self.model, self.data, self._jacp, self._jacr, point, self.ee_body_id)
        # ``_jacp``/``_jacr`` are views into one 6xnv scratch, so the stack is
        # already there; the column slice is what makes the result a fresh array
        # the caller may keep (the state snapshot does) rather than a window
        # onto the buffer the next step overwrites.
        return self._jacobian_scratch[:, self.arm_dofs_idx]

    def _step_ee_readings(self):
        """``(jacobian, ee_pose)`` for the model as it stands, computed once a step.

        :meth:`_read_and_publish_state` and :meth:`arm_control_torque` both want
        both of them, and in :meth:`run_simulation` they run back to back on the
        *same* model state -- read, apply, step. Evaluating them twice cost
        ~11 us of a 1 ms budget on the thread that must not miss its deadline,
        which on a Cartesian motion was a tenth of the whole control law.

        Keyed on ``data.time``, so the memo can only ever be returned for the
        state it was taken from: anything that advances the physics invalidates
        it, and :meth:`step`'s apply-then-step order simply misses every time
        rather than reading a stale one. The public
        :meth:`ee_jacobian`/:meth:`ee_pose` are left uncached -- a caller that
        pokes ``qpos`` directly (tests do) must get an honest answer.

        It also makes the guarantee exact rather than incidental: the pose the
        client is told about, the Jacobian the safety check uses and the frame
        the control law servos are now literally the same two arrays, so a
        ``F_T_EE`` rebound mid-step cannot land between them.
        """
        if self._readings_time != self.data.time:
            self._readings_jacobian = self.ee_jacobian()
            self._readings_pose = self.ee_pose()
            self._readings_time = self.data.time
        return self._readings_jacobian, self._readings_pose

    def measured_ee_velocity(self) -> np.ndarray:
        """Translational velocity of the EE frame in the base frame (m/s).

        ``J_p * dq`` from :meth:`ee_jacobian` -- a Jacobian rather than a
        difference of poses because the difference would carry the integrator's
        step noise divided by 1 ms, which at these magnitudes is louder than the
        signal. The quantity Control watches for ``cartesian_velocity_violation``.
        """
        return self.ee_jacobian()[:3] @ self.data.qvel[self.arm_dofs_idx]

    def _read_and_publish_state(self) -> None:
        """Read the model once and publish one state snapshot for the network threads."""
        # Copies, not views: the snapshot is handed to the network threads and
        # must not change under them when the next step runs.
        q = np.array(self.data.qpos[self.arm_qpos_adr])
        dq = np.array(self.data.qvel[self.arm_dofs_idx])

        # Filtered numerical acceleration.
        ddq_raw = (dq - self.prev_dq) / self.dt
        self.ddq_filtered = self._alpha_acc * self.ddq_filtered + (1 - self._alpha_acc) * ddq_raw
        self.prev_dq = dq

        if self.enable_hand:
            self._finger_snapshot = {
                "q": np.array(self.data.qpos[self.finger_qpos_adr]),
                "dq": np.array(self.data.qvel[self.finger_dofs_idx]),
            }

        # One Jacobian evaluation serves both Cartesian readings below, and one
        # pose evaluation serves ``O_T_EE`` -- and this step's control law gets
        # both of them for free (see _step_ee_readings).
        #
        # The pose is composed with F_T_EE (ee_pose()), not the bare flange: the
        # FCI's O_T_EE is by definition "measured end effector pose in base
        # frame", and publishing the flange there while ee_jacobian()/
        # measured_ee_velocity() -- and the Cartesian generators that servo
        # against them -- all describe the EE frame would put the whole
        # Cartesian interface a tool-length out of agreement with itself. With
        # the default identity F_T_EE the two are the same matrix.
        jacobian, ee_pose = self._step_ee_readings()

        # q_d / tau_J mirror the latest network commands (atomic reads), exactly
        # as the Genesis sim reports them.
        self._state_snapshot = {
            "q": q,
            "dq": dq,
            "ddq": self.ddq_filtered,
            "q_d": self.latest_joint_positions,
            "dq_d": dq,
            "ddq_d": self.ddq_filtered,
            "tau_J": self.latest_torques,
            "O_T_EE": ee_pose.T.flatten(),
            "O_dP_EE": jacobian[:3] @ dq,
            "O_J_EE": jacobian,
            # The geometric third of the safety controller. Published with the
            # rest of the snapshot rather than read from the FCI thread, for the
            # same reason the Jacobian is: mjData belongs to the physics thread.
            "self_collision": self.self_collision(),
        }

    def step(self, steps: int = 1) -> None:
        """Advance the physics by ``steps`` control steps as fast as the host can.

        The unpaced counterpart to :meth:`run_simulation`, for callers that own
        their own clock (tests, batch rollouts) rather than serving a 1 kHz FCI
        client. Not thread-safe against a running :meth:`run_simulation`: both
        write ``qpos``/``qvel``, so use one or the other.
        """
        for _ in range(steps):
            self._apply_control()
            mujoco.mj_step(self.model, self.data)
        self._read_and_publish_state()

    def run_simulation(self) -> None:
        """Physics loop: publish state, apply control, step -- paced to wall-clock realtime.

        One dt of sim time per dt of wall time, so a controller tuned for the
        real 1 kHz FCI sees the same wall-time dynamics rather than physics that
        free-runs many times faster than reality.
        """
        logger.info("Starting MuJoCo simulation loop (dt=%.4fs)", self.dt)

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
            self.viewer, self._viewer_thread = launch_passive_viewer(self.model, self.data)

        self.running = True
        self.run_simulation()

    def stop(self) -> None:
        """Stop the physics loop and close the viewer.

        Idempotent: a second call (a second Ctrl+C, or the finally-block after
        an explicit stop) finds no viewer left and returns immediately.
        """
        self.running = False
        viewer, self.viewer = self.viewer, None
        thread, self._viewer_thread = self._viewer_thread, None
        if viewer is not None:
            close_passive_viewer(viewer, thread, logger)


def main():
    """Run the MuJoCo FR3 simulation standalone (no FCI server)."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("--hand", action="store_true", default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    sim = MujocoFrankaSim(enable_vis=args.vis, enable_hand=args.hand)
    try:
        sim.start()
    except KeyboardInterrupt:
        sim.stop()


if __name__ == "__main__":
    main()
