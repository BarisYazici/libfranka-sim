"""MuJoCo backend for the single-arm FCI simulator (the default physics engine).

Drop-in replacement for :class:`~franka_sim.franka_genesis_sim.FrankaGenesisSim`:
``FrankaSimServer`` and :class:`~franka_sim.gripper_physics.GenesisFrankaHand`
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

from franka_sim.control_modes import ControlMode
from franka_sim.sim_common import (
    FR3_FORCE_LIMITS,
    RealtimeFactorMonitor,
    pose_to_column_major,
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

#: Wall-clock lag (s) the paced loop catches up on before it gives up and
#: resynchronises its deadline. ``viewer.sync()`` blocks on the render thread's
#: mutex for ~12 ms, so every rendered frame leaves the loop a dozen steps
#: behind; resynchronising at less than a frame's worth of lag would *discard*
#: that simulated time rather than make it up. The bound still exists so a
#: genuinely overloaded loop cannot spiral -- past this much lag the backlog is
#: dropped and the RTF monitor reports the truth.
MAX_CATCHUP_LAG_S = 0.25


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
        self.running = False
        self.prefix = ""

        self.arm_qpos_adr: Optional[np.ndarray] = None
        self.arm_dofs_idx: Optional[np.ndarray] = None
        self.finger_qpos_adr: Optional[np.ndarray] = None
        self.finger_dofs_idx: Optional[np.ndarray] = None
        self.ee_body_id: Optional[int] = None

        # Latest commands from the network threads, published lock-free (see the
        # class docstring and the update_* methods).
        self.control_mode = ControlMode.POSITION  # Default to position control
        self.latest_torques = np.zeros(7)
        self.latest_joint_positions = ARM_INITIAL_Q.copy()
        self.latest_joint_velocities = np.zeros(7)

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

        if self.enable_hand:
            finger_names = [f"{self.prefix}finger_joint{i}" for i in (1, 2)]
            self.finger_qpos_adr = _qpos_addresses(self.model, finger_names)
            self.finger_dofs_idx = _dof_addresses(self.model, finger_names)

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

    def set_control_mode(self, mode: ControlMode) -> None:
        """Set the control mode for the robot (lock-free atomic reference swap)."""
        if not isinstance(mode, ControlMode):
            raise ValueError(f"Mode must be a ControlMode enum, got {type(mode)}")
        logger.info("Switching control mode to: %s", mode.value)
        self.control_mode = mode

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
        """
        mode = self.control_mode
        if mode == ControlMode.TORQUE:
            tau = self.latest_torques
        else:
            dq = self.data.qvel[self.arm_dofs_idx]
            if mode == ControlMode.VELOCITY:
                tau = ARM_VELOCITY_KV * (self.latest_joint_velocities - dq)
            else:
                error = self.latest_joint_positions - self.data.qpos[self.arm_qpos_adr]
                tau = ARM_POSITION_KP * error - ARM_POSITION_KD * dq
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
            "O_T_EE": pose_to_column_major(
                self.data.xpos[self.ee_body_id], self.data.xquat[self.ee_body_id]
            ),
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

            next_step += self.dt
            now = time.perf_counter()
            if next_step > now:
                time.sleep(next_step - now)
            elif now - next_step > MAX_CATCHUP_LAG_S:
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
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.running = True
        self.run_simulation()

    def stop(self) -> None:
        """Stop the physics loop and close the viewer."""
        self.running = False
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                logger.exception("Error closing the MuJoCo viewer")
            self.viewer = None


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
