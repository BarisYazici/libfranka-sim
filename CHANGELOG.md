# Changelog

All notable changes to **franka-sim** are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows
[Semantic Versioning](https://semver.org/): from 1.0.0 on, breaking changes only
come with a major version bump.

## [1.0.0] - 2026-08-24

### Changed

- **BREAKING: the mobile-robot modules moved into a `franka_sim.mobile`
  subpackage, and the old top-level import paths are gone.** Eight files that
  had accumulated at the top level of `franka_sim` are now gathered under
  `franka_sim/mobile/`, so the top-level package reads as the arm/FCI core it
  is: `mobile_duo_sim` → `mobile.duo_sim`, `mobile_duo_mujoco_sim` →
  `mobile.duo_mujoco_sim`, `mobile_duo_common` → `mobile.common`,
  `mobile_duo_runner` → `mobile.runner`, and `spine_stub`, `swerve_base`,
  `swerve_kinematics`, `tmr_genesis_sim` keeping their names one level down.
  The move first shipped with `sys.modules` aliases so the old dotted paths
  kept resolving; those aliases are now deleted outright — `0.4`'s adoption
  was negligible, so this repo is taking the breaking change ahead of `1.0.0`
  rather than carrying dead compatibility shims into it. `import
  franka_sim.spine_stub`, `from franka_sim.swerve_base import SwerveBase` and
  `monkeypatch.setattr("franka_sim.mobile_duo_sim.gs", …)` no longer work —
  import from `franka_sim.mobile.spine_stub`, `franka_sim.mobile.swerve_base`,
  `franka_sim.mobile.duo_sim`, and so on. The top-level class exports are
  **unchanged**: `franka_sim.MobileDuoScene` and every other name in
  `franka_sim.__all__` still resolve exactly as before — this only affects
  imports that reached into the old submodule paths directly. The
  `run-franka-spine-stub` entry point already names the new path. The three
  loggers that were pinned to their pre-move dotted names for backward
  compatibility (`franka_sim.swerve_base`, `franka_sim.mobile_duo_runner`,
  `franka_sim.mobile_duo_common`) are un-pinned too, now `logging.getLogger(__name__)`
  under their new `franka_sim.mobile.*` names — update any logging
  configuration or `caplog` assertions that named them explicitly.

- **BREAKING: the gripper modules moved into a `franka_sim.gripper`
  subpackage, and the old top-level import paths are gone.** The four files
  that made up the Franka Hand half of the package are now gathered under
  `franka_sim/gripper/`, matching the `franka_sim.mobile` split above:
  `gripper_backend` → `gripper.backend`, `gripper_physics` →
  `gripper.physics`, `gripper_protocol` → `gripper.protocol`, and
  `gripper_server` → `gripper.server`. No aliases were left at the old paths —
  `import franka_sim.gripper_server`, `from franka_sim.gripper_protocol import
  GRIPPER_COMMAND_PORT` and `monkeypatch.setattr("franka_sim.gripper_backend.…",
  …)` now raise `ModuleNotFoundError`; import from `franka_sim.gripper.*`
  instead. As with the mobile move, this repo is taking the clean break ahead
  of `1.0.0` rather than carrying compatibility shims into it. The gripper
  subpackage re-exports nothing, so `import franka_sim.gripper` stays light.
  The loggers are unaffected: all four modules already used
  `logging.getLogger(__name__)`, so their dotted names simply follow the files
  to `franka_sim.gripper.*` — update any logging configuration or `caplog`
  assertions that named `franka_sim.gripper_server` or
  `franka_sim.gripper_backend` explicitly.

- **BREAKING: `GenesisFrankaHand` is now `FrankaHandPhysics`, and the name in
  `franka_sim.__all__` changed with it.** The class never touched Genesis — it
  only calls `update_finger_positions`/`get_finger_state` on whatever sim it is
  handed, and since the MuJoCo backend landed it has been used mostly against
  MuJoCo scenes, so the old name actively misled about which engine it required.
  The new name matches its new module, `franka_sim.gripper.physics`. No
  deprecation alias is provided: `franka_sim.GenesisFrankaHand` and
  `from franka_sim.gripper.physics import GenesisFrankaHand` both fail. The
  constructor signature and behaviour are otherwise unchanged, and
  `franka_sim.FrankaHandPhysics` resolves eagerly exactly as the old name
  did. (Its first parameter, `genesis_sim` at the time of this rename, is
  itself renamed to `physics_sim` below.)

- **BREAKING: `FrankaSimServer`'s `genesis_sim` constructor kwarg (and the
  `self.genesis_sim`/`FrankaHandPhysics(genesis_sim=...)` attribute it fed)
  are renamed to `physics_sim`.** MuJoCo has been the default engine for two
  releases, so `genesis_sim` was a naming fossil that misleadingly implied
  Genesis specifically, even when injecting a MuJoCo backend or a test
  double. No compatibility alias is provided ahead of `1.0.0`:
  `FrankaSimServer(genesis_sim=...)` now raises `TypeError`, and code reading
  `server.genesis_sim` raises `AttributeError` — use `physics_sim` in both
  places. Genesis-specific names are unaffected: `FrankaGenesisSim`,
  `TMRGenesisSim`, `MobileDuoScene`, and the `franka_genesis_sim`/
  `tmr_genesis_sim` modules keep their names, since those really are Genesis.

- **A second server on the same port now fails loudly with `EADDRINUSE`.**
  The TCP listeners (arm port 1337, gripper port 1338) no longer set
  `SO_REUSEPORT`. With it, two servers co-bound the port with no error on
  either side and the kernel load-balanced incoming clients between them —
  observed in practice as spurious connection timeouts and cross-talk when a
  test run and a live server overlapped. `SO_REUSEADDR` remains, so a
  restart still rebinds through a lingering `TIME_WAIT`. The arm server also
  binds its port *before* physics initialization now, so a busy port kills
  `run-franka-sim-server` immediately instead of after the scene loads, and
  the old silent "force close and rebind" retry is gone.

### Added

- **Cartesian motion generators now drive the arm (MuJoCo backend).** Both
  `kCartesianPosition` and `kCartesianVelocity` were checking-only: the
  commanded stream was validated in full and the arm never moved. They are now
  converted to joint motion by damped-least-squares differential IK on the
  MuJoCo Jacobian at the EE frame (`franka_sim/cartesian_ik.py`), evaluated once
  per physics step.
  - The joint velocity that comes out is fed into the **existing** velocity
    servo, the same one a `kJointVelocity` motion drives, so every measured-side
    safety check applies to a Cartesian motion unchanged. That is what makes a
    Cartesian motion driven into the joint position limits behave as hardware
    does: the commanded trajectory drives joint 7 into its position stop, the
    position-based velocity envelope collapses to zero there, and the sim aborts
    with `joint_velocity_violation` — the error hardware records.
  - **The command-stream checking layer is untouched.** IK tracking is a second
    consumer of the accepted stream, not a filter on it; every limit, start-pose
    and elbow check runs first and on the same signal it always did.
  - `elbow_c[0]`, the redundancy angle, steers the Jacobian's **null space**, so
    it moves the elbow without moving the end effector. `elbow_c[1]` (the branch
    flag) is not followed — flipping it means passing through a singularity,
    which the FCI treats as an error.
  - The IK twist is clamped to the FR3's published Cartesian velocity limits.
    That never binds on a checked stream; it exists so that with
    `--enforce-motion-limits` *off* a client teleporting its commanded pose
    produces a wrong arm rather than an exploding one.
  - Under `kExternalController` nothing changes: the client's torques still
    drive the arm and the pose stream stays a reference.
  - Both internal controller modes drive — `kCartesianImpedance` as well as
    `kJointImpedance` — through the same joint-velocity servo, since that is the
    only law the backends implement. `SetCartesianImpedance` is still accepted
    and not enforced, so `kCartesianImpedance` buys you a tracking arm rather
    than a compliant one; what it no longer buys you is an accepted, checked and
    completely inert motion.
  - Backends without IK (Genesis, the mobile-duo scene view) keep the previous
    checked-but-inert behaviour; the server asks the backend once and dispatches
    accordingly.

- **`cartesian_velocity_violation` (4) — the safety controller's Cartesian
  half.** The measured end-effector speed is now watched against
  `franka::kMaxTranslationalVelocity` (3 − 1e−3 m/s) every cycle, in every
  control mode, computed from the MuJoCo Jacobian at the frame `F_T_EE` defines.
  This is the error hardware raises when the EE is moved 0.5 m out along the
  flange with `setEE` *before* running an
  ordinary joint-velocity ramp: only that lever arm gets the EE past 3 m/s
  before the joints reach their own envelope.
  - **Translation only, and it outranks `joint_velocity_violation` (3).** Both
    are read off hardware rather than assumed: a 3 Nm torque ramp and a
    5 rad/s² `dq_c` ramp each spin joint 6
    through its 4.18 rad/s envelope — so the EE angular speed passes 2.5 rad/s
    first — and hardware still answers `joint_velocity_violation`, which rules a
    rotational term out; and with the 0.5 m lever hardware reports
    `cartesian_velocity_violation` *alone*, which fixes the precedence.
  - `F_T_EE` is now **derived and published** as `F_T_NE · NE_T_EE`, libfranka's
    own decomposition, instead of being a permanent identity. `setEE` therefore
    shows up in `RobotState` and in the frame the sim measures at.

- **A Cartesian `Move` from a singular configuration is rejected** with
  `Move::Status::kStartAtSingularPoseRejected`, which libfranka turns into
  "Move command rejected: cannot start at singular pose!". The test is the
  smallest singular value of the EE Jacobian at the arm's measured `q` against
  0.05 — placed between a known singular FR3 configuration (σ_min ≈ 0.011) and
  the tightest start pose a Cartesian motion is expected to open from (≈ 0.139).
  - Delivered as the motion's **terminal** response, after `kMotionStarted`,
    because libfranka's `executeCommand<Move>` handles the first response
    outside any try/catch: a rejection delivered there escapes as a bare
    `CommandException`, and only the mode-wait loop that follows converts one
    into the `ControlException` a client sees on hardware.
  - Joint interfaces are **not** refused: a singular configuration is reached in
    the first place by joint motion, and driving out of a singularity is the only
    way to leave one.

- **Packet-loss extrapolation — a missed motion-generator cycle now continues
  the trajectory, as the FCI does.** This was the last documented divergence in
  the communication layer. Control "takes the previous waypoints and performs a
  linear extrapolation (keep acceleration constant and integrate) for the missed
  time step" and reports what it extrapolated back as `q_d`/`dq_d`/`ddq_d`
  "even in case of packet losses"; franka-sim held the last command instead.
  - **One substitute waypoint per missed cycle**, at the acceleration **frozen**
    when the gap began — computed once from the backward differences of the last
    two commands that actually arrived, never re-derived from the sim's own
    extrapolated samples. Per interface: `q_c` gets `v += a dt` and then
    `q += v dt`; `dq_c` gets `dq += a dt` with no jerk term; `O_T_EE_c` gets the
    same law per translation axis and a rotation composed on the right by the
    axis-angle increment `(ω + α dt) dt`; `O_dP_EE_c` extends at frozen
    twist-acceleration; `elbow_c[0]` follows the 1-D position law while
    `elbow[1]`, a branch flag with no derivative, is held.

    The order of those two integrals on the position-like signals is
    load-bearing, and it is what makes the law self-consistent with the backward
    differencing it feeds: the first difference of an extrapolated waypoint is
    exactly the velocity stored with it, and the second difference is exactly the
    frozen acceleration. A stream whose acceleration is genuinely constant is
    therefore extrapolated onto its own next waypoints **exactly**, and resumes
    reporting exactly the acceleration it was commanding — pinned for gaps of 1
    to 19 cycles and accelerations of 0.5 to 5 rad/s². The trapezoidal
    `q += v dt + a dt²/2` leaves a half-step of slack against the differencing
    every cycle; it accumulates, and the resume reports `a·(gap/2 + 1)` — enough
    to abort a conforming 5 rad/s² client after a **two**-cycle gap.
  - **A joint-velocity motion's opening command no longer seeds a phantom
    acceleration.** `dq_c` shares its differencer with `dq_c`'s position-mode
    sibling, but the two disagree about which stored field the frozen fallback
    has to zero: on a position stream the frozen quantity is `second`
    (acceleration), on a velocity stream it is `first` — `second` there is an
    unread jerk. Zeroing the wrong one left a flagged acceleration in full
    control of a following gap; an opening `dq_c` within
    `START_VELOCITY_TOLERANCE` (0.1 rad/s) of `dq_d` was excused by the start
    check but still differenced against the seeded history as if it were a
    continuation, implying up to 100 rad/s² from a single sub-tolerance step and
    running a nineteen-cycle gap out to 20× the only value the client ever sent.
    The joint-velocity generator's opening command now rebases the history, the
    same way the position and Cartesian-pose generators already do, and the
    flat-freeze fallback zeroes the field each generator actually integrates.
  - **A flagged command cannot seed a gap.** With enforcement off a command that
    breaks a limit is reported and still applied — that is what the switch is
    for — but its backward differences are no longer what a following gap
    freezes. One duplicated datagram differences to zero velocity and `-v/dt` of
    acceleration; frozen and integrated for nineteen cycles that dispatched a
    reference running backwards at nineteen times the commanded speed
    (0.127 rad/s forwards → 2.419 rad/s backwards) into physics and onto the
    wire. The gap now freezes from the last command the checker did not flag, or,
    where there is no adjacent clean one, from the commanded velocity with the
    acceleration zeroed. Enforced behaviour is unchanged: a refused command never
    enters the history at all.
  - **Physics receives the extrapolated targets and the wire publishes them.**
    The substitute goes through the same dispatch a received datagram does, so
    the arm keeps moving through a gap and `q_d`/`dq_d`/`ddq_d` (and
    `O_T_EE_c`/`O_T_EE_d` on a pose motion) advance with it — which is what stops
    a client's own 100 Hz command low-pass (`ControlLoop::convertMotion`, gain
    0.3859) from being dragged off its trajectory by a frozen reference.
  - **The extrapolation is checked like a command and is not clamped.** An
    extrapolation that runs out past the velocity envelope or a joint stop
    latches the same error a client commanding it would. That is the documented
    behaviour, not collateral damage: it is the mechanism behind libfranka's
    warning that intermittent drops "could trigger `discontinuity` errors even
    when your source signals conform with the interface specification".
  - **It stops at 20 consecutive misses**, where the robot stops. Past the bound
    the last reference holds — a client that has genuinely gone away leaves the
    arm standing still rather than flying off — and the existing
    `communication_constraints_violation` latch and abort run unchanged
    (enforced: reflex and `kReflexAborted`; unenforced: a log line).
  - **Torque still holds**, which is also hardware's behaviour: "if a controller
    command packet is dropped, FCI will reuse the torques of the last successful
    received packet."
  - **Always on**, independent of both strictness switches, because it is not a
    check — it is what the robot's reference *is* while the client is quiet.
  - **A datagram that turns up late replaces the guess it stood in for.** The
    FCI drops a command that missed its 1 ms window; this sim applies it, so the
    two choices collide the moment extrapolation exists — the substitute
    waypoint and the late datagram are the *same* step, one guessed and one
    measured, and differencing them against each other reports a reference that
    travelled nowhere followed by an enormous deceleration. Measured aborting a
    conforming client's approach with a stock libfranka client at a
    `control_command_success_rate` of 0.99. The extrapolations a late datagram
    supersedes are now discarded and it is differenced against the last command
    that actually arrived, over the interval it really travelled. A replay or a
    reordered packet gets no rewind (its id is one the history is already built
    on), and neither does a *fresh* command — which is what makes the resume
    trap below fire rather than being quietly absorbed.

    The run of losses stays rewindable **until a fresh command closes it**, not
    until the first late datagram of it: a receive thread that was descheduled
    hands over a whole backlog at once and every datagram in it is the true
    answer to one of the run's cycles. Rewinding only for the first left the rest
    applied-but-not-recorded, the history frozen *k* cycles in the past while the
    id it claimed to sit at was dragged forward to the current cycle — 127 rad/s²
    out of a two-cycle receive-thread stall, 1273 out of an eleven-cycle one,
    linear in the stall and every one of them an abort under enforcement. The id
    the history sits at now advances by exactly the one cycle each extrapolation
    integrates, so it can never claim motion the reference did not travel.

    Rewind, check and record are also **one operation** on the checker
    (`absorb_command`), under a single hold of its lock. As three separate calls
    they left two windows for the publish thread's own extrapolation to land in,
    and either one re-created the same false abort at 64–191 rad/s² on an
    ordinary conforming ramp. And the rewind is now only final once the command
    is *accepted*: a late datagram that enforcement refuses leaves the
    extrapolated reference standing, where before it rolled the reference back
    and recorded nothing in its place, dispatching the next extrapolated cycle to
    physics as a backward jump the size of the gap.
  - **An extrapolated command is never dispatched under a newer motion.** A
    `Move` accepted between building a substitute on the publish thread and
    dispatching it rewrites `motion_generator_mode`/`controller_mode`, which is
    what the dispatch branches on; the motion token is now re-read inside the
    dispatch's own `_hold_lock`, which the `Move` path takes too.
  - **`extrapolate()` refuses a repeated or backward `message_id`.** Its caller
    (the publish loop) owns a strictly-increasing id, not client data, and
    calling it twice for the same id integrated a *second* cycle of motion into
    the history while `_applied_id` — which advances by at most one per call —
    stood still, leaving the next real command differenced over an interval its
    own bookkeeping said was one cycle when the history had actually moved two.
    Now an assertion, since nothing but a caller bug can trigger it.
  - **The real-robot resume trap is now reproducible in sim.** A client that
    pauses mid-ramp and resumes from *its own* last waypoint commands a step
    backwards the size of the whole gap, and the sim latches the discontinuity
    hardware would; resuming from the reported `q_d` differences clean over the
    standard millisecond. `docs/robot-state.md` documents both halves.
- **Cartesian motion-generator checking — `kCartesianPosition` is no longer a
  silent hole.** The commanded pose stream was decoded off the wire and dropped,
  so a client provoking a Cartesian error never got an abort
  and hung forever. `O_T_EE_c` and `elbow_c` now route into the same checker,
  freshness and coalesced-cycle plumbing as the joint signals, and every error the
  pose interface can raise on hardware is raised here. The indices and their
  precedence are pinned against observed FR3 hardware behaviour:
  - a step in `O_T_EE_c` is `cartesian_motion_generator_velocity_discontinuity`
    (19), **not** the envelope's 18 — the interface-relative naming rule the
    joint generators already followed, now on the Cartesian half: an
    acceleration-limit break on a commanded *pose* is named one derivative down,
    and jerk lands one further up at 20;
  - the first `O_T_EE_c` is checked against the robot's measured `O_T_EE` and
    outranks every discontinuity —
    `cartesian_position_motion_generator_start_pose_invalid` (16);
  - a non-rigid or non-finite matrix is
    `cartesian_position_motion_generator_invalid_frame_flag` (31) and is
    **refused whether or not enforcement is on**, mirroring libfranka's
    client-side `checkMatrix` as a server-side sanity refusal;
  - the elbow gets its three errors: `..._start_elbow_invalid` (22),
    `..._elbow_sign_inconsistent` (21) and `..._elbow_limit_violation` (17), the
    last covering `kMaxElbowVelocity`/`Acceleration`/`Jerk` on `elbow_c[0]`.
  - Rotational rates come from the axis-angle of `R_prev^T · R_curr` — a log map
    written with numpy alone, with clamped-trace and near-π branches so an
    only-1e-5-orthonormal commanded matrix can never hand `acos` a NaN.
  - **The arm still does not move on the pose interface.** There is no inverse
    kinematics stage; this is a checking layer, and the deliverable is the abort.
    With enforcement *off* the checks only log, so a client waiting for an
    abort still hangs — that residual is documented in `docs/compatibility.md`.
- **Arm-role `kCartesianVelocity` is checked too.** The twist checks existed but
  were reachable only from the mobile duo's base role; an arm-role Cartesian
  velocity motion was judged on nothing at all. Its `O_dP_EE_c` now goes through
  the identical path, so a twist step aborts with
  `cartesian_motion_generator_acceleration_discontinuity` (20) on either role.
  The elbow checks that ride along with it are arm-only: `STEERING_DRIVE`
  (the base) has no elbow of any kind and never records one, so a base client
  that happened to set `valid_elbow` used to get `start_elbow_invalid`
  re-latched every cycle against a swerve steering angle — that generator now
  skips the elbow checks entirely, which is a fix, not "unchanged".
- **`O_T_EE_d` and `O_T_EE_c` are no longer a permanent identity.** They are FCI
  fields, not the physics backend's, and they now follow the real robot's
  semantics — the Cartesian twin of what `q_d`/`dq_d` already did:
  - **idle, between motions, and under any non-Cartesian generator** both report
    the *measured* `O_T_EE`, republished every cycle. That is the internal
    controller's hold pose, which is what the robot reports as commanded when it
    is the one holding the flange;
  - **during a `kCartesianPosition` motion** `O_T_EE_c` is the last pose the
    client commanded and `O_T_EE_d` is the pose the generator is tracking; a lost
    cycle *extrapolates* both, exactly as `q_d` is extrapolated;
  - **when the motion ends, however it ends** — motion-finished datagram, reflex
    abort, StopMove, a client that simply vanishes — both snap back to the
    measured pose.

  This was not cosmetic. A libfranka Cartesian-pose motion generator initialises
  and holds from `O_T_EE_d` (the conventional opening for a pose motion is
  `std::array<double, 16> cmd = state.O_T_EE_d;`) and libfranka's command
  low-pass filter blends each new command with `O_T_EE_c`
  (`ControlLoop<CartesianPose>::convertMotion`). With both stubbed to identity,
  *every* pose motion streamed a pose metres and a full rotation from the robot
  and tripped `cartesian_position_motion_generator_start_pose_invalid` on cycle 0
  — a sim artifact that masked every other Cartesian error behind it. With the
  fields published faithfully, five distinct Cartesian error scenarios go from
  false start-pose failures to the errors hardware actually reports:
  a velocity discontinuity on `O_T_EE_c`, an invalid frame matrix, an elbow
  sign flip mid-motion, an invalid start elbow, and an elbow limit violation.
  **Arm roles only** — the mobile-duo base role's
  `O_dP_EE_d`/`O_dP_EE_c` echo and its dead-reckoned `O_T_EE` are untouched,
  guarded by the same role check the rest of the commanded-field ownership uses.
- **`elbow_c` joins `elbow_d`, and both now echo a commanded elbow.** Outside a
  Cartesian motion — and inside one whose commands carry no elbow — they report
  the measured `(q[2], sign(q[3]))`; while either Cartesian generator streams an
  elbow they echo it, and they snap back when the motion ends. `delbow_c` and
  `ddelbow_c` remain zero stubs.
- **`elbow` and `elbow_d` are no longer a zero stub.** They report
  `(q[2], sign(q[3]))` — the redundancy angle is joint 3 on an FR3 and the branch
  flag is the sign of joint 4, so both are a reading of `q`. This is what makes
  the elbow interface reachable at all: libfranka's `checkElbow` throws
  client-side unless the flag is exactly ±1, so a client building
  `CartesianPose{O_T_EE_d, elbow_d}` from a zero-filled `elbow_d` never got a
  datagram onto the wire. Arm roles only.

- **FCI communication-constraints emulation.** franka-sim no longer presents a
  perfect channel. Every published `RobotState` opens a cycle; a conforming
  client answers it with a `RobotCommand` echoing that state's `message_id`
  (which is what libfranka's `sendRobotCommand` stamps), and a cycle with no
  such answer before the next state goes out is a lost cycle. All accounting is
  in cycle space, so a physics stall — which delays state and answer alike —
  never reads as packet loss.
  - **`control_command_success_rate` is now real**: the fraction of the last 100
    cycles answered in time, recomputed every cycle, exactly the window
    libfranka documents.
  - **A missed cycle is extrapolated for a motion generator and held for the
    torque controller**, which is what the FCI does with each — see the
    packet-loss extrapolation entry at the top of this release.
  - Freshness is bounded on **both** sides: an echo older than the open cycle is
    late, and an id the server never published is not an answer at all. A
    datagram that is not fresh is still applied and still checked in full — over
    the same server-observed interval as any other command — but it never
    becomes the *sample* the next command is differenced against. A client running permanently one
    cycle behind is charged for every cycle; the sub-cycle bias at the
    tick/`sendto` boundary is left in the client's disfavour, because nothing in
    cycle space can tell that packet from a one-cycle-late one.
  - `message_id` now starts at **1**, so the first published state can be
    answered like any other (a command echoing `0` is unparsable and was charged
    to the client as loss).
  - **`communication_constraints_violation`** (opt-in, see below): 20
    consecutive lost cycles latch the error in `errors` *and* `reflex_reason`,
    set `robot_mode` to `kReflex`, answer the pending `Move` with
    `kReflexAborted` (which libfranka raises as a `ControlException`), and
    recapture the arm in the idle hold. The state carrying the error reaches the
    wire *before* the TCP response, which is the order libfranka's
    `throwOnMotionError` needs — for motion-limit aborts raised on the UDP
    thread as well, which wait for a state serialised *after* the error rather
    than for the next packet to leave, whatever it happens to contain. `AutomaticErrorRecovery` clears `errors`,
    returns to `kIdle` and re-arms the accounting, so a second violation in the
    same connection aborts exactly like the first.
  - Implemented once in the server layer, so all four physics backends and all
    three mobile-duo bridges get it — the base included, since a body twist is
    a motion command.
- **FCI motion-limit and discontinuity checking.** Every commanded signal is now
  differentiated with backward Euler at the 1 ms cycle and compared against the
  limits libfranka publishes in `rate_limiting.h` — the same formulas its own
  documentation spells out, differenced against the last *applied* command over
  the interval the server itself observed between them, which the client's echoed
  `message_id` may bound from above but never widen.
  - **Joint position generator**: `q_c` inside the FR3 joint range
    (`joint_motion_generator_position_limits_violation`), implied velocity
    (`..._velocity_limits_violation`), acceleration (`..._velocity_discontinuity`)
    and jerk (`..._acceleration_discontinuity`), plus a start-pose check
    (`joint_position_motion_generator_start_pose_invalid`).
  - **Joint velocity generator**: velocity, acceleration and jerk, one derivative
    up from the same machinery.
  - **Cartesian velocity generator** (the mobile base's body twist): translational
    and rotational velocity, acceleration and jerk, compared as norms the way
    `limitRate` does.
  - **Torque controller**: `tau_J_range_violation` for a torque outside the
    joint's effort limit, `controller_torque_discontinuity` for `kMaxTorqueRate`.
  - **Non-finite commands are refused whether or not enforcement is on.** Every
    `value > limit` comparison against a NaN is false, so a NaN passed every
    check, poisoned the backward differences it was recorded into and reached
    both the physics backend and the wire. libfranka will not send one
    (`lowpassFilter` throws on a non-finite value), so this only fires for a
    hand-rolled client.
  - **Every command that reaches the simulator has been checked**, fresh or not,
    first command of a motion or resuming after a gap. The interval it is
    differenced over is the server's own: how many states it has published since
    the command the applied history sits at (`MotionLimitChecker.note_published`).
    A gap of any length is therefore measured at the rate the client actually
    commanded, and an echo naming a state the server never published buys no
    discount, because the interval is bounded by that count. The three-cycle
    `MAX_COALESCED_CYCLES` cap survives only as the fallback for callers that
    drive the checker with no publish loop behind it.
  - **Checking and reporting are always on**: a violation logs a rate-limited
    warning naming the joint or axis, the value and the limit, once per error
    class per motion.
- **`--enforce-motion-limits`** (and `FRANKA_SIM_ENFORCE_MOTION_LIMITS=1`) makes a
  violation abort the motion: the offending command is refused rather than applied,
  the matching bit is latched in `errors` *and* `reflex_reason`, `robot_mode`
  becomes `kReflex`, the pending `Move` is answered `kReflexAborted`, and
  `AutomaticErrorRecovery` clears it. **Off by default** and independent of
  `--enforce-comm-constraints`.
- **`--enforce-comm-constraints`** (and `FRANKA_SIM_ENFORCE_COMM_CONSTRAINTS=1`)
  turns the violation abort on. **Off by default**: tracking and reporting always
  run, but a sim is routinely driven by scripts and teleop bridges that are not
  1 kHz realtime loops, so nothing aborts until asked.
- **`--no-enforce-comm-constraints` / `--no-enforce-motion-limits`**: explicit off
  switches that override the environment variables, for a run inside a shell,
  launch file or container that exports one. The variables themselves are now
  parsed as an allow-list (`1`, `true`, `t`, `yes`, `y`, `on`, `enable`,
  `enabled`, and any nonzero integer);
  previously anything non-empty was truthy, so `=disabled` *enabled* enforcement.
- **The safety controller: `joint_velocity_violation` (index 3).** A per-cycle
  check of *measured* joint velocity against the position-based envelope,
  evaluated at the measured configuration, active in **every** control mode —
  including `startTorqueControl`, where no commanded velocity exists at all and
  a torque that accelerates a joint past the envelope is the only way to see the
  error. This is a different error from
  `joint_motion_generator_velocity_limits_violation` (13), which judges the
  *command*; when 13 fires during a motion, 3 is latched alongside it in the
  same abort rather than one preempting the other. That pairing is a reported
  hardware behaviour rather than one measured here: hardware raises both errors
  for this pair, with `joint_velocity_violation` appearing much earlier because
  the controller shapes the envelope down towards the current velocity. Treat it
  as strong but not independently verified.
  Reporting is always on;
  aborting follows `--enforce-motion-limits` and is identical to every other
  violation (error bit, `kReflex`, `Move` answered `kReflexAborted`). A 0.1 rad/s
  margin over the envelope keeps integrator noise from firing it; the margin is
  not applied when the only question is which name an already-certain violation
  gets. Skipped on a mobile-*base* server, whose joints are not FR3 joints.

### Known issues

- **Velocity-commanded motions are not run-to-run repeatable.** A client that
  runs the same motion sequence twice — once through `robot->control()` and once
  through the read/write active-control interface — and compares where the two
  land will see ~0.01 rad of divergence, enough to fail a 5e-3 rad per-joint
  comparison ~10–30% of the time under host load; a long multi-stage motion
  sequence occasionally goes with it. The physics
  thread steps on the wall clock while the client's motion profile advances on
  the `message_id` clock of the published states, and the publish loop runs a few
  Hz under 1 kHz; every millisecond of slip is an extra step of the last
  commanded velocity that nothing corrects. **Not introduced by the Cartesian
  work** — it reproduces on 0.4.0, where the divergence enters at the
  `kJointVelocity` stage instead (0.044 rad observed), and an interleaved A/B of
  the two revisions does not separate them (4/12 vs 1/12, Fisher p ≈ 0.32).
  Driving the Cartesian generators only gave the same mechanism two more motions
  to accumulate in. Position generators are unaffected. The fix — stepping the
  physics once per published control cycle rather than once per wall millisecond
  — is roadmap, not in this release. See
  [the known-issue section](docs/compatibility.md#known-issue-run-to-run-repeatability-of-velocity-generators).
- ~~**Unexplained:** a `kCartesianPosition` motion very occasionally aborts
  mid-flight with `cartesian_motion_generator_velocity_discontinuity` at a few
  hundred to ~1500 m/s² on a commanded stream that is smooth by construction
  (~1 in 15 parity-test runs). Not root-caused.~~ **Root-caused and fixed** — see
  the drain-gate entry under *Fixed*. The gate released while the receive thread
  still had a read datagram in hand, the published echo froze for one cycle, and
  the client's own 100 Hz command filter emitted a `(1 - gain) × Δcommand` step
  that this server then measured and aborted on. A residual remains, bounded by
  the gate's own 5 ms timeout: a receive-thread stall longer than that still
  publishes a stale echo.

### Changed

- **Arm roles now echo the commanded twist into `O_dP_EE_c`/`O_dP_EE_d`, which
  fixes a silent 0.386x scaling of the whole `kCartesianVelocity` interface.**
  `ControlLoop<CartesianVelocities>::convertMotion` sends
  `lowpassFilter(dt, motion.O_dP_EE, robot_state.O_dP_EE_c, cutoff)`, not the
  twist your callback returned. With `O_dP_EE_c` left at the wire struct's
  permanent zero on an arm role — the mobile base echoed its own twist, the arm
  echoed nothing — the filter's reference was zero on *every* cycle instead of
  the client's previous command, so the expression stopped being a filter and
  became a constant multiplier: `gain = dt / (dt + 1/(2*pi*f_c))` = **0.3859** at
  libfranka's default 100 Hz cutoff and 1 ms. A client asking for 0.1 m/s got
  0.0386 m/s, for the whole motion, with nothing on the wire to show why. The
  fields now carry the client's stream while a twist motion runs (extrapolated
  through a lost cycle like every other commanded field) and return to zero when
  it ends — an arm held by its internal controller commands no end-effector
  motion. This is the exact Cartesian twin of the `q_d` staleness fixed below.

- **`O_T_EE` is now published at the end-effector frame, not the bare flange.**
  It is the measured `link7` pose composed with `F_T_EE`, which is the frame
  `ee_pose()`, the EE Jacobian, the measured `O_dP_EE` and both Cartesian
  generators already worked in — so pose, Jacobian and velocity finally describe
  one and the same frame. A **no-op at the default identity `F_T_EE`**; it
  matters the moment a client sends `SetNEToEE`, where publishing the flange
  while servoing the EE meant the client read a pose a whole tool-length away
  from the frame the sim was controlling, and the first Cartesian command
  computed from it opened with a tool-length pose error.

- **`elbow_d`/`elbow_c` and `O_T_EE_d`/`O_T_EE_c` no longer track the measured
  arm while a Cartesian motion is running.** They are commanded fields: during
  either Cartesian generator they carry the client's own stream, and where the
  stream carries nothing — a pose motion that commands no elbow, a twist motion
  that commands no pose, the cycle before the first command lands — they are now
  **frozen at the value the motion started from** instead of following the arm.
  libfranka builds every Cartesian command out of these fields (they are its
  low-pass filter's reference and its rate limiter's baseline, and the
  conventional opening is `franka::CartesianPose{state.O_T_EE_d,
  state.elbow_d}`), so now that the arm is actually driven from those commands,
  reporting the measured value there closes a feedback loop through the client.
  Idle and joint motions are unchanged: they report the measured pose and elbow,
  which is what a "start from `state.O_T_EE_d`" generator reads on its first
  cycle.

- **Motion-limit errors now use the interface-relative discontinuity names the
  robot uses.** Which discontinuity a violation is called depends on the channel
  the client commands, not on the derivative that broke its limit, and a
  discontinuity outranks the velocity-envelope check when one step breaks both.
  Pinned against observed FR3 hardware behaviour. If you match on error names,
  the ones that changed are:
  - a mid-motion step in **`q_c`** now latches
    `joint_motion_generator_velocity_discontinuity` (14), previously
    `joint_motion_generator_velocity_limits_violation` (13);
  - a step in **`dq_c`** — including one away from the reported `dq_d` on the
    motion's first cycle — now latches
    `joint_motion_generator_acceleration_discontinuity` (15), previously 14 or
    13, and the jerk check on that interface maps to 15 as well;
  - a step in **`O_dP_EE_c`** (the mobile base twist) now latches
    `cartesian_motion_generator_acceleration_discontinuity` (20), previously
    `cartesian_motion_generator_velocity_discontinuity` (19) or the envelope's
    18. Index 19 is now unreachable: it belongs to a Cartesian *pose*
    generator, which this server does not serve.

  Unchanged: `joint_position_motion_generator_start_pose_invalid` (11) still
  outranks every discontinuity on a motion's first cycle, and the torque
  controller still reports `tau_J_range_violation` before
  `controller_torque_discontinuity` — the one ordering here with no hardware
  evidence either way, now pinned by a test so a change to it is deliberate.

- **`control_command_success_rate` reports `0.0` when no control or motion
  generator loop is running**, replacing the previous hard-coded `1.0`. This is
  the documented hardware behaviour ("shows a value of zero if no control or
  motion generator loop is currently running", `include/franka/robot_state.h`),
  and it is what `echo_robot_state` prints against a real robot. A client that
  reads the field outside a control loop and expects `1.0` will see `0.0`.
- **`q_d`, `dq_d`, `ddq_d` and `tau_J_d` are now what was *commanded*.** The FCI
  layer owns them on **arm roles** and no physics backend overwrites them on the
  publish path.
  Previously the MuJoCo backend republished `dq_d`/`ddq_d` as the *measured*
  velocity and a filtered measured acceleration, and its `q_d` copy lagged the
  applied command by a physics step. libfranka's default command low-pass filter
  blends every command with these fields, so the lag came straight back out as a
  wobble in the client's own `q_c` — visible, with the new checks on, as spurious
  discontinuity errors from the stock `generate_joint_position_motion` example.
  `dq_d` and `ddq_d` now carry `dq_{c,k-1}` and `ddq_{c,k-1}`: the commanded
  stream's own backward differences in position mode, the commanded value in
  velocity mode, and zero between motions. A client that read `dq_d` expecting
  measured velocity should read `dq` instead.

  Two consequences worth calling out. The **mobile-duo `base` role is excluded**:
  its motion generator is Cartesian, so the fields the client commands and
  libfranka filters are `O_dP_EE_c`/`O_dP_EE_d`, and nothing commands its
  `q_d`/`dq_d`/`ddq_d` at all — they describe the swerve steer/drive joints, so
  the backend's reading is merged through and they track the wheels. And in a
  **torque-only session** (`kExternalController`, no motion generator) `q_d` now
  stays at the standstill setpoint captured when the `Move` was accepted, with
  `dq_d`/`ddq_d` zero, for the whole session; read `q` for the measured pose.
- **A `Move` is refused while a reflex is latched**, with
  `Move::Status::kCommandNotPossibleRejected` — libfranka renders it as "command
  not possible in the current mode (kReflex)". Previously it was accepted, giving
  a state that claimed `kMove` and carried a latched error at once, and (for
  motion limits) a motion whose every command was silently swallowed.
  `AutomaticErrorRecovery` first, as on hardware.
- **A `Move` that arrives while a motion is running preempts it.** The running
  motion is answered `Move::Status::kPreempted` and the robot is recaptured
  before the new motion is seeded. Previously the new motion inherited the old
  one's difference history and its opening waypoint was validated against the
  start-pose tolerance alone, so every extra `Move` bought an unchecked step.
- **Between motions the state reports the internal controller's hold**, not the
  previous motion's last command: `q_d` becomes the held joint positions and
  `dq_d`/`ddq_d`/`tau_J_d` go to zero when a session ends or a new `Move` starts.
  The mobile base's `O_dP_EE_c`/`O_dP_EE_d` are zeroed by its hold for the same
  reason — it really has stopped.
- Every TCP response now goes out under one send lock, and every transition that
  starts, finishes, preempts or aborts a motion runs under one motion-session
  lock keyed to a monotonic motion id. Reflex aborts are raised from the
  state-publish thread *and* the UDP receive thread while the TCP thread answers
  commands: unserialised, a real violation could be answered `kSuccess`, an abort
  could kill the motion that had just replaced the one that violated, a stale
  `motion_generation_finished` datagram could switch a fresh motion's accounting
  off, and two concurrent `sendall` calls could interleave and desynchronise the
  frame stream.
- `AutomaticErrorRecovery` now clears `errors` (`current_errors`) as well as
  returning `robot_mode` to `kIdle`. `reflex_reason` (`last_motion_errors`) is
  deliberately left alone — libfranka defines it as the record of what aborted
  the *previous* motion.

### Fixed

- **The drain gate now waits for the client's answer to be *applied*, not merely
  read — closing the last window in which this server manufactured the
  discontinuity it then aborted on.** An empty socket is not the same as an
  applied command: `_handle_commands` takes the datagram out of the socket at the
  *start* of its turn and writes the commanded echo — `q_d`, `O_T_EE_c`,
  `elbow_c` — at the *end* of it, with the decoding, the communication accounting
  and the whole limit check in between. `_drain_gate` released on the empty
  socket, so it released straight into the one window where the published echo is
  guaranteed stale, and did so precisely on the cycles where it had engaged at
  all: the moment the receive thread drains the last queued datagram is the moment
  it still has all of its work ahead of it. Worse, `comm.command_received` has
  already run by then, so the cycle counts as *answered* and the publish loop does
  not extrapolate over it either — the reference simply freezes for one cycle,
  libfranka's own 100 Hz command filter blends the next command toward the frozen
  value and emits a `(1 - gain)` × one-cycle step, and this server aborts on the
  kink it caused. The gate now also waits on `_commands_in_flight`, a counter the
  receive thread holds up for the whole journey from socket to simulator.

  Measured with a stock libfranka client on the two motion sequences the issue
  was reported against — a Cartesian pose motion opened from an initial pose
  with an elbow command, and a long multi-stage motion sequence — on a host
  loaded with 12 spinning
  cores: **9 spurious aborts in 30 runs before, 0 in 50 runs after.** Every
  pre-fix abort was caught with a ring-buffer dump of the last cycles, and 8 of
  the 9 were this window (the ninth needed a receive-thread stall longer than the
  gate's own 5 ms bound). The dumps are unambiguous: the client's commanded stream
  runs at |d²/dt²| ≈ 0.05 rad/s² for hundreds of cycles and takes a *single*
  one-cycle step on exactly the cycle whose state carried the stale echo, of size
  `(1 - 0.3859) × Δcommand` — 107.171 rad/s² observed against 107.230 predicted,
  114.443 against 114.392, 222.683 against 222.145, and 184.307 m/s² of
  `O_T_EE_c` translation against a predicted 184.3. That last one is the
  "unexplained" `cartesian_motion_generator_velocity_discontinuity` at a few
  hundred m/s² this file listed as a known issue; it is the same mechanism, and it
  is not Cartesian-specific — one of the nine fired in `kJointPosition` with no
  differential IK running at all.
- **The drain gate no longer charges its own cost to the 1 kHz pacer.** It
  returned the couple of microseconds its `poll(0)` and clock reads take even when
  it had not waited at all, and the publish loop adds that to `next_deadline`
  every cycle. The result was a systematic ~2.5 µs of drift per cycle — a median
  inter-state interval of 1.0025 ms and a publish rate pinned near 997.5 Hz in
  *every* control mode, idle included. The gate now returns exactly `0.0` when it
  did not wait; the median interval measures 1.0000 ms.
- **The state publisher no longer manufactures the discontinuity it then
  reports.** Under `--enforce-motion-limits`, an ordinary approach motion would
  intermittently abort mid-flight with
  `joint_motion_generator_velocity_discontinuity` at a
  `control_command_success_rate` of 0.97-0.99 — hundreds of rad/s² attributed to
  a client whose own signal never left the envelope. Three server-side causes,
  all of which end in the same place: **the client's low-pass filter is closed
  around the `q_d` this server publishes.** `ControlLoop<JointPositions>::
  convertMotion` filters every waypoint toward `robot_state.q_d` with a fixed
  1 ms gain (0.386 at libfranka's default 100 Hz cutoff), so a `q_d` that is
  stale by even two cycles pulls the client's *own* commanded stream off its
  trajectory, and the sim differences the kink it caused.
  - **The 1 kHz pacer fired catch-up bursts.** A cycle that overran was made up
    by publishing the next state immediately behind it — two states 60-90 µs
    apart, measured. The client cannot answer the first in the time the second
    takes to arrive, so the second necessarily carried a `q_d` predating that
    answer. States are now spaced at least 0.8 ms apart however long the cycle
    before them ran, and the pacing sleep moved to just before the send so the
    schedule governs the moment the state leaves.
  - **The publish loop ran ahead of its own receive path.** With the UDP receive
    thread descheduled for a few milliseconds, states went out whose `q_d`
    described the last command the server had managed to apply while the answers
    to those very cycles sat unread in its own socket. A new drain gate
    (`FrankaSimServer._drain_gate`), immediately before the accounting and the
    send, holds the state back while a datagram is queued unread — restoring, for
    a stall that hits one thread and not the other, the invariant the
    communication accounting already assumes: a simulator that stalls delays the
    state publish and the client's answer alike.
  - **A late datagram was differenced over one cycle instead of the cycles it
    travelled.** `fresh=False` forced the interval to 1, so a command two cycles
    ahead of the applied history read as twice its own velocity — 16.9 rad/s²
    where the honest interval gives 6.75. It now gets the same server-observed
    interval as any other command; a replay or reordered packet, whose id is no
    newer than the history's, still floors at one cycle.

  Measured with a stock libfranka client on the two most timing-sensitive error
  scenarios, 40 iterations on one idle host: **14 spurious aborts before, 1
  after.** The
  residual needed a stall long enough to freeze the whole process (10-40 ms
  observed) and was the hold-not-extrapolate divergence, which the packet-loss
  extrapolation entry at the top of this release closes.
- **`StopMove` no longer ends the session -- a second `Move` on the same
  connection now actually receives `RobotState`.** `handle_stop_move_command`
  used to clear both `transmitting_state` and `connection_running`, which are
  what the 1 kHz publish loop (and the connection's own watchdog) read to keep
  going; either flag going false ended the UDP broadcast for good, since
  nothing but a fresh `Connect` ever set them back. On the real robot
  `StopMove` ends the *motion*, not the stream -- `RobotState` keeps
  publishing continuously from `Connect` to disconnect. That gap was
  invisible on a single-motion session but broke every session that does a
  second `Move` after a `StopMove` on the same connection: libfranka's
  `ActiveControl` does exactly that on `cancelMotion()` ->
  `StopMove` -> `AutomaticErrorRecovery` -> `Move`, and the TCP command
  thread (`handle_tcp_messages`) does not check either flag, so the second
  `Move`'s `kMotionStarted` still arrived over TCP -- only for the client's
  next UDP receive to hang until it timed out. Neither flag is cleared by
  `StopMove` any more; both are cleared only by an actual disconnect or
  server shutdown, matching the motion-finished-on-its-own path, which never
  stopped the loop either. See `franka_sim/franka_sim_server.py`'s
  `handle_stop_move_command` and `docs/robot-state.md`.

- **A commanded velocity-envelope violation now also latches
  `joint_velocity_violation`.** A `dq_c` ramp past the envelope -- and a `q_c`
  motion that crosses the position limits the same way -- reports
  `joint_velocity_violation` in the
  `ControlException` message on hardware. That pairing is a reported hardware
  behaviour rather than one measured here: hardware raises both errors, with
  `joint_velocity_violation` appearing much earlier because the controller
  shapes the envelope down towards the current velocity. Treat it as strong but
  not independently verified. The commanded-envelope
  check now latches both 3 and 13 unconditionally whenever a motion is
  running, so either name a caller matches on is present; the pure
  safety-controller path (measured-only, e.g. torque mode with no commanded
  velocity at all) is unchanged and still latches 3 alone. See
  `franka_sim/motion_limits.py`'s `Violation.extra_error_index` and
  `docs/robot-state.md`.

- **`AutomaticErrorRecovery` no longer replies before the arm has stopped.**
  The handler used to answer instantly, so a client recovering from a
  high-speed abort (observed at ~2.6 rad/s) could start its next motion while
  the arm was still decelerating under the idle hold; the new motion's own
  client-side start-pose guard would then see measured `q` drifting off the
  commanded `q` it had just sent and throw a spurious "Performance threshold
  reached" a few milliseconds in, on a motion that had done nothing wrong. The
  handler now polls the physics backend directly (not the cached
  `RobotState.state["dq"]`, which only ever reflects whatever the 1 kHz
  publish loop last copied into it) until measured `dq` has stayed below
  0.005 rad/s for 50 consecutive 1 ms
  cycles, capped at 0.7 s (comfortably under libfranka's own 1 s TCP receive
  timeout on the response -- a 3 s cap was tried first and reliably broke the
  connection instead). On timeout it replies success anyway and logs a
  warning, so a caller can never hang. The wait runs on the per-connection TCP
  thread, not the state-publish loop, so it does not stall state broadcast.
  See `docs/robot-state.md` and `docs/compatibility.md`.

- **A `kCartesianPosition` `Move` no longer aborts with a joint error.** The
  limit checker was seeded with whatever control mode the *previous* motion had
  left behind, because `kCartesianPosition` matches none of the server's physics
  branches. It therefore read the zero-filled `q_c` of a Cartesian
  `RobotCommand` as a joint position command and aborted live clients with
  `joint_motion_generator_position_limits_violation` (joint 4's range does not
  contain 0). Every motion-generator check is now gated on the generator the
  accepted `Move` actually asked for: `q_c` only for `kJointPosition`, `dq_c`
  only for `kJointVelocity`, `tau_J_d` only for `kExternalController`, and
  nothing motion-generator-related for `kCartesianPosition`. `O_dP_EE_c` is
  checked only for `kCartesianVelocity` **on a mobile-base server** — a
  single-arm server has no Cartesian-velocity physics branch, so a
  `kCartesianVelocity` `Move` there falls into the same unchecked category as
  `kCartesianPosition` and its twist is not judged at all. The new
  safety-controller check is the one exception and stays armed in all modes.

- **A fast reconnect no longer times out on the new session's first UDP
  receive.** `_handle_commands` used to read `self.udp_socket` at runtime
  instead of the socket it was actually started with, so its hangup/error
  branch cleared `self.connection_running` unconditionally. After a fast
  reconnect that flag belongs to the *new* session -- a stale thread, still
  unwinding on the *old*, closed socket, killed the new session's flag out
  from under it, and its broadcast loop exited before sending a single
  state datagram. The new client then saw `libfranka: UDP receive: Timeout`
  on a connection that never did anything wrong. `start_command_receiver`
  now captures the socket once and passes it to `_handle_commands` as an
  argument; every place that would act on `connection_running` first checks
  that this thread's captured socket `is` still `self.udp_socket`.

- **A `Move` no longer gets a second, unsolicited response.** The broadcast
  loop used to send a bogus `kSuccess` reply for the current motion right
  after its first UDP state datagram -- even though `handle_move_command`
  had already answered the same `Move` with `kMotionStarted` over TCP.
  libfranka never expected that second reply: it sat unread in the client's
  response map keyed by command id, and when the motion later aborted,
  `Robot::throwOnMotionError` found the stale `kSuccess` ahead of the real
  terminal response and raised `ProtocolException("Unexpected reply to a
  Move command")` instead of the intended `ControlException`. The extra
  send is removed; the terminal response (`kSuccess` via `StopMove` or a
  motion-finished datagram, or an abort status via the pack-stamped
  `_pending_move_response` machinery) is still sent exactly once, as before.

- **franka-sim no longer configures the root logger on import.** A
  module-level `logging.basicConfig(level=logging.ERROR)` in
  `franka_sim_server.py` ran as a side effect of importing the library
  (pulled in transitively by `franka_sim/__init__`), installing a root
  handler that won the first-`basicConfig()`-wins race and silently capped
  every other logger -- including `run_server.main()`'s own, explicitly
  guarded `basicConfig()` call -- at `ERROR`. `python -m franka_sim.run_server`
  now shows its `INFO` startup lines (e.g. "Command handler thread started")
  as intended; embedding applications are free to configure logging
  themselves.

- **`StopMove` no longer clobbers a latched `kReflex`.** Since `StopMove` stopped
  ending the session (see above), the publish loop keeps streaming `RobotState`
  after it -- but `handle_stop_move_command` unconditionally wrote `robot_mode`
  back to `kIdle`, so an enforced abort's `errors` stayed latched while
  `robot_mode` claimed the arm was fine for the whole recovery window. It now
  takes the same guard `_finish_motion` already used: `robot_mode` is only
  written to `kIdle` when it is not already `kReflex`, so a reflex survives a
  `StopMove` exactly as it survives a motion finishing normally, until
  `AutomaticErrorRecovery` clears it.

- **A `Move` arriving before the first `RobotState` publish cycle no longer
  false-aborts on its correct opening pose.** The motion-limit checker's
  Cartesian seed (`_motion_limit_seed_state`) read `O_T_EE` out of
  `self.robot_state.state`, which is still the identity the wire struct was
  constructed with until the publish loop's first cycle has run. A `Move`
  landing in that window got its real first pose judged against that identity
  and aborted with `cartesian_position_motion_generator_start_pose_invalid`
  (observed: `Violation` index 16, 0.57 m off). It now reads the physics
  backend directly for `O_T_EE`, the same way `_publish_hold_setpoint` already
  did for `q`; when the backend cannot answer either, the start-pose and
  start-elbow checks are skipped for that motion instead of judging them
  against a guess.

## [0.4.0] - 2026-08-20

### ⚠ Breaking

- **MuJoCo is the default physics engine; Genesis is an optional extra.**
  `pip install franka-sim` now ships MuJoCo (`mujoco>=3.2,<3.3`) and no longer
  pulls `genesis-world`/`torch`/`numba`. To keep the Genesis engine, install
  `pip install 'franka-sim[genesis]'` and pass `--physics genesis`. The
  `--physics` flag now applies to the single-arm path too and defaults to
  `mujoco` everywhere.

### Added

- **Single-arm MuJoCo backend** (`MujocoFrankaSim`): the MuJoCo Menagerie
  FR3 v2 model with the Franka Hand attached at the flange (same transform as
  the Genesis graft), gravity-compensated PD control clipped to the FR3 torque
  limits, contacts enabled, and a true 1 ms physics step at 1.00x real time.
  Serves the physics gripper backend unchanged.
- **Menagerie / COLLADA visuals for the mobile duo**: the arms render with the
  fr3_v2 obj2mjcf visual set, the TMR base and lift are repainted from their
  COLLADA materials (white shell, black skirt, red tail lights; lift in Franka
  white), and collision geoms are hidden from the viewer.
- **Documentation site** (MkDocs Material, deployed to GitHub Pages on push to
  main): install guide, client compatibility, an evidence-based
  RobotState/TCP-command fidelity reference, the mobile FR3 duo + Quest teleop
  guide, and the backends guide.
- **Genesis-free imports**: `import franka_sim` and the whole MuJoCo path
  (single-arm and mobile-duo) work without Genesis installed.

### Fixed

- **Idle hold on session end.** A client dying mid-torque-stream left its last
  commanded torques applied forever, flinging the gravity-compensated arm into
  its joint limits. Every session-end path (motion finish, `StopMove`,
  disconnect, error recovery) now engages a position hold at the measured
  joint angles, matching the real robot's recapture behaviour.
- **Four unanswered TCP commands.** `SetGuidingMode`, `SetEEToK`, `SetNEToEE`
  and `SetLoad` had no handler, hanging real clients forever; they now reply
  `kSuccess`, and the EE_T_K / NE_T_EE / load values are reflected in
  `RobotState`.
- **Mobile-duo gravity compensation was a silent no-op** (`body_gravcomp`
  written after compile); it is now set at MjSpec level before compilation.
- **Genesis mobile-duo performance**: batched whole-entity reads and
  write-on-change control targets take the scene from 0.40x to ~1.0x real
  time headless.

- **MuJoCo physics backend for the mobile duo** (`--physics mujoco`): `MobileDuoMujocoScene` implements the same scene contract the
  runner and the three FCI bridges already consume, so the protocol surface,
  the joint and link names, the initial pose and the reported state are
  unchanged. Genesis' per-call kernel-launch overhead caps the scene at ~0.4x
  real time at its 2.5 ms step; MuJoCo holds **1.00x real time at a 1 ms
  step** — the rate the bridges actually serve — at about a third of one core.
  Contacts are disabled on this path: the
  chassis' URDF collision meshes interpenetrate as authored, and nothing in
  the scene depends on contact (the base pose is integrated kinematically and
  both arms are servo-driven).
- **Mobile FR3 duo simulation** (`--mobile-duo`): one Genesis scene combining
  the TMR mobile base and two FR3 arms (rigidly mounted, so base motion
  carries both arms), served over **three** FCI bridges — one per role
  (`left`, `right`, `base`) — separated by loopback host IP since libfranka
  clients cannot be told to use a port other than 1337. New flags
  `--scene-urdf`, `--mesh-root`, repeated `--bind ROLE=HOST`; see the README
  for the three-IP loopback convention (base `127.0.0.10`, left `.11`, right
  `.12`, spine `.13`) and `scripts/generate_mobile_duo_urdf.sh` for building
  the combined URDF.
- **Swerve kinematics** for the TMR base, ported from
  `franka_mobile::SwerveKinematics`: body-frame twist -> per-module steering
  angle and wheel speed, with pi-ambiguity resolution that minimises steering
  travel from the previous command.
- **TMR platform emulation** (`TMRGenesisSim`, `SwerveBase`): the base pose is
  advanced kinematically from the commanded twist and written to the entity
  every physics step, while the wheel joints are driven so they look and
  report correctly — mirroring how the real TMR master does swerve IK onboard
  and treats the wheel joints as state-report-only. Currently a
  programmatic/manual bring-up target with no CLI entry point;
  `MobileDuoRunner` is the shipped path.
- **`kCartesianVelocity` protocol mode**: the TMR base has no joint interface,
  so it is driven by libfranka's cartesian-velocity motion generator — the
  commanded body-frame twist (`O_dP_EE_c`) is routed straight to the base's
  swerve inverse kinematics.
- **Fake spine REST device** (`franka_sim.spine_stub`, `--spine` /
  `run-franka-spine-stub`): serves the same HTTPS routes as upstream's real
  spine device from a constant-velocity motion model, so
  `franka_spine_server` can run unmodified against the simulator and a REST
  move visibly raises the prismatic lift (and everything mounted on it) in
  the viewer. New flags `--spine-host`, `--spine-port`, `--spine-cert`,
  `--spine-key`; `FRANKA_SIM_SPINE_PORT` for pointing the end-to-end test
  suite at an unprivileged redirected port instead of the real device's
  port 443.

### Fixed

- **Mobile-duo arms were inert under torque control.** Both FR3 arms hung
  motionless in the `--mobile-duo` scene while every layer above them looked
  healthy (controller `FOLLOWING`, effort interfaces claimed, targets arriving
  at the bridge). The base pose and the spine lift are advanced
  *kinematically*, and Genesis' `set_pos`, `set_quat` and `set_dofs_position`
  all default to `zero_velocity=True`, which zeroes the velocity of **every**
  DOF of the entity — not just the ones named. Running once per physics step on
  the one entity that carries both arms, that acted as an infinite damper: a
  2 Nm command travelled 0.03 rad in 6 s instead of 2.3 rad, and position mode
  under-tracked by ~0.09 rad. The base and the lift still moved (they are
  teleported), which is what made the arms look like the only broken part.
  Both pose writes now pass `zero_velocity=False` and zero only the DOFs the
  teleport actually invalidates — the root joint and the spine joint.
  `tests/test_mobile_duo_physics.py` guards this against real physics.
- **Socket-teardown race in `FrankaSimServer.cleanup()`.** Socket attributes
  (`client_socket`, `server_socket`, `command_socket`, `udp_socket`) were
  re-read between their `shutdown()` and `close()` calls, racing a
  concurrent per-connection teardown that can null those attributes. When it
  did, `None.close()` raised `AttributeError` — not caught by the narrow
  `except socket.error` — aborting the rest of cleanup and leaking listeners
  that coexist via `SO_REUSEPORT` and keep receiving traffic bound to a dead
  scene. Each socket is now cached into a local before use.
  `MobileDuoRunner.stop()` had the same fragility one level up: one bridge's
  `stop()` raising skipped the remaining bridges and the shared scene's
  teardown. Each bridge (and the spine stub) now stops in its own isolated
  `try`/`except`, and the shared scene always stops regardless.

## [0.2.0] - 2026-06-16

### ⚠ Breaking

- **libfranka v10 wire protocol.** `Connect` now negotiates protocol version 10,
  `RobotState` is the float-based **1377-byte** v10 layout (was the double-based
  v9 format), and model loading uses **`GetRobotModel`** — the client builds its
  own Pinocchio model from the served URDF — instead of `LoadModelLibrary`.
  Requires **libfranka ≥ 0.18 / FCI system ≥ 5.9**; v9 clients can no longer
  connect.
- **Default robot is now the FR3, not the Panda.** Genesis loads the MuJoCo
  Menagerie FR3 (7-DOF, hand-less), fetched and cached via `robot_descriptions`
  on first use. Adds a `robot_descriptions` dependency.

### Added

- FR3 simulation **calibrated against a real FR3**: the default joint damping is
  tuned so the sim's joint excursions match a logged real-robot
  `joint_impedance_example_controller` run to within ~5%.
- **Realtime**: the physics loop is paced to wall-clock time, and the `-v` GUI
  holds realtime by decoupling rendering from the physics step (viewer capped to
  30 FPS) instead of letting the viewer throttle physics.
- Environment overrides: `FR3_MJCF` (model path) and `FR3_JOINT_DAMPING` (a
  scalar or 7 comma-separated values).
- Real libfranka v10 client integration test.
- Headless mode (no `-v`) steps Genesis correctly (the TCP/UDP server runs in a
  background thread while physics steps in the main thread).

### Changed

- **Performance**: lock-free `RobotState` snapshot handoff (physics thread →
  network thread), a single precompiled `struct.Struct` for packing, monotonic
  `message_id`, and a 1 kHz deadline-paced UDP broadcast.
- `control_command_success_rate` now reports `1.0` (the sim applies every command
  it receives) — it was hardcoded `0.0`, which libfranka's `communication_test`
  and controllers read as a failing controller.

### Fixed

- **Torque-control clients no longer hang on shutdown.** A pure-torque client
  (`startTorqueControl` / libfranka `communication_test`) signals the end via
  `torque_command_finished`; the server only acted on `motion_generation_finished`,
  so the stop was never acknowledged and the client blocked. Both flags are now
  handled.
- **UDP command handler scoped to the connection.** It looped on the server
  lifetime, so each client spawned another command thread while old ones lived on
  and raced on the socket. It now ends with the connection — repeated
  connect/disconnect on one server process is clean.
- Normal client disconnects are logged at `debug`/`info`, not `error`.
- Review fixes: quoted the CI dependency constraint (an unquoted `>=` was parsed
  as a shell redirect), an actionable error when the FR3 model can't be fetched
  offline, `FR3_JOINT_DAMPING` length validation, and a render-burst after a
  physics stall.

### Known limitations

- First run needs network access to fetch the FR3 model (cached afterwards); set
  `FR3_MJCF` to a local MJCF to run fully offline.
- The Menagerie FR3 joint-zero convention differs from `franka_description`, so
  absolute joint angles don't line up with a real FR3 exactly (peak-to-peak
  motion does).
- Soft realtime (~`dt` 2.5 ms physics on a typical laptop CPU), not hard 1 kHz.
- Validated control paths: joint position, joint impedance (torque), gravity
  compensation, communication test. Cartesian, joint-velocity, gripper, and
  error-recovery paths are not yet exercised against the sim.

## [0.1.13] and earlier

Prior `0.1.x` releases predate this changelog.
