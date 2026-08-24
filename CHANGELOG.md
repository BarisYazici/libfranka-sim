# Changelog

All notable changes to **franka-sim** are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project is pre-1.0,
so per [Semantic Versioning](https://semver.org/) a minor (`0.x`) bump may include
breaking changes — these are called out explicitly.

## [Unreleased]

### Added

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
  - **A missed cycle holds the last applied command**: nothing new is
    dispatched, so the position target stays put, a commanded velocity or twist
    stays applied and the torque is reused — the FCI's own behaviour for a
    dropped *controller* packet, applied to every signal. The real robot
    additionally extrapolates a missed **motion generator** cycle under constant
    acceleration; franka-sim does not, and the divergence (including the
    discontinuity trap it hides on hardware) is documented in
    `docs/robot-state.md`. Emulating it is a roadmap item.
  - Freshness is bounded on **both** sides: an echo older than the open cycle is
    late, and an id the server never published is not an answer at all. A
    datagram that is not fresh is still applied and still checked in full — over
    a single cycle, the strictest reading — but it never becomes the *sample*
    the next command is differenced against. A client running permanently one
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
  the interval its echoed `message_id` says it travelled (capped at three cycles,
  because that interval is the client's own number).
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
    differenced over comes from its echoed `message_id`, capped at three cycles,
    so a genuine one-to-three cycle gap — what the sim's own loss looks like —
    is measured at the rate the client commanded and passes. A long gap can
    still trip a discontinuity, which is the honest consequence of holding
    rather than extrapolating.
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

### Changed

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
- Real libfranka v10 client integration smoke test.
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
