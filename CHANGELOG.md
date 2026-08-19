# Changelog

All notable changes to **franka-sim** are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project is pre-1.0,
so per [Semantic Versioning](https://semver.org/) a minor (`0.x`) bump may include
breaking changes — these are called out explicitly.

## [Unreleased]

### Added

- **MuJoCo physics backend for the mobile duo** (`--physics mujoco`, default
  `genesis`): `MobileDuoMujocoScene` implements the same scene contract the
  runner and the three FCI bridges already consume, so the protocol surface,
  the joint and link names, the initial pose and the reported state are
  unchanged. Genesis' per-call kernel-launch overhead caps the scene at ~0.4x
  real time at its 2.5 ms step; MuJoCo holds **1.00x real time at a 1 ms
  step** — the rate the bridges actually serve — at about a third of one core.
  Needs the new `mujoco` extra. Contacts are disabled on this path: the
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
