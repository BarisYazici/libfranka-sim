# Changelog

All notable changes to **franka-sim** are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). This project is pre-1.0,
so per [Semantic Versioning](https://semver.org/) a minor (`0.x`) bump may include
breaking changes — these are called out explicitly.

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
