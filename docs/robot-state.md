# State &amp; command fidelity

A simulator that speaks the real protocol will happily hand your controller a
`RobotState` with every field populated. Some of those numbers come from physics.
Some are an echo of what you just commanded. Some are constants that have never
changed since the connection opened.

**This page tells you which is which, field by field, with code references.** It is
the page to read before you trust a number.

## How state is produced

Every physics backend runs its own thread and publishes a small snapshot dict per
step. The server's broadcast loop merges that snapshot into the `RobotState` once
per tick — roughly 1 kHz — and sends the packed 1377-byte struct over UDP.

All four backends (single-arm MuJoCo, single-arm Genesis, mobile-duo MuJoCo,
mobile-duo Genesis) publish the **same eight keys**:

```text
q  dq  ddq  q_d  dq_d  ddq_d  tau_J  O_T_EE
```

Everything else in `RobotState` is either written directly by a TCP/UDP command
handler, or never written after initialisation — in which case it is a permanent
constant for the life of the connection.

!!! warning "The snapshot merge wins every tick"

    Because the merge runs on every broadcast tick, it **overwrites** anything a
    command handler wrote into a key the snapshot also owns. `dq_d` is the clearest
    casualty: the UDP handler writes your commanded velocity into it, and the next
    physics tick — under a millisecond later — replaces it with measured velocity.

## Legend

| Marker | Meaning |
| --- | --- |
| **physics** | Measured from the physics engine each step. Trust it. |
| **faithful** | Not physics, but semantically identical to the real robot. Trust it. |
| **echo** | A copy of something you commanded. Real information, but about your command, not the robot. |
| **approximation** | Derived, but by a different mechanism than the real robot uses. |
| **stub** | Never written. A permanent constant for the whole connection. |

## RobotState fields

### Joint state

| Field | Kind | What the sim actually reports |
| --- | --- | --- |
| `q` | **physics** | Measured joint position from `qpos` / `get_dofs_position`, every step. |
| `dq` | **physics** | Measured joint velocity from `qvel` / `get_dofs_velocity`. |
| `q_d` | **echo** | The last commanded joint-position target, seeded to measured `q` on the first tick. For a position-mode client this is a correct setpoint echo. **In velocity or torque mode nothing updates it**, so it silently freezes at whatever it last held. |
| `dq_d` | **approximation** | Reports **measured** velocity, not your commanded velocity — the snapshot key is literally `dq_d: dq`. The command handler's echo is clobbered on the next tick, in *every* mode including velocity control. |
| `ddq_d` | **approximation** | A low-pass–filtered numerical derivative of *measured* `dq`. It is filtered measured acceleration reported under a "desired" name — not a planned trajectory value. |
| `theta`, `dtheta` | **stub** | Permanent zero. There is no motor-side / pre-gearbox encoder model; only link-side `q` exists. |

!!! danger "`tau_J` is not a measured joint torque"

    `tau_J` is an **echo of your last commanded torque**. In torque control
    (`kExternalController`) it equals your `tau_J_d`. In position or velocity
    control it is forced to all zeros on every command. It is never read from
    `qfrc_actuator`, contact forces or any torque-sensor model.

    On the real FR3, `tau_J` is the measured link-side joint torque including
    gravity, friction and contact. Any controller that closes a loop on `tau_J`,
    estimates payload from it, or detects contact through it will behave completely
    differently against the sim.

| Field | Kind | What the sim actually reports |
| --- | --- | --- |
| `tau_J` | **echo** | See above — commanded torque, or zero outside torque mode. |
| `tau_J_d` | **echo** | Correct echo of the commanded torque, but **only written in torque mode**. In position/velocity/Cartesian-velocity modes it keeps its previous value (default zeros), unlike the real robot which always reports the controller's torque command. |
| `dtau_J` | **stub** | Permanent zero. |
| `tau_ext_hat_filtered` | **stub** | Permanent zero. There is no external-torque observer. |

### Cartesian state

| Field | Kind | What the sim actually reports |
| --- | --- | --- |
| `O_T_EE` | **physics** | Measured flange (`link7`) pose, converted to a column-major 4×4. Because all the frame offsets below are identity, this is the **flange** pose — *not* a hand or fingertip TCP pose, even with `--gripper-physics`. |
| `O_T_EE` *(mobile-duo `base` role)* | **approximation** | Not physics-measured. Open-loop dead reckoning: `x, y, theta` Euler-integrated from the commanded body twist. No wheel contact or slip feeds back — the pose is exactly what you commanded, integrated. |
| `O_T_EE_d` | **stub** | Permanent identity. |
| `O_T_EE_c` | **stub** | Permanent identity. The commanded Cartesian pose *is* decoded off the wire and then dropped — not even echoed back. |
| `F_T_EE`, `F_T_NE` | **stub** | Permanent identity. The flange↔EE offset is never modelled. |
| `NE_T_EE`, `EE_T_K` | **echo** | Identity until set; `SetNEToEE` / `SetEEToK` values are reflected back in subsequent states, exactly as the real robot reports them — but the frames are not used in any kinematics. |
| `O_dP_EE_d`, `O_dP_EE_c` | **echo** *(base role only)* | On the mobile-duo base's Cartesian-velocity path, both echo the commanded body twist. For arm roles and the single-arm sim they are permanent-zero stubs. |
| `O_ddP_EE_c`, `O_ddP_O` | **stub** | Permanent zero. |
| `elbow`, `elbow_d` | **stub** | Permanent `[0.0, 0.0]`. The 7-DOF elbow redundancy angle is not modelled at all. |
| `elbow_c`, `delbow_c`, `ddelbow_c` | **stub** | Permanent `[0.0, 0.0]`. Decoded off the wire, never stored. |

### Load and inertia

| Field | Kind | What the sim actually reports |
| --- | --- | --- |
| `m_ee`, `I_ee`, `F_x_Cee` | **stub** | Permanent zero. |
| `m_load`, `I_load`, `F_x_Cload` | **echo** | Zero until set; `SetLoad` values are reflected back in subsequent states, as on the real robot — but the load is not added to the simulated dynamics. |

### Contact, collision and forces

Every field in this group is a **stub — permanent zero**. There is no
collision-threshold detection, no contact estimation and no external-wrench
observer in the sim today.

| Field | Kind |
| --- | --- |
| `joint_contact`, `cartesian_contact` | **stub** — permanent zero |
| `joint_collision`, `cartesian_collision` | **stub** — permanent zero |
| `O_F_ext_hat_K`, `K_F_ext_hat_K` | **stub** — permanent zero |
| `accelerometer_top`, `accelerometer_bottom` | **stub** — permanent zero (6 links × 3 axes each; no IMU model) |

!!! note "Contacts *do* run in the single-arm scene"

    MuJoCo simulates contacts for the single-arm model — the arm will physically
    collide with the ground and with the gripper's fingers. What is missing is the
    **reporting**: none of that reaches the `*_contact` / `*_collision` /
    `*_ext_hat_*` wire fields. Contacts are separately *disabled outright* in the
    [mobile-duo scene](mobile-duo.md#limitations).

### Modes, errors and health

| Field | Kind | What the sim actually reports |
| --- | --- | --- |
| `message_id` | **approximation** | A monotonic counter incremented once per broadcast tick, not a hardware clock. The property libfranka needs — strictly increasing — holds. |
| `motion_generator_mode` | **faithful** | Set from the `Move` request's motion-generator mode, reset to `kIdle` on motion end or `StopMove`. Correct. |
| `controller_mode` | **faithful** | Set from the `Move` request's controller mode, reset to `kOther` on motion end. Also gates which physics branch the UDP command loop takes. Correct. |
| `robot_mode` | **approximation** | Toggles between `kIdle` and `kMove` only. `kOther`, `kGuiding`, `kReflex`, `kUserStopped` and `kAutomaticErrorRecovery` exist in the enum but the sim never produces them. |
| `errors` | **stub** | All 41 booleans permanently `false`. No safety or error condition is ever latched — a client can never observe an error state. |
| `reflex_reason` | **stub** | All 41 booleans permanently `false`. |
| `control_command_success_rate` | **stub** | Hard-coded `1.0`, set at construction, never updated. The real robot computes a rolling window of dropped-vs-received commands. This is deliberate: a value that started at zero would read to most controllers as "every command is being dropped". |

## TCP commands

| Command | Status | Behaviour |
| --- | --- | --- |
| `Connect` (0) | **implemented** | Parses version and UDP port, always replies `kSuccess` with `library_version=10`. It does **not** check the client's requested version — a protocol-9 client gets past `Connect` and then fails on the state layout. |
| `Move` (1) | **implemented** | Full support for joint position, joint velocity, torque (`kExternalController`) and — mobile-duo base only — Cartesian velocity. Validates the controller mode and rejects bad payloads with `kInvalidArgumentRejected`. Replies `kMotionStarted` immediately; the `kSuccess` follows from the state loop. |
| `StopMove` (2) | **implemented** | Freezes the sim (arm holds current `q` at zero torque, base commands zero twist), sends a final idle-mode state frame, and unblocks any pending `Move`. |
| `AutomaticErrorRecovery` (10) | **implemented** | Sets `robot_mode = kIdle` and replies `kSuccess`. A pure protocol-level unblock — it clears no errors (none are ever latched) and restores no physics state. This is what lets `franka_hardware` / franka_ros2 finish activation. |
| `GetRobotModel` (11) | **implemented** | Returns the bundled FR3 URDF (or `--urdf`) as UTF-8 with a `kSuccess` byte, for the client to build its Pinocchio model from. |
| `SetCollisionBehavior` (3) | **accepted, not enforced** | Parses and logs the thresholds, replies `kSuccess`. The values are never stored or used — consistent with the contact/collision fields being permanent zeros. |
| `SetJointImpedance` (4) | **accepted, not enforced** | Parses `K_theta`, replies `kSuccess`. Never applied: the position-mode PD gains are fixed constants in the physics backends. |
| `SetCartesianImpedance` (5) | **accepted, not enforced** | Parses `K_x`, replies `kSuccess`. There is no Cartesian impedance mode in the sim at all. |
| `SetGuidingMode` (6) | **accepted, not enforced** | Acknowledged with `kSuccess`; guiding mode is not entered and `robot_mode` never becomes `kGuiding`. |
| `SetEEToK` (7) | **accepted, not enforced** | Acknowledged with `kSuccess`; `EE_T_K` is reflected in `RobotState` but not used in any kinematics. |
| `SetNEToEE` (8) | **accepted, not enforced** | Acknowledged with `kSuccess`; `NE_T_EE` is reflected in `RobotState` (`F_T_NE` stays identity) but not used in any kinematics. |
| `SetLoad` (9) | **accepted, not enforced** | Acknowledged with `kSuccess`; `m_load` / `I_load` / `F_x_Cload` are reflected in `RobotState` but the load is not applied to the dynamics. |

!!! tip "\"Accepted, not enforced\" is a deliberate contract"

    All twelve v10 commands answer, so no libfranka blocking call ever hangs against
    the sim. Seven of them are acknowledgement-only: your client proceeds exactly as
    it would on hardware, but the sim's behaviour does not change. If your
    controller's correctness depends on the arm honouring a collision threshold, an
    impedance gain or a payload, the sim will not reproduce that.

    `kCartesianPosition` is the one remaining trap: the enum accepts it, so `Move`
    returns `kMotionStarted`, but no physics branch handles it and the arm never
    moves.

## Gripper fidelity

The gripper speaks the standard libfranka gripper protocol (version 3, port 1338)
and broadcasts `GripperState` over UDP at 60 Hz.

| Field | Kind | What the sim reports |
| --- | --- | --- |
| `width` | **physics** *(with `--gripper-physics`)* | Summed finger joint positions, polled live from a real PD servo. |
| `width` | **approximation** *(kinematic default)* | The commanded target, snapped instantly. No motion time, and the commanded `speed` is ignored entirely. |
| `max_width` | constant `0.08` m | Correct — matches the real Franka Hand's 8 cm stroke. |
| `is_grasped` | **approximation** | Physics backend: true if the fingers settled *wider* than the commanded close width within an epsilon band — a position-stall heuristic, no force sensing. Kinematic backend: true only if a test-configured object width falls in the epsilon band. The real hand also infers grasp from a width heuristic, so the approach is directionally right, but **neither backend reads a real grip force**, and the commanded `force` is not applied as a limit. |
| `temperature` | **stub** | Constant 30 °C, always. The real hand's temperature rises with duty cycle. |
| `message_id` | **approximation** | Broadcast-loop counter, independent of the arm's. |

!!! danger "`stop()` reopens instead of halting"

    Both gripper backends implement `stop()` by driving the fingers back to fully
    open — it reuses the homing path. **The real Franka Hand halts in place.** A
    client that stops mid-grasp will see the object released in sim and held in
    reality. This is a known divergence, not intended behaviour.

Command coverage: `Connect`, `Homing`, `Move`, `Grasp` and `Stop` are all
implemented on both backends; an unknown command id replies `kFail`. `Homing` and
`Move` on the physics backend block until the fingers settle or a 4 s timeout
elapses.

## Roadmap: communication-constraints emulation

The largest remaining fidelity gap is not a physics gap — it is that franka-sim
currently presents a **perfect communication channel**. On real hardware, the FCI's
1 ms budget is unforgiving, and a controller that works in an ideal channel can
fail on the robot for reasons that have nothing to do with dynamics.

What the real robot does and the sim does not, yet:

* **`message_id` round-trip tracking.** The real robot correlates each command with
  the state packet it answers and detects out-of-order or missing responses.
  franka-sim's `message_id` is a bare tick counter.
* **`control_command_success_rate`.** The real robot reports a rolling window of
  received-vs-expected commands, and libfranka aborts the motion when it falls
  below the threshold. franka-sim reports a hard-coded `1.0`.
* **Extrapolation on packet loss.** When a command packet does not arrive in its
  cycle, the robot extrapolates the previous command rather than holding or
  faulting — behaviour that shapes how a jittery controller actually feels.
* **Discontinuity checks.** libfranka's `rate_limiting.h` publishes the exact
  velocity, acceleration and jerk limits the robot enforces per joint; violating
  them triggers a `cartesian_reflex` / `joint_motion_generator_*` error. The sim
  accepts any command, however discontinuous.
* **The reflex system.** Collision thresholds, `errors`, `reflex_reason` and the
  `kReflex` robot mode — the whole latch-and-recover cycle that
  `AutomaticErrorRecovery` exists to clear.

The encouraging part: **the protocol machinery for all of this is public.** The
limits, the error enum, the rate-limiting formulas and the success-rate semantics
all live in libfranka's own headers, so this is implementable against a written
specification rather than reverse-engineered from a robot. Cycle-accurate emulation
of these constraints is the next major fidelity milestone.

Until then, treat franka-sim as a faithful *protocol and kinematics* simulator with
an idealised channel: it will catch handshake bugs, mode-switching bugs, kinematics
bugs and controller-structure bugs, and it will not catch timing-margin bugs.
