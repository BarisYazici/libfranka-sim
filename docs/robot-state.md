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

!!! note "The commanded fields are the FCI layer's, not the backend's"

    Four of those keys — `q_d`, `dq_d`, `ddq_d`, `tau_J_d` — describe what you
    **commanded**, not what the arm measured, and libfranka's own command filter
    feeds them straight back into your next command. The server therefore owns
    them: the snapshot merge skips them on arm roles, so a backend cannot echo a
    measured value (or its own copy, a physics step late) into the loop you are
    closing. Everything else in the snapshot is merged as published.

    The **mobile-duo `base` role is the exception**: its motion generator is
    Cartesian, so the fields the client commands and libfranka filters are
    `O_dP_EE_c` / `O_dP_EE_d`, which the server owns and writes on every
    datagram. Nothing commands the base bridge's `q_d` / `dq_d` / `ddq_d` — they
    describe the four swerve steer/drive joints its own onboard controller
    servos — so there the backend's reading is merged through, and those fields
    track the wheels.

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
| `q_d` | **faithful** | The last commanded joint-position target — `q_c` for a position-mode client, the *held* previous waypoint through a lost cycle (see [below](#what-the-sim-does-with-a-lost-cycle)), and the internal controller's own hold setpoint (measured `q`) between motions. Owned by the FCI layer on arm roles, so no physics backend overwrites it. In a **torque-only** session (`kExternalController` with no motion generator) nothing commands a joint position at all, so `q_d` stays at the hold setpoint captured when the `Move` was accepted and `dq_d` / `ddq_d` stay zero for the whole session; see the caveat below the table. |
| `dq_d` | **faithful** | `dq_{c,k-1}`: the backward-difference velocity the commanded stream implies in position mode, the commanded `dq_c` in velocity mode, zero between motions. This is the value libfranka documents ("always sent back to the user in the robot state"), and it is what its default command low-pass filter feeds back in. |
| `ddq_d` | **faithful** | `ddq_{c,k-1}`: the backward-difference acceleration of the commanded stream, from the same differencing the [discontinuity checks](#motion-limits-and-discontinuities) run. Zero between motions. Not a measured acceleration. |
| `theta`, `dtheta` | **stub** | Permanent zero. There is no motor-side / pre-gearbox encoder model; only link-side `q` exists. |

!!! warning "`q_d` is frozen for the whole of a torque-only session"

    `startTorqueControl` runs no motion generator, so there is no commanded joint
    position for `q_d` to carry and no commanded derivative for `dq_d` / `ddq_d`.
    The sim reports the standstill setpoint it captured when the `Move` was
    accepted — the pose the internal controller was holding — and leaves it there
    until the session ends.

    That is the reading we believe matches the robot (libfranka's command filter
    for a torque session blends `tau_J_d` only, and never reads the joint fields),
    but **it is an inference, not a citation**: libfranka documents `q_d` as
    "Desired joint position" and says nothing about a generator-less session. It
    is also a change from earlier versions of the sim, which let the backend echo
    the *measured* `q` into `q_d` here. A torque-mode controller that reads `q_d`
    expecting to see the arm move will see a constant instead; read `q`.

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
| `tau_J_d` | **echo** | Correct echo of the commanded torque in torque mode, and reset to zero when the internal controller takes the joints back. In position/velocity/Cartesian-velocity modes it stays zero rather than reporting the impedance controller's torque command, which the real robot does. |
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
| `robot_mode` | **approximation** | `kIdle`, `kMove` and — on a communication-constraints violation — `kReflex`. `kOther`, `kGuiding`, `kUserStopped` and `kAutomaticErrorRecovery` exist in the enum but the sim never produces them. |
| `errors` (`current_errors`) | **approximation** | Twelve of the 41 booleans are real: `communication_constraints_violation` (25) plus the eleven [motion-limit errors](#motion-limits-and-discontinuities) — indices 11–16, 18–20, 32 and 34. Each is latched by the condition it names and cleared by `AutomaticErrorRecovery`. The rest — collision reflexes, power and Cartesian-position limits — are permanently `false`. |
| `reflex_reason` (`last_motion_errors`) | **approximation** | The same twelve bits, latched as the record of what aborted the previous motion. Deliberately *not* cleared by `AutomaticErrorRecovery`, matching libfranka's "the errors that aborted the previous motion". |
| `control_command_success_rate` | **faithful** | The real thing: the fraction of the last 100 control cycles that were answered in time, recomputed every cycle. `0.0` when no control or motion-generator loop is running — which is what the robot reports then, not a stub. See [communication constraints](#communication-constraints) below. |

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

## Communication constraints

On real hardware the FCI's 1 ms budget is unforgiving, and a controller that works
against an ideal channel can fail on the robot for reasons that have nothing to do
with dynamics. franka-sim emulates the accounting half of that, cycle for cycle.

### The cycle contract

Every published `RobotState` opens a cycle. A conforming client answers it with a
`RobotCommand` echoing that state's `message_id` — which is exactly what libfranka
does: `Robot::Impl::updateState` stores the id of each accepted state and
`sendRobotCommand` stamps the next command with it. franka-sim counts a cycle as
**answered** when a command carrying that id arrives before the next state goes
out, and as **lost** otherwise.

Freshness is bounded on **both** sides: an echo older than the open cycle is late,
and an id the server has never published is not an answer to anything. A command
carrying a future id — a bit-flip, a replay, a client whose counter does not
conform — counts as a lost cycle rather than suppressing the accounting forever.
Message ids start at 1, so the very first state can be answered like any other.

There is a sub-cycle bias, in your disfavour and deliberately so. A cycle closes a
few microseconds before the state that opens the next one reaches the socket, and
a command landing in that sliver is charged as late. Crediting it back is not
possible here: in cycle space — and this page keeps no wall clock, so that a
stalled simulator never reads as packet loss — that command is indistinguishable
from one sent by a client running a whole cycle behind, and crediting it credited
both. A permanently-one-cycle-late client would then have read 1.00 for ever.

The accounting is in cycle space, never wall-clock: a physics stall delays the
state and the client's answer alike, so a slow simulator never reads as packet
loss.

A datagram that arrives but is *not* fresh is still applied — within a cycle a
late command is the freshest intent there is — and it is still **checked in
full**, over a single cycle, which is the strictest reading available. What it
never becomes is a *sample* of your trajectory: it is not recorded, so the next
command is still differenced against the last one that answered its own cycle.

### What the sim does with a lost cycle

| Behaviour | Status | Detail |
| --- | --- | --- |
| `control_command_success_rate` | always on | Fraction of the last 100 cycles answered in time, recomputed every cycle. `0.0` between motions. While the window is still filling, the divisor is what it holds — so three answered cycles read `1.0`, not `0.03`. |
| The last command is **held** | always on | Nothing new is dispatched. The position target stays where it was, a commanded velocity or twist stays applied, the torque is reused. |
| `communication_constraints_violation` | **opt-in** | 20 consecutive lost cycles abort the motion. Off by default. |

Holding is the FCI's behaviour for a dropped *controller* packet — "if a
controller command packet is dropped, FCI will reuse the torques of the last
successful received packet" — applied to every signal.

!!! warning "The sim does not extrapolate a missed motion-generator cycle"

    The real FCI does: "the robot takes the previous waypoints and performs a
    linear extrapolation (keep acceleration constant and integrate) for the missed
    time step" (`docs/system_requirements.rst`). franka-sim holds the last
    waypoint instead. Through a gap your commanded target therefore *stops*
    advancing here where on hardware it would keep moving, and the `q_d` you read
    back says so.

    Emulating the extrapolation is a roadmap item. See
    [the discontinuity trap it hides](#the-real-robot-trap-the-sim-cannot-show-you)
    for why that matters beyond fidelity for its own sake.

A held cycle still counts as lost for the success rate: holding the command is not
the same as pretending the packet arrived.

### Enabling the violation

Aborting the motion is opt-in, because a simulator is routinely driven by clients
that are not 1 kHz realtime control loops at all:

```bash
run-franka-sim-server --enforce-comm-constraints
# or
FRANKA_SIM_ENFORCE_COMM_CONSTRAINTS=1 run-franka-sim-server
```

With it on, 20 consecutive lost cycles reproduce the robot's response: the
`communication_constraints_violation` bit is latched in `errors` **and**
`reflex_reason`, `robot_mode` becomes `kReflex`, the pending `Move` is answered
`kReflexAborted` (which libfranka turns into a `ControlException`), and the arm is
recaptured by the idle hold. The state carrying the error reaches the wire
**before** the TCP response does, which is the order libfranka needs:
`throwOnMotionError` keys off `robot_mode != kMove` in the state and only then
blocks for the response.

That ordering holds for **every** reflex the sim raises, including the
[motion-limit](#motion-limits-and-discontinuities) ones. It is not free: a
motion-limit violation is detected on the UDP receive thread and can latch in the
microseconds between the publish loop serialising a state and putting it on the
socket, so the packet already on its way was built before the error existed. The
`kReflexAborted` is therefore held back until a state serialised *after* the
error has actually been sent — at most one extra millisecond — rather than being
released by the next `sendto` whatever it happens to contain.

While a reflex is latched the robot is out of business until you clear it. A
`Move` arriving in that window is refused with
`Move::Status::kCommandNotPossibleRejected` — libfranka renders it as "Move
command rejected: command not possible in the current mode (kReflex)!" — rather
than being accepted into a state that claims `kMove` and a latched fault at the
same time. `AutomaticErrorRecovery` clears `errors`, returns `robot_mode` to
`kIdle` and re-arms the accounting; a new `Move` then runs normally, and a second
violation in the same connection aborts exactly like the first.

The base bridge of the [mobile duo](mobile-duo.md) is held to the same contract —
a body twist is a motion command — and its stop is a zero twist, not a joint hold.

### One `Move` at a time

A `Move` that arrives while a motion is already running **preempts** it: the
running motion is answered `Move::Status::kPreempted`, the robot is recaptured by
the idle hold, and the new motion is seeded from that standstill. Restarting a
motion is not a way to sidestep the checks — without the recapture the second
motion inherited the first one's history and its opening waypoint was validated
against the start-pose tolerance alone, so every extra `Move` bought an unchecked
step.

## Motion limits and discontinuities

The FCI does not accept whatever you send it. Control differentiates every
commanded signal with backward Euler at the 1 ms cycle and stops the motion when
the result leaves the limits libfranka publishes in `rate_limiting.h`. franka-sim
runs the same arithmetic on every command it receives.

### The difference formulas

Straight out of libfranka's `docs/overview.rst`, for a command `q_c[k]` at cycle
`k` with `dt = 0.001`:

```text
velocity      dq_c[k]   = (q_c[k]   - q_c[k-1])   / dt
acceleration  ddq_c[k]  = (dq_c[k]  - dq_c[k-1])  / dt
jerk          dddq_c[k] = (ddq_c[k] - ddq_c[k-1]) / dt
```

The previous values are the ones the sim *applied*, and they are also the `q_d` /
`dq_d` / `ddq_d` it reports back, so you can compute every one of these numbers
before you send the command.

`dt` is one cycle for a conforming 1 kHz client. Two of your datagrams landing in
one poll is the sim's own artefact — the receive loop keeps only the newest — so
the interval is taken from the echoed `message_id` instead of assumed, capped at
three cycles. The cap matters because the id is *your* number: without it, echoing
an id a thousand cycles old would divide every commanded derivative by a thousand
and walk 50 rad/s steps straight past the checks.

### What is checked, and what it latches

| Generator | Check | Limit | Error latched (index) |
| --- | --- | --- | --- |
| joint position | first `q_c` matches the robot's `q_d` | 0.1 rad (sim choice, see below) | `joint_position_motion_generator_start_pose_invalid` (11) |
| joint position | `q_c` inside the joint range | FR3 URDF `<limit lower= upper=>` | `joint_motion_generator_position_limits_violation` (12) |
| joint position / velocity | implied velocity | [position-based](#the-position-based-joint-velocity-limits): 2.62/5.26/4.18 rad/s away from the stops, shrinking to zero at them | `joint_motion_generator_velocity_limits_violation` (13) |
| joint velocity | *no* joint-range check | a rate says nothing about where it lands without integrating it — but the velocity limit above shrinks to zero at the stop, so commanding into a limit is caught as a velocity violation, which is what the robot reports too | — |
| joint position / velocity | implied acceleration | `kMaxJointAcceleration` = 10 − 1e−3 rad/s² | `joint_motion_generator_velocity_discontinuity` (14) |
| joint position / velocity | implied jerk | `kMaxJointJerk` = 5000 − 1e−3 rad/s³ | `joint_motion_generator_acceleration_discontinuity` (15) |
| Cartesian velocity (the [mobile base](mobile-duo.md) twist) | ‖v‖ / ‖ω‖ | `kMaxTranslationalVelocity` 3 − 1e−3 m/s, `kMaxRotationalVelocity` 2.5 − 1e−3 rad/s | `cartesian_motion_generator_velocity_limits_violation` (18) |
| Cartesian velocity | ‖a‖ / ‖α‖ | 9 − 1e−3 m/s², 17 − 1e−3 rad/s² | `cartesian_motion_generator_velocity_discontinuity` (19) |
| Cartesian velocity | ‖jerk‖ | 4500 − 1e−3 m/s³, 8500 − 1e−3 rad/s³ | `cartesian_motion_generator_acceleration_discontinuity` (20) |
| torque | \|τ\| | FR3 URDF `<limit effort=>`: 87/87/87/87/12/12/12 Nm | `tau_J_range_violation` (34) |
| torque | \|dτ/dt\| | `kMaxTorqueRate` = 1000 − 1e−3 Nm/s | `controller_torque_discontinuity` (32) |
| any | every commanded value is finite | NaN and ±∞ are refused | the limits violation of the generator they arrived in |

**Non-finite commands are refused whether or not enforcement is on.** Every
`value > limit` comparison against a NaN is false, so a NaN passes each check
above, poisons the backward differences it would be recorded into — NaN minus
anything stays NaN for the rest of the motion — and reaches the physics backend
and the wire. libfranka will not send one (`lowpassFilter` throws
`std::invalid_argument` on a non-finite value, `src/lowpass_filter.cpp`), so it
can only arrive from a client that is not libfranka. There is no `Error`
enumerator for it, so it is reported as the limits violation of its generator.

The Cartesian limits are compared as *norms* of the translational and rotational
halves, because that is how `limitRate` treats `O_dP_EE_c`. Note the names: a
`velocity_discontinuity` is an **acceleration** limit and an
`acceleration_discontinuity` is a **jerk** limit — libfranka's own naming, kept as
is so a message can be grepped against the real robot's vocabulary.

### The position-based joint velocity limits

The FR3 has no fixed joint velocity limit. What the robot enforces — and what
latches `joint_motion_generator_velocity_limits_violation` — is a
**position-based envelope**: full speed in the middle of the range, shrinking
along a braking parabola to zero at each stop, so that a joint at the limit of
the envelope can always still decelerate inside its range. For joint *i* at
position `q`:

```text
dq_upper(q) = min( dq_max, max(0, -dq_offset + sqrt(2 * a_dec * (q_upper - q))) ) - 1e-3
dq_lower(q) = max(-dq_max, min(0,  dq_offset - sqrt(2 * a_dec * (q - q_lower))) ) + 1e-3
```

with these per-joint parameters, taken from the `<limit>` and
`<position_based_velocity_limits>` elements of the FR3 URDF — the same URDF this
server hands your client over `GetRobotModel`, and exactly what libfranka v10's
`Robot::getUpperJointVelocityLimits` / `getLowerJointVelocityLimits`
(`src/joint_velocity_limits.cpp`) compute from it:

| Joint | range `q_lower … q_upper` [rad] | `dq_max` [rad/s] | `dq_offset` [rad/s] | `a_dec` [rad/s²] |
| --- | --- | --- | --- | --- |
| 1 | −2.7501 … 2.7501 | 2.62 | 0.30 | 6.0 |
| 2 | −1.7918 … 1.7918 | 2.62 | 0.20 | 2.585 |
| 3 | −2.9065 … 2.9065 | 2.62 | 0.20 | 3.5 |
| 4 | −3.0481 … −0.1458 | 2.62 | 0.30 | 4.0 |
| 5 | −2.8101 … 2.8101 | 5.26 | 0.35 | 17.0 |
| 6 | 0.5409 … 4.5205 | 4.18 | 0.35 | 5.5 |
| 7 | −3.0196 … 3.0196 | 5.26 | 0.35 | 17.0 |

Two practical consequences:

* **The nameplate speeds only exist away from the stops.** 2.62 rad/s on joint 1
  is available for |q| ≤ 2.04 rad — the middle three-quarters of the range; near a stop the
  admissible velocity can be a fraction of that, and a trajectory that is
  comfortably below `dq_max` everywhere can still violate the envelope on its
  approach.
* **Commanding into a stop is a *velocity* violation, not a position one** (for
  a velocity generator): the envelope reaches zero at the limit, so any inward
  velocity there is over it — which is why the sim, like the robot, has no
  separate joint-range check on the velocity interface.

!!! note "Franka's spec page states the same law with different anchors"

    [Franka's robot specification](https://frankarobotics.github.io/docs/robot_specifications.html#position-based-velocity-limits)
    publishes this formula anchored at the *hard* mechanical stops (±2.9007 rad
    on joint 1) with correspondingly larger offsets (0.6599 rad/s, …). libfranka
    and the robot's own URDF anchor it at the *soft* limits above with rounder
    offsets; the two parameterizations describe nearly the same envelope.
    franka-sim follows libfranka and the URDF, because that is the code path
    that decides whether a command is refused.

### Checking is always on; aborting is opt-in

Every command is validated, always. A violation logs a rate-limited warning naming
the joint or axis, the value and the limit — once per error class per motion, not
once per cycle:

```
motion limit violated: joint_motion_generator_velocity_discontinuity:
q_c joint 4 = 294.395 rad/s^2, limit 9.999 rad/s^2 (not enforced)
```

Aborting the motion is opt-in, for the same reason the communication violation is:

```bash
run-franka-sim-server --enforce-motion-limits
# or
FRANKA_SIM_ENFORCE_MOTION_LIMITS=1 run-franka-sim-server
```

It is an independent switch from `--enforce-comm-constraints`. With it on, the
violating command is **not applied to physics** — the real robot rejects it — the
matching bit is latched in `errors` *and* `reflex_reason`, `robot_mode` becomes
`kReflex`, the pending `Move` is answered `kReflexAborted`, and the arm is
recaptured by the idle hold. `AutomaticErrorRecovery` clears it and a new `Move`
runs normally.

### Two deliberate departures

* **The start-pose tolerance is a sim choice.** libfranka publishes no value for
  it — the check lives in Control, and only the remedy is documented ("make sure
  that your control loop starts with the last commanded value observed in the
  robot state"). 0.1 rad is loose enough that the simulator's own tracking error
  can never manufacture the error and tight enough to catch a client that jumps
  into a motion from a stale pose.
* **The interval a command is differenced over comes from its echoed
  `message_id`**, not from an assumed millisecond — capped at three cycles. A
  one-to-three cycle gap, which is what the sim's own loss looks like, is
  therefore measured at the rate you actually commanded and passes.

    There is no grace cycle, and there was: the first command after a gap used to
    skip the differential checks entirely, which let the resume waypoint be
    anywhere in the joint range. A full-range teleport reached physics with the
    checker reporting no violation at all.

    The cap is why a *long* gap can still trip a discontinuity. That is the honest
    consequence of holding rather than extrapolating — the robot would have
    carried your trajectory forward and you would resume on a signal it was
    already tracking, while here the history stays where the gap began.

### The real-robot trap the sim cannot show you

This one is worth knowing about even though — *because* — franka-sim does not
reproduce it.

On real hardware a missed cycle is extrapolated: Control continues the motion
signal under constant acceleration for the cycles it did not hear from you. Its
finite-difference history then straddles that extrapolated segment. When you
resume, the difference is taken against a value **Control invented**, not against
the last waypoint you sent — and the result can be an acceleration or jerk step
large enough to trip `joint_motion_generator_velocity_discontinuity` or
`joint_motion_generator_acceleration_discontinuity`.

The trap is that this happens to *correct* clients. A trajectory that is smooth by
construction can trip it. So can resuming on the value the robot itself
extrapolated to. libfranka warns about the mechanism in one line — intermittent
drops "could trigger `discontinuity` errors even when your source signals conform
with the interface specification" (`docs/overview.rst`) — and it is a well-worn
way to lose an afternoon hunting a bug that is not in your trajectory generator at
all. If a controller runs clean against this sim and then trips discontinuity
reflexes on hardware, packet loss is the first thing to look at, not your
splines.

franka-sim **holds** the last command instead of extrapolating it, so it cannot
put you in that state — a gap here freezes the target, and the grace cycle above
covers the resume. Emulating the extrapolation is a roadmap item, and reproducing
this failure class in sim, where you can drop cycles on purpose and watch it
happen, is much of the point of doing it.

## Roadmap: what is still an ideal channel

* **The reflex system.** Collision thresholds and the remaining bits of `errors` /
  `reflex_reason` — `joint_reflex`, `cartesian_reflex`,
  `self_collision_avoidance_violation`, `power_limit_violation` and the Cartesian
  *position* limits, none of which the sim can detect because it models neither
  contact forces nor an inverse-kinematics stage.
* **Cartesian pose and impedance control.** `kCartesianPosition` is not served at
  all, so the pose motion generator's limits and its `cartesian_motion_generator_joint_*`
  errors have nothing to check.
* **Packet-loss extrapolation.** A missed cycle holds the last command here; the
  robot continues the motion signal under constant acceleration. Emulating it is
  what would let you reproduce
  [the discontinuity trap](#the-real-robot-trap-the-sim-cannot-show-you) in sim,
  on demand, instead of meeting it for the first time on hardware.

The encouraging part: **the protocol machinery for all of this is public.** The
limits, the error enum and the rate-limiting formulas all live in libfranka's own
headers, so this is implementable against a written specification rather than
reverse-engineered from a robot.

Until then, treat franka-sim as a faithful *protocol and kinematics* simulator: it
will catch handshake bugs, mode-switching bugs, kinematics bugs,
controller-structure bugs, lost-packet bugs and — now — discontinuity bugs, and it
will not catch collision-reflex bugs.
