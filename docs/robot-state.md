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
| `theta`, `dtheta` | **approximation** | Derived: `theta = q`, `dtheta = dq`. No backend models joint elasticity, so the motor-side encoder reads exactly what the link-side one does. On hardware the two differ by the joint's elastic-element deflection, `q = theta - tau_J / K_joint` -- order 1e-3 rad at ordinary torques for the FR3. |

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
| `O_T_EE` | **physics** | Measured flange (`link7`) pose composed with `F_T_EE`, converted to a column-major 4×4 — the same frame `O_dP_EE`, the EE Jacobian and the Cartesian generators use, so the pose you are told about is the pose the sim controls. With the default identity `F_T_EE` that is the **flange** pose — *not* a hand or fingertip TCP pose, even with `--gripper-physics`; send `SetNEToEE` to move the frame. |
| `O_T_EE` *(mobile-duo `base` role)* | **approximation** | Not physics-measured. Open-loop dead reckoning: `x, y, theta` Euler-integrated from the commanded body twist. No wheel contact or slip feeds back — the pose is exactly what you commanded, integrated. |
| `O_T_EE_d` | **derived** *(idle / joint motion)* / **echo** *(pose motion)* / **frozen** *(twist motion)* | Three regimes. **Idle, between motions, and under a joint generator** it is the **measured** `O_T_EE` — the pose the internal controller is holding, republished every cycle; that is what a libfranka pose generator reads on its first callback (`std::array<double, 16> cmd = state.O_T_EE_d;`). **Under `kCartesianPosition`** it is the commanded pose the generator is tracking, extrapolated through a lost cycle exactly as `q_d` is. **Under `kCartesianVelocity`** the client commands no pose at all, so rather than resume tracking the arm this motion is itself driving, the field is *frozen* at the pose the motion started from and returns to measured tracking the moment the motion ends, however it ends. Freezing is not cosmetic: a commanded field that follows the arm the commands are moving closes a feedback loop through the client's own filter. It used to be a permanent identity, which opened every pose motion metres away from the robot and tripped `cartesian_position_motion_generator_start_pose_invalid` on cycle 0. *(Arm roles only — the mobile-duo base bridge is unchanged.)* |
| `O_T_EE_c` | **derived** *(idle / joint motion)* / **echo** *(pose motion)* / **frozen** *(twist motion)* | The last pose the client commanded, and otherwise exactly `O_T_EE_d` above, regime for regime. libfranka reads this one too: it is the reference its command low-pass filter blends the next `O_T_EE_c` with (`ControlLoop<CartesianPose>::convertMotion`), so a stub here dragged every command the client sent towards whatever the sim invented. The commanded pose is also decoded, differentiated and [checked](#motion-limits-and-discontinuities) against the pose generator's limits, and — on the default MuJoCo backend — [applied](compatibility.md#cartesian-control) through differential IK, so the arm actually follows it. |
| `F_T_EE` | **derived** | `F_T_NE · NE_T_EE`, libfranka's own decomposition (`Robot::setEE`), recomputed whenever `SetNEToEE` moves the frame and reset to identity on a new connection. Not a stub any more: on the MuJoCo backend it is where `O_T_EE`, the EE Jacobian and the measured `O_dP_EE` the [safety controller](#cartesian_velocity_violation-4-the-safety-controllers-other-half) watches are all evaluated, so a tool mounted 0.5 m out really is measured 0.5 m out, lever arm and all. |
| `F_T_NE` | **stub** | Permanent identity — there is no `SetNEToF`-style command on the v10 wire to move it, so the nominal-EE frame always sits on the flange and `F_T_EE` is exactly `NE_T_EE`. |
| `NE_T_EE` | **echo** | Identity until set; `SetNEToEE` values are reflected back in subsequent states, exactly as the real robot reports them — and, unlike `EE_T_K`, they *do* reach the kinematics through `F_T_EE` above. |
| `EE_T_K` | **echo** | Identity until set; `SetEEToK` values are reflected back in subsequent states, but the stiffness frame is not used in any kinematics — nothing in the sim implements Cartesian impedance for it to mean anything to. |
| `O_dP_EE_d`, `O_dP_EE_c` | **echo** *(twist motion)* / **zero** *(otherwise)* | The last twist the client commanded, on both the mobile-duo base's Cartesian-velocity path (a *body* twist) and an arm role's `kCartesianVelocity` (an *end-effector* twist), extrapolated through a lost cycle like every other commanded field. Zero the rest of the time — idle, between motions, and under any other generator — because an arm held by its internal controller is commanding no end-effector motion. **The arm-role echo is new, and its absence was not a harmless stub:** `ControlLoop<CartesianVelocities>::convertMotion` low-passes every commanded twist toward `O_dP_EE_c`, so with the field pinned at zero the filter stopped being a filter and became a constant multiplier — `dt / (dt + 1/(2πf_c))` = **0.386** at libfranka's default 100 Hz cutoff. Every `kCartesianVelocity` client was moving the arm at 39% of the speed it asked for, silently. `O_ddP_EE_c` is still a permanent zero, so a client that opts into libfranka's `limit_rate = true` is feeding its rate limiter a jerk reference the sim never populates, on both Cartesian interfaces — same caveat as `delbow_c` below. |
| `O_ddP_EE_c`, `O_ddP_O` | **stub** | Permanent zero. |
| `elbow` | **derived** | `(q[2], sign(q[3]))` — the redundancy angle *is* joint 3 on an FR3 and the branch flag *is* the sign of joint 4, so this is a reading of `q` rather than a model. It was a permanent `[0.0, 0.0]` stub until the [elbow checks](#motion-limits-and-discontinuities) landed — and a zero branch flag is one libfranka refuses to send (`checkElbow` throws client-side unless it is exactly ±1), which made the elbow interface unreachable from a real client. *(Arm roles only; the mobile-duo base bridge has no elbow.)* |
| `elbow_d`, `elbow_c` | **derived** *(idle)* / **echo** *(Cartesian motion)* | The same `(q[2], sign(q[3]))` reading whenever nothing commands an elbow — idle, between motions, under a joint generator, or under a Cartesian motion whose commands carry no elbow at all (`valid_elbow` clear). While either Cartesian generator *is* streaming one, both echo it, held through lost cycles and snapping back to the measured elbow when the motion ends. `elbow_c` is also decoded and checked on both Cartesian generators (start elbow, sign consistency, velocity/acceleration/jerk). *(Arm roles only.)* |
| `delbow_c`, `ddelbow_c` | **stub** | Permanent `[0.0, 0.0]` — the elbow's commanded derivatives are never published. libfranka reads them only inside `limitRate`, i.e. only when a client opts into rate limiting. |

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
| `errors` (`current_errors`) | **approximation** | Eighteen of the 41 booleans can really be latched: `communication_constraints_violation` (25), the safety controller's `joint_velocity_violation` (3) and `cartesian_velocity_violation` (4), and the [motion-limit errors](#motion-limits-and-discontinuities) at indices 11–22, 31, 32 and 34. Index 24, `start_elbow_sign_inconsistent`, has a name in the error table but is [deliberately never latched](#what-is-checked-and-what-it-latches). Each is latched by the condition it names and cleared by `AutomaticErrorRecovery`. The rest — collision reflexes, power and Cartesian-position limits — are permanently `false`. |
| `reflex_reason` (`last_motion_errors`) | **approximation** | The same bits, latched as the record of what aborted the previous motion. Deliberately *not* cleared by `AutomaticErrorRecovery`, matching libfranka's "the errors that aborted the previous motion". |
| `control_command_success_rate` | **faithful** | The real thing: the fraction of the last 100 control cycles that were answered in time, recomputed every cycle. `0.0` when no control or motion-generator loop is running — which is what the robot reports then, not a stub. See [communication constraints](#communication-constraints) below. |

## TCP commands

| Command | Status | Behaviour |
| --- | --- | --- |
| `Connect` (0) | **implemented** | Parses version and UDP port, always replies `kSuccess` with `library_version=10`. It does **not** check the client's requested version — a protocol-9 client gets past `Connect` and then fails on the state layout. |
| `Move` (1) | **implemented** | Full support for joint position, joint velocity, torque (`kExternalController`) and — mobile-duo base only — Cartesian velocity. Validates the controller mode and rejects bad payloads with `kInvalidArgumentRejected`. Replies `kMotionStarted` immediately; the `kSuccess` follows from the state loop. |
| `StopMove` (2) | **implemented** | Freezes the sim (arm holds current `q` at zero torque, base commands zero twist) and terminates any running `Move` with `kPreempted` — the status the robot sends for a motion a `StopMove` cut short, and the one libfranka's control loop turns into `ControlException("Move command preempted!")`. Ends the *motion*, not the session: the UDP `RobotState` stream keeps publishing (now reporting the idle hold) and the TCP connection stays open, so a client can send another `Move` on the same connection — matching the real FCI, which streams continuously from `Connect` to disconnect. |
| `AutomaticErrorRecovery` (10) | **implemented** | Clears `errors` (but not `reflex_reason`, the record of what aborted the previous motion), re-arms the communication and motion-limit accounting, sets `robot_mode = kIdle`, and re-engages the idle hold. The reply is deferred until the arm is at (near) standstill or a 0.7 s cap elapses — see [Enabling the violation](#enabling-the-violation) below. This is what lets `franka_hardware` / franka_ros2 finish activation. |
| `GetRobotModel` (11) | **implemented** | Returns the bundled FR3 URDF (or `--urdf`) as UTF-8 with a `kSuccess` byte, for the client to build its Pinocchio model from. |
| `SetCollisionBehavior` (3) | **accepted, not enforced** | Parses and logs the thresholds, replies `kSuccess`. The values are never stored or used — consistent with the contact/collision fields being permanent zeros. |
| `SetJointImpedance` (4) | **accepted, not enforced** | Parses `K_theta`, replies `kSuccess`. Never applied: the position-mode PD gains are fixed constants in the physics backends. |
| `SetCartesianImpedance` (5) | **accepted, not enforced** | Parses `K_x`, replies `kSuccess`. There is no Cartesian impedance mode in the sim at all. |
| `SetGuidingMode` (6) | **accepted, not enforced** | Acknowledged with `kSuccess`; guiding mode is not entered and `robot_mode` never becomes `kGuiding`. |
| `SetEEToK` (7) | **accepted, not enforced** | Acknowledged with `kSuccess`; `EE_T_K` is reflected in `RobotState` but not used in any kinematics. |
| `SetNEToEE` (8) | **implemented** | Acknowledged with `kSuccess`; `NE_T_EE` is reflected in `RobotState`, and — `F_T_NE` staying identity — `F_T_EE` is recomputed from it and handed to the physics backend. On the MuJoCo backend that moves the frame `O_T_EE`, the EE Jacobian, the Cartesian generators and the measured-EE-speed check all work in. The Genesis backend and the mobile-duo bridge still only echo it. |
| `SetLoad` (9) | **accepted, not enforced** | Acknowledged with `kSuccess`; `m_load` / `I_load` / `F_x_Cload` are reflected in `RobotState` but the load is not applied to the dynamics. |

!!! tip "\"Accepted, not enforced\" is a deliberate contract"

    All twelve v10 commands answer, so no libfranka blocking call ever hangs against
    the sim. Seven of them are acknowledgement-only: your client proceeds exactly as
    it would on hardware, but the sim's behaviour does not change. If your
    controller's correctness depends on the arm honouring a collision threshold, an
    impedance gain or a payload, the sim will not reproduce that.

    `kCartesianPosition` is the one remaining trap: the enum accepts it, so `Move`
    returns `kMotionStarted`, but no physics branch handles it and the arm never
    moves. It is no longer *silent*, though — the commanded pose and elbow stream
    is validated against the pose generator's full set of hardware limits, and
    under `--enforce-motion-limits` a bad one aborts with the error the robot
    would give. See [what is checked](#what-is-checked-and-what-it-latches).

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
full**, over the same server-observed interval as any other command: the state
it answers is one this server published, so how far it has travelled is not your
claim to make. What it never becomes is a *sample* of your trajectory: it is not
recorded, so the next command is still differenced against the last one that
answered its own cycle.

### What the sim does with a lost cycle

| Behaviour | Status | Detail |
| --- | --- | --- |
| `control_command_success_rate` | always on | Fraction of the last 100 cycles answered in time, recomputed every cycle. `0.0` between motions. While the window is still filling, the divisor is what it holds — so three answered cycles read `1.0`, not `0.03`. |
| The motion generator is **extrapolated** | always on | One substitute waypoint per missed cycle, at the acceleration frozen when the gap began. It is dispatched to physics and published back in `q_d`/`dq_d`/`ddq_d` (and `O_T_EE_c`/`O_T_EE_d` on a pose motion), so the arm keeps moving and the reference you read back keeps advancing. |
| The last torque is **held** | always on | A torque is not a waypoint. "If a controller command packet is dropped, FCI will reuse the torques of the last successful received packet" — so `tau_J_d` simply stays applied. |
| Extrapolation **stops at 20** | always on | Where the robot stops, the reference stops. Past the bound the last extrapolated value holds, so a client that has genuinely gone away leaves the arm standing still rather than flying off along the trajectory it was on. |
| `communication_constraints_violation` | **opt-in** | 20 consecutive lost cycles abort the motion. Off by default; unenforced, the bound is logged instead. |

The extrapolation is what the robot does — "the robot takes the previous
waypoints and performs a linear extrapolation (keep acceleration constant and
integrate) for the missed time step" (`docs/system_requirements.rst`) — and it
runs whether or not either strictness switch is on, because it is not a check.
It is what the reference *is* while you are quiet. The switches still decide only
whether a violation aborts.

The law, per interface, freezes the highest derivative the last two **real**
commands implied and integrates below it:

| Interface | Missed cycle |
| --- | --- |
| `q_c` | semi-implicit Euler, in this order: `v += a dt` first, then `q += v dt`; `a` frozen |
| `dq_c` | `dq += a dt`, `a` frozen — no jerk term |
| `O_T_EE_c` | translation per axis by the same semi-implicit law; rotation composed on the right by the axis-angle increment `(ω + α dt) dt` |
| `O_dP_EE_c` | twist extended at frozen twist-acceleration |
| `elbow_c` | `elbow[0]` on the 1-D position law; `elbow[1]` **held** — a branch flag has no derivative |
| `tau_J_d` | held |

Frozen, not continued. The acceleration is read once, from the backward
differences of the last two commands that actually arrived, and it does not get
re-derived from the sim's own extrapolated samples: doing that compounds, and
integrating *jerk* instead of freezing acceleration turns twenty milliseconds of
silence into a runaway rather than an extrapolation.

!!! warning "The extrapolation is checked, not exempt — and it is not clamped"

    Each substitute waypoint goes through exactly the same limit checks a command
    you sent would. An extrapolation that runs out past the velocity envelope or
    a joint stop latches the same error, and that is deliberate: it is precisely
    the mechanism behind libfranka's warning that intermittent drops "could
    trigger `discontinuity` errors even when your source signals conform with the
    interface specification". A sim that clamped it would be quietly kinder than
    the robot in the one situation you most need it to be honest about. See
    [the real-robot trap](#the-real-robot-trap-you-can-now-reproduce).

An extrapolated cycle still counts as lost for the success rate: continuing the
trajectory is not the same as pretending the packet arrived.

!!! note "A datagram that turns up late replaces the guess it stood in for"

    One deliberate departure from the robot, and it exists because of an earlier
    one. The FCI *drops* a command that missed its 1 ms window; franka-sim
    applies it, because a simulator is routinely driven by clients that are not
    1 kHz control loops and dropping their datagrams would leave the arm inert.

    Those two choices collide once missed cycles are extrapolated: the
    extrapolation for cycle *N* already took one cycle's worth of motion, and
    the datagram that turns up late for cycle *N* carries the same step —
    measured rather than guessed. Differencing the two against each other would
    report a reference that travelled nowhere and then a huge deceleration, on a
    client that did nothing wrong. So when the real answer arrives, the
    extrapolations it supersedes are thrown away and it is differenced against
    the last command that actually arrived, over the interval it really
    travelled.

    Only for a datagram *inside* the extrapolated run. A replay, a duplicate or
    a reordered packet echoes an id the history is already built on, gets no
    rewind, and never touches the history — and a **fresh** command, the client
    resuming after a real pause, is judged against the extrapolated reference
    unrewound. That last one is what makes the trap below fire.

    This rewind only helps a backlog that drains in the order it was sent. Over
    loopback that is the only order it can arrive in, but on a real network a
    stalled receive thread can hand over the same backlog out of order — the
    datagrams that are not the run's oldest still-unanswered id get no rewind
    and are differenced against the reference as it stands, which can draw a
    spurious `discontinuity` report (and, with enforcement on, an abort) partway
    through an otherwise-conforming drain.



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

**The `AutomaticErrorRecovery` reply is deferred until the arm is at (near)
standstill**, matching the real robot: recovery from a fast abort inherently
completes with the arm stopped, because the internal controller has already
been decelerating it since the abort latched. The sim used to reply the
instant the request arrived, so a client recovering from a high-speed abort
(observed here at ~2.6 rad/s) could start its next motion while the arm was
still decelerating under the idle hold, and its own start-pose guard would
then see measured `q` drifting off the commanded `q` it had just sent and
throw a spurious client-side error a few milliseconds later. The handler now
polls the physics backend directly — not the cached `RobotState`, which only
ever reflects whatever the 1 kHz publish loop last copied into it — until
measured `dq` has stayed below 0.005 rad/s for 50 consecutive 1 ms cycles,
capped at **0.7 s**. That cap is
not 3 s: libfranka's own TCP receive timeout on the response
(`libfranka/src/network.h`, `tcp_timeout` defaults to
`std::chrono::seconds(1)`) is a hard 1 s, and a wait that runs past it does
not degrade gracefully — the client's `Poco::TimeoutException` surfaces as
"libfranka: TCP connection got interrupted" and the connection is gone. 0.7 s
leaves headroom under that ceiling; if the arm has not settled by then the
handler replies success anyway and logs a warning, rather than ever risking
the 1 s wire deadline.

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

`dt` is one cycle for a conforming 1 kHz client. When a cycle goes unanswered, or
the receive path gets to a datagram late, the next command the history records
sits further back than a millisecond, so the interval is **the server's own
count** — how many states it has published since the command the history sits at
— instead of an assumed millisecond. Your echoed `message_id` is honoured only up
to that count: it is *your* number, and without the bound, echoing an id a
thousand cycles old would divide every commanded derivative by a thousand and
walk 50 rad/s steps straight past the checks.

### What is checked, and what it latches

Rows are in **precedence order within each generator**: the first row whose
condition trips is the error you get. Only the signal belonging to the motion's
*active* generator is examined — a joint-position motion is judged on `q_c` and
nothing else, and a `kCartesianPosition` motion is judged on `O_T_EE_c` and
`elbow_c` and nothing else.

The safety controller is **not in this table**, because it is not in this
precedence chain. Its three errors — `joint_velocity_violation` (3) on measured
`dq`, `cartesian_velocity_violation` (4) on measured end-effector speed and
`self_collision_avoidance_violation` (2) on the arm's own geometry — are
judged on the state-publish thread, once per cycle, in every control mode
including pure torque; they never preempt any row below and no row below
preempts them. The two places the velocity halves meet each other: a commanded
velocity-envelope violation (13) latches 3 alongside itself, and 4 outranks 3
when a single cycle breaks both. See
[below](#joint_velocity_violation-3-the-safety-controller).

| Generator | Check | Limit | Error latched (index) |
| --- | --- | --- | --- |
| joint position | `q_c` inside the joint range | FR3 URDF `<limit lower= upper=>` | `joint_motion_generator_position_limits_violation` (12) |
| joint position | first `q_c` matches the robot's `q_d` | 0.1 rad (sim choice, see below) | `joint_position_motion_generator_start_pose_invalid` (11) |
| joint position | implied acceleration | `kMaxJointAcceleration` = 10 − 1e−3 rad/s² | `joint_motion_generator_velocity_discontinuity` (14) |
| joint position | implied velocity | [position-based](#the-position-based-joint-velocity-limits): 2.62/5.26/4.18 rad/s away from the stops, shrinking to zero at them | `joint_motion_generator_velocity_limits_violation` (13) |
| joint position | implied jerk | `kMaxJointJerk` = 5000 − 1e−3 rad/s³ | `joint_motion_generator_acceleration_discontinuity` (15) |
| joint velocity | first `dq_c` continues from the robot's `dq_d` | 0.1 rad/s (sim choice) | `joint_motion_generator_acceleration_discontinuity` (15) |
| joint velocity | implied acceleration | `kMaxJointAcceleration` | `joint_motion_generator_acceleration_discontinuity` (15) |
| joint velocity | implied jerk | `kMaxJointJerk` | `joint_motion_generator_acceleration_discontinuity` (15) |
| joint velocity | commanded `dq_c` in the envelope | as above | `joint_motion_generator_velocity_limits_violation` (13) |
| joint velocity | *no* joint-range check | a rate says nothing about where it lands without integrating it — but the velocity limit above shrinks to zero at the stop, so commanding into a limit is caught as a velocity violation, which is what the robot reports too | — |
| Cartesian velocity (the [mobile base](mobile-duo.md) twist) | ‖a‖ / ‖α‖ | 9 − 1e−3 m/s², 17 − 1e−3 rad/s² | `cartesian_motion_generator_acceleration_discontinuity` (20) |
| Cartesian velocity | ‖jerk‖ | 4500 − 1e−3 m/s³, 8500 − 1e−3 rad/s³ | `cartesian_motion_generator_acceleration_discontinuity` (20) |
| Cartesian velocity | ‖v‖ / ‖ω‖ | `kMaxTranslationalVelocity` 3 − 1e−3 m/s, `kMaxRotationalVelocity` 2.5 − 1e−3 rad/s | `cartesian_motion_generator_velocity_limits_violation` (18) |
| Cartesian pose (`kCartesianPosition`) | `O_T_EE_c` is a homogeneous transformation | `franka::isHomogeneousTransformation`: bottom row `[0,0,0,1]`, rows and columns of the rotation block unit-norm to 1e−5 | `cartesian_position_motion_generator_invalid_frame_flag` (31) — refused even with enforcement off |
| Cartesian pose | first `O_T_EE_c` matches the robot's measured `O_T_EE` | 0.05 m / 0.1 rad (sim choices, see below) | `cartesian_position_motion_generator_start_pose_invalid` (16) |
| Cartesian pose | first `elbow_c` matches the robot's `(q[2], sign(q[3]))` | 0.1 rad (sim choice) or a sign mismatch | `cartesian_motion_generator_start_elbow_invalid` (22) |
| Cartesian pose / velocity | `elbow_c[1]` unchanged mid-motion | any flip of the ±1 branch flag | `cartesian_motion_generator_elbow_sign_inconsistent` (21) |
| Cartesian pose | ‖a‖ / ‖α‖ | 9 − 1e−3 m/s², 17 − 1e−3 rad/s² | `cartesian_motion_generator_velocity_discontinuity` (19) |
| Cartesian pose | ‖v‖ / ‖ω‖ | `kMaxTranslationalVelocity` 3 − 1e−3 m/s, `kMaxRotationalVelocity` 2.5 − 1e−3 rad/s | `cartesian_motion_generator_velocity_limits_violation` (18) |
| Cartesian pose | ‖jerk‖ | 4500 − 1e−3 m/s³, 8500 − 1e−3 rad/s³ | `cartesian_motion_generator_acceleration_discontinuity` (20) |
| Cartesian pose / velocity | commanded `elbow_c[0]` in range | joint 3's own FR3 URDF range, ±2.9065 rad — `elbow[0]` *is* joint 3's angle, and this is the bound Franka's 0.3 rad/s² elbow ramp reaches first (at ~4.49 s, half a second before the velocity cap below) | `cartesian_motion_generator_elbow_limit_violation` (17) |
| Cartesian pose / velocity | elbow velocity, acceleration, jerk | `kMaxElbowVelocity` 1.5 − 1e−3 rad/s, `kMaxElbowAcceleration` 10 − 1e−3 rad/s², `kMaxElbowJerk` 5000 − 1e−3 rad/s³ | `cartesian_motion_generator_elbow_limit_violation` (17) |
| torque | \|τ\| | FR3 URDF `<limit effort=>`: 87/87/87/87/12/12/12 Nm | `tau_J_range_violation` (34) |
| torque | \|dτ/dt\| | `kMaxTorqueRate` = 1000 − 1e−3 Nm/s | `controller_torque_discontinuity` (32) |
| any | every commanded value is finite | NaN and ±∞ are refused | the limits violation of the generator they arrived in |

These checks are independent of whether the arm is *driven* from the stream they
judge. On the default MuJoCo single-arm backend both Cartesian generators move
the arm, through differential IK (see
[Cartesian control](compatibility.md#cartesian-control)); on Genesis and the
mobile-duo scene view they stay checked-but-inert. The validation above is
identical either way, because it judges what the client sent.

One index in the enum is deliberately unreachable here:
`start_elbow_sign_inconsistent` (24), which by its name is the start-time twin of
21. Nothing pins it — the observable start-elbow case on hardware perturbs
`elbow[0]` and
leaves the sign alone, and libfranka refuses to *send* a branch flag that is not
exactly ±1 — so both halves of the start-elbow check report 22 instead, the one
index hardware is confirmed to report. See `START_ELBOW_SIGN_INCONSISTENT_INDEX`.

**With enforcement off, a violating Cartesian motion never ends.** The violation
is logged by name, but nothing aborts, so a client waiting for an error waits
forever. The abort is the deliverable here; `--enforce-motion-limits` is what
turns it on.

### Which discontinuity you get depends on the interface

The enum has two discontinuity names per generator family, and which one a
violation gets is **not** decided by the derivative that broke its limit. It is
decided by the derivative *you command*: the robot names a discontinuity one
step above the commanded channel.

| You command | first difference | second difference |
| --- | --- | --- |
| `q_c` | velocity → **14** `joint_motion_generator_velocity_discontinuity` | acceleration → **15** `joint_motion_generator_acceleration_discontinuity` |
| `dq_c` | acceleration → **15** | jerk → **15** |
| `O_T_EE_c` | velocity → **19** `cartesian_motion_generator_velocity_discontinuity` | acceleration → **20** |
| `O_dP_EE_c` | acceleration → **20** `cartesian_motion_generator_acceleration_discontinuity` | jerk → **20** |
| `tau_J_d` | rate → **32** `controller_torque_discontinuity` | — |

The second half of the rule is a precedence: **a discontinuity beats the
velocity-envelope check.** A step is large enough to break the envelope, the
acceleration limit and the jerk limit all at once, and the robot still returns
exactly one name — the discontinuity. So a 1 rad step in `q_c` is 14, never 13; a
50 rad/s step in `dq_c` is 15, never 13 or 14; a twist step is 20, never 18. The
envelope error (13/18) is what you get for a signal that is *smoothly* too fast.

Start-pose outranks every *discontinuity*: a bad **first** cycle of a
joint-position motion is `joint_position_motion_generator_start_pose_invalid`
(11) however large the jump, because you have not commanded a trajectory yet,
you have commanded a place to begin. It does **not** outrank the joint-range
check — `_check_position` tests `q_c` against the joint range (12) before it
reaches the first-command branch at all, so an opening waypoint that is both
out of range *and* away from `q_d` is reported as 12, not 11.

That ordering (12 before 11) is a sim choice with **no hardware evidence behind
it**. The reference provocation for 11 offsets
`q_d` by 0.2 rad on joint 1, which is comfortably
inside the joint's range, so it exercises 11 alone and says nothing about which
name a first waypoint breaking both should get. If hardware evidence ever turns
up saying start-pose comes first, the fix is to reorder the code, not this
paragraph.

Ordering between `tau_J_range_violation` (34) and `controller_torque_discontinuity`
(32) for a command that breaks both is the one precedence here that is **not**
pinned to hardware evidence; the sim reports the range violation. See the comment
on `MotionLimitChecker._check_torque`.

### `joint_velocity_violation` (3): the safety controller

`joint_velocity_violation` is a **different error** from
`joint_motion_generator_velocity_limits_violation` (13), and the difference is
what it looks at:

* **13 is Control judging your command.** The velocity you asked for — implied by
  `q_c`, or written directly as `dq_c` — is outside the envelope.
* **3 is the robot watching itself.** The *measured* `dq` is outside the envelope,
  whatever you commanded. It fires in every control mode, including
  `startTorqueControl`, where there is no commanded velocity at all: a torque
  that accelerates a joint past the envelope trips it.

When 13 trips while a motion is running, **the sim latches 3 alongside it, in
the same abort** — both bits set in `errors`/`reflex_reason`, so the
`ControlException` your client sees names both errors. That pairing is a
reported hardware behaviour rather than one this project measured directly:
hardware raises **both** errors for this pair — for the position-limits and the
velocity-limits provocation alike — rather than picking one name, with 3 the one
a caller notices first because the limit-shaping controller reacts to the arm
leaving the envelope before Control finishes objecting to the command that put
it there. Treat it as strong but not independently verified.
The sim latches both unconditionally so that either name a caller matches on
is present.

The **pure safety-controller path** is unaffected: a violation raised by
`check_measured_velocity` alone (measured `dq` outside the envelope with no
commanded-envelope check involved at all, e.g. `startTorqueControl`, where
there is no commanded velocity to judge) still latches 3 by itself, with no
13 to pair it with.

The sim allows a 0.1 rad/s margin above the envelope before firing, because the
measured signal comes out of a physics integrator and carries settling ring and
contact spikes that the analytic envelope does not. That is ~4 % of the tightest
envelope value the FR3 has in free space, far below the excursions that matter
and far above the jitter a conforming motion produces. The margin is not applied
when the only question is *which name* an already-certain violation gets.

Like every other check here, this one always logs and only aborts under
`--enforce-motion-limits`; the abort is identical to the others — the bit in
`errors`/`reflex_reason`, `kReflex`, and the pending `Move` answered
`kReflexAborted`. It is skipped on a mobile-*base* server, whose steering and
drive joints are not FR3 joints; the duo's arms are ordinary arm servers and do
get it.

### `cartesian_velocity_violation` (4): the safety controller's other half

The same idea one frame out: the safety controller also watches how fast the
**end effector** is actually travelling, and latches
`cartesian_velocity_violation` when it leaves the FR3's translational Cartesian
limit — `franka::kMaxTranslationalVelocity`, 3 − 1e−3 m/s. Measured, not
commanded, so it is armed in every control mode, torque included; the speed
comes from the MuJoCo Jacobian at the frame `F_T_EE` defines, so a tool set with
`setEE` really is the point being watched.

**MuJoCo backend only.** The check reads a measured end-effector velocity that
only the MuJoCo arm backend publishes; the Genesis backend and the mobile-duo
scene's arm roles publish none, so on those the check simply never fires — no
false positives, no coverage. `joint_velocity_violation` (3) is unaffected and
stays armed on every arm role.

Two things about it are worth stating, because both are read off hardware rather
than assumed:

* **Translation only.** The obvious companion bound would be
  `kMaxRotationalVelocity` (2.5 rad/s) on the EE's angular velocity, and
  hardware behaviour rules it out: a 3 Nm torque ramp and a 5 rad/s² `dq_c`
  ramp both spin *joint 6*
  — whose axis is near enough the EE's own — through its 4.18 rad/s envelope,
  so the EE angular speed passes 2.5 rad/s well before the joint limit, and
  hardware still answers `joint_velocity_violation`. Elbow speed is excluded
  too; the enum's only elbow-speed error is the motion generator's (17).
* **It outranks `joint_velocity_violation`.** Move the EE 0.5 m out along the
  flange with `setEE` and then run an ordinary joint-velocity ramp: hardware
  reports `cartesian_velocity_violation` **alone**, with no
  `joint_velocity_violation` beside it. So when a single cycle breaks both, 4 is
  the bit that latches.

No tolerance is added on top, unlike the 0.1 rad/s allowed on measured `dq`.
That margin exists because ordinary motions ride right up against the joint
envelope; 3 m/s of end-effector travel is an order of magnitude beyond anything
a conforming motion reaches.

**Non-finite commands are refused whether or not enforcement is on.** Every
`value > limit` comparison against a NaN is false, so a NaN passes each check
above, poisons the backward differences it would be recorded into — NaN minus
anything stays NaN for the rest of the motion — and reaches the physics backend
and the wire. libfranka will not send one (`lowpassFilter` throws
`std::invalid_argument` on a non-finite value, `src/lowpass_filter.cpp`), so it
can only arrive from a client that is not libfranka. There is no `Error`
enumerator for it, so it is reported as the limits violation of its generator.

The Cartesian limits are compared as *norms* of the translational and rotational
halves, because that is how `limitRate` treats `O_dP_EE_c`. Note the names: on a
*position* interface a `velocity_discontinuity` is an **acceleration** limit and
an `acceleration_discontinuity` is a **jerk** limit — libfranka's own naming,
kept as is so a message can be grepped against the real robot's vocabulary. On a
*velocity* interface everything shifts by one; see the table above.

### `self_collision_avoidance_violation` (2): the geometric half

The third measured-side check, and the only one that is about *shape* rather
than speed: the safety controller also watches the arm folding onto itself, and
latches `self_collision_avoidance_violation` when two links that are not
neighbours come within **50 mm** of each other. Measured, not commanded, so like
the two velocity halves it is armed in every control mode — including pure
external torque control, which is one of the two ways the reference provocation
runs it.

**It fires before the links touch, and that is the point.** The real controller
does not test the visual meshes; it tests simplified volumes inflated around
each link — the same simplification `franka_description` ships as its collision
meshes — so the reflex has a built-in safety offset and stops the arm short of
contact. The sim gets that offset from MuJoCo's contact `margin` rather than by
fattening geometry: setting `margin == gap` on the arm's collision geoms widens
the set of contacts MuJoCo *reports* to everything within 50 mm while leaving
the set it *simulates*, and every contact force in it, bit for bit unchanged.
Nobody's physics moves; only the sim's field of view widens.

Two calibration facts about the 50 mm, both measured on the sim's own FR3
(`robot_descriptions` `fr3_v2`) over the monitored pairs:

* **The provocation reaches 24.7 mm and never touches.** Folding joint 4 at
  0.1 rad/s while twisting joint 5 at 0.2 rad/s brings link5 down onto link1,
  and the convex hulls stay apart for the whole motion — so a detector waiting
  for an actual contact would wait for ever. Closed loop through the backend's
  velocity servo, 50 mm is crossed at t = 10.80 s.
* **It has to beat the other error that scenario can raise.** The same fold ends
  with joint 4 against its position limit, where the
  [position-based envelope](#the-position-based-joint-velocity-limits) collapses
  towards zero and the commanded −0.1 rad/s becomes
  `joint_motion_generator_velocity_limits_violation`. On the same run that
  happens at t = 11.63 s — 833 control cycles later.

Ordinary operation is nowhere near it: across the home pose and three reference
start poses no monitored pair comes closer than 136 mm, and three of the four
have no monitored pair within 200 mm at all.

**Neighbours are excluded, on measurement rather than principle.** Only links at
least three apart in the chain are monitored. Adjacent links touch by
construction (link0 and link1 sit ~1 mm apart in every configuration). The
once-removed pairs are excluded because they behave the same way: link5 and
link7 are **10–22 mm** apart in *every* configuration — their relative pose
depends only on joints 6 and 7 — which is inside the margin and closer than the
provocation ever gets; link2/link4 and link3/link5 sit at ~70 mm. The wrist's
joint limits, not a reflex, are what hold those apart.

**The grafted hand is not monitored.** With `--gripper-physics` the Franka Hand
sits 26–68 mm off link5 in ordinary poses, inside the margin, so including it
would report a self-collision on a freshly built arm and make the reflex differ
between the two builds. It is an end effector rather than an arm link — and the
reference provocation twists joint 5 precisely to get the gripper "out of the
way … so self-collision between links can be detected".

**MuJoCo backend only**, exactly like `cartesian_velocity_violation` (4): the
Genesis backend and the mobile-duo scene publish no such reading, and a backend
that publishes none is left alone rather than read as "the arm is clear".

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
q_c joint 4 = 294.395 rad/s^2, limit 10 rad/s^2 (not enforced)
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
  into a motion from a stale pose. The reference counter-example on hardware is a
  motion that opens with `std::array<double, 7> discontinuity{{0.2}}`, i.e.
  **+0.2 rad on joint 1** — and libfranka does not attenuate it: the
  command low-pass takes its reference from the command itself on the first cycle
  (`initialized_filter_` is false, `ControlLoop<JointPositions>::convertMotion`),
  so the sim sees the full 0.2 rad. 0.1 rad therefore catches it with
  2× margin, and nothing observed constrains it from below.

    The Cartesian start checks follow that precedent rather than inventing one:
    **0.05 m and 0.1 rad** for the first `O_T_EE_c` against the measured
    `O_T_EE`, and **0.1 rad** for the first `elbow_c` against `q[2]` — the
    rotation and elbow numbers are the joint-space 0.1 rad reused, since both are
    angles, and 0.05 m is its translational counterpart. Nothing pinned is
    sensitive to the choice: the reference hardware provocations offset by 10 m
    and 0.5 rad, three orders of magnitude and five times clear of them. All three are
    constructor arguments on `MotionLimitChecker` if your client needs a
    different contract.

    One more Cartesian departure, in the other direction: libfranka scales its
    *own* rotational limits by `kFactorCartesianRotationPoseInterface` (0.99)
    when it rate-limits a `CartesianPose` client-side. The sim does **not** apply
    that 1% on the server side — nothing published says Control's own bound is
    0.99× rather than the plain `kMaxRotational*`, and shrinking it here would
    refuse a rotation libfranka itself considers legal at the boundary.
* **The interval a command is differenced over is the server's own observation**
  of how many states it published since the applied command, not an assumed
  millisecond and not your echo (which bounds it from above, never widens it). A
  command that reaches the sim late is therefore measured over the time it
  really took, not over an assumed millisecond — so a stalled receive path costs
  you nothing.

    What a *gap* costs you is a separate question, and the answer is not "nothing":
    the reference keeps moving through one under frozen acceleration, so it is
    your own signal's **jerk** that decides whether you resume clean. A stream
    whose acceleration is genuinely constant is extrapolated onto its own next
    waypoints exactly, to the last bit, and resumes reporting exactly the
    acceleration it was commanding — for gaps of 1 to 19 cycles, which is the
    whole window before the robot stops anyway. A stream still ramping its
    acceleration parts company with the frozen one at a rate set by its jerk, and
    a long enough gap will trip a discontinuity. That is the hardware behaviour,
    and the trap below is what it looks like.

    There is no grace cycle, and there was: the first command after a gap used to
    skip the differential checks entirely, which let the resume waypoint be
    anywhere in the joint range. A full-range teleport reached physics with the
    checker reporting no violation at all.

    A *long* gap no longer inflates anything either, because the history does not
    stand still through one: every missed cycle is extrapolated and recorded, so
    the interval back to the applied command is one millisecond again the moment
    you resume — exactly as on hardware.

### The real-robot trap you can now reproduce

This one is a well-worn way to lose an afternoon, and franka-sim reproduces it.

On real hardware a missed cycle is extrapolated: Control continues the motion
signal under constant acceleration for the cycles it did not hear from you. Its
finite-difference history then straddles that extrapolated segment. When you
resume, the difference is taken against a value **Control invented**, not against
the last waypoint you sent — and the result can be an acceleration or jerk step
large enough to trip `joint_motion_generator_velocity_discontinuity` or
`joint_motion_generator_acceleration_discontinuity`.

The trap is that this happens to *correct* clients. A trajectory that is smooth by
construction can trip it. So can resuming on the value **you** last sent, which is
the natural thing for a paused control loop to do: the robot's reference did not
pause with you, so picking up where you left off commands a step backwards the
size of the whole gap. libfranka warns about the mechanism in one line —
intermittent drops "could trigger `discontinuity` errors even when your source
signals conform with the interface specification" (`docs/overview.rst`).

The sim used to **hold** the last command instead of extrapolating it, so it could
not put you in that state — a gap here froze the target, the resume looked clean,
and the one bug a sim2real user most needs to find was exactly the one this
channel could not show them. It does now. Drop cycles on purpose, resume from your
own last waypoint, and the sim latches the discontinuity hardware would:

```text
motion limit violated: joint_motion_generator_velocity_discontinuity:
q_c joint 1 = -763.94 rad/s^2, limit 10 rad/s^2
```

Resume from the `q_d` in the robot state instead — the field libfranka documents
as carrying `q_{c,k-1}` "even in case of packet losses", which during a gap is
the value the sim extrapolated to — and the same resume differences clean over
the standard millisecond. That is the whole remedy, and it is the same one that
works on the robot.

If a controller runs clean against this sim and then trips discontinuity reflexes
on hardware, packet loss is still worth looking at before your splines — but the
sim will now find it for you first if you ask it to.

## Roadmap: what is still an ideal channel

* **The reflex system.** Collision thresholds and the remaining bits of `errors` /
  `reflex_reason` — `joint_reflex`, `cartesian_reflex`, `power_limit_violation`
  and the Cartesian *position* limits, none of which the sim can detect because
  it models neither contact forces nor an external-force estimator.
  (`self_collision_avoidance_violation` is no longer on this list — see
  [above](#self_collision_avoidance_violation-2-the-geometric-half).)
* **Cartesian *impedance* control.** `kCartesianImpedance` as a controller mode
  is accepted but the sim always servos in joint space; and
  `setJointImpedance` / `setCartesianImpedance` are acknowledged without
  changing the servo gains, so a client that lowers its stiffness sees the same
  tracking it saw before. The Cartesian *motion generators* themselves are now
  driven — see [Cartesian control](compatibility.md#cartesian-control).
* **The `cartesian_motion_generator_joint_*` errors** (indices 27–30), which are
  about the joint trajectory the internal IK produces. The sim's IK produces one
  now, but those four errors are not raised from it: a Cartesian command that
  drives a joint into its stop currently surfaces as the safety controller's
  `joint_velocity_violation` — which is also what real hardware reports when a
  Cartesian motion generator is driven into the joint position limits.

The encouraging part: **the protocol machinery for all of this is public.** The
limits, the error enum and the rate-limiting formulas all live in libfranka's own
headers, so this is implementable against a written specification rather than
reverse-engineered from a robot.

Until then, treat franka-sim as a faithful *protocol and kinematics* simulator: it
will catch handshake bugs, mode-switching bugs, kinematics bugs,
controller-structure bugs, lost-packet bugs and — now — discontinuity bugs, and it
will not catch collision-reflex bugs.
