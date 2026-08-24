# Client compatibility

franka-sim has no client library. It has a socket. **Anything that speaks
libfranka's wire protocol works** — the sim does not know or care what is on the
other end, and there is no franka-sim-specific API to port your code to.

![An official libfranka example driving the simulated FR3](assets/direct_libfranka_control.gif){ loading=lazy }

/// caption
Unmodified libfranka client code driving the simulated arm.
///

## Protocol version

franka-sim implements **robot server version 10** and **gripper server version 3**
— the current libfranka wire format.

| libfranka | Robot system | Robot / gripper server | franka-sim |
| --- | --- | --- | --- |
| **>= 0.18.0** | **>= 5.9.0** | **10 / 3** | **supported** |
| >= 0.15.0 | >= 5.7.2 | 9 / 3 | not supported |

Protocol 9 and below are a genuinely different wire format — a `double`-based
`RobotState` and a server-side `LoadModelLibrary` instead of `GetRobotModel` — not
a subset. There is no compatibility shim. See the
[Franka software compatibility matrix](https://frankarobotics.github.io/docs/compatibility.html)
for how libfranka releases map to robot system versions.

!!! info "The model is built client-side in v10"

    During connection the client calls `GetRobotModel`, receives a URDF, and builds
    its own Pinocchio model from it. franka-sim serves the bundled hand-less FR3
    URDF (override with `--urdf`). Nothing about the dynamics model comes from the
    server — which is why a client's `franka::Model` behaves identically against
    sim and hardware.

## Verified clients

### Official libfranka (C++) — verified

`tests/test_real_client_integration.py` compiles a real v10 client probe against a
prebuilt libfranka, links it with Eigen, and runs it against a live
`FrankaSimServer`. It asserts the whole wire contract end to end: `Connect`, the
exact 1377-byte packed `RobotState` layout, and a `GetRobotModel` URDF that the
real client's Pinocchio loader can actually build from.

`examples/communication_test` from a libfranka build tree runs against the sim
unmodified:

```bash
./examples/communication_test 127.0.0.1
```

!!! note "The test skips without a libfranka build"

    The real-client tests are skipped unless a prebuilt libfranka, `g++` and
    `/usr/include/eigen3` are present, so they do not run in every CI job — only
    where the library is available.

### franka_ros2 — verified end to end (torque/effort path)

The full `franka_ros2` stack runs against the sim FCI: `franka_hardware` activates,
the gripper action server homes and moves, and
`joint_impedance_example_controller` drives the arm through `ros2_control` at
1 kHz (re-verified live against the current protocol implementation).

Scope of that verification: **torque/effort controllers**. `franka_hardware`'s
torque path reads only measured `q`/`dq` and writes `tau_J_d`, so the sim's
frozen torque-session `q_d` (see the [state reference](robot-state.md)) is
irrelevant to it. The joint-position and joint-velocity command paths *do*
consume `q_d`/`dq_d`/`ddq_d` when their default-off rate limiters are enabled;
those combinations have not been re-verified.

Getting there required one thing the sim now implements specifically for this
stack: **the `AutomaticErrorRecovery` handshake**. `franka_hardware` issues it
during activation and stalls forever if the robot never answers. franka-sim
handles it as a protocol-level state transition (`robot_mode` → `kIdle`, reply
`kSuccess`), which is enough to unblock activation.

The reply is not instant: it waits (up to 0.7 s) for the arm to actually stop
moving before answering, the way real recovery does — see
[`AutomaticErrorRecovery`](robot-state.md#tcp-commands) in the state
reference. That is well inside libfranka's own 1 s TCP receive timeout on the
response, so it does not change anything about this handshake beyond making
its timing closer to the real robot's.

Point the hardware config at the sim and nothing else changes:

```yaml
robot_ip: 127.0.0.1   # was 172.16.0.2
```

!!! warning "A ROS 2 build-environment gotcha, not a sim one"

    A stale `hardware_interface` overlay (4.40.0 built against a 4.44.0 system) can
    segfault `franka_hardware` in `on_init`, which looks exactly like a sim
    failure. Rebuild the franka packages cleanly before blaming the simulator.

### Python bindings — expected to work, not officially verified

[`libfranka-python`](https://github.com/BarisYazici/libfranka-python), `panda-py`
and other libfranka wrappers all sit *on top of* libfranka: they link the same
library and emit the same bytes. There is nothing between them and the sim that
could differ, so they are expected to work — provided the wrapper links a libfranka
>= 0.18.0.

Being precise about the evidence: **officially verified is libfranka C++ and
franka_ros2.** Python bindings are strongly expected to work by construction, but
franka-sim's test suite does not exercise them.

### Your own driver

If you implement protocol v10 yourself, franka-sim is a perfectly good target to
develop against — it is easier to instrument than a real arm, and
[the fidelity reference](robot-state.md) tells you exactly which state fields carry
real information.

## The `startMotion()` handshake

Worth knowing because it is where non-libfranka clients most often stall:

1. Client sends `Move` over TCP; server responds `kMotionStarted`.
2. Client loops receiving UDP `RobotState`, checking `motion_generator_mode` and
   `controller_mode` against what it requested.
3. When both match, `startJointVelocityControl()` (or its siblings) returns.
4. **Only then** does the client start sending real UDP commands.

franka-sim drives that sequence from its state-broadcast loop, so the modes flip on
the first state tick after `Move`.

## Lazy clients and the command cycle

franka-sim now measures whether your client actually keeps up. Every published
`RobotState` opens a cycle, and the sim expects a `RobotCommand` echoing that
state's `message_id` back before the next state goes out — precisely what
libfranka's own control loop sends. Cycles that go unanswered are counted, the
last command is held through them, and the damage is reported in
`control_command_success_rate` (see
[communication constraints](robot-state.md#communication-constraints)).

Two consequences for client authors:

* **A client that does not echo the state's `message_id`** — a hand-rolled driver
  that stamps its own counter, say — reads as 100% packet loss. It still drives the
  robot, and by default nothing stops it, but the reported success rate is honest
  about it. On real hardware such a client would be dropped outright. An id the
  server never published counts as loss too, in case the counter is not merely
  stale but invented — and a client running permanently one cycle behind is
  charged for every one of them, which is the point.
* **A client that pauses mid-motion** sees the success rate fall, and, if the sim
  was started with `--enforce-comm-constraints`, gets the real
  `communication_constraints_violation` after 20 consecutive lost cycles: the
  pending `Move` is answered `kReflexAborted`, libfranka raises a
  `ControlException`, and the client must call `automaticErrorRecovery()` before
  moving again — a `Move` sent before that recovery is refused with
  `kCommandNotPossibleRejected`, exactly as the robot refuses it.

Enforcement is **off by default** — a simulator is routinely driven by scripts and
teleop bridges that are not 1 kHz realtime loops — so no existing client changes
behaviour until you ask for it. Turn it on when you want the sim to reject the
timing habits hardware would reject:

```bash
run-franka-sim-server --enforce-comm-constraints
```

Both strictness switches read an environment variable as well
(`FRANKA_SIM_ENFORCE_COMM_CONSTRAINTS`, `FRANKA_SIM_ENFORCE_MOTION_LIMITS`), and
both have an explicit off switch — `--no-enforce-comm-constraints`,
`--no-enforce-motion-limits` — for a run inside a shell, launch file or container
that exports one.

!!! warning "A missed cycle is *extrapolated* here, as it is on the robot"

    The FCI continues the motion signal under constant acceleration through a
    dropped cycle, and so does franka-sim: one substitute waypoint per missed
    cycle, at the acceleration frozen when the gap began, dispatched to physics
    and published back in `q_d`/`dq_d`/`ddq_d`. Your commanded target keeps
    advancing during a gap here exactly as it would on hardware, and it stops at
    twenty consecutive misses exactly where the robot stops. Always on, whether
    or not either strictness switch is.

    This matters for more than fidelity. The extrapolated segment ends up
    *inside* Control's finite-difference history, so resuming after packet loss
    is differenced against a value Control invented — which is enough to trip
    `joint_motion_generator_velocity_discontinuity` on a client whose own
    trajectory is perfectly smooth, and which is what happens if you resume from
    the last waypoint *you* sent rather than from the `q_d` the robot reports.
    A controller that hits that on the robot now hits it here first. See
    [the real-robot trap you can now reproduce](robot-state.md#the-real-robot-trap-you-can-now-reproduce).

    The one half that does *not* extrapolate is torque, and that is also
    hardware's behaviour: "if a controller command packet is dropped, FCI will
    reuse the torques of the last successful received packet".

!!! note "`control_command_success_rate` is `0.0` when idle"

    Not a bug and not a stub: libfranka documents the field as showing "a value of
    zero if no control or motion generator loop is currently running". `echo_robot_state`
    against the sim prints `0` for the same reason it does against a real robot.

## Discontinuous commands

The second strictness switch, independent of the first. franka-sim differentiates
every commanded signal with backward Euler at the 1 ms cycle and compares the
result against the limits libfranka publishes in `rate_limiting.h` — joint range,
velocity, acceleration, jerk, torque range and torque rate (see
[motion limits](robot-state.md#motion-limits-and-discontinuities) for the full
table and the error each one latches). Only the signal belonging to the motion's
*active* generator is judged, so a Cartesian motion is never blamed for the
zero-filled `q_c` its datagram happens to carry — and, symmetrically, a joint
motion is never blamed for the zero-filled `O_T_EE_c` its datagram carries, which
is not a valid transform at all.

The checking layer is independent of whether the backend can *drive* the
interface: it judges the stream the client sent, so it behaves identically on a
backend that moves the arm from a Cartesian command and one that does not (see
[Cartesian control](#cartesian-control)). The point of the checks is the
*abort*, which is what lets a client provoking a Cartesian error terminate.
With enforcement off they only log, and a client waiting for an
abort will still wait forever.

Two things about the error *names* are worth knowing before you match on them,
because both cost people time on real hardware:

* **The discontinuity name follows the interface, not the derivative.** A step in
  `q_c` is `joint_motion_generator_velocity_discontinuity`; the same step in
  `dq_c` is `joint_motion_generator_acceleration_discontinuity`; a twist step is
  `cartesian_motion_generator_acceleration_discontinuity`. A step also breaks the
  velocity envelope, and the discontinuity wins —
  `..._velocity_limits_violation` is for a signal that is smoothly too fast, not
  for one that jumps.
* **`joint_velocity_violation` is not `joint_motion_generator_velocity_limits_violation`.**
  The first is the safety controller watching *measured* velocity and fires in
  every mode, torque included; the second is Control judging your command. When
  the second trips during a motion, the sim now latches **both** — not a
  pin this project verified directly, but hardware is reported to raise the
  same pair, so match
  on either name and expect to see both in the `ControlException`. See
  [the safety controller](robot-state.md#joint_velocity_violation-3-the-safety-controller).

Checking and reporting are **always on**: a violation logs a rate-limited warning
naming the joint or axis, the value and the limit. Aborting is opt-in:

```bash
run-franka-sim-server --enforce-motion-limits
```

With it on, a command the robot would refuse never reaches the physics, the
matching error is latched, the pending `Move` is answered `kReflexAborted`, and the
client must call `automaticErrorRecovery()` before moving again — the same shape as
a communication violation, with a different bit.

Five things worth knowing before you turn it on:

* **A stepped target is now an error, not a teleport.** Scripts that jump `q_c`
  straight to a goal work fine by default and abort under enforcement. That is the
  point: on hardware they would have aborted too.
* **`q_c = 0` is not a valid FR3 pose.** Joint 4 lives in `[-3.0481, -0.1458]` and
  joint 6 in `[0.5409, 4.5205]`, so an all-zeros command is a
  `joint_motion_generator_position_limits_violation` on the real robot as much as
  here. Test fixtures that stream zeros are the most common thing this catches.
* **A non-finite command is refused either way.** NaN and ±∞ in `q_c`, `dq_c`,
  `tau_J_d` or `O_dP_EE_c` are dropped with a warning whether or not enforcement is
  on, because every limit comparison against a NaN is false and applying one
  poisons the state, the difference history and the wire. libfranka refuses to send
  one at all, so this only ever fires for a hand-rolled client.
* **A late datagram is checked, not waved through.** A command whose echoed
  `message_id` does not answer the cycle it arrived in is still applied, and it is
  differenced over the interval the *server* observed between it and the last
  applied command — never over a wider one than the server has actually published.
  A hand-rolled client that stamps ids ahead of anything it was sent gets no
  discount on the limits: the interval is bounded by the server's own count, so
  inflating the echo buys nothing.
* **A burst of *late* datagrams no longer aborts a conforming client.** Two
  separate causes, both gone:

    The differencing window used to be capped at three cycles
    (`MAX_COALESCED_CYCLES`) and taken from the client's echo, so four
    back-to-back lost datagrams made the sim judge the resumed command over less
    time than it really took and trip a discontinuity for a client that never
    left its own envelope. The window now comes from the server's own
    observation (`MotionLimitChecker.note_published`), so a burst is measured at
    the rate the client actually commanded. The cap survives only as a fallback
    for callers that drive the checker without a publish loop.

    And a receive thread that was descheduled hands over a whole *backlog* of
    real datagrams at once. Each of them is the true answer to a cycle the
    publish loop had already extrapolated, and each has its guess thrown away
    (`MotionLimitChecker.absorb_command`) — for the whole run of losses, not just
    for the first one of it, and as one indivisible step so the publish thread
    cannot land in the middle of it. Stalls of 1 to 11 cycles are pinned to
    resume clean; before, they reported 127 rad/s² and up, linear in the stall.

    This is about datagrams that *arrive late*. A client that genuinely goes
    quiet and then resumes from its own last waypoint is a different thing, and
    it still aborts — deliberately; see the resume trap below.
* **A host stall long enough to freeze the whole process can still trip a
  discontinuity — but now for the same reason it would on hardware.** libfranka
  closes its own low-pass filter around `q_d` — `ControlLoop::convertMotion`
  filters every waypoint toward `robot_state.q_d` with a fixed 1 ms gain — so a
  reference that goes anywhere the client does not expect drags the client's
  *own* commanded stream with it. A frozen one used to be the sim's own doing;
  that is gone, because a missed cycle now extrapolates and the published `q_d`
  keeps advancing along the trajectory (above). What remains is the genuine
  article: if the stall is long enough that the extrapolated reference and the
  client's resumed stream have parted company, the difference between them is a
  real discontinuity and hardware would report it too. Everything else the sim
  can control here is fixed: the publish loop no longer emits two states
  microseconds apart after an overrun, and it no longer runs ahead of its own
  receive path.

## Cartesian control

Both Cartesian motion generators drive the arm on the default MuJoCo single-arm
backend. `kCartesianPosition` streams `O_T_EE_c`, `kCartesianVelocity` streams
`O_dP_EE_c`, and either becomes joint motion through **damped-least-squares
differential IK** (`franka_sim/cartesian_ik.py`) on the MuJoCo Jacobian at the
EE frame — the frame `F_T_EE` defines, so a tool set with `setEE` really is the
point that is being commanded. The joint velocity that comes out goes into the
*same* velocity servo a `kJointVelocity` motion drives, which is what makes the
measured-side safety checks apply unchanged: an over-fast Cartesian command
reaches the joint velocity envelope and aborts with `joint_velocity_violation`
exactly as it does on hardware.

Details worth knowing:

* **The checking layer is unchanged.** IK tracking is a second consumer of the
  accepted command stream, not a filter on it. Every limit check listed under
  [discontinuous commands](#discontinuous-commands) runs first and on the same
  signal it always did.
* **The elbow steers the null space.** `elbow_c[0]` is the redundancy angle
  (joint 3 on an FR3) and is chased inside the Jacobian's null space, so it
  moves the elbow without moving the end effector. `elbow_c[1]`, the branch
  flag, is not followed: flipping it means passing through a singularity, which
  the FCI treats as an error rather than a command.
* **A Cartesian `Move` is refused from a singular configuration**, with
  `Move::Status::kStartAtSingularPoseRejected` — libfranka's "Move command
  rejected: cannot start at singular pose!". The test is the smallest singular
  value of the EE Jacobian at the arm's measured `q` against a threshold of
  0.05; see `SINGULAR_POSE_MIN_SINGULAR_VALUE` for how that number was placed.
  Joint interfaces are *not* refused — driving out of a singularity is the only
  way to leave one.
* **Under `kExternalController` nothing changes.** There the client's torques
  drive the arm and the Cartesian stream is a reference, so the sim stays in
  torque mode exactly as before.
* **The controller mode is not a control law here.** Both internal controllers —
  `kJointImpedance` and `kCartesianImpedance` — drive the arm through the same
  joint-velocity servo, because that is the only law the backends implement.
  `SetCartesianImpedance`'s `K_x` is [accepted and not
  enforced](robot-state.md#tcp-commands), so a client that picks
  `kCartesianImpedance` to get a compliant end effector gets a stiff one that
  tracks its commands instead. What it will *not* get is silence: naming only
  `kJointImpedance` on the dispatch path would leave a `kCartesianImpedance`
  Cartesian motion accepted, checked and completely inert.
* **`O_T_EE` is measured at `link7`**, not at the true flange (`link8`, 0.107 m
  further along z). That is a pre-existing frame choice shared with the Genesis
  backend and the mobile-duo scene, and `F_T_EE` is composed on top of it — so
  the sim is self-consistent, but an *absolute* Cartesian target taken from a
  real robot lands 0.107 m away from where hardware would put it.
* **Not on every backend.** Genesis and the mobile-duo scene view implement no
  IK; there the Cartesian interfaces stay checked-but-inert.

## Multiple robots on one host

libfranka pins the command port to 1337 and gives you no way to change it, so you
cannot run two robots on one address. The fix is one **loopback address per robot**
— `127.0.0.0/8` is entirely loopback on Linux, so any `127.x.y.z` works and no
interface aliasing tricks are needed beyond adding the address:

```bash
sudo ip addr add 127.0.0.11/8 dev lo
sudo ip addr add 127.0.0.12/8 dev lo
```

Then bind one bridge per address:

```bash
run-franka-sim-server --mobile-duo \
  --bind left=127.0.0.11 --bind right=127.0.0.12 --bind base=127.0.0.10 \
  --scene-urdf ... --mesh-root ...
```

and construct one client per address:

```cpp
franka::Robot left("127.0.0.11");
franka::Robot right("127.0.0.12");
```

This is exactly how the [mobile duo](mobile-duo.md) serves three robots from one
physics scene, and it works for independent single-arm servers too — start each one
with its own `--bind`-equivalent address.

## What will *not* work

| | |
| --- | --- |
| libfranka <= 0.17 (protocol 9) | Different `RobotState` layout. Not supported. |
| `franka::Robot::setGuidingMode`, `setEE`, `setK`, `setLoad` | The commands are **accepted and acknowledged**, so your client will not hang — but the values are not enforced by the sim. See [TCP commands](robot-state.md#tcp-commands). |
| Cartesian control on the **Genesis** backend and the **mobile-duo arms** | The arm does not move. Those backends implement no differential IK, so `kCartesianPosition`/`kCartesianVelocity` get `kMotionStarted` and the commanded stream is checked but never applied. The default MuJoCo single-arm backend *does* drive both interfaces — see [Cartesian control](#cartesian-control). |
| Anything relying on contact/collision reporting | The collision and external-force fields are permanent zeros today. See [the field table](robot-state.md#robotstate-fields). |

## Known issue: run-to-run repeatability of velocity generators

**The sim does not reproduce a velocity-commanded motion to the same joint
configuration twice.** Two runs of the same `kJointVelocity`, `kCartesianVelocity`
or torque motion from the same start pose end tens of milliradians apart under
host load. Position generators are unaffected — they close the loop on a
commanded pose and correct whatever the last cycle got wrong — so this is
specifically the *integrating* interfaces.

**Why.** The physics thread paces itself on the wall clock, one 1 ms step per
millisecond of real time. The client's motion profile advances on a different
clock: the `message_id` of the states this server publishes, which is what
libfranka's `Duration` counts. Those two clocks are the same clock on hardware
and are not here — the publish loop sits a few Hz under 1 kHz (a deliberate
consequence of the [minimum state spacing](#lazy-clients-and-the-command-cycle)
that keeps two states from being sent microseconds apart after an overrun), and
it loses more ticks the busier the box is. Every millisecond of slip is one more
physics step of the last commanded velocity than the client's own clock
accounted for, integrated straight into the joint configuration and never
corrected.

**What you will see.** A client that runs one motion sequence through
`robot->control()` and then the *same* sequence through the read/write
active-control interface, and compares where the two land, will see them
diverge. Held to a 5e-3 rad per-joint tolerance, that comparison fails against
the sim roughly 10–30% of the time depending on load, and a long multi-stage
motion sequence occasionally goes with it. **This is not new in
the Cartesian release:** it reproduces on the release before it, where the
divergence enters at the `kJointVelocity` stage instead (0.044 rad observed), and an
interleaved A/B of the two revisions does not separate them (4/12 vs 1/12 failures,
Fisher p ≈ 0.32). Driving the Cartesian generators simply gave the same mechanism
two more motions to accumulate in — an 8 s open-loop twist trajectory is the
longest integration involved.

**What to do about it.** Nothing, if you are running a controller: this is a
repeatability property, not a correctness one, and nothing else is affected.
If you are diffing two runs of the same trajectory, difference the
*commanded* stream rather than the measured configuration, or drive a position
generator. The real fix is to step the physics once per published control cycle
instead of once per wall millisecond, so that the robot's clock is the clock the
client sees — that is on the roadmap, not in this release.

**One residual is not explained.** A `kCartesianPosition` motion very
occasionally aborts mid-flight with
`cartesian_motion_generator_velocity_discontinuity` at a few hundred to ~1500
m/s² on `O_T_EE_c`, on a commanded stream that is smooth by construction (~1 in
15 runs of that A/B comparison observed). The commanded pose is echoed from the
client's own stream, so a stale echo would explain it — but that has not been
demonstrated, and the mechanism above does not account for it. Treat an isolated
Cartesian discontinuity abort as possibly the sim rather than your client, and
please report it.
