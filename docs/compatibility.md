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

!!! warning "A missed cycle is *held* here, not extrapolated"

    The real FCI continues the motion signal under constant acceleration through a
    dropped cycle; franka-sim holds the last command instead. Your commanded target
    therefore stops advancing during a gap here where on hardware it would keep
    moving.

    This matters for more than fidelity. On hardware the extrapolated segment ends
    up *inside* Control's finite-difference history, so resuming after packet loss
    is differenced against a value Control invented — which is enough to trip
    `joint_motion_generator_velocity_discontinuity` on a client whose own
    trajectory is perfectly smooth. A controller that runs clean against this sim
    can still hit that on the robot, and packet loss, not your splines, is the
    first place to look. See
    [the real-robot trap the sim cannot show you](robot-state.md#the-real-robot-trap-the-sim-cannot-show-you);
    emulating the extrapolation, so the failure is reproducible in sim, is a
    roadmap item.

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
table and the error each one latches).

Checking and reporting are **always on**: a violation logs a rate-limited warning
naming the joint or axis, the value and the limit. Aborting is opt-in:

```bash
run-franka-sim-server --enforce-motion-limits
```

With it on, a command the robot would refuse never reaches the physics, the
matching error is latched, the pending `Move` is answered `kReflexAborted`, and the
client must call `automaticErrorRecovery()` before moving again — the same shape as
a communication violation, with a different bit.

Three things worth knowing before you turn it on:

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
  `message_id` does not answer the cycle it arrived in is still applied, but it is
  judged over a single cycle — the strictest reading — rather than over whatever
  interval its own echo claims. A hand-rolled client that stamps stale ids gets no
  discount on the limits.

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
| Cartesian **position** control | `kCartesianPosition` is accepted by the enum but no physics branch handles it: the client gets `kMotionStarted` and the arm never moves. Cartesian **velocity** works, but only for the mobile duo's base role. |
| Anything relying on contact/collision reporting | The collision and external-force fields are permanent zeros today. See [the field table](robot-state.md#robotstate-fields). |
