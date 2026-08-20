# Physics backends

franka-sim runs the same scenes on two physics engines behind one interface. The
protocol surface, the joint and link names, the initial pose and the reported state
are identical between them — **a client cannot tell them apart from the wire.** Only
the physics differs, and how fast it runs.

```bash
run-franka-sim-server                     # MuJoCo (default)
run-franka-sim-server --physics genesis   # Genesis
```

## Short version

| | MuJoCo | Genesis |
| --- | --- | --- |
| **Install** | core dependency — `pip install franka-sim` | extra — `pip install 'franka-sim[genesis]'` |
| **Physics step** | **1 ms** | 2.5 ms |
| **Real-time factor, single arm** | 1.00x | ~1.0x at its 2.5 ms step |
| **Real-time factor, mobile duo** | **1.00x**, ~⅓ of one core | ~0.4x |
| **Contacts, single arm** | on | on |
| **Contacts, mobile duo** | off (see below) | off |
| **Viewer** | MuJoCo passive viewer | Genesis viewer |
| **Runs on GPU** | no | yes |
| **Install size** | small | multi-GB (`torch`, `numba`) |

**Pick MuJoCo** unless you have a specific reason not to. It is the default because
it is the only one that holds real time at the rate the FCI actually serves.

**Pick Genesis** when you want GPU-parallel scenes for reinforcement learning — many
environments stepped at once — where per-scene wall-clock latency stops mattering
and throughput starts to.

## MuJoCo (default)

The model is the MuJoCo Menagerie `franka_fr3_v2` arm, **unmodified**: its
kinematics, inertias, joint damping, armature and Coulomb friction are the ones
Menagerie calibrated against the real FR3. With the gripper enabled, the Menagerie
Franka Hand is grafted onto the flange through MjSpec attach, at the same transform
the Genesis model uses — so the two backends agree on where the fingers are.

The decisive property is the step size. MuJoCo's zero-copy `qpos`/`qvel` views let
it hold **one physics step per commanded control cycle at 1 ms**, which is what the
FCI serves. No sub-stepping, no interpolation between control cycles, no drift
between what the client thinks it commanded and what the physics saw.

On the mobile duo it holds **1.00x real time at a 1 ms step on roughly a third of
one core** — leaving the rest of the machine for the client stack, which on a
teleop rig is doing considerably more work than the simulator.

!!! note "Contacts on the duo path are disabled"

    For the single arm, contacts are simulated. For the
    [mobile duo](mobile-duo.md#limitations) they are switched off: the chassis'
    URDF collision meshes interpenetrate as authored, and nothing in that scene
    depends on contact — the base pose is integrated kinematically and both arms
    are servo-driven.

### The realtime-factor monitor

Both backends measure their achieved real-time factor after pacing and log it. When
physics keeps up, each iteration is padded back to `dt` and the RTF sits at ~1.0;
when physics falls behind, the sleep disappears and the RTF drops below 1 — and the
monitor reports the truth rather than quietly dropping the backlog.

This matters more than it sounds. A simulator that silently runs at 0.4x still
serves a well-formed 1 kHz state stream to its client; the client's controller sees
correct-looking packets and simply experiences a robot that responds in slow motion.
Watch the RTF log line, not the packet rate.

## Genesis (optional extra)

```bash
pip install 'franka-sim[genesis]'
pip install 'libigl<2.6'   # genesis-world 0.2.1 needs this pin
run-franka-sim-server --physics genesis
```

Genesis was franka-sim's original backend and remains fully supported for both the
single arm and the mobile duo. Its strength is GPU-parallel simulation: many scenes
stepped simultaneously, which is the shape reinforcement-learning workloads want.

Its weakness, for this particular workload, is per-call kernel-launch overhead. The
FCI bridge reads and writes joint state every control cycle, and each of those calls
carries a fixed cost that dominates the actual physics.

??? info "Where the time actually goes — the honest RTF story"

    Profiling the mobile-duo scene at its slowest point found the physics itself
    accounting for about **12%** of wall-clock time. The rest was call overhead on
    the read/write path — not solver work.

    That diagnosis is what made the performance work tractable. Batching the
    per-joint reads into single calls moved the duo scene from 0.40x to 0.87x real
    time; moving the whole scene to MuJoCo's zero-copy state views got it to 1.00x
    at a 1 ms step. Notably, **moving Genesis to the GPU made it worse** — the
    overhead is per call, not per FLOP, so adding a device transfer to each call
    made every call more expensive while the GPU sat idle.

    The general lesson, if you are profiling your own scene: measure before you
    optimise, and be suspicious when a "slow simulator" turns out to be spending
    88% of its time not simulating.

Practical consequence: Genesis needs a 2.5 ms step to hold real time on the single
arm, and still lands around 0.4x on the mobile duo. The FCI bridges serve at 1 kHz
regardless, so on Genesis the physics is being sub-sampled relative to the control
rate.
