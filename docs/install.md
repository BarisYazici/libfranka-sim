# Installation

## Requirements

| | |
| --- | --- |
| **Python** | 3.11 recommended. The package declares `>=3.8` and CI builds 3.9–3.11; 3.11 is what the project is developed and tested against day to day. |
| **OS** | Linux. The loopback-alias multi-robot pattern and the `--mobile-duo` scene assume Linux networking; macOS wheels are published but only the single-arm path is exercised there. |
| **Physics** | MuJoCo (`>=3.2,<3.3`) is a core dependency — installed for you. Genesis is optional. |
| **Robot model** | The FR3 model comes from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) via `robot_descriptions` and **downloads itself on first run** into `~/.cache/robot_descriptions/`. The first launch therefore needs network access; later launches do not. |

!!! note "The upper MuJoCo pin is deliberate"

    `mujoco>=3.3` removed `MjsLight.directional`, which the viewer's light setup
    uses. The pin is `>=3.2,<3.3` until that call site is ported.

## Install

=== "PyPI (recommended)"

    ```bash
    pip install franka-sim
    ```

    This pulls MuJoCo, the default physics backend, plus `numpy`, `trimesh` and
    `robot_descriptions`. Nothing else is required to run the single-arm FCI or
    the mobile duo.

=== "With the Genesis backend"

    ```bash
    pip install 'franka-sim[genesis]'
    pip install 'libigl<2.6'   # (1)!
    ```

    1. `genesis-world==0.2.1` breaks against `libigl>=2.6`, which changed
       `igl.signed_distance`'s return arity. If Genesis import fails with a tuple
       unpacking error, this pin is the fix.

    The extra adds `genesis-world`, `numba` and `torch` — a multi-gigabyte
    install. You only need it to pass [`--physics genesis`](backends.md).

=== "From source"

    ```bash
    git clone https://github.com/BarisYazici/libfranka-sim.git
    cd libfranka-sim
    pip install -e .
    ```

    Add the dev tooling (pre-commit, black, isort, flake8, pytest-cov) with
    `pip install -e '.[dev]'`, then `./install_hooks.sh` to wire up the git hooks.

!!! tip "Running the installed wheel from a source checkout"

    If you `pip install franka-sim` while a `libfranka-sim` checkout is your
    working directory, the local `./franka_sim` package shadows the installed one
    on `sys.path`. Run from a neutral directory (`cd /tmp`) when you mean to test
    the wheel.

## First run

```bash
# Headless — this is the normal mode
run-franka-sim-server

# With the MuJoCo passive viewer
run-franka-sim-server -v
```

Equivalent from a source checkout: `python -m franka_sim.run_server -v`.

The server prints the bound addresses and then idles until a client connects.
You should see:

* **TCP 1337** — the FCI command socket.
* **UDP** — the state stream, on the port the client advertises in its `Connect`
  request (libfranka picks it; you never configure it).
* **TCP 1338** — the gripper command socket, unless you passed `--no-gripper`.

The viewer is MuJoCo's passive viewer and runs on the main thread while physics
runs on its own. It is a debugging convenience — every feature works headless, and
the sim is designed to be run that way on servers and in CI.

## Gripper (Franka Hand)

The gripper server runs **by default** next to the arm and speaks the standard
libfranka gripper protocol (version 3, port 1338). Drive it with `franka::Gripper`
— `homing()`, `move()`, `grasp()`, `stop()`.

| Flag | Backend | Fingers visible in viewer |
| --- | --- | --- |
| *(default)* | `FrankaHandSim` — kinematic, width tracked analytically | no |
| `--gripper-physics` | simulated finger DOFs with a PD servo | **yes**, fingers move |
| `--no-gripper` | none — arm only, port 1338 not bound | no |

```bash
# Physics-backed hand: the hand mesh is loaded and the finger joints are simulated
run-franka-sim-server -v --gripper-physics
```

The kinematic default snaps the width instantly and ignores the commanded speed;
the physics backend runs a real PD servo and infers grasp from a finger stall. See
[gripper fidelity](robot-state.md#gripper-fidelity) for exactly what each backend
does and does not model.

## CLI reference

Everything below is `run-franka-sim-server <flags>`, or equivalently
`python -m franka_sim.run_server <flags>`.

### General

| Flag | Default | Meaning |
| --- | --- | --- |
| `-v`, `--vis` | off | Open the physics viewer. Headless otherwise. |
| `--physics {mujoco,genesis}` | `mujoco` | Physics backend for the single arm and the mobile-duo scene alike. `genesis` needs the `[genesis]` extra. See [backends](backends.md). |
| `--port PORT` | `1337` | TCP command port. Applies to *every* bridge in `--mobile-duo` mode. Real libfranka clients hard-code 1337, so change this only for tests. |
| `--urdf PATH` | bundled hand-less FR3 | The URDF served to the client through `GetRobotModel`. The client builds its Pinocchio model from this, so it must match the arm the physics loads. |

### Gripper

| Flag | Default | Meaning |
| --- | --- | --- |
| `--no-gripper` | off | Do not start the gripper server on 1338. |
| `--gripper-physics` | off | Use the 9-DOF physics hand instead of the kinematic one. |

### Mobile duo

| Flag | Default | Meaning |
| --- | --- | --- |
| `--mobile-duo` | off | Serve the two-arm mobile scene instead of the single arm: one physics scene, three FCI bridges (`left`, `right`, `base`). |
| `--scene-urdf PATH` | — | The combined `mobile_fr3_duo` URDF loaded into the scene. **Required** with `--mobile-duo`. Generate it with `scripts/generate_mobile_duo_urdf.sh`. |
| `--mesh-root PATH` | the URDF's directory | Package root used to resolve `package://` mesh URIs — a `franka_description` checkout. |
| `--bind ROLE=HOST` | — | Bind one bridge to a host address. Repeat once each for `left`, `right` and `base`; all three are required. |

### Spine (mobile duo only)

| Flag | Default | Meaning |
| --- | --- | --- |
| `--spine` | off | Also run the fake spine REST device in-process, driving `franka_spine_vertical_joint`. Requires `--mobile-duo`. |
| `--spine-host HOST` | `127.0.0.13` | Address the spine stub binds. |
| `--spine-port PORT` | `443` | Port the spine stub binds. Upstream's `SpineApiClient` hard-codes 443 with no port override — leave this alone unless you know why you are changing it. |
| `--spine-cert` / `--spine-key` | auto | TLS certificate/key. A throwaway self-signed pair is generated when omitted. |

The spine stub also runs standalone, without a physics scene, via its own console
script:

```bash
run-franka-spine-stub --host 127.0.0.13 --port 443
```

### Environment variables

| Variable | Meaning |
| --- | --- |
| `FRANKA_SIM_SPINE_PORT` | Test-suite only. `SpineApiClient` hard-codes port 443, which needs root to bind; point the mobile-duo end-to-end tests at an unprivileged port instead (e.g. behind an `iptables` `REDIRECT 443 -> 8443`). |

## A complete mobile-duo launch

Full command line, three bridges plus the spine device — see
[Mobile FR3 duo](mobile-duo.md) for what each piece is:

```bash
# Once per boot: loopback aliases, one per role (127.0.0.0/8 is all loopback)
sudo ip addr add 127.0.0.10/8 dev lo   # base
sudo ip addr add 127.0.0.11/8 dev lo   # left arm
sudo ip addr add 127.0.0.12/8 dev lo   # right arm
sudo ip addr add 127.0.0.13/8 dev lo   # spine

run-franka-sim-server --mobile-duo -v \
  --scene-urdf /path/to/mobile_fr3_duo.urdf \
  --mesh-root  /path/to/franka_description \
  --bind base=127.0.0.10 \
  --bind left=127.0.0.11 \
  --bind right=127.0.0.12 \
  --spine
```

## Troubleshooting

??? failure "Genesis import fails with a tuple-unpacking error"

    `genesis-world==0.2.1` expects `igl.signed_distance` to return 3 values;
    `libigl>=2.6` returns 4. Install `pip install 'libigl<2.6'`.

??? failure "`Address already in use` on 1337"

    Another `run-franka-sim-server` is still running, or a previous one did not
    release the socket. In `--mobile-duo` mode every bridge binds 1337 on its own
    host address — if you skipped an `ip addr add`, two bridges collide on
    `127.0.0.1`.

??? failure "The client connects but times out waiting for state"

    Check the libfranka version. franka-sim answers `Connect` with
    `library_version=10` and does not check the client's requested version, so a
    protocol-9 client gets past `Connect` and then fails to parse the
    (float-based, 1377-byte) `RobotState`. Use libfranka >= 0.18.0.

??? failure "Missing mesh files / the first launch hangs"

    The Menagerie FR3 model downloads on first use. The initial run needs network
    access to populate `~/.cache/robot_descriptions/`.
