# Testing in CI

franka-sim speaks the real FCI wire protocol, which makes it a drop-in robot
for continuous integration: every push can run your actual libfranka or
franka_ros2 code against a simulated FR3 — handshake, 1 kHz state stream,
reflexes and all — on a plain GitHub-hosted runner.

Three ways in, from most to least turnkey.

## GitHub Action

The repository doubles as a composite action that starts the server in Docker
(host networking) and blocks until it passes a real-handshake readiness probe:

```yaml
jobs:
  robot-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Start a simulated FR3
        uses: BarisYazici/libfranka-sim@v1
        with:
          # Everything is optional. Fault injection makes the sim abort
          # motions the way the real robot does — see below.
          args: '--enforce-comm-constraints --enforce-motion-limits'

      - name: Run your tests against it
        run: |
          # Your stack connects to 127.0.0.1 exactly as it would to a robot.
          ./run_my_robot_tests.sh 127.0.0.1
```

Inputs (all optional): `image` / `tag` (defaults
`ghcr.io/barisyazici/franka-sim:latest`), `args` (extra
`run-franka-sim-server` flags), `port`, `container-name`, `cpu-shares` (Docker CPU weight for the simulator, default 4096 so the 1 kHz loop wins contention on small runners), `wait-timeout`.
Outputs: `host`, `port`, `container-name`. The container keeps running for
the rest of the job; inspect it with `docker logs franka-sim`.

Linux runners only (the action runs a Linux container with `--network host`).

## Docker image

The image is self-contained — the FR3 and Franka Hand models are baked in, so
it runs with no network access:

```bash
docker run -d --network host ghcr.io/barisyazici/franka-sim        # latest release
docker run -d --network host ghcr.io/barisyazici/franka-sim:edge   # current main
```

Arguments after the image name go straight to `run-franka-sim-server`:

```bash
docker run -d --network host ghcr.io/barisyazici/franka-sim \
  --enforce-comm-constraints --enforce-motion-limits
```

Prefer `--network host`: the FCI is a 1 kHz UDP loop, and published ports put
Docker's NAT in the middle of it.

To wait for readiness — from CI scripts, compose healthgates, anywhere:

```bash
docker exec <container> franka-sim-check --timeout 30
```

`franka-sim-check` (also installed by `pip install franka-sim`) is not a port
scan: it performs the v10 Connect handshake and waits for a RobotState
datagram, so exit 0 means the server is genuinely serving the FCI. It briefly
occupies the single FCI client slot — like a real `franka::Robot` connection —
so use it as a one-shot readiness gate, not a periodic liveness probe.

## pytest fixture

`pip install franka-sim` registers a pytest plugin. Any test can ask for a
running server; one MuJoCo-backed server (on a free port, so parallel CI jobs
never collide) serves the whole session:

```python
def test_my_controller(franka_sim_server):
    robot = connect_my_stack(franka_sim_server.host, franka_sim_server.port)
    ...
```

Customise the server's flags by overriding one fixture in your `conftest.py`:

```python
import pytest

@pytest.fixture(scope="session")
def franka_sim_server_args():
    return ["--no-gripper", "--enforce-motion-limits"]
```

The gripper server is off by default here because its port (1338) is fixed by
the libfranka gripper protocol; drop `--no-gripper` to enable it, and then run
only one such session per machine.

## Fault injection

The flags that make CI runs adversarial, not just green:

- `--enforce-comm-constraints` — after 20 consecutively lost command cycles
  the motion aborts with `communication_constraints_violation`, exactly as the
  real FCI does. Your reconnect/recovery path gets exercised on every push.
- `--enforce-motion-limits` — a commanded discontinuity (position, velocity,
  acceleration, jerk or torque-rate violation) aborts the motion with the
  matching reflex error instead of being silently followed.

Both are also available as environment variables
(`FRANKA_SIM_ENFORCE_COMM_CONSTRAINTS=1`, `FRANKA_SIM_ENFORCE_MOTION_LIMITS=1`),
which is convenient in container definitions.
