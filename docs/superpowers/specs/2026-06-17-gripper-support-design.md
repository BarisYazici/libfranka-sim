# Gripper support — design

**Date:** 2026-06-17
**Status:** Approved (brainstorming)
**Scope:** First-step Franka Hand gripper support for libfranka-sim, with a modular backend
interface that leaves a clean seam for a future Robotiq gripper.

## Goal

Make the simulator wire-compatible with libfranka's hardcoded gripper client (`franka::Gripper`),
so a stock client can `Connect / homing / move / grasp / stop / readOnce` against the sim exactly
as it would against a real Franka Hand. Keep the *backend* (what physically moves) swappable so a
Robotiq gripper can be added later behind the same interface, with the freedom to use its own
richer, non-libfranka transport.

## Decisions (from brainstorming)

1. **Fidelity:** kinematic model first — the backend tracks `width` / `is_grasped` analytically.
   No hand mesh or contact physics in this step. Physical fingers in Genesis can drop in later
   behind the same `GripperBackend` interface.
2. **Topology:** same process — one launch starts both the robot (port 1337) and gripper
   (port 1338) listeners with a shared lifecycle, mirroring how a real Franka exposes both on one
   IP. The gripper runs as a separate listener thread, decoupled from the robot's command loop.
3. **Scope:** implement the Franka Hand now behind a `GripperBackend` interface; Robotiq is
   design-only (documented extension point), not implemented in this step.

## Wire protocol (verified against `libfranka_new`, `research_interface::gripper`, `kVersion = 3`)

The gripper uses a **separate TCP connection** on **port 1338** (`gripper::kCommandPort`), distinct
from the robot's 1337. Everything below is under `#pragma pack(push, 1)` — no alignment padding.

### Command header (10 bytes)

`<HII` = `command (uint16) + command_id (uint32) + size (uint32)`, where `size` is the total
message length including the 10-byte header.

`GripperCommand` enum (uint16): `kConnect=0, kHoming=1, kGrasp=2, kMove=3, kStop=4`.

### Per-command framing

| Command | Request payload | Request msg size | Response payload | Response msg size |
|---|---|---|---|---|
| Connect | `<HH` version, udp_port | 14 | `<HH` status, version | 14 |
| Homing  | none (header only) | 10 | `<H` status | 12 |
| Grasp   | `<ddddd` width, eps_inner, eps_outer, speed, force | 50 | `<H` status | 12 |
| Move    | `<dd` width, speed | 26 | `<H` status | 12 |
| Stop    | none (header only) | 10 | `<H` status | 12 |

Note: commands with an empty request (`Homing`, `Stop`) send the header only — libfranka's
`CommandMessage<RequestBase<T>>` specialization carries no payload.

### Status enums (uint16)

- Connect: `kSuccess=0, kIncompatibleLibraryVersion=1`.
- All other commands: `kSuccess=0, kFail=1, kUnsuccessful=2, kAborted=3`.

### Command/response semantics

Each command sends **exactly one** request and the client blocks for **exactly one** response
(`Gripper::executeCommand`) — unlike the robot's two-phase `Move` (`kMotionStarted` then
`kSuccess`). The client maps the response status: `kSuccess→true`, `kUnsuccessful→false`,
`kFail→CommandException`, `kAborted→CommandException`.

### GripperState (UDP, 23 bytes)

Broadcast **one-way** server→client to the `udp_port` the client gave in its Connect request. The
gripper has **no UDP command channel** (simpler than the arm). Layout `<IddBH`:

`message_id (uint32) + width (double) + max_width (double) + is_grasped (bool, 1 byte) +
temperature (uint16)` = 4 + 8 + 8 + 1 + 2 = **23 bytes**.

`Gripper::readOnce()` drains buffered UDP packets (non-blocking) then blocking-receives one
`GripperState`, so the server must broadcast it continuously (fresh-enough, not 1 kHz).

## Architecture

Three new modules mirroring the existing robot split (`franka_protocol.py` /
`franka_sim_server.py`), plus one integration point.

```
franka_sim/
  gripper_protocol.py   # pure wire format (enums, header, parsers, GripperState pack, responses)
  gripper_backend.py    # GripperBackend ABC + FrankaHandSim kinematic model  <-- modular seam
  gripper_server.py     # FrankaGripperServer: TCP 1338 + one-way UDP broadcast
  franka_sim_server.py  # gains enable_gripper; owns + starts/stops a FrankaGripperServer
  run_server.py         # gains --no-gripper flag
  __init__.py           # exports the new classes
```

### `gripper_protocol.py` — wire format (no I/O)

- Constants: `GRIPPER_COMMAND_PORT = 1338`, `GRIPPER_VERSION = 3`.
- Enums: `GripperCommand`, `GripperStatus`, `GripperConnectStatus`.
- `GripperCommandHeader` dataclass with `from_bytes` / `to_bytes` (`<HII`).
- Request parsers: `ConnectRequest.from_bytes` (`<HH`), `MoveRequest.from_bytes` (`<dd`),
  `GraspRequest.from_bytes` (`<ddddd`). (Homing/Stop have no payload.)
- `GripperState` dataclass + `pack(message_id)` → 23 bytes (`<IddBH`).
- Response builders that return full framed messages (header + payload):
  `build_connect_response(command_id, status, version)` (14 bytes) and
  `build_command_response(command, command_id, status)` (12 bytes).

### `gripper_backend.py` — the modular backend

```python
@dataclass
class GripperStateData:
    width: float
    max_width: float
    is_grasped: bool
    temperature: int

class GripperBackend(ABC):
    def homing(self) -> bool: ...
    def move(self, width: float, speed: float) -> bool: ...
    def grasp(self, width: float, epsilon_inner: float, epsilon_outer: float,
              speed: float, force: float) -> bool: ...
    def stop(self) -> bool: ...
    def get_state(self) -> GripperStateData: ...
```

`FrankaHandSim(GripperBackend)` — kinematic Franka Hand:

- Construction: `max_width=0.08`, `temperature=30`, optional `object_width=None`.
  `set_object_width(w)` hook for tests / future grasp scenarios.
- Pure logic — **no sockets, no threads, no sleeps** — so it is trivially unit-testable.
- `homing()` → `width = max_width`, `is_grasped = False`; return `True`.
- `move(width, speed)` → `width = clamp(width, 0, max_width)`, `is_grasped = False`; return `True`.
- `grasp(width, eps_inner, eps_outer, speed, force)`:
  - If `object_width` is set and `width - eps_inner <= object_width <= width + eps_outer`:
    `width = object_width`, `is_grasped = True`; return `True` (→ `kSuccess`).
  - Else: `width = clamp(width, 0, max_width)`, `is_grasped = False`; return `False`
    (→ `kUnsuccessful`).
- `stop()` → return `True` (aborts a motion; no-op in the instant model).
- `get_state()` → `GripperStateData(width, max_width, is_grasped, temperature)`.

Motion is **instant** in this step (width jumps to target on command completion). Timed servo
motion and physical/contact grasping are non-goals here and can be added later behind this same
interface.

A module docstring documents the Robotiq extension point: a `RobotiqBackend(GripperBackend)`
implements the same five methods; because libfranka's wire client only talks to
`FrankaGripperServer`, a Robotiq integration is free to expose a different/richer transport
(e.g. a native Python API or its own protocol) while reusing the backend contract.

### `gripper_server.py` — `FrankaGripperServer`

- `__init__(host="0.0.0.0", port=1338, backend=None)` — `backend` defaults to `FrankaHandSim()`
  (dependency injection for tests and for swapping backends).
- `run_server()` — TCP accept loop with `SO_REUSEADDR` / `SO_REUSEPORT` and an accept timeout,
  matching the robot server's patterns. On Connect: parse the request, reply
  `build_connect_response(kSuccess, GRIPPER_VERSION)`, record the client's `udp_port`, and start a
  per-connection UDP broadcaster thread that sends `GripperState` to `(client_ip, udp_port)` at a
  modest rate (~30–60 Hz) with a monotonically increasing `message_id`.
- Per command (`Homing/Move/Grasp/Stop`): parse payload, call the backend, map the result to a
  status (`True→kSuccess`, `False→kUnsuccessful`, exception→`kFail`), reply one response.
- Lifecycle: `stop()` / cleanup and per-connection reset mirror `FrankaSimServer`. A client
  disconnect resets per-connection state, stops that connection's UDP broadcaster, and keeps
  listening.

### Integration into `FrankaSimServer`

- New constructor arg `enable_gripper=True`. When set, construct
  `self.gripper_server = FrankaGripperServer(...)`.
- In `start()`, launch `self.gripper_server.run_server()` in a daemon thread alongside the robot
  listener (the robot listener already runs in a background thread while Genesis steps in the main
  thread). The gripper server is self-contained (own backend, own sockets), so it does not touch
  the Genesis physics loop in this step.
- `stop()` also stops the gripper server.
- `run_server.py` gains `--no-gripper` to disable it. `__init__.py` exports `FrankaGripperServer`,
  `GripperBackend`, `FrankaHandSim`.

## Error handling

- Connect with a version mismatch: accept and echo `GRIPPER_VERSION` (the stock client sends v3);
  reserve `kIncompatibleLibraryVersion` for a genuine mismatch.
- Short / malformed payload or unknown command id: reply `kFail`, log, keep the connection alive.
- Backend raising: reply `kFail`, log.
- Client disconnect (clean close or reset): reset per-connection state, stop the UDP broadcaster,
  continue accepting. Normal disconnects log at debug/info, not error (consistent with the robot
  server).

## Testing (TDD — write tests first)

### Unit tests (fast, no sockets)

- `test_gripper_protocol.py`: header `<HII` round-trip; `ConnectRequest` / `MoveRequest` /
  `GraspRequest` parsers; `GripperState.pack` is exactly 23 bytes with the correct `<IddBH`
  field values; `build_connect_response` is 14 bytes and `build_command_response` is 12 bytes with
  correct header + status.
- `test_gripper_backend.py`: `homing` sets `width == max_width`; `move` clamps out-of-range widths
  and clears `is_grasped`; `grasp` returns `True` / `is_grasped` when a configured object is within
  epsilon and `False` / not-grasped otherwise (no object, or outside epsilon); `stop` returns
  `True`.

### Integration tests (real sockets, like `test_commands_mock.py` with a gripper fixture)

- `test_gripper_server.py`: Connect handshake returns `kSuccess` + version 3 as a byte-exact
  14-byte message; `Homing` returns `kSuccess` and a subsequent UDP `GripperState` shows
  `width ≈ 0.08`; `Move` updates the width seen over UDP; `Grasp` with a configured object reports
  `is_grasped` and `kSuccess`, without one reports `kUnsuccessful`; framing byte counts are
  asserted exactly.

### Real-client smoke test (gated, skip-if-unavailable, like `test_real_client_integration.py`)

- If a real `pylibfranka.Gripper` (libfranka_new) is importable, connect it to the sim and run
  `homing` / `move` / `read_once`, asserting the returned state is sane. Skip when unavailable so
  CI without the native client still passes.

## Non-goals (this step)

- Timed / physical finger motion and contact-based grasping in Genesis.
- Robotiq backend implementation (interface + documented extension point only).
- Vacuum gripper (`research_interface::vacuum_gripper`).
