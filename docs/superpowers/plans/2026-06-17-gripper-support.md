# Gripper Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Franka Hand gripper support to libfranka-sim that is wire-compatible with the stock `franka::Gripper` client, behind a modular `GripperBackend` interface that leaves a clean seam for a future Robotiq gripper.

**Architecture:** Mirror the existing robot split (`franka_protocol.py` / `franka_sim_server.py`) with three new modules: a pure wire-format module (`gripper_protocol.py`), a swappable backend (`gripper_backend.py` with a kinematic `FrankaHandSim`), and a TCP/UDP server (`gripper_server.py`) on port 1338. `FrankaSimServer` co-launches the gripper server in a daemon thread so one process serves both the arm (1337) and the gripper (1338).

**Tech Stack:** Python 3.11, stdlib `socket` / `struct` / `threading` / `select`, `pytest`. No new runtime dependencies. Genesis is not touched by this step (the gripper is kinematic).

## Global Constraints

- Code must pass the repo's pre-commit: `black` (line-length 100), `isort` (line_length 100), `flake8` (`max-line-length = 100`). Keep every line ≤ 100 chars.
- All multi-byte wire structs are little-endian and packed (no alignment padding) — always use `struct` format strings beginning with `<`.
- Gripper protocol constants are fixed by libfranka_new: command port `1338`, `kVersion = 3`.
- Run tests from the repo root with the project's conda env active:
  `source /home/yazi_ba/miniforge3/etc/profile.d/conda.sh && conda activate libfranka-sim`.
- New modules must not import `genesis` at module load (the test suite stubs it; the gripper path is Genesis-free).

---

### Task 1: Gripper wire protocol (`gripper_protocol.py`)

**Files:**
- Create: `franka_sim/gripper_protocol.py`
- Test: `tests/test_gripper_protocol.py`

**Interfaces:**
- Consumes: nothing (pure stdlib).
- Produces:
  - Constants `GRIPPER_COMMAND_PORT = 1338`, `GRIPPER_VERSION = 3`, `GRIPPER_HEADER_SIZE = 10`, `GRIPPER_STATE_SIZE = 23`.
  - Enums `GripperCommand` (`kConnect=0, kHoming=1, kGrasp=2, kMove=3, kStop=4`), `GripperConnectStatus` (`kSuccess=0, kIncompatibleLibraryVersion=1`), `GripperStatus` (`kSuccess=0, kFail=1, kUnsuccessful=2, kAborted=3`).
  - `GripperCommandHeader(command, command_id, size)` with `from_bytes(data)->GripperCommandHeader` and `to_bytes()->bytes` (`<HII`).
  - `ConnectRequest(version, udp_port).from_bytes(data)` (`<HH`), `MoveRequest(width, speed).from_bytes(data)` (`<dd`), `GraspRequest(width, epsilon_inner, epsilon_outer, speed, force).from_bytes(data)` (`<ddddd`).
  - `GripperState(width, max_width, is_grasped, temperature)` with `pack(message_id)->bytes` (23 bytes, `<IddBH`) and classmethod `unpack(data)->Tuple[int, GripperState]`.
  - `build_connect_response(command_id, status, version)->bytes` (14 bytes) and `build_command_response(command, command_id, status)->bytes` (12 bytes).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gripper_protocol.py`:

```python
import struct

from franka_sim.gripper_protocol import (
    GRIPPER_HEADER_SIZE,
    GRIPPER_STATE_SIZE,
    GRIPPER_VERSION,
    ConnectRequest,
    GraspRequest,
    GripperCommand,
    GripperCommandHeader,
    GripperConnectStatus,
    GripperState,
    GripperStatus,
    MoveRequest,
    build_command_response,
    build_connect_response,
)


def test_header_roundtrip():
    header = GripperCommandHeader(GripperCommand.kMove, 7, 26)
    data = header.to_bytes()
    assert len(data) == GRIPPER_HEADER_SIZE == 10
    parsed = GripperCommandHeader.from_bytes(data)
    assert parsed.command == GripperCommand.kMove
    assert parsed.command_id == 7
    assert parsed.size == 26


def test_connect_request_parse():
    req = ConnectRequest.from_bytes(struct.pack("<HH", 3, 50000))
    assert req.version == 3
    assert req.udp_port == 50000


def test_move_request_parse():
    req = MoveRequest.from_bytes(struct.pack("<dd", 0.05, 0.1))
    assert req.width == 0.05
    assert req.speed == 0.1


def test_grasp_request_parse():
    req = GraspRequest.from_bytes(struct.pack("<ddddd", 0.03, 0.005, 0.004, 0.1, 60.0))
    assert req.width == 0.03
    assert req.epsilon_inner == 0.005
    assert req.epsilon_outer == 0.004
    assert req.speed == 0.1
    assert req.force == 60.0


def test_gripper_state_pack_roundtrip():
    state = GripperState(width=0.08, max_width=0.08, is_grasped=True, temperature=30)
    data = state.pack(message_id=42)
    assert len(data) == GRIPPER_STATE_SIZE == 23
    message_id, parsed = GripperState.unpack(data)
    assert message_id == 42
    assert parsed.width == 0.08
    assert parsed.max_width == 0.08
    assert parsed.is_grasped is True
    assert parsed.temperature == 30


def test_build_connect_response():
    msg = build_connect_response(1, GripperConnectStatus.kSuccess, GRIPPER_VERSION)
    assert len(msg) == 14
    header = GripperCommandHeader.from_bytes(msg[:10])
    assert header.command == GripperCommand.kConnect
    assert header.command_id == 1
    assert header.size == 14
    status, version = struct.unpack("<HH", msg[10:14])
    assert status == GripperConnectStatus.kSuccess
    assert version == GRIPPER_VERSION


def test_build_command_response():
    msg = build_command_response(GripperCommand.kHoming, 5, GripperStatus.kSuccess)
    assert len(msg) == 12
    header = GripperCommandHeader.from_bytes(msg[:10])
    assert header.command == GripperCommand.kHoming
    assert header.command_id == 5
    assert header.size == 12
    (status,) = struct.unpack("<H", msg[10:12])
    assert status == GripperStatus.kSuccess
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_gripper_protocol.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'franka_sim.gripper_protocol'`.

- [ ] **Step 3: Write the implementation**

Create `franka_sim/gripper_protocol.py`:

```python
import enum
import struct
from dataclasses import dataclass
from typing import Tuple

# Standard libfranka gripper command port and protocol version (research_interface
# ::gripper, kVersion=3). Distinct from the robot's 1337/v10.
GRIPPER_COMMAND_PORT = 1338
GRIPPER_VERSION = 3

# CommandHeader is packed (#pragma pack(1)): command(uint16) + command_id(uint32)
# + size(uint32). size is the total message length including this header.
_HEADER_FORMAT = "<HII"
GRIPPER_HEADER_SIZE = struct.calcsize(_HEADER_FORMAT)  # 10

# GripperState UDP payload, packed: message_id(uint32) + width(double) +
# max_width(double) + is_grasped(bool, 1 byte) + temperature(uint16).
_STATE_FORMAT = "<IddBH"
GRIPPER_STATE_SIZE = struct.calcsize(_STATE_FORMAT)  # 23


class GripperCommand(enum.IntEnum):
    """Gripper commands (research_interface::gripper::Command, uint16)."""

    kConnect = 0
    kHoming = 1
    kGrasp = 2
    kMove = 3
    kStop = 4


class GripperConnectStatus(enum.IntEnum):
    """Status for the gripper Connect command."""

    kSuccess = 0
    kIncompatibleLibraryVersion = 1


class GripperStatus(enum.IntEnum):
    """Status for Homing/Grasp/Move/Stop (CommandBase::Status, uint16)."""

    kSuccess = 0
    kFail = 1
    kUnsuccessful = 2
    kAborted = 3


@dataclass
class GripperCommandHeader:
    command: GripperCommand
    command_id: int
    size: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "GripperCommandHeader":
        command, command_id, size = struct.unpack(_HEADER_FORMAT, data[:GRIPPER_HEADER_SIZE])
        return cls(GripperCommand(command), command_id, size)

    def to_bytes(self) -> bytes:
        return struct.pack(_HEADER_FORMAT, int(self.command), self.command_id, self.size)


@dataclass
class ConnectRequest:
    version: int
    udp_port: int

    @classmethod
    def from_bytes(cls, data: bytes) -> "ConnectRequest":
        version, udp_port = struct.unpack("<HH", data[:4])
        return cls(version, udp_port)


@dataclass
class MoveRequest:
    width: float
    speed: float

    @classmethod
    def from_bytes(cls, data: bytes) -> "MoveRequest":
        width, speed = struct.unpack("<dd", data[:16])
        return cls(width, speed)


@dataclass
class GraspRequest:
    width: float
    epsilon_inner: float
    epsilon_outer: float
    speed: float
    force: float

    @classmethod
    def from_bytes(cls, data: bytes) -> "GraspRequest":
        width, epsilon_inner, epsilon_outer, speed, force = struct.unpack("<ddddd", data[:40])
        return cls(width, epsilon_inner, epsilon_outer, speed, force)


@dataclass
class GripperState:
    width: float
    max_width: float
    is_grasped: bool
    temperature: int

    def pack(self, message_id: int) -> bytes:
        return struct.pack(
            _STATE_FORMAT,
            message_id,
            self.width,
            self.max_width,
            bool(self.is_grasped),
            self.temperature,
        )

    @classmethod
    def unpack(cls, data: bytes) -> Tuple[int, "GripperState"]:
        message_id, width, max_width, is_grasped, temperature = struct.unpack(
            _STATE_FORMAT, data[:GRIPPER_STATE_SIZE]
        )
        return message_id, cls(width, max_width, bool(is_grasped), temperature)


def build_connect_response(command_id: int, status: GripperConnectStatus, version: int) -> bytes:
    """Framed Connect::Response message (header + status(uint16) + version(uint16))."""
    payload = struct.pack("<HH", int(status), version)
    header = GripperCommandHeader(
        GripperCommand.kConnect, command_id, GRIPPER_HEADER_SIZE + len(payload)
    )
    return header.to_bytes() + payload


def build_command_response(
    command: GripperCommand, command_id: int, status: GripperStatus
) -> bytes:
    """Framed response for Homing/Grasp/Move/Stop (header + status(uint16))."""
    payload = struct.pack("<H", int(status))
    header = GripperCommandHeader(command, command_id, GRIPPER_HEADER_SIZE + len(payload))
    return header.to_bytes() + payload
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_gripper_protocol.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Lint and commit**

```bash
black franka_sim/gripper_protocol.py tests/test_gripper_protocol.py
isort franka_sim/gripper_protocol.py tests/test_gripper_protocol.py
flake8 franka_sim/gripper_protocol.py tests/test_gripper_protocol.py
git add franka_sim/gripper_protocol.py tests/test_gripper_protocol.py
git commit -m "feat(gripper): add gripper wire protocol (v3, port 1338)"
```

---

### Task 2: Kinematic backend (`gripper_backend.py`)

**Files:**
- Create: `franka_sim/gripper_backend.py`
- Test: `tests/test_gripper_backend.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - Constants `FRANKA_HAND_MAX_WIDTH = 0.08`, `DEFAULT_TEMPERATURE = 30`.
  - `@dataclass GripperStateData(width: float, max_width: float, is_grasped: bool, temperature: int)`.
  - `GripperBackend(ABC)` with abstract `homing()->bool`, `move(width, speed)->bool`, `grasp(width, epsilon_inner, epsilon_outer, speed, force)->bool`, `stop()->bool`, `get_state()->GripperStateData`.
  - `FrankaHandSim(GripperBackend)` with `__init__(max_width=0.08, temperature=30, object_width=None)`, `set_object_width(width)`, and the five interface methods.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gripper_backend.py`:

```python
from franka_sim.gripper_backend import FRANKA_HAND_MAX_WIDTH, FrankaHandSim


def test_homing_opens_to_max_width():
    gripper = FrankaHandSim()
    assert gripper.homing() is True
    state = gripper.get_state()
    assert state.width == FRANKA_HAND_MAX_WIDTH
    assert state.max_width == FRANKA_HAND_MAX_WIDTH
    assert state.is_grasped is False


def test_move_sets_and_clamps_width():
    gripper = FrankaHandSim()
    assert gripper.move(0.03, 0.1) is True
    assert gripper.get_state().width == 0.03
    gripper.move(0.5, 0.1)  # above max -> clamped
    assert gripper.get_state().width == FRANKA_HAND_MAX_WIDTH
    gripper.move(-0.1, 0.1)  # below zero -> clamped
    assert gripper.get_state().width == 0.0


def test_move_clears_is_grasped():
    gripper = FrankaHandSim(object_width=0.03)
    gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0)
    assert gripper.get_state().is_grasped is True
    gripper.move(0.05, 0.1)
    assert gripper.get_state().is_grasped is False


def test_grasp_succeeds_when_object_within_epsilon():
    gripper = FrankaHandSim()
    gripper.set_object_width(0.03)
    assert gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0) is True
    state = gripper.get_state()
    assert state.is_grasped is True
    assert state.width == 0.03


def test_grasp_fails_without_object():
    gripper = FrankaHandSim()
    assert gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0) is False
    assert gripper.get_state().is_grasped is False


def test_grasp_fails_when_object_outside_epsilon():
    gripper = FrankaHandSim()
    gripper.set_object_width(0.05)
    assert gripper.grasp(0.03, 0.005, 0.005, 0.1, 60.0) is False
    assert gripper.get_state().is_grasped is False


def test_stop_returns_true():
    assert FrankaHandSim().stop() is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_gripper_backend.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'franka_sim.gripper_backend'`.

- [ ] **Step 3: Write the implementation**

Create `franka_sim/gripper_backend.py`:

```python
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Franka Hand maximum stroke (m) between the two fingers.
FRANKA_HAND_MAX_WIDTH = 0.08
# Constant temperature (deg C) reported in GripperState.
DEFAULT_TEMPERATURE = 30


@dataclass
class GripperStateData:
    width: float
    max_width: float
    is_grasped: bool
    temperature: int


class GripperBackend(ABC):
    """Swappable gripper backend.

    The hardcoded libfranka wire client (``franka::Gripper``) only ever talks to
    ``FrankaGripperServer``, which translates its fixed protocol into these five
    calls. A different gripper (e.g. Robotiq) implements the same contract and is
    free to additionally expose its own richer, non-libfranka transport -- the
    server-facing surface stays identical.
    """

    @abstractmethod
    def homing(self) -> bool:
        """Home the gripper; return True on success."""

    @abstractmethod
    def move(self, width: float, speed: float) -> bool:
        """Move fingers to ``width`` (m) at ``speed`` (m/s); return True on success."""

    @abstractmethod
    def grasp(
        self,
        width: float,
        epsilon_inner: float,
        epsilon_outer: float,
        speed: float,
        force: float,
    ) -> bool:
        """Grasp at ``width`` (m); return True only if an object was grasped."""

    @abstractmethod
    def stop(self) -> bool:
        """Abort the current motion; return True on success."""

    @abstractmethod
    def get_state(self) -> GripperStateData:
        """Return the current gripper state."""


class FrankaHandSim(GripperBackend):
    """Kinematic Franka Hand model (no physics, no threads, no sleeps).

    Width updates are instant: a command sets the final width and returns, which
    is enough to be wire-compatible with ``franka::Gripper`` and fully
    unit-testable. ``grasp`` succeeds only if a configured ``object_width`` lies
    within the commanded width's epsilon band -- the same notion of "did the
    fingers catch something" the real hand reports. Timed/physical motion can
    replace the internals later without changing the interface.
    """

    def __init__(
        self,
        max_width: float = FRANKA_HAND_MAX_WIDTH,
        temperature: int = DEFAULT_TEMPERATURE,
        object_width: Optional[float] = None,
    ):
        self.max_width = max_width
        self.temperature = temperature
        self.object_width = object_width
        self.width = max_width  # start fully open
        self.is_grasped = False

    def set_object_width(self, width: Optional[float]) -> None:
        """Configure a graspable object width (m), or None for free space."""
        self.object_width = width

    def _clamp(self, width: float) -> float:
        return max(0.0, min(self.max_width, width))

    def homing(self) -> bool:
        self.width = self.max_width
        self.is_grasped = False
        return True

    def move(self, width: float, speed: float) -> bool:
        self.width = self._clamp(width)
        self.is_grasped = False
        return True

    def grasp(
        self,
        width: float,
        epsilon_inner: float,
        epsilon_outer: float,
        speed: float,
        force: float,
    ) -> bool:
        if self.object_width is not None and (
            width - epsilon_inner <= self.object_width <= width + epsilon_outer
        ):
            self.width = self.object_width
            self.is_grasped = True
            return True
        self.width = self._clamp(width)
        self.is_grasped = False
        return False

    def stop(self) -> bool:
        return True

    def get_state(self) -> GripperStateData:
        return GripperStateData(
            self.width, self.max_width, self.is_grasped, self.temperature
        )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_gripper_backend.py -v`
Expected: PASS (7 passed).

- [ ] **Step 5: Lint and commit**

```bash
black franka_sim/gripper_backend.py tests/test_gripper_backend.py
isort franka_sim/gripper_backend.py tests/test_gripper_backend.py
flake8 franka_sim/gripper_backend.py tests/test_gripper_backend.py
git add franka_sim/gripper_backend.py tests/test_gripper_backend.py
git commit -m "feat(gripper): add modular GripperBackend + kinematic FrankaHandSim"
```

---

### Task 3: Gripper server (`gripper_server.py`) + integration tests

**Files:**
- Create: `franka_sim/gripper_server.py`
- Modify: `tests/conftest.py` (add `gripper_backend` and `gripper_server` fixtures)
- Test: `tests/test_gripper_server.py`

**Interfaces:**
- Consumes: everything from `gripper_protocol` and `gripper_backend` (Tasks 1–2).
- Produces:
  - `FrankaGripperServer(host="0.0.0.0", port=GRIPPER_COMMAND_PORT, backend=None)` with attributes `port`, `backend`, `running`, `connection_running`, and methods `run_server()`, `stop()`, plus internal helpers `receive_exact`, `receive_message`, `handle_client`.
  - conftest fixtures `gripper_backend` (a `FrankaHandSim`) and `gripper_server` (a started `FrankaGripperServer` bound to 1338, torn down after the test).

- [ ] **Step 1: Add fixtures to `tests/conftest.py`**

Append to `tests/conftest.py` (after the existing fixtures). It reuses the existing module-level `wait_for_server` helper:

```python
@pytest.fixture
def gripper_backend():
    """A fresh kinematic Franka Hand backend for gripper tests."""
    from franka_sim.gripper_backend import FrankaHandSim

    return FrankaHandSim()


@pytest.fixture
def gripper_server(gripper_backend):
    """A started gripper server (port 1338) with an injected backend."""
    from franka_sim.gripper_protocol import GRIPPER_COMMAND_PORT
    from franka_sim.gripper_server import FrankaGripperServer

    server = FrankaGripperServer(backend=gripper_backend)
    server_thread = threading.Thread(target=server.run_server, daemon=True)
    server_thread.start()

    if not wait_for_server(GRIPPER_COMMAND_PORT):
        server.stop()
        server_thread.join(timeout=1.0)
        raise RuntimeError("Gripper server failed to start")

    yield server

    server.stop()
    server_thread.join(timeout=2.0)
    time.sleep(0.2)
```

- [ ] **Step 2: Write the failing integration tests**

Create `tests/test_gripper_server.py`:

```python
import socket
import struct

import pytest

from franka_sim.gripper_protocol import (
    GRIPPER_COMMAND_PORT,
    GRIPPER_HEADER_SIZE,
    GRIPPER_VERSION,
    GripperCommand,
    GripperCommandHeader,
    GripperConnectStatus,
    GripperState,
    GripperStatus,
)


def _recv_response(sock):
    header_data = sock.recv(GRIPPER_HEADER_SIZE)
    header = GripperCommandHeader.from_bytes(header_data)
    payload = b""
    remaining = header.size - GRIPPER_HEADER_SIZE
    while len(payload) < remaining:
        payload += sock.recv(remaining - len(payload))
    return header, payload


def _open_client():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5.0)
    sock.connect(("localhost", GRIPPER_COMMAND_PORT))
    return sock


def _open_udp():
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(("127.0.0.1", 0))
    udp.settimeout(5.0)
    return udp, udp.getsockname()[1]


def _connect(sock, udp_port, command_id=1):
    payload = struct.pack("<HH", GRIPPER_VERSION, udp_port)
    header = GripperCommandHeader(
        GripperCommand.kConnect, command_id, GRIPPER_HEADER_SIZE + len(payload)
    )
    sock.sendall(header.to_bytes() + payload)
    return _recv_response(sock)


def _send_header_only(sock, command, command_id):
    header = GripperCommandHeader(command, command_id, GRIPPER_HEADER_SIZE)
    sock.sendall(header.to_bytes())


def _send_with_payload(sock, command, command_id, payload):
    header = GripperCommandHeader(command, command_id, GRIPPER_HEADER_SIZE + len(payload))
    sock.sendall(header.to_bytes() + payload)


def test_connect_handshake(gripper_server):
    sock = _open_client()
    udp, udp_port = _open_udp()
    try:
        header, payload = _connect(sock, udp_port)
        assert header.command == GripperCommand.kConnect
        assert header.size == 14
        status, version = struct.unpack("<HH", payload)
        assert status == GripperConnectStatus.kSuccess
        assert version == GRIPPER_VERSION
    finally:
        sock.close()
        udp.close()


def test_homing_acks_and_broadcasts_state(gripper_server):
    sock = _open_client()
    udp, udp_port = _open_udp()
    try:
        _connect(sock, udp_port)
        _send_header_only(sock, GripperCommand.kHoming, 2)
        header, payload = _recv_response(sock)
        assert header.command == GripperCommand.kHoming
        assert header.size == 12
        (status,) = struct.unpack("<H", payload)
        assert status == GripperStatus.kSuccess
        _, state = GripperState.unpack(udp.recv(64))
        assert state.max_width == pytest.approx(0.08)
        assert state.width == pytest.approx(0.08)
    finally:
        sock.close()
        udp.close()


def test_move_updates_broadcast_width(gripper_server):
    sock = _open_client()
    udp, udp_port = _open_udp()
    try:
        _connect(sock, udp_port)
        _send_with_payload(sock, GripperCommand.kMove, 3, struct.pack("<dd", 0.03, 0.1))
        header, payload = _recv_response(sock)
        assert header.command == GripperCommand.kMove
        (status,) = struct.unpack("<H", payload)
        assert status == GripperStatus.kSuccess
        # The broadcaster runs at ~60 Hz; read a few packets until width settles.
        width = None
        for _ in range(10):
            _, state = GripperState.unpack(udp.recv(64))
            width = state.width
            if width == pytest.approx(0.03):
                break
        assert width == pytest.approx(0.03)
    finally:
        sock.close()
        udp.close()


def test_grasp_success_with_object(gripper_server, gripper_backend):
    gripper_backend.set_object_width(0.03)
    sock = _open_client()
    udp, udp_port = _open_udp()
    try:
        _connect(sock, udp_port)
        payload = struct.pack("<ddddd", 0.03, 0.005, 0.005, 0.1, 60.0)
        _send_with_payload(sock, GripperCommand.kGrasp, 4, payload)
        header, resp = _recv_response(sock)
        (status,) = struct.unpack("<H", resp)
        assert status == GripperStatus.kSuccess
        grasped = False
        for _ in range(10):
            _, state = GripperState.unpack(udp.recv(64))
            grasped = state.is_grasped
            if grasped:
                break
        assert grasped is True
    finally:
        sock.close()
        udp.close()


def test_grasp_unsuccessful_without_object(gripper_server):
    sock = _open_client()
    udp, udp_port = _open_udp()
    try:
        _connect(sock, udp_port)
        payload = struct.pack("<ddddd", 0.03, 0.005, 0.005, 0.1, 60.0)
        _send_with_payload(sock, GripperCommand.kGrasp, 5, payload)
        header, resp = _recv_response(sock)
        (status,) = struct.unpack("<H", resp)
        assert status == GripperStatus.kUnsuccessful
    finally:
        sock.close()
        udp.close()
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_gripper_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'franka_sim.gripper_server'`.

- [ ] **Step 4: Write the implementation**

Create `franka_sim/gripper_server.py`:

```python
#!/usr/bin/env python3
import logging
import select
import socket
import threading
import time
from typing import Optional

from franka_sim.gripper_backend import FrankaHandSim, GripperBackend
from franka_sim.gripper_protocol import (
    GRIPPER_COMMAND_PORT,
    GRIPPER_HEADER_SIZE,
    GRIPPER_VERSION,
    ConnectRequest,
    GraspRequest,
    GripperCommand,
    GripperCommandHeader,
    GripperConnectStatus,
    GripperState,
    GripperStatus,
    MoveRequest,
    build_command_response,
    build_connect_response,
)

logger = logging.getLogger(__name__)

# Rate at which GripperState is broadcast over UDP. readOnce() only needs
# reasonably fresh state, so this is far below the arm's 1 kHz.
_BROADCAST_HZ = 60.0


class FrankaGripperServer:
    """TCP/UDP server implementing the libfranka gripper protocol (port 1338).

    Wire-compatible with the hardcoded ``franka::Gripper`` client: a TCP command
    channel (Connect/Homing/Grasp/Move/Stop, one response each) plus a one-way
    UDP ``GripperState`` broadcast. All physical behaviour is delegated to a
    swappable ``GripperBackend``.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = GRIPPER_COMMAND_PORT,
        backend: Optional[GripperBackend] = None,
    ):
        self.host = host
        self.port = port
        self.backend = backend if backend is not None else FrankaHandSim()
        self.server_socket = None
        self.running = False
        self.connection_running = False
        self.udp_socket = None
        self.client_address = None
        self.client_udp_port = None
        self.message_id = 0
        self.broadcast_thread = None

    # -- low-level receive --------------------------------------------------
    def receive_exact(self, sock: socket.socket, size: int) -> Optional[bytes]:
        data = bytearray()
        while len(data) < size:
            try:
                chunk = sock.recv(size - len(data))
            except socket.error:
                return None
            if not chunk:
                return None
            data.extend(chunk)
        return bytes(data)

    def receive_message(self, sock: socket.socket):
        header_data = self.receive_exact(sock, GRIPPER_HEADER_SIZE)
        if header_data is None:
            return None, None
        header = GripperCommandHeader.from_bytes(header_data)
        payload = b""
        payload_size = header.size - GRIPPER_HEADER_SIZE
        if payload_size > 0:
            payload = self.receive_exact(sock, payload_size)
            if payload is None:
                return None, None
        return header, payload

    # -- UDP broadcast ------------------------------------------------------
    def _broadcast_state(self):
        period = 1.0 / _BROADCAST_HZ
        next_deadline = time.perf_counter()
        while self.running and self.connection_running:
            state = self.backend.get_state()
            self.message_id += 1
            packet = GripperState(
                state.width, state.max_width, state.is_grasped, state.temperature
            ).pack(self.message_id)
            udp = self.udp_socket
            if udp is not None and self.client_address is not None:
                try:
                    udp.sendto(packet, (self.client_address, self.client_udp_port))
                except socket.error as exc:
                    logger.debug(f"Gripper UDP send failed: {exc}")
            next_deadline += period
            remaining = next_deadline - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_deadline = time.perf_counter()

    # -- command dispatch ---------------------------------------------------
    def _handle_connect(self, sock: socket.socket, header: GripperCommandHeader, payload: bytes):
        req = ConnectRequest.from_bytes(payload)
        self.client_address = sock.getpeername()[0]
        self.client_udp_port = req.udp_port
        status = (
            GripperConnectStatus.kSuccess
            if req.version == GRIPPER_VERSION
            else GripperConnectStatus.kIncompatibleLibraryVersion
        )
        sock.sendall(build_connect_response(header.command_id, status, GRIPPER_VERSION))
        logger.info(f"Gripper Connect from {self.client_address} udp_port={self.client_udp_port}")
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.broadcast_thread = threading.Thread(target=self._broadcast_state, daemon=True)
        self.broadcast_thread.start()

    def _dispatch_command(self, sock, header: GripperCommandHeader, payload: bytes):
        cmd = header.command
        try:
            if cmd == GripperCommand.kHoming:
                ok = self.backend.homing()
            elif cmd == GripperCommand.kMove:
                req = MoveRequest.from_bytes(payload)
                ok = self.backend.move(req.width, req.speed)
            elif cmd == GripperCommand.kGrasp:
                req = GraspRequest.from_bytes(payload)
                ok = self.backend.grasp(
                    req.width, req.epsilon_inner, req.epsilon_outer, req.speed, req.force
                )
            elif cmd == GripperCommand.kStop:
                ok = self.backend.stop()
            else:
                logger.warning(f"Unknown gripper command: {cmd}")
                sock.sendall(build_command_response(cmd, header.command_id, GripperStatus.kFail))
                return
            status = GripperStatus.kSuccess if ok else GripperStatus.kUnsuccessful
        except Exception as exc:
            logger.error(f"Gripper command {cmd} failed: {exc}")
            status = GripperStatus.kFail
        sock.sendall(build_command_response(cmd, header.command_id, status))

    # -- connection handling ------------------------------------------------
    def handle_client(self, sock: socket.socket):
        self.connection_running = True
        self.message_id = 0
        try:
            while self.running and self.connection_running:
                readable, _, _ = select.select([sock], [], [], 0.5)
                if not readable:
                    continue
                header, payload = self.receive_message(sock)
                if header is None:
                    break
                if header.command == GripperCommand.kConnect:
                    self._handle_connect(sock, header, payload)
                else:
                    self._dispatch_command(sock, header, payload)
        except Exception as exc:
            logger.error(f"Gripper connection error: {exc}")
        finally:
            self.connection_running = False
            if self.broadcast_thread is not None:
                self.broadcast_thread.join(timeout=1.0)
                self.broadcast_thread = None
            if self.udp_socket is not None:
                try:
                    self.udp_socket.close()
                except socket.error:
                    pass
                self.udp_socket = None
            self.client_address = None
            self.client_udp_port = None
            try:
                sock.close()
            except socket.error:
                pass

    def run_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        self.server_socket.settimeout(1.0)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.running = True
        logger.info(f"Gripper server listening on {self.host}:{self.port}")
        while self.running:
            try:
                client_socket, _ = self.server_socket.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.handle_client(client_socket)

    def stop(self):
        logger.info("Stopping gripper server")
        self.running = False
        self.connection_running = False
        if self.server_socket is not None:
            try:
                self.server_socket.close()
            except socket.error:
                pass
            self.server_socket = None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_gripper_server.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Lint and commit**

```bash
black franka_sim/gripper_server.py tests/test_gripper_server.py tests/conftest.py
isort franka_sim/gripper_server.py tests/test_gripper_server.py tests/conftest.py
flake8 franka_sim/gripper_server.py tests/test_gripper_server.py tests/conftest.py
git add franka_sim/gripper_server.py tests/test_gripper_server.py tests/conftest.py
git commit -m "feat(gripper): add FrankaGripperServer (TCP 1338 + UDP state) with tests"
```

---

### Task 4: Wire the gripper server into `FrankaSimServer`

**Files:**
- Modify: `franka_sim/franka_sim_server.py` (constructor, `start_gripper_server`, `start`, `stop`)
- Modify: `franka_sim/run_server.py` (add `--no-gripper`)
- Modify: `franka_sim/__init__.py` (export new classes)
- Test: `tests/test_gripper_integration.py`

**Interfaces:**
- Consumes: `FrankaGripperServer` (Task 3), `GRIPPER_COMMAND_PORT` (Task 1).
- Produces: `FrankaSimServer(enable_gripper=True, gripper_backend=None, ...)` with attributes `gripper_server` (a `FrankaGripperServer` or `None`) and `gripper_thread`, and a method `start_gripper_server()` that launches the gripper accept loop in a daemon thread.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gripper_integration.py`:

```python
import socket
import struct
import time

from franka_sim.franka_sim_server import FrankaSimServer
from franka_sim.gripper_protocol import (
    GRIPPER_COMMAND_PORT,
    GRIPPER_HEADER_SIZE,
    GRIPPER_VERSION,
    GripperCommand,
    GripperCommandHeader,
    GripperConnectStatus,
)


def _wait_for_port(port, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        probe.settimeout(1.0)
        try:
            probe.connect(("localhost", port))
            return True
        except (ConnectionRefusedError, socket.timeout):
            time.sleep(0.1)
        finally:
            probe.close()
    return False


def test_gripper_server_constructed_when_enabled(mock_genesis_sim):
    server = FrankaSimServer(enable_gripper=True, genesis_sim=mock_genesis_sim)
    assert server.gripper_server is not None
    assert server.gripper_server.port == GRIPPER_COMMAND_PORT


def test_no_gripper_server_when_disabled(mock_genesis_sim):
    server = FrankaSimServer(enable_gripper=False, genesis_sim=mock_genesis_sim)
    assert server.gripper_server is None


def test_start_gripper_server_listens_and_handshakes(mock_genesis_sim):
    server = FrankaSimServer(enable_gripper=True, genesis_sim=mock_genesis_sim)
    server.start_gripper_server()
    try:
        assert _wait_for_port(GRIPPER_COMMAND_PORT), "gripper server did not start"
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(("localhost", GRIPPER_COMMAND_PORT))
        try:
            payload = struct.pack("<HH", GRIPPER_VERSION, 0)
            header = GripperCommandHeader(
                GripperCommand.kConnect, 1, GRIPPER_HEADER_SIZE + len(payload)
            )
            sock.sendall(header.to_bytes() + payload)
            resp_header = GripperCommandHeader.from_bytes(sock.recv(GRIPPER_HEADER_SIZE))
            assert resp_header.command == GripperCommand.kConnect
            status, version = struct.unpack("<HH", sock.recv(4))
            assert status == GripperConnectStatus.kSuccess
            assert version == GRIPPER_VERSION
        finally:
            sock.close()
    finally:
        server.gripper_server.stop()
        if server.gripper_thread is not None:
            server.gripper_thread.join(timeout=2.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_gripper_integration.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'enable_gripper'`.

- [ ] **Step 3: Update `FrankaSimServer.__init__`**

In `franka_sim/franka_sim_server.py`, change the import block near the top to add the gripper server import (after the `from franka_sim.franka_protocol import (...)` block):

```python
from franka_sim.gripper_server import FrankaGripperServer
```

Change the constructor signature from:

```python
    def __init__(
        self,
        host="0.0.0.0",
        port=COMMAND_PORT,
        enable_vis=False,
        genesis_sim=None,
        urdf_path=None,
    ):
```

to add the two new args:

```python
    def __init__(
        self,
        host="0.0.0.0",
        port=COMMAND_PORT,
        enable_vis=False,
        genesis_sim=None,
        urdf_path=None,
        enable_gripper=True,
        gripper_backend=None,
    ):
```

Then, at the end of `__init__` (after `self.urdf_string = self._load_robot_model(urdf_path)`), add:

```python
        # Co-located gripper server (libfranka gripper protocol, port 1338).
        # Self-contained: its own backend and sockets, independent of the arm's
        # Genesis loop. Launched from start() as a daemon thread.
        self.gripper_thread = None
        if enable_gripper:
            self.gripper_server = FrankaGripperServer(host=host, backend=gripper_backend)
        else:
            self.gripper_server = None
```

- [ ] **Step 4: Add `start_gripper_server` and call it from `start`**

In `franka_sim/franka_sim_server.py`, add this method (place it just before `start`):

```python
    def start_gripper_server(self):
        """Launch the co-located gripper server's accept loop in a daemon thread."""
        if self.gripper_server is None:
            return
        self.gripper_thread = threading.Thread(
            target=self.gripper_server.run_server, daemon=True
        )
        self.gripper_thread.start()
        logger.info("Gripper server running in background thread")
```

In `start`, after `self.genesis_sim.initialize_simulation()` and its log line, before the `if self.genesis_sim.enable_vis:` branch, add:

```python
            # Bring up the gripper server alongside the arm (port 1338).
            self.start_gripper_server()
```

- [ ] **Step 5: Stop the gripper server in `stop`**

In `franka_sim/franka_sim_server.py`, in the `stop` method, after `self.cleanup()` and before `self.genesis_sim.stop()`, add:

```python
        if self.gripper_server is not None:
            self.gripper_server.stop()
            if self.gripper_thread is not None:
                self.gripper_thread.join(timeout=2.0)
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_gripper_integration.py -v`
Expected: PASS (3 passed).

- [ ] **Step 7: Add the `--no-gripper` CLI flag**

In `franka_sim/run_server.py`, add this argument after the `--urdf` argument:

```python
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        default=False,
        help="Disable the co-located gripper server (port 1338)",
    )
```

And change the server construction line from:

```python
    server = FrankaSimServer(enable_vis=args.vis, urdf_path=args.urdf)
```

to:

```python
    server = FrankaSimServer(
        enable_vis=args.vis, urdf_path=args.urdf, enable_gripper=not args.no_gripper
    )
```

- [ ] **Step 8: Export the new classes from `__init__.py`**

In `franka_sim/__init__.py`, add these imports after the existing imports:

```python
from franka_sim.gripper_backend import FrankaHandSim, GripperBackend
from franka_sim.gripper_server import FrankaGripperServer
```

And add to `__all__`:

```python
    "GripperBackend",
    "FrankaHandSim",
    "FrankaGripperServer",
```

- [ ] **Step 9: Run the full suite, lint, and commit**

```bash
pytest tests/ -v
black franka_sim/franka_sim_server.py franka_sim/run_server.py franka_sim/__init__.py tests/test_gripper_integration.py
isort franka_sim/franka_sim_server.py franka_sim/run_server.py franka_sim/__init__.py tests/test_gripper_integration.py
flake8 franka_sim/franka_sim_server.py franka_sim/run_server.py franka_sim/__init__.py tests/test_gripper_integration.py
git add franka_sim/franka_sim_server.py franka_sim/run_server.py franka_sim/__init__.py tests/test_gripper_integration.py
git commit -m "feat(gripper): co-locate gripper server in FrankaSimServer (port 1338)"
```

---

### Task 5: Real-client gripper smoke test (gated)

**Files:**
- Create: `tests/_gripper_real_client_probe.cpp`
- Test: `tests/test_gripper_real_client.py`

**Interfaces:**
- Consumes: the `gripper_server` fixture (Task 3) and the prebuilt `libfranka_new`.
- Produces: a gated end-to-end test proving a real `franka::Gripper` completes Connect/homing/readOnce/move against the sim. Skipped when the toolchain/library is absent.

- [ ] **Step 1: Write the C++ probe**

Create `tests/_gripper_real_client_probe.cpp`:

```cpp
// Tiny real-client probe: drives the sim's gripper server with franka::Gripper.
// Prints machine-checkable markers the Python test asserts on.
#include <franka/gripper.h>
#include <franka/gripper_state.h>

#include <iostream>

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: gripper_probe <robot-ip>" << std::endl;
    return 2;
  }
  try {
    franka::Gripper gripper(argv[1]);
    std::cout << "CONNECT_OK" << std::endl;

    bool homed = gripper.homing();
    std::cout << "HOMING=" << (homed ? 1 : 0) << std::endl;

    franka::GripperState state = gripper.readOnce();
    std::cout << "MAX_WIDTH=" << state.max_width << std::endl;
    std::cout << "WIDTH=" << state.width << std::endl;

    bool moved = gripper.move(0.03, 0.1);
    std::cout << "MOVE=" << (moved ? 1 : 0) << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "EXCEPTION: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
```

- [ ] **Step 2: Write the gated Python test**

Create `tests/test_gripper_real_client.py`:

```python
"""Integration smoke test: a real libfranka_new ``franka::Gripper`` vs the sim.

Compiles a tiny C++ probe linked against the prebuilt libfranka_new and points
it at the kinematic gripper server. Skipped unless a prebuilt libfranka_new, a
C++ toolchain, and Eigen are present, so it is a no-op without them.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBFRANKA = REPO_ROOT / "libfranka_new"
LIB_BUILD = LIBFRANKA / "build"
EIGEN_INCLUDE = Path("/usr/include/eigen3")
PROBE_SRC = Path(__file__).parent / "_gripper_real_client_probe.cpp"


def _prereqs_available():
    return (
        shutil.which("g++") is not None
        and (LIB_BUILD / "libfranka.so").exists()
        and (LIBFRANKA / "include" / "franka" / "gripper.h").exists()
        and EIGEN_INCLUDE.exists()
    )


pytestmark = pytest.mark.skipif(
    not _prereqs_available(),
    reason="prebuilt libfranka_new + g++ + eigen3 are required for the gripper smoke test",
)


@pytest.fixture(scope="module")
def gripper_probe_binary(tmp_path_factory):
    out = tmp_path_factory.mktemp("gripper_probe") / "gripper_probe"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            f"-I{LIBFRANKA / 'include'}",
            f"-I{LIBFRANKA / 'common' / 'include'}",
            f"-I{EIGEN_INCLUDE}",
            str(PROBE_SRC),
            f"-L{LIB_BUILD}",
            "-lfranka",
            "-o",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return out


def test_real_gripper_client_connects_homes_and_moves(gripper_server, gripper_probe_binary):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{LIB_BUILD}:" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(
        [str(gripper_probe_binary), "127.0.0.1"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )

    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    assert "CONNECT_OK" in result.stdout
    assert "HOMING=1" in result.stdout
    assert "MAX_WIDTH=0.08" in result.stdout
    assert "MOVE=1" in result.stdout
```

- [ ] **Step 3: Run the test**

Run: `pytest tests/test_gripper_real_client.py -v`
Expected: PASS if the prebuilt libfranka_new + g++ + eigen3 are present; otherwise SKIPPED (one skipped). Either outcome is acceptable.

- [ ] **Step 4: Commit**

```bash
git add tests/_gripper_real_client_probe.cpp tests/test_gripper_real_client.py
git commit -m "test(gripper): add gated real franka::Gripper smoke test"
```

---

## Final verification

- [ ] Run the whole suite: `pytest tests/ -v` — all gripper unit/integration tests pass; the real-client gripper test passes or skips.
- [ ] Confirm lint is clean: `flake8 franka_sim/ tests/`.
- [ ] Optional manual check (needs Genesis): start the server (`run-franka-sim-server`) and run libfranka_new's `move_gripper.py --ip 127.0.0.1` against it; homing/grasp/read_once should behave sanely.

## Self-review notes

- **Spec coverage:** wire protocol → Task 1; `GripperBackend` + `FrankaHandSim` (incl. Robotiq docstring seam) → Task 2; `FrankaGripperServer` (TCP + one-way UDP, status mapping, error handling) → Task 3; co-location in `FrankaSimServer` + `--no-gripper` + exports → Task 4; gated real-client smoke → Task 5. Unit + integration + gated smoke tests all present.
- **Type consistency:** `GripperState.pack/unpack` (`<IddBH`, 23 B), `build_connect_response` (14 B), `build_command_response` (12 B), and the backend's five-method contract are used identically across Tasks 1–5. `set_object_width` is defined in Task 2 and used in Tasks 2/3.
- **Non-goals (unchanged):** timed/physical finger motion & Genesis contact grasping; Robotiq implementation; vacuum gripper.
