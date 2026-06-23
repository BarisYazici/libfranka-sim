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
        try:
            command = GripperCommand(command)
        except ValueError:
            # Unknown command value: keep the raw int so the server can reply
            # kFail and stay connected, rather than crashing the receive loop.
            pass
        return cls(command, command_id, size)

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
