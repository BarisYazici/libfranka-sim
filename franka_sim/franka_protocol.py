"""Wire-format types for the Franka robot network protocol (libfranka v10).

Defines the TCP command/status enums and the ``MessageHeader`` framing shared
by every command exchange, matching ``research_interface::robot`` in
libfranka_new byte-for-byte (packed structs, little-endian).
"""

import enum
import struct
from dataclasses import dataclass

# Standard command port for Franka robot interface
COMMAND_PORT = 1337


class Command(enum.IntEnum):
    """Commands supported by the Franka robot interface protocol.

    Values must match research_interface::robot::Command in libfranka_new
    (protocol v10) exactly. Note that v10 removed kLoadModelLibrary (the model
    is now built client-side from the URDF returned by kGetRobotModel).
    """

    kConnect = 0
    kMove = 1
    kStopMove = 2
    kSetCollisionBehavior = 3
    kSetJointImpedance = 4
    kSetCartesianImpedance = 5
    kSetGuidingMode = 6
    kSetEEToK = 7
    kSetNEToEE = 8
    kSetLoad = 9
    kAutomaticErrorRecovery = 10
    kGetRobotModel = 11


class ConnectStatus(enum.IntEnum):
    """Connection status codes for the Franka protocol"""

    kSuccess = 0
    kIncompatibleLibraryVersion = 1


class MoveStatus(enum.IntEnum):
    """Status codes for Move command"""

    kSuccess = 0
    kMotionStarted = 1
    kPreempted = 2
    kPreemptedDueToActivatedSafetyFunctions = 3
    kCommandRejectedDueToActivatedSafetyFunctions = 4
    kCommandNotPossibleRejected = 5
    kStartAtSingularPoseRejected = 6
    kInvalidArgumentRejected = 7
    kReflexAborted = 8
    kEmergencyAborted = 9
    kInputErrorAborted = 10
    kAborted = 11


class ControllerMode(enum.IntEnum):
    """Controller modes for Move command"""

    kJointImpedance = 0
    kCartesianImpedance = 1
    kExternalController = 2


class MotionGeneratorMode(enum.IntEnum):
    """Motion generator modes for Move command"""

    kJointPosition = 0
    kJointVelocity = 1
    kCartesianPosition = 2
    kCartesianVelocity = 3
    kNone = 4


class LibfrankaControllerMode(enum.IntEnum):
    """Libfranka Controller modes"""

    kJointImpedance = 0
    kCartesianImpedance = 1
    kExternalController = 2
    kOther = 3


class LibfrankaMotionGeneratorMode(enum.IntEnum):
    """Libfranka Motion generator modes"""

    kIdle = 0
    kJointPosition = 1
    kJointVelocity = 2
    kCartesianPosition = 3
    kCartesianVelocity = 4
    kNone = 5


def convert_to_libfranka_motion_mode(mode: MotionGeneratorMode) -> LibfrankaMotionGeneratorMode:
    """Convert Move command motion mode to Libfranka motion mode"""
    conversion_map = {
        MotionGeneratorMode.kJointPosition: LibfrankaMotionGeneratorMode.kJointPosition,
        MotionGeneratorMode.kJointVelocity: LibfrankaMotionGeneratorMode.kJointVelocity,
        MotionGeneratorMode.kCartesianPosition: LibfrankaMotionGeneratorMode.kCartesianPosition,
        MotionGeneratorMode.kCartesianVelocity: LibfrankaMotionGeneratorMode.kCartesianVelocity,
        MotionGeneratorMode.kNone: LibfrankaMotionGeneratorMode.kNone,
    }
    return conversion_map[mode]


def convert_to_libfranka_controller_mode(mode: ControllerMode) -> LibfrankaControllerMode:
    """Convert Move command controller mode to Libfranka controller mode"""
    conversion_map = {
        ControllerMode.kJointImpedance: LibfrankaControllerMode.kJointImpedance,
        ControllerMode.kCartesianImpedance: LibfrankaControllerMode.kCartesianImpedance,
        ControllerMode.kExternalController: LibfrankaControllerMode.kExternalController,
    }
    return conversion_map[mode]


class RobotMode(enum.IntEnum):
    """Operating modes of the Franka robot"""

    kOther = 0
    kIdle = 1
    kMove = 2
    kGuiding = 3
    kReflex = 4
    kUserStopped = 5
    kAutomaticErrorRecovery = 6


@dataclass
class MessageHeader:
    """
    Represents the message header structure from libfranka.
    All messages begin with this 12-byte header.
    """

    command: Command  # Command type (uint32)
    command_id: int  # Unique command identifier (uint32)
    size: int  # Total message size including header (uint32)

    @classmethod
    def from_bytes(cls, data: bytes) -> "MessageHeader":
        """Parse header from binary data using little-endian format"""
        command, command_id, size = struct.unpack("<III", data)
        return cls(Command(command), command_id, size)

    def to_bytes(self) -> bytes:
        """Convert header to binary format using little-endian"""
        return struct.pack("<III", self.command.value, self.command_id, self.size)


@dataclass
class MoveCommand:
    """Represents a Move command request"""

    controller_mode: ControllerMode
    motion_generator_mode: MotionGeneratorMode
    maximum_path_deviation: tuple  # (translation, rotation, elbow)
    maximum_goal_pose_deviation: tuple  # (translation, rotation, elbow)

    @classmethod
    def from_bytes(cls, data: bytes) -> "MoveCommand":
        """Parse Move command from binary data"""
        # Unpack controller mode and motion generator mode
        controller_mode, motion_generator_mode = struct.unpack("<II", data[:8])
        # Validate controller mode and motion generator mode
        try:
            controller_mode = ControllerMode(controller_mode)
            motion_generator_mode = MotionGeneratorMode(motion_generator_mode)
        except ValueError as e:
            raise ValueError(f"Invalid controller mode or motion generator mode: {e}")

        # Unpack maximum path deviation
        path_dev = struct.unpack("<ddd", data[8:32])

        # Unpack maximum goal pose deviation
        goal_dev = struct.unpack("<ddd", data[32:56])

        return cls(controller_mode, motion_generator_mode, path_dev, goal_dev)


@dataclass
class SetCollisionBehaviorCommand:
    """Represents a SetCollisionBehavior command request"""

    lower_torque_thresholds_acceleration: list[float]  # 7 elements
    upper_torque_thresholds_acceleration: list[float]  # 7 elements
    lower_torque_thresholds_nominal: list[float]  # 7 elements
    upper_torque_thresholds_nominal: list[float]  # 7 elements
    lower_force_thresholds_acceleration: list[float]  # 6 elements
    upper_force_thresholds_acceleration: list[float]  # 6 elements
    lower_force_thresholds_nominal: list[float]  # 6 elements
    upper_force_thresholds_nominal: list[float]  # 6 elements

    @classmethod
    def from_bytes(cls, data: bytes) -> "SetCollisionBehaviorCommand":
        """Parse SetCollisionBehavior command from binary data"""
        # Each value is a double (8 bytes)
        # Total expected size: (7+7+7+7)*8 + (6+6+6+6)*8 = 224 + 192 = 416 bytes

        offset = 0
        # Unpack torque thresholds (7 doubles each)
        lower_torque_acc = list(struct.unpack("<7d", data[offset : offset + 56]))
        offset += 56
        upper_torque_acc = list(struct.unpack("<7d", data[offset : offset + 56]))
        offset += 56
        lower_torque_nom = list(struct.unpack("<7d", data[offset : offset + 56]))
        offset += 56
        upper_torque_nom = list(struct.unpack("<7d", data[offset : offset + 56]))
        offset += 56

        # Unpack force thresholds (6 doubles each)
        lower_force_acc = list(struct.unpack("<6d", data[offset : offset + 48]))
        offset += 48
        upper_force_acc = list(struct.unpack("<6d", data[offset : offset + 48]))
        offset += 48
        lower_force_nom = list(struct.unpack("<6d", data[offset : offset + 48]))
        offset += 48
        upper_force_nom = list(struct.unpack("<6d", data[offset : offset + 48]))

        return cls(
            lower_torque_acc,
            upper_torque_acc,
            lower_torque_nom,
            upper_torque_nom,
            lower_force_acc,
            upper_force_acc,
            lower_force_nom,
            upper_force_nom,
        )


@dataclass
class SetJointImpedanceCommand:
    """Represents a SetJointImpedance command request"""

    K_theta: list[float]  # 7 elements for joint stiffness values

    @classmethod
    def from_bytes(cls, data: bytes) -> "SetJointImpedanceCommand":
        """Parse SetJointImpedance command from binary data"""
        # Each value is a double (8 bytes)
        # Total expected size: 7 * 8 = 56 bytes
        K_theta = list(struct.unpack("<7d", data[:56]))
        return cls(K_theta)


@dataclass
class SetCartesianImpedanceCommand:
    """Represents a SetCartesianImpedance command request"""

    K_x: list[float]  # 6 elements for cartesian stiffness values

    @classmethod
    def from_bytes(cls, data: bytes) -> "SetCartesianImpedanceCommand":
        """Parse SetCartesianImpedance command from binary data"""
        # Each value is a double (8 bytes)
        # Total expected size: 6 * 8 = 48 bytes
        K_x = list(struct.unpack("<6d", data[:48]))
        return cls(K_x)


@dataclass
class SetGuidingModeCommand:
    """Represents a SetGuidingMode command request.

    Matches ``research_interface::robot::SetGuidingMode::Request`` (service_types.h):
    ``std::array<bool, 6> guiding_mode`` + ``bool nullspace``, 1 byte each under
    ``#pragma pack(push, 1)`` -> 7 bytes total, no padding.
    """

    guiding_mode: list[bool]  # 6 elements (x, y, z, roll, pitch, yaw)
    nullspace: bool

    @classmethod
    def from_bytes(cls, data: bytes) -> "SetGuidingModeCommand":
        """Parse SetGuidingMode command from binary data (7 bytes: 6 bool + 1 bool)."""
        values = struct.unpack("<7?", data[:7])
        return cls(list(values[:6]), values[6])


@dataclass
class SetEEToKCommand:
    """Represents a SetEEToK command request.

    Matches ``research_interface::robot::SetEEToK::Request``: a single
    ``std::array<double, 16> EE_T_K`` (column-major 4x4 transform) -> 128 bytes.
    """

    EE_T_K: list[float]  # 16 elements, column-major 4x4 transform

    @classmethod
    def from_bytes(cls, data: bytes) -> "SetEEToKCommand":
        """Parse SetEEToK command from binary data (16 doubles = 128 bytes)."""
        return cls(list(struct.unpack("<16d", data[:128])))


@dataclass
class SetNEToEECommand:
    """Represents a SetNEToEE command request.

    Matches ``research_interface::robot::SetNEToEE::Request``: a single
    ``std::array<double, 16> NE_T_EE`` (column-major 4x4 transform) -> 128 bytes.
    """

    NE_T_EE: list[float]  # 16 elements, column-major 4x4 transform

    @classmethod
    def from_bytes(cls, data: bytes) -> "SetNEToEECommand":
        """Parse SetNEToEE command from binary data (16 doubles = 128 bytes)."""
        return cls(list(struct.unpack("<16d", data[:128])))


@dataclass
class SetLoadCommand:
    """Represents a SetLoad command request.

    Matches ``research_interface::robot::SetLoad::Request``: ``double m_load``
    + ``std::array<double, 3> F_x_Cload`` + ``std::array<double, 9> I_load`` ->
    8 + 24 + 72 = 104 bytes, no padding under ``#pragma pack(push, 1)``.
    """

    m_load: float
    F_x_Cload: list[float]  # 3 elements
    I_load: list[float]  # 9 elements (3x3 inertia matrix)

    @classmethod
    def from_bytes(cls, data: bytes) -> "SetLoadCommand":
        """Parse SetLoad command from binary data (1 double + 3 doubles + 9 doubles)."""
        m_load = struct.unpack("<d", data[:8])[0]
        F_x_Cload = list(struct.unpack("<3d", data[8:32]))
        I_load = list(struct.unpack("<9d", data[32:104]))
        return cls(m_load, F_x_Cload, I_load)
