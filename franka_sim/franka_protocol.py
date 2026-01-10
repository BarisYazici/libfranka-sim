import enum
import struct
from dataclasses import dataclass

# Standard command port for Franka robot interface
COMMAND_PORT = 1337


class Command(enum.IntEnum):
    """Commands supported by the Franka robot interface protocol (libfranka 0.9.2)"""

    kConnect = 0
    kMove = 1
    kStopMove = 2
    kGetCartesianLimit = 3  # Added for 0.9.2 compatibility
    kSetCollisionBehavior = 4
    kSetJointImpedance = 5
    kSetCartesianImpedance = 6
    kSetGuidingMode = 7
    kSetEEToK = 8
    kSetNEToEE = 9
    kSetLoad = 10
    kSetFilters = 11  # Added for 0.9.2 compatibility
    kAutomaticErrorRecovery = 12
    kLoadModelLibrary = 13


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


class LoadModelLibraryArchitecture(enum.IntEnum):
    """Architecture types for LoadModelLibrary command"""

    kX64 = 0
    kX86 = 1
    kARM = 2
    kARM64 = 3


class LoadModelLibrarySystem(enum.IntEnum):
    """System types for LoadModelLibrary command"""

    kLinux = 0
    kWindows = 1


class LoadModelLibraryStatus(enum.IntEnum):
    """Status codes for LoadModelLibrary command"""

    kSuccess = 0
    kError = 1


@dataclass
class LoadModelLibraryCommand:
    """Represents a LoadModelLibrary command request"""

    architecture: LoadModelLibraryArchitecture
    system: LoadModelLibrarySystem

    @classmethod
    def from_bytes(cls, data: bytes) -> "LoadModelLibraryCommand":
        """Parse LoadModelLibrary command from binary data"""
        # Architecture (uint8) + System (uint8)
        architecture, system = struct.unpack("<BB", data[:2])
        return cls(LoadModelLibraryArchitecture(architecture), LoadModelLibrarySystem(system))


class GetterSetterStatus(enum.IntEnum):
    """Status codes for getter/setter commands"""

    kSuccess = 0
    kCommandNotPossibleRejected = 1
    kInvalidArgumentRejected = 2
