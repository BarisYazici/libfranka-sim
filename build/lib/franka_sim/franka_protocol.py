import enum
import struct
from dataclasses import dataclass

# Standard command port for Franka robot interface
COMMAND_PORT = 1337

class Command(enum.IntEnum):
    """Commands supported by the Franka robot interface protocol"""
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
    kLoadModelLibrary = 11
    kGetRobotModel = 12

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
    command: int      # Command type (uint32)
    command_id: int   # Unique command identifier (uint32)
    size: int        # Total message size including header (uint32)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'MessageHeader':
        """Parse header from binary data using little-endian format"""
        command, command_id, size = struct.unpack('<III', data)
        return cls(command, command_id, size)

    def to_bytes(self) -> bytes:
        """Convert header to binary format using little-endian"""
        return struct.pack('<III', self.command, self.command_id, self.size)

@dataclass
class MoveCommand:
    """Represents a Move command request"""
    controller_mode: ControllerMode
    motion_generator_mode: MotionGeneratorMode
    maximum_path_deviation: tuple  # (translation, rotation, elbow)
    maximum_goal_pose_deviation: tuple  # (translation, rotation, elbow)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'MoveCommand':
        """Parse Move command from binary data"""
        # Unpack controller mode and motion generator mode
        controller_mode, motion_generator_mode = struct.unpack('<II', data[:8])
        
        # Unpack maximum path deviation
        path_dev = struct.unpack('<ddd', data[8:32])
        
        # Unpack maximum goal pose deviation
        goal_dev = struct.unpack('<ddd', data[32:56])
        
        return cls(
            ControllerMode(controller_mode),
            MotionGeneratorMode(motion_generator_mode),
            path_dev,
            goal_dev
        ) 