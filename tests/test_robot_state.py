import struct
import time

import pytest

from franka_sim.franka_protocol import ControllerMode, MotionGeneratorMode, RobotMode
from franka_sim.robot_state import RobotState


def test_robot_state_initialization():
    """Test robot state initialization with default values"""
    state = RobotState()

    # Check initial values
    assert state.state["message_id"] == 0
    assert state.state["robot_mode"] == RobotMode.kIdle.value
    assert state.state["motion_generator_mode"] == 0
    assert state.state["controller_mode"] == 0
    assert len(state.state["q"]) == 7
    assert len(state.state["dq"]) == 7
    assert len(state.state["tau_J"]) == 7


def test_robot_state_update():
    """Test robot state update mechanism"""
    state = RobotState()
    initial_message_id = state.state["message_id"]

    # Update state
    state.update()

    # Check message_id was incremented
    assert state.state["message_id"] > initial_message_id
    assert isinstance(state.state["time"], float)


def test_robot_state_packing():
    """Test packing robot state into binary format"""
    state = RobotState()

    # Set some test values
    state.state["q"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    state.state["dq"] = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    state.state["tau_J"] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    state.set_robot_mode(RobotMode.kMove)

    # Pack state
    packed_state = state.pack_state()

    # Verify packed data structure
    assert isinstance(packed_state, bytes)

    # Unpack message_id from the start of the packed state
    message_id = struct.unpack("<Q", packed_state[:8])[0]
    assert message_id == state.state["message_id"]

    # Unpack joint positions (after several transformation matrices)
    offset = 8 + (16 * 8 * 6)  # Skip message_id and transformation matrices
    offset += 8  # Skip m_ee
    offset += 9 * 8  # Skip I_ee
    offset += 3 * 8  # Skip F_x_Cee
    offset += 8  # Skip m_load
    offset += 9 * 8  # Skip I_load
    offset += 3 * 8  # Skip F_x_Cload
    offset += 2 * 8  # Skip elbow
    offset += 2 * 8  # Skip elbow_d
    offset += 7 * 8  # Skip tau_J
    offset += 7 * 8  # Skip tau_J_d
    offset += 7 * 8  # Skip dtau_J

    q = struct.unpack("<7d", packed_state[offset : offset + 56])
    assert list(q) == state.state["q"]


def test_robot_state_mode_changes():
    """Test robot state mode changes"""
    state = RobotState()

    # Test mode changes
    state.set_robot_mode(RobotMode.kMove)
    state.state["motion_generator_mode"] = MotionGeneratorMode.kJointPosition.value
    state.state["controller_mode"] = ControllerMode.kJointImpedance.value

    # Verify state values before packing
    assert state.state["robot_mode"] == RobotMode.kMove.value
    assert state.state["motion_generator_mode"] == MotionGeneratorMode.kJointPosition.value
    assert state.state["controller_mode"] == ControllerMode.kJointImpedance.value

    packed_state = state.pack_state()

    # Calculate offsets for mode values in packed state
    total_size = len(packed_state)
    print(f"Total packed state size: {total_size} bytes")

    # The last part of the state should be:
    # - motion_generator_mode (1 byte)
    # - controller_mode (1 byte)
    # - errors (41 bytes)
    # - reflex_reason (41 bytes)
    # - robot_mode (1 byte)
    # - control_command_success_rate (8 bytes)

    # Calculate offsets from the end
    success_rate_offset = total_size - 8
    robot_mode_offset = success_rate_offset - 1
    reflex_reason_offset = robot_mode_offset - 41
    errors_offset = reflex_reason_offset - 41
    controller_mode_offset = errors_offset - 2

    print("Offsets from end:")
    print("success_rate: -8")
    print("robot_mode: -9")
    print("reflex_reason: -50")
    print("errors: -91")
    print("controller_mode: -93")

    # Extract and verify each field
    motion_mode, controller_mode = struct.unpack(
        "<BB", packed_state[controller_mode_offset : controller_mode_offset + 2]
    )
    errors = struct.unpack("<41B", packed_state[errors_offset : errors_offset + 41])
    reflex_reason = struct.unpack(
        "<41B", packed_state[reflex_reason_offset : reflex_reason_offset + 41]
    )
    robot_mode = struct.unpack("<B", packed_state[robot_mode_offset : robot_mode_offset + 1])[0]
    success_rate = struct.unpack("<d", packed_state[success_rate_offset:])[0]

    print("\nExtracted values:")
    print(f"motion_mode: {motion_mode}")
    print(f"controller_mode: {controller_mode}")
    print(f"robot_mode: {robot_mode}")
    print(f"success_rate: {success_rate}")

    # Verify values
    assert motion_mode == MotionGeneratorMode.kJointPosition.value
    assert controller_mode == ControllerMode.kJointImpedance.value
    assert robot_mode == RobotMode.kMove.value
    assert success_rate == state.state["control_command_success_rate"]


def test_robot_state_transformation_matrices():
    """Test handling of transformation matrices in robot state"""
    state = RobotState()

    # Set a test transformation matrix
    test_matrix = [
        1.0,
        0.0,
        0.0,
        1.0,  # Last column is translation
        0.0,
        1.0,
        0.0,
        2.0,
        0.0,
        0.0,
        1.0,
        3.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    state.state["O_T_EE"] = test_matrix

    packed_state = state.pack_state()

    # Unpack the O_T_EE matrix (starts after message_id)
    matrix = struct.unpack("<16d", packed_state[8 : 8 + 128])
    assert list(matrix) == test_matrix
