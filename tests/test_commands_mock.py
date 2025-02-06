import pytest
import struct
import socket
import time
import numpy as np
from franka_sim.franka_protocol import (
    Command, ConnectStatus, MoveStatus, MessageHeader, COMMAND_PORT,
    ControllerMode, MotionGeneratorMode, MoveCommand
)
from franka_sim.franka_sim_server import FrankaSimServer

def perform_handshake(tcp_client):
    """Helper function to perform initial handshake"""
    tcp_client.connect(('localhost', COMMAND_PORT))
    
    # Send connect message
    version = 9
    udp_port = 1338
    payload = struct.pack('<HH', version, udp_port)
    
    header = MessageHeader(
        command=Command.kConnect,
        command_id=1,
        size=12 + len(payload)
    )
    
    tcp_client.sendall(header.to_bytes() + payload)
    
    # Receive and verify response
    response_header_data = tcp_client.recv(12)
    response_data = tcp_client.recv(8)
    status, _ = struct.unpack('<HH4x', response_data)
    
    return status == ConnectStatus.kSuccess

def test_move_command(tcp_client, udp_client, sim_server, mock_genesis_sim):
    """Test sending and handling of Move command with mocked simulator"""
    assert perform_handshake(tcp_client)
    
    # Create Move command
    move_cmd = MoveCommand(
        controller_mode=ControllerMode.kJointImpedance,
        motion_generator_mode=MotionGeneratorMode.kJointPosition,
        maximum_path_deviation=(0.1, 0.1, 0.1),
        maximum_goal_pose_deviation=(0.1, 0.1, 0.1)
    )
    
    # Pack command - format is:
    # - controller_mode (uint32)
    # - motion_generator_mode (uint32)
    # - maximum_path_deviation (3 doubles)
    # - maximum_goal_pose_deviation (3 doubles)
    payload = struct.pack(
        '<II3d3d',  # Changed format to match the actual data
        move_cmd.controller_mode.value,
        move_cmd.motion_generator_mode.value,
        *move_cmd.maximum_path_deviation,
        *move_cmd.maximum_goal_pose_deviation
    )
    
    header = MessageHeader(
        command=Command.kMove,
        command_id=2,
        size=12 + len(payload)
    )
    
    # Send command
    tcp_client.sendall(header.to_bytes() + payload)
    
    # Receive motion started response
    response_header_data = tcp_client.recv(12)
    response_header = MessageHeader.from_bytes(response_header_data)
    assert response_header.command == Command.kMove
    assert response_header.command_id == 2
    
    response_data = tcp_client.recv(4)  # Status (1) + padding (3)
    status = struct.unpack('<B3x', response_data)[0]
    assert status == MoveStatus.kMotionStarted.value  # Compare with the value instead of enum
    
    # Wait a bit for state update
    time.sleep(0.1)
    
    # Receive success response
    response_header_data = tcp_client.recv(12)
    response_header = MessageHeader.from_bytes(response_header_data)
    response_data = tcp_client.recv(4)
    status = struct.unpack('<B3x', response_data)[0]
    assert status == MoveStatus.kSuccess.value  # Compare with the value instead of enum
    
    # Verify simulator interactions
    mock_genesis_sim.set_control_mode.assert_called()

def test_stop_move_command(tcp_client, udp_client, sim_server, mock_genesis_sim):
    """Test sending and handling of StopMove command with mocked simulator"""
    assert perform_handshake(tcp_client)
    
    # First send a Move command
    move_cmd = MoveCommand(
        controller_mode=ControllerMode.kJointImpedance,
        motion_generator_mode=MotionGeneratorMode.kJointPosition,
        maximum_path_deviation=(0.1, 0.1, 0.1),
        maximum_goal_pose_deviation=(0.1, 0.1, 0.1)
    )
    
    payload = struct.pack(
        '<II3d3d',  # Changed format to match the actual data
        move_cmd.controller_mode.value,
        move_cmd.motion_generator_mode.value,
        *move_cmd.maximum_path_deviation,
        *move_cmd.maximum_goal_pose_deviation
    )
    
    header = MessageHeader(
        command=Command.kMove,
        command_id=2,
        size=12 + len(payload)
    )
    
    tcp_client.sendall(header.to_bytes() + payload)
    
    # Skip Move command responses
    tcp_client.recv(16)  # Header (12) + status (1) + padding (3)
    tcp_client.recv(16)  # Second response
    
    # Send StopMove command
    stop_header = MessageHeader(
        command=Command.kStopMove,
        command_id=3,
        size=12
    )
    
    tcp_client.sendall(stop_header.to_bytes())
    
    # Receive both responses
    # First response should be StopMove
    response_header_data = tcp_client.recv(12)
    response_header = MessageHeader.from_bytes(response_header_data)
    assert response_header.command == Command.kStopMove
    assert response_header.command_id == 3
    
    response_data = tcp_client.recv(4)  # Status (1) + padding (3)
    status = struct.unpack('<B3x', response_data)[0]
    assert status == 0  # Success
    
    # Second response should be Move (to break the waiting loop)
    move_response_header_data = tcp_client.recv(12)
    move_response_header = MessageHeader.from_bytes(move_response_header_data)
    assert move_response_header.command == Command.kMove
    assert move_response_header.command_id == 2

def test_invalid_move_parameters(tcp_client, udp_client, sim_server, mock_genesis_sim):
    """Test Move command with invalid parameters using mocked simulator"""
    assert perform_handshake(tcp_client)
    
    # Create Move command with invalid controller mode
    payload = struct.pack(
        '<II3d3d',  # Changed format to match the actual data
        99,  # Invalid controller mode
        MotionGeneratorMode.kJointPosition.value,
        0.1, 0.1, 0.1,  # maximum_path_deviation
        0.1, 0.1, 0.1   # maximum_goal_pose_deviation
    )
    
    header = MessageHeader(
        command=Command.kMove,
        command_id=2,
        size=12 + len(payload)
    )
    
    # Send command
    tcp_client.sendall(header.to_bytes() + payload)
    
    # Receive error response
    response_header_data = tcp_client.recv(12)
    response_header = MessageHeader.from_bytes(response_header_data)
    response_data = tcp_client.recv(4)
    status = struct.unpack('<B3x', response_data)[0]
    assert status == MoveStatus.kInvalidArgumentRejected

def test_robot_state_updates(tcp_client, udp_client, sim_server, mock_genesis_sim):
    """Test that robot state updates are correctly transmitted over UDP"""
    assert perform_handshake(tcp_client)
    
    # Set up mock robot state
    test_state = {
        'q': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        'dq': np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]),
        'tau_J': np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    }
    mock_genesis_sim.get_robot_state.return_value = test_state
    
    # Wait for state update
    time.sleep(0.1)
    
    # Verify that the simulator was called to get state
    mock_genesis_sim.get_robot_state.assert_called()

def test_position_control_desired_states(tcp_client, udp_client, sim_server, mock_genesis_sim):
    """Test that desired joint positions (q_d) are correctly tracked in position control mode"""
    assert perform_handshake(tcp_client)
    
    # Set up initial robot state
    initial_state = {
        'q': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        'dq': np.zeros(7),
        'tau_J': np.zeros(7)
    }
    mock_genesis_sim.get_robot_state.return_value = initial_state
    
    # Send Move command for position control
    move_cmd = MoveCommand(
        controller_mode=ControllerMode.kJointImpedance,
        motion_generator_mode=MotionGeneratorMode.kJointPosition,
        maximum_path_deviation=(0.1, 0.1, 0.1),
        maximum_goal_pose_deviation=(0.1, 0.1, 0.1)
    )
    
    payload = struct.pack(
        '<II3d3d',
        move_cmd.controller_mode.value,
        move_cmd.motion_generator_mode.value,
        *move_cmd.maximum_path_deviation,
        *move_cmd.maximum_goal_pose_deviation
    )
    
    header = MessageHeader(
        command=Command.kMove,
        command_id=2,
        size=12 + len(payload)
    )
    
    tcp_client.sendall(header.to_bytes() + payload)
    
    # Skip Move command responses
    tcp_client.recv(16)  # Header (12) + status (1) + padding (3)
    tcp_client.recv(16)  # Second response
    
    # Send a motion command with desired positions
    desired_positions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    command_msg = struct.pack('<Q', 1)  # message_id
    command_msg += struct.pack('<7d', *desired_positions)  # q_c
    command_msg += struct.pack('<7d', *([0.0] * 7))  # dq_c
    command_msg += struct.pack('<16d', *([0.0] * 16))  # O_T_EE_c
    command_msg += struct.pack('<6d', *([0.0] * 6))  # O_dP_EE_c
    command_msg += struct.pack('<2d', *([0.0] * 2))  # elbow_c
    command_msg += struct.pack('<B', 0)  # valid_elbow
    command_msg += struct.pack('<B', 0)  # motion_generation_finished
    command_msg += struct.pack('<7d', *([0.0] * 7))  # tau_J_d
    command_msg += struct.pack('<B', 0)  # torque_command_finished
    
    udp_client.sendto(command_msg, ('localhost', sim_server.udp_socket.getsockname()[1]))
    
    # Wait for state update
    time.sleep(0.1)
    
    # Verify that q_d was updated to match commanded positions
    assert np.allclose(sim_server.robot_state.state['q_d'], desired_positions)

def test_velocity_control_desired_states(tcp_client, udp_client, sim_server, mock_genesis_sim):
    """Test that desired joint velocities (dq_d) are correctly tracked in velocity control mode"""
    assert perform_handshake(tcp_client)
    
    # Send Move command for velocity control
    move_cmd = MoveCommand(
        controller_mode=ControllerMode.kJointImpedance,
        motion_generator_mode=MotionGeneratorMode.kJointVelocity,
        maximum_path_deviation=(0.1, 0.1, 0.1),
        maximum_goal_pose_deviation=(0.1, 0.1, 0.1)
    )
    
    payload = struct.pack(
        '<II3d3d',
        move_cmd.controller_mode.value,
        move_cmd.motion_generator_mode.value,
        *move_cmd.maximum_path_deviation,
        *move_cmd.maximum_goal_pose_deviation
    )
    
    header = MessageHeader(
        command=Command.kMove,
        command_id=2,
        size=12 + len(payload)
    )
    
    tcp_client.sendall(header.to_bytes() + payload)
    
    # Skip Move command responses
    tcp_client.recv(16)
    tcp_client.recv(16)
    
    # Send a motion command with desired velocities
    desired_velocities = [0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.4]
    command_msg = struct.pack('<Q', 1)  # message_id
    command_msg += struct.pack('<7d', *([0.0] * 7))  # q_c
    command_msg += struct.pack('<7d', *desired_velocities)  # dq_c
    command_msg += struct.pack('<16d', *([0.0] * 16))  # O_T_EE_c
    command_msg += struct.pack('<6d', *([0.0] * 6))  # O_dP_EE_c
    command_msg += struct.pack('<2d', *([0.0] * 2))  # elbow_c
    command_msg += struct.pack('<B', 0)  # valid_elbow
    command_msg += struct.pack('<B', 0)  # motion_generation_finished
    command_msg += struct.pack('<7d', *([0.0] * 7))  # tau_J_d
    command_msg += struct.pack('<B', 0)  # torque_command_finished
    
    udp_client.sendto(command_msg, ('localhost', sim_server.udp_socket.getsockname()[1]))
    
    # Wait for state update
    time.sleep(0.1)
    
    # Verify that dq_d was updated to match commanded velocities
    assert np.allclose(sim_server.robot_state.state['dq_d'], desired_velocities)

def test_torque_control_desired_states(tcp_client, udp_client, sim_server, mock_genesis_sim):
    """Test that desired joint torques (tau_J_d) are correctly tracked in torque control mode"""
    assert perform_handshake(tcp_client)
    
    # Send Move command for external control (torque mode)
    move_cmd = MoveCommand(
        controller_mode=ControllerMode.kExternalController,
        motion_generator_mode=MotionGeneratorMode.kJointPosition,
        maximum_path_deviation=(0.1, 0.1, 0.1),
        maximum_goal_pose_deviation=(0.1, 0.1, 0.1)
    )
    
    payload = struct.pack(
        '<II3d3d',
        move_cmd.controller_mode.value,
        move_cmd.motion_generator_mode.value,
        *move_cmd.maximum_path_deviation,
        *move_cmd.maximum_goal_pose_deviation
    )
    
    header = MessageHeader(
        command=Command.kMove,
        command_id=2,
        size=12 + len(payload)
    )
    
    tcp_client.sendall(header.to_bytes() + payload)
    
    # Skip Move command responses
    tcp_client.recv(16)
    tcp_client.recv(16)
    
    # Send a command with desired torques
    desired_torques = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0]
    command_msg = struct.pack('<Q', 1)  # message_id
    command_msg += struct.pack('<7d', *([0.0] * 7))  # q_c
    command_msg += struct.pack('<7d', *([0.0] * 7))  # dq_c
    command_msg += struct.pack('<16d', *([0.0] * 16))  # O_T_EE_c
    command_msg += struct.pack('<6d', *([0.0] * 6))  # O_dP_EE_c
    command_msg += struct.pack('<2d', *([0.0] * 2))  # elbow_c
    command_msg += struct.pack('<B', 0)  # valid_elbow
    command_msg += struct.pack('<B', 0)  # motion_generation_finished
    command_msg += struct.pack('<7d', *desired_torques)  # tau_J_d
    command_msg += struct.pack('<B', 0)  # torque_command_finished
    
    udp_client.sendto(command_msg, ('localhost', sim_server.udp_socket.getsockname()[1]))
    
    # Wait for state update
    time.sleep(0.1)
    
    # Verify that tau_J_d was updated to match commanded torques
    assert np.allclose(sim_server.robot_state.state['tau_J_d'], desired_torques)

def test_initial_desired_states(tcp_client, udp_client, sim_server, mock_genesis_sim):
    """Test that desired states are correctly initialized"""
    assert perform_handshake(tcp_client)
    
    # Set up initial robot state
    initial_state = {
        'q': np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        'dq': np.zeros(7),
        'tau_J': np.zeros(7)
    }
    mock_genesis_sim.get_robot_state.return_value = initial_state
    
    # Wait for first state update
    time.sleep(0.1)
    
    # Verify that q_d was initialized to match current positions
    assert np.allclose(sim_server.robot_state.state['q_d'], initial_state['q'])
    
    # Verify that dq_d and tau_J_d start at zero
    assert np.allclose(sim_server.robot_state.state['dq_d'], np.zeros(7))
    assert np.allclose(sim_server.robot_state.state['tau_J_d'], np.zeros(7))

def test_set_collision_behavior(tcp_client, udp_client, sim_server, mock_genesis_sim):
    """Test SetCollisionBehavior command handling"""
    assert perform_handshake(tcp_client)
    
    # Test data based on examples_common.cpp setDefaultBehavior
    lower_torque_acc = [20.0] * 7
    upper_torque_acc = [20.0] * 7
    lower_torque_nom = [10.0] * 7
    upper_torque_nom = [10.0] * 7
    lower_force_acc = [20.0] * 6
    upper_force_acc = [20.0] * 6
    lower_force_nom = [10.0] * 6
    upper_force_nom = [10.0] * 6

    # Create command payload
    payload = bytearray()
    payload.extend(struct.pack('<7d', *lower_torque_acc))
    payload.extend(struct.pack('<7d', *upper_torque_acc))
    payload.extend(struct.pack('<7d', *lower_torque_nom))
    payload.extend(struct.pack('<7d', *upper_torque_nom))
    payload.extend(struct.pack('<6d', *lower_force_acc))
    payload.extend(struct.pack('<6d', *upper_force_acc))
    payload.extend(struct.pack('<6d', *lower_force_nom))
    payload.extend(struct.pack('<6d', *upper_force_nom))

    # Create and send command message
    command_id = 1
    header = MessageHeader(Command.kSetCollisionBehavior, command_id, 12 + len(payload))
    message = header.to_bytes() + payload
    tcp_client.sendall(message)

    # Receive and verify response
    response_header_data = tcp_client.recv(12)
    response_header = MessageHeader.from_bytes(response_header_data)
    assert response_header.command == Command.kSetCollisionBehavior
    assert response_header.command_id == command_id
    assert response_header.size == 16  # Header (12) + status (1) + padding (3)

    # Get response status
    response_data = tcp_client.recv(4)  # status (1 byte) + padding (3 bytes)
    status = struct.unpack('<B3x', response_data)[0]
    assert status == 0  # Success 
