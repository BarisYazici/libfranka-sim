import struct
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from franka_sim.franka_genesis_sim import ControlMode
from franka_sim.franka_protocol import (
    COMMAND_PORT,
    Command,
    ConnectStatus,
    ControllerMode,
    LibfrankaMotionGeneratorMode,
    MessageHeader,
    MotionGeneratorMode,
    MoveStatus,
)

BASE_TWIST = [0.25, -0.1, 0.0, 0.0, 0.0, 0.4]


def perform_handshake(tcp_client):
    """Connect handshake (libfranka v10): version + UDP port in, status + version out."""
    tcp_client.connect(("localhost", COMMAND_PORT))
    payload = struct.pack("<HH", 10, 1338)
    header = MessageHeader(command=Command.kConnect, command_id=1, size=12 + len(payload))
    tcp_client.sendall(header.to_bytes() + payload)
    tcp_client.recv(12)
    status, _ = struct.unpack("<BH", tcp_client.recv(3))
    return status == ConnectStatus.kSuccess


def send_move(tcp_client, controller_mode, motion_generator_mode, command_id=2):
    """Send a Move command and consume the kMotionStarted + kSuccess responses."""
    payload = struct.pack(
        "<II3d3d",
        controller_mode.value,
        motion_generator_mode.value,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    )
    header = MessageHeader(command=Command.kMove, command_id=command_id, size=12 + len(payload))
    tcp_client.sendall(header.to_bytes() + payload)
    tcp_client.recv(16)
    tcp_client.recv(16)


def pack_robot_command(message_id=1, o_dp_ee_c=None, tau_j_d=None, motion_finished=False):
    """Pack the UDP RobotCommand exactly as libfranka sends it."""
    o_dp_ee_c = o_dp_ee_c if o_dp_ee_c is not None else [0.0] * 6
    tau_j_d = tau_j_d if tau_j_d is not None else [0.0] * 7
    message = struct.pack("<Q", message_id)
    message += struct.pack("<7d", *([0.0] * 7))  # q_c
    message += struct.pack("<7d", *([0.0] * 7))  # dq_c
    message += struct.pack("<16d", *([0.0] * 16))  # O_T_EE_c
    message += struct.pack("<6d", *o_dp_ee_c)  # O_dP_EE_c
    message += struct.pack("<2d", *([0.0] * 2))  # elbow_c
    message += struct.pack("<B", 0)  # valid_elbow
    message += struct.pack("<B", 1 if motion_finished else 0)  # motion_generation_finished
    message += struct.pack("<7d", *tau_j_d)  # tau_J_d
    message += struct.pack("<B", 0)  # torque_command_finished
    return message


def send_robot_command(udp_client, server, **kwargs):
    udp_client.sendto(
        pack_robot_command(**kwargs), ("localhost", server.udp_socket.getsockname()[1])
    )
    time.sleep(0.2)


def test_move_with_cartesian_velocity_selects_steering_drive(
    tcp_client, base_sim_server, mock_base_sim
):
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)

    mock_base_sim.set_control_mode.assert_called_with(ControlMode.STEERING_DRIVE)
    assert base_sim_server.control_mode is ControlMode.STEERING_DRIVE


def test_cartesian_velocity_command_reaches_the_base(
    tcp_client, udp_client, base_sim_server, mock_base_sim
):
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)
    send_robot_command(udp_client, base_sim_server, o_dp_ee_c=BASE_TWIST)

    mock_base_sim.update_base_twist.assert_called_with(BASE_TWIST)


def test_cartesian_velocity_is_echoed_in_the_reported_state(
    tcp_client, udp_client, base_sim_server
):
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)
    send_robot_command(udp_client, base_sim_server, o_dp_ee_c=BASE_TWIST)

    state = base_sim_server.robot_state.state
    assert state["O_dP_EE_c"] == pytest.approx(BASE_TWIST)
    assert state["O_dP_EE_d"] == pytest.approx(BASE_TWIST)
    assert state["motion_generator_mode"] == LibfrankaMotionGeneratorMode.kCartesianVelocity.value


def test_motion_finished_zeroes_the_base_twist(
    tcp_client, udp_client, base_sim_server, mock_base_sim
):
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)
    send_robot_command(udp_client, base_sim_server, o_dp_ee_c=BASE_TWIST)
    send_robot_command(udp_client, base_sim_server, message_id=2, motion_finished=True)

    mock_base_sim.update_base_twist.assert_called_with([0.0] * 6)
    mock_base_sim.update_joint_positions.assert_not_called()


def test_cartesian_velocity_is_ignored_by_an_arm_server(
    tcp_client, udp_client, sim_server, mock_genesis_sim
):
    """A non-mobile server must drop the twist, not crash and not move the arm."""
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)
    send_robot_command(udp_client, sim_server, o_dp_ee_c=BASE_TWIST)

    mock_genesis_sim.update_base_twist.assert_not_called()
    assert sim_server.control_mode is not ControlMode.STEERING_DRIVE


def test_torque_dispatch_is_unchanged_on_an_arm_server(
    tcp_client, udp_client, sim_server, mock_genesis_sim
):
    """Regression guard: the existing external-controller path still wins."""
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kExternalController, MotionGeneratorMode.kNone)
    torques = [1.0, -1.0, 1.0, -1.0, 0.5, -0.5, 0.5]
    send_robot_command(udp_client, sim_server, tau_j_d=torques)

    assert sim_server.control_mode is ControlMode.TORQUE
    assert np.allclose(sim_server.robot_state.state["tau_J_d"], torques)


def test_arm_server_ignores_a_cartesian_velocity_move(tcp_client, sim_server, mock_genesis_sim):
    """A cartesian-velocity Move must not switch an arm server's control mode.

    FrankaGenesisSim's physics loop has no STEERING_DRIVE branch -- it would
    silently stop actuating the arm.
    """
    assert perform_handshake(tcp_client)
    send_move(tcp_client, ControllerMode.kJointImpedance, MotionGeneratorMode.kCartesianVelocity)

    assert sim_server.control_mode is not ControlMode.STEERING_DRIVE
    for call in mock_genesis_sim.set_control_mode.call_args_list:
        assert call.args[0] is not ControlMode.STEERING_DRIVE


def test_external_controller_keeps_torque_even_with_a_cartesian_motion_generator(
    tcp_client, udp_client, base_sim_server, mock_base_sim
):
    """The cartesian branch must not shadow the external-controller torque path."""
    assert perform_handshake(tcp_client)
    send_move(
        tcp_client, ControllerMode.kExternalController, MotionGeneratorMode.kCartesianVelocity
    )
    torques = [2.0, -2.0, 2.0, -2.0, 1.0, -1.0, 1.0]
    send_robot_command(udp_client, base_sim_server, o_dp_ee_c=BASE_TWIST, tau_j_d=torques)

    assert base_sim_server.control_mode is ControlMode.TORQUE
    assert np.allclose(base_sim_server.robot_state.state["tau_J_d"], torques)
    mock_base_sim.update_base_twist.assert_not_called()


# --- motion-finished hold-log latch (mobile path) ---------------------------


def _move_request(command_id, controller_mode=ControllerMode.kJointImpedance):
    """A ready-to-use (header, payload) pair for handle_move_command()."""
    payload = struct.pack(
        "<II3d3d",
        controller_mode.value,
        MotionGeneratorMode.kCartesianVelocity.value,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
    )
    header = MessageHeader(command=Command.kMove, command_id=command_id, size=12 + len(payload))
    return header, payload


def test_repeated_motion_finished_holds_log_the_hold_once(base_sim_server, mock_base_sim, caplog):
    """1 kHz hot path: libfranka's finishMotion burst calls this once per
    datagram, but the "commanding zero base twist" log must latch.
    """
    with caplog.at_level("INFO", logger="franka_sim.franka_sim_server"):
        for _ in range(5):
            base_sim_server._switch_to_hold_position()

    hold_logs = [r for r in caplog.records if "commanding zero base twist" in r.message]
    assert len(hold_logs) == 1
    # The actual hold command still goes out on every call -- only the log latches.
    assert mock_base_sim.update_base_twist.call_count == 5


def test_a_new_move_rearms_the_hold_log_latch(base_sim_server, mock_base_sim, caplog):
    header, payload = _move_request(command_id=42)
    client_socket = MagicMock()

    with caplog.at_level("INFO", logger="franka_sim.franka_sim_server"):
        base_sim_server._switch_to_hold_position()
        base_sim_server._switch_to_hold_position()
        base_sim_server.handle_move_command(client_socket, header, payload)
        base_sim_server._switch_to_hold_position()

    hold_logs = [r for r in caplog.records if "commanding zero base twist" in r.message]
    assert len(hold_logs) == 2


def test_hold_position_keeps_the_simulator_mode_in_lockstep(base_sim_server, mock_base_sim):
    """Server and simulator must not disagree about control mode after a hold."""
    base_sim_server._switch_to_hold_position()
    mock_base_sim.set_control_mode.assert_called_with(ControlMode.STEERING_DRIVE)
    assert base_sim_server.control_mode is ControlMode.STEERING_DRIVE
