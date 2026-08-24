import struct

import pytest

from franka_sim.franka_protocol import ControllerMode, MotionGeneratorMode, RobotMode
from franka_sim.robot_state import RobotState

# Canonical libfranka_new (v10) RobotState wire layout, transcribed independently
# from research_interface/robot/rbk_types.h (#pragma pack(push, 1)). It is
# float-based and 1377 bytes total. This serves as the spec the packer must match.
ROBOT_STATE_V10_FMT = (
    "<"
    "Q"  # message_id (uint64)
    + "16f" * 6  # O_T_EE, O_T_EE_d, F_T_EE, EE_T_K, F_T_NE, NE_T_EE
    + "f"  # m_ee
    + "9f"  # I_ee
    + "3f"  # F_x_Cee
    + "f"  # m_load
    + "9f"  # I_load
    + "3f"  # F_x_Cload
    + "2f" * 2  # elbow, elbow_d
    + "7f" * 8  # tau_J, tau_J_d, dtau_J, q, q_d, dq, dq_d, ddq_d
    + "7f"  # joint_contact
    + "6f"  # cartesian_contact
    + "7f"  # joint_collision
    + "6f"  # cartesian_collision
    + "7f"  # tau_ext_hat_filtered
    + "6f" * 2  # O_F_ext_hat_K, K_F_ext_hat_K
    + "6f"  # O_dP_EE_d
    + "3f"  # O_ddP_O
    + "2f" * 3  # elbow_c, delbow_c, ddelbow_c
    + "16f"  # O_T_EE_c
    + "6f" * 2  # O_dP_EE_c, O_ddP_EE_c
    + "7f" * 2  # theta, dtheta
    + "18f" * 2  # accelerometer_top, accelerometer_bottom (each 6x3 floats)
    + "BB"  # motion_generator_mode, controller_mode
    + "41B" * 2  # errors, reflex_reason
    + "B"  # robot_mode
    + "f"  # control_command_success_rate
)

# Byte offset of the `q` field (used to verify float encoding at the right place).
_FMT_BEFORE_Q = "<" + "Q" + "16f" * 6 + "f9f3f" + "f9f3f" + "2f2f" + "7f7f7f"

# Byte offset of the `theta` field, transcribed the same independent way: it is
# the last pair of joint arrays before the accelerometers.
_FMT_BEFORE_THETA = (
    "<"
    + "Q"
    + "16f" * 6
    + "f9f3f"
    + "f9f3f"
    + "2f2f"
    + "7f" * 8
    + "7f6f7f6f"
    + "7f6f6f6f"
    + "3f2f2f2f"
    + "16f6f6f"
)


def test_robot_state_initialization():
    """Test robot state initialization with default values"""
    state = RobotState()

    # Check initial values. Ids start at 1: the first state is published before
    # the first update(), and a command echoing 0 is dropped by the server.
    assert state.state["message_id"] == 1
    assert state.state["robot_mode"] == RobotMode.kIdle.value
    assert state.state["motion_generator_mode"] == 0
    assert state.state["controller_mode"] == 0
    assert len(state.state["q"]) == 7
    assert len(state.state["dq"]) == 7
    assert len(state.state["tau_J"]) == 7


def test_robot_state_update():
    """Test robot state update mechanism (monotonic message_id counter)"""
    state = RobotState()
    initial_message_id = state.state["message_id"]

    # Each update advances message_id by exactly one (strictly increasing).
    state.update()
    assert state.state["message_id"] == initial_message_id + 1
    state.update()
    assert state.state["message_id"] == initial_message_id + 2


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

    # Unpack joint positions: float-encoded at the v10 offset of `q`.
    offset = struct.calcsize(_FMT_BEFORE_Q)
    q = struct.unpack("<7f", packed_state[offset : offset + 28])
    assert list(q) == pytest.approx(state.state["q"])


def _unpack_theta_dtheta(packed):
    """Return (theta, dtheta) read off the wire at their v10 offsets."""
    offset = struct.calcsize(_FMT_BEFORE_THETA)
    theta = struct.unpack("<7f", packed[offset : offset + 28])
    dtheta = struct.unpack("<7f", packed[offset + 28 : offset + 56])
    return list(theta), list(dtheta)


def test_theta_and_dtheta_report_the_measured_joint_state():
    """theta/dtheta are the motor-side encoder; rigid joints make them q/dq.

    Regression: they were left at their zero initialiser and never written, so
    every published state claimed the motors sat at 0 rad no matter where the
    arm was. Franka's smoke-test external controller closes its PD loop on
    theta (``test/smoke/src/smoke_common.cpp:272-276``), so a zero theta turned
    ``k_p * (q_d - theta)`` into k_p times the whole joint angle and every
    ``*ExternalController*`` motion aborted with ``tau_J_range_violation``.
    """
    state = RobotState()
    state.state["q"] = [0.1, 0.2, 0.3, -2.0, 0.5, 2.2, 0.7]
    state.state["dq"] = [0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07]

    theta, dtheta = _unpack_theta_dtheta(state.pack_state())

    assert theta == pytest.approx(state.state["q"])
    assert dtheta == pytest.approx(state.state["dq"])
    # The dict the rest of the server reads is kept truthful too, not just the
    # bytes on the wire.
    assert state.state["theta"] == pytest.approx(state.state["q"])
    assert state.state["dtheta"] == pytest.approx(state.state["dq"])


def test_theta_tracks_q_across_successive_packs():
    """A second pack must report the new q, not the one the first pack saw."""
    state = RobotState()
    state.state["q"] = [0.0] * 7
    state.pack_state()

    state.state["q"] = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]
    state.state["dq"] = [0.5] * 7
    theta, dtheta = _unpack_theta_dtheta(state.pack_state())

    assert theta == pytest.approx(state.state["q"])
    assert dtheta == pytest.approx([0.5] * 7)


def test_mirroring_motor_state_does_not_alias_q():
    """Theta must be a copy: mutating q afterwards may not rewrite the packed theta."""
    state = RobotState()
    state.state["q"] = [0.3] * 7
    state.pack_state()

    state.state["theta"][0] = 99.0
    assert state.state["q"][0] == pytest.approx(0.3)


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
    # - control_command_success_rate (4 bytes, float)

    # Calculate offsets from the end
    success_rate_offset = total_size - 4
    robot_mode_offset = success_rate_offset - 1
    reflex_reason_offset = robot_mode_offset - 41
    errors_offset = reflex_reason_offset - 41
    controller_mode_offset = errors_offset - 2

    # Extract and verify each field
    motion_mode, controller_mode = struct.unpack(
        "<BB", packed_state[controller_mode_offset : controller_mode_offset + 2]
    )
    robot_mode = struct.unpack("<B", packed_state[robot_mode_offset : robot_mode_offset + 1])[0]
    success_rate = struct.unpack("<f", packed_state[success_rate_offset:])[0]

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

    # Unpack the O_T_EE matrix (float-encoded, starts after message_id)
    matrix = struct.unpack("<16f", packed_state[8 : 8 + 64])
    assert list(matrix) == pytest.approx(test_matrix)


def test_robot_state_packs_to_v10_size():
    """Verify the packed state matches the libfranka_new v10 size (1377 bytes)."""
    # Sanity check: the independently-written format matches the C++ struct size.
    assert struct.calcsize(ROBOT_STATE_V10_FMT) == 1377
    assert len(RobotState().pack_state()) == 1377


def test_robot_state_v10_float_roundtrip():
    """Joint positions must round-trip through the v10 (float) layout."""
    state = RobotState()
    state.state["message_id"] = 42
    state.state["q"] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    state.set_robot_mode(RobotMode.kMove)

    packed = state.pack_state()

    # message_id sits at the very front.
    assert struct.unpack("<Q", packed[:8])[0] == 42

    # q must be float-encoded at its v10 offset.
    offset = struct.calcsize(_FMT_BEFORE_Q)
    q = struct.unpack("<7f", packed[offset : offset + 28])
    assert q == pytest.approx(state.state["q"])

    # robot_mode is the second-to-last byte (before the float success rate).
    assert packed[-5] == RobotMode.kMove.value
