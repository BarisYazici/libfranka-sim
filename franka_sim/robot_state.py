import logging
import struct
from typing import Any, Dict

from franka_sim.franka_protocol import RobotMode

logger = logging.getLogger(__name__)

# 4x4 identity transform (column-major, 16 elements) used to seed pose fields.
_IDENTITY_4X4 = [
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
]

# libfranka_new (v10) RobotState wire layout (research_interface/robot/rbk_types.h,
# #pragma pack(push, 1)): float-based, 1377 bytes. The Struct is compiled once so
# pack_state() serializes the whole state with a single struct.pack on the 1kHz
# hot path instead of ~50 per-field struct.pack + bytearray.extend calls.
_ROBOT_STATE_FORMAT = (
    "<"
    "Q"  # message_id (uint64)
    + "16f" * 6  # O_T_EE, O_T_EE_d, F_T_EE, EE_T_K, F_T_NE, NE_T_EE
    + "f9f3f"  # m_ee, I_ee, F_x_Cee
    + "f9f3f"  # m_load, I_load, F_x_Cload
    + "2f2f"  # elbow, elbow_d
    + "7f" * 8  # tau_J, tau_J_d, dtau_J, q, q_d, dq, dq_d, ddq_d
    + "7f6f7f6f"  # joint_contact, cartesian_contact, joint_collision, cartesian_collision
    + "7f6f6f6f"  # tau_ext_hat_filtered, O_F_ext_hat_K, K_F_ext_hat_K, O_dP_EE_d
    + "3f2f2f2f"  # O_ddP_O, elbow_c, delbow_c, ddelbow_c
    + "16f6f6f"  # O_T_EE_c, O_dP_EE_c, O_ddP_EE_c
    + "7f7f"  # theta, dtheta
    + "18f18f"  # accelerometer_top, accelerometer_bottom (each 6x3)
    + "BB"  # motion_generator_mode, controller_mode
    + "41B41B"  # errors, reflex_reason
    + "Bf"  # robot_mode, control_command_success_rate
)
_ROBOT_STATE_PACKER = struct.Struct(_ROBOT_STATE_FORMAT)


class RobotState:
    """Manages and packs robot state for the libfranka_new (v10) protocol.

    State is held in a dict and serialized by pack_state() into the 1377-byte,
    float-based binary RobotState wire format.
    """

    def __init__(self):
        """Initialize the robot state with default values."""
        self.state = self._initialize_state()
        # Ids start at 1, not 0. The first state is published *before* the
        # first update(), so a 0 here would put a message_id of 0 on the wire --
        # and _handle_commands drops any command echoing 0 as unparsable, so the
        # client's perfectly good answer to that first state was thrown away and
        # charged to it as a lost cycle.
        self._message_id = self.state["message_id"]
        logger.info("Robot state initialized")

    def _initialize_state(self) -> Dict[str, Any]:
        """Initialize the robot state dictionary with default values."""
        return {
            "message_id": 1,  # see __init__: the first published state uses it
            "q": [0.0] * 7,  # Joint positions
            "q_d": [0.0] * 7,  # Desired joint positions
            "dq": [0.0] * 7,  # Joint velocities
            "dq_d": [0.0] * 7,  # Desired joint velocities
            "ddq_d": [0.0] * 7,  # Desired joint accelerations
            "tau_J": [0.0] * 7,  # Joint torques
            "dtau_J": [0.0] * 7,  # Joint torque derivatives
            "tau_J_d": [0.0] * 7,  # Desired joint torques
            "theta": [0.0] * 7,  # Motor positions
            "dtheta": [0.0] * 7,  # Motor velocities
            "robot_mode": RobotMode.kIdle.value,  # Store as integer value
            # Rewritten every publish cycle from the rolling window in
            # franka_sim.comm_constraints; this is only the value seen before a
            # control loop has run. Zero, because that is what the robot
            # reports then: control_command_success_rate "shows a value of zero
            # if no control or motion generator loop is currently running"
            # (libfranka include/franka/robot_state.h).
            "control_command_success_rate": 0.0,
            # Transformation matrices (4x4 column-major)
            "O_T_EE": list(_IDENTITY_4X4),
            "O_T_EE_d": list(_IDENTITY_4X4),
            "F_T_EE": list(_IDENTITY_4X4),
            "EE_T_K": list(_IDENTITY_4X4),
            "F_T_NE": list(_IDENTITY_4X4),
            "NE_T_EE": list(_IDENTITY_4X4),
            "tau_ext_hat_filtered": [0.0] * 7,
            "F_x_Cee": [0.0] * 3,
            "I_ee": [0.0] * 9,
            "m_ee": 0.0,
            "K_F_ext_hat_K": [0.0] * 6,
            "elbow": [0.0] * 2,
            "elbow_d": [0.0] * 2,
            "joint_contact": [0.0] * 7,
            "cartesian_contact": [0.0] * 6,
            "joint_collision": [0.0] * 7,
            "cartesian_collision": [0.0] * 6,
            "errors": [False] * 41,
            "m_load": 0.0,
            "I_load": [0.0] * 9,
            "F_x_Cload": [0.0] * 3,
            "O_F_ext_hat_K": [0.0] * 6,
            "O_dP_EE_d": [0.0] * 6,
            "O_ddP_O": [0.0] * 3,
            "elbow_c": [0.0] * 2,
            "delbow_c": [0.0] * 2,
            "ddelbow_c": [0.0] * 2,
            "O_T_EE_c": list(_IDENTITY_4X4),
            "O_dP_EE_c": [0.0] * 6,
            "O_ddP_EE_c": [0.0] * 6,
            "accelerometer_top": [0.0] * 18,  # 6 links x 3 axes
            "accelerometer_bottom": [0.0] * 18,  # 6 links x 3 axes
            "motion_generator_mode": 0,
            "controller_mode": 0,
            "reflex_reason": [False] * 41,
        }

    def pack_state(self) -> bytes:
        """Pack robot state into the libfranka_new (v10) binary RobotState.

        The v10 RobotState is float-based and 1377 bytes (rbk_types.h,
        ``#pragma pack(push, 1)``): ``message_id`` is uint64, the mode/error
        fields are uint8, and every floatarray is a 4-byte float (v9 used
        8-byte doubles, i.e. the old 2373-byte format). Serialized in one
        ``struct.pack`` via the module-level precompiled packer.
        """
        s = self.state
        return _ROBOT_STATE_PACKER.pack(
            s["message_id"],
            *s["O_T_EE"],
            *s["O_T_EE_d"],
            *s["F_T_EE"],
            *s["EE_T_K"],
            *s["F_T_NE"],
            *s["NE_T_EE"],
            s["m_ee"],
            *s["I_ee"],
            *s["F_x_Cee"][:3],
            s["m_load"],
            *s["I_load"],
            *s["F_x_Cload"][:3],
            *s["elbow"],
            *s["elbow_d"],
            *s["tau_J"],
            *s["tau_J_d"],
            *s["dtau_J"],
            *s["q"],
            *s["q_d"],
            *s["dq"],
            *s["dq_d"],
            *s["ddq_d"],
            *s["joint_contact"],
            *s["cartesian_contact"],
            *s["joint_collision"],
            *s["cartesian_collision"],
            *s["tau_ext_hat_filtered"],
            *s["O_F_ext_hat_K"],
            *s["K_F_ext_hat_K"],
            *s["O_dP_EE_d"],
            *s["O_ddP_O"][:3],
            *s["elbow_c"],
            *s["delbow_c"],
            *s["ddelbow_c"],
            *s["O_T_EE_c"],
            *s["O_dP_EE_c"],
            *s["O_ddP_EE_c"],
            *s["theta"],
            *s["dtheta"],
            *s["accelerometer_top"],
            *s["accelerometer_bottom"],
            s["motion_generator_mode"],
            s["controller_mode"],
            *s["errors"],
            *s["reflex_reason"],
            s["robot_mode"],
            s["control_command_success_rate"],
        )

    def update(self):
        """Advance to the next state frame with a monotonic message_id.

        A counter (rather than a wall-clock timestamp) keeps message_id strictly
        increasing even above 1kHz, which libfranka requires to accept each new
        state.
        """
        self._message_id += 1
        self.state["message_id"] = self._message_id

    def set_robot_mode(self, mode: RobotMode):
        """Set robot mode and store as integer value"""
        if not isinstance(mode, RobotMode):
            raise ValueError(f"Mode must be a RobotMode enum, got {type(mode)}")
        self.state["robot_mode"] = mode.value

    def set_motion_generator_mode(self, mode: int):
        """Set motion generator mode and store as integer value"""
        self.state["motion_generator_mode"] = mode

    def set_controller_mode(self, mode: int):
        """Set controller mode and store as integer value"""
        self.state["controller_mode"] = mode
