import logging
import struct
import time
from typing import Any, Dict

from franka_sim.franka_protocol import RobotMode

logger = logging.getLogger(__name__)


class RobotState:
    """Manages the state of the Franka robot simulation"""

    def __init__(self):
        """Initialize the robot state with default values"""
        self.state = self._initialize_state()
        logger.info("Robot state initialized")

    def _initialize_state(self) -> Dict[str, Any]:
        """Initialize the robot state dictionary with default values"""
        return {
            "message_id": 0,
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
            "control_command_success_rate": 0.0,
            "time": 0.0,
            # Transformation matrices (4x4 column-major)
            "O_T_EE": [
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
            ],
            "O_T_EE_d": [
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
            ],
            "F_T_EE": [
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
            ],
            "EE_T_K": [
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
            ],
            "F_T_NE": [
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
            ],
            "NE_T_EE": [
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
            ],
            "tau_ext_hat_filtered": [0.0] * 7,
            "F_x_Cee": [0.0] * 6,
            "I_ee": [0.0] * 9,
            "m_ee": 0.0,
            "F_x_Ctotal": [0.0] * 6,
            "F_x_Cee_d": [0.0] * 6,
            "K_F_ext_hat_K": [0.0] * 6,
            "elbow": [0.0] * 2,
            "elbow_d": [0.0] * 2,
            "joint_contact": [0.0] * 7,
            "cartesian_contact": [0.0] * 6,
            "joint_collision": [0.0] * 7,
            "cartesian_collision": [0.0] * 6,
            "errors": [False] * 41,
            "current_errors": [False] * 41,
            "last_motion_errors": [False] * 41,
            "m_load": 0.0,
            "I_load": [0.0] * 9,
            "F_x_Cload": [0.0] * 3,
            "O_F_ext_hat_K": [0.0] * 6,
            "O_dP_EE_d": [0.0] * 6,
            "O_ddP_O": [0.0] * 3,
            "elbow_c": [0.0] * 2,
            "delbow_c": [0.0] * 2,
            "ddelbow_c": [0.0] * 2,
            "O_T_EE_c": [
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
            ],
            "O_dP_EE_c": [0.0] * 6,
            "O_ddP_EE_c": [0.0] * 6,
            "motion_generator_mode": 0,
            "controller_mode": 0,
            "reflex_reason": [False] * 41,
        }

    def pack_state(self) -> bytes:
        """Pack robot state into binary format for UDP transmission"""
        state = bytearray()

        # Pack state in libfranka-expected order
        state.extend(struct.pack("<Q", self.state["message_id"]))
        state.extend(struct.pack("<16d", *self.state["O_T_EE"]))
        state.extend(struct.pack("<16d", *self.state["O_T_EE_d"]))
        state.extend(struct.pack("<16d", *self.state["F_T_EE"]))
        state.extend(struct.pack("<16d", *self.state["EE_T_K"]))
        state.extend(struct.pack("<16d", *self.state["F_T_NE"]))
        state.extend(struct.pack("<16d", *self.state["NE_T_EE"]))
        state.extend(struct.pack("<d", self.state["m_ee"]))
        state.extend(struct.pack("<9d", *self.state["I_ee"]))
        state.extend(struct.pack("<3d", *self.state["F_x_Cee"][:3]))
        state.extend(struct.pack("<d", self.state["m_load"]))
        state.extend(struct.pack("<9d", *self.state["I_load"]))
        state.extend(struct.pack("<3d", *self.state["F_x_Cload"][:3]))
        state.extend(struct.pack("<2d", *self.state["elbow"]))
        state.extend(struct.pack("<2d", *self.state["elbow_d"]))
        state.extend(struct.pack("<7d", *self.state["tau_J"]))
        state.extend(struct.pack("<7d", *self.state["tau_J_d"]))
        state.extend(struct.pack("<7d", *self.state["dtau_J"]))
        state.extend(struct.pack("<7d", *self.state["q"]))
        state.extend(struct.pack("<7d", *self.state["q_d"]))
        state.extend(struct.pack("<7d", *self.state["dq"]))
        state.extend(struct.pack("<7d", *self.state["dq_d"]))
        state.extend(struct.pack("<7d", *self.state["ddq_d"]))
        state.extend(struct.pack("<7d", *self.state["joint_contact"]))
        state.extend(struct.pack("<6d", *self.state["cartesian_contact"]))
        state.extend(struct.pack("<7d", *self.state["joint_collision"]))
        state.extend(struct.pack("<6d", *self.state["cartesian_collision"]))
        state.extend(struct.pack("<7d", *self.state["tau_ext_hat_filtered"]))
        state.extend(struct.pack("<6d", *self.state["O_F_ext_hat_K"]))
        state.extend(struct.pack("<6d", *self.state["K_F_ext_hat_K"]))
        state.extend(struct.pack("<6d", *self.state["O_dP_EE_d"]))
        state.extend(struct.pack("<3d", *self.state["O_ddP_O"][:3]))
        state.extend(struct.pack("<2d", *self.state["elbow_c"]))
        state.extend(struct.pack("<2d", *self.state["delbow_c"]))
        state.extend(struct.pack("<2d", *self.state["ddelbow_c"]))
        state.extend(struct.pack("<16d", *self.state["O_T_EE_c"]))
        state.extend(struct.pack("<6d", *self.state["O_dP_EE_c"]))
        state.extend(struct.pack("<6d", *self.state["O_ddP_EE_c"]))
        state.extend(struct.pack("<7d", *self.state["theta"]))
        state.extend(struct.pack("<7d", *self.state["dtheta"]))
        state.extend(struct.pack("<B", self.state["motion_generator_mode"]))
        state.extend(struct.pack("<B", self.state["controller_mode"]))
        state.extend(struct.pack("<41B", *(1 if e else 0 for e in self.state["errors"])))
        state.extend(struct.pack("<41B", *(1 if r else 0 for r in self.state["reflex_reason"])))
        state.extend(struct.pack("<B", self.state["robot_mode"]))
        state.extend(struct.pack("<d", self.state["control_command_success_rate"]))

        return bytes(state)

    def update(self):
        """Update the robot state for the next iteration"""
        self.state["message_id"] = int(time.time() * 1000)
        self.state["time"] = time.time()
        if self.state["message_id"] % 1000 == 0:  # Log every 1000 updates
            logger.debug(
                f"Robot state updated: message_id={self.state['message_id']}, "
                f"mode={RobotMode(self.state['robot_mode']).name}, "
                f"controller_mode={self.state['controller_mode']}, "
                f"motion_generator_mode={self.state['motion_generator_mode']}"
            )

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
