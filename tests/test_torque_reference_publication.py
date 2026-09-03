"""Regression: the published ``tau_J_d`` must never lag the checker's reference.

``franka_hardware`` rate-limits torques against the robot's own reported
reference (``franka::limitRate(kMaxTorqueRate, tau, current_state_.tau_J_d)``,
``franka_hardware/src/robot.cpp:171``) and libfranka's ``receiveRobotState``
hands it the newest state on the socket, so it cannot route around a reference
the server published one command behind. If the state that goes out carries the
*previous* command's torque while the checker has already recorded this one, the
client saturates its limiter from the stale value and its next command
differences to ``999.999 + |its own previous torque rate|`` over one cycle -- a
conforming controller aborted with ``controller_torque_discontinuity: tau_J_d
joint 4 = 1002.81 Nm/s, limit 1000 Nm/s`` off a 2.81 Nm/s ramp.

Hardware has no such window: the state for cycle k carries the torque Control
applied at cycle k. The sim has two threads and a drain gate that gives up after
``_DRAIN_GATE_TIMEOUT``, which is what a starved 2-core CI runner makes it do.

Suggested home: ``tests/test_motion_limits.py`` (next to the eps-bug family) or
``tests/test_comm_constraints.py``.
"""

import struct
from unittest.mock import Mock

import pytest

from franka_sim.control_modes import ControlMode
from franka_sim.franka_protocol import LibfrankaControllerMode, LibfrankaMotionGeneratorMode
from franka_sim.franka_sim_server import FrankaSimServer

DT = 1e-3
KMAX = 1000.0 - 1e-3  # franka::kMaxTorqueRate
J = 3  # joint 4


def wire_float(values):
    """What the client actually reads: the RobotState struct is float32."""
    return list(struct.unpack("<7f", struct.pack("<7f", *values)))


def limit_rate(target, last):
    """franka::limitRate(max_derivatives, commanded, last_commanded), per joint."""
    return [
        last[i] + max(min((target[i] - last[i]) / DT, KMAX), -KMAX) * DT for i in range(7)
    ]


@pytest.fixture
def torque_server():
    sim = Mock()
    sim.get_robot_state.return_value = {"q": [0.0] * 7, "dq": [0.0] * 7, "self_collision": None}
    server = FrankaSimServer(enable_vis=False, physics_sim=sim, enable_gripper=False)
    server.enforce_motion_limits = True
    server.control_mode = ControlMode.TORQUE
    server.robot_state.state["motion_generator_mode"] = LibfrankaMotionGeneratorMode.kIdle
    server.robot_state.state["controller_mode"] = (
        LibfrankaControllerMode.kExternalController
    )
    server.robot_state.state["tau_J_d"] = [0.0] * 7
    server.motion_limits.start_motion(
        ControlMode.TORQUE, dict(server.robot_state.state), motion_id=1
    )
    return server


def torque_command(tau, message_id):
    return {"tau_J_d": list(tau), "message_id": int(message_id)}


def test_an_accepted_torque_is_published_before_anything_can_read_it(torque_server):
    """The invariant, stated directly: absorb publishes what it recorded."""
    server = torque_server
    server.motion_limits.note_published(1)
    tau = [0.0] * 7
    tau[J] = 0.5

    assert server._absorb_within_motion_limits(torque_command(tau, 1), fresh=True)

    # Before _dispatch_control_command has run: the wire already agrees with the
    # reference the next command will be differenced against.
    assert server.robot_state.state["tau_J_d"][J] == pytest.approx(0.5)


def test_a_rate_limited_client_reading_the_published_state_is_never_a_discontinuity(
    torque_server,
):
    """The franka_ros2 client, end to end through the checker.

    The client ramps gently, then saturates its limiter -- always against the
    newest state the server published, which is all libfranka ever gives it.
    Without the fix the publish loop can emit a state one command stale and this
    aborts at ``1002.81 Nm/s``.
    """
    server = torque_server
    published = []
    prior_rate = 2.81  # Nm/s, the ramp the observed abort came off

    for cycle in range(1, 13):
        # The state-publish thread: it may run *between* the absorb and the
        # dispatch, which is the window this test pins shut.
        server.motion_limits.note_published(cycle)
        published.append(wire_float(server.robot_state.state["tau_J_d"]))

        reference = published[-1]  # franka_hardware's current_state_.tau_J_d
        if cycle <= 6:
            target = [reference[i] - prior_rate * DT for i in range(7)]
        else:
            target = [1e6] * 7  # saturate the limiter
        tau = limit_rate(target, reference)
        tau[J] = min(tau[J], 80.0)

        assert server._absorb_within_motion_limits(
            torque_command(tau, cycle), fresh=True
        ), f"cycle {cycle}: a conforming rate-limited client was aborted"
