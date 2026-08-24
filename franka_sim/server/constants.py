#!/usr/bin/env python3
"""Module-level constants, tunables and enums shared by the server modules.

Every value here is either a protocol constant or a tuning choice the split
server modules read; each carries the note explaining what it is for and what
breaks if it moves. Nothing here holds per-session state.

A library module must not configure the root logger -- that's the
embedding application's call (see run_server.main()'s guarded
basicConfig). A module-level basicConfig() here installs a root handler
on import, which "wins" the first-call race and silently caps every
other logger (including run_server's own) at the level given here.
The logger is created here, once, under the pre-split module's name so that
callers (and tests) capturing ``franka_sim.franka_sim_server`` keep seeing every
record the server modules emit.
"""

import enum
import importlib
import logging

logger = logging.getLogger("franka_sim.franka_sim_server")


#: RobotState fields that report what was *commanded*, not what was measured.
#: The FCI server owns them, so a physics backend's snapshot never overwrites
#: them on the publish path: libfranka defines them as "``q_{c,k-1}``,
#: ``dq_{c,k-1}`` and ``ddq_{c,k-1}`` [...] always sent back to the user in the
#: robot state as ``q_d``, ``dq_d`` and ``ddq_d``" (``docs/overview.rst``), and
#: a conforming client feeds them straight back in -- libfranka's default
#: command low-pass filter blends every command with them. A backend that
#: echoed measured values here, or published its own copy a physics step late,
#: put a wobble into the client's own commands that no client could see or
#: prevent.
#:
#: **Arm roles only.** The mobile base's motion generator is Cartesian: the
#: client commands ``O_dP_EE_c`` and libfranka's filter blends it with
#: ``O_dP_EE_d``, both of which this server owns and writes on every datagram.
#: Nothing commands the base bridge's ``q_d``/``dq_d``/``ddq_d`` -- they
#: describe the four swerve steer/drive joints, which the base's own onboard
#: controller servos -- so the backend's reading of them is the only source
#: there is, and filtering it out simply froze ``q_d`` at the first frame and
#: pinned ``dq_d``/``ddq_d`` to zero on the bridge the teleop reads. See
#: :attr:`FrankaSimServer._server_owned_state_fields`.
#:
#: ``O_T_EE_d``/``O_T_EE_c``, ``O_dP_EE_d``/``O_dP_EE_c`` and
#: ``elbow_d``/``elbow_c`` joined this set once the Cartesian pose, twist and
#: elbow interfaces became FCI-owned fields in their own right (see
#: :meth:`FrankaSimServer._echo_commanded_cartesian`,
#: :meth:`FrankaSimServer._publish_commanded_pose` and
#: :meth:`FrankaSimServer._publish_elbow`), for the identical reason the joint
#: fields are here: none of the physics backends' ``get_robot_state()``
#: snapshots actually carry these keys today, so listing them changes nothing
#: yet, but it keeps this set an honest statement of FCI ownership rather than
#: one that happens to be accidentally correct only because no backend has
#: caught up. (The mobile base takes none of this set -- there the swerve
#: backend really is the only source for ``q_d``/``dq_d``/``ddq_d``, and the
#: base's own twist echo is :meth:`FrankaSimServer._handle_cartesian_velocity`.)
COMMANDED_STATE_FIELDS = (
    "q_d",
    "dq_d",
    "ddq_d",
    "tau_J_d",
    "O_T_EE_d",
    "O_T_EE_c",
    "O_dP_EE_d",
    "O_dP_EE_c",
    "elbow_d",
    "elbow_c",
)

#: Keys a physics backend publishes in its snapshot that are *not* RobotState
#: wire fields: internal readings the FCI layer consumes and never broadcasts.
#: Filtered out of the publish-path state update so ``robot_state.state`` stays
#: exactly the set of fields ``pack_state`` knows about, on every role including
#: the mobile base.
#:
#: ``O_dP_EE`` is the measured end-effector translational velocity the safety
#: controller's Cartesian check reads (see
#: :meth:`FrankaSimServer._run_safety_velocity_check`). libfranka's
#: ``RobotState`` has no measured Cartesian velocity -- only the commanded
#: ``O_dP_EE_d``/``O_dP_EE_c`` -- so there is nowhere on the wire for it to go.
#:
#: ``O_J_EE`` is the 6x7 EE Jacobian a ``Move`` conditions its start pose on
#: (:meth:`FrankaSimServer._refuse_move_at_singular_pose`). libfranka computes
#: its own from the model it downloads and the FCI never sends one, so it has no
#: wire field either -- and unlike the rest of the snapshot it is a 2-D
#: ``ndarray``, which the state dict's consumers (``pack_state``, the
#: state-shaped copies :meth:`FrankaSimServer._publish_hold_setpoint` hands out)
#: have no idea what to do with.
INTERNAL_SIM_STATE_FIELDS = ("O_dP_EE", "O_J_EE")

#: How long :meth:`FrankaSimServer.stop` waits for the gripper server's accept
#: loop to notice its socket was closed. The wait is normally microseconds --
#: closing the listening socket drops accept() out immediately -- so this only
#: bounds the case of a gripper command stuck inside a backend call. Short,
#: because the thread is a daemon and abandoning it costs nothing.
GRIPPER_JOIN_TIMEOUT_S = 1.0

#: How close to zero |measured dq| must be, per joint, for
#: ``FrankaSimServer._wait_for_standstill`` to count a cycle as settled.
#: The real robot's own ``AutomaticErrorRecovery`` inherently completes with
#: the arm at rest -- recovery on hardware clears the reflex and hands the
#: joints back to a holding controller only once the safety layer has stopped
#: reacting to the abort, and by then the arm has stopped. The sim's own
#: reflex handler used to reply the instant the TCP request arrived, so a
#: client that Move'd again straight away started its next motion while the
#: arm was still decelerating from whatever speed the abort caught it at
#: (observed here at ~2.6 rad/s); the new motion's client-side start-pose
#: guard then saw measured q drifting away from the commanded q it had just
#: sent and threw "Performance threshold reached" a few milliseconds in. This
#: is a sim choice -- libfranka publishes no settling tolerance of its own --
#: sized well inside :data:`franka_sim.motion_limits.MEASURED_JOINT_VELOCITY_MARGIN`
#: (0.1 rad/s) so the wait cannot itself look like "still moving" once the
#: idle hold has actually caught the arm.
AUTOMATIC_ERROR_RECOVERY_SETTLE_VELOCITY = 0.005  # rad/s

#: Consecutive settled polls required before the wait is satisfied, at
#: :data:`AUTOMATIC_ERROR_RECOVERY_POLL_PERIOD` apiece -- 50 x 1 ms = 50 ms.
#: "Sustained" rather than "one clean sample" so a single lucky reading
#: between two ring oscillations of a not-yet-caught arm cannot end the wait
#: early.
AUTOMATIC_ERROR_RECOVERY_SETTLE_CYCLES = 50

#: How often ``_wait_for_standstill`` re-reads the latest published ``dq``
#: while waiting. Short on purpose: this runs on the per-connection TCP
#: thread, not the state-publish loop, so a short poll costs that thread
#: nothing else is waiting on, and it is what lets the wait notice
#: :attr:`FrankaSimServer.running` going false promptly during shutdown
#: instead of sleeping through it.
AUTOMATIC_ERROR_RECOVERY_POLL_PERIOD = 0.001  # s

#: Hard ceiling on the wait -- capped well under libfranka's own TCP receive
#: timeout, not the "3 s" figure this constant started at.
#:
#: **Not the sim's choice to make generous.** libfranka's ``Network``
#: defaults ``tcp_timeout`` to ``std::chrono::seconds(1)``
#: (``libfranka/src/network.h``: ``Network(..., std::chrono::milliseconds
#: tcp_timeout = std::chrono::seconds(1), ...)``), and that is the client's
#: own receive timeout on *every* TCP command response, ``AutomaticErrorRecovery``
#: included -- there is no per-command override for it. A 3 s wait was tried
#: against a stock libfranka client and reproduced exactly this: the reply
#: arrived correctly, but ~1.000 s after the request the client had already
#: decided the connection was dead (Poco's ``TimeoutException`` surfaces to
#: the caller as "libfranka: TCP connection got interrupted" /
#: "libfranka: UDP receive: Timeout"), and every later motion on that
#: connection then failed too as the client kept retrying against a server
#: still mid-wait
#: from the *previous* abort. Going over 1 s here is not "slower, but still
#: correct" -- it silently breaks the wire protocol.
#:
#: 0.7 s leaves ~300 ms of margin under that 1 s ceiling for scheduling
#: jitter and the response's own transmission, while still being comfortably
#: above the 50 ms the settle-cycle count needs for a motion that is actually
#: decelerating under the idle hold. On a run where the arm has not settled
#: by then, the timeout path below fires and replies success anyway -- a
#: partial wait that reduces the residual velocity the next motion starts
#: against, which is strictly better than the pre-fix instant reply, even
#: when it cannot certify full standstill within budget.
AUTOMATIC_ERROR_RECOVERY_TIMEOUT = 0.7  # s

#: Single-arm physics backends, mapped to the module and class implementing the
#: simulator contract this server consumes. Imported lazily in
#: :func:`resolve_sim_class` so choosing one backend never pays the other's
#: (multi-second, native) import cost -- and so an install without Genesis can
#: still run the default one.
SINGLE_ARM_PHYSICS = {
    "genesis": ("franka_sim.franka_genesis_sim", "FrankaGenesisSim"),
    "mujoco": ("franka_sim.mujoco_franka_sim", "MujocoFrankaSim"),
}

#: Backend used when the caller does not inject a simulator or name one. MuJoCo
#: holds real time at the 1 ms step the FCI serves; Genesis needs 2.5 ms.
DEFAULT_PHYSICS = "mujoco"

#: Closest two consecutive ``RobotState`` datagrams may be published, as a
#: fraction of the 1 ms cycle. The FCI publishes one state per cycle and expects
#: one answer per state; two states microseconds apart give the client no cycle
#: to answer the first in, and the second then carries a ``q_d`` that predates
#: that answer -- which libfranka's own low-pass filter closes around. See the
#: pacing arithmetic in ``FrankaSimServer.start_state_transmission``.
_MIN_STATE_SPACING = 0.8


def resolve_sim_class(physics: str = DEFAULT_PHYSICS):
    """Import and return the single-arm simulator class for one physics backend."""
    if physics not in SINGLE_ARM_PHYSICS:
        raise ValueError(
            f"Unknown physics backend {physics!r}; expected one of {sorted(SINGLE_ARM_PHYSICS)}"
        )
    module_name, class_name = SINGLE_ARM_PHYSICS[physics]
    return getattr(importlib.import_module(module_name), class_name)


class RobotMode(enum.IntEnum):
    """Operating modes of the Franka robot"""

    kOther = 0
    kIdle = 1
    kMove = 2
    kGuiding = 3
    kReflex = 4
    kUserStopped = 5
    kAutomaticErrorRecovery = 6
