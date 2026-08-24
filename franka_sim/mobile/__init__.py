"""The mobile-robot half of franka-sim: swerve bases, the FR3 duo and the spine.

Everything here serves a *moving* platform rather than a bolted-down arm --
the swerve kinematics and base model (``swerve_kinematics``, ``swerve_base``),
the TMR base scene (``tmr_genesis_sim``), the mobile FR3 duo scenes for both
physics backends (``duo_sim`` for Genesis, ``duo_mujoco_sim`` for MuJoCo) with
the engine-agnostic pieces they share (``common``), the multi-bridge process
that fronts a duo scene with three FCI servers (``runner``) and the fake spine
REST device (``spine_stub``).

Docstring-only by design: it re-exports nothing, so importing this package
pulls in no physics engine. Import the submodule you need -- or reach for the
name through ``franka_sim`` itself, which resolves the engine-backed ones
lazily. Each module's pre-split top-level path (``franka_sim.mobile_duo_sim``
and friends) still resolves, aliased to the module object that lives here.
"""
