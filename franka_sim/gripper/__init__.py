"""The Franka Hand half of franka-sim: the gripper's own FCI server and backends.

Everything here serves the *hand* rather than the arm -- the gripper wire
protocol on port 1338 (``protocol``), the TCP/UDP server that speaks it
(``server``), the backend interface plus the kinematic stand-in that needs no
physics at all (``backend``), and the physics-backed backend that drives real
finger DOFs in a loaded scene (``physics``).

Docstring-only by design: it re-exports nothing, so importing this package
stays light. Import the submodule you need -- or reach for the name through
``franka_sim`` itself, which exports ``GripperBackend``, ``FrankaHandSim``,
``FrankaHandPhysics`` and ``FrankaGripperServer`` directly. The pre-1.0
top-level module paths (``franka_sim.gripper_backend`` and friends, one package
level up) were removed for the 1.0.0 release; import from
``franka_sim.gripper`` instead.
"""
