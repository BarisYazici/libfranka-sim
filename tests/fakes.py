"""Genesis stand-ins shared by the mobile-simulation tests.

Imported as ``from fakes import ...``: pytest puts the ``tests`` directory on
``sys.path`` (there is no ``__init__.py``), not the repository root.
"""

import numpy as np

from franka_sim.swerve_base import TMR_JOINT_ORDER


class FakeTensor:
    """Stands in for a Genesis tensor: ``.cpu().numpy()`` yields an array."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class FakeJoint:
    """A Genesis joint handle carrying only its local DOF index."""

    def __init__(self, dof_idx_local):
        self.dof_idx_local = dof_idx_local


class FakeEntity:
    """Minimal Genesis RigidEntity stand-in that records every control call."""

    def __init__(self, dof_positions=None, dof_velocities=None):
        self.joints = {name: FakeJoint(index) for index, name in enumerate(TMR_JOINT_ORDER)}
        self.dof_positions = dof_positions if dof_positions is not None else [0.1, 0.2, 0.3, 0.4]
        self.dof_velocities = dof_velocities if dof_velocities is not None else [1.0, 2.0, 3.0, 4.0]
        self.position_commands = []
        self.velocity_commands = []
        self.positions = []
        self.quaternions = []

    def get_joint(self, name):
        return self.joints[name]

    def control_dofs_position(self, values, dofs_idx_local):
        self.position_commands.append((np.asarray(values, dtype=float), list(dofs_idx_local)))

    def control_dofs_velocity(self, values, dofs_idx_local):
        self.velocity_commands.append((np.asarray(values, dtype=float), list(dofs_idx_local)))

    def get_dofs_position(self, dofs_idx_local):
        return FakeTensor([self.dof_positions[index] for index in dofs_idx_local])

    def get_dofs_velocity(self, dofs_idx_local):
        return FakeTensor([self.dof_velocities[index] for index in dofs_idx_local])

    def set_dofs_force_range(self, lower, upper, dofs_idx_local):
        pass

    def set_pos(self, position):
        self.positions.append(np.asarray(position, dtype=float))

    def set_quat(self, quaternion):
        self.quaternions.append(np.asarray(quaternion, dtype=float))
