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

    def __init__(self, dof_positions=None, dof_velocities=None, root_dofs=(100, 101, 102)):
        self.joints = {name: FakeJoint(index) for index, name in enumerate(TMR_JOINT_ORDER)}
        self.dof_positions = dof_positions if dof_positions is not None else [0.1, 0.2, 0.3, 0.4]
        self.dof_velocities = dof_velocities if dof_velocities is not None else [1.0, 2.0, 3.0, 4.0]
        self.base_joint = FakeJoint(list(root_dofs)) if root_dofs else None
        self.position_commands = []
        self.velocity_commands = []
        self.positions = []
        self.quaternions = []
        #: ``(values, dofs, zero_velocity)`` for every base-pose write.
        self.pose_writes = []
        #: ``(values, dofs)`` for every set_dofs_velocity call.
        self.set_velocity_calls = []

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

    def set_dofs_velocity(self, values, dofs_idx_local):
        self.set_velocity_calls.append((np.asarray(values, dtype=float), list(dofs_idx_local)))

    def set_pos(self, position, zero_velocity=True):
        self.positions.append(np.asarray(position, dtype=float))
        self.pose_writes.append(("pos", np.asarray(position, dtype=float), zero_velocity))

    def set_quat(self, quaternion, zero_velocity=True):
        self.quaternions.append(np.asarray(quaternion, dtype=float))
        self.pose_writes.append(("quat", np.asarray(quaternion, dtype=float), zero_velocity))


class FakeLink:
    """A Genesis link handle returning a fixed pose."""

    def __init__(self, position, quaternion):
        self._position = np.asarray(position, dtype=float)
        self._quaternion = np.asarray(quaternion, dtype=float)

    def get_pos(self):
        return FakeTensor(self._position)

    def get_quat(self):
        return FakeTensor(self._quaternion)


class FakeDuoEntity:
    """A Genesis RigidEntity stand-in carrying the wheel and both arm joints."""

    def __init__(self):
        from franka_sim.mobile_duo_sim import (
            ARM_EE_LINKS,
            ARM_JOINT_NAMES,
            ROLE_LEFT,
            ROLE_RIGHT,
            SPINE_JOINT_NAME,
        )

        names = list(TMR_JOINT_ORDER)
        names += ARM_JOINT_NAMES[ROLE_LEFT] + ARM_JOINT_NAMES[ROLE_RIGHT]
        names.append(SPINE_JOINT_NAME)
        self.joints = {name: FakeJoint(index) for index, name in enumerate(names)}
        self.n_dofs = len(names)
        self.base_joint = FakeJoint([100, 101, 102])
        self.dof_positions = np.arange(self.n_dofs, dtype=float) / 100.0
        self.dof_velocities = np.arange(self.n_dofs, dtype=float) / 10.0
        self.links = {
            ARM_EE_LINKS[ROLE_LEFT]: FakeLink([0.4, 0.2, 1.1], [1.0, 0.0, 0.0, 0.0]),
            ARM_EE_LINKS[ROLE_RIGHT]: FakeLink([0.4, -0.2, 1.1], [1.0, 0.0, 0.0, 0.0]),
        }
        self.position_commands = []
        self.velocity_commands = []
        self.force_commands = []
        self.set_position_calls = []
        self.positions = []
        self.quaternions = []
        #: ``(values, dofs, zero_velocity)`` for every set_dofs_position call.
        self.set_position_writes = []
        #: ``(values, dofs)`` for every set_dofs_velocity call.
        self.set_velocity_calls = []

    def get_joint(self, name):
        return self.joints[name]

    def get_link(self, name):
        return self.links[name]

    def get_dofs_position(self, dofs_idx_local):
        return FakeTensor([self.dof_positions[index] for index in dofs_idx_local])

    def get_dofs_velocity(self, dofs_idx_local):
        return FakeTensor([self.dof_velocities[index] for index in dofs_idx_local])

    def control_dofs_position(self, values, dofs_idx_local):
        self.position_commands.append((np.asarray(values, dtype=float), list(dofs_idx_local)))

    def control_dofs_velocity(self, values, dofs_idx_local):
        self.velocity_commands.append((np.asarray(values, dtype=float), list(dofs_idx_local)))

    def control_dofs_force(self, values, dofs_idx_local):
        self.force_commands.append((np.asarray(values, dtype=float), list(dofs_idx_local)))

    def set_dofs_force_range(self, lower, upper, dofs_idx_local):
        pass

    def set_dofs_damping(self, damping, dofs_idx_local):
        pass

    def set_dofs_position(self, values, dofs_idx_local, zero_velocity=True):
        values = np.asarray(values, dtype=float)
        self.set_position_calls.append((values, list(dofs_idx_local)))
        self.set_position_writes.append((values, list(dofs_idx_local), zero_velocity))
        for value, index in zip(values, dofs_idx_local):
            self.dof_positions[index] = value

    def set_dofs_velocity(self, values, dofs_idx_local):
        self.set_velocity_calls.append((np.asarray(values, dtype=float), list(dofs_idx_local)))

    def set_pos(self, position, zero_velocity=True):
        self.positions.append(np.asarray(position, dtype=float))

    def set_quat(self, quaternion, zero_velocity=True):
        self.quaternions.append(np.asarray(quaternion, dtype=float))
