"""Genesis stand-ins shared by the mobile-simulation tests.

Imported as ``from fakes import ...``: pytest puts the ``tests`` directory on
``sys.path`` (there is no ``__init__.py``), not the repository root.
"""

import numpy as np

from franka_sim.mobile.swerve_base import TMR_JOINT_ORDER


class FakeTensor:
    """Stands in for a Genesis tensor: ``.cpu().numpy()`` yields an array."""

    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class FakeJoint:
    """A Genesis joint handle carrying its local DOF and qpos indices.

    ``q_idx_local`` is None for the ordinary single-DOF joints (nothing reads it
    there); a free root joint carries seven, which is what ``SwerveBase`` writes
    the base pose through.
    """

    def __init__(self, dof_idx_local, q_idx_local=None):
        self.dof_idx_local = dof_idx_local
        self.q_idx_local = q_idx_local


class FakeEntity:
    """Minimal Genesis RigidEntity stand-in that records every control call."""

    def __init__(
        self,
        dof_positions=None,
        dof_velocities=None,
        root_dofs=(100, 101, 102),
        root_qs=(200, 201, 202, 203, 204, 205, 206),
    ):
        self.joints = {name: FakeJoint(index) for index, name in enumerate(TMR_JOINT_ORDER)}
        self.dof_positions = dof_positions if dof_positions is not None else [0.1, 0.2, 0.3, 0.4]
        self.dof_velocities = dof_velocities if dof_velocities is not None else [1.0, 2.0, 3.0, 4.0]
        self.base_joint = (
            FakeJoint(list(root_dofs), list(root_qs) or None) if root_dofs else None
        )
        self.position_commands = []
        self.velocity_commands = []
        self.positions = []
        self.quaternions = []
        #: ``(values, dofs, zero_velocity)`` for every base-pose write, whether
        #: it arrived as set_qpos or as the set_pos/set_quat fallback pair.
        self.pose_writes = []
        #: ``(values, qs, zero_velocity)`` for every set_qpos call.
        self.qpos_writes = []
        #: ``(values, dofs)`` for every set_dofs_velocity call.
        self.set_velocity_calls = []

    def get_joint(self, name):
        return self.joints[name]

    def control_dofs_position(self, values, dofs_idx_local):
        self.position_commands.append((np.asarray(values, dtype=float), list(dofs_idx_local)))

    def control_dofs_velocity(self, values, dofs_idx_local):
        self.velocity_commands.append((np.asarray(values, dtype=float), list(dofs_idx_local)))

    def get_dofs_position(self, dofs_idx_local=None):
        if dofs_idx_local is None:
            return FakeTensor(self.dof_positions)
        return FakeTensor([self.dof_positions[index] for index in dofs_idx_local])

    def get_dofs_velocity(self, dofs_idx_local=None):
        if dofs_idx_local is None:
            return FakeTensor(self.dof_velocities)
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

    def set_qpos(self, values, qs_idx_local, zero_velocity=True):
        """Record a free root joint's ``[x, y, z, qw, qx, qy, qz]`` qpos write.

        That layout is the base link's pose, so it is also recorded under
        ``positions``/``quaternions``: which setter carried the pose is a
        kernel-launch-count decision, not something the pose assertions care
        about. ``values`` is copied because the caller reuses one scratch array.
        """
        values = np.asarray(values, dtype=float).copy()
        self.qpos_writes.append((values, list(qs_idx_local), zero_velocity))
        self.positions.append(values[:3])
        self.quaternions.append(values[3:7])
        self.pose_writes.append(("qpos", values, zero_velocity))


class FakeLink:
    """A Genesis link handle returning a fixed pose."""

    def __init__(self, position, quaternion, idx_local=0):
        self._position = np.asarray(position, dtype=float)
        self._quaternion = np.asarray(quaternion, dtype=float)
        self.idx_local = idx_local

    def get_pos(self):
        return FakeTensor(self._position)

    def get_quat(self):
        return FakeTensor(self._quaternion)


class FakeDuoEntity:
    """A Genesis RigidEntity stand-in carrying the wheel and both arm joints."""

    def __init__(self):
        from franka_sim.mobile.duo_sim import (
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
        self.base_joint = FakeJoint([100, 101, 102], [200, 201, 202, 203, 204, 205, 206])
        self.dof_positions = np.arange(self.n_dofs, dtype=float) / 100.0
        self.dof_velocities = np.arange(self.n_dofs, dtype=float) / 10.0
        # The real entity carries dozens of links and the two flanges sit at
        # large, non-adjacent local indices, so the filler links below are load
        # bearing: with the flanges at 0 and 1 the whole-entity link reads would
        # still line up under an off-by-one or a swapped pair.
        self.links = {
            ARM_EE_LINKS[ROLE_LEFT]: FakeLink([0.4, 0.2, 1.1], [1.0, 0.0, 0.0, 0.0], idx_local=3),
            ARM_EE_LINKS[ROLE_RIGHT]: FakeLink(
                [0.4, -0.2, 1.1], [0.0, 1.0, 0.0, 0.0], idx_local=9
            ),
        }
        for idx in range(12):
            if idx in (3, 9):
                continue
            self.links[f"filler_link{idx}"] = FakeLink(
                [float(idx), -float(idx), 0.5], [0.0, 0.0, 1.0, 0.0], idx_local=idx
            )
        self.position_commands = []
        self.velocity_commands = []
        self.force_commands = []
        self.set_position_calls = []
        self.positions = []
        self.quaternions = []
        #: ``(values, qs, zero_velocity)`` for every set_qpos call.
        self.qpos_writes = []
        #: ``(values, dofs, zero_velocity)`` for every set_dofs_position call.
        self.set_position_writes = []
        #: ``(values, dofs)`` for every set_dofs_velocity call.
        self.set_velocity_calls = []
        #: ``(damping, dofs)`` for every set_dofs_damping call.
        self.damping_calls = []
        #: ``(kp, dofs)`` / ``(kv, dofs)`` for every control-gain call.
        self.kp_calls = []
        self.kv_calls = []
        #: Whole-entity link-pose reads, which the physics loop decimates.
        self.links_read_count = 0

    def get_joint(self, name):
        return self.joints[name]

    def get_link(self, name):
        return self.links[name]

    def get_dofs_position(self, dofs_idx_local=None):
        if dofs_idx_local is None:
            return FakeTensor(self.dof_positions)
        return FakeTensor([self.dof_positions[index] for index in dofs_idx_local])

    def get_dofs_velocity(self, dofs_idx_local=None):
        if dofs_idx_local is None:
            return FakeTensor(self.dof_velocities)
        return FakeTensor([self.dof_velocities[index] for index in dofs_idx_local])

    def get_links_pos(self):
        """One row per link, in dense ``idx_local`` order, as Genesis returns."""
        self.links_read_count += 1
        ordered = sorted(self.links.values(), key=lambda link: link.idx_local)
        return FakeTensor([link._position for link in ordered])

    def get_links_quat(self):
        ordered = sorted(self.links.values(), key=lambda link: link.idx_local)
        return FakeTensor([link._quaternion for link in ordered])

    def control_dofs_position(self, values, dofs_idx_local):
        self.position_commands.append((np.asarray(values, dtype=float), list(dofs_idx_local)))

    def control_dofs_velocity(self, values, dofs_idx_local):
        self.velocity_commands.append((np.asarray(values, dtype=float), list(dofs_idx_local)))

    def control_dofs_force(self, values, dofs_idx_local):
        self.force_commands.append((np.asarray(values, dtype=float), list(dofs_idx_local)))

    def set_dofs_force_range(self, lower, upper, dofs_idx_local):
        pass

    def set_dofs_damping(self, damping, dofs_idx_local):
        self.damping_calls.append((np.asarray(damping, dtype=float), list(dofs_idx_local)))

    def set_dofs_kp(self, kp, dofs_idx_local):
        self.kp_calls.append((np.asarray(kp, dtype=float), list(dofs_idx_local)))

    def set_dofs_kv(self, kv, dofs_idx_local):
        self.kv_calls.append((np.asarray(kv, dtype=float), list(dofs_idx_local)))

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

    def set_qpos(self, values, qs_idx_local, zero_velocity=True):
        """See :meth:`FakeEntity.set_qpos`: a free root's qpos is its base pose."""
        values = np.asarray(values, dtype=float).copy()
        self.qpos_writes.append((values, list(qs_idx_local), zero_velocity))
        self.positions.append(values[:3])
        self.quaternions.append(values[3:7])
