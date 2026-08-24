"""Alias of :mod:`franka_sim.mobile.duo_mujoco_sim` (moved); kept so old imports work.

Not a copy and not a re-export facade: the line below replaces this module in
``sys.modules`` with the real one, so ``franka_sim.mobile_duo_mujoco_sim`` and
``franka_sim.mobile.duo_mujoco_sim`` are the *same module object*. That is what keeps
``monkeypatch.setattr("franka_sim.mobile_duo_mujoco_sim.<attr>", ...)`` -- and any other
patching through the old path -- landing on the attributes the moved module
actually reads.
"""

import sys

from franka_sim.mobile import duo_mujoco_sim as _moved

sys.modules[__name__] = _moved
