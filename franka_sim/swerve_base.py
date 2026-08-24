"""Alias of :mod:`franka_sim.mobile.swerve_base` (moved); kept so old imports work.

Not a copy and not a re-export facade: the line below replaces this module in
``sys.modules`` with the real one, so ``franka_sim.swerve_base`` and
``franka_sim.mobile.swerve_base`` are the *same module object*. That is what keeps
``monkeypatch.setattr("franka_sim.swerve_base.<attr>", ...)`` -- and any other
patching through the old path -- landing on the attributes the moved module
actually reads.
"""

import sys

from franka_sim.mobile import swerve_base as _moved

sys.modules[__name__] = _moved
