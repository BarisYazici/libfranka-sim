"""Alias of :mod:`franka_sim.mobile.runner` (moved); kept so old imports work.

Not a copy and not a re-export facade: the line below replaces this module in
``sys.modules`` with the real one, so ``franka_sim.mobile_duo_runner`` and
``franka_sim.mobile.runner`` are the *same module object*. That is what keeps
``monkeypatch.setattr("franka_sim.mobile_duo_runner.<attr>", ...)`` -- and any other
patching through the old path -- landing on the attributes the moved module
actually reads.
"""

import sys

from franka_sim.mobile import runner as _moved

sys.modules[__name__] = _moved
