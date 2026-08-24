"""The MuJoCo mobile-duo path must import and (where assets allow) run without
Genesis installed at all.

``franka_sim.mobile.duo_mujoco_sim`` used to import ``ControlMode`` and the
shared constants from ``franka_sim.franka_genesis_sim``/``franka_sim.mobile.duo_sim``,
both of which import ``genesis`` at module level -- so ``--physics mujoco``
would crash on import on a genesis-free (mujoco-only) install, even though it
never touches Genesis at runtime.
``franka_sim.mobile.runner`` had the same problem one level up (it
imported the Genesis-flavoured ``MobileDuoScene`` just for a type reference).

This is machine-verified in a real subprocess with a meta-path finder that
raises on any attempt to import ``genesis`` or ``taichi`` (Genesis' compute
backend) -- the surest way to prove the import graph genuinely never reaches
either package, rather than relying on ``sys.modules`` inspection in-process
(where genesis may already be loaded by another test).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENE_URDF = REPO_ROOT / "assets" / "mobile_duo" / "mobile_fr3_duo.urdf"
MESH_ROOT = Path(os.environ.get("MOBILE_DUO_MESH_ROOT", Path.home() / "franka_description-jazzy"))

#: Common preamble every subprocess script runs first: a meta-path finder that
#: turns any genesis/taichi import into a hard failure, so a stray import
#: anywhere in the chain cannot hide as a silent no-op.
_BLOCK_GENESIS_PREAMBLE = """
import sys


class _BlockGenesis:
    # Meta-path finder that raises on genesis/taichi instead of resolving them.
    BLOCKED = ("genesis", "taichi")

    def find_spec(self, name, path, target=None):
        if name in self.BLOCKED or any(name.startswith(b + ".") for b in self.BLOCKED):
            raise ImportError(f"blocked: {name!r} must not be imported on the mujoco-only path")
        return None


sys.meta_path.insert(0, _BlockGenesis())
"""

#: Runs regardless of whether the real scene assets are present: cheap, and
#: only checks that the modules import and a scene can be constructed.
_IMPORT_ONLY_SCRIPT = (
    _BLOCK_GENESIS_PREAMBLE
    + """
import franka_sim.mobile.duo_mujoco_sim as mujoco_mod
import franka_sim.mobile.runner as runner_mod
import franka_sim.run_server as run_server_mod

assert "genesis" not in sys.modules
assert "taichi" not in sys.modules

# Import-safe construction: __init__ only stores paths/flags, it must not
# touch the filesystem or mujoco until initialize_simulation() is called.
scene = mujoco_mod.MobileDuoMujocoScene("/nonexistent/duo.urdf", enable_vis=False)
assert scene.model is None

# resolve_scene_class("mujoco") is the code path run_server.py actually takes
# for --physics mujoco; exercise it too so the CLI entrypoint is covered.
cls = run_server_mod.resolve_scene_class("mujoco")
assert cls is mujoco_mod.MobileDuoMujocoScene

# MobileDuoRunner itself must be importable/constructible against a mujoco
# scene without ever reaching for the Genesis scene type (only imported under
# TYPE_CHECKING).
assert runner_mod.MobileDuoRunner is not None

assert "genesis" not in sys.modules
assert "taichi" not in sys.modules
print("IMPORT_ONLY_OK")
"""
)

#: Runs only when the generated scene URDF and the franka_description mesh
#: checkout are present: builds the real model and steps it, still under the
#: same genesis-blocking finder, so the whole physics build (URDF resolution,
#: MjSpec compile, visual upgrades) is proven genesis-free too, not just the
#: import statements. Paths come in through the environment so this script
#: needs no string-formatting of Python source (which would collide with the
#: f-strings above).
_FULL_BUILD_SCRIPT = (
    _BLOCK_GENESIS_PREAMBLE
    + """
import os

import mujoco

from franka_sim.mobile.duo_mujoco_sim import MobileDuoMujocoScene

scene = MobileDuoMujocoScene(
    os.environ["MOBILE_DUO_TEST_SCENE_URDF"],
    mesh_root=os.environ["MOBILE_DUO_TEST_MESH_ROOT"],
    enable_vis=False,
)
scene.initialize_simulation()
try:
    for _ in range(5):
        scene._read_and_publish_state()
        scene._apply_control()
        mujoco.mj_step(scene.model, scene.data)
    scene._read_and_publish_state()
finally:
    scene.stop()

assert "genesis" not in sys.modules
assert "taichi" not in sys.modules
print("FULL_BUILD_OK")
"""
)


def _run_blocked(script: str, extra_env=None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_mobile_duo_mujoco_path_imports_without_genesis():
    """The modules --physics mujoco needs must import with genesis blocked."""
    result = _run_blocked(_IMPORT_ONLY_SCRIPT)
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "IMPORT_ONLY_OK" in result.stdout


@pytest.mark.skipif(
    not SCENE_URDF.exists(),
    reason=f"generated scene URDF not present at {SCENE_URDF}",
)
@pytest.mark.skipif(
    not MESH_ROOT.exists(),
    reason=f"franka_description meshes not present at {MESH_ROOT} (set $MOBILE_DUO_MESH_ROOT)",
)
def test_mobile_duo_mujoco_scene_builds_and_steps_without_genesis():
    """The real scene build (URDF resolve, MjSpec compile, a few steps) too."""
    result = _run_blocked(
        _FULL_BUILD_SCRIPT,
        extra_env={
            "MOBILE_DUO_TEST_SCENE_URDF": str(SCENE_URDF),
            "MOBILE_DUO_TEST_MESH_ROOT": str(MESH_ROOT),
        },
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    assert "FULL_BUILD_OK" in result.stdout
