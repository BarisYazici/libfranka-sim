"""Optional integration test for the Genesis GPU backend ($FRANKA_SIM_BACKEND=gpu).

Everything else in this repo's test suite runs Genesis on CPU (or not at
all -- conftest.py stubs the ``genesis`` module with a MagicMock so most
tests can import without the native dependency). This module is the one
place that actually asks Genesis to build and step a scene on the GPU, so
it is heavily gated:

- skipped unless the caller explicitly opted in with $FRANKA_SIM_BACKEND=gpu
  (matching resolve_gs_backend's contract -- this is not exercised by
  default runs of the suite);
- skipped unless a CUDA device is actually available (real Genesis +
  torch.cuda, not just an installed NVIDIA driver);
- skipped if `nvidia-smi` reports an active compute process, so this never
  contends for the GPU with another simulation already running on it.
"""

import importlib
import importlib.util
import shutil
import subprocess
import sys

import pytest

# ---------------------------------------------------------------------------
# Same swap-the-conftest-stub idiom as test_fr3_hand_model.py /
# test_mobile_duo_physics.py: conftest.py stubs `genesis` with a MagicMock so
# other tests can import franka_sim without the native dependency, so the
# cached sys.modules entry cannot be trusted to answer "is genesis really
# installed" -- probe the file system with the stub popped instead.
# ---------------------------------------------------------------------------


def _pop_genesis_modules():
    saved = {}
    for name in list(sys.modules):
        if name == "genesis" or name.startswith("genesis."):
            saved[name] = sys.modules.pop(name)
    return saved


def _restore_genesis_modules(saved):
    for name in list(sys.modules):
        if name == "genesis" or name.startswith("genesis."):
            sys.modules.pop(name)
    sys.modules.update(saved)


def _genesis_available():
    saved = _pop_genesis_modules()
    try:
        spec = importlib.util.find_spec("genesis")
        return spec is not None and isinstance(getattr(spec, "origin", None), str)
    except Exception:
        return False
    finally:
        _restore_genesis_modules(saved)


def _cuda_available():
    """True only when torch reports a real CUDA device (not just a driver)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _gpu_compute_process_active():
    """True if nvidia-smi reports an active compute process on the GPU.

    Conservative on failure (missing binary, timeout, parse error): treated
    as "can't verify it's free" so the test skips rather than risks
    contending with the live sim for the GPU.
    """
    if shutil.which("nvidia-smi") is None:
        return True
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:
        return True
    return bool(result.stdout.strip())


def _requested_gpu_backend():
    import os

    return os.environ.get("FRANKA_SIM_BACKEND", "").strip().lower() == "gpu"


pytestmark = [
    pytest.mark.skipif(
        not _requested_gpu_backend(),
        reason="only runs when FRANKA_SIM_BACKEND=gpu is explicitly requested",
    ),
    pytest.mark.skipif(not _genesis_available(), reason="Genesis is not installed"),
    pytest.mark.skipif(
        not _cuda_available(),
        reason="no CUDA device available (torch.cuda.is_available() is False)",
    ),
    pytest.mark.skipif(
        _gpu_compute_process_active(),
        reason="a GPU compute process is already active; not contending for the GPU",
    ),
]


def test_gpu_backend_builds_and_steps_a_tiny_scene():
    """Sanity check: gs.gpu (via resolve_gs_backend) actually runs physics.

    Deliberately minimal -- a plane and a falling box, five steps -- so the
    one-time CUDA/Taichi JIT compile this incurs on the *first* gs.init(gpu)
    call in a process stays small; it is not a perf benchmark.
    """
    saved = _pop_genesis_modules()
    try:
        try:
            gs = importlib.import_module("genesis")
        except Exception:
            pytest.skip("real genesis could not be loaded")
            return

        import numpy as np

        from franka_sim.franka_genesis_sim import resolve_gs_backend

        if not getattr(gs, "_initialized", False):
            gs.init(backend=resolve_gs_backend(gs), logging_level=None)

        scene = gs.Scene(show_viewer=False, show_FPS=False)
        scene.add_entity(gs.morphs.Plane())
        box = scene.add_entity(
            gs.morphs.Box(size=(0.1, 0.1, 0.1), pos=(0.0, 0.0, 1.0)),
        )
        scene.build()

        for _ in range(5):
            scene.step()

        pos = box.get_pos().cpu().numpy()
        assert np.isfinite(pos).all(), f"GPU-backend step produced a non-finite pose: {pos}"
        # Gravity should have pulled the box down from its z=1.0 start.
        assert pos[2] < 1.0
    finally:
        _restore_genesis_modules(saved)
