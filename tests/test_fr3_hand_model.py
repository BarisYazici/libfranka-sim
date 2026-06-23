"""Tests for the combined FR3 + Franka Hand MJCF generator.

Fast XML-structure tests (no Genesis) verify joint names, slide ranges, mesh paths,
and the hand-body graft.  The gated Genesis test loads the full 9-DOF model and
asserts that direct DOF position control tracks open / close / mid widths to 2 mm.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from franka_sim.assets import build_fr3_with_hand_mjcf


@pytest.fixture(scope="module")
def combined_xml(tmp_path_factory):
    out = tmp_path_factory.mktemp("model") / "fr3_with_hand.xml"
    return build_fr3_with_hand_mjcf(out)


def test_generated_model_has_arm_and_finger_joints(combined_xml):
    root = ET.parse(combined_xml).getroot()
    joints = {j.get("name") for j in root.iter("joint")}
    for i in range(1, 8):
        assert f"fr3_joint{i}" in joints  # arm preserved
    assert "fr3_finger_joint1" in joints
    assert "fr3_finger_joint2" in joints


def test_finger_joints_are_slide_0_to_0p04(combined_xml):
    root = ET.parse(combined_xml).getroot()
    fingers = [j for j in root.iter("joint") if j.get("name", "").startswith("fr3_finger_joint")]
    assert len(fingers) == 2
    for j in fingers:
        assert j.get("type") == "slide"
        assert j.get("axis") == "0 1 0"
        assert j.get("range") == "0 0.04"


def test_all_mesh_paths_are_absolute_and_exist(combined_xml):
    root = ET.parse(combined_xml).getroot()
    files = [m.get("file") for m in root.iter("mesh")]
    assert files, "no meshes in generated model"
    for f in files:
        p = Path(f)
        assert p.is_absolute(), f"mesh path not absolute: {f}"
        assert p.exists(), f"mesh file missing: {f}"


def test_hand_body_grafted_under_link7(combined_xml):
    root = ET.parse(combined_xml).getroot()
    link7 = next(b for b in root.iter("body") if b.get("name") == "fr3_link7")
    child_names = {b.get("name") for b in link7.iter("body")}
    assert "fr3_hand" in child_names


# ---------------------------------------------------------------------------
# Gated Genesis load + tracking test
# ---------------------------------------------------------------------------


def _pop_genesis_modules():
    import sys

    saved = {}
    for name in list(sys.modules):
        if name == "genesis" or name.startswith("genesis."):
            saved[name] = sys.modules.pop(name)
    return saved


def _restore_genesis_modules(saved):
    import sys

    for name in list(sys.modules):
        if name == "genesis" or name.startswith("genesis."):
            sys.modules.pop(name)
    sys.modules.update(saved)


def _genesis_available():
    """Return True only when the *real* Genesis package (not a MagicMock) is installed.

    conftest.py stubs ``genesis`` with ``sys.modules.setdefault("genesis", MagicMock())``
    so that server tests can import without the heavy native dep.  That mock is already in
    ``sys.modules`` by the time this function is evaluated at decoration time.  We detect
    its presence by temporarily removing the cached module and using
    ``importlib.util.find_spec`` to check the file system, then restoring the original
    value so other tests are unaffected.
    """
    import importlib.util

    saved = _pop_genesis_modules()
    try:
        spec = importlib.util.find_spec("genesis")
        genesis_on_fs = spec is not None and isinstance(getattr(spec, "origin", None), str)
    except Exception:
        genesis_on_fs = False
    finally:
        _restore_genesis_modules(saved)

    if not genesis_on_fs:
        return False

    try:
        import robot_descriptions.fr3_mj_description  # noqa: F401

        return True
    except Exception:
        return False


genesis_required = pytest.mark.skipif(
    not _genesis_available(), reason="Genesis + robot_descriptions required for the model load test"
)


@genesis_required
def test_genesis_loads_9dof_and_fingers_track_commanded_width(tmp_path):
    import importlib
    import numpy as np

    # conftest.py stubs ``genesis`` with a MagicMock so server tests don't need the
    # native dep.  Here we need the *real* Genesis; temporarily swap it back.
    saved = _pop_genesis_modules()
    try:
        try:
            gs = importlib.import_module("genesis")
        except Exception:
            pytest.skip("real genesis could not be loaded")
            return

        model = build_fr3_with_hand_mjcf(tmp_path / "m.xml")
        gs.init(backend=gs.cpu, logging_level=None)
        scene = gs.Scene(show_viewer=False)
        robot = scene.add_entity(gs.morphs.MJCF(file=str(model)))
        scene.build()

        assert robot.n_dofs == 9
        finger_idx = [
            robot.get_joint(n).dof_idx_local for n in ("fr3_finger_joint1", "fr3_finger_joint2")
        ]
        assert robot.get_link("fr3_link7") is not None

        # Direct DOF position control must track open / close / mid with no actuator fight.
        for target in (0.04, 0.0, 0.02):
            robot.control_dofs_position(np.array([target, target]), finger_idx)
            for _ in range(400):
                scene.step()
            pos = robot.get_dofs_position(finger_idx).cpu().numpy()
            assert np.allclose(pos, target, atol=2e-3), f"fingers did not track {target}: {pos}"
    finally:
        # Restore conftest's stub so downstream tests are not affected.
        _restore_genesis_modules(saved)
