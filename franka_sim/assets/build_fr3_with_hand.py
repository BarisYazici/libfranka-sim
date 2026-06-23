"""Generate a combined FR3-arm + Franka-Hand MJCF for Genesis (9 DOF).

Production model generator. Produces an MJCF that is the *unmodified* MuJoCo
Menagerie ``franka_fr3/fr3.xml`` arm with the Franka Hand grafted onto the arm
flange (the ``attachment_site`` at ``pos="0 0 0.107"`` under ``fr3_link7``),
mirroring the hand-attachment pattern from ``franka_emika_panda/panda.xml``.

Design goals / invariants
-------------------------
* The arm is preserved bit-for-bit: same 7 joint names ``fr3_joint1..7``, same
  link names (``fr3_link7`` etc.), same inertials / armature / damping / the
  ``fr3`` default class. We only *append* hand assets + a hand subtree + the
  finger tendon/equality/actuator. The arm calibration is therefore untouched.
* Mesh-path strategy: a single MJCF has only one ``meshdir``. The big FR3 ARM
  meshes must keep resolving from the ``robot_descriptions`` cache (never
  vendored). The small HAND meshes are vendored into this repo
  (``franka_sim/assets/franka_hand/``). To satisfy both from one file we drop
  ``meshdir`` and rewrite every ``<mesh file=...>`` to an ABSOLUTE path:
  arm meshes -> cache ``franka_fr3/assets``; hand meshes -> the vendored dir.
* Two finger joints ``fr3_finger_joint1`` / ``fr3_finger_joint2`` (slide, axis
  ``0 1 0``, range ``0 0.04``) coupled by a ``split`` tendon + a joint equality,
  driven by one position-style actuator -- exactly the panda.xml pattern, just
  fr3-prefixed so names never collide with the arm.

Actuator-vs-direct-control outcome (empirically resolved)
----------------------------------------------------------
The MJCF ``<position name="fr3_finger" tendon="split" …>`` actuator is retained
in the generated file.  Empirical testing showed that when Genesis drives the
finger DOFs via ``robot.control_dofs_position()``, the tendon actuator does NOT
interfere: Genesis does not activate MJCF actuators unless the caller explicitly
calls ``robot.control_actuators_position()`` (or equivalent).  The actuator's
default control input sits at 0, but ``control_dofs_position`` overrides the DOF
targets directly and the fingers tracked open (0.04), close (0.0), and mid (0.02)
within 2 mm without any actuator conflict.  The actuator is therefore kept inert
-- it provides a hook for MuJoCo-native use outside Genesis -- and the
``<tendon><fixed name="split">`` remains to couple it.

The generator writes to a temp file by default (so it can run at sim init and
clean up after itself, like the TMR runtime-assembly precedent). Pass an
explicit ``out_path`` to emit a static file instead.

Mesh swap-in note
-----------------
The vendored hand meshes (``franka_sim/assets/franka_hand/``) come from the
MuJoCo Menagerie Franka hand model. For pixel-exact FR3-hand visuals, they can
be replaced by the ``franka_description`` package meshes (``finger.dae`` /
``hand.dae``): convert each DAE to OBJ via ``trimesh`` (they are already in
metres), drop the OBJ files into ``franka_sim/assets/franka_hand/``, and
update the ``HAND_VISUAL_MESHES`` / ``HAND_COLLISION_MESH`` lists above. This
is a cosmetic change only -- the finger kinematics (slide joints, range 0..0.04,
coupled tendon) are identical regardless of which mesh set is used.
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

# Vendored hand meshes live next to this script.
HAND_ASSET_DIR = Path(__file__).resolve().parent / "franka_hand"
HAND_VISUAL_MESHES = [
    "hand_0.obj",
    "hand_1.obj",
    "hand_2.obj",
    "hand_3.obj",
    "hand_4.obj",
    "finger_0.obj",
    "finger_1.obj",
]
HAND_COLLISION_MESH = "hand.stl"


def _resolve_fr3_mjcf() -> Path:
    """Same resolution the production sim uses: $FR3_MJCF or robot_descriptions."""
    override = os.environ.get("FR3_MJCF")
    if override:
        return Path(override)
    from robot_descriptions import fr3_mj_description

    return Path(fr3_mj_description.MJCF_PATH)


def build_fr3_with_hand_mjcf(out_path: str | os.PathLike | None = None) -> Path:
    """Build the combined FR3 + hand MJCF and return its path.

    If ``out_path`` is None, a temp file is created (caller is responsible for
    deleting it; see ``__main__`` for the cleanup-after-itself pattern).
    """
    fr3_path = _resolve_fr3_mjcf()
    fr3_assets = (fr3_path.parent / "assets").resolve()

    tree = ET.parse(fr3_path)
    root = tree.getroot()

    # --- 1. Make all mesh paths absolute, then drop meshdir -------------------
    # Arm meshes -> cache assets dir; we add hand meshes below as absolute too.
    compiler = root.find("compiler")
    if compiler is not None and "meshdir" in compiler.attrib:
        del compiler.attrib["meshdir"]

    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("FR3 MJCF has no <asset> element; cannot graft the hand")
    for mesh in asset.findall("mesh"):
        fname = mesh.get("file")
        mesh.set("file", str((fr3_assets / fname).resolve()))

    # --- 2. Append hand mesh assets (absolute paths to the vendored dir) ------
    for fname in HAND_VISUAL_MESHES:
        ET.SubElement(asset, "mesh", {"file": str((HAND_ASSET_DIR / fname).resolve())})
    ET.SubElement(
        asset,
        "mesh",
        {"name": "hand_c", "file": str((HAND_ASSET_DIR / HAND_COLLISION_MESH).resolve())},
    )

    # --- 3. Hand-specific materials (panda.xml uses off_white; fr3 lacks it) --
    ET.SubElement(asset, "material", {"name": "off_white", "rgba": "0.901961 0.921569 0.929412 1"})

    # --- 4. Locate fr3_link7 and graft the hand subtree -----------------------
    link7 = None
    for body in root.iter("body"):
        if body.get("name") == "fr3_link7":
            link7 = body
            break
    if link7 is None:
        raise RuntimeError("Could not find body 'fr3_link7' in the FR3 MJCF")

    # Hand body at the flange mount: pos 0 0 0.107 (== attachment_site), quat
    # -45 deg about z, mirroring panda.xml. childclass='fr3' so the visual geoms
    # inherit type=mesh/group=2/contype0 from the fr3 'visual' default class.
    hand = ET.SubElement(
        link7,
        "body",
        {
            "name": "fr3_hand",
            "pos": "0 0 0.107",
            "quat": "0.9238795 0 0 -0.3826834",
            "childclass": "fr3",
        },
    )
    ET.SubElement(
        hand,
        "inertial",
        {"mass": "0.73", "pos": "-0.01 0 0.03", "diaginertia": "0.001 0.0025 0.0017"},
    )
    for mesh, mat in [
        ("hand_0", "off_white"),
        ("hand_1", "black"),
        ("hand_2", "black"),
        ("hand_3", "white"),
        ("hand_4", "off_white"),
    ]:
        ET.SubElement(hand, "geom", {"mesh": mesh, "material": mat, "class": "visual"})
    ET.SubElement(hand, "geom", {"mesh": "hand_c", "class": "collision"})

    # Finger bodies. Joints are slide along y, range 0..0.04, fr3-prefixed.
    for body_name, joint_name, quat in [
        ("fr3_left_finger", "fr3_finger_joint1", None),
        ("fr3_right_finger", "fr3_finger_joint2", "0 0 0 1"),
    ]:
        attrs = {"name": body_name, "pos": "0 0 0.0584"}
        if quat is not None:
            attrs["quat"] = quat
        finger = ET.SubElement(hand, "body", attrs)
        ET.SubElement(
            finger,
            "inertial",
            {"mass": "0.015", "pos": "0 0 0", "diaginertia": "2.375e-6 2.375e-6 7.5e-7"},
        )
        ET.SubElement(
            finger,
            "joint",
            {"name": joint_name, "type": "slide", "axis": "0 1 0", "range": "0 0.04"},
        )
        ET.SubElement(
            finger, "geom", {"mesh": "finger_0", "material": "off_white", "class": "visual"}
        )
        ET.SubElement(finger, "geom", {"mesh": "finger_1", "material": "black", "class": "visual"})
        # One collision geom for the finger body (skip the box fingertip pads --
        # not needed for the load/control proof and keeps geom count low).
        ET.SubElement(finger, "geom", {"mesh": "finger_0", "class": "collision"})

    # --- 5. Tendon (couples the two fingers 50/50) ----------------------------
    tendon = ET.SubElement(root, "tendon")
    fixed = ET.SubElement(tendon, "fixed", {"name": "split"})
    ET.SubElement(fixed, "joint", {"joint": "fr3_finger_joint1", "coef": "0.5"})
    ET.SubElement(fixed, "joint", {"joint": "fr3_finger_joint2", "coef": "0.5"})

    # --- 6. Equality (keeps the two fingers in lockstep) ----------------------
    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "joint",
        {
            "joint1": "fr3_finger_joint1",
            "joint2": "fr3_finger_joint2",
            "solimp": "0.95 0.99 0.001",
            "solref": "0.005 1",
        },
    )

    # --- 7. Finger actuator (drives the 'split' tendon) -----------------------
    # Direct-meter actuator on the tendon: ctrlrange 0..0.04 in meters of finger
    # travel (simpler than the panda 0..255 remap; the proof commands meters).
    actuator = root.find("actuator")
    if actuator is None:
        raise RuntimeError("FR3 MJCF has no <actuator> element; cannot add the finger actuator")
    ET.SubElement(
        actuator,
        "position",
        {
            "name": "fr3_finger",
            "tendon": "split",
            "ctrlrange": "0 0.04",
            "kp": "100",
            "forcerange": "-100 100",
        },
    )

    # --- 8. Extend the home keyframe to 9 qpos / matching ctrl ----------------
    # The FR3 keyframe has 7 qpos + 7 ctrl. With 2 finger DOFs and an extra
    # actuator, MuJoCo requires nq=9 and nu=8. Append open-finger values.
    keyframe = root.find("keyframe")
    if keyframe is not None:
        key = keyframe.find("key")
        if key is not None:
            if key.get("qpos"):
                key.set("qpos", key.get("qpos") + " 0.04 0.04")
            if key.get("ctrl"):
                key.set("ctrl", key.get("ctrl") + " 0.04")

    # --- 9. Write out ---------------------------------------------------------
    if out_path is None:
        fd, tmp = tempfile.mkstemp(prefix="fr3_with_hand_", suffix=".xml")
        os.close(fd)
        out_path = Path(tmp)
    else:
        out_path = Path(out_path)

    ET.indent(tree, space="  ")
    tree.write(out_path, encoding="unicode", xml_declaration=False)
    return out_path


if __name__ == "__main__":
    import sys

    # If an explicit path is given, write a static file; else temp + report it.
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    path = build_fr3_with_hand_mjcf(target)
    print(path)
