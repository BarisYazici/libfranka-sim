"""Give the mobile-duo scene's two arms the MuJoCo Menagerie FR3 v2 visuals.

The combined ``mobile_fr3_duo.urdf`` references the franka_description COLLADA
visuals, and :func:`~franka_sim.urdf_assets.resolve_urdf_meshes` has to merge
each of those into a single ``.obj`` before MuJoCo (or Genesis) will load it.
Merging throws the per-submesh materials away, so every arm link renders in one
flat default grey and the two FR3s read as clay models.

The Menagerie's ``franka_fr3_v2`` model carries the same geometry already split
per material by ``obj2mjcf``: one ``.obj`` per (link, material) pair plus the
matching ``<material>`` palette. Its ``fr3v2_link0..7`` body frames are the
identical FR3 kinematic frames the URDF uses -- verified numerically to 1e-15
against the compiled scene -- so the visuals can simply be transplanted onto the
URDF's ``left_fr3v2_link*`` / ``right_fr3v2_link*`` bodies.

This is a *visual-only* transplant: it deletes and adds non-colliding geoms with
zero density, so masses, inertias, collision geoms, joints and actuators are
untouched. When the Menagerie model cannot be resolved the caller keeps the
converted-URDF visuals; nothing here is required for the scene to run.
"""

import logging
from pathlib import Path
from typing import Dict, List, Sequence

import mujoco

logger = logging.getLogger(__name__)

#: Namespace given to every asset copied out of the Menagerie model, so the
#: copies can never collide with the URDF's own mesh and material names (both
#: arms share one set of copies, hence a prefix rather than a per-arm one).
ASSET_PREFIX = "fr3v2_"

#: Arm links whose visuals are swapped. ``link8`` is the flange frame and
#: carries no geometry in either model.
FR3V2_VISUAL_LINKS = tuple(range(8))

#: Body-name prefixes of the two arms inside the combined URDF.
ARM_BODY_PREFIXES = ("left_", "right_")


def resolve_fr3v2_mjcf() -> Path:
    """Return the path to the Menagerie ``fr3v2.xml``.

    ``robot_descriptions`` downloads and caches the Menagerie on first use, so
    this can raise anything from ``ImportError`` to a network error; callers are
    expected to treat a failure as "keep the URDF visuals".
    """
    from robot_descriptions import fr3_v2_mj_description

    return Path(fr3_v2_mj_description.MJCF_PATH)


def _is_visual(geom) -> bool:
    """True for a geom the physics never sees.

    MuJoCo's URDF importer gives ``<visual>`` geoms ``contype``/``conaffinity``
    0 and ``<collision>`` geoms 1, and the Menagerie's ``visual`` class does the
    same, so one predicate discriminates in both models.
    """
    return geom.contype == 0 and geom.conaffinity == 0


def _mesh_files(spec: mujoco.MjSpec, mjcf_path: Path) -> Dict[str, Path]:
    """Map each Menagerie mesh's referenced name to an absolute file path.

    The Menagerie declares its meshes as bare ``<mesh file="link0_0.obj"/>``
    under ``<compiler meshdir="assets">``: the name geoms reference is the file
    stem, and the path is relative to that ``meshdir``. Absolute paths are what
    gets copied into the scene spec, whose own ``meshdir`` is empty.
    """
    mesh_dir = Path(spec.modelfiledir or mjcf_path.parent) / spec.meshdir
    return {
        (mesh.name or Path(mesh.file).stem): (mesh_dir / mesh.file).resolve()
        for mesh in spec.meshes
    }


class _AssetCopier:
    """Copies Menagerie meshes and materials into the scene spec, once each."""

    def __init__(self, scene_spec: mujoco.MjSpec, source_spec: mujoco.MjSpec, mjcf_path: Path):
        self._scene = scene_spec
        self._mesh_files = _mesh_files(source_spec, mjcf_path)
        self._materials = {material.name: material for material in source_spec.materials}
        self.meshes: List[str] = []
        self.material_names: List[str] = []

    def mesh(self, name: str) -> str:
        """Namespaced name of a copied mesh, adding it on first reference."""
        copied = ASSET_PREFIX + name
        if copied not in self.meshes:
            mesh = self._scene.add_mesh()
            mesh.name = copied
            mesh.file = str(self._mesh_files[name])
            self.meshes.append(copied)
        return copied

    def material(self, name: str) -> str:
        """Namespaced name of a copied material, adding it on first reference."""
        copied = ASSET_PREFIX + name
        if copied not in self.material_names:
            source = self._materials[name]
            material = self._scene.add_material()
            material.name = copied
            material.rgba = source.rgba
            material.specular = source.specular
            material.shininess = source.shininess
            material.reflectance = source.reflectance
            material.metallic = source.metallic
            material.roughness = source.roughness
            material.emission = source.emission
            self.material_names.append(copied)
        return copied


def _replace_link_visuals(body, source_body, copier: _AssetCopier) -> int:
    """Swap one link's visual geoms for the source link's. Returns the new count."""
    for geom in [geom for geom in body.geoms if _is_visual(geom)]:
        geom.delete()

    added = 0
    for source in source_body.geoms:
        if not _is_visual(source):
            continue
        geom = body.add_geom()
        geom.name = f"{body.name}_visual{added}"
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = copier.mesh(source.meshname)
        if source.material:
            geom.material = copier.material(source.material)
        else:
            geom.rgba = source.rgba
        geom.group = source.group
        geom.pos = source.pos
        geom.quat = source.quat
        # Visual-only, and provably so: a zero-density massless geom cannot
        # reach the inertia the compiler gives the body, whatever
        # ``inertiafromgeom`` is set to.
        geom.contype = 0
        geom.conaffinity = 0
        geom.density = 0.0
        geom.mass = 0.0
        added += 1
    return added


def apply_fr3v2_visuals(
    scene_spec: mujoco.MjSpec,
    arm_prefixes: Sequence[str] = ARM_BODY_PREFIXES,
    mjcf_path=None,
) -> int:
    """Replace both arms' visual geoms in ``scene_spec`` with Menagerie ones.

    Mutates ``scene_spec`` in place and returns the number of visual geoms
    added. Raises if the Menagerie model cannot be read or does not contain the
    expected bodies -- :class:`~franka_sim.mobile_duo_mujoco_sim.MobileDuoMujocoScene`
    catches that and keeps the converted-URDF visuals.
    """
    mjcf_path = Path(mjcf_path) if mjcf_path is not None else resolve_fr3v2_mjcf()
    source_spec = mujoco.MjSpec.from_file(str(mjcf_path))
    copier = _AssetCopier(scene_spec, source_spec, mjcf_path)

    added = 0
    removed = 0
    for prefix in arm_prefixes:
        for index in FR3V2_VISUAL_LINKS:
            link = f"fr3v2_link{index}"
            body = scene_spec.find_body(prefix + link)
            if body is None:
                raise KeyError(f"Arm link {prefix + link!r} not found in the scene")
            source_body = source_spec.find_body(link)
            if source_body is None:
                raise KeyError(f"Link {link!r} not found in {mjcf_path}")
            removed += sum(1 for geom in body.geoms if _is_visual(geom))
            added += _replace_link_visuals(body, source_body, copier)

    logger.info(
        "Replaced %d converted-URDF arm visuals with %d Menagerie FR3 v2 geoms "
        "(%d meshes, %d materials, shared by %d arms) from %s",
        removed,
        added,
        len(copier.meshes),
        len(copier.material_names),
        len(arm_prefixes),
        mjcf_path,
    )
    return added
