"""Repaint a URDF-imported MuJoCo scene that the COLLADA merge left flat grey.

The combined ``mobile_fr3_duo.urdf`` references the franka_description COLLADA
visuals, and :func:`~franka_sim.urdf_assets.resolve_urdf_meshes` has to merge
each of those into a single ``.obj`` before MuJoCo (or Genesis) will load it.
Merging throws the per-submesh materials away, so every link renders in one flat
default grey: the FR3s read as clay models, the TMR platform as a grey slab and
the lift as a featureless dark box.

Both halves of the fix work on the :class:`mujoco.MjSpec` of an already-imported
URDF, and both are *visual-only*: they only ever delete and add non-colliding,
zero-density geoms, so masses, inertias, collision geoms, joints, DOF properties
and actuators come out bit-identical. Both are optional -- when either source
cannot be read the caller keeps the merged visuals and the scene still runs.

:func:`apply_fr3v2_visuals` handles the two arms. The Menagerie's
``franka_fr3_v2`` model carries the same geometry already split per material by
``obj2mjcf``: one ``.obj`` per (link, material) pair plus the matching
``<material>`` palette. Its ``fr3v2_link0..7`` body frames are the identical FR3
kinematic frames the URDF uses -- verified numerically to 1e-15 against the
compiled scene -- so the visuals can simply be transplanted onto the URDF's
``left_fr3v2_link*`` / ``right_fr3v2_link*`` bodies.

:func:`apply_dae_material_visuals` handles everything else -- the TMR chassis,
the lift column, the arm mount and the head -- which has no Menagerie
counterpart. There the colours are recovered from the source COLLADA itself:
:func:`~franka_sim.urdf_assets.split_dae_by_material` re-splits each ``.dae``
into one ``.obj`` per material, and each of those becomes its own geom wearing
an MJCF material with that material's diffuse rgba.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import mujoco

from franka_sim.urdf_assets import (
    DEFAULT_CACHE_DIR,
    MaterialSubmesh,
    Rgba,
    link_visual_dae_meshes,
    split_dae_by_material,
)

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

#: Namespace for the mesh and material assets derived from the URDF's own
#: COLLADA visuals, so they cannot collide with the names MuJoCo's URDF importer
#: already gave the merged meshes (which stay in the spec, unreferenced).
DAE_ASSET_PREFIX = "dae_"


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
    expected bodies -- :class:`~franka_sim.mobile.duo_mujoco_sim.MobileDuoMujocoScene`
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


class _DaePalette:
    """Adds one ``<mesh>`` per submesh and one ``<material>`` per colour, once each.

    Materials are keyed by colour rather than by (link, material) pair, so the
    white the TMR platform and the lift column share ends up as one MJCF
    material instead of one per link.
    """

    def __init__(self, scene_spec: mujoco.MjSpec):
        self._scene = scene_spec
        self.meshes: List[str] = []
        self.material_names: List[str] = []

    def mesh(self, path: Path) -> str:
        """Namespaced name of a submesh asset, adding it on first reference."""
        name = DAE_ASSET_PREFIX + path.stem
        if name not in self.meshes:
            mesh = self._scene.add_mesh()
            mesh.name = name
            mesh.file = str(path)
            self.meshes.append(name)
        return name

    def material(self, rgba: Rgba) -> str:
        """Namespaced name of a material for one rgba, adding it on first use."""
        name = DAE_ASSET_PREFIX + "".join(f"{int(round(channel * 255)):02x}" for channel in rgba)
        if name not in self.material_names:
            material = self._scene.add_material()
            material.name = name
            material.rgba = rgba
            self.material_names.append(name)
        return name


def _split_link_visual(
    body, geom, submeshes: Sequence[MaterialSubmesh], palette, index: int
) -> int:
    """Replace one merged visual geom with one geom per material. Returns the count.

    The replacements inherit the original geom's ``pos``/``quat``/``group``, so
    a link whose ``<visual>`` carried an ``<origin>`` (the FR3 duo cover sits
    68 mm up its mount) keeps it. ``index`` seeds the generated geom names and
    must be unique within the body: ``mount_link`` splits two COLLADA files.
    """
    pos = tuple(geom.pos)
    quat = tuple(geom.quat)
    group = geom.group
    geom.delete()

    for offset, submesh in enumerate(submeshes):
        replacement = body.add_geom()
        replacement.name = f"{body.name}_visual{index + offset}"
        replacement.type = mujoco.mjtGeom.mjGEOM_MESH
        replacement.meshname = palette.mesh(submesh.path)
        replacement.material = palette.material(submesh.rgba)
        replacement.group = group
        replacement.pos = pos
        replacement.quat = quat
        # Visual-only, and provably so: a zero-density massless geom cannot
        # reach the inertia the compiler gives the body, whatever
        # ``inertiafromgeom`` is set to.
        replacement.contype = 0
        replacement.conaffinity = 0
        replacement.density = 0.0
        replacement.mass = 0.0
    return len(submeshes)


def apply_dae_material_visuals(
    scene_spec: mujoco.MjSpec,
    urdf_path,
    mesh_root=None,
    skip_body_prefixes: Sequence[str] = ARM_BODY_PREFIXES,
    cache_dir=DEFAULT_CACHE_DIR,
) -> int:
    """Repaint the non-arm links from the colours in their source COLLADA files.

    Mutates ``scene_spec`` in place and returns the number of visual geoms
    added. ``urdf_path`` is the *original* URDF (the one whose ``<visual>``
    elements still name ``.dae`` files), and ``mesh_root`` resolves its
    ``package://`` references, exactly as for
    :func:`~franka_sim.urdf_assets.resolve_urdf_meshes`.

    The arm links are skipped by default because :func:`apply_fr3v2_visuals`
    gives them better visuals than their COLLADA carries -- and because it would
    then be replacing geoms this function had already multiplied, which the
    positional link-to-geom mapping below cannot express.

    Every failure mode is per-``<visual>`` and non-fatal: a link whose geoms do
    not line up with its ``<visual>`` elements, or a ``.dae`` ``trimesh`` cannot
    split, simply keeps the merged grey visual it already had.
    """
    palette = _DaePalette(scene_spec)
    replaced = 0
    added = 0

    for link, dae_files in link_visual_dae_meshes(urdf_path, mesh_root).items():
        if any(link.startswith(prefix) for prefix in skip_body_prefixes):
            continue
        if not any(dae_files):
            continue
        body = scene_spec.find_body(link)
        if body is None:
            logger.debug("Link %r has COLLADA visuals but no body in the scene", link)
            continue

        visuals = [geom for geom in body.geoms if _is_visual(geom)]
        if len(visuals) != len(dae_files):
            logger.warning(
                "Keeping %r's merged visuals: it has %d visual geoms but %d <visual> elements",
                link,
                len(visuals),
                len(dae_files),
            )
            continue

        on_body = 0
        for geom, dae_path in zip(visuals, dae_files):
            if dae_path is None:
                continue
            submeshes = _split_or_none(dae_path, cache_dir)
            if not submeshes:
                continue
            replaced += 1
            on_body += _split_link_visual(body, geom, submeshes, palette, on_body)
        added += on_body

    logger.info(
        "Repainted %d merged COLLADA visuals as %d per-material geoms (%d meshes, %d materials)",
        replaced,
        added,
        len(palette.meshes),
        len(palette.material_names),
    )
    return added


#: The lift's grey (and shading-darkened white) materials, keyed by the link
#: that draws them, mapped to a flat neutral Franka white. Picked by rendering
#: the scene and identifying, geom by geom, which material actually forms the
#: visible column -- see the module docstring addendum below for the evidence.
#:
#: ``franka_spine`` (the fixed vertical column) draws 692 of its 2651
#: triangles in a genuine mid-grey (0.412) trim panel; the rest is already a
#: 0.98 near-white. ``mount_link`` (the arm-mount plate the column's prismatic
#: joint carries) is the surprise: its *dominant* submesh by far -- 222,326 of
#: its 254,792 triangles, the whole ``fr3_duo_mount.dae`` body -- is a genuine
#: mid-grey (0.439), with only the smaller ``fr3_duo_cover.dae`` (32,466
#: triangles) already white. Both links' near-white submeshes are included too
#: (colour unchanged) purely so :func:`apply_lift_color_overrides` can give them
#: a private material to brighten without touching the shared palette entry
#: ``base_link``'s correct rim white reuses.
LINK_COLOR_OVERRIDES: Dict[str, Dict[Rgba, Rgba]] = {
    "franka_spine": {
        (0.412, 0.412, 0.412, 1.0): (0.95, 0.95, 0.95, 1.0),
        (0.980, 0.980, 0.980, 1.0): (0.980, 0.980, 0.980, 1.0),
    },
    "mount_link": {
        (0.439, 0.439, 0.439, 1.0): (0.95, 0.95, 0.95, 1.0),
        (1.000, 1.000, 1.000, 1.0): (1.000, 1.000, 1.000, 1.0),
    },
}

#: Emissive boost given to every material :func:`apply_lift_color_overrides`
#: touches. The scene lights the lift with a single top-down directional light
#: plus MuJoCo's default headlight (ambient 0.1) -- there is no fill light --
#: so a correctly-white 0.98 diffuse column face that does not point at either
#: light shades down to as little as 0.42 once rendered (measured on
#: ``franka_spine``'s dominant submesh, the actual visible pillar). A flat rgba
#: swap cannot fix that: the material was never mis-coloured, the shading was
#: doing the darkening. Emission adds a light-independent floor to the
#: material's own colour, which does: 0.45 brings that same worst-case face
#: back up to ~0.83 in the render while leaving the better-lit faces close to
#: fully white, instead of flattening every face to identical white.
LIFT_EMISSION = 0.45


def apply_lift_color_overrides(
    scene_spec: mujoco.MjSpec,
    overrides: Dict[str, Dict[Rgba, Rgba]] = LINK_COLOR_OVERRIDES,
    emission: float = LIFT_EMISSION,
) -> int:
    """Brighten the lift assembly's grey materials to a neutral Franka white.

    Mutates ``scene_spec`` in place and returns the number of geoms
    repainted. Must run after :func:`apply_dae_material_visuals`, once each
    matching geom already wears one of that function's shared-by-colour
    materials -- this only ever swaps a geom's ``material`` reference, it never
    edits mesh geometry, mass, or any other physics property.

    A shared material cannot be brightened in place: ``franka_spine``'s 0.98
    near-white is the exact same MJCF material ``base_link``'s (correct,
    untouched) rim reuses, since :class:`_DaePalette` keys materials by colour
    across the whole scene. So every matching geom instead gets its own
    private, link-scoped copy of the colour -- one add_material() per distinct
    (link, new colour) pair, reused across that link's own geoms -- which is
    also what lets this repaint the lift without ever touching ``base_link``
    or the arms, whose greys/whites are correct as authored.

    ``overrides`` and ``emission`` default to the module constants but take an
    explicit link/colour, so a caller (or a test) can restrict or replace the
    picked colours without editing this function.
    """
    materials = {material.name: material for material in scene_spec.materials}
    private_material_names: Dict[Tuple[str, Rgba], str] = {}
    repainted = 0

    for link, colour_map in overrides.items():
        body = scene_spec.find_body(link)
        if body is None:
            logger.debug("Lift colour override skipped: no body named %r", link)
            continue
        for geom in body.geoms:
            if not _is_visual(geom) or not geom.material:
                continue
            source = materials.get(geom.material)
            if source is None:
                continue
            # ``source.rgba`` is a float32 numpy array: round() on a float32
            # scalar rounds within float32 precision (e.g. 0.9800000190734863
            # for the DAE's 0.98), which compares unequal to the plain Python
            # float64 literals in ``overrides``. Casting to float first rounds
            # in float64, landing on the same value the literals parse to.
            old_rgba = tuple(round(float(channel), 3) for channel in source.rgba)
            new_rgba = colour_map.get(old_rgba)
            if new_rgba is None:
                continue

            key = (link, new_rgba)
            private_name = private_material_names.get(key)
            if private_name is None:
                private_name = f"{DAE_ASSET_PREFIX}lift_{link}_{len(private_material_names)}"
                private = scene_spec.add_material()
                private.name = private_name
                private.rgba = new_rgba
                private.emission = emission
                private_material_names[key] = private_name
                materials[private_name] = private
            geom.material = private_name
            repainted += 1

    logger.info(
        "Brightened %d lift-assembly geoms to Franka white (emission=%.2f) across %d links",
        repainted,
        emission,
        len({link for link, _ in private_material_names}),
    )
    return repainted


def _split_or_none(dae_path: Path, cache_dir) -> Optional[List[MaterialSubmesh]]:
    """Split one ``.dae`` by material, or None when it cannot be split.

    ``trimesh`` is an optional dependency and COLLADA is a large format, so this
    can fail for anything from a missing import to an unsupported ``<effect>``;
    none of that is worth failing a scene build over.
    """
    try:
        return split_dae_by_material(dae_path, cache_dir)
    except Exception as exc:
        logger.info(
            "Keeping the merged visual for %s; it could not be split by material (%s: %s)",
            dae_path,
            type(exc).__name__,
            exc,
        )
        return None
