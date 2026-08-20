"""Prepare a ROS URDF and its meshes for loading into Genesis.

Two problems have to be solved before ``gs.morphs.URDF`` can load a
franka_description robot:

* mesh references use the ROS ``package://`` scheme, which Genesis does not
  resolve;
* the COLLADA visuals contain hundreds of sub-meshes each and Genesis creates a
  ``RigidGeom`` per sub-mesh, which stalls ``scene.build()``.

``resolve_urdf_meshes`` writes a *copy* of the URDF in which every mesh
reference is an absolute path and every ``.dae`` has been merged into a single
``.obj``. Conversions are cached on disk, keyed by the source path, mtime and
size, so repeated server starts pay the conversion cost only once.

Merging is lossy: the COLLADA per-material split is what carries a link's
colours, and a merged ``.obj`` has one (default) material for the whole link.
``split_dae_by_material`` is the non-lossy alternative for renderers that can
draw several geoms per link -- it writes one ``.obj`` *per material* plus that
material's diffuse colour. Genesis keeps using the merged path (one
``RigidGeom`` per sub-mesh is what made ``scene.build()`` stall in the first
place); the MuJoCo backend uses the split, see
:mod:`franka_sim.mujoco_visuals`.
"""

import hashlib
import json
import logging
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

Rgba = Tuple[float, float, float, float]

#: Shared on-disk cache for converted meshes.
DEFAULT_CACHE_DIR = Path(tempfile.gettempdir()) / "franka_sim_mesh_cache"

#: COLLADA files whose largest extent exceeds this (metres) are assumed to be
#: authored in millimetres -- true for the franka_description visuals.
MILLIMETRE_EXTENT_THRESHOLD = 10.0

#: Bump this when ``convert_dae_to_obj``'s output format changes (e.g. what
#: gets written to the .obj/.mtl/texture files). It is folded into the cache
#: key below so a code change invalidates stale cache entries automatically,
#: instead of silently reusing .obj files built by the old, buggy exporter --
#: which is exactly what let the mtl-collision bug (every mesh sharing one
#: "material.mtl"/"material_0.png" name, see ``convert_dae_to_obj``) survive
#: a naive fix that only touched the export call and not the cache key.
MESH_CACHE_FORMAT_VERSION = 2

#: Cache-format version of ``split_dae_by_material``, kept separate from
#: MESH_CACHE_FORMAT_VERSION on purpose. The two conversions read the same
#: ``.dae`` files into the same cache directory but produce unrelated outputs,
#: so folding them into one counter would make every change to the split
#: invalidate the merged ``.obj`` files Genesis depends on -- re-merging
#: 200k-triangle COLLADA on the next Genesis start for no reason. Distinct
#: cache keys (see ``_cache_digest``) also mean neither can ever read the
#: other's entries.
SUBMESH_CACHE_FORMAT_VERSION = 1

#: Cache-key discriminator for the per-material split.
SUBMESH_CACHE_KIND = "split"

#: rgba given to a submesh whose material carries no readable colour. A neutral
#: light grey: the point is to be obviously unremarkable, so an unreadable
#: material degrades to what the merged visual looked like rather than to a
#: colour that would be mistaken for the author's.
DEFAULT_SUBMESH_RGBA: Rgba = (0.7, 0.7, 0.7, 1.0)


class MaterialSubmesh(NamedTuple):
    """One material's worth of geometry split out of a COLLADA file."""

    #: Cached ``.obj`` holding every face that uses this material.
    path: Path
    #: The material's diffuse colour, as MuJoCo/URDF-style 0-1 rgba.
    rgba: Rgba
    #: Triangle count, for logging and for the round-trip tests.
    faces: int


def resolve_mesh_path(filename: str, mesh_root: PathLike) -> Path:
    """Map one URDF ``<mesh filename=...>`` value to an absolute path.

    ``package://<pkg>/<relative>`` resolves to ``<mesh_root>/<relative>``, so
    ``mesh_root`` is the package root (e.g. a ``franka_description`` checkout).
    Relative paths are joined to ``mesh_root``; absolute paths pass through.
    """
    if filename.startswith("package://"):
        without_scheme = filename[len("package://") :]
        _, _, relative = without_scheme.partition("/")
        return (Path(mesh_root) / relative).resolve()

    path = Path(filename)
    if path.is_absolute():
        return path
    return (Path(mesh_root) / path).resolve()


def _cache_digest(dae_path: Path, key: str) -> str:
    """Content-ish cache key for one conversion of one source file.

    ``key`` names the conversion (its kind and format version), so two
    conversions of the same ``.dae`` can never collide in the shared cache
    directory or reuse each other's entries.
    """
    stat = dae_path.stat()
    return hashlib.sha1(
        f"{key}:{dae_path}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
    ).hexdigest()[:16]


def convert_dae_to_obj(dae_path: PathLike, cache_dir: PathLike = DEFAULT_CACHE_DIR) -> Path:
    """Merge a COLLADA file into a single ``.obj`` and return the cached path.

    ``trimesh`` is imported lazily so the module stays importable (and the test
    suite stays runnable) without the optional ``mobile`` extra installed.
    """
    import trimesh

    dae_path = Path(dae_path)
    digest = _cache_digest(dae_path, str(MESH_CACHE_FORMAT_VERSION))

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dae_path.stem}-{digest}"
    obj_path = cache_dir / f"{stem}.obj"
    if obj_path.exists():
        logger.debug("Mesh cache hit: %s", obj_path)
        return obj_path

    loaded = trimesh.load(str(dae_path))
    if isinstance(loaded, trimesh.Scene):
        merged = trimesh.util.concatenate(loaded.dump())
    else:
        merged = loaded

    if merged.extents.max() > MILLIMETRE_EXTENT_THRESHOLD:
        merged.apply_scale(0.001)

    # trimesh's OBJ exporter defaults every material/texture sidecar file to
    # the same generic names ("material.mtl", "material_0.png"). All meshes
    # converted by this function share one cache_dir, so without a per-mesh
    # name every conversion clobbers the previous one's sidecar files: the
    # .obj on disk still says ``mtllib material.mtl``, but that file (and the
    # texture atlas it points at) now belongs to whichever .dae was converted
    # *last*. Every mesh but the last one then loads with a mismatched
    # texture -- which is what made most of the mobile-duo platform render
    # near-black. Naming the material after the same digest-qualified stem as
    # the cached .obj makes every export self-contained, so sidecars can
    # never collide.
    visual = getattr(merged, "visual", None)
    material = getattr(visual, "material", None) if visual is not None else None
    if material is not None:
        material.name = stem

    merged.export(str(obj_path), file_type="obj", mtl_name=f"{stem}.mtl")
    logger.info("Converted %s -> %s", dae_path, obj_path)
    return obj_path


def _as_rgba(value) -> Optional[Rgba]:
    """Normalise a trimesh colour to 0-1 rgba, or None if it is not a colour.

    trimesh hands colours back either as ``uint8`` 0-255 (``main_color``,
    ``baseColorFactor``) or as 0-1 floats, depending on the material class and
    on what the COLLADA author wrote, so the dtype -- not the magnitude -- is
    what decides the scale. Magnitude alone would read an 8-bit ``(1, 1, 1)``
    (near black) as float white.
    """
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.ndim != 1 or array.size not in (3, 4):
        return None
    components = array.astype(float)
    if array.dtype.kind in "iu":
        components = components / 255.0
    if components.size == 3:
        components = np.append(components, 1.0)
    return tuple(float(component) for component in np.clip(components, 0.0, 1.0))


def _submesh_rgba(mesh) -> Rgba:
    """Diffuse colour of one trimesh geometry, or a neutral grey.

    A COLLADA ``<effect>`` reaches trimesh as either a ``PBRMaterial``
    (``baseColorFactor``) or a ``SimpleMaterial`` (``diffuse``); ``main_color``
    is the accessor both share, and is tried last because it falls back to a
    default colour of its own rather than reporting "no colour".
    """
    material = getattr(getattr(mesh, "visual", None), "material", None)
    for attribute in ("baseColorFactor", "diffuse", "main_color"):
        rgba = _as_rgba(getattr(material, attribute, None))
        if rgba is not None:
            return rgba
    return DEFAULT_SUBMESH_RGBA


def _placed_geometries(trimesh, dae_path: Path) -> List:
    """Every mesh in a COLLADA file, in the file's own coordinate frame.

    A ``Scene``'s geometries are stored in their *local* frames and positioned
    by the scene graph, and one geometry may be instanced at several nodes.
    Walking ``nodes_geometry`` and applying each node's transform is therefore
    what puts the sub-meshes where the merged mesh (``Scene.dump()``, which does
    the same thing) draws them; skipping it scatters them around the origin.
    """
    loaded = trimesh.load(str(dae_path))
    if not isinstance(loaded, trimesh.Scene):
        return [loaded]

    placed = []
    for node in loaded.graph.nodes_geometry:
        transform, geometry_name = loaded.graph[node]
        mesh = loaded.geometry[geometry_name].copy()
        mesh.apply_transform(transform)
        placed.append(mesh)
    return placed


def _group_by_material(trimesh, dae_path: Path) -> List[Tuple[Rgba, List]]:
    """Group a COLLADA file's meshes by diffuse colour, largest group first.

    Sorting by triangle count keeps the output order stable across runs
    (``nodes_geometry`` order is not contractual) and puts the link's dominant
    colour at index 0.
    """
    groups: Dict[Rgba, List] = {}
    for mesh in _placed_geometries(trimesh, dae_path):
        groups.setdefault(_submesh_rgba(mesh), []).append(mesh)
    return sorted(
        groups.items(),
        key=lambda group: (-sum(len(mesh.faces) for mesh in group[1]), group[0]),
    )


def _millimetre_scale(groups: List[Tuple[Rgba, List]]) -> float:
    """0.001 when the *whole* file is millimetre-sized, else 1.0.

    The decision has to be made over the union of every group's bounds, exactly
    as ``convert_dae_to_obj`` makes it over the merged mesh: judging each group
    on its own would scale a large chassis panel and leave a small bracket at
    1000x, tearing the link apart.
    """
    bounds = np.array(
        [mesh.bounds for _, meshes in groups for mesh in meshes if mesh.bounds is not None]
    )
    if not bounds.size:
        return 1.0
    extents = bounds[:, 1, :].max(axis=0) - bounds[:, 0, :].min(axis=0)
    return 0.001 if extents.max() > MILLIMETRE_EXTENT_THRESHOLD else 1.0


def _read_submesh_manifest(manifest_path: Path) -> Optional[List[MaterialSubmesh]]:
    """Cached split for one ``.dae``, or None when it is absent or incomplete."""
    if not manifest_path.exists():
        return None
    try:
        entries = json.loads(manifest_path.read_text())["submeshes"]
        submeshes = [
            MaterialSubmesh(
                path=manifest_path.parent / entry["file"],
                rgba=tuple(float(component) for component in entry["rgba"]),
                faces=int(entry["faces"]),
            )
            for entry in entries
        ]
    except (KeyError, TypeError, ValueError, OSError):
        logger.warning("Ignoring an unreadable mesh-split manifest: %s", manifest_path)
        return None

    if not submeshes or not all(submesh.path.exists() for submesh in submeshes):
        return None
    logger.debug("Mesh-split cache hit: %s", manifest_path)
    return submeshes


def split_dae_by_material(
    dae_path: PathLike, cache_dir: PathLike = DEFAULT_CACHE_DIR
) -> List[MaterialSubmesh]:
    """Split a COLLADA file into one cached ``.obj`` per material.

    Returns the submeshes largest-first, each with the diffuse rgba of the
    material it was authored with. A file with a single material yields a single
    submesh -- that is the correct answer for it, not a degenerate one.

    The ``.obj`` files are written without materials of their own: the colour
    travels in :class:`MaterialSubmesh` and the caller is expected to express it
    in its own scene format, which sidesteps the shared-sidecar-name problem
    ``convert_dae_to_obj`` has to work around. Raises whatever ``trimesh`` raises
    on a file it cannot parse; callers fall back to the merged visual.
    """
    import trimesh

    dae_path = Path(dae_path)
    digest = _cache_digest(dae_path, f"{SUBMESH_CACHE_KIND}:{SUBMESH_CACHE_FORMAT_VERSION}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{dae_path.stem}-{digest}"
    manifest_path = cache_dir / f"{stem}.json"
    cached = _read_submesh_manifest(manifest_path)
    if cached is not None:
        return cached

    groups = _group_by_material(trimesh, dae_path)
    if not groups:
        raise ValueError(f"{dae_path} contains no geometry to split")
    scale = _millimetre_scale(groups)

    submeshes = []
    for index, (rgba, meshes) in enumerate(groups):
        merged = trimesh.util.concatenate(meshes)
        if scale != 1.0:
            merged.apply_scale(scale)
        # Rebuilt as bare geometry: a Trimesh with no material exports as a
        # self-contained .obj with no "mtllib" line, so the sidecar files that
        # every mesh in this shared cache directory would otherwise fight over
        # are never written at all.
        bare = trimesh.Trimesh(vertices=merged.vertices, faces=merged.faces, process=False)
        obj_path = cache_dir / f"{stem}-m{index}.obj"
        bare.export(str(obj_path), file_type="obj", include_color=False)
        submeshes.append(MaterialSubmesh(path=obj_path, rgba=rgba, faces=len(bare.faces)))

    manifest_path.write_text(
        json.dumps(
            {
                "source": str(dae_path),
                "submeshes": [
                    {"file": submesh.path.name, "rgba": list(submesh.rgba), "faces": submesh.faces}
                    for submesh in submeshes
                ],
            }
        )
    )
    logger.info(
        "Split %s into %d per-material sub-meshes (%d triangles, scale %g)",
        dae_path,
        len(submeshes),
        sum(submesh.faces for submesh in submeshes),
        scale,
    )
    return submeshes


def link_visual_dae_meshes(
    urdf_path: PathLike, mesh_root: Optional[PathLike] = None
) -> Dict[str, List[Optional[Path]]]:
    """Map each link to its ``<visual>`` COLLADA meshes, in document order.

    The list has one entry per ``<visual>`` element of the link, ``None`` where
    that visual is a primitive or a mesh that is neither a ``.dae`` nor present
    on disk. Keeping the ``None`` placeholders is the point: importers emit one
    geom per ``<visual>`` in this same order, so the index is what ties a link's
    geoms back to the COLLADA files they came from.
    """
    urdf_path = Path(urdf_path)
    root_dir = Path(mesh_root) if mesh_root is not None else urdf_path.resolve().parent

    links: Dict[str, List[Optional[Path]]] = {}
    for link in ET.parse(str(urdf_path)).getroot().iter("link"):
        name = link.get("name")
        if not name:
            continue
        visuals: List[Optional[Path]] = []
        for visual in link.findall("visual"):
            mesh_element = visual.find("geometry/mesh")
            filename = mesh_element.get("filename", "") if mesh_element is not None else ""
            absolute = resolve_mesh_path(filename, root_dir) if filename else None
            keep = absolute is not None and absolute.suffix.lower() == ".dae" and absolute.exists()
            visuals.append(absolute if keep else None)
        if visuals:
            links[name] = visuals
    return links


def resolve_urdf_meshes(
    urdf_path: PathLike,
    mesh_root: Optional[PathLike] = None,
    output_path: Optional[PathLike] = None,
    cache_dir: PathLike = DEFAULT_CACHE_DIR,
) -> Path:
    """Write a Genesis-loadable copy of ``urdf_path`` and return its path.

    ``mesh_root`` defaults to the directory containing the URDF. When
    ``output_path`` is omitted a temporary ``.urdf`` file is created; the caller
    owns it and should delete it on shutdown.
    """
    urdf_path = Path(urdf_path)
    root_dir = Path(mesh_root) if mesh_root is not None else urdf_path.resolve().parent

    tree = ET.parse(str(urdf_path))
    root = tree.getroot()

    for mesh_element in root.iter("mesh"):
        filename = mesh_element.get("filename", "")
        if not filename:
            continue
        absolute = resolve_mesh_path(filename, root_dir)
        if absolute.suffix.lower() == ".dae" and absolute.exists():
            absolute = convert_dae_to_obj(absolute, cache_dir)
        mesh_element.set("filename", str(absolute))

    if output_path is None:
        handle = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False)
        handle.close()
        output_path = Path(handle.name)

    output_path = Path(output_path)
    tree.write(str(output_path), xml_declaration=True)
    logger.info("Wrote resolved URDF: %s", output_path)
    return output_path
