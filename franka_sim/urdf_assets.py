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
"""

import hashlib
import logging
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

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


def convert_dae_to_obj(dae_path: PathLike, cache_dir: PathLike = DEFAULT_CACHE_DIR) -> Path:
    """Merge a COLLADA file into a single ``.obj`` and return the cached path.

    ``trimesh`` is imported lazily so the module stays importable (and the test
    suite stays runnable) without the optional ``mobile`` extra installed.
    """
    import trimesh

    dae_path = Path(dae_path)
    stat = dae_path.stat()
    digest = hashlib.sha1(
        f"{MESH_CACHE_FORMAT_VERSION}:{dae_path}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8")
    ).hexdigest()[:16]

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
