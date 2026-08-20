import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from franka_sim import urdf_assets

URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="fixture">
  <link name="base_link">
    <visual>
      <geometry><mesh filename="{visual}"/></geometry>
    </visual>
    <collision>
      <geometry><mesh filename="{collision}"/></geometry>
    </collision>
  </link>
</robot>
"""


def write_urdf(tmp_path, visual, collision):
    urdf_path = tmp_path / "fixture.urdf"
    urdf_path.write_text(URDF_TEMPLATE.format(visual=visual, collision=collision))
    return urdf_path


def mesh_filenames(urdf_path):
    root = ET.parse(str(urdf_path)).getroot()
    return [element.get("filename") for element in root.iter("mesh")]


def test_resolve_mesh_path_strips_package_scheme(tmp_path):
    resolved = urdf_assets.resolve_mesh_path(
        "package://franka_description/meshes/robots/tmrv0_2/collision/tmrv0_2.stl", tmp_path
    )
    assert resolved == (tmp_path / "meshes/robots/tmrv0_2/collision/tmrv0_2.stl").resolve()


def test_resolve_mesh_path_joins_relative_path(tmp_path):
    resolved = urdf_assets.resolve_mesh_path("meshes/base.stl", tmp_path)
    assert resolved == (tmp_path / "meshes/base.stl").resolve()


def test_resolve_mesh_path_keeps_absolute_path(tmp_path):
    absolute = tmp_path / "already/absolute.stl"
    assert urdf_assets.resolve_mesh_path(str(absolute), tmp_path) == absolute


def test_resolve_urdf_meshes_rewrites_to_absolute_paths(tmp_path):
    mesh_root = tmp_path / "franka_description"
    (mesh_root / "meshes").mkdir(parents=True)
    (mesh_root / "meshes" / "base.stl").write_bytes(b"solid\n")
    urdf_path = write_urdf(
        tmp_path,
        visual="package://franka_description/meshes/base.stl",
        collision="package://franka_description/meshes/base.stl",
    )

    output = urdf_assets.resolve_urdf_meshes(
        urdf_path, mesh_root=mesh_root, output_path=tmp_path / "resolved.urdf"
    )

    expected = str((mesh_root / "meshes" / "base.stl").resolve())
    assert mesh_filenames(output) == [expected, expected]


def test_resolve_urdf_meshes_leaves_source_untouched(tmp_path):
    mesh_root = tmp_path / "franka_description"
    (mesh_root / "meshes").mkdir(parents=True)
    (mesh_root / "meshes" / "base.stl").write_bytes(b"solid\n")
    urdf_path = write_urdf(
        tmp_path,
        visual="package://franka_description/meshes/base.stl",
        collision="package://franka_description/meshes/base.stl",
    )
    original = urdf_path.read_text()

    urdf_assets.resolve_urdf_meshes(
        urdf_path, mesh_root=mesh_root, output_path=tmp_path / "resolved.urdf"
    )

    assert urdf_path.read_text() == original


def test_resolve_urdf_meshes_converts_dae_visuals(tmp_path, monkeypatch):
    """A .dae that exists on disk is routed through the OBJ converter."""
    mesh_root = tmp_path / "franka_description"
    (mesh_root / "meshes").mkdir(parents=True)
    (mesh_root / "meshes" / "body.dae").write_text("<COLLADA/>")
    (mesh_root / "meshes" / "body.stl").write_bytes(b"solid\n")
    urdf_path = write_urdf(
        tmp_path,
        visual="package://franka_description/meshes/body.dae",
        collision="package://franka_description/meshes/body.stl",
    )

    converted = tmp_path / "cache" / "body.obj"
    converted.parent.mkdir()
    converted.write_text("o body\n")
    calls = []

    def fake_convert(dae_path, cache_dir=urdf_assets.DEFAULT_CACHE_DIR):
        calls.append(Path(dae_path))
        return converted

    monkeypatch.setattr(urdf_assets, "convert_dae_to_obj", fake_convert)

    output = urdf_assets.resolve_urdf_meshes(
        urdf_path, mesh_root=mesh_root, output_path=tmp_path / "resolved.urdf"
    )

    assert calls == [(mesh_root / "meshes" / "body.dae").resolve()]
    assert mesh_filenames(output) == [
        str(converted),
        str((mesh_root / "meshes" / "body.stl").resolve()),
    ]


def test_resolve_urdf_meshes_skips_missing_dae(tmp_path, monkeypatch):
    """A .dae that is not on disk is left as an absolute path, not converted."""
    mesh_root = tmp_path / "franka_description"
    (mesh_root / "meshes").mkdir(parents=True)
    urdf_path = write_urdf(
        tmp_path,
        visual="package://franka_description/meshes/missing.dae",
        collision="package://franka_description/meshes/missing.dae",
    )

    def fail_convert(dae_path, cache_dir=urdf_assets.DEFAULT_CACHE_DIR):
        raise AssertionError("converter must not run for a missing file")

    monkeypatch.setattr(urdf_assets, "convert_dae_to_obj", fail_convert)

    output = urdf_assets.resolve_urdf_meshes(
        urdf_path, mesh_root=mesh_root, output_path=tmp_path / "resolved.urdf"
    )
    expected = str((mesh_root / "meshes" / "missing.dae").resolve())
    assert mesh_filenames(output) == [expected, expected]


def test_resolve_urdf_meshes_defaults_mesh_root_to_urdf_directory(tmp_path):
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "base.stl").write_bytes(b"solid\n")
    urdf_path = write_urdf(tmp_path, visual="meshes/base.stl", collision="meshes/base.stl")

    output = urdf_assets.resolve_urdf_meshes(urdf_path, output_path=tmp_path / "resolved.urdf")

    expected = str((tmp_path / "meshes" / "base.stl").resolve())
    assert mesh_filenames(output) == [expected, expected]


def test_resolve_urdf_meshes_writes_a_temp_file_when_no_output_given(tmp_path):
    (tmp_path / "meshes").mkdir()
    (tmp_path / "meshes" / "base.stl").write_bytes(b"solid\n")
    urdf_path = write_urdf(tmp_path, visual="meshes/base.stl", collision="meshes/base.stl")

    output = urdf_assets.resolve_urdf_meshes(urdf_path)
    try:
        assert output.exists()
        assert output.suffix == ".urdf"
        assert output != urdf_path
    finally:
        output.unlink(missing_ok=True)


def test_convert_dae_to_obj_reuses_the_cache(tmp_path, monkeypatch):
    """A second call with an unchanged source file must not re-run trimesh."""
    dae_path = tmp_path / "body.dae"
    dae_path.write_text("<COLLADA/>")
    cache_dir = tmp_path / "cache"

    loads = []

    class FakeMesh:
        extents = type("Extents", (), {"max": staticmethod(lambda: 1.0)})()

        def apply_scale(self, factor):
            raise AssertionError("no rescale expected for a metre-sized mesh")

        def export(self, path, file_type=None, mtl_name=None):
            Path(path).write_text("o body\n")

    class FakeTrimesh:
        Scene = type("Scene", (), {})

        @staticmethod
        def load(path):
            loads.append(path)
            return FakeMesh()

    monkeypatch.setitem(__import__("sys").modules, "trimesh", FakeTrimesh)

    first = urdf_assets.convert_dae_to_obj(dae_path, cache_dir=cache_dir)
    second = urdf_assets.convert_dae_to_obj(dae_path, cache_dir=cache_dir)

    assert first == second
    assert first.exists()
    assert len(loads) == 1


def test_convert_dae_to_obj_names_material_and_mtl_uniquely(tmp_path, monkeypatch):
    """Regression test: trimesh's OBJ exporter defaults sidecar files to the
    generic names ``material.mtl`` / ``material_0.png``. Because every mesh
    converted by this module shares one cache_dir, exporting two different
    .dae files without a per-mesh name clobbers the first mesh's material
    with the second's -- the .obj on disk still references "material.mtl",
    but that file now belongs to a different mesh. This made most of the
    mobile-duo platform render near-black (wrong/mismatched texture atlas).
    The fix names each material/mtl after the same digest-qualified stem as
    the cached .obj so sidecar files can never collide between meshes.
    """
    dae_path = tmp_path / "body.dae"
    dae_path.write_text("<COLLADA/>")
    cache_dir = tmp_path / "cache"

    export_calls = []

    class FakeMaterial:
        name = None

    class FakeVisual:
        material = FakeMaterial()

    class FakeMesh:
        extents = type("Extents", (), {"max": staticmethod(lambda: 1.0)})()
        visual = FakeVisual()

        def apply_scale(self, factor):
            raise AssertionError("no rescale expected for a metre-sized mesh")

        def export(self, path, file_type=None, mtl_name=None):
            export_calls.append((path, file_type, mtl_name))
            Path(path).write_text("o body\n")

    class FakeTrimesh:
        Scene = type("Scene", (), {})

        @staticmethod
        def load(path):
            return FakeMesh()

    monkeypatch.setitem(__import__("sys").modules, "trimesh", FakeTrimesh)

    obj_path = urdf_assets.convert_dae_to_obj(dae_path, cache_dir=cache_dir)

    assert len(export_calls) == 1
    _, file_type, mtl_name = export_calls[0]
    assert file_type == "obj"
    assert mtl_name == f"{obj_path.stem}.mtl"
    assert FakeMesh.visual.material.name == obj_path.stem


# -- per-material split ---------------------------------------------------
#
# These drive the real trimesh grouping/export code over a Scene built in
# memory: only ``trimesh.load`` is faked, because the one thing a fixture
# cannot provide is a COLLADA file (writing one needs pycollada).


@pytest.fixture
def trimesh():
    """The real trimesh, or a skip when the optional ``mobile`` extra is absent."""
    return pytest.importorskip("trimesh")


def coloured_box(trimesh, colour, translation, size=1.0):
    """One box with a COLLADA-style PBR material, in its own local frame."""
    mesh = trimesh.creation.box(extents=(size, size, size))
    mesh.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(baseColorFactor=colour)
    )
    mesh.apply_translation(translation)
    return mesh


def scene_of(trimesh, *placed):
    """A Scene holding ``(mesh, node_translation)`` pairs, like a COLLADA graph.

    The node translation is deliberately *not* baked into the mesh: it is the
    scene-graph transform the split has to apply itself.
    """
    scene = trimesh.Scene()
    for mesh, node_translation in placed:
        scene.add_geometry(
            mesh, transform=trimesh.transformations.translation_matrix(node_translation)
        )
    return scene


@pytest.fixture
def fake_dae(tmp_path, monkeypatch, trimesh):
    """Return a factory that makes ``trimesh.load`` serve a given scene."""

    def make(scene, name="body.dae"):
        dae_path = tmp_path / name
        dae_path.write_text("<COLLADA/>")
        monkeypatch.setattr(trimesh, "load", lambda path: scene)
        return dae_path

    return make


def test_split_dae_by_material_makes_one_obj_per_material(tmp_path, trimesh, fake_dae):
    scene = scene_of(
        trimesh,
        (coloured_box(trimesh, [255, 0, 0, 255], (0.0, 0.0, 0.0)), (0.0, 0.0, 0.0)),
        (coloured_box(trimesh, [0, 0, 255, 255], (0.0, 0.0, 0.0), size=0.5), (0.0, 0.0, 0.0)),
    )
    dae_path = fake_dae(scene)

    submeshes = urdf_assets.split_dae_by_material(dae_path, cache_dir=tmp_path / "cache")

    assert {submesh.rgba for submesh in submeshes} == {
        (1.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 1.0, 1.0),
    }
    assert all(submesh.path.exists() for submesh in submeshes)
    assert len({submesh.path for submesh in submeshes}) == 2
    # Nothing is dropped: two boxes in, two boxes' worth of triangles out.
    assert [submesh.faces for submesh in submeshes] == [12, 12]


def test_split_dae_by_material_merges_every_geometry_of_one_colour(tmp_path, trimesh, fake_dae):
    """1126 sub-meshes in five colours must become five geoms, not 1126."""
    scene = scene_of(
        trimesh,
        *[
            (coloured_box(trimesh, [255, 0, 0, 255], (0.0, 0.0, 0.0)), (float(i), 0.0, 0.0))
            for i in range(5)
        ],
        (coloured_box(trimesh, [0, 0, 255, 255], (0.0, 0.0, 0.0)), (0.0, 3.0, 0.0)),
    )
    dae_path = fake_dae(scene)

    submeshes = urdf_assets.split_dae_by_material(dae_path, cache_dir=tmp_path / "cache")

    assert len(submeshes) == 2
    assert [submesh.faces for submesh in submeshes] == [5 * 12, 12]


def test_split_dae_by_material_applies_the_scene_graph_transforms(tmp_path, trimesh, fake_dae):
    """The sub-meshes must land where the merged mesh draws them, not at the origin."""
    scene = scene_of(
        trimesh,
        (coloured_box(trimesh, [255, 0, 0, 255], (0.0, 0.0, 0.0)), (2.0, 0.0, 0.0)),
        (coloured_box(trimesh, [0, 0, 255, 255], (0.0, 0.0, 0.0), size=0.5), (-2.0, 0.0, 0.0)),
    )
    dae_path = fake_dae(scene)
    merged = trimesh.util.concatenate(scene.dump()).bounds

    submeshes = urdf_assets.split_dae_by_material(dae_path, cache_dir=tmp_path / "cache")

    bounds = np.array([trimesh.load_mesh(str(submesh.path)).bounds for submesh in submeshes])
    assert bounds[:, 0, :].min(axis=0) == pytest.approx(merged[0])
    assert bounds[:, 1, :].max(axis=0) == pytest.approx(merged[1])


def test_split_dae_by_material_rescales_a_millimetre_file(tmp_path, trimesh, fake_dae):
    """Same mm heuristic as convert_dae_to_obj, applied over the whole file.

    The small group is under the threshold on its own; scaling it by anything
    but the large group's factor would tear the link apart.
    """
    scene = scene_of(
        trimesh,
        (coloured_box(trimesh, [255, 0, 0, 255], (0.0, 0.0, 0.0), size=800.0), (0.0, 0.0, 0.0)),
        (coloured_box(trimesh, [0, 0, 255, 255], (0.0, 0.0, 0.0), size=2.0), (0.0, 0.0, 0.0)),
    )
    dae_path = fake_dae(scene)

    submeshes = urdf_assets.split_dae_by_material(dae_path, cache_dir=tmp_path / "cache")

    extents = {submesh.rgba: trimesh.load_mesh(str(submesh.path)).extents for submesh in submeshes}
    assert extents[(1.0, 0.0, 0.0, 1.0)] == pytest.approx([0.8, 0.8, 0.8])
    assert extents[(0.0, 0.0, 1.0, 1.0)] == pytest.approx([0.002, 0.002, 0.002])


def test_split_dae_by_material_keeps_a_metre_file_unscaled(tmp_path, trimesh, fake_dae):
    scene = scene_of(
        trimesh, (coloured_box(trimesh, [255, 0, 0, 255], (0.0, 0.0, 0.0), size=0.8), (0, 0, 0))
    )
    dae_path = fake_dae(scene)

    (submesh,) = urdf_assets.split_dae_by_material(dae_path, cache_dir=tmp_path / "cache")

    assert trimesh.load_mesh(str(submesh.path)).extents == pytest.approx([0.8, 0.8, 0.8])


def test_split_dae_by_material_returns_one_submesh_for_one_material(tmp_path, trimesh, fake_dae):
    """A single-material .dae splits into one coloured geom -- the right answer."""
    scene = scene_of(
        trimesh, (coloured_box(trimesh, [112, 112, 112, 255], (0.0, 0.0, 0.0)), (0, 0, 0))
    )
    dae_path = fake_dae(scene)

    submeshes = urdf_assets.split_dae_by_material(dae_path, cache_dir=tmp_path / "cache")

    assert len(submeshes) == 1
    assert submeshes[0].rgba == pytest.approx((112 / 255, 112 / 255, 112 / 255, 1.0))


def test_split_dae_by_material_handles_a_bare_mesh(tmp_path, trimesh, fake_dae):
    """Not every COLLADA loads as a Scene; a lone Trimesh must still split."""
    dae_path = fake_dae(coloured_box(trimesh, [0, 255, 0, 255], (0.0, 0.0, 0.0)))

    submeshes = urdf_assets.split_dae_by_material(dae_path, cache_dir=tmp_path / "cache")

    assert len(submeshes) == 1
    assert submeshes[0].rgba == (0.0, 1.0, 0.0, 1.0)


def test_split_dae_by_material_falls_back_to_grey_without_a_material(tmp_path, trimesh, fake_dae):
    dae_path = fake_dae(scene_of(trimesh, (trimesh.creation.box(extents=(1, 1, 1)), (0, 0, 0))))

    (submesh,) = urdf_assets.split_dae_by_material(dae_path, cache_dir=tmp_path / "cache")

    assert submesh.rgba == urdf_assets.DEFAULT_SUBMESH_RGBA


def test_split_dae_by_material_writes_self_contained_objs(tmp_path, trimesh, fake_dae):
    """No ``mtllib``: every mesh shares one cache dir, so sidecars would collide.

    ``convert_dae_to_obj`` had to work around exactly that (see its regression
    test); the split sidesteps it by carrying the colour out of band instead.
    """
    cache_dir = tmp_path / "cache"
    dae_path = fake_dae(
        scene_of(trimesh, (coloured_box(trimesh, [255, 0, 0, 255], (0, 0, 0)), (0, 0, 0)))
    )

    (submesh,) = urdf_assets.split_dae_by_material(dae_path, cache_dir=cache_dir)

    assert "mtllib" not in submesh.path.read_text()
    assert not list(cache_dir.glob("*.mtl"))
    assert not list(cache_dir.glob("*.png"))


def test_split_dae_by_material_reuses_the_cache(tmp_path, trimesh, monkeypatch):
    """A second call with an unchanged source must not re-run trimesh."""
    dae_path = tmp_path / "body.dae"
    dae_path.write_text("<COLLADA/>")
    cache_dir = tmp_path / "cache"
    scene = scene_of(
        trimesh,
        (coloured_box(trimesh, [255, 0, 0, 255], (0, 0, 0)), (0, 0, 0)),
        (coloured_box(trimesh, [0, 0, 255, 255], (0, 0, 0), size=0.5), (0, 0, 0)),
    )
    loads = []

    def counting_load(path):
        loads.append(path)
        return scene

    monkeypatch.setattr(trimesh, "load", counting_load)

    first = urdf_assets.split_dae_by_material(dae_path, cache_dir=cache_dir)
    second = urdf_assets.split_dae_by_material(dae_path, cache_dir=cache_dir)

    assert first == second
    assert len(loads) == 1


def test_split_dae_by_material_rebuilds_when_an_obj_is_missing(tmp_path, trimesh, fake_dae):
    """A manifest whose .obj files were swept away must not be trusted."""
    cache_dir = tmp_path / "cache"
    dae_path = fake_dae(
        scene_of(trimesh, (coloured_box(trimesh, [255, 0, 0, 255], (0, 0, 0)), (0, 0, 0)))
    )

    (first,) = urdf_assets.split_dae_by_material(dae_path, cache_dir=cache_dir)
    first.path.unlink()
    (second,) = urdf_assets.split_dae_by_material(dae_path, cache_dir=cache_dir)

    assert second.path == first.path
    assert second.path.exists()


def test_split_dae_by_material_ignores_a_corrupt_manifest(tmp_path, trimesh, fake_dae):
    cache_dir = tmp_path / "cache"
    dae_path = fake_dae(
        scene_of(trimesh, (coloured_box(trimesh, [255, 0, 0, 255], (0, 0, 0)), (0, 0, 0)))
    )

    urdf_assets.split_dae_by_material(dae_path, cache_dir=cache_dir)
    manifest = next(cache_dir.glob("*.json"))
    manifest.write_text("{not json")

    (rebuilt,) = urdf_assets.split_dae_by_material(dae_path, cache_dir=cache_dir)

    assert rebuilt.rgba == (1.0, 0.0, 0.0, 1.0)
    assert json.loads(manifest.read_text())["submeshes"]


def test_the_split_and_merge_caches_do_not_share_entries(tmp_path, trimesh, monkeypatch):
    """Genesis' merged .obj must survive a change to the split, and vice versa.

    Both conversions key on the same (path, mtime, size) triple in the same
    directory, so the only thing keeping them apart is the conversion kind and
    version folded into the digest.
    """
    dae_path = tmp_path / "body.dae"
    dae_path.write_text("<COLLADA/>")
    cache_dir = tmp_path / "cache"
    scene = scene_of(trimesh, (coloured_box(trimesh, [255, 0, 0, 255], (0, 0, 0)), (0, 0, 0)))
    monkeypatch.setattr(trimesh, "load", lambda path: scene)

    merged = urdf_assets.convert_dae_to_obj(dae_path, cache_dir=cache_dir)
    (split,) = urdf_assets.split_dae_by_material(dae_path, cache_dir=cache_dir)

    assert merged != split.path
    assert merged.exists() and split.path.exists()
    # ...and bumping one version leaves the other's entry addressable.
    monkeypatch.setattr(urdf_assets, "SUBMESH_CACHE_FORMAT_VERSION", 99)
    (rebuilt,) = urdf_assets.split_dae_by_material(dae_path, cache_dir=cache_dir)
    assert rebuilt.path != split.path
    assert merged == urdf_assets.convert_dae_to_obj(dae_path, cache_dir=cache_dir)


# -- link -> COLLADA mapping ----------------------------------------------


LINK_URDF = """<?xml version="1.0"?>
<robot name="fixture">
  <link name="base_link">
    <visual><geometry><mesh filename="package://d/meshes/body.dae"/></geometry></visual>
    <visual><geometry><box size="1 1 1"/></geometry></visual>
    <visual><geometry><mesh filename="package://d/meshes/cover.dae"/></geometry></visual>
    <collision><geometry><mesh filename="package://d/meshes/body.stl"/></geometry></collision>
  </link>
  <link name="wheel_link">
    <visual><geometry><cylinder radius="1" length="1"/></geometry></visual>
  </link>
  <link name="frame_link"/>
</robot>
"""


def test_link_visual_dae_meshes_keeps_document_order_and_placeholders(tmp_path):
    mesh_root = tmp_path / "d"
    (mesh_root / "meshes").mkdir(parents=True)
    for name in ("body.dae", "cover.dae"):
        (mesh_root / "meshes" / name).write_text("<COLLADA/>")
    urdf_path = tmp_path / "fixture.urdf"
    urdf_path.write_text(LINK_URDF)

    links = urdf_assets.link_visual_dae_meshes(urdf_path, mesh_root=mesh_root)

    assert links["base_link"] == [
        (mesh_root / "meshes" / "body.dae").resolve(),
        None,
        (mesh_root / "meshes" / "cover.dae").resolve(),
    ]
    assert links["wheel_link"] == [None]
    # A link with no <visual> at all is not listed.
    assert "frame_link" not in links


def test_link_visual_dae_meshes_skips_missing_and_non_dae_files(tmp_path):
    mesh_root = tmp_path / "d"
    (mesh_root / "meshes").mkdir(parents=True)
    (mesh_root / "meshes" / "cover.dae").write_text("<COLLADA/>")
    urdf_path = tmp_path / "fixture.urdf"
    urdf_path.write_text(LINK_URDF.replace("body.dae", "body.stl"))

    links = urdf_assets.link_visual_dae_meshes(urdf_path, mesh_root=mesh_root)

    # body.stl is not COLLADA; the box is not a mesh; only cover.dae survives.
    assert links["base_link"] == [None, None, (mesh_root / "meshes" / "cover.dae").resolve()]
