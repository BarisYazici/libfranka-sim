import xml.etree.ElementTree as ET
from pathlib import Path

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

        def export(self, path, file_type=None):
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
