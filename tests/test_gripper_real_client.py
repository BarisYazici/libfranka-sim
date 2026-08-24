"""Integration test: a real libfranka_new ``franka::Gripper`` vs the sim.

Compiles a tiny C++ probe linked against the prebuilt libfranka_new and points
it at the kinematic gripper server. Skipped unless a prebuilt libfranka_new, a
C++ toolchain, and Eigen are present, so it is a no-op without them.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LIBFRANKA = REPO_ROOT / "libfranka_new"
LIB_BUILD = LIBFRANKA / "build"
EIGEN_INCLUDE = Path("/usr/include/eigen3")
PROBE_SRC = Path(__file__).parent / "_gripper_real_client_probe.cpp"


def _prereqs_available():
    return (
        shutil.which("g++") is not None
        and (LIB_BUILD / "libfranka.so").exists()
        and (LIBFRANKA / "include" / "franka" / "gripper.h").exists()
        and EIGEN_INCLUDE.exists()
    )


pytestmark = pytest.mark.skipif(
    not _prereqs_available(),
    reason="prebuilt libfranka_new + g++ + eigen3 are required for the gripper client test",
)


@pytest.fixture(scope="module")
def gripper_probe_binary(tmp_path_factory):
    out = tmp_path_factory.mktemp("gripper_probe") / "gripper_probe"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            f"-I{LIBFRANKA / 'include'}",
            f"-I{LIBFRANKA / 'common' / 'include'}",
            f"-I{EIGEN_INCLUDE}",
            str(PROBE_SRC),
            f"-L{LIB_BUILD}",
            "-lfranka",
            "-o",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return out


def test_real_gripper_client_connects_homes_and_moves(gripper_server, gripper_probe_binary):
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{LIB_BUILD}:" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(
        [str(gripper_probe_binary), "127.0.0.1"],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )

    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    assert "CONNECT_OK" in result.stdout
    assert "HOMING=1" in result.stdout
    assert "MAX_WIDTH=0.08" in result.stdout
    assert "MOVE=1" in result.stdout
