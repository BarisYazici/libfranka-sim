"""Integration smoke test: a real libfranka_new (v10) client against the sim.

This compiles and runs a tiny C++ client (``_v10_real_client_probe.cpp``) linked
against the prebuilt libfranka_new and points it at the simulator. It proves the
full v10 wire contract end-to-end -- Connect, a 1377-byte ``RobotState``, and
``GetRobotModel`` returning a buildable URDF -- against the actual library, with
Genesis mocked.

It is skipped automatically unless a prebuilt libfranka_new, a C++ toolchain, and
Eigen are present, so it is a no-op in environments without them.
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
PROBE_SRC = Path(__file__).parent / "_v10_real_client_probe.cpp"


def _prereqs_available():
    return (
        shutil.which("g++") is not None
        and (LIB_BUILD / "libfranka.so").exists()
        and (LIBFRANKA / "include" / "franka" / "robot.h").exists()
        and EIGEN_INCLUDE.exists()
    )


pytestmark = pytest.mark.skipif(
    not _prereqs_available(),
    reason="prebuilt libfranka_new + g++ + eigen3 are required for the real-client smoke test",
)


@pytest.fixture(scope="module")
def probe_binary(tmp_path_factory):
    """Compile the real-client probe against the prebuilt libfranka_new."""
    out = tmp_path_factory.mktemp("v10_probe") / "v10_probe"
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


def test_real_v10_client_connects_reads_and_gets_model(sim_server, mock_genesis_sim, probe_binary):
    """A stock libfranka_new client completes Connect, reads state, and gets the URDF."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{LIB_BUILD}:" + env.get("LD_LIBRARY_PATH", "")

    result = subprocess.run(
        [str(probe_binary), "127.0.0.1"],
        capture_output=True,
        text=True,
        timeout=20,
        env=env,
    )

    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    assert "CONNECT_OK" in result.stdout
    assert "has_joint1=1" in result.stdout  # URDF parsed as a 7-DOF arm
    assert "READ_OK states_read=5" in result.stdout
