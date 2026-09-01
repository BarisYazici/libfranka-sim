"""Tests for franka_sim.testing — the pytest plugin users put in their own CI.

The launch/teardown mechanics are tested against a throwaway subprocess (fast,
no physics); one end-to-end test boots the real MuJoCo-backed server through
the public fixture and is skipped where the FR3 model cannot resolve, the same
skip rule the MuJoCo unit tests use.
"""

import subprocess
import sys

import pytest

from franka_sim.testing import FrankaSimProcess, start_server, stop_server


def test_start_server_times_out_on_a_process_that_never_serves():
    # A subprocess that ignores its arguments and just sleeps: the readiness
    # probe must give up within the budget and the child must be reaped.
    with pytest.raises(Exception) as excinfo:
        start_server(
            timeout=2.0,
            _command=[sys.executable, "-c", "import time; time.sleep(60)"],
        )
    assert "franka-sim" in str(excinfo.value)


def test_stop_server_kills_a_process_that_ignores_sigterm():
    ignore_term = (
        "import signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "print('armed', flush=True)\n"
        "time.sleep(600)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", ignore_term],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout.readline().strip() == "armed"
    server = FrankaSimProcess(host="127.0.0.1", port=1, process=proc)
    stop_server(server, grace=1.0)
    assert proc.poll() is not None


def _fr3_model_available() -> bool:
    try:
        from franka_sim.mujoco_franka_sim import default_fr3_mjcf

        return default_fr3_mjcf().exists()
    except Exception:
        return False


@pytest.mark.skipif(not _fr3_model_available(), reason="FR3 MJCF not available")
def test_fixture_boots_a_real_server(franka_sim_server):
    from franka_sim.health_check import check_server

    report = check_server(franka_sim_server.host, franka_sim_server.port, timeout=30.0)
    assert report.server_version == 10
    assert report.state_bytes > 0
    assert franka_sim_server.address == f"127.0.0.1:{franka_sim_server.port}"
