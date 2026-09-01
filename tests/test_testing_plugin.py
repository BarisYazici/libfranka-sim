"""Tests for franka_sim.testing — the pytest plugin users put in their own CI.

The launch/teardown mechanics are tested against throwaway subprocesses (fast,
no physics); one end-to-end test boots the real MuJoCo-backed server through
the public fixture and is skipped where the FR3 model cannot resolve, the same
skip rule the MuJoCo unit tests use.
"""

import subprocess
import sys
import time

import pytest

from franka_sim.testing import (
    FrankaSimProcess,
    FrankaSimStartupError,
    start_server,
    stop_server,
)


def test_start_server_times_out_on_a_process_that_never_serves():
    # A subprocess that ignores its arguments and just sleeps: the readiness
    # probe must give up within the budget and the child must be reaped.
    with pytest.raises(FrankaSimStartupError, match="did not become ready"):
        start_server(
            timeout=2.0,
            _command=[sys.executable, "-c", "import time; time.sleep(60)"],
        )


def test_start_server_fails_fast_when_the_child_dies():
    # A child that exits immediately must fail well before the timeout, with
    # its exit code and output in the error — not burn the whole 30 s budget.
    started = time.monotonic()
    with pytest.raises(FrankaSimStartupError, match="exited with code 3"):
        start_server(
            timeout=30.0,
            _command=[sys.executable, "-c", "print('boom'); import sys; sys.exit(3)"],
        )
    assert time.monotonic() - started < 10.0


def test_server_output_goes_to_a_file_not_a_pipe():
    # Regression: with stdout=PIPE and no reader, a chatty server fills the
    # pipe and blocks — including its 1 kHz state broadcast. The launcher must
    # hand the child a real file instead.
    with pytest.raises(FrankaSimStartupError) as excinfo:
        start_server(
            timeout=2.0,
            _command=[
                sys.executable,
                "-c",
                # Writes far more than a pipe buffer holds; with a pipe and no
                # reader this child would block forever instead of looping.
                "import time\n"
                "for i in range(300):\n"
                "    print('x' * 1024, flush=True)\n"
                "time.sleep(60)\n",
            ],
        )
    # The launcher survived >64 KiB of unread output and captured its tail.
    assert "xxx" in str(excinfo.value)


def test_stop_server_kills_a_process_that_ignores_sigint():
    ignore_int = (
        "import signal, time\n"
        "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
        "print('armed', flush=True)\n"
        "time.sleep(600)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", ignore_int],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
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
    assert franka_sim_server.log_path  # server output is inspectable
