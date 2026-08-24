"""Regression: importing franka_sim must not configure the root logger.

``franka_sim_server.py`` used to run ``logging.basicConfig(level=logging.ERROR)``
at module import time, pulled in transitively by ``franka_sim/__init__``. That
installs a root handler as a side effect of *importing a library* -- the
first ``basicConfig()`` call wins and every later one (including
``run_server.main()``'s own, explicitly guarded call) becomes a no-op, so the
CLI could never show anything below ERROR regardless of ``-v`` or its own
logging setup.

A library must leave root-logger configuration to the application that
embeds it. This has to run in a subprocess: the test suite's own conftest.py
calls ``logging.basicConfig()`` for its own purposes, and once *any*
``basicConfig()`` has run in this process the root logger already has a
handler -- which would make the assertion meaningless (or trivially fail for
a reason that has nothing to do with the import under test).
"""

import subprocess
import sys


def test_importing_franka_sim_leaves_the_root_logger_handler_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import logging, franka_sim.franka_sim_server; "
            "assert not logging.getLogger().handlers, logging.getLogger().handlers",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
