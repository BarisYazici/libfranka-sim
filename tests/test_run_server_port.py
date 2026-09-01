"""Regression: the single-arm path must honor --port, not silently bind 1337.

The mobile-duo path always forwarded ``args.port``; the single-arm path
dropped it, which broke anything that starts servers on ephemeral ports
(the shipped pytest fixture, parallel CI jobs).
"""

from unittest.mock import MagicMock, patch

import franka_sim
from franka_sim.run_server import build_parser, run_single_arm


def test_single_arm_server_receives_cli_port():
    args = build_parser().parse_args(["--port", "42345", "--no-gripper"])
    server = MagicMock()
    with patch.object(franka_sim, "FrankaSimServer", return_value=server) as ctor:
        run_single_arm(args)
    assert ctor.call_args.kwargs["port"] == 42345
    server.start.assert_called_once()
