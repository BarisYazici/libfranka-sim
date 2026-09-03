"""Regression: the single-arm path must honor --port, not silently bind 1337.

The mobile-duo path always forwarded ``args.port``; the single-arm path
dropped it, which broke anything that starts servers on ephemeral ports
(the shipped pytest fixture, parallel CI jobs).

Also covers the other CLI options the single-arm path has to forward rather
than quietly drop.
"""

from unittest.mock import MagicMock, patch

import pytest

import franka_sim
from franka_sim.run_server import build_parser, run_single_arm, validate_args


def test_single_arm_server_receives_cli_port():
    args = build_parser().parse_args(["--port", "42345", "--no-gripper"])
    server = MagicMock()
    with patch.object(franka_sim, "FrankaSimServer", return_value=server) as ctor:
        run_single_arm(args)
    assert ctor.call_args.kwargs["port"] == 42345
    server.start.assert_called_once()


def _ctor_kwargs(argv, env=None, monkeypatch=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    if env is not None:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
    with patch.object(franka_sim, "FrankaSimServer", return_value=MagicMock()) as ctor:
        run_single_arm(args)
    return ctor.call_args.kwargs


def test_gripper_object_width_reaches_the_server(monkeypatch):
    """The flag is the only way a grasp can succeed in the shipped scene.

    Nothing in it is graspable, so an unset width -- the faithful default --
    means every ``franka_gripper`` Grasp action answers false.
    """
    monkeypatch.delenv("FRANKA_SIM_GRIPPER_OBJECT_WIDTH", raising=False)
    assert _ctor_kwargs(["--no-gripper"])["gripper_object_width"] is None
    assert (
        _ctor_kwargs(["--no-gripper", "--gripper-object-width", "0.04"])[
            "gripper_object_width"
        ]
        == 0.04
    )


def test_gripper_object_width_falls_back_to_the_environment(monkeypatch):
    """For a server started from a launch file or a container image."""
    kwargs = _ctor_kwargs(
        ["--no-gripper"], env={"FRANKA_SIM_GRIPPER_OBJECT_WIDTH": "0.035"}, monkeypatch=monkeypatch
    )
    assert kwargs["gripper_object_width"] == 0.035
    # The flag is explicit and wins over the environment.
    kwargs = _ctor_kwargs(
        ["--no-gripper", "--gripper-object-width", "0.02"],
        env={"FRANKA_SIM_GRIPPER_OBJECT_WIDTH": "0.035"},
        monkeypatch=monkeypatch,
    )
    assert kwargs["gripper_object_width"] == 0.02


def test_gripper_object_width_outside_the_stroke_is_rejected():
    """Silently useless otherwise -- neither backend can put it between the fingers."""
    with pytest.raises(ValueError):
        validate_args(build_parser().parse_args(["--gripper-object-width", "0.5"]))
    with pytest.raises(ValueError):
        validate_args(build_parser().parse_args(["--gripper-object-width", "-0.01"]))
