"""Ported from franka_ros2 (jazzy) franka_mobile/test/test_swerve_kinematics.cpp."""

import math

import pytest

from franka_sim.mobile.swerve_kinematics import SwerveKinematics

TOL = 1e-6
L = 1.0
R = 1.0


def make_default_kinematics():
    """Two modules on the x axis at +/-1 m, 1 m wheel radius (the gtest fixture)."""
    return SwerveKinematics([(L, 0.0), (-L, 0.0)], R)


# --- constructor validation ------------------------------------------------


def test_when_parameters_valid_assert_no_throw():
    make_default_kinematics()


@pytest.mark.parametrize(
    "radius",
    [1e-4, 0.0, -0.5, math.inf, math.nan],
    ids=["too_small", "zero", "negative", "infinite", "nan"],
)
def test_when_radius_invalid_assert_raises(radius):
    with pytest.raises(ValueError):
        SwerveKinematics([(1.0, 0.0), (-1.0, 0.0)], radius)


@pytest.mark.parametrize(
    "positions",
    [
        [(0.0, 0.0), (-1.0, 0.0)],
        [(0.0, 0.0), (0.0, 0.0)],
        [(5e-4, 5e-4), (-1.0, 0.0)],
    ],
    ids=["one_zero", "both_zero", "nearly_zero"],
)
def test_when_wheel_position_zero_assert_raises(positions):
    with pytest.raises(ValueError):
        SwerveKinematics(positions, R)


# --- forward kinematics (closed form) --------------------------------------


def test_when_pure_translation_x_assert_velocity_correct():
    vx, vy, wz = make_default_kinematics().forward_kinematics((0.0, 0.0), (1.0, 1.0))
    assert vx == pytest.approx(1.0, abs=TOL)
    assert vy == pytest.approx(0.0, abs=TOL)
    assert wz == pytest.approx(0.0, abs=TOL)


def test_when_pure_translation_x_negative_speed_assert_velocity_correct():
    vx, vy, wz = make_default_kinematics().forward_kinematics((0.0, 0.0), (-1.0, -1.0))
    assert vx == pytest.approx(-1.0, abs=TOL)
    assert vy == pytest.approx(0.0, abs=TOL)
    assert wz == pytest.approx(0.0, abs=TOL)


def test_when_pure_translation_y_assert_velocity_correct():
    half_pi = math.pi / 2.0
    vx, vy, wz = make_default_kinematics().forward_kinematics((half_pi, half_pi), (1.0, 1.0))
    assert vx == pytest.approx(0.0, abs=TOL)
    assert vy == pytest.approx(1.0, abs=TOL)
    assert wz == pytest.approx(0.0, abs=TOL)


def test_when_pure_rotation_assert_angular_velocity_correct():
    half_pi = math.pi / 2.0
    vx, vy, wz = make_default_kinematics().forward_kinematics((half_pi, -half_pi), (1.0, 1.0))
    assert vx == pytest.approx(0.0, abs=TOL)
    assert vy == pytest.approx(0.0, abs=TOL)
    assert wz == pytest.approx(1.0, abs=TOL)


def test_when_pure_rotation_negative_assert_angular_velocity_correct():
    half_pi = math.pi / 2.0
    _, _, wz = make_default_kinematics().forward_kinematics((-half_pi, half_pi), (1.0, 1.0))
    assert wz == pytest.approx(-1.0, abs=TOL)


def test_when_mixed_motion_assert_correct_output():
    angles = (math.pi / 4.0, math.pi / 2.0)
    vx, vy, wz = make_default_kinematics().forward_kinematics(angles, (1.0, 1.0))
    sq2 = math.sqrt(2.0)
    assert vx == pytest.approx(sq2 / 4.0, abs=TOL)
    assert vy == pytest.approx((sq2 / 2.0 + 1.0) / 2.0, abs=TOL)
    assert wz == pytest.approx((sq2 / 2.0 - 1.0) / 2.0, abs=TOL)


def test_when_wheel_radius_changes_assert_scaling():
    radius_2 = 2.0
    sk_2 = SwerveKinematics([(L, 0.0), (-L, 0.0)], radius_2)
    vx, _, _ = make_default_kinematics().forward_kinematics((0.0, 0.0), (1.0, 1.0))
    vx_2, _, _ = sk_2.forward_kinematics((0.0, 0.0), (1.0, 1.0))
    assert vx_2 == pytest.approx(vx * radius_2, abs=TOL)


def test_when_valid_input_assert_forward_returns_result():
    assert make_default_kinematics().forward_kinematics((0.0, 0.0), (1.0, 1.0)) is not None


# --- forward kinematics (QR / least squares) -------------------------------


@pytest.mark.parametrize(
    "angles",
    [(0.0, 0.0), (math.pi / 2.0, -math.pi / 2.0), (math.pi / 4.0, math.pi / 2.0)],
    ids=["translation_x", "rotation", "mixed"],
)
def test_when_qr_assert_matches_closed_form(angles):
    kinematics = make_default_kinematics()
    closed = kinematics.forward_kinematics(angles, (1.0, 1.0))
    qr = kinematics.forward_kinematics_qr(angles, (1.0, 1.0))
    for value, expected in zip(qr, closed):
        assert value == pytest.approx(expected, abs=TOL)


def test_when_qr_zero_input_assert_zero_output():
    vx, vy, wz = make_default_kinematics().forward_kinematics_qr((0.0, 0.0), (0.0, 0.0))
    assert vx == pytest.approx(0.0, abs=TOL)
    assert vy == pytest.approx(0.0, abs=TOL)
    assert wz == pytest.approx(0.0, abs=TOL)


def test_when_valid_input_assert_qr_returns_result():
    assert make_default_kinematics().forward_kinematics_qr((0.0, 0.0), (1.0, 1.0)) is not None


# --- inverse kinematics validation -----------------------------------------


@pytest.mark.parametrize(
    "twist",
    [
        (math.inf, 0.0, 0.0),
        (0.0, math.inf, 0.0),
        (0.0, 0.0, math.inf),
        (math.nan, 0.0, 0.0),
    ],
    ids=["vx_inf", "vy_inf", "wz_inf", "vx_nan"],
)
def test_when_inverse_input_not_finite_assert_none(twist):
    assert make_default_kinematics().inverse_kinematics(*twist) is None


def test_when_inverse_zero_input_assert_result():
    assert make_default_kinematics().inverse_kinematics(0.0, 0.0, 0.0) is not None


# --- inverse kinematics correctness ----------------------------------------


def test_when_pure_translation_x_assert_wheels_point_forward():
    angles, _ = make_default_kinematics().inverse_kinematics(1.0, 0.0, 0.0)
    assert angles[0] == pytest.approx(0.0, abs=TOL)
    assert angles[1] == pytest.approx(0.0, abs=TOL)


def test_when_pure_translation_x_assert_equal_speeds():
    _, speeds = make_default_kinematics().inverse_kinematics(1.0, 0.0, 0.0)
    assert speeds[0] == pytest.approx(speeds[1], abs=TOL)


def test_when_pure_translation_x_assert_correct_speed():
    _, speeds = make_default_kinematics().inverse_kinematics(1.0, 0.0, 0.0)
    assert speeds[0] == pytest.approx(1.0, abs=TOL)


def test_when_pure_translation_y_assert_wheels_point_sideways():
    angles, _ = make_default_kinematics().inverse_kinematics(0.0, 1.0, 0.0)
    assert angles[0] == pytest.approx(math.pi / 2.0, abs=TOL)


def test_when_pure_rotation_assert_equal_speed_magnitudes():
    _, speeds = make_default_kinematics().inverse_kinematics(0.0, 0.0, 1.0)
    assert abs(speeds[0]) == pytest.approx(abs(speeds[1]), abs=TOL)


def test_when_pure_rotation_assert_correct_speed():
    _, speeds = make_default_kinematics().inverse_kinematics(0.0, 0.0, 1.0)
    assert speeds[0] == pytest.approx(1.0, abs=TOL)


def test_when_zero_input_assert_speeds_zero():
    _, speeds = make_default_kinematics().inverse_kinematics(0.0, 0.0, 0.0)
    assert speeds[0] == pytest.approx(0.0, abs=TOL)


def test_when_angle_flip_needed_assert_speed_negated():
    """Reversing the command must not spin the wheel through pi (the stateful part)."""
    kinematics = make_default_kinematics()
    angles, _ = kinematics.inverse_kinematics(1.0, 0.0, 0.0)
    angles_2, speeds_2 = kinematics.inverse_kinematics(-1.0, 0.0, 0.0)
    assert abs(angles_2[0] - angles[0]) <= math.pi / 2.0 + TOL
    assert speeds_2[0] < 0.0


# --- round trip ------------------------------------------------------------


@pytest.mark.parametrize(
    "command",
    [
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 1.0, 0.0),
        (0.5, 0.3, 0.2),
        (-1.0, -0.5, -0.3),
    ],
)
def test_round_trip_closed_form(command):
    kinematics = make_default_kinematics()
    angles, speeds = kinematics.inverse_kinematics(*command)
    recovered = kinematics.forward_kinematics(angles, speeds)
    for value, expected in zip(recovered, command):
        assert value == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize("command", [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.3, 0.2)])
def test_round_trip_qr(command):
    kinematics = make_default_kinematics()
    angles, speeds = kinematics.inverse_kinematics(*command)
    recovered = kinematics.forward_kinematics_qr(angles, speeds)
    for value, expected in zip(recovered, command):
        assert value == pytest.approx(expected, abs=TOL)
