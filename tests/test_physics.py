import numpy as np
import pytest

from ballistics.physics import trajectory

G = 9.81


def test_range_matches_formula():
    # range L = v0² · sin(2α) / g, given h0 = 0
    v0, alpha = 50.0, 45.0
    x, y, flight_time = trajectory(v0, alpha)
    expected = v0**2 * np.sin(np.radians(2 * alpha)) / G
    assert x[-1] == pytest.approx(expected, rel=1e-3)


def test_flight_time_matches_formula():
    # flight_time = 2 · v0 · sin(α) / g, given h0 = 0
    v0, alpha = 50.0, 30.0
    _, _, flight_time = trajectory(v0, alpha)
    assert flight_time == pytest.approx(2 * v0 * np.sin(np.radians(alpha)) / G, rel=1e-6)


def test_max_height_matches_formula():
    # H = v0² · sin²(α) / (2g)
    v0, alpha = 50.0, 60.0
    _, y, _ = trajectory(v0, alpha)
    expected = v0**2 * np.sin(np.radians(alpha)) ** 2 / (2 * G)
    assert y.max() == pytest.approx(expected, rel=1e-3)


def test_starts_at_origin_and_lands_on_ground():
    x, y, _ = trajectory(40.0, 40.0)
    assert x[0] == pytest.approx(0.0)
    assert y[0] == pytest.approx(0.0)
    assert y[-1] == pytest.approx(0.0, abs=1e-6)


def test_complementary_angles_give_same_range():
    x30, _, _ = trajectory(50.0, 30.0)
    x60, _, _ = trajectory(50.0, 60.0)
    assert x30[-1] == pytest.approx(x60[-1], rel=1e-6)


def test_launch_from_height_flies_further_and_lands_at_zero():
    v0, alpha, h0 = 30.0, 30.0, 20.0
    x, y, flight_time = trajectory(v0, alpha, h0=h0)
    vy = v0 * np.sin(np.radians(alpha))
    expected_flight_time = (vy + np.sqrt(vy ** 2 + 2 * G * h0)) / G
    assert flight_time == pytest.approx(expected_flight_time, rel=1e-6)
    assert y[0] == pytest.approx(h0)
    assert y[-1] == pytest.approx(0.0, abs=1e-6)


def test_returns_requested_number_of_points():
    x, y, _ = trajectory(30.0, 45.0, n=50)
    assert len(x) == len(y) == 50


def test_zero_gravity_raises_error():
    with pytest.raises(ValueError):
        trajectory(v0=20, alpha_deg=45, g=0)