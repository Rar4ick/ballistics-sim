"""Projectile trajectory without air drag.

v0: initial speed, m/s; alpha_deg: launch angle, degrees;
h0: initial height, m; g: gravitational acceleration, m/s²;
n: number of trajectory points.
Returns x, y (arrays, m) and flight time (s).
"""
import numpy as np


def trajectory(
    v0: float,
    alpha_deg: float,
    h0: float = 0.0,
    g: float = 9.81,
    n: int = 200,
) -> tuple[np.ndarray, np.ndarray, float]:
    if g <= 0:
        raise ValueError("g must be positive")

    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("n must be an integer")

    if n < 2:
        raise ValueError("n must be at least 2")

    alpha_rad = np.radians(alpha_deg)  # NumPy trig functions expect radians

    vx, vy = v0 * np.cos(alpha_rad), v0 * np.sin(alpha_rad)

    flight_time = (vy + np.sqrt(vy**2 + 2 * g * h0)) / g
    times_array = np.linspace(0, flight_time, n)

    x = vx * times_array
    y = h0 + vy * times_array - g * times_array ** 2 / 2
    return x, y, flight_time