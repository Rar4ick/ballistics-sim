"""Projectile trajectory without air drag.

v0: initial speed, m/s; alpha_deg: launch angle, degrees;
h0: initial height, m; g: gravitational acceleration, m/s²;
n: number of trajectory points.
Returns x, y (arrays, m) and flight time (s).
"""
import numpy as np


def trajectory(v0, alpha_deg, h0=0.0, g=9.81, n=200):
    alpha_rad = np.radians(alpha_deg)  # NumPy trig functions expect radians

    vx, vy = v0 * np.cos(alpha_rad), v0 * np.sin(alpha_rad)

    a, b, c = g, -2 * vy, -2 * h0
    roots = np.roots([a, b, c])
    flight_time = roots.max()
    times_array = np.linspace(0, flight_time, n)

    x = vx * times_array
    y = h0 + vy * times_array - g * times_array ** 2 / 2

    return x, y, flight_time