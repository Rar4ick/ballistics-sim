"""Projectile trajectory without air drag.

v0: initial speed, m/s; alpha_deg: launch angle, degrees;
h0: initial height, m; g: gravitational acceleration, m/s²;
n: number of trajectory points.
Returns x, y (arrays, m) and flight time T (s).
"""
import numpy as np


def trajectory(v0, alpha_deg, h0=0.0, g=9.81, n=200):
    alpha_rad = np.radians(alpha_deg)  # NumPy trig functions expect radians

    vx, vy = v0 * np.cos(alpha_rad), v0 * np.sin(alpha_rad)

    a, b, c = g, -2 * vy, -2 * h0
    roots = np.roots([a, b, c])
    T = roots.max()
    t = np.linspace(0, T, n)

    x = vx * t
    y = h0 + vy * t - g * t ** 2 / 2

    return x, y, T