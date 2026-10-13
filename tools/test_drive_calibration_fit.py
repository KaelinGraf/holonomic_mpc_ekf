"""
Test drive_calibration_fit on synthetic data.

The steps it finds in cmd_vel, the dead-zone fit, and a simulated run whose deadband, gains,
strafe slip and turn radius it must recover.

    python3 -m pytest tools/test_drive_calibration_fit.py -q
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drive_calibration_fit as fit  # noqa: E402, I100


def test_steps_are_runs_of_one_axis():
    """Single-axis runs long enough count; zeros, mixed axes and short blips do not."""
    t = np.arange(0.0, 10.0, 0.02)
    cmd = np.zeros((len(t), 4))
    cmd[:, 0] = t
    cmd[(t >= 1.0) & (t < 2.5), 1] = 0.06
    cmd[(t >= 3.0) & (t < 4.5), 3] = -0.2
    cmd[(t >= 5.0) & (t < 6.5), 1:3] = 0.05
    cmd[(t >= 7.0) & (t < 7.2), 2] = 0.1
    steps = fit.find_steps(cmd)
    assert [(s.axis, s.cmd) for s in steps] == [('x', 0.06), ('r', -0.2)]
    assert steps[0].t0 == pytest.approx(1.0) and steps[0].t1 == pytest.approx(2.5)


def test_the_dead_zone_fit_is_exact_on_clean_data():
    """Recover response = g (|c| - d) above d, and 0 below, to the grid's resolution."""
    c = np.array([0.02, 0.04, 0.06, 0.10, 0.15])
    s = 1.6 * np.maximum(c - 0.033, 0.0)
    d, g, rms = fit.deadzone_fit(c, s)
    assert d == pytest.approx(0.033, abs=0.15 / 800) and g == pytest.approx(1.6, rel=0.01)
    assert rms < 1e-4


def test_a_simulated_run_is_recovered():
    """The --selftest chassis: a 45 us pulse deadband, 80% strafe, a 0.19 m turn radius."""
    fit.selftest()
