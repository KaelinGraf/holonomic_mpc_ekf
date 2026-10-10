"""The parameter file: numbers only, and consistent with its derived values."""

from dock_control import params
import pytest


def test_demo_file_is_consistent(p):
    """The demo file passes the start-up check that every node runs."""
    assert params.check(p) == []


def test_unsigned_exponent_is_rejected(tmp_path):
    """YAML 1.1 reads 5.8e6 as a string, and load() refuses it."""
    bad = tmp_path / 'bad.yaml'
    bad.write_text('dock:\n  mpc: {rho_s: 5.8e6}\n')
    with pytest.raises(ValueError, match='rho_s'):
        params.load(str(bad))


def test_inconsistent_weight_is_reported(p):
    """A Q_y that no longer matches the tolerances is named in the mismatches."""
    p['mpc']['Qy'][0] *= 2.0
    assert any('Qy' in m for m in params.check(p))
