"""
The one parameter file (brief Section 9): load it, check it, hand it to a node.

Every node and every test reads config/dock_demo.yaml through load(), so each
number has exactly one home. The file mirrors dock_params('demo') in the MATLAB
units (degrees, rates); the constructors that use a value convert it once.
"""

import math
import os

import yaml

PACKAGE = 'dock_control'
DEFAULT_FILE = 'dock_demo.yaml'

DERIVED_RTOL = 1e-3


def default_path():
    """Return the path of the installed config/dock_demo.yaml."""
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory(PACKAGE), 'config', DEFAULT_FILE)


def load(path):
    """Load a parameter file and return its ``dock`` mapping."""
    with open(path) as f:
        p = yaml.safe_load(f)['dock']
    _require_numbers(p, 'dock')
    return p


def _require_numbers(node, where):
    if isinstance(node, dict):
        for key, value in node.items():
            _require_numbers(value, f'{where}.{key}')
    elif isinstance(node, list):
        for i, value in enumerate(node):
            _require_numbers(value, f'{where}[{i}]')
    elif isinstance(node, bool) or not isinstance(node, (int, float)):
        raise ValueError(f'{where} = {node!r} is not a number')


def check(p):
    """Recompute the derived MPC weights and return the mismatches (empty if consistent)."""
    lim, mpc, tol = p['lim'], p['mpc'], p['tol']
    du_max = [mpc['Ts'] * lim['a_max'], mpc['Ts'] * lim['a_max'], mpc['Ts'] * lim['alpha_max']]
    expected = {
        'Qy': [tol['y'] ** -2, tol['tgt_y'] ** -2, math.radians(tol['tgt_th_deg']) ** -2],
        'Ru': [d ** -2 for d in du_max],
    }
    mismatches = []
    for key, want in expected.items():
        have = mpc[key]
        if len(have) != len(want) or any(
                abs(h - w) > DERIVED_RTOL * abs(w) for h, w in zip(have, want)):
            mismatches.append(f'mpc.{key} = {have}, expected {[round(w, 1) for w in want]}')
    return mismatches


def from_node(node):
    """Load the file named by the node's ``config`` parameter; refuse to start if inconsistent."""
    path = node.declare_parameter('config', '').value or default_path()
    p = load(path)
    mismatches = check(p)
    if mismatches:
        raise RuntimeError(f'{path} disagrees with its derived values: ' + '; '.join(mismatches))
    node.get_logger().info(f'parameters from {path}')
    return p
