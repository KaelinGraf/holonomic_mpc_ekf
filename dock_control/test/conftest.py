"""Shared fixture: the demo parameter file, loaded fresh for each test."""

import os

from dock_control import params
import pytest

CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config', 'dock_demo.yaml')


@pytest.fixture
def p():
    """Return the demo parameters, a fresh copy for each test so that a test may edit it."""
    return params.load(CONFIG)
