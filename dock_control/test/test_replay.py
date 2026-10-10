"""
Replay tests (brief Section 11): the acceptance tests for the port.

Export the MATLAB sensor fixtures and one MPC tick into test/fixtures/ with
export_fixtures.m first. These two results are the evidence that the port is
the design.
"""

import pytest


@pytest.mark.skip(reason='TODO: needs test/fixtures/ from export_fixtures.m')
def test_ekf_replay_matches_matlab(p):
    """Replay the MATLAB IMU samples and fixes through Ekf: run_ekf's estimates to 1e-9."""


@pytest.mark.skip(reason='TODO: needs test/fixtures/ from export_fixtures.m')
def test_mpc_tick_matches_matlab(p):
    """Run the dumped MATLAB tick through Mpc.step: dU agrees to the solver tolerance."""
