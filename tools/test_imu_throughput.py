"""imu_throughput.summarize on synthetic captures: a clean stream, dropped samples, arrival jitter, a slow
stream, a clock offset between machines, a stream without acceleration, and stamps that repeat.

    python3 -m pytest tools/test_imu_throughput.py -q
"""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imu_throughput import summarize  # noqa: E402

HZ = 150.0


def capture(n=2000, hz=HZ, drop=(), jitter_s=0.0, latency_s=0.001, accel_cov=0.01, seed=0):
    stamps = 1000.0 + np.array([i / hz for i in range(n) if i not in set(drop)])
    arrivals = stamps + latency_s + np.random.default_rng(seed).uniform(0.0, jitter_s, len(stamps))
    return arrivals, stamps, [accel_cov] * len(stamps)


def test_a_clean_stream_reports_its_rate_and_no_gaps():
    r = summarize(*capture(), HZ)
    assert r['messages'] == 2000
    assert r['rate_hz'] == pytest.approx(HZ, rel=1e-3) and r['stamp_rate_hz'] == pytest.approx(HZ, rel=1e-3)
    assert abs(r['rate_error_pct']) < 0.1
    assert r['gaps_over_1.5_periods'] == 0 and r['samples_missing_in_gaps'] == 0
    assert r['stamps_not_increasing'] == 0
    assert r['latency_ms']['p50'] == pytest.approx(1.0, abs=1e-3)
    assert r['acceleration_present']


def test_dropped_samples_are_counted_as_gaps_and_missing_samples():
    r = summarize(*capture(drop=(100, 500, 501, 502)), HZ)
    assert r['messages'] == 1996
    assert r['gaps_over_1.5_periods'] == 2
    assert r['samples_missing_in_gaps'] == 4
    assert r['longest_gap_ms'] == pytest.approx(4 * 1e3 / HZ, abs=1e-3)


def test_arrival_jitter_shows_in_the_arrival_intervals_and_not_in_the_stamps():
    r = summarize(*capture(jitter_s=0.004), HZ)
    assert r['arrival_interval_ms']['std'] > 1.0
    assert r['stamp_interval_ms']['std'] < 1e-3
    assert r['gaps_over_1.5_periods'] == 0


def test_a_uniformly_slow_stream_has_no_gaps_but_misses_the_rate():
    r = summarize(*capture(n=1125, hz=112.5), HZ)
    assert r['gaps_over_1.5_periods'] == 0
    assert r['rate_error_pct'] == pytest.approx(-25.0, abs=0.1)


def test_the_clock_offset_turns_a_clock_difference_into_one_way_latency():
    arrivals, stamps, cov = capture(latency_s=0.030)
    assert summarize(arrivals, stamps, cov, HZ)['latency_ms']['p50'] == pytest.approx(30.0, abs=1e-3)
    assert summarize(arrivals, stamps, cov, HZ, clock_offset=-0.028)['latency_ms']['p50'] == \
        pytest.approx(2.0, abs=1e-3)


def test_a_stream_without_acceleration_is_flagged():
    assert not summarize(*capture(accel_cov=-1.0), HZ)['acceleration_present']


def test_repeated_and_backward_stamps_are_counted():
    arrivals, stamps, cov = capture(n=100)
    stamps[50] = stamps[49]
    stamps[70] = stamps[68]
    assert summarize(arrivals, stamps, cov, HZ)['stamps_not_increasing'] == 2


def test_a_single_message_reports_no_rate_rather_than_failing():
    r = summarize([5.0], [4.999], [0.01], HZ)
    assert r['messages'] == 1 and r['rate_hz'] == 0.0 and r['rate_error_pct'] == -100.0
    assert r['arrival_interval_ms'] == {}
