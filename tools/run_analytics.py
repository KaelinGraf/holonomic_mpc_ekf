#!/usr/bin/env python3
"""
Turn the robot's bags into report-ready records: one folder per run, then an index across the runs.

    python3 tools/run_analytics.py <bag or folder of bags> [--out <records dir>] [--force]
    python3 tools/run_analytics.py /workspace/run_data/pi_bags            # every run not yet recorded
    python3 tools/run_analytics.py <bag> --force --no-sim                  # redo one run, skip its simulation
    python3 tools/run_analytics.py --index-only                            # rebuild INDEX.md and the aggregate
    python3 tools/run_analytics.py <folder> --detectors --finetune         # also the detector comparisons (GPU)

Run it in p4p-gpu with both workspaces sourced, for dock_msgs and convchart_interfaces:

    docker exec p4p-gpu bash -lc 'source /workspace/controller_ws/install/setup.bash; \\
        source /workspace/Conv-ChArT-Wireless-Inference/src/ros/install/setup.bash; \\
        cd /workspace/controller_ws && python3 tools/run_analytics.py /workspace/run_data/pi_bags'

It reads bags with rosbag2_py and never starts a ROS node, so it touches no network.

Per run, <out>/<bag name>/ holds README.md (what ran, the outcome, then a section each for the trajectory,
the EKF, the MPC, the drive, the fixes, the IMU and the simulation against the run, every figure with a
caption and its reasoning), metrics.json (every number in the README), figures/ (PNG and PDF) and data/
(the CSVs behind the figures). Three kinds of run:

  dock      a docking attempt: dock/stage left IDLE (dock_logger records one bag per attempt)
  bringup   recorded with the stack up but docking never armed (the first bags of 2026-10-13)
  drivecal  tools/drive_calibration_run.py: single-axis steps, fitted with drive_calibration_fit.py

Which configuration ran: each run is matched to the controller commit deployed at the time, from
controller_ws's git history (commit times are NZDT, deploys followed commits by minutes), confirmed by what
the bag itself shows (the APPROACH reference, the caps, the horizon, the static transforms) and by
replaying logged MPC ticks through each candidate commit's code (run_analytics_sim.py). The detector comes
from Conv-ChArT-Wireless-Inference's cfg/cfg.yaml history, or from detectors/live_detector.json where the
--detectors comparison identified it from the frames themselves.

Then <out>/INDEX.md (a row per run, the detector comparison and the fine-tuning records), aggregate.csv and
aggregate figures in <out>/figures/. <out>/README.md explains the pipeline and defines every metric.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import functools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

os.environ['ROS_AUTOMATIC_DISCOVERY_RANGE'] = 'OFF'     # belt and braces: nothing here joins a network
# A background batch on the laptop that runs inference during sessions: one thread for the numerics (set before
# numpy loads; the workers inherit it), and main() drops to the lowest CPU priority.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import numpy as np  # noqa: E402

TOOLS = Path(__file__).resolve().parent
CONTROLLER = TOOLS.parent                                   # controller_ws
P4P = CONTROLLER.parent                                     # ~/p4p, /workspace in p4p-gpu
WIRELESS = P4P / 'Conv-ChArT-Wireless-Inference'
NOTES = P4P / 'report' / 'project content' / '7_ros2_implementation'
DEFAULT_OUT = NOTES / 'run_records'
DEFAULT_BAGS = P4P / 'run_data' / 'pi_bags'
SIM_WORKER = TOOLS / 'run_analytics_sim.py'
VERSION = 2                                                 # bump to redo every record

STAGES = ('IDLE', 'ACQUIRE', 'APPROACH', 'FINAL', 'DOCKED', 'HOLD', 'ABORT')
IDLE, ACQUIRE, APPROACH, FINAL, DOCKED, HOLD, ABORT = range(7)
MOVING = (APPROACH, FINAL, HOLD)
DRIVE_STATES = ('NO_LINK', 'NO_TELEMETRY', 'DISARMED', 'STALLED', 'RUNNING')
CHI2_3 = {0.5: 2.366, 0.95: 7.815, 0.99: 11.345}           # chi-square, 3 dof: median, 95th, 99th
CMD_DELAY_S = 0.05          # cmd_vel to the wheels: the bridge and the Mega (7c item 50); in the simulation
DECK_FLASH_UTC = datetime(2026, 10, 13, 5, 27, tzinfo=timezone.utc).timestamp()   # per-wheel firmware (7c 56)
# The drive's channels drove the wrong wheels until a firmware fix between the 16:22 and 16:35 NZDT attempts (7c 30, 34)
CHANNEL_FIX_UTC = datetime(2026, 10, 13, 3, 30, tzinfo=timezone.utc).timestamp()
FOV_HALF_DEG = 42.0         # atan(320 / 356.7): the 640 px stream's half field of view
REANCHOR_RE = re.compile(r'was x ([-+\d.]+) y ([-+\d.]+) theta ([-+\d.]+) deg, now ([-+\d.]+) ([-+\d.]+) ([-+\d.]+)')

# Colours: the dataviz reference palette, categorical in fixed order; ink for text; status only for state.
C = ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, INK2, MUTED, GRID, AXIS = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
STATUS = {'good': '#0ca30c', 'warning': '#fab219', 'serious': '#ec835a', 'critical': '#d03b3b'}
STAGE_WASH = {ACQUIRE: '#f0efec', APPROACH: '#e6e5e0', FINAL: '#d9d8d1', HOLD: STATUS['warning'],
              DOCKED: STATUS['good'], ABORT: STATUS['critical']}


# ============================================================================================ utilities
def nz(t: float) -> datetime:
    """Return the epoch time t in New Zealand time (NZDT in October)."""
    from zoneinfo import ZoneInfo
    return datetime.fromtimestamp(t, ZoneInfo('Pacific/Auckland'))


def utc(t: float) -> datetime:
    return datetime.fromtimestamp(t, timezone.utc)


def fin(v, nd: int = 4):
    """A JSON-safe float: rounded, None for NaN or inf."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if math.isfinite(v) else None


def pct(a, q):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if len(a) else math.nan


def rms(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a * a))) if len(a) else math.nan


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2.0 * np.pi) - np.pi


def fmt(v, spec: str = '.3f', unit: str = '', none: str = 'n/a') -> str:
    """Format a number for prose; 'n/a' for missing."""
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return none
    return f'{v:{spec}}{unit}'


def mode(a, nd: int = 3):
    a = np.round(np.asarray(a, float), nd)
    if not len(a):
        return math.nan
    vals, counts = np.unique(a, return_counts=True)
    return float(vals[np.argmax(counts)])


def write_csv(path: Path, header: list, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(['' if (isinstance(v, float) and not math.isfinite(v)) else
                        (f'{v:.6g}' if isinstance(v, float) else v) for v in r])


def git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True,
                          text=True).stdout


# ============================================================================================ reading
@dataclass
class Bag:
    """Everything the analysis reads from one bag, as arrays in the Pi's clock [s]."""

    name: str
    path: Path
    start: float = math.nan             # first message received
    end: float = math.nan
    closed: bool = True                 # metadata.yaml present (the recorder closed it)
    counts: dict = field(default_factory=dict)
    stage: np.ndarray = None            # t_recv, stage, y_ref x y th, v_max, r_max, enabled
    est: np.ndarray = None              # t_recv, t_stamp, x(5) 2:7, p_diag(5) 7:12, r_gyro 12, imu_rate 13, fix_age 14, nis 15, accepted 16
    mpc: np.ndarray = None              # t_recv, t_stamp, u(3), slack, exit_flag, solve_ms, fallback
    y_pred: list = field(default_factory=list)
    fix: np.ndarray = None              # t_recv, t_stamp, x, y, th, cov(9) 5:14, latency 14, cov_valid 15, roll, pitch, range 18, ambiguous 19
    inf: np.ndarray = None              # t_recv, t_stamp, valid, ambiguous, rms, num_used, t(3), q(4), rms_alt
    reasons: list = field(default_factory=list)
    cmd: np.ndarray = None              # t_recv, u_b, v_b, r_b
    tel: np.ndarray = None              # t_recv, t_stamp, applied(3) 2:5, requested(3) 5:8, armed 8, timed_out 9, saturated 10, flags 11, pulse(4) 12:16, yaw_rate 16
    status: np.ndarray = None           # t_recv, t_stamp, state, cmd_age
    imu: np.ndarray = None              # t_recv, t_stamp, ax, ay, az, gx, gy, gz
    image: np.ndarray = None            # t_recv of each camera frame (not decoded)
    rosout: list = field(default_factory=list)      # (t_recv, node, level, text)
    tf_static: dict = field(default_factory=dict)   # child -> (parent, (x, y, z))


SHAPES = {'stage': 8, 'est': 17, 'mpc': 9, 'fix': 20, 'inf': 17, 'cmd': 4, 'tel': 17, 'status': 4, 'imu': 8}
SUFFIX = {'dock/stage': 'stage', 'dock/estimate': 'est', 'dock/mpc_info': 'mpc', 'fiducial/fix': 'fix',
          'inference_result': 'inf', 'cmd_vel': 'cmd', 'drive/telemetry': 'tel', 'drive/status': 'status',
          'imu/data': 'imu', 'rosout': 'rosout', 'tf_static': 'tf_static', 'image': 'image'}


def bag_storage(path: Path):
    """Return (uri, storage_id, closed) for a bag folder, or None if it is not one."""
    if (path / 'metadata.yaml').exists():
        return str(path), '', True
    for ext, sid in (('*.mcap', 'mcap'), ('*.db3', 'sqlite3')):
        files = sorted(path.glob(ext))
        if files:                           # a recorder killed before it wrote metadata.yaml
            return str(files[0]), sid, False
    return None


def bag_fingerprint(path: Path) -> dict:
    """The bag folder's files and sizes: a bag copied while still recording changes it when copied again."""
    return {f.name: f.stat().st_size for f in sorted(path.iterdir()) if f.is_file()}


def find_bags(target: Path) -> list[Path]:
    if bag_storage(target):
        return [target]
    return sorted(p for p in target.iterdir() if p.is_dir() and bag_storage(p))


def _stamp(m) -> float:
    return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9


def old_telemetry(data: bytes):
    """
    Decode a DriveTelemetry of the layout before 876cd4a (2026-10-13 18:02 NZDT): no requested, no pulse_us.

    Returns the row read_bag keeps (requested NaN, pulses 0), or None if the bytes do not fit that layout. Plain
    little-endian CDR after the 4-byte encapsulation header, each field aligned to its size.
    """
    import struct
    buf = bytes(data)[4:]
    o = 0

    def align(n):
        nonlocal o
        o = (o + n - 1) // n * n
    try:
        sec, nsec = struct.unpack_from('<iI', buf, o)
        o += 8
        (ln,) = struct.unpack_from('<I', buf, o)
        o += 4 + ln
        align(4)
        o += 4                                                  # mega_t_ms
        align(8)
        _, yaw_rate = struct.unpack_from('<dd', buf, o)
        o += 16
        lx, ly, _, _, _, az = struct.unpack_from('<6d', buf, o)
        o += 48
        armed, timed_out, saturated, flags = struct.unpack_from('<4B', buf, o)
        o += 4
        align(2)
        struct.unpack_from('<HH', buf, o)
        if o + 4 != len(buf):
            return None
    except struct.error:
        return None
    return (sec + nsec * 1e-9, lx, ly, az, math.nan, math.nan, math.nan, float(armed), float(timed_out),
            float(saturated), flags, 0, 0, 0, 0, yaw_rate)


def read_bag(path: Path) -> Bag:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    uri, sid, closed = bag_storage(path)
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id=sid), rosbag2_py.ConverterOptions('', ''))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    which = {}
    for name in types:
        bare = name.lstrip('/')
        for suffix, key in SUFFIX.items():
            if bare == suffix or bare.endswith('/' + suffix):
                if key not in which.values() or bare == suffix:
                    which = {n: k for n, k in which.items() if k != key}
                    which[name] = key
    cls = {}
    for name, key in which.items():
        try:
            cls[name] = get_message(types[name])
        except Exception:
            print(f'  {name}: no definition for {types[name]} installed; skipped')
    b = Bag(name=path.name, path=path, closed=closed)
    rows = {k: [] for k in SHAPES}
    images, unreadable = [], 0
    first = last = None
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        t = t_ns * 1e-9
        first = t if first is None else min(first, t)
        last = t if last is None else max(last, t)
        key = which.get(topic)
        b.counts[topic] = b.counts.get(topic, 0) + 1
        if key is None or topic not in cls:
            continue
        if key == 'image':
            images.append(t)
            continue
        try:
            m = deserialize_message(data, cls[topic])
        except Exception:
            row = old_telemetry(data) if key == 'tel' else None
            if row is None:
                unreadable += 1
            else:
                rows[key].append((t, *row))
                b.counts['_old_telemetry'] = b.counts.get('_old_telemetry', 0) + 1
            continue
        if key == 'stage':
            rows[key].append((t, m.stage, *m.y_ref, m.v_max, m.r_max, float(m.command_enabled)))
        elif key == 'est':
            rows[key].append((t, _stamp(m), *m.x, *m.p_diag, m.r_gyro, m.imu_rate, m.fix_age, m.last_nis,
                              float(m.last_fix_accepted)))
        elif key == 'mpc':
            rows[key].append((t, _stamp(m), *m.u, m.slack, m.exit_flag, m.solve_ms, float(m.fallback)))
            b.y_pred.append(np.asarray(m.y_pred, float))
        elif key == 'fix':
            rows[key].append((t, _stamp(m), m.x, m.y, m.theta, *m.covariance, t - _stamp(m), float(m.covariance_valid),
                              m.roll, m.pitch, m.range, float(m.ambiguous)))
        elif key == 'inf':
            p, q = m.pose.pose.position, m.pose.pose.orientation
            rows[key].append((t, _stamp(m), float(not m.reason), float(m.ambiguous), m.rms, m.num_used,
                              p.x, p.y, p.z, q.x, q.y, q.z, q.w, m.rms_alt, float(m.covariance_valid),
                              float(m.covariance_valid_alt), math.sqrt(max(m.pose.covariance[0], 0.0))))
            b.reasons.append(m.reason)
        elif key == 'cmd':
            rows[key].append((t, m.linear.x, m.linear.y, m.angular.z))
        elif key == 'tel':
            pulse = tuple(getattr(m, 'pulse_us', (0, 0, 0, 0)))
            req = getattr(m, 'requested', None)
            rq = (req.linear.x, req.linear.y, req.angular.z) if req is not None else (math.nan,) * 3
            rows[key].append((t, _stamp(m), m.applied.linear.x, m.applied.linear.y, m.applied.angular.z, *rq,
                              float(m.armed), float(m.timed_out), float(m.saturated), m.flags, *pulse,
                              m.yaw_rate))
        elif key == 'status':
            rows[key].append((t, _stamp(m), m.state, m.cmd_age))
        elif key == 'imu':
            a, w = m.linear_acceleration, m.angular_velocity
            rows[key].append((t, _stamp(m), a.x, a.y, a.z, w.x, w.y, w.z))
        elif key == 'rosout':
            b.rosout.append((t, m.name, m.level, m.msg))
        elif key == 'tf_static':
            for tr in m.transforms:
                v = tr.transform.translation
                b.tf_static[tr.child_frame_id] = (tr.header.frame_id, (v.x, v.y, v.z))
    if unreadable:
        print(f'  {unreadable} messages did not deserialise with the installed definitions')
    for k, shape in SHAPES.items():
        a = np.array(rows[k], float).reshape(-1, shape)
        setattr(b, k, a)
    order = np.argsort(b.mpc[:, 0], kind='stable')
    b.mpc = b.mpc[order]
    b.y_pred = [b.y_pred[i] for i in order]
    for k in ('stage', 'est', 'fix', 'cmd', 'tel', 'status', 'imu'):
        a = getattr(b, k)
        setattr(b, k, a[np.argsort(a[:, 0], kind='stable')])
    order = np.argsort(b.inf[:, 0], kind='stable')
    b.inf, b.reasons = b.inf[order], [b.reasons[i] for i in order]
    b.image = np.array(sorted(images))
    b.start, b.end = (first or math.nan), (last or math.nan)
    return b


# ============================================================================================ timeline
@dataclass
class Timeline:
    t0: float                           # arming (the first stage out of IDLE), else the bag's start
    armed: bool
    segments: list                      # (stage, t_start, t_end)
    t_end: float                        # the last armed moment
    t_cmd: float = math.nan             # where commanding ended: DOCKED, ABORT or the disarm that ended the attempt
    approach: float = math.nan          # first APPROACH entry
    docked: float = math.nan
    outcome: str = ''
    abort_cause: str = ''


def stage_at(b: Bag, t: np.ndarray) -> np.ndarray:
    """The stage published last at or before each time (IDLE before the first)."""
    if not len(b.stage):
        return np.zeros(len(np.atleast_1d(t)), int)
    i = np.searchsorted(b.stage[:, 0], np.atleast_1d(t), side='right') - 1
    s = b.stage[np.clip(i, 0, None), 1].astype(int)
    s[i < 0] = IDLE
    return s


def timeline(b: Bag) -> Timeline:
    st = b.stage
    armed_rows = np.nonzero(st[:, 1] != IDLE)[0] if len(st) else np.array([], int)
    if not len(armed_rows):
        return Timeline(t0=b.start, armed=False, segments=[], t_end=b.end, outcome='never armed')
    t0 = st[armed_rows[0], 0]
    segs, cur, ts = [], int(st[armed_rows[0], 1]), t0
    for t, s in st[armed_rows[0]:, :2]:
        if int(s) != cur:
            segs.append((cur, ts, t))
            cur, ts = int(s), t
    segs.append((cur, ts, st[-1, 0]))
    segs = [s for s in segs if not (s[0] == IDLE and s is segs[-1])]
    t_end = st[armed_rows[-1], 0]
    tl = Timeline(t0=t0, armed=True, segments=segs, t_end=t_end)
    app = [s for s in segs if s[0] == APPROACH]
    tl.approach = app[0][1] if app else math.nan
    dk = [s for s in segs if s[0] == DOCKED]
    tl.docked = dk[0][1] if dk else math.nan
    moving = [s for s in segs if s[0] in MOVING]
    tl.t_cmd = moving[-1][2] if moving else t_end
    last = [s for s in segs if s[0] != IDLE][-1][0]
    if math.isfinite(tl.docked):
        tl.outcome = 'DOCKED'
    elif last == ABORT:
        tl.outcome = 'ABORT'
    else:
        tl.outcome = f'disarmed in {STAGES[last]}'
    return tl


def abort_cause(b: Bag, segs: list, cfg: dict) -> str:
    """Which of the supervisor's abort rules fired: HOLD timeout, drive, stale estimate, variance, or the old corridor."""
    i = next(k for k, s in enumerate(segs) if s[0] == ABORT)
    t = segs[i][1]
    prev = segs[i - 1] if i else None
    causes = []
    if corridor_abort(cfg['sha']) and len(b.est):
        k = np.searchsorted(b.est[:, 0], t) - 1
        lim = cfg['settings']['th_max_deg']
        if k >= 0 and abs(math.degrees(b.est[k, 4])) > lim:
            causes.append(f'heading {math.degrees(b.est[k, 4]):+.0f} deg, outside the +-{lim:g} deg corridor (this code '
                          'aborted after 0.5 s outside it; the rule was removed later, 7c item 35)')
    if prev is not None and prev[0] == HOLD and prev[2] - prev[1] >= 1.9:
        causes.append(f'no accepted fix for {prev[2] - prev[1]:.1f} s in HOLD (abort after 2 s)')
    if len(b.est):
        k = np.searchsorted(b.est[:, 0], t) - 1
        if k >= 0:
            p = b.est[k, 7:9]
            if max(p) > 0.05 ** 2:
                causes.append(f'position variance {math.sqrt(max(p)) * 1000:.0f} mm (sd) past the 50 mm limit')
            if t - b.est[k, 1] > 0.1:
                causes.append(f'estimate {t - b.est[k, 1]:.2f} s old (stale past 0.1 s)')
    if len(b.status):
        k = np.searchsorted(b.status[:, 0], t) - 1
        if k >= 0 and int(b.status[k, 2]) not in (3, 4):
            causes.append(f'drive {DRIVE_STATES[int(b.status[k, 2])]}')
    return '; '.join(causes) or 'not identified from the bag'


# ============================================================================================ configuration
@dataclass
class Commit:
    sha: str
    short: str
    t: float
    subject: str


@functools.lru_cache(maxsize=None)
def controller_commits() -> tuple:
    out = git(CONTROLLER, 'log', '--format=%H%x09%h%x09%ct%x09%s', '--', 'dock_control/')
    cs = [Commit(h, s, float(t), subj) for h, s, t, subj in (ln.split('\t', 3) for ln in out.splitlines())]
    return tuple(sorted(cs, key=lambda c: c.t))


@functools.lru_cache(maxsize=None)
def config_at(sha: str) -> dict:
    import yaml
    return yaml.safe_load(git(CONTROLLER, 'show', f'{sha}:dock_control/config/dock_demo.yaml'))['dock']


@functools.lru_cache(maxsize=None)
def cap_type(sha: str) -> str:
    """
    The speed cap's shape at sha: 'L1' (|u_b| + |v_b| <= v_max, the brief's), 'per-axis' (|u_b|, |v_b| <= v_max
    each; 3c397a6), or 'per-axis in APPROACH, L1 in FINAL and HOLD' (the supervisor picks per stage; ef5e42e).
    """
    sup = git(CONTROLLER, 'show', f'{sha}:dock_control/dock_control/supervisor.py')
    if 'l1_cap' in sup:
        return 'per-axis in APPROACH, L1 in FINAL and HOLD'
    text = git(CONTROLLER, 'show', f'{sha}:dock_control/dock_control/mpc.py')
    return 'per-axis' if re.search(r'A_AMP = np\.array\(\[\[1\.0, 0\.0, 0\.0\]', text) else 'L1'


@functools.lru_cache(maxsize=None)
def corridor_abort(sha: str) -> bool:
    """Whether that commit's supervisor aborts on the heading corridor (before 1cfa8f2)."""
    return 'off_corridor_since' in git(CONTROLLER, 'show', f'{sha}:dock_control/dock_control/supervisor.py')


@functools.lru_cache(maxsize=None)
def code_dir(sha: str) -> str:
    """Write dock_control as it was at sha into a temporary folder; return the folder."""
    root = Path(tempfile.gettempdir()) / 'run_analytics_code' / sha[:12]
    pkg = root / 'dock_control'
    if not (pkg / '.complete').exists():
        pkg.mkdir(parents=True, exist_ok=True)
        for name in git(CONTROLLER, 'ls-tree', '--name-only', sha, 'dock_control/dock_control/').split():
            if name.endswith('.py'):
                (pkg / Path(name).name).write_text(git(CONTROLLER, 'show', f'{sha}:{name}'))
        (pkg / '.complete').write_text(sha)
    return str(root)


def observed_settings(b: Bag) -> dict:
    """The settings a bag shows by itself: the APPROACH reference and caps, FINAL's floor, the horizon, the TFs."""
    o = {}
    st = b.stage
    app = st[st[:, 1] == APPROACH] if len(st) else st
    if len(app):
        o['standoff_m'] = -mode(app[:, 2])
        o['v_approach'] = mode(app[:, 5])
        o['r_max'] = mode(app[:, 6])
    fin_ = st[np.isin(st[:, 1], (FINAL, HOLD))] if len(st) else st
    if len(fin_):
        lo = float(np.round(fin_[:, 5].min(), 3))
        if np.sum(np.isclose(fin_[:, 5], lo, atol=5e-4)) >= 5:        # it sat at the floor: v_final
            o['v_final'] = lo
    if b.y_pred:
        o['Np'] = int(mode([len(y) // 3 for y in b.y_pred], 0))
    if 'marker' in b.tf_static:
        o['marker_x_m'] = round(b.tf_static['marker'][1][0], 4)
    if 'pi_camera' in b.tf_static:
        o['cam_x_m'] = round(b.tf_static['pi_camera'][1][0], 4)
    return o


def expected_settings(p: dict) -> dict:
    """The same quantities as observed_settings, as a config file sets them."""
    return {'standoff_m': p['sup']['standoff'], 'v_approach': p['lim']['v_approach'], 'r_max': p['lim']['r_max'],
            'v_final': p['lim']['v_final'], 'Np': p['mpc']['Np'], 'marker_x_m': p['geom']['marker_m'][0],
            'cam_x_m': p['geom']['cam_xyz'][0]}


def mismatches(obs: dict, p: dict) -> list:
    exp = expected_settings(p)
    out = []
    for k, v in obs.items():
        if k in exp and abs(float(exp[k]) - float(v)) > 1e-3 * max(1.0, abs(float(v))):
            out.append(f'{k} {v:g} (config {exp[k]:g})')
    return out


def key_settings(p: dict, sha: str) -> dict:
    m, lim, sup, g = p['mpc'], p['lim'], p['sup'], p['geom']
    return {'standoff_m': sup['standoff'], 'v_approach': lim['v_approach'], 'v_final': lim['v_final'],
            'r_max': lim['r_max'], 'w_max': lim['w_max'], 'a_max': lim['a_max'], 'Np': m['Np'], 'Nc': m['Nc'],
            'rw': m['rw'], 'qw': m.get('qw', 1.0), 'qw_final': m.get('qw_final', m.get('qw', 1.0)),
            'Qy': [round(float(v)) for v in m['Qy']], 'speed_cap': cap_type(sha), 'th_max_deg': m['th_max_deg'],
            'marker_m': g['marker_m'], 'board_square_m': g.get('board_square_m'),
            'align_tol': [sup['align_tol_m'], sup['align_tol_deg']], 'dock_tol': sup['dock_tol'],
            'gate': p['fid']['gate'], 'imu_rate': p['imu']['rate']}


def pre_tuning(p: dict) -> dict:
    """p with the MPC weights from before the robot tuning (7c item 47): rw 1, lateral weight from tol.y."""
    q = json.loads(json.dumps(p))
    q['mpc']['rw'] = 1.0
    q['mpc']['Qy'] = [q['tol']['y'] ** -2, q['tol']['y'] ** -2, q['mpc']['Qy'][2]]
    q['mpc'].pop('qw', None)
    q['mpc'].pop('qw_final', None)
    return q


def run_worker(mode_: str, sha: str, req: dict, timeout: float = 600.0) -> dict | None:
    with tempfile.TemporaryDirectory() as d:
        rq, rs = Path(d) / 'req.json', Path(d) / 'res.json'
        rq.write_text(json.dumps(req))
        p = subprocess.run([sys.executable, str(SIM_WORKER), mode_, code_dir(sha), str(rq), str(rs)],
                           capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0 or not rs.exists():
            print(f'  {mode_} with {sha[:7]} failed: {p.stderr.strip()[-400:]}')
            return None
        return json.loads(rs.read_text())


def replay_ticks(b: Bag, n_max: int = 160) -> list:
    """Logged MPC ticks with everything their solve read, for run_analytics_sim.replay (evenly sampled)."""
    if not len(b.mpc) or not len(b.est):
        return []
    est_by_stamp = {round(s, 6): i for i, s in enumerate(b.est[:, 1])}
    st = b.stage
    ticks, prev_u, prev_t = [], np.zeros(3), -math.inf
    resume = APPROACH
    for i in range(len(b.mpc)):
        t, ts = b.mpc[i, 0], b.mpc[i, 1]
        k = np.searchsorted(st[:, 0], t + 0.002, side='right') - 1
        u_prev = prev_u if t - prev_t < 0.08 else np.zeros(3)
        prev_u, prev_t = b.mpc[i, 2:5].copy(), t
        if k < 0:
            continue
        s = int(st[k, 1])
        if s in (APPROACH, FINAL):
            resume = s
        j = est_by_stamp.get(round(ts, 6))
        if j is None or s not in MOVING or len(b.y_pred[i]) < 3:
            continue
        ticks.append({'x': b.est[j, 2:7].tolist(), 'r_gyro': float(b.est[j, 12]), 'u_prev': u_prev.tolist(),
                      'y_ref': st[k, 2:5].tolist(), 'v_max': float(st[k, 5]), 'r_max': float(st[k, 6]),
                      'stage': STAGES[s], 'resume': STAGES[resume], 'y_pred': b.y_pred[i].tolist(),
                      'u': b.mpc[i, 2:5].tolist(), 't': float(t)})
    if len(ticks) > n_max:
        ticks = [ticks[int(i)] for i in np.linspace(0, len(ticks) - 1, n_max)]
    return ticks


def deployed_config(b: Bag, tl: Timeline, replay: bool = True) -> dict:
    """Match the run to the controller commit that ran, by time, by the bag's own settings and by replay."""
    t = tl.t0
    commits = controller_commits()
    before = [c for c in commits if c.t <= t]
    prev = before[-1] if before else commits[0]
    window = [c for c in commits if t < c.t <= t + 15 * 60]          # deployed before it was committed?
    cands = [(prev.short, prev, None)] + [(c.short, c, None) for c in window]
    after = [c for c in commits if c.t > t]
    obs = observed_settings(b)
    if after and t - prev.t > 6 * 3600:                                # a working tree between commits
        first_after = after[0]
        if first_after not in [c for _, c, _ in cands]:
            cands.append((first_after.short, first_after, None))
        cands.append((f'{first_after.short} with the pre-tuning MPC weights', first_after, 'pre_tuning'))
    ticks = replay_ticks(b) if replay else []
    results = []
    for label, c, variant in cands:
        p = config_at(c.sha)
        if variant == 'pre_tuning':
            p = pre_tuning(p)
        mm = mismatches(obs, p)
        rep = None
        if ticks:
            r = run_worker('replay', c.sha, {'config': p, 'ticks': ticks})
            if r:
                d = np.array(r['plan_diff_m'], float)
                rep = {'ticks': len(d), 'plan_median_mm': fin(np.median(d) * 1000, 1),
                       'plan_p95_mm': fin(pct(d, 95) * 1000, 1),
                       'u_median_mm_s': fin(np.median(r['u_diff']) * 1000, 2)}
        results.append({'label': label, 'commit': c.short, 'sha': c.sha, 'variant': variant,
                        'commit_nzdt': nz(c.t).strftime('%Y-%m-%d %H:%M:%S'), 'subject': c.subject,
                        'minutes_before_run': round((t - c.t) / 60.0, 1), 'mismatches': mm, 'replay': rep,
                        'params': p})

    def score(r):
        rep = r['replay']
        ok = rep and rep['plan_median_mm'] is not None
        return (len(r['mismatches']), round(rep['plan_median_mm'], 1) if ok else 1e9,
                round(rep['plan_p95_mm'] * 2) / 2 if ok else 1e9, 0 if r['commit'] == prev.short and not r['variant'] else 1)
    best = min(results, key=score)
    out = {k: v for k, v in best.items() if k != 'params'}
    out['observed'] = obs
    out['settings'] = key_settings(best['params'], best['sha'])
    out['by_time'] = prev.short
    out['candidates'] = [{k: v for k, v in r.items() if k not in ('params', 'sha')} for r in results]
    rep = best['replay']
    exact = rep is not None and rep['plan_median_mm'] is not None and rep['plan_median_mm'] < 5
    rtxt = (f"the logged MPC ticks replay through its code to {rep['plan_median_mm']:.1f} mm median "
            f"(p95 {rep['plan_p95_mm']} mm)") if rep and rep['plan_median_mm'] is not None else ''
    if best['variant']:
        out['certainty'] = ('reconstructed' + (' and confirmed' if exact else '') + f": no commit matches; {best['commit']}'s "
                            'code with the MPC weights from before the robot tuning (rw 1, lateral weight from tol.y; 7c '
                            f"item 47), a working tree between {prev.short} and {best['commit']}" + (f'; {rtxt}' if rtxt else ''))
    elif best['mismatches'] and exact:
        out['certainty'] = (f"MPC confirmed, rest approximate: {rtxt}, but the bag shows " + ', '.join(best['mismatches'])
                            + ': the deployed working tree had those changes before they were committed')
    elif best['mismatches']:
        out['certainty'] = 'approximate: the bag disagrees with the config on ' + ', '.join(best['mismatches']) + \
                           (f'; {rtxt}' if rtxt else '')
    elif exact:
        out['certainty'] = f'confirmed: every setting the bag shows matches, and {rtxt}'
    elif rep and rep['plan_median_mm'] is not None:
        out['certainty'] = f'likely: the settings the bag shows match, but {rtxt}'
    elif best['commit'] != prev.short and best['minutes_before_run'] < -15:
        out['certainty'] = (f"by the bag's static transforms only (no MPC ticks): the nearest commit whose geometry matches; "
                            f"the deployed code was a working tree between {prev.short} and {best['commit']}")
    else:
        out['certainty'] = 'by time only (no MPC ticks to replay)'
    out['_params'] = best['params']
    return out


@functools.lru_cache(maxsize=None)
def detector_history() -> tuple:
    """(commit time, short, detector file, square_m, fx) for each cfg/cfg.yaml commit, oldest first."""
    import yaml
    out = git(WIRELESS, 'log', '--format=%H%x09%h%x09%ct', '--', 'cfg/cfg.yaml')
    rows = []
    for h, s, t in (ln.split('\t') for ln in out.splitlines()):
        c = yaml.safe_load(git(WIRELESS, 'show', f'{h}:cfg/cfg.yaml')) or {}
        K = (c.get('CAMERA') or {}).get('K')
        rows.append((float(t), s, (c.get('MODEL') or {}).get('detector', ''),
                     (c.get('BOARD') or {}).get('square_length_m'), K[0][0] if K else None))
    return tuple(sorted(rows))


DETECTOR_NAMES = {'detector_882k.onnx': 'release', 'detector_882k_ssl_20261013_r3.onnx': 'SSL live r3',
                  'detector_882k_ssl_20261014_offline_r1.onnx': 'SSL offline r1'}


def deployed_detector(t: float, live: dict | None = None, name: str = '') -> dict:
    hist = detector_history()
    cur = [h for h in hist if h[0] <= t]
    h = cur[-1] if cur else hist[0]
    file = Path(h[2]).name
    out = {'name': DETECTOR_NAMES.get(file, file), 'file': h[2], 'cfg_commit': h[1],
           'source': f'cfg/cfg.yaml at {h[1]} ({nz(h[0]):%Y-%m-%d %H:%M} NZDT)', 'notes': []}
    nxt = [x for x in hist if x[0] > t]
    if nxt and Path(nxt[0][2]).name != file:
        ck = WIRELESS / nxt[0][2]
        if ck.exists() and h[0] < ck.stat().st_mtime < t:            # a new checkpoint, there before the run
            out['notes'].append(f'possibly already {DETECTOR_NAMES.get(ck.name, ck.name)}: its checkpoint was '
                                f'written {nz(ck.stat().st_mtime):%H:%M} NZDT, before this run, and committed '
                                f'to cfg.yaml only at {nz(nxt[0][0]):%H:%M} ({nxt[0][1]})')
    if live and name in live:
        ident = live[name]
        out['identified'] = ident
        if ident.get('detector') and ident['detector'] != out['name']:
            out['notes'].append(f"the frames say {ident['detector']} ({ident['evidence']}), not the "
                                f"cfg.yaml history's {out['name']}")
            out['name'] = ident['detector']
            out['source'] = 'identified from the bag\'s own frames (detectors/live_detector.json)'
        elif ident.get('detector'):
            out['source'] += '; confirmed from the bag\'s own frames'
    elif live:
        idents = sorted((v['t'], v['detector'], k) for k, v in live.items() if v.get('detector'))
        same_era = [x for x in idents if x[0] <= t and not [h for h in hist if x[0] < h[0] <= t]]
        if same_era and same_era[-1][1] != out['name']:
            out['notes'].append(f'the frames of {same_era[-1][2]} (same cfg.yaml era, earlier) show '
                                f'{same_era[-1][1]} was live')
            out['name'] = same_era[-1][1]
            out['source'] = f'identified from {same_era[-1][2]}\'s frames (same cfg.yaml era)'
    return out


def board_square(t: float) -> float:
    """The board's square at time t: cfg.yaml's BOARD.square_length_m; 24 mm before it was committed."""
    cur = [h for h in detector_history() if h[0] <= t]
    sq = cur[-1][3] if cur else None
    return float(sq) if sq else 0.024


# ============================================================================================ signals
def held(t_cmd: np.ndarray, v: np.ndarray, t: np.ndarray, timeout: float = 0.1) -> np.ndarray:
    """The command in force at each t: the last one received within timeout (the bridge's cmd_timeout), else 0."""
    if not len(t_cmd):
        return np.zeros((len(t),) + v.shape[1:])
    i = np.searchsorted(t_cmd, t, side='right') - 1
    out = v[np.clip(i, 0, None)].copy()
    stale = (i < 0) | (t - t_cmd[np.clip(i, 0, None)] > timeout)
    out[stale] = 0.0
    return out


def window_fit(t: np.ndarray, y: np.ndarray, tg: np.ndarray, half: float, min_n: int = 4):
    """Slope and value of a straight line through (t, y) within +-half of each tg; NaN where too few points."""
    i0 = np.searchsorted(t, tg - half)
    i1 = np.searchsorted(t, tg + half, side='right')
    slope = np.full((len(tg),) + y.shape[1:], np.nan)
    value = np.full_like(slope, np.nan)
    for k in range(len(tg)):
        a, b_ = i0[k], i1[k]
        if b_ - a < min_n or t[b_ - 1] - t[a] < 0.6 * half:
            continue
        dt = t[a:b_] - tg[k]
        dm = dt.mean()
        ym = y[a:b_].mean(axis=0)
        sxx = ((dt - dm) ** 2).sum()
        s = ((dt - dm)[:, None] * (y[a:b_] - ym).reshape(len(dt), -1)).sum(axis=0) / sxx
        slope[k] = s.reshape(y.shape[1:])
        value[k] = (ym - s.reshape(ym.shape) * dm)
    return slope, value


def camera_track(b: Bag, t_lo: float, t_hi: float, dt: float = 0.05, half: float = 0.2, ambiguous: bool = False):
    """
    The robot's motion from the camera alone, on a grid: velocity in the target frame, in the body frame, and
    the heading, from straight-line fits of the fixes over 2 x half (mpc_analysis vel_truth.py). Unambiguous
    fixes only, unless ambiguous (then as vel_truth.py: every fix, flips included).
    """
    f = b.fix
    keep = (((f[:, 19] == 0) | ambiguous) & (f[:, 1] >= t_lo - 1.0) & (f[:, 1] <= t_hi + 1.0)) if len(f) \
        else np.array([], bool)
    f = f[keep]
    tg = np.arange(t_lo, t_hi, dt)
    if len(f) < 6 or not len(tg):
        return None
    order = np.argsort(f[:, 1])
    f = f[order]
    th = np.unwrap(f[:, 4])
    slope, value = window_fit(f[:, 1], np.c_[f[:, 2], f[:, 3], th], tg, half)
    vx, vy, r = slope[:, 0], slope[:, 1], slope[:, 2]
    x, y, h = value[:, 0], value[:, 1], value[:, 2]
    c, s = np.cos(h), np.sin(h)
    return {'t': tg, 'x': x, 'y': y, 'th': wrap(h), 'vx': vx, 'vy': vy, 'r': r,
            'u': c * vx + s * vy, 'v': -s * vx + c * vy, 'n_fix': len(f)}


def fit_lag_gain(t: np.ndarray, src_t: np.ndarray, src: np.ndarray, out: np.ndarray, max_lag: float = 0.6,
                 min_range: float = 0.02) -> dict:
    """
    out(t) ~ gain * src(t - lag) + offset, the lag searched in 10 ms steps; least squares at each lag.

    Returns gain, lag, offset, the residual RMS, the samples and the source's 95th-percentile magnitude.
    """
    ok = np.isfinite(out)
    res = {'gain': math.nan, 'lag_s': math.nan, 'offset': math.nan, 'resid_rms': math.nan, 'n': int(ok.sum()),
           'src_p95': math.nan, 'r2': math.nan}
    if ok.sum() < 30:
        res['note'] = f'too few samples to fit ({int(ok.sum())})'
        return res
    tt, yy = t[ok], out[ok]
    best = None
    for L in np.arange(0.0, max_lag + 1e-9, 0.01):
        c = np.interp(tt - L, src_t, src)
        if np.ptp(c) < 1e-6:
            continue
        A = np.c_[c, np.ones_like(c)]
        coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
        rss = float(np.sum((yy - A @ coef) ** 2))
        if best is None or rss < best[0]:
            best = (rss, L, coef, c)
    if best is None:
        res['note'] = 'the command never changed: nothing to fit'
        return res
    rss, L, coef, c = best
    res.update(gain=float(coef[0]), lag_s=float(L), offset=float(coef[1]), resid_rms=math.sqrt(rss / len(yy)),
               src_p95=pct(np.abs(c), 95), r2=1.0 - rss / max(float(np.sum((yy - yy.mean()) ** 2)), 1e-12))
    if res['src_p95'] < min_range:
        res['note'] = 'too little motion on this axis to fit'
    elif res['r2'] < 0.3:
        res['note'] = f"no usable fit (R^2 {res['r2']:.2f})"
    else:                                   # usable: say so when the drive did something odd
        flags = []
        if res['gain'] < 0:
            flags.append('moved against the command')
        elif res['gain'] < 0.3:
            flags.append('barely responded')
        if res['n'] < 40:
            flags.append(f"short: {res['n']} samples")
        if flags:
            res['flag'] = '; '.join(flags)
    return res


# ============================================================================================ analysis
@dataclass
class Fig:
    name: str
    title: str
    what: str
    how: str
    means: str
    caveats: str
    section: str


def analyse_imu(b: Bag, tl: Timeline) -> dict:
    m = {'samples': int(len(b.imu))}
    if len(b.imu) < 10:
        return m
    ts = b.imu[:, 1]
    d = np.diff(np.sort(ts))
    span = ts.max() - ts.min()
    m.update(rate_hz=fin(len(ts) / span if span > 0 else math.nan, 1), dt_median_ms=fin(np.median(d) * 1e3, 2),
             dt_p99_ms=fin(pct(d, 99) * 1e3, 2), dt_max_ms=fin(d.max() * 1e3, 1),
             gaps_over_20ms=int(np.sum(d > 0.020)), repeated_stamps=int(np.sum(d <= 0)),
             recv_delay_median_ms=fin(np.median(b.imu[:, 0] - ts) * 1e3, 2),
             recv_delay_p99_ms=fin(pct(b.imu[:, 0] - ts, 99) * 1e3, 2))
    sec = np.floor(ts - ts.min())
    per_s = np.bincount(sec.astype(int))[:-1] if len(sec) else np.array([])
    if len(per_s):
        m.update(rate_1s_min=int(per_s.min()), rate_1s_p5=fin(pct(per_s, 5), 0), rate_1s_max=int(per_s.max()))
    if len(b.est):
        r = b.est[:, 13]
        r = r[r > 0]
        if len(r):
            m['ekf_imu_rate_median'] = fin(np.median(r), 1)
            m['ekf_imu_rate_min'] = fin(r.min(), 1)
    m['low_rate_warnings'] = sum(1 for _, n, _, s in b.rosout if n == 'ekf_node' and s.startswith('IMU at'))
    return m


def analyse_fixes(b: Bag, tl: Timeline) -> dict:
    """Fix and pose statistics: rate, latency, ambiguity, RMS and corners against range, jitter when still."""
    m = {'inference_results': int(len(b.inf)), 'fixes': int(len(b.fix)), 'images': int(len(b.image))}
    if len(b.inf):
        inf = b.inf
        valid = inf[:, 2] == 1
        span = inf[-1, 0] - inf[0, 0]
        reasons = {}
        for r in b.reasons:
            if r:
                reasons[r] = reasons.get(r, 0) + 1
        rng = np.linalg.norm(inf[:, 6:9], axis=1)
        lat = inf[:, 0] - inf[:, 1]
        m.update(result_rate_hz=fin(len(inf) / span if span > 0 else math.nan, 2),
                 pose_rate_hz=fin(valid.sum() / span if span > 0 else math.nan, 2),
                 valid_share=fin(valid.mean(), 4), refusals=reasons,
                 ambiguous_share=fin(inf[valid, 3].mean() if valid.any() else math.nan, 4),
                 range_median_m=fin(np.median(rng[valid]) if valid.any() else math.nan, 3),
                 range_min_m=fin(rng[valid].min() if valid.any() else math.nan, 3),
                 range_max_m=fin(rng[valid].max() if valid.any() else math.nan, 3),
                 rms_px_median=fin(np.nanmedian(inf[valid, 4]) if valid.any() else math.nan, 3),
                 corners_median=fin(np.median(inf[valid, 5]) if valid.any() else math.nan, 1),
                 latency_median_ms=fin(np.median(lat) * 1e3, 1), latency_p90_ms=fin(pct(lat, 90) * 1e3, 1),
                 latency_max_ms=fin(lat.max() * 1e3, 1))
        if len(b.image) > 1:
            m['image_rate_hz'] = fin((len(b.image) - 1) / (b.image[-1] - b.image[0]), 2)
        edges = np.arange(0.1, 1.6, 0.1)
        rows = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            k = valid & (rng >= lo) & (rng < hi)
            if k.sum():
                rows.append({'range_lo': round(lo, 2), 'range_hi': round(hi, 2), 'poses': int(k.sum()),
                             'ambiguous_share': fin(inf[k, 3].mean(), 4),
                             'rms_px_median': fin(np.nanmedian(inf[k, 4]), 3),
                             'corners_median': fin(np.median(inf[k, 5]), 1)})
        m['by_range'] = rows
    if len(b.fix):
        f = b.fix
        lat = f[:, 0] - f[:, 1]
        m.update(fix_latency_median_ms=fin(np.median(lat) * 1e3, 1), fix_latency_p90_ms=fin(pct(lat, 90) * 1e3, 1),
                 fix_ambiguous_share=fin(f[:, 19].mean(), 4), fix_cov_valid_share=fin(f[:, 15].mean(), 4))
        span = f[-1, 1] - f[0, 1]
        m['fix_rate_hz'] = fin(len(f) / span if span > 0 else math.nan, 2)
        if tl.armed:
            a = f[(f[:, 1] >= tl.t0) & (f[:, 1] <= tl.t_end)]
            if len(a) > 1:
                m['fix_rate_armed_hz'] = fin(len(a) / max(tl.t_end - tl.t0, 1e-6), 2)
                gaps = np.diff(a[:, 1])
                m['fix_gap_max_s'] = fin(gaps.max(), 3)
                m['fix_gaps_over_0p5s'] = int(np.sum(gaps > 0.5))
        m['jitter'] = still_jitter(b)
    return m


def still_spells(b: Bag, min_s: float = 1.0) -> list:
    """(t0, t1) where the robot was still: no moving command for 0.5 s before, and the gyro below 0.02 rad/s."""
    if not len(b.fix):
        return []
    t_lo, t_hi = b.fix[0, 1], b.fix[-1, 1]
    grid = np.arange(t_lo, t_hi, 0.05)
    if not len(grid):
        return []
    moving_cmd = np.zeros(len(grid), bool)
    if len(b.cmd):
        mv = b.cmd[np.abs(b.cmd[:, 1:]).max(axis=1) > 1e-4, 0]
        if len(mv):
            j = np.searchsorted(mv, grid, side='right') - 1
            moving_cmd = (j >= 0) & (grid - mv[np.clip(j, 0, None)] < 0.5)
    gyro_ok = np.ones(len(grid), bool)
    if len(b.imu):
        g = np.abs(b.imu[:, 7])
        gi = np.interp(grid, b.imu[:, 1], g)
        gyro_ok = gi < 0.02
    still = ~moving_cmd & gyro_ok
    spells, start = [], None
    for tg, s in zip(grid, still):
        if s and start is None:
            start = tg
        elif not s and start is not None:
            if tg - start >= min_s:
                spells.append((start, tg))
            start = None
    if start is not None and grid[-1] - start >= min_s:
        spells.append((start, grid[-1]))
    return spells


def still_jitter(b: Bag, win: float = 1.0) -> list:
    """Per 1 s window of a still spell: the unambiguous fixes' sd about a straight line, and the fixes' own R."""
    out = []
    f = b.fix
    for a, c in still_spells(b):
        for w0 in np.arange(a, c - win + 1e-9, win):
            k = (f[:, 1] >= w0) & (f[:, 1] < w0 + win) & (f[:, 19] == 0)
            if k.sum() < 8:
                continue
            t = f[k, 1] - w0
            sd = []
            for col in (2, 3, 4):
                y = np.unwrap(f[k, col]) if col == 4 else f[k, col]
                p = np.polyfit(t, y, 1)
                sd.append(float(np.std(y - np.polyval(p, t), ddof=2)))
            R = f[k, 5:14].reshape(-1, 3, 3)
            out.append({'t': float(w0), 'range_m': float(np.median(f[k, 18])), 'n': int(k.sum()),
                        'sd_x_mm': sd[0] * 1e3, 'sd_y_mm': sd[1] * 1e3, 'sd_th_deg': math.degrees(sd[2]),
                        'R_x_mm': float(np.median(np.sqrt(R[:, 0, 0]))) * 1e3,
                        'R_y_mm': float(np.median(np.sqrt(R[:, 1, 1]))) * 1e3,
                        'R_th_deg': math.degrees(float(np.median(np.sqrt(R[:, 2, 2]))))})
    return out


def processed_fixes(b: Bag) -> np.ndarray:
    """
    One row per fix the EKF processed: t_recv, nis, accepted, stage, ambiguous, range, t_capture.

    dock/estimate carries the last fix's NIS and verdict; each change of the pair marks a processed fix, matched
    to the newest fiducial/fix received before it.
    """
    e = b.est
    if not len(e):
        return np.zeros((0, 7))
    nis, acc = e[:, 15], e[:, 16]
    key_nis = np.where(np.isfinite(nis), np.round(nis, 9), -1.0)
    change = np.r_[True, (key_nis[1:] != key_nis[:-1]) | (acc[1:] != acc[:-1])]
    change &= ~((key_nis == -1.0) & (acc == 0))           # nothing processed yet (NaN, not accepted)
    idx = np.nonzero(change)[0]
    rows = []
    st = stage_at(b, e[idx, 0])
    for i, s in zip(idx, st):
        j = np.searchsorted(b.fix[:, 0], e[i, 0] + 0.002, side='right') - 1 if len(b.fix) else -1
        amb, rng, tc = (b.fix[j, 19], b.fix[j, 18], b.fix[j, 1]) if j >= 0 else (math.nan,) * 3
        rows.append((e[i, 0], nis[i], acc[i], s, amb, rng, tc))
    return np.array(rows, float).reshape(-1, 7)


def ekf_velocity(b: Bag, tl: Timeline, cam: dict) -> dict:
    """The EKF's velocity against a camera track over APPROACH to the end of commanding: RMS, slope, lag."""
    e = b.est[np.argsort(b.est[:, 1])]
    t = cam['t']
    mv = (t >= tl.approach) & (t <= tl.t_cmd)
    vel = {}
    for k, name in ((5, 'x'), (6, 'y')):
        ve, vc = np.interp(t, e[:, 1], e[:, k]), cam['v' + name]
        ok = mv & np.isfinite(vc)
        if ok.sum() >= 30:
            lag = fit_lag_gain(t[ok], t[ok], vc[ok], ve[ok], max_lag=0.4)
            vel[name] = {'rms_mm_s': fin(rms(ve[ok] - vc[ok]) * 1e3, 1), 'slope': fin(np.polyfit(vc[ok], ve[ok], 1)[0], 3),
                         'lag_s': fin(lag['lag_s'], 2), 'n': int(ok.sum()),
                         'camera_p95_mm_s': fin(pct(np.abs(vc[ok]), 95) * 1e3, 0)}
    return vel


def analyse_ekf(b: Bag, tl: Timeline, cam: dict | None, gate: float) -> tuple[dict, dict]:
    """The estimate against the camera-only track: residuals, velocity, NIS, re-anchors, consistency."""
    m, d = {}, {}
    e = b.est
    if not len(e):
        return {'estimates': 0}, d
    m['estimates'] = int(len(e))
    ts = e[:, 1]
    order = np.argsort(ts)
    ts, e = ts[order], e[order]
    lo, hi = (tl.t0, tl.t_cmd) if tl.armed else (ts[0], ts[-1])
    # --- position residuals at the unambiguous fixes' capture times (the estimate the MPC had then)
    f = b.fix[(b.fix[:, 19] == 0) & (b.fix[:, 1] > max(ts[0], lo)) & (b.fix[:, 1] < min(ts[-1], hi))] \
        if len(b.fix) else np.zeros((0, 20))
    if len(f):
        ex = np.interp(f[:, 1], ts, e[:, 2]) - f[:, 2]
        ey = np.interp(f[:, 1], ts, e[:, 3]) - f[:, 3]
        eth = wrap(np.interp(f[:, 1], ts, np.unwrap(e[:, 4])) - f[:, 4])
        P = np.c_[[np.interp(f[:, 1], ts, e[:, 7 + k]) for k in range(3)]].T
        R = f[:, [5, 9, 13]]
        z = np.c_[ex, ey, eth] / np.sqrt(P + R)
        st = stage_at(b, f[:, 0])
        moving = np.isin(st, MOVING)
        m['residual'] = {
            'fixes': int(len(f)), 'x_rms_mm': fin(rms(ex) * 1e3, 1), 'y_rms_mm': fin(rms(ey) * 1e3, 1),
            'th_rms_deg': fin(math.degrees(rms(eth)), 2),
            'moving_y_rms_mm': fin(rms(ey[moving]) * 1e3, 1), 'moving_x_rms_mm': fin(rms(ex[moving]) * 1e3, 1),
            'z2_mean': [fin(np.mean(z[:, k] ** 2), 2) for k in range(3)],
            'within_2sd': [fin(np.mean(np.abs(z[:, k]) < 2), 3) for k in range(3)]}
        d['resid'] = np.c_[f[:, 1], ex, ey, eth, np.sqrt(P + R), st]
    # --- velocity against the camera (moving part only)
    if cam is not None and tl.armed and math.isfinite(tl.approach):
        t = cam['t']
        mv = (t >= tl.approach) & (t <= tl.t_cmd)
        vx_e = np.interp(t, ts, e[:, 5])
        vy_e = np.interp(t, ts, e[:, 6])
        m['velocity'] = ekf_velocity(b, tl, cam)
        cam_all = camera_track(b, tl.t0, tl.t_end, ambiguous=True)
        if cam_all is not None:
            m['velocity_all_fixes'] = ekf_velocity(b, tl, cam_all)
        d['vel'] = (t, vx_e, vy_e, cam['vx'], cam['vy'], mv)
    # --- NIS and acceptance of each processed fix
    pf = processed_fixes(b)
    if len(pf):
        gated = np.isin(pf[:, 3], MOVING)
        n = pf[gated & np.isfinite(pf[:, 1])]
        acc_g = pf[gated, 2]
        m['nis'] = {'processed': int(len(pf)), 'gated': int(gated.sum()),
                    'accepted_share_gated': fin(acc_g.mean() if len(acc_g) else math.nan, 3),
                    'rejected_gated': int(np.sum(acc_g == 0)),
                    'median': fin(np.median(n[:, 1]) if len(n) else math.nan, 2),
                    'mean_accepted': fin(np.mean(n[n[:, 2] == 1, 1]) if (n[:, 2] == 1).any() else math.nan, 2),
                    'share_over_chi2_95': fin(np.mean(n[:, 1] > CHI2_3[0.95]) if len(n) else math.nan, 3),
                    'share_over_gate': fin(np.mean(n[:, 1] > gate) if len(n) else math.nan, 3),
                    'median_unambiguous': fin(np.median(n[n[:, 4] == 0, 1]) if (n[:, 4] == 0).any() else math.nan, 2),
                    'median_ambiguous': fin(np.median(n[n[:, 4] == 1, 1]) if (n[:, 4] == 1).any() else math.nan, 2)}
        d['nis'] = pf
    # --- re-anchors (the node logs each) and steps in the estimate
    re_ = []
    for t, node, _, text in b.rosout:
        if node == 'ekf_node' and 'restarts from them' in text:
            g = REANCHOR_RE.search(text)
            if g:
                a = [float(v) for v in g.groups()]
                re_.append({'t': round(t - tl.t0, 2), 'dx_mm': round((a[3] - a[0]) * 1e3, 0),
                            'dy_mm': round((a[4] - a[1]) * 1e3, 0), 'dth_deg': round(a[5] - a[2], 1),
                            'stage': STAGES[int(stage_at(b, t)[0])]})
    m['reanchors'] = len(re_)
    m['reanchor_list'] = re_
    if re_:
        m['reanchor_max_mm'] = fin(max(math.hypot(r['dx_mm'], r['dy_mm']) for r in re_), 0)
    sel = (ts >= lo) & (ts <= hi)
    steps = np.hypot(np.diff(e[sel, 2]), np.diff(e[sel, 3]))
    big = np.nonzero(steps > 0.01)[0]
    m['steps_over_10mm'] = int(len(big))
    m['step_max_mm'] = fin(steps.max() * 1e3 if len(steps) else math.nan, 0)
    m['step_list'] = [{'t': round(float(ts[sel][i + 1] - tl.t0), 2), 'mm': round(float(steps[i]) * 1e3, 0)}
                      for i in big[np.argsort(-steps[big])][:8]]
    m['p_sd_max_mm'] = fin(math.sqrt(max(e[sel, 7].max(), e[sel, 8].max())) * 1e3 if sel.any() else math.nan, 1)
    return m, d


def analyse_mpc(b: Bag, tl: Timeline, cfg: dict) -> tuple[dict, dict]:
    """Solve times, fallbacks, slack, which limit is active, smoothness, plan pointing, stage times."""
    m, d = {}, {}
    if not len(b.mpc):
        return {'ticks': 0}, d
    p = cfg['_params']
    mp = b.mpc
    st_rows = np.searchsorted(b.stage[:, 0], mp[:, 0] + 0.002, side='right') - 1
    st_rows = np.clip(st_rows, 0, None)
    stage = b.stage[st_rows, 1].astype(int)
    v_max, r_max = b.stage[st_rows, 5], b.stage[st_rows, 6]
    y_ref = b.stage[st_rows, 2:5]
    on = np.isin(stage, MOVING)
    u = mp[:, 2:5]
    solve = mp[:, 7]
    m.update(ticks=int(len(mp)), ticks_commanding=int(on.sum()),
             solve_ms={'median': fin(np.median(solve), 3), 'p95': fin(pct(solve, 95), 3), 'p99': fin(pct(solve, 99), 3),
                       'max': fin(solve.max(), 3), 'over_25ms': int(np.sum(solve > 25.0))},
             fallbacks=int(mp[:, 8].sum()), exit_flags={str(int(k)): int(v) for k, v in
                                                        zip(*np.unique(mp[:, 6], return_counts=True))},
             slack_ticks=int(np.sum(np.nan_to_num(mp[:, 5]) > 1e-6)), slack_max=fin(np.nanmax(mp[:, 5]), 6))
    # which limit binds: per-axis or L1 speed cap, turn cap, wheel speed, increments
    eps = 1e-3
    caps = cfg['settings']['speed_cap']
    g = p['geom']
    L = g['lx'] + g['ly']
    Jinv = np.array([[1, -1, -L], [1, 1, L], [1, 1, -L], [1, -1, L]]) / g['Rw']
    wheel = np.abs(u @ Jinv.T).max(axis=1)
    du = np.abs(np.diff(u, axis=0, prepend=u[:1]))
    du_max = p['mpc']['Ts'] * np.array([p['lim']['a_max'], p['lim']['a_max'], p['lim']['alpha_max']])
    act = {}
    if on.any():
        if caps == 'L1':
            act['speed_L1'] = np.abs(u[:, 0]) + np.abs(u[:, 1]) >= v_max - eps
        act['speed_forward'] = np.abs(u[:, 0]) >= v_max - eps
        act['speed_lateral'] = np.abs(u[:, 1]) >= v_max - eps
        act['turn'] = np.abs(u[:, 2]) >= r_max - eps
        act['wheel'] = wheel >= p['lim']['w_max'] - 0.05
        act['accel_forward'] = du[:, 0] >= du_max[0] - 1e-4
        act['accel_lateral'] = du[:, 1] >= du_max[1] - 1e-4
        act['accel_turn'] = du[:, 2] >= du_max[2] - 1e-4
        m['limits_active_share'] = {k: fin(v[on].mean(), 3) for k, v in act.items()}
        per_stage = {}
        for s in (APPROACH, FINAL):
            k = stage == s
            if k.sum():
                per_stage[STAGES[s]] = {kk: fin(v[k].mean(), 3) for kk, v in act.items()}
        m['limits_active_by_stage'] = per_stage
    # smoothness: increments between consecutive commanding ticks, and r_b reversals
    cons = on[1:] & on[:-1] & (np.diff(mp[:, 0]) < 0.08)
    dd = np.diff(u, axis=0)[cons]
    m['du_rms'] = [fin(rms(dd[:, k]), 4) for k in range(3)]
    rev = {}
    for k, name in ((1, 'v_b'), (2, 'r_b')):
        x = u[on, k]
        sgn = np.sign(np.where(np.abs(x) > 0.02 if k == 1 else np.abs(x) > 0.05, x, 0))
        nzs = sgn[sgn != 0]
        dur = on.sum() * p['mpc']['Ts']
        rev[name] = fin(10.0 * np.sum(nzs[1:] != nzs[:-1]) / max(dur, 1e-6), 2)
    m['reversals_per_10s'] = rev
    # plan pointing: the plan's end against the goal, from the estimate the tick used
    est_by_stamp = {round(s, 6): i for i, s in enumerate(b.est[:, 1])} if len(b.est) else {}
    ang = np.full(len(mp), np.nan)
    pos = np.full((len(mp), 3), np.nan)
    for i in range(len(mp)):
        j = est_by_stamp.get(round(mp[i, 1], 6))
        if j is None or not on[i] or len(b.y_pred[i]) < 3:
            continue
        pos[i] = b.est[j, 2:5]
        yp = b.y_pred[i].reshape(-1, 3)
        dp, dg = yp[-1, :2] - pos[i, :2], y_ref[i, :2] - pos[i, :2]
        if np.linalg.norm(dg) > 0.02 and np.linalg.norm(dp) > 1e-3:
            c = dp @ dg / np.linalg.norm(dp) / np.linalg.norm(dg)
            ang[i] = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    okang = np.isfinite(ang)
    m['pointing'] = {'ticks': int(okang.sum()), 'median_deg': fin(np.median(ang[okang]) if okang.any() else math.nan, 1),
                     'share_over_45deg': fin(np.mean(ang[okang] > 45) if okang.any() else math.nan, 3)}
    # strafe usage
    dt = p['mpc']['Ts']
    m['strafe'] = {'integral_m': fin(np.sum(np.abs(u[on, 1])) * dt, 3), 'max_m_s': fin(np.abs(u[on, 1]).max() if on.any() else math.nan, 3),
                   'share_lateral_dominant': fin(np.mean(np.abs(u[on, 1]) > np.abs(u[on, 0])) if on.any() else math.nan, 3),
                   'forward_integral_m': fin(np.sum(np.abs(u[on, 0])) * dt, 3)}
    d['tick'] = np.c_[mp[:, 0], stage, u, v_max, r_max, solve, mp[:, 8], np.nan_to_num(mp[:, 5]), ang, wheel,
                      *[act.get(k, np.zeros(len(mp), bool)).astype(float) for k in
                        ('speed_L1', 'speed_forward', 'speed_lateral', 'turn', 'wheel')], pos]
    d['caps'] = caps
    return m, d


def stage_metrics(b: Bag, tl: Timeline, cam: dict | None) -> dict:
    """Time in each stage, time to dock, overshoot and final error (from the camera where it can)."""
    m = {'outcome': tl.outcome}
    if not tl.armed:
        return m
    tot = {}
    for s, a, c in tl.segments:
        tot[STAGES[s]] = tot.get(STAGES[s], 0.0) + (c - a)
    m['time_in_stage_s'] = {k: round(v, 2) for k, v in tot.items()}
    m['holds'] = sum(1 for s in tl.segments if s[0] == HOLD)
    m['stages'] = [STAGES[s] for s, _, _ in tl.segments]
    m['acquire_s'] = fin(tl.approach - tl.t0, 2)
    m['approach_to_docked_s'] = fin(tl.docked - tl.approach, 2)
    if tl.abort_cause:
        m['abort_cause'] = tl.abort_cause
    if cam is None or not math.isfinite(tl.approach):
        return m
    m.update(motion_metrics(cam['t'], cam['x'], cam['y'], cam['th'], tl.approach,
                            tl.docked if math.isfinite(tl.docked) else tl.t_cmd, b.fix, b.est))
    return m


def motion_metrics(t, x, y, th, t_app, t_end, fix=None, est=None) -> dict:
    """Start pose, lateral overshoot and final error of a track (real: camera; simulated: true pose)."""
    m = {}
    k0 = np.nonzero((t >= t_app - 0.6) & (t <= t_app) & np.isfinite(y))[0]
    start, src = None, ''
    if fix is not None and len(fix):
        for lo, why in ((0.6, 'unambiguous fixes, last 0.6 s before APPROACH'),
                        (3.0, 'unambiguous fixes, last 3 s before APPROACH (the robot stood in ACQUIRE)')):
            ff = fix[(fix[:, 19] == 0) & (fix[:, 1] >= t_app - lo) & (fix[:, 1] <= t_app + 0.05)]
            if len(ff) >= 3:
                start, src = (np.median(ff[:, 2]), np.median(ff[:, 3]), np.median(ff[:, 4])), why
                break
    if start is None and len(k0):
        start, src = (np.nanmedian(x[k0]), np.nanmedian(y[k0]), np.nanmedian(th[k0])), 'track'
    if start is None and est is not None and len(est):
        j = np.searchsorted(est[:, 1], t_app) - 1
        if j >= 0:
            start, src = tuple(est[j, 2:5]), 'the EKF estimate at APPROACH (every fix around it was ambiguous)'
    if start is None:
        return m
    m['start'] = {'x_m': fin(start[0], 3), 'y_m': fin(start[1], 3), 'th_deg': fin(math.degrees(start[2]), 1),
                  'source': src}
    sel = (t >= t_app) & (t <= t_end) & np.isfinite(y)
    if sel.sum() >= 3:
        y0 = start[1]
        if abs(y0) >= 0.01:
            m['lateral_overshoot_mm'] = fin(max(0.0, float(np.max(-np.sign(y0) * y[sel]))) * 1e3, 0)
        m['max_abs_y_after_x_m0p1_mm'] = fin(np.max(np.abs(y[sel & (x >= -0.1)])) * 1e3
                                             if (sel & (x >= -0.1)).any() else math.nan, 0)
    if fix is not None and len(fix):
        ff = fix[(fix[:, 19] == 0) & (fix[:, 1] >= t_end - 0.5) & (fix[:, 1] <= t_end)]
        if len(ff) >= 3:
            m['final'] = {'x_mm': fin(np.median(ff[:, 2]) * 1e3, 1), 'y_mm': fin(np.median(ff[:, 3]) * 1e3, 1),
                          'th_deg': fin(math.degrees(np.median(ff[:, 4])), 2), 'source': 'camera, last 0.5 s'}
    else:
        k = np.nonzero((t <= t_end) & np.isfinite(y))[0]
        if len(k):
            m['final'] = {'x_mm': fin(x[k[-1]] * 1e3, 1), 'y_mm': fin(y[k[-1]] * 1e3, 1),
                          'th_deg': fin(math.degrees(th[k[-1]]), 2), 'source': 'simulated pose'}
    return m


def analyse_drive(b: Bag, tl: Timeline, cam: dict | None) -> tuple[dict, dict]:
    """Commanded body velocity against what the camera (and the gyro) saw: gain and lag per axis."""
    m, d = {}, {}
    if cam is None or not len(b.cmd) or not math.isfinite(tl.approach):
        return m, d
    t = cam['t']
    mv = (t >= tl.approach) & (t <= tl.t_cmd + 0.5)
    tg = np.arange(t[0] - 1.0, t[-1] + 1.0, 0.01)
    cmd = held(b.cmd[:, 0], b.cmd[:, 1:4], tg)
    gyro = np.full(len(t), np.nan)
    if len(b.imu):
        bias = 0.0
        sp = still_spells(b)
        if sp:
            gs = np.concatenate([b.imu[(b.imu[:, 1] >= a) & (b.imu[:, 1] <= c), 7] for a, c in sp])
            bias = float(np.median(gs)) if len(gs) else 0.0
        _, gv = window_fit(b.imu[:, 1], b.imu[:, 7] - bias, t, 0.2, min_n=10)     # the camera's 0.4 s window
        gyro = gv
    axes = {}
    for k, (name, meas) in enumerate((('forward', cam['u']), ('lateral', cam['v']), ('turn', gyro))):
        sel = mv & np.isfinite(meas)
        r = fit_lag_gain(t[sel], tg, cmd[:, k], meas[sel], min_range=0.05 if k == 2 else 0.02)
        axes[name] = {kk: (fin(v, 4) if isinstance(v, float) else v) for kk, v in r.items()}
    m['axes'] = axes
    m['measured_with'] = 'camera (unambiguous fixes, 0.4 s line fits) for forward and lateral; gyro for turn'
    if len(b.tel):
        a = b.tel[(b.tel[:, 0] >= tl.t0) & (b.tel[:, 0] <= tl.t_end)]
        if len(a):
            m['telemetry'] = {'rows': int(len(a)), 'saturated_share': fin(a[:, 10].mean(), 4),
                              'timed_out_share': fin(a[:, 9].mean(), 4), 'armed_share': fin(a[:, 8].mean(), 4)}
            req = a[:, 5:8]
            if np.isfinite(req).all() and np.abs(req).sum() > 0:
                # bridge to Mega: how late the Mega's copy of the request follows cmd_vel
                rq = fit_lag_gain(a[:, 0], b.cmd[:, 0], b.cmd[:, 1], a[:, 5], max_lag=0.3)
                m['telemetry']['request_lag_s'] = fin(rq['lag_s'], 2)
                app = a[:, 2:5]
                big = np.abs(req).max(axis=1) > 1e-3
                if big.any():
                    ratio = np.linalg.norm(app[big], axis=1) / np.maximum(np.linalg.norm(req[big], axis=1), 1e-9)
                    m['telemetry']['applied_over_requested_median'] = fin(np.median(ratio), 3)
    d['drive'] = (t, tg, cmd, cam['u'], cam['v'], gyro, mv)
    return m, d


def analyse_drivecal(b: Bag) -> dict:
    """A drive calibration run: drive_calibration_fit.py's per-axis deadband and gain."""
    sys.path.insert(0, str(TOOLS))
    import drive_calibration_fit as dcf
    d = {'cmd': b.cmd, 'fix': np.c_[b.fix[:, 1], b.fix[:, 2:5], b.fix[:, 19], b.fix[:, 18]] if len(b.fix) else
         np.zeros((0, 6)), 'imu': np.c_[b.imu[:, 1], b.imu[:, 7]] if len(b.imu) else np.zeros((0, 2)),
         'tel': np.c_[b.tel[:, 1], b.tel[:, 12:16], b.tel[:, 11]] if len(b.tel) else np.zeros((0, 6))}
    steps, info, fits = dcf.analyse(d, dcf.Firmware(), 0.5)
    out = {'steps': len(steps), 'axes': {}, 'pulses_logged': info.get('pulses_logged')}
    for a, f in fits.items():
        if not f:
            continue
        dz, du = f.get('deadzone'), f.get('deadzone_u')
        out['axes'][dcf.NAMES[a]] = {'steps': f['n'], 'motion_threshold': fin(f['thr'], 3),
                                     'deadband': fin(dz[0], 4) if dz else None,
                                     'gain_above_deadband': fin(dz[1], 3) if dz else None,
                                     'fit_rms': fin(dz[2], 4) if dz else None,
                                     'pulse_deadband_us': fin(du[0], 0) if du else None,
                                     'per_us': fin(du[1] * 1000, 3) if du else None,
                                     'moved_against': len(f.get('against', []))}
    out['table'] = [(s.axis, s.cmd, s.response, s.n_fix, int(s.ambiguous), '; '.join(s.notes)) for s in steps]
    return out


# ============================================================================================ simulation
def drive_model(drive: dict) -> tuple:
    """(gain, lag, notes) per axis for the simulation, defaults where the run could not measure an axis."""
    gains, lags, notes = [], [], []
    for name, g0 in (('forward', 1.0), ('lateral', 1.0), ('turn', 1.0)):
        a = (drive.get('axes') or {}).get(name) or {}
        g, lag = a.get('gain'), a.get('lag_s')
        if g is None or lag is None or a.get('note') or not (0.1 <= abs(g) <= 2.0):
            notes.append(f'{name}: not measurable in this run, 1.0x and 0.15 s assumed.')
            g, lag = g0, 0.15
        elif a.get('flag'):
            notes.append(f"{name}: the measured {g:.2f}x is used as it is ({a['flag']}).")
        gains.append(g)
        lags.append(lag)
    return gains, lags, notes


def simulate_run(b: Bag, tl: Timeline, cfg: dict, drive: dict, fixes: dict, stagem: dict) -> dict | None:
    if not math.isfinite(tl.approach) or 'start' not in stagem:
        return None
    s = stagem['start']
    gains, lags, notes = drive_model(drive)
    tau = [max(L - CMD_DELAY_S, 0.02) for L in lags]
    pre_flash = tl.t0 < DECK_FLASH_UTC
    db = (0.005, 0.01, 0.03)          # exp12.py's residual; the old firmware's deadbands are not modelled
    fix_delay = (fixes.get('fix_latency_median_ms') or 120.0) / 1000.0
    real_len = (tl.docked if math.isfinite(tl.docked) else tl.t_cmd) - tl.approach
    # the settings the bag shows win over the matched commit's file (a working tree ahead of its commit)
    params = json.loads(json.dumps(cfg['_params']))
    obs, over = cfg.get('observed') or {}, {}
    for k, (grp, key) in (('standoff_m', ('sup', 'standoff')), ('v_approach', ('lim', 'v_approach')),
                          ('r_max', ('lim', 'r_max')), ('v_final', ('lim', 'v_final')), ('Np', ('mpc', 'Np'))):
        if k in obs and abs(float(params[grp][key]) - float(obs[k])) > 1e-6:
            over[k] = [params[grp][key], obs[k]]
            params[grp][key] = int(obs[k]) if k == 'Np' else float(obs[k])
    for k, (grp, key) in (('marker_x_m', ('geom', 'marker_m')), ('cam_x_m', ('geom', 'cam_xyz'))):
        if k in obs and abs(float(params[grp][key][0]) - float(obs[k])) > 1e-6:
            over[k] = [params[grp][key][0], obs[k]]
            params[grp][key][0] = float(obs[k])
    req = {'config': params, 'start': [s['x_m'], s['y_m'], math.radians(s['th_deg'])],
           'seconds': float(min(90.0, max(30.0, real_len + 25.0))), 'gain': gains, 'tau': tau,
           'delay_cmd': CMD_DELAY_S, 'db': db, 'fov_deg': FOV_HALF_DEG, 'fix_delay': fix_delay, 'seed': 1,
           'log_dt': 0.05}
    r = run_worker('sim', cfg['sha'], req)
    if r is None:
        return None
    log = np.array(r['log'], float)
    app = [t for t, n in r['stages'] if n == 'APPROACH']
    t_app = app[0] if app else 0.0
    dk = [t for t, n in r['stages'] if n == 'DOCKED']
    res = {'model': {'gain': [round(g, 3) for g in gains], 'lag_s': [round(v, 2) for v in lags],
                     'tau_s': [round(v, 3) for v in tau], 'delay_cmd_s': CMD_DELAY_S, 'deadband': list(db),
                     'deadband_basis': ('residual as in exp12.py; this run predates the per-wheel firmware, whose '
                                        'large deadbands from rest are not modelled' if pre_flash else
                                        'per-wheel compensated firmware, residual as in exp12.py'),
                     'pre_flash': pre_flash, 'miswired': tl.t0 < CHANNEL_FIX_UTC,
                     'fix_delay_s': round(fix_delay, 3), 'fov_half_deg': FOV_HALF_DEG, 'notes': notes,
                     'commit': cfg['commit'], 'variant': cfg.get('variant'), 'overrides_from_bag': over,
                     'start_source': s.get('source', '')},
           'outcome': r['final_stage'], 'stages': r['stages'], 'unseen_fixes': r['unseen'],
           'approach_to_docked_s': fin(dk[0] - t_app, 2) if dk else None}
    t = log[:, 0]
    on = np.isin(log[:, 7], MOVING)
    dt = 0.05
    res['strafe'] = {'integral_m': fin(np.sum(np.abs(log[on, 5])) * dt, 3),
                     'max_m_s': fin(np.abs(log[on, 5]).max() if on.any() else math.nan, 3),
                     'share_lateral_dominant': fin(np.mean(np.abs(log[on, 5]) > np.abs(log[on, 4])) if on.any() else math.nan, 3),
                     'forward_integral_m': fin(np.sum(np.abs(log[on, 4])) * dt, 3)}
    t_end = dk[0] if dk else t[-1]
    res.update(motion_metrics(t, log[:, 1], log[:, 2], log[:, 3], t_app, t_end))
    res['holds'] = sum(1 for _, n in r['stages'] if n == 'HOLD')
    # the simulated EKF's velocity against the simulated truth, as the real one is against the camera
    vt = np.gradient(log[:, 1:3], t, axis=0)
    ok = on & np.isfinite(log[:, 11])
    if ok.sum() > 20:
        res['ekf_velocity_rms_mm_s'] = [fin(rms(log[ok, 11] - vt[ok, 0]) * 1e3, 1),
                                        fin(rms(log[ok, 12] - vt[ok, 1]) * 1e3, 1)]
    res['_log'] = log
    res['_t_app'] = t_app
    return res


def sim_verdicts(real: dict, sim: dict, ekf: dict, mpcm: dict) -> list:
    """Plain statements of where the simulation does and does not reproduce the run."""
    out = []
    ro, so = real.get('outcome'), sim.get('outcome')
    if ro == 'DOCKED' and so == 'DOCKED':
        a, c = real.get('approach_to_docked_s'), sim.get('approach_to_docked_s')
        if a and c:
            ratio = c / a
            out.append(f'Both dock. The simulation takes {c:.1f} s from APPROACH to DOCKED against the '
                       f"robot's {a:.1f} s ({ratio:.2f}x)" + ('' if 0.75 <= ratio <= 1.33 else
                                                               ': it does not reproduce the timing.'))
    elif ro == 'disarmed in FINAL' and real.get('final') and abs(real['final']['x_mm']) <= 15 \
            and abs(real['final']['y_mm']) <= 25:
        f = real['final']
        out.append(f"The robot reached the coupler (camera: x {f['x_mm']:.0f} mm, y {f['y_mm']:.0f} mm at the end) but "
                   'never registered DOCKED, which needs the estimate within the dock tolerance at under 20 mm/s, '
                   f"and was disarmed; the simulation {so}" + (f" in {sim.get('approach_to_docked_s')} s"
                                                               if sim.get('approach_to_docked_s') else '')
                   + '. Its fixes carry no bias at contact, so it cannot reproduce a dock the camera reads short.')
    elif so != ro:
        out.append(f'The outcome differs: the robot ended {ro}'
                   + (f' ({real.get("abort_cause")})' if real.get('abort_cause') else '')
                   + f', the simulation {so}. The simulation does not reproduce this run.')
    else:
        out.append(f'Both end {ro}.')
    ho = real.get('lateral_overshoot_mm')
    hs = sim.get('lateral_overshoot_mm')
    if ho is not None and hs is not None:
        out.append(f'Lateral overshoot: robot {ho:.0f} mm, simulation {hs:.0f} mm'
                   + ('.' if abs(ho - hs) <= 20 else ': the simulation does not reproduce it.'))
    rs, ss = mpcm.get('strafe', {}), sim.get('strafe', {})
    if rs.get('integral_m') is not None and ss.get('integral_m') is not None:
        out.append(f"Strafe commanded: robot {rs['integral_m']:.2f} m in all (peak {fmt(rs.get('max_m_s'), '.2f')} m/s), "
                   f"simulation {ss['integral_m']:.2f} m (peak {fmt(ss.get('max_m_s'), '.2f')} m/s).")
    fr, fs = real.get('final'), sim.get('final')
    if fr and fs:
        out.append(f"Final pose: robot ({fr['x_mm']:.0f}, {fr['y_mm']:.0f}) mm, {fr['th_deg']:.1f} deg (camera); "
                   f"simulation ({fs['x_mm']:.0f}, {fs['y_mm']:.0f}) mm, {fs['th_deg']:.1f} deg.")
    if real.get('holds', 0) and not sim.get('holds'):
        out.append(f"The robot went to HOLD {real['holds']} times (no accepted fix for 0.5 s); the simulation never did: "
                   'its fixes are never ambiguous and never fail the gate for long.')
    if ekf.get('reanchors'):
        out.append(f"The robot's EKF re-anchored {ekf['reanchors']} times; the simulated EKF cannot, since its fixes "
                   'are unbiased and unambiguous.')
    if (sim.get('model') or {}).get('miswired'):
        out.append('In this run the drive\'s channels still drove the wrong wheels (fixed between the 16:22 and 16:35 NZDT '
                   'attempts, 7c items 30 and 34): a forward command drove the robot left and a turn turned it the other '
                   'way. Per-axis gains cannot represent commands landing on the wrong axis, so the simulation cannot '
                   'reproduce this run.')
    if (sim.get('model') or {}).get('pre_flash'):
        out.append('This run predates the per-wheel firmware (flashed about 18:27 NZDT): from rest its drive did not move '
                   'below about 0.03 m/s forward, 0.06-0.10 m/s sideways and 0.2 rad/s turning (7c items 49, 54). The '
                   'simulation keeps only the measured average gains, so it misses stalls on small commands.')
    rv = (ekf.get('velocity') or {}).get('y', {}).get('rms_mm_s')
    sv = sim.get('ekf_velocity_rms_mm_s')
    if rv is not None and sv:
        out.append(f"EKF lateral velocity error: robot {rv:.0f} mm/s RMS (against the camera), simulation "
                   f"{sv[1]:.0f} mm/s (against its truth)" + (': the simulated estimate is far better than the real one.'
                                                              if rv > 2 * max(sv[1], 1.0) else '.'))
    return out


# ============================================================================================ figures
def style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.labelsize': 8.5,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK2, 'axes.linewidth': 0.8, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'grid.linestyle': '-', 'xtick.color': MUTED,
        'ytick.color': MUTED, 'xtick.labelcolor': INK2, 'ytick.labelcolor': INK2, 'text.color': INK,
        'axes.titlecolor': INK, 'legend.frameon': False, 'legend.fontsize': 7.5, 'lines.linewidth': 1.4,
        'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
        'axes.prop_cycle': matplotlib.cycler(color=C), 'axes.spines.top': False, 'axes.spines.right': False})
    return plt


def save(fig, out: Path, name: str) -> None:
    (out / 'figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'figures' / f'{name}.png', dpi=130, bbox_inches='tight')
    fig.savefig(out / 'figures' / f'{name}.pdf', bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)


def shade_stages(ax, tl: Timeline, label: bool = True) -> None:
    for s, a, c in tl.segments:
        if s in STAGE_WASH:
            alpha = 0.45 if s in (ACQUIRE, APPROACH, FINAL) else 0.18
            ax.axvspan(a - tl.t0, c - tl.t0, color=STAGE_WASH[s], alpha=alpha, lw=0, zorder=0)
            if label and c - a > 0.04 * max(tl.t_end - tl.t0, 1):
                ax.text((a + c) / 2 - tl.t0, 1.0, STAGES[s], transform=ax.get_xaxis_transform(), ha='center',
                        va='bottom', fontsize=6.5, color=INK2, clip_on=False)


def attempt_xlim(axs, tl) -> str:
    """Show the attempt: from arming to 3 s after commanding ended; return a note if the record goes on."""
    hi = min(tl.t_end, tl.t_cmd + 3.0)
    for ax in np.atleast_1d(axs):
        ax.set_xlim(-0.5, hi - tl.t0 + 0.3)
    rest = tl.t_end - hi
    return f' (the bag goes on {rest:.0f} s more, not shown)' if rest > 1.0 else ''


DET_COLOUR = {'release': C[0], 'SSL live r3': C[1], 'SSL offline r1': C[2]}


def fig_trajectory(b, tl, cfgm, md, out, figs, every=8):
    plt = style()
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    e = b.est[(b.est[:, 1] >= tl.t0) & (b.est[:, 1] <= tl.t_cmd + 0.5)]       # the attempt, not the handling after it
    f = b.fix[(b.fix[:, 1] >= tl.t0) & (b.fix[:, 1] <= tl.t_cmd + 0.5)]
    tick = md.get('tick')
    if tick is not None:
        idx = [i for i in range(len(b.mpc)) if int(tick[i, 1]) in MOVING][::every]
        cm = plt.get_cmap('Blues')
        for n, i in enumerate(idx):
            y = b.y_pred[i].reshape(-1, 3)
            col = cm(0.35 + 0.6 * n / max(len(idx) - 1, 1))
            ax.plot(y[:, 1], y[:, 0], '-', color=col, lw=0.7, alpha=0.8, zorder=2)
    if len(f):
        u = f[f[:, 19] == 0]
        a = f[f[:, 19] == 1]
        ax.plot(u[:, 3], u[:, 2], 'o', ms=2.6, color=C[1], alpha=0.75, mew=0, label='camera fix', zorder=3)
        if len(a):
            ax.plot(a[:, 3], a[:, 2], 'o', ms=3.2, mfc='none', mec=C[4], mew=0.7, alpha=0.9,
                    label='camera fix, ambiguous', zorder=3)
    if len(e):
        ax.plot(e[:, 3], e[:, 2], '-', color=INK, lw=1.4, label='EKF estimate', zorder=4)
        ax.plot(e[0, 3], e[0, 2], 'o', ms=8, color=INK, mec='white', mew=1.5, zorder=5)
    so = (cfgm.get('observed') or {}).get('standoff_m', cfgm['settings']['standoff_m'])
    ax.plot([0], [-so], '+', ms=12, mew=2, color=C[7], label=f'APPROACH target ({so:.2f} m out)', zorder=6)
    ax.plot([0], [0], 'x', ms=10, mew=2, color=C[7], label='docked pose', zorder=6)
    if tick is not None:
        ax.plot([], [], '-', color=plt.get_cmap('Blues')(0.7), lw=0.8, label=f'MPC plan, every {every}th tick')
    ax.set_xlabel('y, sideways (m, target frame; robot left is +)')
    ax.set_ylabel('x, towards the board (m)')
    ax.invert_xaxis()
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='best')
    ax.set_title(f'{b.name}: path, fixes and MPC plans')
    save(fig, out, 'trajectory')
    figs.append(Fig('trajectory', 'Top-down path of the run',
                    'The EKF estimate (black line, dot at the first estimate), every camera fix (orange; hollow '
                    'pink where the pose was ambiguous), every 8th MPC plan (blue, darker later) with the '
                    'APPROACH target (+) and the docked pose (x), in the target frame.',
                    'dock/estimate and fiducial/fix as logged; each plan is dock/mpc_info.y_pred, Np rows of '
                    '(x, y, theta), drawn from the estimate the tick used. The plan is the 10 planned moves then '
                    'the last move held to the horizon, so it shows where that command leads, not a path the '
                    'controller intends to follow.',
                    'Where the fixes and the estimate part, the EKF is not following the camera; plans that '
                    'point away from the target show the MPC starting from a velocity the robot did not have '
                    '(mpc_analysis_2026-10-14 finding 4).',
                    'An ambiguous fix is the solution nearer the EKF prediction, so it is not independent of '
                    'the estimate. Fixes at range carry centimetres of noise; the axes are equal so sideways '
                    'error is not exaggerated.', 'trajectory'))


def fig_timeline(b, tl, md, out, figs):
    plt = style()
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.4), sharex=True)
    e = b.est[(b.est[:, 1] >= tl.t0 - 1) & (b.est[:, 1] <= tl.t_end + 1)]
    f = b.fix[(b.fix[:, 1] >= tl.t0 - 1) & (b.fix[:, 1] <= tl.t_end + 1)]
    for k, (ax, lab, scale) in enumerate(zip(axs, ('x (m)', 'y (m)', 'heading (deg)'), (1, 1, 180 / math.pi))):
        shade_stages(ax, tl, label=(k == 0))
        if len(f):
            u, a = f[f[:, 19] == 0], f[f[:, 19] == 1]
            ax.plot(u[:, 1] - tl.t0, u[:, 2 + k] * scale, 'o', ms=2.4, color=C[1], mew=0, alpha=0.8,
                    label='camera fix')
            if len(a):
                ax.plot(a[:, 1] - tl.t0, a[:, 2 + k] * scale, 'o', ms=3, mfc='none', mec=C[4], mew=0.7,
                        label='ambiguous fix')
        if len(e):
            ax.plot(e[:, 1] - tl.t0, e[:, 2 + k] * scale, '-', color=INK, lw=1.2, label='EKF estimate')
        ax.set_ylabel(lab)
    for r in md.get('reanchor_list', []):
        for ax in axs:
            ax.axvline(r['t'], color=C[7], lw=0.8)
    axs[0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.07), ncol=3)
    cut = attempt_xlim(axs, tl)
    axs[-1].set_xlabel('time since docking was armed (s); red lines: EKF re-anchors' + cut)
    fig.suptitle(f'{b.name}: stages, estimate and camera fixes', fontsize=9.5, y=1.0)
    fig.tight_layout()
    save(fig, out, 'timeline')
    figs.append(Fig('timeline', 'Stages, estimate and fixes over time',
                    'x, y and heading of the EKF estimate (black) and of each camera fix (orange; hollow pink if '
                    'ambiguous) against time since arming, under the supervisor stages (grey bands; HOLD amber, '
                    'DOCKED green, ABORT red). Red vertical lines mark EKF re-anchors.',
                    'Stages from dock/stage (receive time on the Pi), estimates at their IMU stamps, fixes at their '
                    'capture stamps, all on the Pi clock. Re-anchors are parsed from ekf_node\'s log.',
                    'This is the run at a glance: how long ACQUIRE took, whether the estimate stayed on the fixes, '
                    'and where HOLDs and re-anchors came.',
                    'Capture stamps are the frame read time, 10-40 ms after exposure (7c item 51), and the EKF '
                    'does not correct for it.', 'trajectory'))


def fig_ekf(b, tl, ekfm, ed, gate, out, figs):
    plt = style()
    if 'vel' in ed:
        t, vxe, vye, vxc, vyc, mv = ed['vel']
        fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True)
        for ax, ve, vc, lab, key in ((axs[0], vxe, vxc, 'vx, towards the board (m/s)', 'x'),
                                     (axs[1], vye, vyc, 'vy, sideways (m/s)', 'y')):
            shade_stages(ax, tl, label=(ax is axs[0]))
            ax.plot(t - tl.t0, vc, '-', color=C[1], lw=1.4, label='camera (0.4 s line fits)')
            ax.plot(t - tl.t0, ve, '-', color=INK, lw=1.2, label='EKF')
            v = (ekfm.get('velocity') or {}).get(key)
            if v:
                ax.text(0.01, 0.04, f"RMS error {v['rms_mm_s']} mm/s, slope {v['slope']}, EKF lag {v['lag_s']} s",
                        transform=ax.transAxes, fontsize=7, color=INK2)
            ax.set_ylabel(lab)
        axs[0].legend(loc='upper right', ncol=2)
        axs[-1].set_xlabel('time since arming (s)' + attempt_xlim(axs, tl))
        fig.tight_layout()
        save(fig, out, 'ekf_velocity')
        figs.append(Fig('ekf_velocity', "The EKF's velocity against the camera's",
                        'Velocity in the target frame from the camera alone (orange) and the EKF (black) over the '
                        'commanded part of the run.',
                        'Camera velocity: a straight line through the unambiguous fixes within +-0.2 s of each '
                        '50 ms grid point (at least 4 fixes); the EKF velocity interpolated at the same times. '
                        'RMS error and slope (EKF on camera, 1 = right) over APPROACH to the end; lag is the '
                        'shift of the EKF that best fits the camera.',
                        'The MPC starts every prediction from the EKF velocity, so a slope well below 1 or a large '
                        'RMS error bends every plan.',
                        'The camera velocity is itself noisy: about 10-30 mm/s at 0.3-0.5 m, more at range, and it '
                        'is missing where fixes were ambiguous or absent. It is smoothed over 0.4 s, which flattens '
                        'fast changes.', 'ekf'))
    if 'resid' in ed:
        r = ed['resid']
        fig, axs = plt.subplots(3, 1, figsize=(7.2, 5.6), sharex=True)
        for k, (ax, lab, sc) in enumerate(zip(axs, ('x (mm)', 'y (mm)', 'heading (deg)'), (1e3, 1e3, 180 / math.pi))):
            shade_stages(ax, tl, label=(k == 0))
            tt = r[:, 0] - tl.t0
            sd = r[:, 4 + k] * sc
            ax.fill_between(tt, -2 * sd, 2 * sd, color=C[0], alpha=0.12, lw=0, label='+-2 sd (EKF P + fix R)')
            ax.plot(tt, r[:, 1 + k] * sc, 'o', ms=2.4, color=INK, mew=0, label='estimate - fix')
            ax.set_ylabel(lab)
            lim = np.nanpercentile(np.abs(r[:, 1 + k] * sc), 99) if len(r) else 1
            ax.set_ylim(-1.3 * max(lim, 1e-3), 1.3 * max(lim, 1e-3))
        res = ekfm.get('residual', {})
        axs[0].legend(loc='upper right', ncol=2)
        axs[-1].set_xlabel('time since arming (s)' + attempt_xlim(axs, tl))
        fig.tight_layout()
        save(fig, out, 'ekf_residuals')
        figs.append(Fig('ekf_residuals', 'Estimate minus camera fix, against the claimed uncertainty',
                        'At each unambiguous fix, the estimate the EKF had at the capture time minus the fix (black), '
                        'and the band +-2 sqrt(P + R) from the estimate\'s own variance and the fix\'s R (blue).',
                        'dock/estimate interpolated to each fix\'s capture stamp (the real-time estimate, before that '
                        'fix corrected it), P from DockEstimate.p_diag, R from FiducialFix.covariance. '
                        f"Normalised squares (residual^2 / (P + R)) average {res.get('z2_mean')} for x, y, heading; "
                        f"a consistent filter gives about 1, and {res.get('within_2sd')} of residuals fall inside 2 sd "
                        '(about 0.95 if consistent).',
                        'Points outside the band are fixes the filter considered nearly impossible: either the '
                        'estimate drifted (velocity error integrated) or R is too small for the fix.',
                        'R comes from the pipeline\'s range model fitted on synthetic benchmarks, not this camera; '
                        'the residual mixes estimate error with fix error. Ambiguous fixes are left out.', 'ekf'))
    if 'nis' in ed:
        pf = ed['nis']
        sel = np.isin(pf[:, 3], MOVING) & np.isfinite(pf[:, 1]) & (pf[:, 1] > 0)
        if sel.sum() >= 5:
            from scipy import stats
            fig, ax = plt.subplots(figsize=(6.4, 3.6))
            v = pf[sel]
            bins = np.logspace(-2, max(4, math.log10(v[:, 1].max()) + 0.1), 40)
            ax.hist([v[v[:, 2] == 1, 1], v[v[:, 2] == 0, 1]], bins=bins, color=[C[0], C[1]], stacked=True,
                    label=['accepted', 'rejected by the gate'], rwidth=0.9)
            xs = np.logspace(-2, 4, 400)
            dens = stats.chi2.pdf(xs, 3) * len(v) * np.diff(np.log(bins)).mean() * xs
            ax.plot(xs, dens, '-', color=INK, lw=1.4, label='chi-square, 3 dof (expected)')
            ax.axvline(CHI2_3[0.95], color=MUTED, lw=1)
            ax.axvline(gate, color=C[7], lw=1.2)
            ax.text(CHI2_3[0.95], 0.98, ' 95%', transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, va='top')
            ax.text(gate, 0.98, ' gate', transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, va='top')
            ax.set_xscale('log')
            ax.set_xlabel('NIS of each processed fix in APPROACH, FINAL and HOLD')
            ax.set_ylabel('fixes')
            ax.legend(loc='upper left')
            fig.tight_layout()
            save(fig, out, 'ekf_nis')
            n = ekfm['nis']
            figs.append(Fig('ekf_nis', 'Normalised innovation squared against its expected distribution',
                            'The NIS of every fix the EKF processed while the robot was under control (stacked: '
                            'accepted blue, rejected orange), against the chi-square density with 3 degrees of '
                            'freedom that a consistent filter would give.',
                            'DockEstimate.last_nis and last_fix_accepted change once per processed fix; each change '
                            'is one fix. For an ambiguous pose the node fuses the solution with the smaller NIS, so '
                            'those values are biased low.',
                            f"Median {n['median']} against 2.37 expected; {fmt(n['share_over_chi2_95'], '.0%')} above "
                            f"the 95% point (5% expected), {fmt(n['share_over_gate'], '.0%')} above the gate of "
                            f"{gate:g}. A distribution far to the right means the filter is overconfident (P + R too "
                            'small for the real innovation); far to the left, too cautious.',
                            'The innovation includes the latency mismatch (capture stamps 10-40 ms late) and any '
                            'board or camera calibration error, which the chi-square does not allow for.', 'ekf'))


def fig_mpc(b, tl, mm, md, out, figs):
    if 'tick' not in md:
        return
    plt = style()
    tk = md['tick']
    on = np.isin(tk[:, 1], MOVING)
    t = tk[:, 0] - tl.t0
    fig, axs = plt.subplots(4, 1, figsize=(7.2, 7.4), sharex=True)
    for k, (ax, lab) in enumerate(zip(axs[:3], ('u_b forward (m/s)', 'v_b sideways (m/s)', 'r_b turn (rad/s)'))):
        shade_stages(ax, tl, label=(k == 0))
        cap = tk[:, 6] if k == 2 else tk[:, 5]
        ax.plot(t, np.where(on, cap, np.nan), '-', color=MUTED, lw=1.0, label='cap from the supervisor')
        ax.plot(t, np.where(on, -cap, np.nan), '-', color=MUTED, lw=1.0)
        ax.plot(t, np.where(on, tk[:, 2 + k], np.nan), '-', color=C[0], lw=1.4, label='command')
        ax.set_ylabel(lab)
    axs[0].legend(loc='upper right', ncol=2)
    ax = axs[3]
    shade_stages(ax, tl, label=False)
    ax.plot(t, tk[:, 10], 'o', ms=2.4, mew=0, color=INK)
    ax.axhline(45, color=MUTED, lw=1)
    ax.set_ylabel('plan vs goal (deg)')
    ax.set_ylim(0, 180)
    axs[-1].set_xlabel('time since arming (s)' + attempt_xlim(axs, tl))
    fig.tight_layout()
    save(fig, out, 'mpc_commands')
    la = mm.get('limits_active_share', {})
    figs.append(Fig('mpc_commands', 'Commands, caps and plan pointing',
                    'The body-frame command on each tick (blue) inside the supervisor\'s caps (grey: v_max for '
                    'both translations, r_max for turning), and the angle between where the plan ends and where '
                    'the goal is (bottom).',
                    'dock/mpc_info.u and dock/stage of the same tick. Plan pointing: the angle between (plan end - '
                    'estimate) and (goal - estimate), skipped within 2 cm of the goal.',
                    f"Share of commanding ticks at a limit: {', '.join(f'{k} {fmt(v, '.0%')}' for k, v in la.items())}. "
                    'A command riding a cap means the MPC wanted more than it was allowed; plans pointing more than '
                    '45 deg from the goal mean the prediction started from a velocity far from the command.',
                    f"The cap rows are {md.get('caps')} at this commit. 'At a limit' is within 1 mm/s (1 mrad/s) of "
                    'it; the wheel-speed limit is checked through the nominal mecanum inverse, not the firmware\'s '
                    'per-wheel calibration.', 'mpc'))
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    s = tk[on, 7] if on.any() else tk[:, 7]
    ax.hist(s, bins=40, color=C[0], rwidth=0.9)
    for q, lab in ((50, 'median'), (99, 'p99')):
        v = pct(s, q)
        ax.axvline(v, color=INK2, lw=1)
        ax.text(v, 0.97, f' {lab} {v:.2f} ms', transform=ax.get_xaxis_transform(), fontsize=7, color=INK2, va='top')
    ax.set_xlabel('solve time per tick on the Pi (ms; budget 25 ms)')
    ax.set_ylabel('ticks')
    fig.tight_layout()
    save(fig, out, 'mpc_solve')
    sm = mm['solve_ms']
    figs.append(Fig('mpc_solve', 'MPC solve time per tick',
                    'The time each commanding tick spent in Mpc.step on the Pi.',
                    'dock/mpc_info.solve_ms, measured with perf_counter around the solve (linearise, build, DAQP).',
                    f"Median {sm['median']} ms, p99 {sm['p99']} ms, max {sm['max']} ms against the 25 ms budget "
                    f"({sm['over_25ms']} ticks over). Fallbacks: {mm['fallbacks']}; ticks with slack: {mm['slack_ticks']}.",
                    'Wall time on a shared core: a scheduling delay shows up as a slow solve.', 'mpc'))


def fig_drive(b, tl, dm, dd, out, figs):
    if 'drive' not in dd:
        return
    plt = style()
    t, tg, cmd, cu, cv, gyro, mv = dd['drive']
    fig, axs = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)
    for k, (ax, meas, lab, name) in enumerate(zip(axs, (cu, cv, gyro), ('forward (m/s)', 'sideways (m/s)', 'turn (rad/s)'),
                                                 ('forward', 'lateral', 'turn'))):
        shade_stages(ax, tl, label=(k == 0))
        sel = (tg >= t[mv][0] - 0.5) & (tg <= t[mv][-1] + 0.5) if mv.any() else np.ones(len(tg), bool)
        ax.plot(tg[sel] - tl.t0, cmd[sel, k], '-', color=MUTED, lw=1.2, label='commanded (cmd_vel, held)')
        ax.plot(t - tl.t0, meas, '-', color=C[1], lw=1.4, label='measured (camera; gyro for turn)')
        a = dm['axes'].get(name, {})
        if a.get('gain') is not None and a.get('lag_s') is not None:
            pred = a['gain'] * np.interp(t - a['lag_s'], tg, cmd[:, k]) + (a.get('offset') or 0.0)
            ax.plot(t - tl.t0, np.where(mv, pred, np.nan), '-', color=C[0], lw=1.0,
                    label='fit: gain x command, delayed')
            ax.text(0.01, 0.04, f"gain {a['gain']:.2f}, lag {a['lag_s']:.2f} s, residual "
                                f"{a['resid_rms']:.3f}" + (f" ({a.get('note') or a.get('flag')})" if a.get('note') or a.get('flag') else ''),
                    transform=ax.transAxes, fontsize=7, color=INK2)
        ax.set_ylabel(lab)
    axs[0].legend(loc='upper right', ncol=3)
    axs[-1].set_xlabel('time since arming (s)' + attempt_xlim(axs, tl))
    fig.tight_layout()
    save(fig, out, 'drive')
    figs.append(Fig('drive', 'What the drive delivered against what was commanded',
                    'Per body axis, the command in force (grey; cmd_vel held, zero after the bridge\'s 0.1 s '
                    'timeout), what the robot did (orange: camera-derived body velocity, gyro for turning) and the '
                    'fitted model gain x command delayed by the lag (blue).',
                    'Body velocity: the camera\'s target-frame velocity turned by the fixes\' heading. For each lag '
                    'from 0 to 0.6 s in 10 ms steps, least squares for gain and offset; the lag with the smallest '
                    'residual wins. Fitted over APPROACH to the end.',
                    'Gain below 1 means the robot under-delivers that axis; the lag is how late it follows. The MPC '
                    'assumes gain 1 and no lag, so these are its model error, and the simulation below runs on them.',
                    'The lag includes the camera stamp offset (10-40 ms) and the 0.4 s fit window\'s smoothing; '
                    'ambiguous stretches (head-on, at range) drop out, so the lateral fit leans on the parts where '
                    'the board was seen obliquely or close.', 'drive'))


def fig_fixes(b, tl, fm, out, figs):
    if not len(b.inf):
        return
    plt = style()
    inf = b.inf
    valid = inf[:, 2] == 1
    rng = np.linalg.norm(inf[:, 6:9], axis=1)
    rows = fm.get('by_range', [])
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0))
    if rows:
        x = [(r['range_lo'] + r['range_hi']) / 2 for r in rows]
        axs[0].bar(x, [100 * r['ambiguous_share'] for r in rows], width=0.08, color=C[0])
        for xi, r in zip(x, rows):
            axs[0].text(xi, 100 * r['ambiguous_share'], f"{r['poses']}", ha='center', va='bottom', fontsize=6, color=INK2)
    axs[0].set_xlabel('range (m)')
    axs[0].set_ylabel('ambiguous poses (%)')
    axs[0].set_title('ambiguity (n above bars)')
    axs[1].plot(rng[valid], inf[valid, 4], 'o', ms=2.2, mew=0, color=C[0], alpha=0.6)
    axs[1].set_xlabel('range (m)')
    axs[1].set_ylabel('reprojection RMS (px)')
    axs[1].set_title('fit residual')
    axs[2].plot(rng[valid], inf[valid, 5], 'o', ms=2.2, mew=0, color=C[0], alpha=0.4)
    axs[2].set_xlabel('range (m)')
    axs[2].set_ylabel('corners used (of 16)')
    axs[2].set_title('corners in the solve')
    fig.tight_layout()
    save(fig, out, 'fixes_range')
    figs.append(Fig('fixes_range', 'Pose quality against range',
                    'For every solved pose in the run: the share that were ambiguous per 0.1 m of range (count on '
                    'each bar), the reprojection RMS and the number of identified corners.',
                    'inference_result as published by the laptop: range |t| of the board in the camera frame; '
                    'ambiguous when IPPE\'s two solutions fit within 1.5x of each other.',
                    'Ambiguity rises with range as the board shrinks (fewer pixels of perspective); a head-on board '
                    'is the worst case. Corners used shows when the detector loses parts of the board.',
                    'Range is the pipeline\'s own estimate, so it carries the board-scale and intrinsics error. The '
                    'mix of ranges depends on where the run went.', 'fixes'))
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    lat = (inf[:, 0] - inf[:, 1]) * 1e3
    ax.hist(lat[np.isfinite(lat)], bins=50, color=C[0], rwidth=0.9, label='capture to pose back on the Pi')
    if len(b.fix):
        fl = (b.fix[:, 0] - b.fix[:, 1]) * 1e3
        ax.hist(fl, bins=50, histtype='step', color=C[1], lw=1.4, label='capture to fiducial/fix')
    ax.set_xlabel('latency (ms)')
    ax.set_ylabel('poses')
    ax.legend()
    fig.tight_layout()
    save(fig, out, 'fixes_latency')
    figs.append(Fig('fixes_latency', 'Capture-to-fix latency',
                    'How long after a frame\'s stamp its pose came back to the Pi (blue), and its fix left the EKF '
                    '(orange outline).',
                    'Bag receive time minus the header stamp. Both are the Pi\'s clock: the stamp is the camera '
                    'node\'s ROS time when it read the frame and the recorder runs on the Pi, so no clock offset '
                    'between the machines enters.',
                    f"Median {fm.get('latency_median_ms')} ms, p90 {fm.get('latency_p90_ms')} ms, max "
                    f"{fm.get('latency_max_ms')} ms. The EKF applies each fix at its capture time and replays the "
                    'IMU since (0.5 s buffer), so latency costs prediction accuracy, not correctness, until it nears 0.5 s.',
                    'The frame stamp is taken after exposure and readout (10-40 ms, 7c item 51), so the true latency '
                    'is that much longer.', 'fixes'))
    jit = fm.get('jitter') or []
    if len(jit) >= 2:
        fig, axs = plt.subplots(1, 2, figsize=(8.4, 3.0))
        r_ = np.array([j['range_m'] for j in jit])
        for ax, keys, unit in ((axs[0], ('x', 'y'), 'mm'), (axs[1], ('th',), 'deg')):
            for n, k in enumerate(keys):
                ax.plot(r_, [j[f'sd_{k}_{unit}'] for j in jit], 'o', ms=4, mew=0, color=C[n],
                        label=f'{k}: measured sd')
                ax.plot(r_, [j[f'R_{k}_{unit}'] for j in jit], '_', ms=8, mew=1.6, color=C[n],
                        label=f'{k}: the fixes\' own R (sd)')
            ax.set_xlabel('range (m)')
            ax.set_ylabel(f'fix jitter, 1 s windows ({unit})')
            ax.set_yscale('log')
            ax.legend(fontsize=6.5)
        fig.tight_layout()
        save(fig, out, 'fixes_jitter')
        figs.append(Fig('fixes_jitter', 'Fix jitter while the robot stood still, against the fixes\' claimed noise',
                        'For each 1 s window in which the robot was still, the scatter of the unambiguous fixes about '
                        'a straight line (dots) and the standard deviation the fixes claimed in R (bars), against range.',
                        'Still: no moving command for 0.5 s and the gyro under 0.02 rad/s. At least 8 unambiguous fixes '
                        'per window; a fitted line is removed so a slow drift is not counted.',
                        'If the dots sit far above the bars, R is too small and the EKF trusts the fixes too much; far '
                        'below, too large.',
                        'Before arming the board or robot may be moved by hand; the line fit removes only slow '
                        'motion. Few windows exist at long range.', 'fixes'))


def fig_imu(b, tl, im, out, figs):
    if len(b.imu) < 10:
        return
    plt = style()
    ts = np.sort(b.imu[:, 1])
    d = np.diff(ts) * 1e3
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 2.8))
    axs[0].hist(np.clip(d, 0, 30), bins=60, color=C[0], rwidth=0.9)
    axs[0].set_yscale('log')
    axs[0].set_xlabel('interval between IMU stamps (ms, clipped at 30)')
    axs[0].set_ylabel('samples')
    sec = np.floor(ts - tl.t0)
    u, c = np.unique(sec, return_counts=True)
    axs[1].plot(u[1:-1], c[1:-1], '-', color=C[0])
    axs[1].axhline(135, color=C[7], lw=1)
    axs[1].text(u[1] if len(u) > 2 else 0, 136, ' 90% of 150 Hz: below it the supervisor will not start',
                fontsize=6.5, color=INK2, va='bottom')
    axs[1].set_xlabel('time since arming (s)')
    axs[1].set_ylabel('samples per second')
    fig.tight_layout()
    save(fig, out, 'imu')
    figs.append(Fig('imu', 'IMU delivery',
                    'Intervals between consecutive imu/data stamps (log count) and samples per second over the bag.',
                    'Header stamps of imu/data (the Pi\'s clock at read). The EKF steps each sample\'s own interval, '
                    'so the rate matters for prediction quality and for the supervisor\'s start check (>= 135 Hz).',
                    f"{im.get('rate_hz')} Hz on average, intervals median {im.get('dt_median_ms')} ms, p99 "
                    f"{im.get('dt_p99_ms')} ms, longest {im.get('dt_max_ms')} ms; {im.get('gaps_over_20ms')} gaps over 20 ms.",
                    'The BNO085 over I2C tops out near 170 Hz (7c item 1); bursts of short intervals follow a late '
                    'read.', 'imu'))


def fig_sim(b, tl, sim, cam, out, figs):
    if not sim or '_log' not in sim:
        return
    plt = style()
    log = sim['_log']
    ts = log[:, 0] - sim['_t_app']
    fig = plt.figure(figsize=(9.2, 5.4))
    gs = fig.add_gridspec(3, 2, width_ratios=(1.0, 1.25))
    ax = fig.add_subplot(gs[:, 0])
    e = b.est[(b.est[:, 1] >= tl.approach) & (b.est[:, 1] <= tl.t_cmd + 0.5)]
    if cam is not None:
        k = cam['t'] >= tl.approach
        ax.plot(cam['y'][k], cam['x'][k], '-', color=C[1], lw=1.6, label='robot (camera)')
    if len(e):
        ax.plot(e[:, 3], e[:, 2], '-', color=INK, lw=1.0, alpha=0.7, label='robot (EKF)')
    ax.plot(log[:, 2], log[:, 1], '-', color=C[0], lw=1.6, label='simulation')
    ax.plot([0], [0], 'x', ms=10, mew=2, color=C[7])
    ax.set_xlabel('y (m)')
    ax.set_ylabel('x (m)')
    ax.invert_xaxis()
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(loc='best')
    for k, (lab, sc) in enumerate((('x (m)', 1), ('y (m)', 1), ('heading (deg)', 180 / math.pi))):
        a = fig.add_subplot(gs[k, 1])
        if cam is not None:
            kk = cam['t'] >= tl.approach - 0.5
            a.plot(cam['t'][kk] - tl.approach, cam[('x', 'y', 'th')[k]][kk] * sc, '-', color=C[1], lw=1.4)
        a.plot(ts, log[:, 1 + k] * sc, '-', color=C[0], lw=1.4)
        a.set_ylabel(lab)
        if k == 2:
            a.set_xlabel('time since APPROACH began (s)')
    fig.suptitle(f'{b.name}: the robot against the simulation of its configuration and drive', fontsize=9.5)
    fig.tight_layout()
    save(fig, out, 'sim_vs_real')
    mdl = sim['model']
    figs.append(Fig('sim_vs_real', 'The run against its simulation',
                    'The robot\'s path from APPROACH on (orange: camera; black: EKF) and the simulated one (blue) from '
                    'the same start, top-down and as x, y, heading against time.',
                    f"run_analytics_sim.py with commit {mdl['commit']}'s Supervisor, Mpc, Ekf and fiducial chain on a "
                    f"plant with this run's drive model: gains {mdl['gain']} and time constants {mdl['tau_s']} s after a "
                    f"{mdl['delay_cmd_s']} s command delay (forward, lateral, turn), deadband {mdl['deadband']} "
                    f"({mdl['deadband_basis']}), fixes {mdl['fix_delay_s']} s late within a {mdl['fov_half_deg']:.0f} deg "
                    'half field of view, bench fix noise.',
                    'Where the paths agree, the measured drive model explains the run; where they part, something '
                    'the simulation leaves out mattered (see the list above the figure).',
                    'The simulation has no ambiguous poses, no fix bias, no wheel slip or crab, no contact with the '
                    'coupler, and constant drive gains; one noise seed.', 'sim'))


# ============================================================================================ the record
def describe_run(b: Bag) -> str:
    if b.name.startswith('drivecal'):
        return 'drivecal'
    if len(b.stage) and np.any(b.stage[:, 1] != IDLE):
        return 'dock'
    return 'bringup'


def analyse(bag_path: Path, out_root: Path, sim_on: bool = True, live: dict | None = None) -> dict:
    t_start = time.time()
    b = read_bag(bag_path)
    kind = describe_run(b)
    tl = timeline(b)
    out = out_root / b.name
    out.mkdir(parents=True, exist_ok=True)
    figs: list[Fig] = []
    M = {'run': b.name, 'kind': kind, 'version': VERSION, 'bag': str(bag_path), 'bag_closed': b.closed,
         'bag_fingerprint': bag_fingerprint(bag_path),
         'counts': b.counts,
         'time': {'start_utc': utc(b.start).strftime('%Y-%m-%d %H:%M:%S'),
                  'start_nzdt': nz(b.start).strftime('%Y-%m-%d %H:%M:%S'),
                  'armed_nzdt': nz(tl.t0).strftime('%H:%M:%S') if tl.armed else None,
                  'recorded_s': fin(b.end - b.start, 2), 'armed_s': fin(tl.t_end - tl.t0, 2) if tl.armed else 0.0}}
    cfg = deployed_config(b, tl, replay=(kind == 'dock'))
    if tl.outcome == 'ABORT':
        tl.abort_cause = abort_cause(b, tl.segments, cfg)
    M['config'] = {k: v for k, v in cfg.items() if k != '_params'}
    M['detector'] = deployed_detector(tl.t0, live, b.name)
    M['board_square_m'] = board_square(tl.t0)
    try:
        from run_analytics_detectors import camera_at
        cam_epoch = camera_at(tl.t0)
        M['camera'] = {'calibration': cam_epoch[1], 'fx': round(cam_epoch[2][0][0], 2)}
    except (ImportError, subprocess.CalledProcessError) as ex:       # the wireless repo's history unreadable
        M['camera'] = {'calibration': f'unknown ({ex})', 'fx': None}
    gate = cfg['_params']['fid']['gate']
    lo, hi = (tl.t0, tl.t_end) if tl.armed else (b.start, b.end)
    cam = camera_track(b, lo, hi)
    M['imu'] = analyse_imu(b, tl)
    M['fixes'] = analyse_fixes(b, tl)
    M['ekf'], ed = analyse_ekf(b, tl, cam, gate)
    if kind == 'bringup' and M['ekf'].get('estimates'):
        M['ekf'] = {'estimates': M['ekf']['estimates'], 'p_sd_max_mm': M['ekf'].get('p_sd_max_mm'), 'idle': True,
                    'residual_rms_mm': [(M['ekf'].get('residual') or {}).get(k) for k in ('x_rms_mm', 'y_rms_mm')]}
        ed = {}
    M['run_metrics'] = stage_metrics(b, tl, cam)
    M['mpc'], md = analyse_mpc(b, tl, cfg) if kind == 'dock' else ({}, {})
    M['drive'], dd = analyse_drive(b, tl, cam) if kind == 'dock' else ({}, {})
    if kind == 'drivecal':
        try:
            M['drivecal'] = analyse_drivecal(b)
        except Exception as ex:                 # a run with too few steps
            M['drivecal'] = {'error': str(ex)}
    sim = None
    if kind == 'dock' and sim_on and M['run_metrics'].get('start') and len(b.mpc):
        sim = simulate_run(b, tl, cfg, M['drive'], M['fixes'], M['run_metrics'])
        if sim:
            M['sim'] = {k: v for k, v in sim.items() if not k.startswith('_')}
            M['sim']['verdicts'] = sim_verdicts(M['run_metrics'], M['sim'], M['ekf'], M['mpc'])
    # figures
    if len(b.est) and tl.armed:
        fig_trajectory(b, tl, cfg, md, out, figs)
        fig_timeline(b, tl, M['ekf'], out, figs)
    fig_ekf(b, tl, M['ekf'], ed, gate, out, figs)
    fig_mpc(b, tl, M['mpc'], md, out, figs)
    fig_drive(b, tl, M['drive'], dd, out, figs)
    fig_fixes(b, tl, M['fixes'], out, figs)
    fig_imu(b, tl, M['imu'], out, figs)
    fig_sim(b, tl, sim, cam, out, figs)
    if kind == 'drivecal' and 'axes' in M.get('drivecal', {}):
        fig_drivecal(M['drivecal'], out, figs)
    # data
    write_data(b, tl, out, ed, md, dd, sim, cam, M.get('drivecal'))
    M['figures'] = [f.__dict__ for f in figs]
    M['seconds_to_analyse'] = round(time.time() - t_start, 1)
    (out / 'metrics.json').write_text(json.dumps(M, indent=1, default=lambda o: fin(o) if isinstance(o, float) else str(o)))
    (out / 'README.md').write_text(run_readme(M, figs))
    return M


def fig_drivecal(dc, out, figs):
    plt = style()
    tab = dc['table']
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for ax, axis, name, unit in zip(axs, 'xyr', ('forward', 'left', 'turn'), ('m/s', 'm/s', 'rad/s')):
        rows = [r for r in tab if r[0] == axis and r[2] is not None and math.isfinite(r[2])]
        if rows:
            c = np.array([abs(r[1]) for r in rows])
            s = np.array([r[2] for r in rows])
            pos = np.array([r[1] > 0 for r in rows])
            ax.plot(c[pos], s[pos], 'o', ms=5, color=C[0], label='positive')
            ax.plot(c[~pos], s[~pos], 'o', ms=5, color=C[1], label='negative')
            lim = max(c.max(), 1e-3) * 1.1
            ax.plot([0, lim], [0, lim], '-', color=MUTED, lw=1, label='1:1')
            ax.legend()
        ax.set_xlabel(f'|command| ({unit})')
        ax.set_ylabel(f'response along it ({unit})')
        ax.set_title(name)
    fig.tight_layout()
    save(fig, out, 'drivecal')
    figs.append(Fig('drivecal', 'Drive calibration steps',
                    'Each single-axis step\'s steady response against its command, both directions, with the 1:1 line.',
                    'tools/drive_calibration_fit.py: the fixes\' straight-line motion over each step\'s steady part '
                    '(0.5 s after the command to its end) for forward and left, the bias-free gyro for turns.',
                    'Points below the line under-deliver; a cluster at zero on the left is the deadband.',
                    'Lateral responses from ambiguous fixes are suspect (head-on board; 7c item 54); the accelerometer '
                    'based fits of drive_wheel_fit.py are the better reference for the firmware constants.', 'drive'))


def write_data(b, tl, out, ed, md, dd, sim, cam, dc=None):
    dd_ = out / 'data'
    if len(b.est):
        e = b.est
        keep = np.r_[True, np.diff(np.floor(e[:, 1] * 20)) > 0]             # 20 Hz
        write_csv(dd_ / 'estimate.csv', ['t', 'x', 'y', 'theta', 'vx', 'vy', 'P_x', 'P_y', 'P_th', 'P_vx', 'P_vy',
                                         'r_gyro', 'imu_rate', 'fix_age', 'last_nis', 'last_accepted'],
                  [[r[1] - tl.t0, *r[2:17]] for r in e[keep]])
    if len(b.fix):
        write_csv(dd_ / 'fixes.csv', ['t_capture', 't_received', 'x', 'y', 'theta', 'R_xx', 'R_yy', 'R_thth',
                                      'range', 'ambiguous', 'cov_valid'],
                  [[r[1] - tl.t0, r[0] - tl.t0, r[2], r[3], r[4], r[5], r[9], r[13], r[18], int(r[19]), int(r[15])]
                   for r in b.fix])
    if 'nis' in ed:
        write_csv(dd_ / 'nis.csv', ['t', 'nis', 'accepted', 'stage', 'ambiguous', 'range'],
                  [[r[0] - tl.t0, r[1], int(r[2]), STAGES[int(r[3])], r[4], r[5]] for r in ed['nis']])
    if 'tick' in md:
        tk = md['tick']
        write_csv(dd_ / 'mpc_ticks.csv', ['t', 'stage', 'u_b', 'v_b', 'r_b', 'v_max', 'r_max', 'solve_ms', 'fallback',
                                          'slack', 'plan_vs_goal_deg', 'max_wheel_rad_s', 'at_L1_cap', 'at_forward_cap',
                                          'at_lateral_cap', 'at_turn_cap', 'at_wheel_limit', 'est_x', 'est_y', 'est_th'],
                  [[r[0] - tl.t0, STAGES[int(r[1])], *r[2:]] for r in tk])
    if 'drive' in dd:
        t, tg, cmd, cu, cv, gyro, mv = dd['drive']
        c = np.c_[[np.interp(t, tg, cmd[:, k]) for k in range(3)]].T
        write_csv(dd_ / 'drive.csv', ['t', 'cmd_u', 'cmd_v', 'cmd_r', 'cam_u', 'cam_v', 'gyro_r', 'in_fit'],
                  [[t[i] - tl.t0, *c[i], cu[i], cv[i], gyro[i], int(mv[i])] for i in range(len(t))])
    if tl.segments:
        write_csv(dd_ / 'stages.csv', ['stage', 't_start', 't_end'],
                  [[STAGES[s], a - tl.t0, c - tl.t0] for s, a, c in tl.segments])
    if dc and dc.get('table'):
        write_csv(dd_ / 'drivecal_steps.csv', ['axis', 'command', 'response', 'fixes', 'ambiguous_used', 'notes'], dc['table'])
    if sim and '_log' in sim:
        write_csv(dd_ / 'sim.csv', ['t_since_approach', 'x', 'y', 'theta', 'u_b', 'v_b', 'r_b', 'stage', 'est_x',
                                    'est_y', 'est_theta', 'est_vx', 'est_vy'],
                  [[r[0] - sim['_t_app'], *r[1:7], STAGES[int(r[7])], *r[8:]] for r in sim['_log']])


# ============================================================================================ README writing
def table(rows: list, head: list) -> str:
    out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def fig_md(f: Fig, n: int) -> str:
    return (f'![{f.title}](figures/{f.name}.png)\n\n**Figure {n}. {f.title}.** ([PDF](figures/{f.name}.pdf))\n\n'
            f'- *What it shows:* {f.what}\n- *How it was computed:* {f.how}\n- *What it means:* {f.means}\n'
            f'- *Caveats:* {f.caveats}\n')


def run_readme(M: dict, figs: list) -> str:
    c, dt, rm = M['config'], M['detector'], M['run_metrics']
    s = c['settings']
    L = [f"# {M['run']}", '']
    kind = {'dock': 'Docking attempt', 'bringup': 'Bring-up recording (docking never armed)',
            'drivecal': 'Drive calibration run (tools/drive_calibration_run.py)'}[M['kind']]
    L.append(f"{kind}. Recorded {M['time']['start_nzdt']} NZDT ({M['time']['start_utc']} UTC), "
             f"{M['time']['recorded_s']} s" + (f", armed for {M['time']['armed_s']} s" if M['time']['armed_s'] else '')
             + ('' if M['bag_closed'] else '; the recorder never closed this bag (no metadata.yaml), so it was read '
                                           'from its .mcap directly') + '.')
    L.append('')
    if M['kind'] == 'dock':
        out = rm['outcome']
        L.append(f"**Outcome: {out}**" + (f" ({rm['abort_cause']})" if rm.get('abort_cause') else '')
                 + (f", {rm['approach_to_docked_s']} s after APPROACH began" if rm.get('approach_to_docked_s') else '')
                 + f". Stages: {' -> '.join(rm.get('stages', []))}.")
        L.append('')
    L.append(table([
        ['controller', f"{c['commit']}{' (' + c['variant'] + ')' if c.get('variant') else ''} \"{c['subject']}\", "
                       f"committed {c['commit_nzdt']} NZDT " + (f"({c['minutes_before_run']} min before the run)"
                       if c['minutes_before_run'] >= 0 else (f"({-c['minutes_before_run']} min after the run: the base of "
                       'the reconstruction below)' if c.get('variant') or c['minutes_before_run'] < -15 else
                       f"({-c['minutes_before_run']} min after the run began: deployed before it was committed)"))],
        ['how sure', c['certainty']],
        ['key settings', f"standoff {s['standoff_m']} m, v_approach {s['v_approach']} m/s, v_final {s['v_final']} m/s, "
                         f"r_max {s['r_max']} rad/s, w_max {s['w_max']} rad/s, Np {s['Np']}, rw {s['rw']}, qw {s['qw']}, "
                         f"qw_final {s['qw_final']}, speed cap {s['speed_cap']}, Qy {s['Qy']}, marker {s['marker_m']}, "
                         f"gate {s['gate']}"],
        ['seen in the bag', ', '.join(f'{k} {v}' for k, v in c['observed'].items()) or 'nothing (no docking stages)'],
        ['detector', f"{dt['name']} ({Path(dt['file']).name}); {dt['source']}"
                     + ('; ' + '; '.join(dt['notes']) if dt['notes'] else '')],
        ['board', f"{M['board_square_m'] * 1000:.1f} mm squares"],
        ['camera matrix', f"{M['camera']['calibration']}: the one the live pipeline used then (for the image bags of 13:21-13:40 "
                          'NZDT this was checked against their logged poses)']], ['', '']))
    L.append('')
    L.append(headline_md(M))
    sections = [('trajectory', 'Trajectory'), ('ekf', 'EKF'), ('mpc', 'MPC'), ('drive', 'Drive'),
                ('fixes', 'Inference and fixes'), ('imu', 'IMU'), ('sim', 'Simulation against the run')]
    n = 0
    for key, title in sections:
        body = section_md(key, M)
        fs = [f for f in figs if f.section == key]
        if not body and not fs:
            continue
        L += ['', f'## {title}', '']
        if body:
            L.append(body)
            L.append('')
        for f in fs:
            n += 1
            L.append(fig_md(f, n))
    L += ['', '## Files', '', '- `metrics.json`: every number above, and the configuration candidates with their '
          'replay match.', '- `figures/`: each figure as PNG and PDF.', '- `data/`: the CSVs behind the figures '
          '(times in seconds since arming, on the Pi clock).', '',
          f"Generated by `controller_ws/tools/run_analytics.py` (version {M['version']}) from `{M['bag']}`."]
    return '\n'.join(L) + '\n'


def headline_md(M: dict) -> str:
    e, mp, dr, fx, rm = M['ekf'], M.get('mpc', {}), M.get('drive', {}), M['fixes'], M['run_metrics']
    rows = []
    if M['kind'] == 'dock':
        rows.append(['time to dock (APPROACH to DOCKED)', fmt(rm.get('approach_to_docked_s'), '.1f', ' s')])
        rows.append(['ACQUIRE took', fmt(rm.get('acquire_s'), '.1f', ' s')])
        st = rm.get('start')
        if st:
            rows.append(['start (camera, at APPROACH)', f"x {st['x_m']} m, y {st['y_m']} m, heading {st['th_deg']} deg"])
        rows.append(['lateral overshoot (camera)', fmt(rm.get('lateral_overshoot_mm'), '.0f', ' mm')])
        f = rm.get('final')
        if f:
            rows.append(['final pose (camera, last 0.5 s)', f"x {f['x_mm']} mm, y {f['y_mm']} mm, heading {f['th_deg']} deg"])
        rows.append(['HOLDs', rm.get('holds', 0)])
    v = (e.get('velocity') or {})
    if v:
        rows.append(['EKF velocity vs camera (forward / lateral)',
                     f"RMS {fmt((v.get('x') or {}).get('rms_mm_s'), '.0f')} / {fmt((v.get('y') or {}).get('rms_mm_s'), '.0f')} mm/s, "
                     f"slope {fmt((v.get('x') or {}).get('slope'), '.2f')} / {fmt((v.get('y') or {}).get('slope'), '.2f')}"])
    if e.get('nis'):
        rows.append(['fixes accepted (gated stages)', f"{fmt(e['nis']['accepted_share_gated'], '.0%')} of {e['nis']['gated']}; "
                                                      f"NIS median {e['nis']['median']}"])
    if 'reanchors' in e:
        rows.append(['EKF re-anchors / largest estimate step', f"{e['reanchors']} / {fmt(e.get('step_max_mm'), '.0f', ' mm')}"])
    ax = dr.get('axes') or {}
    if ax:
        rows.append(['drive delivered (forward / lateral / turn)',
                     ' / '.join(f"{fmt(ax.get(k, {}).get('gain'), '.2f', 'x')} at {fmt(ax.get(k, {}).get('lag_s'), '.2f', ' s')}"
                                for k in ('forward', 'lateral', 'turn'))])
    if mp.get('solve_ms'):
        rows.append(['MPC solve (median / p99 / max)', f"{mp['solve_ms']['median']} / {mp['solve_ms']['p99']} / {mp['solve_ms']['max']} ms"])
    if fx.get('inference_results'):
        rows.append(['poses: rate / ambiguous / latency', f"{fx.get('pose_rate_hz')} Hz / {fmt(fx.get('ambiguous_share'), '.0%')} / "
                                                         f"{fx.get('latency_median_ms')} ms median"])
    if M['imu'].get('rate_hz'):
        rows.append(['IMU', f"{M['imu']['rate_hz']} Hz, longest gap {M['imu'].get('dt_max_ms')} ms"])
    if M.get('sim'):
        sm = M['sim']
        rows.append(['simulation of this run', f"{sm['outcome']}" + (f" in {sm['approach_to_docked_s']} s" if sm.get('approach_to_docked_s') else '')])
    return '## Headline numbers\n\n' + table(rows, ['', '']) if rows else ''


def section_md(key: str, M: dict) -> str:
    e, mp, dr, fx, im = M['ekf'], M.get('mpc', {}), M.get('drive', {}), M['fixes'], M['imu']
    if key == 'trajectory':
        rm = M['run_metrics']
        if not rm.get('time_in_stage_s'):
            return ''
        return 'Time in each stage: ' + ', '.join(f'{k} {v} s' for k, v in rm['time_in_stage_s'].items()) + '.'
    if key == 'ekf':
        if not e.get('estimates'):
            return 'No estimates in this bag.'
        if e.get('idle'):
            return (f"Docking was never armed, but this code version ran the EKF anyway ({e['estimates']} estimates). Unarmed it "
                    f"drifted: its position sd reached {fmt(e.get('p_sd_max_mm'), '.0f')} mm and its distance from the fixes "
                    f"{fmt((e.get('residual_rms_mm') or [None])[0], '.0f')} mm RMS in x. This idle drift is why the filter now "
                    'runs only while docking is armed (7c item 4); it is not evidence about the docking estimator, so no '
                    'EKF figures are drawn for this recording.')
        L = []
        r = e.get('residual')
        if r:
            L.append(f"- Estimate minus unambiguous fix at its capture time ({r['fixes']} fixes): RMS x {r['x_rms_mm']} mm, "
                     f"y {r['y_rms_mm']} mm, heading {r['th_rms_deg']} deg; while moving, x {r['moving_x_rms_mm']} mm, "
                     f"y {r['moving_y_rms_mm']} mm.")
            L.append(f"- Consistency: residual^2 / (P + R) averages {r['z2_mean']} (x, y, heading; 1 if the claimed "
                     f"variances were right), {r['within_2sd']} within 2 sd (0.95 if right).")
        for k, name in (('x', 'forward (x)'), ('y', 'lateral (y)')):
            v = (e.get('velocity') or {}).get(k)
            va = (e.get('velocity_all_fixes') or {}).get(k)
            if v:
                L.append(f"- Velocity {name} against the camera: RMS error {v['rms_mm_s']} mm/s while the camera saw up to "
                         f"{v['camera_p95_mm_s']} mm/s (p95), slope {v['slope']}, EKF {v['lag_s']} s behind ({v['n']} samples)"
                         + (f"; with the ambiguous fixes kept in the camera track, as vel_truth.py does: {va['rms_mm_s']} mm/s, "
                            f"slope {va['slope']} ({va['n']} samples)" if va else '') + '.')
        if e.get('velocity_all_fixes'):
            L.append('- The two camera tracks bracket the real velocity error: the first leaves out the stretches where every '
                     'fix was ambiguous (head-on, at range), which is where the EKF struggles; the second keeps the solution '
                     'flips in its truth.')
        n = e.get('nis')
        if n:
            L.append(f"- Fixes processed {n['processed']}, {n['gated']} in gated stages, {fmt(n['accepted_share_gated'], '.0%')} accepted "
                     f"({n['rejected_gated']} rejected). NIS median {n['median']} (2.37 expected), unambiguous {n['median_unambiguous']}, "
                     f"ambiguous {n['median_ambiguous']}; {fmt(n['share_over_chi2_95'], '.0%')} over the chi-square 95% point.")
        L.append(f"- Re-anchors: {e.get('reanchors', 0)}" + (': ' + '; '.join(
            f"{x['t']} s in {x['stage']} ({x['dx_mm']:+.0f}, {x['dy_mm']:+.0f}) mm {x['dth_deg']:+.1f} deg"
            for x in e.get('reanchor_list', [])[:10]) if e.get('reanchor_list') else '') + '.')
        L.append(f"- Estimate steps over 10 mm between consecutive estimates: {e.get('steps_over_10mm')}, largest "
                 f"{e.get('step_max_mm')} mm" + (' (' + ', '.join(f"{s['mm']:.0f} mm at {s['t']} s" for s in e.get('step_list', [])[:5]) + ')'
                                                if e.get('step_list') else '') + f". Largest position sd: {e.get('p_sd_max_mm')} mm.")
        return '\n'.join(L)
    if key == 'mpc':
        if not mp.get('ticks'):
            return ''
        L = [f"- {mp['ticks']} ticks, {mp['ticks_commanding']} commanding. Solve {mp['solve_ms']['median']} ms median, "
             f"{mp['solve_ms']['p99']} ms p99, {mp['solve_ms']['max']} ms max; fallbacks {mp['fallbacks']}, ticks with "
             f"slack {mp['slack_ticks']} (max {mp['slack_max']}); exit flags {mp['exit_flags']}."]
        if mp.get('limits_active_by_stage'):
            for stg, v in mp['limits_active_by_stage'].items():
                L.append(f"- At a limit in {stg}: " + ', '.join(f'{k} {fmt(x, ".0%")}' for k, x in v.items()) + '.')
        L.append(f"- Smoothness: RMS change per tick {mp['du_rms']} (u_b, v_b m/s; r_b rad/s); sign reversals per 10 s "
                 f"{mp['reversals_per_10s']}.")
        p = mp['pointing']
        L.append(f"- Plan pointing: median {p['median_deg']} deg from the goal, {fmt(p['share_over_45deg'], '.0%')} of ticks over "
                 f"45 deg ({p['ticks']} ticks).")
        s = mp['strafe']
        L.append(f"- Strafe: {s['integral_m']} m of commanded sideways motion (forward {s['forward_integral_m']} m), peak "
                 f"{s['max_m_s']} m/s, lateral larger than forward on {fmt(s['share_lateral_dominant'], '.0%')} of ticks.")
        return '\n'.join(L)
    if key == 'drive':
        if M['kind'] == 'drivecal' and M.get('drivecal', {}).get('axes'):
            dc = M['drivecal']
            rows = [[k, v['steps'], fmt(v['deadband'], '.3f'), fmt(v['gain_above_deadband'], '.2f'), fmt(v['fit_rms'], '.3f'),
                     fmt(v['pulse_deadband_us'], '.0f'), fmt(v['per_us'], '.3f'), v['moved_against']]
                    for k, v in dc['axes'].items()]
            return (f"{dc['steps']} single-axis steps, fitted with drive_calibration_fit.py: response = gain x (|command| - "
                    'deadband), in m/s or rad/s; in pulse space, the same against the pulse width\'s distance from 1500 us '
                    f"({'logged' if dc.get('pulses_logged') else 'worked out from the command; this firmware did not log them'}).\n\n"
                    + table(rows, ['axis', 'steps', 'deadband', 'gain above it', 'fit RMS', 'pulse deadband (us)',
                                   'per us (mm/s, mrad/s)', 'steps that moved the wrong way'])
                    + '\n\nThe per-step table is in data/drivecal_steps.csv. Lateral responses come from the fixes and are '
                    'suspect where they were ambiguous (head-on board); drive_wheel_fit.py\'s accelerometer fits are the '
                    'reference for the firmware constants (7c items 54-56).')
        ax = dr.get('axes')
        if not ax:
            return ''
        rows = [[k, fmt(v.get('gain'), '.2f'), fmt(v.get('lag_s'), '.2f'), fmt(v.get('offset'), '+.3f'), fmt(v.get('resid_rms'), '.3f'),
                 fmt(v.get('r2'), '.2f'), v.get('n'), fmt(v.get('src_p95'), '.3f'), v.get('note') or v.get('flag') or '']
                for k, v in ax.items()]
        L = [table(rows, ['axis', 'gain', 'lag (s)', 'offset', 'residual RMS', 'R^2', 'samples', 'command p95', 'note'])]
        t = dr.get('telemetry')
        if t:
            L.append('')
            L.append(f"Telemetry: {t['rows']} rows while armed, saturated {fmt(t['saturated_share'], '.1%')}, watchdog timed out "
                     f"{fmt(t['timed_out_share'], '.1%')}" + (f", the Mega's copy of the request {t['request_lag_s']} s after cmd_vel"
                                                              if t.get('request_lag_s') is not None else '')
                     + (f", applied/requested {t['applied_over_requested_median']} median" if t.get('applied_over_requested_median') else '') + '.')
        return '\n'.join(L)
    if key == 'fixes':
        if not fx.get('inference_results'):
            return f"No inference results in this bag ({fx.get('images', 0)} camera frames)." if fx.get('images') else ''
        L = [f"- {fx['inference_results']} inference results ({fx.get('result_rate_hz')} Hz), {fmt(fx.get('valid_share'), '.1%')} with a pose "
             f"({fx.get('pose_rate_hz')} Hz); refusals {fx.get('refusals') or 'none'}.",
             f"- Ambiguous: {fmt(fx.get('ambiguous_share'), '.1%')} of poses. Range {fx.get('range_min_m')}-{fx.get('range_max_m')} m "
             f"(median {fx.get('range_median_m')} m); reprojection RMS median {fx.get('rms_px_median')} px; corners used median "
             f"{fx.get('corners_median')}.",
             f"- Latency, capture to pose on the Pi: median {fx.get('latency_median_ms')} ms, p90 {fx.get('latency_p90_ms')} ms, "
             f"max {fx.get('latency_max_ms')} ms; capture to fix {fx.get('fix_latency_median_ms')} ms median."]
        if fx.get('fix_rate_armed_hz') is not None:
            L.append(f"- Fixes while armed: {fx['fix_rate_armed_hz']} Hz, longest gap {fx.get('fix_gap_max_s')} s, "
                     f"{fx.get('fix_gaps_over_0p5s')} gaps over 0.5 s (HOLD's threshold).")
        if fx.get('images'):
            L.append(f"- Camera frames in the bag: {fx['images']} ({fx.get('image_rate_hz')} Hz); see the detector comparison "
                     'in the index for the three detectors on them.')
        rows = [[f"{r['range_lo']:.1f}-{r['range_hi']:.1f}", r['poses'], fmt(r['ambiguous_share'], '.0%'), r['rms_px_median'],
                 r['corners_median']] for r in fx.get('by_range', [])]
        if rows:
            L += ['', table(rows, ['range (m)', 'poses', 'ambiguous', 'RMS px (median)', 'corners (median)'])]
        return '\n'.join(L)
    if key == 'imu':
        if not im.get('rate_hz'):
            return ''
        return (f"- {im['samples']} samples, {im['rate_hz']} Hz; per-second counts {im.get('rate_1s_min')}-{im.get('rate_1s_max')} "
                f"(5th percentile {im.get('rate_1s_p5')}); intervals median {im['dt_median_ms']} ms, p99 {im['dt_p99_ms']} ms, "
                f"max {im['dt_max_ms']} ms; {im['gaps_over_20ms']} gaps over 20 ms, {im['repeated_stamps']} repeated stamps.\n"
                f"- Stamp to bag: median {im['recv_delay_median_ms']} ms, p99 {im['recv_delay_p99_ms']} ms. The EKF measured "
                f"{im.get('ekf_imu_rate_median', 'n/a')} Hz median (min {im.get('ekf_imu_rate_min', 'n/a')}); "
                f"{im.get('low_rate_warnings', 0)} low-rate warnings.")
    if key == 'sim':
        sm = M.get('sim')
        if not sm:
            return ('Not simulated: ' + ('the run never reached APPROACH.' if M['kind'] == 'dock' else
                                         'not a docking attempt.')) if M['kind'] == 'dock' else ''
        mdl = sm['model']
        rm = M['run_metrics']
        rows = [['outcome', rm.get('outcome'), sm['outcome']],
                ['APPROACH to DOCKED', fmt(rm.get('approach_to_docked_s'), '.1f', ' s'), fmt(sm.get('approach_to_docked_s'), '.1f', ' s')],
                ['lateral overshoot', fmt(rm.get('lateral_overshoot_mm'), '.0f', ' mm'), fmt(sm.get('lateral_overshoot_mm'), '.0f', ' mm')],
                ['strafe commanded (integral)', fmt((M['mpc'].get('strafe') or {}).get('integral_m'), '.2f', ' m'),
                 fmt(sm['strafe'].get('integral_m'), '.2f', ' m')],
                ['peak strafe command', fmt((M['mpc'].get('strafe') or {}).get('max_m_s'), '.2f', ' m/s'),
                 fmt(sm['strafe'].get('max_m_s'), '.2f', ' m/s')],
                ['lateral larger than forward', fmt((M['mpc'].get('strafe') or {}).get('share_lateral_dominant'), '.0%'),
                 fmt(sm['strafe'].get('share_lateral_dominant'), '.0%')],
                ['final pose (x, y, heading)', final_str(rm.get('final')), final_str(sm.get('final'))],
                ['HOLDs', rm.get('holds', 0), sm.get('holds', 0)]]
        L = [f"Simulated from the camera's pose at APPROACH entry with commit {mdl['commit']}"
             + (f" ({mdl['variant']})" if mdl.get('variant') else '') + f" and this run's drive model (gains {mdl['gain']}, "
             f"lags {mdl['lag_s']} s, forward/lateral/turn)"
             + (', with the settings the bag shows where they differ from the file: ' + ', '.join(
                 f'{k} {v[1]} (file {v[0]})' for k, v in mdl.get('overrides_from_bag', {}).items()) if mdl.get('overrides_from_bag') else '')
             + f". Start pose from {mdl.get('start_source') or 'the camera'}. " + ' '.join(mdl['notes']), '',
             table(rows, ['', 'robot', 'simulation']), '', '**Where the simulation does and does not reproduce the run:**', '']
        L += [f'- {v}' for v in sm.get('verdicts', [])]
        L.append('- Always left out: ambiguous poses and the EKF\'s handling of them, bias in the fixes, wheel slip and crab, '
                 'drive gains that change with speed or battery, and the coupler contact.')
        return '\n'.join(L)
    return ''


def final_str(f):
    return f"({f['x_mm']:.0f}, {f['y_mm']:.0f}) mm, {f['th_deg']:.1f} deg" if f else 'n/a'


# ============================================================================================ index
def aggregate(out_root: Path) -> None:
    """INDEX.md, aggregate.csv and the cross-run figures from every <run>/metrics.json."""
    runs = []
    for p in sorted(out_root.glob('*/metrics.json')):
        try:
            runs.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            print(f'  {p}: unreadable, skipped')
    runs.sort(key=lambda r: r['time']['start_utc'])
    flat = [flatten_run(r) for r in runs]
    if flat:
        keys = list(dict.fromkeys(k for f in flat for k in f))
        write_csv(out_root / 'aggregate.csv', keys, [[f.get(k, '') for k in keys] for f in flat])
    figs = aggregate_figures(runs, out_root)
    (out_root / 'INDEX.md').write_text(index_md(runs, out_root, figs))
    (out_root / 'README.md').write_text(PIPELINE_README)


def flatten_run(r: dict) -> dict:
    e, mp, dr, fx, rm, sm = r['ekf'], r.get('mpc') or {}, r.get('drive') or {}, r['fixes'], r['run_metrics'], r.get('sim') or {}
    ax = dr.get('axes') or {}
    v = e.get('velocity') or {}
    f = {'run': r['run'], 'kind': r['kind'], 'start_nzdt': r['time']['start_nzdt'], 'start_utc': r['time']['start_utc'],
         'armed_s': r['time']['armed_s'], 'outcome': rm.get('outcome'), 'abort_cause': rm.get('abort_cause', ''),
         'commit': r['config']['commit'], 'config_variant': r['config'].get('variant') or '',
         'config_certainty': r['config']['certainty'].split(':')[0], 'detector': r['detector']['name'],
         'board_square_mm': r['board_square_m'] * 1000, 'standoff_m': r['config']['settings']['standoff_m'],
         'speed_cap': r['config']['settings']['speed_cap'],
         'acquire_s': rm.get('acquire_s'), 'approach_to_docked_s': rm.get('approach_to_docked_s'),
         'holds': rm.get('holds'), 'start_x_m': (rm.get('start') or {}).get('x_m'), 'start_y_m': (rm.get('start') or {}).get('y_m'),
         'start_th_deg': (rm.get('start') or {}).get('th_deg'), 'lateral_overshoot_mm': rm.get('lateral_overshoot_mm'),
         'final_x_mm': (rm.get('final') or {}).get('x_mm'), 'final_y_mm': (rm.get('final') or {}).get('y_mm'),
         'final_th_deg': (rm.get('final') or {}).get('th_deg'),
         'ekf_vx_rms_mm_s': (v.get('x') or {}).get('rms_mm_s'), 'ekf_vy_rms_mm_s': (v.get('y') or {}).get('rms_mm_s'),
         'ekf_vx_slope': (v.get('x') or {}).get('slope'), 'ekf_vy_slope': (v.get('y') or {}).get('slope'),
         'ekf_vy_lag_s': (v.get('y') or {}).get('lag_s'),
         'ekf_vy_rms_all_fixes_mm_s': ((e.get('velocity_all_fixes') or {}).get('y') or {}).get('rms_mm_s'),
         'ekf_vy_slope_all_fixes': ((e.get('velocity_all_fixes') or {}).get('y') or {}).get('slope'),
         'resid_y_rms_mm': (e.get('residual') or {}).get('y_rms_mm'),
         'nis_median': (e.get('nis') or {}).get('median'), 'fix_accept_share': (e.get('nis') or {}).get('accepted_share_gated'),
         'reanchors': e.get('reanchors'), 'est_step_max_mm': e.get('step_max_mm'),
         'mpc_solve_median_ms': (mp.get('solve_ms') or {}).get('median'), 'mpc_solve_p99_ms': (mp.get('solve_ms') or {}).get('p99'),
         'mpc_solve_max_ms': (mp.get('solve_ms') or {}).get('max'), 'mpc_fallbacks': mp.get('fallbacks'),
         'plan_over_45deg_share': (mp.get('pointing') or {}).get('share_over_45deg'),
         'strafe_integral_m': (mp.get('strafe') or {}).get('integral_m'), 'strafe_max_m_s': (mp.get('strafe') or {}).get('max_m_s')}
    for k in ('forward', 'lateral', 'turn'):
        f[f'drive_{k}_gain'] = (ax.get(k) or {}).get('gain') if not (ax.get(k) or {}).get('note') else None
        f[f'drive_{k}_lag_s'] = (ax.get(k) or {}).get('lag_s') if not (ax.get(k) or {}).get('note') else None
    f.update({'pose_rate_hz': fx.get('pose_rate_hz'), 'ambiguous_share': fx.get('ambiguous_share'),
              'latency_median_ms': fx.get('latency_median_ms'), 'range_median_m': fx.get('range_median_m'),
              'imu_rate_hz': r['imu'].get('rate_hz'), 'imu_gap_max_ms': r['imu'].get('dt_max_ms'),
              'sim_outcome': sm.get('outcome'), 'sim_approach_to_docked_s': sm.get('approach_to_docked_s'),
              'sim_lateral_overshoot_mm': sm.get('lateral_overshoot_mm'), 'sim_strafe_integral_m': (sm.get('strafe') or {}).get('integral_m'),
              'sim_final_y_mm': (sm.get('final') or {}).get('y_mm')})
    return f


def run_label(r: dict) -> str:
    return f"{r['time']['start_nzdt'][11:16]} ({r['run'].split('_')[-1]})"


def firmware_era(r: dict) -> str:
    t = datetime.strptime(r['time']['start_utc'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    if t < CHANNEL_FIX_UTC:
        return 'channels miswired'
    return 'before the per-wheel calibration' if t < DECK_FLASH_UTC else 'per-wheel calibrated'


def era_bands(ax, rows: list, label: bool = True) -> None:
    """Separate the drive's firmware eras between the rows (y = row index); label each at the right edge."""
    eras = [firmware_era(r) for r in rows]
    start = 0
    for i in range(1, len(eras) + 1):
        if i == len(eras) or eras[i] != eras[start]:
            if start:
                ax.axhline(start - 0.5, color=AXIS, lw=0.8)
            if label:
                ax.text(1.0, (start + i - 1) / 2, ' ' + eras[start], transform=ax.get_yaxis_transform(), fontsize=6.5,
                        color=INK2, va='center', ha='left')
            start = i


def span(vals, spec='.2f') -> str:
    v = [x for x in vals if x is not None and math.isfinite(x)]
    return f'{min(v):{spec}}-{max(v):{spec}}' if v else 'n/a'


def aggregate_figures(runs: list, out_root: Path) -> list:
    """The cross-run figures, each with its caption and reasoning (numbers computed from the runs)."""
    plt = style()
    figs = []
    docks = [r for r in runs if r['kind'] == 'dock']
    # drive gains and lags per run
    rows = [r for r in docks if any(a.get('gain') is not None and not a.get('note')
                                    for a in ((r.get('drive') or {}).get('axes') or {}).values())]
    if rows:
        fig, axs = plt.subplots(1, 2, figsize=(10.4, 0.42 * len(rows) + 1.4), sharey=True)
        y = np.arange(len(rows))
        for k, (axis, col) in enumerate((('forward', C[0]), ('lateral', C[1]), ('turn', C[2]))):
            ok = [r['drive']['axes'].get(axis, {}) for r in rows]
            g = [a.get('gain') if a.get('gain') is not None and not a.get('note') else np.nan for a in ok]
            lg = [a.get('lag_s') if a.get('lag_s') is not None and not a.get('note') else np.nan for a in ok]
            axs[0].plot(g, y + (k - 1) * 0.2, 'o', ms=6, color=col, label=axis)
            axs[1].plot(lg, y + (k - 1) * 0.2, 'o', ms=6, color=col, label=axis)
        axs[0].axvline(1.0, color=MUTED, lw=1)
        axs[0].set_xlabel('gain: delivered / commanded (camera; gyro for turn)')
        axs[1].set_xlabel('lag behind the command (s)')
        axs[0].set_yticks(y)
        axs[0].set_yticklabels([run_label(r) for r in rows], fontsize=7)
        axs[0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3)
        era_bands(axs[1], rows)
        era_bands(axs[0], rows, label=False)
        fig.tight_layout()
        save(fig, out_root, 'drive_gains')
        cal = [r for r in rows if firmware_era(r) == 'per-wheel calibrated']

        def g(axis, key='gain'):
            return [(r['drive']['axes'].get(axis) or {}).get(key) for r in cal
                    if not (r['drive']['axes'].get(axis) or {}).get('note')]
        figs.append(Fig('drive_gains', 'What the drive delivered, per run and axis',
                        'For each docking run with a usable fit, the gain (delivered over commanded velocity) and the lag '
                        'behind the command, forward, sideways and turning, grouped by the drive firmware of the day.',
                        'Each run\'s drive fit (its README, Drive): camera-derived body velocity, the gyro for turning, '
                        'least squares for gain, offset and lag over APPROACH to the end of commanding; fits with R^2 < 0.3 '
                        'are left out.',
                        f"On the per-wheel calibrated firmware ({len(cal)} runs) the robot delivered {span(g('forward'))}x the "
                        f"forward command, {span(g('lateral'))}x sideways and {span(g('turn'))}x of the turn rate, "
                        f"{span(g('forward', 'lag_s') + g('lateral', 'lag_s') + g('turn', 'lag_s'))} s late. The MPC predicts "
                        'with gain 1 and no lag, so its predictions overstate sideways motion and turning in every run.',
                        'The camera velocity is noisy and drops head-on, long-range stretches; gains vary with speed and '
                        'battery; a run that barely used an axis has no point for it. The firmware changed twice that day '
                        '(the channel wiring, then the per-wheel calibration about 18:27 NZDT), so compare within an era.',
                        'aggregate'))
    # EKF velocity error
    rows = [(run_label(r), r['ekf'].get('velocity') or {}) for r in docks]
    rows = [(n, v) for n, v in rows if v]
    if rows:
        fig, axs = plt.subplots(1, 2, figsize=(9.6, 0.42 * len(rows) + 1.4), sharey=True)
        y = np.arange(len(rows))
        for k, (axis, col) in enumerate((('x', C[0]), ('y', C[1]))):
            axs[0].plot([(v.get(axis) or {}).get('rms_mm_s', np.nan) or np.nan for _, v in rows], y + (k - 0.5) * 0.25, 'o',
                        ms=6, color=col, label={'x': 'forward', 'y': 'lateral'}[axis])
            axs[1].plot([(v.get(axis) or {}).get('slope', np.nan) or np.nan for _, v in rows], y + (k - 0.5) * 0.25, 'o',
                        ms=6, color=col, label={'x': 'forward', 'y': 'lateral'}[axis])
        axs[1].axvline(1.0, color=MUTED, lw=1)
        axs[0].set_xlabel('EKF velocity error against the camera, RMS (mm/s)')
        axs[1].set_xlabel('slope, EKF on camera (1 = follows it)')
        axs[0].set_yticks(y)
        axs[0].set_yticklabels([n for n, _ in rows], fontsize=7)
        axs[0].legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=2)
        fig.tight_layout()
        save(fig, out_root, 'ekf_velocity_error')
        vy = [(v.get('y') or {}).get('rms_mm_s') for _, v in rows]
        sy = [(v.get('y') or {}).get('slope') for _, v in rows]
        vya = [((r['ekf'].get('velocity_all_fixes') or {}).get('y') or {}).get('rms_mm_s') for r in docks
               if r['ekf'].get('velocity')]
        figs.append(Fig('ekf_velocity_error', "The EKF's velocity against the camera, per run",
                        'For each docking run, the RMS difference between the EKF\'s velocity and the camera-only velocity '
                        '(left) and the slope of one on the other (right; 1 = the EKF follows the camera), forward and sideways.',
                        'Each run\'s EKF section: unambiguous fixes, 0.4 s line fits, over APPROACH to the end of commanding.',
                        f"Lateral RMS {span(vy, '.0f')} mm/s and slopes {span(sy)}; with the ambiguous fixes kept in the camera "
                        f"track (as vel_truth.py) the lateral RMS is {span(vya, '.0f')} mm/s. The MPC starts every prediction "
                        'from this velocity, so the lateral error feeds straight into the plans (mpc_analysis_2026-10-14 finding 3).',
                        'The reference is itself noisy, and leaving out the ambiguous stretches favours the EKF: the true '
                        'error lies between the two figures. Short runs have few samples.', 'aggregate'))
    # ambiguity by range, pooled per detector
    pool = {}
    for r in runs:
        det = r['detector']['name']
        for b in r['fixes'].get('by_range', []):
            key = (b['range_lo'], b['range_hi'])
            n, a = pool.setdefault(det, {}).get(key, (0, 0.0))
            pool[det][key] = (n + b['poses'], a + b['poses'] * (b['ambiguous_share'] or 0))
    if pool:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for k, (det, d) in enumerate(sorted(pool.items(), key=lambda kv: list(DET_COLOUR).index(kv[0])
                                            if kv[0] in DET_COLOUR else 9)[:3]):
            ks = sorted(d)
            x = [(lo + hi) / 2 for lo, hi in ks]
            ax.plot(x, [100 * d[q][1] / max(d[q][0], 1) for q in ks], '-o', ms=5, color=DET_COLOUR.get(det, C[k]),
                    label=f'{det}')
            for xi, q in zip(x, ks):
                if d[q][0] >= 1:
                    ax.text(xi, 100 * d[q][1] / max(d[q][0], 1), f' {d[q][0]}', fontsize=5.5, color=INK2, va='bottom')
        ax.set_xlabel('range (m)')
        ax.set_ylabel('ambiguous poses (%), all runs pooled')
        ax.legend(title='detector deployed', title_fontsize=7)
        fig.tight_layout()
        save(fig, out_root, 'fix_ambiguity_by_range')
        figs.append(Fig('fix_ambiguity_by_range', 'Ambiguous poses by range, all runs pooled per deployed detector',
                        'The share of solved poses that were ambiguous (IPPE\'s two solutions within 1.5x in reprojection '
                        'error) per 0.1 m of range, pooling every run by the detector deployed at the time; the count of '
                        'poses is beside each point.',
                        'inference_result of every bag (docking, bring-up and drive-calibration runs).',
                        'With the 24 mm board, ambiguity climbs from a few per cent inside 0.5 m to over half beyond 0.7 m, '
                        'whichever detector ran: a geometric limit (7c item 57).',
                        'The detectors ran in different sessions, at different angles and light, so a gap between the curves '
                        'here is not a detector effect; the comparison on the same frames (Detectors, below) is the fair '
                        'test. Range is the pipeline\'s own estimate.', 'aggregate'))
    # MPC solve times
    rows = [(run_label(r), r['mpc']['solve_ms']) for r in docks if (r.get('mpc') or {}).get('solve_ms')]
    if rows:
        fig, ax = plt.subplots(figsize=(6.4, 0.38 * len(rows) + 1.4))
        y = np.arange(len(rows))
        for k, (q, col) in enumerate((('median', C[0]), ('p99', C[1]), ('max', C[2]))):
            ax.plot([s[q] for _, s in rows], y, 'o', ms=6, color=col, label=q)
        ax.set_xscale('log')
        ax.axvline(25, color=C[7], lw=1)
        ax.text(25, len(rows) - 0.5, ' budget 25 ms', fontsize=7, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels([n for n, _ in rows], fontsize=7)
        ax.set_xlabel('MPC solve time on the Pi (ms)')
        ax.legend(loc='lower right')
        fig.tight_layout()
        save(fig, out_root, 'mpc_solve_times')
        figs.append(Fig('mpc_solve_times', 'MPC solve time per run on the Pi',
                        'The median, 99th percentile and longest solve of each docking run (log scale), against the 25 ms '
                        'budget of the 50 ms tick.',
                        'dock/mpc_info.solve_ms: wall time around Mpc.step on the Pi (DAQP, Np 40 or 80).',
                        f"Medians {span([s['median'] for _, s in rows])} ms, longest {span([s['max'] for _, s in rows])} ms: "
                        'the solve uses under a tenth of its budget even at the 4 s horizon.',
                        'Wall time on a core shared with other work: a slow tick can be scheduling, not the solver.',
                        'aggregate'))
    # simulation against reality
    rows = [r for r in docks if r.get('sim')]
    if rows:
        fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.8))
        for r in rows:
            a, s = r['run_metrics'].get('approach_to_docked_s'), r['sim'].get('approach_to_docked_s')
            col = C[0] if a and s else C[1]
            axs[0].plot([a if a else 0.0], [s if s else 0.0], 'o', ms=7, color=col)
            axs[0].text(a if a else 0.0, s if s else 0.0, ' ' + r['run'][14:], fontsize=6, color=INK2)
            ho, hs = r['run_metrics'].get('lateral_overshoot_mm'), r['sim'].get('lateral_overshoot_mm')
            if ho is not None and hs is not None:
                axs[1].plot([ho], [hs], 'o', ms=7, color=C[0])
                axs[1].text(ho, hs, ' ' + r['run'][14:], fontsize=6, color=INK2)
        for ax, lab in ((axs[0], 'APPROACH to DOCKED (s); 0 = did not dock'), (axs[1], 'lateral overshoot (mm)')):
            lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
            ax.plot([0, lim], [0, lim], '-', color=MUTED, lw=1)
            ax.set_xlabel(f'robot: {lab}')
            ax.set_ylabel(f'simulation: {lab}')
        fig.tight_layout()
        save(fig, out_root, 'sim_vs_real')
        ov = [(r['run_metrics'].get('lateral_overshoot_mm'), r['sim'].get('lateral_overshoot_mm')) for r in rows]
        ov = [(a, b) for a, b in ov if a is not None and b is not None and a > 0]
        same = sum(1 for r in rows if r['run_metrics'].get('outcome') == r['sim'].get('outcome'))
        figs.append(Fig('sim_vs_real', 'The simulation against the robot, every simulated run',
                        'Per docking run, the time from APPROACH to DOCKED (left; 0 = did not dock) and the lateral '
                        'overshoot (right), robot against its simulation; points on the grey line agree.',
                        'Each run simulated from the camera\'s pose at APPROACH entry with the code and configuration that '
                        'ran and the run\'s own drive gains and lags (run_analytics_sim.py).',
                        f"Outcome the same in {same} of {len(rows)} runs. Where the robot overshot sideways, it did so by "
                        f"{span([a for a, _ in ov], '.0f')} mm against the simulation's {span([b for _, b in ov], '.0f')} mm: "
                        'the drive model alone does not produce the overshoot; what the simulation leaves out (ambiguous '
                        'fixes and the EKF velocity error they cause, re-anchors) does.',
                        'One noise seed per run; the simulation has no ambiguous poses, fix bias, slip or coupler contact, and '
                        'misses the old firmware\'s deadbands and the miswired channels of the first runs.', 'aggregate'))
    return figs


def index_md(runs: list, out_root: Path, figs: list | None = None) -> str:
    L = ['# Run records: index', '',
         f'Generated {nz(time.time()):%Y-%m-%d %H:%M} NZDT by '
         '`controller_ws/tools/run_analytics.py` from every `<run>/metrics.json` here. Definitions and reasoning: '
         '[README.md](README.md). Times are NZDT; bag names are UTC (the Pi\'s clock).', '']
    docks = [r for r in runs if r['kind'] == 'dock']
    rows = []
    for r in docks:
        f = flatten_run(r)
        rows.append([f"[{r['run']}]({r['run']}/README.md)", r['time']['start_nzdt'][11:19],
                     f"{f['outcome']}" + (f" ({f['abort_cause'][:40]})" if f['abort_cause'] else ''),
                     fmt(f['approach_to_docked_s'], '.1f'), f['holds'],
                     f"{f['commit']}{'*' if f['config_variant'] else ''}", f['detector'],
                     fmt(f['ambiguous_share'], '.0%'), fmt(f['ekf_vy_rms_mm_s'], '.0f'), fmt(f['ekf_vy_slope'], '.2f'),
                     f['reanchors'], f"{fmt(f['drive_forward_gain'], '.2f')} / {fmt(f['drive_lateral_gain'], '.2f')}",
                     f"{fmt(f['drive_forward_lag_s'], '.2f')} / {fmt(f['drive_lateral_lag_s'], '.2f')}",
                     fmt(f['mpc_solve_p99_ms'], '.1f'), fmt(f['lateral_overshoot_mm'], '.0f'),
                     (f"{f['sim_outcome']} {fmt(f['sim_approach_to_docked_s'], '.1f')}" if f['sim_outcome'] else '-')])
    L += ['## Docking attempts', '',
          table(rows, ['run', 'NZDT', 'outcome', 'to dock (s)', 'HOLDs', 'config', 'detector', 'ambiguous', 'EKF vy err (mm/s)',
                       'EKF vy slope', 're-anchors', 'drive gain fwd / lat', 'drive lag fwd / lat (s)', 'solve p99 (ms)',
                       'overshoot (mm)', 'simulated: outcome, to dock (s)']),
          '', '`*`: no commit matches; the configuration is a reconstruction (see the run\'s README).', '']
    other = [r for r in runs if r['kind'] != 'dock']
    if other:
        rows = [[f"[{r['run']}]({r['run']}/README.md)", r['kind'], r['time']['start_nzdt'][11:19], r['time']['recorded_s'],
                 r['fixes'].get('inference_results', 0), r['fixes'].get('images', 0), fmt(r['fixes'].get('ambiguous_share'), '.0%'),
                 r['detector']['name']] for r in other]
        L += ['## Other recordings', '', table(rows, ['run', 'kind', 'NZDT', 'length (s)', 'poses', 'frames', 'ambiguous',
                                                       'detector']), '']
    L += ['## Across runs', '']
    for n, f in enumerate(figs or [], 1):
        L.append(fig_md(f, n))
    for sub, title, how in (('detectors', 'Detectors on the real frames', 'run with `--detectors` (the GPU, one model at a time)'),
                            ('finetune', 'Fine-tuning records', 'run with `--finetune`')):
        p = out_root / sub / 'SUMMARY.md'
        L += [f'## {title}', '']
        L.append(embed(p.read_text().split('\n', 2)[2], sub) + f'\n\nFull page: [{sub}/SUMMARY.md]({sub}/SUMMARY.md).'
                 if p.exists() else f'Not yet generated: {how}.')
        L.append('')
    return '\n'.join(L)


def embed(md: str, sub: str) -> str:
    """A sub-page's Markdown inside INDEX.md: headings one level down, relative links rebased from sub/."""
    import posixpath
    md = re.sub(r'^(#+) ', lambda g: '#' + g.group(1) + ' ', md, flags=re.M)

    def rebase(g):
        url = g.group(2)
        if re.match(r'^[a-z]+:|^/|^#', url):
            return g.group(0)
        return f'{g.group(1)}({posixpath.normpath(posixpath.join(sub, url))})'
    return re.sub(r'(\]\s*)\(([^)\s]+)\)', rebase, md)


PIPELINE_README = """# Run records

Report-ready evidence from the robot's real docking runs, to set against the simulations: one folder per
run (`<bag name>/`), [INDEX.md](INDEX.md) across them, `aggregate.csv` (one row per run), `figures/`
(the cross-run figures), `detectors/` (the three detectors on the real frames) and `finetune/` (the
self-supervised fine-tuning records and the detectors on their held-out frames).

## After a session

On the laptop, with the Pi reachable (the script copies the Pi's bags, then runs the analysis in p4p-gpu):

    ~/p4p/controller_ws/tools/pull_pi_bags.sh

Or by hand in p4p-gpu, both workspaces sourced:

    cd /workspace/controller_ws && python3 tools/run_analytics.py /workspace/run_data/pi_bags

A run already recorded is skipped unless `--force`; the index is rebuilt every time. `--no-sim` skips the
simulation, `--detectors` and `--finetune` run the GPU comparisons (one model per process; not while the
camera calibration or a fine-tune holds the GPU), `--index-only` only rebuilds the index.

## What a record holds

The README of each run gives the time (UTC and NZDT), the outcome, the controller commit and its key settings
with how sure the match is, and the detector; then the trajectory, the EKF, the MPC, the drive, the fixes, the
IMU and the simulation, each figure with what it shows, how it was computed, what it means and its caveats.

## Which configuration ran

- **Controller:** the commit of controller_ws (dock_control) deployed at the run's arming time. Commit times
  are NZDT and deploys followed commits by minutes, so the candidates are the last commit before the run and any
  committed up to 15 min after it (deployed before it was committed). Each candidate is checked against what the
  bag shows by itself (the APPROACH reference, i.e. the standoff; the APPROACH and FINAL caps; the turn cap; the
  horizon from y_pred; the target -> marker and base_link -> camera transforms), then up to 160 logged MPC ticks
  are replayed through the candidate's own code (`run_analytics_sim.py replay`). A replay within a few mm of the
  logged plans confirms it. Runs before the 17:58 batch commit of 2026-10-13 ran a working tree that was never
  committed; they are matched to the batch commit's code with the earlier MPC weights (rw 1, lateral weight from
  tol.y, 7c item 47), and marked as reconstructions.
- **Detector:** cfg/cfg.yaml's `MODEL.detector` in Conv-ChArT-Wireless-Inference at the run's time, unless the
  run's own frames identify the live model (detectors/live_detector.json, from re-running each detector on the
  bag's frames and matching the logged poses). A checkpoint written before the run but committed after it is
  flagged.

## Metric definitions, and why

**Camera-only track.** The reference for everything the estimator and the drive do: a straight line through the
unambiguous fixes within +-0.2 s of each 50 ms grid point (at least 4 fixes), its value and slope giving pose and
velocity (as `mpc_analysis_2026-10-14/vel_truth.py`). Ambiguous fixes are left out because the node picks the
solution nearer the EKF's prediction, so they are not independent of the estimate. It is noisy (centimetres at
1 m, roughly 10-30 mm/s of velocity at 0.3-0.5 m) and missing where every fix was ambiguous, typically head-on at
range, which is where the estimator struggles most; read its absence as part of the evidence.

**EKF.**
- *Position residual:* the logged estimate at each unambiguous fix's capture time minus the fix. The logged
  estimate is the real-time one the MPC used, before that fix arrived.
- *Consistency:* residual^2 / (P + R) per axis, P from DockEstimate.p_diag and R from the fix's own covariance.
  About 1 on average (and 95% within 2 sd) if the filter's claimed uncertainty is right; above, overconfident.
- *Velocity:* the EKF's velocity against the camera's over APPROACH to the end of commanding: RMS error, slope of
  EKF on camera (1 = follows it), and the EKF's lag. The MPC predicts from this velocity. It is computed twice:
  against the unambiguous-only track (the headline), and against a track that keeps the ambiguous fixes, as
  `mpc_analysis_2026-10-14/vel_truth.py` does (`velocity_all_fixes`; that one reproduces vel_truth's numbers). The
  first leaves out the head-on, long-range stretches where the EKF struggles most; the second counts solution flips
  as motion. The real error lies between them: on dock_20261013_055413, 31 against 55 mm/s lateral RMS.
- *NIS:* DockEstimate.last_nis per processed fix (a change of last_nis or last_fix_accepted marks one), in the
  gated stages, against chi-square with 3 dof (median 2.37, 95% point 7.81); acceptance share against the gate (100).
- *Re-anchors* from ekf_node's log, with the jump; *estimate steps* over 10 mm between consecutive estimates.

**MPC.** Solve time (dock/mpc_info.solve_ms, the Pi's wall time, budget 25 ms), fallbacks, ticks with slack;
the share of commanding ticks at each limit (speed cap per axis or L1 as the commit had it, turn cap, wheel speed
through the nominal mecanum inverse, increment limits), within 1 mm/s; command smoothness (RMS change per tick,
sign reversals per 10 s); plan pointing (angle between plan end and goal from the tick's estimate, > 45 deg counts
as pointing away; mpc_analysis finding 4); strafe usage (commanded sideways distance, peak, share of ticks with
|v_b| > |u_b|).

**Run.** Time in each stage, APPROACH to DOCKED, HOLDs, the abort rule that fired (from the bag at the abort);
start pose (camera, at APPROACH), lateral overshoot (furthest past the centre line on the side away from the
start, camera), final pose (median of the last 0.5 s of unambiguous fixes before DOCKED or the end).

**Drive.** The command in force (cmd_vel held, zero after the bridge's 0.1 s timeout) against the camera's body
velocity (gyro for turning, over the same 0.4 s window), per axis: gain, offset and lag by least squares over lags
0-0.6 s, over APPROACH to the end of commanding. A fit needs R^2 >= 0.3 and some motion on the axis; a usable fit
with a negative or tiny gain is kept and flagged (the robot moved against, or barely with, the command). Gain < 1:
under-delivery; lag: how late. These are the model errors the MPC ignores, and the simulation's plant. The lag
includes the camera stamp offset (10-40 ms). vel_truth.py fitted the target frame's x and y with the gain fixed at
1 during the lag search; this fits the body axes, so lateral numbers agree (0.73x at 0.15 s on 055413 in both) and
forward ones differ where the robot was turned.

**Fixes.** Pose rate, refusals, ambiguity, reprojection RMS and corners by range; capture-to-pose latency as bag
receive time minus header stamp (both the Pi's clock, so no inter-machine offset; the stamp is 10-40 ms after
exposure); jitter in still 1 s windows against the fixes' own R.

**IMU.** Rate, interval distribution and gaps from header stamps; stamp-to-bag delay; the EKF's own measured rate.

**Simulation against the run.** `run_analytics_sim.py sim`: the matched commit's Supervisor, Mpc, Ekf and fiducial
chain in closed loop from the camera's pose at APPROACH entry, on a plant with this run's measured gains and lags
(time constant = lag - 50 ms command delay), the deadband of the firmware of the day, fixes delayed by the run's
median latency within a 42 deg half field of view, bench fix noise. Compared on outcome, time to dock, lateral
overshoot, strafe, final pose, HOLDs, and the EKF's velocity error. The README lists where it fails to reproduce
the run. It never compares alternative weightings: it simulates what ran.

## Caveats that apply everywhere

- Camera "truth" is noisy at range and absent when ambiguous; it is a reference, not ground truth.
- Commit and detector matches are approximate where marked; check "how sure" in each README.
- All times are the Pi's clock. Capture stamps are the frame read time, not the exposure.
- The drive changed during 2026-10-13 (channel fix, then per-wheel calibration about 18:27 NZDT), so drive numbers
  are comparable only within one firmware.
"""


# ============================================================================================ main
def main() -> None:
    try:
        os.nice(19)             # never compete with a live inference node for the CPU
    except OSError:
        pass
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('target', nargs='?', type=Path, help='a bag folder, or a folder of bags')
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT, help='records folder (default: the report notes)')
    ap.add_argument('--force', action='store_true', help='redo runs already recorded')
    ap.add_argument('--no-sim', action='store_true', help='skip the closed-loop simulation of each run')
    ap.add_argument('--only', default='', help='only bags whose name contains this')
    ap.add_argument('--index-only', action='store_true', help='only rebuild INDEX.md and the aggregate')
    ap.add_argument('--detectors', action='store_true', help='also compare the detectors on the bags\' frames (GPU)')
    ap.add_argument('--finetune', action='store_true', help='also gather the fine-tuning records (CPU)')
    ap.add_argument('--rerun-detectors', action='store_true',
                    help='with --detectors: run every detector again even if its results are cached (GPU)')
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    live_path = a.out / 'detectors' / 'live_detector.json'
    if a.detectors or a.finetune:
        import run_analytics_detectors as rad
        if a.detectors:
            rad.compare_bags(a.target or DEFAULT_BAGS, a.out / 'detectors', rerun=a.rerun_detectors)
        if a.finetune:
            rad.finetune(a.out / 'finetune')
    live = json.loads(live_path.read_text()) if live_path.exists() else None
    if not a.index_only:
        target = a.target or DEFAULT_BAGS
        bags = [p for p in find_bags(target) if a.only in p.name]
        print(f'{len(bags)} bags in {target}')
        for p in bags:
            done = a.out / p.name / 'metrics.json'
            if done.exists() and not a.force:
                try:
                    old = json.loads(done.read_text())
                    if old.get('version') == VERSION and old.get('bag_fingerprint') == bag_fingerprint(p):
                        print(f'{p.name}: recorded already (--force to redo)')
                        continue
                    print(f'{p.name}: the bag or the tool changed since it was recorded')
                except json.JSONDecodeError:
                    pass
            print(f'{p.name}: analysing')
            try:
                M = analyse(p, a.out, sim_on=not a.no_sim, live=live)
                print(f"  {M['kind']}, {M['run_metrics'].get('outcome', '')}, config {M['config']['commit']} "
                      f"({M['config']['certainty'][:60]}), {len(M['figures'])} figures, {M['seconds_to_analyse']} s")
            except Exception as ex:
                import traceback
                traceback.print_exc()
                print(f'{p.name}: FAILED ({ex}); continuing')
    aggregate(a.out)
    print(f'index: {a.out / "INDEX.md"}')


if __name__ == '__main__':
    sys.path.insert(0, str(TOOLS))
    main()
