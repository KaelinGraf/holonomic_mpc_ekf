#!/usr/bin/env python3
"""
The three detectors on real frames, for run_analytics.py: the bags that carry camera frames, and the
self-supervised fine-tune's held-out frames. Then the fine-tuning records gathered into one index.

    python3 tools/run_analytics_detectors.py prepare [<bag folder>]     # CPU: sample frames into the cache
    python3 tools/run_analytics_detectors.py infer <release|r3|r1>      # GPU: ONE detector, in this process only
    python3 tools/run_analytics_detectors.py summarise [--out <records>] # CPU: tables, figures, SUMMARY.md
    python3 tools/run_analytics_detectors.py finetune [--out <records>]  # CPU: the fine-tuning index

run_analytics.py --detectors runs prepare, then each detector in a process of its own, one after the other,
then summarise. One inference_pipeline per process, every call from one thread: a pipeline settles at about
5 GB of the laptop's 8 GB GPU, and ONNX Runtime's CUDA provider adds about 375 MB for each new calling thread
and never gives it back. Do not run it while the camera calibration or a fine-tune holds the GPU.

Each detector runs through the deployed pipeline class (Conv-ChArT-Wireless-Inference/src/inference.py,
inference_pipeline) with a temporary config in /tmp: the cfg.yaml of the day with the detector's absolute
path. The camera matrix is the one in force when the frames were taken (camera_epochs()), passed with each call,
and the board is the 24 mm print every 2026-10-13 frame shows.

Frames: from each bag with an image topic, up to MAX_PER_BAG frames evenly spaced, with the pose the live
pipeline logged for the same frame (inference_result, matched by stamp) where the bag has one; and every
held-out frame of the offline loop's last pass (harvest/gen2/heldout), with the teacher label's square size.
"""
from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np

TOOLS = Path(__file__).resolve().parent
P4P = TOOLS.parent.parent
WIRELESS = P4P / 'Conv-ChArT-Wireless-Inference'
DEFAULT_OUT = P4P / 'report' / 'project content' / '7_ros2_implementation' / 'run_records'
SSL_RECORDS = DEFAULT_OUT.parent / 'ssl_finetune_records'
OFFLINE = WIRELESS / 'ssl_output' / 'offline_20261013-113934'
HELDOUT = OFFLINE / 'harvest' / 'gen2' / 'heldout'
CACHE = Path(tempfile.gettempdir()) / 'run_analytics_frames'
MAX_PER_BAG = 800
SQUARE_M = 0.024            # the 120 mm print: every frame of 2026-10-13
DETECTORS = {'release': 'checkpoints/detector_882k.onnx',
             'r3': 'checkpoints/detector_882k_ssl_20261013_r3.onnx',
             'r1': 'checkpoints/detector_882k_ssl_20261014_offline_r1.onnx'}
LABELS = {'release': 'release', 'r3': 'SSL live r3', 'r1': 'SSL offline r1'}
# The camera matrix in force from each time on (UTC epoch). Up to 2026-10-13 18:03 NZDT the committed cfg.yaml had no
# K, so the first two are set by hand: the pre-refocus calibration cfg.yaml held until the recalibration of 02:05 UTC
# (calibration_output/20261013-020557/cfg.yaml.before; it reproduces the logged poses of the 13:21-13:40 NZDT image
# bags exactly, see detectors/SUMMARY.md), then that recalibration, committed only at 7ed7ced. Every later K comes
# from cfg/cfg.yaml's git history, at its commit time (camera_epochs()).
CAMERA_FIXED = [
    (0.0, 'pre-refocus calibration (cfg.yaml.before of calibration_output/20261013-020557), fx 369.05',
     [[369.05021167731627, 0.0, 324.8834264945149], [0.0, 369.6225129282, 230.89974377954223], [0.0, 0.0, 1.0]],
     [0.03741020990156616, -0.03356877500885844, -0.001614475140159554, -0.00021076142692561133,
      -0.015941719137565636]),
    (1791857157.0, 'recalibration of 2026-10-13 15:05 NZDT (committed in 7ed7ced), fx 356.66',
     [[356.65571428510805, 0.0, 327.55181541731486], [0.0, 356.62625866338544, 234.19377039134744], [0.0, 0.0, 1.0]],
     [0.05350674753450749, -0.10031543312208781, 0.0003128032616877976, 0.00014277460088732907,
      0.03174167274614614]),
]
FIXED_UNTIL = 1791867795.0          # 7ed7ced, 2026-10-13 18:03:15 NZDT: from here on cfg.yaml's history is the record


@functools.lru_cache(maxsize=None)
def camera_epochs() -> tuple:
    """(start, description, K, dist) of each camera matrix, oldest first: the two set by hand, then cfg.yaml's."""
    import yaml
    eps = list(CAMERA_FIXED)
    log = subprocess.run(['git', '-C', str(WIRELESS), 'log', '--reverse', '--format=%h %ct', '--', 'cfg/cfg.yaml'],
                         capture_output=True, text=True, check=True).stdout.split('\n')
    for line in filter(None, log):
        h, t = line.split()
        if float(t) <= FIXED_UNTIL:
            continue
        c = yaml.safe_load(subprocess.run(['git', '-C', str(WIRELESS), 'show', f'{h}:cfg/cfg.yaml'], capture_output=True,
                                          text=True, check=True).stdout) or {}
        cam = c.get('CAMERA') or {}
        if cam.get('K') and not np.allclose(np.array(cam['K'], float), np.array(eps[-1][2], float)):
            from datetime import datetime
            from zoneinfo import ZoneInfo
            when = datetime.fromtimestamp(float(t), ZoneInfo('Pacific/Auckland')).strftime('%Y-%m-%d %H:%M NZDT')
            eps.append((float(t), f"cfg.yaml at {h} ({when}), fx {cam['K'][0][0]:.2f}", cam['K'], cam['dist']))
    return tuple(eps)


K_CHECK_FRAMES = 120        # frames re-solved with the other calibration, to test which one the live pipeline used

C = ('#2a78d6', '#eb6834', '#1baf7a')
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'


def pc(v, spec: str = '.1%') -> str:
    """A share as a percentage, 'n/a' when missing."""
    return 'n/a' if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, spec)


def camera_at(t: float):
    return [e for e in camera_epochs() if e[0] <= t][-1]


# ============================================================================================ prepare (CPU)
def prepare(bag_root: Path) -> list:
    """Sample frames from every bag with an image topic into CACHE/<bag>.npz; return the sets written."""
    import cv2
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    sys.path.insert(0, str(TOOLS))
    from run_analytics import bag_storage, find_bags
    CACHE.mkdir(parents=True, exist_ok=True)
    sets = []
    for path in find_bags(bag_root):
        uri, sid, _ = bag_storage(path)
        out = CACHE / f'{path.name}.npz'
        if out.exists():
            sets.append(out)
            continue
        reader = rosbag2_py.SequentialReader()
        reader.open(rosbag2_py.StorageOptions(uri=uri, storage_id=sid), rosbag2_py.ConverterOptions('', ''))
        types = {t.name: t.type for t in reader.get_all_topics_and_types()}
        img_topic = next((n for n in types if n.rstrip('/').endswith('image') and 'Compressed' in types[n]), None)
        if img_topic is None:
            continue
        inf_topic = next((n for n in types if n.endswith('inference_result')), None)
        topics = [img_topic] + ([inf_topic] if inf_topic else [])
        reader.set_filter(rosbag2_py.StorageFilter(topics=topics))
        img_cls = get_message(types[img_topic])
        inf_cls = get_message(types[inf_topic]) if inf_topic else None
        raw, live = [], {}
        while reader.has_next():
            topic, data, t = reader.read_next()
            if topic == img_topic:
                raw.append((t, data))
            else:
                m = deserialize_message(data, inf_cls)
                p = m.pose.pose.position
                key = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
                live[key] = (float(not m.reason), float(m.ambiguous), p.x, p.y, p.z, float(m.num_used), m.rms)
        if not raw:
            continue
        raw.sort(key=lambda r: r[0])
        idx = np.unique(np.linspace(0, len(raw) - 1, min(MAX_PER_BAG, len(raw))).round().astype(int))
        frames, stamps, lv = [], [], []
        for i in idx:
            m = deserialize_message(raw[i][1], img_cls)
            img = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            key = m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
            frames.append(img)
            stamps.append(key)
            lv.append(live.get(key, (np.nan,) * 7))
        t0 = stamps[0] * 1e-9
        cam = camera_at(t0)
        np.savez(out, frames=np.stack(frames), stamps=np.array(stamps, np.int64), live=np.array(lv, float),
                 total=len(raw), K=np.array(cam[2]), dist=np.array(cam[3]), camera=cam[1], square=SQUARE_M,
                 has_live=bool(live))
        print(f'{path.name}: {len(frames)} of {len(raw)} frames, {len(live)} live results, camera: {cam[1]}')
        sets.append(out)
    return sets


def heldout_frames():
    """(frames, s_px, stamps, brightness level) of the offline loop's gen2 held-out chunks."""
    frames, s_px, stamps, level = [], [], [], []
    for d in sorted(HELDOUT.iterdir()):
        recs = json.loads((d / 'records.json').read_text())['records']
        imgs = np.load(d / 'images.npy', mmap_mode='r')
        for img, r in zip(imgs, recs):
            frames.append(np.ascontiguousarray(img))
            s_px.append(r.get('s_px') or np.nan)
            stamps.append(r.get('stamp') or np.nan)
            level.append(r.get('board_level') or np.nan)
    return frames, np.array(s_px, float), np.array(stamps, float), np.array(level, float)


# ============================================================================================ infer (GPU)
def infer(key: str) -> None:
    """Run ONE detector over every cached set and the held-out frames; one pipeline, this thread only."""
    import yaml
    sys.path.insert(0, str(WIRELESS / 'src'))
    from inference import inference_pipeline
    cfg = yaml.safe_load((WIRELESS / 'cfg' / 'cfg.yaml').read_text())
    cfg['MODEL'] = {'detector': str(WIRELESS / DETECTORS[key]), 'refiner': str(WIRELESS / 'checkpoints' / 'refiner.onnx')}
    cfg['BOARD'] = {'squares': [5, 5], 'square_length_m': SQUARE_M}
    cfg['CAMERA'] = {'K': camera_epochs()[-1][2], 'dist': camera_epochs()[-1][3], 'image_size': [640, 480]}
    tmp = Path('/tmp') / f'run_analytics_cfg_{key}.yaml'
    tmp.write_text(yaml.safe_dump(cfg))
    pipe = inference_pipeline(config=str(tmp))
    res_dir = CACHE / 'results'
    res_dir.mkdir(parents=True, exist_ok=True)

    def solve(img, K, dist):
        t = time.perf_counter()
        r = pipe.run_inference(img, np.asarray(K, float), np.asarray(dist, float))
        ms = (time.perf_counter() - t) * 1e3
        ok = r['rvec'] is not None
        tv = np.asarray(r['tvec'], float).ravel() if ok else np.full(3, np.nan)
        n = sum(c['index'] is not None for c in r['corners'])
        return [float(ok), float(bool(r['ambiguous'])) if ok else np.nan, *tv,
                r['rms'] if r['rms'] is not None else np.nan,
                r['rms_alt'] if r['rms_alt'] is not None else np.nan, float(n), ms]

    for _ in range(3):                                   # warm-up
        solve(np.zeros((480, 640), np.uint8), camera_epochs()[-1][2], camera_epochs()[-1][3])
    for f in sorted(CACHE.glob('*.npz')):
        z = np.load(f)
        frames, K, dist, live = z['frames'], z['K'], z['dist'], z['live']
        rows = [solve(img, K, dist) for img in frames]
        extra = {}
        if bool(z['has_live']):                          # the K check: the other calibration on a subset
            have = np.nonzero(np.isfinite(live[:, 0]) & (live[:, 0] == 1))[0]
            sub = have[np.linspace(0, len(have) - 1, min(K_CHECK_FRAMES, len(have))).round().astype(int)] if len(have) else []
            for e in camera_epochs()[:2]:          # the K check: the two calibrations of 2026-10-13
                if not np.allclose(np.array(e[2]), K):
                    extra[e[1]] = (np.array(sub, int), np.array([solve(frames[i], e[2], e[3]) for i in sub], float))
        np.savez(res_dir / f'{key}__{f.stem}.npz', rows=np.array(rows, float),
                 **{f'kcheck_idx_{i}': v[0] for i, v in enumerate(extra.values())},
                 **{f'kcheck_rows_{i}': v[1] for i, v in enumerate(extra.values())},
                 kcheck_names=np.array(list(extra.keys())))
        print(f'{key} on {f.stem}: {len(rows)} frames, {np.mean([r[-1] for r in rows]):.1f} ms each', flush=True)
    frames, s_px, stamps, level = heldout_frames()
    cam = camera_at(float(np.nanmin(stamps)))          # the held-out frames: 15:18-16:09 NZDT, 2026-10-13
    rows = [solve(img, cam[2], cam[3]) for img in frames]
    np.savez(res_dir / f'{key}__heldout.npz', rows=np.array(rows, float), s_px=s_px, stamps=stamps, level=level,
             camera=cam[1])
    print(f'{key} on held-out: {len(rows)} frames', flush=True)


def run_all_detectors() -> None:
    """Each detector in a process of its own, one after the other (GPU)."""
    for key in DETECTORS:
        t = time.time()
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'infer', key])
        if p.returncode != 0:
            raise RuntimeError(f'the {key} detector process failed ({p.returncode})')
        print(f'{key}: {time.time() - t:.0f} s')


# ============================================================================================ summarise (CPU)
COLS = ['solved', 'ambiguous', 'tx', 'ty', 'tz', 'rms', 'rms_alt', 'corners', 'ms']


def load_results():
    res = {}
    for f in sorted((CACHE / 'results').glob('*.npz')):
        key, name = f.stem.split('__', 1)
        res.setdefault(name, {})[key] = np.load(f, allow_pickle=False)
    return res


def s_px_from(rows_by_key: dict, fx: float) -> np.ndarray:
    """Apparent square size per frame from the solving models' range: fx L / z, the median over models."""
    zs = np.c_[[r['rows'][:, 4] for r in rows_by_key.values()]].T
    with np.errstate(invalid='ignore'):
        return fx * SQUARE_M / np.nanmedian(np.where(np.isfinite(zs), zs, np.nan), axis=1)


def bucket_table(rows_by_key: dict, s_px: np.ndarray, edges) -> list:
    out = []
    keys = list(rows_by_key)
    allsolved = np.all([rows_by_key[k]['rows'][:, 0] == 1 for k in keys], axis=0)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (s_px >= lo) & (s_px < hi)
        if not sel.any():
            continue
        row = {'s_px': f'{lo:g}-{hi:g}' if hi < 1e8 else f'>={lo:g}', 'frames': int(sel.sum()),
               'all_solve': int((sel & allsolved).sum())}
        for k in keys:
            r = rows_by_key[k]['rows']
            sv = sel & (r[:, 0] == 1)
            row[f'{k}_solved'] = float(sv.sum() / sel.sum())
            row[f'{k}_ambiguous_own'] = float(np.nanmean(r[sv, 1])) if sv.any() else math.nan
            m = sel & allsolved
            row[f'{k}_ambiguous_matched'] = float(np.nanmean(r[m, 1])) if m.any() else math.nan
            row[f'{k}_z_median'] = float(np.nanmedian(r[sv, 4])) if sv.any() else math.nan
        out.append(row)
    return out


def live_identification(name: str, z, res: dict) -> dict:
    """Which detector reproduces the poses the live pipeline logged for these frames."""
    live = z['live']
    have = np.isfinite(live[:, 0])
    out = {'frames_with_live': int(have.sum())}
    scores = {}
    for k, r in res.items():
        rows = r['rows']
        both = have & (live[:, 0] == 1) & (rows[:, 0] == 1)
        same_solve = float(np.mean(live[have, 0] == rows[have, 0])) if have.any() else math.nan
        dt = np.linalg.norm(rows[both, 2:5] - live[both, 2:5], axis=1)
        scores[k] = {'same_solved_state': same_solve, 'pose_within_1mm': float(np.mean(dt < 1e-3)) if both.any() else math.nan,
                     'same_ambiguity': float(np.mean(rows[both, 1] == live[both, 1])) if both.any() else math.nan,
                     'same_corners': float(np.mean(rows[both, 7] == live[both, 5])) if both.any() else math.nan,
                     'median_pose_diff_mm': float(np.median(dt) * 1e3) if both.any() else math.nan}
        kc = {}
        names = list(r['kcheck_names']) if 'kcheck_names' in r.files else []
        for i, nm in enumerate(names):
            idx, kr = r[f'kcheck_idx_{i}'], r[f'kcheck_rows_{i}']
            okk = (kr[:, 0] == 1) & (live[idx, 0] == 1)
            d2 = np.linalg.norm(kr[okk, 2:5] - live[idx][okk, 2:5], axis=1)
            d1 = np.linalg.norm(rows[idx][okk, 2:5] - live[idx][okk, 2:5], axis=1)
            kc[str(nm)] = {'frames': int(okk.sum()), 'median_pose_diff_mm': float(np.median(d2) * 1e3) if okk.any() else math.nan,
                           'used_calibration_median_diff_mm': float(np.median(d1) * 1e3) if okk.any() else math.nan}
        if kc:
            scores[k]['k_check'] = kc
    out['scores'] = scores
    # The three detectors share the refiner, which sets the corner positions, so where two detectors find the same
    # corners their poses are identical: the pose cannot tell them apart. What can is which frames each solves (or
    # refuses) and how many corners it identifies.
    rank = sorted(scores, key=lambda k: (np.nan_to_num(scores[k]['same_solved_state']), np.nan_to_num(scores[k]['same_corners']),
                                         np.nan_to_num(scores[k]['pose_within_1mm'])), reverse=True)
    best, second = rank[0], (rank[1] if len(rank) > 1 else None)
    b = scores[best]
    margin = b['same_solved_state'] - scores[second]['same_solved_state'] if second else 1.0
    n = out['frames_with_live']
    tied = [k for k in rank if abs(scores[k]['same_solved_state'] - b['same_solved_state']) * n < 1
            and abs(np.nan_to_num(scores[k]['same_corners']) - np.nan_to_num(b['same_corners'])) * n < 1]
    if len(tied) > 1:
        out['detector'] = None
        out['evidence'] = (f"{', '.join(LABELS[k] for k in tied)} reproduce the live pipeline equally ("
                           f"{b['same_solved_state']:.1%} of {n} logged frames the same solved/refused verdict): these "
                           'frames cannot tell them apart')
    elif b['same_solved_state'] > 0.98 and b['pose_within_1mm'] > 0.95:
        out['detector'] = LABELS[best]
        out['strength'] = 'strong' if margin * n >= 10 else 'weak'
        out['evidence'] = (f"it reproduces the live pipeline's solved/refused verdict on {b['same_solved_state']:.1%} of "
                           f"{n} logged frames (next: {LABELS[second]} {scores[second]['same_solved_state']:.1%}), and its "
                           f"poses within 1 mm on {b['pose_within_1mm']:.0%}; the detectors share the refiner, so "
                           f"poses alone cannot separate them" + (' (a margin of under 10 frames: weak)'
                                                                  if out['strength'] == 'weak' else ''))
    else:
        out['detector'] = None
        out['evidence'] = (f"no detector reproduces the live verdicts (best {LABELS[best]}: {b['same_solved_state']:.0%} same "
                           f"solved state, {b['pose_within_1mm']:.0%} poses within 1 mm)")
    return out


def summarise(out: Path) -> None:
    sys.path.insert(0, str(TOOLS))
    from run_analytics import fin, table
    res = load_results()
    if not res:
        print('no detector results in the cache: run infer first')
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / 'data').mkdir(exist_ok=True)
    keys = [k for k in DETECTORS if all(k in r for r in res.values())]
    summary = {'detectors': {k: DETECTORS[k] for k in keys}, 'sets': {}}
    live_json = {}
    L = ['# Detectors on the real frames', '',
         'The release, SSL live round-3 and SSL offline round-1 detectors run through the deployed pipeline class '
         '(`src/inference.py`, `inference_pipeline`) on the same real frames, one detector per process, by '
         '`controller_ws/tools/run_analytics_detectors.py`. Board: the 24 mm print. Camera matrix: the one in force '
         'when the frames were taken (listed per set).', '']
    bag_sets = sorted(n for n in res if n != 'heldout')
    pooled = {k: [] for k in keys}
    pooled_spx = []
    seen = {}
    for name in bag_sets:
        z = np.load(CACHE / f'{name}.npz')
        stamps = z['stamps']
        dup = np.array([int(x) in seen for x in stamps])
        dup_of = sorted({seen[int(x)] for x in stamps if int(x) in seen})
        for x in stamps:
            seen.setdefault(int(x), name)
        r = {k: res[name][k] for k in keys}
        n = len(z['frames'])
        allsolved = np.all([r[k]['rows'][:, 0] == 1 for k in keys], axis=0)
        fx = float(z['K'][0][0])
        spx = s_px_from(r, fx)
        st = {'frames': n, 'of': int(z['total']), 'camera': str(z['camera']), 'all_solve': int(allsolved.sum())}
        rows = []
        for k in keys:
            rr = r[k]['rows']
            sv = rr[:, 0] == 1
            st[k] = {'solved': fin(sv.mean(), 4), 'ambiguous_own': fin(np.nanmean(rr[sv, 1]) if sv.any() else math.nan, 4),
                     'ambiguous_matched': fin(np.nanmean(rr[allsolved, 1]) if allsolved.any() else math.nan, 4),
                     'range_median_m': fin(np.nanmedian(np.linalg.norm(rr[sv, 2:5], axis=1)) if sv.any() else math.nan, 3),
                     'range_p90_m': fin(np.nanpercentile(np.linalg.norm(rr[sv, 2:5], axis=1), 90) if sv.any() else math.nan, 3),
                     'corners_median': fin(np.nanmedian(rr[sv, 7]) if sv.any() else math.nan, 1),
                     'ms_median': fin(np.median(rr[:, 8]), 1)}
            rows.append([LABELS[k], pc(st[k]['solved']), pc(st[k]['ambiguous_own']),
                         pc(st[k]['ambiguous_matched']) if st[k]['ambiguous_matched'] is not None else 'n/a',
                         st[k]['range_median_m'], st[k]['range_p90_m'], st[k]['corners_median'], st[k]['ms_median']])
            pooled[k].append(rr[~dup])
        pooled_spx.append(spx[~dup])
        if dup.any():
            st['duplicate_of'] = dup_of
            st['duplicate_share'] = fin(dup.mean(), 3)
        if bool(z['has_live']):
            ident = live_identification(name, z, r)
            ident['t'] = float(z['stamps'][0] * 1e-9)
            st['live'] = ident
            live_json[name] = {'detector': ident['detector'], 'evidence': ident['evidence'], 't': ident['t']}
        summary['sets'][name] = st
        L += [f'## {name}', '',
              f"{n} frames sampled evenly from {int(z['total'])}; camera: {z['camera']}. All three solve {int(allsolved.sum())} "
              f'of them ("matched").' + (f" {dup.mean():.0%} of these frames (same stamps, same pixels) are also in "
                                        f"{', '.join(dup_of)}: two recorders ran on one stream. They count once in the "
                                        'pooled table below.' if dup.any() else ''), '',
              table(rows, ['detector', 'solved', 'ambiguous (own solves)', 'ambiguous (matched frames)', 'range median (m)',
                           'range p90 (m)', 'corners (median)', 'ms per frame']), '']
        if 'live' in st:
            idn = st['live']
            L.append(f"**Live detector:** {idn['detector'] or 'not identified'}: {idn['evidence']}. Agreement with the "
                     f"{idn['frames_with_live']} frames the live pipeline logged:")
            L.append('')
            krows = []
            for k, sc in idn['scores'].items():
                kc = sc.get('k_check', {})
                kc_txt = '; '.join(f"{nm.split(',')[0]}: {v['median_pose_diff_mm']:.1f} mm (this set's: "
                                   f"{v['used_calibration_median_diff_mm']:.1f} mm)" for nm, v in kc.items())
                krows.append([LABELS[k], pc(sc['same_solved_state'], '.0%'), pc(sc['pose_within_1mm'], '.0%'),
                              pc(sc['same_ambiguity'], '.0%'), f"{sc['median_pose_diff_mm']:.2f}", kc_txt or '-'])
            L += [table(krows, ['detector', 'same solved/refused', 'pose within 1 mm', 'same ambiguity flag',
                                'median pose difference (mm)', 'other calibration, median difference']), '']
        write = [[i, int(z['stamps'][i]), *[v for k in keys for v in r[k]['rows'][i, :8]]] for i in range(n)]
        from run_analytics import write_csv
        write_csv(out / 'data' / f'{name}.csv', ['frame', 'stamp_ns'] + [f'{k}_{c}' for k in keys for c in COLS[:8]], write)
    # pooled over the bags, by apparent square size
    edges = [0, 8, 10, 12, 14, 16, 20, 32, 48, 1e9]
    if bag_sets:
        P = {k: {'rows': np.concatenate(pooled[k])} for k in keys}
        spx = np.concatenate(pooled_spx)
        bt = bucket_table(P, spx, edges)
        summary['bags_by_square_px'] = bt
        L += ['## All bag frames, by apparent square size', '',
              'Each frame counted once (duplicated recordings removed). Square size s = fx x 24 mm / z, from the median range of the detectors that solved the frame (frames none '
              'solved have no size and are left out here). Head-on, 14 px is about 0.6 m and 10 px about 0.85 m.', '',
              table([[b['s_px'], b['frames'], b['all_solve']] + [f"{pc(b[f'{k}_solved'], '.0%')} / {pc(b[f'{k}_ambiguous_own'], '.0%')} / "
                                                                   f"{pc(b[f'{k}_ambiguous_matched'], '.0%')}" for k in keys] for b in bt],
                    ['s (px)', 'frames', 'all solve'] + [f'{LABELS[k]}: solved / ambiguous own / matched' for k in keys]), '']
    # held-out, by the teacher label's square size (range_buckets.py's method, all three detectors)
    if 'heldout' in res:
        r = {k: res['heldout'][k] for k in keys}
        spx = r[keys[0]]['s_px']
        lab = np.isfinite(spx)
        bt = bucket_table({k: {'rows': r[k]['rows'][lab]} for k in keys}, spx[lab], edges)
        summary['heldout_by_square_px'] = bt
        tot = {}
        allsolved = np.all([r[k]['rows'][:, 0] == 1 for k in keys], axis=0)
        for k in keys:
            rr = r[k]['rows']
            sv = rr[:, 0] == 1
            tot[k] = {'solved': fin(sv.mean(), 4), 'ambiguous_own': fin(np.nanmean(rr[sv, 1]), 4),
                      'ambiguous_matched': fin(np.nanmean(rr[allsolved, 1]), 4)}
        summary['heldout'] = {'frames': int(len(spx)), 'all_solve': int(allsolved.sum()), 'camera': str(r[keys[0]]['camera']),
                              **tot}
        L += ['## Held-out frames of the offline fine-tune, by square size', '',
              f"The {len(spx)} held-out frames of `ssl_output/offline_20261013-113934/harvest/gen2/heldout` (15:18-16:09 NZDT "
              f"on 2026-10-13), bucketed by the teacher label's square size as in "
              '`ssl_finetune_records/offline_20261013-113934/data/range_buckets.py`, now for all three detectors. Camera: '
              f"{r[keys[0]]['camera']}. All three solve {int(allsolved.sum())} frames.", '',
              table([[LABELS[k], pc(tot[k]['solved']), pc(tot[k]['ambiguous_own']), pc(tot[k]['ambiguous_matched'])]
                     for k in keys], ['detector', 'solved', 'ambiguous (own solves)', 'ambiguous (matched)']), '',
              table([[b['s_px'], b['frames'], b['all_solve']] + [f"{pc(b[f'{k}_solved'], '.0%')} / {pc(b[f'{k}_ambiguous_own'], '.0%')} / "
                                                                   f"{pc(b[f'{k}_ambiguous_matched'], '.0%')}" for k in keys] for b in bt],
                    ['s (px)', 'labelled', 'all solve'] + [f'{LABELS[k]}: solved / ambiguous own / matched' for k in keys]), '',
              'The labels are the teacher\'s own detections (pseudo-labels), so a frame no detector found is not in the '
              'set at all; recall here is relative to what some teacher could label.', '']
    plot_detectors(summary, keys, out)
    L += figure_notes(summary)
    (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=str))
    (out / 'live_detector.json').write_text(json.dumps(live_json, indent=1))
    (out / 'SUMMARY.md').write_text('\n'.join(L) + '\n')
    print(f'wrote {out / "SUMMARY.md"}')


def plot_detectors(summary: dict, keys: list, out: Path) -> None:
    sys.path.insert(0, str(TOOLS))
    from run_analytics import save, style
    plt = style()
    for src, name in (('bags_by_square_px', 'detectors_bags_by_size'), ('heldout_by_square_px', 'detectors_heldout_by_size')):
        bt = summary.get(src)
        if not bt:
            continue
        fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.4), sharex=True)
        x = np.arange(len(bt))
        for i, k in enumerate(keys):
            axs[0].plot(x, [100 * b[f'{k}_solved'] for b in bt], '-o', ms=5, color=C[i], label=LABELS[k])
            axs[1].plot(x, [100 * b[f'{k}_ambiguous_matched'] for b in bt], '-o', ms=5, color=C[i], label=LABELS[k])
        for ax, lab in ((axs[0], 'frames with a pose (%)'), (axs[1], 'ambiguous, on frames all three solve (%)')):
            ax.set_xticks(x)
            ax.set_xticklabels([b['s_px'] for b in bt], fontsize=7)
            ax.set_xlabel('apparent square size (px)' + (', teacher label' if 'heldout' in src else ''))
            ax.set_ylabel(lab)
        axs[0].legend()
        fig.tight_layout()
        save(fig, out, name)


def figure_notes(summary: dict) -> list:
    L = []
    if summary.get('bags_by_square_px'):
        L += ['![Detectors on the bag frames by square size](figures/detectors_bags_by_size.png)', '',
              '**Detectors on the docking-day frames, by apparent square size.** ([PDF](figures/detectors_bags_by_size.pdf))', '',
              '- *What it shows:* for the frames sampled from the image-carrying bags, the share each detector solves (left) '
              'and the share of poses that are ambiguous on the frames all three solve (right), by the board\'s apparent '
              'square size.',
              '- *How:* each detector in its own process through the deployed pipeline; size from the solved range '
              '(fx x 24 mm / z). Matching the frames removes the bias of a detector that solves more far, naturally '
              'ambiguous boards.',
              '- *What it means:* left is detection reach; right is how often IPPE cannot tell the two tilts apart on '
              'the same views, which is a property of the corners\' precision and the view, not of reach.',
              '- *Caveats:* the frames are from bring-up recordings (robot moved by hand), not docking motion; frames none '
              'solved have no size and are absent; small samples in the extreme buckets.', '']
    if summary.get('heldout_by_square_px'):
        L += ['![Detectors on the held-out frames by square size](figures/detectors_heldout_by_size.png)', '',
              '**Detectors on the fine-tune\'s held-out frames, by the label\'s square size.** ([PDF](figures/detectors_heldout_by_size.pdf))', '',
              '- *What it shows:* the same two measures on the offline loop\'s 1,792 held-out frames, bucketed by the '
              'teacher label\'s square size.',
              '- *How:* range_buckets.py\'s method for all three detectors, same frames, same camera matrix.',
              '- *What it means:* whether fine-tuning moved the reach (solve rate at small squares) and whether it changed '
              'ambiguity on views every model solves.',
              '- *Caveats:* pseudo-labels from the same session as the training frames (optimistic); the r1 model was '
              'trained on frames from minutes around these.', '']
    return L


# ============================================================================================ fine-tuning index (CPU)
def finetune(out: Path) -> None:
    """Gather the self-supervised fine-tuning records: sessions, rounds, gates, recall, ambiguity, deployments."""
    sys.path.insert(0, str(TOOLS))
    from run_analytics import git, nz, table
    out.mkdir(parents=True, exist_ok=True)
    L = ['# Fine-tuning records', '',
         f'Gathered from `{SSL_RECORDS.relative_to(DEFAULT_OUT.parent)}/` (each session\'s own README and data) by '
         '`controller_ws/tools/run_analytics_detectors.py finetune`. Recall figures there are on pseudo-labelled '
         'held-out frames from the same session (optimistic; see each record\'s caveats).', '']
    sess_rows, round_rows = [], []
    for d in sorted(p for p in SSL_RECORDS.iterdir() if p.is_dir()):
        data = d / 'data'
        rounds = [json.loads(x) for x in (data / 'rounds.jsonl').read_text().splitlines() if x.strip()] \
            if (data / 'rounds.jsonl').exists() else []
        loop = [json.loads(x) for x in (data / 'loop.jsonl').read_text().splitlines() if x.strip()] \
            if (data / 'loop.jsonl').exists() else []
        summ = json.loads((data / 'summary.json').read_text()) if (data / 'summary.json').exists() else {}
        kind = 'offline loop' if loop else 'live'
        exit_ = next((x for x in loop if x.get('kind') == 'exit'), None)
        t0 = rounds[0]['t_start'] if rounds else None
        sess_rows.append([f'[{d.name}](../../ssl_finetune_records/{d.name}/README.md)', kind,
                          nz(t0).strftime('%Y-%m-%d %H:%M') if t0 else '-', len(rounds),
                          sum(1 for r in rounds if r.get('accepted')),
                          (exit_['reason'] if exit_ else ('ended' if summ else 'running?')),
                          Path(exit_['best_onnx']).parent.name if exit_ and exit_.get('best_onnx') else '-'])
        lr = {x.get('round'): x for x in loop if x.get('kind') == 'round'}
        for r in rounds:
            ref, cand = (r.get('reference') or {}).get('all', {}), (r.get('candidate') or {}).get('all', {})
            far = (lr.get(r['round']) or {}).get('far_recall') or {}
            amb = (lr.get(r['round']) or {}).get('ambiguous_rate') or {}
            gate = r.get('gate') or {}
            round_rows.append([d.name, r['round'], nz(r['t_start']).strftime('%m-%d %H:%M'), r.get('train_frames'),
                               r.get('heldout_frames'), f"{ref.get('recall', '-')} -> {cand.get('recall', '-')}",
                               f"{far.get('teacher', '-')} -> {far.get('candidate', '-')}" if far else '-',
                               f"{ref.get('id_precision', '-')} -> {cand.get('id_precision', '-')}",
                               (f"{amb.get('base', 0):.1%} / {amb.get('teacher', 0):.1%} / {amb.get('candidate', 0):.1%}"
                                if amb else '-'),
                               ', '.join(k for k, v in gate.items() if not v) or 'none', 'yes' if r.get('accepted') else 'no'])
    L += ['## Sessions', '', table(sess_rows, ['session', 'kind', 'first round (NZDT)', 'rounds', 'accepted', 'exit',
                                               'best model']), '',
          '## Rounds', '', table(round_rows, ['session', 'round', 'started', 'train', 'held-out', 'held-out recall ref -> cand',
                                              'far (<16 px) recall', 'ID precision', 'ambiguous base / teacher / cand',
                                              'gate failed', 'accepted']), '']
    hist = git(WIRELESS, 'log', '--format=%h%x09%ct%x09%s', '--', 'cfg/cfg.yaml').splitlines()
    dep = []
    import yaml
    for ln in reversed(hist):
        h, t, s = ln.split('\t', 2)
        c = yaml.safe_load(git(WIRELESS, 'show', f'{h}:cfg/cfg.yaml')) or {}
        det = (c.get('MODEL') or {}).get('detector', '')
        K = (c.get('CAMERA') or {}).get('K')
        dep.append([h, nz(float(t)).strftime('%Y-%m-%d %H:%M'), Path(det).name, (c.get('BOARD') or {}).get('square_length_m'),
                    f'{K[0][0]:.2f}' if K else '-', s])
    L += ['## Deployed model history (cfg/cfg.yaml)', '',
          table(dep, ['commit', 'NZDT', 'detector', 'square (m)', 'fx', 'subject']), '',
          'Commit times can trail the deployment: the SSL live r3 checkpoint was written at 16:04 NZDT on 2026-10-13 and '
          'the LIVE notes name it the detector for that day\'s docking tests, but cfg.yaml committed it at 18:03.', '']
    det = DEFAULT_OUT / 'detectors' / 'summary.json'
    if det.exists():
        s = json.loads(det.read_text())
        h = s.get('heldout')
        if h:
            L += ['## The three detectors on the same held-out frames', '',
                  f"{h['frames']} held-out frames; all three solve {h['all_solve']}. See "
                  '[detectors/SUMMARY.md](../detectors/SUMMARY.md) for the table by square size.', '',
                  table([[LABELS.get(k, k), pc(h[k]['solved']), pc(h[k]['ambiguous_own']), pc(h[k]['ambiguous_matched'])]
                         for k in DETECTORS if k in h], ['detector', 'solved', 'ambiguous (own)', 'ambiguous (matched)']), '']
    (out / 'SUMMARY.md').write_text('\n'.join(L) + '\n')
    print(f'wrote {out / "SUMMARY.md"}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument('step', choices=('prepare', 'infer', 'summarise', 'finetune', 'all'))
    ap.add_argument('arg', nargs='?', default=None)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    a = ap.parse_args()
    if a.step == 'prepare':
        prepare(Path(a.arg) if a.arg else P4P / 'run_data' / 'pi_bags')
    elif a.step == 'infer':
        infer(a.arg)
    elif a.step == 'summarise':
        summarise(a.out / 'detectors')
    elif a.step == 'finetune':
        finetune(a.out / 'finetune')
    else:
        prepare(Path(a.arg) if a.arg else P4P / 'run_data' / 'pi_bags')
        run_all_detectors()
        summarise(a.out / 'detectors')
        finetune(a.out / 'finetune')


def compare_bags(bag_root: Path, out: Path, rerun: bool = False) -> None:
    """
    run_analytics.py --detectors: prepare, each detector in its own process, summarise.

    The GPU runs only for detectors with no cached results (or all of them with rerun), so summarising again
    never takes the GPU.
    """
    prepare(bag_root)
    res = CACHE / 'results'
    sets = [f.stem for f in CACHE.glob('*.npz')] + ['heldout']
    missing = [k for k in DETECTORS if rerun or any(not (res / f'{k}__{n}.npz').exists() for n in sets)]
    for key in missing:
        t = time.time()
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'infer', key])
        if p.returncode != 0:
            raise RuntimeError(f'the {key} detector process failed ({p.returncode})')
        print(f'{key}: {time.time() - t:.0f} s on the GPU')
    summarise(out)


def finetune_entry(out: Path, force: bool = False) -> None:
    finetune(out)


if __name__ == '__main__':
    main()
