#!/usr/bin/env python3
"""
How the IMU is rotated against the camera (both are glued on), from a bag of imu/data and the board poses.

    python3 tools/imu_camera_alignment.py <bag dir> [--config dock_demo.yaml] [--max-shift-ms 60]
    python3 tools/imu_camera_alignment.py --selftest

The EKF removes gravity with the IMU's own tilt, so a tilted IMU is harmless; what matters is how the IMU's x/y axes
are turned against base_link, whose axes the camera mounting defines (fiducial.R_LINK_OPTICAL: the camera faces +x).
A yaw offset psi mixes the accelerations: a_base = R(psi) a_imu, so 5 deg leaks 9% of a forward acceleration into
the lateral one. Two independent estimates:

1. Rotations (hand-eye), the accurate one, and the whole 3-D rotation. With the board still, the IMU's turn between
   two times and the camera's turn between the same times are the same rotation seen in two frames:
   A = X' B X, with A from the IMU's orientation (its game rotation vector), B = R_CM(i) R_CM(j)' from the board
   poses, X = R_CI. So the turns' axes satisfy b = X a, and a Kabsch fit over many pairs gives X. Needs turns about
   more than one axis: with the robot disarmed and the board in view, pick it up or tilt it and turn it slowly about
   all three axes for a minute. Yaw-only turns (driving) fix only where the IMU's z points.
2. Accelerations, from a docking bag, no extra test: the base's acceleration in the target frame from the fixes
   (fiducial/fix, smoothed twice) against the IMU's gravity-free (ax, ay) turned into the target frame by the fix
   heading. In 2-D the rotations commute, so the yaw offset and a scale follow from a closed-form Procrustes fit.

Both search a small time shift between the IMU and the camera stamps (the exposure offset) and report the residual.
Reports psi (and, from 1, the IMU's tilt against base_link), with a bootstrap spread.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np

GRAVITY = 9.80665
R_LINK_OPTICAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def quat_to_rot(q):
    """(x, y, z, w) to a rotation matrix; q is (..., 4)."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def rot_to_quat(R):
    R = np.asarray(R, float)
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    return np.array([x, y, z, w])


def log_rot(R):
    """The rotation vector (axis * angle) of R."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if ang < 1e-9:
        return v / 2.0
    return v * ang / (2.0 * math.sin(ang))


def exp_rot(v):
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.eye(3)
    k = v / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def zyx(R):
    """Yaw, pitch, roll [deg] of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return (math.degrees(math.atan2(R[1, 0], R[0, 0])), math.degrees(-math.asin(max(-1.0, min(1.0, R[2, 0])))),
            math.degrees(math.atan2(R[2, 1], R[2, 2])))


def slerp_at(t_src, q_src, t):
    """Quaternions q_src (stamps t_src, ascending) at times t, by slerp; nan outside the span."""
    q_src = np.asarray(q_src, float)
    out = np.full((len(t), 4), np.nan)
    j = np.searchsorted(t_src, t)
    for k, (tk, jk) in enumerate(zip(t, j)):
        if jk <= 0 or jk >= len(t_src):
            continue
        q0, q1 = q_src[jk - 1], q_src[jk]
        if np.dot(q0, q1) < 0:
            q1 = -q1
        u = (tk - t_src[jk - 1]) / max(t_src[jk] - t_src[jk - 1], 1e-9)
        d = min(1.0, float(np.dot(q0, q1)))
        if d > 0.9995:
            q = q0 + u * (q1 - q0)
        else:
            th = math.acos(d)
            q = (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)
        out[k] = q / np.linalg.norm(q)
    return out


def kabsch(a, b, w=None):
    """The rotation X minimising sum w |b - X a|^2 over rows a, b (3-vectors)."""
    w = np.ones(len(a)) if w is None else w
    H = (a * w[:, None]).T @ b
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def rotation_fit(t_imu, q_imu, t_cam, R_cm, shift=0.0, lag_s=(0.3, 1.5), min_deg=3.0, rng=None, boot=0):
    """
    X = R_CI from turn pairs. q_imu: IMU orientation (body to world); R_cm: board in the optical frame.
    Pairs (i, j) of camera frames lag_s apart whose turn exceeds min_deg. Returns (X, residual deg, pairs, boot Xs).
    """
    q_at = slerp_at(t_imu, q_imu, t_cam + shift)
    ok = ~np.isnan(q_at[:, 0])
    t, Rw, Rc = t_cam[ok], quat_to_rot(q_at[ok]), R_cm[ok]
    a, b, w = [], [], []
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            dt = t[j] - t[i]
            if dt < lag_s[0]:
                continue
            if dt > lag_s[1]:
                break
            A = Rw[i].T @ Rw[j]
            B = Rc[i] @ Rc[j].T
            va, vb = log_rot(A), log_rot(B)
            ang = np.linalg.norm(va)
            if math.degrees(ang) < min_deg or abs(np.linalg.norm(vb) - ang) > math.radians(3.0):
                continue
            a.append(va / ang)
            b.append(vb / np.linalg.norm(vb))
            w.append(ang)
    if len(a) < 10:
        return None
    a, b, w = np.array(a), np.array(b), np.array(w)
    X = kabsch(a, b, w)
    ev = np.sort(np.linalg.eigvalsh((a * w[:, None]).T @ a))[::-1]
    spread = float(ev[1] / ev[0])
    res = math.degrees(float(np.sqrt(np.average(np.sum((b - a @ X.T) ** 2, 1), weights=w))))
    boots = []
    rng = rng or np.random.default_rng(0)
    for _ in range(boot):
        k = rng.integers(len(a), size=len(a))
        boots.append(kabsch(a[k], b[k], w[k]))
    return X, res, len(a), boots, spread


def smooth(y, n):
    """Centered moving average over n samples (odd), edges shortened."""
    if n <= 1:
        return y
    k = np.ones(n) / n
    pad = n // 2
    yp = np.pad(y, ((pad, pad),) + ((0, 0),) * (y.ndim - 1), mode="edge")
    return np.apply_along_axis(lambda c: np.convolve(c, k, mode="valid"), 0, yp)


def accel_fit(t_imu, a_imu, t_fix, xyt, shift=0.0, smooth_s=0.4, min_acc=0.05):
    """
    psi (yaw of the IMU against base) and scale from accelerations. a_imu: (n, 2) gravity-free IMU ax, ay;
    xyt: (m, 3) base pose (x, y, theta) in the target frame from the fixes. Returns (psi deg, scale, residual, n).
    """
    rate = 1.0 / np.median(np.diff(t_fix))
    n = max(3, int(round(smooth_s * rate)) | 1)
    p = smooth(xyt[:, :2], n)
    v = np.gradient(p, t_fix, axis=0)
    a_cam = np.gradient(smooth(v, n), t_fix, axis=0)
    th = np.unwrap(xyt[:, 2])
    m = max(3, int(round(smooth_s / np.median(np.diff(t_imu)))) | 1)
    a_s = smooth(a_imu, m)
    ai = np.column_stack([np.interp(t_fix + shift, t_imu, a_s[:, k], left=np.nan, right=np.nan) for k in (0, 1)])
    c, s = np.cos(th), np.sin(th)
    b = np.column_stack([c * ai[:, 0] - s * ai[:, 1], s * ai[:, 0] + c * ai[:, 1]])
    ok = ~np.isnan(b).any(1) & (np.linalg.norm(a_cam, axis=1) > min_acc)
    ok[: 2 * n] = ok[-2 * n:] = False
    if ok.sum() < 30:
        return None
    b, ac = b[ok], a_cam[ok]
    cross = float(np.sum(b[:, 0] * ac[:, 1] - b[:, 1] * ac[:, 0]))
    psi = math.atan2(cross, float(np.sum(b[:, 0] * ac[:, 0] + b[:, 1] * ac[:, 1])))
    Rp = np.array([[math.cos(psi), -math.sin(psi)], [math.sin(psi), math.cos(psi)]])
    rb = b @ Rp.T
    scale = float(np.sum(rb * ac) / np.sum(rb * rb))
    res = float(np.sqrt(np.mean(np.sum((ac - scale * rb) ** 2, 1))))
    return math.degrees(psi), scale, res, int(ok.sum())


def gravity_free(acc, quat):
    """The IMU's ax, ay with gravity removed by its own roll and pitch (ekf_node, geometry.body_gravity_xy)."""
    x, y, z, w = quat.T
    gx, gy = 2.0 * GRAVITY * (x * z - w * y), 2.0 * GRAVITY * (y * z + w * x)
    return np.column_stack([acc[:, 0] - gx, acc[:, 1] - gy])


def read_bag(path: Path) -> dict:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(path), storage_id=""),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    want = {k: v for k, v in types.items() if k.endswith(("imu/data", "inference_result", "fiducial/fix"))}
    cls = {k: get_message(v) for k, v in want.items()}
    out = {"imu": [], "pose": [], "fix": []}
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic not in cls:
            continue
        m = deserialize_message(data, cls[topic])
        st = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if topic.endswith("imu/data"):
            o, a = m.orientation, m.linear_acceleration
            out["imu"].append((st, o.x, o.y, o.z, o.w, a.x, a.y, a.z))
        elif topic.endswith("inference_result"):
            if m.reason or m.ambiguous:
                continue
            q, p = m.pose.pose.orientation, m.pose.pose.position
            out["pose"].append((st, q.x, q.y, q.z, q.w, p.x, p.y, p.z))
        else:
            if m.ambiguous:
                continue
            out["fix"].append((st, m.x, m.y, m.theta))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def report(d: dict, max_shift_ms: float, boot: int = 200) -> None:
    imu, pose, fix = d["imu"], d["pose"], d["fix"]
    print(f"imu/data {len(imu)} samples, board poses {len(pose)} (unambiguous), fixes {len(fix)}")
    if len(imu) < 100:
        sys.exit("not enough imu/data")
    t_imu, q_imu = imu[:, 0], imu[:, 1:5]
    shifts = np.arange(-max_shift_ms, max_shift_ms + 1e-9, 5.0) / 1e3
    if len(pose) >= 30:
        t_cam, R_cm = pose[:, 0], quat_to_rot(pose[:, 1:5])
        fits = [(s, rotation_fit(t_imu, q_imu, t_cam, R_cm, shift=s)) for s in shifts]
        fits = [(s, f) for s, f in fits if f is not None]
        if fits:
            s, (X, res, n, _, spread) = min(fits, key=lambda sf: sf[1][1])
            _, _, _, boots, _ = rotation_fit(t_imu, q_imu, t_cam, R_cm, shift=s, boot=boot)
            R_bi = R_LINK_OPTICAL @ X
            y, p, r = zyx(R_bi)
            ys = np.array([zyx(R_LINK_OPTICAL @ B)[0] for B in boots])
            print(f"\n1. rotations: {n} turn pairs, best stamp shift {s * 1e3:+.0f} ms, axis residual {res:.2f} deg")
            print(f"   IMU against base_link: yaw {y:+.2f} deg (bootstrap sd {ys.std():.2f}), pitch {p:+.2f}, "
                  f"roll {r:+.2f}")
            print(f"   R_base_imu =\n{np.array2string(R_bi, precision=4, suppress_small=True)}")
            if spread < 0.1:
                print(f"   NOT TRUSTWORTHY: the turns are nearly all about one axis (spread {spread:.3f} < 0.1),"
                      " so the IMU's yaw is not observable. Tilt and turn the robot about all three axes"
                      " in front of the board for a minute, and use that bag.")
        else:
            print("\n1. rotations: too few turns (need turns of a few degrees about more than one axis)")
    if len(fix) >= 60:
        a_imu = gravity_free(imu[:, 5:7], q_imu)
        a_imu = a_imu - np.median(a_imu[: max(10, len(a_imu) // 20)], axis=0)
        fits = [(s, accel_fit(t_imu, a_imu, fix[:, 0], fix[:, 1:4], shift=s)) for s in shifts]
        fits = [(s, f) for s, f in fits if f is not None]
        if fits:
            s, (psi, scale, res, n) = min(fits, key=lambda sf: sf[1][2])
            print(f"\n2. accelerations: {n} samples above 0.05 m/s^2, best stamp shift {s * 1e3:+.0f} ms: "
                  f"yaw {psi:+.2f} deg, scale {scale:.3f}, residual {res:.3f} m/s^2")
            if n < 300 or not 0.8 < scale < 1.25:
                print("   NOT TRUSTWORTHY: too few accelerating samples, or a scale far from 1. Drive (or push)"
                      " the robot back and forth and sideways in front of the board for a minute.")
        else:
            print("\n2. accelerations: too little motion in the fixes (needs the base accelerating, board in view)")


def selftest() -> None:
    """Simulate a known mounting, then recover it with both fits."""
    rng = np.random.default_rng(4)
    X_true = R_LINK_OPTICAL.T @ (rot_z(math.radians(4.0)) @ exp_rot(np.radians([1.5, -2.0, 0.0])))
    t = np.arange(0.0, 60.0, 1 / 150)
    rv = np.column_stack([0.35 * np.sin(0.9 * t), 0.3 * np.sin(1.3 * t + 1), 0.5 * np.sin(0.7 * t + 2)])
    R_wb = np.array([exp_rot(v) for v in rv])
    R_wi = R_wb @ (R_LINK_OPTICAL @ X_true)
    q_imu = np.array([rot_to_quat(R) for R in R_wi])
    t_cam = np.arange(0.0, 60.0, 1 / 15) + 0.02
    R_wm = exp_rot(np.array([0.1, -0.2, 2.9]))
    R_wc = np.array([exp_rot(v) for v in np.column_stack([np.interp(t_cam - 0.02, t, rv[:, k]) for k in range(3)])])
    R_wc = R_wc @ R_LINK_OPTICAL
    R_cm = np.array([Rc.T @ R_wm @ exp_rot(rng.normal(0, math.radians(0.15), 3)) for Rc in R_wc])
    s_best = min(((s, rotation_fit(t, q_imu, t_cam, R_cm, shift=s)) for s in np.arange(-0.05, 0.051, 0.005)),
                 key=lambda sf: sf[1][1])
    assert s_best[1][4] > 0.1
    y, p, r = zyx(R_LINK_OPTICAL @ s_best[1][0])
    print(f"rotations: yaw {y:+.2f} (true +4.00), shift {s_best[0] * 1e3:+.0f} ms (true -20), residual "
          f"{s_best[1][1]:.2f} deg")
    assert abs(y - 4.0) < 0.3 and abs(s_best[0] + 0.02) < 0.006
    t = np.arange(0.0, 40.0, 1 / 150)
    pos = np.column_stack([-1.0 + 0.15 * np.sin(0.8 * t), 0.1 * np.sin(1.1 * t + 0.5)])
    th = 0.2 * np.sin(0.3 * t)
    acc_t = np.column_stack([-0.15 * 0.64 * np.sin(0.8 * t), -0.1 * 1.21 * np.sin(1.1 * t + 0.5)])
    c, s = np.cos(th), np.sin(th)
    acc_b = np.column_stack([c * acc_t[:, 0] + s * acc_t[:, 1], -s * acc_t[:, 0] + c * acc_t[:, 1]])
    psi = math.radians(4.0)
    acc_i = acc_b @ np.array([[math.cos(psi), math.sin(psi)], [-math.sin(psi), math.cos(psi)]]).T
    acc_i += rng.normal(0, 0.022, acc_i.shape)
    t_fix = np.arange(0.0, 40.0, 1 / 15)
    xyt = np.column_stack([np.interp(t_fix, t, pos[:, 0]), np.interp(t_fix, t, pos[:, 1]), np.interp(t_fix, t, th)])
    xyt[:, :2] += rng.normal(0, 0.0003, (len(t_fix), 2))
    f = accel_fit(t, acc_i, t_fix, xyt)
    print(f"accelerations: yaw {f[0]:+.2f} (true +4.00), scale {f[1]:.3f}, residual {f[2]:.3f}")
    assert abs(f[0] - 4.0) < 1.0 and abs(f[1] - 1.0) < 0.1
    print("selftest passed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag", nargs="?", type=Path)
    ap.add_argument("--max-shift-ms", type=float, default=60.0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    if a.bag is None:
        ap.error("give a bag, or --selftest")
    report(read_bag(a.bag), a.max_shift_ms)


if __name__ == "__main__":
    main()
