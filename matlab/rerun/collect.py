"""Collect the MATLAB re-run's raw JSON into tables (CSV), figures (PNG) and summary.json.

Usage: python collect.py   (reads ../out/rerun_v2/raw/*.json, writes ../out/rerun_v2/{tables,figures})
"""
import glob
import json
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', 'out', 'rerun_v2')
RAW, TAB, FIG = (os.path.join(OUT, d) for d in ('raw', 'tables', 'figures'))
os.makedirs(TAB, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
TOL = {'fullscale': dict(y=0.048, tgt_y=0.016, th=20.0, tgt_th=1.4),
       'demo': dict(y=0.012, tgt_y=0.004, th=20.0, tgt_th=1.5)}
BAND = (3.71, 6.48)
summary = {}


def runs(job):
    out = []
    for f in sorted(glob.glob(os.path.join(RAW, f'{job}_*.json'))):
        r = json.load(open(f))['runs']
        out += r if isinstance(r, list) else [r]
    return out


def flat(r, drop=('tick', 'imu', 'nis', 'pose0', 'pose_at_dock')):
    d = {k: v for k, v in r.items() if k not in drop}
    if 'pose0' in r:
        d['x0'], d['y0'], d['th0_deg'] = r['pose0'][0], r['pose0'][1], math.degrees(r['pose0'][2])
    if 'pose_at_dock' in r and r['pose_at_dock'] is not None:
        p = [np.nan if v is None else v for v in r['pose_at_dock']]
        d['x_dock'], d['y_dock_m'], d['th_dock_deg'] = p[0], p[1], math.degrees(p[2])
    if 'nis' in r:
        nis = np.array([np.nan if v is None else v for v in np.atleast_1d(r['nis'])], float)
        nis = nis[np.isfinite(nis)]
        d['nis_median'] = float(np.median(nis)) if nis.size else np.nan
        d['nis_share_above_7_81'] = float(np.mean(nis > 7.81)) if nis.size else np.nan
    for k in ('th_dock', 'max_abs_th'):
        if k in d and d[k] is not None:
            d[k + '_deg'] = math.degrees(d.pop(k))
    return d


def stat(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return dict(n=int(x.size), mean=float(x.mean()) if x.size else None, rms=float(np.sqrt((x ** 2).mean())) if x.size else None,
                median=float(np.median(x)) if x.size else None, max=float(np.abs(x).max()) if x.size else None)


# ---------------- R1 / R1sweep: NEES ----------------
def nees_table(rr, label):
    rows = []
    groups = {}
    for r in rr:
        groups.setdefault((r['variant'], r['qv_factor']), []).append(r)
    for (v, fac), g in sorted(groups.items()):
        n = min(len(r['nees']) for r in g)
        mean_t = np.mean([np.array(r['nees'][:n], float) for r in g], axis=0)
        t = np.array(g[0]['t'][:n], float)
        rows.append(dict(variant=v, qv_factor=fac, q_v=g[0]['q_v'], q_th=g[0]['q_th'], runs=len(g),
                         nees20_mean_over_steps=float(mean_t.mean()), nees20_median_over_steps=float(np.median(mean_t)),
                         share_steps_in_band=float(np.mean((mean_t >= BAND[0]) & (mean_t <= BAND[1]))),
                         share_time_inside_3sigma_all5=float(np.mean([r['in3_all'] for r in g])),
                         **{f'share_inside_3sigma_{s}': float(np.mean([r['in3_each'][i] for r in g]))
                            for i, s in enumerate(['x', 'y', 'th', 'vx', 'vy'])},
                         rms_x_mm=1e3 * float(np.mean([r['rms_err'][0] for r in g])),
                         rms_y_mm=1e3 * float(np.mean([r['rms_err'][1] for r in g])),
                         rms_th_deg=math.degrees(float(np.mean([r['rms_err'][2] for r in g]))),
                         max_nis_legit_fix=float(np.max([r['max_nis'] for r in g])),
                         rejected=int(np.sum([r['n_rej'] for r in g]))))
        rows[-1]['_t'], rows[-1]['_nees'] = t, mean_t
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in rows])
    df.to_csv(os.path.join(TAB, f'{label}.csv'), index=False)
    return df, rows


r1, r1rows = nees_table(runs('R1'), 'R1_nees')
sw, swrows = nees_table(runs('R1sweep'), 'R1_nees_sweep_fullscale')
summary['R1'] = r1.to_dict('records')
summary['R1_sweep_fullscale'] = sw.to_dict('records')
fig, ax = plt.subplots(2, 1, figsize=(9, 6), sharex=False)
for a, r in zip(ax, r1rows):
    a.plot(r['_t'], r['_nees'], lw=0.8)
    a.axhspan(*BAND, color='g', alpha=0.15)
    a.axhline(5, color='k', ls=':')
    a.set_title(f"{r['variant']}: 20-run mean NEES (band {BAND[0]}-{BAND[1]}); mean {r['nees20_mean_over_steps']:.2f}, median {r['nees20_median_over_steps']:.2f}")
    a.set_ylabel('NEES')
ax[-1].set_xlabel('time [s]')
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'R1_nees.png'), dpi=110); plt.close(fig)
fig, ax = plt.subplots(figsize=(7, 4))
ax.semilogx(sw.qv_factor, sw.nees20_mean_over_steps, 'o-', label='mean over steps')
ax.semilogx(sw.qv_factor, sw.nees20_median_over_steps, 's--', label='median over steps')
ax.axhspan(*BAND, color='g', alpha=0.15); ax.axhline(5, color='k', ls=':')
ax.set_xlabel('q_v, q_theta factor on the design values (1.5e-6, 1.1e-10)'); ax.set_ylabel('20-run NEES')
ax.set_title('Full-scale NEES sweep, roll-back filter'); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, 'R1_nees_sweep_fullscale.png'), dpi=110); plt.close(fig)

# ---------------- R2 ----------------
r2 = runs('R2')
if r2:
    r2 = r2[0]
    ts = r2.pop('target_step')
    ts = ts if isinstance(ts, list) else [ts]
    pd.DataFrame([r2]).to_csv(os.path.join(TAB, 'R2_unit_tests.csv'), index=False)
    pd.DataFrame(ts).to_csv(os.path.join(TAB, 'R2_target_step.csv'), index=False)
    summary['R2'] = dict(r2, target_step=ts)

# ---------------- R3 ----------------
r3 = pd.DataFrame(runs('R3'))
if len(r3):
    r3.to_csv(os.path.join(TAB, 'R3_dropout_runs.csv'), index=False)
    s3 = r3.groupby('variant').agg(runs=('seed', 'size'), t_inside_norm_min_s=('t_inside_norm_s', 'min'),
                                    t_inside_norm_median_s=('t_inside_norm_s', 'median'),
                                    t_inside_lateral_min_s=('t_inside_lateral_s', 'min'),
                                    t_inside_lateral_median_s=('t_inside_lateral_s', 'median'),
                                    max_err_mm=('max_err_norm_mm', 'max'),
                                    max_err_over_3sigma=('max_err_over_3sigma', 'max'),
                                    fixes_accepted_on_return=('fixes_accepted_on_return', 'sum'),
                                    fixes_checked_on_return=('fixes_checked_on_return', 'sum')).reset_index()
    s3.to_csv(os.path.join(TAB, 'R3_dropout_summary.csv'), index=False)
    summary['R3'] = s3.to_dict('records')

# ---------------- R4 ----------------
r4 = runs('R4')
if r4:
    pd.DataFrame([dict(r, u=str(r['u'])) for r in r4]).to_csv(os.path.join(TAB, 'R4_constrained_tick.csv'), index=False)
    summary['R4'] = r4


# ---------------- R5 ----------------
def r5(job, v):
    rr = runs(job)
    if not rr:
        return
    df = pd.DataFrame([flat(r) for r in rr]).sort_values('start_index')
    df.to_csv(os.path.join(TAB, f'R5_runs_{v}.csv'), index=False)
    rnd = df[df.start_index > 0]
    d = rnd[rnd.docked == True]
    s = dict(variant=v, starts=int(len(rnd)), docked=int(rnd.docked.sum()), inside_window=int(rnd.in_window.sum()),
             inside_target=int(rnd.in_target.sum()), aborted=int(rnd.aborted.sum()),
             lateral_rms_mm=1e3 * float(np.sqrt((d.y_dock ** 2).mean())), lateral_max_mm=1e3 * float(d.y_dock.abs().max()),
             heading_rms_deg=float(np.sqrt((d.th_dock_deg ** 2).mean())), heading_max_deg=float(d.th_dock_deg.abs().max()),
             time_mean_s=float(d.time_to_dock.mean()), time_max_s=float(d.time_to_dock.max()),
             penetration_max_mm=1e3 * float(rnd.max_pen.max()),
             contact_speed_max_mms=1e3 * float(rnd.v_contact.abs().max()) if rnd.v_contact.notna().any() else None,
             runs_with_contact=int(rnd.v_contact.notna().sum()),
             speed_at_declaration_max_mms=1e3 * float(d.v_at_dock.abs().max()),
             fallbacks=int(rnd.n_fallback.sum()), rejected_fixes=int(rnd.n_fix_rej.sum()),
             reanchors=int(rnd.n_reanchor.sum()), ticks_with_slack=int(rnd.n_slack_ticks.sum()),
             max_heading_in_run_deg=float(rnd.max_abs_th_deg.max()),
             nis_median=float(rnd.nis_median.median()))
    nom = df[df.start_index == 0].iloc[0].to_dict()
    summary[f'R5_{v}'] = dict(summary=s, nominal=nom)
    pd.DataFrame([s]).to_csv(os.path.join(TAB, f'R5_summary_{v}.csv'), index=False)
    t = TOL[v]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    ax[0].add_patch(plt.Rectangle((-t['y'] * 1e3, -t['th']), 2 * t['y'] * 1e3, 2 * t['th'], fill=False, ec='r', ls='--'))
    ax[0].add_patch(plt.Rectangle((-t['tgt_y'] * 1e3, -t['tgt_th']), 2 * t['tgt_y'] * 1e3, 2 * t['tgt_th'], fill=False, ec='g', ls='--'))
    ax[0].plot(d.y_dock * 1e3, d.th_dock_deg, 'ko', ms=4)
    ax[0].set_xlim(-1.3 * t['y'] * 1e3, 1.3 * t['y'] * 1e3); ax[0].set_ylim(-3 * t['tgt_th'], 3 * t['tgt_th'])
    ax[0].set_xlabel('lateral error at declaration [mm]'); ax[0].set_ylabel('heading error [deg]')
    ax[0].set_title(f"{v}: {s['docked']}/{s['starts']} docked, {s['inside_window']} window, {s['inside_target']} target")
    ax[1].hist(d.time_to_dock, bins=12); ax[1].set_xlabel('time to dock from APPROACH [s]')
    ax[2].hist(rnd.max_pen * 1e3, bins=12); ax[2].axvline(10, color='r', ls='--'); ax[2].set_xlabel('largest penetration past first contact [mm]')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f'R5_montecarlo_{v}.png'), dpi=110); plt.close(fig)
    # starts and paths
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(rnd.x0, rnd.y0, 'k.', label='random starts')
    ax.plot(nom['x0'], nom['y0'], 'r*', ms=10, label='nominal start')
    ax.plot(0, 0, 'ks', label='docked pose'); ax.set_aspect('equal'); ax.legend(); ax.grid(True)
    ax.set_xlabel('x_T [m]'); ax.set_ylabel('y_T [m]'); ax.set_title(f'R5 starts ({v}): camera-to-board range <= 1.2 m, board in view')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f'R5_starts_{v}.png'), dpi=110); plt.close(fig)
    # nominal run, per IMU sample, estimate +-2 sigma against truth (R9)
    rn = [r for r in rr if r['start_index'] == 0][0]
    if rn.get('imu'):
        im = pd.DataFrame(rn['imu'])
        im.to_csv(os.path.join(TAB, f'R9_nominal_run_per_sample_{v}.csv'), index=False)
        pd.DataFrame(rn['tick']).to_csv(os.path.join(TAB, f'R9_nominal_run_per_tick_{v}.csv'), index=False)
        fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for a, (c, ch, pc, sc, unit) in zip(ax, [('x', 'xh', 'P11', 1e3, 'mm'), ('y', 'yh', 'P22', 1e3, 'mm'), ('th', 'thh', 'P33', 180 / math.pi, 'deg')]):
            e = (im[ch] - im[c]) * sc
            if c == 'th':
                e = ((im[ch] - im[c] + math.pi) % (2 * math.pi) - math.pi) * sc
            s2 = 2 * np.sqrt(im[pc]) * sc
            a.plot(im.t, e, lw=0.8, label='estimate - truth'); a.plot(im.t, s2, 'r--', lw=0.6, label='+-2 sigma'); a.plot(im.t, -s2, 'r--', lw=0.6)
            a.set_ylabel(f'{c} error [{unit}]'); a.grid(True)
            lim = np.nanpercentile(np.abs(e), 99.5) * 3 + 1e-6
            a.set_ylim(-lim, lim)
        ax[0].legend(); ax[0].set_title(f'Nominal run ({v}): EKF error against the truth with its 2-sigma bound')
        ax[-1].set_xlabel('time [s]')
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f'R9_nominal_ekf_convergence_{v}.png'), dpi=110); plt.close(fig)
        tk = pd.DataFrame(rn['tick'])
        fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
        ax[0].plot(tk.t, tk.x, 'k', tk.t, tk.xh, 'b--'); ax[0].set_ylabel('x [m]')
        ax[1].plot(tk.t, tk.y * 1e3, 'k', tk.t, tk.yh * 1e3, 'b--'); ax[1].set_ylabel('y [mm]')
        ax[2].plot(tk.t, np.degrees(tk.th), 'k', tk.t, np.degrees(tk.thh), 'b--'); ax[2].set_ylabel('theta [deg]')
        ax[3].step(tk.t, tk.u_b, where='post', label='u_b'); ax[3].step(tk.t, tk.v_b, where='post', label='v_b'); ax[3].step(tk.t, tk.r_b, where='post', label='r_b')
        ax[3].legend(); ax[3].set_ylabel('command'); ax[3].set_xlabel('time [s]')
        for a in ax:
            a.grid(True)
            for tt, st in zip(tk.t[1:], np.diff(tk.stage)):
                if st:
                    a.axvline(tt, color='m', ls=':', lw=0.8)
        ax[0].set_title(f'Nominal run ({v}): truth (black), estimate (blue), stage switches (magenta)')
        fig.tight_layout(); fig.savefig(os.path.join(FIG, f'R5_nominal_timeseries_{v}.png'), dpi=110); plt.close(fig)


r5('R5', 'fullscale')
r5('R5demo', 'demo')

# ---------------- R6 ----------------
r6 = runs('R6')
if r6:
    df = pd.DataFrame([flat(r) for r in r6])
    df.to_csv(os.path.join(TAB, 'R6_fixed_starts.csv'), index=False)
    cols = ['variant', 'use_ekf', 'start_index', 'x0', 'y0', 'th0_deg', 'docked', 'time_to_dock', 'y_dock_m', 'th_dock_deg',
            'max_pen', 'v_contact', 'v_at_dock', 'n_fallback']
    summary['R6'] = df[cols].to_dict('records')
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5))
    for a, v in zip(axs, ['fullscale', 'demo']):
        for r in [r for r in r6 if r['variant'] == v]:
            tk = pd.DataFrame(r['tick'])
            a.plot(tk.x, tk.y, '-' if not r['use_ekf'] else '--', label=f"start {r['start_index']}{' EKF' if r['use_ekf'] else ' true state'}")
        a.plot(0, 0, 'ks'); a.set_aspect('equal'); a.grid(True); a.legend(fontsize=8); a.set_title(f'R6 fixed starts ({v})')
        a.set_xlabel('x_T [m]'); a.set_ylabel('y_T [m]')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'R6_fixed_starts.png'), dpi=110); plt.close(fig)

# ---------------- R7 ----------------
for job, v in (('R7full', 'fullscale'), ('R7demo', 'demo')):
    rr = runs(job)
    if not rr:
        continue
    df = pd.DataFrame([flat(r) for r in rr])
    df = df[df.param != 'bias']   # IMU-bias sensitivity removed at the author's request (2026-10-16)
    df.to_csv(os.path.join(TAB, f'R7_runs_{v}.csv'), index=False)
    rows = []
    for (p, val), g in df.groupby(['param', 'value']):
        d = g[g.docked == True]
        rows.append(dict(variant=v, param=p, value=val, runs=len(g), share_docked=float(g.docked.mean()),
                         share_in_window=float(g.in_window.mean()), share_in_target=float(g.in_target.mean()),
                         lateral_median_mm=1e3 * float(d.y_dock.abs().median()) if len(d) else None,
                         lateral_max_mm=1e3 * float(d.y_dock.abs().max()) if len(d) else None,
                         heading_median_deg=float(d.th_dock_deg.abs().median()) if len(d) else None,
                         heading_max_deg=float(d.th_dock_deg.abs().max()) if len(d) else None,
                         time_median_s=float(d.time_to_dock.median()) if len(d) else None,
                         time_max_s=float(d.time_to_dock.max()) if len(d) else None,
                         penetration_max_mm=1e3 * float(g.max_pen.max()), aborted=int(g.aborted.sum()),
                         fallbacks=int(g.n_fallback.sum())))
    s7 = pd.DataFrame(rows)
    s7.to_csv(os.path.join(TAB, f'R7_summary_{v}.csv'), index=False)
    summary[f'R7_{v}'] = rows
    fig, ax = plt.subplots(3, 3, figsize=(12, 9))
    for c, p in enumerate(['tau', 'delay', 'noise']):
        g = s7[s7.param == p].sort_values('value')
        ax[0, c].plot(g.value, g.share_docked, 'o-'); ax[0, c].set_ylim(-0.05, 1.05); ax[0, c].set_title(p)
        ax[1, c].plot(g.value, g.lateral_median_mm, 'o-', label='median'); ax[1, c].plot(g.value, g.lateral_max_mm, 's--', label='max')
        ax[1, c].axhline(TOL[v]['tgt_y'] * 1e3, color='g', ls='--')
        ax[2, c].plot(g.value, g.time_median_s, 'o-', label='median'); ax[2, c].plot(g.value, g.time_max_s, 's--', label='max')
        ax[2, c].set_xlabel(p)
        for r in range(3):
            ax[r, c].grid(True)
    ax[0, 0].set_ylabel('share docked'); ax[1, 0].set_ylabel('|lateral| at dock [mm]'); ax[2, 0].set_ylabel('time to dock [s]')
    ax[1, 0].legend(); ax[2, 0].legend()
    fig.suptitle(f'R7 sensitivity ({v}), 10 seeds per value, filter and controller unaware')
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f'R7_sensitivity_{v}.png'), dpi=110); plt.close(fig)


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


json.dump(clean(summary), open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
print('collected:', ', '.join(summary))
