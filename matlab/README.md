# MATLAB model of the docking estimator and controller

v2 (2026-10-16): the design as deployed on the scale prototype (`dock_control`), at both scales,
re-run for the report per `MATLAB_RERUN_BRIEF.md`. Results and their summary:
[`results/rerun_v2/RESULTS.md`](results/rerun_v2/RESULTS.md).

## Layout

| Folder | Contents |
|---|---|
| `params/dock_params.m` | every parameter, both scales (`'fullscale'`, `'demo'`), each value marked with its source |
| `ekf/` | roll-back-and-replay EKF: `ekf_create`, `ekf_advance` (step rule), `ekf_fix` (gated / ungated / init / re-anchor + replay), `ekf_node_fix` (ambiguity, ACQUIRE, re-anchoring), `ekf_zv` (zero-velocity reset), `fid_chain` (pre-transform and Jacobian), `run_ekf` (filter alone) |
| `mpc/` | `mpc_step` (QP, shared slack, fallback), `mpc_constraints` (440 rows, per-axis or L1 caps), `mpc_linearise`, `mpc_predict_matrices`, `mecanum_ik/fk`, `sup_step` (seven-stage supervisor) |
| `sim/` | `run_closed_loop` (1 ms truth plant, sensors, EKF, supervisor, MPC), `fiducial_measure` (detector range law through the chain), `make_starts`, `make_truth_trajectory`, `apply_target_velocity_step` |
| `rerun/` | `rerun_job.m` (jobs R1-R7, one MATLAB process per slice), `r1_traj`, `r2_tests`, `r4_tick`, `collect.py` (raw JSON -> CSV, PNG, summary.json) |
| `results/rerun_v2/` | `raw/*.json` (every run, incl. per-tick logs for R5/R6 and per-IMU-sample logs of the nominal run), `tables/*.csv`, `figures/*.png`, `summary.json`, `RESULTS.md`, `logs/` |
| `results/v1_design_2026-10-09/` | the v1 design's figures, tables and documents, for comparison |
| `archive/design_v1_2026-10-09/` | the v1 code (shifted delayed update, N_p 40, L1 cap, three-stage supervisor) |
| `tests/` | the v1 test suite (targets the archived v1 interfaces; the v2 checks are R2 and R4 in `rerun/`) |

## Re-running

```matlab
cd matlab/rerun
rerun_job('R5', 1, 1)        % or any of R1 R1sweep R2 R3 R4 R5 R5demo R6 R7full R7demo
```
then `python collect.py` (pandas, matplotlib) to rebuild `out/rerun_v2/{tables,figures,summary.json}`.
Each job writes `out/rerun_v2/raw/<job>_<part>.json`; slices run in parallel as separate
`matlab -batch "rerun_job('R7full', p, 8)"` processes.

Raw JSON layout: `{job, part, model, runs: [...]}`; closed-loop runs carry the summary metrics
(`docked`, `time_to_dock`, `y_dock`, `th_dock`, `max_pen`, `v_contact`, ...), `nis` (every fix), and
for R5/R6 `tick` (per 50 ms tick: true pose, estimate, command, slack, exit flag, solve time).
