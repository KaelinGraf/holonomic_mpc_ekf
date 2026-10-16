# MATLAB re-run of the deployed design: results (2026-10-16)

Answers `MATLAB_RERUN_BRIEF.md`. Model version `dock_params v2-deployed-2026-10-16` (branch `MATLAB`, folder `matlab/`).
Every number below is in `tables/*.csv` and `summary.json`; per-run raw data in `raw/*.json`.

## What was done and what was not

| Run | Status |
|---|---|
| Section 1 model changes | Done, ported from `dock_control` (repo `holonomic_mpc_ekf`, commit efae8d2, 2026-10-15) where the brief was silent; the brief governs where they differ (no conflict found). |
| R1, R2, R3, R4, R5, R6, R7 | Done. R5 also run for the prototype. R7: tau, delay and noise only; the IMU-bias case was removed. |
| R8 (exactness of the softening) | Not run (optional). The slack was logged on every tick: zero on every tick of R5 at both scales. |
| R9 | Partly: the nominal R5 run is exported per IMU sample and per tick (`tables/R9_*`), with an EKF error +-2 sigma figure. The Table F4 policy paths were not made. |
| R10 (over-confidence stress cases) | Not run (optional). |
| R11 (Python-MATLAB parity) | Not run. |
| Section 2.4 stress effects | Not included in any run: every run uses the baseline plant. |

## Model as run (Section 1)

- **EKF:** roll back and replay (`ekf/ekf_fix.m`): fix applied at the latest entry with stamp <= capture time against that entry's state and covariance, Joseph form, then every later step replayed with its own input, length and zero-velocity flag. Fixes older than the buffer or the last fix are refused. Buffer `round(2 x 0.5 s x rate)` = 200 (full scale, 200 Hz) / 150 (prototype, 150 Hz). Nothing is buffered before the first pose (as `ekf.py`).
- **Step rule** (`ekf_advance.m`): n = max(1, round(dt/5 ms)) steps, floor x h/T0; repeated stamp: nothing; gap > 0.5 s: advance 0.5 s and reset the clock.
- **Start-up:** accelerometer zeroed over 2 s at rest after arming; the first pose after that initialises the filter with R_z + diag(d^2). ACQUIRE fuses a gate-failing pose ungated with R_z + diag(d^2).
- **Gate** NIS <= 100. **Ambiguity rule** implemented (`ekf_node_fix.m`); no ambiguous poses were injected (one solution per fix, d = 0).
- **Re-anchoring:** 5 consecutive gate rejections each within 120 mm (full scale) / 30 mm (prototype) and 3 deg of the one before.
- **Zero-velocity reset:** commands within 1e-4 for 0.3 s; velocity std 20 mm/s (full) / 5 mm/s (proto); position variance +(0.20 m)^2/s (full) / (0.05 m)^2/s (proto); heading +(10 deg)^2/s.
- **IMU mounting yaw:** 0 at both scales (the simulation has no misalignment to correct).
- **MPC:** N_p 80, N_c 10, T_s 50 ms, 440 rows. Q_y = diag(y_tol^-2, y_tgt^-2, theta_tgt^-2) = diag(434, 3906, 1675) full / diag(6944, 62500, 1459) proto; R_u diag(4444, 4444, 1600) / diag(1600, 1600, 400); r_w [16 16 64]; q_w APPROACH [1 0.25 2] -> [1 1 2] (smoothstep), FINAL [1 1 10], HOLD keeps the stage it left. Penetration x <= 10 mm soft, heading corridor off (180 deg), one shared slack, rho_s 5.8e6, slack Hessian entry 200. quadprog active-set, interior-point retry, fallback decelerates by one increment.
- **Caps:** APPROACH per axis, falling (smoothstep) from v_approach to v_align, never faster than one increment a tick; FINAL forward <= v_final, lateral <= v_lat_final, each falling by one increment a tick; HOLD L1 falling to v_final (as `supervisor.py`).
- **Supervisor:** seven stages as `supervisor.py`. Distances to go are measured to the standoff along x (as the code; the brief does not say). ACQUIRE is run at both scales (it passes on the first tick at full scale).

## Settings as run (Section 2)

- **Starts:** 30 random per scale, x uniform in [-1.3, -standoff] m, y in [-0.4, 0.4] m, theta in [-20, 20] deg, kept if camera-to-board range <= 1.2 m, board centre inside the half-field, and at least the standoff from the docked pose; seed 2026. Nominal start (R5 index 0, R6 start 1, R7): (-0.85 m, 0.25 m, -10 deg). R6 starts: (-0.85, 0.25, -10 deg), (-0.80, -0.30, +20 deg), (-0.40, 0.02, +2 deg).
- **Fix noise, full scale:** the detector's range law at **z' = (672/580) l / 0.040 m** (pixel-matched), Gaussian draws of forward (optical z), lateral (optical x) and yaw (rotation about optical y) in the camera frame, carried through the chain's Jacobian into R_z, heading std floored at 1 deg. Fixes refused beyond 1.45 m (16 px squares) or outside the 28.9 deg half-field. 15 Hz, delivered 50 ms after capture.
- **Fix noise, prototype:** the design's diagonal model (4 mm at 1 m, 0.5 deg), heading floored at 1 deg in R_z; 42 deg half-field, 2.5 m range.
- **Truth plant:** first-order wheels (tau 0.3 s full / 0.1 s proto), 1 ms integration, slip offset (std 10 mm/s, ~1 s correlation) scaled with the wheel-implied speed (none at rest: an additive slip made a stopped robot creep up to 30 mm), IMU from the plant's motion. IMU bias: a constant at-rest offset (removed by the zeroing) plus the design's residual after zeroing, [0.008, -0.005] m/s^2 full / [0.03, -0.02] m/s^2 proto, and gyro bias 0.002 / 0.05 deg/s.

## Values for Table F1 (full scale; prototype as deployed)

| Row | Value used |
|---|---|
| q; q_v, q_theta | q = 4 (filter sigma_a 0.004 m/s^2, sigma_r 0.06 deg/s); **q_v 4.5e-8, q_theta 3.3e-12** per 5 ms (0.03 x the design's 1.5e-6, 1.1e-10, from the R1 sweep) |
| fix noise in simulation | detector range law, 200 mm board, 4.0 mm lens at 580 px/rad, evaluated at z' = (672/580) l / 0.040 m, Gaussian |
| re-anchoring; zero-velocity reset | 5 rejections within 120 mm and 3 deg; reset after 0.3 s of zero commands, velocity std 20 mm/s, position (0.20 m)^2/s, heading (10 deg)^2/s |
| R_w, l_x, l_y | 0.1016 m, **0.36 m, 0.24 m** (inside 998 x 598 mm with an 80 mm wheel: margins 37 mm and 19 mm; layout not taken from Meleisea 2023, Fig. 9) |
| v_max, v_lat | v_approach 0.30, **v_align 0.05**, v_final 0.05, **v_lat_final 0.02** m/s; r_max 0.3 rad/s; omega_max 4 rad/s |
| standoff; handover; docked | 0.32 m; \|y\| < 40 mm, \|theta\| < 2 deg, within 60 mm of the standoff along x; x >= -20 mm, speed < 20 mm/s, \|y\| <= 16 mm, \|theta\| <= 1.4 deg |
| start / abort std; ramps | start 120 mm, abort 200 mm; look-at 1.20 -> 0.24 m, slow-down 1.20 -> 0.32 m, q_w 3.20 -> 1.00 m (distance to go to the standoff; with starts inside 1.2 m the q_w ramp is fully at [1 1 2] throughout) |
| camera, board | camera 0.45 m ahead of the chassis centre on the centre line; board centre 0.65 m into the target frame, normal facing the robot (docked camera-to-board range 0.20 m) |

## Results by placeholder

**R1, NEES** (`tables/R1_nees.csv`, `figures/R1_nees.png`, sweep `R1_nees_sweep_fullscale.*`). Filter alone, analytic quintic from (-0.85, 0.20 m) after 2.5 s at rest, board always in view, 20 seeds. Time averaging: mean and median over all filter steps of the 20-run mean NEES.

| | full scale (q_v 4.5e-8) | prototype (deployed) |
|---|---|---|
| 20-run NEES, mean over steps | 5.43 | 2.58 |
| median over steps | 6.28 | 2.80 |
| share of steps inside 3.71-6.48 | 30.5 % | 17.4 % |
| share of time inside 3 sigma (all five states) | 99.8 % | 99.9 % |
| largest NIS of a legitimate fix | 14.4 | 17.8 |

At the design's q_v (1.5e-6) the full-scale NEES was 1.03. The prototype filter is conservative (heading floor 1 deg against 0.5 deg simulated noise; q = 4).

**R2, unit tests** (`tables/R2_*.csv`). Jacobians F, G against central differences 1.4e-10, 3.0e-11; chain Jacobian 3.4e-10. Noise-free tracking: 0.011 mm, 0.003 deg, 0.13 mm/s. Late fix (147 ms) against on time: state and covariance identical (difference 0). Target-velocity step (moving away and sideways, so the board stays in range on the shorter approach): 0.05 m/s recovery 1.45 s, peak 9.3 mm, no fix rejected; 0.2 m/s recovery 0.94 s, peak 71 mm, **5 fixes rejected and 1 re-anchor**. These replace the Table F3 values (0.90 mm ...; 0.75/1.20 s, 10/35 mm, none rejected).

**R3, loss of the board 2 s** (`tables/R3_dropout_*.csv`, 20 seeds, dropout at t = 14-16 s while driving in). Full scale: inside +-48 mm for the whole 2 s in every run; largest error 37 mm; error/3 sigma up to 1.15 (slightly over-confident during the coast at the retuned q_v); every fix accepted on return. Prototype: inside +-12 mm for 0.05 s minimum, 0.39 s median (position norm; lateral alone 0.29 s minimum, 0.85 s median); largest error 174 mm; error/3 sigma <= 0.78; every fix accepted on return.

**R4, constrained tick** (`tables/R4_constrained_tick.csv`). Every limit held at both scales; 440 rows. Full scale u = [0.285, 0.015, 0.017]; largest predicted x 93 mm against x_max 10 mm with slack 83 mm: from 0.3 m/s, N_c du_max = 10 x 0.015 = 0.15 m/s of braking inside the control horizon cannot stop the 4 s plan, so the soft row is violated in the plan (never in closed loop: zero slack ticks in R5). Prototype u = [0.18, -0.025, -0.023]; largest predicted x 10.0 mm, slack 0.

**R5, 30 random starts, estimator in the loop** (`tables/R5_*`, `figures/R5_*`).

| | full scale | prototype |
|---|---|---|
| docked within 40 s | 30 / 30 | 30 / 30 |
| inside the window | 30 / 30 | 30 / 30 |
| inside the internal targets | 30 / 30 | 28 / 30 |
| lateral error at declaration, rms / max | 2.7 / 5.8 mm | 2.2 / 4.6 mm |
| heading error, rms / max | 0.02 / 0.08 deg | 0.30 / 0.73 deg |
| time to dock (from APPROACH), mean / max | 14.2 / 20.0 s | 7.1 / 10.2 s |
| largest penetration past first contact | 0 (no run reaches x = 0) | 3.6 mm |
| contact speed, max | none: declared up to 20 mm short at <= 14 mm/s | 55 mm/s (14 runs touch) |
| fallbacks / rejected fixes / re-anchors / ticks with slack | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |

Nominal start: full scale docked in 17.5 s, y 2.2 mm, theta -0.005 deg; prototype 8.1 s, 1.3 mm, -0.30 deg.

**R6, three fixed starts** (`tables/R6_fixed_starts.csv`, `figures/R6_fixed_starts.png`).

| Scale | Start | Feedback | Time [s] | \|y\| [mm] | \|theta\| [deg] | Penetration [mm] | Contact [mm/s] |
|---|---|---|---|---|---|---|---|
| full | 1 | true state | 16.8 | 0.8 | 0.00 | 0 | none |
| full | 2 | true state | 17.4 | 3.7 | 0.00 | 0 | none |
| full | 3 | true state | 8.8 | 1.2 | 0.03 | 0 | none |
| full | 1 | EKF | 17.4 | 1.5 | 0.02 | 0 | none |
| proto | 1 | true state | 8.2 | 0.2 | 0.00 | 4.8 | 40 |
| proto | 2 | true state | 9.2 | 0.1 | 0.00 | 5.5 | 34 |
| proto | 3 | true state | 6.9 | 0.3 | 0.00 | 4.9 | 39 |
| proto | 1 | EKF | 8.6 | 1.5 | 0.16 | 0.9 | 16 |

**R7, sensitivity** (`tables/R7_summary_*.csv`, `figures/R7_sensitivity_*.png`), nominal start, 10 seeds per value, unknown to filter and controller. Full scale: every value of tau (0.05-0.5 s), delay (0.01-0.2 s) and noise (1-4x) docks 10/10 inside the targets; lateral median 1.9-3.4 mm, max 7.1 mm (tau 0.5 s); time median 17.8-18.1 s. Prototype: 10/10 dock for every value; inside the 4 mm target 6/10 at 0.2 s delay, 3/10 at 4x noise (lateral max 15.8 mm, one run outside the window), 7/10 at tau 0.5 s (penetration 26 mm, time median 12.9 s).

## Notes for the report

- Docking at full scale is declared 20 mm before first contact (the x4 docked threshold), so the simulation has no contact speed or penetration to report at full scale; the speed at the declaration is <= 14 mm/s.
- The "one tick took about 17 ms" figure: at N_p 80 a full-scale tick takes 10 ms mean in MATLAB quadprog on this machine (nominal run).
- The v1 results (design of 2026-10-09) are kept in `matlab/archive/` and `matlab/results/v1_design_2026-10-09/` for comparison.
