# Report figures

All figures are generated from `matlab/` with the implemented estimator and controller (`setup_paths`, then `make_ekf_figures`, `make_mpc_figures`, `make_report_figures`; `DOCK_VARIANT = demo | fullscale`). Every figure exists for both instances: `demo` (tabletop VEX robot) and `fullscale` (600 kg barrier robot, simulation only). The numbers quoted below are in `report_summary_<variant>.txt` and in the test output of `run_all_tests`. `figure_catalogue.pdf` in this folder shows every figure with the same explanations.

Common conventions: target frame with the origin at the docked chassis position and `x` into the target; the robot approaches from `x < 0`; black = truth, blue dashed = EKF estimate, green dashed = internal target or engagement band, red dashed = limit or tolerance, magenta dotted = supervisor stage switch.

---

## Estimator figures (`ekf_<variant>_*.png`)

Produced by `make_ekf_figures`: analytic truth trajectory (`make_truth_trajectory`: quintic staged approach from (−2 m, 0.3 m, 15°) to the standoff and on to the dock, 27.5 s), simulated sensors (`simulate_sensors`: 200 Hz IMU with white noise and residual bias, 15 Hz fiducial fixes with range-dependent noise and 50 ms delay), a 2 s marker dropout at t = 18 to 20 s, "handover" initialisation (truth plus a draw from P0).

| File | What it shows | What it supports |
|---|---|---|
| `ekf_<v>_plane.png` | Plane view in the target frame: truth path, EKF estimate, the fiducial fixes as points, the docked pose. | Context figure for the estimator section: the estimate rides on the truth, the fixes scatter with the range-dependent noise, the estimate coasts through the dropout. |
| `ekf_<v>_errors.png` | Five panels, estimation error of x, y, θ, v_x, v_y against time with the filter's own ±3σ bounds; grey band = dropout; green = internal target. | Consistency by inspection (errors inside the bounds about 99 % of the time), error growth during the dropout and re-convergence after it, the velocity-error level (10 to 15 mm/s) that the MPC inherits through its initial increment. |
| `ekf_<v>_nees.png` | Top: single-run NEES against time with the 95 % chi-square bounds (5 dof). Bottom: mean NEES of 20 Monte Carlo runs with the band. | The course's consistency test, used to choose the velocity process noise q_v. Full scale 3.21 (slightly below the band, conservative), demo 5.32 (inside). |
| `ekf_<v>_delay.png` | Position-error magnitude during the constant-velocity cruise with delay compensation on and off, with the line speed × τ_d; true speed below. | Justifies the ring-buffer delayed innovation: without it a 10 mm lag bias at 0.2 m/s (comparable to the demo tolerance), with it 1.6 mm. |
| `ekf_<v>_dropout.png` | Zoom on the dropout: position-error magnitude against the 3σ bound, v_x error below. | Sets the supervisor's hold and abort times: full scale stays inside the 80 mm tolerance for the whole 2 s; the demo leaves 12 mm after about 0.2 s, so the demonstrator needs continuous marker visibility. |
| `ekf_<v>_target_motion.png` | The target starts moving at 0.2 m/s at t_step (the robot's IMU cannot see it): x error and v_x error over the next 6 s. | The stationary-target assumption degrades gracefully: the filter re-converges from fixes alone within 1.2 s with no fix rejected by the gate. Also the evidence for the gate value (a tight gate rejected every fix here and diverged). |

---

## Controller figures (`mpc_<variant>_*.png`)

Produced by `make_mpc_figures`: closed loop (`run_closed_loop`) with the EKF in the loop from (−2 m, 0.3 m, 15°), seed 1, truth plant with wheel lag (0.3 s full scale, 0.1 s demo) and a slip offset, three-stage supervisor.

| File | What it shows | What it supports |
|---|---|---|
| `mpc_<v>_plane.png` | Path in the target frame: true path, EKF estimate, and the MPC's predicted trajectories (red) at ticks 1, 40, 120, 200, 300; standoff and docked pose markers. | Receding horizon with successive linearisation: each prediction is the linear model's 2 s forecast from that tick; the forecasts curve as the linearisation point moves and converge onto the path. |
| `mpc_<v>_timeseries.png` | x, y, θ against time, true and estimated, with the stage switches, the standoff and contact lines and the internal targets. | The two-stage approach: alignment at the standoff, final approach at 50 mm/s, docking time (demo 12.4 s, full scale 14.5 s). Demo heading settles within ±1° after the initial turn; full scale shows the ±1° yaw ripple against the 0.3 s wheel lag the kinematic model omits. |
| `mpc_<v>_commands.png` | Body-velocity commands with the approach and final caps; increments with the translation and yaw limits; the four wheel speeds with the wheel limit (the course's output / input / increment layout). | Which constraints are active when: the increment limit during the initial turn and the deceleration at the standoff; at full scale the wheel limit during the approach; the L1 speed cap. The wheel limit is never exceeded. |
| `mpc_<v>_rw.png` | x(t) and the forward command for r_w = 0.1, 1 and 10, with docking times. | The course's single tuning knob: smaller r_w gives faster, more aggressive increments; the design uses r_w = 1. |
| `mpc_<v>_tick.png` | One tick's predicted x over the 2 s horizon (red) against what actually happened (black), with the tick and the end of the control horizon marked. | Prediction quality of the per-tick linear model, the Example 5.3 flavour of the notes. |

---

## Report figures (`report_<variant>_*.png`)

Produced by `make_report_figures`, all closed loop with the EKF in the loop.

| File | What it shows | What it supports |
|---|---|---|
| `report_<v>_montecarlo.png` | 30 runs from start poses drawn uniformly in x ∈ [−2.5, −0.8] m, y ∈ [−0.4, 0.4] m, θ ∈ [−20°, 20°] with independent noise seeds. Top left: final lateral and yaw error against the coupler tolerance (red box) and the internal target (green box). Then docking time, penetration past first contact (red line: end of the 10 mm engagement band) and closing speed at contact (red line: final cap). | The main repeatability result. Full scale: 30/30 inside tolerance and target, lateral 3.5 mm rms / 9.2 mm max, yaw ≤ 0.61°, docking 13.3 s mean / 17.9 s max, penetration ≤ 8.2 mm, contact ≤ 12 mm/s. Demo: 30/30 inside tolerance, 26/30 inside the 4 mm target, lateral 2.4 mm rms / 5.6 mm max, yaw ≤ 0.26°, docking 12.0 s mean / 15.2 s max, penetration ≤ 10.1 mm, contact ≤ 43 mm/s. The four demo runs outside the target are estimator-limited (4 mm fiducial noise per fix equals the target). No solver fallback in any run. |
| `report_<v>_sensitivity.png` | 3 × 4 grid: lateral error at docking, penetration and docking time against the wheel-loop time constant τ, the fiducial delay, the sensor-noise scale and the IMU-bias scale, one at a time, with the filter and controller unaware of the change. Blue dotted = nominal value, green = target or band, red × = did not dock within 40 s. | The design's margins. Full scale is insensitive: τ up to 0.5 s, delay up to 0.2 s, bias up to 4× all dock inside the targets; only noise at 4× prevents the alignment condition being met. Demo: docks with τ up to 0.3 s, delay up to 0.1 s, noise up to 4×, bias up to 2×; does not dock at τ = 0.5 s, delay 0.2 s or bias 4×. These fix the bench requirements: τ < 0.2 s, latency < 0.1 s, residual bias within 2× of 0.03 m/s². |
| `report_<v>_constraints.png` | The nominal run with the constraints in the optimiser (black) against the same controller with every limit removed from the QP and the command clipped to the limits afterwards (red): x(t), x near the contact plane, largest wheel speed. | Honest comparison: in the nominal run the two behave alike because the active limits (speed cap, wheel limit) are enforced equally by clipping and the output constraints are inactive. The constraints matter for the output rows, which clipping cannot enforce, and for the feasibility guarantee; the nominal performance does not rest on them. |
| `report_<v>_estimator.png` | Perfect state feedback (black) against the EKF in the loop, true pose (blue) and estimate (dotted): x, y, θ. | The estimator's cost: one to two seconds of docking time (15.8 vs 14.6 s full scale, 15.3 vs 13.5 s demo) and a few millimetres of lateral error (2.6 vs 0.6 mm; 1.3 vs 0.2 mm). Certainty equivalence holds for this loop. |
| `report_summary_<v>.txt` | The numbers behind the four figures above, as printed by the script. | Source for the tables in the report section. |

---

## Which figures go where in the report

- Estimator design and tuning: `ekf_<v>_nees` (tuning), `ekf_<v>_delay` (delayed innovation), `ekf_<v>_errors` (consistency), `ekf_<v>_dropout` (supervisor timing), `ekf_<v>_target_motion` (assumption robustness, gate).
- Controller design: `mpc_<v>_plane` (receding horizon), `mpc_<v>_commands` (constraints active), `mpc_<v>_rw` (tuning knob), `mpc_<v>_tick` (prediction quality).
- Results: `mpc_<v>_timeseries` (one annotated run), `report_<v>_montecarlo` (repeatability), `report_<v>_sensitivity` (margins), `report_<v>_estimator` (estimator in the loop), `report_<v>_constraints` (why MPC, stated honestly).
- Usually one variant in the body and the other in an appendix; the demo is the one the hardware results will sit beside.
