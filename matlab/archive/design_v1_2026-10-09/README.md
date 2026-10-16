# MATLAB implementation

Companion code for `design/docking_formulation.pdf`. The design code is written by the
thesis author; the test harness, simulation fixtures and report-figure scripts are provided.

## Who owns what

| Folder | Owner | Contents |
|---|---|---|
| `ekf/` | implementation | `ekf_predict`, `ekf_update`, `run_ekf`. Implemented (2026-10-09); the help text is the contract and matches `design/ekf_equations.pdf`. |
| `mpc/` | implementation | `mecanum_ik`, `mecanum_fk`, `mpc_linearise`, `mpc_predict_matrices`, `mpc_constraints`, `mpc_step`. Implemented (2026-10-09); the help text is the contract and matches `design/mpc_bridge.pdf`. |
| `params/` | shared | `dock_params('fullscale' | 'demo')`. You own the values; the tests read them. |
| `sim/` | fixtures | `make_truth_trajectory` (quintic staged approach with analytic IMU truth), `simulate_sensors` (IMU, fiducial with delay and dropout, RTK for the navigation studies), `plant_init` / `plant_step` (truth chassis with wheel lag and slip for the closed loop) |
| `tests/` | harness | pass/fail tests, tuning sweep, report figures |
| `utils/` | shared | `wrap_pi`, `rot2`, `chi2_bounds`, `getfield_default` |
| `reference/` | sealed | a complete reference implementation. Use it to check that the tests behave (`setup_paths('reference')`) or to diff against your code after you have written it. Delete it if you prefer. |
| `out/` | generated | figures; `out/reference/` holds figures made with the reference |

## Workflow

```matlab
setup_paths                 % your implementation on the path
run_all_tests               % PASS / WARN / FAIL table; stubs show up as FAIL with "notImplemented"
make_ekf_figures            % estimator figures to out/ (DOCK_VARIANT = 'fullscale' or 'demo')
make_mpc_figures            % controller figures, same variant switch
make_report_figures         % Monte Carlo, sensitivity, with/without constraints, estimator-in-loop (about 20 min)
sweep_ekf_tuning            % mean NEES against the velocity process noise, both variants
```

The Python/ROS 2 port for the demonstrator is specified in `design/ros2_implementation_brief.md`;
the files here are its executable reference.

Suggested order, EKF: `ekf_predict` (unlocks `test_ekf_jacobians`), then `ekf_update`, then
`run_ekf` (unlocks the rest). MPC: `mecanum_ik` and `mecanum_fk` (unlocks `test_mpc_kinematics`),
`mpc_linearise`, `mpc_predict_matrices` (unlocks the prediction test, which also compares against
the course's `mpcgain.m`), `mpc_constraints`, then `mpc_step` (unlocks the unconstrained and
constrained tick tests). Each test's help text says what mistake it catches.

## EKF interface

Read the help of the three stubs; it is the contract the tests call. In short:

- `[x, P, F, G] = ekf_predict(x, P, u, T, Sigma_u, Qfloor)` with `x = [px; py; th; vx; vy]`
  in the target frame and `u = [ax; ay; r]` in the body frame. `F`, `G` at the input state.
- `[x, P, nu, S, accepted] = ekf_update(x, P, z, H, R, x_ref, ang_idx, gate)`; the innovation
  is formed against `x_ref` (buffered estimate at capture time), the correction applied to `x`.
- `out = run_ekf(sens, prm, tr, opts)` runs the multi-rate schedule of Section 3.6 and returns
  `t, x, Pdiag, fid_used, fid_rej` and, when truth is given, `err, nees`. The docking filter
  uses the IMU and the fiducial only; RTK belongs to the navigation filter.

## Tests

| Test | What it checks | What a failure usually means |
|---|---|---|
| `test_ekf_jacobians` | `F`, `G` against central finite differences, 1e-6 | Jacobian at the wrong point, missing term, deg/rad |
| `test_ekf_noise_free` | zero noise and bias: errors under 2 mm, 0.05 deg, 5 mm/s | frame or sign error in the rotation, delay bookkeeping |
| `test_ekf_delay` | cruise at 0.2 m/s: lag of about 10 mm without compensation, under 2 mm with | buffer indexed by availability instead of capture time |
| `test_ekf_target_motion` | target starts moving at 0.05 and 0.2 m/s: velocity re-converges within 3 s from fiducial fixes alone | velocity process noise too small, velocity treated as a pure IMU integral |
| `test_ekf_dropout` | 2 s marker loss, both variants: time the error stays inside the coupler tolerance (sets the supervisor timeout), error inside its 3-sigma bound, first fixes after the gap accepted | process noise too small to cover the coast, gate too tight |
| `test_ekf_consistency` | 20-run NEES not above the 95 % band (FAIL) and not below it (WARN); error inside 3 sigma 95 % of the time | Q or R mis-scaled, covariance not propagated, unwrapped heading in the NEES |
| `test_mpc_kinematics` | wheel signs for forward, left and yaw; `fk(ik(u)) = u`; null vector `[1 1 -1 -1]` | sign convention, lever arm `lx + ly`, forward map not the left inverse |
| `test_mpc_linearisation` | `A_k`, `B_k` against finite differences of the kinematic map | continuous Jacobian without `I + Ts`, transposed rotation, linearised at `u = 0` |
| `test_mpc_prediction` | `F`, `Phi` against brute-force stepping of the augmented model, and against `mpcgain.m` for the SISO case at zero heading | block indexing, powers of `Ap` off by one, control horizon not zero-padded |
| `test_mpc_unconstrained` | with limits removed, `mpc_step` equals the closed-form solution | weights or `Yr` wrong, whole `dU` applied, sign of `f` |
| `test_mpc_constraints` | increments, the L1 speed cap `|u_b| + |v_b| <= v_max`, yaw rate, wheel speeds, predicted `x` and heading all inside their limits; `20 Nc + 3 Np` rows; the speed cap binds | constraint blocks in the wrong order or sign, `C1`/`C2`, a box instead of the L1 cap, wheel rows with the wrong kinematics |
| `test_mpc_closed_loop` | perfect-state docking from 3 start poses, both variants: docked in 40 s, inside the internal targets, overshoot below the lag stopping distance, contact speed, wheel limit, no fallbacks | command rotation in the wrong frame, staging never switches, constraints not binding |
| `test_integrated_docking` | EKF and MPC together on the truth plant, both variants | estimator/controller frame or timing mismatches |

Figures: `make_ekf_figures` and `make_mpc_figures` (plane view with predictions, pose time series,
commands/increments/wheel speeds against limits, `r_w` comparison, one tick's prediction against
the outcome). The closed-loop harness is `sim/run_closed_loop`, which runs the truth plant, the
sensors, the EKF on the path, the MPC on the path and the three-stage supervisor; pass
`use_ekf = false` for perfect state feedback to test the controller before the EKF exists.

## MPC interface

`mpc_step(xhat, r_gyro, u_prev, yr, lim, prm)` is one tick: build `x_p` from the EKF estimate
and the gyro, linearise about the current pose and last command, build `F` and `Phi`, assemble the
cost from `prm.mpc.Qy`, `prm.mpc.Ru` and `prm.mpc.rw`, assemble `M`, `gam` from `lim`, solve with
`quadprog`, apply the first increment. The no-penetration and heading-corridor rows are soft with
a linear, exact slack penalty; the other rows are hard. The translational speed cap is the L1
bound `|u_b| + |v_b| <= lim.v_max` (four rows per step), not a box: a box is 41 % faster along
its diagonal and the optimiser yaws the body to use it. The supervisor (in the closed-loop
harness) chooses `yr` and `lim` per stage: standoff pose with the approach cap until aligned, then
the origin with the final cap lowered at the increment rate. On an infeasible QP the safe action
is to decelerate every axis by the largest allowed increment and flag `info.fallback`.

From a shell without the MATLAB desktop:

```bash
matlab -batch "cd('C:/Users/OMEN/Documents/p4p/controller_design/matlab'); setup_paths; run_all_tests"
```
