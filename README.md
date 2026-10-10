# holonomic_mpc_ekf

The docking estimator (EKF), supervisor and MPC of the tabletop demonstrator: the
ROS 2 Jazzy port of the MATLAB design in the docking report, built from the ROS 2
implementation brief and fitted to the nodes that already run on the rig
(`Conv-ChArT-Wireless-Inference`).

## Where it sits

The camera, the pose pipeline and the drive are existing nodes. This repository
adds two, each in its own process (one GIL each, so the solve never delays the
EKF), pinned to its own core with `rt:=true`.

| node | in | out |
|---|---|---|
| `ekf_node` | `imu/data` (p4p_serial_bridge), `inference_result` (convchart_ros) | `dock/estimate` (every IMU sample), `fiducial/fix` (every pose), TF `target -> base_link` (20 Hz) |
| `mpc_node` | `dock/estimate`, `fiducial/fix`, `drive/status` (bridge), service `dock/arm` | `cmd_vel` (to the bridge, 20 Hz), `dock/stage`, `dock/mpc_info` |

The launch file also publishes the static transforms `target -> marker` and
`base_link -> pi_camera`, built from the same constants the EKF uses.

## How it runs

- **ekf_node.** Each IMU message triggers a predict: the filter steps its fixed
  T over the sample's elapsed time and publishes the estimate (every sample) and,
  every 10 steps, the transform. Each `inference_result` triggers a correct: the
  board pose goes through the chain to z, its live covariance through the
  chain's Jacobian to R, and the update is made against the state buffered at
  the image's capture time, then applied to the current state; the corrected
  state goes out with the next IMU sample. Both callbacks share one
  single-threaded executor, so a predict and a correct never overlap.
- **mpc_node.** The estimate callback only keeps the newest message. A 20 Hz
  timer, independent of it, runs the supervisor and then the MPC on that newest
  estimate (at most one IMU period old) and publishes `cmd_vel`, `dock/stage`
  and `dock/mpc_info`.
- **p4p_serial_bridge** subscribes to `cmd_vel`, resends the newest command to
  the firmware on its own timer (`command_rate_hz`), and sends zeros once it is
  older than `cmd_timeout`.

## Layout

```
dock_msgs/                    DockEstimate, DockStage, FiducialFix, MpcInfo
dock_control/
  config/dock_demo.yaml       the one parameter file; mirrors dock_params('demo')
  launch/dock.launch.py
  dock_control/
    ekf.py                    EKF and its ring buffer               (report 2, brief 5)
    fiducial.py               board pose -> z, the noise model     (brief 2)
    mpc.py                    linearisation, prediction, QP        (report 3.1-3.5, brief 6)
    supervisor.py             the stage machine                    (report 3.6, brief 7)
    kinematics.py             mecanum J^- and J^+                  (report eq. 18)
    geometry.py, params.py, rosutil.py
    ekf_node.py, mpc_node.py  the ROS side: subscriptions, messages, timers
  test/                       pytest: one file per module, plus a closed-loop run
```

Everything that does not change at run time is computed once in a constructor:
the static transforms and their inverses, the noise matrices, F and G with their
constant entries, the weights as diagonals, J^-, and the 200 constraint rows
whose left-hand sides never change.

## Build

`dock_control` needs `convchart_interfaces` and `convchart_qos`, so overlay the
inference workspace first:

```bash
source /opt/ros/jazzy/setup.bash
source ~/p4p/Conv-ChArT-Wireless-Inference/src/ros/install/setup.bash
cd ~/p4p/controller_ws
colcon build && source install/setup.bash
pip install daqp==0.10.3            # the QP solver; osqp too for qp_solver:=osqp
colcon test --packages-select dock_control && colcon test-result --verbose
```

On the host with conda active, CMake picks conda's Python for the message
packages and fails (`No module named 'em'`): `conda deactivate` first, or add
`--cmake-args -DPython3_EXECUTABLE=/usr/bin/python3`. The Pi image has neither
problem and carries daqp (see "On the Pi").

## Run

On the Pi, the existing nodes first, the bridge with the brief's timing (a 100 ms
command watchdog, set-points at 50 Hz or faster):

```bash
ros2 launch p4p_serial_bridge bridge.launch.py cmd_timeout:=0.1 command_rate_hz:=50.0
ros2 launch p4p_camera camera.launch.py
ros2 launch dock_control dock.launch.py rt:=true          # records a bag into bags/
```

and the convchart_ros pipeline on the laptop. Then arm the drive and the docking:

```bash
ros2 service call /serial_bridge/arm std_srvs/srv/SetBool "{data: true}"
ros2 service call /dock/arm std_srvs/srv/SetBool "{data: true}"   # false: back to IDLE
```

Launch arguments: `config`, `rt`, `ekf_core`, `mpc_core`, `exposure_offset` (image
stamp minus capture time, from the LED test), `qp_solver` (`daqp` or `osqp`),
`imu_topic`, `record`.

### On the Pi

The Pi runs everything in the `convchart-wireless/pi:jazzy` container of
`Conv-ChArT-Wireless-Inference`, whose image carries `daqp`. Its compose file
mounts this repository from `~/holonomic_mpc_ekf` at `/controller_ws`, and the
container's shell sources `/controller_ws/install` on top of the inference
overlay once it is built:

```bash
cd ~/holonomic_mpc_ekf && git pull                # on the Pi host
docker compose exec pi bash                        # or pi-camera
cd /controller_ws && colcon build && source install/setup.bash
```

## What the existing nodes still have to provide

- **IMU.** The EKF is driven by body-frame linear acceleration (gravity removed)
  and yaw rate at 200 Hz, in `base_link`, stamped at the sample time (the
  sensor's clock re-anchored to ROS time, brief Section 4). The bridge's
  `imu/data` is 50 Hz with yaw and yaw rate only
  (`linear_acceleration_covariance[0] = -1`, which `ekf_node` reports), so the
  dedicated IMU node provides it. Exactly one node may publish the topic the
  EKF reads: run the bridge with `publish_imu:=false`, or point the EKF at the
  IMU node's topic with `imu_topic:=...`. The subscription is BEST_EFFORT, so a
  RELIABLE or a sensor-data publisher both match.
- **Limits.** The firmware saturates the wheels itself (`config.h`: `WHEEL_MAX_MPS`
  0.5, `CHASSIS_L_PLUS_W` 0.24), while the MPC's wheel rows use `Rw`, `lx + ly`
  (0.30) and `w_max` (15 rad/s = 0.76 m/s at the rim) from the YAML. Make them
  agree, or the firmware will scale commands the MPC thought were feasible.
- **Board frame.** `fiducial.R_MARKER_BOARD` assumes the pipeline's lattice
  corner 0 is top-left seen from the front; `marker_m` locates the board frame's
  origin (the lattice's outer corner), not the board centre. Check both with the
  four-pose test before the EKF is switched on (brief Section 10.b).

## Where the code departs from the brief

- **Two nodes, not six.** The camera/detector, IMU and wheel nodes are the
  existing p4p_camera, convchart_ros and p4p_serial_bridge. The fiducial chain is
  a library called in `ekf_node`, and the supervisor a library called in
  `mpc_node`, so the stage is computed in the tick that uses it.
- **QP solver.** DAQP by default; OSQP (the brief's choice) with `qp_solver:=osqp`.
  In closed-loop simulation OSQP ran out of iterations on a few per cent of
  ticks near the dock, even on QPs with no active constraint, each one a
  fallback; DAQP solved every tick exactly in about 0.15 ms. For OSQP the
  objective is divided by `rho_s` (the minimiser is unchanged), without which
  the exact-penalty slack kept it from converging at all.
- **FINAL and HOLD cap.** Ramped down by `min(|u_b|, du_max) + min(|v_b|, du_max)`
  a tick rather than `2 du_max`: driving straight in (`v_b = 0`) only `du_max` is
  reachable, and `2 du_max` made the hard amplitude rows infeasible.
- **Corridor abort.** Only when the estimate stays outside the heading corridor
  for longer than `hold_after_s`: the corridor is soft in the MPC, and an
  immediate abort stopped 3 of 30 runs from the report's start region at once.
- **Covariance abort** checks both position variances, `p_diag[0]` and `p_diag[1]`.
- **Two MPC misses** (fallback or a blown 25 ms budget) hold, as in brief
  Sections 4 and 7; Section 12 says abort. HOLD aborts after 2 s anyway.
- **Ring buffer.** `correct_from` adds each correction to the entries from the
  capture stamp on (the brief's `overwrite_from`). Over WiFi the pose delay
  (107 ms median) exceeds the 67 ms fix period, so consecutive fixes overlap and
  would otherwise correct the same error twice.
- **Measurement noise.** R comes with every pose: the pipeline's live covariance
  (its measured noise at that range) carried through the chain's Jacobian into a
  full 3x3 for [x, y, theta]. A pose without a usable covariance is not fused.
  The brief's range model (`fid.sigma_p0`, `d0`, `sigma_th_deg`) only simulates
  fixes in the tests. `FiducialFix` logs that R, not two fixed sigmas.
- **IPPE ambiguity.** When the pipeline flags a pose `ambiguous`, `ekf_node`
  keeps the candidate with the smaller NIS, and never starts the filter from one.
- **E-stop.** No `/estop` topic: the bridge's `arm` service (false is the software
  e-stop) and `drive/status` take its place.
- **Messages.** `MpcInfo` carries the command `u`; there is no `dock/cmd_body`.
- **IMU zeroing** happens in `ekf_node`, so it works whatever publishes `imu/data`.
- **IMU rate.** The EKF steps its fixed T over each sample's elapsed time
  (`Ekf.advance`), so a dropped sample or a slower stream costs accuracy, not
  time; at 200 Hz that is one step a sample, as in the MATLAB. `ekf_node`
  publishes the measured rate in `DockEstimate.imu_rate`, and the supervisor
  starts moving only with it within 10 % of `imu.rate`.
- **Stale estimate.** A moving stage aborts once the estimate is more than
  100 ms old, so a dead IMU node cannot leave the MPC steering on a frozen state.

## Tests

`test_kinematics`, `test_ekf`, `test_fiducial` and `test_mpc` are the unit tests
of report Tables 2 and 3 and brief Section 11; `test_supervisor` walks the stage
table of brief Section 7; `test_closed_loop` docks a simulated demonstrator from
(-2 m, 0.3 m, 15 deg) with the EKF in the loop against the criteria of report
Table 4. The two replay tests in `test_replay.py`, the acceptance tests for the
port, wait for the MATLAB fixtures (`export_fixtures.m`).

## From the MATLAB

| MATLAB | Python |
|---|---|
| `ekf_predict.m`, `ekf_update.m` | `ekf.Ekf.predict`, `ekf.Ekf.update` |
| `run_ekf.m` (schedule) | `ekf_node.py` |
| fiducial pre-transform | `fiducial.FiducialChain.chain` |
| `mecanum_ik.m`, `mecanum_fk.m` | `kinematics.ik_matrix`, `kinematics.fk_matrix` |
| `mpc_linearise.m` | `mpc.linearise` |
| `mpc_predict_matrices.m` | `mpc.predict_matrices` (with `mpc.augment`) |
| `mpc_constraints.m` | `mpc.Mpc._constant_rows`, `mpc.Mpc.constraints` |
| `mpc_step.m` | `mpc.Mpc.step` |
| `run_closed_loop.m` supervisor | `supervisor.Supervisor` |
| `dock_params.m` | `config/dock_demo.yaml` |
