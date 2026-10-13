# holonomic_mpc_ekf

The docking estimator (EKF), supervisor and MPC of the tabletop demonstrator: the
ROS 2 Jazzy port of the MATLAB design in the docking report, built from the ROS 2
implementation brief and fitted to the nodes that already run on the rig
(`Conv-ChArT-Wireless-Inference`).

## Where it sits

The camera, the pose pipeline and the drive are existing nodes. This repository
adds two, each in its own process (one GIL each, so the solve never delays the
EKF), pinned to its own core with `rt:=true`. On the Pi, `pi.launch.py` (with its
default `rt:=true`) gives the four cores one job each: the camera on core 1, the
EKF on core 2 and the MPC on core 3, each at SCHED_FIFO 50, and everything else
(the IMU, the bridge, the bag, the transforms) on core 0 (`camera_core`,
`ekf_core`, `mpc_core`, `other_core`).

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
p4p_nodes/                    the Pi's own nodes, moved here from Conv-ChArT-Wireless-Inference
  p4p_camera/                 the OV2311 stream and the image_full_res service
  p4p_serial_bridge/          cmd_vel to the Mega, drive/status and drive/telemetry back
  p4p_imu/                    the BNO085 on imu/data
  p4p_bringup/                launch/pi.launch.py, the ordered bring-up, and scripts/pi_bringup.sh
tools/                        fake_mega.py (the firmware on a pty), grab_full_res.py,
                              imu_throughput.py (rate, jitter, gaps and latency of imu/data),
                              drive_calibration_run.py (on the Pi: one axis and speed at a time,
                              bagged), drive_calibration_fit.py (each axis's deadband and gain)
                              and drive_wheel_fit.py (each wheel's, for the firmware's
                              DRIVE_WHEEL_CAL constants)
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

On the Pi, one launch boots the nodes in order: the camera and the IMU, then the
bridge once `imu/data` streams, then this stack once `drive/status` reports the
Mega. It uses the brief's timing (a 100 ms command watchdog, set-points at
50 Hz, the IMU at 150 Hz) and arms nothing:

```bash
/controller_ws/p4p_nodes/p4p_bringup/scripts/pi_bringup.sh             # in the Pi container
ros2 launch p4p_bringup pi.launch.py stack:=sensors                    # camera and IMU only
ros2 launch p4p_bringup pi.launch.py drive:=fake                       # no Arduino: tools/fake_mega.py stands in
```

`drive:=fake` runs everything, arming and MPC commands included, against
`tools/fake_mega.py`, so nothing moves: carry the robot by hand to watch the EKF
and the MPC. The supervisor only starts moving with the IMU at 90 % of
`imu.rate` or faster, so the IMU's `sample_rate_hz` and the YAML's `imu.rate`
should agree.
The nodes can still be started by hand:

```bash
ros2 launch p4p_serial_bridge bridge.launch.py cmd_timeout:=0.1 command_rate_hz:=50.0
ros2 launch p4p_camera camera.launch.py
ros2 launch p4p_imu imu.launch.py sample_rate_hz:=150
ros2 launch dock_control dock.launch.py rt:=true          # a bag per docking attempt into bags/, no camera
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

- **IMU.** The EKF is driven by body-frame acceleration and yaw rate at 150 Hz,
  in `base_link`, stamped at the sample time (the sensor's clock re-anchored to
  ROS time, brief Section 4). The acceleration includes gravity, and each
  message's orientation is what `ekf_node` removes it with (Gravity, below).
  `p4p_imu` provides both from the BNO085 on the Pi's I2C bus; the bridge no longer publishes
  `imu/data`, and the heading and gyro fields on `drive/telemetry` have no
  sensor behind them. Exactly one node may publish the topic the EKF reads, or
  point the EKF at another with `imu_topic:=...`. A stream without acceleration
  (`linear_acceleration_covariance[0] = -1`) is reported by `ekf_node`. The
  subscription is BEST_EFFORT, so a RELIABLE or a sensor-data publisher both
  match.
- **Limits.** The firmware saturates the wheels itself, with `config.h` values
  `WHEEL_MAX_MPS` 0.5 (a placeholder) and `CHASSIS_L_PLUS_W` 0.1655.
  - **The MPC's wheel rows** use the measured `Rw` (27.5 mm) and `lx + ly`
    (0.1655 m) from the YAML, with `w_max` (15 rad/s, 0.41 m/s at the rim, still to
    measure).
  - **The firmware's `CHASSIS_L_PLUS_W`** was 0.24, the 153 mm frame length taken as
    a half-length. It is the measured 0.1655 since 2026-10-13, as are `fake_mega.py`
    and the bridge's `max_angular` (0.5 / 0.1655 = 3.02 rad/s).
  - **Keep them in agreement,** or the firmware will rotate faster than commanded
    and scale commands the MPC thought were feasible.
- **Board frame.** The board's rotation about its own normal is normalised out:
  `fiducial.FiducialChain` uses only the board centre's position and the outward
  normal's yaw, laid onto the ground plane, as the planar marker of report eq. 5.
  So the board can be mounted at any rotation in its plane, and which lattice
  corner the pipeline numbers first does not matter. `geom.marker_m` is the board
  centre in the target frame, (0.236, 0) with the dock's coupler 35 mm out from
  the board on its centre line (0.326 until late on 2026-10-13, with it 125 mm
  out), and `geom.board_squares` and `board_square_m` place that centre in
  the pipeline's board frame. The fix's `roll` reports the board's rotation in
  its plane and `pitch` its normal's elevation; both are monitored only.

## Where the code departs from the brief

- **Two nodes, not six.** The camera/detector, IMU and wheel nodes are the
  existing p4p_camera and convchart_ros, p4p_imu and p4p_serial_bridge. The fiducial chain is
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
- **No corridor abort** (removed 2026-10-13). The supervisor used to abort once the
  estimate stayed outside the heading corridor for `hold_after_s`. On the robot that
  stopped every start more than 20 deg off the board's normal, at once: its timer
  ran from ACQUIRE, and the turn rate ramps too slowly to get back inside 0.5 s.
  Alignment with the board is now left to the MPC, whose cost weights the heading
  and whose corridor is soft.
- **Covariance abort** checks both position variances, `p_diag[0]` and `p_diag[1]`.
- **Two MPC misses** (fallback or a blown 25 ms budget) hold, as in brief
  Sections 4 and 7; Section 12 says abort. HOLD aborts after 2 s anyway.
- **Late fixes: roll back and replay.** Each buffer slot holds a step's state and
  covariance, and the IMU input and length that produced it. A fix is applied at
  its photo time, with that step's own covariance, and the steps since are replayed
  to now, so a late fix leaves exactly the state an on-time one would
  (`test_ekf.py`). A fix older than one already applied is refused.
  - **Before 2026-10-13:** the brief's `overwrite_from` shifted every later state
    by the photo-time correction, with the current covariance in the gain.
  - **Why it changed:** on the robot over WiFi, the fixes were 0.2-0.5 mm precise
    and arrived 140 ms late (median; 270 ms p90). The shift left each velocity
    correction misplacing the present by itself times the delay, and that grew
    into a 0.3 Hz oscillation of ±0.27 m/s with the robot standing still. The
    simulation reproduced it (0.4 m/s, metres of error at a 150 ms delay) and the
    replay removes it, which a regression test now checks at 150 and 300 ms.
- **Measurement noise.** R comes with every pose: the pipeline's live covariance
  (its measured noise at that range) carried through the chain's Jacobian into a
  full 3x3 for [x, y, theta]. A pose without a usable covariance is not fused.
  The brief's range model (`fid.sigma_p0`, `d0`, `sigma_th_deg`) only simulates
  fixes in the tests. `FiducialFix` logs that R, not two fixed sigmas.
- **IPPE ambiguity.** An `ambiguous` pose is not refused. With the filter
  running, `ekf_node` fuses the solution with the smaller NIS, and the gate still
  rejects it if even that one is too far. It starts the filter from one only if
  its two solutions agree on x, y and theta to within 3 sigma of its own R (a
  tilt about the horizontal leaves them equal); otherwise it waits for the next.
  Idle or zeroing, it publishes the first solution as the fix, flagged ambiguous.
- **E-stop.** No `/estop` topic: the bridge's `arm` service (false is the software
  e-stop) and `drive/status` take its place.
- **Messages.** `MpcInfo` carries the command `u`; there is no `dock/cmd_body`.
- **IMU zeroing** happens in `ekf_node`, so it works whatever publishes `imu/data`.
  It runs at every arm: the EKF estimates only while docking is armed (`dock/stage`
  out of IDLE), so arming restarts it with 2 s of zeroing (keep the robot still)
  and the next accepted pose, and disarming idles it.
- **Gravity.** `imu/data` carries acceleration including gravity, as
  `sensor_msgs/Imu` specifies and `p4p_imu` publishes it (the BNO085's calibrated
  accelerometer report). `ekf_node` removes it with each sample's own roll and
  pitch, from the message's orientation, before the zeroing and the predict.
- **IMU rate.** The MATLAB design steps a fixed 5 ms. The port steps each
  sample's own interval instead (`Ekf.advance`), split into equal steps near
  `ekf.step_s` (5 ms), never longer than 7.5 ms. That is one step a sample at
  150 Hz, and more over a dropped sample.
  - **Noise floors.** `q_v`, `q_th` and `p_floor` are the design's per-step
    floors from its NEES sweep. They are scaled by `h / step_s`, so they add the
    tuned noise per second at any rate.
  - **Equivalence.** A stream at exactly 5 ms reproduces the design's filter.
  - **Measured rate.** On the robot the IMU delivers 150–181 Hz when asked for
    150. `ekf_node` publishes the measured rate in `DockEstimate.imu_rate`.
  - **Start check.** The supervisor starts moving only with that rate at 90 % of
    `imu.rate` or more. A slow IMU is the sign of a failing node; a fast one is
    harmless.
  - **TF.** It goes out at 20 Hz of filter time.
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
