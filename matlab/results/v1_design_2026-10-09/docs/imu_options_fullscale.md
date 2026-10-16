# Inertial sensor options for the full-scale barrier unit

Research note for the P4P docking-controller thesis. Compiled 2026-10-08 from vendor datasheets, vendor documentation sites and distributor listings (all URLs in section 5). Prices are approximate, single-unit or small-quantity, as listed on the date given; fleet pricing (1000+ units) would need quotes. Anything that could not be read from a datasheet or vendor page is marked **unverified**.

## Summary

1. In-run bias instability is **not** the limiting IMU parameter for this EKF: even consumer-grade in-run numbers give under 0.1 deg of yaw drift over a 10 s fiducial outage and under 0.5 deg over 60 s. What decides whether the "bias-free IMU" model holds is the *residual* yaw-gyro bias after the sensor's own estimator (turn-on and temperature bias, 15-110 deg/h on MEMS units), which is unobservable on a steel bridge without a magnetometer unless a heading aid or a stationary zero-rate calibration exists.
2. Accelerometer bias is irrelevant next to gravity leakage from attitude error: 0.2 deg of tilt error is 3.5 mg of fake acceleration, 1.7 m of position drift after 10 s. The IMU only has to bridge the 67-100 ms between fiducial/RTK updates, where every grade in the table gives under 0.5 mm.
3. Best engineering choice: a dual-antenna RTK GNSS/INS, specifically the HBK/MicroStrain **3DM-GQ7** (1.5 deg/h, 0.25 deg heading at 1.5 m baseline, RTK 1 cm + 1 ppm built in, IP68, 1 kHz, official ROS 2, about US$4,000) or the SBG **Ellipse-D** (0.2 deg INS heading, IP68, CAN + RS-422, official ROS 2, price on quote). It adds a direct heading measurement and removes the separate RTK receiver.
4. Fleet-cost choice: industrial MEMS AHRS/INS module (Xsens MTi-7/MTi-8 class, or ADIS16470 if the robot computer does the fusion) plus two u-blox ZED-F9P receivers in moving-base mode: 0.4 deg heading at 1 m baseline, about 0.2 deg at 2 m, RTK 1 cm, roughly US$600-1,300 in parts per unit; a Septentrio mosaic-H (0.15 deg at 1 m, official ROS 2 driver, about EUR 610-700) is the better-supported variant of the same idea.
5. Minimum acceptable: an industrial AHRS with hardware sync (Xsens MTi-630 or VectorNav VN-100) plus the existing RTK receiver, magnetometer disabled, with a mandatory stationary gyro-bias calibration before each move. The BNO085 baseline fails on timestamps, gravity-leak error (1.5 deg static angle error on its linear-acceleration output) and black-box bias handling.

## 1. Requirements derivation: what the no-bias-state EKF can tolerate

**Setup.** The EKF uses body-frame gravity-compensated acceleration and yaw rate as process inputs at >= 200 Hz with no bias states. Fiducial pose arrives at 15 Hz (50 ms latency) and RTK position at 10 Hz. Targets: docking lateral tolerance +-80 mm with a 27 mm design target, yaw 1.5 deg; worst fiducial outage during final approach 10 s; navigation on RTK alone up to 60 s. Coupler lever arm 1 m. The drift formulas requested are: velocity error `v = b_a * t`, position error `p = 0.5 * b_a * t^2`, heading drift `psi = b_g * t`, lateral error at the coupler `e = L * psi` with L = 1 m.

**Table 1. Open-loop drift from in-run bias instability alone** (representative values per grade; g = 9.80665 m/s^2).

| Grade (representative in-run values) | b_g (deg/h) | b_a (ug) | 10 s: v (mm/s) | 10 s: p (mm) | 10 s: yaw (deg) | 10 s: lateral @1 m (mm) | 60 s: v (mm/s) | 60 s: p (mm) | 60 s: yaw (deg) | 60 s: lateral @1 m (mm) |
|---|---|---|---|---|---|---|---|---|---|---|
| Consumer MEMS, BNO085 class (Luxonis Allan data, units assumed deg/h and ug, unverified) | 5 | 50 | 4.9 | 25 | 0.014 | 0.24 | 29 | 883 | 0.083 | 1.5 |
| Consumer MEMS using the BNO085 fused-heading drift spec (0.5 deg/min) | 30 | 50 | 4.9 | 25 | 0.083 | 1.5 | 29 | 883 | 0.50 | 8.7 |
| Industrial MEMS (MTi-1, VN-100, Ellipse, ADIS16470 class) | 10 | 40 | 3.9 | 20 | 0.028 | 0.49 | 24 | 706 | 0.17 | 2.9 |
| Industrial MEMS with better accelerometer (MTi-600 15 ug, ADIS16470 13 ug) | 8 | 15 | 1.5 | 7.4 | 0.022 | 0.39 | 8.8 | 265 | 0.13 | 2.3 |
| Tactical-entry MEMS (3DM-CV7/GQ7, ADIS16505, INS-D class) | 1.5 | 5 | 0.49 | 2.5 | 0.004 | 0.07 | 2.9 | 88 | 0.025 | 0.44 |
| Navigation-grade FOG (reference only) | 0.1 | 5 | 0.49 | 2.5 | 0.0003 | 0.005 | 2.9 | 88 | 0.002 | 0.03 |

Reading of Table 1:

- **Yaw.** Against the 1.5 deg target, in-run drift is negligible for every grade over 10 s and still under 0.5 deg over 60 s even for the consumer row. The yaw requirement is therefore not about in-run bias instability.
- **Position.** Over 10 s the industrial-grade accelerometer bias alone gives 7-20 mm, inside the 27 mm target, but this is a false comfort, for two reasons given next.

**Attitude coupling dominates the accelerometer channel.** Gravity removal needs the robot's tilt; a tilt error `d_theta` injects `a_err = g * sin(d_theta)` into the horizontal acceleration. On a bridge with a few percent grade (1-3 deg) this is the real error source:

| Tilt error | Equivalent accel bias | p after 0.1 s | p after 1 s | p after 10 s |
|---|---|---|---|---|
| 0.05 deg (Ellipse-D / GQ7 RTK roll-pitch spec) | 0.87 mg | 0.04 mm | 4.3 mm | 0.43 m |
| 0.1 deg (Certus Mini D spec) | 1.7 mg | 0.09 mm | 8.6 mm | 0.86 m |
| 0.2 deg (MTi-600 spec) | 3.5 mg | 0.17 mm | 17 mm | 1.7 m |
| 0.5 deg (VN-100 static, MTi-1 spec) | 8.7 mg | 0.43 mm | 43 mm | 4.3 m |
| 1.5 deg (BNO085 linear-acceleration static angle error, datasheet) | 26 mg | 1.3 mm | 0.13 m | 12.8 m |

So no MEMS unit can hold 27 mm by inertial coasting for 10 s. It does not need to: during a fiducial outage the 10 Hz RTK still bounds position at the receiver's 1-2 cm, and the IMU only propagates over 0.1 s, where even the BNO085 row contributes about 1 mm. The 60 s "RTK only" case is identical for position. The accelerometer channel therefore only has to be smooth and correctly time-stamped; its bias grade hardly matters.

**Residual gyro bias is the real test of the "bias-free" assumption.** Table 1 assumes the sensor's own estimator has already removed the turn-on bias. Datasheet turn-on repeatability and over-temperature bias are far larger than in-run instability, and without a heading reference (no magnetometer on a steel bridge, and GNSS course is meaningless for a holonomic vehicle) the yaw bias is unobservable except when the vehicle is stationary:

| Residual yaw-rate bias | Source | Yaw after 10 s | Yaw after 60 s | Lateral @1 m after 60 s |
|---|---|---|---|---|
| 0.004 deg/s = 14 deg/h | 3DM-CV7 turn-on-to-turn-on bias | 0.04 deg | 0.24 deg | 4 mm |
| 0.5 deg/min = 30 deg/h | BNO085 geomagnetic-rotation-vector heading drift spec | 0.08 deg | 0.50 deg | 9 mm |
| 0.03 deg/s = 108 deg/h | 3DM-CV7 / GQ7 bias error over temperature | 0.30 deg | 1.8 deg | 31 mm |
| 0.5 deg/s = 1800 deg/h | typical uncalibrated consumer gyro zero-rate offset | 5 deg | 30 deg | 0.52 m |

The budget is 1.5 deg / 60 s = 90 deg/h of residual bias, or 30 deg/h with a factor-3 margin. A stationary zero-rate average pins the bias to roughly `ARW / sqrt(T)`: with ARW 0.3 deg/sqrt(h), 10 s of standstill gives about 6 deg/h and 60 s about 2 deg/h; with ARW 0.15 deg/sqrt(h) (CV7/GQ7) about 3 and 1 deg/h. Because a barrier unit is stationary before every move, a 10 s pre-move calibration makes the industrial-MEMS residual bias small enough for both the 10 s and 60 s cases.

**Which grades keep the bias-free model honest:**

- *Dual-antenna GNSS/INS (GQ7, Ellipse-D, Certus Mini D, INS-D, VN-300):* yaw bias is continuously estimated inside the unit against the GNSS heading, so the robot EKF can treat yaw rate (and heading) as bias-free by construction. Reasonable without conditions.
- *Industrial / tactical-entry AHRS with onboard bias tracking (MTi-600, MTi-1 v3, VN-100, 3DM-CV7):* reasonable for the 10 s docking outage, and for 60 s navigation provided the stationary calibration above is enforced and the magnetometer is switched off (it would otherwise inject steel-structure heading errors of tens of degrees). Roll/pitch bias is self-correcting through the accelerometer, which is what the gravity compensation needs.
- *Component IMUs without fusion (ADIS16470/16505):* the raw outputs carry the full turn-on and temperature bias; either the robot computer runs its own AHRS plus bias tracking, or bias states must go into the filter. Not bias-free as delivered.
- *Consumer fused (BNO085):* the fusion is a black box with magnetometer-dependent heading, no hardware timestamp, and a 1.5 deg gravity-leak angle error; it would force bias states and a timing model, which is precisely the gap the demonstrator is meant to illustrate.

## 2. Main comparison

Derived values are marked (d): ARW from gyro noise density as `ND * 60`, VRW from accelerometer noise density as `ND * g * 60`. Prices are as listed on the date given; currency conversions use approximate 2026 rates (GBP 1.30, EUR 1.10, CAD 0.72, SGD 0.76 USD) and are flagged.

**Table 2a. Sensor performance**

| # | Unit (type) | Gyro in-run bias (deg/h) | ARW (deg/sqrt(h)) | Accel in-run bias (ug) | VRW (m/s/sqrt(h)) | Max output rate | Heading method and accuracy | Roll/pitch |
|---|---|---|---|---|---|---|---|---|
| 1 | **Bosch/CEVA BNO085** (consumer 9-DoF, onboard fusion) baseline | 4.0-5.3 per axis (Luxonis Allan test; units not stated, unverified) | 0.39-0.59 (Luxonis; units not stated, unverified) | 39-59 (Luxonis; units not stated, unverified) | 0.069-0.085 (Luxonis; units not stated, unverified) | Rotation vector 400 Hz, linear accel 400 Hz, raw gyro 1 kHz, raw accel 512 Hz | Magnetometer; rotation-vector error 3.5 deg dynamic / 2.0 deg static; geomagnetic heading drift 0.5 deg/min | Game rotation vector non-heading error 2.5 deg dyn / 1.5 deg static |
| 2 | **Xsens MTi-3** (AHRS module) / **MTi-7** (GNSS/INS, external receiver) | 6 | 0.18 (d, from 0.003 deg/s/sqrt(Hz)) | 40 | 0.041 (d, from 70 ug/sqrt(Hz)) | up to 1 kHz | Mag: yaw 2 deg RMS (MTi-3); MTi-7 yaw 1.5 deg RMS using mag + GNSS course (not valid for a holonomic vehicle) | 0.5 deg RMS |
| 3 | **Xsens MTi-630** (AHRS) / **MTi-670** (GNSS/INS, external receiver) | 8 | 0.42 (d, from 0.007 deg/s/sqrt(Hz)) | 10 (x,y), 15 (z) | 0.035 (d, from 60 ug/sqrt(Hz)) | 2 kHz, 400 Hz SDI | Mag: yaw 1 deg RMS | 0.2 deg RMS |
| 4 | **VectorNav VN-100** (AHRS, SMD or Rugged) | < 10 (5 typ) | 0.21 (d, from 0.0035 deg/s/sqrt(Hz)) | < 40 | 0.082 (d, from 0.14 mg/sqrt(Hz)) | IMU 800 Hz, attitude 400 Hz | Mag: 2.0 deg RMS | 0.5 deg static / 1.0 deg dynamic RMS |
| 5 | **HBK/MicroStrain 3DM-CV7-AHRS** (tactical-entry OEM) | 1.5 | 0.14 | 18 | 0.018 (d, from 30 ug/sqrt(Hz)) | 1-1000 Hz | Mag; turn-on bias 0.004 deg/s, over-temperature 0.03 deg/s | 0.25 deg static / 0.5 deg dynamic |
| 6 | **ADI ADIS16470** (component IMU, no fusion) | 8 | 0.34 | 13 | 0.037 | 2 kSPS | None (no heading) | None |
| 7 | **ADI ADIS16505-1** (component IMU, no fusion) | 2.3 | 0.13 | 26.5 um/s^2 (= 2.7 ug) | unverified | 2 kSPS (unverified) | None | None |
| 8 | **VectorNav VN-300** (dual-antenna GNSS-compass INS) | < 10 (5 typ) | 0.21 (d) | < 40 | 0.082 (d) | IMU 400 Hz, PVA 400 Hz | GNSS-compass (moving-baseline RTK): 0.3-0.6 deg @0.5 m, 0.15-0.3 deg @1 m, 0.08-0.15 deg @2 m RMS; INS heading 0.2 deg | INS 0.03 deg, static 0.5 deg |
| 9 | **SBG Ellipse-D** (dual-antenna RTK GNSS/INS) | 7 | 0.18 | 14 | 0.034 (d, from 57 ug/sqrt(Hz)) | 200 Hz nav, 1 kHz IMU | Dual antenna RTK: INS heading 0.2 deg; receiver-only 0.55 deg (>0.5 m), 0.4 deg (>1 m), 0.3 deg (>2 m) | 0.05 deg (RTK), 0.1 deg single point |
| 10 | **Advanced Navigation Certus Mini D** (dual-antenna RTK GNSS/INS; the 2013 Spatial Dual is superseded by the Certus line) | unverified (initial bias < 0.2 deg/s; full-size Certus 3 deg/h) | 0.24 | < 20 | 0.0588 | 500 Hz (docs; press material says 1000 Hz) | Dual antenna: 0.1 deg @1 m, 0.07 deg @2 m | 0.1 deg |
| 11 | **Xsens MTi-680G** (single-antenna RTK GNSS/INS) | 8 | 0.42 (d) | 10 (x,y), 15 (z) | 0.035 (d) | 2 kHz, 400 Hz SDI | Mag + GNSS velocity: 0.5 deg RMS; no dual-antenna input documented (MTi-680 external-receiver NMEA inputs are GGA/RMC/GSA/GST; HDT heading input unverified) | 0.2 deg RMS |
| 12 | **HBK/MicroStrain 3DM-GQ7** (dual-antenna RTK GNSS/INS) | 1.5 | 0.15 | 5 | 0.012 (d, from 20 ug/sqrt(Hz)) | 1-1000 Hz | Dual antenna: 0.25 deg RMS @1.5 m baseline; turn-on bias 0.002 deg/s | 0.05 deg |
| 13 | **Inertial Labs INS-D** (dual-antenna RTK GNSS/INS) | 1 | < 0.2 | 5 | 0.015 | 200 Hz | Dual antenna: 0.08 deg RMS @2 m baseline | 0.05 deg static |
| 14 | **Cost-engineered: 2x u-blox ZED-F9P moving base (or F9P + F9H) + ADIS16470 or MTi-7/MTi-8** | per IMU chosen (rows 2 or 6) | per IMU | per IMU | per IMU | GNSS 7-20 Hz RTK nav rate depending on constellation set; IMU as above | Moving-base RTK: 0.4 deg (50 %) @1 m baseline per datasheet; about 0.8 deg @0.5 m and 0.2 deg @2 m by 1/L scaling (d; u-blox Figure 1 curve, not quoted numerically) | from IMU |
| 14b | **Variant: Septentrio mosaic-H + MTi-3/ADIS16470** | per IMU | per IMU | per IMU | per IMU | GNSS 100 Hz, latency < 10 ms | Dual antenna: 0.15 deg @1 m, 0.03 deg @5 m; pitch 0.25 deg @1 m | from IMU |

**Table 2b. Integration, environment and cost**

| # | Unit | RTK | IP rating | Operating temp (C) | Interfaces | Timestamp / sync | ROS 2 driver | Approx. price (USD unless stated; date; source) |
|---|---|---|---|---|---|---|---|---|
| 1 | BNO085 | None | None (IC / breakout) | -40 to +85 | I2C, SPI, UART-RVC | Internal relative timestamps only; no PPS, no hardware sync; I2C clock stretching is a known Raspberry Pi issue | Community: `bno08x_driver` (ROS Index, Humble); `bnbhat/bno08x_ros2_driver` | IC $21.55 (LCSC, 2026); Adafruit breakout $29.50 (2026-10) |
| 2 | MTi-3 / MTi-7 | MTi-7: external receiver (NMEA/UBX/SBF), position 1 m CEP; MTi-8 is the RTK variant | IP00 (bare module) | -40 to +85 | UART, SPI, I2C; MTi-7 adds GNSS UART | SyncIn; 10 ppm clock, 1 ppm when disciplined by external GNSS (MTi-7) | Official Movella ROS 2 node in the MT SDK; `xsenssupport/Xsens_MTi_ROS_Driver_and_Ntrip_Client` ros2 branch; `DEMCON/ros2_xsens_mti_driver` | MTi-3 GBP 437 ex VAT (about $570); MTi-7 GBP 592 ex VAT (about $770) (RS, 2025-26, converted); MTi-8 unverified |
| 3 | MTi-630 / MTi-670 | MTi-670: external receiver | IP51 | -40 to +85 | UART, CAN, RS-232 | SyncIn/SyncOut, ClockSync, 10 ppm or external | Official (as row 2) | MTi-630 about $1,619 (Cytech reference listing, 2025, unverified); MTi-670 about $1,967-2,318 (distributor listings, unverified) |
| 4 | VN-100 | None | None stated (Rugged = aluminium case) | -40 to +85 | Rugged: RS-232, TTL; SMD: TTL, SPI | Sync in/out; 0.001 deg angular resolution | Community: `dawonn/vectornav` ros2 branch (no official ROS 2 driver found) | SMD $600 / Rugged $950 standard calibration; +$350 for thermal calibration (VectorNav purchasing guide via Metromatics, 2025) |
| 5 | 3DM-CV7-AHRS | None (CV7-INS variant accepts external GNSS) | Not IP rated (OEM); the 3DM-GV7-AHRS is the IP68 version | -40 to +85 | UART (TTL), USB, 4x GPIO | PPS in/out, event triggering | Official `LORD-MicroStrain/microstrain_inertial` (ROS build farm, Humble and later) | About $700-750 converted from Mouser CA C$1,035 / SG S$928 / IN Rs 59,988 listings (2025-26); US list price unverified |
| 6 | ADIS16470 | None | None (BGA module) | -25 to +85 as indexed (datasheet range unverified) | SPI, data-ready line | Host timestamps on data-ready; no PPS | Official `analogdevicesinc/imu-ros2` (Humble, Linux IIO over SPI; documented on Pi 3/4, Pi 5 unverified) | $253 (LCSC) to $408 (Richardson RFPD, qty 1), 2025-26 |
| 7 | ADIS16505-1 | None | None | -40 to +105 | SPI | Host timestamps on data-ready | Official `imu-ros2` (listed device) | $687.63 (ADI 1 ku list price, 2026) |
| 8 | VN-300 | **No RTK positioning** (SBAS, 1.0 m RMS); moving-baseline RTK used only for heading | None stated | -40 to +85 | Rugged: RS-232, TTL; SMD: TTL, SPI | PPS 30 ns RMS, sync in/out | Community: `dawonn/vectornav` ros2 branch | SMD $3,400; Rugged $4,850 (VectorNav store, 2026-10) |
| 9 | Ellipse-D | Built-in L1/L2 quad-constellation RTK, 1 cm + 1 ppm; RTCM in | IP68 | -40 to +85 | CAN 2.0 A/B, RS-232/422 (up to 3 ports); Ethernet unverified | Timestamp < 200 ns, PPS < 1 us | Official `SBG-Systems/sbg_ros2_driver` (Humble; RTCM topic input) | Quote only (resellers list "on request"); **unverified** |
| 10 | Certus Mini D | Built-in L1/L5 RTK, 10 mm H / 15 mm V | IP67 (rugged); OEM board unrated | -40 to +85 | RS-232/422, aux RS-232, CAN, 2x GPIO (OEM: UART) | GPIO/PPS (details unverified) | Official `advanced-navigation/ros2-driver` (ANPP; Humble) | **Unverified** (not published) |
| 11 | MTi-680G | Built-in u-blox ZED-F9, 1 cm + 1 ppm; RTCM 3.2/3.3 in via RS-232 | IP68 (MTi-680 external-receiver version IP51) | -40 to +85 | CAN, RS-232 | SyncIn/SyncOut, ClockSync, 1 ppm | Official (as row 2) | $3,016 (DigiKey, 2025-26); GBP 2,729 ex VAT (RS) |
| 12 | 3DM-GQ7 | Built-in dual multiband receivers, 1 cm + 1 ppm (2 cm at 10 km); RTCM 3.1 in or 3DM-RTK modem | IP68 | -40 to +85 | 2x RS-232, 2x USB, 4x GPIO (no CAN listed) | PPS, event GPIO | Official `microstrain_inertial` | $4,023 (Mouser, 2025-26) |
| 13 | INS-D | Built-in, 1 cm + 1 ppm; RTCM 2.x/3.x | IP67 (enclosed unit; OEM board unrated) | -40 to +75 | RS-232 or RS-422, CAN, Ethernet optional | PPS (details unverified) | Official `inertiallabs/inertiallabs-ros2-pkgs` | $3,750 (Unmanned Systems Source listing, date not stated; unverified) |
| 14 | 2x ZED-F9P (+ IMU) | Yes, 1 cm + 1 ppm, < 10 s convergence; RTCM 3 in | Module unrated; needs IP67 enclosure | -40 to +85 | UART, SPI, I2C, USB | PPS out (disciplines MTi-7/8 or host) | Community: `aussierobots/ublox_dgnss` (Humble-Rolling; moving base via NAV-RELPOSNED, NTRIP client) | Module $116-119 (Future, -02B) to $248 (LCSC, -04B); SparkFun board $259.95 (2026-10); pair plus two antennas about $400-700; plus IMU ($250-800) |
| 14b | mosaic-H (+ IMU) | Yes, 0.6 cm + 0.5 ppm; RTCM 2/3 in | Module unrated | -40 to +85 | 4x UART, Ethernet, USB | PPS, event input | Official `septentrio-gnss/septentrio_gnss_driver` (Foxy-Rolling; publishes AttEuler heading; NTRIP) | Boards EUR 610-700 (about $670-770) (gnss.store, 2026-10) |

Notes on the table: the VN-100/VN-300 and MTi-1 ARW/VRW figures are derived from noise density, so they are not vendor Allan-variance numbers. The 3DM-CV7 IMU figures are taken from the CV7-series datasheet (shared IMU across -AR/-AHRS/-INS). All units except the INS-D cover the -5 to +45 C ambient with margin; the INS-D's +75 C upper limit is still adequate for an IP67 box in sun only if the enclosure temperature is checked.

## 3. Shortlist and recommendation

### (a) Best engineering choice for the theoretical full-scale design: dual-antenna RTK GNSS/INS

**Primary: HBK/MicroStrain 3DM-GQ7.** Verified tactical-entry IMU (1.5 deg/h, 0.15 deg/sqrt(h), 5 ug, 20 ug/sqrt(Hz)), dual multiband receivers with RTK built in, 0.25 deg RMS heading at a 1.5 m baseline (a lane-barrier unit can carry a 2 m baseline, which should push this toward 0.2 deg), 0.05 deg roll/pitch for gravity compensation, IP68, 1 kHz output, PPS/GPIO timing, an official ROS 2 driver on the build farm, and a published price of about US$4,000. Its one integration weakness is interfaces: RS-232/USB only, no CAN.

**Equal alternative: SBG Ellipse-D.** The raw IMU is one class lower (7 deg/h, 14 ug), but the INS heading is 0.2 deg with RTK, the roll/pitch 0.05 deg, it is IP68, offers CAN plus RS-422 (better suited to a 600 kg vehicle harness), timestamps to 200 ns and ships an official ROS 2 driver with RTCM subscription. Price is quote-only, so it cannot be ranked on cost here. The Advanced Navigation Certus Mini D has the best heading figure (0.1 deg at 1 m, 0.07 deg at 2 m) and IP67 in 55 g, but its gyro in-run bias and price could not be verified. The Inertial Labs INS-D matches the GQ7 on the IMU and beats it on heading (0.08 deg at 2 m) at a listed US$3,750, but the listing date is unknown and its +75 C ceiling is the lowest. The VN-300 is excluded from this tier because it has no RTK positioning: it would give heading but not replace the RTK receiver.

### (b) Fleet-cost-engineered choice that still meets the requirements

**Industrial MEMS module + two u-blox ZED-F9P in moving-base mode.** The F9P pair gives RTK position (1 cm + 1 ppm) and a 0.4 deg (50 %) heading at 1 m baseline, about 0.2 deg at 2 m, both well inside the 1.5 deg yaw target (0.4 deg is 7 mm at the 1 m coupler arm). The u-blox datasheet quotes the 1 m figure; the 2 m figure is the 1/L scaling of the same relative-position error, consistent with the VN-300's published 0.5/1/2 m table, and should be confirmed on the vehicle. For the IMU, the Xsens MTi-7 (or the RTK-capable MTi-8) is the natural partner: 6 deg/h, 40 ug, an onboard AHRS with bias tracking, a direct UART input for the u-blox receiver and PPS clock discipline, so the module delivers the gravity-compensated, hardware-time-stamped signals the EKF wants. If the fusion is to run on the Pi instead, the ADIS16470 (8 deg/h, 13 ug, 2 kSPS, official IIO-based ROS 2 driver) is the cheapest honest industrial IMU, at the cost of implementing attitude and bias tracking in software (which breaks the "bias-free input" assumption unless the stationary calibration is enforced). Parts cost per unit is roughly US$600-1,300 (two F9P boards, two multiband antennas, IMU), versus US$3,000-4,000 for an integrated unit, before enclosure and harness. The hidden costs are an IP67 enclosure, cabling, antenna mounts, time synchronisation and community-maintained drivers.

**Better-supported variant:** replace the F9P pair with one Septentrio mosaic-H (EUR 610-700 as a board): 0.15 deg heading at 1 m, 100 Hz attitude with under 10 ms latency, strong multipath/interference mitigation (relevant on a steel truss), and an official ROS 2 driver that publishes dual-antenna attitude. On a 1000-unit fleet this is roughly the same money as two F9P boards with much less integration risk.

### (c) Minimum acceptable

**Xsens MTi-630 (IP51, CAN/RS-232, 8 deg/h, 0.2 deg roll/pitch, SyncIn/ClockSync, official ROS 2 driver, about US$1,600) or VectorNav VN-100 Rugged (US$950)** with the existing separate RTK receiver, magnetometer disabled, and a mandatory stationary gyro-bias calibration (>= 10 s) before each move. This meets the docking case: heading drift over a 10 s fiducial outage stays under 0.1 deg even with worst-case turn-on bias, and roll/pitch for gravity compensation is self-correcting. It does **not** cover the 60 s navigation case on its own: with no heading aid, yaw drift depends entirely on how well the pre-move calibration held (0.24 deg at 14 deg/h, 1.8 deg at 108 deg/h), so the operating concept would need stops every 60 s or a later heading aid. The BNO085 remains below the minimum: no hardware timestamp or PPS, I2C timing problems on the Pi, magnetometer-driven heading, and a 1.5 deg static angle error on its linear-acceleration output (26 mg of gravity leak).

### How a dual-antenna GNSS/INS changes the estimator

A single-antenna RTK receiver observes position only; on a holonomic robot the yaw state is observable solely through the fiducial, so between fiducial fixes the yaw covariance grows with the gyro noise and whatever residual bias the no-bias-state filter cannot represent. A dual-antenna unit adds a direct measurement `z_psi = psi + v`, with sigma 0.1-0.4 deg depending on unit and baseline, at 5-20 Hz from the receiver or at 200-1000 Hz as a fused INS output. Consequences: (1) yaw is observable everywhere, so the 60 s navigation case stops being a drift problem; (2) the unit's internal EKF estimates and removes gyro biases against that heading, so the "bias-free yaw rate" assumption becomes valid by construction rather than by procedure; (3) the fiducial heading becomes a second measurement of the same state and must be given consistent covariance and lever-arm/mounting-misalignment treatment (the VN-300 datasheet quotes 0.15 deg heading mounting misalignment for the rugged case; this is a calibration term, not noise); (4) the unit absorbs the RTK receiver, antenna lever arms and time synchronisation (PPS-disciplined timestamps, 200 ns on the Ellipse-D), which removes the 50 ms-class latency bookkeeping from the robot side for everything except the camera; (5) the multipath environment of a steel bridge becomes a shared risk for position and heading, which argues for receivers with explicit multipath mitigation and for keeping the fiducial as the authoritative sensor in the final metre. The baseline should be as long as the unit allows (2 m halves the heading error relative to 1 m) and the antennas mounted with clear sky and identical ground planes.

## 4. Caveats on the evidence

- Allan-variance numbers for the BNO085 come from a third-party (Luxonis) characterisation table that does not state units; the CEVA datasheet gives only system-level error figures. Treat the BNO085 row as indicative.
- Several PDFs (ADI, Mouser-hosted MicroStrain) were not reachable from this network; ADIS16470 values are from ADI parametric data and secondary mirrors, ADIS16505 VRW and ODR are unverified.
- The Certus Mini D gyro in-run bias instability is not published on the pages reached; the VN-100/VN-300 and MTi-1 ARW/VRW are derived from noise density.
- Prices are single-unit list or distributor prices; the three currency conversions are approximate. No vendor publishes 1000-unit pricing.

## 5. Sources

Baseline and consumer:
- CEVA BNO080/BNO085 datasheet (performance table, report rates, temperature): https://www.ceva-ip.com/wp-content/uploads/BNO080_085-Datasheet.pdf
- Luxonis BNO08x characterised noise parameters: https://docs.luxonis.com/hardware/platform/sensors/imu/bno08x.md
- Bosch Sensortec forum on BMI085-class gyro bias: https://community.bosch-sensortec.com/t5/MEMS-sensors-forum/THE-bias-instability-of-BNO055/td-p/94733
- Adafruit BNO085 breakout price: https://www.adafruit.com/product/4754
- LCSC BNO085 price: https://fat.lcsc.com/product-image/C5189642.html
- ROS 2 drivers: https://index.ros.org/p/bno08x_driver/ and https://github.com/bnbhat/bno08x_ros2_driver

Xsens / Movella:
- MTi-3 leaflet (Feb 2025): https://www.xsens.com/hubfs/MTi%20product%20leaflet_07022025_MTi-3.pdf
- MTi-7 leaflet (Feb 2025): https://www.xsens.com/hubfs/MTi%20product%20leaflet_07022025_MTi-7.pdf
- MTi-630 leaflet (Feb 2025): https://www.xsens.com/hubfs/MTi%20product%20leaflet_07022025_MTi-630.pdf
- MTi-680 leaflet: https://www.xsens.com/hubfs/Downloads/Leaflets/MTi-680.pdf
- MTi-680G leaflet (Feb 2025): https://www.xsens.com/hubfs/MTi%20product%20leaflet_07022025_MTi-680G.pdf
- Movella application note on external GNSS inputs (NMEA message set): https://www.movella.com/hubfs/Automation%20and%20Mobility%20-%20Application%20Notes/Supercharge%20your%20GNSS-INS%20Receiver/Xsens%20GNSS-INS%20-%20Supercharge%20Your%20GNSS%20Receiver%20-%20MTAN001-A.pdf
- Movella sensor specification pages: https://mtidocs.xsens.com/sensor-specifications and https://mtidocs.xsens.com/sensor-specifications-2
- RS Components Xsens listings (MTi-3, MTi-7 GBP prices): https://uk.rs-online.com/web/b/xsens-by-movella/?pn=3
- MTi-630 / MTi-670 distributor reference prices: https://www.cytechsystems.com/de/product/mti-630-dk and https://www.cytechsystems.com/de/product/mti-670-dk
- MTi-680G prices: https://punchouttest.digikey.com.mx/en/products/detail/xsens-a-movella-brand/MTI-680G/12702435 and https://uk.rs-online.com/web/p/multi-function-sensors/2614490
- ROS 2 drivers: https://github.com/xsenssupport/Xsens_MTi_ROS_Driver_and_Ntrip_Client and https://github.com/DEMCON/ros2_xsens_mti_driver

VectorNav:
- VN-100 product page: https://www.vectornav.com/products/detail/vn-100
- VN-100 product brief: https://vectornav.com/docs/default-source/product-brief/vn-100-product-brief.pdf
- VN-100 purchasing guide (prices): https://metromatics.com.au/wp-content/uploads/2025/12/VN-100-Purchasing-Guide.pdf
- VN-300 product page: https://www.vectornav.com/products/detail/vn-300
- VN-300 datasheet (baseline table): https://www.navtechgps.com/wp-content/uploads/VN-300_DS.pdf
- VN-300 store prices: https://vectornav.com/store/products/dual-gnss-ins/p/vn-300-smd-dual-gnssins and https://www.vectornav.com/store/products/dual-gnss-ins/p/vn-300-rugged-dual-gnssins
- Community ROS 2 driver: https://github.com/dawonn/vectornav

HBK / MicroStrain:
- 3DM-CV7-AHRS product page: https://www.hbkworld.com/en/products/transducers/inertial-sensors/attitude-and-heading/3dm-cv7-ahrs
- 3DM-CV7 series datasheet (Metromatics mirror): https://metromatics.com.au/wp-content/uploads/2024/04/3DM-CV7-AR.pdf
- 3DM-CV7 distributor listings used for price conversion: https://www.mouser.ca/c/semiconductors/sensor-ics/imus-inertial-measurement-units and https://www.mouser.sg/c/semiconductors/sensor-ics/imus-inertial-measurement-units and https://www.mouser.in/new/microstrain/microstrain-3dm-cv7-imus/
- 3DM-GQ7 datasheet (Metromatics mirror): https://metromatics.com.au/wp-content/uploads/2024/04/3DM-GQ7-GNSS-INS.pdf
- 3DM-GQ7 product page: https://www.hbkworld.com/en/products/transducers/inertial-sensors/navigation/3dm-gq7-gnss-ins
- 3DM-GQ7 documentation (IP68): https://files.microstrain.com/GQ7+User+Manual/Home.htm
- 3DM-GQ7 Mouser price: https://www.mouser.com/en/ProductDetail/MicroStrain-by-HBK/3DMGQ7-GNSS-INS?qs=DPoM0jnrROUOf6M42mFhSg%3D%3D
- Official ROS 2 driver: https://github.com/LORD-MicroStrain/microstrain_inertial

Analog Devices:
- ADIS16470 product page and datasheet: https://www.analog.com/en/products/adis16470.html and https://www.analog.com/media/en/technical-documentation/data-sheets/adis16470.pdf
- ADIS16470 parametric mirror: https://www.radiolocman.com/datasheet/data.html?di=449139
- ADIS16470 prices: https://shop.richardsonrfpd.com/Products/Product/ADIS16470AMLZ and https://www.lcsc.com/product-image/C459878.html
- ADIS16505 product page and datasheet: https://www.analog.com/en/products/adis16505.html and https://www.analog.com/media/en/technical-documentation/data-sheets/adis16505.pdf
- Official ROS 2 driver: https://github.com/analogdevicesinc/imu-ros2

SBG Systems:
- Ellipse-D product page: https://www.sbg-systems.com/ins/ellipse-d/
- Ellipse raw IMU specifications: https://support.sbg-systems.com/sc/el/latest/ellipse-documentation/performance-specifications/raw-imu-specifications
- Ellipse performance specifications (heading vs baseline): https://support.sbg-systems.com/sc/el/latest/ellipse-documentation/performance-specifications
- Reseller quote-only listing: https://skysat.fr/en-us/products/sbg-ellipse-d-inertial-navigation-system-gnss-kit
- Official ROS 2 driver: https://github.com/SBG-Systems/sbg_ros2_driver

Advanced Navigation:
- Certus Mini D product page: https://www.advancednavigation.com/inertial-navigation-systems/mems-gnss-ins/certus-mini-d/
- Certus Mini navigation specifications: https://docs.advancednavigation.com/certus-mini-oem/NavigationSpecifications.htm
- Certus Mini sensor specifications: https://docs.advancednavigation.com/certus-mini-oem/SensorSpecifications.htm
- Certus (full size) page: https://www.advancednavigation.com/inertial-navigation-systems/mems-gnss-ins/certus/
- Spatial Dual 2013 launch (superseded product): https://www.gpsworld.com/advanced-navigation-releases-dual-antenna-gnssins/
- Spatial FOG Dual NRND notice: https://www.advancednavigation.com/product/spatial-fog-dual
- Official ROS 2 driver: https://github.com/advanced-navigation/ros2-driver

Inertial Labs:
- INS-D / INS-DL OEM datasheet: https://cdn.bestech.com.au/wp-content/uploads/2020/03/INS-D-OEM-INS-DL-OEM_Dual-antenna-gps-aided-inertial-navigation-system-ID-3846.pdf and https://inertiallabs.com/ins-d-oem-datasheet
- INS-D IP67 listing: https://www.everythingrf.com/products/inertial-navigation-systems/inertial-labs/1092-2404-ins-d
- INS-D price listing: https://www.unmannedsystemssource.com/shop/ins-sensors/dual-gps-ins/gps-aided-ins-d-ins-dl-dual-antenna/
- Official ROS 2 driver: https://github.com/inertiallabs/inertiallabs-ros2-pkgs

u-blox and Septentrio:
- ZED-F9P-04B datasheet (moving-base heading 0.4 deg at 1 m, RTK rates): https://content.u-blox.com/sites/default/files/ZED-F9P-04B_DataSheet_UBX-21044850.pdf
- ZED-F9H-01B datasheet: https://content.u-blox.com/sites/default/files/ZED-F9H-01B_DataSheet_UBX-21025012.pdf
- ZED-F9P moving-base application note: https://content.u-blox.com/sites/default/files/documents/ZED-F9P-MovingBase_AppNote_UBX-19009093.pdf
- ZED-F9P prices: https://www.futureelectronics.com/p/semiconductors--wireless-rf--rf-modules-solutions--gps/zed-f9p-02b-u-blox-3131722 and https://www.sparkfun.com/products/15136
- ArduSimple heading kit (contents; product retired): https://www.ardusimple.com/product/simplertk2b-heading-basic-starter-kit-ip67/
- ROS 2 driver with moving base: https://github.com/aussierobots/ublox_dgnss
- Septentrio mosaic-H datasheet (mirror): https://xcopter.com/download/mosaic-H.pdf
- Septentrio mosaic-H product page: https://www.septentrio.com/en/products/gnss-receivers/gnss-receiver-modules/mosaic-h
- mosaic-H board prices: https://gnss.store/collections/mosaic-h
- Official ROS 2 driver: https://github.com/septentrio-gnss/septentrio_gnss_driver
