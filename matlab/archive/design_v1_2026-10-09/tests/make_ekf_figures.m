%MAKE_EKF_FIGURES Report figures for the docking EKF.
%   Runs the staged approach of make_truth_trajectory with a 2 s fiducial dropout,
%   compares delay compensation on/off, plots a 20-run NEES, and shows the response
%   to a target that starts moving. Figures go to matlab/out (or matlab/out/reference
%   when the reference implementation is on the path).
%
%   Usage:  >> setup_paths; make_ekf_figures
%   Select the variant with the environment variable DOCK_VARIANT ('fullscale' | 'demo').

close all;
variant = getenv('DOCK_VARIANT');
if isempty(variant), variant = 'fullscale'; end
prm = dock_params(variant);
root = fullfile(fileparts(mfilename('fullpath')), '..');
is_ref = contains(which('ekf_predict'), [filesep, 'reference', filesep]);
if is_ref
    outdir = fullfile(root, 'out', 'reference');  impl = 'reference';
else
    outdir = fullfile(root, 'out');               impl = 'user';
end
if ~exist(outdir, 'dir'), mkdir(outdir); end
fig_file = @(name) fullfile(outdir, sprintf('ekf_%s_%s.png', variant, name));
fprintf('=== Docking EKF figures, variant: %s (%s implementation) ===\n', variant, impl);

%% Truth trajectory and sensors
tr = make_truth_trajectory(prm.imu.T);
dropout = [18, 20];                     % fiducial lost during the final approach [s]
t_standoff = 15.5;                      % end of the approach segment + hold [s]
sens = simulate_sensors(tr, prm, 1, struct('dropout', dropout));
t = tr.t;

%% EKF runs
res_on  = run_ekf(sens, prm, tr, struct('delay_comp', true));
res_off = run_ekf(sens, prm, tr, struct('delay_comp', false));

%% Figure 1: plane view
f1 = figure('Visible', 'off', 'Position', [100 100 800 500]);
plot(tr.pose(:,1), tr.pose(:,2), 'k-', 'LineWidth', 1.2); hold on; grid on; axis equal;
plot(res_on.x(:,1), res_on.x(:,2), 'b--', 'LineWidth', 1.0);
v = sens.fid.valid;
plot(sens.fid.z(v,1), sens.fid.z(v,2), 'r.', 'MarkerSize', 5);
plot(0, 0, 'ks', 'MarkerFaceColor', 'k');
xlabel('x_T [m]'); ylabel('y_T [m]');
legend('truth', 'EKF', 'fiducial fixes', 'docked pose', 'Location', 'best');
title(sprintf('Docking approach in the target frame (%s)', variant));
exportgraphics(f1, fig_file('plane'), 'Resolution', 110);

%% Figure 2: estimation error with 3-sigma bounds
names = {'x [mm]', 'y [mm]', '\theta [deg]', 'v_x [mm/s]', 'v_y [mm/s]'};
scale = [1e3, 1e3, 180/pi, 1e3, 1e3];
f2 = figure('Visible', 'off', 'Position', [100 100 900 950]);
tl = tiledlayout(5, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
for i = 1:5
    nexttile; hold on; grid on;
    xregion(dropout(1), dropout(2), 'FaceColor', [0.9 0.9 0.9]);
    sig3 = 3*sqrt(res_on.Pdiag(:,i))*scale(i);
    plot(t, sig3, 'r:', t, -sig3, 'r:', 'LineWidth', 0.8);
    plot(t, res_on.err(:,i)*scale(i), 'b-', 'LineWidth', 0.9);
    if i == 2, yline( prm.tgt.y*1e3, 'g--'); yline(-prm.tgt.y*1e3, 'g--'); end
    if i == 3, yline( rad2deg(prm.tgt.th), 'g--'); yline(-rad2deg(prm.tgt.th), 'g--'); end
    ylabel(names{i});
    if i == 1
        title(sprintf('EKF error and 3\\sigma bounds (%s). Grey: fiducial dropout. Green: internal target', variant));
    end
end
xlabel(tl, 'time [s]');
exportgraphics(f2, fig_file('errors'), 'Resolution', 110);

%% Monte Carlo consistency check (averaged NEES)
Nmc = 20;
nees_sum = zeros(numel(t), 1);
for i = 1:Nmc
    s_i = simulate_sensors(tr, prm, 100 + i, struct('dropout', dropout));
    r_i = run_ekf(s_i, prm, tr, struct('delay_comp', true, 'seed', 200 + i));
    nees_sum = nees_sum + r_i.nees;
end
nees_avg = nees_sum / Nmc;
[lo1, hi1] = chi2_bounds(5, 0.05);
[loN, hiN] = chi2_bounds(5*Nmc, 0.05);  loN = loN/Nmc;  hiN = hiN/Nmc;

%% Figure 3: NEES
f3 = figure('Visible', 'off', 'Position', [100 100 900 550]);
tl = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; hold on; grid on;
xregion(dropout(1), dropout(2), 'FaceColor', [0.9 0.9 0.9]);
plot(t, res_on.nees, 'b-'); yline(lo1, 'r--'); yline(hi1, 'r--');
set(gca, 'YScale', 'log');
ylabel('NEES, single run');
title(sprintf('Filter consistency (%s): 95%% bounds, 5 dof', variant));
nexttile; hold on; grid on;
xregion(dropout(1), dropout(2), 'FaceColor', [0.9 0.9 0.9]);
plot(t, nees_avg, 'b-'); yline(loN, 'r--'); yline(hiN, 'r--'); yline(5, 'k:');
ylabel(sprintf('NEES, mean of %d runs', Nmc)); xlabel('time [s]');
exportgraphics(f3, fig_file('nees'), 'Resolution', 110);

%% Figure 4: delay compensation on versus off
epos_on  = hypot(res_on.err(:,1),  res_on.err(:,2))*1e3;
epos_off = hypot(res_off.err(:,1), res_off.err(:,2))*1e3;
speed = hypot(tr.vel(:,1), tr.vel(:,2));
f4 = figure('Visible', 'off', 'Position', [100 100 900 550]);
tl = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; hold on; grid on;
xregion(dropout(1), dropout(2), 'FaceColor', [0.9 0.9 0.9]);
plot(t, epos_off, 'Color', [0.85 0.33 0.10]);
plot(t, epos_on, 'b-');
plot(t, speed*prm.fid.delay*1e3, 'k:', 'LineWidth', 1.2);
ylabel('|position error| [mm]');
legend('no delay compensation', 'delay compensation', 'speed \times \tau_d', 'Location', 'northeast');
title(sprintf('Effect of the %d ms fiducial delay (%s)', round(prm.fid.delay*1e3), variant));
nexttile; hold on; grid on;
plot(t, speed, 'k-'); ylabel('true speed [m/s]'); xlabel('time [s]');
exportgraphics(f4, fig_file('delay'), 'Resolution', 110);

%% Figure 5: marker dropout, coasting on the IMU alone (zoom of figure 2)
f5 = figure('Visible', 'off', 'Position', [100 100 900 550]);
tl = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; hold on; grid on;
xregion(dropout(1), dropout(2), 'FaceColor', [0.9 0.9 0.9]);
plot(t, 3*sqrt(res_on.Pdiag(:,1) + res_on.Pdiag(:,2))*1e3, 'r:', 'LineWidth', 1.0);
plot(t, epos_on, 'b-');
ylabel('|position error| [mm]'); xlim([dropout(1)-2, dropout(2)+3]);
legend('3\sigma bound', 'estimate', 'Location', 'northwest');
title(sprintf('Marker lost for %g s: coasting on the IMU (%s)', diff(dropout), variant));
nexttile; hold on; grid on;
xregion(dropout(1), dropout(2), 'FaceColor', [0.9 0.9 0.9]);
plot(t, 3*sqrt(res_on.Pdiag(:,4))*1e3, 'r:', t, -3*sqrt(res_on.Pdiag(:,4))*1e3, 'r:');
plot(t, res_on.err(:,4)*1e3, 'b-');
ylabel('v_x error [mm/s]'); xlabel('time [s]'); xlim([dropout(1)-2, dropout(2)+3]);
exportgraphics(f5, fig_file('dropout'), 'Resolution', 110);

%% Figure 6: target starts moving at 0.2 m/s during the approach (robot IMU cannot see it)
t_step = 6;  dv = [-0.2; 0.06];
tr_m = apply_target_velocity_step(tr, t_step, dv, 0.5);
s_m  = simulate_sensors(tr_m, prm, 1, struct('dropout', dropout));
r_m  = run_ekf(s_m, prm, tr_m, struct('delay_comp', true));
f6 = figure('Visible', 'off', 'Position', [100 100 900 550]);
tl = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; hold on; grid on;
xregion(t_step, t_step + 0.5, 'FaceColor', [0.9 0.9 0.9]);
plot(t, 3*sqrt(r_m.Pdiag(:,1))*1e3, 'r:', t, -3*sqrt(r_m.Pdiag(:,1))*1e3, 'r:');
plot(t, r_m.err(:,1)*1e3, 'b-');
ylabel('x error [mm]'); xlim([t_step - 1, t_step + 6]);
title(sprintf('Target starts moving at %.2f m/s at t = %g s (%s): recovery from fiducial fixes alone', norm(dv), t_step, variant));
nexttile; hold on; grid on;
xregion(t_step, t_step + 0.5, 'FaceColor', [0.9 0.9 0.9]);
plot(t, 3*sqrt(r_m.Pdiag(:,4))*1e3, 'r:', t, -3*sqrt(r_m.Pdiag(:,4))*1e3, 'r:');
plot(t, r_m.err(:,4)*1e3, 'b-');
ylabel('v_x error [mm/s]'); xlabel('time [s]'); xlim([t_step - 1, t_step + 6]);
exportgraphics(f6, fig_file('target_motion'), 'Resolution', 110);

%% Console summary
seg = {'approach (t < 15.5 s)',            t < t_standoff; ...
       'final approach, marker visible',   t >= t_standoff & ~(t >= dropout(1) & t < dropout(2) + 0.5); ...
       'dropout window',                   t >= dropout(1) & t < dropout(2)};
fprintf('\nRMS estimation error, delay compensation ON:\n');
fprintf('%-34s %8s %8s %8s %9s %9s\n', 'segment', 'x[mm]', 'y[mm]', 'th[deg]', 'vx[mm/s]', 'vy[mm/s]');
for i = 1:size(seg, 1)
    e = res_on.err(seg{i,2}, :);
    fprintf('%-34s %8.2f %8.2f %8.3f %9.2f %9.2f\n', seg{i,1}, ...
        rms(e(:,1))*1e3, rms(e(:,2))*1e3, rad2deg(rms(e(:,3))), rms(e(:,4))*1e3, rms(e(:,5))*1e3);
end
fprintf('\nApproach RMS position error: delay compensation OFF %.2f mm, ON %.2f mm\n', ...
    rms(epos_off(t < t_standoff)), rms(epos_on(t < t_standoff)));
fprintf('Final pose error at t = %.1f s: x %.2f mm, y %.2f mm, theta %.3f deg\n', ...
    t(end), res_on.err(end,1)*1e3, res_on.err(end,2)*1e3, rad2deg(res_on.err(end,3)));
in3 = mean(abs(res_on.err) <= 3*sqrt(res_on.Pdiag), 1)*100;
fprintf('Samples inside 3 sigma: x %.1f%%, y %.1f%%, th %.1f%%, vx %.1f%%, vy %.1f%%\n', in3);
fprintf('NEES single run: mean %.2f (95%% band for 5 dof: %.2f to %.2f)\n', mean(res_on.nees), lo1, hi1);
fprintf('NEES %d-run mean: %.2f, inside band %.1f%% of the time (band %.2f to %.2f)\n', ...
    Nmc, mean(nees_avg), mean(nees_avg >= loN & nees_avg <= hiN)*100, loN, hiN);
fprintf('Fiducial updates: %d, rejected by the gate: %d\n', nnz(res_on.fid_used), nnz(res_on.fid_rej));
fprintf('Figures written to %s\n', outdir);
