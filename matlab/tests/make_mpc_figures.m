%MAKE_MPC_FIGURES Report figures for the docking MPC (closed loop with the EKF in the loop).
%   Figures go to matlab/out (or matlab/out/reference when the reference is on the path).
%     plane:      trajectory in the target frame with predicted trajectories at a few ticks
%     timeseries: pose vs time with stage switches and internal targets
%     commands:   body-velocity commands, increments and wheel speeds against their limits
%                 (the course's cart-pendulum layout: output / input / input increment)
%     rw:         effect of the scalar weight rw on the approach, three values
%     tick:       one tick's prediction against what actually happened (Example 5.3 flavour)
%
%   Usage:  >> setup_paths; make_mpc_figures      DOCK_VARIANT = 'fullscale' | 'demo'

close all;
variant = getenv('DOCK_VARIANT');
if isempty(variant), variant = 'fullscale'; end
prm = dock_params(variant);
root = fullfile(fileparts(mfilename('fullpath')), '..');
is_ref = contains(which('mpc_step'), [filesep, 'reference', filesep]);
if is_ref, outdir = fullfile(root, 'out', 'reference'); else, outdir = fullfile(root, 'out'); end
if ~exist(outdir, 'dir'), mkdir(outdir); end
fig_file = @(name) fullfile(outdir, sprintf('mpc_%s_%s.png', variant, name));
fprintf('=== Docking MPC figures, variant: %s ===\n', variant);

pose0 = [-2.0; 0.30; deg2rad(15)];
rec = [1, 40, 120, 200, 300];
L = run_closed_loop(prm, pose0, struct('use_ekf', true, 'seed', 1, 't_max', 40, 'record_ticks', rec));
t = L.t;
fprintf('docked at %.1f s; pose at dock x %.1f mm, y %.1f mm, th %.2f deg; penetration %.1f mm; fallbacks %d\n', ...
        L.t_dock, L.pose_at_dock(1)*1e3, L.pose_at_dock(2)*1e3, rad2deg(L.pose_at_dock(3)), max(0, L.max_x)*1e3, L.n_fallback);

f1 = figure('Visible', 'off', 'Position', [100 100 900 520]);
h1 = plot(L.pose(:,1), L.pose(:,2), 'k-', 'LineWidth', 1.3); hold on; grid on; axis equal;
h2 = plot(L.xhat(:,1), L.xhat(:,2), 'b--', 'LineWidth', 0.8);
h3 = [];
for i = 1:numel(L.preds)
    Y = L.preds(i).Y;
    h3 = plot(Y(:,1), Y(:,2), 'r-', 'LineWidth', 0.8);
    plot(Y(1,1), Y(1,2), 'ro', 'MarkerSize', 4);
end
h4 = plot(-prm.mpc.standoff, 0, 'gs', 'MarkerFaceColor', 'g');
h5 = plot(0, 0, 'ks', 'MarkerFaceColor', 'k');
xline(0, 'k:');
xlabel('x_T [m]'); ylabel('y_T [m]');
legend([h1, h2, h3, h4, h5], 'true path', 'EKF estimate', 'MPC predictions', 'standoff', 'docked pose', 'Location', 'best');
title(sprintf('Closed-loop docking (%s): predictions at ticks %s', variant, mat2str(rec)));
exportgraphics(f1, fig_file('plane'), 'Resolution', 110);

f2 = figure('Visible', 'off', 'Position', [100 100 900 700]);
tl = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
sw = find(diff(L.stage) ~= 0);
nexttile; hold on; grid on; plot(t, L.pose(:,1), 'k-'); plot(t, L.xhat(:,1), 'b--');
yline(-prm.mpc.standoff, 'g--'); yline(0, 'k:'); for s = sw', xline(t(s), 'm:'); end
ylabel('x [m]'); legend('true', 'estimate', 'standoff', 'contact', 'stage switch', 'Location', 'southeast');
title(sprintf('Pose during docking (%s); dock at %.1f s', variant, L.t_dock));
nexttile; hold on; grid on; plot(t, L.pose(:,2)*1e3, 'k-'); plot(t, L.xhat(:,2)*1e3, 'b--');
yline(prm.tgt.y*1e3, 'g--'); yline(-prm.tgt.y*1e3, 'g--'); for s = sw', xline(t(s), 'm:'); end
ylabel('y [mm]');
nexttile; hold on; grid on; plot(t, rad2deg(L.pose(:,3)), 'k-'); plot(t, rad2deg(L.xhat(:,3)), 'b--');
yline(rad2deg(prm.tgt.th), 'g--'); yline(-rad2deg(prm.tgt.th), 'g--'); for s = sw', xline(t(s), 'm:'); end
ylabel('\theta [deg]'); xlabel(tl, 'time [s]');
exportgraphics(f2, fig_file('timeseries'), 'Resolution', 110);

n_mpc = round(prm.mpc.Ts/prm.imu.T);
kt = 1:n_mpc:numel(t);
U  = L.u(kt, :);  dU = [zeros(1,3); diff(U)];  W = L.w(kt, :);  tt = t(kt);
f3 = figure('Visible', 'off', 'Position', [100 100 900 800]);
tl = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; hold on; grid on; stairs(tt, U, 'LineWidth', 0.9);
yline(prm.lim.v_approach, 'r--'); yline(-prm.lim.v_approach, 'r--'); yline(prm.lim.v_final, 'r:'); yline(-prm.lim.v_final, 'r:');
ylabel('u [m/s, rad/s]'); legend('u_b', 'v_b', 'r_b', 'approach limit', '', 'final limit', 'Location', 'northeast');
title(sprintf('Commands, increments and wheel speeds (%s)', variant));
nexttile; hold on; grid on; stairs(tt, dU, 'LineWidth', 0.9);
yline(prm.mpc.du_max(1), 'r--'); yline(-prm.mpc.du_max(1), 'r--');
yline(prm.mpc.du_max(3), 'r:'); yline(-prm.mpc.du_max(3), 'r:');
ylabel('\Delta u per tick'); legend('\Delta u_b', '\Delta v_b', '\Delta r_b', 'translation limit', '', 'yaw limit', 'Location', 'northeast');
nexttile; hold on; grid on; stairs(tt, W, 'LineWidth', 0.9);
yline(prm.lim.w_max, 'r--'); yline(-prm.lim.w_max, 'r--');
ylabel('wheel speed [rad/s]'); legend('FL', 'FR', 'RL', 'RR', 'limit', 'Location', 'northeast'); xlabel(tl, 'time [s]');
exportgraphics(f3, fig_file('commands'), 'Resolution', 110);

rws = [0.1, 1, 10];
f4 = figure('Visible', 'off', 'Position', [100 100 900 600]);
tl = tiledlayout(2, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
ax1 = nexttile; hold on; grid on; ax2 = nexttile; hold on; grid on;
for rw = rws
    Lr = run_closed_loop(prm, pose0, struct('use_ekf', true, 'seed', 1, 't_max', 40, 'rw', rw));
    plot(ax1, Lr.t, Lr.pose(:,1), 'DisplayName', sprintf('r_w = %g, dock %.1f s', rw, Lr.t_dock));
    plot(ax2, Lr.t, Lr.u(:,1), 'DisplayName', sprintf('r_w = %g', rw));
end
yline(ax1, 0, 'k:', 'HandleVisibility', 'off'); ylabel(ax1, 'x [m]'); legend(ax1, 'Location', 'southeast');
title(ax1, sprintf('Effect of the increment weight r_w (%s)', variant));
yline(ax2, prm.lim.v_approach, 'r--', 'HandleVisibility', 'off'); ylabel(ax2, 'u_b [m/s]'); xlabel(ax2, 'time [s]'); legend(ax2, 'Location', 'northeast');
exportgraphics(f4, fig_file('rw'), 'Resolution', 110);

if numel(L.preds) >= 4
    pr = L.preds(4);  Y = pr.Y;  tp = pr.t + prm.mpc.Ts*(1:prm.mpc.Np)';
    f5 = figure('Visible', 'off', 'Position', [100 100 900 420]);
    hold on; grid on;
    plot(tp, Y(:,1), 'r-', 'LineWidth', 1.2);
    plot(t, L.pose(:,1), 'k-');
    xline(pr.t, 'm:'); xline(pr.t + prm.mpc.Nc*prm.mpc.Ts, 'b:');
    win = t >= pr.t - 1 & t <= pr.t + prm.mpc.Np*prm.mpc.Ts + 1;
    xlim([pr.t - 1, pr.t + prm.mpc.Np*prm.mpc.Ts + 1]);
    ylim([min([Y(:,1); L.pose(win,1)]) - 0.02, max([Y(:,1); L.pose(win,1)]) + 0.02]);
    xlabel('time [s]'); ylabel('x [m]');
    legend('predicted over N_p', 'actual', 'tick', 'end of N_c', 'Location', 'southeast');
    title(sprintf('Receding horizon at tick %d (t = %.2f s): prediction vs outcome', pr.tick, pr.t));
    exportgraphics(f5, fig_file('tick'), 'Resolution', 110);
end
fprintf('Figures written to %s\n', outdir);
