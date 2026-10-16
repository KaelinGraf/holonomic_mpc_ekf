%MAKE_REPORT_FIGURES Statistical, sensitivity and comparison figures for the report.
%   All runs are closed loop with the EKF in the loop, both variants:
%     montecarlo:  N_MC random start poses and noise seeds; scatter of the lateral and yaw
%                  error at docking against the coupler tolerance and the internal target,
%                  with docking time, penetration and contact-speed distributions
%     sensitivity: wheel lag tau, fiducial delay, sensor-noise scale and IMU-bias scale,
%                  one at a time, with the filter unaware of the change
%     constraints: the same run with the constraints removed from the QP and the command
%                  clipped afterwards (the course's with/without constraints comparison)
%     estimator:   EKF in the loop against perfect state feedback, same seed
%   Figures go to matlab/out (or matlab/out/reference when the reference is on the path);
%   the numbers go to report_summary_<variant>.txt next to them.
%   Usage:  >> setup_paths; make_report_figures        (about 20 min; lower N_MC to shorten)

close all;
N_MC = 30;
root = fullfile(fileparts(mfilename('fullpath')), '..');
is_ref = contains(which('mpc_step'), [filesep, 'reference', filesep]);
if is_ref, outdir = fullfile(root, 'out', 'reference'); else, outdir = fullfile(root, 'out'); end
if ~exist(outdir, 'dir'), mkdir(outdir); end
pose0 = [-2.0; 0.30; deg2rad(15)];
nominal = struct('use_ekf', true, 'seed', 21, 't_max', 40);
variants = {'fullscale', 'demo'};
if ~isempty(getenv('DOCK_VARIANT')), variants = {getenv('DOCK_VARIANT')}; end   % one variant only

for variant = variants
    v = variant{1};  prm = dock_params(v);
    fig_file = @(name) fullfile(outdir, sprintf('report_%s_%s.png', v, name));
    ft = fopen(fullfile(outdir, sprintf('report_summary_%s.txt', v)), 'w');
    say(ft, '=== %s ===\n', v);

    %% Monte Carlo over start poses and seeds
    rng(7);
    y = nan(N_MC,1); th = y; td = y; pen = y; vc = y; myaw = y; fb = y; docked = false(N_MC,1); X0 = zeros(3, N_MC);
    for i = 1:N_MC
        p0 = [-2.5 + 1.7*rand; -0.4 + 0.8*rand; deg2rad(-20 + 40*rand)];
        X0(:,i) = p0;
        L = run_closed_loop(prm, p0, struct('use_ekf', true, 'seed', 1000 + i, 't_max', 40));
        docked(i) = ~isnan(L.t_dock);
        if docked(i)
            y(i) = L.pose_at_dock(2);  th(i) = wrap_pi(L.pose_at_dock(3));  td(i) = L.t_dock;
        end
        pen(i) = max(0, L.max_x);  vc(i) = L.v_contact;  myaw(i) = L.max_th;  fb(i) = L.n_fallback;
    end
    ok = docked & abs(y) <= prm.tol.y & abs(th) <= prm.tol.th;
    in_tgt = docked & abs(y) <= prm.tgt.y & abs(th) <= prm.tgt.th;
    say(ft, 'Monte Carlo, N = %d, start x in [-2.5,-0.8] m, y in [-0.4,0.4] m, yaw in [-20,20] deg\n', N_MC);
    say(ft, '  docked %d/%d, inside coupler tolerance %d/%d, inside internal target %d/%d\n', sum(docked), N_MC, sum(ok), N_MC, sum(in_tgt), N_MC);
    say(ft, '  lateral error at dock: rms %.1f mm, max %.1f mm (tolerance %.0f, target %.0f)\n', sqrt(mean(y(docked).^2))*1e3, max(abs(y(docked)))*1e3, prm.tol.y*1e3, prm.tgt.y*1e3);
    say(ft, '  yaw error at dock: rms %.2f deg, max %.2f deg (tolerance %.0f, target %.1f)\n', rad2deg(sqrt(mean(th(docked).^2))), rad2deg(max(abs(th(docked)))), rad2deg(prm.tol.th), rad2deg(prm.tgt.th));
    say(ft, '  docking time: mean %.1f s, max %.1f s; penetration max %.1f mm; contact speed max %.0f mm/s; max yaw %.0f deg; fallbacks %d\n', ...
        mean(td(docked)), max(td(docked)), max(pen)*1e3, max(abs(vc))*1e3, rad2deg(max(myaw)), sum(fb));
    f1 = figure('Visible', 'off', 'Position', [100 100 1000 760]);
    tl = tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile; hold on; grid on;
    rectangle('Position', [-prm.tol.y*1e3, -rad2deg(prm.tol.th), 2*prm.tol.y*1e3, 2*rad2deg(prm.tol.th)], 'EdgeColor', 'r', 'LineStyle', '--');
    rectangle('Position', [-prm.tgt.y*1e3, -rad2deg(prm.tgt.th), 2*prm.tgt.y*1e3, 2*rad2deg(prm.tgt.th)], 'EdgeColor', 'g', 'LineStyle', '--');
    plot(y(docked)*1e3, rad2deg(th(docked)), 'ko', 'MarkerFaceColor', 'k', 'MarkerSize', 4);
    xlabel('lateral error at dock [mm]'); ylabel('yaw error at dock [deg]');
    xlim([-1.2, 1.2]*prm.tol.y*1e3); ylim([-1.2, 1.2]*rad2deg(prm.tol.th));
    title(sprintf('%s: %d runs, %d docked, %d inside tolerance (red), %d inside target (green)', v, N_MC, sum(docked), sum(ok), sum(in_tgt)));
    nexttile; histogram(td(docked), 12); grid on; xlabel('docking time [s]'); ylabel('runs'); title('docking time');
    nexttile; histogram(pen*1e3, 12); grid on; hold on; xline(prm.mpc.contact_band*1e3, 'r--'); xlabel('penetration past first contact [mm]'); ylabel('runs'); title('penetration (red: end of engagement band)');
    nexttile; histogram(abs(vc)*1e3, 12); grid on; hold on; xline(prm.lim.v_final*1e3, 'r--'); xlabel('|closing speed| at contact [mm/s]'); ylabel('runs'); title('contact speed (red: final speed cap)');
    exportgraphics(f1, fig_file('montecarlo'), 'Resolution', 110);

    %% Sensitivity sweeps, filter unaware
    taus = [0.05 0.1 0.2 0.3 0.5];  Mt = nan(numel(taus), 6);
    for i = 1:numel(taus), p = prm; p.plant.tau = taus(i); Mt(i,:) = metrics(run_closed_loop(p, pose0, nominal)); end
    dels = [0.01 0.05 0.1 0.2];      Md = nan(numel(dels), 6);
    for i = 1:numel(dels), p = prm; p.fid.delay = dels(i); Md(i,:) = metrics(run_closed_loop(p, pose0, nominal)); end
    nsc = [1 2 4];                   Mn = nan(numel(nsc), 6);
    for i = 1:numel(nsc), o = nominal; o.noise_scale = nsc(i); Mn(i,:) = metrics(run_closed_loop(prm, pose0, o)); end
    bsc = [1 2 4];                   Mb = nan(numel(bsc), 6);
    for i = 1:numel(bsc), o = nominal; o.bias_scale = bsc(i); Mb(i,:) = metrics(run_closed_loop(prm, pose0, o)); end
    say(ft, 'Sensitivity (start pose (-2, 0.3, 15 deg), seed 21): columns = lateral mm, yaw deg, dock s, penetration mm, max yaw deg, fallbacks\n');
    say(ft, '  tau [s]      %s\n', mat2str([taus', Mt], 3));
    say(ft, '  delay [s]    %s\n', mat2str([dels', Md], 3));
    say(ft, '  noise scale  %s\n', mat2str([nsc', Mn], 3));
    say(ft, '  bias scale   %s\n', mat2str([bsc', Mb], 3));
    f2 = figure('Visible', 'off', 'Position', [100 100 1100 720]);
    tl = tiledlayout(3, 4, 'TileSpacing', 'compact', 'Padding', 'compact');
    sweeps = {taus, Mt, 'wheel lag \tau [s]', prm.plant.tau; dels, Md, 'fiducial delay [s]', prm.fid.delay; nsc, Mn, 'sensor noise scale', 1; bsc, Mb, 'IMU bias scale', 1};
    rows = {1, 'lateral error at dock [mm]', prm.tgt.y*1e3; 4, 'penetration [mm]', prm.mpc.contact_band*1e3; 3, 'docking time [s]', NaN};
    for r = 1:3
        for c = 1:4
            nexttile((r-1)*4 + c); hold on; grid on;
            plot(sweeps{c,1}, sweeps{c,2}(:, rows{r,1}), 'ko-', 'MarkerFaceColor', 'k');
            nd = isnan(sweeps{c,2}(:, 3));                       % runs that did not dock within 40 s
            if any(nd)
                yl = ylim;
                plot(sweeps{c,1}(nd), yl(2)*ones(1, sum(nd)), 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
                if r == 1, text(sweeps{c,1}(find(nd, 1)), yl(2), ' no dock', 'Color', 'r', 'VerticalAlignment', 'top', 'FontSize', 8); end
            end
            xline(sweeps{c,4}, 'b:');
            if ~isnan(rows{r,3}), yline(rows{r,3}, 'g--'); end
            if r == 3, xlabel(sweeps{c,3}); end
            if c == 1, ylabel(rows{r,2}); end
            if r == 1, title(sweeps{c,3}); end
        end
    end
    title(tl, sprintf('Sensitivity, %s (blue: nominal, green: target or band)', v));
    exportgraphics(f2, fig_file('sensitivity'), 'Resolution', 110);

    %% With and without constraints
    Lc = run_closed_loop(prm, pose0, nominal);
    o = nominal; o.unconstrained = true;
    Lu = run_closed_loop(prm, pose0, o);
    say(ft, 'Constraints: with -> dock %.1f s, penetration %.1f mm, max wheel %.1f rad/s; without (clipped afterwards) -> dock %.1f s, penetration %.1f mm, max wheel %.1f rad/s\n', ...
        Lc.t_dock, max(0, Lc.max_x)*1e3, Lc.max_w_cmd, Lu.t_dock, max(0, Lu.max_x)*1e3, Lu.max_w_cmd);
    f3 = figure('Visible', 'off', 'Position', [100 100 1000 760]);
    tl = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile; hold on; grid on; plot(Lc.t, Lc.pose(:,1), 'k-'); plot(Lu.t, Lu.pose(:,1), 'r-');
    yline(0, 'k:'); yline(prm.mpc.contact_band, 'g--'); ylabel('x [m]'); legend('constrained MPC', 'limits removed, command clipped', 'first contact', 'end of band', 'Location', 'southeast');
    title(sprintf('With and without constraints in the optimiser (%s)', v));
    nexttile; hold on; grid on; plot(Lc.t, Lc.pose(:,1)*1e3, 'k-'); plot(Lu.t, Lu.pose(:,1)*1e3, 'r-');
    yline(0, 'k:'); yline(prm.mpc.contact_band*1e3, 'g--'); ylim([-60, 60]); ylabel('x near contact [mm]');
    nexttile; hold on; grid on; plot(Lc.t, max(abs(Lc.w), [], 2), 'k-'); plot(Lu.t, max(abs(Lu.w), [], 2), 'r-');
    yline(prm.lim.w_max, 'r--'); ylabel('max |wheel speed| [rad/s]'); xlabel('time [s]');
    exportgraphics(f3, fig_file('constraints'), 'Resolution', 110);

    %% Estimator in the loop against perfect state
    o = nominal; o.use_ekf = false;
    Lp = run_closed_loop(prm, pose0, o);
    say(ft, 'Estimator: EKF loop -> dock %.1f s, y %.1f mm, yaw %.2f deg; perfect state -> dock %.1f s, y %.1f mm, yaw %.2f deg\n', ...
        Lc.t_dock, Lc.pose_at_dock(2)*1e3, rad2deg(wrap_pi(Lc.pose_at_dock(3))), Lp.t_dock, Lp.pose_at_dock(2)*1e3, rad2deg(wrap_pi(Lp.pose_at_dock(3))));
    f4 = figure('Visible', 'off', 'Position', [100 100 1000 760]);
    tl = tiledlayout(3, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
    nexttile; hold on; grid on; plot(Lp.t, Lp.pose(:,1), 'k-'); plot(Lc.t, Lc.pose(:,1), 'b-'); plot(Lc.t, Lc.xhat(:,1), 'b:');
    yline(0, 'k:'); ylabel('x [m]'); legend('perfect state', 'EKF in loop, true', 'EKF in loop, estimate', 'Location', 'southeast');
    title(sprintf('Estimator in the loop against perfect state feedback (%s)', v));
    nexttile; hold on; grid on; plot(Lp.t, Lp.pose(:,2)*1e3, 'k-'); plot(Lc.t, Lc.pose(:,2)*1e3, 'b-'); plot(Lc.t, Lc.xhat(:,2)*1e3, 'b:');
    yline(prm.tgt.y*1e3, 'g--'); yline(-prm.tgt.y*1e3, 'g--'); ylim([-60, 60]); ylabel('y [mm]');
    nexttile; hold on; grid on; plot(Lp.t, rad2deg(Lp.pose(:,3)), 'k-'); plot(Lc.t, rad2deg(Lc.pose(:,3)), 'b-'); plot(Lc.t, rad2deg(Lc.xhat(:,3)), 'b:');
    yline(rad2deg(prm.tgt.th), 'g--'); yline(-rad2deg(prm.tgt.th), 'g--'); ylabel('\theta [deg]'); xlabel('time [s]');
    exportgraphics(f4, fig_file('estimator'), 'Resolution', 110);

    fclose(ft);
end
fprintf('Report figures written to %s\n', outdir);

function m = metrics(L)
    if isnan(L.t_dock)
        m = [NaN, NaN, NaN, max(0, L.max_x)*1e3, rad2deg(L.max_th), L.n_fallback];
    else
        m = [abs(L.pose_at_dock(2))*1e3, rad2deg(abs(wrap_pi(L.pose_at_dock(3)))), L.t_dock, max(0, L.max_x)*1e3, rad2deg(L.max_th), L.n_fallback];
    end
end

function say(ft, varargin)
    fprintf(ft, varargin{:});
    fprintf(1, varargin{:});
end
