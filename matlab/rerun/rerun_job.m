function rerun_job(job, part, nparts)
%RERUN_JOB One slice of the MATLAB re-run (MATLAB_RERUN_BRIEF.md, Section 3). Writes JSON
%   to matlab/out/rerun_v2/raw/<job>_<part>.json. Jobs: R1, R1sweep, R2, R3, R4, R5, R5demo,
%   R6, R7full, R7demo. part/nparts split the run list for parallel MATLAB processes.
if nargin < 2, part = 1; nparts = 1; end
root = fileparts(fileparts(mfilename('fullpath')));
addpath(genpath(root));  rmpath(genpath(fullfile(root, 'archive')));
outdir = fullfile(root, 'out', 'rerun_v2', 'raw');  if ~exist(outdir, 'dir'), mkdir(outdir); end
warning('off', 'all');
res = struct('job', job, 'part', part, 'model', 'dock_params v2-deployed-2026-10-16', 'runs', []);
switch job
    case 'R1'
        runs = {};
        for v = {'fullscale', 'demo'}
            prm = dock_params(v{1});  tr = r1_traj(prm);
            for s = 1:20, runs{end+1} = r1_run(prm, tr, s, 1); end
        end
        res.runs = [runs{:}];
    case 'R1sweep'
        fac = [0.01 0.03 0.1 0.3 1 3];  runs = {};
        for a = fac
            prm = dock_params('fullscale');  prm = retune(prm, a/0.03);  tr = r1_traj(prm);
            for s = 1:20, runs{end+1} = r1_run(prm, tr, s, a); end
        end
        res.runs = [runs{:}];
    case 'R2'
        res.runs = r2_tests();
    case 'R3'
        runs = {};
        for v = {'fullscale', 'demo'}
            prm = dock_params(v{1});  tr = r1_traj(prm);  drop = [14, 16];
            for s = 1:20
                o = run_ekf(tr, prm, struct('seed', 300 + s, 'dropout', drop));
                win = tr.t >= drop(1) & tr.t < drop(2) + 0.1;  tw = tr.t(win) - drop(1);
                ep = hypot(o.err(win, 1), o.err(win, 2));  ey = abs(o.err(win, 2));
                sig3 = 3*sqrt(o.Pdiag(win, 1) + o.Pdiag(win, 2));
                over = find(ep > prm.tol.y, 1);  t_in = 2;  if ~isempty(over), t_in = min(2, tw(over)); end
                overy = find(ey > prm.tol.y, 1);  t_in_y = 2;  if ~isempty(overy), t_in_y = min(2, tw(overy)); end
                after = o.fixes(o.fixes(:, 1) >= drop(2), :);  after = after(1:min(5, end), :);
                runs{end+1} = struct('variant', v{1}, 'seed', 300 + s, 't_inside_norm_s', t_in, ...
                    't_inside_lateral_s', t_in_y, 'max_err_norm_mm', 1e3*max(ep), 'max_err_over_3sigma', max(ep./sig3), ...
                    'fixes_accepted_on_return', sum(after(:, 3)), 'fixes_checked_on_return', size(after, 1));
            end
        end
        res.runs = [runs{:}];
    case 'R4'
        res.runs = r4_tick();
    case {'R5', 'R5demo'}
        if strcmp(job, 'R5'), v = 'fullscale'; else, v = 'demo'; end
        prm = dock_params(v);  P = [[-0.85; 0.25; deg2rad(-10)], make_starts(prm, 30, 2026)];
        idx = part:nparts:size(P, 2);  runs = cell(1, numel(idx));
        for i = 1:numel(idx)
            j = idx(i);
            L = run_closed_loop(prm, P(:, j), struct('seed', 5000 + j, 'log', j == 1));
            runs{i} = pack(L, v, j - 1, j == 1);
        end
        res.runs = [runs{:}];
    case 'R6'
        S0 = [-0.85 0.25 -10; -0.80 -0.30 20; -0.40 0.02 2]';  S0(3, :) = deg2rad(S0(3, :));
        runs = {};
        for v = {'fullscale', 'demo'}
            prm = dock_params(v{1});
            for j = 1:3
                L = run_closed_loop(prm, S0(:, j), struct('seed', 600 + j, 'use_ekf', false));
                runs{end+1} = pack(L, v{1}, j, true);
            end
            L = run_closed_loop(prm, S0(:, 1), struct('seed', 601, 'use_ekf', true));
            runs{end+1} = pack(L, v{1}, 1, true);
        end
        res.runs = [runs{:}];
    case {'R7full', 'R7demo'}
        if strcmp(job, 'R7full'), v = 'fullscale'; else, v = 'demo'; end
        cases = {};
        for x = [0.05 0.1 0.2 0.3 0.5], cases{end+1} = {'tau', x}; end
        for x = [0.01 0.05 0.1 0.2],     cases{end+1} = {'delay', x}; end
        for x = [1 2 4],                 cases{end+1} = {'noise', x}; end
        list = {};
        for c = 1:numel(cases), for s = 1:10, list{end+1} = {cases{c}{1}, cases{c}{2}, s}; end, end
        idx = part:nparts:numel(list);  runs = cell(1, numel(idx));
        for i = 1:numel(idx)
            c = list{idx(i)};  prm = dock_params(v);  o = struct('seed', 7000 + c{3});
            switch c{1}
                case 'tau',   prm.plant.tau = c{2};
                case 'delay', prm.fid.delay = c{2};
                case 'noise', o.noise_scale = c{2};
                case 'bias',  o.bias_scale = c{2};
            end
            L = run_closed_loop(prm, [-0.85; 0.25; deg2rad(-10)], o);
            r = pack(L, v, c{3}, false);  r.param = c{1};  r.value = c{2};
            runs{i} = r;
        end
        res.runs = [runs{:}];
end
fid = fopen(fullfile(outdir, sprintf('%s_%d.json', job, part)), 'w');
fprintf(fid, '%s', jsonencode(res));  fclose(fid);
fprintf('%s part %d: %d runs written\n', job, part, numel(res.runs));
end

function prm = retune(prm, a)
prm.ekf.q_v = a*prm.ekf.q_v;  prm.ekf.q_th = a*prm.ekf.q_th;
prm.ekf.Qfloor = diag([1e-8, 1e-8, prm.ekf.q_th, prm.ekf.q_v, prm.ekf.q_v]);
end

function r = r1_run(prm, tr, s, fac)
o = run_ekf(tr, prm, struct('seed', 100 + s));
m = ~isnan(o.nees);
in3 = abs(o.err(m, :)) <= 3*sqrt(o.Pdiag(m, :));
r = struct('variant', prm.variant, 'seed', 100 + s, 'qv_factor', fac, 'q_v', prm.ekf.q_v, 'q_th', prm.ekf.q_th, ...
    't', o.t(m)', 'nees', o.nees(m)', 'in3_all', mean(all(in3, 2)), 'in3_each', mean(in3, 1), ...
    'rms_err', sqrt(mean(o.err(m, :).^2, 1)), 'max_nis', max(o.nis), 'nis', o.nis, ...
    'n_acc', o.n_acc, 'n_rej', o.n_rej);
end

function r = pack(L, v, idx, with_log)
f = {'pose0', 'seed', 'use_ekf', 'docked', 'aborted', 't_motion', 't_dock', 'time_to_dock', 'pose_at_dock', ...
     'v_at_dock', 'y_dock', 'th_dock', 'in_window', 'in_target', 'max_pen', 'v_contact', 'n_fallback', ...
     'n_slack_ticks', 'n_fix_acc', 'n_fix_rej', 'n_fix_refused', 'n_reanchor', 'final_stage', 'max_w_cmd', 'max_abs_th'};
r = struct('variant', v, 'start_index', idx);
for i = 1:numel(f), r.(f{i}) = L.(f{i}); end
r.nis = L.nis;
if with_log
    r.tick = table2struct(L.tick, 'ToScalar', true);
    if isfield(L, 'imu'), r.imu = table2struct(L.imu, 'ToScalar', true); end
else
    r.tick = [];
end
if ~isfield(r, 'imu'), r.imu = []; end
end
