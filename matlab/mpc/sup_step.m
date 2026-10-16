function [sv, out] = sup_step(sv, in, prm)
%SUP_STEP Seven-stage supervisor, one 20 Hz tick (supervisor.py Supervisor.step).
%   Stages: 0 IDLE, 1 ACQUIRE, 2 APPROACH, 3 FINAL, 4 DOCKED, 5 HOLD, 6 ABORT.
%   in: now, armed, x (5x1 or []), p_diag, fix_age, est_age, imu_rate (fraction of
%       nominal), misses (consecutive solver failures), u_last
%   out: stage, yr, lim (caps for mpc_step), qw, move (command enabled)
%   Create with sv = struct('stage', 0, 'entered', 0, 'resume', 2, 'hold_ref', zeros(3,1)).
S = prm.sup;  du_t = prm.mpc.du_max(1);
stage = sv.stage;  x = in.x;
moving = any(stage == [2 3 5]);
unsafe = in.est_age > S.stale || isempty(x) || in.p_diag(1) > S.abort_sd^2 || in.p_diag(2) > S.abort_sd^2;
if ~in.armed
    nxt = 0;
elseif moving && unsafe
    nxt = 6;
elseif stage == 0
    nxt = 1;
elseif stage == 1
    ready = ~isempty(x) && in.est_age <= S.stale && in.imu_rate >= S.imu_rate_min ...
            && in.p_diag(1) < S.start_sd^2 && in.p_diag(2) < S.start_sd^2;
    nxt = 1 + ready;
elseif any(stage == [2 3]) && (in.fix_age > S.hold_after || in.misses >= S.miss_max)
    nxt = 5;
elseif stage == 2
    aligned = abs(x(2)) < S.handover_y && abs(x(3)) < S.handover_th && abs(x(1) + S.standoff) < S.handover_x;
    nxt = 2 + aligned;
elseif stage == 3
    docked = x(1) >= -S.dock_x && x(4)^2 + x(5)^2 < S.dock_v^2 && abs(x(2)) <= prm.tgt.y && abs(x(3)) <= prm.tgt.th;
    nxt = 3 + docked;
elseif stage == 5
    if in.fix_age <= S.hold_after && in.misses == 0
        nxt = sv.resume;
    elseif in.now - sv.entered > S.abort_after
        nxt = 6;
    else
        nxt = 5;
    end
else
    nxt = stage;
end
if nxt ~= stage
    sv.entered = in.now;
    if nxt == 5, sv.resume = stage; sv.hold_ref = x(1:3); end
end
sv.stage = nxt;
u = abs(in.u_last(:));
lim = struct('du_max', prm.mpc.du_max, 'v_max', 0, 'v_lat', 0, 'l1', false, 'r_max', prm.lim.r_max, ...
             'w_max', prm.lim.w_max, 'x_max', prm.mpc.x_max, 'th_max', prm.mpc.th_max);
out = struct('stage', nxt, 'yr', zeros(3, 1), 'lim', lim, 'qw', [1; 1; 1], 'move', false);
ramped = max(prm.cap.v_final, u(1) + u(2) - min(u(1), du_t) - min(u(2), du_t));
switch nxt
    case 2
        to_go = max(0, -S.standoff - x(1));
        out.yr = [-S.standoff; 0; look_at(x, to_go, S, prm.geom)];
        s = ss((S.slow_d(1) - to_go) / (S.slow_d(1) - S.slow_d(2)));
        cap = prm.cap.v_approach + s*(prm.cap.v_align - prm.cap.v_approach);
        out.lim.v_max = min(prm.cap.v_approach, max(cap, max(u(1), u(2)) - du_t));
        out.lim.v_lat = out.lim.v_max;
        out.qw = approach_qw(x, S, prm);  out.move = true;
    case 3
        out.lim.v_max = max(prm.cap.v_final, min(prm.cap.v_approach, u(1) - du_t));
        out.lim.v_lat = max(prm.cap.v_lat_final, u(2) - du_t);
        out.qw = prm.mpc.qw_final;  out.move = true;
    case 5
        out.yr = sv.hold_ref;  out.lim.v_max = ramped;  out.lim.v_lat = ramped;  out.lim.l1 = true;
        if sv.resume == 3, out.qw = prm.mpc.qw_final; else, out.qw = approach_qw(sv.hold_ref, S, prm); end
        out.move = true;
end
end

function s = ss(s)
s = min(1, max(0, s));  s = s*s*(3 - 2*s);
end

function q = approach_qw(x, S, prm)
to_go = max(0, -S.standoff - x(1));
s = ss((S.qw_d(1) - to_go) / (S.qw_d(1) - S.qw_d(2)));
q = prm.mpc.qw_approach_far + s*(prm.mpc.qw_approach_near - prm.mpc.qw_approach_far);
end

function th = look_at(x, to_go, S, g)
w = ss((to_go - S.lookat_d(2)) / (S.lookat_d(1) - S.lookat_d(2)));
c = cos(x(3));  s = sin(x(3));
cx = x(1) + c*g.cam(1) - s*g.cam(2);  cy = x(2) + s*g.cam(1) + c*g.cam(2);
th = w * atan2(g.board(2) - cy, g.board(1) - cx);
end
