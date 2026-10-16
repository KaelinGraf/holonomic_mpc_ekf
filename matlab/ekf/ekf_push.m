function f = ekf_push(f, stamp, u, h, zv)
%EKF_PUSH Store the current state as the newest buffer entry (ekf.py RingBuffer.push).
f.head = mod(f.head, f.L) + 1;
f.count = min(f.count + 1, f.L);
k = f.head;
f.stamp(k) = stamp;  f.X(:, k) = f.x;  f.PP(:, :, k) = f.P;
f.U(:, k) = u(:);    f.Tk(k) = h;      f.ZV(k) = zv;  f.valid(k) = true;
end
