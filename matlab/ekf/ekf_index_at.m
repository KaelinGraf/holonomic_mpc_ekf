function k = ekf_index_at(f, stamp)
%EKF_INDEX_AT Slot of the latest buffer entry with stamp <= stamp; [] before the buffer.
k = [];
i = f.head;
for j = 1:f.count
    if f.stamp(i) <= stamp, k = i; return; end
    i = i - 1;  if i == 0, i = f.L; end
end
end
