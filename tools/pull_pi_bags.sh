#!/usr/bin/env bash
# After a session: copy the Pi's docking bags to the laptop, then record every new run.
#
#     tools/pull_pi_bags.sh                              # the Pi as ssh's "pi" host (Ethernet pi-share)
#     PI=p4p@10.43.0.2 tools/pull_pi_bags.sh             # over the Pi's hotspot (p4p-dock-pi)
#     PI=p4p@172.23.152.148 tools/pull_pi_bags.sh        # over the university WiFi
#     tools/pull_pi_bags.sh --detectors                  # extra arguments go to run_analytics.py
#
# 1. rsync ~/holonomic_mpc_ekf/bags/ from the Pi into ~/p4p/run_data/pi_bags/. It copies what is new or has
#    grown and never deletes, on either side. Disarm docking first: a bag still being recorded arrives without
#    its metadata.yaml; it is read from its .mcap, and recorded again once a later pull brings it closed.
# 2. tools/run_analytics.py in the p4p-gpu container over the whole folder. Runs already recorded from the same
#    bag are skipped, and the index in report/project content/7_ros2_implementation/run_records/ is rebuilt.
#    Reading bags starts no ROS node and joins no network. --detectors also re-runs the detector comparison,
#    which needs the GPU (one model at a time): leave it out while a calibration or a fine-tune holds the GPU.
set -euo pipefail

PI=${PI:-pi}
SRC=${PI_BAGS:-holonomic_mpc_ekf/bags/}
P4P=${P4P:-$HOME/p4p}
DEST=$P4P/run_data/pi_bags/
CONTAINER=${CONTAINER:-p4p-gpu}

mkdir -p "$DEST"
echo "copying $PI:$SRC -> $DEST"
rsync -a --info=progress2,stats1 "$PI:$SRC" "$DEST"

args=""
[ $# -gt 0 ] && args=$(printf '%q ' "$@")
echo "recording the runs in $CONTAINER"
docker exec "$CONTAINER" bash -lc "source /workspace/controller_ws/install/setup.bash && \
    source /workspace/Conv-ChArT-Wireless-Inference/src/ros/install/setup.bash && \
    cd /workspace/controller_ws && python3 tools/run_analytics.py /workspace/run_data/pi_bags $args"
