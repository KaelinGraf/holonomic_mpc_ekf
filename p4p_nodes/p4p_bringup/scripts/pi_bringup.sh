#!/usr/bin/env bash
# Boot the Pi side inside the Pi container: sources ROS, the inference overlay and this workspace, then
# runs p4p_bringup's ordered launch from a run directory, so dock_control's bag lands in <run dir>/bags.
#
#   docker compose exec pi-camera /controller_ws/p4p_nodes/p4p_bringup/scripts/pi_bringup.sh [launch args]
#   P4P_RUN_DIR=/controller_ws bash .../pi_bringup.sh stack:=sensors
#
# Launch arguments pass through unchanged (stack:=, camera:=, rt:=, port:=, ...).
set -eo pipefail
source "/opt/ros/${ROS_DISTRO:-jazzy}/setup.bash"
source /workspace/src/ros/install/setup.bash
source /controller_ws/install/setup.bash
mkdir -p "${P4P_RUN_DIR:-/controller_ws}"
cd "${P4P_RUN_DIR:-/controller_ws}"
exec ros2 launch p4p_bringup pi.launch.py "$@"
