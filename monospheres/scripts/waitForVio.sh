#!/bin/bash
# Block until OpenVINS publishes odometry (it only does so once initialized).
# Usage: waitForVio.sh [timeout_s] [topic]. Exits 1 (with a loud message) on timeout.
TIMEOUT=${1:-120}
TOPIC=${2:-/$UAV_NAME/open_vins/odomimu}
echo "waiting for VIO odometry on $TOPIC (timeout ${TIMEOUT}s)"
if timeout "$TIMEOUT" ros2 topic echo --once "$TOPIC" nav_msgs/msg/Odometry > /dev/null 2>&1; then
  echo "VIO initialized"
  exit 0
fi
echo "!!! VIO NOT INITIALIZED after ${TIMEOUT}s (no message on $TOPIC) !!!"
exit 1
