#!/usr/bin/env bash
# Run under a dedicated Xvfb display with -ac, never a user's desktop session.
set -euo pipefail
image=${1:?Usage: check_docker_viewer.sh IMAGE}
: "${DISPLAY:?Run this check with xvfb-run}"
container="franka-viewer-check-$$"
cleanup() {
  result=$?
  if [ "$result" -ne 0 ]; then docker logs "$container" >&2 || true; fi
  docker rm -f "$container" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# No published ports or host network: this check can run beside the headless test.
docker run -d --name "$container" --network none \
  --user "$(id -u):$(id -g)" -e DISPLAY -e LIBGL_ALWAYS_SOFTWARE=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro \
  "$image" --physics mujoco --vis >/dev/null

for attempt in $(seq 1 30); do
  if xwininfo -root -tree | grep -q '"MuJoCo :'; then
    docker exec "$container" franka-sim-check --timeout 30
    echo 'MuJoCo window opened and the FCI readiness check passed.'
    exit 0
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$container")" != true ]; then
    echo 'Viewer container exited before creating its window.' >&2
    exit 1
  fi
  sleep 1
done
echo 'MuJoCo did not create a window within 30 seconds.' >&2
exit 1
