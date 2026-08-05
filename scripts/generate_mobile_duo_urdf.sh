#!/usr/bin/env bash
# Generate the combined mobile_fr3_duo URDF used by the libfranka-sim mobile
# scene, from a franka_description checkout on branch `jazzy`.
#
# The generated URDF and the meshes it references MUST come from the same
# checkout: mesh paths differ between franka_description branches (for example
# `meshes/accessories/franka_spine` on jazzy vs `franka_spine_v0_1` on main).
#
# Requires a sourced ROS 2 Jazzy environment (xacro needs `$(find ...)` to
# resolve through the ament index).
#
# Usage:
#   scripts/generate_mobile_duo_urdf.sh <franka_description_dir> <output.urdf>
set -euo pipefail

DESCRIPTION_DIR="${1:?usage: generate_mobile_duo_urdf.sh <franka_description_dir> <output.urdf>}"
OUTPUT="${2:?usage: generate_mobile_duo_urdf.sh <franka_description_dir> <output.urdf>}"

# The generated URDF and meshes MUST come from the pinned sha (see the
# robot_types note below): a different checkout can silently change mesh
# paths or joint names out from under the sim.
git -C "${DESCRIPTION_DIR}" rev-parse --verify HEAD | grep -q ^72baf5b || {
  echo "wrong franka_description sha: expected HEAD at 72baf5b..., got \
$(git -C "${DESCRIPTION_DIR}" rev-parse --short HEAD 2>/dev/null || echo 'unknown')" >&2
  exit 1
}

command -v xacro >/dev/null || {
  echo "xacro not found: source a ROS 2 Jazzy environment first" >&2
  exit 1
}

XACRO_INPUT="${DESCRIPTION_DIR}/robots/mobile_fr3_duo_v0_2/mobile_fr3_duo_v0_2.urdf.xacro"
[ -f "${XACRO_INPUT}" ] || { echo "not found: ${XACRO_INPUT}" >&2; exit 1; }

# robot_types is passed explicitly rather than relying on the xacro default.
# At the pinned sha 72baf5b the default already is fr3v2, but it has changed
# across refs (older jazzy revisions defaulted to fr3v2_1, which would name the
# joints left_fr3v2_1_joint1..7). Passing it explicitly makes the generated
# joint names left_fr3v2_joint1..7 / right_fr3v2_joint1..7 independent of the
# ref, matching the labs fr3_mobile_duo controller configuration.
xacro "${XACRO_INPUT}" \
  robot_types:="['tmrv0_2','fr3v2','fr3v2']" \
  hand:=false \
  ros2_control:=false \
  gazebo:=false \
  use_arms:=true \
  with_sc:=false \
  reduced_version:=false \
  > "${OUTPUT}"

echo "wrote ${OUTPUT} from $(git -C "${DESCRIPTION_DIR}" rev-parse --short HEAD)"
