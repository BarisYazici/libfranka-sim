# franka-sim: headless libfranka FCI simulator, ready for CI.
#
#   docker run --network host ghcr.io/barisyazici/franka-sim
#
# then point any libfranka client / franka_ros2 at 127.0.0.1. The FR3 and
# Franka Hand MJCF models are baked in and pinned via $FR3_MJCF /
# $FRANKA_HAND_MJCF, so the container needs no network access at runtime.
#
# No HEALTHCHECK on purpose: the server serves a single FCI client and
# fail-fasts on a busy command port, exactly like the real robot — a periodic
# probe would report "unhealthy" precisely while a client is connected. Use a
# one-shot readiness wait instead (the bundled GitHub Action does):
#
#   docker exec <container> franka-sim-check --timeout 30

# --- Stage 1: fetch the MuJoCo Menagerie models the server uses -------------
FROM python:3.11-slim AS models

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir 'robot_descriptions>=1.12'

# Download via robot_descriptions (same code path the package uses), then keep
# only the two model directories — not the whole Menagerie clone.
RUN python -c "\
import pathlib, shutil; \
from robot_descriptions import fr3_v2_mj_description as fr3, panda_mj_description as panda; \
out = pathlib.Path('/models'); out.mkdir(); \
fr3_dir = pathlib.Path(fr3.MJCF_PATH).parent; \
hand_dir = pathlib.Path(panda.MJCF_PATH).parent; \
assert (fr3_dir / 'fr3v2.xml').exists(), fr3_dir; \
assert (hand_dir / 'hand.xml').exists(), hand_dir; \
shutil.copytree(fr3_dir, out / 'franka_fr3_v2'); \
shutil.copytree(hand_dir, out / 'franka_emika_panda')"

# --- Stage 2: build the wheel ------------------------------------------------
# A separate stage so the final image carries the installed package only —
# no source tree layer.
FROM python:3.11-slim AS build

# Version stamp for setuptools_scm: the build context carries no .git, so the
# workflow passes the version explicitly (git describe / tag). Without it the
# wheel gets pyproject's fallback version.
ARG FRANKA_SIM_VERSION=""

WORKDIR /src
COPY . .
RUN pip install --no-cache-dir build \
    && if [ -n "$FRANKA_SIM_VERSION" ]; then \
        export SETUPTOOLS_SCM_PRETEND_VERSION="$FRANKA_SIM_VERSION"; \
    fi \
    && python -m build --wheel --outdir /wheels

# --- Stage 3: the server image ----------------------------------------------
FROM python:3.11-slim

COPY --from=models /models /opt/mujoco_menagerie
ENV FR3_MJCF=/opt/mujoco_menagerie/franka_fr3_v2/fr3v2.xml \
    FRANKA_HAND_MJCF=/opt/mujoco_menagerie/franka_emika_panda/hand.xml

RUN --mount=type=bind,from=build,source=/wheels,target=/wheels \
    pip install --no-cache-dir /wheels/*.whl

RUN useradd --create-home franka
USER franka
WORKDIR /home/franka

# 1337: FCI command port (TCP). 1338: gripper command port (TCP). Robot and
# gripper state stream back over UDP to the connecting client's own port.
# Prefer --network host in CI: port publishing adds NAT on a 1 kHz UDP loop.
EXPOSE 1337/tcp 1338/tcp

ENTRYPOINT ["run-franka-sim-server"]
