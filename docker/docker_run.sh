#!/bin/bash
xhost +local:

CONTAINER_NAME="scene-reconstruction-colmap"

# this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_FILE="${DIR}/../env"

if [ ! -f "${ENV_FILE}" ]; then
    echo "Missing env file: ${ENV_FILE}" >&2
    exit 1
fi

set -a
source "${ENV_FILE}"
set +a

: "${ITRI_3D_RECONSTRUCTION:?Set ITRI_3D_RECONSTRUCTION in ${ENV_FILE}}"

if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' is already running. Attaching..."
    docker exec -it "${CONTAINER_NAME}" bash
else
    echo "Starting new container '${CONTAINER_NAME}'..."
    docker run -itd \
        --rm \
        --gpus all --runtime=nvidia \
        --name "${CONTAINER_NAME}" \
        --privileged \
        --network=host --ipc=host \
        --env-file "${ENV_FILE}" \
        --env="DISPLAY=${DISPLAY}" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="${DOCKER_UTILS}:/docker_utils" \
        --volume="${AGENT_SKILLS}:/agent-skills" \
        --volume="${DIR}/../:/workspace" \
        --volume="${ITRI_3D_RECONSTRUCTION}:/data/itri-3d-reconstruction" \
        -e QT_XCB_GL_INTEGRATION=xcb_egl \
        scene-reconstruction-colmap:dev
fi
