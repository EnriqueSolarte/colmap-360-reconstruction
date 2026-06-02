#!/bin/bash
xhost +local:

CONTAINER_NAME="scene-reconstruction"

# this directory
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

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
        --env="DISPLAY=${DISPLAY}" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="${DOCKER_UTILS}:/docker_utils" \
        --volume="${DIR}/../:/workspace" \
        -e QT_XCB_GL_INTEGRATION=xcb_egl \
        scene-reconstruction:dev
fi
