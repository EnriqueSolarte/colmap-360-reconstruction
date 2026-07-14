#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
docker build -f ${DIR}/dockerfile.dev-container --no-cache -t scene-reconstruction-colmap:dev ${DIR}
