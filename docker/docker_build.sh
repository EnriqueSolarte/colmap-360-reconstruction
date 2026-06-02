#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
docker build -f dockerfile.dev-container -t scene-reconstruction:dev ${DIR}
