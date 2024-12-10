#!/bin/bash

xhost +local:docker || true

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

docker run  --gpus all \
            -ti --rm \
            -e "DISPLAY" \
            -e "QT_X11_NO_MITSHM=1" \
            -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            -e XAUTHORITY \
            -v /dev:/dev \
            -v $ROOT_DIR:/stable-baseline-playground \
            --net=host \
            --privileged \
            --name stable-baseline-ppo-visual stable-baseline-ppo-visual-img
