#!/usr/bin/env bash

ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
EXEC_PATH=$PWD

cd $ROOT_DIR

echo "[!] If you use nvidia gpu, please rebuild with -n or --nvidia argument"
docker build -t stable-baseline-ppo-visual-img -f $ROOT_DIR/docker/Dockerfile $ROOT_DIR \
                                                 --network=host \
                                                 --build-arg from=nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
                                                 
cd $EXEC_PATH
