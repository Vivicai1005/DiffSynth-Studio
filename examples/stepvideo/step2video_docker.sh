#!/bin/bash
# This script runs the Docker container in detached mode
sudo docker run -it --name step2video --ipc=host --cap-add=SYS_PTRACE --network=host \
 --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined \
 -v /home/weicai:/work --group-add video --privileged -w /work \
 rocm/pytorch:rocm6.3.3_ubuntu22.04_py3.9_pytorch_release_2.4.0