#!/bin/bash

echo "INSTALLING THE APT DEPENDENCIES"
apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

echo "INSTALLING PYTHON DEPENDENCIES"
python -m pip install -U pip
pip install fastapi uvicorn
pip install -q diffusers transformers accelerate opencv-python controlnet-aux

echo "SETTING UP THE HF CACHE"
export HF_HOME="/workspace"
export DISABLE_TELEMETRY="YES"

echo "RUNNING THE SD SERVER"
cd /workspace
python sd_xl_generic_server_runpod_controlnet_only.py

echo "ALL DONE"
