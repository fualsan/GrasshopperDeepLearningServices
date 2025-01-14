#!/bin/bash

echo "INSTALLING THE APT DEPENDENCIES"
apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

echo "INSTALLING PYTHON DEPENDENCIES"
python -m pip install -U pip
pip install fastapi uvicorn
pip install -q diffusers transformers accelerate opencv-python controlnet-aux
pip install python-dotenv openai

echo "SETTING UP THE HF CACHE"
export HF_HOME="/workspace"
export DISABLE_TELEMETRY="YES"

echo "RUNNING THE COMBINED SERVER"
cd /workspace
python combined_service.py

echo "ALL DONE"
