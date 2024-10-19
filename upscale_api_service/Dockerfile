FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

RUN pip install fastapi uvicorn
RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
RUN pip install -q diffusers transformers accelerate opencv-python controlnet-aux

RUN mkdir -p /root/generative_app
RUN mkdir -p /root/generative_app/generated_images

COPY i2i_upscale_generic_server.py /root/generative_app

WORKDIR /root/generative_app

# disable huggingface telemetry
ENV DISABLE_TELEMETRY="YES"

# expose port 9000
EXPOSE 9000

# command to run the FastAPI application
CMD ["python", "i2i_upscale_generic_server.py"]
