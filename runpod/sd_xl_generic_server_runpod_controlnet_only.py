from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL

import torch

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from PIL import Image
from io import BytesIO
import base64
import uvicorn
import gc
import logging
import os
from time import time_ns

from PIL import Image, ImageDraw
import cv2
import numpy as np


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


# this folder is generated by Dockerfile
# however, standalone runs might need it
os.makedirs('./generated_images/', exist_ok=True)


def make_mask(size, x1, y1, x2, y2):
    custom_mask = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(custom_mask)
    draw.rectangle([x1, y1, x2, y2], fill=(200, 200, 200))
    return custom_mask


################# LOGGER ###################
logger = logging.getLogger('sd_xl_generic_server_controlnet_only')
logger.setLevel(logging.DEBUG)

# print to a log file
file_handler = logging.handlers.RotatingFileHandler(
    filename='sd_xl_generic_server_controlnet_only.log',
    encoding='utf-8',
    maxBytes=32 * 1024 * 1024, # 32 MB
    backupCount=5,  # Rotate through 5 files
)

# hour minute, seconds, day, month, year
log_format = '%H:%M:%S %d-%m-%Y'
formatter = logging.Formatter('[{asctime}] [{levelname:<8}] {name}: {message}', log_format, style='{')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

# print to terminal (console)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)
############################################


######### IMAGE 2 IMAGE CONTROLNET #########
print('LOADING IMG2IMG CONTROLNET MODEL...')

controlnet = ControlNetModel.from_pretrained(
    'diffusers/controlnet-canny-sdxl-1.0',
        cache_dir='/workspace/hub',
    torch_dtype=torch.float16,
)

controlnet_vae = AutoencoderKL.from_pretrained(
    'madebyollin/sdxl-vae-fp16-fix',
        cache_dir='/workspace/hub',
    torch_dtype=torch.float16
)

i2i_controlnet_pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
        cache_dir='/workspace/hub',
    controlnet=controlnet,
    vae=controlnet_vae,
    torch_dtype=torch.float16
).to('cuda')

print('IMG2IMG CONTROLNET MODEL LOADED SUCCESSFULLY!')
############################################


# uvicorn runs this
app = FastAPI()


class Image2ImageControlNetRequest(BaseModel):
        image: str # base64 encoded image as string
        prompt: str
        negative_prompt: str = Field(default=None) # TODO: max_length ?
        seed: int = Field(default=1234, gt=0, description='Seed must be greater than zero')
        guidance_scale: float = Field(default=10.5, ge=0.0, description='Guidence scale must be greater than or equal to zero')
        # a lower strength value means the generated image is more similar to the initial image
        strength: float = Field(default=0.8, ge=0.0, description='Strength scale must be greater than or equal to zero')
        height: int = Field(default=1024, gt=0, description='Height must be greater than zero')
        width: int = Field(default=1024, gt=0, description='Width must be greater than zero')
        num_inference_steps: int = Field(default=50, gt=0, description='Number of inference steps scale must be greater than zero')
        # added in controlnet
        model: str = Field(default='canny')
        controlnet_conditioning_scale: float = Field(default=0.5, ge=0.0, description='Conditioning scale must be greater than or equal to zero')
        threshold1: int = Field(default=100, gt=0, description='Threshold 1 must be greater than zero')
        threshold2: int = Field(default=200, gt=0, description='Threshold 2 must be greater than zero')


@app.post('/i2i_controlnet')
async def image2image_controlnet_generation(request: Image2ImageControlNetRequest):

        try:
                generator = torch.Generator(device).manual_seed(request.seed)

                # decode base64 encoded image
                decoded_image = base64.b64decode(request.image, validate=True)
                decoded_image = Image.open(BytesIO(decoded_image))#.convert('RGB')

                ################## CANNY ##################
                image_np = np.array(decoded_image)
                image_np = cv2.Canny(image_np, request.threshold1, request.threshold2)
                image_np = image_np[:, :, None]
                image_np = np.concatenate([image_np, image_np, image_np], axis=2)
                canny_image = Image.fromarray(image_np)
                ############################################

                image = i2i_controlnet_pipeline(
                        image=canny_image,
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt,
                        generator=generator,
                        guidance_scale=request.guidance_scale,
                        strength=request.strength,
                        height=request.height,
                        width=request.height,
                        num_inference_steps=request.num_inference_steps,
                controlnet_conditioning_scale=request.controlnet_conditioning_scale # added in controlnet
                ).images[0]

                # save image (OPTIONAL)
                #image.save(f'./generated_images/gen_img_{time_ns()}.jpg')

                buffer = BytesIO()
                image.save(buffer, format='JPEG')
                generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

                generation_status = 'ok'
        except:
                generation_status = 'error'
                generated_image = None

        return {'generation_status': generation_status,'generated_image': generated_image}


@app.middleware('http')
async def log_request_info(request, call_next):
        """
        request.client.host
        request.client.port
        request.method
        request.url
        request.url.path
        request.query_params
        request.headers
        """
        logger.debug(f'{request.url.path} endpoint received {request.method} request from {request.client.host}:{request.client.port} using agent: {request.headers["user-agent"]}')

        response = await call_next(request)
        return response


if __name__ == '__main__':
        logger.debug(f'SD XL Grasshopper Deep Learning Services (GHDLS) is starting...')
        # accept every connection (not only local connections)
        uvicorn.run(app, host='0.0.0.0', port=8000)