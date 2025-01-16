import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from PIL import Image
from io import BytesIO
import base64
import uvicorn
import os
import logging
import uvicorn
from time import time_ns

from dotenv import load_dotenv
load_dotenv()


STABILITY_API_KEY = os.getenv('STABILITY_API_KEY')


################# LOGGER ###################
logger = logging.getLogger('stability_ai_server')
logger.setLevel(logging.DEBUG)

# print to a log file
file_handler = logging.handlers.RotatingFileHandler(
    filename='stability_ai_server.log',
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


# uvicorn runs this
app = FastAPI()


class Text2ImageRequest(BaseModel):
	prompt: str
	negative_prompt: str = Field(default=None) # TODO: max_length ?
	model: str = Field(default='sd3-medium')
	seed: int = Field(default=1234, gt=0, description='Seed must be greater than zero')
    # converted to int later
	guidance_scale: float = Field(default=8.0, ge=0.0, description='Guidence scale must be greater than or equal to zero')
	# DEPRACATED!
	height: int = Field(default=1024, gt=0, description='Height must be greater than zero')
	width: int = Field(default=1024, gt=0, description='Width must be greater than zero')
	num_inference_steps: int = Field(default=50, gt=0, description='Number of inference steps scale must be greater than zero')


class Image2ImageRequest(BaseModel):
    image: str # base64 encoded image as string
    prompt: str
    negative_prompt: str = Field(default=None) # TODO: max_length ?
    model: str = Field(default='sd3-medium')
    seed: int = Field(default=1234, gt=0, description='Seed must be greater than zero')
    # converted to int later
    guidance_scale: float = Field(default=8.0, ge=0.0, description='Guidence scale must be greater than or equal to zero')
    # a lower strength value means the generated image is more similar to the initial image
    strength: float = Field(default=0.8, ge=0.0, description='Strength scale must be greater than or equal to zero')
    # DEPRACATED!
    height: int = Field(default=1024, gt=0, description='Height must be greater than zero')
    width: int = Field(default=1024, gt=0, description='Width must be greater than zero')
    num_inference_steps: int = Field(default=50, gt=0, description='Number of inference steps scale must be greater than zero')


class InpaintingRequest(BaseModel):
    image: str # base64 encoded image as string
    mask: str # base64 encoded mask image as string
    prompt: str
    negative_prompt: str = Field(default=None) # TODO: max_length ?
    model: str = Field(default='sd3-medium')
    seed: int = Field(default=1234, gt=0, description='Seed must be greater than zero')
    # converted to int later
    guidance_scale: float = Field(default=8.0, ge=0.0, description='Guidence scale must be greater than or equal to zero')
    # a lower strength value means the generated image is more similar to the initial image
    strength: float = Field(default=0.8, ge=0.0, description='Strength scale must be greater than or equal to zero')
    # DEPRACATED!
    height: int = Field(default=1024, gt=0, description='Height must be greater than zero')
    width: int = Field(default=1024, gt=0, description='Width must be greater than zero')
    num_inference_steps: int = Field(default=50, gt=0, description='Number of inference steps scale must be greater than zero')


# acceptable models from stability ai
# see: https://platform.stability.ai/docs/api-reference
MODEL_NAMES = (
     'core',
     'sd3.5-large',
     'sd3.5-large-turbo',
     'sd3.5-medium',
     'sd3-large',
     'sd3-large-turbo',
     'sd3-medium',
     'ultra'
)


@app.post('/t2i')
async def text2image_generation(request: Text2ImageRequest):

    if request.model.lower() not in MODEL_NAMES:
        return {'generation_status': f'error: model {request.model} is not supported!','generated_image': None}

    if request.model.lower() == 'core':
        URL = 'https://api.stability.ai/v2beta/stable-image/generate/core'
    elif request.model.lower() == 'ultra':
        URL = 'https://api.stability.ai/v2beta/stable-image/generate/ultra'
    else:
        URL = 'https://api.stability.ai/v2beta/stable-image/generate/sd3'

    response = requests.post(
        URL,
        headers={
            'authorization': f'Bearer {STABILITY_API_KEY}',
            'accept': 'image/*'
        },
        files={'none': ''},
        data={
            'prompt': request.prompt,
            'model': request.model.lower(),
            'seed': request.seed,
            'negative_prompt': request.negative_prompt,
            'cfg_scale': int(request.guidance_scale),
            'output_format': 'jpeg',
            'mode': 'text-to-image'
        },
    )

    if response.status_code == 200:
        generation_status = 'ok'
    else:
        generation_status = 'error'
        return {'generation_status': generation_status,'generated_image': None}

	# save image (OPTIONAL)
	#image.save(f'./generated_images/gen_img_{time_ns()}.jpg')

    buffer = BytesIO(response.content)
    generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
	
    return {'generation_status': generation_status,'generated_image': generated_image}


@app.post('/i2i')
async def image2image_generation(request: Image2ImageRequest):

    if request.model.lower() not in MODEL_NAMES:
        return {'generation_status': f'error: model {request.model} is not supported!','generated_image': None}

    if request.model.lower() == 'core':
        URL = 'https://api.stability.ai/v2beta/stable-image/generate/core'
    elif request.model.lower() == 'ultra':
        URL = 'https://api.stability.ai/v2beta/stable-image/generate/ultra'
    else:
        URL = 'https://api.stability.ai/v2beta/stable-image/generate/sd3'


    # decode base64 encoded image
    decoded_image = base64.b64decode(request.image, validate=True)
    decoded_image = BytesIO(decoded_image)

    response = requests.post(
        URL,
        headers={
            'authorization': f'Bearer {STABILITY_API_KEY}',
            'accept': 'image/*'
        },
        files={'image': decoded_image},
        data={
            'prompt': request.prompt,
            'model': request.model,
            'seed': request.seed,
            'negative_prompt': request.negative_prompt,
            'cfg_scale': int(request.guidance_scale),
            'strength': request.strength,
            'output_format': 'jpeg',
            'mode': 'image-to-image'
        },
    )

    if response.status_code == 200:
        generation_status = 'ok'
    else:
        generation_status = 'error'
        return {'generation_status': generation_status,'generated_image': None}

	# save image (OPTIONAL)
	#image.save(f'./generated_images/gen_img_{time_ns()}.jpg')

    buffer = BytesIO(response.content)
    generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
	
    return {'generation_status': generation_status,'generated_image': generated_image}


@app.post('/upscale4x')
async def image2image_generation(request: Image2ImageRequest):

    # decode base64 encoded image
    decoded_image = base64.b64decode(request.image, validate=True)
    decoded_image = BytesIO(decoded_image)

    response = requests.post(
        'https://api.stability.ai/v2beta/stable-image/upscale/fast',
        headers={
            'authorization': f'Bearer {STABILITY_API_KEY}',
            'accept': 'image/*'
        },
        files={'image': decoded_image},
        data={
            'output_format': 'jpeg',
        },
    )

    if response.status_code == 200:
        generation_status = 'ok'
    else:
        generation_status = 'error'
        return {'generation_status': generation_status,'generated_image': None}

	# save image (OPTIONAL)
	#image.save(f'./generated_images/gen_img_{time_ns()}.jpg')

    buffer = BytesIO(response.content)
    generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
	
    return {'generation_status': generation_status,'generated_image': generated_image}


@app.post('/inpainting')
async def image2image_generation(request: InpaintingRequest):
    
    # decode base64 encoded image
    decoded_image = base64.b64decode(request.image, validate=True)
    decoded_image = BytesIO(decoded_image)

    # decode base64 encoded image
    decoded_mask = base64.b64decode(request.mask, validate=True)
    decoded_mask = BytesIO(decoded_mask)

    response = requests.post(
        'https://api.stability.ai/v2beta/stable-image/edit/inpaint',
        headers={
            'authorization': f'Bearer {STABILITY_API_KEY}',
            'accept': 'image/*'
        },
        files={'image': decoded_image, 'mask': decoded_mask},
        data={
            'prompt': request.prompt,
            'model': request.model,
            'seed': request.seed,
            'negative_prompt': request.negative_prompt,
            'cfg_scale': int(request.guidance_scale),
            'strength': request.strength,
            'output_format': 'jpeg',
        },
    )

    if response.status_code == 200:
        generation_status = 'ok'
    else:
        generation_status = 'error'
        return {'generation_status': generation_status,'generated_image': None}

	# save image (OPTIONAL)
	#image.save(f'./generated_images/gen_img_{time_ns()}.jpg')

    buffer = BytesIO(response.content)
    generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
	
    return {'generation_status': generation_status,'generated_image': generated_image}



@app.post('/control')
async def image2image_generation(request: Image2ImageRequest):

    # decode base64 encoded image
    decoded_image = base64.b64decode(request.image, validate=True)
    decoded_image = BytesIO(decoded_image)

    response = requests.post(
        'https://api.stability.ai/v2beta/stable-image/control/sketch',
        headers={
            'authorization': f'Bearer {STABILITY_API_KEY}',
            'accept': 'image/*'
        },
        files={'image': decoded_image},
        data={
            'prompt': request.prompt,
            'seed': request.seed,
            'negative_prompt': request.negative_prompt,
            'control_strength': request.strength,
            'output_format': 'jpeg',
        },
    )

    if response.status_code == 200:
        generation_status = 'ok'
    else:
        generation_status = 'error'
        return {'generation_status': generation_status,'generated_image': None}

	# save image (OPTIONAL)
	#image.save(f'./generated_images/gen_img_{time_ns()}.jpg')

    buffer = BytesIO(response.content)
    generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
	
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
    logger.debug(f'Stability AI Grasshopper Deep Learning Services (GHDLS) is starting...')
	# accept every connection (not only local connections)
    uvicorn.run(app, host='0.0.0.0', port=8000)