import requests

from openai import OpenAI

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


client = OpenAI(
    api_key=os.getenv('OPENAI_KEY')
)


################# LOGGER ###################
logger = logging.getLogger('uniform_cloud_server')
logger.setLevel(logging.DEBUG)

# print to a log file
file_handler = logging.handlers.RotatingFileHandler(
    filename='uniform_cloud_server.log',
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


################# CONVERSATION ###################
INIT_CONTENT = {
    'role': 'developer',
    'content': 'You are a software engineer working on computational design. You will write Python code that runs on Rhino Grasshopper. Respond only the generated Python code, do not respond anything else. Code should be in ```python ... ``` brackets.'
}


conversation = [INIT_CONTENT]


def reset_conversation_history():
    global conversation
    conversation = [INIT_CONTENT]
##################################################


# uvicorn runs this
app = FastAPI()


class ScriptGenerationRequest(BaseModel):
	prompt: str
	model: str = Field(default='gpt-4o')
	temperature: float = Field(default=0.8, gt=0, description='Temperature must be between 0.0 and 2.0')


class DiffusionRequest(BaseModel):
    mode: str = Field(default=None)
    # used in image-to-image and control
    image: str = Field(default=None) # base64 encoded image as string
    # used in inpainting
    mask: str = Field(default=None) # base64 encoded mask image as string
    prompt: str = Field(default='')
    negative_prompt: str = Field(default='ugly, deformed, disfigured, poor details, bad anatomy, background') # TODO: max_length ?
    model: str = Field(default='sd3-medium')
    seed: int = Field(default=1234, gt=0, le=4294967294, description='Seed must be greater than zero and below 4294967294')
    # converted to int later
    # (known as cfg_scale in documentation)
    # NOTE: Point Aware 3D specific also uses this
    guidance_scale: float = Field(default=8.0, ge=0.0, le=10.0, description='Guidence scale must be between 0 and 10')
    # a lower strength value means the generated image is more similar to the initial image
    strength: float = Field(default=0.95, ge=0.0, le=1.0, description='Strength scale must be between 0.0 and 1.0')
    # used in control_style
    fidelity: float = Field(default=0.5, ge=0.0, le=1.0, description='Fidelity scale must be between 0.0 and 1.0')
    # used in Fast 3D & Point Aware 3D
    texture_resolution: str = Field(default='1024')
    # NOTE: DIFFERENT FOR 3D FAST AND POINT AWARE!
    foreground_ratio: float = Field(default=0.85, ge=0.1, le=2.0, description='Foreground ratio must be between 0.1 and 2.0')
    remesh: str = Field(default='none')
    # NOTE: Point Aware 3D specific does not use this
    # ("-1" means that a limit is not set)
    vertex_count: int = Field(default=-1, ge=-1, le=20000, description='Vertex count must be between -1 and 20000')
    # Point Aware 3D specific 
    target_type: str = Field(default='none')
    target_count: int = Field(default=1000, ge=100, le=20000, description='Target count must be between 100 and 20000')
    # DEPRACATED!
    height: int = Field(default=1024, gt=0, description='Height must be greater than zero')
    width: int = Field(default=1024, gt=0, description='Width must be greater than zero')
    num_inference_steps: int = Field(default=50, gt=0, description='Number of inference steps scale must be greater than zero')


# acceptable models from open ai
OPENAI_MODEL_NAMES = (
    'gpt-4o',
    'chatgpt-4o-latest',
    'gpt-4o-mini',
    'o1',
    'o1-mini',
    'o1-preview',
    #'gpt-4o-realtime-preview',
    #'gpt-4o-mini-realtime-preview',
    #'gpt-4o-audio-preview',
)


# acceptable models from stability ai
# see: https://platform.stability.ai/docs/api-reference
# NOTE: these models only available in text-to-image and image-to-image modes
SD_MODEL_NAMES = (
     'core',
     'sd3.5-large',
     'sd3.5-large-turbo',
     'sd3.5-medium',
     'sd3-large',
     'sd3-large-turbo',
     'sd3-medium',
     'ultra'
)


MODES = (
    'text-to-image',
    'image-to-image',
    'upscale4x_fast',
    'inpainting',
    'control_sketch',
    'control_structure',
    'control_style',
    '3d_fast',
    '3d_pointaware'
)


####### USED IN 3D #######
TEXTURE_RESOLUTIONS = (
    '512',
    '1024',
    '2048',
)

REMESH_ALGORITHMS = (
    'none',
    'quad',
    'triangle',
)

# Point Aware 3D
TARGET_TYPES = (
    'none',
    'face',
    'vertex'
)
##########################


@app.post('/diffusion')
async def diffusion_endpoint(request: DiffusionRequest):
    #logger.debug(request)

    if request.mode is None:
        return {'generation_status': 'error, mode is missing','generated_image': None}

    if request.mode.lower() not in MODES:
        return {'generation_status': f'error, mode "{request.mode}" is not supported!','generated_image': None}

    logger.debug(f'Executing task "{request.mode}"...')

    if request.mode.lower() == 'text-to-image':
        return text2image_generation(request)
    elif request.mode.lower() == 'image-to-image':
        return image2image_generation(request)
    elif request.mode.lower() == 'upscale4x_fast':
        return upscale_generation(request)
    elif request.mode.lower() == 'inpainting':
        return inpainting_generation(request)
    elif 'control' in request.mode.lower():
        # control_sketch, control_structure, control_style
        return control_generation(request)
    elif '3d' in request.mode.lower():
        # 3d_fast, 3d_pointaware
        return three_dimensional_generation(request)
    
    return {'generation_status': 'error: unknown','generated_image': None}


def text2image_generation(request):

    if request.model.lower() not in SD_MODEL_NAMES:
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
        generation_status = f'error: {str(response.json())}'
        logger.debug(generation_status)
        return {'generation_status': generation_status,'generated_image': None}

    buffer = BytesIO(response.content)
    generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
	
    return {'generation_status': generation_status,'generated_image': generated_image}


def image2image_generation(request):

    if request.model.lower() not in SD_MODEL_NAMES:
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
        generation_status = f'error: {str(response.json())}'
        logger.debug(generation_status)
        return {'generation_status': generation_status,'generated_image': None}
    
    buffer = BytesIO(response.content)
    generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
	
    return {'generation_status': generation_status,'generated_image': generated_image}


def upscale_generation(request):

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
        generation_status = f'error: {str(response.json())}'
        logger.debug(generation_status)
        return {'generation_status': generation_status,'generated_image': None}

    buffer = BytesIO(response.content)
    generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
	
    return {'generation_status': generation_status,'generated_image': generated_image}


def inpainting_generation(request):
    
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
            'seed': request.seed,
            'negative_prompt': request.negative_prompt,
            'output_format': 'jpeg',
        },
    )

    if response.status_code == 200:
        generation_status = 'ok'
    else:
        generation_status = f'error: {str(response.json())}'
        logger.debug(generation_status)
        return {'generation_status': generation_status,'generated_image': None}

    buffer = BytesIO(response.content)
    generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
	
    return {'generation_status': generation_status,'generated_image': generated_image}



def control_generation(request):

    submode = request.mode.lower().split('_')[1]

    if submode == 'sketch':
        URL = 'https://api.stability.ai/v2beta/stable-image/control/sketch'
    elif submode == 'structure':
        URL = 'https://api.stability.ai/v2beta/stable-image/control/structure'
    elif submode == 'style':
        URL = 'https://api.stability.ai/v2beta/stable-image/control/style'


    # decode base64 encoded image
    decoded_image = base64.b64decode(request.image, validate=True)
    decoded_image = BytesIO(decoded_image)

    _data = {
        'prompt': request.prompt,
        'seed': request.seed,
        'negative_prompt': request.negative_prompt,
        'control_strength': request.strength,
        'output_format': 'jpeg',
    }

    if submode == 'style':
        _data['fidelity'] = request.fidelity

    response = requests.post(
        URL,
        headers={
            'authorization': f'Bearer {STABILITY_API_KEY}',
            'accept': 'image/*'
        },
        files={'image': decoded_image},
        data=_data,
    )

    if response.status_code == 200:
        generation_status = 'ok'
    else:
        generation_status = f'error: {str(response.json())}'
        logger.debug(generation_status)
        return {'generation_status': generation_status,'generated_image': None}

    buffer = BytesIO(response.content)
    generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
	
    return {'generation_status': generation_status,'generated_image': generated_image}


def three_dimensional_generation(request):

    submode = request.mode.lower().split('_')[1]

    if submode == 'fast':
        URL = 'https://api.stability.ai/v2beta/3d/stable-fast-3d'
    elif submode == 'pointaware':
        URL = 'https://api.stability.ai/v2beta/3d/stable-point-aware-3d'


    # decode base64 encoded image
    decoded_image = base64.b64decode(request.image, validate=True)
    decoded_image = BytesIO(decoded_image)

    _data = {
        'texture_resolution': request.texture_resolution,
        'foreground_ratio': request.foreground_ratio,
        'remesh': request.remesh,
        'vertex_count': request.vertex_count,
    }

    if submode == 'pointaware':
        _data['target_type'] = request.target_type
        _data['target_count'] = request.target_count
        _data['guidance_scale'] = request.guidance_scale
        _data['seed'] = request.seed
        # ensure the correct range of foreground_ratio for point aware 3D
        if (request.foreground_ratio < 1.0) or (request.foreground_ratio > 2.0):
            logger.debug(f'WARNING: value of {_data["foreground_ratio"]} is invalid for point aware 3D, falling back to default value of 1.3')
            _data['foreground_ratio'] = 1.3

    # ensure the correct range of foreground_ratio for fast 3D
    if (request.foreground_ratio < 0.1) or (request.foreground_ratio > 1.0):
        logger.debug(f'WARNING: value of {_data["foreground_ratio"]} is invalid for fast 3D, falling back to default value of 0.85')
        _data['foreground_ratio'] = 0.85

    response = requests.post(
        URL,
        headers={
            'authorization': f'Bearer {STABILITY_API_KEY}',
        },
        files={'image': decoded_image},
        data=_data,
    )

    if response.status_code == 200:
        generation_status = 'ok'
    else:
        generation_status = f'error: {str(response.json())}'
        logger.debug(generation_status)
        return {'generation_status': generation_status,'generated_image': None}

    buffer = BytesIO(response.content)
    generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
	
    return {'generation_status': generation_status,'generated_image': generated_image}


#################################### LLM SCRIPTING AGENT ####################################
@app.post('/generate_script')
async def generate_script_endpoint(request: ScriptGenerationRequest):
    global conversation
    
    conversation.append({'role': 'user', 'content': request.prompt})
    
    # response in: chat_completion.choices[0].message.content
    # response role (assistant) in: chat_completion.choices[0].message.role 
    chat_completion = client.chat.completions.create(
        model=request.model,
        messages=conversation,
        temperature=request.temperature,
    )

    response_content = chat_completion.choices[0].message.content
    conversation.append({'role': 'assistant', 'content': response_content})

    try:
        script_from_content = response_content.split('```python')[1].split('```')[0]
    except:
        script_from_content = ''

    if len(script_from_content) < 1:
        generation_status = 'error'
    else:
        generation_status = 'ok'
    
    response = {
        'generation_status': generation_status,
        'generated_script': script_from_content,
    }
    
    return response


# NOTE: GET REQUEST, NOT POST!
@app.get('/get_conversation_history')
async def get_conversation_history_endpoint():
    global conversation
    
    response = {
        'conversation_history': conversation,
    }
    
    return response


# NOTE: GET REQUEST, NOT POST!
@app.get('/reset_conversation_history')
async def reset_conversation_history_endpoint():

    try:
        reset_conversation_history()
        reset_conversation_status = 'ok'
    except:
        reset_conversation_status = 'error'
    
    response = {
        'reset_conversation_status': reset_conversation_status
    }
    
    return response
#############################################################################################


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
    logger.debug(f'Uniform API Grasshopper Deep Learning Services (GHDLS) is starting...')
	# accept every connection (not only local connections)
    uvicorn.run(app, host='0.0.0.0', port=8000)