from diffusers import AutoPipelineForText2Image
from diffusers import AutoPipelineForImage2Image
from diffusers import AutoPipelineForInpainting

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
logger = logging.getLogger('sd_xl_generic_server')
logger.setLevel(logging.DEBUG)

# print to a log file
file_handler = logging.handlers.RotatingFileHandler(
    filename='sd_xl_generic_server.log',
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


############## TEXT 2 IMAGE ################
print('LOADING TXT2IMG MODEL...')

t2i_pipeline = AutoPipelineForText2Image.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0', 
    torch_dtype=torch.float16, 
    variant='fp16', 
    use_safetensors=True
).to('cuda')

t2i_pipeline.enable_model_cpu_offload()
t2i_pipeline.enable_xformers_memory_efficient_attention()

print('TXT2IMG MODEL LOADED SUCCESSFULLY!')
############################################


############## IMAGE 2 IMAGE ###############
print('LOADING IMG2IMG MODEL...')

i2i_pipeline = AutoPipelineForImage2Image.from_pipe(
    t2i_pipeline
).to('cuda')

print('IMG2IMG MODEL LOADED SUCCESSFULLY!')
############################################


############### INPAINTING ################
print('LOADING INPAINTING MODEL...')

inpainting_pipeline = AutoPipelineForInpainting.from_pipe(
    t2i_pipeline
).to('cuda')

print('IMG2IMG MODEL LOADED SUCCESSFULLY!')
############################################


# uvicorn runs this
app = FastAPI()


class Text2ImageRequest(BaseModel):
	prompt: str
	negative_prompt: str = Field(default=None) # TODO: max_length ?
	seed: int = Field(default=1234, gt=0, description='Seed must be greater than zero')
	guidance_scale: float = Field(default=10.5, ge=0.0, description='Guidence scale must be greater than or equal to zero')
	height: int = Field(default=1024, gt=0, description='Height must be greater than zero')
	width: int = Field(default=1024, gt=0, description='Width must be greater than zero')
	num_inference_steps: int = Field(default=50, gt=0, description='Number of inference steps scale must be greater than zero')


class Image2ImageRequest(BaseModel):
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


class InpaintingRequest(BaseModel):
    image: str # base64 encoded image as string
    mask: str # base64 encoded mask image as string
    prompt: str
    negative_prompt: str = Field(default=None) # TODO: max_length ?
    seed: int = Field(default=1234, gt=0, description='Seed must be greater than zero')
    guidance_scale: float = Field(default=12.5, ge=0.0, description='Guidence scale must be greater than or equal to zero')
    # a lower strength value means the generated image is more similar to the initial image
    strength: float = Field(default=0.8, ge=0.0, description='Strength scale must be greater than or equal to zero')
    height: int = Field(default=1024, gt=0, description='Height must be greater than zero')
    width: int = Field(default=1024, gt=0, description='Width must be greater than zero')
    num_inference_steps: int = Field(default=50, gt=0, description='Number of inference steps scale must be greater than zero')


@app.post('/t2i')
async def text2image_generation(request: Text2ImageRequest):

	generator = torch.Generator(device).manual_seed(request.seed)

	# TODO: multiple images maybe? 
	image = t2i_pipeline(
		prompt=request.prompt, 
		negative_prompt=request.negative_prompt, 
		generator=generator, 
		guidance_scale=request.guidance_scale, 
		height=request.height, 
		width=request.height,
		num_inference_steps=request.num_inference_steps
	).images[0]

	# save image (OPTIONAL)
	#image.save(f'./generated_images/gen_img_{time_ns()}.jpg')

	buffer = BytesIO()
	image.save(buffer, format='JPEG')
	generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

	return {'generated_image': generated_image}


@app.post('/i2i')
async def image2image_generation(request: Image2ImageRequest):

	generator = torch.Generator(device).manual_seed(request.seed)

	# decode base64 encoded image
	decoded_image = base64.b64decode(request.image, validate=True)
	decoded_image = Image.open(BytesIO(decoded_image))#.convert('RGB')

	image = i2i_pipeline(
		image=decoded_image,
		prompt=request.prompt, 
		negative_prompt=request.negative_prompt, 
		generator=generator, 
		guidance_scale=request.guidance_scale, 
		strength=request.strength,
		height=request.height, 
		width=request.height,
		num_inference_steps=request.num_inference_steps
	).images[0]
	
	# save image (OPTIONAL)
	#image.save(f'./generated_images/gen_img_{time_ns()}.jpg')

	buffer = BytesIO()
	image.save(buffer, format='JPEG')
	generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

	return {'generated_image': generated_image}


@app.post('/inpainting')
async def image2image_generation(request: InpaintingRequest):

	generator = torch.Generator(device).manual_seed(request.seed)

	# decode base64 encoded image
	decoded_image = base64.b64decode(request.image, validate=True)
	decoded_image = Image.open(BytesIO(decoded_image))#.convert('RGB')

    # decode base64 encoded image
	decoded_mask = base64.b64decode(request.mask, validate=True)
	decoded_mask = Image.open(BytesIO(decoded_mask))#.convert('RGB')

	# TODO: multiple images maybe? 
	image = inpainting_pipeline(
		image=decoded_image,
		prompt=request.prompt, 
        mask_image=decoded_mask,
		negative_prompt=request.negative_prompt, 
		generator=generator, 
		guidance_scale=request.guidance_scale, 
		strength=request.strength,
		height=request.height, 
		width=request.height,
		num_inference_steps=request.num_inference_steps
	).images[0]
	
	# save image (OPTIONAL)
	#image.save(f'./generated_images/gen_img_{time_ns()}.jpg')

	buffer = BytesIO()
	image.save(buffer, format='JPEG')
	generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

	return {'generated_image': generated_image}

    
@app.post('/reset_all')
async def reset_models(request: Request):
	global i2i_pipeline, t2i_pipeline, inpainting_pipeline

	if inpainting_pipeline is not None:
		del inpainting_pipeline
		inpainting_pipeline = None
	else:
		print('Inpainting already removed from memory!')

	if t2i_pipeline is not None:
		del t2i_pipeline
		t2i_pipeline = None
	else:
		print('Text to image already removed from memory!')

	if i2i_pipeline is not None:
		del i2i_pipeline
		i2i_pipeline = None
	else:
		print('Image to image already removed from memory!')

	gc.collect()
	torch.cuda.empty_cache()

	return {'result': 'all removed from memory!'}


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