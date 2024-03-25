from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import pipeline
from controlnet_aux import PidiNetDetector, HEDdetector

import torch

import numpy as np
import cv2

from fastapi import FastAPI
from pydantic import BaseModel, Field
from PIL import Image
from io import BytesIO
import base64
import uvicorn
import gc


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


############## TEXT 2 IMAGE ################
t2i_pipeline = AutoPipelineForText2Image.from_pretrained(
	'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16, variant='fp16', use_safetensors=True
).to(device)


t2i_pipeline.enable_model_cpu_offload()
# remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
t2i_pipeline.enable_xformers_memory_efficient_attention()
############################################


############## IMAGE 2 IMAGE ###############
SUPPORTED_NAMES = (
	'depth_default',
	'canny',
	'softedge',
	'scribble'
)

def load_depth_default():
	control_img_generator = pipeline('depth-estimation')

	def preprocess_control_img(control_img_generator, image):
		image = control_img_generator(image)['depth']
		image = np.array(image)
		image = image[:, :, None]
		image = np.concatenate([image, image, image], axis=2)
		image = Image.fromarray(image)
		# save image (OPTIONAL)
		#image.save('/tmp/control_img_controlnet.jpg')
		return image

	controlnet = ControlNetModel.from_pretrained(
		'lllyasviel/sd-controlnet-depth', 
		torch_dtype=torch.float16
	)

	i2i_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
		'runwayml/stable-diffusion-v1-5', 
		controlnet=controlnet, 
		safety_checker=None, 
		torch_dtype=torch.float16,
		variant='fp16'
	).to(device)

	i2i_pipeline.scheduler = UniPCMultistepScheduler.from_config(i2i_pipeline.scheduler.config)

	#i2i_pipeline.enable_xformers_memory_efficient_attention()
	i2i_pipeline.enable_model_cpu_offload()

	return control_img_generator, preprocess_control_img, controlnet, i2i_pipeline


def load_canny():
	# Another pipeline is not required for canny
	# However, this is kept for compatibility
	control_img_generator = None

	# TODO: take low_threshold, high_threshold from request
	def preprocess_control_img(control_img_generator, image, low_threshold=100, high_threshold=200):
		image = cv2.Canny(image, low_threshold, high_threshold)
		image = image[:, :, None]
		image = np.concatenate([image, image, image], axis=2)
		image = Image.fromarray(image)

		# save image (OPTIONAL)
		#image.save('/tmp/control_img_controlnet.jpg')
		return image

	controlnet = ControlNetModel.from_pretrained(
		'lllyasviel/sd-controlnet-canny', 
		torch_dtype=torch.float16,
		#variant='fp16',
	)

	i2i_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
		'runwayml/stable-diffusion-v1-5', 
		controlnet=controlnet, 
		safety_checker=None, 
		torch_dtype=torch.float16,
		variant='fp16',
	).to(device)

	i2i_pipeline.scheduler = UniPCMultistepScheduler.from_config(i2i_pipeline.scheduler.config)

	#i2i_pipeline.enable_xformers_memory_efficient_attention()
	i2i_pipeline.enable_model_cpu_offload()

	return control_img_generator, preprocess_control_img, controlnet, i2i_pipeline


def load_softedge():

	control_img_generator = PidiNetDetector.from_pretrained('lllyasviel/Annotators')

	# Not very useful in this case
	# However, this is kept for compatibility
	def preprocess_control_img(control_img_generator, image):
		image = control_img_generator(image, safe=True)

		# save image (OPTIONAL)
		#image.save('/tmp/control_img_controlnet.jpg')
		return image

	controlnet = ControlNetModel.from_pretrained(
		'lllyasviel/control_v11p_sd15_softedge',
		torch_dtype=torch.float16
	)

	i2i_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
		'runwayml/stable-diffusion-v1-5', 
		controlnet=controlnet, 
		torch_dtype=torch.float16
	).to(device)

	i2i_pipeline.scheduler = UniPCMultistepScheduler.from_config(i2i_pipeline.scheduler.config)

	#i2i_pipeline.enable_xformers_memory_efficient_attention()
	i2i_pipeline.enable_model_cpu_offload()

	return control_img_generator, preprocess_control_img, controlnet, i2i_pipeline


def load_scribble():

	control_img_generator = HEDdetector.from_pretrained('lllyasviel/ControlNet')

	# Not very useful in this case
	# However, this is kept for compatibility
	def preprocess_control_img(control_img_generator, image):
		image = control_img_generator(image, scribble=True)

		# save image (OPTIONAL)
		#image.save('/tmp/control_img_controlnet.jpg')
		return image

	controlnet = ControlNetModel.from_pretrained(
		'lllyasviel/sd-controlnet-scribble',
		torch_dtype=torch.float16
	)

	i2i_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
		'runwayml/stable-diffusion-v1-5', 
		controlnet=controlnet, 
		torch_dtype=torch.float16
	).to(device)

	i2i_pipeline.scheduler = UniPCMultistepScheduler.from_config(i2i_pipeline.scheduler.config)

	#i2i_pipeline.enable_xformers_memory_efficient_attention()
	i2i_pipeline.enable_model_cpu_offload()

	return control_img_generator, preprocess_control_img, controlnet, i2i_pipeline
############################################


############### LOAD DEFAULT ###############
# This is the defaul model/pipeline loaded when service starts
control_img_generator, preprocess_control_img, controlnet, i2i_pipeline = load_depth_default()
############################################


# uvicorn runs this
app = FastAPI()


class Text2ImageRequest(BaseModel):
	prompt: str
	negative_prompt: str = Field(default=None) # TODO: max_length ?
	seed: int = Field(default=1234, gt=0, description='Seed must be greater than zero')
	guidance_scale: float = Field(default=7.5, ge=0.0, description='Guidence scale must be greater than or equal to zero')
	height: int = Field(default=512, gt=0, description='Height must be greater than zero')
	width: int = Field(default=512, gt=0, description='Width must be greater than zero')
	num_inference_steps: int = Field(default=50, gt=0, description='Number of inference steps scale must be greater than zero')


class Image2ImageRequest(BaseModel):
	image: str # base64 encoded image as string
	prompt: str
	negative_prompt: str = Field(default=None) # TODO: max_length ?
	seed: int = Field(default=1234, gt=0, description='Seed must be greater than zero')
	guidance_scale: float = Field(default=7.5, ge=0.0, description='Guidence scale must be greater than or equal to zero')
	# a lower strength value means the generated image is more similar to the initial image
	strength: float = Field(default=0.8, ge=0.0, description='Strength scale must be greater than or equal to zero')
	height: int = Field(default=512, gt=0, description='Height must be greater than zero')
	width: int = Field(default=512, gt=0, description='Width must be greater than zero')
	num_inference_steps: int = Field(default=50, gt=0, description='Number of inference steps scale must be greater than zero')


class LoadRequest(BaseModel):
	name: str # name of the model/pipeline
	reset: bool = Field(default=False) # removes previously loaded


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
	#image.save('/tmp/gen_img.jpg')

	buffer = BytesIO()
	image.save(buffer, format='JPEG')
	generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

	return {'generated_image': generated_image}


@app.post('/i2i')
async def image2image_generation(request: Image2ImageRequest):

	generator = torch.Generator(device).manual_seed(request.seed)

	# decode base64 encoded low res image
	decoded_image = base64.b64decode(request.image, validate=True)
	decoded_image = Image.open(BytesIO(decoded_image))#.convert('RGB')

	# TODO: multiple images maybe? 
	depth_image = preprocess_control_img(control_img_generator, decoded_image)

	image = i2i_pipeline(
		image=depth_image,
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
	#image.save('/tmp/gen_img.jpg')

	buffer = BytesIO()
	image.save(buffer, format='JPEG')
	generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

	return {'generated_image': generated_image}


@app.post('/load')
async def load_model(request: LoadRequest):
	global control_img_generator, preprocess_control_img, controlnet, i2i_pipeline

	_name = request.name

	if _name not in SUPPORTED_NAMES:
		_supported_names_str = ', '.join(SUPPORTED_NAMES) 
		return {'result': f'{_name} is not supported! Please select from: {_supported_names_str}'}

	# OPTIONAL
	if request.reset:
		reset_models()

	if _name == 'depth_default':
		control_img_generator, preprocess_control_img, controlnet, i2i_pipeline = load_depth_default()
	elif _name == 'canny':
		control_img_generator, preprocess_control_img, controlnet, i2i_pipeline = load_canny()
	elif _name == 'softedge':
		control_img_generator, preprocess_control_img, controlnet, i2i_pipeline = load_softedge()
	elif _name == 'scribble':
		control_img_generator, preprocess_control_img, controlnet, i2i_pipeline = load_scribble()

	return {'result': f'{_name} loaded succesfully!'}


@app.post('/reset_all')
async def reset_models():
	global control_img_generator, i2i_pipeline, t2i_pipeline

	if control_img_generator is not None:
		del control_img_generator
		control_img_generator = None
	else:
		print('Depth estimator already removed from memory!')

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


if __name__ == '__main__':
	# optionally run from terminal: uvicorn t2i_server:app --host 0.0.0.0 --port 8000 --reload
	# accept every connection (not only local connections)
    uvicorn.run(app, host='0.0.0.0', port=8000)