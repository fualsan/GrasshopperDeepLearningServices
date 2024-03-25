from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import pipeline

import torch

import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel, Field
from PIL import Image
from io import BytesIO
import base64
import uvicorn


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
depth_estimator = pipeline('depth-estimation')

def preprocess_image_depth(depth_estimator, image):
	image = depth_estimator(image)['depth']
	image = np.array(image)
	image = image[:, :, None]
	image = np.concatenate([image, image, image], axis=2)
	image = Image.fromarray(image)
	# save image (OPTIONAL)
	#image.save('/tmp/control_img_controlnet_depth.jpg')
	return image

controlnet = ControlNetModel.from_pretrained(
    'lllyasviel/sd-controlnet-depth', 
    torch_dtype=torch.float16
)

i2i_pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    'runwayml/stable-diffusion-v1-5', 
    controlnet=controlnet, 
    safety_checker=None, 
    torch_dtype=torch.float16
).to(device)

i2i_pipeline.scheduler = UniPCMultistepScheduler.from_config(i2i_pipeline.scheduler.config)

#i2i_pipeline.enable_xformers_memory_efficient_attention()
i2i_pipeline.enable_model_cpu_offload()
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


@app.post('/t2i')
async def process_image(request: Text2ImageRequest):

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
async def process_image(request: Image2ImageRequest):

	generator = torch.Generator(device).manual_seed(request.seed)

	# decode base64 encoded low res image
	decoded_image = base64.b64decode(request.image, validate=True)
	decoded_image = Image.open(BytesIO(decoded_image))#.convert('RGB')

	# TODO: multiple images maybe? 
	depth_image = preprocess_image_depth(depth_estimator, decoded_image)

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


if __name__ == '__main__':
	# optionally run from terminal: uvicorn t2i_server:app --host 0.0.0.0 --port 8000 --reload
	# accept every connection (not only local connections)
    uvicorn.run(app, host='0.0.0.0', port=8000)