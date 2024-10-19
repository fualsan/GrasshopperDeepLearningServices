from diffusers import StableDiffusionLatentUpscalePipeline
import torch

from fastapi import FastAPI
from pydantic import BaseModel, Field
from PIL import Image
from io import BytesIO
import base64
import uvicorn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


pipeline = StableDiffusionLatentUpscalePipeline.from_pretrained(
	'stabilityai/sd-x2-latent-upscaler', use_safetensors=True
).to(device)


pipeline.enable_model_cpu_offload()
pipeline.enable_xformers_memory_efficient_attention()


# uvicorn runs this
app = FastAPI()


class UpscaleRequest(BaseModel):
	image: str # base64 encoded image as string
	prompt: str = Field(default='')
	negative_prompt: str = Field(default=None) # TODO: max_length ?
	seed: int = Field(default=1234, gt=0, description='Seed must be greater than zero')
	# NOTE: guidance_scale is set to 0.0 for pure upscaling
	guidance_scale: float = Field(default=0.0, ge=0, description='Guidence scale must be greater than or equal to zero')
	num_inference_steps: int = Field(default=50, gt=0, description='Number of inference steps scale must be greater than zero')


@app.post('/upscale2x')
async def process_image(request: UpscaleRequest):

	generator = torch.Generator(device).manual_seed(request.seed)

	# decode base64 encoded low res image
	decoded_image = base64.b64decode(request.image, validate=True)
	decoded_image = Image.open(BytesIO(decoded_image))

	# TODO: multiple images maybe? 
	image = pipeline(
		image=decoded_image,
		prompt=request.prompt, 
		negative_prompt=request.negative_prompt, 
		generator=generator, 
		guidance_scale=request.guidance_scale,
		num_inference_steps=request.num_inference_steps
	).images[0]

	# save image (OPTIONAL)
	#image.save('/tmp/gen_img.jpg')

	buffer = BytesIO()
	image.save(buffer, format='JPEG')
	generated_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

	return {'generated_image': generated_image}


if __name__ == '__main__':
	# accept every connection (not only local connections)
    uvicorn.run(app, host='0.0.0.0', port=9000)