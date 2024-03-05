from diffusers import AutoPipelineForText2Image
import torch

from fastapi import FastAPI
from pydantic import BaseModel, Field
from io import BytesIO
import base64
import uvicorn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')


pipeline = AutoPipelineForText2Image.from_pretrained(
	'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16, variant='fp16'
).to(device)


# uvicorn runs this
app = FastAPI()


class Text2ImageRequest(BaseModel):
	prompt: str
	negative_prompt: str = Field(default=None) # TODO: max_length ?
	seed: int = Field(default=1234, gt=0, description='Seed must be greater than zero')
	guidance_scale: float = Field(default=7.5, gt=0, description='Guidence scale must be greater than zero')
	height: int = Field(default=512, gt=0, description='Height must be greater than zero')
	width: int = Field(default=512, gt=0, description='Width must be greater than zero')
	num_inference_steps: int = Field(default=50, gt=0, description='Number of inference steps scale must be greater than zero')


@app.post('/t2i')
async def process_image(request: Text2ImageRequest):

	generator = torch.Generator(device).manual_seed(request.seed)

	# TODO: multiple images maybe? 
	image = pipeline(
		request.prompt, 
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


if __name__ == '__main__':
	# optionally run from terminal: uvicorn t2i_server:app --host 0.0.0.0 --port 8000 --reload
	# accept every connection (not only local connections)
    uvicorn.run(app, host='0.0.0.0', port=8000)