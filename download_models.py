from diffusers import AutoPipelineForText2Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from transformers import pipeline
from controlnet_aux import PidiNetDetector, HEDdetector

import torch

import numpy as np
import cv2

from PIL import Image


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
	)


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
	)


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
	)


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
	)



def load_text2image():
	t2i_pipeline = AutoPipelineForText2Image.from_pretrained(
		'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16, variant='fp16', use_safetensors=True
	)


SUPPORTED_NAMES = {
	'depth_default': load_depth_default,
	'canny': load_canny,
	'softedge': load_softedge,
	'scribble': load_scribble,
	'text2image': load_text2image,
}


def download_all():
	
	for _name, _load_fn in SUPPORTED_NAMES.items():
		print(f'Downloading {_name}...'.upper())
		_load_fn()
		print(f'Successfully downloaded {_name}!'.upper())
		print('\n')


if __name__ == '__main__':
    download_all()