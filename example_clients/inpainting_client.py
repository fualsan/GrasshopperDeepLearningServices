import base64
import json
from urllib import request

from diffusers.utils import make_image_grid
from PIL import Image, ImageDraw
import io


def make_mask(size, x1, y1, x2, y2, fill_value=200):
    """
    utility function for testing inpainting

    fill_value: should be between 0 and 255 (uint8)
    """
    custom_mask = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(custom_mask)
    draw.rectangle([x1, y1, x2, y2], fill=(fill_value, fill_value, fill_value))
    return custom_mask


x1, y1 =   0, 200
x2, y2 = 450, 600

mask_inpainting = make_mask((1024, 1024), x1, y1, x2, y2)
pil_to_bytes = io.BytesIO()
mask_inpainting.save(pil_to_bytes, format='PNG')
encoded_mask = base64.b64encode(pil_to_bytes.getvalue()).decode('utf-8')

# NOTE: first run t2i_client.py to generate folder and images
# (not required for Text2Image)
SEND_IMG_URL = '../generated_images/gen_img.jpg'

# Text2Image
#SAVE_IMG_URL = '../generated_images/gen_img.jpg'
# Image2Image
#SAVE_IMG_URL = '../generated_images/gen_img_i2i.jpg'
# Inpainting
SAVE_IMG_URL = '../generated_images/gen_img_inpainting.jpg'


# encode image for sending over API
with open(SEND_IMG_URL, 'rb') as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    

# the JSON data to be sent
# prompts are useful for generic i2i, not necessery in upscaling
data_t2i = {
    'prompt': 'A realistic photo of a traditional house with zen gardens, warm color palette, muted colors, detailed, 8k',
    'negative_prompt': 'ugly, deformed, disfigured, poor details, bad anatomy, background',
    'num_inference_steps': 50,
    'guidance_scale': 10.5,
    #'seed': 1,
}

data_i2i = {
    'image': encoded_image,
    'prompt': 'A cyberpunk house with neon lights',
    'negative_prompt': 'ugly, deformed, disfigured, poor details, bad anatomy, background',
    'num_inference_steps': 50,
    'strength': 0.75,
}

data_inpainting = {
    'image': encoded_image,
    'mask': encoded_mask,
    'prompt': 'A house with shiny windows and red lights',
    'negative_prompt': 'ugly, deformed, disfigured, poor details, bad anatomy, background',
    'num_inference_steps': 20,
    'strength': 0.75,
}


### send request as JSON to server ###
def send_request(url, json_data):
    data = json.dumps(json_data).encode('utf-8')
    req = request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    response = request.urlopen(req)
    response_data = json.loads(response.read().decode('utf-8'))
    return response_data['generated_image']


# Text2Image
#URL = 'http://localhost:8000/t2i'
# Image2Image
#URL = 'http://localhost:8000/i2i'
# Inpainting
URL = 'http://localhost:8000/inpainting'

# Text2Image
#generated_image_data = send_request(URL, data_t2i)
# Image2Image
#generated_image_data = send_request(URL, data_i2i)
# Inpainting
generated_image_data = send_request(URL, data_inpainting)


with open(SAVE_IMG_URL, 'wb') as image_file:
    # decode received base64 encoded image
    image_file.write(base64.b64decode(generated_image_data, validate=True))


inpainting_grid = make_image_grid([Image.open(SEND_IMG_URL), mask_inpainting, Image.open(SAVE_IMG_URL)], rows=1, cols=3)
inpainting_grid.save('../generated_images/gen_img_inpainting_grid.jpg')
