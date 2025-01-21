import base64
import json
from urllib import request
import time
from PIL import Image, ImageDraw
import io
import os


os.makedirs('../generated_images/', exist_ok=True)


# NOTE: first run text-to-image mode to generate folder and images


IMG_URL = '../generated_images/gen_img_text-to-image.jpg'
#IMG_URL = '../generated_images/sketch.jpg'


try:
    # encode image for sending over API
    with open(IMG_URL, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
except:
    encoded_image = None
    print(f'WARNING: cannot read file {IMG_URL}, run text-to-image first to generate an image')


######################### INPAINTING #########################
def make_mask(size, x1, y1, x2, y2, fill_value=255):
    """
    utility function for testing inpainting

    fill_value: should be between 0 and 255 (uint8)
    """
    custom_mask = Image.new('RGB', size, (0, 0, 0))
    draw = ImageDraw.Draw(custom_mask)
    draw.rectangle([x1, y1, x2, y2], fill=(fill_value, fill_value, fill_value))
    return custom_mask


x1, y1 =   0, 200
x2, y2 = 650, 800

mask_inpainting = make_mask((1024, 1024), x1, y1, x2, y2)
pil_to_bytes = io.BytesIO()
mask_inpainting.save(pil_to_bytes, format='PNG')
encoded_mask = base64.b64encode(pil_to_bytes.getvalue()).decode('utf-8')
##############################################################


#MODE = 'text-to-image'
#MODE = 'image-to-image'
#MODE = 'upscale4x_fast'
#MODE = 'inpainting'
#MODE = 'control_sketch'
#MODE = 'control_structure'
#MODE = 'control_style'
MODE = '3d_fast'
#MODE = '3d_pointaware'


# the JSON data to be sent
data = {
    'mode': MODE,
    'image': encoded_image,
    'mask': encoded_mask,
    'prompt': 'A red sports car speeding on the highway',
    #'prompt': 'A red car speeding on the highway',
    #'prompt': 'Cyberpunk neon lights',
    #'negative_prompt': 'ugly, deformed, disfigured, poor details, bad anatomy, background',
    #'strength': 0.95,
    'model': 'sd3-medium', # required for initial text-to-image run
    #'model': 'ultra',
    # BELOW ARE ONLY USED IN 3D
    'texture_resolution': '1024',
    'foreground_ratio': 1.3,
    'remesh': 'none',
    'vertex_count': -1,
    'target_type': 'none',
    'target_count': '1000',
}


data = json.dumps(data).encode('utf-8')


# send request as JSON to server
# Generic Stability AI
URL = 'http://localhost:8000/diffusion'
# OpenAI LLM Scripting Agent
#URL = 'http://localhost:8000/control'


print(f'Running request for "{MODE}" mode')
start_time = time.time()
req = request.Request(URL, data=data, headers={'Content-Type': 'application/json'})
response = request.urlopen(req)
print(f'elapsed: {time.time()-start_time:.4f}')


response_data = json.loads(response.read().decode('utf-8'))
print('Generation status:', response_data.get('generation_status'))

generated_image_data = response_data.get('generated_image')

if generated_image_data is not None:
    
    if MODE == '3d_fast' or MODE == '3d_pointaware':
        save_url = f'../generated_images/gen_statue_{MODE}.glb'
    else:
        save_url = f'../generated_images/gen_img_{MODE}.jpg'

    with open(save_url, 'wb') as image_file:
        # decode received base64 encoded image
        image_file.write(base64.b64decode(generated_image_data, validate=True))

else:
    print('generated_image_data:', generated_image_data)