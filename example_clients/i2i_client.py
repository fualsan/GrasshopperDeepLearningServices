import base64
import json
from urllib import request
import time


# NOTE: first run t2i_client.py to generate folder and images


IMG_URL = '../generated_images/gen_img.jpg'
#IMG_URL = '../generated_images/sketch.jpg'


# encode image for sending over API
with open(IMG_URL, 'rb') as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    

# the JSON data to be sent
# prompts are useful for generic i2i, not necessery in upscaling
data = {
    'image': encoded_image,
    #'prompt': 'an upscaled image',
    #'prompt': 'A modern building',
    'prompt': 'A red truck speeding on the highway',
    #'negative_prompt': 'ugly, deformed, disfigured, poor details, bad anatomy, background',
    'num_inference_steps': 50,
    #'strength': 0.1,
    'model': 'sd3-medium',
    #'model': 'ultra',
}


data = json.dumps(data).encode('utf-8')


# send request as JSON to server
# Generic Image2Image
URL = 'http://localhost:8000/i2i'
# 2x upscale
#URL = 'http://localhost:8000/upscale2x'
# 4x upscale (consumes more VRAM)
#URL = 'http://localhost:8000/upscale4x'
# Control (stablity AI)
#URL = 'http://localhost:8000/control'


start_time = time.time()
req = request.Request(URL, data=data, headers={'Content-Type': 'application/json'})
response = request.urlopen(req)
print(f'elapsed: {time.time()-start_time:.4f}')


response_data = json.loads(response.read().decode('utf-8'))
print('Generation status:', response_data.get('generation_status'))

generated_image_data = response_data['generated_image']


with open('../generated_images/gen_img_i2i.jpg', 'wb') as image_file:
    # decode received base64 encoded image
    image_file.write(base64.b64decode(generated_image_data, validate=True))
