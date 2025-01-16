import base64
import json
from urllib import request
import os
import time


os.makedirs('../generated_images/', exist_ok=True)


# the JSON data to be sent
data = {
    'prompt': 'A car speeding on the highway, image contains only one ca and it is blue',
    'negative_prompt': 'ugly, deformed, disfigured, poor details, bad anatomy, background',
    'height': 512,
    'width': 512,
    #'num_inference_steps': 20,
    'model': 'sd3-medium',
    #'model': 'ultra',
}


data = json.dumps(data).encode('utf-8')


start_time = time.time()
# send request as JSON to server
req = request.Request('http://localhost:8000/t2i', data=data, headers={'Content-Type': 'application/json'})
response = request.urlopen(req)
print(f'elapsed: {time.time()-start_time:.4f}')


response_data = json.loads(response.read().decode('utf-8'))
print('Generation status:', response_data.get('generation_status'))

generated_image_data = response_data['generated_image']


with open('../generated_images/gen_img.jpg', 'wb') as image_file:
    # decode received base64 encoded image
    image_file.write(base64.b64decode(generated_image_data, validate=True))
