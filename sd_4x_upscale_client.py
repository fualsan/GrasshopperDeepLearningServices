import base64
import json
from urllib import request


IMG_URL = '/tmp/gen_img.jpg'

# encode image for sending over API
with open(IMG_URL, 'rb') as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    

# the JSON data to be sent
data = {
    'image': encoded_image,
    'prompt': 'an upscaled image',
    #'negative_prompt': 'ugly, deformed, disfigured, poor details, bad anatomy, background',
}


data = json.dumps(data).encode('utf-8')


# send request as JSON to server
req = request.Request('http://localhost:8000/upscale', data=data, headers={'Content-Type': 'application/json'})
response = request.urlopen(req)


response_data = json.loads(response.read().decode('utf-8'))
generated_image_data = response_data['generated_image']


with open('/tmp/gen_img_upscaled.jpg', 'wb') as image_file:
    # decode received base64 encoded image
    image_file.write(base64.b64decode(generated_image_data, validate=True))
