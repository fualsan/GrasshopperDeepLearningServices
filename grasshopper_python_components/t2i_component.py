import base64
import json
from urllib import request

img_base64_data = 'nothing yet!'

def send_request():
    # Prepare the data to be sent
    data = {
        #'prompt': 'sketch portrait of a car speeding on the highway, image contains only one car and it is white',
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'num_inference_steps': num_inference_steps,
        'guidance_scale': guidance_scale,
    }

    data = json.dumps(data).encode('utf-8')

    # send request to server
    # LOCAL
    URL = 'http://localhost:8000/t2i'
    # REMOTE (VPN)
    #URL = 'http://10.1.7.24:8000/t2i'
    req = request.Request(URL, data=data, headers={'Content-Type': 'application/json'})
    response = request.urlopen(req)

    # decode received image
    response_data = json.loads(response.read().decode('utf-8'))
    generated_image_data = response_data['generated_image']


    save_path = 'C:/Users/hfuat/Documents/rhino_projects/text2image_test/'
    with open(save_path+'gen_img_t2i.jpg', 'wb') as image_file:
        image_file.write(base64.b64decode(generated_image_data, validate=True))


    img_base64_data = generated_image_data[:100]

if send:
    send_request()