import base64
import json
import urllib2


img_base64_data = 'nothing yet!'


def read_and_encode_img(img_url):
    # encode image for sending over API
    with open(img_url, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image


def send_request():
    # Read image data from files
    IMG_URL = 'PATH/TO/IMAGE'

    encoded_img = read_and_encode_img(IMG_URL)

    # Prepare the data to be sent
    data = {
        'image': encoded_img,
        'prompt': prompt,
        'negative_prompt': negative_prompt,
        'num_inference_steps': num_inference_steps,
        'strength': strength,
    }

    data = json.dumps(data).encode('utf-8')

    # send request to server
    # LOCAL
    URL = 'http://localhost:8000/i2i'
    # REMOTE (VPN)
    # URL = 'http://10.1.7.24:8000/i2i'
    req = urllib2.Request(URL, data=data, headers={'Content-Type': 'application/json'})
    response = urllib2.urlopen(req)

    # decode received image
    response_data = json.loads(response.read().decode('utf-8'))
    generated_image_data = response_data['generated_image']

    save_path = 'SAVE/PATH'
    with open(save_path + 'gen_img_i2i.jpg', 'wb') as image_file:
        image_file.write(base64.b64decode(generated_image_data))

    global img_base64_data
    img_base64_data = generated_image_data[:100]


if send:
    send_request()
