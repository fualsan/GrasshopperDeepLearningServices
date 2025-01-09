# Grasshopper Deep Learning Services
Deep learning models implemented as web services for integration in Rhino Grasshopper

## Demo Screenshots
### Text to Image
![demo](assets/t2i_demo_video_thumbnail.png)

### Image to Image
![demo](assets/i2i_demo_video_thumbnail.png)

### Demo Video
[Click here to download and play the demo video](./assets/demo_video.mp4)

## Examples
### Text to Image
**prompt: a traditional japanese house in a green garden**

![text to image demo image](./assets/t2i_demo_example_image.jpg)


### Image to Image
Using the image above as input, **prompt: cyberpunk building with neon lights, night**

![text to image demo image](./assets/i2i_demo_example_image.jpg)

### NEW: Inpainting

Inpainting is implemented with Stable Diffusion XL (which offers text to image and image to image in 1024x1024 resolution as well)

See [inpainting_client.py](./example_clients/inpainting_client.py) and [sd_xl_generic_server.py](./diffusion_api_service/sd_xl_generic_server.py)

# Use Locally
## Rhino Grasshopper Demo 
* Install Rhino (tested on Rhino 8 but Rhino 7 is also supported)
* Download **grasshopper_demo.gh** file to your working directory and open it with Rhino. 
* Right click Text2Img and Image2Image Grasshopper components and select "Open Script Editor". Here, edit the "save_path" for selecting image save folder and "URL" if server is running on remote machine.
* Edit the prompt (and other inputs) and click "Send Request" to send to the server. Depending on your hardware, response can take a while (GPUs are highly recommended for faster response). 

*NOTE: this demo only works in Rhino 8 (Python 3.x) environment. However, Python 2.x scripts are also provided but not fully tested. Python 2.x scripts end with **_p2.py**.*

# Use Official Docker Images

## Diffusion API

This service exposes two endpoints **/t2i** and **/i2i** for text to image and image to image. For an example, you can check out client code: [t2i_client.py](./example_clients/t2i_client.py) and [i2i_client.py](./example_clients/i2i_client.py).

[Diffusion API Docker Hub Image Page](https://hub.docker.com/repository/docker/fualsan/diffusion-api/general)


```bash
$ docker run -d --gpus all -p 8000:8000 -v diffusion_api_volume:/root/generative_app fualsan/diffusion-api
```

## Upscale Diffusion API

This service exposes two endpoints **/upscale2x** and **/upscale4x** for 2x and 4x image upscaling. For an example, you can check out client code: [i2i_client.py](./example_clients/i2i_client.py).

[Upscale Diffusion API Docker Hub Image Page](https://hub.docker.com/repository/docker/fualsan/diffusion-upscale-api/general)

```bash
$ docker run -d --gpus all -p 9000:9000 -v diffusion_upscale_api_volume:/root/generative_app fualsan/diffusion-upscale-api
```