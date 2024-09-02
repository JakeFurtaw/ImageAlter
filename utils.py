from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
import torch
from torch.utils.tensorboard.summary import image

"""------------Text to Image Pipeline------------"""

pipe = StableDiffusionXLPipeline.from_pretrained(
    model="stabilityai/sdxl-turbo",
    device_map="auto",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16")

prompt = ""

textToImage = pipe(prompt, num_inference_steps=70,
                   guidance_scale=0.0,
                   num_images_per_prompt=1).images[0]

"""--------Image to Image Pipeline----------"""

base = StableDiffusionXLPipeline.from_pretrained(
    model="stabilityai/sdxl-turbo",
    device_map="auto",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16")

refiner = StableDiffusionXLPipeline.from_pretrained(
    model= "stabilityai/stable-diffusion-xl-refiner-1.0",
    device_map="auto",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

init_image = load_image(image="") # original image

prompt = ""

imageToImage = base(prompt,
                    image=init_image,
                    num_inference_steps=75, # how many steps the model takes to make the image higher quality, takes longer for inference higher the number
                    strength=0.3, # more noise that gets added to the image or how much the image gets changed
                    guidance_scale=0.3,  # How close the image should be to the original
                    denoising_end=.75,
                    num_images_per_prompt=3, # how many images get generated
                    output_type="latent"
                    ).images[0]
imageToImage = refiner(prompt,
                       image=imageToImage,
                       num_inference_steps=75,
                       strength=.3,
                       guidance_scale=0.3,
                       denoising_start=.75,
                       num_images_per_prompt=3
                       ).images[0]