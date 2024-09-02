from diffusers import StableDiffusionPipeline
from diffusers.utils import load_image
import torch

"""------------Text to Image Pipeline------------"""

pipe = StableDiffusionPipeline.from_pretrained(
    model="stabilityai/sdxl-turbo",
    device_map="auto",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16")

prompt = ""

textToImage = pipe(prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

"""--------Image to Image Pipeline----------"""

pipe = StableDiffusionPipeline.from_pretrained(
    model="stabilityai/sdxl-turbo",
    device_map="auto",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16")
init_image = load_image(image="")

prompt = ""

imageToImage = pipe(prompt, image=init_image, num_inference_steps=2, strength=0.5, guidance_scale=0.0).images[0]