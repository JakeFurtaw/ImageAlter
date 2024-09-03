import numpy as np
import torch
from PIL import Image
from diffusers import FluxPipeline, DiffusionPipeline
import random

MAX_SEED = np.iinfo(np.int32).max
base_model = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    device_map="balanced",
    torch_dtype=torch.bfloat16,
    use_safetensors=True
)
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    device_map="balanced",
    torch_dtype=torch.bfloat16,
    use_safetensors=True,
)


def text_to_image(prompt, height, width, num_images, num_inference_steps, guidance_scale):
    #TODO add seed slider???
    #TODO add negative prompt
    seed = random.randint(0, MAX_SEED)
    image = base_model(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        max_sequence_length=256,
        generator=torch.Generator("cuda:1").manual_seed(seed)
    ).images
    return image, image, [(None, f"Generating image(s) for prompt: {prompt}.....")]


def image_to_image(prompt, init_image, height, width, num_images, num_inference_steps, guidance_scale):
    # TODO fix image input to make input image work
    img = Image.fromarray(init_image.astype('uint8'), 'RGB').resize((height, width), Image.Resampling.LANCZOS) # Maybe try BICUBIC or HAMMING
    #TODO add seed slider???
    seed = random.randint(0, MAX_SEED)
    base_output = refiner(
        prompt,
        image=img,
        height=height,
        width=width,
        num_images_per_prompt=num_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=256,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images
    return base_output, base_output, [(None, f"Generating image(s) for prompt: {prompt}......")]
