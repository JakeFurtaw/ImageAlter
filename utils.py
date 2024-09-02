import numpy as np
import torch
from diffusers import FluxPipeline, DiffusionPipeline
import random

MAX_SEED = np.iinfo(np.int32).max

# Load the models
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
    use_safetensors=True
)

def text_to_image(prompt, height, width, num_images, num_inference_steps, guidance_scale):
    seed = random.randint(0, MAX_SEED)
    image = base_model(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        max_sequence_length=256,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images
    return image[0], image, [(None, f"Generated image(s) for prompt: {prompt}")]

def image_to_image( prompt, init_image, height, width, num_images, num_inference_steps, guidance_scale):
    seed = random.randint(0, MAX_SEED)
    base_output = refiner(
        prompt,
        image=init_image,
        height=height,
        width=width,
        num_images_per_prompt=num_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=256,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images
    return base_output[0], base_output, [(None, f"Generated image(s) for prompt: {prompt}")]