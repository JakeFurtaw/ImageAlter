import numpy as np
import torch
from diffusers import FluxPipeline
import random

MAX_SEED = np.iinfo(np.int32).max

# Load the models
base_model = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    device_map="balanced",
    torch_dtype=torch.bfloat16,
    use_safetensors=True
)

def text_to_image(prompt, num_images, num_inference_steps, guidance_scale):
    seed = random.randint(0, MAX_SEED)
    images = base_model(
        prompt,
        height=1024,
        width=1024,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        max_sequence_length=256,
        generator=torch.Generator("cuda:1").manual_seed(seed)
    ).images
    return images[0], images, [(None, f"Generated image(s) for prompt: {prompt}")]

def image_to_image(init_image, prompt, num_images, num_inference_steps, strength, guidance_scale):
    seed = random.randint(0, MAX_SEED)
    base_output = base_model(
        prompt,
        image=init_image,
        height=1024,
        width=1024,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images,
        max_sequence_length=256,
        generator=torch.Generator("cuda:1").manual_seed(seed)
    ).images



    return base_output[0], base_output, [(None, f"Generated image(s) for prompt: {prompt}")]