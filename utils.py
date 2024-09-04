import numpy as np
import torch
from PIL import Image
from diffusers import FluxPipeline, DiffusionPipeline
import random

MAX_SEED = np.iinfo(np.int32).max
TORCH_DTYPE = torch.float16
device="cuda"

#SET MODELS HERE
flux_model = "black-forest-labs/FLUX.1-schnell"
sdxl = "stabilityai/stable-diffusion-xl-refiner-1.0"

flux = FluxPipeline.from_pretrained(
    flux_model,
    device_map="balanced",
    torch_dtype=TORCH_DTYPE,
    use_safetensors=True
)
refiner = DiffusionPipeline.from_pretrained(
    sdxl,
    device_map="balanced",
    torch_dtype=TORCH_DTYPE,
    use_safetensors=True
)


def text_to_image(prompt, height, width, num_images, num_inference_steps, guidance_scale, seed):
    seed = random.randint(0, MAX_SEED) if seed == 0 else seed
    images = flux(
        prompt=prompt + "Make this image super high quality, a masterpiece, ultra-detailed, high quality photography, photo realistic, 8k, DSLR.",
        height=height,
        width=width,
        num_images_per_prompt=num_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=256,
        generator=torch.Generator("cuda:1").manual_seed(seed)
    ).images
    return images, images, [(None, f"Generated image(s) for prompt: {prompt}")]


def image_to_image(prompt, init_image, height, width, num_images, num_inference_steps, guidance_scale):
    # TODO fix image input to make input image work
    img = Image.fromarray(init_image.astype('uint8'), 'RGB').resize((height, width), Image.Resampling.LANCZOS)  # Maybe try BICUBIC or HAMMING
    seed = random.randint(0, MAX_SEED)
    i2i_output = refiner(
        prompt=prompt + "Make this image best quality, masterpiece, ultra-detailed, high quality photography, photo realistic, 8k, DSLR.",
        image=img,
        height=height,
        width=width,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, normal quality, low quality, blurry, jpeg artifacts, sketch",
        num_images_per_prompt=num_images,
        num_inference_steps=(num_inference_steps*3),  #TODO fix inference step to make it work properly
        guidance_scale=guidance_scale,  #Also called CFG
        max_sequence_length=256,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images
    return i2i_output, i2i_output, [(None, f"Generated image(s) for prompt: {prompt}")]
