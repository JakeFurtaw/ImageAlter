import numpy as np
import torch
from PIL import Image
from diffusers import FluxPipeline, DiffusionPipeline, AutoencoderKL
from transformers import CLIPTokenizer, CLIPTextModel
import random

MAX_SEED = np.iinfo(np.int32).max
TORCH_DTYPE = torch.float16

#SET MODELS HERE
flux_schnell = "black-forest-labs/FLUX.1-schnell"
flux_dev = "black-forest-labs/FLUX.1-dev"
flux_dev_shakker_labs = "Shakker-Labs/AWPortrait-FL"

flux = FluxPipeline.from_pretrained(
    flux_dev,
    device_map="balanced",
    torch_dtype=TORCH_DTYPE,
)


def text_to_image(prompt, height, width, num_images, num_inference_steps, guidance_scale, seed):
    torch.cuda.empty_cache()
    seed = random.randint(0, MAX_SEED) if seed == 0 else seed
    images = flux(
        prompt=prompt + " Make this image super high quality, a masterpiece, ultra-detailed, high quality photography, photo realistic, 8k, DSLR.",
        height=height,
        width=width,
        num_images_per_prompt=num_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=256,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images
    return images, images


def image_to_image(prompt, init_image, height, width, num_images, num_inference_steps, guidance_scale, seed):
    torch.cuda.empty_cache()
    # TODO fix image input to make input image work
    input_img = Image.fromarray(init_image.astype('uint8'), 'RGB').resize((height, width), Image.Resampling.LANCZOS)  # Maybe try BICUBIC or HAMMING
    seed = random.randint(0, MAX_SEED) if seed == 0 else seed
    altered_image = flux(
        prompt=prompt + "Make this image best quality, masterpiece, ultra-detailed, high quality photography, photo realistic, 8k, DSLR.",
        image=input_img,
        height=height,
        width=width,
        num_images_per_prompt=num_images,
        num_inference_steps=num_inference_steps,  #TODO fix inference step to make it work properly
        guidance_scale=guidance_scale,  #Also called CFG
        max_sequence_length=256,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images
    return altered_image, altered_image
