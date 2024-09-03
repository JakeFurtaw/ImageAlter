import numpy as np
import torch
from PIL import Image
from diffusers import (
    FluxPipeline,
    StableDiffusionImg2ImgPipeline,
    DDPMScheduler,
    AutoencoderKL,
    UNet2DConditionModel)
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizerFast
)
import random

MAX_SEED = np.iinfo(np.int32).max
TORCH_DTYPE = torch.float16
device="cuda"

#SET MODELS HERE
flux_model = "black-forest-labs/FLUX.1-schnell"
sdxl = "stabilityai/stable-diffusion-xl-refiner-1.0"

#I2I Helpers
unet = UNet2DConditionModel.from_pretrained(
    sdxl,
    subfolder="unet",
    torch_dtype=TORCH_DTYPE,
)
vae = AutoencoderKL.from_pretrained(
    sdxl,
    subfolder="vae",
    torch_dtype=TORCH_DTYPE
)
tokenizer = CLIPTokenizerFast.from_pretrained(
    sdxl,
    subfolder="tokenizer_2"
)
scheduler = DDPMScheduler.from_pretrained(
    sdxl,
    subfolder="scheduler"
)
text_encoder = CLIPTextModel.from_pretrained(
    sdxl,
    subfolder="text_encoder_2",
    torch_dtype=TORCH_DTYPE
)
# image_encoder= CLIPVisionModelWithProjection.from_pretrained(
#     sdxl,
#     subfolder="image_encoder",
#     torch_dtype=TORCH_DTYPE
# )

#MODEL INSTANTIATED HERE
flux = FluxPipeline.from_pretrained(
    flux_model,
    device_map="balanced",
    torch_dtype=TORCH_DTYPE,
    use_safetensors=True
)
refiner = StableDiffusionImg2ImgPipeline.from_pretrained(
    sdxl,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    scheduler=scheduler,
    # image_encoder=image_encoder,
    torch_dtype=TORCH_DTYPE,
)


def text_to_image(prompt, height, width, num_images, num_inference_steps, guidance_scale, seed):
    #TODO add seed slider???
    images = flux(
        prompt=prompt + "Make this image super high quality, a masterpiece, ultra-detailed, high quality photography, photo realistic, 8k, DSLR.",
        height=height,
        width=width,
        num_images_per_prompt=num_images,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        max_sequence_length=256,
        generator=torch.Generator("cuda:1").manual_seed(seed=seed)
    ).images
    return images, images, [(None, f"Generated image(s) for prompt: {prompt}")]


def image_to_image(prompt, init_image, height, width, num_images, num_inference_steps, guidance_scale):
    # TODO fix image input to make input image work
    img = Image.fromarray(init_image.astype('uint8'), 'RGB').resize((height, width), Image.Resampling.LANCZOS)  # Maybe try BICUBIC or HAMMING
    seed = random.randint(0, MAX_SEED)  #TODO add seed slider???
    i2i_output = refiner(
        prompt=prompt + "Make this image best quality, masterpiece, ultra-detailed, high quality photography, photo realistic, 8k, DSLR.",
        image=img,
        height=height,
        width=width,
        negative_prompt="monochrome, lowres, bad anatomy, worst quality, normal quality, low quality, blurry, jpeg artifacts, sketch",
        num_images_per_prompt=num_images,
        num_inference_steps=num_inference_steps,  #TODO fix inference step to make it work properly
        guidance_scale=guidance_scale,  #Also called CFG
        max_sequence_length=256,
        generator=torch.Generator("cuda").manual_seed(seed)
    ).to(device).images
    return i2i_output, i2i_output, [(None, f"Generated image(s) for prompt: {prompt}")]
