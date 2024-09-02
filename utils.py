import torch
from diffusers import StableDiffusionXLPipeline

# Load the models
base_model = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    device_map="balanced",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

refiner_model = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    device_map="balanced",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16"
)

def text_to_image(prompt, num_images, num_inference_steps, strength, guidance_scale):
    images = base_model(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        num_images_per_prompt=num_images
    ).images
    return images[0], images, f"Generated image(s) for prompt: {prompt}"

def image_to_image(init_image, prompt, num_images, num_inference_steps, strength, guidance_scale):
    base_output = base_model(
        prompt,
        image=init_image,
        num_inference_steps=num_inference_steps,
        strength=strength,
        guidance_scale=guidance_scale,
        denoising_end=0.75,
        num_images_per_prompt=num_images,
        output_type="latent"
    ).images

    refined_images = refiner_model(
        prompt,
        image=base_output,
        num_inference_steps=num_inference_steps,
        strength=strength,
        guidance_scale=guidance_scale,
        denoising_start=0.75,
        num_images_per_prompt=num_images
    ).images

    return refined_images[0], refined_images, f"Generated image(s) for prompt: {prompt}"