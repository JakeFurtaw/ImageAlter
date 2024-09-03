# Image Alter

Image Alter is a Gradio-based web application that allows users to create and edit images using advanced AI models. 
The app provides two main functionalities: Text-to-Image generation and Image-to-Image transformation, powered by 
state-of-the-art Stable Diffusion models.

## Features

### Text-to-Image Generation
- Create images from text prompts
- Customize generation parameters:
  - Number of images (1-5)
  - Number of inference steps (1-24)
  - Guidance scale (0.0-5.0)
  - Image height and width (256-2048 pixels)
- View generated images in an interactive gallery
- Accumulate generated images in an output gallery

### Image-to-Image Transformation (Work In Progress)
- Edit existing images using text prompts
- Customize transformation parameters (same as Text-to-Image)
- View transformed images in an interactive gallery
- Accumulate transformed images in an output gallery

### User Interface
- Tabbed interface for easy navigation between Text-to-Image and Image-to-Image modes
- Chatbot-style interaction for prompts and responses
- Advanced settings accordion for fine-tuning generation parameters
- Responsive image galleries with download and fullscreen options

## Installation

1. Clone the repository:
```
git clone https://github.com/JakeFurtaw/ImageAlter.git
```

2. Install the required dependencies:
```
pip install gradio torch diffusers transformers
```

3. Run the Gradio app:
```
gradio imagealter.py
```

4. The app will automatically open in your default web browser.

## How to Use the App

### Text to Image
1. Enter a text prompt in the "Image Prompt" field
2. Adjust the generation parameters in the "Advanced Settings" accordion (optional)
3. Press Enter or click Submit to generate images
4. View the generated images in the output gallery and accumulated gallery

### Image to Image
1. Upload an input image to the left input section
2. Enter a text prompt for editing in the "Image Prompt" field
3. Adjust the transformation parameters in the "Advanced Settings" accordion (optional)
4. Press Enter or click Submit to generate edited images
5. View the transformed images in the output gallery and accumulated gallery

## Models Used

- Base Model for Text-to-Image: FluxPipeline ("black-forest-labs/FLUX.1-schnell")
- Refiner Model for Image-to-Image: DiffusionPipeline ("stabilityai/stable-diffusion-xl-refiner-1.0")

## File Structure

- `imagealter.py`: Main Gradio application file
- `utils.py`: Utility functions for image generation and transformation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

This project uses the FLUX.1-schnell model from Black Forest Labs and the Stable Diffusion XL Refiner model 
from Stability AI.

