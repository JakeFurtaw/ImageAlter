# Image Alter
Image Alter is a Gradio-based web application that allows users to create and edit images using Stable Diffusion models. The app provides two main functionalities: Text-to-Image generation and Image-to-Image transformation.


## Features
### Text-to-Image Generation
- Create images from text prompts
- Customize generation parameters (number of images, inference steps, guidance scale)
- View generated images in a gallery
### Image-to-Image Transformation
- Edit existing images using text prompts
- Customize transformation parameters (number of images, inference steps, strength, guidance scale)
- View transformed images in a gallery


## Pictures and Videos
***COMING SOON***


## Installation
1. Clone the repository:
```commandline
git clone https://github.com/JakeFurtaw/ImageAlter.git
```
2. Install the required dependencies:
```commandline
pip install gradio torch diffusers transformers
```
3. Run the Gradio app:
```commandline
gradio imagealter.py
```
4. The app will automatically load in your browser.


## How Use the app
### In the "Text to Image" tab:
1. Enter a text prompt
2. Adjust the generation parameters
3. Click "Enter/Submit" to generate images
### In the "Image to Image" tab:
1. Upload an input image to the left input section
2. Enter a text prompt for editing
3. Adjust the transformation parameters
4. Click "Enter/Submit" to generate images


### Models Used
- Base Model: stabilityai/sdxl-turbo
- Refiner Model: stabilityai/stable-diffusion-xl-refiner-1.0


### File Structure
***imagealter.py***: Main Gradio application file
***utils.py***: Utility functions for image generation and transformation


### Contributing
Contributions are welcome! Please feel free to submit a Pull Request.


### Acknowledgements
This project uses the ***Stable Diffusion XL*** models from Stability AI.
