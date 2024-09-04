import gradio as gr
from utils import text_to_image, image_to_image
import numpy as np

MAX_SEED = np.iinfo(np.int32).max

css = """
.gradio-container {
    background: radial-gradient(#416e8a, #000000);
}
.custom-title {
    font-size: 75px;
}
.custom-title-two {
    font-size: 60px;
    margin-top: 40px;
}
"""

with gr.Blocks(title="Image Alter", theme="default", fill_width=True, css=css) as demo:
    gr.Markdown('<h1 class="custom-title"><center>Image Alter</center></h1>')
    gr.Markdown("## <center>Image Alter is a Gradio-based web application that allows users to create and edit images "
                "using advanced state-of-the-art Stable Diffusion models.</center>")
    text_to_image_gallery = gr.State([])
    image_to_image_gallery = gr.State([])
    with gr.Tab("Text to Image"):
        with gr.Row():
            gr.Column(scale=2)
            with gr.Column(scale=8, show_progress=True, variant="compact"):
                gr.Markdown("## <center>Output Image(s)</center>")
                output_image = gr.Gallery(height="50vh",
                                          rows=[1],
                                          columns=[num_images.value],
                                          show_label=False,
                                          interactive=False,
                                          object_fit="contain",
                                          show_download_button=True,
                                          show_fullscreen_button=True)
                prompt = gr.Textbox(label="Image Prompt",
                                    placeholder="Enter image prompt...",)
                gr.Examples(examples=["Can you generate me an image of an astronaut in a blue space suit on the moon with an alien space rifle in his hands and the sun in the background?",
                                      "Can you generate me an image of a flying car that is being operated by a pink unicorn on a dark night with stars in the sky?",
                                      "Can you generate me an image of a person wearing a blue swimsuit, riding a mountain bike in a park with a dragon flying over top of them and snow mountains in the background?",
                                      "Can you generate me an image of a soldier in a orange wetsuit with a musket in their hands on a raft in the middle of the ocean?"], inputs=prompt)
                with gr.Accordion(label="Advanced Settings", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            num_images = gr.Slider(minimum=1, maximum=5, value=3, step=1,
                                                   label="Number of Images to Generate",
                                                   info="How many images you want the model to generate.",
                                                   interactive=True)
                            num_inference_steps = gr.Slider(minimum=1, maximum=124, value=4, step=1,
                                                            label="Number of Inference Steps",
                                                            info="Selected how many steps the model takes to make the image higher quality. Takes longer for inference higher you make the number.",
                                                            interactive=True)
                            guidance_scale = gr.Slider(minimum=0.0, maximum=5, value=0.0, step=0.1,
                                                       label="Guidance Scale",
                                                       info="How closely the image should follow the prompt. Higher values make the image more closely follow the prompt but will loose image quality.",
                                                       interactive=True)
                        with gr.Column(scale=1):
                            height = gr.Slider(minimum=256, maximum=2048, value=1024, step=256,
                                               label="Height",
                                               info="Height of the generated Image.",
                                               interactive=True)
                            width = gr.Slider(minimum=256, maximum=2048, value=1024, step=256,
                                              label="Width",
                                              info="Width of the generated Image.",
                                              interactive=True)
                            seed = gr.Slider(minimum=0, maximum=MAX_SEED, value=0, step=1,
                                             label="Seed",
                                             info="Select a seed for the image to be generated from. Make sure you pick a good one! If you leave seed set to 0 a random seed will be chosen.",
                                             interactive=True)
            gr.Column(scale=2)
        gr.Tabs()
        gr.Markdown('<h1 class="custom-title-two"><center>Output Image Gallery</center></h1>')
        output_gallery = gr.Gallery(height="auto",
                                    rows=[10],
                                    columns=[num_images.value],
                                    show_download_button=True,
                                    show_fullscreen_button=True,
                                    label="Output Image Gallery",
                                    show_label=False,
                                    object_fit="contain",
                                    interactive=False, )

        def update_output_image_columns(num):
            return gr.Gallery(height="50vh", rows=[1], columns=[num])
        num_images.change(
            fn=update_output_image_columns,
            inputs=[num_images],
            outputs=[output_image]
        )


        def process_text_to_image(prompt, height, width, num_images, num_inference_steps, guidance_scale, seed, gallery):
            images, _ = text_to_image(prompt, height, width, num_images, num_inference_steps, guidance_scale, seed)
            updated_gallery = images + gallery
            return images, updated_gallery, updated_gallery
        prompt.submit(
            fn=process_text_to_image,
            inputs=[prompt, height, width, num_images, num_inference_steps, guidance_scale, seed, text_to_image_gallery],
            outputs=[output_image, output_gallery, text_to_image_gallery]
        )



# ----------------------------Image to Image Code Below-----------------------------------------------------


    with gr.Tab("Image to Image"):
        with gr.Column(variant="compact"):
            with gr.Row():
                with gr.Column(scale=2, show_progress=True):
                    gr.Markdown("## <center>Input Image</center>")
                    input_image = gr.Image(height="50vh", show_label=False)
                with gr.Column(scale=4, show_progress=True):
                    gr.Markdown("## <center>Output Image(s)</center>")
                    i2i_output_image = gr.Gallery(height="50vh",
                                                  rows=[1],
                                                  columns=[i2i_num_images.value],
                                                  object_fit="contain",
                                                  show_fullscreen_button=True,
                                                  show_label=False,
                                                  interactive=False)
            with gr.Row():
                with gr.Column(scale=8, show_progress=True):
                    i2i_prompt = gr.Textbox(label="Image Prompt", placeholder="Enter image edit prompt...", autoscroll=True)
                    with gr.Accordion(label="Advanced Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                i2i_num_images = gr.Slider(minimum=1, maximum=5, value=3, step=1,
                                                       label="Number of Images to Generate",
                                                       info="How many images you want the model to generate.",
                                                       interactive=True)
                                i2i_num_inference_steps = gr.Slider(minimum=6, maximum=500, value=4, step=1,
                                                                    label="Number of Inference Steps",
                                                                    info="Selected how many steps the model takes to make the image higher quality. Takes longer for inference higher you make the number.",
                                                                    interactive=True)
                                i2i_guidance_scale = gr.Slider(minimum=0.0, maximum=5, value=0.0, step=0.1,
                                                               label="Guidance Scale",
                                                               info="How closely the image should follow the prompt. Higher values make the image more closely follow the prompt but will loose image quality.",
                                                               interactive=True)
                            with gr.Column():
                                i2i_height = gr.Slider(minimum=256, maximum=2048, value=1024, step=256,
                                                       label="Height",
                                                       info="Height of the generated Image.",
                                                       interactive=True)
                                i2i_width = gr.Slider(minimum=256, maximum=2048, value=1024, step=256,
                                                      label="Width",
                                                      info="Width of the generated Image.",
                                                      interactive=True)
                                i2i_seed = gr.Slider(minimum=0, maximum=MAX_SEED, value=0, step=1,
                                                    label="Seed",
                                                    info="Select a seed for the image to be generated from. Make sure you pick a good one! If you leave seed set to 0 a random seed will be chosen.",
                                                    interactive=True)
        gr.Tabs()
        gr.Markdown('<h1 class="custom-title-two"><center>Output Image Gallery</center></h1>')
        i2i_output_gallery = gr.Gallery(height="auto",
                                        rows=[10],
                                        columns=[i2i_num_images.value],
                                        show_download_button=True,
                                        show_fullscreen_button=True,
                                        preview=True,
                                        label="Output Image Gallery",
                                        show_label=False,
                                        object_fit="contain",
                                        interactive=False)

        def process_image_to_image(prompt, init_image, height, width, num_images, num_inference_steps, guidance_scale, seed, gallery):
            single_image, new_images, _ = image_to_image(prompt, init_image, height, width, num_images, num_inference_steps, guidance_scale, seed)
            updated_gallery = gallery + new_images
            return single_image, updated_gallery, updated_gallery

        i2i_prompt.submit(
            fn=process_image_to_image,
            inputs=[i2i_prompt, input_image, i2i_height, i2i_width, i2i_num_images, i2i_num_inference_steps, i2i_guidance_scale, i2i_seed, image_to_image_gallery],
            outputs=[i2i_output_image, i2i_output_gallery, image_to_image_gallery]
        )
demo.launch(inbrowser=True)