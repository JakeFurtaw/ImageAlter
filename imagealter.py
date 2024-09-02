import gradio as gr
from utils import text_to_image, image_to_image

css = """
.gradio-container {
    background: radial-gradient(#416e8a, #000000);
}
"""


def update_chatbot(history, user_message):
    history.append((user_message, None))
    return history


with gr.Blocks(title="Image Alter",theme="default", fill_width=True, css=css) as demo:
    gr.Markdown("# <center>Image Alter</center>")
    gr.Markdown("### <center>This app is used to create and edit images using Stable Diffusion.</center>")

    # State variables to store accumulated images
    text_to_image_gallery = gr.State([])
    image_to_image_gallery = gr.State([])

    with gr.Tab("Text to Image"):
        with gr.Row():
            with gr.Column(scale=100, variant="compact"):
                chatbot = gr.Chatbot(height="30vh",
                                     show_label=False,
                                     show_copy_button=True)
                prompt = gr.Textbox(label="Image Prompt",
                                    placeholder="Enter image prompt...")
                with gr.Accordion(label="Advanced Settings"):
                    num_images = gr.Slider(minimum=1, maximum=5, value=3, step=1,
                                           label="Number of Images to Generate",
                                           info="How many images you want the model to generate.",
                                           interactive=True)
                    num_inference_steps = gr.Slider(minimum=1, maximum=24, value=4, step=1,
                                                    label="Number of Inference Steps",
                                                    info="Selected how many steps the model takes to make the image higher quality. Takes longer for inference higher you make the number",
                                                    interactive=True)
                    guidance_scale = gr.Slider(minimum=0.0, maximum=5, value=0.0, step=0.1,
                                               label="Guidance Scale",
                                               info="How closely the image should follow the prompt. Higher values make the image more closely follow the prompt.",
                                               interactive=True)
                    height = gr.Slider(minimum=256, maximum=2048, value=1024, step=256,
                                       label="Height",
                                       info="Height of the generated Image",
                                       interactive=True)
                    width = gr.Slider(minimum=256, maximum=2048, value=1024, step=256,
                                      label="Height",
                                      info="Height of the generated Image",
                                      interactive=True)
            with gr.Column(scale=126, show_progress=True):
                gr.Markdown("## <center>Output Image(s)</center>")
                output_image = gr.Gallery(height="auto",
                                          rows=[1],
                                          columns=[num_images.value],
                                          show_label=False,
                                          interactive=False,
                                          object_fit="contain",
                                          show_download_button=True,
                                          show_fullscreen_button=True)

        gr.Markdown("# <center>Output Image Gallery</center>")
        output_gallery = gr.Gallery(height="auto",
                                    rows=[6],
                                    columns=[num_images.value],
                                    show_download_button=True,
                                    show_fullscreen_button=True,
                                    label="Output Image Gallery",
                                    show_label=False,
                                    object_fit="contain",
                                    interactive=False)


        def process_text_to_image(prompt, height, width, num_images, num_inference_steps, guidance_scale, history, gallery):
            single_image, new_images, new_history = text_to_image(prompt, height, width, num_images, num_inference_steps, guidance_scale)
            updated_history = history + new_history
            updated_gallery = gallery + new_images
            return single_image, updated_gallery, updated_history, updated_gallery


        prompt.submit(
            fn=process_text_to_image,
            inputs=[prompt, height, width, num_images, num_inference_steps, guidance_scale, chatbot, text_to_image_gallery],
            outputs=[output_image, output_gallery, chatbot, text_to_image_gallery]
        ).then(
            fn=update_chatbot,
            inputs=[chatbot, prompt],
            outputs=[chatbot]
        )

    with gr.Tab("Image to Image"):
        with gr.Row():
            with gr.Column(scale=4, show_progress=True):
                gr.Markdown("## <center>Input Image</center>")
                input_image = gr.Image(height="50vh", show_label=False)

            with gr.Column(scale=3, show_progress=True, variant="compact"):
                i2i_chatbot = gr.Chatbot(height="25.5vh", show_label=False)
                i2i_prompt = gr.Textbox(label="Image Prompt", placeholder="Enter image edit prompt...", autoscroll=True)
                with gr.Accordion(label="Advanced Settings"):
                    i2i_num_images = gr.Slider(minimum=1, maximum=5, value=3, step=1,
                                           label="Number of Images to Generate",
                                           info="How many images you want the model to generate.",
                                           interactive=True)
                    i2i_num_inference_steps = gr.Slider(minimum=1, maximum=24, value=4, step=1,
                                                    label="Number of Inference Steps",
                                                    info="Selected how many steps the model takes to make the image higher quality. Takes longer for inference higher you make the number",
                                                    interactive=True)
                    i2i_guidance_scale = gr.Slider(minimum=0.0, maximum=5, value=0.0, step=0.1,
                                               label="Guidance Scale",
                                               info="How closely the image should follow the prompt. Higher values make the image more closely follow the prompt.",
                                               interactive=True)
                    i2i_height = gr.Slider(minimum=256, maximum=2048, value=1024, step=256,
                                       label="Height",
                                       info="Height of the generated Image",
                                       interactive=True)
                    i2i_width = gr.Slider(minimum=256, maximum=2048, value=1024, step=256,
                                      label="Height",
                                      info="Height of the generated Image",
                                      interactive=True)
            with gr.Column(scale=4, show_progress=True):
                gr.Markdown("## <center>Output Image</center>")
                i2i_output_image = gr.Image(height="50vh", show_label=False, interactive=False)

        gr.Markdown("# <center>Output Image Gallery</center>")
        i2i_output_gallery = gr.Gallery(height="auto",
                                        rows=[6],
                                        columns=[3],
                                        show_download_button=True,
                                        show_fullscreen_button=True,
                                        preview=True,
                                        label="Output Image Gallery",
                                        show_label=False,
                                        object_fit="contain",
                                        interactive=False)


        def process_image_to_image(prompt, init_image, height, width, num_images, num_inference_steps, guidance_scale,
                                   history, gallery):
            single_image, new_images, new_history = image_to_image(prompt, height, width, init_image, num_images, num_inference_steps, guidance_scale)
            updated_history = history + new_history
            updated_gallery = gallery + new_images
            return single_image, updated_gallery, updated_history, updated_gallery


        i2i_prompt.submit(
            fn=process_image_to_image,
            inputs=[i2i_prompt, input_image, i2i_height, i2i_width, i2i_num_images, i2i_num_inference_steps, i2i_guidance_scale,
                    i2i_chatbot, image_to_image_gallery],
            outputs=[i2i_output_image, i2i_output_gallery, i2i_chatbot, image_to_image_gallery]
        ).then(
            fn=update_chatbot,
            inputs=[i2i_chatbot, i2i_prompt],
            outputs=[i2i_chatbot]
        )

demo.launch(inbrowser=True)