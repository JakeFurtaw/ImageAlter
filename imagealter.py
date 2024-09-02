import gradio as gr
from utils import text_to_image, image_to_image

css = """
.gradio-container {
    background: radial-gradient(#416e8a, #000000);
}
"""


def update_chatbot(history, message):
    history.append((None, message))
    return history


with gr.Blocks(theme="default", fill_width=True, css=css) as demo:
    gr.Markdown("# <center>Image Alter</center>")
    gr.Markdown("### <center>This app is used to create and edit images using Stable Diffusion.</center>")

    with gr.Tab("Text to Image"):
        with gr.Row():
            with gr.Column(scale=2, show_progress=True, variant="compact"):
                num_images = gr.Slider(minimum=1, maximum=5, value=3, step=1,
                                       label="Number of Images to Generate",
                                       info="How many images you want the model to generate.",
                                       interactive=True)
                num_inference_steps = gr.Slider(minimum=1, maximum=100, value=75, step=1,
                                                label="Number of Inference Steps",
                                                info="Selected how many steps the model takes to make the image higher quality. Takes longer for inference higher you make the number",
                                                interactive=True)
                strength = gr.Slider(minimum=0, maximum=1, value=.75, step=.01,
                                     label="Strength",
                                     info="How much noise gets add to the photo or how much the photo changes.",
                                     interactive=True)
                guidance_scale = gr.Slider(minimum=0, maximum=30, value=7.5, step=0.1,
                                           label="Guidance Scale",
                                           info="How closely the image should follow the prompt. Higher values make the image more closely follow the prompt.",
                                           interactive=True)
                chatbot = gr.Chatbot(height="44.5vh", show_label=False)
                prompt = gr.Textbox(label="Image Prompt", placeholder="Enter image prompt...")

            with gr.Column(scale=4, show_progress=True):
                gr.Markdown("## <center>Output Image</center>")
                output_image = gr.Image(height="70vh",
                                        show_label=False,
                                        interactive=False,
                                        show_download_button=True,
                                        show_fullscreen_button=True)

        gr.Markdown("# <center>Output Image Gallery</center>")
        output_gallery = gr.Gallery(height="500",
                                    rows=[6],
                                    columns=[3],
                                    show_download_button=True,
                                    show_fullscreen_button=True,
                                    preview=True,
                                    label="Output Image Gallery",
                                    show_label=False,
                                    object_fit="contain",
                                    interactive=False)

        prompt.submit(
            fn=text_to_image,
            inputs=[prompt, num_images, num_inference_steps, strength, guidance_scale],
            outputs=[output_image, output_gallery, chatbot],
            postprocess=False
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
                t2i_num_images = gr.Slider(minimum=1, maximum=5, value=1, step=1,
                                           label="Number of Images to Generate",
                                           info="How many images you want the model to generate.",
                                           interactive=True)
                t2i_num_inference_steps = gr.Slider(minimum=1, maximum=100, value=75, step=1,
                                                    label="Number of Inference Steps",
                                                    info="Selected how many steps the model takes to make the image higher quality. Takes longer for inference higher you make the number",
                                                    interactive=True)
                t2i_strength = gr.Slider(minimum=0, maximum=1, value=.75, step=.01,
                                         label="Strength",
                                         info="How much noise gets add to the photo or how much the photo changes.",
                                         interactive=True)
                t2i_guidance_scale = gr.Slider(minimum=0, maximum=30, value=7.5, step=0.1,
                                               label="Guidance Scale",
                                               info="How closely the image should follow the prompt. Higher values make the image more closely follow the prompt.",
                                               interactive=True)
                t2i_chatbot = gr.Chatbot(height="25.5vh", show_label=False)
                t2i_prompt = gr.Textbox(label="Image Prompt", placeholder="Enter image edit prompt...", autoscroll=True)

            with gr.Column(scale=4, show_progress=True):
                gr.Markdown("## <center>Output Image</center>")
                t2i_output_image = gr.Image(height="50vh", show_label=False, interactive=False)

        gr.Markdown("# <center>Output Image Gallery</center>")
        t2i_output_gallery = gr.Gallery(height="30vh",
                                        rows=[6],
                                        columns=[3],
                                        show_download_button=True,
                                        show_fullscreen_button=True,
                                        preview=True,
                                        label="Output Image Gallery",
                                        show_label=False,
                                        object_fit="contain",
                                        interactive=False)

        t2i_prompt.submit(
            fn=image_to_image,
            inputs=[input_image, t2i_prompt, t2i_num_images, t2i_num_inference_steps, t2i_strength, t2i_guidance_scale],
            outputs=[t2i_output_image, t2i_output_gallery, t2i_chatbot],
            postprocess=False
        ).then(
            fn=update_chatbot,
            inputs=[t2i_chatbot, t2i_prompt],
            outputs=[t2i_chatbot]
        )

demo.launch(inbrowser=True)