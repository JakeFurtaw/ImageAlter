import gradio as gr

css= """
.gradio-container{
background:radial-gradient(#416e8a, #000000);
}
"""

with gr.Blocks(theme="default", fill_width=True, css=css) as demo:
    gr.Markdown("# <center>Image Alter</center>")
    gr.Markdown("### <center>This app is used to create and edit images using Stable Diffusion.</center>")
    with gr.Tab("Text to Image"):
        with gr.Row():
            with gr.Column(scale=2, show_progress=True, variant="compact"):
                num_images = gr.Slider(minimum=1, maximum=5, value=1, step=1,
                                       label="Number of Images to Generate",
                                       info="How many images you want the model to generate.",
                                       interactive=True)
                num_inference_steps=gr.Slider(minimum=1, maximum=100, value=75, step=1,
                                              label="Number of Inference Steps",
                                              info="Selected how many steps the model takes to make the image higher "
                                                   "quality. Takes longer for inference higher you make the number number",
                                              interactive=True)
                strength=gr.Slider(minimum=0, maximum=1, value=.75, step=.01,
                                   label="Strength",
                                   info="How much noise gets add to the photo or how much the photo changes.",
                                   interactive=True)
                guidance_scale=gr.Slider(minimum=0, maximum=1, value=.3, step=.01,
                                         label="Guidance Scale",
                                         info="How close the image should be to the original.",
                                         interactive=True)
                gr.Chatbot(height="50vh", show_label=False)
                prompt = gr.Textbox(label="Image Prompt", placeholder="Enter image edit prompt...")
            with gr.Column(scale=4, show_progress=True):
                gr.Markdown("## <center>Output Image</center>")
                gr.Image(height="70vh", show_label=False, interactive=False)
        gr.Markdown("# <center>Output Image Gallery</center>")
        gr.Tabs()
        gr.Gallery(height="500",
                   rows=[6],
                   columns=[3],
                   show_download_button=True,
                   show_fullscreen_button=True,
                   preview=True,
                   label="Output Image Gallery",
                   show_label=False,
                   object_fit="contain")
    with gr.Tab("Image to Image"):
        with gr.Row():
            with gr.Column(scale=4, show_progress=True):
                gr.Markdown("## <center>Input Image</center>")
                gr.Tabs()
                gr.Image(height="50vh", show_label=False)
            with gr.Column(scale=3, show_progress=True, variant="compact"):
                t2i_num_images=gr.Slider(minimum=1, maximum=5, value=1, step=1,
                                         label="Number of Images to Generate",
                                         info="How many images you want the model to generate.",
                                         interactive=True)
                t2i_num_inference_steps=gr.Slider(minimum=1, maximum=100, value=75, step=1,
                                                  label="Number of Inference Steps",
                                                  info="Selected how many steps the model takes to make the image higher"
                                                       " quality.Takes longer for inference higher you make the number"
                                                       " number",
                                                  interactive=True)
                t2i_strength=gr.Slider(minimum=0, maximum=1, value=.75, step=.01,
                                       label="Strength",
                                       info="How much noise gets add to the photo or how much the photo changes.",
                                       interactive=True)
                t2i_guidance_scale=gr.Slider(minimum=0, maximum=1, value=.3, step=.01,
                                             label="Guidance Steps",
                                             info="How much noise gets add to the photo or how much the photo changes.",
                                             interactive=True)
                gr.Chatbot(height="30vh", show_label=False)
                t2i_prompt = gr.Textbox(label="Image Prompt",placeholder="Enter image edit prompt...")
            with gr.Column(scale=4, show_progress=True):
                gr.Markdown("## <center>Output Image</center>")
                gr.Tabs()
                gr.Image(height="50vh", show_label=False, interactive=False)
        gr.Markdown("# <center>Output Image Gallery</center>")
        gr.Tabs()
        gr.Gallery(height="30vh",
                   rows=[6],
                   columns=[3],
                   show_download_button=True,
                   show_fullscreen_button=True,
                   preview=True,
                   label="Output Image Gallery",
                   show_label=False,
                   object_fit="contain")


    demo.launch(inbrowser=True)