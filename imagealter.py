import gradio as gr

css= """
.gradio-container{
background:radial-gradient(#416e8a, #000000);
}
"""

with gr.Blocks(theme="default", fill_width=True, css=css) as demo:
    gr.Markdown("# <center>Image Alter</center>")
    gr.Markdown("<center>This app is used to edit images using Stable Diffusion.</center>")
    gr.Tabs()
    with gr.Row():
        with gr.Column(scale=4, show_progress=True):
            gr.Markdown("## <center>Input Image</center>")
            gr.Tabs()
            gr.Image(height="50vh", show_label=False)
        with gr.Column(scale=3, show_progress=True, variant="compact"):
            gr.Chatbot(height="50vh")
            gr.Textbox(placeholder="Enter image edit prompt...")
        with gr.Column(scale=4, show_progress=True):
            gr.Markdown("## <center>Output Image</center>")
            gr.Tabs()
            gr.Image(height="50vh", show_label=False, interactive=False)
    gr.Markdown("# <center>Output Image Gallery</center>")
    gr.Tabs()
    gr.Gallery(height="500",
               rows=[6],
               columns=[6],
               show_download_button=True,
               show_fullscreen_button=True,
               preview=True,
               label="Output Image Gallery",
               show_label=False,
               object_fit="scale-down")


    demo.launch(inbrowser=True)