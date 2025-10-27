import gradio as gr
import cv2
from colorize import colorize_image
import numpy as np
from PIL import Image
import io
import base64

def process_image(image):
    if image is None:
        return None, None
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    colorized = colorize_image(image_bgr)
    colorized_rgb = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)
    return image, colorized_rgb

def get_download_link(image_np):
    if image_np is None:
        return None
    img = Image.fromarray(image_np)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    b64 = base64.b64encode(byte_im).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="colorized.png">⬇️ Download Colorized Image</a>'
    return href

custom_css = """
body, .gradio-container {background-color: #0d0d0d !important; color: #e0e0e0 !important;}
h1, h2, h3 {color: #00bcd4 !important;}
a {color: #00e5ff !important;}
#slider-container {position: relative; width: 100%; max-width: 700px; overflow: hidden; border-radius: 20px; margin: 0 auto; box-shadow: 0 0 20px #000;}
#slider-container img {width: 100%; display: block;}
#slider {position: absolute; width: 100%; top: 0; left: 0;}
#slider input {width: 100%; cursor: ew-resize; opacity: 0; height: 100%;}
.before {position: absolute; top: 0; left: 0; width: 50%; overflow: hidden;}
.before img {width: 200%; transform: translateX(-25%);}
"""

with gr.Blocks(css=custom_css, theme="gradio/monochrome") as demo:
    gr.HTML("""
    <h1 style="text-align:center;">🎨 AI Image Colorizer</h1>
    <p style="text-align:center;">Bring your black & white images to life with AI</p>
    <div id="slider-container">
        <div class="before">
            <img src="assets/demo_gray.jpg" alt="Before">
        </div>
        <img src="assets/demo_color.png" alt="After">
        <input type="range" min="0" max="100" value="50" oninput="this.previousElementSibling.style.width=this.value+'%'">
    </div>
    <br>
    """)

    with gr.Row():
        input_image = gr.Image(type="numpy", label="Upload Grayscale Image")
        output_image = gr.Image(label="Colorized Output")
    
    with gr.Row():
        toggle = gr.Button("👁️ Toggle Preview")
        download_html = gr.HTML("")

    def toggle_preview(input_img, output_img, state):
        if input_img is None or output_img is None:
            return None, None, state
        return (output_img if not state else input_img), (input_img if not state else output_img), not state

    preview_state = gr.State(False)
    
    toggle.click(toggle_preview, [input_image, output_image, preview_state], [input_image, output_image, preview_state])
    btn = gr.Button("🚀 Colorize")
    btn.click(process_image, inputs=[input_image], outputs=[input_image, output_image])
    output_image.change(lambda img: get_download_link(img), inputs=[output_image], outputs=[download_html])

    gr.HTML("<p style='text-align:center;'>Made with ❤️ using OpenCV + Gradio</p>")

if __name__ == "__main__":
    demo.launch()
