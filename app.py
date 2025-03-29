import gradio as gr
from gradio_imageslider import ImageSlider
import torch
from depth_net import DepthNet
import matplotlib
import numpy as np

model = DepthNet()
model.load_state_dict(torch.load("weights/best_depth_model.pth",weights_only=True,map_location='cpu'))
model.eval()

def predict_depth(image):
    return model.infer_image(image)

with gr.Blocks() as demo:
    gr.Markdown("### Depth Prediction demo")
    
    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output',type='numpy', position=0.5)
    submit = gr.Button(value="Compute Depth")
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    def on_submit(image):
        original_image = image.copy()

        depth = predict_depth(torch.tensor(image))
        
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.numpy(force=True).astype(np.uint8)
        colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
        
        return (original_image, colored_depth)
    
    submit.click(on_submit, inputs=input_image, outputs=depth_image_slider)
    
demo.launch()