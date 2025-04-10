import gradio as gr
from diffusers import StableDiffusionPipeline
import torch

# Use a smaller model for CPU
model_id = "Ojimi/anime-kawai-diffusion"  # Lightweight anime model
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")  # Explicitly set to CPU

def generate_art(prompt):
    with torch.inference_mode():
        image = pipe(prompt, num_inference_steps=25).images[0]
    return image

with gr.Blocks() as demo:
    gr.Markdown("# 🎨 AI Art Generator (CPU)")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Enter your prompt")
            generate_btn = gr.Button("Generate")
        with gr.Column():
            output = gr.Image(label="Generated Art")
    
    generate_btn.click(fn=generate_art, inputs=prompt, outputs=output)

demo.launch()