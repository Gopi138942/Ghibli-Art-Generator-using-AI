
# Ghibli Art Generator using AI

This project utilizes a pre-trained diffusion model to generate Studio Ghibli-style artwork from text prompts. By leveraging **Stable Diffusion** and a **fine-tuned model**, the goal is to bring the magical and serene world of Studio Ghibli to life through AI.

## Features:
- **Text-to-Art Generation**: Create Ghibli-inspired images from custom text prompts.
- **Pre-trained Model**: Use a fine-tuned Ghibli model (`nitrosocke/Ghibli-Diffusion`).
- **Web Interface**: Easily generate artwork using Gradio or Streamlit.

## Getting Started

### Requirements:
- Python 3.8+
- CUDA-enabled GPU (Recommended for fast performance)
- Stable Diffusion pre-trained model

---

### 📓 **Kaggle Notebook Summary**


# Ghibli Art Generator using AI

In this notebook, we will generate **Studio Ghibli-style art** using an AI model powered by **Stable Diffusion**. This project demonstrates the magic of text-to-image generation, allowing users to create whimsical Ghibli-inspired visuals from custom prompts.

## Overview
This project leverages a pre-trained diffusion model, **Ghibli-Diffusion**, and fine-tunes it to produce artwork in the unique, magical style of Studio Ghibli.

### Libraries Used:
- `diffusers`: For Stable Diffusion models.
- `torch`: PyTorch for model processing.
- `gradio`: For creating a simple web interface to interact with the model.

## Step 1: Load the Pre-trained Model
First, we load the fine-tuned model designed specifically to produce Ghibli-style artwork.
#python
from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "nitrosocke/Ghibli-Diffusion",
    torch_dtype=torch.float16,
    revision="fp16"
).to("cuda")

### Installation:
follow this link to analyse
https://huggingface.co/spaces/gopi135942/AI_art_generator

# Set up Python environment
python -m venv ghibli-env
source ghibli-env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio
pip install diffusers transformers accelerate gradio
