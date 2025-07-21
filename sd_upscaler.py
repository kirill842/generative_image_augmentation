from diffusers import StableDiffusionUpscalePipeline
import torch
from PIL import Image
import os

SOURCE_DIR = "./images_for_upscaling"
TARGET_DIR = "./upscaled_images"
os.makedirs(TARGET_DIR, exist_ok=True)

device = "cuda:6"

pipe = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler"
).to(device)

pipe.safety_checker = None

for filename in os.listdir(SOURCE_DIR):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        continue

    lr_image = Image.open(os.path.join(SOURCE_DIR, filename)).convert("RGB")
    lr_image = lr_image.resize((512, 512), resample=Image.LANCZOS)

    hr_image = pipe(
        prompt="flying bird",
        image=lr_image,
        noise_level=0.2,
        num_inference_steps=30
    ).images[0]

    out_path = os.path.join(TARGET_DIR, filename)
    hr_image.save(out_path)
