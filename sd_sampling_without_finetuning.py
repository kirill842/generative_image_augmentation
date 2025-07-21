from diffusers import StableDiffusionPipeline
import os
import torch


PATH_TO_SAVE = "./inference_images_stable-diffusion-2_no_finetuning/"
os.makedirs(PATH_TO_SAVE, exist_ok=True)

device = torch.device("cuda:0")

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
pipe = pipe.to(device)


prompt = "flying bird"
for i in range(250):
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(f"{PATH_TO_SAVE}flying_bird_{i:04d}.png")
