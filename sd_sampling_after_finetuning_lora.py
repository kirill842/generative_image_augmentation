from diffusers import StableDiffusionPipeline
import os
import torch

model_path = "./saved_model_v10/checkpoint-1258"
PATH_TO_SAVE = "./inference_images_model_v10_ckpt_1258_photo_white_prompt_3/"
os.makedirs(PATH_TO_SAVE, exist_ok=True)

device = torch.device("cuda:5")

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
pipe.unet.load_attn_procs(model_path)
pipe.to(device)

# ###
# def count_params(model):
#     return sum(p.numel() for p in model.parameters())
#
# # пример для UNet, VAE и текстового кодера в вашем pipeline:
# unet_params = count_params(pipe.unet)
# vae_params  = count_params(pipe.vae)
# text_params = count_params(pipe.text_encoder)
#
# total_params = unet_params + vae_params + text_params
#
# print(f"UNet:  {unet_params/1e6:.1f}M параметров")
# print(f"VAE:   {vae_params/1e6:.1f}M параметров")
# print(f"Text:  {text_params/1e6:.1f}M параметров")
# print(f"Всего: {total_params/1e6:.1f}M параметров")
#
# bytes_per_param = 2  # для float16
# mem_bytes = total_params * bytes_per_param
# print(f"≈{mem_bytes/1024**3:.2f} GiB для весов (только параметры)")
# ###

# resume = int(os.listdir(PATH_TO_SAVE)[-1].split(".")[0][-4:]) if len(os.listdir(PATH_TO_SAVE)) > 0 else -1
resume = max([-1] + [int(img_name.split(".")[0][-4:]) for img_name in os.listdir(PATH_TO_SAVE)])
prompt = "A photo of flying photorealistic white bird in a photorealistic environment: in a city or in an urban area or in a forest or in a field or in the mountains"
for i in range(resume+1, 10000):
    print("Generating", i, "image")
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save(f"{PATH_TO_SAVE}_image_{i:04d}.png")
