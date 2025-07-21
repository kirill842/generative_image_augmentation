from diffusers import StableDiffusionPipeline
import torch

PATH_TO_SAVE = "./inference_images_model_v5/"
RUN = 1

pipeline = StableDiffusionPipeline.from_pretrained("./saved_model_v5", use_safetensors=True).to("cuda")

for i in range(250):
    image = pipeline(prompt="flying bird").images[0]
    image.save(PATH_TO_SAVE + "flying_bird" + "_" + str.zfill(str(RUN), 3) + "_" + str.zfill(str(i), 3) + ".png")
