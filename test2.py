import os


PATH = "./inference_images_model_v10_ckpt_1445_photo_prompt"
int(img_name.split(".")[0][-4:])
resume = max([-1] + [int(img_name.split(".")[0][-4:]) for img_name in os.listdir(PATH)])
print(resume+1)
