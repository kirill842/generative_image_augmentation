from diffusers import StableDiffusionPipeline
import os
import torch
import argparse

def run_inference(
    pretrained_model_name_or_path,
    lora_checkpoint_dir,
    output_dir,
    prompt,
    device,
    num_inference_steps,
    guidance_scale,
    num_images,
):
    os.makedirs(output_dir, exist_ok=True)
    _device = torch.device(device)
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)
    pipe.unet.load_attn_procs(lora_checkpoint_dir)
    pipe.to(_device)

    resume = max([-1] + [int(img_name.split(".")[0][-4:]) for img_name in os.listdir(output_dir)])
    for i in range(resume+1, num_images):
        print("Generating", i, "image")
        image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]
        image.save(f"{output_dir}_image_{i:04d}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with Stable Diffusion + LoRA checkpoint')
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='stabilityai/stable-diffusion-2',
                        help='Pretrained Stable Diffusion repo or path')
    parser.add_argument('--lora_checkpoint_dir', type=str, required=True,
                        help='Directory with LoRA attention processors')
    parser.add_argument('--output_dir', type=str, default='./inference_images/',
                        help='Where to save generated images')
    parser.add_argument('--prompt', type=str,
                        default='A photo of flying photorealistic white bird in a photorealistic environment',
                        help='Text prompt for generation')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Torch device (e.g., cuda:0 or cpu)')
    parser.add_argument('--num_inference_steps', type=int, default=30,
                        help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='Guidance scale')
    parser.add_argument('--num_images', type=int, default=10000,
                        help='Total number of images to generate')
    args = parser.parse_args()

    run_inference(
        pretrained_model_name_or_path=args.pretrained,
        lora_checkpoint_dir=args.lora_checkpoint_dir,
        output_dir=args.output_dir,
        prompt=args.prompt,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_images=args.num_images
    )
