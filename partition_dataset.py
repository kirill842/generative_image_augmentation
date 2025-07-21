import os
import shutil
import argparse
from math import ceil

def split_dataset(input_dir, output_dir, num_parts):
    # Get list of image files
    images = [f for f in os.listdir(input_dir)
              if os.path.isfile(os.path.join(input_dir, f))]
    total = len(images)
    if total == 0:
        print("No files found in the input directory.")
        return

    # Calculate sizes for each part
    base_size = total // num_parts
    remainder = total % num_parts

    # Create output directories and distribute files
    start = 0
    for part in range(1, num_parts + 1):
        # Determine part size (distribute remainder)
        size = base_size + (1 if part <= remainder else 0)
        end = start + size
        part_files = images[start:end]

        part_dir = os.path.join(output_dir, f"part_{part}")
        os.makedirs(part_dir, exist_ok=True)

        # Copy files
        for filename in part_files:
            src_path = os.path.join(input_dir, filename)
            dst_path = os.path.join(part_dir, filename)
            shutil.copy2(src_path, dst_path)

        print(f"Part {part}: copied {len(part_files)} files to {part_dir}")
        start = end

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description="Split image dataset into N parts of approximately equal size"
    # )
    # parser.add_argument(
    #     'input_dir', help='Path to the directory with images'
    # )
    # parser.add_argument(
    #     'output_dir', help='Path to save the image parts'
    # )
    # parser.add_argument(
    #     'num_parts', type=int, help='Number of parts to split into'
    # )
    # args = parser.parse_args()

    # split_dataset(args.input_dir, args.output_dir, args.num_parts)

    split_dataset("./[HAND FILTERED] finish_labeled/inference_images_model_v10_ckpt_612_photo_prompt/images with labels",
                  "./[HAND FILTERED] finish_labeled/inference_images_model_v10_ckpt_612_photo_prompt/images_with_labels_partioned",
                  10)
