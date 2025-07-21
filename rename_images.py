import os
import argparse


def rename_dataset(files_dir, new_prefix, start_index=0, padding=4):
    """
    Rename files in a directory to a new prefix with zero-padded indices.

    Args:
        files_dir (str): Path to the directory containing files to rename.
        new_prefix (str): New prefix for renamed files.
        start_index (int): Starting index for numbering (default: 0).
        padding (int): Number of digits for zero-padding (default: 4).
    """
    # List and sort files
    files = sorted([f for f in os.listdir(files_dir) if os.path.isfile(os.path.join(files_dir, f))])
    for idx, original_name in enumerate(files, start=start_index):
        # Extract extension
        name, ext = os.path.splitext(original_name)
        # Build new filename
        new_name = f"{new_prefix}_{idx:0{padding}d}{ext}"
        # Rename
        src = os.path.join(files_dir, original_name)
        dst = os.path.join(files_dir, new_name)
        print(f"Renaming '{src}' to '{dst}'")
        os.rename(src, dst)


def main():
    # parser = argparse.ArgumentParser(description="Rename paired image and label files to a custom naming scheme.")
    # parser.add_argument('--images_dir', required=True, help='Path to the images directory')
    # parser.add_argument('--labels_dir', required=True, help='Path to the labels directory')
    # parser.add_argument('--prefix', required=True, help='New prefix for renamed files')
    # parser.add_argument('--start', type=int, default=0, help='Starting index for numbering')
    # parser.add_argument('--padding', type=int, default=4, help='Zero-padding width for indices')
    # args = parser.parse_args()

    # # Rename images and labels
    # rename_dataset(args.images_dir, args.prefix, args.start, args.padding)
    # rename_dataset(args.labels_dir, args.prefix, args.start, args.padding)

    rename_dataset("./finish_labeled/inference_images_model_v10_ckpt_1258_photo_brown_prompt_4/images",
                   "model_v10_ckpt_1258_brown_2", 0, 4)
    rename_dataset("./finish_labeled/inference_images_model_v10_ckpt_1258_photo_brown_prompt_4/labels",
                   "model_v10_ckpt_1258_brown_2", 0, 4)


if __name__ == '__main__':
    main()
