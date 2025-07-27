#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from PIL import Image
import argparse
import cv2


def transfer_and_crop_good_birds(
    images_dir, labels_dir, output_dir, crop_size, percent, stride
):
    """
    For each image in train/val/test, read YOLO annotations,
    find "good" birds (class 0 with w,h > 0.03),
    then slide a window of size crop_size and save all crops
    that fully contain at least one good bird bounding box.
    Also writes metadata.jsonl with {'file_name': ..., 'text': ...}.
    """
    # Use full step if stride not provided
    stride = stride or crop_size

    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, 'metadata.jsonl')

    with open(meta_path, 'w') as meta_file:
        for img_name in os.listdir(images_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            base_name = os.path.splitext(img_name)[0]
            lbl_path = os.path.join(labels_dir, base_name + '.txt')
            img_path = os.path.join(images_dir, img_name)

            # Skip images without labels
            if not os.path.exists(lbl_path):
                continue

            # Read annotations
            birds = []  # list of (xmin, ymin, xmax, ymax)
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls, cx, cy, w_norm, h_norm = parts
                    cls = int(cls)
                    cx, cy, w_n, h_n = map(float, [cx, cy, w_norm, h_norm])
                    # Filter good birds
                    if cls == 0 and w_n > percent and h_n > percent:
                        # Convert normalized coords to pixel coords
                        with Image.open(img_path) as img:
                            W, H = img.size
                        bw = w_n * W
                        bh = h_n * H
                        bx = cx * W
                        by = cy * H
                        xmin = bx - bw/2
                        ymin = by - bh/2
                        xmax = bx + bw/2
                        ymax = by + bh/2
                        birds.append((xmin, ymin, xmax, ymax))

            if not birds:
                continue

            # Load image once per file
            image = Image.open(img_path)
            W, H = image.size
            crop_counter = 0

            # Slide window
            for y in range(0, H - crop_size + 1, stride):
                for x in range(0, W - crop_size + 1, stride):
                    x2 = x + crop_size
                    y2 = y + crop_size
                    # Check if any bird bbox is fully inside window
                    has_good = False
                    for (xmin, ymin, xmax, ymax) in birds:
                        if xmin >= x and ymin >= y and xmax <= x2 and ymax <= y2:
                            has_good = True
                            break
                    if not has_good:
                        continue

                    # Crop and save
                    crop = image.crop((x, y, x2, y2))
                    crop_name = f"{base_name}_crop_{crop_counter:03d}.jpg"
                    crop_path = os.path.join(output_dir, crop_name)
                    crop.save(crop_path)

                    # Metadata
                    text = 'flying bird' if len(birds) == 1 else 'flying birds'
                    meta = {'file_name': crop_name, 'text': text}
                    meta_file.write(json.dumps(meta, ensure_ascii=False) + '\n')
                    crop_counter += 1

            image.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transfer and crop images of good birds from a dataset with YOLO annotations."
    )
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Directory with input images.")
    parser.add_argument("--labels_dir", type=str, required=True,
                        help="Directory with YOLO label files.")
    parser.add_argument("--output_dir", type=str, default="./dataset_for_finetuning",
                        help="Directory to save cropped images and metadata.")
    parser.add_argument("--crop_size", type=int, required=True,
                        help="Size of the square crop window in pixels.")
    parser.add_argument("--percent", type=float, default=0.015,
                        help="Minimum normalized width/height for a bird to be considered 'good'.")
    parser.add_argument("--stride", type=int, default=None,
                        help="Stride for sliding window. Defaults to crop_size if not set.")
    parser.add_argument("--text_label", type=str, required=True,
                        help="Text label for every image")
    args = parser.parse_args()

    # If stride not provided, set to crop_size
    stride = args.stride or args.crop_size

    transfer_and_crop_good_birds(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        crop_size=args.crop_size,
        percent=args.percent,
        stride=stride
    )

