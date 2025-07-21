import os
import shutil
import json
from tqdm import tqdm
from PIL import Image
import cv2

def transfer_good_images_to_dataset_folder_and_make_metadata_file(percent=0.015):
    sub_folders = ["train", "val", "test"]
    images_path = "./dataset/images"
    labels_path = "./dataset/labels"

    transfer_path = "./dataset_for_finetuning_v7_no_crop_p_0015"
    os.makedirs(transfer_path, exist_ok=True)

    meta_file = open(transfer_path + "/" + "metadata.jsonl", 'w')
    for sub_folder in sub_folders:
        _images_path = images_path + "/" + sub_folder
        _labels_path = labels_path + "/" + sub_folder
        for image_name in os.listdir(_images_path):
            label_name = image_name[:-4] + ".txt"

            assert image_name[:-4] == label_name[:-4]

            # if we don't have annotation for an image -> going to the next image
            if not os.path.exists(_labels_path + "/" + label_name):
                continue

            label_file = open(_labels_path + "/" + label_name, "r")

            # counting birds and good birds in the image
            n_birds = 0
            n_good_birds = 0
            for obj_annot in label_file.readlines():
                if len(obj_annot.split()) != 5:
                    continue
                cls, x1, y1, w, h = map(float, obj_annot.split())
                cls = int(cls)
                if cls == 0:
                    n_birds += 1
                # if bird occupies more than 3% of an image it is good bird
                if cls == 0 and w >= percent and h >= percent:
                    n_good_birds += 1

            if n_good_birds >= 1:
                if n_good_birds == 1:
                    text = "flying bird"
                elif n_good_birds > 1:
                    text = "flying birds"
                meta_info = {"file_name": image_name, "text": text}
                meta_file.write(json.dumps(meta_info) + "\n")
                shutil.copy(_images_path + "/" + image_name, transfer_path + "/" + image_name)

                label_file.close()

    meta_file.close()


def transfer_good_images_to_dataset_folder_and_make_metadata_file_with_rectangles(percent=0.015):
    sub_folders = ["train", "val", "test"]
    images_path = "./dataset/images"
    labels_path = "./dataset/labels"

    transfer_path = "./dataset_for_finetuning_v7_no_crop_p_0015_with_rectangles"
    os.makedirs(transfer_path, exist_ok=True)

    # Открываем JSONL файл для записи метаданных
    with open(os.path.join(transfer_path, "metadata.jsonl"), 'w') as meta_file:
        for sub_folder in sub_folders:
            img_folder = os.path.join(images_path, sub_folder)
            lbl_folder = os.path.join(labels_path, sub_folder)

            for image_name in os.listdir(img_folder):
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                label_name = os.path.splitext(image_name)[0] + ".txt"
                label_path = os.path.join(lbl_folder, label_name)

                # пропускаем, если разметки нет
                if not os.path.exists(label_path):
                    continue

                # Загружаем изображение
                img_path = os.path.join(img_folder, image_name)
                image = cv2.imread(img_path)
                if image is None:
                    print(f"Не удалось загрузить изображение: {img_path}")
                    continue
                height, width = image.shape[:2]

                n_good_birds = 0

                # Читаем аннотации и рисуем прямоугольники
                with open(label_path, 'r') as lf:
                    for line in lf:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls_id, x_center, y_center, w_norm, h_norm = parts
                        cls_id = int(cls_id)
                        x_center, y_center, w_norm, h_norm = map(float, (x_center, y_center, w_norm, h_norm))

                        if cls_id != 0:
                            continue

                        # рассчитываем координаты бокса в пикселях
                        box_w = w_norm * width
                        box_h = h_norm * height
                        x1 = int((x_center * width) - box_w / 2)
                        y1 = int((y_center * height) - box_h / 2)
                        x2 = int(x1 + box_w)
                        y2 = int(y1 + box_h)

                        # считаем хорошие птицы по порогу площади
                        if w_norm >= percent and h_norm >= percent:
                            n_good_birds += 1
                            # рисуем зеленый прямоугольник для хорошей птицы
                            color = (0, 255, 0)
                        else:
                            # рисуем красный прямоугольник для остальных
                            color = (0, 0, 255)

                        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Если есть хотя бы одна хорошая птица, сохраняем и копируем
                if n_good_birds >= 1:
                    # сохраняем аннотированное изображение в папку transfer
                    save_path = os.path.join(transfer_path, image_name)
                    cv2.imwrite(save_path, image)

                    # формируем описание для metadata
                    text = "flying bird" if n_good_birds == 1 else "flying birds"
                    meta_info = {"file_name": image_name, "text": text}
                    meta_file.write(json.dumps(meta_info, ensure_ascii=False) + "\n")

    print("Готово! Изображения с аннотацией сохранены, metadata.jsonl обновлён.")


def transfer_and_crop_good_birds(
    images_dir='./dataset/images',
    labels_dir='./dataset/labels',
    output_dir='./dataset_for_finetuning_v7_cropped',
    crop_size=768,
    stride=None
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

    sub_folders = ['train', 'val', 'test']
    os.makedirs(output_dir, exist_ok=True)
    meta_path = os.path.join(output_dir, 'metadata.jsonl')

    with open(meta_path, 'w') as meta_file:
        for split in sub_folders:
            img_folder = os.path.join(images_dir, split)
            lbl_folder = os.path.join(labels_dir, split)
            if not os.path.isdir(img_folder) or not os.path.isdir(lbl_folder):
                continue

            for img_name in os.listdir(img_folder):
                if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                base_name = os.path.splitext(img_name)[0]
                lbl_path = os.path.join(lbl_folder, base_name + '.txt')
                img_path = os.path.join(img_folder, img_name)

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
                        if cls == 0 and w_n > 0.03 and h_n > 0.03:
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

def transfer_and_crop_good_birds_2(
    images_dir="./dataset_for_finetuning_v7_no_crop_p_0015",
    labels_dir='./dataset/labels',
    output_dir='./dataset_for_finetuning_v8',
    crop_size=768,
    percent=0.015,
    stride=None
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

def fix_metafile(dataset_dir: str):
    meta_path = os.path.join(dataset_dir, 'metadata.jsonl')
    with open(meta_path, 'w') as meta_file:
        for img_name in os.listdir(dataset_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            meta = {'file_name': img_name, 'text': 'A photo of flying photorealistic bird in a photorealistic environment: in a city or in an urban area or in a forest or in a field or in the mountains'}
            meta_file.write(json.dumps(meta, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # transfer_good_images_to_dataset_folder_and_make_metadata_file()
    # transfer_good_images_to_dataset_folder_and_make_metadata_file_with_rectangles()
    # transfer_and_crop_good_birds(stride=250)
    # transfer_and_crop_good_birds_2(stride=250)
    fix_metafile('./dataset_for_finetuning_v9')
    pass
