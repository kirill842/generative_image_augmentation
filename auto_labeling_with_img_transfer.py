import os
import cv2
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tqdm import tqdm
import argparse

# Обход всех изображений
def run_detection(
    model_path,
    image_dir,
    label_out_dir,
    image_out_dir,
    conf_threshold,
    slice_size,
    overlap_ratio,
    device,
    object_class_id,
    output_class_id
):
    # Создаем выходные папки
    os.makedirs(label_out_dir, exist_ok=True)
    os.makedirs(image_out_dir, exist_ok=True)

    # Инициализация SAHI-модели
    n_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=conf_threshold,
        device=device
    )

    for img_name in tqdm(os.listdir(image_dir)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(image_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Не удалось загрузить {img_path}")
            continue
        h, w = image.shape[:2]

        # Слайсинг-инференс
        result = get_sliced_prediction(
            image,
            n_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio
        )

        # Сборка аннотаций
        annotations = []
        for obj in result.object_prediction_list:
            if obj.category.id != object_class_id:
                continue
            xmin, ymin = obj.bbox.minx, obj.bbox.miny
            xmax, ymax = obj.bbox.maxx, obj.bbox.maxy
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            width    = (xmax - xmin) / w
            height   = (ymax - ymin) / h
            annotations.append((output_class_id, x_center, y_center, width, height))

        # Запись разметки и сохранение картинки
        if annotations:
            # сохраняем текстовый файл разметки
            txt_path = os.path.join(label_out_dir, os.path.splitext(img_name)[0] + '.txt')
            with open(txt_path, 'w') as f:
                for cid, xc, yc, bw, bh in annotations:
                    f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            # сохраняем изображение в выходную папку
            out_path = os.path.join(image_out_dir, img_name)
            cv2.imwrite(out_path, image)
            # print(f"[INFO] {img_name}: сохранено изображение и {len(annotations)} аннотаций")

    print("Генерация разметки и сохранение изображений завершены — пустые файлы не создаются.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run YOLOv8 + SAHI sliced inference and save YOLO-format labels')
    parser.add_argument('--model_path', type=str, default='yolov8l.pt', help='Path to YOLOv8 model weights')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with input images')
    parser.add_argument('--label_out_dir', type=str, required=True, help='Directory to save label txt files')
    parser.add_argument('--image_out_dir', type=str, required=True, help='Directory to save annotated images')
    parser.add_argument('--conf_threshold', type=float, default=0.7, help='Confidence threshold for detection')
    parser.add_argument('--slice_size', type=int, default=768, help='Size of slice for SAHI inference')
    parser.add_argument('--overlap_ratio', type=float, default=0.2, help='Overlap ratio for slicing')
    parser.add_argument('--device', type=str, default='cuda:0', help='Torch device for inference')
    parser.add_argument('--object_class_id', type=int, default=14, help='14 is original bird class ID in YOLO')
    parser.add_argument('--output_class_id', type=int, default=0, help='Class ID to write in output labels')
    args = parser.parse_args()

    run_detection(
        model_path=args.model_path,
        image_dir=args.image_dir,
        label_out_dir=args.label_out_dir,
        image_out_dir=args.image_out_dir,
        conf_threshold=args.conf_threshold,
        slice_size=args.slice_size,
        overlap_ratio=args.overlap_ratio,
        device=args.device,
        object_class_id=args.object_class_id,
        output_class_id=args.output_class_id
    )
