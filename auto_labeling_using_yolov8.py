import os
import cv2
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Параметры
MODEL_PATH       = 'yolov8l.pt'
IMAGE_DIR        = './[FILTERED] inference_images_model_v7_ckpt_2997_run_2'
LABEL_DIR        = './[FILTERED] inference_images_model_v7_ckpt_2997_run_2/labels'
CONF_THRESHOLD   = 0.7                    # порог доверия :contentReference[oaicite:2]{index=2}
SLICE_SIZE       = 768
OVERLAP_RATIO    = 0.2
DEVICE           = 'cuda:0'
BIRD_CLASS_ID    = 14                     # bird в YOLOv8 :contentReference[oaicite:3]{index=3}
OUTPUT_CLASS_ID  = 0

os.makedirs(LABEL_DIR, exist_ok=True)

# Инициализация SAHI-модели
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=MODEL_PATH,
    confidence_threshold=CONF_THRESHOLD,
    device=DEVICE
)                                              # SAHI: слайсинг для мелких объектов :contentReference[oaicite:4]{index=4}

# Обход всех изображений
for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith(('.jpg','.jpeg','.png')):
        continue
    img_path = os.path.join(IMAGE_DIR, img_name)
    image = cv2.imread(img_path)
    if image is None:
        print(f"[WARN] Не удалось загрузить {img_path}")
        continue
    h, w = image.shape[:2]

    # Слайсинг-инференс
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO
    )                                            # slice-size=768, overlap=20% :contentReference[oaicite:5]{index=5}

    # Сборка аннотаций
    annotations = []
    for obj in result.object_prediction_list:
        # фильтруем только «bird» по индексу 14
        if obj.category.id != BIRD_CLASS_ID:
            continue
        xmin, ymin = obj.bbox.minx, obj.bbox.miny
        xmax, ymax = obj.bbox.maxx, obj.bbox.maxy
        x_center = ((xmin + xmax) / 2) / w
        y_center = ((ymin + ymax) / 2) / h
        width    = (xmax - xmin) / w
        height   = (ymax - ymin) / h
        annotations.append((OUTPUT_CLASS_ID, x_center, y_center, width, height))

    # # Отладочный вывод
    # print(f"[INFO] {img_name}: найдено {len(annotations)} аннотаций")

    # Запись только при наличии хотя бы одного бокса
    if annotations:
        txt_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + '.txt')
        with open(txt_path, 'w') as f:
            for cid, xc, yc, bw, bh in annotations:
                f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

print("Генерация разметки завершена — пустые файлы не создаются.")
