import os
import cv2
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from tqdm import tqdm

# Параметры
MODEL_PATH       = 'yolov8l.pt'
IMAGE_DIR        = './inference_images_model_v10_ckpt_1258_photo_brown_prompt_4'
LABEL_DIR        = './finish_labeled/inference_images_model_v10_ckpt_1258_photo_brown_prompt_4/labels'
IMAGE_OUT_DIR    = './finish_labeled/inference_images_model_v10_ckpt_1258_photo_brown_prompt_4/images'
CONF_THRESHOLD   = 0.7
SLICE_SIZE       = 768
OVERLAP_RATIO    = 0.2
DEVICE           = 'cuda:2'
BIRD_CLASS_ID    = 14
OUTPUT_CLASS_ID  = 0

# Создаем выходные папки
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(IMAGE_OUT_DIR, exist_ok=True)

# Инициализация SAHI-модели
n_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path=MODEL_PATH,
    confidence_threshold=CONF_THRESHOLD,
    device=DEVICE
)

# Обход всех изображений
def main():
    for img_name in tqdm(os.listdir(IMAGE_DIR)):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
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
            n_model,
            slice_height=SLICE_SIZE,
            slice_width=SLICE_SIZE,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO
        )

        # Сборка аннотаций
        annotations = []
        for obj in result.object_prediction_list:
            if obj.category.id != BIRD_CLASS_ID:
                continue
            xmin, ymin = obj.bbox.minx, obj.bbox.miny
            xmax, ymax = obj.bbox.maxx, obj.bbox.maxy
            x_center = ((xmin + xmax) / 2) / w
            y_center = ((ymin + ymax) / 2) / h
            width    = (xmax - xmin) / w
            height   = (ymax - ymin) / h
            annotations.append((OUTPUT_CLASS_ID, x_center, y_center, width, height))

        # Запись разметки и сохранение картинки
        if annotations:
            # сохраняем текстовый файл разметки
            txt_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + '.txt')
            with open(txt_path, 'w') as f:
                for cid, xc, yc, bw, bh in annotations:
                    f.write(f"{cid} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
            # сохраняем изображение в выходную папку
            out_path = os.path.join(IMAGE_OUT_DIR, img_name)
            cv2.imwrite(out_path, image)
            # print(f"[INFO] {img_name}: сохранено изображение и {len(annotations)} аннотаций")

    print("Генерация разметки и сохранение изображений завершены — пустые файлы не создаются.")

if __name__ == '__main__':
    main()
