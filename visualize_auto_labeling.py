import os
import cv2
from tqdm import tqdm

# Параметры
IMAGE_DIR = './generated_dataset/images'      # папка с изображениями 768x768
LABEL_DIR = './generated_dataset/labels'       # папка с YOLO .txt файлами
OUTPUT_DIR = './generated_dataset/images with labels'  # папка для сохранения изображений с рамками

# Убедимся, что выходная папка существует
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Проходим по всем изображениям
for img_name in tqdm(os.listdir(IMAGE_DIR)):
    # проверяем расширение
    if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    # формируем пути к файлам
    img_path = os.path.join(IMAGE_DIR, img_name)
    label_path = os.path.join(LABEL_DIR, os.path.splitext(img_name)[0] + '.txt')

    # пропускаем, если нет файла разметки
    if not os.path.isfile(label_path):
        continue

    # читаем изображение
    image = cv2.imread(img_path)
    if image is None:
        print(f"[WARN] Не удалось загрузить изображение: {img_path}")
        continue
    h, w = image.shape[:2]

    # читаем аннотации
    with open(label_path, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # рисуем боксы
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue  # пропускаем некорректные строки
        _, x_center, y_center, box_w, box_h = parts
        x_center, y_center = float(x_center), float(y_center)
        box_w, box_h = float(box_w), float(box_h)

        # конвертация нормализованных координат в пиксели
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        # рисуем прямоугольник (зелёный цвет, толщина 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # сохраняем результат только для размеченных изображений
    out_path = os.path.join(OUTPUT_DIR, img_name)
    cv2.imwrite(out_path, image)

print(f"Визуализация завершена. Файлы сохранены в: {OUTPUT_DIR}")
