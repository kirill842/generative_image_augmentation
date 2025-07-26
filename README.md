# [Читать диплом](https://www.dropbox.com/scl/fi/8fvhxu0z3e09av9e7o2uz/2025-110508-_2.pdf?rlkey=9ukrmp6q3okqfvnuqkjqosctj&st=7fkuwf6z&dl=0) 

---

## About

**Generative Image Augmentation** — это Python-проект, реализующий метод расширения обучающих выборок изображений с помощью дообучения предобученной диффузионной модели Stable Diffusion 2 и техники Low-Rank Adaptation (LoRA). Основная цель — синтезировать изображения, соответствующие гладкому распределению исходной выборки, и автоматически разметить их для последующего обучения детекторов.

**Ключевые особенности:**

* **Дообучение Stable Diffusion 2.0 с LoRA**
  Использование скрипта `train_text_to_image_lora.py` (Diffusers) для эффективной адаптации предобученной модели под узкую предметную область с минимальными вычислительными затратами.
* **Генерация синтетических изображений**
  Набор python-скриптов, которые в совокупности составляют единый алгоритм генерации синтетических изображений с разметкой
* **Автоматическая разметка с YOLOv8**
  Детектирование на сгенерированных изображениях с кастомным порогом надёжности, что позволяет исключать некорректные образцы.
* **Модульность и расширяемость**
  Лёгкая замена модели генерации, меток и модели разметки под любые другие классы объектов.

## Quick Start

1. **Клонировать репозиторий**

   ```bash
   git clone https://github.com/kirill842/generative_image_augmentation.git
   cd generative_image_augmentation
   ```

2. **Установить зависимости**

   ```bash
   pip install -r requirements.txt
   ```

3. **Дообучение модели**

   ```bash
   python train_text_to_image_lora.py \
     --pretrained_model_name_or_path="stabilityai/stable-diffusion-2" \
     --train_data_dir="data/birds_crops" \
     --output_dir="lora_checkpoints" \
     --text_prompts_file="data/prompts.txt" \
     --resolution=768 \
     --train_batch_size=1 \
     --learning_rate=1e-4 \
     --num_train_epochs=20 \
     --lora_rank=4
   ```

4. **Генерация**

   ```bash
   python generate.py \
     --lora_checkpoint="lora_checkpoints/best.ckpt" \
     --prompt="A flying or sitting realistic bird in a realistic environment" \
     --num_images=500 \
     --output_dir="generated_images"
   ```

5. **Разметка**

   ```bash
   python annotate.py \
     --images_dir="generated_images" \
     --model="yolov8n.pt" \
     --confidence=0.5 \
     --output_annotations="annotations/"
   ```

## Результаты

* **Визуальные**: сгенерированные птицы выглядят реалистично и органично вписываются в разные среды (см. папку `samples/`).
* **Качество разметки**: YOLOv8 демонстрирует высокую точность детекции новых изображений без дополнительной ручной корректировки.

---

Добавьте или уберите разделы (например, «Contributing» или «License») согласно требованиям вашего проекта. Такое описание даст инженерам чёткое понимание методологии, архитектуры и быстрого старта с вашим кодом.


