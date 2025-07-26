# [Читать диплом](https://www.dropbox.com/scl/fi/8fvhxu0z3e09av9e7o2uz/2025-110508-_2.pdf?rlkey=9ukrmp6q3okqfvnuqkjqosctj&st=7fkuwf6z&dl=0) 

---

## About

**Generative Image Augmentation** — это Python-проект, реализующий метод расширения обучающих выборок изображений с помощью дообучения предобученной диффузионной модели Stable Diffusion 2 и техники Low-Rank Adaptation (LoRA). Основная цель — синтезировать изображения, соответствующие гладкому распределению исходной выборки, и автоматически разметить их для последующего обучения детекторов.

**Ключевые особенности:**

* **Дообучение Stable Diffusion 2.0 с LoRA**
  
  Использование скрипта `train_text_to_image_lora.py` (Diffusers) для эффективной адаптации предобученной модели под узкую предметную область с минимальными вычислительными затратами.
  
* **Генерация синтетических изображений**
  
  Набор python-скриптов, которые в совокупности составляют единый алгоритм генерации синтетических изображений с разметкой
  
* **Автоматическая разметка с использованием модели для детекции**
  
  Детектирование на сгенерированных изображениях с кастомным порогом по confidence score, что позволяет исключать некорректные образцы.
  
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
   accelerate launch train_text_to_image_lora.py \
     --pretrained_model_name_or_path="stabilityai/stable-diffusion-2" \
     --train_data_dir="../../../dataset_for_finetuning_v9" \
     --caption_column="text" --center_crop --resolution=768 \
     --random_flip --train_batch_size=1 --num_train_epochs=40
     --checkpointing_steps=17 --learning_rate=1e-04 --lr_scheduler="constant" \
     --lr_warmup_steps=0 --output_dir="../../../saved_model_v10" \
     --validation_prompt="A photo of flying photorealistic bird in a photorealistic environment: in a city or in an urban area or in a forest or in a field or in the mountains" \
     --report_to="wandb" --num_validation_images 20
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

