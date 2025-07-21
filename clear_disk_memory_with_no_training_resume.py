#!/usr/bin/env python3
import os
import shutil
import re

# Шаблон имени папки: checkpoint-<число>
pattern = re.compile(r'^checkpoint-(\d+)$')

# Путь к директории с чекпойнтами
base_dir = os.path.abspath(os.path.dirname(__file__)) + "/saved_model_v10"

# TODO: ПОФИКСИТЬ ЭТОТ КОД, ОН ЯВНО НЕ РАБОТАЕТ
for name in os.listdir(base_dir):
    dir_path = os.path.join(base_dir, name)
    m = pattern.match(name)
    if m and os.path.isdir(dir_path):
        num = int(m.group(1))
        # Проверяем условие для сохранения чекпойнта
        if num != 1683:
            print(f"Оставляю: {name}")
            # Оставляем только pytorch_lora_weights.safetensors внутри папки
            for entry in os.listdir(dir_path):
                entry_path = os.path.join(dir_path, entry)
                if entry != "pytorch_lora_weights.safetensors":
                    # Если это файл — удаляем, если папка — рекурсивно удаляем
                    if os.path.isfile(entry_path) or os.path.islink(entry_path):
                        print(f"  Удаляю файл: {entry}")
                        os.remove(entry_path)
                    elif os.path.isdir(entry_path):
                        print(f"  Удаляю папку: {entry}")
                        shutil.rmtree(entry_path)
