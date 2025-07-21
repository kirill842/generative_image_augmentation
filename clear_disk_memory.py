#!/usr/bin/env python3
import os
import shutil
import re

# Шаблон имени папки: checkpoint-<число>
pattern = re.compile(r'^checkpoint-(\d+)$')

# Путь к директории, в которой лежат чекпоинты.
# По умолчанию — текущая директория скрипта.
base_dir = os.path.abspath(os.path.dirname(__file__))
base_dir += "/saved_model_v10"

for name in os.listdir(base_dir):
    path = os.path.join(base_dir, name)
    m = pattern.match(name)
    if m and os.path.isdir(path):
        num = int(m.group(1))
        # Если номер не кратен 3584 — удаляем папку
        if num % 51 != 0 and num not in {612, 1445, 1258}:
            print(f"Удаляю: {name}")
            shutil.rmtree(path)
        else:
            print(f"Оставляю: {name}")
