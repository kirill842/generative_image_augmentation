import os
import shutil
import argparse
import subprocess


def find_7z_executable():
    """
    Ищет доступную утилиту 7z среди возможных вариантов.
    """
    candidates = ['7z', '7za', '7zr']
    for cmd in candidates:
        path = shutil.which(cmd)
        if path:
            return path
    return None


def split_and_archive(input_dir, output_dir, num_parts, seven_zip_cmd):
    # Получаем список файлов изображений
    images = [f for f in os.listdir(input_dir)
              if os.path.isfile(os.path.join(input_dir, f))]
    total = len(images)
    if total == 0:
        print("В папке нет файлов.")
        return

    # Размер базовой части и остаток
    base_size = total // num_parts
    remainder = total % num_parts

    start = 0
    for part in range(1, num_parts + 1):
        # Распределяем остаток по первым частям
        size = base_size + (1 if part <= remainder else 0)
        end = start + size
        part_files = images[start:end]

        # Создаём директорию для части
        part_dir = os.path.join(output_dir, f"part_{part}")
        os.makedirs(part_dir, exist_ok=True)

        # Копируем файлы
        for filename in part_files:
            shutil.copy2(os.path.join(input_dir, filename),
                         os.path.join(part_dir, filename))

        print(f"Part {part}: скопировано {len(part_files)} файлов в {part_dir}")

        # Упаковываем в архив 7z
        archive_path = os.path.join(output_dir, f"part_{part}.7z")
        try:
            subprocess.run([
                seven_zip_cmd, "a", "-t7z", archive_path, part_dir
            ], check=True)
            print(f"Архив создан: {archive_path} (использован: {seven_zip_cmd})")
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при архивации части {part}: {e}")

        start = end

if __name__ == '__main__':
    split_and_archive("./finish_labeled/inference_images_model_v10_ckpt_1445_photo_prompt/images with labels",
                      "./finish_labeled/inference_images_model_v10_ckpt_1445_photo_prompt/images with labels partioned",
                      5,
                      "C:/Program Files/7-Zip/7z.exe")
