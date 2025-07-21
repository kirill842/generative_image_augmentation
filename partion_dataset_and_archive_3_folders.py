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
    # Категории папок, которые нужно разделить
    categories = ['images', 'labels', 'images with labels']
    # Собираем файлы для каждой категории
    files_by_category = {}
    for category in categories:
        cat_path = os.path.join(input_dir, category)
        if not os.path.isdir(cat_path):
            print(f"Категория не найдена: {cat_path}")
            return
        files = [f for f in os.listdir(cat_path)
                 if os.path.isfile(os.path.join(cat_path, f))]
        if not files:
            print(f"В папке {cat_path} нет файлов.")
            return
        files_by_category[category] = files

    # Предполагаем, что у всех категорий одинаковое количество файлов
    total = len(files_by_category[categories[0]])
    base_size = total // num_parts
    remainder = total % num_parts

    start = 0
    for part in range(1, num_parts + 1):
        size = base_size + (1 if part <= remainder else 0)
        end = start + size

        # Создаем директорию части
        part_dir = os.path.join(output_dir, f"part_{part}")
        os.makedirs(part_dir, exist_ok=True)

        # Для каждой категории копируем соответствующие файлы
        for category in categories:
            part_cat_dir = os.path.join(part_dir, category.replace(' ', '_'))
            os.makedirs(part_cat_dir, exist_ok=True)

            # Берем соответствующий срез списка
            part_files = files_by_category[category][start:end]
            for filename in part_files:
                src = os.path.join(input_dir, category, filename)
                dst = os.path.join(part_cat_dir, filename)
                shutil.copy2(src, dst)

            print(f"Part {part}/{num_parts} - {category}: скопировано {len(part_files)} файлов в {part_cat_dir}")

        # Архивация части
        archive_path = os.path.join(output_dir, f"part_{part}.7z")
        try:
            subprocess.run([
                seven_zip_cmd, "a", "-t7z", archive_path, part_dir
            ], check=True)
            print(f"Архив создан: {archive_path}")
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при архивации части {part}: {e}")

        start = end

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(
    #     description='Split dataset into parts with subfolders and archive with 7z.'
    # )
    # parser.add_argument('input_dir', help='Путь к директории с папками images, labels, images with labels')
    # parser.add_argument('output_dir', help='Путь к директории-назначению')
    # parser.add_argument('-n', '--num_parts', type=int, default=5, help='Количество частей')
    # parser.add_argument('-7z', '--seven_zip_cmd', default=find_7z_executable() or '7z',
    #                     help='Команда для запуска 7z')
    # args = parser.parse_args()
    #
    # split_and_archive(
    #     args.input_dir,
    #     args.output_dir,
    #     args.num_parts,
    #     args.seven_zip_cmd
    # )
    split_and_archive(
        "./generated_dataset",
        "./generated_dataset_partioned",
        20,
        "C:/Program Files/7-Zip/7z.exe"
    )
