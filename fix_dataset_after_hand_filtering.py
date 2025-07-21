import os
import shutil
from pathlib import Path


def sync_folders_with_extras(root_dir, folder_names, extras_dir_name="extras"):
    """
    Синхронизирует несколько подпапок в root_dir, оставляя в них только общие файлы по basename,
    а все остальные — перемещает в root_dir/extras/<folder_name>.

    Args:
        root_dir (str or Path): путь к корневой папке, где лежат папки из folder_names.
        folder_names (list of str): названия папок для синхронизации, например
            ["images", "labels", "images with labels"].
        extras_dir_name (str): название общей папки для «лишних» файлов (по умолчанию "extras").
    """
    root = Path(root_dir)
    # Проверяем, что корень существует
    if not root.is_dir():
        raise ValueError(f"Корневая папка не найдена: {root}")

    # Полные пути к папкам-источникам
    dirs = [root / name for name in folder_names]
    for d in dirs:
        if not d.is_dir():
            raise ValueError(f"Папка не найдена: {d}")

    # 1) собираем множества basenames в каждой папке
    sets = []
    for d in dirs:
        names = {f.stem for f in d.iterdir() if f.is_file()}
        sets.append(names)

    # 2) пересечение по всем папкам
    common = set.intersection(*sets)
    print(f"Общих файлов по basename: {len(common)}")

    # 3) создаём root/extras и поддиректории
    extras_root = root / extras_dir_name
    extras_root.mkdir(exist_ok=True)
    extras_subdirs = {}
    for name in folder_names:
        sub = extras_root / name
        sub.mkdir(exist_ok=True)
        extras_subdirs[name] = sub

    # 4) для каждой исходной папки перемещаем все файлы,
    #    basename которых нет в common, в соответствующую extras-подпапку
    for d, name in zip(dirs, folder_names):
        for f in d.iterdir():
            if not f.is_file():
                continue
            if f.stem not in common:
                target = extras_subdirs[name] / f.name
                shutil.move(str(f), str(target))
                print(f"Переместил «лишний» файл {f.name} -> {target}")

    print("Синхронизация завершена.")


if __name__ == "__main__":
    # пример: папка проекта — текущая директория
    PATH = "./generated_dataset"
    folders = ["images", "labels", "images with labels"]
    sync_folders_with_extras(PATH, folders)
