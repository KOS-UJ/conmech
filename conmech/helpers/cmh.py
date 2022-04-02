"""
conmech helpers
"""
import os
import shutil
import time

from tqdm import tqdm

from conmech.helpers.config import Config


def get_timestamp(config: Config):
    return int(time.time() * config.TIMESTAMP_SKIP)


def get_tqdm(iterable, config: Config, desc=None, position=None) -> tqdm:
    return tqdm(iterable, desc=desc, position=position, ascii=config.SHELL)


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_folders(path):
    all_folders = path.split("/")
    final_path = ""
    for folder in all_folders:
        final_path += f"{folder}/"
        create_folder(final_path)


def get_all_contents(directory):
    return os.listdir(directory)


def clear_folder(directory):
    if not os.path.exists(directory):
        return
    shutil.rmtree(directory)


def recreate_folder(directory):
    clear_folder(directory)
    create_folder(directory)


def find_files_by_extension(directory, extension):
    files = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(f".{extension}")]:
            path = os.path.join(dirpath, filename)
            files.append(path)
    return files
