"""
conmech helpers
"""
import os
import shutil
import time
from datetime import datetime

from tqdm import tqdm


def get_timestamp():
    return int(time.time() * 10000)


RUN_TIMESTEMP = get_timestamp()

CURRENT_TIME = datetime.now().strftime("%d-%H.%M.%S")


def get_tqdm(iterable, desc=None, position=None) -> tqdm:
    return tqdm(iterable, desc=desc, position=position)#, ascii=True)


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
