"""
conmech helpers
"""
import os

from tqdm import tqdm

from conmech.helpers.config import Config


def get_tqdm(iterable, config: Config, desc=None, position=None) -> tqdm:
    return tqdm(iterable, desc=desc, position=position, ascii=config.shell)


def create_folders(path):
    all_folders = path.split("/")
    final_path = ""
    for folder in all_folders:
        final_path += f"{folder}/"
        if not os.path.exists(final_path):
            os.mkdir(final_path)
