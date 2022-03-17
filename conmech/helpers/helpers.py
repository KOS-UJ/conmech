import glob
import os
import shutil
import resource
import time
from datetime import datetime

import deep_conmech.common.config as config
import numpy as np
import psutil
from tqdm import tqdm

# np.random.seed(42)


def get_timestamp():
    return int(time.time() * 10000)

RUN_TIMESTEMP = get_timestamp()

CURRENT_TIME = datetime.now().strftime("%d-%H.%M.%S")


def get_tqdm(iterable, desc=None, position=None):
    return tqdm(iterable, desc=desc, position=position, ascii=True)


def get_used_memory_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3


def set_memory_limit():
    rsrc = resource.RLIMIT_DATA
    soft, hard = resource.getrlimit(rsrc)
    # print('Soft limit starts as  :', soft)
    limit = config.TOTAL_MEMORY_LIMIT_GB * (1024 ** 3)  # (b -> kb -> mb -> gb)
    resource.setrlimit(rsrc, (limit, hard))  # limit to one kilobyte
    soft, hard = resource.getrlimit(rsrc)
    # print('Soft limit changed to :', soft)



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