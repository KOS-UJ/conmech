import os
import resource
import time
from ctypes import ArgumentError
from datetime import datetime

import numpy as np
import pandas
import psutil
import torch
from tqdm import tqdm

import deep_conmech.common.config as config

# np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_timestamp():
    return int(time.time() * 10000)

RUN_TIMESTEMP = get_timestamp()

CURRENT_TIME = datetime.now().strftime("%d-%H.%M.%S")


# torch.autograd.set_detect_anomaly(True)
print(f"RUNNING USING {device}")


def cuda_launch_blocking():
    print("CUDA_LAUNCH_BLOCKING !!!!")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"






def set_precision(data):
    return data.float()  # float


def to_torch_double(data):
    return torch.tensor(data, dtype=torch.float64)  # .to(device)


def to_torch_long(data):
    return torch.tensor(data, dtype=torch.long)  # .to(device)


def to_np_double(data):
    return data.cpu().detach().numpy().astype(np.float64)

def to_np_long(data):
    return data.cpu().detach().numpy().astype(np.long)










def euclidean_norm_torch(vector):
    return torch.sqrt(torch.sum(vector ** 2, axis=-1))


def max_norm(data):
    return torch.max(torch.linalg.norm(data, axis=1)) #-1 ?


def rmse_torch(predicted, exact):
    return torch.sqrt(torch.mean(torch.linalg.norm(predicted - exact, axis=-1) ** 2))



def get_tqdm(iterable, desc=None, position=None):
    return tqdm(iterable, desc=desc, position=position, ascii=True)





def get_oriented_tangential_torch(normal):
    tangential = torch.tensor([normal[1], -normal[0]]).to(device)
    return tangential

def rotate_up_torch(old_vectors, up_vector):
    tangential = get_oriented_tangential_torch(up_vector)
    result = torch.vstack((old_vectors @ tangential, old_vectors @ up_vector)).T
    return result






class MaxData:
    def __init__(self, description, episode_steps):
        self.description = description
        self.max_forces = torch.zeros(episode_steps)
        self.max_u = torch.zeros(episode_steps)
        self.max_v = torch.zeros(episode_steps)

    def set(self, setting, i):
        self.max_forces[i] = max_norm(setting.normalized_forces_torch)
        self.max_u[i] = max_norm(setting.normalized_u_old_torch)
        self.max_v[i] = max_norm(setting.normalized_v_old_torch)

    def print(self):
        print(
            f"max norms -{self.description} | f: {float(torch.max(self.max_forces)):.4f} | u: {float(torch.max(self.max_u)):.4f} | v: {float(torch.max(self.max_v))}"
        )


def print_pandas(data):
    name = f"{data=}".split("=")[0]
    print(f">>> {name} <<<")
    print(pandas.DataFrame(data).round(4))


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





def skip(time, skip):
    return np.allclose(time % skip, 0.0) or np.allclose(time % skip, skip)


def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def create_folders(path):
    all_folders = path.split("/")
    final_path = ""
    for folder in all_folders:
        final_path += f"{folder}/"
        create_folder(final_path)


