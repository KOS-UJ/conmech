import os
import resource
import deep_conmech.common.config as config
import numpy as np
import pandas
import psutil
import torch
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.autograd.set_detect_anomaly(True)
print(f"Running using {device}")


def cuda_launch_blocking():
    print("CUDA_LAUNCH_BLOCKING !!!!")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def set_precision(data):
    return data.float()


def to_torch_double(data):
    return torch.tensor(data, dtype=torch.float64)


def to_torch_long(data):
    return torch.tensor(data, dtype=torch.long)


def to_np_double(data):
    return data.cpu().detach().numpy().astype(np.float64)


def to_np_long(data):
    return data.cpu().detach().numpy().astype(np.long)


def get_contiguous_torch(data):
    return to_torch_long(data).t().contiguous()


def append_euclidean_norm(data):
    return torch.hstack((data, torch.linalg.norm(data, keepdim=True, dim=1)))


def euclidean_norm_torch(vector):
    return torch.sqrt(torch.sum(vector ** 2, axis=-1))


def max_norm(data):
    return torch.max(torch.linalg.norm(data, axis=1))  # -1 ?


def rmse_torch(predicted, exact):
    return torch.sqrt(torch.mean(torch.linalg.norm(predicted - exact, axis=-1) ** 2))


def get_tqdm(iterable, desc=None, position=None):
    return tqdm(iterable, desc=desc, position=position, ascii=True)


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

#####

def print_pandas(data):
    name = f"{data=}".split("=")[0]
    print(f">>> {name} <<<")
    print(pandas.DataFrame(data).round(4))


def get_used_memory_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3

def set_memory_limit():
    rsrc = resource.RLIMIT_DATA
    soft, hard = resource.getrlimit(rsrc)
    soft_limit = config.TOTAL_MEMORY_LIMIT_GB * (1024 ** 3)  # (b -> kb -> mb -> gb)
    resource.setrlimit(rsrc, (soft_limit, hard))

