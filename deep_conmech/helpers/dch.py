"""
deep_conmech helpers
"""
import os
import resource

import pandas
import torch.multiprocessing

from deep_conmech.training_config import TrainingConfig


def print_pandas(data):
    name = f"{data}=".split("=")[0]
    print(f">>> {name} <<<")
    print(pandas.DataFrame(data).round(4))


def cuda_launch_blocking():
    print("CUDA_LAUNCH_BLOCKING !!!")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def set_memory_limit(config: TrainingConfig):
    rsrc = resource.RLIMIT_DATA
    _, hard = resource.getrlimit(rsrc)
    new_limit_gb = config.total_memory_limit_gb
    new_soft = int(new_limit_gb * 1024**3)
    resource.setrlimit(rsrc, (new_soft, hard))
    print(f"Memory limit set to {new_limit_gb:.2f} GB")


def set_torch_sharing_strategy():
    torch.multiprocessing.set_sharing_strategy("file_system")
