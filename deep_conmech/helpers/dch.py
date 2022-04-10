"""
deep_conmech helpers
"""
import os

import pandas
import psutil
import resource

from deep_conmech.training_config import TrainingConfig


def print_pandas(data):
    name = f"{data}=".split("=")[0]
    print(f">>> {name} <<<")
    print(pandas.DataFrame(data).round(4))


def get_used_memory_gb():
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3  # (b -> kb -> mb -> gb)


def cuda_launch_blocking():
    print("CUDA_LAUNCH_BLOCKING !!!")
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def set_memory_limit(config: TrainingConfig):
    rsrc = resource.RLIMIT_DATA
    soft, hard = resource.getrlimit(rsrc)
    new_limit_gb = config.TOTAL_MEMORY_LIMIT_GB
    new_soft = int(new_limit_gb * 1024**3)
    resource.setrlimit(rsrc, (new_soft, hard))
    print(f"Memory limit set to {new_limit_gb:.2f} GB")
