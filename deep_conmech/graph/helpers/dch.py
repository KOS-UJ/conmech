"""
deep_conmech helpers
"""
import os
import resource

import deep_conmech.common.config as config
import pandas
import psutil


def print_pandas(data):
    name = f"{data=}".split("=")[0]
    print(f">>> {name} <<<")
    print(pandas.DataFrame(data).round(4))


def get_used_memory_gb():
    return (
        psutil.Process(os.getpid()).memory_info().rss / 1024 ** 3
    )  # (b -> kb -> mb -> gb)


TOTAL_MEMORY_GB = psutil.virtual_memory().total / 1024 ** 3
TOTAL_MEMORY_LIMIT_GB = TOTAL_MEMORY_GB * 0.9
GENERATION_MEMORY_LIMIT_GB = (TOTAL_MEMORY_GB * 0.8) / config.GENERATION_WORKERS


def set_memory_limit():
    rsrc = resource.RLIMIT_DATA
    soft, hard = resource.getrlimit(rsrc)
    soft_limit = TOTAL_MEMORY_LIMIT_GB * 1024 ** 3
    resource.setrlimit(rsrc, (soft_limit, hard))
    print(f"Setting memory limit to {TOTAL_MEMORY_LIMIT_GB:.2f} GB")

set_memory_limit()