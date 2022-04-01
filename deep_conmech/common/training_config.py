from dataclasses import dataclass

import psutil

from conmech.helpers.config import Config
from deep_conmech.common.training_data import TrainingData


@dataclass
class TrainingConfig(Config):
    td: TrainingData = TrainingData()
    DEVICE: str = "_"
    # torch.autograd.set_detect_anomaly(True)
    # print(numba.cuda.gpus)

    DATALOADER_WORKERS = 4
    GENERATION_WORKERS = 2

    ############

    TOTAL_MEMORY_GB = psutil.virtual_memory().total / 1024 ** 3
    TOTAL_MEMORY_LIMIT_GB = round(TOTAL_MEMORY_GB * 0.9, 2)
    GENERATION_MEMORY_LIMIT_GB = round((TOTAL_MEMORY_GB * 0.8) / GENERATION_WORKERS, 2)

    ############

    DATA_FOLDER: str = f"{td.MESH_DENSITY}"
    PRINT_DATA_CUTOFF: float = 0.1
