from dataclasses import dataclass
from typing import Optional

import psutil
from conmech.helpers.config import Config


@dataclass
class TrainingData:
    TRAIN_SCALE: int = 1
    VALIDATION_SCALE: int = 1
    PRINT_SCALE: int = 1

    DATASET: str = "synthetic"  # synthetic # calculator # live
    FINAL_TIME: float = 5  # !# 5 #8
    MESH_DENSITY: int = 16  # !# 8 #16
    ADAPTIVE_TRAINING_MESH: bool = False  # True

    FORCES_RANDOM_SCALE: int = 4
    OBSTACLE_ORIGIN_SCALE: float = 3.0 * TRAIN_SCALE
    OBSTACLE_MIN_SCALE: float = 0.7 * TRAIN_SCALE
    U_RANDOM_SCALE: float = 0.2
    V_RANDOM_SCALE: float = 2.5

    ROTATE_VELOCITY_PROPORTION: float = 0.3
    ZERO_FORCES_PROPORTION: float = 0.9
    CORNERS_SCALE_PROPORTION: float = 0.2
    ROTATE_SCALE_PROPORTION: float = 0.5

    U_NOISE_GAMMA: float = 0.1
    U_IN_RANDOM_FACTOR: float = 0.005 * U_RANDOM_SCALE
    V_IN_RANDOM_FACTOR: float = 0.005 * V_RANDOM_SCALE

    SAVE_AT_MINUTES: int = 10
    VALIDATE_AT_EPOCHS: int = 20
    UPDATE_AT_EPOCHS: int = 100

    USE_ENERGY_AS_LOSS: bool = True  # !#
    BATCH_SIZE: int = 128  # !# 64 # 128
    VALID_BATCH_SIZE: int = 128  # !#
    SYNTHETIC_BATCHES_IN_EPOCH: int = 32  # 96 # 32 !# 64 # 512
    SYNTHETIC_SOLVERS_COUNT: int = BATCH_SIZE * SYNTHETIC_BATCHES_IN_EPOCH

    USE_DATASET_STATS: bool = False
    INPUT_BATCH_NORM: bool = True
    INTERNAL_BATCH_NORM: bool = False
    LAYER_NORM: bool = True

    DROPOUT_RATE: Optional[float] = None  # 0.0  # 0.1  # 0.2  0.05
    SKIP: bool = True
    GRADIENT_CLIP = 10.0  # None

    ATTENTION_HEADS: Optional[int] = None  # None 1 3 5

    INITIAL_LR: float = 1e-3  # 1e-3  # 1e-4 # 1e-5
    LR_GAMMA: float = 1.0  # 0.999
    FINAL_LR: float = 1e-6

    LATENT_DIM: int = 128
    ENC_LAYER_COUNT: int = 2
    PROC_LAYER_COUNT: int = 0
    DEC_LAYER_COUNT: int = 2
    MESSAGE_PASSES: int = 8  # 5 # 10


@dataclass
class TrainingConfig(Config):
    td: TrainingData = TrainingData()
    DEVICE: str = "_"
    # torch.autograd.set_detect_anomaly(True)
    # print(numba.cuda.gpus)

    DATALOADER_WORKERS = 4
    SYNTHETIC_GENERATION_WORKERS = 1  # 2

    TOTAL_MEMORY_GB = psutil.virtual_memory().total / 1024**3
    TOTAL_MEMORY_LIMIT_GB = round(TOTAL_MEMORY_GB * 0.9, 2)
    SYNTHETIC_GENERATION_MEMORY_LIMIT_GB = round(
        (TOTAL_MEMORY_GB * 0.8) / SYNTHETIC_GENERATION_WORKERS, 2
    )

    DATASET_IMAGES_COUNT: float = 100

    LOG_DATASET_STATS = True
    LOAD_TRAIN_DATASET_TO_RAM = True
