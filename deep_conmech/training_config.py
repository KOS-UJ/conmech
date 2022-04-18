from dataclasses import dataclass
from typing import Optional

import psutil

from conmech.helpers.config import Config


@dataclass
class TrainingData:
    train_scale: int = 1
    validation_scale: int = 1
    print_scale: int = 1

    dataset: str = "synthetic"  # synthetic # calculator # live
    final_time: float = 8  # !# 5 #8
    mesh_density: int = 16  # !# 8 #16
    adaptive_training_mesh: bool = False  # True

    forces_random_scale: int = 4
    obstacle_origin_scale: float = 3.0 * train_scale  # less
    obstacle_min_scale: float = 0.4 * train_scale
    displacement_random_scale: float = 0.2
    velocity_random_scale: float = 2.5

    rotate_velocity_proportion: float = 0.5
    zero_forces_proportion: float = 0.2  ## 0.8
    corners_scale_proportion: float = 0.8  # less
    rotate_scale_proportion: float = 0.5

    displacement_to_velocity_noise: float = 0.1
    displacement_in_random_factor: float = 0.005 * displacement_random_scale
    velocity_in_random_factor: float = 0.005 * velocity_random_scale

    save_at_minutes: int = 10
    validate_at_epochs: int = 10
    validate_scenarios_at_epochs: int = 30
    update_at_epochs: int = 100

    use_energy_as_loss: bool = True
    batch_size: int = 128  #
    valid_batch_size: int = 128  #
    synthetic_batches_in_epoch: int = 96  # 2

    USE_DATASET_STATS: bool = False
    INPUT_BATCH_NORM: bool = True
    INTERNAL_BATCH_NORM: bool = False
    LAYER_NORM: bool = True

    DROPOUT_RATE: Optional[float] = None  # 0.0  # 0.1 # 0.2  0.05
    SKIP: bool = True
    GRADIENT_CLIP = 10.0  # None

    ATTENTION_HEADS: Optional[int] = 1  # None 1 3 5

    INITIAL_LR: float = 1e-3  # 1e-3  # 1e-4 # 1e-5
    LR_GAMMA: float = 0.999  # 1.0
    FINAL_LR: float = 1e-6

    LATENT_DIM: int = 128
    ENC_LAYER_COUNT: int = 0  # 2
    PROC_LAYER_COUNT: int = 0
    DEC_LAYER_COUNT: int = 0  # 2
    MESSAGE_PASSES: int = 8  # 5 # 10


@dataclass
class TrainingConfig(Config):
    td: TrainingData = TrainingData()
    device: str = "_"
    # torch.autograd.set_detect_anomaly(True)
    # print(numba.cuda.gpus)

    DATALOADER_WORKERS = 4
    SYNTHETIC_GENERATION_WORKERS = 1  # 2

    TOTAL_MEMORY_GB = psutil.virtual_memory().total / 1024**3
    TOTAL_MEMORY_LIMIT_GB = round(TOTAL_MEMORY_GB * 0.9, 2)
    SYNTHETIC_GENERATION_MEMORY_LIMIT_GB = round(
        (TOTAL_MEMORY_GB * 0.8) / SYNTHETIC_GENERATION_WORKERS, 2
    )

    dataset_images_count: float = 100

    log_dataset_stats = True
    load_train_dataset_to_ram = True
    compare_with_base_setting = False

    max_epoch_number: Optional[int] = None
    datasets_main_path: str = "datasets"
    log_catalog: str = "log"
