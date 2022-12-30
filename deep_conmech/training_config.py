from dataclasses import dataclass
from typing import Optional

import psutil
from torch import nn

from conmech.helpers.config import Config

TEST = False  # False
DIMENSION = 3
MESH_LAYERS_PROPORTION = 2
CLOSEST_COUNT = 3  # 4
CLOSEST_BOUNDARY_COUNT = CLOSEST_COUNT - 1


@dataclass
class TrainingData:
    dimension: int = DIMENSION

    train_scale: int = 1
    validation_scale: int = 1
    print_scale: int = 1

    dataset: str = "calculator"  # synthetic # calculator
    final_time: float = 0.5 if TEST else 4  # 4  # 8
    mesh_density: int = 32  # 16  # 8  # 16 # 64 if dimension == 2 else 16
    adaptive_training_mesh_scale: Optional[float] = 0.0  # 0.8  # 0.1

    forces_random_scale: float = 4.0
    obstacle_origin_max_scale: float = 3.0 * train_scale
    obstacle_origin_min_scale: float = 0.4 * train_scale
    initial_corners_scale: float = 0.1
    displacement_random_scale: float = 0.2
    velocity_random_scale: float = 2.5

    zero_forces_proportion: float = 0.2
    zero_displacement_proportion: float = 0.2
    zero_velocity_proportion: float = 0.2
    corners_scale_proportion: float = 0.8

    displacement_to_velocity_noise: float = 0.1
    displacement_in_random_factor: float = 0.0  # 0.2 * (0.01**2)  # 0.1 0.5 0.1 0.01 0
    # same as net error, so that a_correction is similar
    velocity_in_random_factor: float = displacement_in_random_factor * 100.0
    # 0.005 * velocity_random_scale

    save_at_minutes: int = 5
    raport_at_examples: int = 256 * 64
    validate_at_minutes: int = 15  # 0 # 10 100
    validate_at_epochs: int = 2  # 5
    validate_scenarios_at_epochs: Optional[int] = None  # 30  # None 3

    batch_size: int = 1  # 4  # 8  # 1  # 16  # 32  # 16  # 32 # 256
    dataset_size: int = 32  # 256 * (1 if TEST else 1) #8)  # 2048)

    use_dataset_statistics: bool = False
    input_batch_norm: bool = False  # False #True
    internal_batch_norm: bool = False
    layer_norm: bool = False  # True

    dropout_rate: Optional[float] = 0.5  # None  # 0.0  # 0.1 # 0.2  0.05
    skip_connections: bool = True
    gradient_clip = None  # 10.0

    attention_heads_count: Optional[int] = None  # None 1 3 5

    initial_learning_rate: float = 1e-4  # 1e-4  # 1e-3  # 1e-3  # 1e-4 # 1e-5
    learning_rate_decay: float = 1.0  # 0.995
    final_learning_rate: float = initial_learning_rate  # 1e-6

    activation = nn.ReLU()  # PReLU LeakyReLU
    latent_dimension: int = 64  # 128
    encoder_layers_count: int = 0  # 3
    processor_layers_count: int = 0
    decoder_layers_count: int = 0  # 3
    message_passes: int = 8  # 10  # 8  # 4 # 3


@dataclass
class TrainingConfig(Config):
    use_jax: bool = False #True

    td: TrainingData = TrainingData()
    device: str = "cuda"  # "cpu" if TEST else "cuda"
    #:" + ",".join(map(str, DEVICE_IDS)))  # torch.cuda.is_available()

    distributed_training = False and not use_jax
    dataloader_workers = 4
    synthetic_generation_workers = 4
    scenario_generation_workers = 2

    total_mempry_gb = psutil.virtual_memory().total / 1024**3
    total_memory_limit_gb = round(total_mempry_gb * 0.7, 2)
    synthetic_generation_memory_limit_gb = round(
        (total_mempry_gb * 0.8) / synthetic_generation_workers, 2
    )
    loaded_data_memory_limit_gb = round((total_mempry_gb * 0.8), 2)

    dataset_images_count: Optional[float] = None  # 8

    log_dataset_stats: bool = False
    with_train_scenes_file: bool = False

    compare_with_base_scene = False
    max_epoch_number: Optional[int] = None
    datasets_main_path: str = "datasets"
    log_catalog: str = "log"
    load_newest_train: bool = False

    load_training_data_to_ram: bool = False
    load_validation_data_to_ram: bool = False
    profile_training: bool = False
