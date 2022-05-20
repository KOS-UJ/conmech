import numpy as np
import torch

from conmech.helpers import cmh
from deep_conmech.data import base_dataset
from deep_conmech.data.synthetic_dataset import SyntheticDataset
from deep_conmech.scene import scene_input
from deep_conmech.scene.scene_layers import SceneLayers
from deep_conmech.training_config import TrainingConfig, TrainingData


def check_layer_data_approximation(layer_list):
    if len(layer_list) == 1:
        return

    def check_diff(diff, layer, precision):
        diff_norm = torch.linalg.norm(diff, dim=1)
        base_norm = torch.linalg.norm(layer.pos.double(), dim=1)
        result = torch.sum(diff_norm) / torch.sum(base_norm)
        assert result < precision

    layer_base = layer_list[0]
    for layer_number in range(1, len(layer_list)):
        layer = layer_list[layer_number]
        layer_down = layer_list[layer_number - 1]

        diff = layer.pos.double() - SceneLayers.approximate_internal(
            from_values=layer_down.pos,
            closest_nodes=layer.closest_nodes_from_down,
            closest_weights=layer.closest_weights_from_down,
        )
        check_diff(diff, layer, precision=0.02)

        diff = layer_down.pos.double() - SceneLayers.approximate_internal(
            from_values=layer.pos,
            closest_nodes=layer.closest_nodes_to_down,
            closest_weights=layer.closest_weights_to_down,
        )
        check_diff(diff, layer, precision=0.1)


def test_graph_layers():
    output_catalog = "output/TEST_LAYERS"
    databases_main_path = f"{output_catalog}/DATA"
    log_catalog = f"{output_catalog}/LOG"

    td = TrainingData(
        dataset="synthetic",
        mesh_density=16,
        batch_size=16,
        synthetic_batches_in_epoch=2,
        mesh_layers_count=3,
        final_time=0.1,
        save_at_minutes=0,
        validate_at_epochs=1,
    )
    config = TrainingConfig(
        td=td,
        device="cpu",
        max_epoch_number=2,
        datasets_main_path=databases_main_path,
        dataset_images_count=1,
        with_train_scenes_file=True,
        output_catalog=output_catalog,
        log_catalog=log_catalog,
    )

    cmh.clear_folder(output_catalog)
    dataset = SyntheticDataset(
        description="train",
        layers_count=config.td.mesh_layers_count,
        load_features_to_ram=config.load_train_features_to_ram,
        load_targets_to_ram=config.load_train_targets_to_ram,
        with_scenes_file=config.with_train_scenes_file,
        randomize_at_load=True,
        config=config,
    )

    dataloader = base_dataset.get_train_dataloader(dataset)
    for _, layer_list in enumerate(dataloader):
        check_layer_data_approximation(layer_list)

    cmh.clear_folder(output_catalog)
