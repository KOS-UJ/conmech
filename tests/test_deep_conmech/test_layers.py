import torch

from conmech.helpers import cmh
from deep_conmech.data import base_dataset
from deep_conmech.data.interpolation_helpers import approximate_internal
from deep_conmech.data.synthetic_dataset import SyntheticDataset
from deep_conmech.training_config import TrainingConfig, TrainingData


# def check_layer_data_approximation(layer_list):
#     if len(layer_list) == 1:
#         return

#     def check_diff(diff, layer, precision):
#         diff_norm = torch.linalg.norm(diff, dim=1)
#         base_norm = torch.linalg.norm(layer.pos.double(), dim=1)
#         result = torch.sum(diff_norm) / torch.sum(base_norm)
#         assert result < precision

#     layer_base = layer_list[0]
#     for layer_number in range(1, len(layer_list)):
#         layer = layer_list[layer_number]
#         layer_down = layer_list[layer_number - 1]

#         diff = layer.pos.double() - approximate_internal(
#             base_values=layer_down.pos,
#             closest_nodes=layer.closest_nodes_from_down,
#             closest_weights=layer.closest_weights_from_down,
#         )
#         check_diff(diff, layer, precision=0.02)

#         diff = layer_down.pos.double() - approximate_internal(
#             base_values=layer.pos,
#             closest_nodes=layer.closest_nodes_to_down,
#             closest_weights=layer.closest_weights_to_down,
#         )
#         check_diff(diff, layer, precision=0.1)


# def test_graph_layers():
#     output_catalog = "output/TEST_LAYERS"
#     databases_main_path = f"{output_catalog}/DATA"
#     log_catalog = f"{output_catalog}/LOG"

#     td = TrainingData(
#         dataset="synthetic",
#         mesh_density=16,
#         batch_size=16,
#         dataset_size=32,
#         # mesh_layers_count=3,
#         final_time=0.1,
#         save_at_minutes=0,
#         validate_at_epochs=1,
#     )
#     config = TrainingConfig(
#         td=td,
#         device="cpu",
#         max_epoch_number=2,
#         datasets_main_path=databases_main_path,
#         dataset_images_count=1,
#         with_train_scenes_file=True,
#         output_catalog=output_catalog,
#         log_catalog=log_catalog,
#     )

#     cmh.clear_folder(output_catalog)
#     dataset = SyntheticDataset(
#         description="train",
#         with_scenes_file=config.with_train_scenes_file,
#         randomize=True,
#         config=config,
#         rank=0,
#         world_size=1,
#         load_data_to_ram=True
#     )

#     dataloader = base_dataset.get_train_dataloader(dataset, rank=0, world_size=1)
#     for _, layer_list in enumerate(dataloader):
#         check_layer_data_approximation(layer_list)

#     cmh.clear_folder(output_catalog)
