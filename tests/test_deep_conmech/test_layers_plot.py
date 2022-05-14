from email.mime import base

import numpy as np
import torch

from conmech.helpers import cmh
from conmech.plotting import plotter_2d, plotter_common
from deep_conmech.data import base_dataset
from deep_conmech.data.synthetic_dataset import SyntheticDataset
from deep_conmech.helpers import thh
from deep_conmech.scene import scene_layers
from deep_conmech.scene.scene_layers import SceneLayers
from deep_conmech.training_config import TrainingConfig, TrainingData


def get_axs():
    fig = plotter_2d.get_fig()
    axs = plotter_2d.get_axs(fig)

    axs.set_aspect("equal", "box")
    plotter_common.set_ax(axs)
    return axs


def draw_nodes(nodes, ptr, axs, shift):
    position = np.array([0, -1.5])
    description_offset = np.array([-0.05, -0.05])

    scene = 0
    for i, initial_node in enumerate(nodes):
        if i == ptr[scene + 1]:
            scene += 1
        node = initial_node + scene * position + np.array([shift, 0])
        axs.annotate(str(i), node + description_offset, color="w", fontsize=0.1)
        axs.scatter(node[0], node[1], s=0.05, marker=".", c="tab:orange")

    # plotter_2d.draw_nodes(nodes, position, "orange", axs)


def save_plot(description):
    plotter_common.plt_save(f"./output/LAYERS {description}.png", "png")


def plot_graph_layers():
    output_catalog = "output/TEST_TMP"
    databases_main_path = f"{output_catalog}/DATA"
    log_catalog = f"{output_catalog}/LOG"

    td = TrainingData(
        dataset="synthetic",
        mesh_density=8,
        adaptive_training_mesh=False,
        batch_size=1,
        synthetic_batches_in_epoch=1,
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
    for batch_number, layer_list in enumerate(dataloader):
        base_dataset.order_batch_layer_indices(layer_list)
        up_layer = layer_list[1]
        down_layer = layer_list[0]
        _ = """
        ptr = up_layer.ptr
        nodes_1 = up_layer.pos.numpy()
        nodes_2 = SceneLayers.approximate_internal(
            from_values=down_layer.pos,
            closest_nodes=up_layer.closest_nodes_from_down,
            closest_weights=up_layer.closest_weights_from_down,
        ).numpy()
        axs = get_axs()
        draw_nodes(nodes_1, ptr, axs=axs, shift=0)
        draw_nodes(nodes_2, ptr, axs=axs, shift=1.5)
        save_plot(f"batch_number={batch_number} up_layer")
        """

        ptr = down_layer.ptr
        nodes_1 = down_layer.pos.double()
        base_nodes = up_layer.pos.numpy()[:4]
        closest_nodes, closest_weights = scene_layers.get_interlayer_data(
            old_nodes=base_nodes, new_nodes=down_layer.pos.numpy(), closest_count=3
        )
        nodes_2 = SceneLayers.approximate_internal(
            from_values=base_nodes,
            closest_nodes=closest_nodes,
            closest_weights=closest_weights,
        )
        nodes_3 = base_nodes
        _ = """
        nodes_2 = SceneLayers.approximate_internal(
            from_values=up_layer.pos,
            closest_nodes=up_layer.closest_nodes_to_down,
            closest_weights=up_layer.closest_weights_to_down,
        )
        """
        axs = get_axs()
        draw_nodes(nodes_1, ptr, axs=axs, shift=0)
        draw_nodes(nodes_2, ptr, axs=axs, shift=1.5)
        draw_nodes(nodes_3, ptr, axs=axs, shift=3.0)

        save_plot(f"batch_number={batch_number} down_layer")

        a = 0

    cmh.clear_folder(output_catalog)


plot_graph_layers()
