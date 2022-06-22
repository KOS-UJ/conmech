from ctypes import ArgumentError

import numpy as np

from conmech.helpers import cmh
from conmech.plotting import plotter_2d, plotter_common
from deep_conmech.data import base_dataset
from deep_conmech.data.interpolation_helpers import approximate_internal
from deep_conmech.data.synthetic_dataset import SyntheticDataset
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


def plot_comparison(from_layer, to_layer, approximated_nodes, description):
    def get_ptr(nodes):
        if len(nodes) == from_layer.ptr[-1]:
            return from_layer.ptr
        if len(nodes) == to_layer.ptr[-1]:
            return to_layer.ptr
        raise ArgumentError

    axs = get_axs()
    draw_nodes(from_layer.pos.numpy(), get_ptr(from_layer.pos), axs=axs, shift=0)
    draw_nodes(approximated_nodes, get_ptr(approximated_nodes), axs=axs, shift=1.5)
    draw_nodes(to_layer.pos.numpy(), get_ptr(to_layer.pos), axs=axs, shift=3.0)
    save_plot(description)


def plot_graph_layers():
    output_catalog = "output/TEST_TMP"
    cmh.clear_folder(output_catalog)
    dataset = get_dataset(output_catalog)

    dataloader = base_dataset.get_train_dataloader(dataset, rank=0, world_size=1)
    for batch_number, layer_list in enumerate(dataloader):
        for layer_number in range(1, len(layer_list)):
            up_layer = layer_list[layer_number]
            down_layer = layer_list[layer_number - 1]
            desc = f"batch_number={batch_number} layer_number={layer_number}"

            approximated_nodes = approximate_internal(
                base_values=up_layer.pos,
                closest_nodes=up_layer.closest_nodes_to_down,
                closest_weights=up_layer.closest_weights_to_down,
            )
            plot_comparison(
                from_layer=up_layer,
                to_layer=down_layer,
                approximated_nodes=approximated_nodes,
                description=f"{desc} to_down",
            )

            approximated_nodes = approximate_internal(
                base_values=down_layer.pos,
                closest_nodes=up_layer.closest_nodes_from_down,
                closest_weights=up_layer.closest_weights_from_down,
            )
            plot_comparison(
                from_layer=down_layer,
                to_layer=up_layer,
                approximated_nodes=approximated_nodes,
                description=f"{desc} from_down",
            )

    cmh.clear_folder(output_catalog)


def get_dataset(output_catalog):
    databases_main_path = f"{output_catalog}/DATA"
    log_catalog = f"{output_catalog}/LOG"

    td = TrainingData(
        dataset="synthetic",
        mesh_density=8,
        adaptive_training_mesh_scale=0,
        batch_size=3,
        dataset_size=2,
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
    dataset = SyntheticDataset(
        description="train",
        layers_count=config.td.mesh_layers_count,
        with_scenes_file=config.with_train_scenes_file,
        randomize_at_load=True,
        config=config,
    )

    return dataset


if __name__ == "__main__":
    plot_graph_layers()
