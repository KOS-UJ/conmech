import copy
import enum
import functools
import json
import os

import numpy as np
import tensorflow.compat.v1 as tf
from conmech.helpers import cmh
from deep_conmech.common import config, mapper
from deep_conmech.common.plotter import plotter_3d, plotter_mapper
from deep_conmech.common.plotter.plotter_2d import Plotter
from deep_conmech.graph.setting.setting_randomized import SettingRandomized
from deep_conmech.scenarios import *
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.setting.setting_forces import *

tf.enable_eager_execution()


#########


def _parse(proto, meta):
    feature_lists = {k: tf.io.VarLenFeature(tf.string) for k in meta["field_names"]}
    features = tf.io.parse_single_example(proto, feature_lists)
    out = {}
    for key, field in meta["features"].items():
        data = tf.io.decode_raw(features[key].values, getattr(tf, field["dtype"]))
        data = tf.reshape(data, field["shape"])
        if field["type"] == "static":
            data = tf.tile(data, [meta["trajectory_length"], 1, 1])
        elif field["type"] == "dynamic_varlen":
            length = tf.io.decode_raw(features["length_" + key].values, tf.int32)
            length = tf.reshape(length, [-1])
            data = tf.RaggedTensor.from_row_lengths(data, row_lengths=length)
        elif field["type"] != "dynamic":
            raise ValueError("invalid data format")
        out[key] = data
    return out


def save_tf_data(data, path: str):
    writer = tf.io.TFRecordWriter(path)

    out = tf.train.Example(features=tf.train.Features(feature=data))
    writer.write(out.SerializeToString())

    writer.close()


def save_meta(meta: dict, path: str):
    with open(path, "w") as file:
        json.dump(meta, file)


def load_meta(path: str):
    with open(path, "r") as file:
        meta = json.loads(file.read())
    return meta


def to_bytes(array):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[array.tobytes()]))


def to_dict(type, array):
    return dict(type=type, shape=[*array.shape], dtype=str(array.dtype))


def simulate(scenario, directory):
    all_images_paths = []
    all_figs = []
    all_settings = []
    extension = "png"
    images_directory = f"{directory}/saved_images"
    cmh.create_folders(images_directory)

    def save_example(time, setting, base_setting, a, base_a):
        all_settings.append(copy.deepcopy(setting))
        plotter_mapper.print_at_interval(
            time=time,
            setting=setting,
            path=f"{images_directory}/{scenario.id} {int(time * 100)}.{extension}",
            base_setting=None,
            draw_detailed=True,
            all_images_paths=all_images_paths,
            all_figs=all_figs,
            extension=extension,
            skip=config.PRINT_SKIP,
        )

    mapper.map_time(
        compare_with_base_setting=False,
        operation=save_example,
        solve_function=Calculator.solve,
        scenario=scenario,
        get_setting_function=SettingRandomized.get_setting,
        simulate_dirty_data=False,
        description="Performing simulation",
    )

    Plotter.draw_animation(
        f"{images_directory}/{scenario.id} scale_{scenario.mesh_data.scale_x} ANIMATION.gif",
        all_images_paths, all_figs
    )
    return all_settings


def prepare_data(all_settings):
    base_setting = all_settings[0]

    elements = base_setting.elements[np.newaxis, ...].astype("int32")
    initial_nodes = base_setting.initial_nodes[np.newaxis, ...].astype("float32")
    node_type = np.zeros(
        (1, base_setting.nodes_count, 1), dtype="int32"
    )  # TODO: Mask boundary points

    moved_nodes = np.array([s.moved_nodes for s in all_settings], dtype="float32")
    forces = np.array([s.forces for s in all_settings], dtype="float32")

    meta = dict(
        simulator="conmech",
        dt=base_setting.time_step,
        collision_radius=None,
        features=dict(
            cells=to_dict("static", elements),
            node_type=to_dict("static", node_type),
            mesh_pos=to_dict("static", initial_nodes),
            forces=to_dict("dynamic", forces),
            world_pos=to_dict("dynamic", moved_nodes),
        ),
        field_names=["cells", "node_type", "mesh_pos", "forces", "world_pos"],
        trajectory_length=len(all_settings),
    )

    data = dict(
        cells=to_bytes(elements),
        node_type=to_bytes(node_type),
        mesh_pos=to_bytes(initial_nodes),
        forces=to_bytes(forces),
        world_pos=to_bytes(moved_nodes),
    )

    return meta, data


def main():
    directory = "/home/michal/Desktop/DATA/conmech"  # "data/flag_simple"
    cmh.recreate_folder(directory)
    scenario = Scenario(
        "polygon_rotate",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[1],
            mesh_density=[8],
            is_adaptive=False,
        ),
        body_prop,
        obstacle_prop,
        time_data=TimeData(final_time=2.0),
        forces_function=f_rotate,
        obstacles=o_side,
    )

    # all_settings = []
    all_settings = simulate(scenario, directory)
    meta, data = prepare_data(all_settings)

    meta_path = os.path.join(directory, "meta.json")
    save_meta(meta, meta_path)

    for mode in ["train", "test", "valid"]:
        data_path = f"{directory}/{mode}.tfrecord"
        save_tf_data(data, data_path)

        inputs = load_data(meta_path, data_path)
        print_result(inputs, directory)


def print_result(inputs, directory):
    images_directory = f"{directory}/loaded_images"
    cmh.create_folders(images_directory)

    elements = inputs["cells"][0].numpy()
    all_moved_nodes = inputs["world_pos"].numpy()

    all_images_paths = []
    for i in range(len(all_moved_nodes)):
        time = (i+1)*0.01 #refactor
        if time % config.PRINT_SKIP == 0:
            moved_nodes = all_moved_nodes[i]
            path = f"{images_directory}/loaded_result {i}.png"
            all_images_paths.append(path)
            plotter_mapper.print_simple_data(elements, moved_nodes, path)
            
    Plotter.draw_animation(
        f"{images_directory}/loaded_result ANIMATION.gif", all_images_paths
    )


def load_data(meta_path, data_path):
    meta = load_meta(meta_path)

    ds = tf.data.TFRecordDataset(data_path)
    parser = functools.partial(_parse, meta=meta)
    first_input = tf.data.make_one_shot_iterator(ds).get_next()
    result = parser(first_input)

    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    ds = ds.prefetch(1)  # 10
    inputs = tf.data.make_one_shot_iterator(ds).get_next()
    return inputs


if __name__ == "__main__":
    main()
