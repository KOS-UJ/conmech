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



def get_tf_data_path(directory):
    return f"{directory}/train.tfrecord"

def get_meta_path(directory):
    return os.path.join(directory, "meta.json")


def save_tf_data(data, directory):
    writer = tf.io.TFRecordWriter(get_tf_data_path(directory))

    out = tf.train.Example(features=tf.train.Features(feature=data))
    writer.write(out.SerializeToString())

    writer.close()


def save_meta(meta: dict, directory: str):
    with open(get_meta_path(directory), "w") as file:
        json.dump(meta, file)


def load_meta(directory: str):
    with open(get_meta_path(directory), "r") as file:
        meta = json.loads(file.read())
    return meta


def to_bytes(array):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[array.tobytes()]))


def to_dict(type, array):
    return dict(type=type, shape=[*array.shape], dtype=str(array.dtype))


def simulate(scenario, path):
    all_images_paths = []
    all_settings=[]
    extension = "png"
    images_path = f"{path}/images"
    cmh.create_folders(images_path)

    def save_example(time, setting, base_setting, a, base_a):
        all_settings.append(copy.deepcopy(setting))
        plotter_mapper.print_at_interval(
            time=time,
            setting=setting,
            path=f"{images_path}/{scenario.id} {int(time * 100)}.{extension}",
            base_setting=None,
            draw_detailed=True,
            all_images_paths=all_images_paths,
            extension=extension,
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
        f"{images_path}/{scenario.id} scale_{scenario.mesh_data.scale_x} ANIMATION.gif",
        all_images_paths,
    )
    return all_settings



def prepare_data(all_settings):
    time_step = 0.01
    dim = 2
    elements_count = 100
    nodes_count = 50
    episode_steps = 20

    elements = np.ones((1, elements_count, dim + 1), dtype="int32")
    initial_nodes = np.ones((1, nodes_count, dim), dtype="float32")
    moved_nodes = np.ones((episode_steps, nodes_count, dim), dtype="float32")

    meta = dict(
        simulator="conmech",
        dt=time_step,
        collision_radius=None,
        features=dict(
            cells=to_dict("static", elements),
            mesh_pos=to_dict("static", initial_nodes),
            world_pos=to_dict("dynamic", moved_nodes),
        ),
        field_names=["cells", "mesh_pos", "world_pos"],
        trajectory_length=episode_steps,
    )

    data = dict(
        cells=to_bytes(elements),
        mesh_pos=to_bytes(initial_nodes),
        world_pos=to_bytes(moved_nodes),
    )  # serialize_array(example)

    return meta, data



def main():
    directory = "datasets/data/conmech"  # "data/flag_simple"
    scenario = Scenario(
        "polygon_rotate",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[1],
            mesh_density=[3],
            is_adaptive=False,
        ),
        body_coeff,
        obstacle_coeff,
        time_data=TimeData(final_time=0.5),
        forces_function=f_rotate,
        obstacles=o_side,
    )

    #all_settings = []
    all_settings = simulate(scenario, directory)
    meta, data = prepare_data(all_settings)

    save_meta(meta, directory)
    save_tf_data(data, directory)

    load_data(directory, parse_all=False)


def load_data(directory, parse_all=True):
    meta = load_meta(directory)

    ds = tf.data.TFRecordDataset(get_tf_data_path(directory))
    if parse_all:
        ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
        ds = ds.prefetch(1)
    else:
        parser = functools.partial(_parse, meta=meta)
        first_input = tf.data.make_one_shot_iterator(ds).get_next()
        result = parser(first_input)

    a = 0


if __name__ == "__main__":
    main()
