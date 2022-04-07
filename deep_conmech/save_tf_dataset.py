import functools
import json
import os

import numpy as np
import tensorflow.compat.v1 as tf
from deep_conmech.graph.helpers import dch

from conmech.helpers import cmh, pkh
from conmech.helpers.config import Config
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios.scenarios import (Scenario, default_body_prop,
                                         default_obstacle_prop, f_rotate, m_polygon,
                                         o_side)
from conmech.simulations import simulation_runner
from conmech.solvers.calculator import Calculator
from deep_conmech.training_config import TrainingConfig

tf.enable_eager_execution()


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


def simulate(config: Config, scenario) -> str:
    _, data_path = simulation_runner.run_scenario(
        solve_function=Calculator.solve,
        scenario=scenario,
        catalog="SAVE_TF",
        config=config,
        save_all=True
    )
    return data_path


def prepare_data(config: TrainingConfig, data_path: str):
    all_indices = pkh.get_all_indices_pickle(data_path)
    data_count = len(all_indices)
    settings_file = pkh.open_file_settings_read_pickle(data_path)
    with settings_file:
        load_function = lambda index: pkh.load_index_pickle(index=index, all_indices=all_indices,
                                                            settings_file=settings_file)
        base_setting = load_function(index=0)
        elements = base_setting.elements[np.newaxis, ...].astype("int32")
        initial_nodes = base_setting.initial_nodes[np.newaxis, ...].astype("float32")
        node_type = np.zeros(
            (1, base_setting.nodes_count, 1), dtype="int32"
        )  # TODO: Mask boundary points

        moved_nodes_list = []
        forces_list = []

        for index in cmh.get_tqdm(range(data_count), config=config, desc="Preparing data to save"):
            setting = load_function(index=index)
            moved_nodes_list.append(setting.moved_nodes)
            forces_list.append(setting.forces)
    moved_nodes = np.array(moved_nodes_list, dtype="float32")
    forces = np.array(forces_list, dtype="float32")

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
        trajectory_length=data_count,
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
    config = TrainingConfig()
    dch.set_memory_limit(config=config)
    directory = "/home/michal/Desktop/DATA/conmech"
    cmh.recreate_folder(directory)

    scenario = Scenario(
        "polygon_rotate",
        MeshProperties(
            dimension=2,
            mesh_type=m_polygon,
            scale=[1],
            mesh_density=[32],
            is_adaptive=False,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=8.0),
        forces_function=f_rotate,
        obstacles=o_side,
    )

    data_path = simulate(config=config, scenario=scenario)
    meta, data = prepare_data(config=config, data_path=data_path)

    meta_path = os.path.join(directory, "meta.json")
    save_meta(meta, meta_path)

    for mode in ["train", "test", "valid"]:
        data_path = f"{directory}/{mode}.tfrecord"
        save_tf_data(data, data_path)

    print("DONE")


if __name__ == "__main__":
    main()
