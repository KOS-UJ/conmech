import functools
import json
import os

import tensorflow.compat.v1 as tf

from deep_conmech.common import simulation_runner
from deep_conmech.graph.helpers import dch
from deep_conmech.scenarios import *
from deep_conmech.simulator.setting.setting_forces import *
from deep_conmech.simulator.solver import Solver

tf.enable_eager_execution()


#########


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


#########

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


def simulate(scenario):
    return simulation_runner.plot_scenario(
        solve_function=Solver.solve,
        scenario=scenario,
        catalog="SAVE_TF",
        save_all=True
    )


def prepare_data(all_setting_paths):
    # all_settings = [SettingIterable.load_pickle(path) for path in all_setting_paths]
    base_setting = SettingIterable.load_pickle(all_setting_paths[0])

    elements = base_setting.elements[np.newaxis, ...].astype("int32")
    initial_nodes = base_setting.initial_nodes[np.newaxis, ...].astype("float32")
    node_type = np.zeros(
        (1, base_setting.nodes_count, 1), dtype="int32"
    )  # TODO: Mask boundary points

    moved_nodes_list = []
    forces_list = []
    for path in cmh.get_tqdm(all_setting_paths, desc="Preparing data to save"):
        setting = SettingIterable.load_pickle(path)
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
        trajectory_length=len(all_setting_paths),
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
        MeshData(
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

    all_setting_paths = simulate(scenario)
    meta, data = prepare_data(all_setting_paths)

    meta_path = os.path.join(directory, "meta.json")
    save_meta(meta, meta_path)

    for mode in ["train", "test", "valid"]:
        data_path = f"{directory}/{mode}.tfrecord"
        save_tf_data(data, data_path)


if __name__ == "__main__":
    main()
