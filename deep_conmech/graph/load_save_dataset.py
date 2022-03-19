import enum
import functools
import json
import os

import numpy as np
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


######################


def to_bytes(array):
    return _bytes_feature(array.tobytes())


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    return tf.io.serialize_tensor(array)


#################################


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


def write_data_to_tf(data, datapath):
    writer = tf.io.TFRecordWriter(datapath)

    out = tf.train.Example(features=tf.train.Features(feature=data))
    writer.write(out.SerializeToString())

    writer.close()
    print("Wrote data to TFRecord")


def save_meta(path, meta):
    with open(os.path.join(path, "meta.json"), "w") as file:
        json.dump(meta, file)


def load_meta(path):
    with open(os.path.join(path, "meta.json"), "r") as file:
        meta = json.loads(file.read())
    return meta

def to_dict(type, array):
    return dict(type=type, shape=[*array.shape], dtype=str(array.dtype))


def main():
    path = "datasets/data/conmech"  # "data/flag_simple"
    datapath = os.path.join(path, "train.tfrecord")

    time_step=0.01
    dim = 2
    elements_count = 100
    nodes_count=50
    episode_steps=20

    elements = np.ones((1, elements_count, dim+1), dtype="int32")
    elements[0, 1, 1] = 10
    initial_nodes = np.ones((1, nodes_count, dim), dtype="float32")
    moved_nodes = np.ones((episode_steps, nodes_count, dim), dtype="float32")

    meta_start = dict(
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
        cells=to_bytes(elements), mesh_pos=to_bytes(initial_nodes), world_pos=to_bytes(moved_nodes),
    )  # serialize_array(example)

    save_meta(path, meta_start)
    write_data_to_tf(data, datapath)

    ###
    meta = load_meta(path)

    ds = tf.data.TFRecordDataset(datapath)
    ds = ds.map(functools.partial(_parse, meta=meta), num_parallel_calls=8)
    
    #parser = functools.partial(_parse, meta=meta)
    #first_input = tf.data.make_one_shot_iterator(ds).get_next()
    #parser(first_input)

    ds = ds.prefetch(1)
    a = 0


if __name__ == "__main__":
    main()


################################3


image_small_shape = (250, 250, 3)
number_of_images_small = 100

images_small = np.random.randint(
    low=0, high=256, size=(number_of_images_small, *image_small_shape), dtype=np.int16
)
print(images_small.shape)

labels_small = np.random.randint(low=0, high=5, size=(number_of_images_small))  ###, 1))
print(labels_small.shape)
print(labels_small[:10])


FILEPATH = "./data/conmech/train.tfrecord"


count = write_images_to_tfr_short(images_small, labels_small, filepath=FILEPATH)


#####


def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "raw_image": tf.io.FixedLenFeature([], tf.string),
        "depth": tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data)
    height = content["height"]
    width = content["width"]
    depth = content["depth"]
    label = content["label"]
    raw_image = content["raw_image"]

    # get our 'feature'-- our image -- and reshape it appropriately
    feature = tf.io.parse_tensor(raw_image, out_type=tf.int16)
    feature = tf.reshape(feature, shape=[height, width, depth])
    return (feature, label)


def get_dataset_small(filepath):
    # create the dataset
    dataset = tf.data.TFRecordDataset(filepath)

    # pass every single feature through our mapping function
    dataset = dataset.map(parse_tfr_element)

    return dataset


dataset_small = get_dataset_small(FILEPATH)

for sample in dataset_small.take(1):
    print(sample[0].shape)
    print(sample[1].shape)

