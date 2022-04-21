"""
pickle helpers
"""
import pickle
import sys
from io import BufferedReader
from typing import Callable, Iterable, List


def open_files_append(path: str):
    return open(f"{path}.scenes", "ab+"), open(f"{path}.indices", "ab+")


def open_file_scenes_read(path: str):
    return open(f"{path}.scenes", "rb")


def get_all_indices(all_scenes_path):
    all_indices = []
    try:
        with open(f"{all_scenes_path}.indices", "rb") as file:
            try:
                while True:
                    all_indices.append(pickle.load(file))
            except EOFError:
                pass
    except IOError:
        pass
    return all_indices


def internal_load(scenes_file):
    # return pickle.load(scenes_file)

    state_dict = pickle.load(scenes_file)
    module_name = state_dict.pop("MODULE", None)
    class_name = state_dict.pop("CLASS", None)
    scene_class = getattr(sys.modules[module_name], class_name)  # __name__
    scene = scene_class.__new__(scene_class)
    scene.load_state_dict(state_dict)
    return scene


def append(scene, scenes_file: BufferedReader, file_meta: BufferedReader) -> None:
    index = scenes_file.tell()
    state_dict = scene.get_state_dict()
    state_dict["MODULE"] = scene.__module__
    state_dict["CLASS"] = scene.__class__.__name__
    pickle.dump(state_dict, scenes_file)  # self #copy.deepcopy(self)
    # pickle.dump(scene, scenes_file)
    pickle.dump(index, file_meta)


def load_index(index: int, all_indices: List[int], scenes_file: BufferedReader):
    byte_index = all_indices[index]
    scenes_file.seek(byte_index)
    scene = internal_load(scenes_file)
    return scene


def get_iterator(data_path: str, scene_tqdm: Iterable[int], preprocess_example: Callable):
    with open(f"{data_path}.scenes", "rb") as file:
        all_features_data = []
        all_target_data = []
        for index, _ in enumerate(scene_tqdm):
            features_data, target_data = preprocess_example(internal_load(file), index)
            all_features_data.append(features_data)
            all_target_data.append(target_data)
    return all_features_data, all_target_data

    # with open(f"{path}.scenes", "rb") as file:
    #     for _ in range(data_count):
    #         # try:
    #         yield internal_load_pickle(file)
    #         # except EOFError:
    #         #    break
