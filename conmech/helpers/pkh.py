"""
pickle helpers
"""
import pickle
import sys
from io import BufferedReader
from typing import Callable, Iterable, List


def open_files_append(path: str):
    return open(path, "ab+"), open(f"{path}_indices", "ab+")


def open_file_read(path: str):
    return open(path, "rb")


def get_all_indices(data_path):
    all_indices = []
    try:
        with open(f"{data_path}_indices", "rb") as file:
            try:
                while True:
                    all_indices.append(pickle.load(file))
            except EOFError:
                pass
    except IOError:
        pass
    return all_indices


def append_data(data, data_file: BufferedReader, indices_file: BufferedReader) -> None:
    index = data_file.tell()
    pickle.dump(data, data_file)
    pickle.dump(index, indices_file)


def load_index(index: int, all_indices: List[int], data_file: BufferedReader):
    byte_index = all_indices[index]
    data_file.seek(byte_index)
    data = pickle.load(data_file)
    return data


def get_iterator(scenes_path: str, data_tqdm: Iterable[int], preprocess_example: Callable):
    with open(scenes_path, "rb") as file:
        for index in data_tqdm:
            # try:
            features_data, target_data = preprocess_example(pickle.load(file), index)
            yield (features_data, target_data)
            # except EOFError:
            #    break
