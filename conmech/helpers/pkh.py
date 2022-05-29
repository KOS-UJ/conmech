"""
pickle helpers
"""
import pickle
from io import BufferedReader
from threading import Lock
from typing import List


def open_files_write(path: str):
    return open(path, "wb+"), open(f"{path}_indices", "wb+")


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


def append_data(data, data_path: str, lock: Lock) -> None:
    return append_multiple_data(all_data=[data], all_data_paths=[data_path], lock=lock)


def append_multiple_data(all_data: List, all_data_paths: List[str], lock: Lock) -> None:
    with lock:
        for i, data in enumerate(all_data):
            data_file, indices_file = open_files_append(all_data_paths[i])
            with data_file, indices_file:
                index = data_file.tell()
                pickle.dump(data, data_file)
                pickle.dump(index, indices_file)


def load_index(index: int, all_indices: List[int], data_file: BufferedReader):
    byte_index = all_indices[index]
    return load_byte_index(byte_index=byte_index, data_file=data_file)


def load_byte_index(byte_index: int, data_file: BufferedReader):
    data_file.seek(byte_index)
    data = pickle.load(data_file)
    return data
