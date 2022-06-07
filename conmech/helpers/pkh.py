"""
pickle helpers
"""
import pickle
from io import BufferedReader
from threading import Lock
from typing import List, Optional


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


def append_data(data, data_path: str, lock: Optional[Lock]) -> None:
    def append_data_internal():
        data_file, indices_file = open_files_append(data_path)
        with data_file, indices_file:
            index = data_file.tell()
            pickle.dump(data, data_file)
            pickle.dump(index, indices_file)

    if lock is None:
        append_data_internal()
    else:
        with lock:
            append_data_internal()


def load_index(index: int, all_indices: List[int], data_file: BufferedReader):
    return load_byte_index(byte_index=all_indices[index], data_file=data_file)


def load_byte_index(byte_index: int, data_file: BufferedReader):
    data_file.seek(byte_index)
    data = pickle.load(data_file)
    return data
