"""
pickle helpers
"""
import pickle
import sys
from io import BufferedReader
from typing import List


def open_files_append_pickle(path: str):
        return open(f"{path}.settings", 'ab+'), open(f"{path}.indices", 'ab+')


def open_file_settings_read_pickle(path: str):
    return open(f"{path}.settings", 'rb')


def get_all_indices_pickle(all_settings_path):
    all_indices = []
    try:
        with open(f"{all_settings_path}.indices", 'rb') as file:
            try:
                while True:
                    all_indices.append(pickle.load(file))
            except EOFError:
                pass
    except:        
        pass
    return all_indices



def internal_load_pickle(settings_file):
    #return pickle.load(settings_file)
    
    state_dict = pickle.load(settings_file)
    module_name = state_dict.pop("MODULE", None)
    class_name = state_dict.pop("CLASS", None)
    setting_class = getattr(sys.modules[module_name], class_name) # __name__
    setting = setting_class.__new__(setting_class)
    setting.load_state_dict(state_dict)
    return setting
    


def append_pickle(setting, settings_file: BufferedReader, file_meta: BufferedReader) -> None:
    index = settings_file.tell()
    state_dict = setting.get_state_dict()
    state_dict["MODULE"] = setting.__module__
    state_dict["CLASS"] = setting.__class__.__name__
    pickle.dump(state_dict, settings_file) #self #copy.deepcopy(self)
    #pickle.dump(setting, settings_file)
    pickle.dump(index, file_meta)


def load_index_pickle(index: int, all_indices: List[int], settings_file: BufferedReader):
    byte_index = all_indices[index]
    settings_file.seek(byte_index)
    setting = internal_load_pickle(settings_file)
    return setting


def get_iterator_pickle(path: str, data_count: int):
    with open(f"{path}.settings", "rb") as file:
        for i in range(data_count):
            #try:
            yield internal_load_pickle(file)
            #except EOFError:
            #    break
