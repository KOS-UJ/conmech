import gc
import os
import time
from torchaudio import datasets
import config 
import examples 
from drawer import Drawer
from helpers import *
from mesh import Mesh
from mesh_features import MeshFeatures 
from setting import *
import mapper




def create_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_final_path(catalog, forces_function_name, file_name, extension):
    final_path = "output"
    create_folder(final_path)
    final_path += f"/{helpers.CURRENT_TIME}"
    create_folder(final_path)
    final_path += f"/{catalog}"
    create_folder(final_path)
    #final_path += f"/{forces_function_name}"
    #create_folder(final_path)
    final_path += f"/{helpers.get_timestamp()} {forces_function_name} {file_name}.{extension}"
    return final_path


############################


def print_one_dynamic(
    solve_function,
    forces_function,
    catalog,
    simulate_dirty_data,
    print_base,
    print_max_data,
    description
):
    drawer = Drawer()
    all_images_paths = []
    extension = "png"  # pdf

    _print_at_interval = lambda time, setting, base_setting, a, base_a: print_at_interval(
        time,
        setting,
        get_final_path(
            catalog, forces_function.__name__, str(int(time * 100)), extension
        ),
        get_base_setting(base_setting, print_base),
        all_images_paths,
        extension,
    )
    mapper.map_time(
        _print_at_interval,
        config.PRINT_CORNERS,
        config.DRAW_EPISODE_STEPS,
        solve_function,
        forces_function,
        simulate_dirty_data,
        print_max_data,
        description,
    )

    animation_path = get_final_path(
        catalog, forces_function.__name__, "ANIMATION", "gif"
    )
    drawer.draw_animation(animation_path, all_images_paths)


def get_base_setting(base_setting, print_base):
    if(print_base):
        return base_setting
    else:
        return None


def print_at_interval(time, setting, path, base_setting, all_images_paths, extension):
    draw_skip = config.DRAW_SKIP
    if np.allclose(time % draw_skip, 0.0) or np.allclose(
        time % draw_skip, draw_skip
    ):  # TODO: clean
        print_setting(setting, path, base_setting, extension, time)
        all_images_paths.append(path)


def print_setting(setting, path, base_setting, extension, time):
    drawer = Drawer()
    ax = drawer.get_one_ax()
    if base_setting is None:
        height = 1
    else:
        height = 2
    drawer.draw_setting_ax(setting, ax, [3, height, 12, 3], base_setting, time)
    drawer.plt_save(path, extension)

###############################