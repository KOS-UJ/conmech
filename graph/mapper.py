import gc
import time
import config 
import examples 
from drawer import Drawer
from calculator import Calculator
from setting_input import SettingInput
from helpers import *
from mesh import Mesh
from mesh_features import MeshFeatures 
from setting import *



def map_time(
    operation,
    corners,
    episode_steps,
    solve_function,
    forces_function,
    simulate_dirty_data,
    print_max_data,
    description,
):
    mt, ms = config.MESH_TYPE, config.MESH_SIZE_PRINT
    setting = SettingInput(ms, mt, corners, is_adaptive=False, randomized_inputs=simulate_dirty_data)
    base_setting = SettingInput(ms, mt, corners, is_adaptive=False, randomized_inputs=False)
    max_data = MaxData(forces_function.__name__, episode_steps)

    time_tqdm = helpers.get_tqdm(range(episode_steps), description)
    for timestep in time_tqdm:
        time = (timestep + 1) * config.TIMESTEP

        setting.set_forces_from_function(forces_function, time)
        base_setting.set_forces_from_function(forces_function, time)
        max_data.set(setting, timestep)

        a = solve_function(setting)
        base_a = Calculator(base_setting).solve_function()  ## save in setting

        if simulate_dirty_data:
            setting.make_dirty()

        operation(time, setting, base_setting, a, base_a)

        setting.iterate_self(a, randomized_inputs=simulate_dirty_data)
        base_setting.iterate_self(base_a, randomized_inputs=False)

    if print_max_data:
        max_data.print()

    ######
    #del setting, base_setting
    #gc.collect()
