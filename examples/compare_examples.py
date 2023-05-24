import os
import pickle
from datetime import datetime
from io import BufferedReader

import numpy as np
import pandas as pd
from tqdm import tqdm

from conmech.helpers import cmh, pkh
from conmech.helpers.config import Config, SimulationConfig
from conmech.scenarios.scenarios import bunny_fall_3d
from conmech.simulations import simulation_runner


def main():
    cmh.print_jax_configuration()

    simulation_config = SimulationConfig(
        use_linear_solver=False,
        use_normalization=False,
        use_green_strain=True,
        use_nonconvex_friction_law=False,
        use_constant_contact_integral=False,  # False,
        use_lhs_preconditioner=False,
        with_self_collisions=True,
        use_pca=False,
        mesh_layer_proportion=4,  # 2 8,
        mode="compare_net",
    )

    all_scenarios = [
        # all_train(config.td, config.sc)[0]
        bunny_fall_3d(
            mesh_density=32,
            scale=1,
            final_time=2.5,
            simulation_config=simulation_config,
            scale_forces=5.0,
        ),
    ]

    simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=True,
        config=Config(shell=False),
    )



##############

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

def load_simulation(simulation_path):
    all_indices = get_all_indices(simulation_path)
    simulation = []
    with open(simulation_path, "rb") as scenes_file:
        for byte_index in all_indices:
            scenes_file.seek(byte_index)
            data = pickle.load(scenes_file)
            simulation.append(data)
    return simulation


def get_simulation(scene_files, label):
    labels = [s for s in scene_files if label in s]
    # assert len(labels) == 1
    labels.sort()
    label=labels[-1]
    print(label)
    return load_simulation(label)


def get_error(simulation_1, simulation_2, index, key):
        return np.linalg.norm(simulation_1[index][key] - simulation_2[index][key])

def compare_latest(label=None):
    current_time: str = datetime.now().strftime("%m.%d-%H.%M.%S")
    
    input_path = 'output'

    all_scene_files = cmh.find_files_by_extension(input_path, "scenes")

    dense = True
    path_id = "/scenarios/" if dense else "/scenarios_reduced/"
    scene_files = [f for f in all_scene_files if path_id in f]
    # all_arrays_path = max(scene_files, key=os.path.getctime)

    normal = get_simulation(scene_files, 'normal')
    skinning = get_simulation(scene_files, 'skinning')
    net = get_simulation(scene_files, 'net')

    simulation_len = min(len(normal), len(skinning), len(net))
    for key in ['displacement_old', 'exact_acceleration', 'normalized_nodes']:
        errors_skinning = []
        errors_net= []
        for index in tqdm(range(simulation_len)):
            errors_skinning.append(get_error(skinning, normal, index, key))
            errors_net.append(get_error(net, normal, index, key))
                
        errors_df = pd.DataFrame(np.array([errors_skinning, errors_net]).T, columns=['skinning', 'net'])
        plot = errors_df.plot()
        fig = plot.get_figure()
        fig.savefig(f"output/{current_time}_{label}_dense:{dense}_errors_{key}.png")
        print(errors_df)

    ####

    dense = False
    path_id = "/scenarios/" if dense else "/scenarios_reduced/"
    scene_files = [f for f in scene_files if path_id in f]

    skinning = get_simulation(all_scene_files, 'skinning')
    net = get_simulation(all_scene_files, 'net')

    simulation_len = min(len(normal), len(skinning), len(net))
    for key in ['displacement_old', 'exact_acceleration', 'normalized_nodes']:
        errors_reduced = []
        for index in tqdm(range(simulation_len)):
            errors_reduced.append(get_error(skinning, net, index, key))
                
        errors_df = pd.DataFrame(np.array([errors_reduced]).T, columns=['reduced'])
        plot = errors_df.plot()
        fig = plot.get_figure()
        fig.savefig(f"output/{current_time}_{label}_dense:{dense}_errors_{key}.png")
        print(errors_df)



if __name__ == "__main__":
    # main()
    compare_latest()