import deep_conmech.graph.data.data_interpolation as data_interpolation
import numpy as np
from deep_conmech.common import config
from deep_conmech.graph.data.data_base import *
from deep_conmech.graph.helpers import thh
from conmech.helpers import mph
from deep_conmech.graph.setting.setting_input import SettingInput
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.setting.setting_forces import *
from torch_geometric.loader import DataLoader


def create_forces(setting):
    if data_interpolation.decide(config.DATA_ZERO_FORCES):
        forces = np.zeros([setting.nodes_count, config.DIM])
    else:
        forces = data_interpolation.interpolate_four(
            setting.nodes_count,
            setting.initial_nodes,
            config.FORCES_RANDOM_SCALE,
            setting.scale,
        )
    return forces


def create_u_old(setting):
    u_old = data_interpolation.interpolate_four(
        setting.nodes_count,
        setting.initial_nodes,
        config.U_RANDOM_SCALE,
        setting.scale,
    )
    return u_old


def create_v_old(setting):
    if data_interpolation.decide(config.DATA_ROTATE_VELOCITY):
        v_old = data_interpolation.interpolate_rotate(
            setting.nodes_count,
            setting.initial_nodes,
            config.V_RANDOM_SCALE,
            setting.scale,
        )
    else:
        v_old = data_interpolation.interpolate_four(
            setting.nodes_count,
            setting.initial_nodes,
            config.V_RANDOM_SCALE,
            setting.scale,
        )
    return v_old


def create_obstacles(setting):
    obstacle_normals_unnormaized = nph.get_random_normal_circle(
        1, config.OBSTACLE_ORIGIN_SCALE
    )
    obstacle_origins = -obstacle_normals_unnormaized + setting.mean_moved_points
    return np.stack((obstacle_normals_unnormaized, obstacle_origins))


def create_mesh_type():
    return data_interpolation.choose(
        ["pygmsh_rectangle", "pygmsh_circle", "pygmsh_spline", "pygmsh_polygon"]
    )


def create_obstacles(setting):
    obstacle_normals_unnormaized = thh.get_random_normal_circle(
        1, config.OBSTACLE_ORIGIN_SCALE
    )
    obstacle_origins = -obstacle_normals_unnormaized + setting.mean_moved_points
    return np.stack((obstacle_normals_unnormaized, obstacle_origins))


def create_mesh_type():
    return data_interpolation.choose(
        ["pygmsh_rectangle", "pygmsh_circle", "pygmsh_spline", "pygmsh_polygon"]
    )


def get_base_setting(mesh_type):
    return SettingInput(
        mesh_type=mesh_type,
        mesh_density_x=config.MESH_DENSITY,
        mesh_density_y=config.MESH_DENSITY,
        scale_x=config.TRAIN_SCALE,
        scale_y=config.TRAIN_SCALE,
        is_adaptive=config.ADAPTIVE_MESH,
        create_in_subprocess=False,
    )


def generate_random_setting_data(index):
    mesh_type = create_mesh_type()
    setting = get_base_setting(mesh_type)
    # setting.set_randomization(True)

    obstacles_unnormaized = create_obstacles(setting)
    forces = create_forces(setting)
    u_old = create_u_old(setting)
    v_old = create_v_old(setting)

    setting.set_obstacles(obstacles_unnormaized)
    setting.set_u_old(u_old)
    setting.set_v_old(v_old)
    setting.prepare(forces)

    # if Calculator.is_fast():
    normalized_a = Calculator.solve_normalized(setting)
    exact_normalized_a_torch = thh.to_torch_double(normalized_a)

    # data = setting.get_data(index, exact_normalized_a_torch)
    return setting, exact_normalized_a_torch  # data, setting


def generate_synthetic_data_process(dataset, data_part_count, queue, process_id):
    indices_to_do = get_process_indices_to_do(process_id, data_part_count, dataset.path)
    tqdm_description = f"Process {process_id} - generating {dataset.relative_path} data"
    step_tqdm = thh.get_tqdm(indices_to_do, desc=tqdm_description, position=process_id,)

    if not indices_to_do:
        return end_process(queue, step_tqdm, process_id, "done", True)

    for index in step_tqdm:
        if is_memory_overflow(step_tqdm, tqdm_description):
            return end_process(queue, step_tqdm, process_id, "memory overflow", False)

        setting, exact_normalized_a_torch = generate_random_setting_data(index)
        dataset.save(setting, exact_normalized_a_torch, index)
        dataset.check_and_print(0, index, setting, step_tqdm, tqdm_description)

    return end_process(queue, step_tqdm, process_id, "done", True)


class TrainingSyntheticDatasetDynamic(BaseDatasetDynamic):
    def __init__(self):
        super().__init__(
            relative_path="training_synthetic",
            data_count=config.TRAIN_SOLVERS_COUNT,
            randomize_at_load=True,
        )
        self.initialize_data()

    def generate_all_data(self):
        num_workers = config.GENERATION_WORKERS
        if self.data_count % num_workers != 0:
            raise Exception("Cannot divide data generation work")
        data_part_count = int(self.data_count / num_workers)

        result = False
        while result is False:
            result = mph.run_processes(
                generate_synthetic_data_process, (self, data_part_count), num_workers,
            )
            if result is False:
                print("Restarting data generation")


class StepDataset:
    def __init__(self, batch_size):
        self.all_data = [None] * batch_size

    def set(self, i, setting):
        self.all_data[i] = setting.data

    def __getitem__(self, index):
        return self.all_data[index]

    def __len__(self):
        return len(self.all_data)

    def get_dataloader(self):
        return DataLoader(
            dataset=self,
            batch_size=len(self),
            shuffle=False,
            # num_workers=1,
        )
