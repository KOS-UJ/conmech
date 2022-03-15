import deep_conmech.graph.data.data_interpolation as data_interpolation
import numpy as np
from conmech.helpers import mph
from deep_conmech.common import config
from deep_conmech.graph.data.data_base import *
from deep_conmech.graph.helpers import thh
from deep_conmech.graph.setting.setting_input import SettingInput
from deep_conmech.simulator.calculator import Calculator
from deep_conmech.simulator.setting.setting_forces import *
from torch_geometric.loader import DataLoader


def create_forces(setting):
    if data_interpolation.decide(config.DATA_ZERO_FORCES):
        forces = np.zeros([setting.nodes_count, setting.dim])
    else:
        forces = data_interpolation.interpolate_four(
            setting.nodes_count,
            setting.initial_nodes,
            config.FORCES_RANDOM_SCALE,
            setting.scale_x,
            setting.scale_y,
        )
    return forces


def create_u_old(setting):
    u_old = data_interpolation.interpolate_four(
        setting.nodes_count,
        setting.initial_nodes,
        config.U_RANDOM_SCALE,
        setting.scale_x,
        setting.scale_y,
    )
    return u_old


def create_v_old(setting):
    if data_interpolation.decide(config.DATA_ROTATE_VELOCITY):
        v_old = data_interpolation.interpolate_rotate(
            setting.nodes_count,
            setting.initial_nodes,
            config.V_RANDOM_SCALE,
            setting.scale_x,
            setting.scale_y,
        )
    else:
        v_old = data_interpolation.interpolate_four(
            setting.nodes_count,
            setting.initial_nodes,
            config.V_RANDOM_SCALE,
            setting.scale_x,
            setting.scale_y,
        )
    return v_old


def create_obstacles(setting):
    obstacle_normals_unnormaized = nph.get_random_normal_circle_numba(
        setting.dim, 1, config.OBSTACLE_ORIGIN_SCALE
    )
    obstacle_origins = -obstacle_normals_unnormaized + setting.mean_moved_nodes
    return np.stack((obstacle_normals_unnormaized, obstacle_origins))


def create_mesh_type():
    return data_interpolation.choose(
        ["pygmsh_rectangle", "pygmsh_circle", "pygmsh_spline", "pygmsh_polygon"]
    )


def create_obstacles(setting):
    obstacle_normals_unnormaized = nph.get_random_normal_circle_numba(
        setting.dim, 1, config.OBSTACLE_ORIGIN_SCALE
    )
    obstacle_origins = -obstacle_normals_unnormaized + setting.mean_moved_nodes
    return np.stack((obstacle_normals_unnormaized, obstacle_origins))


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





class TrainingSyntheticDatasetDynamic(BaseDatasetDynamic):
    def __init__(self, dim):
        super().__init__(
            dim=dim,
            relative_path="training_synthetic",
            data_count=config.SYNTHETIC_SOLVERS_COUNT,
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
                self.generate_random_data_process, (self, data_part_count), num_workers,
            )
            if result is False:
                print("Restarting data generation")
        


    def generate_setting(self, index):
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

        add_label = False
        # if Calculator.is_fast():
        if add_label:
            normalized_a = Calculator.solve_normalized(setting)
            exact_normalized_a_torch = thh.to_torch_double(normalized_a)
        else:
            exact_normalized_a_torch = None
            
        # data = setting.get_data(index, exact_normalized_a_torch)
        return setting, exact_normalized_a_torch  # data, setting






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
