import numpy as np

import deep_conmech.data.interpolation_helpers as interpolation_helpers
from conmech.helpers import nph, cmh, pkh
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios import scenarios
from conmech.solvers.calculator import Calculator
from deep_conmech.data import base_dataset
from deep_conmech.data.base_dataset import BaseDataset
from deep_conmech.graph.setting.setting_input import SettingInput
from deep_conmech.helpers import thh
from deep_conmech.training_config import TrainingConfig

def create_mesh_type():
    return interpolation_helpers.choose(
        ["pygmsh_rectangle", "pygmsh_circle", "pygmsh_spline", "pygmsh_polygon"]
    )


def create_forces(config, setting):
    if interpolation_helpers.decide(config.td.DATA_ZERO_FORCES):
        forces = np.zeros([setting.nodes_count, setting.dimension])
    else:
        forces = interpolation_helpers.interpolate_four(
            count=setting.nodes_count,
            initial_nodes=setting.initial_nodes,
            randomization_scale=config.td.FORCES_RANDOM_SCALE,
            setting_scale_x=setting.mesh_data.scale_x,
            setting_scale_y=setting.mesh_data.scale_y,
        )
    return forces


def create_u_old(config, setting):
    u_old = interpolation_helpers.interpolate_four(
        count=setting.nodes_count,
        initial_nodes=setting.initial_nodes,
        randomization_scale=config.td.U_RANDOM_SCALE,
        setting_scale_x=setting.mesh_data.scale_x,
        setting_scale_y=setting.mesh_data.scale_y,
    )
    return u_old


def create_v_old(config, setting):
    function = interpolation_helpers.interpolate_rotate if interpolation_helpers.decide(
        config.td.DATA_ROTATE_VELOCITY) else interpolation_helpers.interpolate_four
    v_old = function(
        setting.nodes_count,
        setting.initial_nodes,
        config.td.V_RANDOM_SCALE,
        setting.mesh_data.scale_x,
        setting.mesh_data.scale_y,
    )
    return v_old


def create_obstacles(config, setting):
    obstacle_normals_unnormaized = nph.get_random_normal_circle_numba(
        setting.dimension, 1, config.td.OBSTACLE_ORIGIN_SCALE
    )
    obstacle_origins = -obstacle_normals_unnormaized + setting.mean_moved_nodes
    return np.stack((obstacle_normals_unnormaized, obstacle_origins))


def get_base_setting(config, mesh_type):
    return SettingInput(
        mesh_data=MeshProperties(
            mesh_type=mesh_type,
            mesh_density=[config.td.MESH_DENSITY],
            scale=[config.td.TRAIN_SCALE],
            is_adaptive=config.td.ADAPTIVE_TRAINING_MESH,
        ),
        body_prop=scenarios.default_body_prop,
        obstacle_prop=scenarios.default_obstacle_prop,
        schedule=Schedule(final_time=config.td.FINAL_TIME),
        config=config,
        create_in_subprocess=False,
    )


class SyntheticDataset(BaseDataset):
    def __init__(self, description: str, dimension: int, load_to_ram: bool, config: TrainingConfig):
        num_workers = config.SYNTHETIC_GENERATION_WORKERS

        if self.data_count % num_workers != 0:
            raise Exception("Cannot divide data generation work")
        self.data_part_count = int(self.data_count / num_workers)

        super().__init__(
            description=f"{description}_synthetic",
            dimension=dimension,
            randomize_at_load=True,
            num_workers=num_workers,
            load_to_ram=load_to_ram,
            config=config
        )
        self.initialize_data()

    @property
    def data_count(self):
        return TrainingConfig.td.SYNTHETIC_SOLVERS_COUNT

    def generate_setting(self, index):
        mesh_type = create_mesh_type()
        setting = get_base_setting(self.config, mesh_type)
        setting.set_randomization(False)  # TODO #65: Check

        obstacles_unnormaized = create_obstacles(self.config, setting)
        forces = create_forces(self.config, setting)
        u_old = create_u_old(self.config, setting)
        v_old = create_v_old(self.config, setting)

        setting.normalize_and_set_obstacles(obstacles_unnormaized)
        setting.set_u_old(u_old)
        setting.set_v_old(v_old)
        setting.prepare(forces)

        add_label = False
        exact_normalized_a_torch = thh.to_torch_double(
            Calculator.solve(setting)) if add_label else None

        return setting, exact_normalized_a_torch

    def generate_data_process(self, num_workers, process_id):
        assigned_data_range = base_dataset.get_process_data_range(process_id, self.data_part_count)

        tqdm_description = (
            f"Process {process_id} - generating {self.data_id} data"
        )
        step_tqdm = cmh.get_tqdm(
            assigned_data_range, desc=tqdm_description, config=self.config, position=process_id,
        )

        settings_file, file_meta = pkh.open_files_append_pickle(self.data_path)
        with settings_file, file_meta:
            for index in step_tqdm:
                if base_dataset.is_memory_overflow(config=self.config, step_tqdm=step_tqdm,
                                                   tqdm_description=tqdm_description):
                    return False

                setting, exact_normalized_a_torch = self.generate_setting(index)
                pkh.append_pickle(setting=setting, settings_file=settings_file,
                                  file_meta=file_meta)  # exact_normalized_a_torch

                self.check_and_print(
                    len(assigned_data_range), index, setting, step_tqdm, tqdm_description
                )

        step_tqdm.set_description(f"{step_tqdm.desc} - done")
        return True
