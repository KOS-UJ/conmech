import numpy as np

import deep_conmech.data.interpolation_helpers as interpolation_helpers
from conmech.helpers import cmh, nph, pkh
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios import scenarios
from conmech.solvers.calculator import Calculator
from deep_conmech.data import base_dataset
from deep_conmech.data.base_dataset import BaseDataset
from deep_conmech.graph.scene.scene_input import SceneInput
from deep_conmech.helpers import thh
from deep_conmech.training_config import TrainingConfig


def draw_mesh_type(config: TrainingConfig):
    if config.td.dimension == 2:
        return interpolation_helpers.choose(
            [scenarios.M_RECTANGLE, scenarios.M_CIRCLE, scenarios.M_POLYGON]  # "pygmsh_spline"
        )
    else:
        return interpolation_helpers.choose(
            [scenarios.M_CUBE_3D, scenarios.M_BALL_3D, scenarios.M_POLYGON_3D]
        )


def draw_base_setting(config: TrainingConfig, base: np.ndarray):
    return SceneInput(
        mesh_prop=MeshProperties(
            dimension=config.td.dimension,
            mesh_type=draw_mesh_type(config),
            mesh_density=[config.td.mesh_density],
            scale=[config.td.train_scale],
            is_adaptive=config.td.adaptive_training_mesh,
            initial_base=base,
        ),
        body_prop=scenarios.default_body_prop,
        obstacle_prop=scenarios.default_obstacle_prop,
        schedule=Schedule(final_time=config.td.final_time),
        config=config,
        create_in_subprocess=False,
    )


def draw_forces(config: TrainingConfig, setting, base: np.ndarray):
    if interpolation_helpers.decide(config.td.zero_forces_proportion):
        forces = np.zeros([setting.nodes_count, setting.dimension])
    else:
        forces = interpolation_helpers.interpolate_four(
            initial_nodes=setting.initial_nodes,
            scale=config.td.forces_random_scale,
            corners_scale_proportion=config.td.corners_scale_proportion,
            mesh_prop=setting.mesh_prop,
            base=base,
            interpolate_rotate=False,
        )
    return forces


def draw_displacement_old(config: TrainingConfig, setting, base: np.ndarray):
    displacement_old = interpolation_helpers.interpolate_four(
        initial_nodes=setting.initial_nodes,
        scale=config.td.displacement_random_scale,
        corners_scale_proportion=config.td.corners_scale_proportion,
        mesh_prop=setting.mesh_prop,
        base=base,
        interpolate_rotate=False,
    )
    return displacement_old


def draw_velocity_old(config: TrainingConfig, setting, base: np.ndarray):
    interpolate_rotate = interpolation_helpers.decide(config.td.rotate_velocity_proportion)
    velocity_old = interpolation_helpers.interpolate_four(
        initial_nodes=setting.initial_nodes,
        scale=config.td.velocity_random_scale,
        corners_scale_proportion=config.td.corners_scale_proportion,
        mesh_prop=setting.mesh_prop,
        base=base,
        interpolate_rotate=interpolate_rotate,
    )
    return velocity_old


def draw_obstacles(config: TrainingConfig, scene: SceneInput):
    obstacle_nodes_unnormaized = nph.draw_uniform_circle(
        rows=1,
        columns=scene.dimension,
        low=config.td.obstacle_min_scale,
        high=config.td.obstacle_origin_scale,
    )
    obstacle_nodes = obstacle_nodes_unnormaized + scene.mean_moved_nodes
    obstacle_normals_unnormaized = -obstacle_nodes_unnormaized
    return np.stack((obstacle_normals_unnormaized, obstacle_nodes))


def draw_base(config: TrainingConfig):
    dimension = config.td.dimension
    base = nph.draw_normal_circle(rows=dimension, columns=dimension, scale=1)
    base = nph.normalize_euclidean_numba(base)
    base = nph.orthogonalize_gram_schmidt(base)
    base = nph.normalize_euclidean_numba(base)  # second time for numerical stability
    return base


class SyntheticDataset(BaseDataset):
    def __init__(
        self,
        description: str,
        load_to_ram: bool,
        randomize_at_load: bool,
        config: TrainingConfig,
    ):
        num_workers = config.synthetic_generation_workers
        super().__init__(
            description=f"{description}_synthetic",
            dimension=config.td.dimension,
            data_count=config.td.batch_size * config.td.synthetic_batches_in_epoch,
            randomize_at_load=randomize_at_load,
            num_workers=num_workers,
            load_to_ram=load_to_ram,
            config=config,
        )

        if self.data_count % num_workers != 0:
            raise Exception("Cannot divide data generation work")
        self.data_part_count = int(self.data_count / num_workers)

        self.initialize_data()

    @property
    def data_size_id(self):
        return f"s:{self.data_count}_a:{self.config.td.adaptive_training_mesh}"

    def generate_scene(self, index):
        _ = index

        base = draw_base(self.config)
        setting = draw_base_setting(self.config, base)
        setting.set_randomization(False)  # TODO #65: Check

        obstacles_unnormalized = draw_obstacles(self.config, setting)
        forces = draw_forces(self.config, setting, base)
        displacement_old = draw_displacement_old(self.config, setting, base)
        velocity_old = draw_velocity_old(self.config, setting, base)

        setting.normalize_and_set_obstacles(
            obstacles_unnormalized=obstacles_unnormalized, all_mesh_prop=[]
        )
        setting.set_displacement_old(displacement_old)
        setting.set_velocity_old(velocity_old)
        setting.prepare(forces)

        add_label = False
        exact_normalized_a_torch = (
            thh.to_torch_double(Calculator.solve(setting)) if add_label else None
        )

        return setting, exact_normalized_a_torch

    def generate_data_process(self, num_workers, process_id):
        assigned_data_range = base_dataset.get_process_data_range(process_id, self.data_part_count)

        tqdm_description = f"Process {process_id} - generating {self.data_id} data"
        step_tqdm = cmh.get_tqdm(
            assigned_data_range,
            desc=tqdm_description,
            config=self.config,
            position=process_id,
        )

        scenes_file, file_meta = pkh.open_files_append(self.data_path)
        with scenes_file, file_meta:
            for index in step_tqdm:
                if base_dataset.is_memory_overflow(
                    config=self.config,
                    step_tqdm=step_tqdm,
                    tqdm_description=tqdm_description,
                ):
                    return False

                scene, exact_normalized_a_torch = self.generate_scene(index)
                pkh.append(
                    scene=scene, scenes_file=scenes_file, file_meta=file_meta
                )  # exact_normalized_a_torch

                self.check_and_print(
                    len(assigned_data_range),
                    index,
                    scene,
                    step_tqdm,
                    tqdm_description,
                )

        step_tqdm.set_description(f"{step_tqdm.desc} - done")
        return True
