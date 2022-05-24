from typing import Optional

import numpy as np

import deep_conmech.data.interpolation_helpers as interpolation_helpers
from conmech.helpers import cmh, nph, pkh
from conmech.mesh.mesh_builders_helpers import get_random_corner_data
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios import scenarios
from conmech.scene.scene import Scene
from deep_conmech.data import base_dataset
from deep_conmech.data.base_dataset import BaseDataset
from deep_conmech.scene.scene_input import SceneInput
from deep_conmech.training_config import TrainingConfig


def generate_mesh_type(config: TrainingConfig):
    if config.td.dimension == 2:
        return interpolation_helpers.choose(
            [scenarios.M_RECTANGLE, scenarios.M_CIRCLE, scenarios.M_POLYGON]  # "pygmsh_spline"
        )
    else:
        return interpolation_helpers.choose(
            [scenarios.M_CUBE_3D, scenarios.M_BALL_3D, scenarios.M_POLYGON_3D]
        )


def generate_base_scene(base: np.ndarray, layers_count: int, config: TrainingConfig):
    corner_vectors = interpolation_helpers.get_corner_vectors_all(
        dimension=config.td.dimension, scale=config.td.initial_corners_scale
    )

    corner_mesh_data = get_random_corner_data() if config.td.adaptive_training_mesh else None
    scene = SceneInput(
        mesh_prop=MeshProperties(
            dimension=config.td.dimension,
            mesh_type=generate_mesh_type(config),
            mesh_density=[config.td.mesh_density],
            scale=[config.td.train_scale],
            initial_base=base,
            mean_at_origin=True,
            corners_vector=corner_vectors,
            corner_mesh_data=corner_mesh_data,
        ),
        body_prop=scenarios.default_body_prop,
        obstacle_prop=scenarios.default_obstacle_prop,
        schedule=Schedule(final_time=config.td.final_time),
        normalize_by_rotation=config.normalize_by_rotation,
        create_in_subprocess=False,
        layers_count=layers_count,
        with_schur=False,
    )
    scene.unset_randomization()
    return scene


def generate_forces(config: TrainingConfig, scene: Scene, base: np.ndarray):
    forces = interpolation_helpers.interpolate_four(
        nodes=scene.initial_nodes,
        mean_scale=config.td.forces_random_scale,
        corners_scale_proportion=config.td.corners_scale_proportion,
        base=base,
        interpolate_rotate=False,
        zero_out_proportion=config.td.zero_forces_proportion,
    )
    return forces


def generate_displacement_old(config: TrainingConfig, scene: Scene, base: np.ndarray):
    displacement_old = interpolation_helpers.interpolate_four(
        nodes=scene.initial_nodes,
        mean_scale=config.td.displacement_random_scale,
        corners_scale_proportion=config.td.corners_scale_proportion,
        base=base,
        interpolate_rotate=False,
        zero_out_proportion=config.td.zero_displacement_proportion,
    )
    return displacement_old


def generate_velocity_old(config: TrainingConfig, scene: Scene, base: np.ndarray):
    interpolate_rotate = interpolation_helpers.decide(config.td.rotate_velocity_proportion)
    velocity_old = interpolation_helpers.interpolate_four(
        nodes=scene.initial_nodes,
        mean_scale=config.td.velocity_random_scale,
        corners_scale_proportion=config.td.corners_scale_proportion,
        base=base,
        interpolate_rotate=interpolate_rotate,
        zero_out_proportion=config.td.zero_velocity_proportion,
    )
    return velocity_old


def generate_obstacles(config: TrainingConfig, scene: SceneInput):
    obstacle_nodes_unnormaized = nph.generate_uniform_circle(
        rows=1,
        columns=scene.dimension,
        low=config.td.obstacle_origin_min_scale,
        high=config.td.obstacle_origin_max_scale,
    )
    obstacle_nodes = obstacle_nodes_unnormaized + scene.mean_moved_nodes
    obstacle_normals_unnormaized = -obstacle_nodes_unnormaized
    return np.stack((obstacle_normals_unnormaized, obstacle_nodes))


def generate_base(config: TrainingConfig):
    dimension = config.td.dimension
    base = nph.generate_normal(rows=dimension, columns=dimension, scale=1)
    base = nph.normalize_euclidean_numba(base)
    base = nph.orthogonalize_gram_schmidt(base)
    base = nph.normalize_euclidean_numba(base)  # second time for numerical stability
    return base


class SyntheticDataset(BaseDataset):
    def __init__(
        self,
        description: str,
        layers_count: int,
        load_features_to_ram: bool,
        load_targets_to_ram: bool,
        randomize_at_load: bool,
        with_scenes_file: bool,
        config: TrainingConfig,
    ):
        num_workers = config.synthetic_generation_workers
        super().__init__(
            description=f"{description}_synthetic",
            dimension=config.td.dimension,
            data_count=config.td.batch_size * config.td.synthetic_batches_in_epoch,
            layers_count=layers_count,
            randomize_at_load=randomize_at_load,
            num_workers=num_workers,
            load_features_to_ram=load_features_to_ram,
            load_targets_to_ram=load_targets_to_ram,
            with_scenes_file=with_scenes_file,
            config=config,
        )

        if self.data_count % num_workers != 0:
            raise Exception("Cannot divide data generation work")
        self.scenes_part_count = int(self.data_count / num_workers)

        self.initialize_data()

    @property
    def data_size_id(self):
        return f"s:{self.data_count}_a:{self.config.td.adaptive_training_mesh}"

    def generate_scene(self, index: int):
        _ = index

        base = generate_base(self.config)
        scene = generate_base_scene(base=base, layers_count=self.layers_count, config=self.config)

        obstacles_unnormalized = generate_obstacles(self.config, scene)
        forces = generate_forces(self.config, scene, base)
        displacement_old = generate_displacement_old(self.config, scene, base)
        velocity_old = generate_velocity_old(self.config, scene, base)

        scene.normalize_and_set_obstacles(
            obstacles_unnormalized=obstacles_unnormalized, all_mesh_prop=[]
        )
        scene.set_displacement_old(displacement_old)
        scene.set_velocity_old(velocity_old)
        scene.prepare(forces)

        # exact_normalized_a_torch = thh.to_torch_double(Calculator.solve(scene))
        exact_normalized_a_torch = None
        return scene, exact_normalized_a_torch

    def generate_data_process(self, num_workers, process_id):
        assigned_data_range = base_dataset.get_process_data_range(
            process_id, self.scenes_part_count
        )
        tqdm_description = f"Process {process_id} - generating data"
        self.generate_data_internal(
            assigned_data_range=assigned_data_range,
            tqdm_description=tqdm_description,
            position=process_id,
        )

    def generate_data_simple(self):
        assigned_data_range = range(self.scenes_part_count)
        tqdm_description = "Generating data"
        self.generate_data_internal(
            assigned_data_range=assigned_data_range,
            tqdm_description=tqdm_description,
            position=None,
        )

    def generate_data_internal(
        self, assigned_data_range, tqdm_description: str, position: Optional[int]
    ):
        step_tqdm = cmh.get_tqdm(
            assigned_data_range,
            desc=tqdm_description,
            config=self.config,
            position=position,
        )

        scenes_file, indices_file = pkh.open_files_append(self.scenes_data_path)
        with scenes_file, indices_file:
            for index in step_tqdm:
                if base_dataset.is_memory_overflow(
                    config=self.config,
                    step_tqdm=step_tqdm,
                    tqdm_description=tqdm_description,
                ):
                    return False

                scene, exact_normalized_a_torch = self.generate_scene(index)
                _ = exact_normalized_a_torch

                self.save_scene(scene=scene, scenes_file=scenes_file, indices_file=indices_file)

                self.check_and_print(
                    len(assigned_data_range),
                    index,
                    scene,
                    step_tqdm,
                    tqdm_description,
                )

        step_tqdm.set_description(f"{step_tqdm.desc} - done")
        return True
