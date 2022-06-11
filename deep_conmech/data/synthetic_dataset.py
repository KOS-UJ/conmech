from queue import Empty

import numpy as np

import deep_conmech.data.interpolation_helpers as interpolation_helpers
from conmech.helpers import cmh, lnh, mph, nph
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios import scenarios
from conmech.scene.scene import Scene
from deep_conmech.data.base_dataset import BaseDataset
from deep_conmech.scene.scene_input import SceneInput
from deep_conmech.training_config import TrainingConfig


def generate_mesh_type(config: TrainingConfig):
    if config.td.dimension == 2:
        return interpolation_helpers.choose(
            [
                scenarios.M_RECTANGLE,
                scenarios.M_CIRCLE,
            ]  # , scenarios.M_POLYGON, scenarios.M_SPLINE]
        )
    else:
        return interpolation_helpers.choose(
            [scenarios.M_CUBE_3D, scenarios.M_BALL_3D]  # , scenarios.M_POLYGON_3D]
        )


def generate_base_scene(base: np.ndarray, layers_count: int, config: TrainingConfig):
    initial_nodes_corner_vectors = interpolation_helpers.generate_corner_vectors(
        dimension=config.td.dimension, scale=config.td.initial_corners_scale
    )
    mesh_corner_scalars = (
        None
        if config.td.adaptive_training_mesh_scale is None
        else interpolation_helpers.generate_mesh_corner_scalars(
            dimension=config.td.dimension, scale=config.td.adaptive_training_mesh_scale
        )
    )
    switch_orientation = interpolation_helpers.decide(0.5)
    scene = SceneInput(
        mesh_prop=MeshProperties(
            dimension=config.td.dimension,
            mesh_type=generate_mesh_type(config),
            mesh_density=[config.td.mesh_density],
            scale=[config.td.train_scale],
            initial_base=base,
            mean_at_origin=True,
            switch_orientation=switch_orientation,
            initial_nodes_corner_vectors=initial_nodes_corner_vectors,
            mesh_corner_scalars=mesh_corner_scalars,
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
    forces = interpolation_helpers.interpolate_corners(
        initial_nodes=scene.initial_nodes,
        mean_scale=config.td.forces_random_scale,
        corners_scale_proportion=config.td.corners_scale_proportion,
        base=base,
        zero_out_proportion=config.td.zero_forces_proportion,
    )
    return forces


def generate_displacement_old(config: TrainingConfig, scene: Scene, base: np.ndarray):
    displacement_old = interpolation_helpers.interpolate_corners(
        initial_nodes=scene.initial_nodes,
        mean_scale=config.td.displacement_random_scale,
        corners_scale_proportion=config.td.corners_scale_proportion,
        base=base,
        zero_out_proportion=config.td.zero_displacement_proportion,
    )
    return displacement_old


def generate_velocity_old(config: TrainingConfig, scene: Scene, base: np.ndarray):
    velocity_old = interpolation_helpers.interpolate_corners(
        initial_nodes=scene.initial_nodes,
        mean_scale=config.td.velocity_random_scale,
        corners_scale_proportion=config.td.corners_scale_proportion,
        base=base,
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


class SyntheticDataset(BaseDataset):
    def __init__(
        self,
        description: str,
        layers_count: int,
        randomize_at_load: bool,
        with_scenes_file: bool,
        config: TrainingConfig,
        rank: int,
        world_size: int,
    ):
        num_workers = config.synthetic_generation_workers
        super().__init__(
            description=f"{description}_synthetic",
            dimension=config.td.dimension,
            data_count=config.td.dataset_size,
            layers_count=layers_count,
            randomize_at_load=randomize_at_load,
            num_workers=num_workers,
            with_scenes_file=with_scenes_file,
            config=config,
            rank=rank,
            world_size=world_size,
        )

    @property
    def data_size_id(self):
        return f"s:{self.data_count}"

    def generate_scene(self):
        base = lnh.generate_base(self.config.td.dimension)
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
        return scene  # , exact_normalized_a_torch

    def generate_data_process(self, num_workers: int = 1, process_id: int = 0):
        assigned_data_range = self.get_process_data_range(
            data_count=self.data_count, process_id=process_id, num_workers=num_workers
        )
        tqdm_description = f"Process {process_id+1}/{num_workers} - generating data"
        step_tqdm = cmh.get_tqdm(
            assigned_data_range,
            desc=tqdm_description,
            config=self.config,
            position=process_id,
        )

        def generate_data_inner(queue):
            while not self.is_synthetic_generation_memory_overflow:
                queue.put(self.generate_scene())

        up_queue = mph.get_queue()
        inner_process = mph.start_process(generate_data_inner, up_queue)
        for index in step_tqdm:
            # scene = self.generate_scene()
            while True:
                try:
                    scene = up_queue.get(timeout=30.0)
                    break
                except Empty:
                    if not inner_process.is_alive():
                        print("Process terminated, restarting...")
                        inner_process = mph.start_process(generate_data_inner, up_queue)

            self.safe_save_scene(scene=scene, data_path=self.scenes_data_path)
            self.check_and_print(
                all_data_count=self.data_count,
                current_index=index,
                scene=scene,
                step_tqdm=step_tqdm,
                tqdm_description=tqdm_description,
            )
        inner_process.kill()

        step_tqdm.set_description(f"{step_tqdm.desc} - done")
        return True
