from ctypes import ArgumentError
from typing import Callable, List, Optional

import numpy as np

from conmech.helpers import cmh, pkh
from conmech.scenarios.scenarios import Scenario
from conmech.scene.scene import Scene
from deep_conmech.data.base_dataset import (
    BaseDataset,
    get_assigned_scenarios,
    is_memory_overflow,
)
from deep_conmech.helpers import thh
from deep_conmech.scene.scene_input import SceneInput
from deep_conmech.training_config import TrainingConfig


def check_and_get_dimension(scenarios):
    dimensions = set([s.mesh_prop.dimension for s in scenarios])
    if len(dimensions) != 1:
        raise ArgumentError("Incorrect data")
    return dimensions.pop()


class ScenariosDataset(BaseDataset):
    def __init__(
        self,
        description: str,
        all_scenarios: List[Scenario],
        skip_index: int,
        solve_function: Callable,
        load_features_to_ram: bool,
        load_targets_to_ram: bool,
        randomize_at_load: bool,
        config: TrainingConfig,
    ):
        self.all_scenarios = all_scenarios
        self.skip_index = skip_index
        self.solve_function = solve_function

        super().__init__(
            description=description,
            dimension=check_and_get_dimension(all_scenarios),
            data_count=self.get_data_count(self.all_scenarios),
            randomize_at_load=randomize_at_load,
            num_workers=1,  # TODO: #65 Check
            load_features_to_ram=load_features_to_ram,
            load_targets_to_ram=load_targets_to_ram,
            with_scenes_file=True,
            config=config,
        )
        self.initialize_data()

    def get_data_count(self, scenarios):
        return np.sum([int(s.schedule.episode_steps / self.skip_index) for s in scenarios])

    @property
    def data_size_id(self):
        return f"f:{self.config.td.final_time}_i:{self.skip_index}"

    def get_scene(self, scenario: Scenario, config: TrainingConfig) -> Scene:
        scene = SceneInput(
            mesh_prop=scenario.mesh_prop,
            body_prop=scenario.body_prop,
            obstacle_prop=scenario.obstacle_prop,
            schedule=scenario.schedule,
            normalize_by_rotation=config.normalize_by_rotation,
            create_in_subprocess=False,
        )
        scene.normalize_and_set_obstacles(scenario.linear_obstacles, scenario.mesh_obstacles)
        return scene

    def generate_data_process(self, num_workers, process_id):
        assigned_scenarios = get_assigned_scenarios(self.all_scenarios, num_workers, process_id)
        tqdm_description = f"P{process_id}: Generating {self.description}"
        self.generate_data_internal(
            assigned_scenarios=assigned_scenarios,
            tqdm_description=tqdm_description,
            position=process_id,
        )

    def generate_data_simple(self):
        tqdm_description = "Generating data"
        self.generate_data_internal(
            assigned_scenarios=self.all_scenarios,
            tqdm_description=tqdm_description,
            position=None,
        )

    def generate_data_internal(
        self, assigned_scenarios, tqdm_description: str, position: Optional[int]
    ):
        simulation_data_count = np.sum([s.schedule.episode_steps for s in self.all_scenarios])
        start_index = 0 if position is None else position * simulation_data_count
        current_index = start_index
        step_tqdm = cmh.get_tqdm(
            range(simulation_data_count),
            config=self.config,
            desc=tqdm_description,
            position=position,
        )
        scenario = assigned_scenarios[0]

        scenes_file, indices_file = pkh.open_files_append(self.scenes_data_path)
        with scenes_file, indices_file:
            for index in step_tqdm:
                episode_steps = scenario.schedule.episode_steps
                ts = (index % episode_steps) + 1
                if ts == 1:
                    scenario = assigned_scenarios[int(index / episode_steps)]
                    scene = self.get_scene(scenario=scenario, config=self.config)

                if is_memory_overflow(
                    config=self.config,
                    step_tqdm=step_tqdm,
                    tqdm_description=tqdm_description,
                ):
                    return False

                current_time = ts * scene.time_step
                forces = scenario.get_forces_by_function(scene, current_time)
                scene.prepare(forces)

                a, normalized_a = self.solve_function(scene)
                exact_normalized_a_torch = thh.to_torch_double(normalized_a)
                _ = exact_normalized_a_torch

                if index % self.skip_index == 0:
                    pkh.append_data(
                        data=scene, data_file=scenes_file, indices_file=indices_file
                    )  # exact_normalized_a_torch

                self.check_and_print(
                    simulation_data_count,
                    current_index,
                    scene,
                    step_tqdm,
                    tqdm_description,
                )

                # setting = setting.get_copy()
                scene.iterate_self(a)
                current_index += 1

        step_tqdm.set_description(f"{step_tqdm.desc} - done")
        return True
