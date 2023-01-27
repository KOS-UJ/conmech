from ctypes import ArgumentError
from typing import Callable, List, Optional

import numpy as np

from conmech.helpers import cmh
from conmech.scenarios.scenarios import Scenario
from conmech.scene.energy_functions import EnergyFunctions
from conmech.scene.scene import Scene
from conmech.solvers.calculator import Calculator
from deep_conmech.data.base_dataset import BaseDataset
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
        use_jax: bool,
        all_scenarios: List[Scenario],
        solve_function: Callable,
        load_data_to_ram: bool,
        with_scenes_file: bool,
        randomize: bool,
        config: TrainingConfig,
        rank: int,
        world_size: int,
        device_count: int,
        item_fn,
    ):
        self.all_scenarios = all_scenarios

        super().__init__(
            description=description,
            use_jax=use_jax,
            dimension=check_and_get_dimension(all_scenarios),
            data_count=self.get_data_count(self.all_scenarios),
            solve_function=solve_function,
            load_data_to_ram=load_data_to_ram,
            randomize=randomize,
            num_workers=config.scenario_generation_workers,
            with_scenes_file=with_scenes_file,
            config=config,
            rank=rank,
            world_size=world_size,
            device_count=device_count,
            item_fn=item_fn,
        )

    def get_data_count(self, scenarios):
        return np.sum([int(s.schedule.episode_steps) for s in scenarios])

    @property
    def data_size_id(self):
        return f"f:{self.config.td.final_time}"

    def get_assigned_scenarios(self, num_workers, process_id):
        scenarios_count = len(self.all_scenarios)
        if scenarios_count % num_workers != 0:
            raise Exception("Cannot divide data generation work")
        assigned_scenarios_count = int(scenarios_count / num_workers)
        assigned_scenarios = self.all_scenarios[
            process_id * assigned_scenarios_count : (process_id + 1) * assigned_scenarios_count
        ]
        return assigned_scenarios

    def get_scene(self, scenario: Scenario, config: TrainingConfig) -> Scene:
        scene = SceneInput(
            mesh_prop=scenario.mesh_prop,
            body_prop=scenario.body_prop,
            obstacle_prop=scenario.obstacle_prop,
            schedule=scenario.schedule,
            simulation_config=scenario.simulation_config,
            create_in_subprocess=False,
        )
        scene.set_randomization(config)

        scene.normalize_and_set_obstacles(scenario.linear_obstacles, scenario.mesh_obstacles)
        return scene

    def print_stats(self, scene):
        print(len(scene.initial_nodes))
        print(len(scene.reduced.initial_nodes))
        print(np.min(scene.initial_nodes, axis=0))
        print(np.max(scene.initial_nodes, axis=0))

    def generate_data(self):
        self.generate_data_process()
        # mph.run_process(self.generate_data_process)
        # done = mph.run_processes(self.generate_data_process, num_workers=self.num_workers)
        # if not done:
        #     print("NOT DONE")

    def generate_data_process(self, num_workers: int = 1, process_id: int = 0):
        assigned_scenarios = self.get_assigned_scenarios(num_workers, process_id)
        tqdm_description = f"Generating data - process {process_id+1}/{num_workers}"
        simulation_data_count = np.sum([s.schedule.episode_steps for s in assigned_scenarios])
        start_index = process_id * simulation_data_count
        current_index = start_index
        step_tqdm = cmh.get_tqdm(
            range(simulation_data_count),
            config=self.config,
            desc=tqdm_description,
            position=process_id,
        )
        scenario = assigned_scenarios[0]

        for index in step_tqdm:
            episode_steps = scenario.schedule.episode_steps
            ts = (index % episode_steps) + 1
            if ts == 1:
                scenario = assigned_scenarios[int(index / episode_steps)]
                scene = self.get_scene(scenario=scenario, config=self.config)
                energy_functions = EnergyFunctions(simulation_config=scene.simulation_config)

            current_time = ts * scene.time_step

            forces = scenario.get_forces_by_function(scene, current_time)
            scene, acceleration = self.solve_and_prepare_scene(scene, forces, energy_functions)

            if self.with_scenes_file:
                self.safe_save_scene(scene=scene, data_path=self.scenes_data_path)
            else:
                self.save_features_and_target(scene=scene)

            self.check_and_print(
                self.data_count, current_index, scene, step_tqdm, tqdm_description, current_time
            )
            current_index += 1

            scene.iterate_self(acceleration)

        step_tqdm.set_description(f"{step_tqdm.desc} - done")
        return True
