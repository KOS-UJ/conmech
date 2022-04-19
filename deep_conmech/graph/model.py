import json
import time
from ctypes import ArgumentError
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch.utils.tensorboard.writer import SummaryWriter

from conmech.helpers import cmh, nph
from conmech.scenarios import scenarios
from conmech.scenarios.scenarios import Scenario
from conmech.scene.scene import EnergyObstacleArguments
from conmech.simulations import simulation_runner
from conmech.solvers.calculator import Calculator
from deep_conmech.data import base_dataset
from deep_conmech.data.dataset_statistics import FeaturesStatistics
from deep_conmech.graph.logger import Logger
from deep_conmech.graph.net import CustomGraphNet
from deep_conmech.graph.scene import scene_input
from deep_conmech.graph.scene.scene_input import SceneInput
from deep_conmech.helpers import thh
from deep_conmech.training_config import TrainingConfig


class ErrorResult:
    value = 0


class GraphModelDynamic:
    def __init__(
        self,
        train_dataset,
        all_val_datasets,
        print_scenarios: List[Scenario],
        net: CustomGraphNet,
        config: TrainingConfig,
    ):
        self.config = config
        self.all_val_datasets = all_val_datasets
        self.dim = train_dataset.dimension  # TODO: Check validation datasets
        self.train_dataset = train_dataset
        self.print_scenarios = print_scenarios
        self.loss_labels = [
            "energy",
            # "energy_diff",
            # "RMSE_acc",
        ]  # "energy_diff", "energy_no_acc"]  # . "energy_main", "v_step_diff"]
        self.labels_count = len(self.loss_labels)
        self.tqdm_loss_index = 0

        self.net = net
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.config.td.initial_learning_rate,  # weight_decay=5e-4
        )
        lr_lambda = lambda epoch: max(
            self.config.td.learning_rate_decay**epoch,
            self.config.td.final_learning_rate / self.config.td.initial_learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self.logger = Logger(dataset=self.train_dataset, config=config)
        self.logger.save_parameters_and_statistics()

    @property
    def lr(self):
        return float(self.scheduler.get_last_lr()[0])

    def graph_sizes(self, batch):
        graph_sizes = np.ediff1d(thh.to_np_long(batch.ptr)).tolist()
        return graph_sizes

    def boundary_nodes_counts(self, batch):
        return thh.to_np_long(batch.boundary_nodes_count).tolist()

    def get_split(self, batch, index, dim, graph_sizes):
        value = batch.x[:, index * dim : (index + 1) * dim]
        value_split = value.split(graph_sizes)
        return value_split

    def train(self):
        # epoch_tqdm = tqdm(range(config.EPOCHS), desc="EPOCH")
        # for epoch in epoch_tqdm:
        start_time = time.time()
        last_save_time = start_time
        examples_seen = 0
        epoch_number = 0
        print("----TRAINING----")
        while self.config.max_epoch_number is None or epoch_number < self.config.max_epoch_number:
            epoch_number += 1
            # with profile(with_stack=True, profile_memory=True) as prof:

            loss_array, es = self.iterate_dataset(
                dataset=self.train_dataset,
                dataloader_function=base_dataset.get_train_dataloader,
                step_function=self.train_step,
                description=f"EPOCH: {epoch_number}",  # , lr: {self.lr:.6f}",
            )
            examples_seen += es
            self.training_raport(loss_array=loss_array, examples_seen=examples_seen)

            self.scheduler.step()

            current_time = time.time()
            elapsed_time = current_time - last_save_time
            if elapsed_time > self.config.td.save_at_minutes * 60:
                # print(f"--Training time: {(elapsed_time / 60):.4f} min")
                self.save_net()
                last_save_time = time.time()

            if epoch_number % self.config.td.validate_at_epochs == 0:
                self.validation_raport(examples_seen=examples_seen)
            if epoch_number % self.config.td.validate_scenarios_at_epochs == 0:
                self.validate_all_scenarios_raport(examples_seen=examples_seen)

            # print(prof.key_averages().table(row_limit=10))

    def save_net(self):
        print("----SAVING----")
        timestamp = cmh.get_timestamp(self.config)
        catalog = f"{self.config.output_catalog}/{self.config.current_time} - GRAPH MODELS"
        cmh.create_folders(catalog)
        path = f"{catalog}/{timestamp} - MODEL.pt"
        self.net.save(path)

    @staticmethod
    def get_newest_saved_model_path(config: TrainingConfig):
        def get_index(path):
            return int(path.split("/")[-1].split(" ")[0])

        saved_model_paths = cmh.find_files_by_extension(config.output_catalog, "pt")
        if not saved_model_paths:
            raise ArgumentError("No saved models")

        newest_index = np.argmax(np.array([get_index(path) for path in saved_model_paths]))
        path = saved_model_paths[newest_index]

        print(f"Taking saved model {path.split('/')[-1]}")
        return path

    @staticmethod
    def get_setting_function(
        scenario: Scenario,
        config: TrainingConfig,
        randomize=False,
        create_in_subprocess: bool = False,
    ) -> SceneInput:  # "Scene":
        setting = SceneInput(
            mesh_prop=scenario.mesh_prop,
            body_prop=scenario.body_prop,
            obstacle_prop=scenario.obstacle_prop,
            schedule=scenario.schedule,
            config=config,
            create_in_subprocess=create_in_subprocess,
        )
        setting.set_randomization(randomize)
        setting.normalize_and_set_obstacles(scenario.linear_obstacles, scenario.mesh_obstacles)
        return setting

    @staticmethod
    def plot_all_scenarios(net: CustomGraphNet, print_scenarios, config: TrainingConfig):
        print("----PLOTTING----")
        start_time = time.time()
        timestamp = cmh.get_timestamp(config)
        catalog = f"GRAPH PLOT/{timestamp} - RESULT"
        for scenario in print_scenarios:
            simulation_runner.run_scenario(
                solve_function=net.solve,
                scenario=scenario,
                config=config,
                run_config=simulation_runner.RunScenarioConfig(
                    catalog=catalog,
                    simulate_dirty_data=False,
                    compare_with_base_setting=config.compare_with_base_setting,
                    plot_animation=True,
                ),
                get_setting_function=GraphModelDynamic.get_setting_function,
            )
            print("---")
        print(f"Plotting time: {int((time.time() - start_time) / 60)} min")
        # return catalog

    def train_step(self, batch):
        self.net.train()
        self.net.zero_grad()

        loss, loss_array_np = self.E(batch)
        loss.backward()
        if self.config.td.gradient_clip is not None:
            self.clip_gradients(self.config.td.gradient_clip)
        self.optimizer.step()

        return loss_array_np

    def test_step(self, batch):
        self.net.eval()

        with torch.no_grad():  # with tc.set_grad_enabled(train):
            _, loss_array_np = self.E(batch)

        return loss_array_np

    def clip_gradients(self, max_norm: float):
        parameters = self.net.parameters()
        # norms = [np.max(np.abs(p.grad.cpu().detach().numpy())) for p in parameters]
        # total_norm = np.max(norms)_
        # print("total_norm", total_norm)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def iterate_dataset(self, dataset, dataloader_function, step_function, description):
        dataloader = dataloader_function(dataset)
        batch_tqdm = cmh.get_tqdm(dataloader, desc=description, config=self.config)

        examples_seen = 0
        mean_loss_array = np.zeros(self.labels_count)
        for _, batch in enumerate(batch_tqdm):
            # len(batch) ?
            loss_array = step_function(batch)

            old_examples_seen = examples_seen
            examples_seen += batch.num_graphs

            mean_loss_array = mean_loss_array * (old_examples_seen / examples_seen) + loss_array * (
                batch.num_graphs / examples_seen
            )

            batch_tqdm.set_description(
                f"{description} loss: {(mean_loss_array[self.tqdm_loss_index]):.4f}"
            )
        return mean_loss_array, examples_seen

    def training_raport(self, loss_array, examples_seen):
        self.logger.writer.add_scalar(
            "Loss/Training/LearningRate",
            self.lr,
            examples_seen,
        )
        for i, loss in enumerate(loss_array):
            self.logger.writer.add_scalar(
                f"Loss/Training/{self.loss_labels[i]}",
                loss,
                examples_seen,
            )

    def validation_raport(self, examples_seen):
        print("----VALIDATING----")
        start_time = time.time()

        for dataset in self.all_val_datasets:
            loss_array, _ = self.iterate_dataset(
                dataset=dataset,
                dataloader_function=base_dataset.get_valid_dataloader,
                step_function=self.test_step,
                description=dataset.data_id,
            )
            for i in range(self.labels_count):
                self.logger.writer.add_scalar(
                    f"Loss/Validation/{dataset.data_id}/{self.loss_labels[i]}",
                    loss_array[i],
                    examples_seen,
                )
        validation_time = time.time() - start_time
        print(f"--Validation time: {(validation_time / 60):.4f} min")

    def validate_all_scenarios_raport(self, examples_seen):
        print("----VALIDATING SCENARIOS----")
        start_time = time.time()
        total_mean_energy = 0.0
        for scenario in self.print_scenarios:
            _, _, mean_energy = simulation_runner.run_scenario(
                solve_function=self.net.solve,
                scenario=scenario,
                config=self.config,
                run_config=simulation_runner.RunScenarioConfig(),
                get_setting_function=GraphModelDynamic.get_setting_function,
            )
            self.logger.writer.add_scalar(
                f"Loss/Validation/{scenario.name}/mean_energy",
                mean_energy,
                examples_seen,
            )
            total_mean_energy += mean_energy / len(self.print_scenarios)
            print("---")

        self.logger.writer.add_scalar(
            f"Loss/Validation/total_mean_energy",
            total_mean_energy,
            examples_seen,
        )
        print(f"--Validating scenarios time: {int((time.time() - start_time) / 60)} min")

    def E(self, batch, test_using_true_solution=False):
        # graph_couts = [1 for i in range(batch.num_graphs)]
        graph_sizes = self.graph_sizes(batch)
        boundary_nodes_counts = self.boundary_nodes_counts(batch)
        dim_graph_sizes = [size * self.dim for size in graph_sizes]
        dim_dim_graph_sizes = [(size * self.dim) ** 2 for size in graph_sizes]

        loss = 0.0
        loss_array = np.zeros(self.labels_count)

        batch_cuda = batch.to(self.net.device)
        predicted_normalized_a_split = self.net(batch_cuda).split(graph_sizes)

        reshaped_C_split = batch.reshaped_C.split(dim_dim_graph_sizes)
        normalized_E_split = batch.normalized_E.split(dim_graph_sizes)
        normalized_a_correction_split = batch.normalized_a_correction.split(graph_sizes)
        normalized_boundary_velocity_old_split = batch.normalized_boundary_velocity_old.split(
            boundary_nodes_counts
        )
        normalized_boundary_nodes_split = batch.normalized_boundary_nodes.split(
            boundary_nodes_counts
        )
        normalized_boundary_normals_split = batch.normalized_boundary_normals.split(
            boundary_nodes_counts
        )

        normalized_boundary_obstacle_nodes_split = batch.normalized_boundary_obstacle_nodes.split(
            boundary_nodes_counts
        )
        normalized_boundary_obstacle_normals_split = (
            batch.normalized_boundary_obstacle_normals.split(boundary_nodes_counts)
        )
        surface_per_boundary_node_split = batch.surf_per_boundary_node.split(boundary_nodes_counts)

        if hasattr(batch, "exact_normalized_a"):
            exact_normalized_a_split = batch.exact_normalized_a.split(graph_sizes)

        for i in range(batch.num_graphs):
            C_side_len = graph_sizes[i] * self.dim
            C = reshaped_C_split[i].reshape(C_side_len, C_side_len)
            normalized_E = normalized_E_split[i]
            normalized_a_correction = normalized_a_correction_split[i]
            predicted_normalized_a = predicted_normalized_a_split[i]

            energy_args = dict(
                a_correction=normalized_a_correction,
                args=EnergyObstacleArguments(
                    lhs=C,
                    rhs=normalized_E,
                    boundary_velocity_old=normalized_boundary_velocity_old_split[i],
                    boundary_nodes=normalized_boundary_nodes_split[i],
                    boundary_normals=normalized_boundary_normals_split[i],
                    boundary_obstacle_nodes=normalized_boundary_obstacle_nodes_split[i],
                    boundary_obstacle_normals=normalized_boundary_obstacle_normals_split[i],
                    surface_per_boundary_node=surface_per_boundary_node_split[i],
                    obstacle_prop=scenarios.default_obstacle_prop,  # TODO: generalize
                    time_step=0.01,  # TODO: generalize
                ),
            )

            if test_using_true_solution:
                predicted_normalized_a = self.use_true_solution(predicted_normalized_a, energy_args)

            predicted_normalized_energy = scene_input.energy_normalized_obstacle_correction(
                cleaned_a=predicted_normalized_a, **energy_args
            )
            if hasattr(batch, "exact_normalized_a"):
                exact_normalized_a = exact_normalized_a_split[i]

            if self.config.td.use_energy_as_loss:
                loss += predicted_normalized_energy
            else:
                loss += thh.rmse_torch(predicted_normalized_a, exact_normalized_a)

            loss_array[0] += predicted_normalized_energy
            if hasattr(batch, "exact_normalized_a"):
                exact_normalized_energy = scene_input.energy_normalized_obstacle_correction(
                    cleaned_a=exact_normalized_a, **energy_args
                )
                loss_array[1] += float(
                    (predicted_normalized_energy - exact_normalized_energy)
                    / torch.abs(exact_normalized_energy)
                )
                loss_array[2] += float(thh.rmse_torch(predicted_normalized_a, exact_normalized_a))

        loss /= batch.num_graphs
        loss_array /= batch.num_graphs
        return loss, loss_array

    def use_true_solution(self, predicted_normalized_a, energy_args):
        function = lambda normalized_a_vector: scene_input.energy_normalized_obstacle_correction(
            cleaned_a=thh.to_torch_double(nph.unstack(normalized_a_vector, dim=2)).to(
                self.net.device
            ),
            **energy_args,
        ).item()

        # @v = function(thh.to_np_double(torch.zeros_like(predicted_normalized_a)))
        predicted_normalized_a = thh.to_torch_double(
            nph.unstack(
                Calculator.minimize(
                    function,
                    thh.to_np_double(torch.zeros_like(predicted_normalized_a)),
                ),
                dim=2,
            )
        ).to(self.net.device)
        return predicted_normalized_a
