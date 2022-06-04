import time
from ctypes import ArgumentError
from typing import List

import numpy as np
import torch
from torch_geometric.data.batch import Data

from conmech.helpers import cmh
from conmech.scenarios.scenarios import Scenario
from conmech.simulations import simulation_runner
from deep_conmech.data import base_dataset
from deep_conmech.graph.logger import Logger
from deep_conmech.graph.loss_raport import LossRaport
from deep_conmech.graph.net import CustomGraphNet
from deep_conmech.helpers import thh
from deep_conmech.scene import scene_input
from deep_conmech.scene.scene_input import SceneInput
from deep_conmech.training_config import TrainingConfig


class ErrorResult:
    value = 0


def get_graph_sizes(batch):
    return np.ediff1d(thh.to_np_long(batch.ptr)).tolist()


class GraphModelDynamic:
    def __init__(
        self,
        train_dataset,
        all_val_datasets,
        print_scenarios: List[Scenario],
        net: CustomGraphNet,
        config: TrainingConfig,
    ):
        print("----CREATING MODEL----")
        self.config = config
        self.all_val_datasets = all_val_datasets
        self.dim = train_dataset.dimension  # TODO: Check validation datasets
        self.train_dataset = train_dataset
        self.print_scenarios = print_scenarios
        self.loss_labels = [
            "energy",
            "mean"
            # "energy_diff",
            # "RMSE_acc",
        ]  # "energy_diff", "energy_no_acc"]  # . "energy_main", "v_step_diff"]
        self.labels_count = len(self.loss_labels)

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

            mean_loss_raport, es = self.iterate_dataset(
                dataset=self.train_dataset,
                dataloader_function=base_dataset.get_train_dataloader,
                step_function=self.train_step,
                description=f"EPOCH: {epoch_number}",  # , lr: {self.lr:.6f}",
            )
            examples_seen += es
            self.training_raport(mean_loss_raport=mean_loss_raport, examples_seen=examples_seen)

            self.scheduler.step()

            current_time = time.time()
            elapsed_time = current_time - last_save_time
            if elapsed_time > self.config.td.save_at_minutes * 60:
                # print(f"--Training time: {(elapsed_time / 60):.4f} min")
                self.save_net()
                last_save_time = time.time()

            def is_at_skip(skip):
                return skip is not None and epoch_number % skip == 0

            if is_at_skip(self.config.td.validate_at_epochs):
                self.validation_raport(examples_seen=examples_seen)
            if is_at_skip(self.config.td.validate_scenarios_at_epochs):
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
    def get_scene_function(
        scenario: Scenario,
        config: TrainingConfig,
        randomize=False,
        create_in_subprocess: bool = False,
    ) -> SceneInput:
        scene = SceneInput(
            mesh_prop=scenario.mesh_prop,
            body_prop=scenario.body_prop,
            obstacle_prop=scenario.obstacle_prop,
            schedule=scenario.schedule,
            normalize_by_rotation=config.normalize_by_rotation,
            layers_count=config.td.mesh_layers_count,
            create_in_subprocess=create_in_subprocess,
        )
        if randomize:
            scene.set_randomization(config)
        else:
            scene.unset_randomization()
        scene.normalize_and_set_obstacles(scenario.linear_obstacles, scenario.mesh_obstacles)
        return scene

    @staticmethod
    def plot_all_scenarios(
        net: CustomGraphNet, print_scenarios: List[Scenario], config: TrainingConfig
    ):
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
                    compare_with_base_scene=config.compare_with_base_scene,
                    plot_animation=True,
                ),
                get_scene_function=GraphModelDynamic.get_scene_function,
            )
            print("---")
        print(f"Plotting time: {int((time.time() - start_time) / 60)} min")
        # return catalog

    def train_step(self, layer_list: List[Data], dataset):
        self.net.train()
        self.net.zero_grad()

        main_loss, loss_raport = self.calculate_loss(
            layer_list=layer_list, layer_number=0, dataset=dataset
        )
        main_loss.backward()
        if self.config.td.gradient_clip is not None:
            self.clip_gradients(self.config.td.gradient_clip)
        self.optimizer.step()

        return loss_raport

    def test_step(self, layer_list: List[Data], dataset):
        self.net.eval()

        with torch.no_grad():  # with tc.set_grad_enabled(train):
            _, loss_raport = self.calculate_loss(
                layer_list=layer_list, layer_number=0, dataset=dataset
            )

        return loss_raport

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
        mean_loss_raport = LossRaport()
        for _, layer_list in enumerate(batch_tqdm):
            # len(batch) ?

            loss_raport = step_function(layer_list, dataset)
            mean_loss_raport.add(loss_raport)

            examples_seen += layer_list[0].num_graphs

            batch_tqdm.set_description(f"{description} loss: {(mean_loss_raport.main):.4f}")
        return mean_loss_raport, examples_seen

    def training_raport(self, mean_loss_raport, examples_seen):
        self.logger.writer.add_scalar(
            "Loss/Training/LearningRate",
            self.lr,
            examples_seen,
        )
        for key, value in mean_loss_raport.get_iterator():
            self.logger.writer.add_scalar(
                f"Loss/Training/{key}",
                value,
                examples_seen,
            )

    def validation_raport(self, examples_seen):
        # print("----VALIDATING----")
        start_time = time.time()

        for dataset in self.all_val_datasets:
            mean_loss_raport, _ = self.iterate_dataset(
                dataset=dataset,
                dataloader_function=base_dataset.get_valid_dataloader,
                step_function=self.test_step,
                description=dataset.data_id,
            )
            for key, value in mean_loss_raport.get_iterator():
                self.logger.writer.add_scalar(
                    f"Loss/Validating/{key}",
                    value,
                    examples_seen,
                )
        # validation_time = time.time() - start_time
        # print(f"--Validation time: {(validation_time / 60):.4f} min")

    def validate_all_scenarios_raport(self, examples_seen):
        print("----VALIDATING SCENARIOS----")
        start_time = time.time()
        episode_steps = self.print_scenarios[0].schedule.episode_steps
        all_energy_values = np.zeros(episode_steps)
        for scenario in self.print_scenarios:
            assert episode_steps == scenario.schedule.episode_steps
            _, _, energy_values = simulation_runner.run_scenario(
                solve_function=self.net.solve,
                scenario=scenario,
                config=self.config,
                run_config=simulation_runner.RunScenarioConfig(),
                get_scene_function=GraphModelDynamic.get_scene_function,
            )
            # time_step = scenario.time_step
            # integrated_energy = np.sum(time_step * energy_values) /  scenario.episode_length
            _ = """
            self.logger.writer.add_scalar(
                f"Loss/Validation/{scenario.name}/mean_energy_all",
                mean_energy,
                examples_seen,
            )
            """
            all_energy_values += energy_values / len(self.print_scenarios)
            print("---")

        for i in [1, 10, 50, 100, 200, 800]:
            self.logger.writer.add_scalar(
                f"Loss/Validation/energy_mean_{i}_steps",
                np.mean(all_energy_values[:i]),
                examples_seen,
            )

        print(f"--Validating scenarios time: {int((time.time() - start_time) / 60)} min")

    def calculate_loss(
        self, layer_list: List[Data], layer_number: int, dataset: base_dataset.BaseDataset
    ):
        dimension = self.config.td.dimension
        # non_blocking=True
        layer_list_cuda = [layer.to(self.net.device, non_blocking=True) for layer in layer_list]
        batch_main_layer = layer_list[layer_number]
        graph_sizes_base = get_graph_sizes(layer_list[0])

        all_predicted_normalized_a = self.net(layer_list_cuda, layer_number)

        predicted_normalized_a_split = all_predicted_normalized_a.split(
            graph_sizes_base
        )  # .to("cpu")
        node_features_split = batch_main_layer.x.split(graph_sizes_base)  # .to("cpu")

        loss_raport = LossRaport()
        main_loss = 0.0
        for batch_graph_index, scene_index in enumerate(batch_main_layer.scene_id):
            targets_data = dataset.get_targets_data(scene_index).to(
                self.net.device, non_blocking=True
            )

            predicted_normalized_a = predicted_normalized_a_split[batch_graph_index]
            node_features = node_features_split[batch_graph_index]
            forces = node_features[:, :dimension]

            # if hasattr(energy_args, "exact_normalized_a"):
            #    exact_normalized_a = exact_normalized_a_split[i]
            exact_normalized_a = None

            main_example_loss, example_loss = scene_input.loss_normalized_obstacle_correction(
                cleaned_a=predicted_normalized_a,
                a_correction=targets_data.a_correction,
                forces=forces,
                energy_args=targets_data.energy_args,
                exact_a=exact_normalized_a,
            )
            main_loss += main_example_loss
            loss_raport.add(example_loss, normalize=False)

        main_loss /= batch_main_layer.num_graphs
        loss_raport.normalize()
        return main_loss, loss_raport

    def get_derivatives(self, layer_list_cuda, layer_number, dimension):
        main_layer_cuda = layer_list_cuda[0]
        main_layer_cuda.x.requires_grad_(True)
        acceleration = self.net(layer_list_cuda, layer_number)

        for d in range(dimension):
            out_i = torch.zeros_like(acceleration)
            out_i[:, d] = 1.0

            acceleration_grad_i = torch.autograd.grad(
                outputs=acceleration,
                inputs=main_layer_cuda.x,
                grad_outputs=out_i,
                retain_graph=True,
                create_graph=True,
            )

            da_di = acceleration_grad_i[0][:, 0:dimension]

        # dr_dx_T = torch.stack((dr_dx_x, dr_dx_y), axis=1)
        return acceleration
