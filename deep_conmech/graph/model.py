import gc
import time
from ctypes import ArgumentError
from typing import Callable, List

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
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
        rank: int,
        world_size: int,
    ):
        self.rank = rank
        print(f"----{self.rank}: CREATING MODEL----")
        self.config = config
        self.all_val_datasets = all_val_datasets
        self.dim = train_dataset.dimension  # TODO: Check validation datasets
        self.train_dataset = train_dataset
        self.print_scenarios = print_scenarios
        self.world_size = world_size

        self.ddp_net = (
            DistributedDataParallel(net, device_ids=[rank], find_unused_parameters=True)
            if config.distributed_training
            else net
        )

        self.optimizer = torch.optim.Adam(
            self.ddp_net.parameters(),
            lr=self.config.td.initial_learning_rate,  # weight_decay=5e-4
        )
        lr_lambda = lambda epoch: max(
            self.config.td.learning_rate_decay**epoch,
            self.config.td.final_learning_rate / self.config.td.initial_learning_rate,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self.logger = Logger(dataset=self.train_dataset, config=config)
        self.epoch = 0
        self.examples_seen = 0
        if self.is_main:
            self.logger.save_parameters_and_statistics()
        if self.config.distributed_training:
            dist.barrier()

    @property
    def is_main(self):
        return self.rank == 0

    @property
    def lr(self):
        return float(self.scheduler.get_last_lr()[0])

    def train(self):
        # epoch_tqdm = tqdm(range(config.EPOCHS), desc="EPOCH")
        # for epoch in epoch_tqdm:
        start_time = time.time()
        last_save_time = start_time
        print("----TRAINING----")

        dataloader = base_dataset.get_train_dataloader(
            self.train_dataset, world_size=self.world_size, rank=self.rank
        )
        while self.config.max_epoch_number is None or self.epoch < self.config.max_epoch_number:
            self.epoch += 1

            mean_loss_raport = self.iterate_dataset(
                dataloader=dataloader,
                step_function=self.train_step,
                description=f"EPOCH: {self.epoch}",  # , lr: {self.lr:.6f}",
            )
            self.examples_seen += len(self.train_dataset)
            self.training_raport(mean_loss_raport=mean_loss_raport)

            self.scheduler.step()

            if self.config.distributed_training:
                dist.barrier()

            if self.is_main:
                current_time = time.time()
                elapsed_time = current_time - last_save_time
                if elapsed_time > self.config.td.save_at_minutes * 60:
                    # print(f"--Training time: {(elapsed_time / 60):.4f} min")
                    self.save_checkpoint()
                    last_save_time = time.time()

                def is_at_skip(skip):
                    return skip is not None and self.epoch % skip == 0

                if is_at_skip(self.config.td.validate_at_epochs):
                    self.validation_raport()
                if is_at_skip(self.config.td.validate_scenarios_at_epochs):
                    self.validate_all_scenarios_raport()

            if self.config.distributed_training:
                dist.barrier()

    def save_checkpoint(self):
        print("----SAVING CHECKPOINT----")
        timestamp = cmh.get_timestamp(self.config)
        catalog = f"{self.config.output_catalog}/{self.config.current_time} - GRAPH MODELS"
        cmh.create_folders(catalog)
        path = f"{catalog}/{timestamp} - MODEL.pt"

        checkpoint = {
            "epoch": self.epoch,
            "examples_seen": self.examples_seen,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "net": self.ddp_net.state_dict(),
        }
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpointed_net(net, path: str):
        print("----LOADING NET----")
        checkpoint = torch.load(path)
        net.load_state_dict(checkpoint["net"])
        net.eval()
        return net

    def load_checkpoint(self, path: str):
        print("----LOADING CHECKPOINT----")
        map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank}
        checkpoint = torch.load(path, map_location=map_location)

        self.epoch = checkpoint["epoch"]
        self.examples_seen = checkpoint["examples_seen"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.ddp_net.load_state_dict(checkpoint["net"])

        self.ddp_net.eval()

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

    def train_step(self, batch_data: List[Data]):
        self.ddp_net.train()
        self.ddp_net.zero_grad()

        # cmh.profile(lambda: self.calculate_loss(batch_data=batch_data, layer_number=0))
        main_loss, loss_raport = self.calculate_loss(batch_data=batch_data)
        main_loss.backward()
        if self.config.td.gradient_clip is not None:
            self.clip_gradients(self.config.td.gradient_clip)
        self.optimizer.step()

        return loss_raport

    def test_step(self, batch_data: List[Data]):
        self.ddp_net.eval()

        with torch.no_grad():  # with tc.set_grad_enabled(train):
            _, loss_raport = self.calculate_loss(batch_data=batch_data)

        return loss_raport

    def clip_gradients(self, max_norm: float):
        parameters = self.ddp_net.parameters()
        # norms = [np.max(np.abs(p.grad.cpu().detach().numpy())) for p in parameters]
        # total_norm = np.max(norms)_
        # print("total_norm", total_norm)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm)

    def iterate_dataset(self, dataloader, step_function: Callable, description: str):
        batch_tqdm = cmh.get_tqdm(
            dataloader, desc=description, config=self.config, position=self.rank
        )

        mean_loss_raport = LossRaport()

        gc.disable()
        profile = True
        if profile:
            prof = self.logger.get_profiler()
            prof.start()
        for _, batch_data in enumerate(batch_tqdm):

            loss_raport = step_function(batch_data)
            mean_loss_raport.add(loss_raport)

            batch_tqdm.set_description(f"{description} loss: {(mean_loss_raport.main):.4f}")
            if profile:
                prof.step()
        if profile:
            prof.stop()
        gc.enable()

        return mean_loss_raport

    def training_raport(self, mean_loss_raport):
        self.logger.writer.add_scalar(
            "Loss/Training/LearningRate",
            self.lr,
            self.examples_seen,
        )
        for key, value in mean_loss_raport.get_iterator():
            self.logger.writer.add_scalar(
                f"Loss/Training/{key}",
                value,
                self.examples_seen,
            )

    def validation_raport(self):
        # print("----VALIDATING----")
        start_time = time.time()

        dataloader = base_dataset.get_valid_dataloader(
            self.all_val_datasets[0], world_size=self.world_size, rank=self.rank
        )
        # for dataset in self.all_val_datasets:
        mean_loss_raport = self.iterate_dataset(
            dataloader=dataloader,
            step_function=self.test_step,
            description=dataset.data_id,
        )
        for key, value in mean_loss_raport.get_iterator():
            self.logger.writer.add_scalar(
                f"Loss/Validating/{key}",
                value,
                self.examples_seen,
            )
        # validation_time = time.time() - start_time
        # print(f"--Validation time: {(validation_time / 60):.4f} min")

    def validate_all_scenarios_raport(self):
        print("----VALIDATING SCENARIOS----")
        start_time = time.time()
        episode_steps = self.print_scenarios[0].schedule.episode_steps
        all_energy_values = np.zeros(episode_steps)
        for scenario in self.print_scenarios:
            assert episode_steps == scenario.schedule.episode_steps
            _, _, energy_values = simulation_runner.run_scenario(
                solve_function=self.ddp_net.solve,
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
                self.examples_seen,
            )

        print(f"--Validating scenarios time: {int((time.time() - start_time) / 60)} min")

    def calculate_loss_all(
        self, dimension, node_features, target_data, all_acceleration, graph_sizes_base
    ):
        big_forces = node_features[:, :dimension]
        big_lhs_size = target_data.a_correction.numel()
        big_lhs_sparse = torch.sparse_coo_tensor(
            indices=target_data.lhs_index,
            values=target_data.lhs_values,
            size=(big_lhs_size, big_lhs_size),
        )
        big_main_loss, big_loss_raport = scene_input.loss_normalized_obstacle_scatter(
            acceleration=all_acceleration,
            forces=big_forces,
            lhs=big_lhs_sparse,
            rhs=target_data.rhs,
            energy_args=None,
            graph_sizes_base=graph_sizes_base,
        )

        return big_main_loss, big_loss_raport

    def calculate_loss_single(
        self, dimension, node_features, target_data, all_acceleration, graph_sizes_base
    ):
        num_graphs = len(graph_sizes_base)
        node_features_split = node_features.split(graph_sizes_base)
        predicted_acceleration_split = all_acceleration.split(graph_sizes_base)

        loss_raport = LossRaport()
        main_loss = 0.0
        for batch_graph_index in range(num_graphs):
            predicted_acceleration = predicted_acceleration_split[batch_graph_index]
            node_features = node_features_split[batch_graph_index]
            forces = node_features[:, :dimension]

            energy_args = target_data.energy_args[batch_graph_index]

            energy_args.lhs_sparse = torch.sparse_coo_tensor(
                indices=energy_args.lhs_indices,
                values=energy_args.lhs_values,
                size=energy_args.lhs_size,
            )

            # if hasattr(energy_args, "exact_normalized_a"):
            #    exact_normalized_a = exact_normalized_a_split[i]
            exact_normalized_a = None

            main_example_loss, example_loss_raport = scene_input.loss_normalized_obstacle(
                acceleration=predicted_acceleration,
                forces=forces,
                lhs=energy_args.lhs_sparse,
                rhs=energy_args.rhs,
                energy_args=energy_args,
                exact_a=exact_normalized_a,
            )
            main_loss += main_example_loss
            loss_raport.add(example_loss_raport, normalize=False)

        main_loss /= num_graphs
        loss_raport.normalize()

        return main_loss, loss_raport

    def calculate_loss(self, batch_data: List[Data]):
        dimension = self.config.td.dimension
        layer_list = [layer.to(self.rank, non_blocking=True) for layer in batch_data[0]]
        target_data = batch_data[1].to(self.rank, non_blocking=True)
        batch_main_layer = layer_list[0]
        graph_sizes_base = get_graph_sizes(batch_main_layer)
        node_features = batch_main_layer.x  # .to("cpu")

        all_predicted_normalized_a = self.ddp_net(layer_list)  # .to("cpu")
        all_acceleration = scene_input.clean_acceleration(
            cleaned_a=all_predicted_normalized_a, a_correction=target_data.a_correction
        )

        loss_all_tuple = self.calculate_loss_all(
            dimension, node_features, target_data, all_acceleration, graph_sizes_base
        )

        # loss_single_tuple = self.calculate_loss_single(
        #     dimension, node_features, target_data, all_acceleration, graph_sizes_base
        # )
        return loss_all_tuple

    def get_derivatives(self, layer_list, layer_number, dimension):
        main_layer_cuda = layer_list[0]
        main_layer_cuda.x.requires_grad_(True)
        acceleration = self.ddp_net(layer_list, layer_number)

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
