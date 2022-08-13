import gc
import time
from typing import Callable, List

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data.batch import Data

from conmech.helpers import cmh
from conmech.scenarios.scenarios import Scenario
from conmech.simulations import simulation_runner
from conmech.solvers.calculator import Calculator
from deep_conmech.data import base_dataset
from deep_conmech.graph.logger import Logger
from deep_conmech.graph.loss_calculation import (
    clean_acceleration,
    loss_normalized_obstacle_scatter,
)
from deep_conmech.graph.loss_raport import LossRaport
from deep_conmech.graph.net import CustomGraphNet
from deep_conmech.helpers import thh
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
        validation_dataset,
        print_scenarios: List[Scenario],
        net: CustomGraphNet,
        config: TrainingConfig,
        rank: int,
        world_size: int,
    ):
        self.rank = rank
        print(f"----NODE {self.rank}: CREATING MODEL----")
        self.config = config
        self.validation_dataset = validation_dataset
        self.dim = train_dataset.dimension  # TODO: Check validation datasets
        self.train_dataset = train_dataset
        self.print_scenarios = print_scenarios
        self.world_size = world_size

        print("UNUSED PARAMETERS")
        if config.distributed_training:
            self.ddp_net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
            self.ddp_net = DistributedDataParallel(
                self.ddp_net,
                device_ids=[rank],
                find_unused_parameters=True
            )
        else:
            self.ddp_net = net

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
        self.fp16_scaler = torch.cuda.amp.GradScaler(enabled=True)
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

    def is_at_skip(self, skip):
        return skip is not None and self.epoch % skip == 0

    def train(self):
        # epoch_tqdm = tqdm(range(config.EPOCHS), desc="EPOCH")
        # for epoch in epoch_tqdm:
        start_time = time.time()
        last_save_time = start_time
        if self.is_main:
            print("----TRAINING----")

        train_dataloader = base_dataset.get_train_dataloader(
            self.train_dataset, world_size=self.world_size, rank=self.rank
        )
        valid_dataloader = base_dataset.get_valid_dataloader(
            self.validation_dataset, world_size=self.world_size, rank=self.rank
        )
        while self.config.max_epoch_number is None or self.epoch < self.config.max_epoch_number:
            # self.train_dataset.reset()
            # for _ in range(2):
            self.epoch += 1
            _ = self.iterate_dataset(
                dataloader=train_dataloader,
                train=True,
                step_function=self.train_step,
                tqdm_description=f"EPOCH: {self.epoch}",  # , lr: {self.lr:.6f}",
                raport_description="Training",
            )  # , all_acceleration
            # self.train_dataset.update(all_acceleration)

            self.scheduler.step()
            self.optional_barrier()

            if self.is_main:
                current_time = time.time()
                elapsed_time = current_time - last_save_time
                if elapsed_time > self.config.td.save_at_minutes * 60:
                    # print(f"--Training time: {(elapsed_time / 60):.4f} min")
                    self.save_checkpoint()
                    last_save_time = time.time()

            self.optional_barrier()
            if self.is_at_skip(self.config.td.validate_at_epochs):
                _ = self.iterate_dataset(
                    dataloader=valid_dataloader,
                    train=False,
                    step_function=self.test_step,
                    tqdm_description=f"Validation: {self.epoch}",
                    raport_description="Validation",
                )
                # if self.is_at_skip(self.config.td.validate_scenarios_at_epochs):
                #     self.validate_all_scenarios_raport()

            self.optional_barrier()

    def optional_barrier(self):
        if self.config.distributed_training:
            dist.barrier()

    def save_checkpoint(self):
        print("----SAVING CHECKPOINT----")
        timestamp = cmh.get_timestamp(self.config)
        catalog = f"{self.config.output_catalog}/{self.config.current_time} - GRAPH MODELS"
        cmh.create_folders(catalog)
        path = f"{catalog}/{timestamp} - MODEL.pt"

        net = self.ddp_net.module if self.config.distributed_training else self.ddp_net
        checkpoint = {
            "epoch": self.epoch,
            "examples_seen": self.examples_seen,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "net": net.state_dict(),
        }
        torch.save(checkpoint, path)

    @staticmethod
    def get_checkpoint(rank: int, path: str):
        return torch.load(path, map_location={"cuda:0": f"cuda:{rank}"})

    @staticmethod
    def load_checkpointed_net(net, rank: int, path: str):
        print("----LOADING NET----")
        checkpoint = GraphModelDynamic.get_checkpoint(rank=rank, path=path)
        # consume_prefix_in_state_dict_if_present()
        net.load_state_dict(checkpoint["net"])
        net.eval()
        return net

    def load_checkpoint(self, path: str):
        print("----LOADING CHECKPOINT----")
        checkpoint = GraphModelDynamic.get_checkpoint(rank=self.rank, path=path)

        self.epoch = checkpoint["epoch"]
        self.examples_seen = checkpoint["examples_seen"]
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

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
                solve_function=net.solve,  # (net.solve, Calculator.solve),
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
        with torch.cuda.amp.autocast():
            main_loss, loss_raport = self.calculate_loss(batch_data=batch_data)  # acceleration_list
        self.fp16_scaler.scale(main_loss).backward()

        if self.config.td.gradient_clip is not None:
            self.clip_gradients(self.config.td.gradient_clip)
        self.fp16_scaler.step(self.optimizer)
        self.fp16_scaler.update()

        return loss_raport  # , acceleration_list

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

    def iterate_dataset(
        self,
        dataloader,
        train: bool,
        step_function: Callable,
        tqdm_description: str,
        raport_description: str,
    ):
        batch_tqdm = cmh.get_tqdm(
            dataloader, desc=tqdm_description, config=self.config, position=self.rank
        )
        if train:
            dataloader.sampler.set_epoch(self.epoch)

        mean_loss_raport = LossRaport()

        gc.disable()
        if self.config.profile_training:
            profiler = self.logger.get_and_start_profiler()

        # rae = self.config.td.raport_at_examples
        # all_acceleration = []
        for batch_id, batch_data in enumerate(batch_tqdm):

            loss_raport = step_function(batch_data)  # , acceleration_list
            # all_acceleration.extend(acceleration_list)

            mean_loss_raport.add(loss_raport)
            if train:
                self.examples_seen += loss_raport._count * self.world_size

            loss_description = f"{tqdm_description} loss: {(mean_loss_raport.main):.4f}"
            if batch_id == len(batch_tqdm) - 1:  # or self.examples_seen % rae == 0:
                if self.is_main:
                    self.save_raport(
                        mean_loss_raport=mean_loss_raport, description=raport_description
                    )
                mean_loss_raport = LossRaport()
                loss_description += " - raport saved"
            batch_tqdm.set_description(loss_description)

            if self.config.profile_training:
                profiler.step()
        if self.config.profile_training:
            profiler.stop()
        gc.enable()

        return mean_loss_raport  # , all_acceleration

    def should_raport_training(self, batch_id: int, batches_count: int):
        return (
            batch_id == batches_count - 1
            or self.examples_seen % self.config.td.raport_at_examples == 0
        )

    def save_raport(self, mean_loss_raport, description: str):
        self.logger.writer.add_scalar(
            f"Loss/{description}/LearningRate",
            self.lr,
            self.examples_seen,
        )
        for key, value in mean_loss_raport.get_iterator():
            self.logger.writer.add_scalar(
                f"Loss/{description}/{key}",
                value,
                self.examples_seen,
            )

    def validate_all_scenarios_raport(self):
        print("----VALIDATING SCENARIOS----")
        start_time = time.time()
        episode_steps = self.print_scenarios[0].schedule.episode_steps
        all_energy_values = np.zeros(episode_steps)
        for scenario in self.print_scenarios:
            assert episode_steps == scenario.schedule.episode_steps
            _, _, energy_values = simulation_runner.run_scenario(
                solve_function=self.ddp_net.module.solve,
                scenario=scenario,
                config=self.config,
                run_config=simulation_runner.RunScenarioConfig(),
                get_scene_function=GraphModelDynamic.get_scene_function,
            )
            all_energy_values += energy_values / len(self.print_scenarios)

        for i in [1, 10, 50, 100, 200, 800]:
            self.logger.writer.add_scalar(
                f"Loss/Validation/energy_mean_{i}_steps",
                np.mean(all_energy_values[:i]),
                self.examples_seen,
            )

        print(f"--Validating scenarios time: {int((time.time() - start_time) / 60)} min")

    def calculate_loss_all(
        self,
        dimension,
        node_features,
        target_data,
        all_acceleration,
        graph_sizes_base,
        all_exact_acceleration,
        all_linear_acceleration,
    ):
        # big_forces = node_features[:, :dimension]
        # big_lhs_size = target_data.a_correction.numel()
        # big_lhs_sparse = torch.sparse_coo_tensor(
        #     indices=target_data.lhs_index,
        #     values=target_data.lhs_values,
        #     size=(big_lhs_size, big_lhs_size),
        # )
        # big_lhs_sparse = big_lhs_sparse_coo.to_sparse_csr()
        big_main_loss, big_loss_raport = loss_normalized_obstacle_scatter(
            acceleration=all_acceleration,
            # forces=big_forces,
            # lhs=big_lhs_sparse,
            # rhs=target_data.rhs,
            # energy_args=target_data.energy_args,
            graph_sizes_base=graph_sizes_base,
            exact_acceleration=all_exact_acceleration,
            linear_acceleration=all_linear_acceleration,
        )

        return big_main_loss, big_loss_raport

    def calculate_loss(self, batch_data: List[Data]):
        dimension = self.config.td.dimension
        batch_layers = batch_data[0][: self.config.td.mesh_layers_count]
        layer_list = [layer.to(self.rank, non_blocking=True) for layer in batch_layers]
        target_data = batch_data[1].to(self.rank, non_blocking=True)
        batch_main_layer = layer_list[0]
        graph_sizes_base = get_graph_sizes(batch_main_layer)
        node_features = batch_main_layer.x  # .to("cpu")

        all_predicted_normalized_a = self.ddp_net(layer_list)  # .to("cpu")
        all_acceleration = clean_acceleration(
            cleaned_a=all_predicted_normalized_a, a_correction=target_data.a_correction
        )

        loss_tuple = self.calculate_loss_all(
            dimension=dimension,
            node_features=node_features,
            target_data=target_data,
            all_acceleration=all_acceleration,
            graph_sizes_base=graph_sizes_base,
            all_exact_acceleration=target_data.exact_acceleration,
            all_linear_acceleration=None,  # target_data.linear_acceleration,
        )
        # acceleration_list = [*all_acceleration.detach().split(graph_sizes_base)]

        return loss_tuple  # *, acceleration_list
