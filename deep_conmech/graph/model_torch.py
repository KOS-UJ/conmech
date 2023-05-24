import gc
import time
from functools import partial
from typing import Callable, List

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.data.batch import Data

from conmech.helpers import cmh
from conmech.scenarios.scenarios import Scenario
from conmech.scene.energy_functions import EnergyFunctions
from conmech.simulations import simulation_runner
from conmech.solvers.calculator import Calculator
from deep_conmech.data import base_dataset
from deep_conmech.graph.logger import Logger
from deep_conmech.graph.loss_raport import LossRaport
from deep_conmech.graph.net_torch import CustomGraphNet
from deep_conmech.helpers import thh
from deep_conmech.scene.scene_input import SceneInput
from deep_conmech.training_config import TrainingConfig


class ErrorResult:
    value = 0


def get_graph_sizes(batch):
    return np.ediff1d(thh.to_np_long(batch.ptr)).tolist()


def clean_acceleration(cleaned_a, a_correction):
    return cleaned_a if (a_correction is None) else (cleaned_a - a_correction)


def get_mean_loss(acceleration, forces, mass_density, boundary_integral):
    # F = m * a
    return (boundary_integral == 0) * (
        torch.norm(torch.mean(forces, axis=0) - torch.mean(mass_density * acceleration, axis=0))
        ** 2
    )


class GraphModelDynamicTorch:
    def __init__(
        self,
        train_dataset,
        all_validation_datasets,
        print_scenarios: List[Scenario],
        net: CustomGraphNet,
        config: TrainingConfig,
        rank: int,
        world_size: int,
    ):
        self.rank = rank
        print(f"----NODE {self.rank}: CREATING MODEL----")
        self.config = config
        self.all_validation_datasets = all_validation_datasets
        self.dim = train_dataset.dimension  # TODO: Check validation datasets
        self.train_dataset = train_dataset
        self.print_scenarios = print_scenarios
        self.world_size = world_size
        self.net = net

        ###
        # def plot_weights(data_jax, name):
        #     data = thh.to_np_double(data_jax.flatten())

        #     import matplotlib.pyplot as plt

        #     _ = plt.hist(data, bins=100)
        #     plt.title(f"Weight histogram {name} shape: {data_jax.shape}")
        #     plt.savefig(f"{name}.png")

        # net.sparse_processor_layers[0].node_processor.net[0].blocks[0].weight

        # plot_weights(net.node_encoder_sparse.net[0].blocks[0].weight, "TForwardKernel1")
        # plot_weights(net.node_encoder_sparse.net[0].blocks[0].bias, "TForwardBias1")

        # plot_weights(net.sparse_processor_layers[0].node_processor.net[0].blocks[0].weight, "TProcessorKernel1")
        # plot_weights(net.sparse_processor_layers[0].node_processor.net[0].blocks[0].bias, "TProcessorBias1")
        ###

        print("UNUSED PARAMETERS")
        if config.torch_distributed_training:
            self.ddp_net = nn.SyncBatchNorm.convert_sync_batchnorm(net)  # TODO: Add this to JAX
            self.ddp_net = DistributedDataParallel(
                self.ddp_net, device_ids=[rank], find_unused_parameters=True
            )
        else:
            self.ddp_net = net

        self.optimizer = torch.optim.AdamW(
            self.ddp_net.parameters(),
            lr=self.config.td.initial_learning_rate,  # weight_decay=5e-4
        )

        def lr_lambda(epoch):
            return max(
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
        if self.config.torch_distributed_training:
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
        # start_time = time.time()
        # last_save_time = start_time
        if self.is_main:
            print("----TRAINING----")

        train_dataloader = base_dataset.get_train_dataloader(self.train_dataset)
        all_valid_dataloaders = [
            base_dataset.get_valid_dataloader(dataset) for dataset in self.all_validation_datasets
        ]
        while self.config.max_epoch_number is None or self.epoch < self.config.max_epoch_number:
            # self.train_dataset.reset()
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
                # elapsed_save_time = time.time() - last_save_time
                # if elapsed_save_time > self.config.td.save_at_minutes * 60:
                if self.is_at_skip(self.config.td.save_at_epochs):
                    # print(f"--Training time: {(elapsed_time / 60):.4f} min")
                    self.save_checkpoint()
                    # last_save_time = time.time()

            self.optional_barrier()
            if self.is_at_skip(self.config.td.validate_at_epochs):
                for dataloader in all_valid_dataloaders:
                    _ = self.iterate_dataset(
                        dataloader=dataloader,
                        train=False,
                        step_function=self.test_step,
                        tqdm_description="VAL:",
                        raport_description=dataloader.dataset.description,
                    )
                # if self.is_at_skip(self.config.td.validate_scenarios_at_epochs):
                #     self.validate_all_scenarios_raport()

            self.optional_barrier()

    def optional_barrier(self):
        if self.config.torch_distributed_training:
            dist.barrier()

    def save_checkpoint(self):
        print("----SAVING CHECKPOINT----")
        timestamp = cmh.get_timestamp(self.config)
        catalog = f"{self.config.output_catalog}/{self.config.current_time} - TORCH GRAPH MODELS"
        cmh.create_folders(catalog)
        path = f"{catalog}/{timestamp} - MODEL.pt"

        net = self.ddp_net.module if self.config.torch_distributed_training else self.ddp_net
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
        checkpoint = GraphModelDynamicTorch.get_checkpoint(rank=rank, path=path)
        # consume_prefix_in_state_dict_if_present()
        net.load_state_dict(checkpoint["net"])
        net.eval()
        return net

    def load_checkpoint(self, path: str):
        print("----LOADING CHECKPOINT----")
        checkpoint = GraphModelDynamicTorch.get_checkpoint(rank=self.rank, path=path)

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
        for scenario in print_scenarios:
            simulation_runner.run_scenario(
                solve_function=partial(solve, net=net),  # (net.solve, Calculator.solve),
                scenario=scenario,
                config=config,
                run_config=simulation_runner.RunScenarioConfig(
                    catalog="GRAPH PLOT",
                    simulate_dirty_data=False,
                    plot_animation=True,
                ),
                get_scene_function=GraphModelDynamicTorch.get_scene_function,
            )
            print("---")
        print(f"Plotting time: {int((time.time() - start_time) / 60)} min")
        # return catalog

    def train_step(self, batch_data: List[Data]):
        scale = False
        self.ddp_net.train()
        self.ddp_net.zero_grad()

        # cmh.profile(lambda: self.calculate_loss(batch_data=batch_data, layer_number=0))
        if scale:
            with torch.cuda.amp.autocast():
                main_loss, loss_raport = self.calculate_loss(
                    batch_data=batch_data
                )  # acceleration_list
            self.fp16_scaler.scale(main_loss).backward()
        else:
            main_loss, loss_raport = self.calculate_loss(batch_data=batch_data)  # acceleration_list
            main_loss.backward()

        if self.config.td.gradient_clip is not None:
            self.clip_gradients(self.config.td.gradient_clip)

        if scale:
            self.fp16_scaler.step(self.optimizer)
            self.fp16_scaler.update()
        else:
            self.optimizer.step()

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
                self.examples_seen += loss_raport.count * self.world_size

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

        return mean_loss_raport

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
        # TODO: add
        # self.logger.writer.add_graph(self) #args

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
                get_scene_function=GraphModelDynamicTorch.get_scene_function,
            )
            all_energy_values += energy_values / len(self.print_scenarios)

        for i in [1, 10, 50, 100, 200, 800]:
            self.logger.writer.add_scalar(
                f"Loss/Validation/energy_mean_{i}_steps",
                np.mean(all_energy_values[:i]),
                self.examples_seen,
            )

        print(f"--Validating scenarios time: {int((time.time() - start_time) / 60)} min")

    def calculate_loss(self, batch_data: List[Data]):
        batch_layers = batch_data[0]
        layer_list = [layer.to(self.rank, non_blocking=True) for layer in batch_layers]
        target_data = batch_data[1].to(self.rank, non_blocking=True)
        batch_main_layer = layer_list[0]
        graph_sizes_base = get_graph_sizes(batch_main_layer)
        num_graphs = len(graph_sizes_base)

        net_new_displacement = self.ddp_net(layer_list)

        displacement_loss = thh.root_mean_square_error_torch(
            net_new_displacement, target_data.normalized_new_displacement
        )
        loss_raport = LossRaport(
            main=displacement_loss.item(),
            displacement_loss=displacement_loss.item(),
            _count=num_graphs,
        )

        return displacement_loss, loss_raport

    @staticmethod
    def get_sample_onnx_args(dataset):
        dataloader = base_dataset.get_train_dataloader(dataset)
        sample_batch_data = next(iter(dataloader))

        layer_list = sample_batch_data[0]

        def get_dict(layer):
            d = vars(layer)["_store"]
            return cmh.DotDict(d)

        layer_list = [get_dict(layer) for layer in layer_list]
        return layer_list

    @staticmethod
    def save_onnx_model(model_path, net, dataset):
        layer_list = GraphModelDynamicTorch.get_sample_onnx_args(dataset)
        # model = lambda : net(layer_list)
        torch.onnx.export(
            net,
            {"layer_list": layer_list},
            model_path,
            verbose=True,
            # input_names=input_names,
            # output_names=output_names,
            # export_params=True,
        )


def solve(net, scene: SceneInput, energy_functions: EnergyFunctions, initial_a, initial_t, timer):
    # return Calculator.solve(scene=scene, energy_functions=energy_functions, initial_a=initial_a)
    _ = initial_a, initial_t, timer

    scene.reduced.exact_acceleration, _ = Calculator.solve(
        scene=scene.reduced,
        energy_functions=energy_functions,
        initial_a=scene.reduced.exact_acceleration,
    )

    layers_list = [
        scene.get_features_data(layer_number=layer_number).to(net.device)
        for layer_number, _ in enumerate(scene.all_layers)
    ]

    net.eval()
    net_result = net(layer_list=layers_list)
    net_displacement = thh.to_np_double(net_result)

    # base = scene.moved_base
    # position = scene.position
    reduced_displacement_new = scene.reduced.to_displacement(scene.reduced.exact_acceleration)
    base = scene.reduced.get_rotation(reduced_displacement_new)
    position = np.mean(reduced_displacement_new, axis=0)

    new_displacement = scene.get_displacement(
        base=base, position=position, base_displacement=net_displacement
    )

    acceleration_from_displacement = scene.from_displacement(new_displacement)
    scene.reduced.lifted_acceleration = scene.reduced.exact_acceleration

    return acceleration_from_displacement, None
