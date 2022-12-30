import gc
import time
from functools import partial
from typing import Callable, List

import flax
import flax.jax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch
import torch.utils
from flax.training import checkpoints
from flax.training.train_state import TrainState
from torch_geometric.data.batch import Data

from conmech.helpers import cmh
from conmech.scenarios.scenarios import Scenario
from conmech.scene.energy_functions import EnergyFunctions
from conmech.simulations import simulation_runner
from conmech.solvers.calculator import Calculator
from deep_conmech.data import base_dataset
from deep_conmech.graph.logger import Logger
from deep_conmech.graph.loss_raport import LossRaport
from deep_conmech.graph.net_jax import CustomGraphNetJax, GraphNetArguments
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


class NetState(TrainState):
    batch_stats: float


class GraphModelDynamicJax:
    def __init__(
        self,
        train_dataset,
        all_validation_datasets,
        print_scenarios: List[Scenario],
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
        self.train_state = None

        self.logger = Logger(dataset=self.train_dataset, config=config)
        self.epoch = 0
        self.examples_seen = 0
        if self.is_main:
            self.logger.save_parameters_and_statistics()

    @property
    def is_main(self):
        return self.rank == 0

    @property
    def lr(self):
        return 0  # TODO

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
        all_valid_dataloaders = [
            base_dataset.get_valid_dataloader(dataset, world_size=self.world_size, rank=self.rank)
            for dataset in self.all_validation_datasets
        ]

        state = self.initialize_state(train_dataloader)

        ###
        # def plot_weights(data_jax, name):
        #     data = np.array(data_jax.flatten())

        #     import matplotlib.pyplot as plt

        #     _ = plt.hist(data, bins=100)
        #     plt.title(f"Weight histogram {name} shape: {data_jax.shape}")
        #     plt.savefig(f"{name}.png")

        # plot_weights(state.params['ForwardNet_0']['Dense_0']['kernel'], "Forward0Kernel")
        # plot_weights(state.params['ForwardNet_0']['Dense_0']['bias'], "Forward0Bias")

        # plot_weights(state.params['ProcessorLayer_1']['ForwardNet_1']['Dense_0']['kernel'], "ProcessorKernel1")
        # plot_weights(state.params['ProcessorLayer_1']['ForwardNet_1']['Dense_0']['bias'], "ProcessorBias1")
        ###

        while self.config.max_epoch_number is None or self.epoch < self.config.max_epoch_number:
            # self.train_dataset.reset()
            # for _ in range(2):
            self.epoch += 1

            state = self.iterate_dataset(
                state=state,
                dataloader=train_dataloader,
                train=True,
                step_function=self.train_step,
                tqdm_description=f"EPOCH: {self.epoch}",  # , lr: {self.lr:.6f}",
                raport_description="Training",
            )  # , all_acceleration
            # self.train_dataset.update(all_acceleration)

            current_time = time.time()
            elapsed_time = current_time - last_save_time
            if True:  # elapsed_time > self.config.td.save_at_minutes * 60:
                # print(f"--Training time: {(elapsed_time / 60):.4f} min")
                self.save_checkpoint(state=state)
                last_save_time = time.time()

            # if self.is_at_skip(self.config.td.validate_at_epochs):
            #     # if elapsed_time > self.config.td.validate_at_minutes * 60:
            #     for dataloader in all_valid_dataloaders:
            #         _ = self.iterate_dataset(
            #             dataloader=dataloader,
            #             train=False,
            #             step_function=self.test_step,
            #             tqdm_description=f"Validation: {self.epoch}",
            #             raport_description=dataloader.dataset.description,
            #         )
            #     # if self.is_at_skip(self.config.td.validate_scenarios_at_epochs):
            #     #     self.validate_all_scenarios_raport()

    def save_checkpoint(self, state):
        print("----SAVING CHECKPOINT----")

        target = flax.jax_utils.unreplicate(state)
        timestamp = cmh.get_timestamp(self.config)
        catalog = f"{self.config.output_catalog}/{self.config.current_time} - JAX GRAPH MODELS"
        cmh.create_folders(catalog)
        path = f"{catalog}/{timestamp} - MODEL.pt"
        checkpoints.save_checkpoint(ckpt_dir=path, target=target, step=0)

    @staticmethod
    def get_checkpoint(rank: int, path: str):
        return torch.load(path, map_location={"cuda:0": f"cuda:{rank}"})

    @staticmethod
    def load_checkpointed_net(net, rank: int, path: str):
        print("----LOADING NET----")
        state = checkpoints.restore_checkpoint(ckpt_dir=path, target=None)
        return state

    def load_checkpoint(self, path: str):
        print("----LOADING CHECKPOINT----")
        raise NotImplementedError()

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
    def plot_all_scenarios(state, print_scenarios: List[Scenario], config: TrainingConfig):
        print("----PLOTTING----")
        start_time = time.time()

        variables = {"params": state["params"]}

        @jax.jit
        def apply_net(args):
            args = jax.lax.stop_gradient(args)
            return CustomGraphNetJax().apply(variables, args, train=False)

        solve_function = lambda scene, energy_functions, initial_a: solve(
            apply_net, scene, energy_functions, initial_a
        )  # (net.solve, Calculator.solve),

        for scenario in print_scenarios:
            simulation_runner.run_scenario(
                solve_function=solve_function,  # (net.solve, Calculator.solve),
                scenario=scenario,
                config=config,
                run_config=simulation_runner.RunScenarioConfig(
                    catalog="GRAPH PLOT",
                    simulate_dirty_data=False,
                    compare_with_base_scene=config.compare_with_base_scene,
                    plot_animation=True,
                ),
                get_scene_function=GraphModelDynamicJax.get_scene_function,
            )
            print("---")
        print(f"Plotting time: {int((time.time() - start_time) / 60)} min")
        # return catalog

    def train_step(self, state, batch_data: List[Data]):
        state, main_loss, loss_raport = self.calculate_loss(
            state, batch_data=batch_data
        )  # acceleration_list
        return state, loss_raport  # , acceleration_list

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
        state,
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
            state, loss_raport = step_function(state, batch_data)  # , acceleration_list
            # all_acceleration.extend(acceleration_list)

            # TODO: check / assert state consistency across GPUs
            # TODO: Check if data are randomized
            mean_loss_raport.add(loss_raport)
            if train:
                self.examples_seen += loss_raport._count * self.world_size

            loss_description = f"{tqdm_description} loss: {(mean_loss_raport.main):.4f}"
            if batch_id == len(batch_tqdm) - 1:  # or self.examples_seen % rae == 0:
                self.save_raport(mean_loss_raport=mean_loss_raport, description=raport_description)
                mean_loss_raport = LossRaport()
                loss_description += " - raport saved"
            batch_tqdm.set_description(loss_description)

            if self.config.profile_training:
                profiler.step()
        if self.config.profile_training:
            profiler.stop()
        gc.enable()

        return state  # mean_loss_raport  # , all_acceleration

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

    #####

    def get_layer_list_and_target_data(self, batch_data):
        batch_layers = batch_data[0]
        target_data = batch_data[1].to(self.rank, non_blocking=True)
        layer_list = [layer.to(self.rank, non_blocking=True) for layer in batch_layers]
        layer_list, target_data = convert_to_jax(layer_list, target_data)
        return layer_list, target_data

    def calculate_loss(self, state, batch_data: List[List[Data]]):
        local_devices = jax.local_devices()
        devices_count = len(local_devices)
        data = [self.get_layer_list_and_target_data(bd) for bd in batch_data]

        all_args = [
            prepare_input(layer_list) for layer_list in [data[d][0] for d in range(devices_count)]
        ]
        all_target_data = [
            data[d][1].normalized_new_displacement.astype(np.float32) for d in range(devices_count)
        ]

        sharded_targets = jax.device_put_sharded(
            all_target_data, local_devices
        )  # TODO: check order with pmap
        # flax.jax_utils.replicate(
        #        target_data.normalized_new_displacement.astype(np.float32)
        # )
        sharded_args = jax.device_put_sharded(
            all_args, local_devices
        )  # flax.jax_utils.replicate(args)

        state, displacement_loss = train_step(
            state,
            sharded_targets,
            sharded_args,
        )

        batch_main_layer = data[0][0][0]
        graph_sizes_base = get_graph_sizes(batch_main_layer)
        num_graphs = len(graph_sizes_base) * len(jax.local_devices())  # TODO: generalize

        loss_raport = LossRaport(
            main=displacement_loss.item(),
            displacement_loss=displacement_loss.item(),
            _count=num_graphs,
        )

        return state, displacement_loss, loss_raport

    def initialize_state(self, dataloader):
        sample_batch_data = next(iter(dataloader))
        sample_layer_list, _ = self.get_layer_list_and_target_data(
            sample_batch_data[0]
        )  # get data for first device
        sample_args = prepare_input(sample_layer_list)

        # init_rng = jax.random.PRNGKey(0)  # the same initialization instead of training an ensemble
        # rng = flax.jax_utils.replicate(init_rng)  # jax.random.split(init_rng, jax.device_count())

        # def cts(rng):
        #     return create_train_state(rng, sample_args, self.config.td.initial_learning_rate)

        # state = jax.pmap(cts)(rng)
        # # print("Sample weights:", state.params['ForwardNet_0']['Dense_0']['kernel'][0])
        init_state = create_train_state(
            jax.random.PRNGKey(42), sample_args, self.config.td.initial_learning_rate
        )

        state = flax.jax_utils.replicate(init_state)
        # , sample_args, sample_static_args, self.config.td.initial_learning_rate) # static_broadcasted_argnums=(2, 3))
        return state


def convert_to_jax(layer_list, target_data=None):
    for layer in layer_list:
        layer["x"] = thh.convert_cuda_tensor_to_jax(layer["x"])
        layer.edge_attr = thh.convert_cuda_tensor_to_jax(layer.edge_attr)
        layer.edge_index = thh.convert_cuda_tensor_to_jax(layer.edge_index)

    layer_sparse = layer_list[1]
    layer_sparse.edge_index_to_down = thh.convert_cuda_tensor_to_jax(
        layer_sparse.edge_index_to_down
    )
    layer_sparse.edge_attr_to_down = thh.convert_cuda_tensor_to_jax(layer_sparse.edge_attr_to_down)
    if target_data is None:
        return layer_list
    target_data.normalized_new_displacement = thh.convert_cuda_tensor_to_jax(
        target_data.normalized_new_displacement
    )
    return layer_list, target_data


# TODO: all in Jax?
def solve(apply_net, scene: SceneInput, energy_functions: EnergyFunctions, initial_a):
    # return Calculator.solve(scene=scene, energy_functions=energy_functions, initial_a=initial_a)

    t_start = time.time()
    scene.reduced.exact_acceleration = Calculator.solve(
        scene=scene.reduced,
        energy_functions=energy_functions,
        initial_a=scene.reduced.exact_acceleration,
    )

    print("1", time.time() - t_start)
    t_start = time.time()

    layers_list = [
        scene.get_features_data(layer_number=layer_number)  # .to(device)
        for layer_number, _ in enumerate(scene.all_layers)
    ]

    args = prepare_input(convert_to_jax(layers_list))
    local_devices = jax.local_devices()
    args = jax.device_put(args, local_devices[0])

    print("2", time.time() - t_start)
    t_start = time.time()

    # with jax.profiler.trace("./log", create_perfetto_link=True):
    # https://github.com/google/jax/issues/13009
    def fun():
        return apply_net(args)
    net_result = cmh.profile(fun, baypass=True)

    print("3", time.time() - t_start)
    t_start = time.time()

    net_displacement = np.array(net_result)

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

    print("4", time.time() - t_start)
    t_start = time.time()

    ###
    # displacement_new = scene.to_displacement(acceleration_from_displacement)
    # reduced_displacement_new = scene.lift_data(displacement_new)
    # lifted_acceleration = scene.reduced.from_displacement(reduced_displacement_new)

    # alpha = 0.9
    # scene.reduced.lifted_acceleration = (
    #     alpha * scene.reduced.exact_acceleration + (1 - alpha) * lifted_acceleration
    # )
    ###

    return acceleration_from_displacement


def prepare_input(layer_list):
    def unpack(layer):
        return layer["x"], layer.edge_attr, layer.edge_index

    layer_dense = layer_list[0]
    layer_sparse = layer_list[1]

    dense_x, dense_edge_attr, dense_edge_index = unpack(layer_dense)
    sparse_x, sparse_edge_attr, sparse_edge_index = unpack(layer_sparse)
    multilayer_edge_attr = layer_sparse.edge_attr_to_down
    multilayer_edge_index = layer_sparse.edge_index_to_down

    sparse_edge_index = np.array(sparse_edge_index)
    dense_edge_index = np.array(dense_edge_index)
    multilayer_edge_index = np.array(multilayer_edge_index)

    args = GraphNetArguments(
        sparse_x=sparse_x,
        sparse_edge_attr=sparse_edge_attr,
        dense_x=dense_x,
        dense_edge_attr=dense_edge_attr,
        multilayer_edge_attr=multilayer_edge_attr,
        sparse_edge_index=sparse_edge_index,
        dense_edge_index=dense_edge_index,
        multilayer_edge_index=multilayer_edge_index,
    )
    return args


def create_train_state(rng, sample_args, learning_rate):
    params = CustomGraphNetJax().get_params(sample_args, rng)
    jax.tree_util.tree_map(lambda x: x.shape, params)  # Checking output shapes

    optimizer = optax.adam(learning_rate=learning_rate)

    def a_fn(variables, args, train):
        return CustomGraphNetJax().apply(variables, args, train)

    return NetState.create(apply_fn=a_fn, params=params, tx=optimizer, batch_stats=None)


def MSE(predicted, exact):
    return jnp.mean(jnp.linalg.norm(predicted - exact, axis=-1) ** 2)


def RMSE(predicted, exact):
    return jnp.sqrt(MSE(predicted, exact))


@partial(jax.pmap, axis_name="models")  # , static_broadcasted_argnums=(2))
# @partial(jax.jit, static_argnames=("static_args"))
def apply_model(state, args, normalized_new_displacement):
    def loss_fn(params):
        variables = {"params": params}
        net_new_displacement = state.apply_fn(variables, args, train=True)
        losses = RMSE(net_new_displacement, normalized_new_displacement)
        return losses

    losses, grads = jax.value_and_grad(loss_fn)(state.params)

    losses_pmean = jax.lax.pmean(losses, axis_name="models")
    grads_pmean = jax.lax.pmean(grads, axis_name="models")
    return losses, losses_pmean, grads_pmean


def train_step(state, normalized_new_displacement, args):
    losses, losses_pmean, grads_pmean = cmh.profile(
        lambda: apply_model(state, args, normalized_new_displacement), baypass=True
    )

    # print("Losses:", losses)
    state = update_model(state, grads_pmean)
    return state, jnp.mean(losses_pmean)


@jax.pmap
def update_model(state, grads):
    return state.apply_gradients(grads=grads) # TODO: move to apply_map
