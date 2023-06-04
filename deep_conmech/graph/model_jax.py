import gc
import time
from functools import partial
from typing import Any, Callable, List, Optional

import flax
import flax.jax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint
import tensorflow as tf
import torch
import torch.utils
from flax.training import train_state
from jax import lax
from jax.experimental import jax2tf
from torch_geometric.data.batch import Data

from conmech.helpers import cmh
from conmech.helpers.tmh import Timer
from conmech.scenarios.scenarios import Scenario
from conmech.scene.energy_functions import EnergyFunctions
from conmech.simulations import simulation_runner
from conmech.solvers.calculator import Calculator
from deep_conmech.data import base_dataset
from deep_conmech.data.dataset_statistics import FeaturesStatistics
from deep_conmech.graph.logger import Logger
from deep_conmech.graph.loss_raport import LossRaport
from deep_conmech.graph.net_jax import CustomGraphNetJax, GraphNetArguments
from deep_conmech.helpers import thh
from deep_conmech.scene.scene_input import SceneInput
from deep_conmech.training_config import TrainingConfig

SCALE = 1e3


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


class TrainState(train_state.TrainState):
    batch_stats: Any


class GraphModelDynamicJax:
    def __init__(
        self,
        train_dataset,
        all_validation_datasets,
        print_scenarios: List[Scenario],
        config: TrainingConfig,
        statistics: Optional[dict[str, FeaturesStatistics]] = None,
    ):
        print("----CREATING MODEL----")
        self.config = config
        self.all_validation_datasets = all_validation_datasets
        self.dim = train_dataset.dimension  # TODO: Check validation datasets
        self.train_dataset = train_dataset
        self.print_scenarios = print_scenarios
        self.statistics = statistics
        self.train_state = None

        self.logger = Logger(dataset=self.train_dataset, config=config)
        self.checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        self.epoch = 0
        self.examples_seen = 0

        # self.logger.save_parameters_and_statistics() ###

    def is_at_skip(self, skip):
        return skip is not None and self.epoch % skip == 0

    def train(self):
        print("----TRAINING----")

        train_dataloader = base_dataset.get_train_dataloader(self.train_dataset)
        all_valid_dataloaders = [
            base_dataset.get_valid_dataloader(dataset) for dataset in self.all_validation_datasets
        ]

        train_devices = jax.local_devices()
        validate = len(self.all_validation_datasets) > 0
        if validate:
            validation_devices_count = self.all_validation_datasets[0].device_count
            validation_devices = train_devices[:validation_devices_count]

        train_states = initialize_states(
            config=self.config,
            dataloader=train_dataloader,
            devices=train_devices,
            statistics=self.statistics,
        )

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
            self.epoch += 1

            train_states = sync_batch_stats(train_states)

            def training_fun():
                return self.iterate_dataset(
                    states=train_states,
                    dataloader=train_dataloader,
                    train=True,
                    tqdm_description=f"GPUS: {len(train_devices)} EPOCH: {self.epoch}",  # , lr: {self.lr:.6f}",
                    raport_description="Training",
                    devices=train_devices,
                )

            if self.config.profile_training:
                # https://github.com/google/jax/issues/13009
                with jax.profiler.trace("./log", create_perfetto_link=True):
                    train_states = training_fun()
            else:
                train_states = training_fun()

            if self.is_at_skip(self.config.td.save_at_epochs):
                self.save_checkpoint(states=train_states)

            if self.is_at_skip(self.config.td.validate_at_epochs) and validate:
                validation_states = rereplicate_states(train_states, validation_devices)
                for dataloader in all_valid_dataloaders:
                    _ = self.iterate_dataset(
                        states=validation_states,
                        dataloader=dataloader,
                        train=False,
                        tqdm_description=f"GPUS: {len(validation_devices)} VAL:",
                        raport_description=dataloader.dataset.description,
                        devices=validation_devices,
                    )

                # TODO: Check if needed, add assert
                print("----REREPLICATING TRAIN STATE----")
                train_states = rereplicate_states(train_states, train_devices)

                # if self.is_at_skip(self.config.td.validate_scenarios_at_epochs):
                #     self.validate_all_scenarios_raport()

    def save_checkpoint(self, states):
        print("----SAVING CHECKPOINT----")

        state = flax.jax_utils.unreplicate(states)
        timestamp = cmh.get_timestamp(self.config)
        catalog = f"{self.config.output_catalog}/{self.config.current_time} - JAX GRAPH MODELS"
        cmh.create_folders(catalog)
        path = f"{catalog}/{timestamp} - MODEL"
        self.checkpointer.save(directory=path, item=state)

    @staticmethod
    def get_checkpoint(rank: int, path: str):
        return torch.load(path, map_location={"cuda:0": f"cuda:{rank}"})

    @staticmethod
    def load_checkpointed_net(path: str):
        print("----LOADING NET----")
        state = orbax.checkpoint.PyTreeCheckpointer().restore(directory=path)
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
            simulation_config=scenario.simulation_config,
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

        apply_net = get_apply_net(state)

        for scene in print_scenarios:
            simulation_runner.run_scenario(
                solve_function=partial(solve, apply_net=apply_net),
                scene=scene,
                config=config,
                run_config=simulation_runner.RunScenarioConfig(
                    catalog="GRAPH PLOT",
                    simulate_dirty_data=False,
                    plot_animation=True,
                ),
                get_scene_function=GraphModelDynamicJax.get_scene_function,
            )
            print("---")
        print(f"Plotting time: {int((time.time() - start_time) / 60)} min")
        # return catalog

    def clip_gradients(self, max_norm: float):
        raise NotImplementedError

    def iterate_dataset(
        self,
        states,
        dataloader,
        train: bool,
        tqdm_description: str,
        raport_description: str,
        devices,
    ):
        batch_tqdm = cmh.get_tqdm(dataloader, desc=tqdm_description, config=self.config, position=0)
        if train:
            dataloader.sampler.set_epoch(self.epoch)

        mean_loss_raport = LossRaport()

        gc.disable()

        for batch_id, batch_data in enumerate(batch_tqdm):
            states, loss_raport = self.calculate_loss(
                states, batch_data=batch_data, devices=devices, train=train
            )

            # TODO: Check / assert state consistency across GPUs
            # TODO: Check if data are randomized
            mean_loss_raport.add(loss_raport)
            if train:
                self.examples_seen += loss_raport.count  # * self.world_size

            loss_description = f"{tqdm_description} loss: {(mean_loss_raport.main):.4f}"
            if batch_id == len(batch_tqdm) - 1:
                self.save_raport(
                    states=states,
                    mean_loss_raport=mean_loss_raport,
                    description=raport_description,
                )
                mean_loss_raport = LossRaport()
                loss_description += " - raport saved"
            batch_tqdm.set_description(loss_description)

        gc.enable()
        return states

    def should_raport_training(self, batch_id: int, batches_count: int):
        return (
            batch_id == batches_count - 1
            or self.examples_seen % self.config.td.raport_at_examples == 0
        )

    def save_raport(self, states, mean_loss_raport, description: str):
        self.logger.writer.add_scalar(
            f"Loss/{description}/main",
            mean_loss_raport.main,
            self.examples_seen,
        )
        # state = flax.jax_utils.unreplicate(states)
        # learning_rate = state.opt_state.hyperparams["learning_rate"].item()
        # self.logger.writer.add_scalar(
        #     f"Loss/{description}/LearningRate",
        #     learning_rate,
        #     self.examples_seen,
        # )
        # for key, value in mean_loss_raport.get_iterator():
        #     self.logger.writer.add_scalar(
        #         f"Loss/{description}/{key}",
        #         value,
        #         self.examples_seen,
        #     )
        # self.logger.writer.add_graph(selFf)

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

    def calculate_loss(self, states, batch_data: List[List[Data]], devices, train):
        devices_count = len(devices)

        data = [get_layer_list_and_target_data(bd) for bd in batch_data]

        compare=True

        all_target_data = [
            data[d][1].normalized_new_displacement for d in range(devices_count)
        ]  ### NO AS TYPE .astype(np.float32)
        sharded_targets = jax.device_put_sharded(
            all_target_data, devices
        )  # TODO: check order with pmap
        
        if not compare:
            all_args = [
                prepare_input(layer_list) for layer_list in [data[d][0] for d in range(devices_count)]
            ]
            sharded_args = jax.device_put_sharded(all_args, devices)
            
            if train:
                states, losses = apply_model_train(states, sharded_args, sharded_targets)
            else:
                losses = apply_model_test(states, sharded_args, sharded_targets)

        else:
            other_target_data = [
                data[d][1].normalized_new_displacement_skinning for d in range(devices_count)
            ]
            sharded_other_targets = jax.device_put_sharded(
                other_target_data, devices
            )
            losses = apply_model_compare(sharded_targets=sharded_targets, sharded_other_targets=sharded_other_targets)
  
        displacement_loss = jnp.mean(losses) / SCALE
        # print(displacement_loss)

        batch_main_layer = data[0][0][0]
        graph_sizes_base = get_graph_sizes(batch_main_layer)
        num_graphs = len(graph_sizes_base) * devices_count

        loss_raport = LossRaport(
            main=displacement_loss.item(),
            displacement_loss=displacement_loss.item(),
            _count=num_graphs,
        )

        return states, loss_raport


def get_sample_args(dataloader):
    sample_batch_data = next(iter(dataloader))
    sample_layer_list, _ = get_layer_list_and_target_data(
        sample_batch_data[0]
    )  # get data for first device
    sample_args = prepare_input(sample_layer_list)
    return sample_args


def initialize_states(config, dataloader, devices, statistics):
    sample_args = get_sample_args(dataloader)
    init_state = create_train_state(
        jax.random.PRNGKey(42), sample_args, config.td.initial_learning_rate, statistics
    )

    states = flax.jax_utils.replicate(init_state, devices=devices)
    return states


def get_layer_list_and_target_data(batch_data):
    batch_layers = batch_data[0]
    target_data = batch_data[1]
    layer_list = [layer for layer in batch_layers]
    layer_list, target_data = convert_to_jax(layer_list, target_data)
    return layer_list, target_data


def rereplicate_states(states, devices):
    return flax.jax_utils.replicate(flax.jax_utils.unreplicate(states), devices)


def sync_batch_stats(states):
    cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, "x"), "x")
    return states.replace(batch_stats=cross_replica_mean(states.batch_stats))


def convert_to_jax(layer_list, target_data=None):
    for layer in layer_list:
        layer["x"] = thh.convert_tensor_to_jax(layer["x"])
        layer.edge_attr = thh.convert_tensor_to_jax(layer.edge_attr)
        layer.edge_index = thh.convert_tensor_to_jax(layer.edge_index)

    layer_sparse = layer_list[1]
    layer_sparse.edge_index_to_down = thh.convert_tensor_to_jax(layer_sparse.edge_index_to_down)
    layer_sparse.edge_attr_to_down = thh.convert_tensor_to_jax(layer_sparse.edge_attr_to_down)
    if target_data is None:
        return layer_list
    target_data.normalized_new_displacement = (
        thh.convert_tensor_to_jax(target_data.normalized_new_displacement) * SCALE
    )
    target_data.normalized_new_displacement_skinning = (
        thh.convert_tensor_to_jax(target_data.normalized_new_displacement_skinning) * SCALE
    )
    return layer_list, target_data


# TODO: all in Jax?
def solve(
    apply_net,
    scene: SceneInput,
    energy_functions: EnergyFunctions,
    initial_a,
    initial_t,
    timer=Timer(),
):
    _ = initial_a, initial_t

    dense_path = cmh.get_base_for_comarison()
    with timer["jax_calculator"]:
        if dense_path is None:
            scene.reduced.exact_acceleration, _ = Calculator.solve(
                scene=scene.reduced,
                energy_functions=energy_functions[1],  # 0],
                initial_a=scene.reduced.lifted_acceleration, #scene.reduced.exact_acceleration, #initial_reduced,
                timer=timer,
            )
        else:
            scene.exact_acceleration, scene.reduced.exact_acceleration = cmh.get_exact_acceleration(scene=scene, path=dense_path)

        scene.reduced.lifted_acceleration = scene.reduced.exact_acceleration

    device_number = 0  # using GPU 0

    with timer["jax_features_constructon"]:
        layers_list_0 = cmh.profile(lambda: scene.get_features_data(layer_number=0), baypass=True)
        layers_list_1 = cmh.profile(lambda: scene.get_features_data(layer_number=1), baypass=True)
        layers_list = [layers_list_0, layers_list_1]

    with timer["jax_data_movement"]:
        args = prepare_input(convert_to_jax(layers_list))
        args = jax.device_put(args, jax.local_devices()[device_number])
        # TODO: ADD STOP GRADIENT

    with timer["jax_net"]:
        scene.norm_lifted_new_displacement = apply_net(args) / SCALE

    with timer["jax_translation"]:
        scene.recentered_norm_lifted_new_displacement  = scene.recenter_by_reduced(new_displacement=scene.norm_lifted_new_displacement, reduced_exact_acceleration=scene.reduced.exact_acceleration)
        scene.lifted_acceleration = np.array(
            scene.from_displacement(scene.recentered_norm_lifted_new_displacement)
        )

    if dense_path is None:
        return scene.lifted_acceleration, None
    return scene.exact_acceleration, None


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


def create_train_state(rng, sample_args, learning_rate, statistics):
    params, batch_stats = CustomGraphNetJax(statistics=statistics).get_params(sample_args, rng)
    # jax.tree_util.tree_map(lambda x: x.shape, params)  # Checking output shapes

    # optimizer = optax.adam(learning_rate=learning_rate)
    optimizer = optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate)

    def a_fn(variables, args, train):
        return CustomGraphNetJax().apply(variables, args, train, mutable=["batch_stats"])

    return TrainState.create(apply_fn=a_fn, params=params, tx=optimizer, batch_stats=batch_stats)


def get_apply_net(state):
    variables = {"params": state["params"], "batch_stats": state["batch_stats"]}

    @jax.jit
    def apply_net(args):
        args = jax.lax.stop_gradient(args)
        return CustomGraphNetJax().apply(variables, args, train=False)

    return apply_net


def MSE(predicted, exact):
    return jnp.mean(jnp.linalg.norm(predicted - exact, axis=-1) ** 2)


def RMSE(predicted, exact):
    return jnp.sqrt(MSE(predicted, exact))


def get_loss_function(states, sharded_args, sharded_targets, train):
    def loss_function(params):
        variables = {"params": params, "batch_stats": states.batch_stats}
        sharded_net_result, non_trainable_params = states.apply_fn(variables, sharded_args, train)
        losses = RMSE(sharded_net_result, sharded_targets)
        new_batch_stats = non_trainable_params["batch_stats"]
        ###
        # new_batch_stats = flax.core.frozen_dict.unfreeze(new_batch_stats)
        # for key in new_batch_stats.keys():
        #     new_batch_stats[key]['BatchNorm_0']['mean'] = 0.
        #     new_batch_stats[key]['BatchNorm_0']['var'] = SCALE
        # new_batch_stats = flax.core.frozen_dict.freeze(new_batch_stats)
        ###
        return losses, new_batch_stats

    return loss_function



@partial(jax.pmap, axis_name="models")
def apply_model_compare(sharded_targets, sharded_other_targets):
    sharded_other_targets = jax.lax.stop_gradient(sharded_other_targets)
    losses = RMSE(sharded_other_targets, sharded_targets)
    return losses


@partial(jax.pmap, axis_name="models")
def apply_model_test(states, sharded_args, sharded_targets):
    sharded_args = jax.lax.stop_gradient(sharded_args)
    loss_fn = get_loss_function(states, sharded_args, sharded_targets, train=False)

    (losses, _) = loss_fn(states.params)
    return losses


@partial(jax.pmap, axis_name="models")
# @partial(jax.jit, static_argnames=("static_args"))
def apply_model_train(states, sharded_args, sharded_targets):
    loss_fn = get_loss_function(states, sharded_args, sharded_targets, train=True)

    (losses, new_batch_stats), grads = jax.value_and_grad(loss_fn, has_aux=True)(states.params)
    grads_pmean = jax.lax.pmean(grads, axis_name="models")
    states = states.apply_gradients(grads=grads_pmean, batch_stats=new_batch_stats)
    return states, losses


def save_tf_model(model_path, state, dataset):
    apply_net = get_apply_net(state)

    dataloader = base_dataset.get_train_dataloader(dataset)
    sample_args = get_sample_args(dataloader)

    apply_net_tf = tf.function(
        jax2tf.convert(apply_net, enable_xla=False),
        autograph=False,
    )

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [apply_net_tf.get_concrete_function(sample_args)], apply_net_tf
    )

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]
    tflite_float_model = converter.convert()

    with open(model_path, "wb") as f:
        f.write(tflite_float_model)

    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # tflite_quantized_model = converter.convert()

    # with open("./quantized.tflite", "wb") as f:
    #     f.write(tflite_quantized_model)
