import json
import time
from argparse import ArgumentError

import torch
from torch.utils.tensorboard.writer import SummaryWriter

from deep_conmech import scenarios
from deep_conmech.common.training_config import TrainingConfig
from deep_conmech.graph.data import data_base
from deep_conmech.graph.data.data_base import *
from deep_conmech.graph.helpers import thh
from deep_conmech.graph.net import CustomGraphNet
from deep_conmech.graph.setting import setting_input


def get_and_init_writer(config: TrainingConfig):
    writer = SummaryWriter(f"./log/{config.CURRENT_TIME}")

    def pretty_json(value):
        dictionary = vars(value)
        json_str = json.dumps(dictionary, indent=2)
        return "".join("\t" + line for line in json_str.splitlines(True))

    writer.add_text(f"{config.CURRENT_TIME}_PARAMETERS.txt", pretty_json(config.td), global_step=0)
    return writer


# | ung {config.U_NOISE_GAMMA} - rf u {config.U_IN_RANDOM_FACTOR} v {config.V_IN_RANDOM_FACTOR} \
# | dzf {training_config.DATA_ZERO_FORCES} drv {training_config.DATA_ROTATE_VELOCITY}  \
# | vpes {config.EPISODE_STEPS} \


class ErrorResult:
    value = 0


class GraphModelDynamic:
    def __init__(
            self,
            train_dataset,
            all_val_datasets,
            net: CustomGraphNet,
            config: TrainingConfig,
    ):
        self.config = config
        self.all_val_datasets = all_val_datasets
        self.dim = train_dataset.dimension  # TODO: Check validation datasets
        self.train_dataset = train_dataset
        self.writer = get_and_init_writer(self.config)
        self.loss_labels = [
            "L2",
            "L2_diff",
            "RMSE_acc",
        ]  # "L2_diff", "L2_no_acc"]  # . "L2_main", "v_step_diff"]
        self.labels_count = len(self.loss_labels)
        self.tqdm_loss_index = 0

        self.net = net
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.config.td.INITIAL_LR,  # weight_decay=5e-4
        )
        lr_lambda = lambda epoch: max(
            self.config.td.LR_GAMMA ** epoch,
            self.config.td.FINAL_LR / self.config.td.INITIAL_LR,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lr_lambda
        )

    @property
    def lr(self):
        return float(self.scheduler.get_last_lr()[0])

    def graph_sizes(self, batch):
        graph_sizes = np.ediff1d(thh.to_np_long(batch.ptr)).tolist()
        return graph_sizes

    def boundary_nodes_counts(self, batch):
        return thh.to_np_long(batch.boundary_nodes_count).tolist()

    def get_split(self, batch, index, dim, graph_sizes):
        value = batch.x[:, index * dim: (index + 1) * dim]
        value_split = value.split(graph_sizes)
        return value_split

    ################

    def train(self):
        # epoch_tqdm = tqdm(range(config.EPOCHS), desc="EPOCH")
        # for epoch in epoch_tqdm:
        start_time = time.time()
        last_valid_time = start_time
        examples_seen = 0
        epoch_number = 0
        print("----TRAINING----")
        while True:
            epoch_number += 1
            # with profile(with_stack=True, profile_memory=True) as prof:

            loss_array, es = self.iterate_dataset(
                dataset=self.train_dataset,
                dataloader_function=data_base.get_train_dataloader,
                step_function=self.train_step,
                description=f"EPOCH: {epoch_number}",  # , lr: {self.lr:.6f}",
            )
            examples_seen += es
            self.training_raport(loss_array=loss_array, examples_seen=examples_seen)

            self.scheduler.step()

            current_time = time.time()
            elapsed_time = current_time - last_valid_time
            if elapsed_time > self.config.td.VALIDATE_AT_MINUTES * 60:
                print(f"--Training time: {(elapsed_time / 60):.4f} min")
                self.save_net()
                self.validation_raport(
                    examples_seen=examples_seen
                )
                self.train_dataset.update_data()
                last_valid_time = time.time()

            # print(prof.key_averages().table(row_limit=10))

    ################

    def save_net(self):
        print("----SAVING----")
        timestamp = cmh.get_timestamp(self.config)
        catalog = f"output/{self.config.CURRENT_TIME} - GRAPH MODELS"
        cmh.create_folders(catalog)
        path = f"{catalog}/{timestamp} - MODEL.pt"
        self.net.save(path)

    ################

    @staticmethod
    def get_newest_saved_model_path():
        def get_index(path):
            return int(path.split("/")[-1].split(" ")[0])

        saved_model_paths = cmh.find_files_by_extension("output", "pt")
        if not saved_model_paths:
            raise ArgumentError("No saved models")

        newest_index = np.argmax(
            np.array([get_index(path) for path in saved_model_paths])
        )
        path = saved_model_paths[newest_index]

        print(f"Taking saved model {path.split('/')[-1]}")
        return path

    @staticmethod
    def get_setting_function(
            scenario: Scenario,
            config: TrainingConfig,
            randomize=False,
            create_in_subprocess: bool = False,
    ) -> SettingInput:  # "SettingIterable":
        setting = SettingInput(
            mesh_data=scenario.mesh_data,
            body_prop=scenario.body_prop,
            obstacle_prop=scenario.obstacle_prop,
            schedule=scenario.schedule,
            config=config,
            create_in_subprocess=create_in_subprocess,
        )
        setting.set_randomization(randomize)
        setting.set_obstacles(scenario.obstacles)
        return setting

    @staticmethod
    def plot_all_scenarios(net: CustomGraphNet, print_scenarios, config: Config):
        print("----PLOTTING----")
        start_time = time.time()
        timestamp = cmh.get_timestamp(config)
        catalog = f"GRAPH PLOT/{timestamp} - RESULT"
        for scenario in print_scenarios:
            simulation_runner.plot_scenario(
                solve_function=net.solve,
                scenario=scenario,
                config=config,
                catalog=catalog,
                simulate_dirty_data=False,
                plot_animation=True,
                get_setting_function=GraphModelDynamic.get_setting_function,
            )
            print("---")
        print(f"Plotting time: {int((time.time() - start_time) / 60)} min")
        # return catalog

    #################

    def train_step(self, batch):
        self.net.train()
        self.net.zero_grad()

        loss, loss_array_np, batch = self.E(batch)
        loss.backward()
        # self.clip_gradients()
        self.optimizer.step()

        return loss_array_np

    def test_step(self, batch):
        self.net.eval()

        with torch.no_grad():  # with tc.set_grad_enabled(train):
            _, loss_array_np, _ = self.E(batch)

        return loss_array_np

    def clip_gradients(self):
        """clip_grad_norm (which is actually deprecated in favor of clip_grad_norm_ 
        following the more consistent syntax of a trailing _ when in-place modification is performed)"""
        parameters = self.net.parameters()
        # norms = [np.max(np.abs(p.grad.cpu().detach().numpy())) for p in parameters]
        # total_norm = np.max(norms)_
        # print("total_norm", total_norm)
        torch.nn.utils.clip_grad_norm_(parameters, self.config.td.GRADIENT_CLIP)

    #################

    def iterate_dataset(self, dataset, dataloader_function, step_function, description):
        dataloader = dataloader_function(dataset)
        batch_tqdm = cmh.get_tqdm(dataloader, desc=description, config=self.config)
        # range(len()) -> enumerate

        examples_seen = 0
        mean_loss_array = np.zeros(self.labels_count)
        for _, batch in enumerate(batch_tqdm):
            # len(batch) ?
            loss_array = step_function(batch)
            examples_seen += batch.num_graphs
            mean_loss_array += loss_array * (batch.num_graphs / len(dataset))
            batch_tqdm.set_description(
                f"{description} loss: {(loss_array[self.tqdm_loss_index]):.4f}"
            )
        return mean_loss_array, examples_seen

    def training_raport(self, loss_array, examples_seen):
        self.writer.add_scalar(
            "Loss/Training/LearningRate", self.lr, examples_seen,
        )
        for i in range(len(loss_array)):
            self.writer.add_scalar(
                f"Loss/Training/{self.loss_labels[i]}", loss_array[i], examples_seen,
            )

    def validation_raport(self, examples_seen):
        print("----VALIDATING----")
        start_time = time.time()

        mean_loss_array = np.zeros(self.labels_count)
        for dataset in self.all_val_datasets:
            loss_array, _ = self.iterate_dataset(
                dataset=dataset,
                dataloader_function=data_base.get_valid_dataloader,
                step_function=self.test_step,
                description=dataset.relative_path,
            )
            mean_loss_array += loss_array / len(self.all_val_datasets)
            for i in range(self.labels_count):
                self.writer.add_scalar(
                    f"Loss/Validation/{dataset.relative_path}/{self.loss_labels[i]}",
                    loss_array[i],
                    examples_seen,
                )

        for i in range(self.labels_count):
            self.writer.add_scalar(
                f"Loss/Validation/{self.loss_labels[i]}",
                mean_loss_array[i],
                examples_seen,
            )

        validation_time = time.time() - start_time
        print(f"--Validation time: {(validation_time / 60):.4f} min")

    #################

    def E(self, batch, test_using_true_solution=False):
        # graph_couts = [1 for i in range(batch.num_graphs)]
        graph_sizes = self.graph_sizes(batch)
        boundary_nodes_counts = self.boundary_nodes_counts(batch)
        dim_graph_sizes = [size * self.dim for size in graph_sizes]
        dim_dim_graph_sizes = [(size * self.dim) ** self.dim for size in graph_sizes]

        loss = 0.0
        loss_array = np.zeros(self.labels_count)

        batch_cuda = batch.to(self.net.device)
        predicted_normalized_a_split = self.net(batch_cuda).split(graph_sizes)

        reshaped_C_split = batch.reshaped_C.split(dim_dim_graph_sizes)
        normalized_E_split = batch.normalized_E.split(dim_graph_sizes)
        normalized_a_correction_split = batch.normalized_a_correction.split(graph_sizes)
        normalized_boundary_v_old_split = batch.normalized_boundary_v_old.split(
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
        normalized_boundary_obstacle_normals_split = batch.normalized_boundary_obstacle_normals.split(
            boundary_nodes_counts
        )
        boundary_nodes_volume_split = batch.boundary_nodes_volume.split(
            boundary_nodes_counts
        )

        if hasattr(batch, "exact_normalized_a"):
            exact_normalized_a_split = batch.exact_normalized_a.split(graph_sizes)

        # dataset = StepDataset(batch.num_graphs)
        for i in range(batch.num_graphs):
            C_side_len = graph_sizes[i] * self.dim
            C = reshaped_C_split[i].reshape(C_side_len, C_side_len)
            normalized_E = normalized_E_split[i]
            normalized_a_correction = normalized_a_correction_split[i]
            predicted_normalized_a = predicted_normalized_a_split[i]

            L2_args = dict(
                a_correction=normalized_a_correction,
                C=C,
                E=normalized_E,
                boundary_v_old=normalized_boundary_v_old_split[i],
                boundary_nodes=normalized_boundary_nodes_split[i],
                boundary_normals=normalized_boundary_normals_split[i],
                boundary_obstacle_nodes=normalized_boundary_obstacle_nodes_split[i],
                boundary_obstacle_normals=normalized_boundary_obstacle_normals_split[i],
                boundary_nodes_volume=boundary_nodes_volume_split[i],
                obstacle_prop=scenarios.default_obstacle_prop,  # TODO: generalize
                time_step=0.01,  # TODO: generalize
            )

            if test_using_true_solution:
                predicted_normalized_a = self.use_true_solution(
                    predicted_normalized_a, L2_args
                )

            predicted_normalized_L2 = setting_input.L2_normalized_obstacle_correction(
                cleaned_a=predicted_normalized_a, **L2_args
            )
            if hasattr(batch, "exact_normalized_a"):
                exact_normalized_a = exact_normalized_a_split[i]

            if self.config.td.L2_LOSS:
                loss += predicted_normalized_L2
            else:
                loss += thh.rmse_torch(predicted_normalized_a, exact_normalized_a)

            loss_array[0] += predicted_normalized_L2
            if hasattr(batch, "exact_normalized_a"):
                exact_normalized_L2 = setting_input.L2_normalized_obstacle_correction(
                    cleaned_a=exact_normalized_a, **L2_args
                )
                loss_array[1] += float(
                    (predicted_normalized_L2 - exact_normalized_L2)
                    / torch.abs(exact_normalized_L2)
                )
                loss_array[2] += float(
                    thh.rmse_torch(predicted_normalized_a, exact_normalized_a)
                )

        loss /= batch.num_graphs
        loss_array /= batch.num_graphs
        return loss, loss_array, None  # new_batch

    def use_true_solution(self, predicted_normalized_a, L2_args):
        function = lambda normalized_a_vector: setting_input.L2_normalized_obstacle_correction(
            cleaned_a=thh.to_torch_double(nph.unstack(normalized_a_vector, dim=2)).to(
                self.net.device),
            **L2_args,
        ).item()

        # @v = function(thh.to_np_double(torch.zeros_like(predicted_normalized_a)))
        predicted_normalized_a = thh.to_torch_double(
            nph.unstack(
                Solver.minimize(
                    function,
                    thh.to_np_double(torch.zeros_like(predicted_normalized_a)),
                ),
                dim=2,
            )
        ).to(self.net.device)
        return predicted_normalized_a
