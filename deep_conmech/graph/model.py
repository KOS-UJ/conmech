import time
from re import A

import numpy as np
import torch
from conmech.helpers import cmh
from deep_conmech import scenarios
from deep_conmech.common import *
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.graph.data import data_base
from deep_conmech.graph.data.data_base import *
from deep_conmech.graph.helpers import thh
from deep_conmech.graph.setting import setting_input
from torch.utils.tensorboard import SummaryWriter

start = time.time()


def get_writer():
    return SummaryWriter(
        f"./log/{cmh.CURRENT_TIME} \
| lr {config.INITIAL_LR} - {config.FINAL_LR} ({config.LR_GAMMA}) \
| dr {config.DROPOUT_RATE} \
| ah {config.ATTENTION_HEADS} \
| ln {config.LAYER_NORM} \
| l2l {config.L2_LOSS} \
| dzf {config.DATA_ZERO_FORCES} drv {config.DATA_ROTATE_VELOCITY}  \
| md {config.MESH_DENSITY} ad {config.ADAPTIVE_TRAINING_MESH} \
| ung {config.U_NOISE_GAMMA} - rf u {config.U_IN_RANDOM_FACTOR} v {config.V_IN_RANDOM_FACTOR} \
| bs {config.BATCH_SIZE} vbs {config.VALID_BATCH_SIZE} bie {config.SYNTHETIC_BATCHES_IN_EPOCH} \
| ld {config.LATENT_DIM} \
| lc {config.ENC_LAYER_COUNT}-{config.PROC_LAYER_COUNT}-{config.DEC_LAYER_COUNT} \
| mp {config.MESSAGE_PASSES}"
    )


# | vpes {config.EPISODE_STEPS} \


class ErrorResult:
    value = 0


class GraphModelDynamic:
    def __init__(self, train_dataset, all_val_datasets, print_scenarios, net):
        self.all_val_datasets = all_val_datasets
        self.print_scenarios = print_scenarios
        self.dim = train_dataset.dimension  # TODO: Check validation datasets
        self.train_dataset = train_dataset
        self.train_dataloader = data_base.get_train_dataloader(train_dataset)
        self.all_val_data = [
            (dataset, data_base.get_valid_dataloader(dataset))
            for dataset in all_val_datasets
        ]
        self.writer = get_writer()
        self.loss_labels = [
            "L2",
            "L2_diff",
            "RMSE_acc",
        ]  # "L2_diff", "L2_no_acc"]  # . "L2_main", "v_step_diff"]
        self.labels_count = len(self.loss_labels)
        self.tqdm_loss_index = -1

        self.net = net
        self.optimizer = torch.optim.Adam(
            self.net.parameters(), lr=config.INITIAL_LR,  # weight_decay=5e-4
        )
        lr_lambda = lambda epoch: max(
            config.LR_GAMMA ** epoch, config.FINAL_LR / config.INITIAL_LR
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
        value = batch.x[:, index * dim : (index + 1) * dim]
        value_split = value.split(graph_sizes)
        return value_split

    ################

    def save(self):
        print("Saving model")
        timestamp = cmh.get_timestamp()
        catalog = f"output/{cmh.CURRENT_TIME} - GRAPH"
        cmh.create_folders(catalog)
        path = f"{catalog}/{timestamp} - MODEL.pt"
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        print(f"Loading model {path}")
        self.net.load_state_dict(torch.load(path))
        self.net.eval()

    ################

    def get_epoch_description(self, epoch_number, loss_array=None):
        loss_description = (
            ""
            if loss_array is None
            else f", loss: {loss_array[self.tqdm_loss_index]:.4f}"
        )
        return f"EPOCH: {epoch_number}, lr: {self.lr:.6f}{loss_description}"

    def train(self):
        # epoch_tqdm = tqdm(range(config.EPOCHS), desc="EPOCH")
        # for epoch in epoch_tqdm:
        start_time = time.time()
        last_plotting_time = start_time
        examples_seen = 0
        epoch_number = 0
        print("----TRAINING----")
        while True:
            epoch_number += 1
            # with profile(with_stack=True, profile_memory=True) as prof:

            epoch_tqdm = cmh.get_tqdm(
                self.train_dataloader, desc=self.get_epoch_description(epoch_number)
            )
            batch_count = len(epoch_tqdm)

            total_loss_array = np.zeros(self.labels_count)
            for batch_number, batch in enumerate(epoch_tqdm):
                epoch_tqdm.set_description(
                    self.get_epoch_description(
                        epoch_number, total_loss_array / (batch_number + 1)
                    )
                )
                total_loss_array += self.train_step(batch)
                examples_seen += len(batch)

                if batch_number == batch_count - 1:
                    total_loss_array /= batch_count
                    self.training_raport(
                        epoch_tqdm, total_loss_array, examples_seen, epoch_number
                    )

            self.scheduler.step()

            current_time = time.time()
            elapsed_time = current_time - start_time
            if epoch_number % config.VALIDATE_AT_EPOCHS == 0:
                self.validation_raport(examples_seen, epoch_number, elapsed_time)
                self.train_dataset.update_data()

            if current_time > config.DRAW_AT_MINUTES * 60 + last_plotting_time:
                # self.save()
                self.plot_scenarios(elapsed_time)
                last_plotting_time = time.time()

            # print(prof.key_averages().table(row_limit=10))

    #################

    def train_step(self, batch):
        self.net.train()

        # for _ in range(config.TRAIN_ITERATIONS):
        self.net.zero_grad()

        loss, loss_array_np, batch = self.E(batch)
        loss.backward()
        # self.clip_gradients()
        self.optimizer.step()

        return loss_array_np

    def test_step(self, batch):
        ###############self.net.eval()
        self.net.train()

        # forward
        with torch.no_grad():  # with tc.set_grad_enabled(train):
            _, loss_array_np, _ = self.E(batch)

        return loss_array_np

    def clip_gradients(self):
        """
        clip_grad_norm (which is actually deprecated in favor of clip_grad_norm_ 
        following the more consistent syntax of a trailing _ when in-place modification is performed)"""
        parameters = self.net.parameters()
        # norms = [np.max(np.abs(p.grad.cpu().detach().numpy())) for p in parameters]
        # total_norm = np.max(norms)_
        # print("total_norm", total_norm)
        torch.nn.utils.clip_grad_norm_(parameters, config.GRADIENT_CLIP)

    #################

    def training_raport(self, tqdm, loss_array, examples_seen, epoch_number):
        tqdm.set_description(self.get_epoch_description(epoch_number, loss_array))
        self.writer.add_scalar(
            "Loss/Training/LearningRate", self.lr, examples_seen,
        )
        for i in range(len(loss_array)):
            self.writer.add_scalar(
                f"Loss/Training/{self.loss_labels[i]}", loss_array[i], examples_seen,
            )

    def print_elapsed_time(self, elapsed_time):
        print(f"Time elapsed: {(elapsed_time / 60):.4f} min")

    def validation_raport(self, examples_seen, epoch_number, elapsed_time):
        print("----VALIDATING----")
        self.print_elapsed_time(elapsed_time)
        total_loss_array = np.zeros(self.labels_count)
        for dataset, dataloader in self.all_val_data:
            mean_loss_array = np.zeros(self.labels_count)

            batch_tqdm = cmh.get_tqdm(dataloader, desc=dataset.relative_path)
            # range(len()) -> enumerate

            for _, batch in enumerate(batch_tqdm):
                loss_array = self.test_step(batch)
                mean_loss_array += loss_array
                batch_tqdm.set_description(
                    f"{dataset.relative_path} loss: {(loss_array[self.tqdm_loss_index]):.4f}"
                )
            mean_loss_array = mean_loss_array / len(dataloader)

            for i in range(self.labels_count):
                self.writer.add_scalar(
                    f"Loss/Validation/{dataset.relative_path}/{self.loss_labels[i]}",
                    mean_loss_array[i],
                    examples_seen,
                )
            total_loss_array += mean_loss_array

        total_loss_array /= len(self.all_val_data)
        for i in range(self.labels_count):
            self.writer.add_scalar(
                f"Loss/Validation/{self.loss_labels[i]}",
                total_loss_array[i],
                examples_seen,
            )
        print("---")

    def plot_scenarios(self, elapsed_time):
        print("----PLOTTING----")
        self.print_elapsed_time(elapsed_time)
        start_time = time.time()
        timestamp = cmh.get_timestamp()
        for scenario in self.print_scenarios:
            plotter_mapper.print_one_dynamic(
                self.net.solve,
                scenario,
                SettingInput.get_setting,
                catalog=f"GRAPH/{timestamp} - RESULT",
                simulate_dirty_data=False,
                draw_base=False,  ###
                draw_detailed=False,
                description="Raport",
                plot_images=False,
                plot_animation=True,
            )

        print(f"Plotting time: {int((time.time() - start_time)/60)} min")
        print("----")

    #################

    def E(self, batch):
        # graph_couts = [1 for i in range(batch.num_graphs)]
        graph_sizes = self.graph_sizes(batch)
        boundary_nodes_counts = self.boundary_nodes_counts(batch)
        dim_graph_sizes = [size * self.dim for size in graph_sizes]
        dim_dim_graph_sizes = [(size * self.dim) ** self.dim for size in graph_sizes]

        loss = 0.0
        loss_array = np.zeros(self.labels_count)

        batch_cuda = batch.to(thh.device)
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
                obstacle_prop=scenarios.obstacle_prop,  # TODO: generalize
                time_step=0.01,  # TODO: generalize
            )

            predicted_normalized_L2 = setting_input.L2_normalized_obstacle_correction(
                cleaned_a=predicted_normalized_a, **L2_args
            )

            if config.L2_LOSS:
                loss += predicted_normalized_L2
            else:
                exact_normalized_a = exact_normalized_a_split[i]
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
