import time

import deep_conmech.graph.setting.setting_input
import numpy as np
import torch
from deep_conmech.common import *
from deep_conmech.common.plotter import plotter_mapper
from deep_conmech.graph.data import data_base
from deep_conmech.graph.data.data_base import *
from deep_conmech.graph.net import CustomGraphNet
from deep_conmech.graph.setting import setting_input
from torch.utils.tensorboard import SummaryWriter

start = time.time()


def get_writer():
    return SummaryWriter(
        f"./log/{thh.CURRENT_TIME} \
| lr {config.INITIAL_LR} - {config.FINAL_LR} ({config.LR_GAMMA}) \
| dr {config.DROPOUT_RATE} \
| ah {config.ATTENTION_HEADS} \
| dzf {config.DATA_ZERO_FORCES} drv {config.DATA_ROTATE_VELOCITY}  \
| md {config.MESH_DENSITY} ad {config.ADAPTIVE_MESH} \
| vpes {config.VAL_PRINT_EPISODE_STEPS} \
| ung {config.U_NOISE_GAMMA} - rf u {config.U_IN_RANDOM_FACTOR} v {config.V_IN_RANDOM_FACTOR} \
| bs {config.BATCH_SIZE} bie {config.BATCHES_IN_EPOCH} \
| ld {config.LATENT_DIM} \
| lc {config.ENC_LAYER_COUNT}-{config.PROC_LAYER_COUNT}-{config.DEC_LAYER_COUNT} \
| mp {config.MESSAGE_PASSES}"
    )


class ErrorResult:
    value = 0


class GraphModelDynamic:
    def __init__(
        self, train_dataset, all_val_datasets, print_scenarios,
    ):
        self.train_dataloader = data_base.get_fast_dataloader(train_dataset)
        self.all_val_data = [
            (dataset, data_base.get_print_dataloader(dataset))
            for dataset in all_val_datasets
        ]
        self.print_scenarios = print_scenarios
        self.writer = get_writer()
        self.loss_labels = [
            "Main",
            "L2_diff",
            "RMSE_acc",
        ]  # "L2_diff", "L2_no_acc"]  # . "L2_main", "v_step_diff"]

        self.net = CustomGraphNet().to(thh.device)
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

    def boundary_edges_counts(self, batch):
        return thh.to_np_long(batch.boundary_edges_count).tolist()

    def get_split(self, batch, index, graph_sizes):
        value = batch.x[:, index * config.DIM : (index + 1) * config.DIM]
        value_split = value.split(graph_sizes)
        return value_split

    ################

    def solve(self, setting, print_time=False):
        self.net.eval()
        batch = setting.get_data().to(thh.device)

        start = time.time()
        normalized_a_cuda = self.net(
            batch
        )  # + setting.predicted_normalized_a_mean_cuda
        if print_time:
            print("Graph solve time: ", time.time() - start)

        normalized_a = thh.to_np_double(normalized_a_cuda)
        a = setting.rotate_from_upward(normalized_a)
        return a

    ################

    def save(self):
        print("Saving model")
        timestamp = thh.get_timestamp()
        catalog = f"output/GRAPH - {thh.CURRENT_TIME}"
        thh.create_folders(catalog)
        path = f"{catalog}/{timestamp} - MODEL.pt"
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        print(f"Loading model {path}")
        self.net.load_state_dict(torch.load(path))
        self.net.eval()

    ################

    def train(self):
        # epoch_tqdm = tqdm(range(config.EPOCHS), desc="EPOCH")
        # for epoch in epoch_tqdm:
        examples_seen = 0

        start_time = time.time()
        epoch = 0
        while True:
            epoch += 1
            # with profile(with_stack=True, profile_memory=True) as prof:

            batch_tqdm = thh.get_tqdm(self.train_dataloader, desc="Batch")
            for batch_number, batch in enumerate(batch_tqdm):

                train_loss, train_loss_array = self.train_step(batch)
                examples_seen += len(batch)
                self.batch_raport(
                    batch_tqdm, train_loss, train_loss_array, examples_seen, epoch
                )

            self.scheduler.step()

            if epoch % config.VALIDATE_AT_EPOCHS == 0:
                self.epoch_raport(batch_tqdm, examples_seen, epoch)
            # if epoch % config.VAL_ROLLOUT_AT_EPOCHS == 0:
            #    self.validate_rollout(examples_seen)

            time_enlapsed = time.time() - start_time
            if time_enlapsed > config.DRAW_AT_MINUTES * 60:
                self.save()
                self.print_raport()
                start_time = time.time()

            # print(prof.key_averages().table(row_limit=10))

    #################

    def train_step(self, base_batch):
        self.net.train()
        batch = base_batch

        # for _ in range(config.TRAIN_ITERATIONS):
        self.net.zero_grad()

        loss, loss_array_np, batch = self.E(batch)
        loss.backward()
        # self.clip_gradients()
        self.optimizer.step()

        return loss, loss_array_np

    def test_step(self, batch):
        self.net.eval()

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

    def E(self, batch):
        graph_couts = [1 for i in range(batch.num_graphs)]
        graph_sizes = self.graph_sizes(batch)
        boundary_nodes_counts = self.boundary_nodes_counts(batch)
        boundary_edges_counts = self.boundary_edges_counts(batch)
        dim_graph_sizes = [size * config.DIM for size in graph_sizes]
        dim_dim_graph_sizes = [
            (size * config.DIM) ** config.DIM for size in graph_sizes
        ]

        loss = 0.0
        loss_array = np.zeros([3])

        batch_cuda = batch.to(thh.device)
        predicted_normalized_a_split = self.net(batch_cuda).split(graph_sizes)

        reshaped_C_split = batch.reshaped_C.split(dim_dim_graph_sizes)
        normalized_E_split = batch.normalized_E.split(dim_graph_sizes)
        normalized_a_correction_split = batch.normalized_a_correction.split(graph_sizes)
        normalized_boundary_v_old_split = batch.normalized_boundary_v_old.split(
            boundary_nodes_counts
        )
        normalized_boundary_points_split = batch.normalized_boundary_points.split(
            boundary_nodes_counts
        )
        boundary_edges_split = batch.boundary_edges.split(boundary_edges_counts)
        normalized_closest_obstacle_normals_split = batch.normalized_closest_obstacle_normals.split(
            boundary_edges_counts
        )
        normalized_closest_obstacle_origins_split = batch.normalized_closest_obstacle_origins.split(
            boundary_edges_counts
        )

        if hasattr(batch, "exact_normalized_a_torch"):
            exact_normalized_a_split = batch.exact_normalized_a_torch.split(graph_sizes)

        # dataset = StepDataset(batch.num_graphs)
        for i in range(batch.num_graphs):
            C_side_len = graph_sizes[i] * config.DIM
            C = reshaped_C_split[i].reshape(C_side_len, C_side_len)
            normalized_E = normalized_E_split[i]
            normalized_a_correction = normalized_a_correction_split[i]
            #
            normalized_boundary_v_old = normalized_boundary_v_old_split[i]
            normalized_boundary_points = normalized_boundary_points_split[i]
            boundary_edges = boundary_edges_split[i]
            normalized_closest_obstacle_normals = normalized_closest_obstacle_normals_split[
                i
            ]
            normalized_closest_obstacle_origins = normalized_closest_obstacle_origins_split[
                i
            ]

            predicted_normalized_a = predicted_normalized_a_split[i]
            # + setting.predicted_normalized_a_mean_cuda
            # predicted_normalized_L2 = setting_input.L2_normalized_correction_cuda(
            #    predicted_normalized_a, C, normalized_E, normalized_a_correction
            # )

            predicted_normalized_L2 = setting_input.L2_normalized_obstacle_correction_cuda(
                predicted_normalized_a,
                C,
                normalized_E,
                normalized_boundary_v_old,
                normalized_boundary_points,
                boundary_edges,
                normalized_closest_obstacle_normals,
                normalized_closest_obstacle_origins,
                normalized_a_correction,
            )

            loss += predicted_normalized_L2

            if hasattr(batch, "exact_normalized_a_torch"):
                exact_normalized_a = exact_normalized_a_split[i]
                if exact_normalized_a is not None:
                    exact_normalized_L2 = setting_input.L2_normalized_correction_cuda(
                        exact_normalized_a, C, normalized_E, normalized_a_correction
                    )
                    loss_array[1] += float(
                        (predicted_normalized_L2 - exact_normalized_L2)
                        / torch.abs(exact_normalized_L2)
                    )

                    # no_acc_normalized_L2 = setting_input.L2_normalized_correction_cuda(
                    #    torch.zeros_like(predicted_normalized_a), C, normalized_E, normalized_a_correction
                    # )
                    # loss_array[2] += float(
                    #    (no_acc_normalized_L2 - exact_normalized_L2)
                    #    / torch.abs(exact_normalized_L2)
                    # )
                    loss_array[2] += float(
                        thh.rmse_torch(predicted_normalized_a, exact_normalized_a)
                    )

                # new_setting = setting.iterate(helpers.to_np(a_predicted))
                # new_setting.set_forces(copy.deepcopy(setting.forces))
                # dataset.set(i, new_setting)

            # new_batch = next(iter(dataset.get_dataloader()))
        loss_array[0] = loss

        loss /= batch.num_graphs
        loss_array /= batch.num_graphs
        return loss, loss_array, None  # new_batch

    #################

    def batch_raport(self, tqdm, loss, loss_array, examples_seen, epoch):
        loss = float(loss)
        tqdm.set_description(
            f"EPOCH: {epoch}, lr: {self.lr:.6f}, train loss: {loss:.4f}"
        )
        self.writer.add_scalar(
            "Loss/Training/LearningRate", self.lr, examples_seen,
        )
        for i in range(len(loss_array)):
            self.writer.add_scalar(
                f"Loss/Training/{self.loss_labels[i]}", loss_array[i], examples_seen,
            )

    def epoch_raport(self, tqdm, examples_seen, epoch):
        for dataset, dataloader in self.all_val_data:
            # print(f"Validating {dataset.relative_path} |", end='')
            labels_count = len(self.loss_labels)
            mean_loss_array = np.zeros([labels_count])

            batch_tqdm = thh.get_tqdm(dataloader, desc=dataset.relative_path)
            for _, batch in enumerate(batch_tqdm):
                mean_loss_array += self.test_step(batch)
            mean_loss_array = mean_loss_array / len(dataloader)

            # tqdm.set_description(
            #    f"EPOCH: {epoch}, lr: {self.lr:.6f}, {dataset.relative_path} val loss {mean_loss_array[0]:.4f}"
            # )

            for i in range(labels_count):
                self.writer.add_scalar(
                    f"Loss/Validation/{dataset.relative_path}/{self.loss_labels[i]}",
                    mean_loss_array[i],
                    examples_seen,
                )
        print("---")

    #################

    def print_raport(self):
        path = f"GRAPH - {thh.CURRENT_TIME}/{thh.get_timestamp()} - RESULT"
        for scenario in self.print_scenarios:

            plotter_mapper.print_one_dynamic(
                lambda setting: self.solve(setting, print_time=False),
                scenario,
                path,
                simulate_dirty_data=False,
                print_base=False,  #######################
                description="Printing raport",
            )

    """
    def validate_rollout(self, examples_seen):
        for forces_function in self.rollout_forces_functions:
            error_result = ErrorResult()

            episode_steps = config.VAL_PRINT_EPISODE_STEPS
            _validate = lambda time, setting, base_setting, a, base_a: self.calculate_error(
                setting, base_setting, a, base_a, error_result, episode_steps
            )

            mapper.map_time(
                True,
                _validate,
                config.TRAIN_CORNERS,
                episode_steps,
                lambda setting: self.solve(setting, print_time=False),
                forces_function,
                self.obstacles,
                simulate_dirty_data=False,
                description=f"Validating rollout - {forces_function.__name__}",
                mesh_type=config.MESH_TYPE_VALIDATION,
                mesh_density=config.MESH_DENSITY,
            )

            self.writer.add_scalar(
                f"Loss/Validation/rollout {forces_function.__name__}",
                error_result.value,
                examples_seen,
            )

    def calculate_error(
        self, setting, base_setting, a, base_a, error_result, episode_steps
    ):
        cleaned_normalized_a = setting.rotate_to_upward(a)
        function = setting.get_L2_full_normalized_correction_np()
        predicted_normalized_L2 = function(cleaned_normalized_a) * 0. ##################################
        error_result.value += float(predicted_normalized_L2 / episode_steps)
    """


#####################


class GraphModelStatic(GraphModelDynamic):
    def b(self, batch):
        x = batch.pos
        rx = (x[..., 0] - config.min[0]) / nph.len_x(corners)
        ry = (x[..., 0] - config.min[0]) / nph.len_y(corners)
        result = torch.stack((rx, ry), -1)
        return result

    def predict(self, batch):
        psi = self.net(batch)
        # b = self.b(batch)
        # return torch.multiply(psi, b)
        return psi

    def E(self, batch):
        graph_sizes = np.ediff1d(batch.ptr).tolist()
        u = self.predict(batch)
        u_split = u.split(graph_sizes)

        # forces_split = self.get_split(batch, 0, graph_sizes)

        loss_array = torch.zeros(batch.num_graphs)
        for i in range(batch.num_graphs):
            # forces_i = forces_split[i]
            # forces_i =thh.to_torch_float([-0.1, 0.1])
            forces_i = batch.setting[i].FORCES()
            l2 = batch.setting[i].L2_torch(u_split[i], forces_i)
            loss_array[i] = l2

        loss = torch.mean(loss_array)
        return loss
