import torch
from deep_conmech.common import *
from deep_conmech.graph.helpers import thh
from deep_conmech.graph.setting.setting_randomized import *
from deep_conmech.simulator.setting.setting_forces import *
from torch_geometric.data import Data


def L2_normalized_obstacle_correction_cuda(
    cleaned_a, a_correction, *args,
):
    a = cleaned_a if (a_correction is None) else (cleaned_a - a_correction)
    return L2_obstacle(a, *args)


#################################


@njit
def set_diff(data, position, row, i, j):
    vector = data[j] - data[i]
    row[position : position + 2] = vector
    row[position + 2] = np.linalg.norm(vector)


@njit  # (parallel=True)
def get_edges_data(
    edges,
    initial_nodes,
    u_old,
    v_old,
    forces,
    boundary_faces_count,
    obstacle_normal,
    boundary_obstacle_penetration,
):  # , forces
    edges_number = edges.shape[0]
    edges_data = np.zeros((edges_number, 12))
    for e in range(edges_number):
        i = edges[e, 0]
        j = edges[e, 1]

        set_diff(initial_nodes, 0, edges_data[e], i, j)
        set_diff(u_old, 3, edges_data[e], i, j)
        set_diff(v_old, 6, edges_data[e], i, j)
        set_diff(forces, 9, edges_data[e], i, j)
    return edges_data


###################################3


def L2_obstacle_nvt(
    boundary_a,
    C_boundary,
    E_boundary,
    boundary_v_old,
    boundary_nodes,
    boundary_normals,
    boundary_obstacle_nodes,
    boundary_obstacle_normals,
    boundary_nodes_volume,
):  # np via torch
    value_torch = L2_normalized_obstacle_correction_cuda(
        thh.to_torch_double(boundary_a).to(thh.device),
        None,
        thh.to_torch_double(C_boundary).to(thh.device),
        thh.to_torch_double(E_boundary).to(thh.device),
        thh.to_torch_double(boundary_v_old).to(thh.device),
        thh.to_torch_double(boundary_nodes).to(thh.device),
        thh.to_torch_long(boundary_normals).to(thh.device),
        thh.to_torch_double(boundary_obstacle_nodes).to(thh.device),
        thh.to_torch_double(boundary_obstacle_normals).to(thh.device),
        thh.to_torch_double(boundary_nodes_volume).to(thh.device),
    )
    value = thh.to_np_double(value_torch)
    return value  # .item()


class SettingInput(SettingRandomized):
    def __init__(
        self,
        mesh_type,
        mesh_density_x,
        mesh_density_y,
        scale_x,
        scale_y,
        is_adaptive,
        create_in_subprocess,
    ):
        super().__init__(
            mesh_type,
            mesh_density_x,
            mesh_density_y,
            scale_x,
            scale_y,
            is_adaptive,
            create_in_subprocess,
        )

    def get_edges_data_torch(self, edges):
        edges_data = get_edges_data(
            edges,
            self.normalized_initial_nodes,
            self.input_u_old,
            self.input_v_old,
            self.input_forces,
            self.boundary_faces_count,
            self.normalized_obstacle_normal,
            self.boundary_obstacle_penetration,
        )
        return thh.to_torch_double(edges_data)

    def get_nodes_data(self):

        penetration = self.complete_boundary_data_with_zeros(
            self.normalized_boundary_obstacle_penetration_vectors_torch
        )
        normals = self.complete_boundary_data_with_zeros(
            self.normalized_boundary_normals_torch
        )
        volume = self.complete_boundary_data_with_zeros(self.boundary_nodes_volume_torch)

        data = torch.hstack(
            (
                thh.append_euclidean_norm(self.input_forces_torch),
                # thh.append_euclidean_norm(self.input_u_old_torch),
                # thh.append_euclidean_norm(self.input_v_old_torch) #TODO: Add v tangential?
                thh.append_euclidean_norm(penetration),
                thh.append_euclidean_norm(normals),
                volume
            )
        )
        return data

    def get_data(self, setting_index=None, exact_normalized_a_torch=None):
        # edge_index_torch, edge_attr = remove_self_loops(
        #    self.contiguous_edges_torch, self.edges_data_torch
        # )
        # Do not use "face" in name (probably reserved in PyG)
        directional_edges = np.vstack((self.edges, np.flip(self.edges, axis=1)))
        data = Data(
            pos=thh.set_precision(self.normalized_initial_nodes_torch),
            x=thh.set_precision(self.get_nodes_data()),
            edge_index=thh.get_contiguous_torch(directional_edges),
            edge_attr=thh.set_precision(self.get_edges_data_torch(directional_edges)),
            setting_index=setting_index,
            normalized_a_correction=self.normalized_a_correction_torch,
            reshaped_C=self.C_torch.reshape(-1, 1),
            normalized_E=self.normalized_E_torch,
            exact_normalized_a=exact_normalized_a_torch,
            normalized_boundary_v_old=self.normalized_boundary_v_old_torch,
            normalized_boundary_nodes=self.normalized_boundary_nodes_torch,
            normalized_boundary_normals=self.normalized_boundary_normals_torch,
            normalized_boundary_obstacle_nodes=self.normalized_boundary_obstacle_nodes_torch,
            normalized_boundary_obstacle_normals=self.normalized_boundary_obstacle_normals_torch,
            boundary_nodes_volume=self.boundary_nodes_volume_torch,
            boundary_nodes_count=self.boundary_nodes_count_torch,
            # pin_memory=True,
            # num_workers=1
        )
        """
        transform = T.Compose(
            [
                T.TargetIndegree(norm=False),
                T.Cartesian(norm=False),
                T.Polar(norm=False),
            ]  # add custom for multiple 'pos' types
        )  # T.OneHotDegree(),
        transform(data)
        """
        return data

    def normalized_L2_obstacle_nvt(self, normalized_boundary_a_vector):
        return L2_obstacle_nvt(
            nph.unstack(normalized_boundary_a_vector, self.dim),
            self.C_boundary,
            self.normalized_E_boundary,
            self.normalized_boundary_v_old,
            self.normalized_boundary_nodes,
            self.normalized_boundary_normals,
            self.normalized_boundary_obstacle_nodes,
            self.normalized_boundary_obstacle_normals,
            self.boundary_nodes_volume,
        )

