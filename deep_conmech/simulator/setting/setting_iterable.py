import copy
import pickle

import deep_conmech.simulator.mesh.remesher as remesher
from conmech.dataclass.body_properties import DynamicBodyProperties
from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.obstacle_properties import ObstacleProperties
from conmech.dataclass.schedule import Schedule
from deep_conmech.simulator.setting.setting_obstacles import SettingObstacles


class SettingIterable(SettingObstacles):
    def __init__(
            self,
            mesh_data: MeshData,
            body_prop: DynamicBodyProperties,
            obstacle_prop: ObstacleProperties,
            schedule: Schedule,
            normalize_by_rotation: bool,
            create_in_subprocess,
    ):
        super().__init__(
            mesh_data=mesh_data,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            normalize_by_rotation=normalize_by_rotation,
            create_in_subprocess=create_in_subprocess,
        )

    @property
    def input_v_old(self):
        return self.normalized_v_old

    @property
    def input_u_old(self):
        return self.normalized_u_old

    @property
    def input_forces(self):
        return self.normalized_forces

    def get_copy(self):
        setting = copy.deepcopy(self)
        return setting

    def iterate_self(self, a, randomized_inputs=False):
        v = self.v_old + self.time_step * a
        u = self.u_old + self.time_step * v

        self.set_u_old(u)
        self.set_v_old(v)
        self.set_a_old(a)

        self.clear()
        return self

    def remesh_self(self):
        old_initial_nodes = self.initial_nodes.copy()
        old_elements = self.elements.copy()
        u_old = self.u_old.copy()
        v_old = self.v_old.copy()
        a_old = self.a_old.copy()

        self.remesh()

        u = remesher.approximate_all_numba(
            self.initial_nodes, old_initial_nodes, u_old, old_elements
        )
        v = remesher.approximate_all_numba(
            self.initial_nodes, old_initial_nodes, v_old, old_elements
        )
        a = remesher.approximate_all_numba(
            self.initial_nodes, old_initial_nodes, a_old, old_elements
        )

        self.set_u_old(u)
        self.set_v_old(v)
        self.set_a_old(a)

    def save_pickle(self, path: str) -> None:
        with open(f"{path}.st", "wb") as file:
            setting_copy = copy.deepcopy(self)
            setting_copy.clear_save()
            pickle.dump(setting_copy, file)

    @staticmethod
    def load_pickle(path: str):
        with open(f"{path}.st", "rb") as file:
            setting = pickle.load(file)
            return setting

    def clear_save(self):
        self.is_contact = None
        self.is_dirichlet = None

        self.element_initial_volume = None
        # self.const_volume = None
        # self.const_elasticity = None
        # self.const_viscosity = None
        self.ACC = None
        self.K = None
        self.C2T = None
        # self.visco_plus_elast_times_ts = None

        self.C_boundary = None
        self.free_x_contact = None
        self.contact_x_free = None
        self.free_x_free_inverted = None

        self.T_boundary = None
        self.T_free_x_contact = None
        self.T_contact_x_free = None
        self.T_free_x_free_inverted = None
