import copy
from conmech.dataclass.body_properties import BodyProperties
from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.obstacle_properties import ObstacleProperties
from conmech.dataclass.schedule import Schedule

import deep_conmech.simulator.mesh.remesher as remesher
from conmech.helpers import nph
from deep_conmech.common import config
from deep_conmech.simulator.setting.setting_forces import *
from deep_conmech.simulator.setting.setting_obstacles import SettingObstacles
from deep_conmech.scenarios import Scenario


class SettingIterable(SettingObstacles):
    def __init__(
        self,
        mesh_data: MeshData,
        body_prop: BodyProperties,
        obstacle_prop: ObstacleProperties,
        schedule: Schedule,
        create_in_subprocess,
    ):
        super().__init__(
            mesh_data=mesh_data,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            create_in_subprocess=create_in_subprocess,
        )

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


    @staticmethod
    def get_setting(
        scenario: Scenario, randomize: bool = False, create_in_subprocess: bool = False
    ):
        setting = SettingIterable(
            mesh_data=scenario.mesh_data,
            body_prop=scenario.body_prop,
            obstacle_prop=scenario.obstacle_prop,
            schedule=scenario.schedule,
            create_in_subprocess=create_in_subprocess,
        )
        setting.set_randomization(randomize)
        setting.set_obstacles(scenario.obstacles)
        return setting
