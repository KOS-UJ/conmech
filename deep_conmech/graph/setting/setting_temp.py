import numpy as np
from conmech.dataclass.body_properties import BodyProperties
from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.obstacle_properties import ObstacleProperties
from conmech.dataclass.schedule import Schedule
from conmech.helpers import nph
from deep_conmech.common import config
from deep_conmech.simulator.setting.setting_forces import *
from numba import njit
from deep_conmech.graph.setting.setting_randomized import SettingRandomized


def L2_temp(
    t,
    C,
    E,
):
    value = L2_new(a, C, E)
    return value



class SettingTemp(SettingRandomized):
    def __init__(
        self, mesh_data, body_prop, obstacle_prop, schedule, create_in_subprocess,
    ):
        super().__init__(
            mesh_data=mesh_data,
            body_prop=body_prop,
            obstacle_prop=obstacle_prop,
            schedule=schedule,
            create_in_subprocess=create_in_subprocess,
        )
