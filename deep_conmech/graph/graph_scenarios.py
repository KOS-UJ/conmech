from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.schedule import Schedule
from deep_conmech.graph.setting.setting_input import SettingInput
from deep_conmech.scenarios import Scenario
from deep_conmech import scenarios
from deep_conmech.common import config

class GraphScenario(Scenario):
    
    def get_setting(
        self, randomize=False, create_in_subprocess: bool = False
    ) -> SettingInput:  # "SettingIterable":
        setting = SettingInput(
            mesh_data=self.mesh_data,
            body_prop=self.body_prop,
            obstacle_prop=self.obstacle_prop,
            schedule=self.schedule,
            create_in_subprocess=create_in_subprocess,
        )
        setting.set_randomization(randomize)
        setting.set_obstacles(self.obstacles)
        return setting


def all_print():
    return [
        GraphScenario(
            "polygon_rotate",
            MeshData(
                dimension=2,
                mesh_type=scenarios.m_polygon,
                scale=[config.PRINT_SCALE],
                mesh_density=[config.MESH_DENSITY],
                is_adaptive=False,
            ),
            scenarios.default_body_prop,
            scenarios.default_obstacle_prop,
            schedule=Schedule(final_time=config.FINAL_TIME),
            forces_function=scenarios.f_rotate,
            obstacles=scenarios.o_side,
        )
    ]
