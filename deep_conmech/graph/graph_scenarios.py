from conmech.dataclass.mesh_data import MeshData
from conmech.dataclass.schedule import Schedule
from deep_conmech.common import training_config
from deep_conmech.graph.setting.setting_input import SettingInput
from deep_conmech.scenarios import *


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


###################################################


def circle_slope(mesh_density, scale, is_adaptive, final_time):
    return GraphScenario(
        id="circle_slope",
        mesh_data=MeshData(
            dimension=2,
            mesh_type=m_circle,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        body_prop=default_body_prop,
        obstacle_prop=default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_slide,
        obstacles=o_slope,
    )


def spline_right(mesh_density, scale, is_adaptive, final_time):
    return GraphScenario(
        id="spline_right",
        mesh_data=MeshData(
            dimension=2,
            mesh_type=m_spline,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        body_prop=default_body_prop,
        obstacle_prop=default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_accelerate_slow_right,
        obstacles=o_front,
    )


def circle_left(mesh_density, scale, is_adaptive, final_time):
    return GraphScenario(
        "circle_left",
        MeshData(
            dimension=2,
            mesh_type=m_circle,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_accelerate_slow_left,
        obstacles=o_back,
    )


def polygon_left(mesh_density, scale, is_adaptive, final_time):
    return GraphScenario(
        "polygon_left",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_accelerate_slow_left,
        obstacles=o_back * scale,
    )


def polygon_slope(mesh_density, scale, is_adaptive, final_time):
    return GraphScenario(
        "polygon_slope",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_slide,
        obstacles=o_slope,
    )


def circle_rotate(mesh_density, scale, is_adaptive, final_time):
    return GraphScenario(
        "circle_rotate",
        MeshData(
            dimension=2,
            mesh_type=m_circle,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_rotate,
        obstacles=o_side,
    )


def polygon_rotate(mesh_density, scale, is_adaptive, final_time):
    return GraphScenario(
        "polygon_rotate",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_rotate,
        obstacles=o_side,
    )


def polygon_stay(mesh_density, scale, is_adaptive, final_time):
    return GraphScenario(
        "polygon_stay",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_stay,
        obstacles=o_side,
    )


def polygon_two(mesh_density, scale, is_adaptive, final_time):
    return GraphScenario(
        "polygon_two",
        MeshData(
            dimension=2,
            mesh_type=m_polygon,
            scale=[scale],
            mesh_density=[mesh_density],
            is_adaptive=is_adaptive,
        ),
        default_body_prop,
        default_obstacle_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_slide,
        obstacles=o_two,
    )


#########################


def get_data(**args):
    return [
        polygon_rotate(**args),
        polygon_two(**args),
        circle_slope(**args),
        spline_right(**args),
        circle_left(**args),
        polygon_left(**args),
        circle_rotate(**args),
        polygon_rotate(**args),
        polygon_rotate(**args),
        polygon_stay(**args),
    ]


all_train = get_data(
    mesh_density=training_config.MESH_DENSITY,
    scale=training_config.TRAIN_SCALE,
    is_adaptive=training_config.ADAPTIVE_TRAINING_MESH,
    final_time=training_config.FINAL_TIME,
)

all_validation = get_data(
    mesh_density=training_config.MESH_DENSITY,
    scale=training_config.VALIDATION_SCALE,
    is_adaptive=False,
    final_time=training_config.FINAL_TIME,
)

print_args = dict(
    mesh_density=training_config.MESH_DENSITY,
    scale=training_config.PRINT_SCALE,
    is_adaptive=False,
    final_time=training_config.FINAL_TIME,
)


def all_print(
    mesh_density=training_config.MESH_DENSITY, final_time=training_config.FINAL_TIME
):  # config.FINAL_TIME):
    return [
        *get_data(
            mesh_density=mesh_density,
            scale=training_config.PRINT_SCALE,
            is_adaptive=False,
            final_time=final_time,
        ),
        # *get_data(scale=config.VALIDATION_SCALE, is_adaptive=False, final_time=config.FINAL_TIME),
        # polygon_two(**print_args),
    ]
