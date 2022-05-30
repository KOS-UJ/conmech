from typing import Callable, Union

import numpy as np

from conmech.helpers import cmh
from conmech.helpers.config import Config
from conmech.properties.body_properties import (
    DynamicBodyProperties,
    DynamicTemperatureBodyProperties,
)
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import (
    ObstacleProperties,
    TemperatureObstacleProperties,
)
from conmech.properties.schedule import Schedule
from conmech.scene.scene import Scene
from conmech.scene.scene_temperature import SceneTemperature
from conmech.solvers.calculator import Calculator
from conmech.state.obstacle import Obstacle


class Scenario:
    def __init__(
        self,
        name: str,
        mesh_prop: MeshProperties,
        body_prop: DynamicBodyProperties,
        schedule: Schedule,
        forces_function: Union[Callable[..., np.ndarray], np.ndarray],
        obstacle: Obstacle,
    ):
        self.name = name
        self.mesh_prop = mesh_prop
        self.body_prop = body_prop
        self.obstacle_prop = obstacle.properties
        self.schedule = schedule
        self.linear_obstacles = (
            None if obstacle.geometry is None else obstacle.geometry * mesh_prop.scale_x
        )
        self.mesh_obstacles = None if obstacle.all_mesh is None else obstacle.all_mesh
        self.forces_function = forces_function

    @staticmethod
    def get_by_function(function, setting, current_time):
        if isinstance(function, np.ndarray):
            return np.tile(function, (setting.nodes_count, 1))
        return np.array(
            [
                function(*nodes_pairs, setting.mesh_prop, current_time)
                for nodes_pairs in zip(setting.initial_nodes, setting.moved_nodes)
            ]
        )

    def get_forces_by_function(self, setting, current_time):
        return Scenario.get_by_function(self.forces_function, setting, current_time)

    def get_tqdm(self, desc: str, config: Config):
        return cmh.get_tqdm(
            iterable=range(self.schedule.episode_steps),
            config=config,
            desc=f"{desc} {self.name}",
        )

    @staticmethod
    def get_solve_function():
        return Calculator.solve

    def get_scene(
        self,
        normalize_by_rotation=True,
        randomize=False,
        create_in_subprocess: bool = False,
    ) -> Scene:
        _ = randomize
        setting = Scene(
            mesh_prop=self.mesh_prop,
            body_prop=self.body_prop,
            obstacle_prop=self.obstacle_prop,
            schedule=self.schedule,
            normalize_by_rotation=normalize_by_rotation,
            create_in_subprocess=create_in_subprocess,
        )
        setting.normalize_and_set_obstacles(self.linear_obstacles, self.mesh_obstacles)
        return setting

    @property
    def dimension(self):
        return self.mesh_prop.dimension

    @property
    def time_step(self):
        return self.schedule.time_step

    @property
    def final_time(self):
        return self.schedule.final_time


class TemperatureScenario(Scenario):
    def __init__(
        self,
        name: str,
        mesh_prop: MeshProperties,
        body_prop: DynamicTemperatureBodyProperties,
        schedule: Schedule,
        forces_function: Union[Callable, np.ndarray],
        obstacle: Obstacle,
        heat_function: Union[Callable, np.ndarray],
    ):
        super().__init__(
            name=name,
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            schedule=schedule,
            forces_function=forces_function,
            obstacle=obstacle,
        )
        self.heat_function = heat_function

    def get_heat_by_function(self, setting, current_time):
        return Scenario.get_by_function(self.heat_function, setting, current_time)

    @staticmethod
    def get_solve_function():
        return Calculator.solve_with_temperature

    def get_scene(
        self,
        normalize_by_rotation=True,
        randomize=False,
        create_in_subprocess: bool = False,
    ) -> SceneTemperature:
        _ = randomize
        setting = SceneTemperature(
            mesh_prop=self.mesh_prop,
            body_prop=self.body_prop,
            obstacle_prop=self.obstacle_prop,
            schedule=self.schedule,
            normalize_by_rotation=normalize_by_rotation,
            create_in_subprocess=create_in_subprocess,
        )
        setting.normalize_and_set_obstacles(self.linear_obstacles, self.mesh_obstacles)
        return setting


default_schedule = Schedule(time_step=0.01, final_time=4.0)

default_body_prop = DynamicBodyProperties(
    mu=4.0, lambda_=4.0, theta=4.0, zeta=4.0, mass_density=1.0
)
# body_prop = DynamicBodyProperties(mu=0.01, lambda_=0.01, theta=0.01, zeta=0.01, mass_density=0.01)

default_thermal_expansion_coefficients = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
default_thermal_conductivity_coefficients = np.array(
    [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
)
default_temp_body_prop = DynamicTemperatureBodyProperties(
    mass_density=1.0,
    mu=4.0,
    lambda_=4.0,
    theta=4.0,
    zeta=4.0,
    thermal_expansion=default_thermal_expansion_coefficients,
    thermal_conductivity=default_thermal_conductivity_coefficients,
)

obstacle_mesh_prop = [
    MeshProperties(
        dimension=2,
        mesh_type="pygmsh_circle",
        scale=[1],
        mesh_density=[4],
        initial_position=np.array([1.5, 0.0]),
    ),
    MeshProperties(
        dimension=2,
        mesh_type="pygmsh_rectangle",
        scale=[1],
        mesh_density=[4],
        initial_position=np.array([-1.5, 0.0]),
    ),
]


def get_temp_body_prop(thermal_expansion_coeff, thermal_conductivity_coeff):
    return DynamicTemperatureBodyProperties(
        mass_density=1.0,
        mu=4.0,
        lambda_=4.0,
        theta=4.0,
        zeta=4.0,
        thermal_expansion=thermal_expansion_coeff,
        thermal_conductivity=thermal_conductivity_coeff,
    )


default_obstacle_prop = ObstacleProperties(hardness=100.0, friction=5.0)
default_temp_obstacle_prop = TemperatureObstacleProperties(hardness=100.0, friction=5.0, heat=0.01)

M_RECTANGLE = "pygmsh_rectangle"
M_SPLINE = "pygmsh_spline"
M_CIRCLE = "pygmsh_circle"
M_POLYGON = "pygmsh_polygon"

M_CUBE_3D = "meshzoo_cube_3d"
M_BALL_3D = "meshzoo_ball_3d"
M_POLYGON_3D = "pygmsh_polygon_3d"
M_TWIST_3D = "pygmsh_twist_3d"


def f_fall(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_prop, time
    force = np.array([2.0, -1.0])
    return force


def f_slide(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_prop
    force = np.array([0.0, 0.0])
    if time <= 0.5:
        force = np.array([4.0, 0.0])
    return force


def f_accelerate_fast(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_prop, time
    force = np.array([2.0, 0.0])
    return force


def f_accelerate_slow_right(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_prop, time
    force = np.array([0.5, 0.0])
    return force


def f_accelerate_slow_left(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_prop, time
    force = np.array([-0.5, 0.0])
    return force


def f_accelerate_slow_up(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_prop, time
    force = np.array([0.0, 0.5])
    return force


def f_accelerate_slow_down(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_prop, time
    force = np.array([0.0, -0.5])
    return force


def f_accelerate_slow_up_left(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_prop, time
    force = np.array([-0.5, 0.5])
    return force


def f_stay(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_prop, time
    return np.array([0.0, 0.0])


def f_rotate(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    time: float,
):
    _ = moved_node
    if time <= 0.5:
        y_scaled = initial_node[1] / mesh_prop.scale_y
        return y_scaled * np.array([1.5, 0.0])
    return np.array([0.0, 0.0])


def f_rotate_fast(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    time: float,
):
    _ = moved_node
    if time <= 0.5:
        y_scaled = initial_node[1] / mesh_prop.scale_y
        return y_scaled * np.array([3.0, 0.0])
    return np.array([0.0, 0.0])


def f_push_3d(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    time: float,
):
    _ = initial_node, moved_node, mesh_prop, time
    return np.array([1.0, 1.0, 1.0])


def f_rotate_3d(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    time: float,
):
    _ = moved_node, mesh_prop
    if time <= 0.5:
        scale = initial_node[1] * initial_node[2]
        return scale * np.array([4.0, 0.0, 0.0])
    return np.array([0.0, 0.0, 0.0])


def polygon_mesh_obstacles(mesh_density, scale, final_time, tag=""):
    obstacle = Obstacle(
        geometry=None, properties=default_obstacle_prop, all_mesh=obstacle_mesh_prop
    )
    return Scenario(
        name=f"polygon_mesh_obstacles{tag}",
        mesh_prop=MeshProperties(
            dimension=2, mesh_type=M_POLYGON, scale=[scale], mesh_density=[mesh_density]
        ),
        body_prop=default_body_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_slide,
        obstacle=obstacle,
    )


def circle_slope(mesh_density, scale, final_time, tag=""):
    obstacle = Obstacle.get_linear_obstacle("slope", default_obstacle_prop)
    return Scenario(
        name=f"circle_slope{tag}",
        mesh_prop=MeshProperties(
            dimension=2, mesh_type=M_CIRCLE, scale=[scale], mesh_density=[mesh_density]
        ),
        body_prop=default_body_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_slide,
        obstacle=obstacle,
    )


def spline_down(mesh_density, scale, final_time, tag=""):
    obstacle = Obstacle.get_linear_obstacle("bottom", default_obstacle_prop)
    return Scenario(
        name=f"spline_down{tag}",
        mesh_prop=MeshProperties(
            dimension=2, mesh_type=M_SPLINE, scale=[scale], mesh_density=[mesh_density]
        ),
        body_prop=default_body_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_accelerate_slow_down,
        obstacle=obstacle,
    )


def circle_up_left(mesh_density, scale, final_time, tag=""):
    obstacle = Obstacle.get_linear_obstacle("back", default_obstacle_prop)
    return Scenario(
        name=f"circle_up_left{tag}",
        mesh_prop=MeshProperties(
            dimension=2, mesh_type=M_CIRCLE, scale=[scale], mesh_density=[mesh_density]
        ),
        body_prop=default_body_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_accelerate_slow_up_left,
        obstacle=obstacle,
    )


def polygon_left(mesh_density, scale, final_time, tag=""):
    obstacle = Obstacle.get_linear_obstacle("back", default_obstacle_prop)
    obstacle.geometry *= scale
    return Scenario(
        name=f"polygon_left{tag}",
        mesh_prop=MeshProperties(
            dimension=2, mesh_type=M_POLYGON, scale=[scale], mesh_density=[mesh_density]
        ),
        body_prop=default_body_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_accelerate_slow_left,
        obstacle=obstacle,
    )


def polygon_slope(mesh_density, scale, final_time, tag=""):
    obstacle = Obstacle.get_linear_obstacle("slope", default_obstacle_prop)
    return Scenario(
        name=f"polygon_slope{tag}",
        mesh_prop=MeshProperties(
            dimension=2, mesh_type=M_POLYGON, scale=[scale], mesh_density=[mesh_density]
        ),
        body_prop=default_body_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_slide,
        obstacle=obstacle,
    )


def circle_rotate(mesh_density, scale, final_time, tag=""):
    obstacle = Obstacle.get_linear_obstacle("side", default_obstacle_prop)
    return Scenario(
        name=f"circle_rotate{tag}",
        mesh_prop=MeshProperties(
            dimension=2, mesh_type=M_CIRCLE, scale=[scale], mesh_density=[mesh_density]
        ),
        body_prop=default_body_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_rotate,
        obstacle=obstacle,
    )


def polygon_rotate(mesh_density, scale, final_time, tag=""):
    obstacle = Obstacle.get_linear_obstacle("side", default_obstacle_prop)
    return Scenario(
        name=f"polygon_rotate{tag}",
        mesh_prop=MeshProperties(
            dimension=2, mesh_type=M_POLYGON, scale=[scale], mesh_density=[mesh_density]
        ),
        body_prop=default_body_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_rotate,
        obstacle=obstacle,
    )


def polygon_stay(mesh_density, scale, final_time, tag=""):
    obstacle = Obstacle.get_linear_obstacle("side", default_obstacle_prop)
    return Scenario(
        name=f"polygon_stay{tag}",
        mesh_prop=MeshProperties(
            dimension=2,
            mesh_type=M_POLYGON,
            scale=[scale],
            mesh_density=[mesh_density],
        ),
        body_prop=default_body_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_stay,
        obstacle=obstacle,
    )


def polygon_two(mesh_density, scale, final_time, tag=""):
    obstacle = Obstacle.get_linear_obstacle("two", default_obstacle_prop)
    return Scenario(
        name=f"polygon_two{tag}",
        mesh_prop=MeshProperties(
            dimension=2,
            mesh_type=M_POLYGON,
            scale=[scale],
            mesh_density=[mesh_density],
        ),
        body_prop=default_body_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=f_slide,
        obstacle=obstacle,
    )


def scenario_3d(mesh_density: int, scale: int, final_time: float, tag=""):
    return Scenario(
        name="ball_roll",
        mesh_prop=MeshProperties(
            dimension=3, mesh_type=M_BALL_3D, scale=[scale], mesh_density=[mesh_density]
        ),
        body_prop=default_body_prop,
        schedule=Schedule(final_time=final_time),
        forces_function=np.array([0.0, 0.0, -0.5]),
        obstacle=Obstacle(
            np.array([[[0.3, 0.2, 1.0]], [[0.0, 0.0, -0.01]]]), default_obstacle_prop
        ),
    )


def get_train_data(**args):
    tag = "_train"
    return [
        polygon_two(**args, tag=tag),
        spline_down(**args, tag=tag),
        circle_up_left(**args, tag=tag),
        circle_rotate(**args, tag=tag),
        polygon_stay(**args, tag=tag),
    ]


def get_valid_data(**args):
    tag = "_val"
    return [
        polygon_left(**args, tag=tag),
        polygon_rotate(**args, tag=tag),
        circle_slope(**args, tag=tag),
    ]


def get_args(td):
    return dict(
        mesh_density=td.mesh_density,
        scale=td.train_scale,
        final_time=td.final_time,
    )


def all_train(td):
    args = get_args(td)
    if td.dimension == 3:
        return [scenario_3d(**args)]
    return get_train_data(**args)


def all_validation(td):
    args = get_args(td)
    if td.dimension == 3:
        return [scenario_3d(**args)]
    return get_valid_data(**args)


def all_train_2(td):
    return [*all_train(td), *all_validation(td)]


def all_print(td):
    args = get_args(td)
    if td.dimension == 3:
        return [scenario_3d(**args)]
    return [
        *get_valid_data(**args),
        *get_train_data(**args),
    ]
