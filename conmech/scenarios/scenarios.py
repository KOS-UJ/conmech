from typing import Callable, Optional, Union

import numpy as np

from conmech.helpers import cmh
from conmech.helpers.config import Config, SimulationConfig
from conmech.properties.body_properties import (
    TimeDependentBodyProperties,
    TimeDependentTemperatureBodyProperties,
)
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import (
    ObstacleProperties,
    TemperatureObstacleProperties,
)
from conmech.properties.schedule import Schedule
from conmech.state.obstacle import Obstacle


class Scenario:
    def __init__(
        self,
        name: str,
        mesh_prop: MeshProperties,
        body_prop: TimeDependentBodyProperties,
        schedule: Schedule,
        forces_function: Union[Callable[..., np.ndarray], np.ndarray],
        obstacle: Obstacle,
        simulation_config: SimulationConfig,
        forces_function_parameter: Optional[float] = None,
    ):  # pylint: disable=too-many-arguments
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
        self.forces_function_parameter = forces_function_parameter
        self.simulation_config = simulation_config

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

    def get_forces_by_function(self, scene, current_time):
        if self.forces_function_parameter is not None:

            def function(*args):
                return self.forces_function(*args, self.forces_function_parameter)

        else:
            function = self.forces_function
        return Scenario.get_by_function(function, scene, current_time)

    def get_tqdm(self, desc: str, config: Config):
        return cmh.get_tqdm(
            iterable=range(self.schedule.episode_steps),
            config=config,
            desc=f"{desc} {self.name}",
        )

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
        body_prop: TimeDependentTemperatureBodyProperties,
        schedule: Schedule,
        forces_function: Union[Callable, np.ndarray],
        obstacle: Obstacle,
        heat_function: Union[Callable, np.ndarray],
        simulation_config: SimulationConfig,
    ):  # pylint: disable=too-many-arguments
        simulation_config.mode = "temperature"
        super().__init__(
            name=name,
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            schedule=schedule,
            forces_function=forces_function,
            obstacle=obstacle,
            simulation_config=simulation_config,
        )
        self.heat_function = heat_function

    def get_heat_by_function(self, setting, current_time):
        return Scenario.get_by_function(self.heat_function, setting, current_time)


default_schedule = Schedule(time_step=0.01, final_time=4.0)

SCALE_MASS = 1.0
SCALE_COEFF = 1.0

default_body_prop = TimeDependentBodyProperties(
    mu=4.0 * SCALE_COEFF,
    lambda_=4.0 * SCALE_COEFF,
    theta=4.0 * SCALE_COEFF,
    zeta=4.0 * SCALE_COEFF,
    mass_density=SCALE_MASS,
)
default_body_prop_3d = TimeDependentBodyProperties(
    mu=12.0,  # 8,
    lambda_=12.0,  # 8,
    theta=4.0,
    zeta=4.0,
    mass_density=1.0,
)


default_thermal_expansion_coefficients = np.array(
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
default_thermal_conductivity_coefficients = np.array(
    [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
)
default_temp_body_prop = TimeDependentTemperatureBodyProperties(
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
    return TimeDependentTemperatureBodyProperties(
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
M_BUNNY_3D = "pygmsh_bunny_3d"
M_ARMADILLO_3D = "pygmsh_armadillo_3d"


def f_fall(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    apply_time: float,
    scale_forces=1.0,
):
    _ = initial_node, moved_node, mesh_prop, apply_time
    force = np.array([2.0, -1.0])
    return force * scale_forces


def f_slide(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    apply_time: float,
    scale_forces=1.0,
):
    _ = initial_node, moved_node, mesh_prop
    force = np.array([0.0, 0.0])
    if apply_time <= 0.5:
        force = np.array([4.0, 0.0])
    return force * scale_forces


def f_accelerate_fast(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    apply_time: float,
    scale_forces=1.0,
):
    _ = initial_node, moved_node, mesh_prop, apply_time
    force = np.array([2.0, 0.0])
    return force * scale_forces


def f_accelerate_slow_right(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    apply_time: float,
    scale_forces=1.0,
):
    _ = initial_node, moved_node, mesh_prop, apply_time
    force = np.array([0.5, 0.0])
    return force * scale_forces


def f_accelerate_slow_left(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    apply_time: float,
    scale_forces=1.0,
):
    _ = initial_node, moved_node, mesh_prop, apply_time
    force = np.array([-0.5, 0.0])
    return force * scale_forces


def f_accelerate_slow_up(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    apply_time: float,
    scale_forces=1.0,
):
    _ = initial_node, moved_node, mesh_prop, apply_time
    force = np.array([0.0, 0.5])
    return force * scale_forces


def f_accelerate_slow_down(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    apply_time: float,
    scale_forces=1.0,
):
    _ = initial_node, moved_node, mesh_prop, apply_time
    force = np.array([0.0, -0.5])
    return force * scale_forces


def f_accelerate_slow_up_left(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    apply_time: float,
    scale_forces=1.0,
):
    _ = initial_node, moved_node, mesh_prop, apply_time
    force = np.array([-0.5, 0.5])
    return force * scale_forces


def f_stay(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    apply_time: float,
    scale_forces=1.0,
):
    _ = initial_node, moved_node, mesh_prop, apply_time
    force = np.array([0.0, 0.0])
    return force * scale_forces


def f_rotate(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    apply_time: float,
    scale_forces=1.0,
):
    _ = moved_node
    if apply_time <= 0.5:
        y_scaled = initial_node[1] / mesh_prop.scale_y
        return y_scaled * np.array([1.5, 0.0]) * scale_forces
    return np.array([0.0, 0.0]) * scale_forces


def f_rotate_fast(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    time: float,
    scale_forces=1.0,
):
    _ = moved_node
    if time <= 0.5:
        y_scaled = initial_node[1] / mesh_prop.scale_y
        return y_scaled * np.array([3.0, 0.0]) * scale_forces
    return np.array([0.0, 0.0]) * scale_forces


def f_swing_3d(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    apply_time: float,
    scale_forces=1.0,
):
    _ = initial_node, moved_node, mesh_prop
    force = np.array([1.0, 1.0, 1.0]) * scale_forces
    if apply_time <= 1.5:
        return force
    return -force


def f_rotate_3d(
    initial_node: np.ndarray,
    moved_node: np.ndarray,
    mesh_prop: MeshProperties,
    apply_time: float,
    arg: float = 1.0,
    scale_forces=1.0,
):
    _ = moved_node, mesh_prop
    if apply_time <= np.abs(arg):  # 1.0 0.5: # if (time % 4.0) <= 2.0:
        scale = initial_node[1] * initial_node[2]
        return scale * np.array([4.0, 0.0, 0.0]) * scale_forces * np.sign(arg)
    return np.array([0.0, 0.0, 0.0]) * scale_forces


def polygon_mesh_obstacles(
    mesh_density, scale, final_time, simulation_config: SimulationConfig, tag=""
):
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
        simulation_config=simulation_config,
    )


def circle_slope(mesh_density, scale, final_time, simulation_config: SimulationConfig, tag=""):
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
        simulation_config=simulation_config,
    )


def spline_down(mesh_density, scale, final_time, simulation_config: SimulationConfig, tag=""):
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
        simulation_config=simulation_config,
    )


def circle_up_left(mesh_density, scale, final_time, simulation_config: SimulationConfig, tag=""):
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
        simulation_config=simulation_config,
    )


def polygon_left(mesh_density, scale, final_time, simulation_config: SimulationConfig, tag=""):
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
        simulation_config=simulation_config,
    )


def polygon_slope(mesh_density, scale, final_time, simulation_config: SimulationConfig, tag=""):
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
        simulation_config=simulation_config,
    )


def circle_rotate(mesh_density, scale, final_time, simulation_config: SimulationConfig, tag=""):
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
        simulation_config=simulation_config,
    )


def polygon_rotate(mesh_density, scale, final_time, simulation_config: SimulationConfig, tag=""):
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
        simulation_config=simulation_config,
    )


def polygon_stay(mesh_density, scale, final_time, simulation_config: SimulationConfig, tag=""):
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
        simulation_config=simulation_config,
    )


def polygon_two(mesh_density, scale, final_time, simulation_config: SimulationConfig, tag=""):
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
        simulation_config=simulation_config,
    )


bottom_obstacle_3d = Obstacle(
    np.array([[[0.0, 0.01, 1.0]], [[0.0, 0.01, -2.0]]]), default_obstacle_prop
)


def ball_rotate_3d(
    mesh_density: int,
    scale: int,
    final_time: float,
    simulation_config: SimulationConfig,
    tag="",
    arg=1.0,
):
    _ = tag
    return Scenario(
        name="ball_rotate",
        mesh_prop=MeshProperties(
            dimension=3, mesh_type=M_BALL_3D, scale=[scale], mesh_density=[mesh_density]
        ),
        body_prop=default_body_prop_3d,
        schedule=Schedule(final_time=final_time),
        forces_function=f_rotate_3d,  # np.array([0.0, 0.0, -0.5]),
        obstacle=bottom_obstacle_3d,
        forces_function_parameter=arg,
        simulation_config=simulation_config,
    )


def ball_swing_3d(
    mesh_density: int,
    scale: int,
    final_time: float,
    simulation_config: SimulationConfig,
    tag="",
    arg=1.0,
):
    _ = tag
    _ = arg
    return Scenario(
        name="ball_swing",
        mesh_prop=MeshProperties(
            dimension=3, mesh_type=M_BALL_3D, scale=[scale], mesh_density=[mesh_density]
        ),
        body_prop=default_body_prop_3d,
        schedule=Schedule(final_time=final_time),
        forces_function=f_swing_3d,
        obstacle=bottom_obstacle_3d,
        simulation_config=simulation_config,
    )


def cube_rotate_3d(
    mesh_density: int,
    scale: int,
    final_time: float,
    simulation_config: SimulationConfig,
    tag="",
    arg=1.0,
):
    _ = tag
    return Scenario(
        name="cube_rotate",
        mesh_prop=MeshProperties(
            dimension=3, mesh_type=M_CUBE_3D, scale=[scale], mesh_density=[mesh_density]
        ),
        body_prop=default_body_prop_3d,
        schedule=Schedule(final_time=final_time),
        forces_function=f_rotate_3d,  # np.array([0.0, 0.0, -0.5]),
        obstacle=bottom_obstacle_3d,
        forces_function_parameter=arg,
        simulation_config=simulation_config,
    )


def cube_move_3d(
    mesh_density: int,
    scale: int,
    final_time: float,
    simulation_config: SimulationConfig,
    tag="",
    arg=1.0,
):
    _ = tag
    _ = arg
    return Scenario(
        name="cube_move",
        mesh_prop=MeshProperties(
            dimension=3, mesh_type=M_CUBE_3D, scale=[scale], mesh_density=[mesh_density]
        ),
        body_prop=default_body_prop_3d,
        schedule=Schedule(final_time=final_time),
        forces_function=np.array([0.0, 0.5, 0.0]),
        obstacle=bottom_obstacle_3d,
        simulation_config=simulation_config,
    )


def bunny_fall_3d(
    mesh_density: int,
    scale: int,
    final_time: float,
    simulation_config: SimulationConfig,
    tag="",
    arg=0.7,
    scale_forces=1.0,
):
    _ = tag
    _ = scale
    _ = mesh_density
    return Scenario(
        name="bunny_fall",
        mesh_prop=MeshProperties(
            dimension=3,
            mesh_type=M_BUNNY_3D,
            scale=[1],
            mesh_density=[mesh_density],
        ),
        body_prop=default_body_prop_3d,
        schedule=Schedule(final_time=final_time),
        forces_function=scale_forces * np.array([0.0, 0.0, -1.0]),
        obstacle=Obstacle(  # 0.3
            np.array([[[0.0, arg, 1.0]], [[0.0, 0.0, -0.5]]]),
            ObstacleProperties(hardness=100.0, friction=5.0),
        ),
        simulation_config=simulation_config,
    )


def bunny_rotate_3d(
    mesh_density: int,
    scale: int,
    final_time: float,
    simulation_config: SimulationConfig,
    tag="",
    arg=1.0,
    scale_forces=5.0,
):
    _ = tag
    _ = scale
    return Scenario(
        name="bunny_rotate",
        mesh_prop=MeshProperties(
            dimension=3,
            mesh_type=M_BUNNY_3D,
            scale=[1],
            mesh_density=[mesh_density],
        ),
        body_prop=default_body_prop_3d,
        schedule=Schedule(final_time=final_time),
        forces_function=lambda *args: f_rotate_3d(*args, scale_forces=scale_forces),
        obstacle=bottom_obstacle_3d,
        forces_function_parameter=arg,
        simulation_config=simulation_config,
    )


def bunny_swing_3d(
    mesh_density: int,
    scale: int,
    final_time: float,
    simulation_config: SimulationConfig,
    tag="",
    arg=1.0,
    scale_forces=5.0,
):
    _, _, _ = scale, tag, arg
    return Scenario(
        name="bunny_swing",
        mesh_prop=MeshProperties(
            dimension=3,
            mesh_type=M_BUNNY_3D,
            scale=[1],
            mesh_density=[mesh_density],
        ),
        body_prop=default_body_prop_3d,
        schedule=Schedule(final_time=final_time),
        forces_function=lambda *args: f_swing_3d(*args, scale_forces=scale_forces),
        obstacle=bottom_obstacle_3d,
        simulation_config=simulation_config,
    )


def bunny_obstacles(
    mesh_density: int,
    scale: int,
    final_time: float,
    simulation_config: SimulationConfig,
    tag="",
    arg=1.0,
    scale_forces=1.0,
):
    _, _, _ = scale, tag, arg
    return Scenario(
        name="bunny_obstacles",
        mesh_prop=MeshProperties(
            dimension=3,
            mesh_type=M_BUNNY_3D,
            scale=[1],
            mesh_density=[mesh_density],
        ),
        body_prop=default_body_prop_3d,
        schedule=Schedule(final_time=final_time),
        forces_function=scale_forces * np.array([0.0, 0.0, -1.0]),
        obstacle=Obstacle(
            geometry=None,
            properties=ObstacleProperties(hardness=100.0, friction=0.0),
            all_mesh=[
                MeshProperties(
                    dimension=3,
                    mesh_type="slide_left",
                    scale=[1],
                    mesh_density=[16],
                    initial_position=[0, 0, -0.2],
                ),
                MeshProperties(
                    dimension=3,
                    mesh_type="slide_right",
                    scale=[1],
                    mesh_density=[16],
                    initial_position=[0, -1, -1.0],
                ),
                MeshProperties(
                    dimension=3,
                    mesh_type="slide_left",
                    scale=[1],
                    mesh_density=[16],
                    initial_position=[0, 0, -1.8],
                ),
                MeshProperties(
                    dimension=3,
                    mesh_type="slide_right",
                    scale=[1],
                    mesh_density=[16],
                    initial_position=[0, -1, -2.6],
                ),
            ],
        ),
        simulation_config=simulation_config,
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


def get_args(td, sc):
    return dict(
        mesh_density=td.mesh_density,
        scale=td.train_scale,
        final_time=td.final_time,
        simulation_config=sc,
        scale_forces=1.0,
    )


def all_train(td, sc):
    # args = dict(
    #     mesh_density=td.mesh_density,
    #     scale=td.train_scale,
    #     final_time=td.final_time,
    #     simulation_config=sc,
    # )
    if td.dimension != 3:
        return []  # get_train_data(**args)
    args = []
    for forces_dim in [0, 1, 2]:
        for forces_dir in [1.0, -1.0]:
            for normals_dim_plus in [1]:  # , 2]:
                for normals_dir in [1]:  # , -1]:
                    forces = [0.0, 0.0, 0.0]
                    forces[forces_dim] = forces_dir

                    normals = [0.0, 0.0, 0.0]
                    normals[forces_dim] = -forces_dir
                    normals[(forces_dim + normals_dim_plus) % 3] = normals_dir
                    args.append(
                        {
                            "scale_forces": 2.0,
                            "forces_and_nodes": forces,
                            "obstacle_normals": normals,
                        }
                    )
    data = []
    data.extend(
        [
            Scenario(
                name="bunny_train",
                mesh_prop=MeshProperties(
                    dimension=3,
                    mesh_type=M_BUNNY_3D,
                    scale=[1],
                    mesh_density=[td.mesh_density],
                ),
                body_prop=default_body_prop_3d,
                schedule=Schedule(final_time=td.final_time),
                forces_function=arg["scale_forces"]
                * np.array(arg["forces_and_nodes"]),  # scale_forces * [0.0, 0.0, -1.0]),
                obstacle=Obstacle(
                    np.array([[arg["obstacle_normals"]], [arg["forces_and_nodes"]]]),
                    ObstacleProperties(hardness=100.0, friction=5.0),
                ),
                simulation_config=sc,
            )
            for arg in args
            # bunny_fall_3d(**args, arg=arg, scale_forces=scale_forces)
            # for (arg, scale_forces) in [(-0.7, 1.0), (0.8, 6.0), (1.2, 2.0), (-0.5, 4.0)]
        ]
    )
    # data.extend(
    #     [
    #         bunny_rotate_3d(**args, arg=arg, scale_forces=scale_forces)
    #         for (arg, scale_forces) in [(-2.0, 6.0), (1.0, 1.0)]
    #     ]
    # )
    return data


def all_validation(td, sc):
    args = get_args(td, sc)
    if td.dimension == 3:
        final_time = 2.0
        return [
            [
                bunny_fall_3d(
                    mesh_density=32,
                    scale=1,
                    final_time=final_time,
                    simulation_config=sc,
                    scale_forces=5.0,
                )
            ],
            [
                bunny_rotate_3d(
                    mesh_density=32,
                    scale=1,
                    final_time=final_time,
                    simulation_config=sc,
                    scale_forces=5.0,
                )
            ],
        ]
        args["final_time"] = 2.0  # 8.0
        return [
            # [bunny_fall_3d(**args)],
            # [bunny_rotate_3d(**args)],
            # [bunny_swing_3d(**args)],
            # [ball_rotate_3d(**args)],
            # [ball_swing_3d(**args)],
            # [cube_rotate_3d(**args)],
        ]
    return get_valid_data(**args)


def all_print(td, sc):
    args = get_args(td, sc)
    if td.dimension == 3:
        args["final_time"] = 10.0  # 12.0
        return [
            # bunny_obstacles(**args),
            bunny_fall_3d(**args),
            bunny_rotate_3d(**args),
            bunny_swing_3d(**args),
            # ball_rotate_3d(**args),
            # ball_swing_3d(**args),
            # cube_rotate_3d(**args),
        ]
    return [
        *get_valid_data(**args),
        *get_train_data(**args),
    ]
