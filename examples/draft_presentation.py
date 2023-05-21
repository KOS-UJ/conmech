import copy

import numpy as np

from conmech.helpers.config import Config, SimulationConfig
from conmech.properties.body_properties import TimeDependentBodyProperties
from conmech.properties.mesh_properties import MeshProperties
from conmech.properties.obstacle_properties import ObstacleProperties
from conmech.properties.schedule import Schedule
from conmech.scenarios.scenarios import M_BUNNY_3D, Scenario, f_rotate_3d
from conmech.simulations import simulation_runner
from conmech.state.obstacle import Obstacle

simulation_config = SimulationConfig(
    use_normalization=False,
    use_linear_solver=False,
    use_green_strain=True,
    use_nonconvex_friction_law=False,
    use_constant_contact_integral=False,
    use_lhs_preconditioner=False,
    with_self_collisions=False,
    use_pca=False,
)
name = "linear_16"
simulation_config.use_green_strain = False
simulation_config.use_linear_solver = True
simulation_config.use_constant_contact_integral = True

animation_backend = "three"  # None


def main(mesh_density=16, final_time=3, plot_animation=True):  # 100
    schedule = Schedule(final_time=final_time, time_step=0.01)
    body_prop = TimeDependentBodyProperties(
        mu=8,
        lambda_=8,
        theta=8,
        zeta=8,
        mass_density=1.0,
    )
    obstacle = Obstacle(
        np.array([[[0.0, 0.7, 1.0]], [[0.0, 0.0, -1.0]]]),
        ObstacleProperties(hardness=100.0, friction=5.0),
        all_mesh=None,
    )
    mesh_prop = MeshProperties(
        dimension=3,
        mesh_type=M_BUNNY_3D,
        scale=[1],
        mesh_density=[mesh_density],
    )

    advanced_scenarios = [
        Scenario(
            name=name,
            mesh_prop=mesh_prop,
            body_prop=body_prop,
            schedule=schedule,
            forces_function=np.array([0.0, 0.0, -1.0]),  # f_rotate_3d
            obstacle=obstacle,
            simulation_config=simulation_config,
        ),
    ]

    return simulation_runner.run_examples(
        all_scenarios=advanced_scenarios,
        file=__file__,
        plot_animation=plot_animation,
        config=Config(shell=False, animation_backend=animation_backend),
    )


if __name__ == "__main__":
    main()
