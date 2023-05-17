from conmech.helpers import cmh
from conmech.helpers.config import Config, SimulationConfig
from conmech.scenarios.scenarios import bunny_fall_3d
from conmech.simulations import simulation_runner


def main():
    cmh.print_jax_configuration()

    simulation_config = SimulationConfig(
        use_linear_solver=False,
        use_normalization=False,
        use_green_strain=True,
        use_nonconvex_friction_law=False,
        use_constant_contact_integral=False,  # False,
        use_lhs_preconditioner=False,
        with_self_collisions=True,
        use_pca=False,
        mesh_layer_proportion=4,  # 2 8,
        mode="compare_net",
    )

    all_scenarios = [
        # all_train(config.td, config.sc)[0]
        bunny_fall_3d(
            mesh_density=32,
            scale=1,
            final_time=2.5,
            simulation_config=simulation_config,
            scale_forces=5.0,
        ),
    ]

    simulation_runner.run_examples(
        all_scenarios=all_scenarios,
        file=__file__,
        plot_animation=True,
        config=Config(shell=False),
    )


if __name__ == "__main__":
    main()
