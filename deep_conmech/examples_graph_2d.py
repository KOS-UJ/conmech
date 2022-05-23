from conmech.scenarios import scenarios
from conmech.simulations import simulation_runner
from deep_conmech.graph.model import GraphModelDynamic
from deep_conmech.training_config import TrainingConfig


def main(mesh_density=4, final_time=5, plot_animation=True):
    config = TrainingConfig()
    simulation_runner.run_examples(
        all_scenarios=scenarios.get_train_data(
            mesh_density=mesh_density, scale=1, final_time=final_time
        ),
        file=__file__,
        simulate_dirty_data=True,
        plot_animation=plot_animation,
        config=config,
        get_scene_function=GraphModelDynamic.get_scene_function,
    )


if __name__ == "__main__":
    main()
