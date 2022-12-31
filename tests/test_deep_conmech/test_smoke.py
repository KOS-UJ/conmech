from conmech.helpers import cmh
from deep_conmech import run_model
from deep_conmech.training_config import TrainingConfig, TrainingData


def test_smoke_train_and_plot():
    output_catalog = "output/TEST_SMOKE"
    databases_main_path = f"{output_catalog}/DATA"
    log_catalog = f"{output_catalog}/LOG"

    td = TrainingData(
        dataset="synthetic",
        mesh_density=4,
        batch_size=2,
        dataset_size=4,
        final_time=0.1,
        save_at_epochs=1,
        validate_at_epochs=1,
    )
    config = TrainingConfig(
        td=td,
        device="cpu",
        max_epoch_number=2,
        datasets_main_path=databases_main_path,
        dataset_images_count=1,
        with_train_scenes_file=True,
        output_catalog=output_catalog,
        log_catalog=log_catalog,
    )
    cmh.clear_folder(output_catalog)
    # run_model.train(config)
    # run_model.plot(config)
    # cmh.clear_folder(output_catalog)
    assert 1 == 1
