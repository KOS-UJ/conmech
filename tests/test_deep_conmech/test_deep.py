from conmech.helpers import cmh
from deep_conmech import run_model
from deep_conmech.training_config import TrainingConfig, TrainingData


# Caution: may take some time to complete
def test_smoke_train_and_plot():
    output_catalog = "output/TEST_TMP"
    databases_main_path = f"{output_catalog}/DATA"
    log_catalog = f"{output_catalog}/LOG"

    td = TrainingData(
        DATASET="synthetic",
        MESH_DENSITY=4,
        batch_size=2,
        synthetic_batches_in_epoch=2,
        FINAL_TIME=0.1,
        save_at_minutes=0,
        validate_at_epochs=1,
    )
    config = TrainingConfig(
        td=td,
        device="cpu",
        max_epoch_number=2,
        datasets_main_path=databases_main_path,
        dataset_images_count=1,
        output_catalog=output_catalog,
        log_catalog=log_catalog,
    )
    cmh.clear_folder(output_catalog)
    run_model.train(config)
    run_model.plot(config)
    cmh.clear_folder(output_catalog)
    assert 1 == 1
