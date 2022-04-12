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
        BATCH_SIZE=2,
        SYNTHETIC_BATCHES_IN_EPOCH=2,
        FINAL_TIME=0.1,
        SAVE_AT_MINUTES=0,
        VALIDATE_AT_EPOCHS=1,
    )
    config = TrainingConfig(
        td=td,
        DEVICE="cpu",
        MAX_EPOCH_NUMBER=2,
        DATASETS_MAIN_PATH=databases_main_path,
        DATASET_IMAGES_COUNT=1,
        output_catalog=output_catalog,
        LOG_CATALOG=log_catalog,
    )
    cmh.clear_folder(output_catalog)
    run_model.train(config)
    run_model.plot(config)
    cmh.clear_folder(output_catalog)
    assert 1 == 1
