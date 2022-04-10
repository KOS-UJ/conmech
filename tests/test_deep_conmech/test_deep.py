from conmech.helpers import cmh
from deep_conmech import run_model
from deep_conmech.graph.net import CustomGraphNet
from deep_conmech.helpers import thh
from deep_conmech.training_config import TrainingConfig, TrainingData


# Caution: may take some time to complete
def test_smoke_train_and_plot():
    output_catalog = "output/TEST_TMP"
    databases_main_path = f"{output_catalog}/DATA"
    cmh.clear_folder(databases_main_path)
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
    )
    run_model.train(config)
    assert 1 == 1
    run_model.plot(config)
    assert 2 == 2
    cmh.clear_folder(output_catalog)
