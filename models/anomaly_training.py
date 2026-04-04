import os
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import Folder
import torch

torch.set_float32_matmul_precision("medium")


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(BASE_DIR, "anomaly_data", "multi_source_dataset")

    dataset = Folder(
        name = "multi_source_dataset",
        root = DATA_PATH,
        normal_dir = "train/good",
        abnormal_dir = "test",
        train_batch_size = 4,
        eval_batch_size = 4,   
        num_workers=4
        )

    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=1,
        num_neighbors=9   # PatchCore only stores x% of the learnings. Otherwise RAM filled and crash
    )

    engine = Engine(
        default_root_dir=os.path.join(BASE_DIR, "results"),
        max_epochs = 3,
        accelerator = "auto",
        devices = 1
    )

    #Training
    engine.fit(model=model, datamodule=dataset)

    #Testing
    engine.test(model=model, datamodule=dataset)