import os
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import Folder
import torch


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    DATA_PATH = os.path.join(BASE_DIR, "anomaly_data", "metal_nut")

    dataset = Folder(
        name = "metal_nut",
        root = DATA_PATH,
        normal_dir = "train/good",
        abnormal_dir = "test",
        train_batch_size = 4,
        eval_batch_size = 4,   
        num_workers=0
        )

    model = Patchcore(
        backbone="resnet18",
        layers=["layer2", "layer3"],
        pre_trained=True,
        coreset_sampling_ratio=1   # PatchCore only stores x% of the learnings. Otherwise RAM filled and crash
    )

    engine = Engine(
        default_root_dir=os.path.join(BASE_DIR, "results"),
        max_epochs = 1,
        accelerator = "auto",
        devices = 1
    )

    #Training
    engine.fit(model=model, datamodule=dataset)

    #Testing
    engine.test(model=model, datamodule=dataset)