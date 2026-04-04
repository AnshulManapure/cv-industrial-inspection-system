from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.data import Folder
import anomalib
import torch
import os
import pathlib
import matplotlib.pyplot as plt
import math

torch.serialization.add_safe_globals([anomalib.PrecisionType])
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "results", "Patchcore", "multi_source_dataset", "latest", "weights", "lightning", "model.ckpt")
IMAGE_PATH = os.path.join(BASE_DIR, "anomaly_data", "multi_source_dataset", "sample")


def analyze_results(results):

    n = len(results)
    cols = 4  # number of image pairs per row
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols * 2, figsize=(4 * cols * 2, 4 * rows))

    axes = axes.reshape(rows, cols * 2)

    for i, r in enumerate(results):
        row = i // cols
        col = (i % cols) * 2

        image = r.image.squeeze().permute(1, 2, 0).cpu().numpy()
        heatmap = r.anomaly_map.squeeze().cpu().numpy()
        score = r.pred_score.item()

        image = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Original
        axes[row][col].imshow(image)
        axes[row][col].set_title("Original")
        axes[row][col].axis("off")

        # Heatmap
        threshold = 0.6
        color = "red" if score > threshold else "green"
        axes[row][col + 1].set_title(f"{score:.3f}", color=color)
        axes[row][col + 1].imshow(image)
        axes[row][col + 1].imshow(heatmap, alpha=0.5, cmap="jet")
        axes[row][col + 1].axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    #Loading model
    model = Patchcore.load_from_checkpoint(
    checkpoint_path=str(MODEL_PATH),
    map_location="cuda"
    )

    engine = Engine()
    data = Folder(
        name="Inference",
        root=IMAGE_PATH,
        normal_dir='.',
        abnormal_dir='.',
        train_batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        val_split_mode="none"
    )

    results = engine.predict(
        model=model,
        datamodule=data,
        ckpt_path=str(MODEL_PATH)
    )

    analyze_results(results=results)