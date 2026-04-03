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
MODEL_PATH = pathlib.Path(r"D:\Upskill\Mini_Projects\cv-industrial-inspection-system\results\Patchcore\metal_nut\latest\weights\lightning\model.ckpt")
IMAGE_PATH = os.path.join(BASE_DIR, "anomaly_data", "metal_nut", "sample")


# def analyze_results(results):
#     fig, axes = plt.subplots(len(results), 2, figsize=(20, 6*len(results)))
#     for i, r in enumerate(results):
#         #r.image is of the format (Batch, Channels, Height, Width) but plt expects (Height, Width, Channels)
#         #squeeze() removes the batch parameter
#         #permute rearranges things
#         #cpu().numpy() moves the tensor to cpu and converts to numpy
#         #Same process with heatmap but no need to permute
#         image = r.image.squeeze().permute(1, 2, 0).cpu().numpy()
#         heatmap = r.anomaly_map.squeeze().cpu().numpy()
#         score = r.pred_score.item()

#         #Normalise the image values to be between 0 and 1 as that is what matplotlib expects
#         image = (image - image.min()) / (image.max() - image.min() + 1e-8)

#         # Original image
#         axes[i][0].imshow(image)
#         axes[i][0].set_title("Original")
#         axes[i][0].axis("off")

#         # Heatmap overlay
#         axes[i][1].imshow(image)
#         axes[i][1].imshow(heatmap, alpha=0.5, cmap="jet")
#         axes[i][1].set_title(f"Anomaly Score: {score:.3f}")
#         axes[i][1].axis("off")

#     plt.tight_layout()
#     plt.show()

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
        axes[row][col + 1].imshow(image)
        axes[row][col + 1].imshow(heatmap, alpha=0.5, cmap="jet")
        axes[row][col + 1].set_title(f"{score:.3f}")
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