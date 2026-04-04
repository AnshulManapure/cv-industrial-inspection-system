import torch
from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Resize
import torch
import cv2
import numpy as np
import anomalib

torch.serialization.add_safe_globals([anomalib.PrecisionType])

transform = Compose([
    ToImage(),
    Resize(size=(256, 256)),
    ToDtype(dtype=torch.float32, scale=True)
])

device = "cuda" if torch.cuda.is_available() else "cpu"
def run_patchcore_frame(frame, model):    
    image = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        result = model(image)
        score = result.pred_score.item()
        anomaly_map = result.anomaly_map.squeeze().cpu().numpy()
        anomaly_map = cv2.normalize(src=anomaly_map, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        anomaly_map = anomaly_map.astype(np.uint8)
        heatmap = cv2.applyColorMap(anomaly_map, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(src=heatmap, code=cv2.COLOR_BGR2RGB)
        heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    
    return score, heatmap

