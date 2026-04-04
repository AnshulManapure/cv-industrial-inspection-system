from anomalib.models import Patchcore
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from system.patchcore_inference import *
from system.yolo_inference import *
from app.encoder import *

#Loading environment variables
BASE_DIR = os.path.dirname(__file__)
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))
MODEL_DIR = os.getenv("MODEL_DIR")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH")
PATCHCORE_MODEL_PATH = os.getenv("PATCHCORE_MODEL_PATH")

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

#Loading YOLO Model
if not os.path.exists(YOLO_MODEL_PATH):
    yolo_found = 0
    print("No YOLO model found. Please train a model by executing the script 'yolo_model_training.py' in the 'models' folder.")
else:
    yolo_found = 1
    yolo_model = YOLO(YOLO_MODEL_PATH)

#Loading PatchCore Model
if not os.path.exists(PATCHCORE_MODEL_PATH):
    patchcore_found = 0
    print("No PatchCore model found. Please train a model by executing the script 'anomaly_training.py' in the 'models' folder.")
else:
    patchcore_found = 1
    patchcore_model = Patchcore.load_from_checkpoint(checkpoint_path=str(PATCHCORE_MODEL_PATH), map_location="cuda")
    patchcore_model.post_processor = None
    patchcore_model.eval()

if yolo_found == 0 or patchcore_found == 0:
    raise RuntimeError("No models found for YOLO and/or PatchCore. Please train the models first.")


app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.post('/inspect')
async def inspect(file: UploadFile = File(...)):
    detections = []
    contents = await file.read()
    nparr = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    #Inference
    data = {
            'Status':'CLEAN'
        }
    yolo_results = run_yolo(frame=image, model=yolo_model)
    if yolo_results is not None:
        boxes, classes = yolo_results
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{yolo_model.names[cls]} {conf:.2f}"
            cv2.rectangle(img=image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=1)
            cv2.putText(img=image, text=label, org=(x1, y1-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.5, color=(0, 0, 255), thickness=1)
            data = {
                'Status':'DEFECT',
                'Class':yolo_model.names[cls]
            }
            detections.append(data)
    else:
        score, heatmap = run_patchcore_frame(frame=image, model=patchcore_model)
        if score > 9:
            image = cv2.addWeighted(src1=image, alpha=0.6, src2=heatmap, beta=0.4, gamma=0)
            data = {
                'Status':'ANOMALY',
                'Score':score
            }
        detections.append(data)
    
    encoded_image = encode_image(image=image)
    
    return {
        "results":detections,
        "image":encoded_image
    }