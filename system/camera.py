from anomalib.models import Patchcore
from ultralytics import YOLO
import cv2
import numpy as np
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import csv

from yolo_inference import *
from patchcore_inference import *

#Loading environment variables
load_dotenv()
MODEL_DIR = os.getenv("MODEL_DIR")
YOLO_MODELS = os.getenv("YOLO_MODELS")
PATCHCORE_MODEL_PATH = os.getenv("PATCHCORE_MODEL_PATH")

BASE_DIR = os.path.dirname(__file__)


#Loading YOLO Model
latest_yolo_model = os.listdir(YOLO_MODELS)[-1]
if not latest_yolo_model:
    yolo_found = 0
    print("No YOLO model found. Please train a model by executing the script 'yolo_model_training.py' in the 'models' folder.")
else:
    yolo_found = 1
    yolo_model = YOLO(os.path.join(YOLO_MODELS, latest_yolo_model, "weights", "best.pt"))

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

#Loading CSV file to log data
csvFile = open(os.path.join(BASE_DIR, f"Log_{datetime.date(datetime.now())}.csv"), 'a', newline='')
fieldNames = ['Timestamp','Status','Score/Class']
writer = csv.DictWriter(csvFile, fieldnames=fieldNames)
if csvFile.tell() == 0:
    writer.writeheader()

#Camera Loop
def stream(source=0):
    if source == 0:
        print("Attempting to open camera...")
        cap = cv2.VideoCapture(source, apiPreference=cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error opening camera.")
        return
    
    print("Stream started successfully.")
    
    cam_fps = 30
    cap.set(cv2.CAP_PROP_FPS, cam_fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cam_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cam_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    print("CAMERA HEIGHT: ", cam_height)
    print("CAMERA WIDTH: ", cam_width)

    ret, frame = cap.read()
    if ret:
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

    #Define a Region of Interest. A part of the frame that will be processed.
    x = int(frame_width)//4
    w = int(frame_width)//2
    y = int(frame_height)//4
    h = int(frame_height)//2
    roi = np.zeros(shape=frame.shape[:2], dtype=np.uint8)
    roi[y:y+h, x:x+w] = 255

    frame_count = 0
    start_time = time.perf_counter()
    fps = 0
    last_heatmap = None

    while cap.isOpened():
        loop_start = time.perf_counter()
        ret, frame = cap.read()

        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1

        #Overlay metrics on the stream
        cv2.putText(frame, f"FPS: {fps}", org=(10,20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,255))
        cv2.putText(frame, f"Image Resolution: {cam_width}x{cam_height}", org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,255))
        cv2.putText(frame, f"ROI Resolution: {w}x{h}", org=(10,60), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,255))

        #Preprocessing the ROI and adding it back to the frame. Saves resources if we only process the ROI and leave the rest untouched.
        processed_frame = frame.copy()
        roi_frame = processed_frame[y:y+h, x:x+w]
        cv2.rectangle(processed_frame, pt1=(x, y), pt2=(x+w, y+h), color=[0,255,0], thickness=3)

        #Inference
        data = {
                'Timestamp':datetime.time(datetime.now()),
                'Status':'CLEAN',
                'Score/Class':None
            }
        yolo_results = run_yolo(frame=roi_frame, model=yolo_model)
        if yolo_results is not None:
            boxes, classes = yolo_results
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{yolo_model.names[cls]} {conf:.2f}"
                cv2.rectangle(img=roi_frame, pt1=(x1, y1), pt2=(x2, y2), color=(0, 0, 255), thickness=1)
                cv2.putText(img=roi_frame, text=label, org=(x1, y1-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.5, color=(0, 0, 255), thickness=1)
                data = {
                    'Timestamp':datetime.time(datetime.now()),
                    'Status':'DEFECT',
                    'Score/Class':yolo_model.names[cls]
                }
                writer.writerow(data)
        else:
            if frame_count % 10 == 0:
                score, heatmap = run_patchcore_frame(frame=roi_frame, model=patchcore_model)
                if score > 9:
                    roi_frame = cv2.addWeighted(src1=roi_frame, alpha=0.6, src2=heatmap, beta=0.4, gamma=0)                   
                    last_heatmap = heatmap
                    data = {
                        'Timestamp':datetime.time(datetime.now()),
                        'Status':'ANOMALY',
                        'Score/Class':score
                    }
            elif last_heatmap is not None:
                roi_frame = cv2.addWeighted(src1=roi_frame, alpha=0.6, src2=last_heatmap, beta=0.4, gamma=0)
        
            writer.writerow(data)

        
        processed_frame[y:y+h, x:x+w] = roi_frame


        #Display frames
        cv2.imshow("Processed Frame", processed_frame)            

        #Wait for user input to close windows
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #Calculate FPS with Smoothening (Signal Filtering)
        alpha = 0.1
        loop_end = time.perf_counter()
        frame_time = loop_end - loop_start
        new_fps = 1/frame_time
        fps = ((1-alpha) * fps) + (alpha * new_fps)
    
    csvFile.close()
    cap.release()
    cv2.destroyAllWindows()

    end_time = time.perf_counter()
    print()
    print("AVERAGE FPS: ", frame_count/(end_time-start_time))
    print("TOTAL TIME: ", end_time - start_time)