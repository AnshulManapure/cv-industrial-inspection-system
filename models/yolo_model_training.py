from ultralytics import YOLO
import os


BASE_DIR = os.path.dirname(__file__)
data_path = os.path.join(BASE_DIR, "data")

def main():
    model = YOLO("yolov8s.pt")

    model.train(
        project=os.path.join(BASE_DIR, "runs"),
        data=os.path.join(data_path, "data.yaml"),
        epochs=30,
        imgsz=640,
        batch=24,
        workers=3,
        device=0
    )

if __name__ == "__main__":
    main()