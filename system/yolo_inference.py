def run_yolo(frame, model):
    results = model(frame, verbose=False)

    boxes = results[0].boxes
    classes = results[0].names

    if boxes is None or len(boxes) == 0:
        return None
    else:
        return boxes, classes