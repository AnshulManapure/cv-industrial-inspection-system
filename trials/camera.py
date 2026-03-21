import cv2
import time
import numpy as np

from trials.pre_processing import *

def stream():
    print("Attempting to open camera...")
    cap = cv2.VideoCapture(index=0, apiPreference=cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("Error opening camera.")
        return
    
    print("Camera opened successfully.")
    
    cam_fps = 30
    cap.set(cv2.CAP_PROP_FPS, cam_fps)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

    while cap.isOpened():
        loop_start = time.perf_counter()

        ret, frame = cap.read()
        
        if not ret:
            break

        frame_count += 1

        #Overlay metrics on the stream
        cv2.putText(frame, f"FPS: {fps}", org=(10,20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,255))
        cv2.putText(frame, f"Image Resolution: {cam_width}x{cam_height}", org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,255))
        cv2.putText(frame, f"ROI Resolution: {w}x{h}", org=(10,60), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,255,255))

        #Preprocessing the ROI and adding it back to the frame. Saves resources if we only process the ROI and leave the rest untouched.
        processed_frames = {x: None for x in funcs.keys()}

        for technique in processed_frames:
            processed_frame = frame.copy()
            roi_frame = processed_frame[y:y+h, x:x+w].copy()
            processed_roi = pre_process(roi_frame, technique)
            processed_frame[y:y+h, x:x+w] = processed_roi
            #Draw rectangle around ROI
            cv2.rectangle(processed_frame, pt1=(x, y), pt2=(x+w, y+h), color=[0,255,0], thickness=3)
            processed_frames[technique] = processed_frame

        #Mask everything outside the ROI in the frame with black
        masked_frame = cv2.bitwise_and(src1=frame, src2=frame, mask=roi)

        #Display frames
        cv2.imshow("Original", frame)
        cv2.imshow("Original ROI", roi_frame)
        cv2.imshow("Processed Frame", processed_frame)
        # for technique in processed_frames:
        #     cv2.imshow(technique, processed_frames[technique])
            

        #Wait for user input to close windows
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # #Calculate FPS (Noisy Method) (FPS jumps around)
        # loop_end = time.perf_counter()
        # frame_time = loop_end - loop_start
        # fps = 1/frame_time

        #Calculate FPS with Smoothening (Signal Filtering)
        alpha = 0.1
        loop_end = time.perf_counter()
        frame_time = loop_end - loop_start
        new_fps = 1/frame_time
        fps = ((1-alpha) * fps) + (alpha * new_fps)
    
    cap.release()
    cv2.destroyAllWindows()

    end_time = time.perf_counter()
    print()
    print("AVERAGE FPS: ", frame_count/(end_time-start_time))
    print("TOTAL TIME: ", end_time - start_time)



if __name__ == "__main__":
    stream()