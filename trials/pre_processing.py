import cv2
import numpy as np

# def global_thresholding(roi_frame):
#     _, processed_roi = cv2.threshold(src=roi_frame, thresh=127, maxval=255, type=cv2.THRESH_BINARY)
#     return processed_roi

# def otsu_thresholding(roi_frame):    
#     _, processed_roi = cv2.threshold(src=roi_frame, thresh=127, maxval=255, type=cv2.THRESH_OTSU)
#     return processed_roi

# def adaptive_thresholding(roi_frame):
#     processed_roi = cv2.adaptiveThreshold(src=roi_frame, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=21, C=2)
#     return processed_roi

def canny_edge(roi_frame):
    median = np.median(roi_frame)
    # low = int(0.25 * median)
    # high = int(1.5 * median)
    canny_frame = cv2.Canny(roi_frame, threshold1=20, threshold2=60)
    return canny_frame

def morphological_ops(roi_frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
    dilation = cv2.dilate(roi_frame, kernel)
    return dilation

def get_contours(roi_frame):
    contours, _ = cv2.findContours(image=roi_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = []
    for contour in contours:
        area = cv2.contourArea(contour=contour)
        x, y, w, h = cv2.boundingRect(contour)
        if area > 50 and area < 1000:
            valid_contours.append((x,y,w,h))
    return valid_contours

funcs = {
    # "Global Thresholding": global_thresholding,
    # "Otsu Thresholding": otsu_thresholding,
    # "Adaptive Thresholding": adaptive_thresholding,
    "Canny Edge": canny_edge
}

def pre_process(roi_frame, technique):
    func = funcs[technique]
    # roi_frame = cv2.GaussianBlur(roi_frame, ksize=(1,1), sigmaX=0) #Blurring image to remove noise

    #Converting to grayscale and getting Canny edges
    processed_roi = func(cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY))

    #Performing morphological ops on the canny edges
    processed_roi = morphological_ops(processed_roi)
    
    #Generating and filtering out contours
    valid_contours = get_contours(processed_roi)

    #Converting grayscale to BGR to introduce color channel again
    processed_roi = cv2.cvtColor(processed_roi, cv2.COLOR_GRAY2BGR)

    #Overlaying Bounding boxes over the contours
    for contour in valid_contours:
        x, y, w, h = contour
        processed_roi = cv2.rectangle(processed_roi, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=2)

    return processed_roi