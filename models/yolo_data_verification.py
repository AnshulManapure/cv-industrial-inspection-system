import yaml
import os
import random
import cv2

def apply_BB(img, labels, classes):
    img_height, img_width, _ = img.shape
    for label in labels:
        label = label.strip().split(" ")
        label = list(map(lambda x:float(x), label))

        #Extracting class and location of the label; YOLO Format: [class x_centre y_centre bb_width bb_height]
        defect_class = classes[int(label[0])]
        x_center = label[1] * img_width
        y_center = label[2] * img_height
        box_width = label[3] * img_width
        box_height = label[4] * img_height

        #Getting upper left and lower right corners from the extracted values
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)
        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        #Fixing clipping
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_width, x2)
        y2 = min(img_height, y2)

        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0,0,255), thickness=2)
        cv2.putText(img, defect_class, (x1, y1), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0,0,255), thickness=1)

    return img


BASE_DIR = os.path.dirname(__file__)
data_path = os.path.join(BASE_DIR, "YOLO_data", "data.yaml")

#Reading metadata
with open(data_path) as f:
    data = yaml.safe_load(f)

#Extracting the YOLO classes
classes = {}
for i in range(data['nc']):
    classes[i] = data['names'][i]
print(classes)

#Getting images and labels
training_path = os.path.join(BASE_DIR, "YOLO_data", "train")
training_images = os.listdir(os.path.join(training_path, "images"))
training_labels = os.listdir(os.path.join(training_path, "labels"))

for i in range(5):
    selected_image = random.choice(training_images)
    selected_label = selected_image.replace('.jpg','.txt')
    print(selected_image)

    with open(os.path.join(training_path, "labels", selected_label)) as f:
        labels = f.readlines()

    img = cv2.imread(os.path.join(training_path, "images", selected_image))
    img = apply_BB(img=img, labels=labels, classes=classes)
    cv2.imshow(f"Image {i}", img)

if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()





