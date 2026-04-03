import yaml
import os
import time

def clean_labels(label_path, defect_classes):
    total_boxes = 0
    removed_boxes = 0
    removed_images = 0
    labels = [f for f in os.listdir(label_path) if f.endswith(".txt")]
    for label in labels:
        with open(os.path.join(label_path, label), 'r') as f:
            lines = f.readlines()
            valid_lines = []
            for line in lines:
                parts = line.split()
                
                #If the image has no BB
                if not parts:
                    continue

                total_boxes += 1
                    
                #Ignore segments. Only deal with bounding boxes
                if len(parts) != 5:
                    removed_boxes += 1
                    continue
                
                bb_class, x, y, w, h = parts
                bb_class = int(bb_class)
                x, y, w, h = map(float, (x, y, w, h))

                if h == 0:
                    removed_boxes += 1
                    continue
                
                bb_area = w * h
                bb_aspect_ratio = w / h
                
                #Removing the problem class from the labels
                if bb_class in defect_classes:
                    removed_boxes += 1
                    continue
                
                shift = sum(1 for d in defect_classes if bb_class > d)
                bb_class -= shift
                
                #Removing the labels where bounding box is bad
                if bb_area > 0.7 or bb_area < 0.00005: #Remove bounding boxes that cover too much or too little of the image
                    removed_boxes += 1
                    continue
                
                if bb_aspect_ratio > 10 or bb_aspect_ratio < 0.1: #Removing very thin boxes
                    removed_boxes += 1
                    continue

                # if x < 0.02 or x > 0.98: #Removing boxes at the extreme edges of the image
                #     removed_boxes += 1
                #     continue
                
                parts[0] = str(bb_class)
                valid_lines.append(" ".join(parts) + "\n")
        
        with open(os.path.join(label_path, label), 'w') as f:
            f.writelines(valid_lines)

        if len(valid_lines) == 0:
            removed_images += 1
            label_file_path = os.path.join(label_path, label)
            # remove label file
            os.remove(label_file_path)
            
            # remove corresponding image
            for ext in [".jpg", ".png", ".jpeg"]:
                image_path = label_file_path.replace("labels", "images").replace(".txt", ext)
                if os.path.exists(image_path):
                    os.remove(image_path)            
            continue
        
    print(f"Removed {(removed_boxes/total_boxes)*100}% boxes.")
    print(f"Removed {removed_images} images.")


BASE_DIR = os.path.dirname(__file__)
data_path = os.path.join(BASE_DIR, "YOLO_data", "data.yaml")

#Reading metadata
with open(data_path) as f:
    data = yaml.safe_load(f)

#Extracting the YOLO classes
classes = {}
defect_classes = []
for i in range(data['nc']):
    classes[i] = data['names'][i]
    if classes[i] == "Defect" or classes[i] == "Rust":
        defect_classes.append(i)

print(classes)

start = time.perf_counter()
print("Cleaning Started...")
#Cleaning Training data
label_path = os.path.join(BASE_DIR, "YOLO_data", "train", "labels")
clean_labels(label_path, defect_classes)

#Cleaning Testing data
label_path = os.path.join(BASE_DIR, "YOLO_data", "test", "labels")
clean_labels(label_path, defect_classes)

#Cleaning Validation data
label_path = os.path.join(BASE_DIR, "YOLO_data", "valid", "labels")
clean_labels(label_path, defect_classes)

#Updating data.yaml
data['nc'] = 2
data['names'].remove("Defect")
data['names'].remove("Rust")
with open(data_path, 'w') as f:
    yaml.dump(data, f, default_flow_style = False, sort_keys=False)

end = time.perf_counter()
print(f"Cleaning completed in {end-start} seconds.")