import yaml
import os
import time

def clean_labels(label_path, defect_class):
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
                    
                #Ignore segments. Only deal with bounding boxes
                if len(parts) != 5:
                    removed_boxes += 1
                    continue

                total_boxes += 1
                bb_class, x, y, w, h = parts
                bb_class = int(bb_class)
                x, y, w, h = map(float, (x, y, w, h))
                
                bb_area = w * h
                bb_aspect_ratio = w / h
                
                #Removing the problem class from the labels
                if defect_class != -1:
                    if bb_class == defect_class:
                        removed_boxes += 1
                        continue
                    else:
                        if bb_class > defect_class:
                            bb_class -= 1
                
                #Removing the labels where bounding box is bad
                if bb_area > 0.5 or bb_area < 0.0002: #Remove bounding boxes that cover too much or too little of the image
                    removed_boxes += 1
                    continue
                
                if bb_aspect_ratio > 10 or bb_aspect_ratio < 0.1: #Removing very thin boxes
                    removed_boxes += 1
                    continue

                if x < 0.02 or x > 0.98: #Removing boxes at the extreme edges of the image
                    removed_boxes += 1
                    continue
                
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
data_path = os.path.join(BASE_DIR, "data", "data.yaml")

#Reading metadata
with open(data_path) as f:
    data = yaml.safe_load(f)

#Extracting the YOLO classes
classes = {}
for i in range(data['nc']):
    classes[i] = data['names'][i]
    if classes[i] == "Defect":
        defect_class = i

print(classes)

if "Defect" not in data['names']:
    # raise ValueError("Defect class not found in dataset")
    defect_class = -1

start = time.perf_counter()
print("Cleaning Started...")
#Cleaning Training data
label_path = os.path.join(BASE_DIR, "data", "train", "labels")
clean_labels(label_path, defect_class)

#Cleaning Testing data
label_path = os.path.join(BASE_DIR, "data", "test", "labels")
clean_labels(label_path, defect_class)

#Cleaning Validation data
label_path = os.path.join(BASE_DIR, "data", "valid", "labels")
clean_labels(label_path, defect_class)

#Updating data.yaml
data['nc'] = 3
data['names'].remove("Defect")
with open(data_path, 'w') as f:
    yaml.dump(data, f, default_flow_style = False, sort_keys=False)

end = time.perf_counter()
print(f"Cleaning completed in {end-start} seconds.")