import sys
import yaml
import os
import numpy as np
import cv2 as cv
import random
from tensorflow import keras
from Anchor import AnchorBox
from utilities.utils_3_test import predict_image
from utilities.utils_1_extract_data import parse_xml, convert_chars_to_class_id
from utilities.utils_4_evaluation import calculate_average_precision_over_images
from utilities.utils import load_images_from_folder, resize_and_pad_image,  preprocessing, num_classes, dictionary

# Check inputs
if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython 4_evaluate.py model\n")
    sys.exit(1)

# Extract parameters
params_prepare = yaml.safe_load(open("params.yaml"))["prepare"]
dim = (params_prepare["dim"], params_prepare["dim"])
random.seed(params_prepare["seed"])
split_train = params_prepare["train"]
split_val = params_prepare["val"]

params_test = yaml.safe_load(open("params.yaml"))["test"]
threshold = params_test["threshold"]
threshold_nms = params_test["threshold_nms"]

# Load model
model_path = sys.argv[1]
model = keras.models.load_model(model_path, compile=False)

# Extract data
folder_path = 'data/Img'
image_extensions = ['.jpg', '.jpeg', '.png']
num_images = 0
for filename in os.listdir(folder_path):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        num_images += 1

chars, coords_all, files = parse_xml("data/XML")
files_test = files[int(num_images*split_train+num_images*split_val):]
images_test = load_images_from_folder("data/Img", files_test)
chars_test = chars[int(num_images*split_train+num_images*split_val):]
labels_test = convert_chars_to_class_id(chars_test)
coords_test = coords_all[int(num_images*split_train+num_images*split_val):]

# Calculate APs
os.makedirs("info", exist_ok=True)
APs = {}
total_ap = 0
for class_id in range(num_classes):
    ground_truth_list = []
    predictions_list = []
    for i, image_file in enumerate(files_test):
        image_path = os.path.join('data/Img', image_file)
        img = cv.imread(image_path)
        img = preprocessing(img)
        img = cv.resize(img, dim, interpolation = cv.INTER_CUBIC) 
        anchor_boxes = AnchorBox().get_anchors(dim[0], dim[1])
        bbox_new_l, class_l, predictions_new_l = predict_image(model, anchor_boxes, threshold, img)
        index = cv.dnn.NMSBoxes((bbox_new_l), (predictions_new_l), threshold, threshold_nms)
        nms_bbox = np.array(bbox_new_l)[index]
        nms_pred = np.array(predictions_new_l)[index]
        nms_class = np.array(class_l)[index]
        if len(nms_pred) == 0:
            break
        predictions = [list(values) for values in zip(nms_class, nms_pred, nms_bbox)]
        _, bbox_test, x_alter, y_alter = resize_and_pad_image(images_test[i], coords_test[i], dim)
        bbox_test = bbox_test.numpy().tolist()
        prob = [1] * len(bbox_test)
        ground_truth = [list(values) for values in zip(labels_test[i], prob, bbox_test)]
        img_ground_truth = [gt for gt in ground_truth if gt[0] == class_id]
        img_predictions = [pred for pred in predictions if pred[0] == class_id]
        ground_truth_list.append(img_ground_truth)
        predictions_list.append(img_predictions)
    ap = calculate_average_precision_over_images(predictions_list, ground_truth_list, confidence_threshold=threshold, iou_threshold=0.5)
    print(f"class_id: {class_id} \tap: {ap}")
    APs[class_id] = ap
    total_ap += ap

# Calculate mAP
mAP = np.mean(list(APs.values()))
print("mAP: ", mAP)

# Save APs and MAP
with open('info/mAP.txt', 'w') as f:
    f.write("Mean Average Precision (mAP): " + str(mAP))
    
with open('info/APs.txt', 'w') as file:
    for class_id, ap in APs.items():
        class_letter = [k for k, v in dictionary.items() if v == str(class_id)][0]
        file.write(f"Class {class_id} -> {class_letter}: AP = {ap}\n")
        
print("Mean Average Precision (mAP):", mAP)
print("APs for individual classes:", APs)
