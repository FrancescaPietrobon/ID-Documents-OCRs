import cv2 as cv
import sys
import os
import yaml
import numpy as np
from tensorflow import keras
from Anchor import AnchorBox
from utilities.utils_3_test import save_detections, predict_image
from utilities.utils import preprocessing

# Check inputs
if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython 3_test.py model folder_path\n")
    sys.exit(1)

# Extract parameters
params_prepare = yaml.safe_load(open("params.yaml"))["prepare"]
dim = (params_prepare["dim"], params_prepare["dim"])
split_train = params_prepare["train"]
split_val = params_prepare["val"]

params_test = yaml.safe_load(open("params.yaml"))["test"]
threshold = params_test["threshold"]
threshold_nms = params_test["threshold_nms"]

# Load model
model_path = sys.argv[1]
model = keras.models.load_model(model_path, compile=False)

# Load images from folder
folder_path = sys.argv[2]
num_images = len(os.listdir(folder_path))
num_skip = int(num_images*split_train+num_images*split_val)
image_files = [filename for filename in os.listdir(folder_path) if filename.lower().endswith(('.jpg', '.png', '.jpeg'))][num_skip:]

output_folder = os.path.join("evaluation_model", "predictions")
os.makedirs(output_folder, exist_ok=True)

for image_file in image_files:
    print("Image: ", image_file)
    image_path = os.path.join(folder_path, image_file)
    img = cv.imread(image_path)
    img = preprocessing(img)
    img = cv.resize(img, dim, interpolation = cv.INTER_CUBIC) 
    anchor_boxes = AnchorBox().get_anchors(dim[0], dim[1])
    bbox_new_l, class_l, predictions_new_l = predict_image(model, anchor_boxes, threshold, img)
    print("Pre NMS: ", len(bbox_new_l))

    save_detections(
        img,
        bbox_new_l,
        class_l,
        predictions_new_l,
        title="PreNMS",
        path=output_folder,
        filename=image_file
    )

    index = cv.dnn.NMSBoxes((bbox_new_l), (predictions_new_l), threshold, threshold_nms)
    print("Post NMS: ", len(index))

    nms_bbox = np.array(bbox_new_l)[index]
    nms_pred = np.array(predictions_new_l)[index]
    nms_class = np.array(class_l)[index]

    # POST NMS
    save_detections(
        img,
        nms_bbox,
        nms_class,
        nms_pred,
        title="PostNMS",
        path=output_folder,
        filename=image_file
    )
