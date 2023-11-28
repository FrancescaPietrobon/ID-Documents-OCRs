import numpy as np
import pandas as pd
import os
import sys
import yaml
import random
import itertools as it
from collections import Counter
from LabelEncoder import LabelEncoder
from utilities.utils_1_extract_data import parse_xml, convert_chars_to_class_id, preprocess_data
from utilities.utils import load_images_from_folder, get_weights, get_prior

# Check inputs
if len(sys.argv) != 3:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write("\tpython 1_extract_data.py data\n")
    sys.exit(1)

# Extract parameters
params = yaml.safe_load(open("params.yaml"))["prepare"]
dim = (params["dim"], params["dim"])
random.seed(params["seed"])
split_train = params["train"]
split_val = params["val"]

# Path of the dataset's folder
folder_path_doc = sys.argv[1]
folder_path_xml = sys.argv[2]

# Extract data
chars, coords_all, files = parse_xml(folder_path_xml)
images = load_images_from_folder(folder_path_doc, files)

# Extract weights of each char
weights = get_weights(chars)
prior = get_prior(chars)

# Convert characters in the corresponding value in the dictionary
classes_id = convert_chars_to_class_id(chars)

# Get occurence of each of the element
list_all = list(map(str, it.chain.from_iterable(chars)))
duplicate_dict = Counter(list_all)
data = []
for k,v in  duplicate_dict.most_common():
    data.append([k, v])
dataset_info = pd.DataFrame(data, columns=["Character", "Occurrences"])

os.makedirs("info", exist_ok=True)
with open("info/characters_occurrences.txt", 'w') as outfile:
    outfile.write(dataset_info.to_markdown())

# Preprocess data
images_new, bboxes_new, x_alter_new, y_alter_new = preprocess_data(images, coords_all, dim)

# Split in train, validation and test
num_images = len(images_new)

x_train = np.asarray(images_new[:int(num_images*split_train)]).astype('float32')
x_val = np.asarray(images_new[int(num_images*split_train):int(num_images*split_train+num_images*split_val)]).astype('float32')
x_test = np.asarray(images_new[int(num_images*split_train+num_images*split_val):]).astype('float32')

boxes_train = bboxes_new[:int(num_images*split_train)]
boxes_val = bboxes_new[int(num_images*split_train):int(num_images*split_train+num_images*split_val)]
boxes_test = bboxes_new[int(num_images*split_train+num_images*split_val):]

labels_train = classes_id[:int(num_images*split_train)]
labels_val = classes_id[int(num_images*split_train):int(num_images*split_train+num_images*split_val)]
labels_test = classes_id[int(num_images*split_train+num_images*split_val):]

# Transforms the raw labels into targets for training
label_encoder = LabelEncoder()
images_train, lab_train = label_encoder.encode_batch(x_train, boxes_train, labels_train)
images_val, lab_val = label_encoder.encode_batch(x_val, boxes_val, labels_val)

# Save numpy array of pairs
output_folder = os.path.join("extracted_data")
os.makedirs(output_folder, exist_ok=True)

np.save(os.path.join(output_folder, "train_images.npy"), images_train)
np.save(os.path.join(output_folder, "train_labels.npy"), lab_train)
np.save(os.path.join(output_folder, "val_images.npy"), images_val)
np.save(os.path.join(output_folder, "val_labels.npy"), lab_val)
np.save(os.path.join(output_folder, "test_images.npy"), x_test)
np.save(os.path.join(output_folder, "chars_weights.npy"), weights)
np.save(os.path.join(output_folder, "chars_prior.npy"), prior)

# Save number of images in train, val and test
data = [len(x_train), len(x_val), len(labels_test)]
dataset_info = pd.DataFrame(data, index=["Train", "Validation", "Test"], columns=["Number of images"])
with open("info/dataset_info.txt", "w") as outfile:
    outfile.write(dataset_info.to_markdown())
    