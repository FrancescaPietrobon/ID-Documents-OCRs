import os
from utilsPy.utils_fixAnnotations import process_label_files_in_directory

annotations_in_path = './data/labels/'
annotations_out_path = './data/fixed_annotations/'

os.makedirs(annotations_out_path, exist_ok=True)

label_mapping = {}
with open('data/label_mapping.txt', 'r') as file:
    lines = file.readlines()
for line in lines:
    parts = line.strip().split(':')
    if len(parts) == 2:
        label, value = parts[0].strip(), parts[1].strip()
        label_mapping[label] = int(value)

process_label_files_in_directory(annotations_in_path, annotations_out_path, label_mapping)

