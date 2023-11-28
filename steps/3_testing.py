import yaml
import os
from ultralytics import YOLO
from utilsPy.utils_testing import plot_metrics

params = yaml.safe_load(open("params.yaml"))["test"]
directory_output = params['directory_output']

# Import the model
model = YOLO('output/train/weights/best.onnx', task='detect')

results = model.val(data="yolov8ConfigTest.yaml", project = directory_output)

values = [round(results.results_dict['metrics/mAP50(B)'], 2), round(results.results_dict['metrics/mAP50-95(B)'], 2), round(results.results_dict['fitness'], 2)]
categories = ['mAP50(B)', 'mAP50-95(B)', 'fitness']
plot_metrics(categories, values)

file_path = 'datasets/test/labels.cache'
if os.path.exists(file_path):
    os.remove(file_path)
    