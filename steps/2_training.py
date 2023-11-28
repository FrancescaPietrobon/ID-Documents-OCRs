import yaml
import os
from ultralytics import YOLO

params = yaml.safe_load(open("params.yaml"))["train"]
image_size = params['image_size']
epochs = params['epochs']
batch_size = params['batch_size']
learning_rate_initial = float(params['learning_rate_initial'])
learning_rate_final = float(params['learning_rate_final'])
box = params['box']
cls = params['cls']
patience = params['patience']

hsv_h = params['hsv_h']
hsv_s = params['hsv_s']
hsv_v = params['hsv_v']
degrees = params['degrees']
translate = params['translate']
scale = params['scale']
shear = params['shear']
perspective = params['perspective']
flipud = params['flipud']
fliplr = params['fliplr']
mosaic = params['mosaic']
mixup = params['mixup']
copy_paste = params['copy_paste']

seed = params['seed']
directory_output = params['directory_output']
optimizer = params['optimizer']

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results = model.train(data = "yolov8Config.yaml",
                      imgsz = image_size,
                      epochs = epochs,
                      batch = batch_size,
                      lr0 = learning_rate_initial,
                      lrf = learning_rate_final,
                      box = box,
                      cls = cls,
                      patience = patience,
                      seed = seed,
                      hsv_h = hsv_h,
                      hsv_s = hsv_s,
                      hsv_v = hsv_v,
                      degrees = degrees,
                      translate = translate,
                      scale = scale,
                      shear = shear,
                      perspective = perspective,
                      flipud = flipud,
                      fliplr = fliplr,
                      mosaic = mosaic,
                      mixup = mixup,
                      copy_paste = copy_paste,
                      project = directory_output,
                      optimizer = optimizer)  # train the model

model.export(format='onnx', imgsz=image_size)

# Remove the chache file that create a problem with the dvc pipeline
file_paths = ['datasets/train/labels.cache', 'datasets/validation/labels.cache']
for path in file_paths:
    if os.path.exists(path):
        os.remove(path)
        