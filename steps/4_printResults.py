import os
import pandas as pd
import shutil
from utilsPy.utils_printResults import correct_column_names, plot_single, plot_generic

lista_path = ['output', 'plots', 'models']
for path in lista_path:
    if not os.path.exists(path):
        os.mkdir(path)

df = pd.read_csv('output/train/results.csv')
columns_names = correct_column_names(df.columns.to_list())
df.columns = columns_names

# Creating plot for training/validation
plot_single(df['metrics/mAP50-95(B)'].to_list(), 'MAP_50-95')
plot_generic(df['train/box_loss'].to_list(), df['val/box_loss'].to_list(), 'box_loss')
plot_generic(df['train/cls_loss'].to_list(), df['val/cls_loss'].to_list(), 'classification_loss')
plot_generic(df['train/dfl_loss'].to_list(), df['val/dfl_loss'].to_list(), 'generalized_focal_loss')

shutil.copy("output/train/confusion_matrix_normalized.png", "plots/confusion_matrix_normalized_validation.png")
shutil.copy("output/train/confusion_matrix.png", "plots/confusion_matrix_validation.png")

# Creating plot for test
shutil.copy("output/val/confusion_matrix_normalized.png", "plots/confusion_matrix_normalized_test.png")
shutil.copy("output/val/confusion_matrix.png", "plots/confusion_matrix_test.png")
shutil.copy("output/val/metrics_test.png", "plots/metrics_test.png")

# Move model
shutil.copy("output/train/weights/best.onnx", "models/ocr-yolov8-binary.onnx")

