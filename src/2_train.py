import os
import yaml
import sys
import numpy as np
import pandas as pd
import nvgpu
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from utilities.utils import num_classes
from utilities.utils_2_train import get_model, plt_metric
from losses import retinanet_loss
from metrics import box_loss, class_loss, mean_iou

# Load parameters
params = yaml.safe_load(open("params.yaml"))["train"]
learning_rate = params["learning_rate"]
epochs = params["epochs"]
batch_size = params["batch_size"]
patience = params["patience"]

# Load data
data = "extracted_data"

# Load images
train_images = np.load(os.path.join(data, "train_images.npy"))
val_images = np.load(os.path.join(data, "val_images.npy"))

# Load labels
train_labels = np.load(os.path.join(data, "train_labels.npy"))
val_labels = np.load(os.path.join(data, "val_labels.npy"))
weights = np.load(os.path.join(data, "chars_weights.npy"))

# Initializing and compiling model
model = get_model(weights, num_classes)
loss_fn = retinanet_loss

optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
model.compile(loss=loss_fn, optimizer=optimizer, metrics=[box_loss, class_loss, mean_iou])

# Set callbacks
early_stopping_callbac = tf.keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=patience, restore_best_weights=True)
callbacks_list = [early_stopping_callbac]

# Make a table of GPU info
g = nvgpu.gpu_info()
df = pd.DataFrame.from_dict(g[0], orient="index", columns=["Value"])
with open("info/gpu_info.txt", "w") as outfile:
    outfile.write(df.to_markdown())

# Fit
history = model.fit(
          train_images,
          train_labels,
          validation_data=(val_images, val_labels),
          epochs=epochs,
          batch_size=batch_size,
          callbacks=callbacks_list,
          verbose=1
)

# Create frozen graph
os.makedirs("fit_res", exist_ok=True)
full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
frozen_func = convert_variables_to_constants_v2(full_model)                                                                                                                              
tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir="fit_res",
                  name='ocr_model.pb', as_text=False)

# Make directory for plots
evaluation_model_path = os.path.join("evaluation_model", "metric_acc")
os.makedirs(evaluation_model_path, exist_ok=True)

# Plot the losses
plt_metric(history=history.history, metric="loss", title="Loss", path=evaluation_model_path)
plt_metric(history=history.history, metric="mean_iou", title="MeanIOU", path=evaluation_model_path)
plt_metric(history=history.history, metric="box_loss", title="Box Loss", path=evaluation_model_path)
plt_metric(history=history.history, metric="class_loss", title="Classification Loss", path=evaluation_model_path)

# Save model
os.makedirs("model", exist_ok=True)
model.save("model")

# Save history
np.save(os.path.join("fit_res", "my_history.npy"), history.history)
