import os
import numpy as np
import cv2 as cv
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from utilities.utils import inv_dictionary, preprocessing

def rgb_to_bgr(x):
  # 'RGB'->'BGR'
  #x = x[..., ::-1]
  mean = [103.939, 116.779, 123.68]
  x = x.astype(float)
  # Zero-center by mean pixel
  x[..., 0] = x[..., 0] - mean[0]
  x[..., 1] = x[..., 1] - mean[1]
  x[..., 2] = x[..., 2] - mean[2]
  return x


def predict_image(model, anchor_boxes, threshold, img):
  img = rgb_to_bgr(img)
  img_exp = np.array(tf.expand_dims(img, axis=0))
  predictions = model.predict(img_exp)
  box_predictions = tf.reshape(predictions[0,:,0,:4], [predictions.shape[1], 4]) * np.array([0.1, 0.1, 0.2, 0.2])
  cls_predictions_pre = tf.reshape(predictions[0,:,0,4:], [predictions.shape[1], 63])
  cls_predictions = tf.nn.sigmoid(cls_predictions_pre)
  boxes = tf.concat([box_predictions[:, :2] * anchor_boxes[:, 2:] + anchor_boxes[:, :2],
                     tf.math.exp(box_predictions[:, 2:]) * anchor_boxes[:, 2:]], axis=-1)
  corners = tf.concat([boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0], axis=-1)
  bbox = np.array(corners, dtype="int32")
  predictions = np.array(cls_predictions)
  indexes_max = np.argmax(predictions, axis=1)
  maxes = np.max(predictions, axis=1)
  bbox_new_l = []
  predictions_new_l = []
  class_l = []
  for i in range(len(indexes_max)):
    if(maxes[i]>=threshold):
      bbox_new_l.append(bbox[i])
      predictions_new_l.append(maxes[i])
      class_l.append(indexes_max[i])
  return bbox_new_l, class_l, predictions_new_l


def save_detections(image, boxes, classes, scores, title, path, filename):
  plt.figure(figsize=(25,25))
  plt.axis("off")
  plt.imshow(image, aspect="auto")
  ax = plt.gca()
  for box, _cls, score in zip(boxes, classes, scores):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    rect = patches.Rectangle((x1, y1), w, h, linewidth=1,
                              edgecolor=color_map_color(int(_cls)), facecolor="none")
    ax.add_patch(rect)
    text = inv_dictionary[str(int(_cls))]
    ax.text(x1, y1, text)
  save_filename = f"{filename}_{title}.jpg"
  plt.savefig(os.path.join(path, save_filename), bbox_inches="tight", pad_inches=0)
  plt.close()


def color_map_color(value, cmap_name='rainbow', vmin=0, vmax=80):
  norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
  cmap = cm.get_cmap(cmap_name)  # PiYG
  rgb = cmap(norm(abs(value)))[:3]  # will return rgba, we take only first 3 so we get rgb
  color = matplotlib.colors.rgb2hex(rgb)
  return color
