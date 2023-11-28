import os
import numpy as np
import cv2 as cv
import tensorflow as tf
import itertools as it
from collections import Counter

dictionary = {"0": "0",
              "1": "1",
              "2": "2",
              "3": "3",
              "4": "4",
              "5": "5",
              "6": "6",
              "7": "7",
              "8": "8",
              "9": "9",
              "A": "10",
              "a": "11",
              "B": "12",
              "b": "13",
              "C": "14",
              "c": "15",
              "D": "16",
              "d": "17",
              "E": "18",
              "e": "19",
              "F": "20",
              "f": "21",
              "G": "22",
              "g": "23",
              "H": "24",
              "h": "25",
              "I": "26",
              "i": "27",
              "J": "28",
              "j": "29",
              "K": "30",
              "k": "31",
              "L": "32",
              "l": "33",
              "M": "34",
              "m": "35",
              "N": "36",
              "n": "37",
              "O": "38",
              "o": "39",
              "P": "40",
              "p": "41",
              "Q": "42",
              "q": "43",
              "R": "44",
              "r": "45",
              "S": "46",
              "s": "47",
              "T": "48",
              "t": "49",
              "U": "50",
              "u": "51",
              "V": "52",
              "v": "53",
              "W": "54",
              "w": "55",
              "X": "56",
              "x": "57",
              "Y": "58",
              "y": "59",
              "Z": "60",
              "z": "61",
              "<": "62"
              }

inv_dictionary = dict((v, k) for k, v in dictionary.items())

num_classes = len(dictionary)

def load_images_from_folder(folder, filenames):
  """Extract images form the folder.

  Arguments:
      folder: path of the images's folder.
      dim: tuple (height, width) of the dimension required for the image.

  Returns:
      list of numpy arrays of int values with dimension (dim[0], dim[1], 3) (3 for colors channels)
  """
  images = []
  for i in range(len(filenames)):
    img = cv.imread(os.path.join(folder,filenames[i]), cv.IMREAD_GRAYSCALE)
    if img is not None:
      img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
      img = preprocessing(img)
      images.append(img)               
  return images


def preprocessing(img):
  img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
  img = cv.fastNlMeansDenoising(img, h = 10)
  return img


def get_weights(chars):
  list_all = list(map(str, it.chain.from_iterable(chars)))
  duplicate_dict = Counter(list_all)
  tot = len(list_all)
  # Construct prior with the frequency of every class (freq_item/tot_items)
  frequences = []
  prior = []
  weights = []
  for i in range(num_classes):
    freq = (duplicate_dict[inv_dictionary[str(i)]])/tot
    frequences.append(freq)
    prior_val = (duplicate_dict[inv_dictionary[str(i)]]+1)/tot #*0.5
    prior.append(prior_val)
    weight = 10/((duplicate_dict[inv_dictionary[str(i)]]+1) * num_classes)
    weights.append(weight)
  return weight


def get_prior(chars):
  list_all = list(map(str, it.chain.from_iterable(chars)))
  duplicate_dict = Counter(list_all)
  tot = len(list_all)
  # Construct prior with the frequency of every class (freq_item/tot_items)
  frequences = []
  prior = []
  weights = []
  for i in range(num_classes):
    freq = (duplicate_dict[inv_dictionary[str(i)]])/tot
    frequences.append(freq)
    prior_val = (duplicate_dict[inv_dictionary[str(i)]]+1)/tot *0.5
    prior.append(prior_val)
    weight = 10/((duplicate_dict[inv_dictionary[str(i)]]+1) * num_classes)
    weights.append(weight)
  return prior


def resize_and_pad_image(img, bbox, dim):
  w = dim[0]
  h = dim[1]
  img_w = img.shape[1]
  img_h = img.shape[0]
  img = cv.resize(np.squeeze(img), dim, interpolation = cv.INTER_CUBIC)
  # x_alter and y_alter are like ratio
  x_alter = w / img_w
  y_alter = h / img_h
  bbox = tf.stack(
      [
          bbox[:, 0] * x_alter,
          bbox[:, 1] * y_alter,
          bbox[:, 2] * x_alter,
          bbox[:, 3] * y_alter,
      ],
      axis=-1,
  )
  return img, bbox, x_alter, y_alter


def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """Changes the box format to corner coordinates

    Arguments:
      boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.

    Returns:
      converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )


def compute_iou(boxes1, boxes2):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    boxes1_corners = convert_to_corners(boxes1)
    boxes2_corners = convert_to_corners(boxes2)
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = boxes1[:, 2] * boxes1[:, 3]
    boxes2_area = boxes2[:, 2] * boxes2[:, 3]
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)
