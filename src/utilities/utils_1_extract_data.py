import os
from lxml import etree
import numpy as np
from utilities.utils import dictionary, resize_and_pad_image, convert_to_xywh

def parse_xml(xmlFolder):
  chars = []
  files = []
  coords_all = []
  for xmlFile in os.listdir(xmlFolder):
    with open(os.path.join(xmlFolder,xmlFile)) as fobj:
      xml = fobj.read()
      root = etree.fromstring(xml)
      names = list()
      boxes = list()
      file_name = root.find('filename').text
      for objects in root.findall('.//object'):
        name = objects.find('name').text
        names.append(name)
      for box in root.findall('.//bndbox'):
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        coors = [xmin, ymin, xmax, ymax]
        boxes.append(np.array(coors))
    coords_all.append(np.array(boxes))
    chars.append(names)
    files.append(file_name)
  return chars, coords_all, files

def convert_chars_to_class_id(chars):
  classes_id = []
  for i in range(len(chars)):
    labels_int = []
    for j in range(len(chars[i])):
      labels_int.append(int(dictionary[str(chars[i][j])]))
    classes_id.append(np.array(labels_int).astype('float32'))
  return classes_id

def preprocess_data(images, bboxes, dim):
    """Applies preprocessing step to a single sample

    Arguments:
      dataset: A dataset with training or testing data

    Returns:
      image: Resized and padded image with random horizontal flipping applied.
      bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
        of the format `[x, y, width, height]`.
      class_id: An tensor representing the class id of the objects, having
        shape `(num_objects,)`.
    """
    images_list = []
    bboxes_list = []
    x_alter_list = []
    y_alter_list = []
    for i in range(len(images)):
      image, bbox, x_alter, y_alter = resize_and_pad_image(images[i], bboxes[i], dim)
      bbox = np.array(convert_to_xywh(bbox)).astype('float32')
      images_list.append(image)
      bboxes_list.append(bbox)
      x_alter_list.append(x_alter)
      y_alter_list.append(y_alter)
    images_new = np.array(images_list)
    bboxes_new = np.array(bboxes_list, dtype=object)
    x_alter_new = np.array(x_alter_list)
    y_alter_new = np.array(y_alter_list)
    return images_new, bboxes_new, x_alter_new, y_alter_new
