import tensorflow as tf
import yaml
from utilities.utils import num_classes

params = yaml.safe_load(open("params.yaml"))["loss"]
delta = params["delta"]
alpha = params["alpha"]

def box_loss(y_true, y_pred):
    positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
    y_true = y_true[:, :, :4]
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_pred = y_pred[:, :, :, :4]
    y_pred = tf.reshape(y_pred, (1, 200145, 4))
    difference = y_true - y_pred
    absolute_difference = tf.abs(difference)
    squared_difference = difference ** 2
    box_loss = tf.where(
        tf.less(absolute_difference, delta),
        0.5 * squared_difference,
        absolute_difference - 0.5,
    )
    box_loss = tf.reduce_sum(box_loss, axis=-1)
    box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
    normalizer = tf.reduce_sum(positive_mask, axis=-1)
    box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
    return box_loss


def class_loss(y_true, y_pred):
    positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
    ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
    y_true = tf.one_hot(
        tf.cast(y_true[:, :, 4], dtype=tf.int32),
        depth=num_classes,
        dtype=tf.float32,
    )
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    y_pred = y_pred[:, :, :, 4:]
    y_pred = tf.reshape(y_pred, (1, 200145, 63))
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    alpha_new = tf.where(tf.equal(y_true, 1.0), alpha, 1-alpha)
    clf_loss = alpha_new * cross_entropy
    clf_loss = tf.reduce_sum(clf_loss, axis=-1) 
    clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
    normalizer = tf.reduce_sum(positive_mask, axis=-1)
    clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
    return clf_loss


def mean_iou(y_true, y_pred):
  cls_labels = tf.one_hot(tf.cast(y_true[:, :, 4], 'int32'), depth=num_classes)
  cls_predictions = y_pred[:, :, :, 4:]
  cls_predictions = tf.reshape(cls_predictions, (1, 200145, 63))
  true_pixels = tf.math.argmax(cls_labels, axis=-1) # exclude background
  pred_pixels = tf.math.argmax(cls_predictions, axis=-1)
  iou = []
  flag = tf.convert_to_tensor(-1, dtype='float64')
  for i in range(num_classes-1):
      true_labels = tf.math.equal(true_pixels, i)
      pred_labels = tf.math.equal(pred_pixels, i)
      inter = tf.cast(true_labels & pred_labels, 'int32')
      union = tf.cast(true_labels | pred_labels, 'int32')
      cond = (tf.reduce_sum(union) > 0) & (tf.reduce_sum(tf.cast((true_labels), 'int32')) > 0)
      res = tf.cond(cond, lambda: tf.reduce_sum(inter)/tf.reduce_sum(union), lambda: flag)
      iou.append(res)
  iou = tf.stack(iou)
  legal_labels = tf.greater(iou, flag)
  iou = tf.gather(iou, indices=tf.where(legal_labels))
  return tf.reduce_mean(iou)
