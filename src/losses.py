import tensorflow as tf
import yaml
from utilities.utils import num_classes

params = yaml.safe_load(open("params.yaml"))["loss"]
delta = params["delta"]
alpha = params["alpha"]

def retinanet_loss(y_true, y_pred):
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    box_labels = y_true[:, :, :4]
    box_predictions = y_pred[:, :, :, :4]
    cls_labels = tf.one_hot(
        tf.cast(y_true[:, :, 4], dtype=tf.int32),
        depth=num_classes,
        dtype=tf.float32,
    )
    cls_predictions = y_pred[:, :, :, 4:]
    positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
    ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
    clf_loss = retinanet_classificaion_loss(cls_labels, cls_predictions)
    box_loss = retinanet_box_loss(box_labels, box_predictions)
    clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
    box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
    normalizer = tf.reduce_sum(positive_mask, axis=-1)
    clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
    box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
    loss = clf_loss + box_loss
    return loss


def retinanet_classificaion_loss(y_true, y_pred):
    y_pred = tf.reshape(y_pred, (1, 200145, 63))
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    alpha_new = tf.where(tf.equal(y_true, 1.0), alpha, 1-alpha)
    loss = alpha_new * cross_entropy
    return tf.reduce_sum(loss, axis=-1)


def retinanet_box_loss(y_true, y_pred):
    y_pred = tf.reshape(y_pred, (1, 200145, 4))
    difference = y_true - y_pred
    absolute_difference = tf.abs(difference)
    squared_difference = difference ** 2
    loss = tf.where(
        tf.less(absolute_difference, delta),
        0.5 * squared_difference,
        absolute_difference - 0.5,
    )
    return tf.reduce_sum(loss, axis=-1)
