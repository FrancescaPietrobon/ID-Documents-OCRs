import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

def get_model(weights, num_classes):
    base_cnn = keras.applications.ResNet50(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=(800, 800, 3),
        include_top=False) 

    class_weights = np.array(weights * 15)
    prior_probability = tf.constant_initializer(-np.log((1 - class_weights) / class_weights))
    cls_head = build_head(num_classes * 15, prior_probability) # #aspect_ratios*#scales
    box_head = build_head(4 * 15, "zeros")

    c3_output = base_cnn.get_layer("conv3_block4_out").output
    c4_output = base_cnn.get_layer("conv4_block6_out").output
    c5_output = base_cnn.get_layer("conv5_block3_out").output

    conv3 = tf.keras.layers.Conv2D(256, 1, 1, "same")(c3_output)
    conv4 = tf.keras.layers.Conv2D(256, 1, 1, "same")(c4_output)
    conv5 = tf.keras.layers.Conv2D(256, 1, 1, "same")(c5_output)

    up_output = tf.keras.layers.Resizing(50,50)(conv5)
    up1_output = tf.keras.layers.Resizing(100,100)(conv4)

    add1 = tf.keras.layers.Add() ([conv4, up_output])
    add2 = tf.keras.layers.Add() ([conv3, up1_output])
    p3_output = tf.keras.layers.Conv2D(256, 3, 1, "same")(add2)
    p4_output = tf.keras.layers.Conv2D(256, 3, 1, "same")(add1)
    p5_output = tf.keras.layers.Conv2D(256, 3, 1, "same")(conv5)
    p6_output = tf.keras.layers.Conv2D(256, 3, 2, "same")(c5_output)
    relu_output = tf.keras.layers.ReLU()(p6_output)
    p7_output = tf.keras.layers.Conv2D(256, 3, 2, "same")(relu_output)

    box_output_1 = tf.keras.layers.Reshape([150000, 1, 4])(box_head(p3_output))
    box_output_2 = tf.keras.layers.Reshape([37500, 1, 4])(box_head(p4_output)) 
    box_output_3 = tf.keras.layers.Reshape([9375, 1, 4])(box_head(p5_output))  
    box_output_4 = tf.keras.layers.Reshape([2535, 1, 4])(box_head(p6_output)) 
    box_output_5 = tf.keras.layers.Reshape([735, 1, 4])(box_head(p7_output))

    cls_output_1 = tf.keras.layers.Reshape([150000, 1, num_classes])(cls_head(p3_output))
    cls_output_2 = tf.keras.layers.Reshape([37500, 1, num_classes])(cls_head(p4_output))
    cls_output_3 = tf.keras.layers.Reshape([9375, 1, num_classes])(cls_head(p5_output))
    cls_output_4 = tf.keras.layers.Reshape([2535, 1, num_classes])(cls_head(p6_output))
    cls_output_5 = tf.keras.layers.Reshape([735, 1, num_classes])(cls_head(p7_output))

    box_outputs = tf.keras.layers.Concatenate(axis=1)([box_output_1, box_output_2, box_output_3, box_output_4, box_output_5])
    cls_outputs = tf.keras.layers.Concatenate(axis=1)([cls_output_1, cls_output_2, cls_output_3, cls_output_4, cls_output_5])

    model = keras.Model(base_cnn.input,tf.keras.layers.Concatenate(axis=3)([box_outputs, cls_outputs]))
    return model
    

def build_head(output_filters, bias_init):
  """Builds the class/box predictions head.

  Arguments:
    output_filters: Number of convolution filters in the final layer.
    bias_init: Bias Initializer for the final convolution layer.

  Returns:
    A keras sequential model representing either the classification
      or the box regression head depending on `output_filters`.
  """
  model_input = tf.keras.Input(shape=[None, None, 256])
  kernel_init = tf.initializers.RandomNormal(0.0, 0.01)

  for _ in range(4):
    conv = tf.keras.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)(model_input)
    relu = tf.keras.layers.ReLU()(conv)
  output = tf.keras.layers.Conv2D(output_filters,3,1,padding="same", kernel_initializer=kernel_init,bias_initializer=bias_init)(relu)

  head = keras.Model(model_input, output)
  
  return head

def plt_metric(history, metric, title, path, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        path: string with the path in which save the plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.savefig(os.path.join(path, metric + ".png"))
    plt.close()
    