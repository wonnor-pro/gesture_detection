import tensorflow as tf
from generate_train_test import get_train_test_list
import numpy as np
import cv2

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):

  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])


  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)


  logits = tf.layers.dense(inputs=dropout, units=8)

  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
  # Load training and eval data

  dataset_path = "dataset_sorted/"
  testdata_amount = 500
  train_file_path_list, train_label_list, test_file_path_list, test_label_list = get_train_test_list(dataset_path, testdata_amount)

  train_data = []
  train_labels = []
  eval_data = []
  eval_labels = []

  for file in train_file_path_list:
      im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
      im = im.astype(float)
      im = im.flatten()
      train_data.append(im)

  for file in test_file_path_list:
      im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
      im = im.astype(float)
      im = im.flatten()
      eval_data.append(im)

  for label in train_label_list:
      label_array = np.zeros(8)
      label_array[label] = 1
      train_labels.append(label)

  for label in test_label_list:
      label_array = np.zeros(8)
      label_array[label] = 1
      eval_labels.append(label)

  train_data = np.array(train_data)
  train_labels = np.array(train_labels)
  eval_data = np.array(eval_data)
  eval_labels = np.array(eval_labels)

  print(train_data.shape)
  print(train_data[0])
  print(train_labels.shape)
  print(train_labels)
  print(eval_data.shape)
  print(eval_labels.shape)

  num_train_sample = len(train_data)
  num_test_sample = len(eval_data)
  BATCH_SIZE = 32

  print('--------------------------')
  print("Number of training examples: {}".format(num_train_sample))
  print("Number of test examples:     {}".format(num_test_sample))
  print('--------------------------')


  # Create the Estimator
  gesture_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=BATCH_SIZE,
      num_epochs=None,
      shuffle=True)

  gesture_classifier.train(
      input_fn=train_input_fn,
      steps=20000,
      hooks=[logging_hook])

  # Evaluate

  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = gesture_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()