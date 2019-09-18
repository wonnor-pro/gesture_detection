import numpy as np
import tensorflow as tf

# This is my playground.

def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    print(train_data.shape)
    print(train_data[0])
    print(train_labels.shape)
    print(train_labels)
    print(eval_data.shape)
    print(eval_labels.shape)


if __name__ == "__main__":
    tf.app.run()