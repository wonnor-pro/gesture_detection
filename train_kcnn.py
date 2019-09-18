import tensorflow as tf
from generate_train_test import get_train_test_list
from dataset import load_dataset

import math
import numpy as np
import matplotlib.pyplot as plt
import os

tf.enable_eager_execution()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
dataset_path = "dataset_sorted/"
testdata_amount = 500
train_file_path_list, train_label_list, test_file_path_list, test_label_list = get_train_test_list(dataset_path, testdata_amount)
train_dataset = load_dataset(train_file_path_list, train_label_list, batchsize=32, repeat=10)
test_dataset = load_dataset(test_file_path_list, test_label_list, batchsize=32, repeat=10)

print(train_dataset)

plot = False
img_rows, img_cols = 28, 28

class_names = ['0SingleOne', '1SingleTwo', '2SingleFour', '3SingleSix',
               '4SingleEight', '5SingleNine', '6SingleBad', '7SingleGood']


print('--------------------------')
num_train_examples = len(train_file_path_list)
num_test_examples = len(test_file_path_list)
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))
print('--------------------------')


def normalize(images, labels):
  images = tf.cast(images, tf.float32)
  images /= 255
  return images, labels

# The map function applies the normalize function to each element in the train
# and test datasets
train_dataset =  train_dataset.map(normalize)
test_dataset  =  test_dataset.map(normalize)

print(train_dataset)
print(test_dataset)

# Take a single image, and remove the color dimension by reshaping
for image, label in test_dataset.take(1):
    print(image.shape)
    break
image = image.numpy().reshape((28,28))

if plot:
    # Plot the image - voila a piece of fashion clothing
    plt.figure()
    plt.imshow(image, cmap=plt.cm.binary)
    plt.colorbar()
    plt.grid(False)
    plt.show()

    plt.figure(figsize=(10,10))
    i = 0
    for (image, label) in test_dataset.shuffle(num_train_examples).take(25):
        image = image.numpy().reshape((28,28))
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)
        plt.xlabel(class_names[label])
        i += 1
    plt.show()

# --------------TRAINING------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(8,  activation=tf.nn.softmax),
])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation=tf.nn.relu,input_shape=(img_rows, img_cols, 1)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(8, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

BATCH_SIZE = 32
train_dataset = train_dataset.repeat().shuffle(num_train_examples).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model.fit(train_dataset, epochs=50, steps_per_epoch=math.ceil(num_train_examples/BATCH_SIZE))


model.save_weights('checkpoints/Sep18')


# ---------------TESTING-----------------------
test_loss, test_accuracy = model.evaluate(test_dataset, steps=math.ceil(num_test_examples/32))
print('Accuracy on test dataset:', test_accuracy)

# ---------------INFERENCING-------------------
for test_images, test_labels in test_dataset.shuffle(num_test_examples).take(1):
    test_images = test_images.numpy()
    test_labels = test_labels.numpy()
    predictions = model.predict(test_images)

# ---------------SHOW THE RESULT-------------------
prediction_labels = []
for i in range(predictions.shape[0]):
    prediction_labels.append(np.argmax(predictions[i]))
print("predictions:", prediction_labels)
print("Ground truth:", test_labels)


def plot_image(i, predictions_array, true_labels, images):
    predictions_array, true_label, img = predictions_array[i], true_labels[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img[..., 0], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')



# # Plot the first X test images, their predicted label, and the true label
# # Color correct predictions in blue, incorrect predictions in red
# num_rows = 5
# num_cols = 3
# num_images = num_rows*num_cols
# plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# for i in range(num_images):
#   plt.subplot(num_rows, 2*num_cols, 2*i+1)
#   plot_image(i, predictions, test_labels, test_images)
#   plt.subplot(num_rows, 2*num_cols, 2*i+2)
#   plot_value_array(i, predictions, test_labels)
# plt.show()

