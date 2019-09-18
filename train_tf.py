import tensorflow as tf

import numpy as np
import cv2
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from generate_train_test import get_train_test_list

dataset_path = "dataset_sorted/"
testdata_amount = 500
train_file_path_list, train_label_list, test_file_path_list, test_label_list = get_train_test_list(dataset_path, testdata_amount)

train_images = []
train_labels = []
eval_images = []
eval_labels = []

for file in train_file_path_list:
    im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    im = im.astype(float)
    im = im.flatten()
    train_images.append(im)

for file in test_file_path_list:
    im = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    im = im.astype(float)
    im = im.flatten()
    eval_images.append(im)

for label in train_label_list:
    label_array = np.zeros(8)
    label_array[label] = 1
    train_labels.append(label)

for label in test_label_list:
    label_array = np.zeros(8)
    label_array[label] = 1
    eval_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
eval_images = np.array(eval_images)
eval_labels = np.array(eval_labels)

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

BUFFER_SIZE = len(train_images)

print('--------------------------')
num_train_examples = len(train_file_path_list)
num_test_examples = len(test_file_path_list)
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))
print('--------------------------')

batch_size = 128
num_classes = 8
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets

x_train = train_images
y_train = train_labels
x_test = eval_images
y_test = eval_labels

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
              optimizer=tensorflow.keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])