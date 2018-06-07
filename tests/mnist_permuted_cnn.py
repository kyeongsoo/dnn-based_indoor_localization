'''Trains a simple convnet on the permuted MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import sys
import numpy as np
# if 'matplotlib.pyplot' not in sys.modules:
#     if 'pylab' not in sys.modules:
#         matplotlib.use('Agg') # directly plot to a file when no GUI is available
#                               # (e.g., remote running)
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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

# samples of original images
digit_idx = np.empty([10], dtype=int)
for i in range(10):
    digit_idx[i] = np.where(y_train == i)[0][0]
samples_original = x_train[digit_idx]

# permute images
num_pixels = img_rows*img_cols
permute_idx = np.random.permutation(num_pixels)  # random permutation for 28*28 images
# permute_idx = np.roll(np.arange(num_pixels), 1)  # roll the elements of 1-dim array for 28*28 images
# for i in range(len(x_train)):
#     x_train[i] = ((np.ndarray.flatten(x_train[i]))[permute_idx]).reshape((img_rows, img_cols, 1))
tmp = x_train.reshape(len(x_train), num_pixels)
tmp = tmp[:, permute_idx]
x_train = tmp.reshape(len(x_train), img_rows, img_cols, 1)
# for i in range(len(x_test)):
#     x_test[i] = ((np.ndarray.flatten(x_test[i]))[permute_idx]).reshape((img_rows, img_cols, 1))
tmp = x_test.reshape(len(x_test), num_pixels)
tmp = tmp[:, permute_idx]
x_test = tmp.reshape(len(x_test), img_rows, img_cols, 1)

# samples of permutted images
samples_permuted = x_train[digit_idx]

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

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

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# display samples of original and permuted images
fig = plt.figure()
rows = 4
cols = 5
for i in range(5):
    fig.add_subplot(rows, cols, i+1)
    plt.imshow(samples_original[i,:,:,0], cmap='gray')
    plt.axis('off')
    fig.add_subplot(rows, cols, i+6)
    plt.imshow(samples_permuted[i,:,:,0], cmap='gray')
    plt.axis('off')
for i in range(5, 10):
    fig.add_subplot(rows, cols, i+6)
    plt.imshow(samples_original[i,:,:,0], cmap='gray')
    plt.axis('off')
    fig.add_subplot(rows, cols, i+11)
    plt.imshow(samples_permuted[i,:,:,0], cmap='gray')
    plt.axis('off')
fig.savefig('digits.png')
