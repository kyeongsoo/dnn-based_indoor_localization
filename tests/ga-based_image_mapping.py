#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     test_siso_classifier.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-06-06
#
# @brief Prototype GA-based mapping of unstructured data to 2-D images.
#
# @remarks The results will be published in a paper submitted to the <a
#          href="http://www.sciencedirect.com/science/journal/08936080">Elsevier
#          Neural Networks</a> journal.

### import basic modules and a model to test
from __future__ import print_function
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
# from keras import backend as K
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

# permute train images
num_pixels = img_rows*img_cols
permute_idx = np.random.permutation(num_pixels)  # random permutation for 28*28 images
tmp = x_train.reshape(len(x_train), num_pixels)
tmp = tmp[:, permute_idx]
x_permuted = tmp.reshape(len(x_train), img_rows, img_cols, 1)
# tmp = x_test.reshape(len(x_test), num_pixels)
# tmp = tmp[:, permute_idx]
# x_test = tmp.reshape(len(x_test), img_rows, img_cols, 1)

# samples of original and permuted images
digit_idx = np.empty([10], dtype=int)
for i in range(10):
    digit_idx[i] = np.where(y_train == i)[0][0]
samples_original = x_train[digit_idx]
samples_permuted = x_permuted[digit_idx]

