#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     ga-based_image_mapping.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-06-06
#
# @brief    Prototype GA-based mapping of unstructured data to 2-D images.
#
# @remarks   The results will be published in a paper submitted to the <a
#            href="http://www.sciencedirect.com/science/journal/08936080">Elsevier
#            Neural Networks</a> journal.


### TODO: Remove dependence on Keras/TensrFlow


### import basic modules and a model to test
from __future__ import print_function
import keras
from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import sys
import numpy as np
if 'matplotlib.pyplot' not in sys.modules:
    if 'pylab' not in sys.modules:
        import matplotlib
        matplotlib.use('Agg') # directly plot to a file when no GUI is available
                              # (e.g., remote running)
import matplotlib.pyplot as plt
import pickle
from skimage.measure import label

### import packages for GA algorithm
import array
import multiprocessing
import random
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from functools import partial


# input image dimensions
img_rows, img_cols = 28, 28


# function evaluating image-likeness of permuted 2D arrays based on the average number of connected regions
def connectivity(imgs, permutation):
    tmp = imgs.reshape(imgs.shape[0], np.prod(imgs.shape[1:]))
    imgs_permuted = tmp[:, permutation].reshape(imgs.shape)
    return sum([label(img, return_num=True)[1] for img in imgs_permuted])/imgs.shape[0],  # must return a sequence of numbers to be used as DEAP evaluation function


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
permutation = np.random.permutation(num_pixels)  # random permutation for 28*28 images
tmp = x_train.reshape(x_train.shape[0], num_pixels)
x_permuted = tmp[:, permutation].reshape(x_train.shape)

# sample original and permuted images
digit_idx = np.empty([10], dtype=int)
for i in range(10):
    digit_idx[i] = np.where(y_train == i)[0][0]
samples_original = x_train[digit_idx]
samples_permuted = x_permuted[digit_idx]

# obtain the binary version of permuted train images (reshaped without channel)
tmp = np.copy(x_permuted)
tmp[np.where(tmp > 0)] = 1
x_bp = tmp.astype(int).reshape(len(tmp), img_rows, img_cols)

### GA to find optimal permutation for image mapping
# n_iter = 100000
# min_obj = 1000
# inv_permutation = None
# for i in range(n_iter):
#     tmp_ip = np.random.permutation(num_pixels)
#     tmp_obj = connectivity(x_bp, tmp_ip)
#     print("Obj. value for iter={0:d}: {1:f}".format(i, tmp_obj))
#     if tmp_obj < min_obj:
#         min_obj = tmp_obj
#         inv_permutation = tmp_ip

# print("min. obj. value: {0:f}".format(min_obj))
# outfile = open('inv_permutation.pkl', 'wb')
# pickle.dump(inv_permutation, outfile)
# outfile.close()

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(num_pixels), num_pixels)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

evalImageMapping = partial(connectivity, x_bp)

# toolbox.register("mate", tools.cxOrdered)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalImageMapping)


def main():
    random.seed(64)
    
    pool = multiprocessing.Pool(processes=4)
    toolbox.register("map", pool.map)
    
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=100, stats=stats,
                        halloffame=hof)
    pool.close()
    return pop, logbook, hof


if __name__ == "__main__":
    pop, logbook, hof = main()
    outfile = open('ga-based_image_mapping.pkl', 'wb')
    pickle.dump(pop, outfile)
    pickle.dump(logbook, outfile)
    pickle.dump(hof, outfile)
    outfile.close()
