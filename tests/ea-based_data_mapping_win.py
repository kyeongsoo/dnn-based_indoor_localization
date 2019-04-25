#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     ea-based_data_mapping.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-07-20
#
# @brief    Prototype evolutionary algorithm (EA)-based mapping of unstructured
#           data to 2-D images.
#
# @remarks


### import modules
import os
import sys
# to directly plot to a file when no GUI is available (e.g., remote running)
if 'matplotlib.pyplot' not in sys.modules:
    if 'pylab' not in sys.modules:
        import matplotlib
        matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
import array
import gzip
import mnist
import multiprocessing
import numpy as np
import pickle
import random
import time
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from functools import partial
from skimage.measure import label


# to parse predownloaded MNIST data
def parse_mnist_file(fname):
    fopen = gzip.open if os.path.splitext(fname)[1] == '.gz' else open
    with fopen(fname, 'rb') as fd:
        return mnist.parse_idx(fd)


# for execution time measurement
t_start = time.time()

# parse argument parameters first
parser = argparse.ArgumentParser()
parser.add_argument(
    "-G",
    "--ngen",
    help=
    "the number of generations; default is 100",
    default=100,
    type=int)
parser.add_argument(
    "-P",
    "--processes",
    help=
    "the number of processes for multiprocessing; default is 4",
    default=4,
    type=int)
args = parser.parse_args()

ngen = args.ngen
processes = args.processes


# load and preprocess the MNIST train dataset
mnist_data_dir = '../data/mnist/'
x_train = parse_mnist_file(mnist_data_dir + 'train-images-idx3-ubyte.gz')
y_train = parse_mnist_file(mnist_data_dir + 'train-labels-idx1-ubyte.gz')
# x_train = mnist.train_images()
# y_train = mnist.train_labels()
x_train = x_train.astype('float32')
x_train /= 255
num_samples = x_train.shape[0]
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
num_pixels = img_rows*img_cols

# permute train images
permutation = np.random.permutation(num_pixels)  # random permutation for 28*28 images
tmp = x_train.reshape(x_train.shape[0], num_pixels)
x_permuted = tmp[:, permutation].reshape(x_train.shape)

# obtain the binary version of permuted train images (reshaped without channel)
tmp = np.copy(x_permuted)
tmp[np.where(tmp > 0)] = 1
x_bp = tmp.astype(int).reshape(num_samples, img_rows, img_cols)

### EA to find optimal permutation for image mapping
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode='i', fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(num_pixels), num_pixels)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# evaluating image-likeness of permuted (i.e., of 'individual') 2D arrays (i.e.,
# x_bp) based on the average number of connected regions
def evalImageMapping(individual):
    tmp = x_bp.reshape(x_bp.shape[0], num_pixels)
    x_bp_permuted = tmp[:, individual].reshape(x_bp.shape)
    return sum([label(img, return_num=True)[1] for img in x_bp_permuted])/x_bp.shape[0],  # must return a sequence of numbers to be used as DEAP evaluation function

# toolbox.register("mate", tools.cxOrdered)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evalImageMapping)


if __name__ == "__main__":
    random.seed(64)
    pool = multiprocessing.Pool(processes=processes)
    toolbox.register("map", pool.map)
    
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=ngen, stats=stats,
                        halloffame=hof)
    pool.close()
    
    cp = dict(permutation=permutation, population=pop, halloffame=hof,
              logbook=logbook)
    with open("ea-based_data_mapping.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)
    cp_file.close()

    print("Permutation:")
    print(permutation)
    print("Hall of fame (mapping back to images):")
    print(hof)
    
    # for execution time measurement
    t_end = time.time()
    print("Elapsed time: {0:8.2e} second".format(t_end - t_start))
