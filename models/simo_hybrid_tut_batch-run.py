#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
sys.path.insert(0, '../models')
sys.path.insert(0, '../utils')
from simo_hybrid_tut import simo_hybrid_tut
from mean_ci import mean_ci
import argparse
import datetime
import numpy as np
from num2words import num2words

# set coordinates loss weight using command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '-C',
    '--coordinates_weight',
    help='loss weight for a coordinates; default 1.0',
    default=1.0,
    type=float)
args = parser.parse_args()
coordinates_weight = args.coordinates_weight

# set system parameters
num_runs = 20
# num_runs = 2                    # for test

# set default parameters for simo_hybrid_tut()
gpu_id = 0
dataset = 'tut'
frac = 1.0
validation_split = 0.2
preprocessor = 'standard_scaler'
batch_size = 64
epochs = 100
optimizer = 'nadam'
dropout = 0.25
corruption_level = 0.1
dae_hidden_layers = ''
sdae_hidden_layers = [1024, 1024, 1024]
cache = True
common_hidden_layers = [1024]
floor_hidden_layers = [256]
coordinates_hidden_layers = [256]
floor_weight = 1.0
verbose = 0

# inialize results arrays
flr_accs = np.empty(num_runs)
mean_error_2ds = np.empty(num_runs)
median_error_2ds = np.empty(num_runs)
mean_error_3ds = np.empty(num_runs)
median_error_3ds = np.empty(num_runs)
elapsedTimes = np.empty(num_runs)

# run experiments
for i in range(num_runs):
    print("\n########## Coordinates loss weight={0:.2f}: {1:s} run ##########".format(coordinates_weight, num2words(i+1, to='ordinal_num')))
    rst = simo_hybrid_tut(gpu_id, dataset, frac, validation_split,
                          preprocessor, batch_size, epochs, optimizer,
                          dropout, corruption_level, dae_hidden_layers,
                          sdae_hidden_layers, cache, common_hidden_layers,
                          floor_hidden_layers, coordinates_hidden_layers,
                          floor_weight, coordinates_weight, verbose)
    flr_accs[i] = rst.flr_acc
    mean_error_2ds[i] = rst.mean_error_2d
    median_error_2ds[i] = rst.median_error_2d
    mean_error_3ds[i] = rst.mean_error_3d
    median_error_3ds[i] = rst.median_error_3d
    elapsedTimes[i] = rst.elapsedTime

# print out results
base_file_name = '../results/test/simo_hybrid_tut/tut/cw{0:.1f}_'.format(coordinates_weight)

with open(base_file_name + 'floor_accuracy.csv', 'a') as output_file:
    output_file.write("{0:.2f},{1:.4f},{2:.4f},{3:.4f},{4:.4f}\n".format(coordinates_weight, *[i*100 for i in mean_ci(flr_accs)], 100*flr_accs.max(), 100*flr_accs.min()))

with open(base_file_name + 'mean_error_2d.csv', 'a') as output_file:
    output_file.write("{0:.2f},{1:.4f},{2:.4f},{3:.4f},{4:.4f}\n".format(coordinates_weight, *mean_ci(mean_error_2ds), mean_error_2ds.max(), mean_error_2ds.min()))

with open(base_file_name + 'mean_error_3d.csv', 'a') as output_file:
    output_file.write("{0:.2f},{1:.4f},{2:.4f},{3:.4f},{4:.4f}\n".format(coordinates_weight, *mean_ci(mean_error_3ds), mean_error_3ds.max(), mean_error_3ds.min()))
