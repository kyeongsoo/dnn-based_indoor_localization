#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     seq_classification.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-02-13
#
# @brief A scalable indoor localization system (up to reference points) based on
#        Wi-Fi fingerprinting using a sequential multi-class classification of
#        building, floor, and location (reference point) with multiple
#        single-input and single-output (SISO) deep neural network (DNN) models
#
# @remarks The results will be published in a paper submitted to the <a
#          href="http://www.sciencedirect.com/science/journal/08936080">Elsevier
#          Neural Networks</a> journal.

### import basic modules and a model to test
import os
os.environ['PYTHONHASHSEED'] = '0'  # for reproducibility
import platform
if platform.system() == 'Windows':
    data_path = os.path.expanduser(
        '~kks/Research/Ongoing/localization/xjtlu_surf_indoor_localization/data/UJIIndoorLoc'
    )
    models_path = os.path.expanduser(
        '~kks/Research/Ongoing/localization/elsevier_nn_scalable_indoor_localization/program/models'
    )
    utils_path = os.path.expanduser(
        '~kks/Research/Ongoing/localization/elsevier_nn_scalable_indoor_localization/program/utils'
    )
else:
    data_path = os.path.expanduser(
        '~kks/research/ongoing/localization/xjtlu_surf_indoor_localization/data/UJIIndoorLoc'
    )
    models_path = os.path.expanduser(
        '~kks/research/ongoing/localization/elsevier_nn_scalable_indoor_localization/program/models'
    )
    utils_path = os.path.expanduser(
        '~kks/research/ongoing/localization/elsevier_nn_scalable_indoor_localization/program/utils'
    )
import sys
sys.path.insert(0, utils_path)
sys.path.insert(0, models_path)
from siso_classifier import siso_classifier
from ujiindoorloc import UJIIndoorLoc
### import other modules; keras and its backend will be loaded later
import argparse
import datetime
# import math
import numpy as np
import pandas as pd
import pathlib
import random as rn
# from configparser import ConfigParser
# from numpy.linalg import norm
from time import time
from timeit import default_timer as timer
### import keras and its backend (e.g., tensorflow)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # supress warning messages
import tensorflow as tf
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
)  # force TF to use single thread for reproducibility
from keras import backend as K
import keras
# from keras.callbacks import TensorBoard
from keras.models import Model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-G",
        "--gpu_id",
        help=
        "ID of GPU device to run this script; default is 0; set it to a negative number for CPU (i.e., no GPU)",
        default=0,
        type=int)
    parser.add_argument(
        "-R", "--random_seed", help="random seed", default=0, type=int)
    parser.add_argument(
        "-B",
        "--batch_size",
        help="batch size; default is 32",
        default=32,
        type=int)
    parser.add_argument(
        "-E",
        "--epochs",
        help="number of epochs; default is 50",
        default=50,
        type=int)
    parser.add_argument(
        "--validation_split",
        help=
        "fraction of training data to be used as validation data: default is 0.2",
        default=0.2,
        type=float)
    parser.add_argument(
        "--building_hidden_layers",
        help=
        "comma-separated numbers of units in hidden layers for buildings; default is '128,128'",
        default='128,128',
        type=str)
    parser.add_argument(
        "--floor_hidden_layers",
        help=
        "comma-separated numbers of units in additional hidden layers for floors; default is '128,128'",
        default='128,128',
        type=str)
    parser.add_argument(
        "--location_hidden_layers",
        help=
        "comma-separated numbers of units in additional hidden layers for locations; default is '128,128'",
        default='128,128',
        type=str)
    parser.add_argument(
        "-O",
        "--optimizer",
        help="optimizer; default is 'nadam'",
        default='nadam',
        type=str)
    parser.add_argument(
        "-D",
        "--dropout",
        help="dropout rate before and after hidden layers; default is 0.0",
        default=0.0,
        type=float)
    parser.add_argument(
        "-F",
        "--frac",
        help=
        "fraction of input data to load for training and validation; default is 1.0",
        default=1.0,
        type=float)
    parser.add_argument(
        "-V",
        "--verbose",
        help=
        "verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch; default is 1",
        default=1,
        type=int)
    # parser.add_argument(
    #     "-N",
    #     "--neighbours",
    #     help="number of (nearest) neighbour locations to consider in positioning; default is 1",
    #     default=1,
    #     type=int)
    # parser.add_argument(
    #     "--scaling",
    #     help=
    #     "scaling factor for threshold (i.e., threshold=scaling*maximum) for the inclusion of nighbour locations to consider in positioning; default is 0.0",
    #     default=0.0,
    #     type=float)
    args = parser.parse_args()

    # set variables using command-line arguments
    gpu_id = args.gpu_id
    random_seed = args.random_seed
    batch_size = args.batch_size
    epochs = args.epochs
    validation_split = args.validation_split
    # sae_hidden_layers = [int(i) for i in (args.sae_hidden_layers).split(',')]
    if args.building_hidden_layers == '':
        building_hidden_layers = ''
    else:
        building_hidden_layers = [
            int(i) for i in (args.building_hidden_layers).split(',')
        ]
    if args.floor_hidden_layers == '':
        floor_hidden_layers = ''
    else:
        floor_hidden_layers = [
            int(i) for i in (args.floor_hidden_layers).split(',')
        ]
    if args.location_hidden_layers == '':
        location_hidden_layers = ''
    else:
        location_hidden_layers = [
            int(i) for i in (args.location_hidden_layers).split(',')
        ]
    optimizer = args.optimizer
    dropout = args.dropout
    frac = args.frac
    verbose = args.verbose
    # N = args.neighbours
    # scaling = args.scaling

    ### initialize numpy, random, TensorFlow, and keras
    np.random.seed(random_seed)
    rn.seed(random_seed)
    tf.set_random_seed(random_seed)
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    sess = tf.Session(
        graph=tf.get_default_graph(),
        config=session_conf)  # for reproducibility
    K.set_session(sess)

    ### load dataset after scaling
    ujiindoorloc = UJIIndoorLoc(
        data_path, frac=frac, scale=True, classification_mode='hierarchical')
    rss, labels = ujiindoorloc.load_data()

    # create, train, and evaluate a model for multi-class classification of a building
    bld_model = siso_classifier(
        input_dim=rss.shape[1],
        input_name='bld_input',
        output_dim=labels.building.shape[1],
        output_name='bld_output',
        base_model=None,
        hidden_layers=building_hidden_layers,
        optimizer=optimizer,
        dropout=dropout)
    startTime = timer()
    bld_history = bld_model.fit(
        x=rss,
        y=labels.building,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_split=validation_split,
        shuffle=True)
    elapsedTime = timer() - startTime
    print("Building classifier trained in %e s." % elapsedTime)

    # create a model for multi-class classification of a floor based on the building model
    # cloned_model = keras.models.clone_model(bld_model)
    # num_to_remove = 6  # remove the output and relevant layers: 1*Dropout layers, 2*BatchNormalization, 2*Activation, 1*Dense,
    # for i in range(num_to_remove):
    #     cloned_model.layers.pop()
    #     cloned_model.outputs = [cloned_model.layers[-1].output]
    #     cloned_model.layers[-1].outbound_nodes = []
    pruned_model = Model(bld_model.inputs,
                         bld_model.layers[-7].output)  # prune last 6 layers

    flr_model = siso_classifier(
        input_dim=rss.shape[1],
        input_name='flr_input',
        output_dim=labels.floor.shape[1],
        output_name='flr_output',
        # base_model=cloned_model,
        base_model=pruned_model,
        hidden_layers=floor_hidden_layers,
        optimizer=optimizer,
        dropout=dropout)
    startTime = timer()
    flr_model.fit(
        x=rss,
        y=labels.floor,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_split=validation_split,
        shuffle=True)
    elapsedTime = timer() - startTime
    print("Floor classifier trained in %e s." % elapsedTime)

    # create a model for multi-class classification of a location based on the floor model
    # cloned_model = keras.models.clone_model(flr_model)
    # num_to_remove = 6  # remove the output and relevant layers: 1*Dropout layers, 2*BatchNormalization, 2*Activation, 1*Dense,
    # for i in range(num_to_remove):
    #     cloned_model.layers.pop()
    #     cloned_model.outputs = [cloned_model.layers[-1].output]
    #     cloned_model.layers[-1].outbound_nodes = []
    pruned_model = Model(flr_model.inputs,
                         flr_model.layers[-7].output)  # prune last 6 layers

    loc_model = siso_classifier(
        input_dim=rss.shape[1],
        input_name='loc_input',
        output_dim=labels.location.shape[1],
        output_name='loc_output',
        # base_model=cloned_model,
        base_model=pruned_model,
        hidden_layers=location_hidden_layers,
        optimizer=optimizer,
        dropout=dropout)
    loc_model.fit(
        x=rss,
        y=labels.location,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_split=validation_split,
        shuffle=True)
    elapsedTime = timer() - startTime
    print("Location classifier trained in %e s." % elapsedTime)

    # # turn the given validation set into a testing set
    # # testing_df = pd.read_csv(validation_data_file, header=0)
    # test_AP_features = scale(np.asarray(testing_df.iloc[:,0:520]).astype(float), axis=1) # convert integer to float and scale jointly (axis=1)
    # x_test_utm = np.asarray(testing_df['LONGITUDE'])
    # y_test_utm = np.asarray(testing_df['LATITUDE'])
    # # blds = np.asarray(pd.get_dummies(testing_df['BUILDINGID']))
    # blds = blds_all[len_train:]
    # # flrs = np.asarray(pd.get_dummies(testing_df['FLOOR']))
    # flrs = flrs_all[len_train:]

    # ### evaluate the model
    # print("\nPart 3: evaluating the model ...")

    # # calculate the accuracy of building and floor estimation
    # preds = model.predict(test_AP_features, batch_size=batch_size)
    # n_preds = preds.shape[0]
    # # blds_results = (np.equal(np.argmax(test_labels[:, :3], axis=1), np.argmax(preds[:, :3], axis=1))).astype(int)
    # blds_results = (np.equal(np.argmax(blds, axis=1), np.argmax(preds[:, :3], axis=1))).astype(int)
    # acc_bld = blds_results.mean()
    # # flrs_results = (np.equal(np.argmax(test_labels[:, 3:8], axis=1), np.argmax(preds[:, 3:8], axis=1))).astype(int)
    # flrs_results = (np.equal(np.argmax(flrs, axis=1), np.argmax(preds[:, 3:8], axis=1))).astype(int)
    # acc_flr = flrs_results.mean()
    # acc_bf = (blds_results*flrs_results).mean()
    # # locs_results = (np.equal(np.argmax(test_labels[:, 8:118], axis=1), np.argmax(preds[:, 8:118], axis=1))).astype(int)
    # # acc_loc = locs_results.mean()
    # # acc = (blds_results*flrs_results*locs_results).mean()

    # # calculate positioning error when building and floor are correctly estimated
    # mask = np.logical_and(blds_results, flrs_results) # mask index array for correct location of building and floor
    # x_test_utm = x_test_utm[mask]
    # y_test_utm = y_test_utm[mask]
    # blds = blds[mask]
    # flrs = flrs[mask]
    # locs = (preds[mask])[:, 8:118]

    # n_success = len(blds)       # number of correct building and floor location
    # # blds = np.greater_equal(blds, np.tile(np.amax(blds, axis=1).reshape(n_success, 1), (1, 3))).astype(int) # set maximum column to 1 and others to 0 (row-wise)
    # # flrs = np.greater_equal(flrs, np.tile(np.amax(flrs, axis=1).reshape(n_success, 1), (1, 5))).astype(int) # ditto

    # n_loc_failure = 0
    # sum_pos_err = 0.0
    # sum_pos_err_weighted = 0.0
    # idxs = np.argpartition(locs, -N)[:, -N:]  # (unsorted) indexes of up to N nearest neighbors
    # threshold = scaling*np.amax(locs, axis=1)
    # for i in range(n_success):
    #     xs = []
    #     ys = []
    #     ws = []
    #     for j in idxs[i]:
    #         loc = np.zeros(110)
    #         loc[j] = 1
    #         rows = np.where((train_labels == np.concatenate((blds[i], flrs[i], loc))).all(axis=1)) # tuple of row indexes
    #         if rows[0].size > 0:
    #             if locs[i][j] >= threshold[i]:
    #                 xs.append(training_df.loc[training_df.index[rows[0][0]], 'LONGITUDE'])
    #                 ys.append(training_df.loc[training_df.index[rows[0][0]], 'LATITUDE'])
    #                 ws.append(locs[i][j])
    #     if len(xs) > 0:
    #         sum_pos_err += math.sqrt((np.mean(xs)-x_test_utm[i])**2 + (np.mean(ys)-y_test_utm[i])**2)
    #         sum_pos_err_weighted += math.sqrt((np.average(xs, weights=ws)-x_test_utm[i])**2 + (np.average(ys, weights=ws)-y_test_utm[i])**2)
    #     else:
    #         n_loc_failure += 1
    #         key = str(np.argmax(blds[i])) + '-' + str(np.argmax(flrs[i]))
    #         pos_err = math.sqrt((x_avg[key]-x_test_utm[i])**2 + (y_avg[key]-y_test_utm[i])**2)
    #         sum_pos_err += pos_err
    #         sum_pos_err_weighted += pos_err
    # # mean_pos_err = sum_pos_err / (n_success - n_loc_failure)
    # mean_pos_err = sum_pos_err / n_success
    # # mean_pos_err_weighted = sum_pos_err_weighted / (n_success - n_loc_failure)
    # mean_pos_err_weighted = sum_pos_err_weighted / n_success
    # loc_failure = n_loc_failure / n_success # rate of location estimation failure given that building and floor are correctly located

    ### print out final results
    base_dir = '../results/test/' + (os.path.splitext(
        os.path.basename(__file__))[0]).replace('test_', '')
    pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)
    base_file_name = base_dir + "/E{0:d}_B{1:d}_D{2:.2f}_H{3:s}".format(
        epochs, batch_size, dropout, args.hidden_layers.replace(',', '-'))
    # + '_T' + "{0:.2f}".format(args.training_ratio) \
    # sae_model_file = base_file_name + '.hdf5'
    now = datetime.datetime.now()
    output_file_base = base_file_name + '_' + now.strftime("%Y%m%d-%H%M%S")

    with open(output_file_base + '.org', 'w') as output_file:
        output_file.write(
            "#+STARTUP: showall\n")  # unfold everything when opening
        output_file.write("* System parameters\n")
        output_file.write("  - Optimizer: %s\n" % optimizer)
        output_file.write("  - Random number seed: %d\n" % random_seed)
        # output_file.write("  - Ratio of training data to overall data: %.2f\n" % training_ratio)
        output_file.write("  - Epochs: %d\n" % epochs)
        output_file.write("  - Batch size: %d\n" % batch_size)
        output_file.write("  - Dropout rate: %.2f\n" % dropout)
        output_file.write("  - Hidden layers: %d" % hidden_layers[0])
        for units in hidden_layers[1:]:
            output_file.write("-%d" % units)
        output_file.write("\n")
        output_file.write("* Performance\n")
        output_file.write("  - Accuracy: {0:.2f}% ({1:.2f}%)".format(
            (100 * results.mean()), (100 * results.std())))
        # output_file.write("  - Loss (overall): %e\n" % results.losses.overall)
        # output_file.write("  - Accuracy (overall): %e\n" % results.accuracy.overall)
        # output_file.write("  - Building hit rate [%%]: %.2f\n" % (100*results.metrics.building_acc))
        # output_file.write("  - Floor hit rate [%%]: %.2f\n" % (100*results.metrics.floor_acc))
        # output_file.write("  - Building-floor hit rate [%%]: %.2f\n" % (100*results.metrics.bf_acc))
        # output_file.write("  - MSE (location): %e\n" % results.metrics.location_mse)
        # output_file.write("  - Mean error [m]: %.2f\n" % results.metrics.mean_error)  # according to EvAAL/IPIN 2015 competition rule
        # output_file.write("  - Median error [m]: %.2f\n" % results.metrics.median_error)  # ditto
