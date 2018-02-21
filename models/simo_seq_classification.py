#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     simo_seq_classification.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-02-15
#
# @brief A scalable indoor localization system (up to reference points) based on
#        Wi-Fi fingerprinting using a sequential multi-class classification of
#        building, floor, and location (reference point) with a single-input and
#        multi-output (SIMO) deep neural network (DNN) model.
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
from deep_autoencoder import deep_autoencoder
from ujiindoorloc import UJIIndoorLoc
### import other modules; keras and its backend will be loaded later
import argparse
import datetime
import math
import numpy as np
import pandas as pd
import pathlib
import random as rn
# from configparser import ConfigParser
from numpy.linalg import norm
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
from keras.callbacks import TensorBoard
from keras.layers import Activation, Dense, Dropout, Input
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
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
        "-F",
        "--frac",
        help=
        "fraction of input data to load for training and validation; default is 1.0",
        default=1.0,
        type=float)
    parser.add_argument(
        "--validation_split",
        help=
        "fraction of training data to be used as validation data: default is 0.2",
        default=0.2,
        type=float)
    parser.add_argument(
        "-P",
        "--preprocessor",
        help=
        "preprocessor to scale/normalize input data before training and validation; default is 'standard_scaler'",
        default='standard_scaler',
        type=str)
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
        "-N",
        "--neighbours",
        help=
        "number of (nearest) neighbour locations to consider in positioning; default is 1",
        default=1,
        type=int)
    parser.add_argument(
        "--scaling",
        help=
        "scaling factor for threshold (i.e., threshold=scaling*maximum) for the inclusion of nighbour locations to consider in positioning; default is 0.0",
        default=0.0,
        type=float)
    parser.add_argument(
        "--dae_hidden_layers",
        help=
        "comma-separated numbers of units in hidden layers for deep autoencoder; default is '256,128,256'",
        default='256,128,256',
        type=str)
    parser.add_argument(
        "--building_hidden_layers",
        help=
        "comma-separated numbers of units in hidden layers for building; default is '128,128'",
        default='128,128',
        type=str)
    parser.add_argument(
        "--floor_hidden_layers",
        help=
        "comma-separated numbers of units in additional hidden layers for floor; default is '128,128'",
        default='128,128',
        type=str)
    parser.add_argument(
        "--location_hidden_layers",
        help=
        "comma-separated numbers of units in additional hidden layers for location; default is '128,128'",
        default='128,128',
        type=str)
    parser.add_argument(
        "--building_weight",
        help="loss weight for building; default 1.0",
        default=1.0,
        type=float)
    parser.add_argument(
        "--floor_weight",
        help="loss weight for floor; default 1.0",
        default=1.0,
        type=float)
    parser.add_argument(
        "--location_weight",
        help="loss weight for location; default 1.0",
        default=1.0,
        type=float)
    parser.add_argument(
        "-V",
        "--verbose",
        help=
        "verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch; default is 1",
        default=1,
        type=int)
    args = parser.parse_args()

    # set variables using command-line arguments
    gpu_id = args.gpu_id
    random_seed = args.random_seed
    frac = args.frac
    validation_split = args.validation_split
    preprocessor = args.preprocessor
    batch_size = args.batch_size
    epochs = args.epochs
    optimizer = args.optimizer
    dropout = args.dropout
    N = args.neighbours
    scaling = args.scaling
    if args.dae_hidden_layers == '':
        dae_hidden_layers = ''
    else:
        dae_hidden_layers = [
            int(i) for i in (args.dae_hidden_layers).split(',')
        ]
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
    building_weight = args.building_weight
    floor_weight = args.floor_weight
    location_weight = args.location_weight
    verbose = args.verbose
    
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
    print("\nPart 1: loading UJIIndoorLoc data ...")

    if preprocessor == 'standard_scaler':
        from sklearn.preprocessing import StandardScaler
        rss_scaler = StandardScaler()
        utm_scaler = StandardScaler()
    elif preprocessor == 'minmax_scaler':
        from sklearn.preprocessing import MinMaxScaler
        rss_scaler = MinMaxScaler()
        utm_scaler = MinMaxScaler()
    elif preprocessor == 'normalizer':
        from sklearn.preprocessing import Normalizer
        rss_scaler = Normalizer()
        utm_scaler = Normalizer()
    else:
        rss_scaler = None
        utm_scaler = None

    ujiindoorloc = UJIIndoorLoc(
        data_path,
        frac=frac,
        rss_scaler=rss_scaler,
        utm_scaler=utm_scaler,
        classification_mode='hierarchical')
    training_df, training_data, testing_df, testing_data = ujiindoorloc.load_data(
    )

    ### build and train sequentially a SIMO model
    print(
        "\nPart 2: buidling and training a SIMO model for sequential classification ..."
    )
    rss = training_data.rss_scaled
    labels = training_data.labels
    input = Input(shape=(rss.shape[1], ), name='input')  # common input
    tensorboard = TensorBoard(
        log_dir="logs/{}".format(time()), write_graph=True)

    ### (optional) build deep autoencoder
    if dae_hidden_layers != '':
        print("\nPart 2.0: buidling a DAE model ...")
        model = deep_autoencoder(
            rss,
            hidden_layers=dae_hidden_layers,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split)
        x = model(input)
    else:
        x = input

    print("\nPart 2.1: buidling and training a building classifier ...")
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    for units in building_hidden_layers:
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)
    bld_hl_output = x

    x = Dense(labels.building.shape[1])(x)
    x = BatchNormalization()(x)
    bld_output = Activation(
        'softmax', name='building_output')(x)  # no dropout for an output layer

    bld_model = Model(inputs=input, outputs=bld_output)
    bld_model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    startTime = timer()
    bld_history = bld_model.fit(
        x=rss,
        y=labels.building,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_split=validation_split,
        # validation_data=({
        #     'input': testing_data.rss_scaled
        # }, {
        #     'building_output': testing_data.labels.building
        # }),
        callbacks=[tensorboard],
        shuffle=True)
    elapsedTime = timer() - startTime
    print("Building classifier trained in %e s." % elapsedTime)

    print("\nPart 2.2: buidling and training a building-floor classifier ...")
    x = bld_hl_output
    for units in floor_hidden_layers:
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)
    flr_hl_output = x

    x = Dense(labels.floor.shape[1])(x)
    x = BatchNormalization()(x)
    flr_output = Activation(
        'softmax', name='floor_output')(x)  # no dropout for an output layer

    bf_model = Model(inputs=input, outputs=[bld_output, flr_output])
    bf_model.compile(
        optimizer=optimizer,
        loss=['categorical_crossentropy', 'categorical_crossentropy'],
        loss_weights={
            'building_output': building_weight,
            'floor_output': floor_weight
        },
        metrics={
            'building_output': 'accuracy',
            'floor_output': 'accuracy'
        })

    startTime = timer()
    bf_history = bf_model.fit(
        x={'input': rss},
        y={'building_output': labels.building,
           'floor_output': labels.floor},
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_split=validation_split,
        # validation_data=({
        #     'input': testing_data.rss_scaled
        # }, {
        #     'building_output': testing_data.labels.building,
        #     'floor_output': testing_data.labels.floor
        # }),
        callbacks=[tensorboard],
        shuffle=True)
    elapsedTime = timer() - startTime
    print("Building-floor classifier trained in %e s." % elapsedTime)

    print(
        "\nPart 2.3: buidling and training a building-floor-location classifier ..."
    )
    x = flr_hl_output
    for units in location_hidden_layers:
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)

    x = Dense(labels.location.shape[1])(x)
    x = BatchNormalization()(x)
    loc_output = Activation(
        'softmax', name='location_output')(x)  # no dropout for an output layer

    bfl_model = Model(
        inputs=input, outputs=[bld_output, flr_output, loc_output])
    bfl_model.compile(
        optimizer=optimizer,
        loss=[
            'categorical_crossentropy', 'categorical_crossentropy',
            'categorical_crossentropy'
        ],
        loss_weights={
            'building_output': building_weight,
            'floor_output': floor_weight,
            'location_output': location_weight
        },
        metrics={
            'building_output': 'accuracy',
            'floor_output': 'accuracy',
            'location_output': 'accuracy'
        })

    startTime = timer()
    bfl_history = bfl_model.fit(
        x={'input': rss},
        y={
            'building_output': labels.building,
            'floor_output': labels.floor,
            'location_output': labels.location
        },
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_split=validation_split,
        # validation_data=({
        #     'input': testing_data.rss_scaled
        # }, {
        #     'building_output': testing_data.labels.building,
        #     'floor_output': testing_data.labels.floor,
        #     'location_output': testing_data.labels.location
        # }),
        callbacks=[tensorboard],
        shuffle=True)
    elapsedTime = timer() - startTime
    print("Building-floor-location classifier trained in %e s." % elapsedTime)

    ### evaluate the model
    print("\nPart 3: evaluating the model ...")
    rss = testing_data.rss_scaled
    labels = testing_data.labels
    blds = labels.building
    flrs = labels.floor
    utm = testing_data.utm  # original UTM coordinates

    # calculate the classification accuracies and localization errors
    preds = bfl_model.predict(rss, batch_size=batch_size)
    bld_results = (np.equal(
        np.argmax(blds, axis=1), np.argmax(preds[0], axis=1))).astype(int)
    bld_acc = bld_results.mean()
    flr_results = (np.equal(
        np.argmax(flrs, axis=1), np.argmax(preds[1], axis=1))).astype(int)
    flr_acc = flr_results.mean()
    bf_acc = (bld_results * flr_results).mean()

    # calculate positioning error
    locs = preds[2]
    n_samples = len(blds)
    n_locs = locs.shape[1]  # number of locations (reference points)
    # sum_pos_err = 0.0
    # sum_pos_err_weighted = 0.0
    idxs = np.argpartition(
        locs, -N)[:, -N:]  # (unsorted) indexes of up to N nearest neighbors
    threshold = scaling * np.amax(locs, axis=1)
    training_labels = np.concatenate(
        (training_data.labels.building, training_data.labels.floor,
         training_data.labels.location),
        axis=1)
    training_utm_avg = training_data.utm_avg
    utm_est = np.zeros((n_samples, 2))
    utm_est_weighted = np.zeros((n_samples, 2))
    for i in range(n_samples):
        xs = []
        ys = []
        ws = []
        for j in idxs[i]:
            if locs[i][j] >= threshold[i]:
                loc = np.zeros(n_locs)
                loc[j] = 1
                rows = np.where((training_labels == np.concatenate(
                    (blds[i], flrs[i],
                     loc))).all(axis=1))  # tuple of row indexes
                if rows[0].size > 0:
                    xs.append(training_df.loc[training_df.index[rows[0][0]],
                                              'LONGITUDE'])
                    ys.append(training_df.loc[training_df.index[rows[0][0]],
                                              'LATITUDE'])
                    ws.append(locs[i][j])
        if len(xs) > 0:
            utm_est[i] = np.array((xs, ys)).mean(axis=1)
            utm_est_weighted[i] = np.array((np.average(xs, weights=ws),
                                            np.average(ys, weights=ws)))
        else:
            if rows[0].size > 0:
                key = str(np.argmax(blds[i])) + '-' + str(np.argmax(flrs[i]))
            else:
                key = str(np.argmax(blds[i]))
            utm_est[i] = utm_est_weighted[i] = training_utm_avg[key]

    # calculate localization errors per EvAAL/IPIN 2015 competition
    dist = norm(utm - utm_est, axis=1)  # Euclidean distance
    dist_weighted = norm(utm - utm_est_weighted, axis=1)
    flr_diff = np.absolute(
        np.argmax(flrs, axis=1) - np.argmax(preds[1], axis=1))
    error = dist + 50 * (
        1 - bld_results) + 4 * flr_diff  # individual error [m]
    error_weighted = dist_weighted + 50 * (
        1 - bld_results) + 4 * flr_diff  # individual error [m]
    mean_error = error.mean()
    mean_error_weighted = error_weighted.mean()
    median_error = np.median(error)
    median_error_weighted = np.median(error_weighted)

    ### print out final results
    base_dir = '../results/test/' + (os.path.splitext(
        os.path.basename(__file__))[0]).replace('test_', '')
    pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)
    # base_file_name = base_dir + "/E{0:d}_B{1:d}_D{2:.2f}_H{3:s}".format(
    #     epochs, batch_size, dropout, args.hidden_layers.replace(',', '-'))
    base_file_name = base_dir + "/E{0:d}_B{1:d}_D{2:.2f}".format(
        epochs, batch_size, dropout)
    # + '_T' + "{0:.2f}".format(args.training_ratio) \
    # sae_model_file = base_file_name + '.hdf5'
    now = datetime.datetime.now()
    output_file_base = base_file_name + '_' + now.strftime("%Y%m%d-%H%M%S")

    with open(output_file_base + '.org', 'w') as output_file:
        output_file.write(
            "#+STARTUP: showall\n")  # unfold everything when opening
        output_file.write("* System parameters\n")
        output_file.write("  - GPU ID: %d\n" % gpu_id)
        output_file.write("  - Random number seed: %d\n" % random_seed)
        output_file.write("  - Fraction of data loaded for training and validation: %.2f\n" % frac)
        output_file.write("  - Validation split: %.2f\n" % validation_split)
        output_file.write("  - Preprocessor for scaling/normalizing input data: %s\n" % preprocessor)
        output_file.write("  - Batch size: %d\n" % batch_size)
        output_file.write("  - Epochs: %d\n" % epochs)
        output_file.write("  - Optimizer: %s\n" % optimizer)
        output_file.write("  - Dropout rate: %.2f\n" % dropout)
        output_file.write("  - Number of (nearest) neighbour locations: %d\n" % N)
        output_file.write("  - Scaling factor for threshold: %.2f\n" % scaling)
        output_file.write("  - Deep autoencoder hidden layers: ")
        if dae_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % dae_hidden_layers[0])
            for units in dae_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("  - Building hidden layers: ")
        if building_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % building_hidden_layers[0])
            for units in building_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("  - Floor hidden layers: ")
        if floor_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % floor_hidden_layers[0])
            for units in floor_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("  - Location hidden layers: ")
        if location_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % location_hidden_layers[0])
            for units in location_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("  - Building loss weight: %.2f\n" % building_weight)
        output_file.write("  - Floor loss weight: %.2f\n" % floor_weight)
        output_file.write("  - Location loss weight: %.2f\n" % location_weight)
        output_file.write("\n")
        output_file.write("* Performance\n")
        output_file.write(" - Building hit rate [%%]: %.2f\n" %
                          (100 * bld_acc))
        output_file.write(" - Floor hit rate [%%]: %.2f\n" % (100 * flr_acc))
        output_file.write(" - Building-floor hit rate [%%]: %.2f\n" %
                          (100 * bf_acc))
        output_file.write(
            "  - Mean error [m]: %.2f\n" %
            mean_error)  # according to EvAAL/IPIN 2015 competition rule
        output_file.write(
            "  - Mean error (weighted) [m]: %.2f\n" % mean_error_weighted)
        output_file.write("  - Median error [m]: %.2f\n" % median_error)
        output_file.write(
            "  - Median error (weighted) [m]: %.2f\n" % median_error_weighted)
        # output_file.write(
        #     "  - Location estimation failure rate (given the correct building/floor): %e\n"
        #     % loc_failure)
        # output_file.write(
        #     "  - Positioning error w/ B-F hit [m]: %e\n" % mean_pos_err)
        # output_file.write(
        #     "  - Positioning error w/ B-F hit and weighted [m]: %e\n" %
        #     mean_pos_err_weighted)
