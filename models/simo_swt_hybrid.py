#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     simo_swt_hybrid.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-02-21
#
# @brief A scalable indoor localization system (up to reference points) based on
#        Wi-Fi fingerprinting using a stage-wise-trained multi-class
#        classification of building and floor and regression of location
#        (reference point) coordiates with a single-input and multi-output
#        (SIMO) deep neural network (DNN) model.
#
# @remarks The results will be published in a paper submitted to the <a
#          href="http://www.sciencedirect.com/science/journal/08936080">Elsevier
#          Neural Networks</a> journal.

### import basic modules and a model to test
import os
os.environ['PYTHONHASHSEED'] = '0'  # for reproducibility
import sys
sys.path.insert(0, '../models')
sys.path.insert(0, '../utils')
from deep_autoencoder import deep_autoencoder
from sdae import sdae
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
# from keras.callbacks import TensorBoard
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
        "-R",
        "--random_seed",
        help="random seed",
        default=0,
        type=int)
    parser.add_argument(
        "--dataset",
        help="a data set for training, validation, and testing; choices are 'uji' (default) and 'tut'",
        default='uji',
        type=str)
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
        help="dropout rate before and after hidden layers; default is 0.2",
        default=0.2,
        type=float)
    parser.add_argument(
        "-C",
        "--corruption_level",
        help=
        "corruption level of masking noise for stacked denoising autoencoder; default is 0.1",
        default=0.1,
        type=float)
    parser.add_argument(
        "-N",
        "--neighbours",
        help=
        "number of (nearest) neighbour locations to consider in positioning; default is 8",
        default=8,
        type=int)
    parser.add_argument(
        "--scaling",
        help=
        "scaling factor for threshold (i.e., threshold=scaling*maximum) for the inclusion of nighbour locations to consider in positioning; default is 0.2",
        default=0.2,
        type=float)
    parser.add_argument(
        "--dae_hidden_layers",
        help=
        "comma-separated numbers of units in hidden layers for deep autoencoder; default is ''",
        default='',
        type=str)
    parser.add_argument(
        "--sdae_hidden_layers",
        help=
        "comma-separated numbers of units in hidden layers for stacked denoising autoencoder; default is '1024,1024,1024'",
        default='1024,1024,1024',
        type=str)
    parser.add_argument(
        "--common_hidden_layers",
        help=
        "comma-separated numbers of units in common hidden layers; default is '1024,1024'",
        default='1024,1024',
        type=str)
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
    dataset = args.dataset
    frac = args.frac
    validation_split = args.validation_split
    preprocessor = args.preprocessor
    batch_size = args.batch_size
    epochs = args.epochs
    optimizer = args.optimizer
    dropout = args.dropout
    corruption_level = args.corruption_level
    N = args.neighbours
    scaling = args.scaling
    if args.dae_hidden_layers == '':
        dae_hidden_layers = ''
    else:
        dae_hidden_layers = [
            int(i) for i in (args.dae_hidden_layers).split(',')
        ]
    if args.sdae_hidden_layers == '':
        sdae_hidden_layers = ''
    else:
        sdae_hidden_layers = [
            int(i) for i in (args.sdae_hidden_layers).split(',')
        ]
    if args.common_hidden_layers == '':
        common_hidden_layers = ''
    else:
        common_hidden_layers = [
            int(i) for i in (args.common_hidden_layers).split(',')
        ]
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

    ### load datasets after scaling
    print("\nPart 1: loading data ...")

    if dataset == 'uji':
        from ujiindoorloc import UJIIndoorLoc
        ujiindoorloc = UJIIndoorLoc(
            '../data/ujiindoorloc',
            frac=frac,
            preprocessor=preprocessor,
            classification_mode='hierarchical')
        training_df, training_data, testing_df, testing_data = ujiindoorloc.load_data(
        )
    elif dataset == 'tut':
        from tut import TUT
        tut = TUT(
            path='../data/tut',
            frac=frac,
            preprocessor=preprocessor,
            classification_mode='hierarchical')
        training_df = tut.training_df
        training_data = tut.training_data
        testing_df = tut.testing_df
        testing_data = tut.testing_data
    elif dataset == 'tut2':
        from tut import TUT2
        tut2 = TUT2(
            path='../data/tut',
            frac=frac,
            preprocessor=preprocessor,
            classification_mode='hierarchical',
            testing_split=0.2)
        training_df = tut2.training_df
        training_data = tut2.training_data
        testing_df = tut2.testing_df
        testing_data = tut2.testing_data
    else:
        print("'{0}' is not a supported data set.".format(dataset))
        sys.exit(0)
        
    ### build and do stage-wise training of a SIMO model
    print(
        "\nPart 2: building and stage-wise training a SIMO model for hybrid classification and regression ..."
    )
    rss = training_data.rss_scaled
    coord = training_data.coord_scaled
    coord_scaler = training_data.coord_scaler  # for inverse transform
    labels = training_data.labels
    input = Input(shape=(rss.shape[1], ), name='input')  # common input
    # tensorboard = TensorBoard(
    #     log_dir="logs/{}".format(time()), write_graph=True)

    # (optional) build deep autoencoder or stacked denoising autoencoder
    if dae_hidden_layers != '':
        print("\nPart 2.0: building a DAE model ...")
        model = deep_autoencoder(
            input_data=rss,
            preprocessor=preprocessor,
            hidden_layers=dae_hidden_layers,
            model_fname=None,
            optimizer=optimizer,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split)
        x = model(input)
    elif sdae_hidden_layers != '':
        print("\nPart 2.0: building an SDAE model ...")
        model = sdae(
            input_data=rss,
            preprocessor=preprocessor,
            hidden_layers=sdae_hidden_layers,
            model_fname=None,
            optimizer=optimizer,
            corruption_level=corruption_level,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split)
        x = model(input)
    else:
        x = input

    # common hidden layers
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)
    if common_hidden_layers != '':
        for units in common_hidden_layers:
            x = Dense(units)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(dropout)(x)
    common_hl_output = x

    # building classification output
    x = Dense(labels.building.shape[1])(common_hl_output)
    x = BatchNormalization()(x)
    building_output = Activation(
        'softmax', name='building_output')(x)  # no dropout for an output layer

    # floor classification output
    x = Dense(labels.floor.shape[1])(common_hl_output)
    x = BatchNormalization()(x)
    floor_output = Activation(
        'softmax', name='floor_output')(x)  # no dropout for an output layer

    # location classification output
    x = Dense(labels.location.shape[1])(common_hl_output)
    x = BatchNormalization()(x)
    location_output = Activation(
        'softmax', name='location_output')(x)  # no dropout for an output layer

    # coordinates regression output
    x = Dense(coord.shape[1], kernel_initializer='normal')(common_hl_output)
    x = BatchNormalization()(x)
    coordinates_output = Activation(
        'linear', name='coordinates_output')(x)  # 'linear' activation

    # build model
    model = Model(
        inputs=input,
        outputs=[
            building_output, floor_output, location_output, coordinates_output
        ])

    print("\nPart 2.1: stage-wise training with building information ...")

    # to handle single-building datasets (e.g., TUT)
    if labels.building.shape[1] > 1:
        building_loss = 'categorical_crossentropy'
    else:
        building_loss = 'sparse_categorical_crossentropy'
        
    model.compile(
        optimizer=optimizer,
        loss=[
            building_loss, 'categorical_crossentropy',
            'categorical_crossentropy', 'mean_squared_error'
        ],
        loss_weights={
            'building_output': 1.0,
            'floor_output': 0.0,
            'location_output': 0.0,
            'coordinates_output': 0.0
        },
        metrics={
            'building_output': 'accuracy',
            'floor_output': 'accuracy',
            'location_output': 'accuracy',
            'coordinates_output': 'mean_squared_error'
        })

    startTime = timer()
    b_history = model.fit(
        x={'input': rss},
        y={
            'building_output': labels.building,
            'floor_output': labels.floor,
            'location_output': labels.location,
            'coordinates_output': coord
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
        # callbacks=[tensorboard],
        shuffle=True)
    elapsedTime = timer() - startTime
    print("Stage-wise training with building information completed in %e s." %
          elapsedTime)

    print(
        "\nPart 2.2: stage-wise training with building-floor information ...")
    model.compile(
        optimizer=optimizer,
        loss=[
            building_loss, 'categorical_crossentropy',
            'categorical_crossentropy', 'mean_squared_error'
        ],
        loss_weights={
            'building_output': 1.0,
            'floor_output': 1.0,
            'location_output': 0.0,
            'coordinates_output': 0.0
        },
        metrics={
            'building_output': 'accuracy',
            'floor_output': 'accuracy',
            'location_output': 'accuracy',
            'coordinates_output': 'mean_squared_error'
        })

    startTime = timer()
    bf_history = model.fit(
        x={'input': rss},
        y={
            'building_output': labels.building,
            'floor_output': labels.floor,
            'location_output': labels.location,
            'coordinates_output': coord
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
        # callbacks=[tensorboard],
        shuffle=True)
    elapsedTime = timer() - startTime
    print(
        "Stage-wise training with building-floor information completed in %e s."
        % elapsedTime)

    print(
        "\nPart 2.3: stage-wise training with building-floor-location information ..."
    )
    model.compile(
        optimizer=optimizer,
        loss=[
            building_loss, 'categorical_crossentropy',
            'categorical_crossentropy', 'mean_squared_error'
        ],
        loss_weights={
            'building_output': 1.0,
            'floor_output': 1.0,
            'location_output': 1.0,
            'coordinates_output': 0.0
        },
        metrics={
            'building_output': 'accuracy',
            'floor_output': 'accuracy',
            'location_output': 'accuracy',
            'coordinates_output': 'mean_squared_error'
        })

    startTime = timer()
    bfl_history = model.fit(
        x={'input': rss},
        y={
            'building_output': labels.building,
            'floor_output': labels.floor,
            'location_output': labels.location,
            'coordinates_output': coord
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
        # callbacks=[tensorboard],
        shuffle=True)
    elapsedTime = timer() - startTime
    print(
        "Stage-wise training with building-floor-location information completed in %e s."
        % elapsedTime)

    print(
        "\nPart 2.4: stage-wise training with building-floor-location-coordinates information ..."
    )
    model.compile(
        optimizer=optimizer,
        loss=[
            building_loss, 'categorical_crossentropy',
            'categorical_crossentropy', 'mean_squared_error'
        ],
        loss_weights={
            'building_output': 1.0,
            'floor_output': 1.0,
            'location_output': 1.0,
            'coordinates_output': 1.0
        },
        metrics={
            'building_output': 'accuracy',
            'floor_output': 'accuracy',
            'location_output': 'accuracy',
            'coordinates_output': 'mean_squared_error'
        })

    startTime = timer()
    bflc_history = model.fit(
        x={'input': rss},
        y={
            'building_output': labels.building,
            'floor_output': labels.floor,
            'location_output': labels.location,
            'coordinates_output': coord
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
        # callbacks=[tensorboard],
        shuffle=True)
    elapsedTime = timer() - startTime
    print(
        "Stage-wise training with building-floor-location-coordinates information completed in %e s."
        % elapsedTime)

    ### evaluate the model
    print("\nPart 3: evaluating the model ...")
    rss = testing_data.rss_scaled
    labels = testing_data.labels
    blds = labels.building
    flrs = labels.floor
    coord = testing_data.coord  # original coordinates
    if dataset == 'uji':
        x_col_name = 'LONGITUDE'
        y_col_name = 'LATITUDE'
    else:                       # including 'tut'
        x_col_name = 'X'
        y_col_name = 'Y'

    # calculate the classification accuracies and localization errors
    preds = model.predict(rss, batch_size=batch_size)
    bld_results = (np.equal(
        np.argmax(blds, axis=1), np.argmax(preds[0], axis=1))).astype(int)
    bld_acc = bld_results.mean()
    flr_results = (np.equal(
        np.argmax(flrs, axis=1), np.argmax(preds[1], axis=1))).astype(int)
    flr_acc = flr_results.mean()
    bf_acc = (bld_results * flr_results).mean()

    # calculate positioning error based on locations
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
    training_coord_avg = training_data.coord_avg
    coord_est = np.zeros((n_samples, 2))
    coord_est_weighted = np.zeros((n_samples, 2))
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
                                              x_col_name])
                    ys.append(training_df.loc[training_df.index[rows[0][0]],
                                              y_col_name])
                    ws.append(locs[i][j])
        if len(xs) > 0:
            coord_est[i] = np.array((xs, ys)).mean(axis=1)
            coord_est_weighted[i] = np.array((np.average(xs, weights=ws),
                                            np.average(ys, weights=ws)))
        else:
            if rows[0].size > 0:
                key = str(np.argmax(blds[i])) + '-' + str(np.argmax(flrs[i]))
            else:
                key = str(np.argmax(blds[i]))
            coord_est[i] = coord_est_weighted[i] = training_coord_avg[key]

    # calculate localization errors per EvAAL/IPIN 2015 competition
    dist = norm(coord - coord_est, axis=1)  # Euclidean distance
    dist_weighted = norm(coord - coord_est_weighted, axis=1)
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

    # calculate positioning error based on coordinates regression
    coord_preds = coord_scaler.inverse_transform(
        preds[3])  # inverse-scaled version
    location_mse = ((coord - coord_preds)**2).mean()

    # calculate localization errors per EvAAL/IPIN 2015 competition
    dist = norm(coord - coord_preds, axis=1)  # Euclidean distance
    error = dist + 50 * (
        1 - bld_results) + 4 * flr_diff  # individual error [m]
    mean_error_regression = error.mean()
    median_error_regression = np.median(error)

    ### print out final results
    base_dir = '../results/test/' + (os.path.splitext(
        os.path.basename(__file__))[0]).replace('test_', '') + '/' + dataset
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
        output_file.write(
            "  - Fraction of data loaded for training and validation: %.2f\n" %
            frac)
        output_file.write("  - Validation split: %.2f\n" % validation_split)
        output_file.write(
            "  - Preprocessor for scaling/normalizing input data: %s\n" %
            preprocessor)
        output_file.write("  - Batch size: %d\n" % batch_size)
        output_file.write("  - Epochs: %d\n" % epochs)
        output_file.write("  - Optimizer: %s\n" % optimizer)
        output_file.write("  - Dropout rate: %.2f\n" % dropout)
        output_file.write(
            "  - Number of (nearest) neighbour locations: %d\n" % N)
        output_file.write("  - Scaling factor for threshold: %.2f\n" % scaling)
        output_file.write("  - Deep autoencoder hidden layers: ")
        if dae_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % dae_hidden_layers[0])
            for units in dae_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("  - Stacked denoising autoencoder hidden layers: ")
        if sdae_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % sdae_hidden_layers[0])
            for units in sdae_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("  - Common hidden layers: ")
        if common_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % common_hidden_layers[0])
            for units in common_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("\n")
        output_file.write("* Model Summary\n")
        model.summary(print_fn=lambda x: output_file.write(x + '\n'))
        output_file.write("\n")
        output_file.write("* Performance\n")
        output_file.write(" - Building hit rate [%%]: %.2f\n" %
                          (100 * bld_acc))
        output_file.write(" - Floor hit rate [%%]: %.2f\n" % (100 * flr_acc))
        output_file.write(" - Building-floor hit rate [%%]: %.2f\n" %
                          (100 * bf_acc))
        # below are based on EvAAL/IPIN 2015 competition rules
        output_file.write("  - Mean error [m]: %.2f\n" % mean_error)
        output_file.write(
            "  - Mean error (weighted) [m]: %.2f\n" % mean_error_weighted)
        output_file.write(
            "  - Mean error (regression) [m]: %.2f\n" % mean_error_regression)
        output_file.write("  - Median error [m]: %.2f\n" % median_error)
        output_file.write(
            "  - Median error (weighted) [m]: %.2f\n" % median_error_weighted)
        output_file.write("  - Median error (regression) [m]: %.2f\n" %
                          median_error_regression)
