#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     simo_swt_hybrid_tut.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-08-04
#
# @brief A scalable indoor localization system based on Wi-Fi fingerprinting
#        using a stage-wise-trained multi-class classification of building,
#        floor, and regression of location coordiates with a single-input and
#        multi-output (SIMO) deep neural network (DNN) model and TUT datasets.
#
# @remarks TBD

### import basic modules and a model to test
import os
# os.environ['PYTHONHASHSEED'] = '0'  # for reproducibility
import sys
sys.path.insert(0, '../models')
sys.path.insert(0, '../utils')
from deep_autoencoder import deep_autoencoder
from sdae import sdae
from mean_ci import mean_ci
### import other modules; keras and its backend will be loaded later
import argparse
import datetime
import math
import multiprocessing
import numpy as np
import pandas as pd
import pathlib
import random as rn
from collections import namedtuple
from num2words import num2words
from numpy.linalg import norm
from time import time
from timeit import default_timer as timer
### import keras and tensorflow backend
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # supress warning messages
import tensorflow as tf
num_cpus = multiprocessing.cpu_count()
session_conf = tf.ConfigProto(
    intra_op_parallelism_threads=num_cpus,
    inter_op_parallelism_threads=num_cpus
)
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dense, Dropout, Input
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.models import Model


def simo_swt_hybrid_tut(
        gpu_id: int,
        dataset: str,
        frac: float,
        validation_split: float,
        preprocessor: str,
        batch_size: int,
        epochs: int,
        optimizer: str,
        dropout: float,
        corruption_level: float,
        dae_hidden_layers: list,
        sdae_hidden_layers: list,
        cache: bool,
        common_hidden_layers: list,
        floor_hidden_layers: list,
        coordinates_hidden_layers: list,
        verbose: int
):
    """Multi-floor indoor localization based on hybrid floor classification and
    coordinates regression using a stage-wise trained single-input and
    multi-output (SIMO) deep neural network (DNN) model and TUT datasets.

    Keyword arguments:

    """

    ### initialize numpy, random, TensorFlow, and keras
    np.random.seed()            # based on current time or OS-specific randomness source
    rn.seed()                   # "
    tf.set_random_seed(rn.randint(0, 1000000))
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    sess = tf.Session(
        graph=tf.get_default_graph(),
        config=session_conf)
    K.set_session(sess)

    ### load datasets after scaling
    print("Loading data ...")
    if dataset == 'tut':
        from tut import TUT
        tut = TUT(
            cache=cache,
            frac=frac,
            preprocessor=preprocessor,
            classification_mode='hierarchical',
            grid_size=0)
    elif dataset == 'tut2':
        from tut import TUT2
        tut = TUT2(
            cache=cache,
            frac=frac,
            preprocessor=preprocessor,
            classification_mode='hierarchical',
            grid_size=0,
            testing_split=0.2)
    elif dataset == 'tut3':
        from tut import TUT3
        tut = TUT3(
            cache=cache,
            frac=frac,
            preprocessor=preprocessor,
            classification_mode='hierarchical',
            grid_size=0)
    else:
        print("'{0}' is not a supported data set.".format(dataset))
        sys.exit(0)
    flr_height = tut.floor_height
    training_df = tut.training_df
    training_data = tut.training_data
    testing_df = tut.testing_df
    testing_data = tut.testing_data
        
    ### build and do stage-wise training of a SIMO model
    print(
        "Building and stage-wise training a SIMO model for hybrid classification and regression ..."
    )
    rss = training_data.rss_scaled
    coord = training_data.coord_scaled
    coord_scaler = training_data.coord_scaler  # for inverse transform
    labels = training_data.labels
    input = Input(shape=(rss.shape[1], ), name='input')  # common input

    # (optional) build deep autoencoder or stacked denoising autoencoder
    if dae_hidden_layers != '':
        print("- Building a DAE model ...")
        model = deep_autoencoder(
            dataset=dataset,
            input_data=rss,
            preprocessor=preprocessor,
            hidden_layers=dae_hidden_layers,
            cache=cache,
            model_fname=None,
            optimizer=optimizer,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split)
        x = model(input)
    elif sdae_hidden_layers != '':
        print("- Building an SDAE model ...")
        model = sdae(
            dataset=dataset,
            input_data=rss,
            preprocessor=preprocessor,
            hidden_layers=sdae_hidden_layers,
            cache=cache,
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
        for i in range(len(common_hidden_layers)):
            x = Dense(common_hidden_layers[i], name='common_hidden_layer_{:d}'.format(i))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(dropout)(x)
    common_hl_output = x

    # floor classification output
    if floor_hidden_layers != '':
        for i in range(len(floor_hidden_layers)):
            x = Dense(floor_hidden_layers[i], name='floor_hidden_layer_{:d}'.format(i))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(dropout)(x)
        i += 1
    else:
        i = 0
    x = Dense(labels.floor.shape[1], name='floor_hidden_layer_{:d}'.format(i))(x)
    x = BatchNormalization()(x)
    floor_output = Activation(
        'softmax', name='floor_output')(x)  # no dropout for an output layer

    # coordinates regression output
    x = common_hl_output
    if coordinates_hidden_layers != '':
        for i in range(len(coordinates_hidden_layers)):
            x = Dense(coordinates_hidden_layers[i], kernel_initializer='normal', name='coordinates_hidden_layer_{:d}'.format(i))(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(dropout)(x)
        i += 1
    else:
        i = 0
    x = Dense(coord.shape[1], kernel_initializer='normal', name='coordinates_hidden_layer_{:d}'.format(i))(x)
    x = BatchNormalization()(x)
    coordinates_output = Activation(
        'linear', name='coordinates_output')(x)  # 'linear' activation
    
    # build model
    model = Model(
        inputs=input,
        outputs=[
            floor_output,
            coordinates_output
        ])

    print("- Stage-wise training with floor information ...", end='')
    model.compile(
        optimizer=optimizer,
        loss=[
            'categorical_crossentropy',
            'mean_squared_error'
        ],
        loss_weights={
            'floor_output': 1.0,
            'coordinates_output': 0.0
        },
        metrics={
            'floor_output': 'accuracy',
            'coordinates_output': 'mean_squared_error'
        })
    weights_file = os.path.expanduser("~/tmp/best_f-weights.h5")
    checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', save_best_only=True, verbose=0)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
    
    startTime = timer()
    f_history = model.fit(
        x={'input': rss},
        y={
            'floor_output': labels.floor,
            'coordinates_output': coord
        },
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=[checkpoint, early_stop],
        validation_split=validation_split,
        shuffle=True)
    elapsedTime = timer() - startTime
    elapsedTime_total = elapsedTime
    print(" completed in {0:.4e} s".format(elapsedTime))
    model.load_weights(weights_file)  # load weights from the best model

    print(
        "- Stage-wise training with floor-coordinates information ...", end=''
    )

    # reinitialize hidden layers based on
    # https://www.codementor.io/nitinsurya/how-to-re-initialize-keras-model-weights-et41zre2g
    # - common
    if common_hidden_layers != '':
        for i in range(len(common_hidden_layers)):
            layer = model.get_layer('common_hidden_layer_{:d}'.format(i))
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=sess)
    # # - floor
    # if floor_hidden_layers != '':
    #     for i in range(len(floor_hidden_layers)):
    #         layer = model.get_layer('floor_hidden_layer_{:d}'.format(i))
    #         if hasattr(layer, 'kernel_initializer'):
    #             layer.kernel.initializer.run(session=sess)
    #     i += 1
    # else:
    #     i = 0
    # layer = model.get_layer('floor_hidden_layer_{:d}'.format(i))
    # if hasattr(layer, 'kernel_initializer'):
    #     layer.kernel.initializer.run(session=sess)
    # - coordinats
    if coordinates_hidden_layers != '':
        for i in range(len(coordinates_hidden_layers)):
            layer = model.get_layer('coordinates_hidden_layer_{:d}'.format(i))
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=sess)
        i += 1
    else:
        i = 0
    layer = model.get_layer('coordinates_hidden_layer_{:d}'.format(i))
    if hasattr(layer, 'kernel_initializer'):
        layer.kernel.initializer.run(session=sess)
                
    model.compile(
        optimizer=optimizer,
        loss=[
            'categorical_crossentropy',
            'mean_squared_error'
        ],
        loss_weights={
            'floor_output': 1.0,
            'coordinates_output': 1.0
        },
        metrics={
            'floor_output': 'accuracy',
            'coordinates_output': 'mean_squared_error'
        })
    weights_file = os.path.expanduser("~/tmp/best_fc-weights.h5")
    checkpoint = ModelCheckpoint(weights_file, monitor='val_loss', save_best_only=True, verbose=0)
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

    startTime = timer()
    fc_history = model.fit(
        x={'input': rss},
        y={
            'floor_output': labels.floor,
            'coordinates_output': coord
        },
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=[checkpoint, early_stop],
        validation_split=validation_split,
        shuffle=True)
    elapsedTime = timer() - startTime
    elapsedTime_total += elapsedTime
    print(" completed in {0:.4e} s".format(elapsedTime))
    model.load_weights(weights_file)  # load weights from the best model

    ### evaluate the model
    print("Evaluating the model ...")
    rss = testing_data.rss_scaled
    labels = testing_data.labels
    flrs = labels.floor
    coord = testing_data.coord  # original coordinates
    x_col_name = 'X'
    y_col_name = 'Y'

    # calculate the classification accuracies and localization errors
    flrs_pred, coords_scaled_pred = model.predict(rss, batch_size=batch_size)
    flr_results = (np.equal(
        np.argmax(flrs, axis=1), np.argmax(flrs_pred, axis=1))).astype(int)
    flr_acc = flr_results.mean()
    coord_est = coord_scaler.inverse_transform(coords_scaled_pred)  # inverse-scaling

    # calculate 2D localization errors
    dist_2d = norm(coord - coord_est, axis=1)
    mean_error_2d = dist_2d.mean()
    median_error_2d = np.median(dist_2d)

    # calculate 3D localization errors
    flr_diff = np.absolute(
        np.argmax(flrs, axis=1) - np.argmax(flrs_pred, axis=1))
    z_diff_squared = (flr_height**2)*np.square(flr_diff)
    dist_3d = np.sqrt(np.sum(np.square(coord - coord_est), axis=1) + z_diff_squared)
    mean_error_3d = dist_3d.mean()
    median_error_3d = np.median(dist_3d)

    LocalizationResults = namedtuple('LocalizationResults', ['flr_acc',
                                                             'mean_error_2d',
                                                             'median_error_2d',
                                                             'mean_error_3d',
                                                             'median_error_3d',
                                                             'elapsedTime'])
    return LocalizationResults(flr_acc=flr_acc, mean_error_2d=mean_error_2d,
                               median_error_2d=median_error_2d,
                               mean_error_3d=mean_error_3d,
                               median_error_3d=median_error_3d,
                               elapsedTime=elapsedTime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-N",
        "--num_runs",
        help=
        "number of runs; default is 20",
        default=20,
        type=int)
    parser.add_argument(
        "-G",
        "--gpu_id",
        help=
        "ID of GPU device to run this script; default is 0; set it to a negative number for CPU (i.e., no GPU)",
        default=0,
        type=int)
    parser.add_argument(
        "--dataset",
        help="a data set for training, validation, and testing; choices are 'tut' (default), 'tut2', and 'tut3'",
        default='tut',
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
        help="batch size; default is 64",
        default=64,
        type=int)
    parser.add_argument(
        "-E",
        "--epochs",
        help="number of epochs; default is 100",
        default=100,
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
        help="dropout rate before and after hidden layers; default is 0.25",
        default=0.25,
        type=float)
    parser.add_argument(
        "-C",
        "--corruption_level",
        help=
        "corruption level of masking noise for stacked denoising autoencoder; default is 0.1",
        default=0.1,
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
        "--no_cache",
        help=
        "disable loading a trained model from/saving it to a cache",
        action='store_true')
    parser.add_argument(
        "--common_hidden_layers",
        help=
        "comma-separated numbers of units in common hidden layers; default is '1024'",
        default='1024',
        type=str)
    parser.add_argument(
        "--floor_hidden_layers",
        help=
        "comma-separated numbers of units in additional hidden layers for floor; default is '256'",
        default='256',
        type=str)
    parser.add_argument(
        "--coordinates_hidden_layers",
        help=
        "comma-separated numbers of units in additional hidden layers for coordinates; default is '256'",
        default='256',
        type=str)
    parser.add_argument(
        "-V",
        "--verbose",
        help=
        "verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch; default is 0",
        default=0,
        type=int)
    args = parser.parse_args()

    # set variables using command-line arguments
    num_runs = args.num_runs
    gpu_id = args.gpu_id
    dataset = args.dataset
    frac = args.frac
    validation_split = args.validation_split
    preprocessor = args.preprocessor
    batch_size = args.batch_size
    epochs = args.epochs
    optimizer = args.optimizer
    dropout = args.dropout
    corruption_level = args.corruption_level
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
    cache = not args.no_cache
    if args.common_hidden_layers == '':
        common_hidden_layers = ''
    else:
        common_hidden_layers = [
            int(i) for i in (args.common_hidden_layers).split(',')
        ]
    if args.floor_hidden_layers == '':
        floor_hidden_layers = ''
    else:
        floor_hidden_layers = [
            int(i) for i in (args.floor_hidden_layers).split(',')
        ]
    if args.coordinates_hidden_layers == '':
        coordinates_hidden_layers = ''
    else:
        coordinates_hidden_layers = [
            int(i) for i in (args.coordinates_hidden_layers).split(',')
        ]
    verbose = args.verbose

    ### run simo_hybrid_tut() num_runs times
    flr_accs = np.empty(num_runs)
    mean_error_2ds = np.empty(num_runs)
    median_error_2ds = np.empty(num_runs)
    mean_error_3ds = np.empty(num_runs)
    median_error_3ds = np.empty(num_runs)
    elapsedTimes = np.empty(num_runs)
    for i in range(num_runs):
        print("\n########## {0:s} run ##########".format(num2words(i+1, to='ordinal_num')))
        rst = simo_swt_hybrid_tut(gpu_id, dataset, frac, validation_split,
                                  preprocessor, batch_size, epochs, optimizer,
                                  dropout, corruption_level, dae_hidden_layers,
                                  sdae_hidden_layers, cache,
                                  common_hidden_layers, floor_hidden_layers,
                                  coordinates_hidden_layers, verbose)
        flr_accs[i] = rst.flr_acc
        mean_error_2ds[i] = rst.mean_error_2d
        median_error_2ds[i] = rst.median_error_2d
        mean_error_3ds[i] = rst.mean_error_3d
        median_error_3ds[i] = rst.median_error_3d
        elapsedTimes[i] = rst.elapsedTime

    ### print out final results
    base_dir = '../results/test/' + (os.path.splitext(
        os.path.basename(__file__))[0]).replace('test_', '') + '/' + dataset
    pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)
    base_file_name = base_dir + "/E{0:d}_B{1:d}_D{2:.2f}".format(
        epochs, batch_size, dropout)
    now = datetime.datetime.now()
    output_file_base = base_file_name + '_' + now.strftime("%Y%m%d-%H%M%S")

    with open(output_file_base + '.org', 'w') as output_file:
        output_file.write(
            "#+STARTUP: showall\n")  # unfold everything when opening
        output_file.write("* System parameters\n")
        output_file.write("  - Command line: %s\n" % ' '.join(sys.argv))
        output_file.write("  - Number of runs: %d\n" % num_runs)
        output_file.write("  - GPU ID: %d\n" % gpu_id)
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
        output_file.write("  - Floor hidden layers: ")
        if floor_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % floor_hidden_layers[0])
            for units in floor_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("  - Coordinates hidden layers: ")
        if coordinates_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % coordinates_hidden_layers[0])
            for units in coordinates_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("\n")
        # output_file.write("* Model Summary\n")
        # model.summary(print_fn=lambda x: output_file.write(x + '\n'))
        # output_file.write("\n")
        output_file.write("* Performance\n")
        output_file.write("  - Floor hit rate [%]: Mean (w/ 95% CI)={0:.4f}+-{1:{ci_fs}}, Max={2:.4f}, Min={3:.4f}\n".format(*[i*100 for i in mean_ci(flr_accs)], 100*flr_accs.max(), 100*flr_accs.min(), ci_fs=('.4f' if num_runs > 1 else '')))
        output_file.write("  - Mean 2D error [m]: Mean (w/ 95% CI)={0:.4f}+-{1:.4f}, Max={2:.4f}, Min={3:.4f}\n".format(*mean_ci(mean_error_2ds), mean_error_2ds.max(), mean_error_2ds.min()))
        output_file.write("  - Median 2D error [m]: Mean (w/ 95% CI)={0:.4f}+-{1:.4f}, Max={3:.4f}, Min={3:.4f}\n".format(*mean_ci(median_error_2ds), median_error_2ds.max(), median_error_2ds.min()))
        output_file.write("  - Mean 3D error [m]: Mean (w/ 95% CI)={0:.4f}+-{1:.4f}, Max={2:.4f}, Min={3:.4f}\n".format(*mean_ci(mean_error_3ds), mean_error_3ds.max(), mean_error_3ds.min()))
        output_file.write("  - Median 3D error [m]: Mean (w/ 95% CI)={0:.4f}+-{1:.4f}, Max={2:.4f}, Min={3:.4f}\n".format(*mean_ci(median_error_3ds), median_error_3ds.max(), median_error_3ds.min()))
        output_file.write("  - Training time [s]: Mean (w/ 95% CI)={0:.4f}+-{1:.4f}, Max={2:.4f}, Min={3:.4f}\n".format(*mean_ci(elapsedTimes), elapsedTimes.max(), elapsedTimes.min()))
