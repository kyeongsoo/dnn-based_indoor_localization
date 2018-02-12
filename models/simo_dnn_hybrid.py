#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     simo_dnn_hybrid.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-02-02
#
# @brief    A scalable indoor localization system (up to location coordinates)
#           based on Wi-Fi fingerprinting using a single-input and multi-output
#           (SIMO) deep neural network (DNN) model for hybrid building and floor
#           classification and locaton coordinates regression.
#
# @remarks  The results will be published in a paper submitted to the
#           <a href="http://www.sciencedirect.com/science/journal/08936080">Elsevier Neural Networks</a>
#           journal.


### import basic modules first
import os
import sys
### import other modules (except keras and its backend which will be loaded later for reproducibility)
import argparse
import datetime
import math
import matplotlib
if 'matplotlib.pyplot' not in sys.modules:
    if 'pylab' not in sys.modules:
        matplotlib.use('Agg') # directly plot to a file when no GUI is available (e.g., remote running)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# from configparser import ConfigParser
from collections import namedtuple
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
from time import time
from timeit import default_timer as timer


### global variables
#------------------------------------------------------------------------
# general
#------------------------------------------------------------------------
VERBOSE = 1                     # 0 for turning off logging
OPTIMIZER = 'adam'              # common for all outputs
#------------------------------------------------------------------------
# stacked auto encoder (sae)
#------------------------------------------------------------------------
SAE_ACTIVATION = 'relu'
SAE_OPTIMIZER = 'adam'
SAE_LOSS = 'mean_squared_error'
#------------------------------------------------------------------------
# common hidden layers
#------------------------------------------------------------------------
COMMON_ACTIVATION = 'relu'
#------------------------------------------------------------------------
# classifier
#------------------------------------------------------------------------
CLASSIFIER_ACTIVATION = 'relu'
# CLASSIFIER_OPTIMIZER = 'adam'
#------------------------------------------------------------------------
# regressor
#------------------------------------------------------------------------
REGRESSOR_ACTIVATION = 'tanh'   # for nonlinear regression
# REGRESSOR_ACTIVATION = 'relu'   # for linear regression
# REGRESSOR_OPTIMIZER = 'adam'
#------------------------------------------------------------------------
# input files
#------------------------------------------------------------------------
training_data_file = os.path.expanduser(
    '~kks/Research/Ongoing/localization/xjtlu_surf_indoor_localization/data/UJIIndoorLoc/trainingData2.csv'
)  # '-110' for the lack of AP.
validation_data_file = os.path.expanduser(
    '~kks/Research/Ongoing/localization/xjtlu_surf_indoor_localization/data/UJIIndoorLoc/validationData2.csv'
)  # ditto


def simo_dnn_hybrid(gpu_id, random_seed, epochs, batch_size, training_ratio,
                    dropout, sae_model_file, sae_hidden_layers,
                    common_hidden_layers, floor_location_hidden_layers,
                    building_hidden_layers, floor_hidden_layers,
                    location_hidden_layers, building_weight, floor_weight,
                    location_weight):
    """Multi-building and multi-floor indoor localisztion based on Wi-Fi fingerprinting with a single-input and multi-output deep neural network

    Keyword arguments:
    gpu_id -- ID of GPU device to run this script; set it to a negative number for CPU (i.e., no GPU)
    random_seed -- a seed for random number generator
    epoch -- number of epochs
    batch_size -- batch size
    training_ratio -- ratio of training data to the overall data
    dropout -- dropout rate before and after hidden layers
    sae_model_file -- full path name for SAE model load & save
    sae_hidden_layers -- list of numbers of units in SAE hidden layers
    common_hidden_layers -- list of numbers of units in common hidden layers
    floor_location_hidden_layers -- list of numbers of units in floor/location hidden layers
    building_hidden_layers --list of numbers of units in building classifier hidden layers
    floor_hidden_layers --list of numbers of units in floor classifier hidden layers
    location_hidden_layers --list of numbers of units in location hidden layers
    building_weight -- loss weight for a building classifier
    floor_weight -- loss weight for a floor classifier
    location_weight -- loss weight for a location regressor
    """

    np.random.seed(random_seed) # initialize random number generator
    
    #--------------------------------------------------------------------
    # import keras and its backend (e.g., tensorflow)
    #--------------------------------------------------------------------
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # supress warning messages
    import tensorflow as tf
    from keras import backend as K
    from keras.engine.topology import Input
    from keras.layers import Activation, Dense, Dropout
    from keras.layers.normalization import BatchNormalization
    from keras.models import Model, Sequential, load_model
    from keras.callbacks import TensorBoard
    K.clear_session()           # avoid clutter from old models / layers.

    # read both train and test dataframes for consistent label formation through one-hot encoding
    training_df = pd.read_csv(training_data_file, header=0)  # pass header=0 to be able to replace existing names
    testing_df = pd.read_csv(validation_data_file, header=0)  # turn the validation set into a testing set
    

    # scale numerical data (over their flattened versions for joint scaling)
    rss_scaler = StandardScaler()  # the same scaling will be applied to test data later
    utm_scaler = StandardScaler()  # ditto
    
    col_aps = [col for col in training_df.columns if 'WAP' in col]
    num_aps = len(col_aps)
    rss = np.asarray(training_df[col_aps], dtype=np.float32)
    rss = (rss_scaler.fit_transform(rss.reshape((-1, 1)))).reshape(rss.shape)
    
    utm_x = np.asarray(training_df['LONGITUDE'], dtype=np.float32)
    utm_y = np.asarray(training_df['LATITUDE'], dtype=np.float32)
    utm = utm_scaler.fit_transform(np.column_stack((utm_x, utm_y)))
    num_coords = utm.shape[1]
    
    # # map reference points to sequential IDs per building & floor before building labels
    # training_df['REFPOINT'] = training_df.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])), axis=1) # add a new column
    # blds = np.unique(training_df[['BUILDINGID']])
    # flrs = np.unique(training_df[['FLOOR']])
    # for bld in blds:
    #     for flr in flrs:
    #         cond = (training_df['BUILDINGID']==bld) & (training_df['FLOOR']==flr)
    #         _, idx = np.unique(training_df.loc[cond, 'REFPOINT'], return_inverse=True) # refer to numpy.unique manual
    #         training_df.loc[cond, 'REFPOINT'] = idx

    # build labels for the classification of a building, a floor, and a reference point
    num_training_samples = len(training_df)
    num_testing_samples = len(testing_df)
    blds_all = np.asarray(pd.get_dummies(pd.concat([training_df['BUILDINGID'], testing_df['BUILDINGID']])))  # for consistency in one-hot encoding for both dataframes
    num_blds = blds_all.shape[1]
    flrs_all = np.asarray(pd.get_dummies(pd.concat([training_df['FLOOR'], testing_df['FLOOR']])))  # ditto
    num_flrs = flrs_all.shape[1]
    blds = blds_all[:num_training_samples]
    flrs = flrs_all[:num_training_samples]
    # rfps = np.asarray(pd.get_dummies(training_df['REFPOINT']))
    # num_rfps = rfps.shape[1]    
    # labels is an array of 19937 x 118
    # - 3 for BUILDINGID
    # - 5 for FLOOR,
    # - 110 for REFPOINT
    # OUTPUT_DIM = training_labels.shape[1]

    # split the training set into a training and a validation set; the original
    # validation set is used as a testing set.
    mask_training = np.random.rand(len(rss)) < training_ratio # mask index array
    
    rss_training = rss[mask_training]
    utm_training = utm[mask_training]
    blds_training = blds[mask_training]
    flrs_training = flrs[mask_training]
    # rfps_training = rfps[mask_training]
    
    rss_validation = rss[~mask_training]
    utm_validation = utm[~mask_training]
    blds_validation = blds[~mask_training]
    flrs_validation = flrs[~mask_training]
    # rfps_validation = rfps[~mask_training]
    
    ### build SAE encoder model
    print("\nPart 1: buidling an SAE encoder ...")

    if os.path.isfile(sae_model_file) and (os.path.getmtime(sae_model_file) > os.path.getmtime(__file__)):
        model = load_model(sae_model_file)
        # # below are the workaround from oarriaga@GitHub: https://github.com/keras-team/keras/issues/4044
        # model = load_model(sae_model_file, compile=False) 
        # model.compile(optimizer=SAE_OPTIMIZER, loss=SAE_LOSS)
    else:
        # N.B. each layer is named explicitly to avoid any conflicts with
        # calling model.compile() later
        model = Sequential()
        model.add(Dense(sae_hidden_layers[0], input_dim=num_aps, activation=SAE_ACTIVATION, use_bias=False, name='sae_input'))
        n_hl = 0
        for units in sae_hidden_layers[1:]:
            n_hl += 1
            model.add(Dense(units, activation='relu', use_bias=False, name='sae_hidden_' + str(n_hl)))  
        model.add(Dense(num_aps, activation='sigmoid', use_bias=False, name='sae_output'))
        model.compile(optimizer=SAE_OPTIMIZER, loss=SAE_LOSS)

        # train the model
        model.fit(rss_training, rss_training, batch_size=batch_size, epochs=epochs, verbose=VERBOSE, shuffle=True)

        # remove the decoder part
        num_to_remove = (len(sae_hidden_layers) + 1) // 2
        for i in range(num_to_remove):
            model.pop()

        # # set all layers (i.e., SAE encoder) to non-trainable (weights will not be updated)
        # # N.B. the effect of freezing seems to be negative
        # for layer in model.layers[:]:
        #     layer.trainable = False
        
        # save the SAE model for later use
        model.save(sae_model_file)

    ### build and train a complete model with the trained SAE encoder and a new classifier
    print("\nPart 2: buidling a complete model ...")

    input = Input(shape=(num_aps,), name='input')
    # denoised_input = model(input) # denoise the input data using the SAE encoder
    x = model(input) # denoise the input data using the SAE encoder
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    denoised_input = Dropout(dropout)(x)
    
    # common hidden layers for all
    if common_hidden_layers == '':
        common_input = denoised_input
    else:
        # x = Dropout(dropout)(denoised_input)
        for units in common_hidden_layers[:-1]:
            # x = Dense(units, activation=CLASSIFIER_ACTIVATION, use_bias=False)(x)
            x = Dense(units, use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(dropout)(x)
        x = Dense(common_hidden_layers[-1], use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        common_input = Dropout(dropout, name='common_input')(x)
        # common_input = Dense(
        #     common_hidden_layers[-1],
        #     activation='relu',
        #     use_bias=False,
        #     name='common_input'
        # )(x)
    
    # building classifier output
    # x = Dropout(dropout)(common_input)
    for units in building_hidden_layers:
        # x = Dense(units, activation=CLASSIFIER_ACTIVATION, use_bias=False)(x)
        x = Dense(units, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)
    x = Dense(num_blds, use_bias=False)(x)
    x = BatchNormalization()(x)
    building_output = Activation('softmax', name='building_output')(x)  # no dropout for an output layer
    # building_output = Dense(
    #     num_blds,
    #     activation='softmax',   # for multi-class classification; resulting in
    #                             # predicted probabilities in [0, 1]
    #     use_bias=False,
    #     name='building_output'
    # )(x)

    # common hidden layers for floor and location/position
    if floor_location_hidden_layers == '':
        floor_location_input = common_input
    else:
        # x = Dropout(dropout)(common_input)
        for units in floor_location_hidden_layers[:-1]:
            # x = Dense(units, activation=CLASSIFIER_ACTIVATION, use_bias=False)(x)
            x = Dense(units, use_bias=False)(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Dropout(dropout)(x)
        x = Dense(floor_location_hidden_layers[-1], use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        floor_location_input = Dropout(dropout, name='floor_location_input')(x)
        # floor_location_input = Dense(
        #     floor_location_hidden_layers[-1],
        #     activation=CLASSIFIER_ACTIVATION,
        #     use_bias=False,
        #     name='floor_location_input'
        # )(x)
    
    # floor classifier output
    # x = Dropout(dropout)(floor_location_input)
    for units in floor_hidden_layers:
        # x = Dense(units, activation=CLASSIFIER_ACTIVATION, use_bias=False)(x)
        x = Dense(units, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)
    x = Dense(num_flrs, use_bias=False)(x)
    x = BatchNormalization()(x)
    floor_output = Activation('softmax', name='floor_output')(x)  # no dropout for an output layer
    # floor_output = Dense(
    #     num_flrs,
    #     activation='softmax',
    #     use_bias=False,
    #     name='floor_output'
    # )(x)

    # location/position regressor output
    # x = Dropout(dropout)(floor_location_input)
    for units in location_hidden_layers:
        # x = Dense(units, activation=REGRESSOR_ACTIVATION, use_bias=False, kernel_initializer='normal')(x)
        x = Dense(units, use_bias=False, kernel_initializer='normal')(x)
        x = BatchNormalization()(x)
        x = Activation(REGRESSOR_ACTIVATION)(x)
        x = Dropout(dropout)(x)
    x = Dense(num_coords, use_bias=False, kernel_initializer='normal')(x)
    x = BatchNormalization()(x)
    location_output = Activation(REGRESSOR_ACTIVATION, name='location_output')(x)
    # location_output = Dense(
    #     num_coords,
    #     activation='linear',    # for a regressor output layer
    #     use_bias=False,
    #     kernel_initializer='normal',
    #     name='location_output'
    # )(x)

    # build and compile a SIMO model
    model = Model(inputs=[input], outputs=[building_output, floor_output, location_output])
    model.compile(
        optimizer=OPTIMIZER,
        loss=['categorical_crossentropy', 'categorical_crossentropy', 'mean_squared_error'],
        loss_weights={'building_output': building_weight, 'floor_output': floor_weight, 'location_output': location_weight},
        metrics={'building_output': 'accuracy', 'floor_output': 'accuracy', 'location_output': 'mean_squared_error'}
    )
    
    # train the model
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), write_graph=True)
    startTime = timer()
    history = model.fit(
        x={'input': rss_training},
        y={'building_output': blds_training, 'floor_output': flrs_training, 'location_output': utm_training},
        # validation_split=1.0-training_ratio,
        validation_data=(
            {'input': rss_validation},
            {'building_output': blds_validation, 'floor_output': flrs_validation, 'location_output': utm_validation}
        ),
        batch_size=batch_size,
        epochs=epochs,
        verbose=VERBOSE,
        callbacks=[tensorboard],
        shuffle=True
    )
    elapsedTime = timer() - startTime
    print("Model trained in %e s." % elapsedTime)
    
    ### evaluate the model
    print("\nPart 3: evaluating the model ...")

    # turn the given validation set into a testing set
    rss_testing = np.asarray(testing_df[col_aps], dtype=np.float32)
    rss_testing = (rss_scaler.transform(rss_testing.reshape((-1, 1)))).reshape(rss_testing.shape)
    utm_x_testing = np.asarray(testing_df['LONGITUDE'], dtype=np.float32)
    utm_y_testing = np.asarray(testing_df['LATITUDE'], dtype=np.float32)
    utm_testing_original = np.column_stack((utm_x_testing, utm_y_testing))
    utm_testing = utm_scaler.transform(utm_testing_original)  # scaled version
    blds_testing = blds_all[num_training_samples:]
    flrs_testing = flrs_all[num_training_samples:]

    # rst = model.evaluate(
    #     x={'input': rss_testing},
    #     y={'building_output': blds_testing, 'floor_output': flrs_testing, 'location_output': utm_testing}
    # )

    # Results = namedtuple('Results', ['losses', 'metrics', 'history'])
    # Losses = namedtuple('Losses', ['overall', 'building', 'floor', 'location'])
    # Metrics = namedtuple('Metrics', ['building_acc', 'floor_acc', 'location_mse'])
    # results = Results(
    #     losses=Losses(overall=rst[0], building=rst[1], floor=rst[2], location=rst[3]),
    #     metrics=Metrics(building_acc=rst[4], floor_acc=rst[5], location_mse=rst[6]),
    #     history=history
    # )

    # calculate the classification accuracies and localization errors
    preds = model.predict(rss_testing, batch_size=batch_size)  # a list of arrays (one for each output) returned
    blds_results = (np.equal(np.argmax(blds_testing, axis=1), np.argmax(preds[0], axis=1))).astype(int)
    blds_acc = blds_results.mean()
    flrs_results = (np.equal(np.argmax(flrs_testing, axis=1), np.argmax(preds[1], axis=1))).astype(int)
    flrs_acc = flrs_results.mean()
    bf_acc = (blds_results*flrs_results).mean()
    # rfps_results = (np.equal(np.argmax(test_labels[:, 8:118], axis=1), np.argmax(preds[:, 8:118], axis=1))).astype(int)
    # acc_rfp = rfps_results.mean()
    # acc = (blds_results*flrs_results*rfps_results).mean()
    utm_preds = utm_scaler.inverse_transform(preds[2])  # inverse-scaled version
    location_mse = ((utm_testing_original-utm_preds)**2).mean()

    # calculate localization errors per EvAAL/IPIN 2015 competition
    dist = norm(utm_testing_original-utm_preds, axis=1)  # Euclidean distance
    flrs_diff = np.absolute(np.argmax(flrs_testing, axis=1) -
                            np.argmax(preds[1], axis=1))
    error = dist + 50*(1-blds_results) + 4*flrs_diff # individual error [m]
    mean_error = error.mean()
    median_error = np.median(error)

    Results = namedtuple('Results', ['metrics', 'history'])
    Metrics = namedtuple('Metrics', ['building_acc', 'floor_acc', 'bf_acc',
                                     'location_mse', 'mean_error',
                                     'median_error'])
    results = Results(metrics=Metrics(building_acc=blds_acc, floor_acc=flrs_acc,
                                      bf_acc=bf_acc, location_mse=location_mse,
                                      mean_error=mean_error,
                                      median_error=median_error),
                      history=history )
    return results

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-G",
        "--gpu_id",
        help="ID of GPU device to run this script; default is 0; set it to a negative number for CPU (i.e., no GPU)",
        default=0,
        type=int)
    parser.add_argument(
        "-R",
        "--random_seed",
        help="random seed",
        default=0,
        type=int)
    parser.add_argument(
        "-E",
        "--epochs",
        help="number of epochs; default is 20",
        default=20,
        type=int)
    parser.add_argument(
        "-B",
        "--batch_size",
        help="batch size; default is 10",
        default=10,
        type=int)
    parser.add_argument(
        "-T",
        "--training_ratio",
        help="ratio of training data to the overall data: default is 0.9",
        default=0.9,
        type=float)
    parser.add_argument(
        "-D",
        "--dropout",
        help=
        "dropout rate before and after hidden layers; default 0.0",
        default=0.0,
        type=float)
    parser.add_argument(
        "-S",
        "--sae_hidden_layers",
        help=
        "comma-separated numbers of units in SAE hidden layers; default is '128,32,128'",
        default='128,32,128',
        type=str)
    parser.add_argument(
        "-C",
        "--common_hidden_layers",
        help=
        "comma-separated numbers of units in common hidden layers for building, floor, and location; default is '16,4'",
        default='16,16',
        type=str)
    parser.add_argument(
        "--floor_location_hidden_layers",
        help=
        "comma-separated numbers of units in common hidden layers for floor and location/position; default is ''",
        default='',
        type=str)
    parser.add_argument(
        "--building_hidden_layers",
        help=
        "comma-separated numbers of units in building hidden layers; default is '4,2'",
        default='4,2',
        type=str)
    parser.add_argument(
        "--floor_hidden_layers",
        help=
        "comma-separated numbers of units in floor hidden layers; default is '8,4'",
        default='8,4',
        type=str)
    parser.add_argument(
        "--location_hidden_layers",
        help=
        "comma-separated numbers of units in location hidden layers; default is '8,4'",
        default='8,4',
        type=str)
    parser.add_argument(
        "--building_weight",
        help=
        "loss weight for a building; default 1.0",
        default=1.0,
        type=float)
    parser.add_argument(
        "--floor_weight",
        help=
        "loss weight for a floor; default 1.0",
        default=1.0,
        type=float)
    parser.add_argument(
        "--location_weight",
        help=
        "loss weight for a location; default 1.0",
        default=1.0,
        type=float)
    args = parser.parse_args()

    # set variables using command-line arguments
    gpu_id = args.gpu_id
    random_seed = args.random_seed
    epochs = args.epochs
    batch_size = args.batch_size
    training_ratio = args.training_ratio
    dropout = args.dropout
    
    sae_hidden_layers = [int(i) for i in (args.sae_hidden_layers).split(',')]
    if args.common_hidden_layers == '':
        common_hidden_layers = ''
    else:
        common_hidden_layers = [int(i) for i in (args.common_hidden_layers).split(',')]
    if args.floor_location_hidden_layers == '':
        floor_location_hidden_layers = ''
    else:
        floor_location_hidden_layers = [int(i) for i in (args.floor_location_hidden_layers).split(',')]
    if args.building_hidden_layers == '':
        building_hidden_layers = ''
    else:
        building_hidden_layers = [int(i) for i in (args.building_hidden_layers).split(',')]
    if args.floor_hidden_layers == '':
        floor_hidden_layers = ''
    else:
        floor_hidden_layers = [int(i) for i in (args.floor_hidden_layers).split(',')]
    if args.location_hidden_layers == '':
        location_hidden_layers = ''
    else:
        location_hidden_layers = [int(i) for i in (args.location_hidden_layers).split(',')]
    building_weight = args.building_weight
    floor_weight = args.floor_weight
    location_weight = args.location_weight

    # set full path and base for file names based on input parameter values
    base_dir = '../results/' + os.path.splitext(os.path.basename(__file__))[0] + '/'
    base_file_name = base_dir \
               + 'B' + "{0:d}".format(batch_size) \
               + '_T' + "{0:.2f}".format(args.training_ratio) \
               + '_S' + args.sae_hidden_layers.replace(',', '-')
    sae_model_file = base_file_name + '.hdf5'
    # output_file_base = base_file_name + '_C' + args.classifier_hidden_layers.replace(',', '-') \
    #              + '_D' + "{0:.2f}".format(dropout)
    now = datetime.datetime.now()
    output_file_base = base_file_name + "_D{0:.2f}_".format(dropout) + now.strftime("%Y%m%d-%H%M%S")
    
    ### call simo_dnn_hybrid()
    results = simo_dnn_hybrid(
        gpu_id=gpu_id,
        random_seed=random_seed,
        epochs=epochs,
        batch_size=batch_size,
        training_ratio=training_ratio,
        dropout=dropout,
        sae_model_file=sae_model_file,
        sae_hidden_layers=sae_hidden_layers,
        common_hidden_layers=common_hidden_layers,
        floor_location_hidden_layers=floor_location_hidden_layers,
        building_hidden_layers=building_hidden_layers,
        floor_hidden_layers=floor_hidden_layers,
        location_hidden_layers=location_hidden_layers,
        building_weight=building_weight,
        floor_weight=floor_weight,
        location_weight=location_weight
    )
    
    ### print out final results
    with open(output_file_base + '.org', 'w') as output_file:
        output_file.write("#+STARTUP: showall\n")  # unfold everything when opening
        output_file.write("* System parameters\n")
        output_file.write("  - Random number seed: %d\n" % random_seed)
        output_file.write("  - Ratio of training data to overall data: %.2f\n" % training_ratio)
        output_file.write("  - Epochs: %d\n" % epochs)
        output_file.write("  - Batch size: %d\n" % batch_size)
        output_file.write("  - Optimizer: %s\n" % OPTIMIZER)
        output_file.write("  - SAE hidden layers: %d" % sae_hidden_layers[0])
        for units in sae_hidden_layers[1:]:
            output_file.write("-%d" % units)
        output_file.write("\n")
        output_file.write("  - SAE activation: %s\n" % SAE_ACTIVATION)
        # output_file.write("  - SAE optimizer: %s\n" % SAE_OPTIMIZER)
        output_file.write("  - SAE loss: %s\n" % SAE_LOSS)
        output_file.write("  - Common hidden layers: ")
        if common_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % common_hidden_layers[0])
            for units in common_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("  - Floor/location hidden layers: ")
        if floor_location_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % floor_location_hidden_layers[0])
            for units in floor_location_hidden_layers[1:]:
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
        output_file.write("  - Classifier activation: %s\n" % CLASSIFIER_ACTIVATION)
        # output_file.write("  - Classifier optimizer: %s\n" % CLASSIFIER_OPTIMIZER)
        # output_file.write("  - Classifier loss: %s\n" % CLASSIFIER_LOSS)
        output_file.write("  - Dropout rate: %.2f\n" % dropout)
        output_file.write("  - Loss weight for buildings: %.2f\n" % building_weight)
        output_file.write("  - Loss weight for floors: %.2f\n" % floor_weight)
        output_file.write("  - Loss weight for location: %.2f\n" % location_weight)
        output_file.write("* Performance\n")
        # output_file.write("  - Loss (overall): %e\n" % results.losses.overall)
        # output_file.write("  - Accuracy (overall): %e\n" % results.accuracy.overall)
        output_file.write("  - Building hit rate [%%]: %.2f\n" % (100*results.metrics.building_acc))
        output_file.write("  - Floor hit rate [%%]: %.2f\n" % (100*results.metrics.floor_acc))
        output_file.write("  - Building-floor hit rate [%%]: %.2f\n" % (100*results.metrics.bf_acc))
        output_file.write("  - MSE (location): %e\n" % results.metrics.location_mse)
        output_file.write("  - Mean error [m]: %.2f\n" % results.metrics.mean_error)  # according to EvAAL/IPIN 2015 competition rule
        output_file.write("  - Median error [m]: %.2f\n" % results.metrics.median_error)  # ditto

    ### plot history of overall loss during the training and validation (other metrics to be added)
    plt.plot(results.history.history['loss'])
    plt.plot(results.history.history['val_loss'])
    # plt.title('model accuracy')
    plt.ylabel('Overall Loss')
    plt.xlabel('Epoch')
    plt.legend(['Traininig', 'Validation'], loc='upper left')
    plt.savefig(os.path.expanduser(output_file_base) + '.pdf', format='pdf')
    plt.close('all')
