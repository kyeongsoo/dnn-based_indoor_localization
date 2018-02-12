#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     siso_dnn_classification.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-02-12
#
# @brief    A scalable indoor localization system (up to reference points)
#           based on Wi-Fi fingerprinting using a single-input and single-output
#           (SIMO) deep neural network (DNN) model for multi-class
#           classification of building, floor, and reference point.
#
# @remarks  The results will be published in a paper submitted to the
#           <a href="http://www.sciencedirect.com/science/journal/08936080">Elsevier Neural Networks</a>
#           journal.


### import basic modules first
import os
os.environ['PYTHONHASHSEED'] = '0'  # for reproducibility
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
import random as rn
# from configparser import ConfigParser
from collections import namedtuple
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
from time import time
from timeit import default_timer as timer
### import keras and its backend (e.g., tensorflow)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # supress warning messages
import tensorflow as tf
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)  # force TF to use single thread for reproducibility
from keras import backend as K
from keras.engine.topology import Input
from keras.layers import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


### global constant variables
#------------------------------------------------------------------------
# general
#------------------------------------------------------------------------
VERBOSE = 1                     # 0 for turning off logging
OPTIMIZER = 'adam'              # common for all outputs
#------------------------------------------------------------------------
# input files
#------------------------------------------------------------------------
training_data_file = os.path.expanduser(
    '~kks/Research/Ongoing/localization/xjtlu_surf_indoor_localization/data/UJIIndoorLoc/trainingData2.csv'
)  # '-110' for the lack of AP.
validation_data_file = os.path.expanduser(
    '~kks/Research/Ongoing/localization/xjtlu_surf_indoor_localization/data/UJIIndoorLoc/validationData2.csv'
)  # ditto
#------------------------------------------------------------------------
# output files
#------------------------------------------------------------------------
path_base = '../results/' + os.path.splitext(os.path.basename(__file__))[0]
path_out =  path_base + '_out'
path_sae_model = path_base + '_sae_model.hdf5'


def siso_dnn_classification(input_dim, output_dim, hidden_layers):
    input = Input(shape=(input_dim,), name='input')
    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)

    for units in hidden_layers:
        x = Dense(units, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)
    x = Dense(output_dim, use_bias=False)(x)
    x = BatchNormalization()(x)
    output = Activation('softmax', name='output')(x)  # no dropout for an output layer

    model = Model(inputs=input, outputs=output)
    model.compile(
        optimizer=OPTIMIZER,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


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
        help="number of epochs; default is 50",
        default=50,
        type=int)
    parser.add_argument(
        "-B",
        "--batch_size",
        help="batch size; default is 32",
        default=32,
        type=int)
    # parser.add_argument(
    #     "-T",
    #     "--training_ratio",
    #     help="ratio of training data to overall data: default is 0.9",
    #     default=0.9,
    #     type=float)
    parser.add_argument(
        "-H",
        "--hidden_layers",
        help=
        "comma-separated numbers of units in hidden layers; default is '128,128'",
        default='128,128',
        type=str)
    parser.add_argument(
        "-D",
        "--dropout",
        help=
        "dropout rate before and after classifier hidden layers; default 0.0",
        default=0.0,
        type=float)
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
    epochs = args.epochs
    batch_size = args.batch_size
    # training_ratio = args.training_ratio
    # sae_hidden_layers = [int(i) for i in (args.sae_hidden_layers).split(',')]
    if args.hidden_layers == '':
        hidden_layers = ''
    else:
        hidden_layers = [int(i) for i in (args.hidden_layers).split(',')]
    dropout = args.dropout
    # N = args.neighbours
    # scaling = args.scaling

    ### initialize numpy, TensorFlow, and keras
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)    
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)  # for reproducibility
    K.set_session(sess)

    ### load and pre-process the dataset
    # option 1: with full data
    # training_df = pd.read_csv(training_data_file, header=0) # pass header=0 to be able to replace existing names
    # option 2: with 10% of data for development and hyperparameter tuning
    training_df = (pd.read_csv(training_data_file, header=0)).sample(frac=0.1)  # pass header=0 to be able to replace existing names
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
    
    # map reference points to sequential IDs per building & floor before building labels
    training_df['REFPOINT'] = training_df.apply(lambda row: str(int(row['SPACEID'])) + str(int(row['RELATIVEPOSITION'])), axis=1) # add a new column
    blds = np.unique(training_df[['BUILDINGID']])
    flrs = np.unique(training_df[['FLOOR']])
    x_avg = {}
    y_avg = {}
    for bld in blds:
        for flr in flrs:
            # map reference points to sequential IDs per building-floor before building labels
            cond = (training_df['BUILDINGID']==bld) & (training_df['FLOOR']==flr)
            _, idx = np.unique(training_df.loc[cond, 'REFPOINT'], return_inverse=True) # refer to numpy.unique manual
            training_df.loc[cond, 'REFPOINT'] = idx
            
            # calculate the average coordinates of each building/floor
            x_avg[str(bld) + '-' + str(flr)] = np.mean(training_df.loc[cond, 'LONGITUDE'])
            y_avg[str(bld) + '-' + str(flr)] = np.mean(training_df.loc[cond, 'LATITUDE'])

    # build labels for the multi-class classification of a building, a floor, and a reference point
    num_training_samples = len(training_df)
    num_testing_samples = len(testing_df)
    blds = training_df['BUILDINGID'].map(str)
    flrs = training_df['FLOOR'].map(str)
    rfps = training_df['REFPOINT'].map(str)
    tv_labels = np.asarray(pd.get_dummies(blds+'-'+flrs+'-'+rfps))  # labels for training/validation
    # labels is an array of 19937 x 905
    # - 3 for BUILDINGID
    # - 5 for FLOOR,
    # - 110 for REFPOINT
    output_dim = tv_labels.shape[1]
    
    # # split the training set into training and validation sets; we will use the
    # # validation set at a testing set.
    # train_val_split = np.random.rand(len(train_AP_features)) < training_ratio # mask index array
    # x_train = train_AP_features[train_val_split]
    # y_train = train_labels[train_val_split]
    # x_val = train_AP_features[~train_val_split]
    # y_val = train_labels[~train_val_split]

    # create a model
    model = KerasClassifier(build_fn=siso_dnn_classification, input_dim=num_aps,
                            output_dim=output_dim, hidden_layers=hidden_layers,
                            epochs=epochs, batch_size=batch_size, verbose=VERBOSE)

    # train and evaluate the model with k-fold cross validation
    startTime = timer()
    kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)
    # model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=epochs, verbose=VERBOSE)
    results = cross_val_score(model, rss, tv_labels, cv=kfold)
    elapsedTime = timer() - startTime
    print("Model trained in %e s." % elapsedTime)
    
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
    # # rfps_results = (np.equal(np.argmax(test_labels[:, 8:118], axis=1), np.argmax(preds[:, 8:118], axis=1))).astype(int)
    # # acc_rfp = rfps_results.mean()
    # # acc = (blds_results*flrs_results*rfps_results).mean()
    
    # # calculate positioning error when building and floor are correctly estimated
    # mask = np.logical_and(blds_results, flrs_results) # mask index array for correct location of building and floor
    # x_test_utm = x_test_utm[mask]
    # y_test_utm = y_test_utm[mask]
    # blds = blds[mask]
    # flrs = flrs[mask]
    # rfps = (preds[mask])[:, 8:118]

    # n_success = len(blds)       # number of correct building and floor location
    # # blds = np.greater_equal(blds, np.tile(np.amax(blds, axis=1).reshape(n_success, 1), (1, 3))).astype(int) # set maximum column to 1 and others to 0 (row-wise)
    # # flrs = np.greater_equal(flrs, np.tile(np.amax(flrs, axis=1).reshape(n_success, 1), (1, 5))).astype(int) # ditto

    # n_loc_failure = 0
    # sum_pos_err = 0.0
    # sum_pos_err_weighted = 0.0
    # idxs = np.argpartition(rfps, -N)[:, -N:]  # (unsorted) indexes of up to N nearest neighbors
    # threshold = scaling*np.amax(rfps, axis=1)
    # for i in range(n_success):
    #     xs = []
    #     ys = []
    #     ws = []
    #     for j in idxs[i]:
    #         rfp = np.zeros(110)
    #         rfp[j] = 1
    #         rows = np.where((train_labels == np.concatenate((blds[i], flrs[i], rfp))).all(axis=1)) # tuple of row indexes
    #         if rows[0].size > 0:
    #             if rfps[i][j] >= threshold[i]:
    #                 xs.append(training_df.loc[training_df.index[rows[0][0]], 'LONGITUDE'])
    #                 ys.append(training_df.loc[training_df.index[rows[0][0]], 'LATITUDE'])
    #                 ws.append(rfps[i][j])
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
    base_dir = '../results/' + os.path.splitext(os.path.basename(__file__))[0] + '/'
    base_file_name = base_dir \
                     + "E{0:d}_B{1:d}_S{2:s}".format(epoch, batch_size, args.sae_hidden_layers.replace(',', '-'))
                     # + '_T' + "{0:.2f}".format(args.training_ratio) \
    sae_model_file = base_file_name + '.hdf5'
    # output_file_base = base_file_name + '_C' + args.classifier_hidden_layers.replace(',', '-') \
    #              + '_D' + "{0:.2f}".format(dropout)
    now = datetime.datetime.now()
    output_file_base = base_file_name + "_D{0:.2f}_".format(dropout) + now.strftime("%Y%m%d-%H%M%S")

    with open(output_file_base + '.org', 'w') as output_file:
        output_file.write("#+STARTUP: showall\n")  # unfold everything when opening
        output_file.write("* System parameters\n")
        output_file.write("  - Random number seed: %d\n" % random_seed)
        # output_file.write("  - Ratio of training data to overall data: %.2f\n" % training_ratio)
        output_file.write("  - Epochs: %d\n" % epochs)
        output_file.write("  - Batch size: %d\n" % batch_size)
        output_file.write("  - Optimizer: %s\n" % OPTIMIZER)
        output_file.write("  - Hidden layers: %d" % hidden_layers[0])
        for units in hidden_layers[1:]:
            output_file.write("-%d" % units)
        output_file.write("\n")
        output_file.write("  - Dropout rate: %.2f\n" % dropout)
        output_file.write("* Performance\n")
        output_file.write("  - Accuracy: {0:.2f}%% ({1:.2f}%%)".format((100*results.mean()), (100*results.std())))
        # output_file.write("  - Loss (overall): %e\n" % results.losses.overall)
        # output_file.write("  - Accuracy (overall): %e\n" % results.accuracy.overall)
        # output_file.write("  - Building hit rate [%%]: %.2f\n" % (100*results.metrics.building_acc))
        # output_file.write("  - Floor hit rate [%%]: %.2f\n" % (100*results.metrics.floor_acc))
        # output_file.write("  - Building-floor hit rate [%%]: %.2f\n" % (100*results.metrics.bf_acc))
        # output_file.write("  - MSE (location): %e\n" % results.metrics.location_mse)
        # output_file.write("  - Mean error [m]: %.2f\n" % results.metrics.mean_error)  # according to EvAAL/IPIN 2015 competition rule
        # output_file.write("  - Median error [m]: %.2f\n" % results.metrics.median_error)  # ditto
