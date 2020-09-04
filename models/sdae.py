#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     sdae.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-02-21
#
# @brief A stacked denoising autoencoder (SDAE) for pretraining of hidden layers
#        in Wi-Fi fingerprinting.
#
# @remarks It is inspired by the implementation by Ben Ogorek@kaggle [1]
#
#          [1] Ben Ogorek, Autoencoder with greedy layer-wise pretraining,
#              Available online:
#              https://www.kaggle.com/baogorek/autoencoder-with-greedy-layer-wise-pretraining/notebook

import os
import numpy as np
import pathlib
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, load_model


def masking_noise(x, corruption_level):
    x_corrupted = x
    x_corrupted[np.random.rand(len(x)) < corruption_level] = 0.0
    return x_corrupted


def sdae(dataset='tut',
         input_data=None,
         preprocessor='standard_scaler',
         hidden_layers=[],
         cache=False,
         model_fname=None,
         optimizer='nadam',
         corruption_level=0.1,
         batch_size=32,
         epochs=100,
         validation_split=0.0):
    """Stacked denoising autoencoder

    Keyword arguments:
    dataset -- a data set for training, validation, and testing; choices are 'tut' (default), 'tut2', 'tut3', and 'ujiindoorloc'
    input_data -- two-dimensional array of RSSs
    preprocessor -- preprocessor used to scale/normalize the original input data (information only)
    hidden_layers -- list of numbers of units in SDAE hidden layers
    cache -- whether to load a trained model from/save it to a cache
    model_fname -- full path name for SDAE model load & save
    optimizer -- optimizer for training
    corruption_level -- corruption level of masking noise
    batch_size -- size of batch
    epochs -- number of epochs
    validation_split -- fraction of training data to be used as validation data
    """

    if (preprocessor == 'standard_scaler') or (preprocessor == 'normalizer'):
        loss = 'mean_squared_error'
    elif preprocessor == 'minmax_scaler':
        loss = 'binary_crossentropy'
    else:
        print("{0:s} preprocessor is not supported.".format(preprocessor))
        sys.exit()

    if cache:
        if model_fname is None:
            model_fname = './saved/sdae/' + dataset + '/H' \
                          + '-'.join(map(str, hidden_layers)) \
                          + "_B{0:d}_E{1:d}_L{2:s}_P{3:s}".format(batch_size,epochs, loss, preprocessor) \
                          + '.h5'
        if os.path.isfile(model_fname) and (os.path.getmtime(model_fname) >
                                            os.path.getmtime(__file__)):
            # # below are the workaround from oarriaga@GitHub: https://github.com/keras-team/keras/issues/4044
            # model = load_model(model_fname, compile=False)
            # model.compile(optimizer=SDAE_OPTIMIZER, loss=SDAE_LOSS)
            return load_model(model_fname)

    # each layer is named explicitly to avoid any conflicts in
    # model.compile() by models using SDAE

    input_dim = input_data.shape[1]  # number of RSSs per sample
    input = Input(shape=(input_dim, ), name='sdae_input')

    encoded_input = []
    distorted_input = []
    encoded = []
    # encoded_bn = []
    decoded = []
    autoencoder = []
    encoder = []
    x = input_data
    n_hl = len(hidden_layers)
    all_layers = [input_dim] + hidden_layers
    for i in range(n_hl):
        encoded_input.append(
            Input(
                shape=(all_layers[i], ),
                name='sdae_encoded_input' + str(i)))
        encoded.append(
            Dense(all_layers[i + 1],
                  activation='sigmoid')(encoded_input[i]))
        decoded.append(
            Dense(all_layers[i], activation='sigmoid')(encoded[i]))
        autoencoder.append(
            Model(inputs=encoded_input[i], outputs=decoded[i]))
        # encoder.append(
        #     Model(inputs=encoded_input[i], outputs=encoded_bn[i]))
        encoder.append(Model(inputs=encoded_input[i], outputs=encoded[i]))
        autoencoder[i].compile(optimizer=optimizer, loss=loss)
        encoder[i].compile(optimizer=optimizer, loss=loss)
        autoencoder[i].fit(
            x=masking_noise(x, corruption_level),
            y=x,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            shuffle=True)
        x = encoder[i].predict(x)

    x = input
    for i in range(n_hl):
        x = encoder[i](x)
    output = x
    model = Model(inputs=input, outputs=output)
    model.compile(optimizer=optimizer, loss=loss)

    # # set all layers (i.e., SDAE encoder) to non-trainable (weights will not be updated)
    # # N.B. the effect of freezing seems to be negative
    # for layer in model.layers[:]:
    #     layer.trainable = False

    if cache:
        pathlib.Path(os.path.dirname(model_fname)).mkdir(
            parents=True, exist_ok=True)
        model.save(model_fname)  # save for later use

        with open(os.path.splitext(model_fname)[0] + '.org',
                  'w') as output_file:
            model.summary(print_fn=lambda x: output_file.write(x + '\n'))
            # output_file.write(
            #     "Training loss: %.4e\n" % history.history['loss'][-1])
            # output_file.write(
            #     "Validation loss: %.4ef\n" % history.history['val_loss'][-1])

    return model


if __name__ == "__main__":
    # import basic modules and a model to test
    os.environ['PYTHONHASHSEED'] = '0'  # for reproducibility
    import sys
    sys.path.insert(0, '../utils')
    from ujiindoorloc import UJIIndoorLoc
    # import other modules; keras and its backend will be loaded later
    import argparse
    import random as rn
    # import keras and its backend (e.g., tensorflow)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # supress warning messages
    # session_conf = tf.ConfigProto(
    #     intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    # )  # force TF to use single thread for reproducibility
    # from keras import backend as K

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
        help="number of epochs; default is 20",
        default=20,
        type=int)
    parser.add_argument(
        "--no_cache",
        help=
        "disable loading a trained model from/saving it to a cache",
        action='store_true')
    parser.add_argument(
        "--validation_split",
        help=
        "fraction of training data to be used as validation data: default is 0.2",
        default=0.2,
        type=float)
    parser.add_argument(
        "-H",
        "--hidden_layers",
        help=
        "comma-separated numbers of units in SDAE hidden layers; default is '128,128,128'",
        default='128,128,128',
        type=str)
    parser.add_argument(
        "-F",
        "--frac",
        help=
        "fraction of input data to load for training and validation; default is 1.0",
        default=1.0,
        type=float)
    parser.add_argument(
        "-P",
        "--preprocessor",
        help=
        "preprocessor to scale/normalize input data before training and validation; default is 'standard_scaler'",
        default='standard_scaler',
        type=str)
    parser.add_argument(
        "-O",
        "--optimizer",
        help="optimizer for training; default is 'nadam'",
        default='nadam',
        type=str)
    parser.add_argument(
        "-C",
        "--corruption_level",
        help="corruption level of masking noise; default is 0.1",
        default=0.1,
        type=float)
    args = parser.parse_args()

    gpu_id = args.gpu_id
    random_seed = args.random_seed
    batch_size = args.batch_size
    epochs = args.epochs
    validation_split = args.validation_split
    hidden_layers = [int(i) for i in (args.hidden_layers).split(',')]
    cache = not args.no_cache
    frac = args.frac
    preprocessor = args.preprocessor
    optimizer = args.optimizer
    corruption_level = args.corruption_level

    # initialize numpy, random, TensorFlow, and keras
    np.random.seed(random_seed)
    rn.seed(random_seed)
    tf.random.set_seed(random_seed)
    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ''

    # load dataset after scaling
    print("Loading UJIIndoorLoc data ...")

    ujiindoorloc = UJIIndoorLoc(
        path='../data/ujiindoorloc', frac=frac, preprocessor=preprocessor)
    _, training_data, _, _ = ujiindoorloc.load_data()

    # build SDAE model
    print("Buidling SDAE model ...")
    model = sdae(
        training_data.rss_scaled,
        preprocessor=preprocessor,
        hidden_layers=hidden_layers,
        cache=cache,
        model_fname=None,
        optimizer=optimizer,
        corruption_level=corruption_level,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split)
    print(model.summary())
