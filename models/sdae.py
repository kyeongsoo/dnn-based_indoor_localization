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
import pathlib
### import keras and its backend (e.g., tensorflow)
from keras.layers import Activation, Dense, Dropout, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential, load_model


def sdae(input_data=None,
         preprocessor='standard_scaler',
         hidden_layers=[],
         model_fname=None,
         optimizer='nadam',
         loss='mse',
         batch_size=32,
         epochs=100,
         validation_split=0.0):
    """Deep autoencoder

    Keyword arguments:
    input_data -- two-dimensional array of RSSs
    preprocessor -- preprocessor used to scale/normalize input data (information only)
    hidden_layers -- list of numbers of units in SDAE hidden layers
    model_fname -- full path name for SDAE model load & save
    optimizer -- optimizer for training
    loss -- loss function for training
    batch_size -- size of batch
    epochs -- number of epochs
    validation_split -- fraction of training data to be used as validation data
    """

    if model_fname == None:
        model_fname = './saved/sdae_H' + '-'.join(map(
            str, hidden_layers)) + "_B{0:d}_E{1:d}_L{2:s}_P{3:s}".format(
                batch_size, epochs, loss, preprocessor) + '.hdf5'

    if os.path.isfile(model_fname) and (os.path.getmtime(model_fname) >
                                        os.path.getmtime(__file__)):
        model = load_model(model_fname)
        # # below are the workaround from oarriaga@GitHub: https://github.com/keras-team/keras/issues/4044
        # model = load_model(model_fname, compile=False)
        # model.compile(optimizer=SDAE_OPTIMIZER, loss=SDAE_LOSS)
    else:
        # each layer is named explicitly to avoid any conflicts in
        # model.compile() by models using SDAE

        # TODO: generlize the following
        # Layer by layer pretraining Models

        # Layer 1
        input_img = Input(shape=(784, ))
        distorted_input1 = Dropout(.1)(input_img)
        encoded1 = Dense(800, activation='sigmoid')(distorted_input1)
        encoded1_bn = BatchNormalization()(encoded1)
        decoded1 = Dense(784, activation='sigmoid')(encoded1_bn)

        autoencoder1 = Model(input=input_img, output=decoded1)
        encoder1 = Model(input=input_img, output=encoded1_bn)

        # Layer 2
        encoded1_input = Input(shape=(800, ))
        distorted_input2 = Dropout(.2)(encoded1_input)
        encoded2 = Dense(400, activation='sigmoid')(distorted_input2)
        encoded2_bn = BatchNormalization()(encoded2)
        decoded2 = Dense(800, activation='sigmoid')(encoded2_bn)

        autoencoder2 = Model(input=encoded1_input, output=decoded2)
        encoder2 = Model(input=encoded1_input, output=encoded2_bn)

        # Layer 3 - which we won't end up fitting in the interest of time
        encoded2_input = Input(shape=(400, ))
        distorted_input3 = Dropout(.3)(encoded2_input)
        encoded3 = Dense(200, activation='sigmoid')(distorted_input3)
        encoded3_bn = BatchNormalization()(encoded3)
        decoded3 = Dense(400, activation='sigmoid')(encoded3_bn)

        autoencoder3 = Model(input=encoded2_input, output=decoded3)
        encoder3 = Model(input=encoded2_input, output=encoded3_bn)

        # Deep Autoencoder
        encoded1_da = Dense(800, activation='sigmoid')(input_img)
        encoded1_da_bn = BatchNormalization()(encoded1_da)
        encoded2_da = Dense(400, activation='sigmoid')(encoded1_da_bn)
        encoded2_da_bn = BatchNormalization()(encoded2_da)
        encoded3_da = Dense(200, activation='sigmoid')(encoded2_da_bn)
        encoded3_da_bn = BatchNormalization()(encoded3_da)
        decoded3_da = Dense(400, activation='sigmoid')(encoded3_da_bn)
        decoded2_da = Dense(800, activation='sigmoid')(decoded3_da)
        decoded1_da = Dense(784, activation='sigmoid')(decoded2_da)

        deep_autoencoder = Model(input=input_img, output=decoded1_da)

        # Not as Deep Autoencoder
        nad_encoded1_da = Dense(800, activation='sigmoid')(input_img)
        nad_encoded1_da_bn = BatchNormalization()(nad_encoded1_da)
        nad_encoded2_da = Dense(400, activation='sigmoid')(nad_encoded1_da_bn)
        nad_encoded2_da_bn = BatchNormalization()(nad_encoded2_da)
        nad_decoded2_da = Dense(800, activation='sigmoid')(nad_encoded2_da_bn)
        nad_decoded1_da = Dense(784, activation='sigmoid')(nad_decoded2_da)

        nad_deep_autoencoder = Model(input=input_img, output=nad_decoded1_da)
        sgd1 = SGD(lr=5, decay=0.5, momentum=.85, nesterov=True)
        sgd2 = SGD(lr=5, decay=0.5, momentum=.85, nesterov=True)
        sgd3 = SGD(lr=5, decay=0.5, momentum=.85, nesterov=True)

        autoencoder1.compile(loss='binary_crossentropy', optimizer=sgd1)
        autoencoder2.compile(loss='binary_crossentropy', optimizer=sgd2)
        autoencoder3.compile(loss='binary_crossentropy', optimizer=sgd3)

        encoder1.compile(loss='binary_crossentropy', optimizer=sgd1)
        encoder2.compile(loss='binary_crossentropy', optimizer=sgd1)
        encoder3.compile(loss='binary_crossentropy', optimizer=sgd1)

        deep_autoencoder.compile(loss='binary_crossentropy', optimizer=sgd1)
        nad_deep_autoencoder.compile(
            loss='binary_crossentropy', optimizer=sgd1)

        # model = Sequential()
        # input_dim = input_data.shape[1]  # number of RSSs per sample
        # model.add(
        #     Dense(
        #         hidden_layers[0],
        #         input_dim=input_dim,
        #         activation='relu',
        #         name='sdae_hidden_1'))
        # n_hl = 1
        # for units in hidden_layers[1:]:
        #     n_hl += 1
        #     model.add(
        #         Dense(
        #             units, activation='relu', name='sdae_hidden_' + str(n_hl)))
        # model.add(Dense(input_dim, activation='sigmoid', name='sdae_output'))
        # model.compile(optimizer=optimizer, loss=loss)

        # history = model.fit(
        #     input_data,
        #     input_data,
        #     batch_size=batch_size,
        #     epochs=epochs,
        #     validation_split=validation_split,
        #     shuffle=True)

        # # remove the decoder part
        # num_to_remove = (len(hidden_layers) + 1) // 2
        # for i in range(num_to_remove):
        #     model.pop()

        # # # set all layers (i.e., SDAE encoder) to non-trainable (weights will not be updated)
        # # # N.B. the effect of freezing seems to be negative
        # # for layer in model.layers[:]:
        # #     layer.trainable = False

        # pathlib.Path(os.path.dirname(model_fname)).mkdir(
        #     parents=True, exist_ok=True)
        # model.save(model_fname)  # save for later use

        # with open(os.path.splitext(model_fname)[0] + '.org',
        #           'w') as output_file:
        #     model.summary(print_fn=lambda x: output_file.write(x + '\n'))
        #     output_file.write(
        #         "Training loss: %.4e\n" % history.history['loss'][-1])
        #     output_file.write(
        #         "Validation loss: %.4ef\n" % history.history['val_loss'][-1])

    return model


if __name__ == "__main__":
    ### import basic modules and a model to test
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
    from ujiindoorloc import UJIIndoorLoc
    ### import other modules; keras and its backend will be loaded later
    import argparse
    import numpy as np
    import random as rn
    ### import keras and its backend (e.g., tensorflow)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # supress warning messages
    import tensorflow as tf
    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )  # force TF to use single thread for reproducibility
    from keras import backend as K

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
        help="batch size; default is 10",
        default=10,
        type=int)
    parser.add_argument(
        "-E",
        "--epochs",
        help="number of epochs; default is 20",
        default=20,
        type=int)
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
        "comma-separated numbers of units in SDAE hidden layers; default is '128,32,128'",
        default='128,32,128',
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
        "-L",
        "--loss",
        help="loss function for training; default is 'mse'",
        default='mse',
        type=str)
    args = parser.parse_args()

    gpu_id = args.gpu_id
    random_seed = args.random_seed
    batch_size = args.batch_size
    epochs = args.epochs
    validation_split = args.validation_split
    hidden_layers = [int(i) for i in (args.hidden_layers).split(',')]
    frac = args.frac
    preprocessor = args.preprocessor
    optimizer = args.optimizer
    loss = args.loss

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
    print("Loading UJIIndoorLoc data ...")

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
    _, training_data, _, _ = ujiindoorloc.load_data()

    ### build SDAE model
    print("Buidling SDAE model ...")
    model = sdae(
        training_data.rss_scaled,
        preprocessor=preprocessor,
        hidden_layers=hidden_layers,
        model_fname=None,
        optimizer=optimizer,
        loss=loss,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split)
    print(model.summary())
