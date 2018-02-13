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

from keras.engine.topology import Input
from keras.layers import Activation, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def siso_classification(input_dim=512,
                        output_dim=905,
                        hidden_layers=[],
                        optimizer='adam',
                        dropout=0.0):
    input = Input(shape=(input_dim, ), name='input')
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
    output = Activation(
        'softmax', name='output')(x)  # no dropout for an output layer

    model = Model(inputs=input, outputs=output)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model
