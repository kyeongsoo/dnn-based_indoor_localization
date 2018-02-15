#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     siso_hl.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-02-15
#
# @brief Build a single-input and single-output (SIMO) deep neural network (DNN)
#        model with a base model, if provided, and a given hidden layer
#        structure.

from keras.layers import Activation, Dense, Dropout, Input
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def siso_hl(input,
            base_model=None,
            hidden_layers=[],
            optimizer='nadam',
            dropout=0.0) -> 'model':

    if base_model != None:
        x = BatchNormalization()(base_model(input))
    else:
        x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Dropout(dropout)(x)

    for units in hidden_layers:
        x = Dense(units, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout)(x)

    model = Model(inputs=input, outputs=x)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model
