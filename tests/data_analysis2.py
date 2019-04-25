#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     data_analysis2.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-09-25
#
# @brief    Analyze UJIIndoorLoc data sets
#


#import itertools
import numpy as np
import pandas as pd

# train_df = (pd.read_csv('../data/UJIIndoorLoc/trainingData2.csv', header=0))[:19930]
train_df = pd.read_csv('../data/ujiindoorloc/trainingData.csv', header=0)

blds = np.unique(train_df[['BUILDINGID']])
flrs = np.unique(train_df[['FLOOR']])
# spcs = np.unique(train_df[['SPACEID']])
# rps = np.unique(train_df[['RELATIVEPOSITION']])

# num_blds = len(blds)
# num_flrs = len(flrs)
spcs = {}
for bld in blds:
    for flr in flrs:
        key = str(bld) + '-' + str(flr)
        spcs[key] = np.unique(
            (train_df[(train_df['BUILDINGID'] == bld)
                      & (train_df['FLOOR'] == flr)])[['SPACEID']])
        print("Number of spaces in (%s): %d" % (key, len(spcs[key])))
        print("Intersection of (0-0) and (%s)" % (key))
        print(np.intersect1d(spcs['0-0'], spcs[key]))
