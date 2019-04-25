#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     data_analysis1.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-09-25
#
# @brief    Analyze UJIIndoorLoc data sets
#


import itertools
import numpy as np
import pandas as pd


train_df = pd.read_csv('../data/ujiindoorloc/trainingData.csv', header=0)
blds = np.unique(train_df[['BUILDINGID']])
flrs = np.unique(train_df[['FLOOR']])

max_aps = {}
keys = []
for bld in blds:
    for flr in flrs:
        df = (train_df[(train_df['BUILDINGID']==bld) & (train_df['FLOOR']==flr)]).iloc[:,:520]
        key = str(bld) + '-' + str(flr)  # dictionary key
        max_aps[key] = set(df.idxmax(axis=1))
        keys.append(key)

for pair in itertools.combinations(keys, 2):
    intersection = max_aps[pair[0]].intersection(max_aps[pair[1]])
    print("Intersection of %s and %s: %s" % (pair[0], pair[1], str(intersection)))
