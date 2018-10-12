#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     wap_to_building.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-10-12
#
# @brief Assign building ID to each WAP of the UJIIndoorLoc dataset
#


#import itertools
import numpy as np
import pandas as pd


# variables
N = 10                          # the number of largest RSS values to consider
                                # in determining WAP's building ID
lack_of_ap = -110               # RSS value indicating no detection of the
                                # corresponding AP


train_df = pd.read_csv('../data/ujiindoorloc/trainingData.csv', header=0)

blds = np.unique(train_df[['BUILDINGID']])
flrs = np.unique(train_df[['FLOOR']])

# replace 100 with lack_of_ap in RSS
waps = ['WAP'+"{0:0=3d}".format(i) for i in range(1, 521)]
train_df[waps] = train_df[waps].replace(100, -110)

wap_bld = {}
for wap in waps:
    rss = np.asarray(train_df[wap])
    row_idx = np.argsort(rss)[-N:]  # indexes of N largest RSS values

    # obtain the majority value
    (values, counts) = np.unique(train_df.loc[row_idx, 'BUILDINGID'], return_counts=True)
    wap_bld[wap] = values[np.argmax(counts)]

    print(wap + "--> Building {0:d}".format(wap_bld[wap]))
