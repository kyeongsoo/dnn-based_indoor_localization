#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     ujiindoorloc.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-02-14
#
# @brief A parser and preprocessor for the UJIIndoorLoc WLAN fingerprinting
#        database.
#
# @remarks For details of the UJIIndoorLoc database, please refer [1].
#
#          [1] J. Torres-Sospedra et al., "UJIIndoorLoc: A new multi-building
#              and multi-floor database for WLAN fingerprint-based indoor
#              localization problems," Proc. International Conference on Indoor
#              Positioning and Indoor Navigation (IPIN), Busan, Korea,
#              pp. 261-270, Oct. 2014.

### import basic modules and a model to test
import numpy as np
import pandas as pd

class UJIIndoorLoc(object):
    def __init__(self, path='.', scale=True):
        self.training_fname = path + '/' + 'trainingData2.csv'  # '-110' for the lack of AP.
        self.testing_fname = path + '/' + 'validationData2.csv'  # use validation data as testing data
        self.scale=scale

    def load_data():
        self.training_df = pd.read_csv(training_fname, header=0) # pass header=0 to be able to replace existing names
        self.testing_df = pd.read_csv(testing_fname, header=0)  # ditto
        
        col_aps = [col for col in training_df.columns if 'WAP' in col]
        num_aps = len(col_aps)
        rss = np.asarray(training_df[col_aps], dtype=np.float32)
        utm_x = np.asarray(training_df['LONGITUDE'], dtype=np.float32)
        utm_y = np.asarray(training_df['LATITUDE'], dtype=np.float32)
        utm = np.column_stack((utm_x, utm_y))
        num_coords = utm.shape[1]
        if scale == True:
            # scale numerical data (over their flattened versions for joint scaling)
            rss_scaler = StandardScaler()  # the same scaling will be applied to test data later
            utm_scaler = StandardScaler()  # ditto
            rss = (rss_scaler.fit_transform(rss.reshape((-1, 1)))).reshape(rss.shape)
            utm = utm_scaler.fit_transform(utm)

        # map locations (i.e., reference points) to sequential IDs per building &
        # floor before building labels
        training_df['REFPOINT'] = training_df.apply(lambda row:
                                                    str(int(row['SPACEID'])) +
                                                    str(int(row['RELATIVEPOSITION'])),
                                                    axis=1) # add a new column
        blds = np.unique(training_df[['BUILDINGID']])
        flrs = np.unique(training_df[['FLOOR']])
        x_avg = {}
        y_avg = {}
        for bld in blds:
            for flr in flrs:
                # map reference points to sequential IDs per building-floor before building labels
                cond = (training_df['BUILDINGID'] == bld) & (
                    training_df['FLOOR'] == flr)
                _, idx = np.unique(
                    training_df.loc[cond, 'REFPOINT'],
                    return_inverse=True)  # refer to numpy.unique manual
                training_df.loc[cond, 'REFPOINT'] = idx

                # calculate the average coordinates of each building/floor
                x_avg[str(bld) + '-' + str(flr)] = np.mean(
                    training_df.loc[cond, 'LONGITUDE'])
                y_avg[str(bld) + '-' + str(flr)] = np.mean(
                    training_df.loc[cond, 'LATITUDE'])

        # build labels for sequential multi-class classification of a building, a floor, and a location (reference point)
        num_training_samples = len(training_df)
        num_testing_samples = len(testing_df)
        bld_labels = np.asarray(pd.get_dummies(training_df['BUILDINGID']))
        flr_labels = np.asarray(pd.get_dummies(training_df['FLOOR']))
        loc_labels = np.asarray(pd.get_dummies(training_df['REFPOINT']))
        # tv_labels = np.asarray(pd.get_dummies(
        #     blds + '-' + flrs + '-' + locs))  # labels for training/validation
        # labels is an array of 19937 x 905
        # - 3 for BUILDINGID
        # - 5 for FLOOR,
        # - 110 for REFPOINT
        # output_dim = tv_labels.shape[1]

        # # split the training set into training and validation sets; we will use the
        # # validation set at a testing set.
        # train_val_split = np.random.rand(len(train_AP_features)) < training_ratio # mask index array
        # x_train = train_AP_features[train_val_split]
        # y_train = train_labels[train_val_split]
        # x_val = train_AP_features[~train_val_split]
        # y_val = train_labels[~train_val_split]
