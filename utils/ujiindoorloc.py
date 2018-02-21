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
from collections import namedtuple


class UJIIndoorLoc(object):
    def __init__(self,
                 path='.',
                 frac=1.0,
                 rss_scaler=None,
                 utm_scaler=None,
                 classification_mode='hierarchical'):
        self.training_fname = path + '/' + 'trainingData2.csv'  # '-110' for the lack of AP.
        self.testing_fname = path + '/' + 'validationData2.csv'  # use validation data as testing data
        self.frac = frac
        self.rss_scaler = rss_scaler  # a scaler provided by sklearn.preprocessing module
        self.utm_scaler = utm_scaler  # ditto
        self.classification_mode = classification_mode

    def load_data(self):
        self.training_df = (pd.read_csv(self.training_fname, header=0)).sample(
            frac=self.frac
        )  # pass header=0 to be able to replace existing names
        self.testing_df = pd.read_csv(self.testing_fname, header=0)  # ditto

        col_aps = [col for col in self.training_df.columns if 'WAP' in col]
        num_aps = len(col_aps)

        # process RSS
        # N.B. double precision needed for proper working with scaler
        training_rss = np.asarray(self.training_df[col_aps], dtype=np.float)
        testing_rss = np.asarray(self.testing_df[col_aps], dtype=np.float)
        if self.rss_scaler != None:
            # scale numerical data (over their flattened versions for joint scaling)
            training_rss_scaled = (self.rss_scaler.fit_transform(
                training_rss.reshape((-1, 1)))).reshape(training_rss.shape)
            testing_rss_scaled = (self.rss_scaler.transform(
                testing_rss.reshape(
                    (-1, 1)))).reshape(testing_rss.shape)  # scaled version
        else:
            training_rss_scaled = training_rss
            testing_rss_scaled = testing_rss

        # process UTM coordinates
        training_utm_x = np.asarray(
            self.training_df['LONGITUDE'], dtype=np.float)
        training_utm_y = np.asarray(
            self.training_df['LATITUDE'], dtype=np.float)
        training_utm = np.column_stack((training_utm_x, training_utm_y))
        num_coords = training_utm.shape[1]
        testing_utm_x = np.asarray(
            self.testing_df['LONGITUDE'], dtype=np.float)
        testing_utm_y = np.asarray(self.testing_df['LATITUDE'], dtype=np.float)
        testing_utm = np.column_stack((testing_utm_x, testing_utm_y))
        if self.utm_scaler != None:
            training_utm_scaled = self.utm_scaler.fit_transform(training_utm)
            testing_utm_scaled = self.utm_scaler.transform(
                testing_utm)  # scaled version
        else:
            training_utm_scaled = training_utm
            testing_utm_scaled = testing_utm

        # map locations (reference points) to sequential IDs per building &
        # floor before building labels
        self.training_df['REFPOINT'] = self.training_df.apply(lambda row:
                                                    str(int(row['SPACEID'])) +
                                                    str(int(row['RELATIVEPOSITION'])),
                                                    axis=1) # add a new column
        blds = np.unique(self.training_df[['BUILDINGID']])
        flrs = np.unique(self.training_df[['FLOOR']])
        training_utm_avg = {}

        for bld in blds:
            n_rfps = 0
            sum_x = sum_y = 0.0
            for flr in flrs:
                # map reference points to sequential IDs per building-floor before building labels
                cond = (self.training_df['BUILDINGID'] == bld) & (
                    self.training_df['FLOOR'] == flr)
                if len(self.training_df.loc[cond]) > 0:
                    _, idx = np.unique(
                        self.training_df.loc[cond, 'REFPOINT'],
                        return_inverse=True)  # refer to numpy.unique manual
                    self.training_df.loc[cond, 'REFPOINT'] = idx

                    # calculate the average coordinates of each building/floor
                    df = self.training_df.loc[cond, [
                        'REFPOINT', 'LONGITUDE', 'LATITUDE'
                    ]].drop_duplicates(subset='REFPOINT')
                    x = np.mean(df.loc[cond, 'LONGITUDE'])
                    y = np.mean(df.loc[cond, 'LATITUDE'])
                    training_utm_avg[str(bld) + '-' + str(flr)] = np.array((x,
                                                                            y))
                    n = len(df)
                    sum_x += n * x
                    sum_y += n * y
                    n_rfps += n

            # calculate the average coordinates of each building
            training_utm_avg[str(bld)] = np.array((sum_x / n_rfps,
                                                   sum_y / n_rfps))

        # build labels for sequential multi-class classification of a building, a floor, and a location (reference point)
        num_training_samples = len(self.training_df)
        num_testing_samples = len(self.testing_df)
        labels_bld = np.asarray(
            pd.get_dummies(
                pd.concat([
                    self.training_df['BUILDINGID'],
                    self.testing_df['BUILDINGID']
                ])))  # for consistency in one-hot encoding for both dataframes
        labels_flr = np.asarray(
            pd.get_dummies(
                pd.concat(
                    [self.training_df['FLOOR'],
                     self.testing_df['FLOOR']])))  # ditto
        training_labels_bld = labels_bld[:num_training_samples]
        training_labels_flr = labels_flr[:num_training_samples]
        testing_labels_bld = labels_bld[num_training_samples:]
        testing_labels_flr = labels_flr[num_training_samples:]
        training_lables_loc = np.asarray(
            pd.get_dummies(self.training_df['REFPOINT']))
        # BUILDINGID: 3
        # FLOOR: 5
        # REFPOINT: 110
        # multi-label labels: array of 19937 x 905

        if self.classification_mode == 'hierarchical':
            TrainingData = namedtuple('TrainingData', [
                'rss', 'rss_scaled', 'utm', 'utm_avg', 'utm_scaled', 'labels'
            ])
            TrainingLabels = namedtuple('TrainingLabels',
                                        ['building', 'floor', 'location'])
            training_labels = TrainingLabels(
                building=training_labels_bld,
                floor=training_labels_flr,
                location=training_lables_loc)
            training_data = TrainingData(
                rss=training_rss,
                rss_scaled=training_rss_scaled,
                utm=training_utm,
                utm_avg=training_utm_avg,
                utm_scaled=training_utm_scaled,
                labels=training_labels)

            TestingData = namedtuple(
                'TestingData',
                ['rss', 'rss_scaled', 'utm', 'utm_scaled', 'labels'])
            TestingLabels = namedtuple('TestingLabels', ['building', 'floor'])
            testing_labels = TestingLabels(
                building=testing_labels_bld, floor=testing_labels_flr)
            testing_data = TestingData(
                rss=testing_rss,
                rss_scaled=testing_rss_scaled,
                utm=testing_utm,
                utm_scaled=testing_utm_scaled,
                labels=testing_labels)

        self.training_data = training_data
        self.testing_data = testing_data

        return self.training_df, self.training_data, self.testing_df, self.testing_data


if __name__ == "__main__":
    ### import basic modules and a model to test
    import os
    import platform
    if platform.system() == 'Windows':
        data_path = os.path.expanduser(
            '~kks/Research/Ongoing/localization/xjtlu_surf_indoor_localization/data/UJIIndoorLoc'
        )
    else:
        data_path = os.path.expanduser(
            '~kks/research/ongoing/localization/xjtlu_surf_indoor_localization/data/UJIIndoorLoc'
        )
    ### import other modules; keras and its backend will be loaded later
    import argparse
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    frac = args.frac
    preprocessor = args.preprocessor

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
    training_df, training_data, testing_df, testing_data = ujiindoorloc.load_data(
    )
