#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     tut.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-07-24
#
# @brief A parser and preprocessor for the TUT Wi-Fi fingerprinting
#        dataset.
#
# @remarks For details of the TUT dataset, please refer [1].
#
#          [1] E. S. Lohan et al., "Wi-Fi crowdsourced fingerprinting dataset
#          for indoor positioning," Data, vol. 2, no. 4, article no. 32,
#          pp. 1-16, 2017. 

### import basic modules and a model to test
import os
import sys
### import other modules; keras and its backend will be loaded later
import cloudpickle  # for storing namedtuples
import numpy as np
import pandas as pd
import pathlib
from collections import namedtuple


class TUT(object):
    def __init__(self,
                 path='.',
                 frac=1.0,
                 preprocessor='standard_scaler',
                 classification_mode='hierarchical',
                 lack_of_ap=-110):
        self.training_fname = path + '/' + 'trainingData.csv'  # RSS=100 for lack of AP
        self.testing_fname = path + '/' + 'validationData.csv'  # validation data as testing data
        self.frac = frac
        if preprocessor == 'standard_scaler':
            from sklearn.preprocessing import StandardScaler
            self.rss_scaler = StandardScaler()
            self.utm_scaler = StandardScaler()
        elif preprocessor == 'minmax_scaler':
            from sklearn.preprocessing import MinMaxScaler
            self.rss_scaler = MinMaxScaler()
            self.utm_scaler = MinMaxScaler()
        elif preprocessor == 'normalizer':
            from sklearn.preprocessing import Normalizer
            self.rss_scaler = Normalizer()
            self.utm_scaler = Normalizer()
        else:
            print("{0:s} preprocessor is not supported.".format(preprocessor))
            sys.exit()
        self.classification_mode = classification_mode
        self.lack_of_ap = lack_of_ap
        self.saved_fname = path + '/saved/ujiindoorloc' + '_F{0:.1f}_L{1:d}_P{2:s}.cpkl'.format(
            frac, lack_of_ap,
            preprocessor)  # cloudpickle file name for saved objects

    def load_data(self):
        if os.path.isfile(self.saved_fname) and (os.path.getmtime(
                self.saved_fname) > os.path.getmtime(__file__)):
            with open(self.saved_fname, 'rb') as input_file:
                self.training_df = cloudpickle.load(input_file)
                self.training_data = cloudpickle.load(input_file)
                self.testing_df = cloudpickle.load(input_file)
                self.testing_data = cloudpickle.load(input_file)
        else:
            # training data
            FILE_NAME_TRAIN_RSS = path_to_data + '/Training_rss_21Aug17.csv'
            FILE_NAME_TRAIN_COORDS = path_to_data + '/Training_coordinates_21Aug17.csv'
            # read training data
            X_train = genfromtxt(FILE_NAME_TRAIN_RSS, delimiter=',')
            y_train = genfromtxt(FILE_NAME_TRAIN_COORDS, delimiter=',')
            X_train[X_train==100] = np.nan

            # test data
            FILE_NAME_TEST_RSS = path_to_data + '/Test_rss_21Aug17.csv'
            FILE_NAME_TEST_COORDS = path_to_data + '/Test_coordinates_21Aug17.csv'
            # read test data
            X_test = genfromtxt(FILE_NAME_TEST_RSS, delimiter=',')
            y_test = genfromtxt(FILE_NAME_TEST_COORDS, delimiter=',')
            X_test[X_test==100] = np.nan
            return (X_train, y_train, X_test, y_test)

    def load_data(self):
        if os.path.isfile(self.saved_fname) and (os.path.getmtime(
                self.saved_fname) > os.path.getmtime(__file__)):
            with open(self.saved_fname, 'rb') as input_file:
                self.training_df = cloudpickle.load(input_file)
                self.training_data = cloudpickle.load(input_file)
                self.testing_df = cloudpickle.load(input_file)
                self.testing_data = cloudpickle.load(input_file)
        else:
            self.training_df = (pd.read_csv(
                self.training_fname, header=0)).sample(
                    frac=self.frac
                )  # pass header=0 to be able to replace existing names
            self.testing_df = pd.read_csv(
                self.testing_fname, header=0)  # ditto

            col_aps = [col for col in self.training_df.columns if 'WAP' in col]
            num_aps = len(col_aps)

            # process RSS
            # N.B. double precision needed for proper working with scaler
            training_rss = np.asarray(
                self.training_df[col_aps], dtype=np.float)
            training_rss[training_rss ==
                         100] = self.lack_of_ap  # RSS value for lack of AP
            testing_rss = np.asarray(self.testing_df[col_aps], dtype=np.float)
            testing_rss[testing_rss ==
                        100] = self.lack_of_ap  # RSS value for lack of AP
            if self.rss_scaler != None:
                # scale over flattened data for joint scaling
                training_rss_scaled = (self.rss_scaler.fit_transform(
                    training_rss.reshape((-1, 1)))).reshape(
                        training_rss.shape)
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
            testing_utm_y = np.asarray(
                self.testing_df['LATITUDE'], dtype=np.float)
            testing_utm = np.column_stack((testing_utm_x, testing_utm_y))
            if self.utm_scaler != None:
                training_utm_scaled = self.utm_scaler.fit_transform(
                    training_utm)
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
                            return_inverse=True
                        )  # refer to numpy.unique manual
                        self.training_df.loc[cond, 'REFPOINT'] = idx

                        # calculate the average coordinates of each building/floor
                        df = self.training_df.loc[cond, [
                            'REFPOINT', 'LONGITUDE', 'LATITUDE'
                        ]].drop_duplicates(subset='REFPOINT')
                        x = np.mean(df.loc[cond, 'LONGITUDE'])
                        y = np.mean(df.loc[cond, 'LATITUDE'])
                        training_utm_avg[str(bld) + '-' + str(flr)] = np.array(
                            (x, y))
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
                    ]))
            )  # for consistency in one-hot encoding for both dataframes
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
                    'rss', 'rss_scaled', 'rss_scaler', 'utm', 'utm_avg',
                    'utm_scaled', 'utm_scaler', 'labels'
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
                    rss_scaler=self.rss_scaler,
                    utm=training_utm,
                    utm_avg=training_utm_avg,
                    utm_scaled=training_utm_scaled,
                    utm_scaler=self.utm_scaler,
                    labels=training_labels)

                TestingData = namedtuple(
                    'TestingData',
                    ['rss', 'rss_scaled', 'utm', 'utm_scaled', 'labels'])
                TestingLabels = namedtuple('TestingLabels',
                                           ['building', 'floor'])
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

            pathlib.Path(os.path.dirname(self.saved_fname)).mkdir(
                parents=True, exist_ok=True)
            with open(self.saved_fname, 'wb') as output_file:
                cloudpickle.dump(self.training_df, output_file)
                cloudpickle.dump(self.training_data, output_file)
                cloudpickle.dump(self.testing_df, output_file)
                cloudpickle.dump(self.testing_data, output_file)

        return self.training_df, self.training_data, self.testing_df, self.testing_data


if __name__ == "__main__":
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
    parser.add_argument(
        "-L",
        "--lack_of_ap",
        help="RSS value for indicating lack of AP; default is -110",
        default=-110,
        type=int)
    args = parser.parse_args()
    frac = args.frac
    preprocessor = args.preprocessor
    lack_of_ap = args.lack_of_ap

    ### load dataset after scaling
    data_path = '../data/tut'
    print("Loading TUT data ...")
    tut = TUT(
        path='../data/tut',
        frac=frac,
        preprocessor=preprocessor,
        classification_mode='hierarchical',
        lack_of_ap=lack_of_ap)
    training_df, training_data, testing_df, testing_data = tut.load_data(
    )
