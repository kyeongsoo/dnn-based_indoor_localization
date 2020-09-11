#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     simo_rnn_tut.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2020-09-07
#
# @brief A hierarchical indoor localization system based on Wi-Fi
#        fingerprinting using a stacked denoising autoencoder (SDAE)
#        and a single-input multi-output (SIMO) recurrent neural network
#        (RNN) with TUT dataset. PyTorch version.
#
# @remarks The results are submitted to <a
#          href="https://is-candar.org/GCA20/">The 5th International
#          Workshop on GPU Computing and AI (GCA'20)</a>.


import os
import sys
import pathlib
import argparse
import datetime
import numpy as np
from collections import namedtuple
from num2words import num2words
from numpy.linalg import norm
from sklearn.metrics import accuracy_score
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchsummary import summary
# user modules
sys.path.insert(0, '../models')
from sdae_pt import sdae_pt
sys.path.insert(0, '../utils')
from mean_ci import mean_ci
from tut import TUT


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TutDataset(Dataset):
    """Convert TUT training dataset to a PyTorch dataset."""

    def __init__(self, tut):
        self.rss = tut.rss_scaled.astype('float32')
        # convert one-hot encoded labels to class-index-based ones
        self.floor = np.argmax(tut.labels.floor, axis=1)
        self.coord = tut.coord_scaled.astype('float32')

    def __len__(self):
        return (len(self.rss))

    def __getitem__(self, idx):
        return (self.rss[idx], self.floor[idx], self.coord[idx])


class SimoRnn(nn.Module):
    """ SIMO RNN for hierarchical indoor localization."""

    def __init__(self, input_size, hidden_size, floor_size, coord_size):
        super(SimoRnn, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size)
        self.fnn_floor = nn.Linear(hidden_size, floor_size)
        self.fnn_coord = nn.Linear(hidden_size, coord_size)

    def forward(self, input, hidden):
        output, hidden = self.rnn(input, hidden)
        output_floor = self.fnn_floor(output.view(1, -1))
        # output_floor = F.softmax(output, dim=1)

        output, hidden = self.rnn(input, hidden)
        output_coord = self.fnn_coord(output.view(1, -1))
        # output_coord = F.linear(output)

        return output_floor, output_coord, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class SdaeSimoRnn(nn.Module):
    def __init__(self, sdae, rnn):
        super(SdaeSimoRnn, self).__init__()
        self.sdae = sdae
        self.rnn = rnn
        self.sdae_output_size = self.sdae.fc2.out_features

    def forward(self, input, hidden):
        x = self.sdae(input)
        return self.rnn(x.view(-1, 1, self.sdae_output_size), hidden)

    def initHidden(self):
        return self.rnn.initHidden()


def simo_rnn_tut_pt(
        frac: float,
        validation_split: float,
        preprocessor: str,
        batch_size: int,
        epochs: int,
        optimizer: str,
        dropout: float,
        corruption_level: float,
        dae_hidden_layers: list,
        sdae_hidden_layers: list,
        cache: bool,
        common_hidden_layers: list,
        floor_hidden_layers: list,
        coordinates_hidden_layers: list,
        rnn_hidden_size: int,
        floor_weight: float,
        coordinates_weight: float,
        verbose: int
):
    """Multi-building and multi-floor indoor localization based on hybrid
    buidling/floor classification and coordinates regression using SDAE and
    SIMO RNN and TUT dataset.

    Keyword arguments:

    """

    # load datasets after scaling
    print("Loading the data ...")
    tut = TUT(
        cache=cache,
        frac=frac,
        preprocessor=preprocessor,
        classification_mode='hierarchical')
    flr_height = tut.floor_height
    # training_df = tut.training_df
    training_data = tut.training_data
    # testing_df = tut.testing_df
    testing_data = tut.testing_data

    print("Building the model ...")
    rss = training_data.rss_scaled
    coord = training_data.coord_scaled
    coord_scaler = training_data.coord_scaler  # for inverse transform
    labels = training_data.labels
    rss_size = rss.shape[1]
    floor_size = labels.floor.shape[1]
    coord_size = coord.shape[1]

    if sdae_hidden_layers != '':
        sdae = sdae_pt(
            dataset='tut',
            input_data=rss,
            preprocessor=preprocessor,
            hidden_layers=sdae_hidden_layers,
            cache=cache,
            model_fname=None,
            optimizer=optimizer,
            corruption_level=corruption_level,
            batch_size=batch_size,
            # epochs=epochs,
            epochs=300,
            validation_split=validation_split)
        rnn = SimoRnn(
            input_size=sdae_hidden_layers[-1],
            hidden_size=rnn_hidden_size,
            floor_size=floor_size,
            coord_size=coord_size)
        model = SdaeSimoRnn(sdae, rnn)
    else:
        model = SimoRnn(
            input_size=rss_size,
            hidden_size=rnn_hidden_size,
            floor_size=floor_size,
            coord_size=coord_size)

    print("Training the model ...", end='')
    startTime = timer()
    # N.B.: CrossEntropyLoss combines nn.LogSoftmax() and nn.NLLLoss() in one
    # single class. So we don't need softmax activation function in
    # classification.
    criterion_floor = nn.CrossEntropyLoss()
    criterion_coord = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    dataset = TutDataset(tut.training_data)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for rss, floor, coord in dataloader:
            hidden = model.initHidden()

            # move data to GPU if available
            hidden = hidden.to(device, non_blocking=True)
            rss = rss.to(device, non_blocking=True)
            floor = floor.to(device, non_blocking=True)
            coord = coord.to(device, non_blocking=True)
            optimizer.zero_grad()

            # run the model recursively twice for floor and location
            for _ in range(2):
                output_floor, output_coord, hidden = model(rss, hidden)

            loss = floor_weight*criterion_floor(output_floor, floor)
            loss += coordinates_weight*criterion_coord(output_coord, coord)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print("[Epoch {0:3d}] loss: {1:.3f}".format(epoch+1, running_loss/len(dataloader)))

    elapsedTime = timer() - startTime
    print(" completed in {0:.4e} s".format(elapsedTime))

    print("Evaluating the model ...")
    model.eval()
    rss = testing_data.rss_scaled
    flrs = np.argmax(testing_data.labels.floor, axis=1)
    coord = testing_data.coord  # original coordinates

    dataset = TutDataset(tut.testing_data)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # calculate the classification accuracies and localization errors
    flrs_pred = list()
    coords_scaled_pred = list()
    for rss, _, _ in dataloader:
        hidden = model.initHidden()

        # move data to GPU if available
        hidden = hidden.to(device, non_blocking=True)
        rss = rss.to(device, non_blocking=True)

        # run the model recursively twice for floor and location
        for _ in range(2):
            output_floor, output_coord, hidden = model(rss, hidden)
        if device == torch.device("cuda"):
            output_floor = output_floor.detach().cpu().clone().numpy()
            output_coord = output_coord.detach().cpu().clone().numpy()
        else:
            output_floor = output_floor.detach().clone().numpy()
            output_coord = output_coord.detach().clone().numpy()
        flrs_pred.append(output_floor)
        coords_scaled_pred.append(output_coord)

    flrs_pred = np.argmax(np.vstack(flrs_pred), axis=1)
    flr_acc = accuracy_score(flrs, flrs_pred)
    # flr_results = (np.equal(
    #      flrs, np.argmax(flrs_pred, axis=1))).astype(int)
    # flr_acc = flr_results.mean()
    coords_scaled_pred = np.vstack(coords_scaled_pred)
    coord_est = coord_scaler.inverse_transform(coords_scaled_pred)  # inverse-scaling

    # calculate 2D localization errors
    dist_2d = norm(coord - coord_est, axis=1)
    mean_error_2d = dist_2d.mean()
    median_error_2d = np.median(dist_2d)

    # calculate 3D localization errors
    flr_diff = np.absolute(flrs - flrs_pred)
    z_diff_squared = (flr_height**2)*np.square(flr_diff)
    dist_3d = np.sqrt(np.sum(np.square(coord - coord_est), axis=1) + z_diff_squared)
    mean_error_3d = dist_3d.mean()
    median_error_3d = np.median(dist_3d)

    LocalizationResults = namedtuple('LocalizationResults', ['flr_acc',
                                                             'mean_error_2d',
                                                             'median_error_2d',
                                                             'mean_error_3d',
                                                             'median_error_3d',
                                                             'elapsedTime'])
    return LocalizationResults(flr_acc=flr_acc, mean_error_2d=mean_error_2d,
                               median_error_2d=median_error_2d,
                               mean_error_3d=mean_error_3d,
                               median_error_3d=median_error_3d,
                               elapsedTime=elapsedTime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-N",
        "--num_runs",
        help=
        "number of runs; default is 20",
        default=20,
        type=int)
    parser.add_argument(
        "-F",
        "--frac",
        help=
        "fraction of input data to load for training and validation; default is 1.0",
        default=1.0,
        type=float)
    parser.add_argument(
        "--validation_split",
        help=
        "fraction of training data to be used as validation data: default is 0.2",
        default=0.2,
        type=float)
    parser.add_argument(
        "-P",
        "--preprocessor",
        help=
        "preprocessor to scale/normalize input data before training and validation; default is 'standard_scaler'",
        default='standard_scaler',
        type=str)
    parser.add_argument(
        "-B",
        "--batch_size",
        help="batch size; default is 64",
        default=64,
        type=int)
    parser.add_argument(
        "-E",
        "--epochs",
        help="number of epochs; default is 100",
        default=100,
        type=int)
    parser.add_argument(
        "-O",
        "--optimizer",
        help="optimizer; default is 'nadam'",
        default='nadam',
        type=str)
    parser.add_argument(
        "-D",
        "--dropout",
        help="dropout rate before and after hidden layers; default is 0.25",
        default=0.25,
        type=float)
    parser.add_argument(
        "-C",
        "--corruption_level",
        help=
        "corruption level of masking noise for stacked denoising autoencoder; default is 0.1",
        default=0.1,
        type=float)
    parser.add_argument(
        "--dae_hidden_layers",
        help=
        "comma-separated numbers of units in hidden layers for deep autoencoder; default is ''",
        default='',
        type=str)
    parser.add_argument(
        "--sdae_hidden_layers",
        help=
        "comma-separated numbers of units in hidden layers for stacked denoising autoencoder; default is '1024,1024,1024'",
        default='1024,1024,1024',
        type=str)
    parser.add_argument(
        "--no_cache",
        help=
        "disable loading a trained model from/saving it to a cache",
        action='store_true')
    parser.add_argument(
        "--common_hidden_layers",
        help=
        "comma-separated numbers of units in common hidden layers; default is '1024'",
        default='1024',
        type=str)
    parser.add_argument(
        "--floor_hidden_layers",
        help=
        "comma-separated numbers of units in additional hidden layers for floor; default is '256'",
        default='256',
        type=str)
    parser.add_argument(
        "--coordinates_hidden_layers",
        help=
        "comma-separated numbers of units in additional hidden layers for coordinates; default is '256'",
        default='256',
        type=str)
    parser.add_argument(
        "--rnn_hidden_size",
        help="number of units in RNN hidden layer; default is 1024",
        default=1024,
        type=int)
    parser.add_argument(
        "--floor_weight",
        help="loss weight for a floor; default 1.0",
        default=1.0,
        type=float)
    parser.add_argument(
        "--coordinates_weight",
        help="loss weight for coordinates; default 1.0",
        default=1.0,
        type=float)
    parser.add_argument(
        "-V",
        "--verbose",
        help=
        "verbosity mode: 0 = silent, 1 = progress bar, 2 = one line per epoch; default is 0",
        default=0,
        type=int)
    args = parser.parse_args()

    # set variables using command-line arguments
    num_runs = args.num_runs
    frac = args.frac
    validation_split = args.validation_split
    preprocessor = args.preprocessor
    batch_size = args.batch_size
    epochs = args.epochs
    optimizer = args.optimizer
    dropout = args.dropout
    corruption_level = args.corruption_level
    if args.dae_hidden_layers == '':
        dae_hidden_layers = ''
    else:
        dae_hidden_layers = [
            int(i) for i in (args.dae_hidden_layers).split(',')
        ]
    if args.sdae_hidden_layers == '':
        sdae_hidden_layers = ''
    else:
        sdae_hidden_layers = [
            int(i) for i in (args.sdae_hidden_layers).split(',')
        ]
    cache = not args.no_cache
    if args.common_hidden_layers == '':
        common_hidden_layers = ''
    else:
        common_hidden_layers = [
            int(i) for i in (args.common_hidden_layers).split(',')
        ]
    if args.floor_hidden_layers == '':
        floor_hidden_layers = ''
    else:
        floor_hidden_layers = [
            int(i) for i in (args.floor_hidden_layers).split(',')
        ]
    if args.coordinates_hidden_layers == '':
        coordinates_hidden_layers = ''
    else:
        coordinates_hidden_layers = [
            int(i) for i in (args.coordinates_hidden_layers).split(',')
        ]
    rnn_hidden_size = args.rnn_hidden_size
    floor_weight = args.floor_weight
    coordinates_weight = args.coordinates_weight
    verbose = args.verbose

    # run simo_rnn_tut_pt() num_runs times
    flr_accs = np.empty(num_runs)
    mean_error_2ds = np.empty(num_runs)
    median_error_2ds = np.empty(num_runs)
    mean_error_3ds = np.empty(num_runs)
    median_error_3ds = np.empty(num_runs)
    elapsedTimes = np.empty(num_runs)
    for i in range(num_runs):
        print("\n########## {0:s} run ##########".format(num2words(i+1, to='ordinal_num')))
        rst = simo_rnn_tut_pt(frac, validation_split, preprocessor, batch_size,
                              epochs, optimizer, dropout, corruption_level,
                              dae_hidden_layers, sdae_hidden_layers, cache,
                              common_hidden_layers, floor_hidden_layers,
                              coordinates_hidden_layers, rnn_hidden_size,
                              floor_weight, coordinates_weight, verbose)
        flr_accs[i] = rst.flr_acc
        mean_error_2ds[i] = rst.mean_error_2d
        median_error_2ds[i] = rst.median_error_2d
        mean_error_3ds[i] = rst.mean_error_3d
        median_error_3ds[i] = rst.median_error_3d
        elapsedTimes[i] = rst.elapsedTime

    # print out final results
    base_dir = '../results/test/' + (os.path.splitext(
        os.path.basename(__file__))[0]).replace('test_', '') + '/' + 'tut'
    pathlib.Path(base_dir).mkdir(parents=True, exist_ok=True)
    base_file_name = base_dir + "/E{0:d}_B{1:d}_D{2:.2f}".format(
        epochs, batch_size, dropout)
    now = datetime.datetime.now()
    output_file_base = base_file_name + '_' + now.strftime("%Y%m%d-%H%M%S")

    with open(output_file_base + '.org', 'w') as output_file:
        output_file.write(
            "#+STARTUP: showall\n")  # unfold everything when opening
        output_file.write("* System parameters\n")
        output_file.write("  - Command line: %s\n" % ' '.join(sys.argv))
        output_file.write("  - Number of runs: %d\n" % num_runs)
        output_file.write(
            "  - Fraction of data loaded for training and validation: %.2f\n" %
            frac)
        output_file.write("  - Validation split: %.2f\n" % validation_split)
        output_file.write(
            "  - Preprocessor for scaling/normalizing input data: %s\n" %
            preprocessor)
        output_file.write("  - Batch size: %d\n" % batch_size)
        output_file.write("  - Epochs: %d\n" % epochs)
        output_file.write("  - Optimizer: %s\n" % optimizer)
        output_file.write("  - Dropout rate: %.2f\n" % dropout)
        output_file.write("  - Deep autoencoder hidden layers: ")
        if dae_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % dae_hidden_layers[0])
            for units in dae_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("  - Stacked denoising autoencoder hidden layers: ")
        if sdae_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % sdae_hidden_layers[0])
            for units in sdae_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("  - Common hidden layers: ")
        if common_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % common_hidden_layers[0])
            for units in common_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("  - Floor hidden layers: ")
        if floor_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % floor_hidden_layers[0])
            for units in floor_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("  - Coordinates hidden layers: ")
        if coordinates_hidden_layers == '':
            output_file.write("N/A\n")
        else:
            output_file.write("%d" % coordinates_hidden_layers[0])
            for units in coordinates_hidden_layers[1:]:
                output_file.write("-%d" % units)
            output_file.write("\n")
        output_file.write("  - Floor loss weight: %.2f\n" % floor_weight)
        output_file.write(
            "  - Coordinates loss weight: %.2f\n" % coordinates_weight)
        output_file.write("\n")
        # output_file.write("* Model Summary\n")
        # model.summary(print_fn=lambda x: output_file.write(x + '\n'))
        # output_file.write("\n")
        output_file.write("* Performance\n")
        output_file.write("  - Floor hit rate [%]: Mean (w/ 95% CI)={0:.4f}+-{1:{ci_fs}}, Max={2:.4f}, Min={3:.4f}\n".format(*[i*100 for i in mean_ci(flr_accs)], 100*flr_accs.max(), 100*flr_accs.min(), ci_fs=('.4f' if num_runs > 1 else '')))
        output_file.write("  - Mean 2D error [m]: Mean (w/ 95% CI)={0:.4f}+-{1:.4f}, Max={2:.4f}, Min={3:.4f}\n".format(*mean_ci(mean_error_2ds), mean_error_2ds.max(), mean_error_2ds.min()))
        output_file.write("  - Median 2D error [m]: Mean (w/ 95% CI)={0:.4f}+-{1:.4f}, Max={3:.4f}, Min={3:.4f}\n".format(*mean_ci(median_error_2ds), median_error_2ds.max(), median_error_2ds.min()))
        output_file.write("  - Mean 3D error [m]: Mean (w/ 95% CI)={0:.4f}+-{1:.4f}, Max={2:.4f}, Min={3:.4f}\n".format(*mean_ci(mean_error_3ds), mean_error_3ds.max(), mean_error_3ds.min()))
        output_file.write("  - Median 3D error [m]: Mean (w/ 95% CI)={0:.4f}+-{1:.4f}, Max={2:.4f}, Min={3:.4f}\n".format(*mean_ci(median_error_3ds), median_error_3ds.max(), median_error_3ds.min()))
        output_file.write("  - Training time [s]: Mean (w/ 95% CI)={0:.4f}+-{1:.4f}, Max={2:.4f}, Min={3:.4f}\n".format(*mean_ci(elapsedTimes), elapsedTimes.max(), elapsedTimes.min()))