#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     rnn_hl_uji.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2020-09-07
#
# @brief A hierarchical indoor localization system based on Wi-Fi
#        fingerprinting using a stacked denoising autoencoder (SDAE)
#        and a single-input multi-output (SIMO) recurrent neural network
#        (RNN) with UJIIndoorLoc dataset. PyTorch version.
#
# @remarks The results are submitted to <a
#          href="https://is-candar.org/GCA20/">The 5th International
#          Workshop on GPU Computing and AI (GCA'20)</a>.

# OS & system modules
import os
import sys
# other modules
import argparse
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pathlib
import random as rn
import time
from collections import namedtuple
from num2words import num2words
from numpy.linalg import norm
from timeit import default_timer as timer
# PyTorch
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
# user modules
sys.path.insert(0, '../models')
sys.path.insert(0, '../utils')
from deep_autoencoder import deep_autoencoder
from sdae_pt import sdae_pt
from mean_ci import mean_ci


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimoRNN(nn.Module):
    """ SIMO RNN for hierarchical indoor localization."""

    def __init__(self, input_size, hidden_size, bld_size, flr_size, crd_size):
        super(SimoRNN, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.fnn_bld = nn.Linear(hidden_size, bld_size)
        self.fnn_flr = nn.Linear(hidden_size, flr_size)
        self.fnn_crd = nn.Linear(hidden_size, crd_size)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.fnn_bld(output)
        output_bld = F.softmax(output)

        output, hidden = self.gru(input, hidden)
        output = self.fnn_flr(output)
        output_flr = F.softmax(output)

        output, hidden = self.gru(input, hidden)
        output = self.fnn_crd(output)
        output_crd = F.linear(output)

        return output_bld, output_flr, output_crd, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def rnn_hl_uji_pt(
        gpu_id: int,
        dataset: str,
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
        floor_weight: float,
        coordinates_weight: float,
        verbose: int
):
    """Multi-building and multi-floor indoor localization based on hybrid
    buidling/floor classification and coordinates regression using SDAE and
    SIMO RNN and UJIIndoorloc dataset.

    Keyword arguments:

    """

    # load datasets after scaling
    print("Loading data ...")
    if dataset == 'uji':
        from ujiindoorloc import UJIIndoorLoc
        uji = UJIIndoorLoc(
            cache=cache,
            frac=frac,
            preprocessor=preprocessor,
            classification_mode='hierarchical')
    else:
        print("'{0}' is not a supported data set.".format(dataset))
        sys.exit(0)
    flr_height = uji.floor_height
    training_df = uji.training_df
    training_data = uji.training_data
    testing_df = uji.testing_df
    testing_data = uji.testing_data

    # build and train a SIMO RNN model
    print("Building and training a SIMO RNN model for hybrid classification and regression ...")
    rss = training_data.rss_scaled
    coord = training_data.coord_scaled
    coord_scaler = training_data.coord_scaler  # for inverse transform
    labels = training_data.labels
    input = Input(shape=(rss.shape[1], ), name='input')  # common input

    if sdae_hidden_layers != '':

        Class MyModel(nn.Module):
            def __init__(self):
                super(MyModel, self).__init__()
                self.sdae = sdae_pt(
                    dataset=dataset,
                    input_data=rss,
                    preprocessor=preprocessor,
                    hidden_layers=sdae_hidden_layers,
                    cache=cache,
                    model_fname=None,
                    optimizer=optimizer,
                    corruption_level=corruption_level,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_split=validation_split)
                self.rnn = SimoRNN(
                    input_size=100,
                    hidden_size=10,
                    bld_size=3,
                    flr_size=5,
                    crd_size=2)

            def forward(self, x):
                x = self.sdae(x)
                x = self.rnn(x)
                return x

        model = MyModel
    else:
        model = SimoRNN(
            input_size=100,
            hidden_size=10,
            bld_size=3,
            flr_size=5,
            crd_size=2)

    print("- Training a hybrid building/floor classifier and coordinates regressor ...", end='')

    def train(category_tensor, line_tensor):
        hidden = rnn.initHidden()
        rnn.zero_grad()

        for i in range(line_tensor.size()[0]):
            output, hidden = model(line_tensor[i], hidden)

        loss = criterion(output, category_tensor)
        loss.backward()

        # Add parameters' gradients to their values, multiplied by learning rate
        for p in rnn.parameters():
            p.data.add_(p.grad.data, alpha=-learning_rate)

        return output, loss.item()

    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    def timeSince(since):
        now = time.time()
        s = now - since
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = ' ✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0

    # plot the evolution of loss over time
    plt.figure()
    plt.plot(all_losses)

    print("Evaluating the model ...")
    rss = testing_data.rss_scaled
    labels = testing_data.labels
    flrs = labels.floor
    coord = testing_data.coord  # original coordinates

    # calculate the classification accuracies and localization errors
    flrs_pred, coords_scaled_pred = model.predict(rss, batch_size=batch_size)
    flr_results = (np.equal(
        np.argmax(flrs, axis=1), np.argmax(flrs_pred, axis=1))).astype(int)
    flr_acc = flr_results.mean()
    coord_est = coord_scaler.inverse_transform(coords_scaled_pred)  # inverse-scaling

    # calculate 2D localization errors
    dist_2d = norm(coord - coord_est, axis=1)
    mean_error_2d = dist_2d.mean()
    median_error_2d = np.median(dist_2d)

    # calculate 3D localization errors
    flr_diff = np.absolute(
        np.argmax(flrs, axis=1) - np.argmax(flrs_pred, axis=1))
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
        "-G",
        "--gpu_id",
        help=
        "ID of GPU device to run this script; default is 0; set it to a negative number for CPU (i.e., no GPU)",
        default=0,
        type=int)
    parser.add_argument(
        "--dataset",
        help="a data set for training, validation, and testing; choices are 'tut' (default), 'tut2', and 'tut3'",
        default='tut',
        type=str)
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
    gpu_id = args.gpu_id
    dataset = args.dataset
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
    floor_weight = args.floor_weight
    coordinates_weight = args.coordinates_weight
    verbose = args.verbose

    # run simo_hybrid_tut() num_runs times
    flr_accs = np.empty(num_runs)
    mean_error_2ds = np.empty(num_runs)
    median_error_2ds = np.empty(num_runs)
    mean_error_3ds = np.empty(num_runs)
    median_error_3ds = np.empty(num_runs)
    elapsedTimes = np.empty(num_runs)
    for i in range(num_runs):
        print("\n########## {0:s} run ##########".format(num2words(i+1, to='ordinal_num')))
        rst = simo_hybrid_tut(gpu_id, dataset, frac, validation_split,
                              preprocessor, batch_size, epochs, optimizer,
                              dropout, corruption_level, dae_hidden_layers,
                              sdae_hidden_layers, cache, common_hidden_layers,
                              floor_hidden_layers, coordinates_hidden_layers,
                              floor_weight, coordinates_weight, verbose)
        flr_accs[i] = rst.flr_acc
        mean_error_2ds[i] = rst.mean_error_2d
        median_error_2ds[i] = rst.median_error_2d
        mean_error_3ds[i] = rst.mean_error_3d
        median_error_3ds[i] = rst.median_error_3d
        elapsedTimes[i] = rst.elapsedTime

    # print out final results
    base_dir = '../results/test/' + (os.path.splitext(
        os.path.basename(__file__))[0]).replace('test_', '') + '/' + dataset
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
        output_file.write("  - GPU ID: %d\n" % gpu_id)
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
