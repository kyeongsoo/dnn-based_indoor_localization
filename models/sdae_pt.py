#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     sdae_pt.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2020-09-07
#
# @brief A stacked denoising autoencoder (SDAE) for pretraining of hidden
#        layers in Wi-Fi fingerprinting based on PyTorch.
#
# @remarks It is based on the SDAE implementation based on PyTorch by Vladimir
#          Lukiyanov [1].
#
#          [1] Vladimir Lukiyanov, pt-sdae. Available online:
#              https://github.com/vlukiyanov/pt-sdae


import os
import sys
import pathlib
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchsummary import summary


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NoiseRssDataset(Dataset):
    """Convert a numpy RSS input to a PyTorch dataset."""

    def __init__(self, rss, corruption_level):
        self.rss = rss.astype('float32')
        self.rss_corrupted = self.rss
        # apply masking noise
        self.rss_corrupted[np.random.rand(len(rss)) < corruption_level] = 0.0

    def __len__(self):
        return (len(self.rss))

    def __getitem__(self, idx):
        return (self.rss_corrupted[idx], self.rss[idx])


# def masking_noise(x, corruption_level):
#     x_corrupted = x
#     x_corrupted[np.random.rand(len(x)) < corruption_level] = 0.0
#     return x_corrupted


def sdae_pt(dataset='tut',
            input_data=None,
            preprocessor='standard_scaler',
            hidden_layers=[],
            cache=False,
            model_fname=None,
            optimizer='nadam',
            corruption_level=0.1,
            batch_size=32,
            epochs=100,
            validation_split=0.0):
    """Stacked denoising autoencoder

    Keyword arguments:
    dataset -- a data set for training, validation, and testing; choices are 'tut' (default), 'tut2', 'tut3', and 'uji'
    input_data -- two-dimensional array of RSSs
    preprocessor -- preprocessor used to scale/normalize the original input data (information only)
    hidden_layers -- list of numbers of units in SDAE hidden layers
    cache -- whether to load a trained model from/save it to a cache
    model_fname -- full path name for SDAE model load & save
    optimizer -- optimizer for training
    corruption_level -- corruption level of masking noise
    batch_size -- size of batch
    epochs -- number of epochs
    validation_split -- fraction of training data to be used as validation data
    """

    if (preprocessor == 'standard_scaler') or (preprocessor == 'normalizer'):
        loss_criterion = 'mse'  # mean squared error
        criterion = nn.MSELoss()
    elif preprocessor == 'minmax_scaler':
        loss_criterion = 'bce'  # binary crossentropy
        criterion = nn.BCELoss()
    else:
        print("{0:s} preprocessor is not supported.".format(preprocessor))
        sys.exit()

    if cache:
        if model_fname is None:
            model_fname = './saved/sdae_pt/' + dataset + '/H' \
                          + '-'.join(map(str, hidden_layers)) \
                          + "_B{0:d}_E{1:d}_L{2:s}_P{3:s}".format(batch_size, epochs, loss_criterion, preprocessor) \
                          + '.pt'
        if os.path.isfile(model_fname) and (os.path.getmtime(model_fname) >
                                            os.path.getmtime(__file__)):
            model = torch.load(model_fname)
            model.eval()
            return model

    x = input_data
    n_hl = len(hidden_layers)
    input_dim = input_data.shape[1]  # number of RSSs per sample
    all_layers = [input_dim] + hidden_layers
    units = []

    # build SDAE
    for i in range(n_hl):
        units.append(('fc'+str(i), nn.Linear(all_layers[i], all_layers[i+1])))
        units.append(('af'+str(i), nn.Sigmoid()))
    model = nn.Sequential(OrderedDict(units)).to(device)

    # layer-wise pretraining
    for i in range(n_hl):
        autoencoder = nn.Sequential(
            nn.Linear(all_layers[i], all_layers[i+1]),
            nn.Sigmoid(),
            nn.Linear(all_layers[i+1], all_layers[i])).to(device)
        autoencoder.train()
        optimizer = optim.SGD(autoencoder.parameters(), lr=0.001, momentum=0.9)
        dataset = NoiseRssDataset(x, corruption_level=corruption_level)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            running_loss = 0
            for x, y in dataloader:
                # move data to GPU if available
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad()
                y_pred = autoencoder(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print("[Epoch {0:3d}] loss: {1:.3f}".format(epoch+1, running_loss/len(dataloader)))
        print('Pretraining done ... ')

        # update input data for the next layer
        encoder = nn.Sequential(*list(autoencoder.children())[0:2])  # remove decoder
        x = encoder(x)
        if device == torch.device("cuda"):
            x = x.detach().cpu().clone().numpy()  # for custom dataset based on numpy
        else:
            x = x.detach().clone().numpy()

        # copy the weights and biases of the pretrained layer
        model[2*i].weight = nn.Parameter(autoencoder[0].weight.detach().clone())
        model[2*i].bias = nn.Parameter(autoencoder[0].bias.detach().clone())

    if cache:
        pathlib.Path(os.path.dirname(model_fname)).mkdir(
            parents=True, exist_ok=True)
        torch.save(model, model_fname)

    return model


if __name__ == "__main__":
    # import modules and a model to test
    os.environ['PYTHONHASHSEED'] = '0'  # for reproducibility
    sys.path.insert(0, '../utils')
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        help="a data set for training, validation, and testing; choices are 'tut' (default), 'tut2', 'tut3' and 'uji'",
        default='tut',
        type=str)
    parser.add_argument(
        "-B",
        "--batch_size",
        help="batch size; default is 32",
        default=32,
        type=int)
    parser.add_argument(
        "-E",
        "--epochs",
        help="number of epochs; default is 20",
        default=20,
        type=int)
    parser.add_argument(
        "--no_cache",
        help=
        "disable loading a trained model from/saving it to a cache",
        action='store_true')
    parser.add_argument(
        "--validation_split",
        help=
        "fraction of training data to be used as validation data: default is 0.2",
        default=0.2,
        type=float)
    parser.add_argument(
        "-H",
        "--hidden_layers",
        help=
        "comma-separated numbers of units in SDAE hidden layers; default is '128,128,128'",
        default='128,128,128',
        type=str)
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
        "-O",
        "--optimizer",
        help="optimizer for training; default is 'nadam'",
        default='nadam',
        type=str)
    parser.add_argument(
        "-C",
        "--corruption_level",
        help="corruption level of masking noise; default is 0.1",
        default=0.1,
        type=float)
    args = parser.parse_args()

    dataset = args.dataset
    batch_size = args.batch_size
    epochs = args.epochs
    validation_split = args.validation_split
    hidden_layers = [int(i) for i in (args.hidden_layers).split(',')]
    cache = not args.no_cache
    frac = args.frac
    preprocessor = args.preprocessor
    optimizer = args.optimizer
    corruption_level = args.corruption_level

    print("Loading  data ...")
    if dataset == 'uji':
        from ujiindoorloc import UJIIndoorLoc
        ds = UJIIndoorLoc(
            cache=cache,
            frac=frac,
            preprocessor=preprocessor,
            classification_mode='hierarchical')
    elif dataset == 'tut':
        from tut import TUT
        ds = TUT(
            cache=cache,
            frac=frac,
            preprocessor=preprocessor,
            classification_mode='hierarchical',
            grid_size=0)
    elif dataset == 'tut2':
        from tut import TUT2
        ds = TUT2(
            cache=cache,
            frac=frac,
            preprocessor=preprocessor,
            classification_mode='hierarchical',
            grid_size=0,
            testing_split=0.2)
    elif dataset == 'tut3':
        from tut import TUT3
        ds = TUT3(
            cache=cache,
            frac=frac,
            preprocessor=preprocessor,
            classification_mode='hierarchical',
            grid_size=0)
    else:
        print("'{0}' is not a supported data set.".format(dataset))
        sys.exit(0)

    print("Buidling SDAE model ...")
    input_data = ds.training_data.rss_scaled
    model = sdae_pt(
        dataset=dataset,
        input_data=input_data,
        preprocessor=preprocessor,
        hidden_layers=hidden_layers,
        cache=cache,
        model_fname=None,
        optimizer=optimizer,
        corruption_level=corruption_level,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split)
    summary(model, input_size=(1, input_data.shape[0], input_data.shape[1]))
