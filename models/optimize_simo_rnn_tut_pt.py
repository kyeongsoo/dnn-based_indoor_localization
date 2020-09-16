#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render, init_notebook_plotting
#from ax.utils.tutorials.cnn_utils import load_mnist, train, evaluate, CNN
from simo_rnn_tut_pt import simo_rnn_tut_pt


torch.manual_seed(12345)
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_runs = 10
mean_error_3ds = np.empty(num_runs)


def train_evaluate(parameterization):
    for i in range(num_runs):
        rst = simo_rnn_tut_pt(frac=1.0,
                              validation_split=0.2,
                              preprocessor='standard_scaler',
                              batch_size=parameterization.get('batch_size'),
                              epochs=300,
                              optimizer='SGD',
                              dropout=parameterization.get('dropout'),
                              corruption_level=0.1,
                              dae_hidden_layers='',
                              sdae_hidden_layers='',
                              cache=True,
                              rnn_hidden_size=parameterization.get('rnn_hidden_size'),
                              rnn_num_layers=parameterization.get('rnn_num_layers'),
                              floor_hidden_size=parameterization.get('floor_hidden_size'),
                              floor_num_layers=parameterization.get('floor_num_layers'),
                              coordinates_hidden_size=parameterization.get('coordinates_hidden_size'),
                              coordinates_num_layers=parameterization.get('coordinates_num_layers'),
                              floor_weight=1.0,
                              coordinates_weight=1.0,
                              log_level='WARNING',
                              device=device)
        mean_error_3ds[i] = rst.mean_error_3d

    return {"mean_3d_error": (mean_error_3ds.mean(), mean_error_3ds.std())}


best_parameters, values, experiment, model = optimize(
    parameters=[
        {
            "name": "batch_size",
            "type": "choice",
            "values": [4, 8, 16, 32, 64, 128, 256],
        },
        {
            "name": "dropout",
            "type": "range",
            "bounds": [0.05, 0.5],
            "log_scale": True,
        },
        {
            "name": "rnn_hidden_size",
            "type": "choice",
            "values": [16, 64, 128, 256, 512, 1024]
        },
        {
            "name": "rnn_num_layers",
            "type": "choice",
            "values": [1, 2, 3]
        },
        {
            "name": "floor_hidden_size",
            "type": "choice",
            "values": [16, 64, 128, 256, 512, 1024]
        },
        {
            "name": "floor_num_layers",
            "type": "choice",
            "values": [1, 2, 3]
        },
        {
            "name": "coordinates_hidden_size",
            "type": "choice",
            "values": [16, 64, 128, 256, 512, 1024]
        },
        {
            "name": "coordinates_num_layers",
            "type": "choice",
            "values": [1, 2, 3]
        },
    ],
    evaluation_function=train_evaluate,
    objective_name='mean_3d_error',
    total_trials=100            # default is 20
)

print('Best parameters')
for x in best_parameters:
    print("- {0}: {1}".format(x, best_parameters[x]))
print('Values')
avg, cov = values
print(" - Avg. of {0}: {1}".format(list(avg.keys())[0], list(avg.values())[0]))
print(" - Cov. of {0}: {1}".format(list(cov.keys())[0], list(list(cov.values())[0].values())[0]))
