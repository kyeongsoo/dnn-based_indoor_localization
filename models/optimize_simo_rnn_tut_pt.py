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

init_notebook_plotting()


# In[2]:


torch.manual_seed(12345)
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_runs = 2
mean_error_3ds = np.empty(num_runs)


# In[3]:


def train_evaluate(parameterization):
    for i in range(num_runs):
        rst = simo_rnn_tut_pt(frac=1.0,
                              validation_split=0.2,
                              preprocessor='standard_scaler',
                              batch_size=parameterization.get('batch_size'),
                              epochs=10,
                              optimizer='SGD',
                              dropout=parameterization.get('dropout'),
                              corruption_level=0.1,
                              dae_hidden_layers='',
                              sdae_hidden_layers='',
                              cache=True,
                              rnn_hidden_size=parameterization.get('rnn_hidden_size'),
                              rnn_num_layers=1,
                              floor_hidden_layers=[128],
                              coordinates_hidden_layers=[128],
                              floor_weight=1.0,
                              coordinates_weight=1.0,
                              verbose=0,
                              device=device)
        mean_error_3ds[i] = rst.mean_error_3d

    return {"mean_3d_error": (mean_error_3ds.mean(), mean_error_3ds.std())}


# In[ ]:


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
            "bounds": [0.01, 0.5],
            "log_scale": True,
        },
        {
            "name": "rnn_hidden_size",
            "type": "choice",
            "values": [16, 64, 128, 256, 512, 1024]
        }
    ],
    evaluation_function=train_evaluate,
    objective_name='mean_3d_error',
)

