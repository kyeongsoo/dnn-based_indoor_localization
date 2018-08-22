#!/usr/bin/env python3
# -*- coding: utf-8 -*-
##
# @file     mean_ci.py
# @author   Kyeong Soo (Joseph) Kim <kyeongsoo.kim@gmail.com>
# @date     2018-08-22
#
# @brief A function calculating the mean and the (half) confidence interval of
#        sample data.
#
# @remarks 

### import modules
import numpy as np
import scipy.stats


def mean_ci(data, confidence=0.95):
    """Calculate the mean and (half) confidence interval of sample data.
    """
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    if n > 1:
        se = scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    else:
        h = np.nan
    return m, h
