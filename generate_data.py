import numpy as np
import time
import math
import random
from scipy.stats import norm
import matplotlib.pyplot as plt

def generate_data(m, d): 
    X = np.random.normal(0, 1, (m, d))
    theta_true = np.random.randint(2, size=d)
    y = X @ theta_true + np.random.normal(0, 1, m)
    theta_random = np.random.randint(2, size=d)

    return X, y, theta_true, theta_random


def generate_data_fixed_ones(m, d, s): 
    """Generate theta with s ones and d-s zeros

    Args:
        m (int): number of samples
        d (int): signal dimension
        s (int): number of ones in signal

    Returns:
        _type_: _description_
    """
    X = np.random.normal(0, 1, (m, d))
    theta_true = [0] * int(d - s) + [1] * int(s)
    random.shuffle(theta_true)
    theta_true = np.array(theta_true)
    y = X @ theta_true + np.random.normal(0, 1, m)

    theta_random = [0] * int(d - s) + [1] * int(s)
    random.shuffle(theta_random) # initialize theta randomly with 0 and 1s
    theta_random = np.array(theta_random)


    return X, y, theta_true, theta_random

def generate_data_sign(m, d, s): 
    X = np.random.normal(0, 1, (m, d))
    theta_true = [0] * int(d - s) + [1] * int(s)
    random.shuffle(theta_true)
    theta_true = np.array(theta_true)
    y = np.sign(X @ theta_true + np.random.normal(0, 1, m))

    theta_random = [0] * int(d - s) + [1] * int(s)
    random.shuffle(theta_random) # initialize theta randomly with 0 and 1s
    theta_random = np.array(theta_random)

    return X, y, theta_true, theta_random






