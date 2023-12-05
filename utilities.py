import numpy as np
import math
import random
from scipy.stats import norm
from utilities import *
import matplotlib.pyplot as plt

def noise(X, y, theta): 
    return ((y - X @ theta) ** 2).mean()

def generate_data(m, d): 
    X = np.random.normal(0, 1, (m, d))
    theta_true = np.random.randint(2, size=d)
    y = X @ theta_true + np.random.normal(0, 1, m)
    theta_random = np.random.randint(2, size=d)

    return X, y, theta_true, theta_random


def generate_data_s_ones(m, d, s): 
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

def log_likelihood(X, y, theta): 
    
    min_one_indices = np.where(y == -1)[0]
    ones_indices = np.where(y == 1)[0]
    
    #print('inputt', X @ theta)
    alpha = norm.cdf((X @ theta) / 1)
    #print('alpha', alpha)
    return -(np.log(alpha[ones_indices]).sum() + np.log(1 - alpha[min_one_indices]).sum())



def get_sampling_losses(iterations, beta, m, d):


    errors = []
    X, y, theta_true, theta = generate_data(m, d)

    for _ in range(iterations): 
        pos = np.random.randint(0, d) 
        theta1 = theta.copy()
        theta1[pos] = not theta[pos]  # get new theta with only one digit changed

        comp = np.exp(-beta * (noise(X, y, theta1) - noise(X, y, theta)))
        acceptance = min(1, comp)

        # change state with acceptance probability
        if np.random.rand(1)[0] < acceptance : 
            theta = theta1

        # compute error
        mse_val = ((theta-theta_true)**2).sum()*(2/d)
        errors.append(mse_val)

        # early stopping
        if mse_val == 0: 
            break

    return errors

def get_simulation_annealing_losses(iterations, beta, m, d, change_beta_every, update_beta): 

    errors = []
    X, y, theta_true, theta = generate_data(m, d)
    
    for _ in range(iterations): 
        pos = np.random.randint(0, d) 
        theta1 = theta.copy()
        theta1[pos] = not theta[pos]  # get new theta with only one digit changed

        comp = np.exp(-beta * (noise(X, y, theta1) - noise(X, y, theta)))
        acceptance = min(1, comp)

        # change state with acceptance probability
        if np.random.rand(1)[0] < acceptance : 
            theta = theta1
        
        if iterations % change_beta_every == 0: 
            beta = update_beta(beta)
        
        # compute error
        mse_val = ((theta-theta_true)**2).sum()*(2/d)
        errors.append(mse_val)

        # early stopping
        if mse_val == 0: 
            break

    return errors



def get_sampling_losses_fixed_ones(iterations, beta, m, d, s):

    X, y, theta_true, theta = generate_data_s_ones(m, d, s)
    errors = []

    for _ in range(iterations): 
        
        theta1 = theta.copy()
        
        zeros_indices = np.where(theta == 0)[0]
        ones_indices = np.where(theta == 1)[0]
        pos_zero = np.random.choice(zeros_indices)
        pos_one = np.random.choice(ones_indices)
        
        # switch position between one 1 and one 0
        theta1[pos_zero] = 1
        theta1[pos_one] = 0

        comp = np.exp(-beta * (noise(X, y, theta1) - noise(X, y, theta)))
        acceptance = min(1, comp)

        # change state with acceptance probability
        if np.random.rand(1)[0] < acceptance : 
            theta = theta1

        # compute error
        mse_val = ((theta-theta_true)**2).sum()*(1/(2*s))
        errors.append(mse_val)

        # early stopping
        if mse_val == 0: 
            break

    return errors


def get_sampling_losses_sign(iterations, beta, m, d, s):

    X, y, theta_true, theta = generate_data_sign(m, d, s)
    errors = []

    for _ in range(iterations): 
        
        theta1 = theta.copy()
        
        zeros_indices = np.where(theta == 0)[0]
        ones_indices = np.where(theta == 1)[0]
        pos_zero = np.random.choice(zeros_indices)
        pos_one = np.random.choice(ones_indices)
        
        theta1[pos_zero] = 1
        theta1[pos_one] = 0

        comp = np.exp(-beta * (log_likelihood(X, y, theta1) - log_likelihood(X, y, theta)))
        acceptance = min(1, comp)

        # change state with acceptance probability
        if np.random.rand(1)[0] < acceptance : 
            theta = theta1

        # compute error
        mse_val = ((theta-theta_true)**2).sum()*(1/(2*s))
        errors.append(mse_val)

        # early stopping
        if mse_val == 0: 
            break

    return errors
