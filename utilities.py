import numpy as np
import time
import math
import random
from scipy.stats import norm
from utilities import *
import matplotlib.pyplot as plt



class NoiseComputer: 
    def __init__(self, X, y, theta): 
        self.X = X
        self.y = y
        self.theta = theta
        self.residual = y - X @ theta
    
    def compute_noise(self, new_theta): 
        
        if (self.theta == new_theta).all(): 
            return self.residual, (self.residual**2).mean()

        # element that became 0 will have -1 the one that became one will have +1 
        difference = self.theta - new_theta
        indices = np.nonzero(difference)

        column0 = self.X[:, indices[0]]  * difference[indices[0]][0]
        new_residual = self.residual + np.squeeze(column0) 

        return new_residual, (new_residual**2).mean()
    
    def compute_noise_fixed_ones(self, new_theta): 
        
        if (self.theta == new_theta).all(): 
            return self.residual, (self.residual**2).mean()

        # element that became 0 will have -1 the one that became one will have +1 
        difference =   self.theta  - new_theta
        indices = np.nonzero(difference)[0]

        column0 = self.X[:, indices[0]] * difference[indices[0]]
        column1 = self.X[:, indices[1]] * difference[indices[1]]
        new_residual = self.residual + np.squeeze(column0) +  np.squeeze(column1) 

        return new_residual, (new_residual**2).mean()

    def update_theta_residual(self, theta, residual): 
        self.theta = theta
        self.residual = residual


class NoiseComputer1: 
    def __init__(self, X, y, theta): 
        self.X = X
        self.y = y
        self.theta = theta
        self.residual = X @ theta

        self.min_one_indices = np.where(y == -1)[0]
        self.ones_indices = np.where(y == 1)[0]
    
    def compute_noise_fixed_ones(self, new_theta): 
        
        if (self.theta == new_theta).all(): 
            return self.residual, self.loss(self.residual)

        # element that became 0 will have -1 the one that became one will have +1 
        difference =   self.theta  - new_theta
        indices = np.nonzero(difference)[0]

        column0 = self.X[:, indices[0]] * difference[indices[0]]
        column1 = self.X[:, indices[1]] * difference[indices[1]]
        new_residual = self.residual + np.squeeze(column0) +  np.squeeze(column1) 

        return new_residual, self.loss(new_residual)

    def loss(self, residual): 
        #print('inputt', X @ theta)
        alpha = norm.cdf((residual) / 1)
        print('class alpha', residual[0])
        #print('alpha', alpha)
        return -(np.log(alpha[self.ones_indices]).sum() + np.log(1 - alpha[self.min_one_indices]).sum())
        
    def update_theta_residual(self, theta, residual): 
        self.theta = theta
        self.residual = residual


def noise(X, y, theta): 
    # take the mean so the loss is indipendent from m
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
    print('log alpha', (X @ theta)[0])
    #print('alpha', alpha)
    return -(np.log(alpha[ones_indices]).sum() + np.log(1 - alpha[min_one_indices]).sum())


def run_multiple_experiments(repeat_n_times, get_samples, *args): 
    list_errors = []
    number_iterations = []

    for i in range(repeat_n_times): 
        errors = get_samples(*args)
        list_errors.append(errors[-1])
        number_iterations.append(len(errors))

    return np.mean(list_errors), np.mean(number_iterations)


def get_sampling_losses(iterations, beta, m, d):

    X, y, theta_true, theta = generate_data(m, d)
    noise_comp = NoiseComputer(X, y, theta)

    errors = []
    for _ in range(iterations): 
        pos = np.random.randint(0, d) 
        theta1 = theta.copy()
        theta1[pos] = not theta[pos]  # get new theta with only one digit changed

        residual, old_noise = noise_comp.compute_noise(theta)
        residual, new_noise = noise_comp.compute_noise(theta1) 

        comp = np.exp(-beta * (new_noise - old_noise))
        acceptance = min(1, comp)

        # change state with acceptance probability
        if np.random.rand(1)[0] < acceptance : 
            noise_comp.update_theta_residual(theta1, residual)
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
    noise_comp = NoiseComputer(X, y, theta)
    
    for _ in range(iterations): 
        pos = np.random.randint(0, d) 
        theta1 = theta.copy()
        theta1[pos] = not theta[pos]  # get new theta with only one digit changed


        residual, old_noise = noise_comp.compute_noise(theta)
        residual, new_noise = noise_comp.compute_noise(theta1) 

        comp = np.exp(-beta * (new_noise - old_noise))
        acceptance = min(1, comp)

        # change state with acceptance probability
        if np.random.rand(1)[0] < acceptance : 
            noise_comp.update_theta_residual(theta1, residual)
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
    noise_comp = NoiseComputer(X, y, theta)
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

        residual, old_noise = noise_comp.compute_noise_fixed_ones(theta)
        residual, new_noise = noise_comp.compute_noise_fixed_ones(theta1) 

        comp = np.exp(-beta * (new_noise - old_noise))
        acceptance = min(1, comp)

        # change state with acceptance probability
        if np.random.rand(1)[0] < acceptance : 
            noise_comp.update_theta_residual(theta1, residual)
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
    noise_comp = NoiseComputer1(X, y, theta)
    errors = []

    for _ in range(iterations): 
        
        theta1 = theta.copy()
        
        zeros_indices = np.where(theta == 0)[0]
        ones_indices = np.where(theta == 1)[0]
        pos_zero = np.random.choice(zeros_indices)
        pos_one = np.random.choice(ones_indices)
        
        theta1[pos_zero] = 1
        theta1[pos_one] = 0


        residual, old_noise = noise_comp.compute_noise_fixed_ones(theta)
        residual, new_noise = noise_comp.compute_noise_fixed_ones(theta1) 

        comp = np.exp(-beta * (new_noise - old_noise))
        acceptance = min(1, comp)

        # change state with acceptance probability
        if np.random.rand(1)[0] < acceptance : 
            noise_comp.update_theta_residual(theta1, residual)
            theta = theta1

        # comp = np.exp(-beta * (log_likelihood(X, y, theta1) - log_likelihood(X, y, theta)))
        # acceptance = min(1, comp)

        # # change state with acceptance probability
        # if np.random.rand(1)[0] < acceptance : 
        #     theta = theta1


        # compute error
        mse_val = ((theta-theta_true)**2).sum()*(1/(2*s))
        errors.append(mse_val)

        # early stopping
        if mse_val == 0: 
            break

    return errors


def test_trick(): 
    m = 2000
    d = 2000
    s = d/100
    beta = 2
    iterations = 100*d
    X, y, theta_true, theta = generate_data(m, d)
    noise_comp = NoiseComputer(X, y, theta)

    residual = noise_comp.residual

    s = time.time()

    for i in range(1000): 
        pos = np.random.randint(0, d) 
        theta1 = theta.copy()
        if i % 3 == 0: 
            theta1[pos] = not theta[pos]  # get new theta with only one digit changed

        new_residual, square_sum  = noise_comp.compute_noise(theta1)
        
        if np.abs(noise(X, y, theta1) - square_sum) > 0.001: 
            print(noise(X, y, theta1),  square_sum)
            print('PROBLEM')
            break
        
        if i % 3 == 0: 
            theta = theta1
            noise_comp.update_theta_residual(theta1, new_residual)

        # print(theta.shape)
    print(time.time() -s )