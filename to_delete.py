import numpy as np
import random
from scipy.stats import norm
import concurrent.futures
from utilities import *
import matplotlib.pyplot as plt

m3 = 100
d3 = 500
s3 = d3/100

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
        difference =    new_theta - self.theta
        indices = np.nonzero(difference)[0]

        column0 = self.X[:, indices[0]] * difference[indices[0]]
        column1 = self.X[:, indices[1]] * difference[indices[1]]
        new_residual = self.residual + np.squeeze(column0) +  np.squeeze(column1) 

        return new_residual, self.loss(new_residual)

    def loss(self, residual): 
        #print('inputt', X @ theta)
        alpha = norm.cdf((residual) / 1)
        #print('alpha', alpha)
        return -(np.log(alpha[self.ones_indices]).sum() + np.log(1 - alpha[self.min_one_indices]).sum())
        
    def update_theta_residual(self, theta, residual): 
        self.theta = theta
        self.residual = residual




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

        print(log_likelihood(X, y, theta), old_noise)
        print(log_likelihood(X, y, theta1), new_noise)



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


m3 = 100
d3 = 500
s3 = d3/100
#beta = 2
iterations3 = 100*d3

loss = get_sampling_losses_sign(10, 1, m3, d3, s3)
plt.plot(loss)