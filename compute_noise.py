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
        self.residual_fixed_ones = X @ theta

    def compute_residual(self, new_theta, fixed_ones = False, sign = False): 
        if (self.theta == new_theta).all(): 
            return self.residual, (self.residual**2).mean()

        # element that became 0 will have -1 the one that became one will have +1 
        difference = self.theta - new_theta
        indices = np.nonzero(difference)

        update_residual = np.squeeze(self.X[:, indices[0]]  * difference[indices[0]][0])
        if fixed_ones: 
            update_residual =+ np.squeeze(self.X[:, indices[1]] * difference[indices[1]])

        if sign: 
            new_residual = self.residual - update_residual
        else: 
            new_residual = self.residual + update_residual

        return new_residual
    
    
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




def log_likelihood(X, y, theta): 
    
    min_one_indices = np.where(y == -1)[0]
    ones_indices = np.where(y == 1)[0]
    
    #print('inputt', X @ theta)
    alpha = norm.cdf((X @ theta) / 1)
    print('log alpha', (X @ theta)[0])
    #print('alpha', alpha)
    return -(np.log(alpha[ones_indices]).sum() + np.log(1 - alpha[min_one_indices]).sum())
