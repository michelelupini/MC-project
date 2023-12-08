import numpy as np
import time
import math
import random
from scipy.stats import norm
from utilities import *
import matplotlib.pyplot as plt
from generate_data import * 

class NoiseComputer: 
    def __init__(self, X, y, theta, sign = False): 
        self.X = X
        self.y = y
        self.theta = theta
        self.sign = sign
        if self.sign: 
            self.residual = X @ theta
        else:
            self.residual = y - X @ theta

        self.min_one_indices = np.where(y == -1)[0]
        self.ones_indices = np.where(y == 1)[0]

    def compute_residual(self, new_theta, fixed_ones = False, sign = False): 
        if (self.theta == new_theta).all(): 
            return self.residual

        # element that became 0 will have -1 the one that became one will have +1 
        difference = self.theta - new_theta
        indices = np.nonzero(difference)

        update_residual = np.squeeze(self.X[:, indices[0]]  * difference[indices[0]][0])
        # 
        if fixed_ones: 
            indices = indices[0]
            update_residual =  np.squeeze(self.X[:, indices[0]]  * difference[indices[0]])
            update_residual = update_residual + np.squeeze(self.X[:, indices[1]] * difference[indices[1]])

        if self.sign: 
            new_residual = self.residual - np.squeeze(update_residual)
        else: 
            new_residual = self.residual + np.squeeze(update_residual)


        return new_residual
    
    def compute_noise(self, new_theta): 
        new_residual = self.compute_residual(new_theta)
        # print(new_residual)
        return new_residual, (new_residual**2).mean()
    
    def compute_noise_fixed_ones(self, new_theta): 
        new_residual = self.compute_residual(new_theta, fixed_ones=True)
        return new_residual, (new_residual**2).mean()

    def compute_noise_fixed_ones_sign(self, new_theta): 
        new_residual = self.compute_residual(new_theta, fixed_ones=True, sign=True)
        return new_residual, self.loss(new_residual)

    def update_theta_residual(self, theta, residual): 
        self.theta = theta.copy()
        self.residual = residual

    def loss(self, residual): 
        #print('inputt', X @ theta)
        alpha = norm.cdf((residual) / 1)
        # print('class alpha', alpha[0])
        #print('alpha', alpha)
        return -(np.log(alpha[self.ones_indices]).sum() + np.log(1 - alpha[self.min_one_indices]).sum())


def noise(X, y, theta): 
    # take the mean so the loss is indipendent from m
    return ((y - X @ theta) ** 2).mean()


def log_likelihood(X, y, theta): 
    
    min_one_indices = np.where(y == -1)[0]
    ones_indices = np.where(y == 1)[0]
    
    #print('inputt', X @ theta)
    alpha = norm.cdf((X @ theta) / 1)
    # print('log alpha', (X @ theta)[0])
    #print('alpha', alpha)
    return -(np.log(alpha[ones_indices]).sum() + np.log(1 - alpha[min_one_indices]).sum())



def change_theta(theta, d): 
    pos = np.random.randint(0, d) 
    new_theta = theta.copy()
    new_theta[pos] = not theta[pos]  # get new theta with only one digit changed
    return new_theta


def change_theta_fixed_ones(theta): 
    new_theta = theta.copy()
    
    zeros_indices = np.where(theta == 0)[0]
    ones_indices = np.where(theta == 1)[0]
    pos_zero = np.random.choice(zeros_indices)
    pos_one = np.random.choice(ones_indices)
    
    # switch position between one 1 and one 0
    new_theta[pos_zero] = 1
    new_theta[pos_one] = 0

    return new_theta

if __name__ == '__main__': 
    # test to make sure everything is working

    m = 1000
    d = 1000
    beta = 10
    s = d/100


    X, y, theta_true, theta = generate_data(m, d)
    noise_comp = NoiseComputer(X, y, theta)

    for i in range(1000): 
        theta1 = change_theta(theta, d)
        
        residual, old_noise = noise_comp.compute_noise(theta)
        residual, new_noise = noise_comp.compute_noise(theta1) 

        assert round(noise(X, y, theta), 5) == round(old_noise, 5)
        assert round(noise(X, y, theta1), 5) == round(new_noise, 5)

        if i % 4 == 0: 
            noise_comp.update_theta_residual(theta1, residual)
            theta = theta1.copy()


    X, y, theta_true, theta = generate_data_fixed_ones(m, d, s)
    noise_comp = NoiseComputer(X, y, theta)

    for i in range(1000): 
        theta1 = change_theta_fixed_ones(theta)
        
        residual, old_noise = noise_comp.compute_noise_fixed_ones(theta)
        residual, new_noise = noise_comp.compute_noise_fixed_ones(theta1) 

        assert round(noise(X, y, theta), 5) == round(old_noise, 5)
        assert round(noise(X, y, theta1), 5) == round(new_noise, 5)

        if i % 4 == 0: 
            noise_comp.update_theta_residual(theta1, residual)
            theta = theta1.copy()



    m3 = 100
    d3 = 500
    s3 = d3/100

    X, y, theta_true, theta = generate_data_sign(m3, d3, s3)
    noise_comp = NoiseComputer(X, y, theta, sign=True)

    for i in range(1000): 
        theta1 = change_theta_fixed_ones(theta)

        residual, old_noise = noise_comp.compute_noise_fixed_ones_sign(theta)
        residual, new_noise = noise_comp.compute_noise_fixed_ones_sign(theta1)

        # print(round(log_likelihood(X, y, theta), 5), round(old_noise, 5))
        # print(round(log_likelihood(X, y, theta1), 5), round(new_noise, 5))
        assert round(log_likelihood(X, y, theta), 5) == round(old_noise, 5)
        assert round(log_likelihood(X, y, theta1), 5) == round(new_noise, 5)


        if i % 4 == 0: 
            noise_comp.update_theta_residual(theta1, residual)
            theta = theta1.copy()


    print('TESTS PASSED!')