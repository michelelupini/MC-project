import numpy as np
import time
import math
import random
from scipy.stats import norm
from compute_noise import *
import matplotlib.pyplot as plt
from generate_data import * 

class RunExperiment: 

    def __init__(self, beta, m, d, s, fixed_ones=False, sign=False) -> None:
        self.d = d
        self.m = m
        self.beta = beta

        if fixed_ones and sign: 
            self.change_state = self.change_theta_fixed_ones
            self.X, self.y, self.theta_true, self.theta = generate_data_sign(m, d, s)
            self.noise_comp = NoiseComputer(self.X, self.y, self.theta, sign= True)
            self.compute_noise = self.noise_comp.compute_noise_fixed_ones_sign
            self.compute_mse = lambda theta : ((theta-self.theta_true)**2).sum()*(1/(2*s))
        
        elif fixed_ones and not sign: 
            self.change_state = self.change_theta_fixed_ones
            X, y, self.theta_true, self.theta = generate_data_fixed_ones(m, d, s)
            self.noise_comp = NoiseComputer(X, y, self.theta)
            self.compute_noise = self.noise_comp.compute_noise_fixed_ones
            self.compute_mse = lambda theta : ((theta-self.theta_true)**2).sum()*(1/(2*s))

        elif not fixed_ones and not sign : 
            self.change_state = self.change_theta
            X, y, self.theta_true, self.theta = generate_data(m, d)
            self.noise_comp = NoiseComputer(X, y, self.theta)
            self.compute_noise = self.noise_comp.compute_noise
            self.compute_mse = lambda theta : ((theta-self.theta_true)**2).sum()*(2/d)

        else: 
            raise Exception('Combination of fixed_ones and sign not supported')


    def change_theta(self, theta): 
        pos = np.random.randint(0, self.d) 
        new_theta = theta.copy()
        new_theta[pos] = not theta[pos]  # get new theta with only one digit changed
        return new_theta

    
    def change_theta_fixed_ones(self, theta): 
        new_theta = theta.copy()
        
        zeros_indices = np.where(theta == 0)[0]
        ones_indices = np.where(theta == 1)[0]
        pos_zero = np.random.choice(zeros_indices)
        pos_one = np.random.choice(ones_indices)
        
        # switch position between one 1 and one 0
        new_theta[pos_zero] = 1
        new_theta[pos_one] = 0

        return new_theta
    

    def log_likelihood(self, X, y, theta): 
        
        min_one_indices = np.where(y == -1)[0]
        ones_indices = np.where(y == 1)[0]
        
        #print('inputt', X @ theta)
        alpha = norm.cdf((X @ theta) / 1)
        #print('alpha', alpha)
        return -(np.log(alpha[ones_indices]).sum() + np.log(1 - alpha[min_one_indices]).sum())

    
    def get_sampling_losses(self, iterations, change_beta_every = None, update_beta = None): 
        errors = []

        theta = self.theta
        
        for _ in range(iterations): 
            theta1 = self.change_state(theta)

            residual, old_noise = self.compute_noise(theta)
            residual, new_noise = self.compute_noise(theta1)

            comp = np.exp(-self.beta * (new_noise - old_noise))
            acceptance = min(1, comp)

            # change state with acceptance probability
            if np.random.rand(1)[0] < acceptance : 
                self.noise_comp.update_theta_residual(theta1, residual)
                theta = theta1.copy()
            
            if change_beta_every and iterations % change_beta_every == 0: 
                self.beta = update_beta(self.beta)
            
            # compute error
            mse_val = self.compute_mse(theta)
            errors.append(mse_val)

            # early stopping
            if mse_val == 0: 
                break

        return errors


def run_multiple_experiments(repeat_n_times, beta, m, d, s, fixed_ones=False, sign=False, **args): 
    list_errors = []
    steps_to_converge = []

    for i in range(repeat_n_times): 
        run_exp = RunExperiment(beta, m, d, s, fixed_ones, sign)
        errors = run_exp.get_sampling_losses(**args)
        list_errors.append(errors[-1])
        steps_to_converge.append(len(errors))

    return list_errors, np.mean(steps_to_converge)





















    


# class run experiment
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

    X, y, theta_true, theta = generate_data_fixed_ones(m, d, s)
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

        # print(log_likelihood(X, y, theta), old_noise)
        # print(log_likelihood(X, y, theta1), new_noise)



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

