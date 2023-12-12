import concurrent.futures
import numpy as np
import time
import math
import random
from scipy.stats import norm
from compute_noise import *
import matplotlib.pyplot as plt
import math
from concurrent.futures import ProcessPoolExecutor
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
                self.beta = self.beta *  update_beta 
            
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

    return beta, np.mean(list_errors), np.mean(steps_to_converge)




def run_multiple_multi_proc(repeat_n_times, betas, m, d, s, fixed_ones=False, sign=False, **args): 
    best_beta = 0
    best_loss = math.inf
    best_iterations = math.inf

    with ProcessPoolExecutor() as executor:
        futures = []
        for beta in betas: 
            futures.append(executor.submit(run_multiple_experiments, repeat_n_times, beta, m, d, s, fixed_ones=fixed_ones, sign=sign, **args))

        for future in concurrent.futures.as_completed(futures):
            beta, loss, n_iterations = future.result()

            if loss <= best_loss and n_iterations <= best_iterations:
                best_beta = beta
                best_loss = loss
                best_iterations = n_iterations

            print('beta=', beta, 'n_iterations=', n_iterations, 'loss', loss)

    print('Best result: beta=', best_beta, 'n_iterations=', best_iterations, 'loss', best_loss)



def run_multiple_multi_proc_sim_annealing(repeat_n_times, betas, m, d, s, mul_increases, increase_everys, fixed_ones=False, sign=False, iterations = 0): 


    best_beta = 0
    best_loss = math.inf
    best_iterations = math.inf

    with ProcessPoolExecutor() as executor:
        futures = []
        for beta in betas: 
            for mul_inc in mul_increases: 
                for increase_every in increase_everys: 
                    futures.append(executor.submit(run_multiple_experiments, repeat_n_times, beta, m, d, s, fixed_ones=fixed_ones, 
                                                   sign=sign, change_beta_every= increase_every, update_beta=mul_inc, iterations=iterations))

        for future in concurrent.futures.as_completed(futures):
            beta, loss, n_iterations = future.result()

            if loss <= best_loss and n_iterations <= best_iterations:
                best_beta = beta
                best_loss = loss
                best_iterations = n_iterations

            print('beta=', beta, 'n_iterations=', n_iterations, 'loss', loss)

    print('Best result: beta=', best_beta, 'n_iterations=', best_iterations, 'loss', best_loss)






def run_multiple_ms(repeat_n_times, beta, m_values, d, s, mul_increase, increase_every, fixed_ones=False, sign=False, iterations = 0):
    with ProcessPoolExecutor() as executor:
        futures = []
        for m_value in m_values: 
            futures.append(executor.submit(repeat_n_times, beta, m_value, d, s, mul_increase, increase_every, fixed_ones=fixed_ones, sign=sign, iterations = iterations))

        losses_with_different_m = [result[0] for result in futures]
        
    return losses_with_different_m