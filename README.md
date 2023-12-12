# Random Signal Recovery

This repository contains code and documentation related to the problem of recovering sparse signals from measurements. The problem is approached using Markov Chain Monte Carlo (MCMC) techniques, with a focus on optimizing over binary hypercube and recovering sparse binary signals.

## Table of Contents
- [Introduction](#introduction)
- [Optimizing over the Binary Hypercube](#optimizing-over-the-binary-hypercube)
  - [Minimum Number of Measurements](#minimum-number-of-measurements)
  - [Computing Probability](#computing-probability)
  - [Optimizing Parameters](#optimizing-parameters)
  - [Results](#results)
- [Recovering a Sparse, Binary Signal](#recovering-a-sparse-binary-signal)
  - [Base Chain Modification](#base-chain-modification)
  - [Exploration and Convergence](#exploration-and-convergence)
  - [Different Number of Measurements](#different-number-of-measurements)
- [Recovering a Sparse Signal from 1-Bit Measurements](#recovering-a-sparse-signal-from-1-bit-measurements)
  - [Formulating the Optimization Problem](#formulating-the-optimization-problem)
  - [Probability Computation](#probability-computation)
  - [Code](#code)
  - [Results](#results-1)

## Introduction

Provide a brief overview of the problem and the goals of this project.

## Optimizing over the Binary Hypercube

### Minimum Number of Measurements

Explain the approach to finding the minimum number of measurements required for signal recovery with a focus on the Subset Sum Algorithm.

### Computing Probability

Describe the probability computation and the formulation of the maximization problem.

### Optimizing Parameters

Discuss the optimization of hyperparameters using both fixed values and simulated annealing.

### Results

Present the results and analysis of the experiments conducted in optimizing over the binary hypercube.

## Recovering a Sparse, Binary Signal

### Base Chain Modification

Explain the modification made to the base chain to improve convergence for sparse binary signals.

### Exploration and Convergence

Discuss the exploration strategy and convergence results, comparing fixed and simulated annealing approaches.

### Different Number of Measurements

Explore the impact of varying the number of measurements on signal recovery and provide visualizations of the results.

## Recovering a Sparse Signal from 1-Bit Measurements

### Formulating the Optimization Problem

Discuss the formulation of the optimization problem for recovering sparse signals from 1-bit measurements.

### Probability Computation

Explain the computation of probability and the approach to solve the optimization problem.

### Code

Provide information about the code structure and organization.

### Results

Present the results of the experiments on recovering sparse signals from 1-bit measurements.

## Conclusion

Summarize the key findings, lessons learned, and potential future work.

## Acknowledgments

Give credit to any external libraries, resources, or contributors.

## License

Specify the license under which the code and documentation are released.
