"""
This code is for Model choice problem
on the end_of_term_exam for the
SDS386D exam

@cnyahia
"""

import numpy.random as nprand
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math


# define sampling from Poisson
def sample_Poisson(lam, sample_size=1):
    """
    sample from a Poisson
    :param lam: Poisson rate lambda
    :param sample_size: number of sampled values
    :return: sampled values
    """
    if sample_size == 1:
        samples = nprand.poisson(lam=lam)
    else:
        samples = list(nprand.poisson(lam=lam, size=sample_size))
    return samples


# compute likelihood from Poisson M1
def like_M1(theta, data):
    """
    computes the likelihood for the Poisson model M1
    :param theta: poisson parameter
    :param data: list of data points
    :return: likelihood
    """
    likelihood = 1
    for data_point in data:
        likelihood = likelihood * (theta**data_point) * (np.exp(-theta)) * (1.0 / math.factorial(data_point))
    
    return likelihood


# compute likelihood from Geometric M2
def like_M2(theta, data):
    """
    computes the likelihood for the Geometric model M2
    :param theta: parameter of geometric dist
    :param data: list of data points
    :return: likelihood
    """
    likelihood = 1
    for data_point in data:
        likelihood = likelihood * theta * ((1 - theta)**data_point)
    
    return likelihood


# def evaluate lognormal
def eval_lognormal(point):
    """
    evaluates a standard log normal
    :param point: point of evaluation
    :return: value
    """
    eval = stats.lognorm.pdf(point, 1)
    return eval


# def sample lognormal
def sample_lognormal(mean=0.0, sigma=1.0):
    """
    samples a lognormal
    :param mean:
    :param sigma:
    :return: sample
    """
    sample = nprand.lognormal(mean, sigma)
    return sample


# implement algorithm
if __name__ == '__main__':
    # generate 100 samples from Poisson, lambda=1
    true_data = sample_Poisson(1, 100)
    
    # initialize algorithm
    theta1 = 6
    theta2 = 0.5
    current_model = 2
    
    # MH algorithm parameters
    num_of_iter = 1
    total_iter = 5000
    
    # number of times visit M1
    visits = 0
    
    while num_of_iter <= total_iter:
        u = nprand.uniform(low=0, high=1)
        if current_model == 1:
            visits += 1
            theta2 = nprand.uniform(low=0, high=1)
            numerator = like_M2(theta2, true_data) * eval_lognormal(theta1)
            denom = like_M1(theta1, true_data) * np.exp(-theta1)
            print('likeM1', like_M1(theta1, true_data))
            print('theta1', theta1)
            print('theta2', theta2)
            print('likeM2', like_M2(theta2, true_data))
            ratio = float(numerator / denom)
            alpha = min(1.0, ratio)
            if u < alpha:
                current_model = 2
        
        elif current_model == 2:
            theta1 = sample_lognormal(mean=0.0, sigma=1.0)
            numerator = like_M1(theta1, true_data) * np.exp(-theta1)
            denom = like_M2(theta2, true_data) * eval_lognormal(theta1)
            ratio = float(numerator / denom)
            alpha = min(1.0, ratio)
            if u < alpha:
                current_model = 1
        
        num_of_iter += 1
    
    print('number of time visit M1:', visits)
    print('total number of iterations', total_iter)
    print('posterior probability of M1', float(visits) / total_iter)



