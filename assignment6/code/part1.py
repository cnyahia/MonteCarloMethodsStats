"""
This code is for the sequential Monte Carlo
problem in assignment 6

@cnyahia
"""

import numpy.random as nprand
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


# define the sampling from p(xt|xt-1)
def pxx(x_prev, stdev):
    """
    implements sampling P(xt|xt-1)=N(xt-1,var)
    This is normal around xt with variance of 1
    :param x_prev: previous sample
    :param stdev: standard deviation
    :return: sample
    """
    sample = nprand.normal(loc=x_prev, scale=stdev)
    return sample


# define the sampling for p(yt|xt)
def pyx(x, stdev):
    """
    implements sampling P(yt|xt)=N(yt|xt,1)
    :param x: current value of x
    :param stdev: observation density standard dev.
    :return: sample
    """
    sample = nprand.normal(loc=x, scale=stdev)
    return sample


# define update of mean from Kalman filter equations
def update_mean(observation, prev_mean, prev_var, pred_var, obs_var):
    """
    update of mean in KF equations for P(xt|y1:t)
    :param observation: observation yt
    :param prev_mean: previous mean of P(xt|y1:t)
    :param prev_var: previous var of P(xt|y1:t)
    :param pred_var: prediction variance of P(xt|xt-1)
    :param obs_var: observation variance of P(yt|xt)
    :return: new_mean
    """
    numerator = observation * (pred_var + prev_var) + obs_var * prev_mean
    denom = pred_var + prev_var + obs_var
    new_mean = float(numerator) / denom
    return new_mean


# define update of variance for Kalman filter equations
def update_var(prev_var, pred_var, obs_var):
    """
    update variance in KF equations for P(xt|y1:t)
    :param prev_var: previous variance of P(xt|y1:t)
    :param pred_var: prediction variance of P(xt|xt-1)
    :param obs_var: observation variance of P(yt|xt)
    :return: new_var
    """
    numerator = obs_var * (pred_var + prev_var)
    denominator = pred_var + prev_var + obs_var
    new_var = float(numerator)/denominator
    return new_var


# implement algorithm
if __name__ == '__main__':
    # actual initial value
    x = 1
    
    y_samples = list()  # generated observations from model
    # model parameters
    pred_var = 1
    obs_var = 1
    
    # KF initial distribution
    mean = 1
    var = 1
    
    # generate initial samples for PF
    particles = list()
    particle_weights = list()
    normalized_weights = list()
    N = 10000  # number of particles
    for particle in range(1, N+1):
        particles.append(nprand.normal(loc=1, scale=1))
        # initialize with uniform weight
        particle_weights.append(1.0 / N)
    
    print(len(particles))
    print(particles)
    print(particle_weights)
    for idx in range(1, 50 + 1):
        # Generate observations
        x = pxx(x, np.sqrt(pred_var))
        y = pyx(x, np.sqrt(obs_var))
        y_samples.append(y)

        # Kalman filter
        mean = update_mean(y, mean, var, pred_var, obs_var)
        var = update_var(var, pred_var, obs_var)
        
        # particle filter
        # get new particles at t+1 using proposal
        for key, particle in enumerate(particles):
            particles[key] = pxx(particle, np.sqrt(pred_var))
        
        # update particle weights
        for key, particle in enumerate(particles):
            particle_weights[key] = particle_weights[key] * norm.pdf(y, particle, np.sqrt(obs_var))
        
        # re-normalize particles
        sum_weights = sum(particle_weights)
        normalized_weights = [float(p_weight)/sum_weights for p_weight in particle_weights]

    # the distribution P(x50|y1:50) is normal(mean, var)
    # mean, var what results at the end from updates above
    print("... plotting Kalman filter density ...")
    print('mean: ', mean)
    print('std: ', np.sqrt(var))
    x_range = np.arange(mean-10, mean+10, 0.001)
    plt.plot(x_range, norm.pdf(x_range, mean, np.sqrt(var)), 'g')
    # plot particle filter
    print("... plotting PF results ...")
    plt.bar(particles, normalized_weights, width=0.65)
    plt.title("density from KF and PF updates")
    plt.xlabel("x")
    plt.ylabel("density f(x)")
    plt.tight_layout()
    plt.savefig('KfPf.png')
    plt.show()

