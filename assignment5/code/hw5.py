"""
This code implements a Markov Chain Monte Carlo
algorithm for the Gibbs sampler in Homework 4 -
SDS386D Monte Carlo methods in statistics  (q1)

The code samples a bimodal Gaussian mixture model
Note that in this code di takes zero or one instead
of one or two. This is just a label and does not
have implications on outputs.

@cnyahia
"""

import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import random
from random import randrange
import copy as cp

# define sample exponential
def sample_expon(lam, n):
    """
    this function samples exponentials
    :param lam: lambda
    :param n: size
    :return: samples
    """
    scale = 1.0 / lam
    samples = list(nprand.exponential(scale=scale, size=n))
    return samples


# sample w from a dirichlet distribution given lables
def sample_weights(labels, M):
    """
    samples the weights from a Dirichlet distribution given the
    list of labels and the size M
    :param labels: list of labels
    :param M: number of components
    :return: sample from a Dirichlet distribution
    """
    # count the number of labels referring to each component
    label_numbers = [0]*M
    for key, label in enumerate(list(range(1, M+1))):
        label_numbers[key] = labels.count(label)

    for key, label_number in enumerate(label_numbers):
        label_numbers[key] = label_number + 1

    weights = list(nprand.dirichlet(label_numbers))
    return weights


# sample the labels given the weights and M
def samples_labels(weights, M, data_points):
    """
    This method samples the labels given a list
    of weights and the number of components
    :param weights: weights sampled from Dirichlet
    :param M: number of components
    :param data_points: the generated data points
    :return: label of data points
    """
    label_samples = [0]*len(data_points)
    for data_point_key, data_point in enumerate(data_points):
        # probability of a specific data point belonging to each component
        probability_di = [0]*len(weights)
        for weight_key, weight in enumerate(weights):
            numerator = weight * (weight_key+1) * np.exp(-(weight_key+1)*data_point)
            denominator = evaluate_mixture(weights, data_point)
            probability_di[weight_key] = float(numerator) / denominator
        # sample the data_point from the multivariate bernoulli distribution
        label_samples[data_point_key] = sample_multivar_bernoulli(probability_di)

    return label_samples


# evaluate the mixture
def evaluate_mixture(weights, data_point):
    """
    evaluates the mixture model for the given weights
    and the value of x (data_point)
    :param weights: weights
    :param data_point: value of x
    :return: mixture output
    """
    result = 0
    for key, weight in enumerate(weights):
        result += weight * (key+1) * np.exp(-(key+1)*data_point)

    return result


# sample a multivariate bernoulli random variable given the probabilities
def sample_multivar_bernoulli(probabilities):
    """
    sample a multivariate bernoulli for labels 1,2,3,...
    :param probabilities: probabilities of 1,2,3...
    :return: sample
    """
    cumulative = [0]*len(probabilities)
    accumulate = 0
    for key, probability in enumerate(probabilities):
        accumulate += probability
        cumulative[key] = accumulate

    u = nprand.uniform(low=0, high=1)
    sample = 0
    for key, cumu in enumerate(cumulative):
        if u <= cumu:
            sample = key + 1
            break

    return sample


# define prior for M
def M_prior(M):
    """
    returns prior for M
    :param M: value of M
    :return: prior probability
    """
    prior = 1.0 / math.factorial(M-1)
    return prior


# define sampling of increased weight vector from transition
def w_plus(w):
    """
    generates an augmented weight vector for M+1
    from the transition for w
    :param w: current vector w
    :return: augmented vector w
    """
    new_weights = cp.deepcopy(w)
    random_index = randrange(0, len(w))
    split = nprand.uniform(low=0, high=1)
    new_w1 = split * new_weights[random_index]
    new_w2 = (1-split) * new_weights[random_index]
    new_weights[random_index:random_index+1] = (new_w1, new_w2)
    return new_weights


# define sampling of decreased weight vector from the transition
def w_minus(w):
    """
    generates a weight vector with one less item by combining 2
    :param w: current vector w
    :return: decreased vector w
    """
    new_weights = cp.deepcopy(w)
    two_weights = random.sample(w, 2)
    combined_weight = sum(two_weights)
    index = new_weights.index(two_weights[0])
    new_weights.remove(two_weights[0])
    new_weights.remove(two_weights[1])
    new_weights.insert(index, combined_weight)
    return new_weights


# def probability of moving to wM+1
def prob_w_plus(M):
    ans = 1.0 / M
    return ans


# def probability of decreasing size to wM-1
def prob_w_minus(M):
    ans = (2.0) / (M * (M - 1))
    return ans


# define alpha increasing
def alpha_increasing(w, M, data_points):
    """
    returns alpha increasing
    :param w: weights
    :param M: M
    :return: alpha
    """
    new_weights = w_plus(w)
    product_numerator = 1
    for data_point in data_points:
        product_numerator = product_numerator * evaluate_mixture(new_weights, data_point)

    numerator = M_prior(M+1) * product_numerator * prob_w_minus(M+1)
    product_denominator = 1
    for data_point in data_points:
        product_denominator = product_denominator * evaluate_mixture(w, data_point)

    denominator = M_prior(M) * product_denominator * prob_w_plus(M)
    ratio = numerator / denominator
    alpha = min(1.0, ratio)
    return alpha


# define alpha decreasing
def alpha_decreasing(w, M, data_points):
    """
    returns alpha decreasing
    :param w: weights
    :param M: M
    :param data_points: data points
    :return: alpha value
    """
    new_weights = w_minus(w)
    product_numerator = 1
    for data_point in data_points:
        product_numerator = product_numerator * evaluate_mixture(new_weights, data_point)

    numerator = M_prior(M-1) * product_numerator * prob_w_plus(M-1)

    product_denominator = 1
    for data_point in data_points:
        product_denominator = product_denominator * evaluate_mixture(w, data_point)

    denominator = M_prior(M) * product_denominator * prob_w_minus(M)
    ratio = numerator / denominator
    alpha = min(1.0, ratio)
    return alpha


# define alpha increasing if current M is 1
def alpha_increasingM1(w, M, data_points):
    """
    returns alpha increasing for special case
    :param w: weights
    :param M: M
    :return: alpha
    """
    new_weights = w_plus(w)
    product_numerator = 1
    for data_point in data_points:
        product_numerator = product_numerator * evaluate_mixture(new_weights, data_point)

    numerator = M_prior(M + 1) * product_numerator * prob_w_minus(M + 1)
    product_denominator = 1
    for data_point in data_points:
        product_denominator = product_denominator * evaluate_mixture(w, data_point)

    denominator = M_prior(M) * product_denominator * prob_w_plus(M)
    ratio = (numerator / denominator) * (0.5)
    alpha = min(1.0, ratio)
    return alpha


# define alpha decreasing for special case M1
def alpha_decreasingM1(w, M, data_points):
    """
    returns alpha decreasing
    :param w: weights
    :param M: M
    :param data_points: data points
    :return: alpha value
    """
    new_weights = w_minus(w)
    product_numerator = 1
    for data_point in data_points:
        product_numerator = product_numerator * evaluate_mixture(new_weights, data_point)

    numerator = M_prior(M - 1) * product_numerator * prob_w_plus(M - 1)

    product_denominator = 1
    for data_point in data_points:
        product_denominator = product_denominator * evaluate_mixture(w, data_point)

    denominator = M_prior(M) * product_denominator * prob_w_minus(M)
    ratio = (numerator / denominator) * (0.5)
    alpha = min(1.0, ratio)
    return alpha


# define alpha increasing if current M is 2
def alpha_increasingM2(w, M, data_points):
    """
    returns alpha increasing for special case
    :param w: weights
    :param M: M
    :return: alpha
    """
    new_weights = w_plus(w)
    product_numerator = 1
    for data_point in data_points:
        product_numerator = product_numerator * evaluate_mixture(new_weights, data_point)

    numerator = M_prior(M + 1) * product_numerator * prob_w_minus(M + 1)
    product_denominator = 1
    for data_point in data_points:
        product_denominator = product_denominator * evaluate_mixture(w, data_point)

    denominator = M_prior(M) * product_denominator * prob_w_plus(M)
    ratio = (numerator / denominator) * 2.0
    alpha = min(1.0, ratio)
    return alpha


# define alpha decreasing for special case M1
def alpha_decreasingM2(w, M, data_points):
    """
    returns alpha decreasing
    :param w: weights
    :param M: M
    :param data_points: data points
    :return: alpha value
    """
    new_weights = w_minus(w)
    product_numerator = 1
    for data_point in data_points:
        product_numerator = product_numerator * evaluate_mixture(new_weights, data_point)

    numerator = M_prior(M - 1) * product_numerator * prob_w_plus(M - 1)

    product_denominator = 1
    for data_point in data_points:
        product_denominator = product_denominator * evaluate_mixture(w, data_point)

    denominator = M_prior(M) * product_denominator * prob_w_minus(M)
    ratio = (numerator / denominator) * 2.0
    alpha = min(1.0, ratio)
    return alpha


# implement the sampling from q
def sample_MH_prop(M):
    """
    sample from q given the previous value
    q is a random walk
    :param xm: previous value
    :return: sample
    """
    die = nprand.uniform(low=0, high=1)
    if M == 1:
        sample = 2

    else:
        if die <= 0.5:
            sample = M + 1
        else:
            sample = M - 1

    return sample


# sample from mixture
def sample_mix_model(weights):
    """
    samples the mixture
    :param weights: weight parameters
    :return:
    """
    key = sample_multivar_bernoulli(weights)
    value = sample_expon(key, 1)
    return value


# evaluate exponential
def eval_expon(lam, x):
    val = lam * np.exp(-lam * x)
    return val


if __name__ == '__main__':
    data_points = sample_expon(3, 100)
    M = 3  # initial value for M
    possible_labels = [1,2,3]
    labels = [0]*len(data_points)
    # set initial labels randomly
    for key, data_point in enumerate(data_points):
        labels[key] = random.choice(possible_labels)

    samples_w = []
    samples_M = []
    samples_x = []
    numiter = 1
    alpha = 1
    while numiter <= 3000:
        # sample w
        weights = sample_weights(labels=labels, M=M)
        samples_w.append(weights)

        # sample M
        M_star = sample_MH_prop(M)
        if (M_star != 1) and (M_star != 2):
            if M_star > M:
                alpha = alpha_increasing(w=weights, M=M, data_points=data_points)
            elif M_star < M:
                alpha = alpha_decreasing(weights, M, data_points)
        elif M_star == 1:
            if M_star > M:
                alpha = alpha_increasingM1(weights, M, data_points)
            elif M_star < M:
                alpha = alpha_decreasing(weights, M, data_points)
        elif M_star == 2:
            if M_star > M:
                alpha = alpha_increasingM2(weights, M, data_points)
            elif M_star < M:
                alpha = alpha_decreasingM2(weights, M, data_points)
        u = nprand.uniform(low=0, high=1)
        if u < alpha:
            M = M_star

        # sample d
        labels = samples_labels(weights, M, data_points)

        # sample x
        samples_x.append(sample_mix_model(weights))

        numiter += 1

    density_evaluate = list(np.arange(0, 3, 0.05))
    exponvals = [eval_expon(3, x) for x in density_evaluate]

    plt.hist(samples_x, bins=20, facecolor='g', alpha=0.7, normed=True, label='predictive_density')
    plt.plot(density_evaluate, exponvals, label='exponential')
    plt.title("predictive density (sampled) vs. exponential")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('pred_.png')
    plt.show()
