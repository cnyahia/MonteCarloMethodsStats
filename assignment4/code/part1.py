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


# define a sample from f(d|...)
def sample_d(x_i, mew_1, mew_2, w, lam):
    """
    samples the Bernoulli conditional for di
    :param x_i: data point
    :param mew_1: mean of one mode
    :param mew_2: mean of another mode
    :param w: probability of data point from mode
    :param lam: lambda (precision)
    :return:
    """
    st_dev = 1.0 / np.sqrt(lam)
    term_1 = w * stats.norm(mew_1, st_dev).pdf(x_i)
    term_2 = (1 - w) * stats.norm(mew_2, st_dev).pdf(x_i)
    prob_d_equal_1 = term_1 / (term_1 + term_2)
    di = nprand.binomial(1, prob_d_equal_1)
    return di


# define a sample from f(lambda|...)
def sample_lambda(a, b, mew_1, mew_2, data, labels):
    """
    sample lambda from the corresponding conditional
    distribution
    :param a: prior parameter
    :param b: prior parameter
    :param mew_1: mean of first mode
    :param mew_2: mean of second mode
    :param data: list of data points sampled
    :param labels: list of generated labels for data points
    :return: new_lambda
    """
    n = len(data)
    new_a = a + float(n) / 2
    data_mean_1 = list()
    data_mean_2 = list()
    for key, label in enumerate(labels):
        if label == 0:
            data_mean_1.append(data[key])
        elif label == 1:
            data_mean_2.append(data[key])
    sum_squares_1 = sum([(data_point - mew_1)**2 for data_point in data_mean_1])
    sum_squares_2 = sum([(data_point - mew_2)**2 for data_point in data_mean_2])
    new_b = b + (1.0 / 2) * sum_squares_1 + (1.0 / 2) * sum_squares_2
    scale = 1.0 / new_b
    new_lambda = nprand.gamma(new_a, scale=scale)
    return new_lambda


# sampling from posterior for mew_1 f(mew_1|..)
def sample_mew_1(labels, data, lam):
    if 0 not in labels:
        sample = nprand.normal(loc=0, scale=10)
    else:
        data_mean_1 = list()
        for key, label in enumerate(labels):
            if label == 0:
                data_mean_1.append(data[key])
        nA = len(data_mean_1)
        mean_data_1 = sum(data_mean_1) / len(data_mean_1)
        new_mean = (lam * nA * mean_data_1) / ((nA * lam) + (1.0 / 100))
        new_var = 1.0 / ((1.0 / 100) + (nA * lam))
        new_std = np.sqrt(new_var)
        sample = nprand.normal(loc=new_mean, scale=new_std)
    return sample


# sampling from posterior for mew_2 f(mew_2|..)
def sample_mew_2(labels, data, lam):
    if 1 not in labels:
        sample = nprand.normal(loc=0, scale=10)
    else:
        data_mean_2 = list()
        for key, label in enumerate(labels):
            if label == 1:
                data_mean_2.append(data[key])

        nB = len(data_mean_2)
        mean_data_2 = sum(data_mean_2) / len(data_mean_2)
        new_mean = (lam * nB * mean_data_2) / ((nB * lam) + (1.0 / 100))
        new_var = 1.0 / ((1.0 / 100) + (nB * lam))
        new_std = np.sqrt(new_var)
        sample = nprand.normal(loc=new_mean, scale=new_std)
    return sample


# sample from f(w|..)
def sample_w(labels):
    nA = 0
    nB = 0
    for label in labels:
        if label == 0:
            nA += 1
        elif label == 1:
            nB += 1
    alpha = nA + 1
    beta = nB + 1
    sample = nprand.beta(alpha, beta)
    return sample


# evaluate the density
def mix_model(x, w, mew_1, mew_2, lam):
    var = 1.0 / lam
    std_dev = np.sqrt(var)
    g = w * stats.norm(mew_1, std_dev).pdf(x) + (1 - w) * stats.norm(mew_2, std_dev).pdf(x)
    return g


# sample the mixture model
def sample_mix_model(w, mew_1, mew_2, lam):
    var = 1.0 / lam
    std_dev = np.sqrt(var)
    u = nprand.uniform(low=0, high=1)
    if u <= w:
        sample = nprand.normal(loc=mew_1, scale=std_dev)
    else:
        sample = nprand.normal(loc=mew_2, scale=std_dev)

    return sample


# implement sampling
if __name__ == '__main__':
    lam = 1  # initial values
    mew_1 = 0
    mew_2 = 0
    w = 0.5

    # prior params for precision
    a = 1
    b = 1

    # generate data points
    data = list(nprand.normal(0, 1, 100))

    samples_mew1 = list()  # store samples mew_1
    samples_mew2 = list()  # store samples mew_2
    samples_w = list()  # store samples_w
    samples_lam = list()  # store samples lam

    numofiter = 1
    total_iter = 2000

    labels = [0]*100
    dlabel = list()

    density_evaluate = list(np.arange(-5, 5.1, 0.05))
    predictive_density = [0]*len(density_evaluate)   # average densities for all the values

    x_samples = list()
    Ihat = list()
    estimator = list()

    while numofiter <= total_iter:
        for key, val in enumerate(data):
            labels[key] = sample_d(val, mew_1, mew_2, w, lam)
        # print(labels)
        dlabel.append(labels[10]+1)
        lam = sample_lambda(a, b, mew_1, mew_2, data, labels)
        mew_1 = sample_mew_1(labels, data, lam)
        mew_2 = sample_mew_2(labels, data, lam)
        w = sample_w(labels)

        # append lists
        samples_mew1.append(mew_1)
        samples_mew2.append(mew_2)
        samples_w.append(w)
        samples_lam.append(lam)
        x_samples.append(sample_mix_model(w, mew_1, mew_2, lam))
        Ihat.append(sum(x_samples) / len(x_samples))
        average_w = sum(samples_w) / len(samples_w)
        average_mew1 = sum(samples_mew1) / len(samples_mew1)
        average_mew2 = sum(samples_mew2) / len(samples_mew2)
        estimator_val = average_w * average_mew1 + (1.0 - average_w) * average_mew2
        estimator.append(estimator_val)
        for key, item in enumerate(density_evaluate):
            predictive_density[key] = (predictive_density[key]*(numofiter - 1) +
                                       mix_model(item, w, mew_1, mew_2, lam)) / numofiter

        numofiter += 1

    print("... plot expected value ...")
    print(Ihat[-1])
    Iterations = range(1, len(Ihat) + 1)
    plt.plot(Iterations, Ihat, 'g')
    plt.title("integral estimator across iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Ihat")
    plt.tight_layout()
    plt.savefig('expected-value.png')
    plt.show()

    print("... plot wmew_1 + (1-w)mew_2 ...")
    print(estimator[-1])
    iterations = range(1, len(estimator) + 1)
    plt.plot(iterations, estimator, 'g')
    plt.title("estimator w*mu_1 + (1-w)(mu_2) across iterations")
    plt.xlabel("iterations")
    plt.ylabel("w*mu_1 + (1-w)(mu_2)")
    plt.tight_layout()
    plt.savefig('estimator.png')
    plt.show()

    print("... plotting samples of mu 1 ...")
    # plot a graph showing distribution
    plt.hist(samples_mew1, bins=50, facecolor='g', normed=True)
    plt.title("generated samples for \mu_1")
    plt.xlabel("samples")
    plt.ylabel("normalized frequency")
    plt.tight_layout()
    plt.savefig('samples_mu1.png')
    plt.show()

    print("... plotting samples of mu 2 ...")
    # plot a graph showing distribution
    plt.hist(samples_mew2, bins=50, facecolor='g', normed=True)
    plt.title("generated samples for \mu_2")
    plt.xlabel("samples")
    plt.ylabel("normalized frequency")
    plt.tight_layout()
    plt.savefig('samples_mu2.png')
    plt.show()

    print("... plotting samples of lambda ...")
    # plot a graph showing distribution
    plt.hist(samples_lam, bins=50, facecolor='g', normed=True)
    plt.title("generated samples for lambda")
    plt.xlabel("samples")
    plt.ylabel("normalized frequency")
    plt.tight_layout()
    plt.savefig('samples_lambda.png')
    plt.show()

    print("... plotting samples from w ...")
    plt.hist(samples_w, bins=50, facecolor='g', normed=True)
    plt.title("generated samples for w")
    plt.xlabel("samples")
    plt.ylabel("normalized frequency of w samples")
    plt.tight_layout()
    plt.savefig('samples_w.png')
    plt.show()

    print("... plotting samples from d ...")
    plt.hist(dlabel, bins=50, facecolor='g', normed=True)
    plt.title("generated labels for the 10th data point")
    plt.xlabel("samples")
    plt.ylabel("normalized frequency of label samples")
    plt.tight_layout()
    plt.savefig('samples_d.png')
    plt.show()

    standard_norm = [stats.norm(0, 1).pdf(x) for x in density_evaluate]
    print("... print predictive density ...")
    plt.plot(density_evaluate, predictive_density, label='predictive')
    plt.plot(density_evaluate, standard_norm, '--', label='standard_normal')
    plt.title("predictive density vs. standard normal")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('pred_std.png')
    plt.show()

    print(" ... plotting density generated from samples ...")
    plt.hist(x_samples, bins=50, facecolor='g', alpha=0.7, normed=True, label='predictive_density')
    plt.plot(density_evaluate, standard_norm, label='standard_normal')
    plt.title("predictive density (sampled) vs. standard normal")
    plt.xlabel("x")
    plt.ylabel("density")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('pred_std_sampled.png')
    plt.show()

    '''
    print("... plotting priors ...")
    prior_samples_mu = []  # set of samples from the prior for mu
    prior_samples_lambda = []  # set of samples from the prior for lambda
    numofiter = 1
    lambda_prior = 3
    while numofiter <= total_iter:
        mu_prior = sample_prior_mu(tau)
        lambda_prior = sample_prior_lamda(a, b)
        prior_samples_mu.append(mu_prior)
        prior_samples_lambda.append(lambda_prior)
        numofiter += 1

    
    plt.hist(samples_lambda, bins=50, alpha=0.7, label='posterior', facecolor='g', normed=True)
    plt.hist(prior_samples_lambda, bins=50, alpha=0.3, label='prior', facecolor='g', normed=True)
    plt.title("prior vs. posterior for lambda")
    plt.xlabel("samples")
    plt.ylabel("normalized frequency")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('lambda_prior_post.png')
    plt.show()

    fig, ax = plt.subplots()
    h = ax.hist2d(prior_samples_mu, prior_samples_lambda, bins=50, normed=True)
    plt.title("prior distribution on parameters")
    # plt.tight_layout()
    plt.xlabel("mu")
    plt.ylabel("lambda")
    plt.colorbar(h[3], ax=ax)
    plt.savefig('samples_prior_2d.png')
    plt.show()

    print("... plotting posterior in two dimensions ...")
    fig, ax = plt.subplots()
    h = ax.hist2d(samples_mu, samples_lambda, bins=50, normed=True)
    plt.title("posterior distribution on parameters")
    # plt.tight_layout()
    plt.xlabel("mu")
    plt.ylabel("lambda")
    plt.colorbar(h[3], ax=ax)
    plt.savefig('samples_posterior_2d.png')
    plt.show()
    '''

