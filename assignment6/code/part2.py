"""
This code implements problem 2 on the
midterm for Monte Carlo methods in
statistics SDS386D
We implement Bayesian posterior sampling
for a normal likelihood using conjugate
priors

@cnyahia
"""

import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt

# define a sample from f(mu|lamda, data)
def sample_mu(prev_lambda, sum_nums, n, tau_squared):
    """
    sampling of mu
    :param prev_lambda: lambda at the previous iteration
    :param sum_nums: sum of the data
    :param n: total data points
    :param tau_squared: prior parameter on mew
    :return: mu
    """
    new_mean = 1.0 / ((1.0 / tau_squared) + float(n) * prev_lambda) * float(sum_nums) * prev_lambda
    new_var = 1.0 / ((1.0 / tau_squared) + float(n) * prev_lambda)
    new_std = np.sqrt(new_var)
    mu = nprand.normal(loc=new_mean, scale=new_std)
    return mu


# define a sample from f(lambda|mu, data)
def sample_lambda(mew, sum_nums, sum_squares, n, a, b):
    """
    sampling of lambda
    :param mew: value of mew conditioned upon
    :param sum_nums: sum(data)
    :param sum_squares: sum(data**2)
    :param n: total data points
    :param a: prior parameter
    :param b: prior parameter
    :return: new_lambda
    """
    new_a = a + float(n) / 2
    new_b = b + (1.0 / 2) * (sum_squares - 2 * mew * sum_nums + n * mew**2)
    scale = 1.0 / new_b
    new_lambda = nprand.gamma(new_a, scale=scale)
    return new_lambda


# sampling from prior for mu, mean around which data is distributed
def sample_prior_mu(tau):
    sample = nprand.normal(loc=0, scale=tau)
    return sample


# sampling from the prior for lambda
def sample_prior_lamda(a, b):
    scale = 1.0 / b
    sample = nprand.gamma(a, scale=scale)
    return sample


# implement sampling
if __name__ == '__main__':
    numofiter = 1  # iterator
    total_iter = 80000  # total iterations
    lam = 3  # initial value for lambda
    tau = 1.5
    tau_squared = tau**2  # mean prior
    a = 1.5  # precision prior
    b = 1.5  # precision prior
    n = 51  # sample total
    sum_squared = 39.6
    sum_nums = 10.2
    samples_mu = []  # store samples mu
    samples_lambda = []  # store samples lambda
    while numofiter <= total_iter:
        mu = sample_mu(lam, sum_nums, n, tau_squared)
        lam = sample_lambda(mu, sum_nums, sum_squared, n, a, b)
        samples_mu.append(mu)
        samples_lambda.append(lam)
        numofiter += 1

    print("... plotting samples of mu ...")
    # plot a graph showing distribution
    plt.hist(samples_mu, bins=50, facecolor='g', normed=True)
    plt.title("generated samples for mu")
    plt.xlabel("samples")
    plt.ylabel("normalized frequency")
    plt.tight_layout()
    plt.savefig('samples_mu.png')
    plt.show()

    print("... plotting samples of lambda ...")
    # plot a graph showing distribution
    plt.hist(samples_lambda, bins=50, facecolor='g', normed=True)
    plt.title("generated samples for lambda")
    plt.xlabel("samples")
    plt.ylabel("normalized frequency")
    plt.tight_layout()
    plt.savefig('samples_lambda.png')
    plt.show()

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

    plt.hist(samples_mu, bins=50, alpha=0.7, label='posterior', facecolor='g', normed=True)
    plt.hist(prior_samples_mu, bins=50, alpha=0.3, label='prior', facecolor='g', normed=True)
    plt.title("prior vs. posterior for mu")
    plt.xlabel("samples")
    plt.ylabel("normalized frequency")
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('mu_prior_post.png')
    plt.show()

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
