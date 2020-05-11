"""
This code is for generating a sequence Xn
and sample values of X.
Use this to find the stationary distribution.

@cnyahia
"""

import numpy.random as rn
import scipy.integrate as intgrate


# define a bernoulli random variable
def bernoulli(p):
    """
    defines a bernoulli random variable
    :param p: probability of success
    :return: bernoulli sample
    """
    bern = rn.binomial(1,p)
    return bern


# define the transition probability
def ptrans(Xn):
    """
    for a value of Xn, returns the probability
    that Xn+1 = 1
    :param Xn: realization of R.V. Xn
    :return: P(Xn+1 | Xn)
    """
    p = intgrate.quad(lambda q: q**(Xn + 1) * (1-q)**(2-Xn-1), 0, 1)
    prob = 2.0 * p[0]
    return prob


if __name__ == '__main__':
    # start by generating a bernoulli X0 with parameter 1/2
    X = bernoulli(0.5)
    samples = []  # list of generated samples
    iter = 1
    numiter = 2000
    while iter <= numiter:
        p = ptrans(X)
        X = bernoulli(p)  # generate samples from transition prob.
        samples.append(X)
        iter += 1

    successes = samples.count(1)
    failures = samples.count(0)
    stationary_success = float(successes) / numiter
    stationary_failures = float(failures) / numiter
    print(failures, successes)
    print(stationary_failures, stationary_success)










