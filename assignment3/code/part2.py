"""
This code implements a Markov Chain Monte Carlo
algorithm for the integral given in Homework 3 -
SDS386D Monte Carlo methods in statistics (q2)

The code uses the Metropolis-Hastings algorithm
to sample from a density and evaluate an integral

@cnyahia
"""

import numpy.random as nprand
import matplotlib.pyplot as plt
import numpy as np


# define alpha
def alpha(xn1, xn):
    numerator = np.exp(-0.5 * xn1**2) * (1 + xn**2)
    denominator = np.exp(-0.5 * xn**2) * (1 + xn1**2)
    fraction = numerator / denominator
    alfa = min(1.0, fraction)
    return alfa


# define function for sampling from a normal density at xn with variance var
def normal_sample(xn, var):
    standard_dev = np.sqrt(var)
    sample = nprand.normal(loc=-xn, scale=standard_dev)
    return sample


# implement metropolis hastings algorithm
if __name__ == '__main__':
    numofiter = 1  # iterator
    xn = 3  # start with an arbitrary value of x0
    variance = 0.3  # choose a variance for proposal
    number_of_iterations = 5000
    number_accept = 0
    number_reject = 0
    samples = []
    Ihat = []  # store expected value
    while numofiter <= number_of_iterations:
        samples.append(xn)
        xn1 = normal_sample(xn, variance)
        u = nprand.uniform(low=0.0, high=1.0)
        alpha_val = alpha(xn1, xn)
        if u < alpha_val:
            number_accept += 1
            xn = xn1  # for next round take xn1
        else:
            number_reject += 1
            # do nothing, for next round xn = xn
        numofiter += 1
        Ihat.append(sum(samples) / len(samples))

    # print fraction accepted
    print(float(number_accept / number_of_iterations))

    Iterations = range(1, len(Ihat) + 1)
    plt.plot(Iterations, Ihat, 'g')
    plt.title("integral estimator across iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Ihat")
    plt.tight_layout()
    plt.savefig('expected-neg.png')
    plt.show()


    # plot a histogram of samples
    plt.hist(samples, bins=50, facecolor='g', normed=True)
    plt.title("pi(x)")
    plt.xlabel("x")
    plt.ylabel("frequency")
    # fig = plt.gcf()
    plt.savefig('piofx-neg.png')
    plt.show()



