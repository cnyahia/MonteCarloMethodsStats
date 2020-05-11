"""
This code implements a Markov Chain Monte Carlo
algorithm for sampling the posterior in Homework 3 -
SDS386D Monte Carlo methods in statistics (q3)

The code uses the Metropolis-Hastings algorithm
to sample from a density and evaluate an integral

@cnyahia
"""


import numpy.random as nprand
import matplotlib.pyplot as plt
import numpy as np


# define alpha
def alpha(thetan1, thetan):
    numerator = np.exp(thetan1) * np.exp(-3.0 * np.exp(thetan1)) * np.exp(-0.5 * thetan1**2) * \
                 np.exp(0.5 * (thetan1**2 - thetan**2))
    denominator = np.exp(thetan) * np.exp(-3.0 * np.exp(thetan)) * np.exp(-0.5 * thetan**2)
    fraction = numerator / denominator
    alfa = min(1.0, fraction)
    return alfa


# implement metropolis hastings algorithm
if __name__ == '__main__':
    numofiter = 1  # iterator
    thetan = 3  # start with an arbitrary value of theta0
    number_of_iterations = 5000
    number_accept = 0
    number_reject = 0
    samples = []
    while numofiter <= number_of_iterations:
        samples.append(thetan)
        thetan1 = nprand.normal(loc=0, scale=1.0)  # sample from standard normal
        u = nprand.uniform(low=0.0, high=1.0)
        alpha_val = alpha(thetan1, thetan)
        if u < alpha_val:
            number_accept += 1
            thetan = thetan1  # for next round take xn1
        else:
            number_reject += 1
            # do nothing, for next round thetan = thetan
        numofiter += 1

    # print fraction accepted
    print(float(number_accept / number_of_iterations))

    # plot a histogram of samples
    plt.hist(samples, bins=50, facecolor='g', normed=True)
    plt.title("f(theta)")
    plt.xlabel("theta")
    plt.ylabel("pdf")
    # fig = plt.gcf()
    plt.savefig('part3-m3.png')
    plt.show()
