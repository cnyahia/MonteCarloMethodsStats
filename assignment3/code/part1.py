"""
This code implements a Markov Chain Monte Carlo
algorithm for the integral give in Homework 3 -
SDS386D Monte Carlo methods in statistics  (q1)

The code samples from gamma marginal densities
to evaluete the expected value of ab over the
joing pdf

@cnyahia
"""

import numpy.random as nprand
import matplotlib.pyplot as plt


# define pi(a_{n+1}|b_{n+1})
def piab(bnp1):
    """
    This is the marginal of pi(a|b)
    :param bnp1: b_{n+1} from previous pi(b|a)
    :return: a_{n+1}
    """
    anp1 = nprand.gamma(5, 1.0 / (1 + 3 * bnp1))
    return anp1


# define pi(b_{n+1} | a_{n})
def piba(an):
    bnp1 = nprand.gamma(7, 1.0 / (1 + 3 * an))
    return bnp1

# implement the sampling and integration
if __name__ == '__main__':
    numofiter = 1  # iterator
    total_iter = 5000  # total iterations
    numerator = []
    Ihat = []  # store integral value
    an = 2  # initial value of a
    while numofiter <= total_iter:
        bnplus1 = piba(an)
        anplus1 = piab(bnplus1)
        product = anplus1 * bnplus1
        numerator.append(product)
        an = anplus1  # update the current value of an
        Ihat.append(sum(numerator) / len(numerator))
        numofiter += 1

    print("... Integral estimator ...")
    print(Ihat[-1])

    # plot a graph showing convergence of integral
    Iterations = range(1, len(Ihat) + 1)
    plt.plot(Iterations, Ihat, 'g')
    plt.title("integral estimator across iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Ihat")
    plt.tight_layout()
    plt.savefig('Ihat.png')
    plt.show()

