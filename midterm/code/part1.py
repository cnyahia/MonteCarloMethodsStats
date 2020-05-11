"""
This code implements Gibbs sampling
to evaluate the integral in problem 1
of the mid-term exam.
We specifically implement a data
augmentation slice sampling version of
Gibbs

Monte Carlo methods in statistics
SDS386D

@cnyahia
"""

import numpy.random as nprand
import matplotlib.pyplot as plt


# define a sample from f(u|x)
def fux(x):
    """
    This samples from the marginal f(u|x)
    :param: x
    :return: u, a sample from f(u|x)
    """
    u = nprand.uniform(low=0, high=1.0 / (1 + x**4))
    return u


# define a sample from f(x|u)
def fxu(u):
    fun = ((1.0 / u) - 1.0)**(1.0 / 4)
    x = nprand.uniform(low=-fun, high=fun)
    return x

# implement sampling
if __name__ == '__main__':
    numofiter = 1  # iterator
    total_iter = 100000  # total iterations
    x = 0  # starting value of x
    samples = []  # store samples
    numerator = []
    Ihat = []  # integral estimator
    while numofiter <= total_iter:
        u = fux(x)
        x = fxu(u)
        samples.append(x)
        numerator.append(x**2)
        Ihat.append(float(sum(numerator)) / len(numerator))
        numofiter += 1

    print("... plotting samples ...")
    # plot a graph showing distribution
    plt.hist(samples, bins=50, facecolor='g', normed=True)
    plt.title("generated samples from f(x)")
    plt.xlabel("samples")
    plt.ylabel("normalized frequency")
    plt.tight_layout()
    plt.savefig('samples.png')
    plt.show()

    # print Ihat and plot a graph showing its convergence
    print("... integral estimator ...")
    print(Ihat[-1])
    Iterations = range(1, len(Ihat) + 1)
    plt.plot(Iterations, Ihat, 'g')
    plt.title("integral estimator across iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Ihat")
    plt.tight_layout()
    plt.savefig('ihat.png')
    plt.show()

