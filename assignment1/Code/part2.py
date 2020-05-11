"""
This code implements rejection sampling for
the function given in Homework 1 - SDS386D
Monte Carlo methods in Statistics.

The code uses the sampled density to evaluate
the integral in part 2 of the homework.

@cnyahia
"""

import numpy.random as nprand
import math
import matplotlib.pyplot as plt
import numpy as np


# Define the function g(x)
def gx(x):
    if x <= 0 or x >= 1:
        raise Exception("Warning x is out of domain")
    else:
        y = float(((-math.log(x))**2) * (x**3) * ((1 - x)**2))
    return y


# Define the function l(x)
def lx(x):
    l = float((1.0 - x)**0.5)
    return l


# The function f(x)
def fx(x):
    y = gx(x)
    return y

# implement the sampling algorithm
if __name__ == '__main__':
    M = 0.01996559  # set the upper bound on g
    accepted = []
    numofiter = 0
    numaccept = 0  # number of accepted x_i's
    lxi = []  # numerator of estimator
    fxi = []
    Ihat = []
    VarIhat = []
    while numofiter <= 1000:
        x = nprand.uniform(low=0, high=1)
        gofx = gx(x)  # calculate g(x)
        ratio = float(gofx) / M
        u = nprand.uniform(low=0, high=1)
        if u < ratio:
            accepted.append(x)
            # computationally calculate acceptance probability
            numaccept += 1
            lxi.append(lx(x))
        fxi.append(fx(u))  # sample f(x) using uniform R.V. to obtain c
        numofiter += 1
        if numaccept > 0:
            Itildaelem = float(sum(lxi)) / numaccept
            celem = numofiter / float(sum(fxi))
            Ihat.append(Itildaelem / celem)
            if len(Ihat) > 1:
                VarIhat.append(np.var(Ihat, ddof=1))


    Itilda = float(sum(lxi)) / numaccept
    c = numofiter / float(sum(fxi))
    estimator = Itilda / c
    print("... Integral estimator ...")
    print(estimator)
    acceptprob = float(numaccept) / numofiter
    print("... acceptance probability ...")
    print(acceptprob)  # Should get in range of 40%
    print("... number of accepted proposals ...")
    print(numaccept)
    print("... total number of proposals ...")
    print(numofiter)
    print("... variance of estimator ...")
    print(VarIhat[-1])

    # plot a graph showing convergence to value of integral I
    Iterations = range(1, len(VarIhat) + 1)
    plt.plot(Iterations, VarIhat, 'g')
    plt.title("Variance for integral estimator across iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Variance of Ihat")
    plt.tight_layout()
    plt.savefig('variance.png')
    plt.show()







