"""
This code implements rejection sampling for
the function given in Homework 1 - SDS386D
Monte Carlo methods in Statistics.

@cnyahia
"""

import numpy.random as nprand
import math
import matplotlib.pyplot as plt


# Define the function g(x)
def gx(x):
    if x <= 0 or x >= 1:
        raise Exception("Warning x is out of domain")
    else:
        y = float(((-math.log(x))**2) * (x**3) * ((1 - x)**2))
    return y

# implement the sampling algorithm
if __name__ == '__main__':
    M = 0.019966  # set the upper bound on g
    accepted = []
    numofiter = 0
    numaccept = 0
    while numofiter <= 50000:
        x = nprand.uniform(low=0, high=1)
        gofx = gx(x)  # calculate g(x)
        ratio = gofx / M
        u = nprand.uniform(low=0, high=1)
        if u < ratio:
            accepted.append(x)
            # computationally calculate acceptance probability
            numaccept += 1
        numofiter += 1

    acceptprob = float(numaccept) / numofiter
    print(acceptprob)  # Should get in range of 40%
    print(numaccept)
    print(numofiter)

    # plot a histogram of accepted values
    plt.hist(accepted, bins=20, facecolor='g')
    plt.title("f(x)")
    plt.xlabel("x")
    plt.ylabel("frequency")
    # fig = plt.gcf()
    plt.savefig('from_uniform.png')
    plt.show()





