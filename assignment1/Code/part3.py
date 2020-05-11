"""
This code is for part 3 of Homework 1
on Monte Carlo methods in statistics.

The exercise asks for sampling a
standard normal density using a Cauchy
proposal

@cnyahia
"""
import numpy.random as nprand
import numpy as np
import matplotlib.pyplot as plt


# define a Cauchy R.V. obtained using
# inverse CDF technique
def cauchyRV():
    X = np.tan(np.pi * (nprand.uniform(low=0, high=1) - 0.5))
    return X


# Define the function g(x)
def gx(x):
    numerator = np.pi * (1 + x**2) * np.exp(-0.5 * x**2)
    denominator = np.sqrt(2 * np.pi)
    y = float(numerator) / denominator
    return y

if __name__ == '__main__':
    M = 1.52  # set the upper bound on g
    accepted = []
    numofiter = 0
    numaccept = 0  # number of accepted x_i's
    while numofiter <= 1000:
        x = cauchyRV()  # sample a Cauchy random variable
        gofx = gx(x)  # calculate g(x)
        ratio = float(gofx) / M
        u = nprand.uniform(low=0, high=1)  # sample a uniform on (0,1)
        if u < ratio:
            accepted.append(x)
            # computationally calculate acceptance probability
            numaccept += 1
        numofiter += 1

    acceptprob = float(numaccept) / numofiter
    print("... acceptance probability ...")
    print(acceptprob)  # Should get in range of 65%
    print("... number of accepted proposals ...")
    print(numaccept)
    print("... total number of proposals ...")
    print(numofiter)

    # plot a histogram of accepted values
    plt.hist(accepted, bins=20, facecolor='g')
    plt.title("f(x)")
    plt.xlabel("accepted x")
    plt.ylabel("frequency")
    # fig = plt.gcf()
    plt.savefig('standardnormal.png')
    plt.show()
