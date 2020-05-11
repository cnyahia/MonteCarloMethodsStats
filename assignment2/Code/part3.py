"""
This code is used to evaluate the integral using
a Gaussian Markov process.

@cnyahia
"""
import numpy as np
import matplotlib.pyplot as plt
import math


def round_nearest(x):
    return round(round(x / 0.05) * 0.05, -int(math.floor(math.log10(0.05))))


def gx(Xn):
    """
    This is the function g(x)
    :param x: sample from the Markov chain
    :return: g(x)
    """
    gofx = np.sqrt(2 * np.pi) / (1 + Xn**4)
    return gofx

if __name__ == '__main__':
    rows = np.arange(-1, 1, 0.05).tolist()
    dictvar = {}
    for row in rows:
        # start by generating a standard normal R.V.
        X = np.random.normal(0, 1)
        samples = []
        S = 0  # the sum in Ihat
        currentIhat = []
        iter = 1
        rho = 0.0
        numiter = 5000
        VarIhat = []
        while iter <= numiter:
            S = S + gx(X)
            samples.append(X)
            X = rho * X + np.sqrt(1 - rho**2) * np.random.normal(0, 1)
            currentIhat.append(S/iter)
            iter += 1
            if len(currentIhat) > 1:
                VarIhat.append(np.var(currentIhat, ddof=1))

        Ihat = S / numiter
        # print(Ihat)
        dictvar[round_nearest(row)] = VarIhat[-1]
        # print(row)
    print(dictvar)
    print(min(dictvar, key=dictvar.get))

    '''
    # plot a graph showing convergence to value of integral I
    Iterations = range(1, len(currentIhat) + 1)
    plt.plot(Iterations, currentIhat, 'g')
    plt.title("Convergence of Ihat to the Integral value")
    plt.xlabel("Iterations")
    plt.ylabel("Value of Ihat")
    plt.tight_layout()
    plt.savefig('Ihat0.png')
    plt.show()
    '''
