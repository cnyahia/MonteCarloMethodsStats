"""
This code is for problem 3 on the
midterm for Monte Carlo methods in
statistics course SDS386D.

The code implements a Metropolis-
Hastings algorithm were the proposal
density is a random walk bounded below
by 1

@cnyahia
"""

import numpy.random as nprand
import matplotlib.pyplot as plt


# define alpha
def alpha(xp, xm, power):
    """
    define the function alpha
    with all the special cases
    :param xp: x_plus
    :param xm: x_minus
    :param power: power law constant
    :return: alfa
    """
    if (xp != 1) and (xm != 1):
        ratio = float(xp ** -power) / (xm ** -power)
        alfa = min(1, ratio)
    elif (xp == 2) and (xm == 1):
        ratio = (1.0/2) * (float(xp ** -power) / (xm ** -power))
        alfa = min(1, ratio)
    elif (xp == 1) and (xm == 2):
        ratio = 2.0 * (float(xp ** -power) / (xm ** -power))
        alfa = min(1, ratio)
    else:
        raise Exception('... error computing alpha ...')

    return alfa


# implement the sampling from q
def sample_q(xm):
    """
    sample from q given the previous value
    q is a random walk
    :param xm: previous value
    :return: sample
    """
    die = nprand.uniform(low=0, high=1)
    if xm == 1:
        sample = 2

    else:
        if die <= 0.5:
            sample = xm + 1
        else:
            sample = xm - 1

    return sample


# implement the Metropolis-Hastings algorithm
if __name__ == '__main__':
    numofiter = 1  # iterator
    total_iter = 80000  # total iterations
    power = 3  # power alpha for the power law
    x_n = 3  # initial value
    samples = []
    number_accept = 0
    while numofiter <= total_iter:
        samples.append(x_n)
        x_star = sample_q(x_n)  # sample x_star from x_{n}
        u = nprand.uniform(low=0, high=1)  # sample from uniform
        if u < alpha(x_star, x_n, power):
            x_n = x_star  # set the value at the next iteration
            number_accept += 1
        # else, do nothing, x_n at next iteration is x_n
        numofiter += 1

    print("... acceptance probability ...")
    print(number_accept / total_iter)
    print("... plotting samples of power law ...")
    # plot a graph showing distribution
    plt.hist(samples, bins=50, facecolor='g', normed=True)
    plt.title("generated samples for power law")
    plt.xlabel("samples")
    plt.ylabel("normalized frequency")
    plt.tight_layout()
    plt.savefig('samples_plaw.png')
    plt.show()

    # verify
    k = 1
    k_total = 30
    plaw_fun = []
    k_vals = []
    while k <= k_total:
        k_vals.append(k)
        plaw_fun.append(k**(-power))
        k += 1

    print("... power law evaluated using function ...")
    plt.bar(k_vals, plaw_fun)
    plt.title("power law evaluated using function")
    plt.xlabel("k")
    plt.ylabel("k^{-3}")
    plt.tight_layout()
    plt.savefig('plaw_fun.png')
    plt.show()
