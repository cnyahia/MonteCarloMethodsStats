"""
This is the last problem of the Midterm
for Monte Carlo methods in statistics
SDS386D

The code implements Gibbs by using RS and
MH repeatedly

@cnyahia
"""

import numpy as np
import numpy.random as nprand
import matplotlib.pyplot as plt


# sample from h(x) for rejection sampling
def RS_sample_hx():
    """
    This function is used to sample h(x)
    :return: exponential sample
    """
    beta = 1.0 / 4  # scale parameter
    expon = nprand.exponential(beta)
    return expon


# rejection sampling, evaluate ratio g(x)/M
def RS_evaluate_ratio(x, y):
    """
    for a given value of x
    evaluate g(x)/M
    :param x: current val of x
    :return: ratio
    """
    ratio = np.exp(3 * y * (1 - np.exp(x)))
    return ratio


# MH proposal
def MH_proposal(y, sigma=1):
    """
    sample the log-normal MH proposal
    the parameters are mean and sigma of the
    normally distributed log(Y), where
    Y is log-normally distributed
    :param y: log-normal mean
    :param sigma: log-normal variance
    :return: sample
    """
    mean_of_logx = np.log(y)
    sample = nprand.lognormal(mean_of_logx, sigma)
    return sample


# define alpha for MH
def MH_alpha(yp, ym, x):
    """
    define alpha for the MH sampling of
    pi(yp|ym, x)
    :param yp: y_plus
    :param ym: y_minus
    :param x: current x
    :return: alpha
    """
    ratio_numerator = (yp**4) * np.exp(-3 * yp * np.exp(x))
    ratio_denom = (ym**4) * np.exp(-3 * ym * np.exp(x))
    ratio = ratio_numerator / ratio_denom
    alpha = min(1, ratio)
    return alpha


# implement the sampling algorithm
if __name__ == '__main__':
    samples_y = []
    samples_x = []
    Ihat = []  # integral estimator
    numerator = []  # numerator of Ihat
    x = 1  # initial starting value
    y = 1  # initial starting value
    numofiter = 1
    total_iter = 80000  # total iterations
    while numofiter <= total_iter:
        # sample y from x using MH
        samples_y.append(y)
        y_star = MH_proposal(y)  # q in MH
        u_MH = nprand.uniform(low=0, high=1)
        if u_MH < MH_alpha(y_star, y, x):
            y = y_star  # set the value at the next iteration
        # you have sampled y|x, now sample x|y using RS
        sampled_RS = False
        while not sampled_RS:
            x = RS_sample_hx()  # sample hx in RS
            u = nprand.uniform(low=0, high=1)
            g_by_M = RS_evaluate_ratio(x, y)
            if u < g_by_M:
                samples_x.append(x)  # append x
                sampled_RS = True  # end once sample is
                # obtained

        numerator.append(x * y)
        Ihat.append(float(sum(numerator)) / len(numerator))
        numofiter += 1

    print("... plotting samples of x ...")
    plt.hist(samples_x, bins=50, facecolor='g', normed=True)
    plt.title("normalized samples of x")
    plt.xlabel("samples")
    plt.ylabel("normalized frequency")
    plt.tight_layout()
    plt.savefig('p4_x_samples.png')
    plt.show()

    print("... plotting samples of y ...")
    plt.hist(samples_y, bins=50, facecolor='g', normed=True)
    plt.title("normalized samples of y")
    plt.xlabel("samples")
    plt.ylabel("normalized frequency")
    plt.tight_layout()
    plt.savefig('p4_y_samples.png')
    plt.show()

    print("... integral estimator ...")
    print(Ihat[-1])
    Iterations = range(1, len(Ihat) + 1)
    plt.plot(Iterations, Ihat, 'g')
    plt.title("integral estimator across iterations")
    plt.xlabel("Iterations")
    plt.ylabel("Ihat")
    plt.tight_layout()
    plt.savefig('p4_ihat.png')
    plt.show()

    print("... plotting pi(x,y) in two dimensions ...")
    fig, ax = plt.subplots()
    h = ax.hist2d(samples_x, samples_y, bins=50, normed=True)
    plt.title("distribution of pi(x, y)")
    # plt.tight_layout()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(h[3], ax=ax)
    plt.savefig('p4_pixy.png')
    plt.show()


