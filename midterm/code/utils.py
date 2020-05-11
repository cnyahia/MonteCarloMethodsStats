"""
Midterm for Monte Carlo methods in statistics
SDS386D.

This script implements some needed plots and
utilities

@cnyahia
"""
import numpy as np
import matplotlib.pyplot as plt


def fofx(x, y):
    """
    returns the value of fofx for x
    :param x: variable value
    :param y: function parameter
    :return: function value
    """
    fun = np.exp(-3*y*np.exp(x))
    return fun

if __name__ == '__main__':
    x_range = np.linspace(0,5,num=100)
    fun_vals = []
    y_range = [0.25]
    for y in y_range:
        temp = []
        for x in x_range:
            temp.append(fofx(x, y))
        fun_vals.append(temp)
    print("... plotting ...")
    for key, val in enumerate(y_range):
        plt.plot(x_range, fun_vals[key], label="y=%s" % str(val))
    plt.xlim(xmin=0, xmax=5)
    plt.legend()
    plt.title("g(x)=exp(-3yexp(x))")
    plt.xlabel("x")
    plt.ylabel("g(x)")
    plt.tight_layout()
    plt.savefig('gofx.png')
    plt.show()



