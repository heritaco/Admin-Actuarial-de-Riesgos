import numpy as np


def d1():
    return (np.log(St/K2) + (r-q+s2/2)(T-t))/(s*np.sqrt(T-t))


def d2():
    return d1 - s*(T-t)


def cgap():
    return S0*np.exp()
