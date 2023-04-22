import numpy as np


# SIMPLE MATHEMATICAL FUNCTIONS
def fun_exp(x, a, b, c):
    return a * np.exp(b * (x - c))

def fun_log(x,a,b,c,d):
    return a*np.log(-b*(x-c))-d*x**2