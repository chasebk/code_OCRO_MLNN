#!/usr/bin/env python
# ------------------------------------------------------------------------------------------------------%
# Created by "Thieu Nguyen" at 00:51, 29/03/2020                                                        %
#                                                                                                       %
#       Email:      nguyenthieu2102@gmail.com                                                           %
#       Homepage:   https://www.researchgate.net/profile/Thieu_Nguyen6                                  %
#       Github:     https://github.com/thieunguyen5991                                                  %
# -------------------------------------------------------------------------------------------------------%

from numpy import where, maximum, power, multiply, exp
from numpy import tanh as mytanh

def itself(x):
    return x
def elu(x, alpha=1):
    return where(x < 0, alpha * (exp(x) - 1), x)
def relu(x):
    return maximum(0, x)
def tanh(x):
    return mytanh(x)
def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

def derivative_self(x):
    return 1
def derivative_elu(x, alpha=1):
    return where(x < 0, x + alpha, 1)
def derivative_relu(x):
    return where(x < 0, 0, 1)
def derivative_tanh(x):
    return 1 - power(x, 2)
def derivative_sigmoid(x):
    return multiply(x, 1-x)
