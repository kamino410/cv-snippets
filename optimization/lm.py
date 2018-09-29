# coding: UTF-8

# Levenberg-Marquardt Method (Least squares problem)
# Sample for cosine curve fitting
#   (a, w, p, c) = minarg |y - f(t)|
#     f(x) = a * cos(w*t + p) + c
#   Truth : a = 2, w = 10, p = pi/2, c = 1

import numpy as np
from scipy.optimize import least_squares

ts = np.linspace(0.0, 1.0, 101)
ys = 2*np.cos(10*ts + np.pi/2)+1


def res_func(xs, ts, ys):
    return ys - xs[0] * np.cos(xs[1]*ts + xs[2]) - xs[3]


x0 = np.array([1.5, 8.0, 1.0, 0.5])

res = least_squares(res_func, x0, args=(ts, ys), method='lm')
print('success:', res.success)
print('x:', res.x)
