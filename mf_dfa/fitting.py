import numpy as np
import pandas as pd
from numba import njit


@njit
def coeff_matrix(X:np.ndarray, deg:int)->np.ndarray:

    matrix = np.zeros(shape=(X.shape[0], deg+1))

    const = np.ones_like(X)

    matrix[:, 0] = const

    matrix[:, 1] = X

    if deg > 1:
        for n in range(2, deg+1):
            matrix[:, n] = X**n

    return matrix

@njit
def solve(a, b):

    coeffs = np.linalg.lstsq(a, b)[0]

    return coeffs

@njit
def polyfit(X:np.ndarray, y:np.ndarray, deg:int)->np.ndarray:

    a = coeff_matrix(X, deg)

    coeffs = solve(a, y)

    return coeffs[::-1]

@njit
def polyeval(coeffs:np.ndarray, X:np.ndarray)->np.ndarray:

    result = np.zeros(shape=(len(X), coeffs.shape[1]))

    for j in range(coeffs.shape[1]):
        for coeff in coeffs[:, j]:
            result[:, j] = X * result[:, j] + coeff

    return result
