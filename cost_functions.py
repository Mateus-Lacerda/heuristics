import numpy as np


def rotated_high_conditional_elliptic_function(d: int, x_array: list) -> float:
    result = 0
    for i in range(1, d):
        result += pow(10, (i-1) / (d-1)) * pow(x_array[i], 2)
    return result


def wierstrass_function(d: int, x_array: list) -> float:
    result = 0
    a = 0.5
    b = 3
    kmax = 20
    op_1 = 0
    for i in range(d):
        for k in range(kmax+1):
            op_1 += (a ** k) * np.cos(2*np.pi*(b ** k)*(x_array[i] + 0.5))

    op_2 = 0
    for k in range(kmax+1):
        op_2 += (a ** k) * np.cos(2*np.pi*(b ** k)*0.5)

    result = op_1 - d * op_2

    return result
