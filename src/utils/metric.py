import numpy as np


def KL(arr_x, arr_y):
    out_arr = arr_x * np.log(np.divide(arr_x, 0.5 * (arr_x + arr_y)))
    out_arr = out_arr[~np.isnan(out_arr)]
    out = np.sum(out_arr)
    return out


def jsdiv(y_tst, y_mod):
    out = 0.5 * (KL(y_tst, y_mod) + KL(y_mod, y_tst))
    return out
