import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def gray_scott(t, y, lap_op, Du, Dv, F, k):
    n_samples = len(y)
    len_u = int(n_samples/2)
    u = y[:len_u]
    v = y[len_u:]
    # Parameters
    #Du, Dv, F, k = 0.16, 0.08, 0.035, 0.065
    #u, v = y
    # Laplacian
    u_xx = np.dot(lap_op, u)
    v_xx = np.dot(lap_op, v)
    # Reaction-diffusion equations
    du = Du * u_xx - u * v**2 + F * (1 - u)
    dv = Dv * v_xx + u * v**2 - (F + k) * v
    return np.concatenate((du, dv))

