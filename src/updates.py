import numpy as np

from param import *
from stats import post_var


Npre = int(T_hat / dt)
Vhat = (sigma_Y ** 2) / T_hat  # prior variance
DELbias_mat = np.zeros(Nc)
Delta_s_t_mat = np.zeros((Nc, Nt))

T_hatS = dt * np.arange(1,13,1)

def updates(tau, sigma_Y, dZ_build, dZ, T_hat):
    Delta_s_t = np.zeros(1)
    Npre = int(T_hat / dt)
    Vhat = (sigma_Y ** 2) / T_hat  # prior variance
    DELbias_mat = np.zeros(Nt)
    Delta_s_t_mat = np.zeros((Nc, Nt))
    biasvec = dZ_build[-Npre:]

    for i in range(1, Nc):
        tau_short = tau[-i]
        dDelta_s_t = (
                             post_var(sigma_Y, Vhat, tau_short) / sigma_Y ** 2
              ) * (
                -Delta_s_t * dt + dZ_build[i - 1]
             )  # from eq(5)
        if i < Npre:
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(Delta_s_t, 0)  # newborns begin with 0 bias when there are not enough earlier observations
        else:
            DELbias = np.average(dZ_build[int(i - Npre): i]) / dt
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(
                Delta_s_t, DELbias
            )  # newborns begin with Npre earlier observations

    for j in range(Nt):
        dDelta_s_t = (
            post_var(sigma_Y, Vhat, tau) / sigma_Y**2
                     ) * (
            -Delta_s_t * dt + dZ[j]
        )  # from eq(5)
        if j < (Npre - 1):
            init_bias = (np.sum(biasvec[j+1:]) + np.sum(dZ[:j+1])) / T_hat
        else:
            init_bias = np.average(dZ[j+1 - Npre: j+1]) / dt

        DELbias_mat[j] = init_bias
        Delta_s_t += dDelta_s_t
        Delta_s_t = np.append(Delta_s_t[1:], init_bias)
        Delta_s_t_mat[j, :] = Delta_s_t

    return np.mean(Delta_s_t_mat[:, -100:])

m = 200
n = len(T_hatS)

A = np.empty((m, n))


for a in range(m):
    dZ_build = dt ** 0.5 * np.random.randn(int(Nc - 1))
    dZ = dt ** 0.5 * np.random.randn(Nt)
    for b, t_hat in enumerate(T_hatS):
        # print(t_hat)
        A[a, b] = updates(tau, sigma_Y, dZ_build, dZ, t_hat)
    print(a)
