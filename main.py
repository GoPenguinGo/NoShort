import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable, Tuple
from src.cohort_builder import build_cohorts
from src.cohort_simulator import simulate_cohorts
from src.param import *
import concurrent.futures

# TODO: @chingyulin: make cohort a class

# The main loop builds up the economy with a large number of cohorts, and simulates the stationary economy forward
for k in range(Mpaths):
# def simulate(k, Nc, dt, rho, nu, Vhat, mu_Y, sigma_Y, beta, T_hat):
    s = time.time()
    time_s = time.time()
    dZt = dt**0.5 * np.random.randn(int(Nt - 1))
    (
        IntVec,
        Xt,
        Delta_s_t,
        Yt,
        Zt,
        tau,
        MaxThetaDelta_s_t,
    ) = build_cohorts(dZt, Nc, dt, rho, nu, Vhat, mu_Y, sigma_Y, beta, T_hat)
    if time.time() - time_s > time_tolerance:
        print(f"It takes more than {time_tolerance}s to build up the cohorts")

    dZforbias = np.diff(Zt)  # dZt used in the build_cohorts function
    biasvec = dZforbias[-Npre:]

    dZt = dt**0.5 * np.random.randn(Nt)  # dZt forward
    Zt = np.cumsum(dZt)

    (
     Xt2,
     part1,
     mu_S,
     mu_S_s,
     mu_hat_S,
     r,
     theta,
     BIGF,
     BIGDELTA,
     BIGMAX,
     BIGPORT,
     BIGPOPU,
     BIGFCONDI,
     BIGDELTABARCONDI,
     dR,

    ) = simulate_cohorts(
        biasvec,
        dZt,
        Nt,
        Nc,
        tau,
        IntVec,
        Delta_s_t,
        MaxThetaDelta_s_t,
        dt,
        rho,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        sigma_S,
        beta,
        T_hat,
        Npre,
    )

    erp_S = mu_S - r
    erp_hat_S = mu_hat_S - r
    erp_S_s = mu_S_s - np.reshape(r, (Nt, 1))

    Z_matrix[k, :] = Zt
    dR_matrix[k, :] = dR
    # EtMAT[k, :] = np.transpose(Et)
    # VtMAT[k, :] = np.transpose(Vt)
    delta_matrix[k, :, :] = BIGDELTA
    r_matrix[k, :] = r
    theta_matrix[k, :] = theta
    mu_S_matrix[k, :] = mu_S
    mu_S_s_matrix[k, :, :] = mu_S_s
    mu_hat_S_matrix[k, :] = mu_hat_S
    erp_S_matrix[k, :] = erp_S
    erp_S_s_matrix[k, :, :] = erp_S_s
    erp_hat_S_matrix[k, :] = erp_hat_S
    port_matrix[k, :, :] = BIGPORT

    f_matrix[k, :, :] = BIGF
    fcondi_matrix[k, :] = BIGFCONDI
    popu_matrix[k, :] = BIGPOPU
    delta_condi_matrix[k, :] = BIGDELTABARCONDI
    
    print(time.time() - s)

change_popu_matrix = BIGPOPU[1:]/BIGPOPU[:-1]

########################################################################################################################
### GRAPHS:

# keep only the last 100 years data

# (1) Zt and Participation Rate from one random path
t = np.arange(0, T_cohort, dt)  # has the same length as Nc
random_paths = 1  # 1 can be any random number from 0 to Mpaths
y1 = Z_matrix[random_paths, :]
y2 = popu_matrix[random_paths, :]
fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)

ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Population holding stocks', color = color2)
ax2.plot(t, y2, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)

fig.suptitle('Zt and Participation Rate')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()

# (2) dZt and Log Change in Participation Rate
sample = np.arange(Nc - Nkeep - 1, Nc - 1, 20)
x1 = Z_matrix[:, sample] - Z_matrix[:, sample - 1]
y1 = np.log(popu_matrix[:, sample + 1] / popu_matrix[:, sample])
plt.scatter(x1, y1, marker = '.', c = 'b', alpha = 0.8)
plt.xlabel('dZt')
plt.ylabel('log changes in participation rate')
plt.title('dZt and Log Change in Participation Rate')
plt.show()


for k in range(Mpaths):
    corrMuSmuHat[k] = np.corrcoef(mu_hat_S_matrix[k], mu_S_matrix[k])[0, 1]
    for l in range(Nsamples):
        # a = int(l * stepcorr)
        # b = int((l + 1) * stepcorr)
        # corrZMUs_t[k, l] = np.corrcoef(Z_matrix[k, a:b], mu_S_s_matrix[k, a:b, Nc-1])[0, 1]
        # corrZport[k, l] = np.corrcoef(Z_matrix[k, a:b], port_matrix[k, a:b, Nc-1])[0, 1]
        # corrMU_sMUs_t[k, l] = np.corrcoef(mu_S_matrix[k, a:b], mu_S_s_matrix[k, a:b, Nc-1])[0, 1]
        corrZMUs_t[k, l] = np.corrcoef(Z_matrix[k, ], erp_S_s_matrix[k, :, -l])[0, 1]
        corrZport[k, l] = np.corrcoef(Z_matrix[k, ], port_matrix[k, :, -l])[0, 1]
        corrMU_sMUs_t[k, l] = np.corrcoef(erp_S_matrix[k, ], erp_S_s_matrix[k, :, -l])[0, 1]
        # muCst[k, l] = np.mean(muC_s_t[a:b])
        # logmuCst[k, l] = np.mean(muC_s_t[a:b]) - 0.5 * sum((sigmaC_s_t[a:b]) ** 2)
        # sigCst[k, l] = np.mean(sigmaC_s_t[a:b])
        # stdCst[k, l] = np.mean(abs(sigmaC_s_t[a:b]))





MaxAge = 100
MaxAgeN = int(MaxAge / Tsample)
tperiod = range(Tsample, 100 + Tsample, Tsample)
meanZport = np.mean(corrZport, axis=0)
meanZmus_t = np.mean(corrZMUs_t, axis=0)

# Compute the mean values from the simulation
meanMus = np.mean(corrMU_sMUs_t, axis=0)
# meanMuCst = np.mean(muCst, axis=0)
# meanSCst = np.mean(sigCst, axis=0)
# meanStdCst = np.mean(stdCst, axis=0)
# meanLogMuCst = np.mean(logmuCst, axis=0)

# Figures
# Figure 1 in the paper
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 10))
ax1.plot(tperiod, meanMus[:MaxAgeN])
ax2.plot(tperiod, meanZmus_t[:MaxAgeN])

# Figure 2 in the paper



#
# ks = [k for k in range(10)]
#
# def main():
#     with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
#         for k, result in zip(ks, executor.map(simulate, ks)):
#             print(f"{k} is done.")
#
#
# if __name__ == "__main__":
#     time_s = time.time()
#     main()
#     print(time.time() - time_s)
