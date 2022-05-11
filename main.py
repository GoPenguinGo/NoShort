import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable, Tuple
from src.cohort_builder import build_cohorts
from src.cohort_simulator import simulate_cohorts
from src.param import *
from src.stats import shocks, tau_calculator
import concurrent.futures
from numba import jit


# generate values that are fixed in the main loop
tau = np.arange(T_cohort, 0, -dt)
cohort_size = nu * np.exp(-nu * (tau - dt)) * dt  # cohort size when a new cohort is just born
cummu_popu = np.cumsum(cohort_size)
tau_cutoff1 = np.searchsorted(cummu_popu, 0.75)
tau_cutoff2 = np.searchsorted(cummu_popu, 0.5)
tau_cutoff3 = np.searchsorted(cummu_popu, 0.25)

# The main loop builds up the economy with a large number of cohorts, and simulates the stationary economy forward
for k in range(Mpaths):
# def simulate(k, Nc, dt, rho, nu, Vhat, mu_Y, sigma_Y, beta, T_hat):
    s = time.time()
    time_s = time.time()
    dZ_build = dt**0.5 * np.random.randn(int(Nc - 1))  # dZt for the build function
    biasvec = dZ_build[-Npre:]  # dZt used in the build_cohorts function
    dZ = dt ** 0.5 * np.random.randn(Nt)  # dZt for the simulate function
    (
        Z,
        Y,
    ) = shocks(
        dZ,
        mu_Y,
        sigma_Y,
        dt,
    )

    # baseline scenario
    (
        f_st,
        Delta_s_t,
        eta_st_ss,
        eta_bar,
        MaxThetaDelta_s_t,
        invest_tracker,
    ) = build_cohorts(dZ_build, Nc, dt, tau, rho, nu, Vhat, mu_Y, sigma_Y, beta, Npre, T_hat, mode1)

   # drop scenario
    (
        f_st_drop,
        Delta_s_t_drop,
        eta_st_ss_drop,
        eta_bar_drop,
        MaxThetaDelta_s_t_drop,
        invest_tracker_drop,
    ) = build_cohorts(dZ_build, Nc, dt, tau, rho, nu, Vhat, mu_Y, sigma_Y, beta, Npre, T_hat, mode2)

    # complete market scenario
    (
        f_st_comp,
        Delta_s_t_comp,
        eta_st_ss_comp,
        eta_bar_comp,
        MaxThetaDelta_s_t_comp,
        invest_tracker_comp,
    ) = build_cohorts(dZ_build, Nc, dt, tau, rho, nu, Vhat, mu_Y, sigma_Y, beta, Npre, T_hat, mode3)
    # if time.time() - time_s > time_tolerance:
    #     print(f"It takes more than {time_tolerance}s to build up the cohorts")

    # partial constraint scenario
    (
        f_st_free,
        Delta_s_t_free,
        eta_st_ss_free,
        eta_bar_free,
        d_eta_st_ss_free,
        invest_tracker_free,
        can_short_tracker_free,
    ) = build_cohorts_partial_constraint(dZ_build, Nc, dt, tau, cohort_size, rho, nu, Vhat, mu_Y, sigma_Y, beta, Npre, T_hat, mode4)

    (
        mu_S,
        mu_S_s,
        # mu_hat_S,
        r,
        theta,
        f,
        Delta,
        max,
        pi,
        parti,
        f_parti,
        Delta_bar_parti,
        dR,
        w,
        w_cohort,
        age,
        n_parti,
    ) = simulate_cohorts(
        Y,
        biasvec,
        dZ,
        Nt,
        Nc,
        tau,
        dt,
        rho,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        sigma_S,
        beta,
        omega,
        T_hat,
        Npre,
        mode1,
        cohort_size,
        f_st,
        Delta_s_t,
        eta_st_ss,
        eta_bar,
        MaxThetaDelta_s_t,
        invest_tracker,
    )

    (
        mu_S_drop,
        mu_S_s_drop,
        # mu_hat_S_drop,
        r_drop,
        theta_drop,
        f_drop,
        Delta_drop,
        max_drop,
        pi_drop,
        parti_drop,
        f_parti_drop,
        Delta_bar_parti_drop,
        dR_drop,
        w_drop,
        w_cohort_drop,
        age_drop,
        n_parti_drop,
    ) = simulate_cohorts(
        Y,
        biasvec,
        dZ,
        Nt,
        Nc,
        tau,
        dt,
        rho,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        sigma_S,
        beta,
        omega,
        T_hat,
        Npre,
        mode2,
        cohort_size,
        f_st_drop,
        Delta_s_t_drop,
        eta_st_ss_drop,
        eta_bar_drop,
        MaxThetaDelta_s_t_drop,
        invest_tracker_drop,
    )

    (
        mu_S_comp,
        mu_S_s_comp,
        # mu_hat_S_comp,
        r_comp,
        theta_comp,
        f_comp,
        Delta_comp,
        max_comp,
        pi_comp,
        parti_comp,
        f_parti_comp,
        Delta_bar_parti_comp,
        dR_comp,
        w_comp,
        w_cohort_comp,
        age_comp,
        n_parti_comp,
    ) = simulate_cohorts(
        Y,
        biasvec,
        dZ,
        Nt,
        Nc,
        tau,
        dt,
        rho,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        sigma_S,
        beta,
        omega,
        T_hat,
        Npre,
        mode3,
        cohort_size,
        f_st_comp,
        Delta_s_t_comp,
        eta_st_ss_comp,
        eta_bar_comp,
        MaxThetaDelta_s_t_comp,
        invest_tracker_comp,
    )

    (
        mu_S_free,
        mu_S_s_free,
        # mu_hat_S_free,
        r_free,
        theta_free,
        f_free,
        Delta_free,
        d_eta_free,
        pi_free,
        f_parti_free,
        Delta_bar_parti_free,
        dR_free,
        w_free,
        w_cohort_free,
        popu_parti_free,
        popu_can_short_free,
        popu_short_free,
        popu_long_free,
        f_parti_free,
        f_short_free,
        f_long_free,
        age_parti_free,
        age_short_free,
        age_long_free,
        n_parti_free,
    ) = simulate_cohorts_partial_constraint(
        Y,
        biasvec,
        dZ,
        Nt,
        Nc,
        tau,
        dt,
        rho,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        sigma_S,
        beta,
        omega,
        T_hat,
        Npre,
        mode4,
        cohort_size,
        f_st_free,
        Delta_s_t_free,
        eta_st_ss_free,
        eta_bar_free,
        d_eta_st_ss_free,
        invest_tracker_free,
        can_short_tracker_free,
    )

    erp_S = mu_S - r
    # erp_hat_S = mu_hat_S - r
    erp_S_s = mu_S_s - np.reshape(r, (Nt, 1))

    dZ_matrix[k, :] = dZ
    Z_matrix[k, :] = Z

    dR_matrix[k, :] = dR
    delta_matrix[k, :, :] = Delta
    r_matrix[k, :] = r
    theta_matrix[k, :] = theta
    mu_S_matrix[k, :] = mu_S
    mu_S_s_matrix[k, :, :] = mu_S_s
    # mu_hat_S_matrix[k, :] = mu_hat_S
    erp_S_matrix[k, :] = erp_S
    erp_S_s_matrix[k, :, :] = erp_S_s
    # erp_hat_S_matrix[k, :] = erp_hat_S
    pi_matrix[k, :, :] = pi

    f_matrix[k, :, :] = f
    f_parti_matrix[k, :] = f_parti
    parti_matrix[k, :] = parti
    Delta_bar_parti_matrix[k, :] = Delta_bar_parti
    w_matrix[k, :, :] = w
    w_cohort_matrix[k, :, :] = w_cohort
    age_matrix[k, :] = age

    ############ for the drop case: ###########

    erp_S_drop = mu_S_drop - r_drop
    # erp_hat_S_drop = mu_hat_S_drop - r_drop
    erp_S_s_drop = mu_S_s_drop - np.reshape(r_drop, (Nt, 1))

    dR_drop_matrix[k, :] = dR_drop
    delta_drop_matrix[k, :, :] = Delta_drop
    r_drop_matrix[k, :] = r_drop
    theta_drop_matrix[k, :] = theta_drop
    mu_S_drop_matrix[k, :] = mu_S_drop
    mu_S_s_drop_matrix[k, :, :] = mu_S_s_drop
    # mu_hat_S_matrix_drop[k, :] = mu_hat_S_drop
    erp_S_drop_matrix[k, :] = erp_S_drop
    erp_S_s_drop_matrix[k, :, :] = erp_S_s_drop
    # erp_hat_S_matrix_drop[k, :] = erp_hat_S_drop
    pi_drop_matrix[k, :, :] = pi_drop

    f_drop_matrix[k, :, :] = f_drop
    f_parti_drop_matrix[k, :] = f_parti_drop
    parti_drop_matrix[k, :] = parti_drop
    Delta_bar_parti_drop_matrix[k, :] = Delta_bar_parti_drop
    w_drop_matrix[k, :, :] = w_drop
    w_cohort_drop_matrix[k, :, :] = w_cohort_drop
    age_drop_matrix[k, :] = age_drop

    print(time.time() - s)
    print(k)

########################################################################################################################
### GRAPHS:

# keep only the last 100 years data
# todo: save the graphs
# (1.1) Zt and Participation Rate from one random path
t = np.arange(0, T_cohort, dt)  # has the same length as Nc
random_paths = 2  # can be any random number from 0 to Mpaths
y1 = Z_matrix[random_paths, :]
y2 = parti_matrix[random_paths, :]
y3 = age_matrix[random_paths, :]

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Population holding stocks', color = color2)
ax2.set_ylim([0,1])
ax2.plot(t, y2, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and Participation Rate')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Participation Rate' + '_keep' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Average age holding stocks', color = color2)
ax2.set_ylim([0,100])
ax2.plot(t, y3, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and Average Participant Age')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Average Participant Age' + '_keep' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('% of cohorts holding stocks', color = color2)
ax2.set_ylim([0,1])
ax2.plot(t, y4, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and % of Cohorts Participate')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Cohorts Participate' + '_keep' + '.jpg')
plt.show()

# (1.2) dZt and Log Change in Participation Rate
sample = np.arange(Nc - Nkeep, Nc, 20)
x1 = dZ_matrix[:, sample]
y1 = np.log(parti_matrix[:, sample] / parti_matrix[:, sample - 1])
plt.figure(2)
plt.scatter(x1, y1, marker = '.', c = 'b', alpha = 0.8)
plt.xlim([-1.5, 1.5])
plt.ylim([-0.8, 0.8])
plt.xlabel('dZt')
plt.ylabel('log changes in participation rate')
plt.title('dZt and Log Change in Participation Rate')
plt.savefig('dZt and Log Change in Participation Rate' + '_keep' + '.jpg')
plt.show()

############## for the drop case: ################

# (1.1) Zt and Participation Rate from one random path
t = np.arange(0, T_cohort, dt)  # has the same length as Nc
random_paths = 2  # 1 can be any random number from 0 to Mpaths
y1 = Z_matrix[random_paths, :]
y2 = parti_drop_matrix[random_paths, :]
y3 = age_drop_matrix[random_paths, :]


fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Population holding stocks, drop', color = color2)
ax2.set_ylim([0,1])
ax2.plot(t, y2, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and Participation Rate, if Drop')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Participation Rate' + '_drop' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Average age holding stocks', color = color2)
ax2.set_ylim([0,100])
ax2.plot(t, y3, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and Average Participant Age, if Drop')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Average Participant Age' + '_drop' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('% of cohorts holding stocks', color = color2)
ax2.set_ylim([0,1])
ax2.plot(t, y4, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and % of Cohorts Participate, if Drop')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Cohorts Participate' + '_drop' + '.jpg')
plt.show()

# (1.2) dZt and Log Change in Participation Rate
sample = np.arange(Nc - Nkeep, Nc, 20)
x1 = dZ_matrix[:, sample]
y1 = np.log(parti_drop_matrix[:, sample] / parti_drop_matrix[:, sample - 1])
plt.scatter(x1, y1, marker = '.', c = 'b', alpha = 0.8)
plt.xlim([-1.5, 1.5])
plt.ylim([-0.8, 0.8])
plt.xlabel('dZt')
plt.ylabel('log changes in participation rate')
plt.title('dZt and Log Change in Participation Rate, if Drop')
plt.savefig('dZt and Log Change in Participation Rate' + '_drop' + '.jpg')
plt.show()

###################################################

y1 = Z
y2 = parti_comp
y3 = age_comp
y4 = n_parti_comp

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Population longing stocks', color = color2)
ax2.set_ylim([0,1])
ax2.plot(t, y2, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and Participation Rate, complete market')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Participation Rate' + '_comp' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('Average age longing stocks', color = color2)
ax2.set_ylim([0,100])
ax2.plot(t, y3, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and Average Participant Age, complete market')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Average Participant Age' + '_comp' + '.jpg')
plt.show()

fig, ax1 = plt.subplots()
color1 = 'r'
ax1.set_xlabel('time in simulation, one random path')
ax1.set_ylabel('Zt', color = color1)
ax1.plot(t, y1, color = color1, linewidth = 0.8)
ax1.tick_params(axis='y', labelcolor = color1)
ax2 = ax1.twinx()
color2 = 'b'
ax2.set_ylabel('% of cohorts longing stocks', color = color2)
ax2.set_ylim([0,1])
ax2.plot(t, y4, color = color2, linewidth = 0.8)
ax2.tick_params(axis='y', labelcolor = color2)
fig.suptitle('Zt and % of Cohorts Participate, complete market')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Zt and Cohorts Participate' + '_comp' + '.jpg')
plt.show()

#############################
#### comparative graphs


#
# # (2.1)
# for k in range(Mpaths):
#     corrMuSmuHat[k] = np.corrcoef(mu_hat_S_matrix[k], mu_S_matrix[k])[0, 1]
#     for l in range(Nsamples):
#         # a = int(l * stepcorr)
#         # b = int((l + 1) * stepcorr)
#         # corrZMUs_t[k, l] = np.corrcoef(Z_matrix[k, a:b], mu_S_s_matrix[k, a:b, Nc-1])[0, 1]
#         # corrZport[k, l] = np.corrcoef(Z_matrix[k, a:b], port_matrix[k, a:b, Nc-1])[0, 1]
#         # corrMU_sMUs_t[k, l] = np.corrcoef(mu_S_matrix[k, a:b], mu_S_s_matrix[k, a:b, Nc-1])[0, 1]
#         corrZMUs_t[k, l] = np.corrcoef(Z_matrix[k, ], erp_S_s_matrix[k, :, -l])[0, 1]
#         corrZport[k, l] = np.corrcoef(Z_matrix[k, ], port_matrix[k, :, -l])[0, 1]
#         corrMU_sMUs_t[k, l] = np.corrcoef(erp_S_matrix[k, ], erp_S_s_matrix[k, :, -l])[0, 1]
#         # muCst[k, l] = np.mean(muC_s_t[a:b])
#         # logmuCst[k, l] = np.mean(muC_s_t[a:b]) - 0.5 * sum((sigmaC_s_t[a:b]) ** 2)
#         # sigCst[k, l] = np.mean(sigmaC_s_t[a:b])
#         # stdCst[k, l] = np.mean(abs(sigmaC_s_t[a:b]))
#
#
#
#
#
# MaxAge = 100
# MaxAgeN = int(MaxAge / Tsample)
# tperiod = range(Tsample, 100 + Tsample, Tsample)
# meanZport = np.mean(corrZport, axis=0)
# meanZmus_t = np.mean(corrZMUs_t, axis=0)
#
# # Compute the mean values from the simulation
# meanMus = np.mean(corrMU_sMUs_t, axis=0)
# # meanMuCst = np.mean(muCst, axis=0)
# # meanSCst = np.mean(sigCst, axis=0)
# # meanStdCst = np.mean(stdCst, axis=0)
# # meanLogMuCst = np.mean(logmuCst, axis=0)
#
# # Figures
# # Figure 1 in the paper
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 10))
# ax1.plot(tperiod, meanMus[:MaxAgeN])
# ax2.plot(tperiod, meanZmus_t[:MaxAgeN])

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
