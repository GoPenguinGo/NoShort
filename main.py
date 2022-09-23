import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable, Tuple
from src.simulation import simulate_SI
from src.cohort_builder import build_cohorts_SI
from src.cohort_simulator import simulate_cohorts_SI
from src.param import *
from src.stats import shocks, tau_calculator, good_times
from numba import jit
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tabulate as tabulate

# different scenarios
# mode_learn = 'drop'
mode_learn = 'keep'
# mode_trade = 'complete'
# mode_trade = 'w_constraint'
mode_trade = 'partial_constraint_old'
# mode_trade = 'partial_constraint_rich'

dZ_matrix = np.load('dZ_matrix.npy')
dZ_build_matrix = np.load('dZ_build_matrix.npy')
dZ_SI_matrix = np.load('dZ_SI_matrix.npy')
dZ_SI_build_matrix = np.load('dZ_SI_build_matrix.npy')

# The main loop builds up the economy with a large number of cohorts, and simulates the stationary economy forward
# survey_view_parti_matrix = np.zeros((Mpaths, Nt))  # average perceived risk premia among investors
# survey_view_long_only_matrix = np.zeros((Mpaths, Nt))
# survey_view_can_short_matrix = np.zeros((Mpaths, Nt))
# survey_view_parti_young_matrix = np.zeros((Mpaths, Nt))
# survey_view_parti_old_matrix = np.zeros((Mpaths, Nt))
# popu_parti_young_matrix = np.zeros((Mpaths, Nt))
# popu_parti_old_matrix = np.zeros((Mpaths, Nt))
# age_parti_matrix = np.zeros((Mpaths, Nt))
# r_matrix = np.zeros((Mpaths, Nt))
# erp_S_matrix = np.zeros((Mpaths, Nt))
# Delta_bar_parti_matrix = np.zeros((Mpaths, Nt))
# dY_Y_matrix = np.zeros((Mpaths, Nt))
# obj_rp_matrix = np.zeros((Mpaths, Nt))


# dZ_matrix = np.load('dZ_matrix.npy')
# dZ_build_matrix = np.load('dZ_build_matrix.npy')
# dZ_SI_matrix = np.load('dZ_SI_matrix.npy')
# dZ_SI_build_matrix = np.load('dZ_SI_build_matrix.npy')

N = 50
phi_vector = [0, 0.4, 0.8]
n_phi = len(phi_vector)

# define colors
color1 = 'black'
color2 = 'mediumblue'
color3 = 'darkgreen'
color4 = 'orange'
color5 = 'red'
color6 = 'gold'
color7 = 'g'

colors = ['darkmagenta', 'midnightblue', 'green', 'saddlebrown', 'darkgreen', 'firebrick', 'purple', 'blue',
          'olivedrab', 'darkviolet']

modes_trade = ['complete', 'w_constraint', 'partial_constraint']
modes_learn = ['keep', 'drop']
n_modes_trade = len(modes_trade)
n_modes_learn = len(modes_learn)
index_Z_Ys = [6, 0]  # indices of a good and a bad shock for Z^Y
index_Z_SIs = [0, 13]  # indices of a good and a bad shock for Z^SI



theta_matrix = np.empty((N, n_phi, Nt))
popu_parti_matrix = np.empty((N, n_phi, Nt))
market_view_matrix = np.empty((N, n_phi, Nt))
survey_view_matrix = np.empty((N, n_phi, Nt))
Delta_matrix = np.empty((N, n_phi, Nt, Nc))
pi_matrix = np.empty((N, n_phi, Nt, Nc))
mu_st_rt_matrix = np.empty((N, n_phi, Nt, Nc))
r_matrix = np.zeros((N, n_phi, Nt))
belief_dispersion_matrix = np.zeros((N, n_phi, Nt))
dR_matrix = np.zeros((N, n_phi, Nt))
delta_bar_matrix = np.zeros((N, n_phi, Nt))

# run the program for different values of phi, and store the results
for j in range(N):
    print(j)
    dZ = dZ_matrix[j]
    dZ_build = dZ_build_matrix[j]
    dZ_SI = dZ_SI_matrix[j]
    dZ_SI_build = dZ_SI_build_matrix[j]

    for i, phi in enumerate(phi_vector):
        (
            r,
            theta,
            f,
            Delta,
            max,
            pi,
            popu_parti,
            f_parti,
            Delta_bar_parti,
            dR,
            w_cohort,
            age_parti,
            n_parti,
        ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta, phi,
                        Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size)
        # invest_tracker = pi > 0
        Delta_matrix[j, i] = Delta
        pi_matrix[j, i] = pi
        theta_matrix[j, i] = theta
        r_matrix[j, i] = r
        theta_mat = np.transpose(np.tile(theta, (Nc, 1)))
        mu_st_rt_matrix[j, i] = theta_mat + Delta
        popu_parti_matrix[j, i] = popu_parti
        market_view_matrix[j, i] = np.average(Delta, axis=1, weights=f)
        delta_bar_matrix[j, i] = Delta_bar_parti
        survey_view_matrix[j, i] = np.average(Delta, axis=1, weights=cohort_size)
        belief_dispersion_matrix[j, i] = np.std(Delta, axis = 1)  # todo: maybe add weights
        dR_matrix[j, i] = dR

# for k in range(Mpaths):
#     s = time.time()
#     time_s = time.time()
#
#     dZ = dZ_matrix[k]
#     dZ_build = dZ_build_matrix[k]
#     dZ_SI = dZ_SI_matrix[k]
#     dZ_SI_build = dZ_SI_build_matrix[k]
#
#     # if k == Mpaths - 1:  # in the last round, use the shocks seen in the slides
#     #     dZ_build = np.load('dZt_build_demo.npy')  # dZt for the build function
#     #     dZ = np.load('dZt_demo.npy')  # dZt for the simulate function
#     #
#     # else:
#     #     dZ_build = dt ** 0.5 * np.random.randn(int(Nc - 1))  # dZt for the build function
#     #     dZ = dt ** 0.5 * np.random.randn(Nt)  # dZt for the simulate function
#
#     dY_Y_matrix[k, :] = mu_Y * dt + sigma_Y * dZ
#
#     # dZ_matrix[k, :] = dZ
#     # Z = np.cumsum(dZ)
#     # Z_matrix[k, :] = Z
#
#     if mode_trade == 'complete' or mode_trade == 'w_constraint':
#         (
#             r,
#             theta,
#             f,
#             Delta,
#             max,
#             pi,
#             popu_parti,
#             f_parti,
#             Delta_bar_parti,
#             dR,
#             w_cohort,
#             age_parti,
#             n_parti,
#         ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta, phi,
#                         Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size)
#         invest_tracker = pi > 0
#
#         y1 = Delta
#
#     # else:
#     #     print('Error! Mode not defined')
#     #     break
#     theta_mat = np.transpose(np.tile(theta, (Nc, 1)))
#
#     erp_S = mu_S - r
#     # erp_S_s = mu_S_s - np.reshape(r, (Nt, 1))
#
#     dR_matrix[k] = dR
#     # delta_matrix[k] = Delta
#     r_matrix[k] = r
#     # theta_matrix[k] = theta
#     # mu_S_matrix[k] = mu_S
#     # mu_S_s_matrix[k] = mu_S_s
#     erp_S_matrix[k] = erp_S
#     # erp_S_s_matrix[k] = erp_S_s
#     # pi_matrix[k] = pi
#
#     # w_matrix[k] = w
#     # w_cohort_matrix[k] = w_cohort
#     # n_parti_matrix[k] = n_parti
#
#     popu_parti_matrix[k] = popu_parti
#     # popu_can_short_matrix[k] = popu_can_short
#     # popu_short_matrix[k] = popu_short
#     # popu_long_matrix[k] = popu_long
#     # f_matrix[k] = f
#     # f_parti_matrix[k] = f_parti
#     # f_short_matrix[k] = f_short
#     # f_long_matrix[k] = f_long
#     age_parti_matrix[k] = age_parti
#     # age_short_matrix[k] = age_short
#     # age_long_matrix[k] = age_long
#
#     # invest_tracker_matrix[k] = invest_tracker
#     # can_short_tracker_matrix[k] = can_short_tracker
#     # long_indicator_matrix[k] = long
#     # short_indicator_matrix[k] = short
#
#     Delta_bar_parti_matrix[k] = Delta_bar_parti
#     # Delta_bar_long_matrix[k] = Delta_bar_long
#     # Delta_bar_short_matrix[k] = Delta_bar_short
#
#     obj_rp_matrix[k] = theta * sigma_S
#
#     parti_track = cohort_size_mat * invest_tracker
#     popu_parti_young_matrix[k] = np.sum(parti_track[:, tau_cutoff1:], axis=1)  # the first age quartile
#     popu_parti_old_matrix[k] = np.sum(parti_track[:, tau_cutoff3:tau_cutoff2], axis=1)  # the third age quartile
#
#     # # for graphs specific to one random path:
#     # if dZ_build == np.load('dZt_build_demo.npy'):
#     #     if mode == 'comp':
#     #         y21 = theta
#     #         y31 = r
#     #         y41 = Delta_bar_parti
#     #     if mode == 'drop':
#     #         y1 = Delta
#     #         y22 = theta
#     #         y32 = r
#     #         y42 = Delta_bar_parti / f_parti
#     #         y43 = f_parti
#     #         y44 = popu_parti
#     #         invest = pi > 0
#     #         parti_rate = invest * cohort_size
#     #         y5 = parti_rate
#     #         y6 = pi
#     #     if mode == 'free':
#     #         y71 = theta
#     #         y72 = popu_parti
#     #         y73 = popu_short
#     #         y74 = Delta_bar_long
#     #         y75 = Delta_bar_short
#
#     print(time.time() - s)
#     print(mode)
#     print(k)
#
#     # for graphs using information from all simulated paths


# #######################################
# ########## ONE RANDOM PATH ############
# #######################################
#
# # the x-axis






# ######################################
# ############ GRAPH ONE ###############
# ######################################

# ONE SPECIFIC PATH:

# generate data for the graphs:

dZ_build = dZ_build_matrix[0]
dZ_SI_build = dZ_SI_build_matrix[0]  # fix the shocks at the buildup stage


theta_compare = np.empty((n_modes, 2, 2, n_phi, Nt))
popu_parti_compare = np.empty((n_modes, 2, 2, n_phi, Nt))
market_view_compare = np.empty((n_modes, 2, 2, n_phi, Nt))
survey_view_compare = np.empty((n_modes, 2, 2, n_phi, Nt))
Delta_compare = np.empty((n_modes, 2, 2, n_phi, Nt, Nc))
pi_compare = np.empty((n_modes, 2, 2, n_phi, Nt, Nc))
r_compare = np.zeros((n_modes, 2, 2, n_phi, Nt))
belief_dispersion_compare = np.zeros((n_modes, 2, 2, n_phi, Nt))
delta_bar_compare = np.zeros((n_modes, 2, 2, n_phi, Nt))
cons_compare = np.zeros((n_modes, 2, 2, n_phi, Nt, Nc))
cohort_size_mat = np.transpose(np.tile(cohort_size, (Nc, 1)))

for g, mode_trade in enumerate(modes_trade):
    for i, index_Z_Y in enumerate(index_Z_Ys):
        dZ = dZ_matrix[index_Z_Y]
        log_Yt = np.cumsum((mu_Y - 0.5 * sigma_Y ** 2) * dt + sigma_Y * dZ)
        log_Yt_mat = np.transpose(np.tile(log_Yt, (Nc, 1)))

        for j, index_Z_SI in enumerate(index_Z_SIs):
            dZ_SI = dZ_SI_matrix[index_Z_SI]
            for k, phi in enumerate(phi_vector):
                (
                    r,
                    theta,
                    f,
                    Delta,
                    max,
                    pi,
                    popu_parti,
                    f_parti,
                    Delta_bar_parti,
                    dR,
                    w_cohort,
                    age_parti,
                    n_parti,
                ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta,
                                phi,
                                Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size)
                # invest_tracker = pi > 0
                Delta_compare[g, i, j, k] = Delta
                pi_compare[g, i, j, k] = pi
                theta_compare[g, i, j, k] = theta
                r_compare[g, i, j, k] = r
                theta_mat = np.transpose(np.tile(theta, (Nc, 1)))
                popu_parti_compare[g, i, j, k] = popu_parti
                market_view_compare[g, i, j, k] = np.average(Delta, axis=1, weights=f)
                delta_bar_compare[g, i, j, k] = Delta_bar_parti
                survey_view_compare[g, i, j, k] = np.average(Delta, axis=1, weights=cohort_size)
                belief_dispersion_compare[g, i, j, k] = np.std(Delta, axis=1)  # todo: maybe add weights
                cons_compare[g, i, j, k] = log_Yt_mat + np.log(f / cohort_size_mat)

# graphs:
cohort_matrix_list = [pi_compare, Delta_compare, cons_compare]

# Delta
red_cases = ['G', 'B']
yellow_cases = ['G', 'B']
nn = 3  # number of cohorts illustrated
t = np.arange(0, T_cohort, dt)
length = len(t)
starts = np.zeros(nn)
cohort_labels = ['cohort 1', 'cohort 2', 'cohort 3']
var_y_labels = ['Investment in stock market', 'Estimation error', 'Log consumption']
figure_labels = ['pi', 'Delta', 'Log cons']

for i, cohort_matrix in enumerate(cohort_matrix_list):
    for j, index_Z_Y in enumerate(index_Z_Ys):
        Z = np.cumsum(dZ_matrix[index_Z_Y])
        red_case = red_cases[j]
        for k, index_Z_SI in enumerate(index_Z_SIs):
            Z_SI = np.cumsum(dZ_SI_matrix[index_Z_SI])
            yellow_case = yellow_cases[k]
            for l, phi in enumerate(phi_vector):
                phi = phi_vector[l]
                y_interest = cohort_matrix[1, j, k, l]  # with short-sale constraint
                y_interest_time_series = np.zeros((nn, length))
                for m in range(nn):
                    start = int((m + 1) * 100 * (1 / dt))
                    starts[m] = start * dt
                    for n in range(length):
                        if n < start:
                            y_interest_time_series[m, n] = np.nan
                        else:
                            cohort_rank = length - (n - start) - 1
                            y_interest_time_series[m, n] = y_interest[n, cohort_rank]
                if i == 0:
                    parti_time_series = (y_interest_time_series > 0)
                    fig, ax1 = plt.subplots(figsize=(15, 5))
                    ax1.set_xlabel('Time in simulation, one random path')
                    ax1.set_ylabel('Zt', color=color5)
                    ax1.plot(t, Z, color=color5, linewidth=0.5, label='Z^Y_t')
                    ax1.plot(t, Z_SI, color=color6, linewidth=0.5, label='Z^SI_t')
                    ax1.tick_params(axis='y', labelcolor=color5)
                    ax2 = ax1.twinx()
                    ax2.set_ylabel(var_y_labels[i], color=colors[0])
                    ax2.set_ylim([0.01, 20])
                    for m in range(nn):
                        y_cohort = y_interest_time_series[m]
                        plt.vlines(starts[m], ymax=20, ymin=0, color='grey', linestyle='--', linewidth=0.4)
                        ax2.plot(t, y_cohort, label = cohort_labels[m], color=colors[m], linewidth=0.4)
                    ax2.tick_params(axis='y', labelcolor=colors[0])
                    fig.tight_layout()  # otherwise the right y-label is slightly clipped
                    plt.savefig(red_case + yellow_case + ' Zt and ' + figure_labels[i] + ' time series' + str(round(phi, 2)) + '.png', dpi=500)
                    plt.show()
                    plt.close()

                else:
                    if i == 1:
                        lower = -0.5
                        upper = 0.5
                    else:
                        lower = 4
                        upper = 15
                    switch = abs(parti_time_series[:, 1:] ^ parti_time_series[:, :-1])
                    col = np.reshape(switch[:, -1], (3, -1))
                    switch = np.append(switch, col, axis=1)
                    fig, ax1 = plt.subplots(figsize=(15, 5))
                    ax1.set_xlabel('Time in simulation, one random path')
                    ax1.set_ylabel('Zt', color=color5)
                    ax1.plot(t, Z, color=color5, linewidth=0.5, label='Z^Y_t')
                    ax1.plot(t, Z_SI, color=color6, linewidth=0.5, label='Z^SI_t')
                    ax1.tick_params(axis='y', labelcolor=color5)
                    ax2 = ax1.twinx()
                    ax2.set_ylabel(var_y_labels[i], color=color2)
                    ax2.set_ylim([lower, upper])
                    for m in range(nn):
                        # switch[m, starts[m]] = 1
                        y_cohort = y_interest_time_series[m]
                        y_cohort_N = np.ma.masked_where(parti_time_series[0] == 1, y_cohort)
                        y_cohort_P = np.ma.masked_where(parti_time_series[0] == 0, y_cohort)
                        y_cohort_switch = np.ma.masked_where(switch[0] == 0, y_cohort)
                        plt.vlines(starts[m], ymax=upper, ymin=lower, color='grey', linestyle='--', linewidth=0.4)
                        ax2.plot(t, y_cohort_P, label=cohort_labels[m], color=colors[m], linewidth=0.4)
                        ax2.plot(t, y_cohort_N,  color=colors[m], linewidth=0.4, linestyle='dotted')
                        if m == 0:
                            ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o', label='state switch')
                        else:
                            ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o')
                    ax2.tick_params(axis='y', labelcolor=color2)
                    fig.tight_layout()  # otherwise the right y-label is slightly clipped
                    plt.legend()
                    plt.savefig(red_case + yellow_case + ' Zt and ' + figure_labels[i] + ' time series' + str(round(phi, 2)) + '.png', dpi=500)
                    plt.show()
                    plt.close()




# for k in range(n_phi):
#     y1 = Delta_matrix[0, k]
#     y6 = pi_matrix[0, k]
#     phi = phi_vector[k]
#
#     # Delta:
#
#     Delta_time_series = np.zeros((nn, length))
#     parti_time_series = np.zeros((nn, length))
#     for i in range(nn):
#         start = int((i + 1) * 100 * (1 / dt))
#         for j in range(length):
#             if j < start:
#                 Delta_time_series[i, j] = np.nan
#                 parti_time_series[i, j] = np.nan
#             else:
#                 cohort_rank = length - (j - start) - 1
#                 Delta_time_series[i, j] = y1[j, cohort_rank]
#                 parti_time_series[i, j] = (y6[j, cohort_rank] > 0)
#     switch = abs(parti_time_series[:, 1:] - parti_time_series[:,:-1])
#     col = np.reshape(switch[:, -1], (3, -1))
#     switch = np.append(switch, col, axis=1)
#
#     y11 = Delta_time_series[0]
#     y11_N = np.ma.masked_where(parti_time_series[0] == 1, y11)
#     y11_P = np.ma.masked_where(parti_time_series[0] == 0, y11)
#     y11_switch = np.ma.masked_where(switch[0] == 0, y11)
#     y12 = Delta_time_series[1]
#     y12_N = np.ma.masked_where(parti_time_series[1] == 1, y12)
#     y12_P = np.ma.masked_where(parti_time_series[1] == 0, y12)
#     y12_switch = np.ma.masked_where(switch[1] == 0, y12)
#     y13 = Delta_time_series[2]
#     y13_N = np.ma.masked_where(parti_time_series[2] == 1, y13)
#     y13_P = np.ma.masked_where(parti_time_series[2] == 0, y13)
#     y13_switch = np.ma.masked_where(switch[2] == 0, y13)
#
#     fig, ax1 = plt.subplots(figsize=(15, 5))
#     ax1.set_xlabel('Time in simulation, one random path')
#     ax1.set_ylabel('Zt', color=color5)
#     ax1.plot(t, y0, color=color5, linewidth=0.5, label = 'Z^Y_t')
#     ax1.plot(t, y01, color=color6, linewidth=0.5, label = 'Z^SI_t')
#     ax1.tick_params(axis='y', labelcolor=color5)
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Estimation error', color=color2)
#     ax2.set_ylim([-0.5, 0.5])
#     ax2.plot(t, y11_P, color=color2, linewidth=0.4, label = 'cohort 1')
#     ax2.plot(t, y11_N, color=color2, linewidth=0.4, linestyle = 'dotted')
#     ax2.scatter(t, y11_switch, color='red', s=10, marker = 'o', label = 'state switch')
#     ax2.plot(t, y12_P, color=color3, linewidth=0.4, label = 'cohort 2')
#     ax2.plot(t, y12_N, color=color3, linewidth=0.4, linestyle = 'dotted')
#     ax2.scatter(t, y12_switch, color='red', s=10, marker = 'o')
#     ax2.plot(t, y13_P, color=color4, linewidth=0.4, label = 'cohort 3')
#     ax2.plot(t, y13_N, color=color4, linewidth=0.4, linestyle = 'dotted')
#     ax2.scatter(t, y13_switch, color='red', s=10, marker = 'o')
#     ax2.tick_params(axis='y', labelcolor=color2)
#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#     plt.legend()
#     plt.savefig('Zt and Delta time series' + str(round(phi, 2)) + '.png', dpi=500)
#     plt.show()
#     plt.close()
#
#
# # consumption
# y_log_Yt = np.cumsum((mu_Y - 0.5 * sigma_Y ** 2) * dt + sigma_Y * dZ_matrix[0])
# y_log_Yt_mat = np.transpose(np.tile(y_log_Yt, (Nc, 1)))
# cohort_size_mat = np.transpose(np.tile(cohort_size, (Nc, 1)))
# for k in range(n_phi):
#     y6 = pi_matrix[0, k]
#     phi = phi_vector[k]
#     y_fst = f_matrix[k]
#     # y_fst = f_matrix[0, k]
#     y1 = y_log_Yt_mat + np.log(y_fst / cohort_size_mat)
#     # y1 = np.log(y_fst / cohort_size_mat)
#
#     # Delta:
#     nn = 3
#     length = len(t)
#     Delta_time_series = np.zeros((nn, length))
#     parti_time_series = np.zeros((nn, length))
#     for i in range(nn):
#         start = int((i + 1) * 100 * (1 / dt))
#         for j in range(length):
#             if j < start:
#                 Delta_time_series[i, j] = np.nan
#                 parti_time_series[i, j] = np.nan
#             else:
#                 cohort_rank = length - (j - start) - 1
#                 Delta_time_series[i, j] = y1[j, cohort_rank]
#                 parti_time_series[i, j] = (y6[j, cohort_rank] > 0)
#     switch = abs(parti_time_series[:, 1:] - parti_time_series[:,:-1])
#     col = np.reshape(switch[:, -1], (3, -1))
#     switch = np.append(switch, col, axis=1)
#
#     y11 = Delta_time_series[0]
#     y11_N = np.ma.masked_where(parti_time_series[0] == 1, y11)
#     y11_P = np.ma.masked_where(parti_time_series[0] == 0, y11)
#     y11_switch = np.ma.masked_where(switch[0] == 0, y11)
#     y12 = Delta_time_series[1]
#     y12_N = np.ma.masked_where(parti_time_series[1] == 1, y12)
#     y12_P = np.ma.masked_where(parti_time_series[1] == 0, y12)
#     y12_switch = np.ma.masked_where(switch[1] == 0, y12)
#     y13 = Delta_time_series[2]
#     y13_N = np.ma.masked_where(parti_time_series[2] == 1, y13)
#     y13_P = np.ma.masked_where(parti_time_series[2] == 0, y13)
#     y13_switch = np.ma.masked_where(switch[2] == 0, y13)
#
#     fig, ax1 = plt.subplots(figsize=(15, 5))
#     ax1.set_xlabel('Time in simulation, one random path')
#     ax1.set_ylabel('Zt', color=color5)
#     ax1.plot(t, y0, color=color5, linewidth=0.5, label = 'Z^Y_t')
#     ax1.plot(t, y01, color=color6, linewidth=0.5, label = 'Z^SI_t')
#     ax1.tick_params(axis='y', labelcolor=color5)
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Log consumption', color=color2)
#     ax2.set_ylim([0, 20])
#     ax2.plot(t, y11_P, color=color2, linewidth=0.4, label = 'cohort 1')
#     ax2.plot(t, y11_N, color=color2, linewidth=0.4, linestyle = 'dotted')
#     ax2.scatter(t, y11_switch, color='red', s=10, marker = 'o', label = 'state switch')
#     ax2.plot(t, y12_P, color=color3, linewidth=0.4, label = 'cohort 2')
#     ax2.plot(t, y12_N, color=color3, linewidth=0.4, linestyle = 'dotted')
#     ax2.scatter(t, y12_switch, color='red', s=10, marker = 'o')
#     ax2.plot(t, y13_P, color=color4, linewidth=0.4, label = 'cohort 3')
#     ax2.plot(t, y13_N, color=color4, linewidth=0.4, linestyle = 'dotted')
#     ax2.scatter(t, y13_switch, color='red', s=10, marker = 'o')
#     ax2.tick_params(axis='y', labelcolor=color2)
#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#     plt.legend()
#     plt.savefig('Zt and Consumption time series' + str(round(phi, 2)) + '.png', dpi=500)
#     plt.show()
#     # plt.close()
#
# # pi: illustrate who are investing and when do they quit
# nn = 10
# length = len(t)
# pi_time_series = np.zeros((nn, length))
# starts = np.zeros(nn)
# for k in range(n_phi):
#     phi = phi_vector[k]
#     y6 = pi_matrix[0, k]
#     for i in range(nn):
#         start = int((i + 5) * 25 * (1 / dt))
#         starts[i] = start * dt
#         for j in range(length):
#             if j < start:
#                 pi_time_series[i, j] = np.nan
#             else:
#                 cohort_rank = length - (j - start) - 1
#                 a = y6[j, cohort_rank]
#                 pi_time_series[i, j] = a
#                 # if a == 0:
#                 #     pi_time_series[i, j + 1: j + 8] = 0
#                 #     pi_time_series[i, j + 8:] = np.nan
#                 #     break
#
#     fig, ax1 = plt.subplots(figsize=(15, 5))
#     ax1.set_xlabel('Time in simulation, one random path')
#     ax1.set_ylabel('Zt', color=color5)
#     ax1.plot(t, y0, color=color5, linewidth=0.5, label='Z^Y_t')
#     ax1.plot(t, y01, color=color6, linewidth=0.5, label='Z^SI_t')
#     ax1.tick_params(axis='y', labelcolor=color5)
#     ax2 = ax1.twinx()
#     ax2.set_ylabel('Investment in stock market', color=color2)
#     ax2.set_ylim([0.01, 25])
#     for i in range(nn):
#         y6i = pi_time_series[i]
#         plt.vlines(starts[i], ymax=25, ymin=0, color='grey', linestyle='--', linewidth=0.4)
#         ax2.plot(t, y6i, color=colors[i], linewidth=0.4)
#     ax2.tick_params(axis='y', labelcolor=color2)
#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#     plt.savefig('Zt and pi time series' + str(round(phi, 2)) + '.png', dpi=500)
#     plt.show()
#     # plt.close()

# popu, theta, market view and survey view:
market_matrix_list = [theta_compare, r_compare, popu_parti_compare, survey_view_compare, market_view_compare, delta_bar_compare, belief_dispersion_compare]
y_title_list = ['Market price of risk', 'Interest rate', 'Participation rate', 'Survey view', 'Market view', 'Market view_parti', 'Belief dispersion']

# Graphs for the short-sale constraint case, comparing different phi values
for i, market_matrix in enumerate(market_matrix_list):
    y_title = y_title_list[i]
    for j, index_Z_Y in enumerate(index_Z_Ys):
        Z = np.cumsum(dZ_matrix[index_Z_Y])
        red_case = red_cases[j]
        for k, index_Z_SI in enumerate(index_Z_SIs):
            Z_SI = np.cumsum(dZ_SI_matrix[index_Z_SI])
            yellow_case = yellow_cases[k]
            y = market_matrix[1, j, k]

            fig, ax1 = plt.subplots(figsize=(15, 5))
            ax1.set_xlabel('Time in simulation, one random path')
            ax1.set_ylabel('Zt', color=color5)
            ax1.plot(t, Z, color=color5, linewidth=0.5, label='Z^Y_t')
            ax1.plot(t, Z_SI, color=color6, linewidth=0.5, label='Z^SI_t')
            ax1.tick_params(axis='y', labelcolor=color5)
            ax2 = ax1.twinx()
            ax2.set_ylabel(y_title, color=color2)
            lower = np.nanmin(market_matrix)
            upper = np.nanmax(market_matrix)
            ax2.set_ylim([lower, upper])

            for i in range(n_phi):
                label_i = 'phi = ' + str(round(phi_vector[i], 2))
                yy = y[i]
                ax2.plot(t, yy, color=colors[i], linewidth=0.4, label=label_i)
            ax2.tick_params(axis='y', labelcolor=color2)
            plt.legend()
            # fig.suptitle('Zt and Market Price of Risk')
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.savefig(red_case + yellow_case + ' Zt and ' + y_title + ' different phi'+ '.png', dpi=500)
            plt.show()
            plt.close()


# Comparing with the complete market case:
y_title_list = ['Market price of risk', 'Interest rate', 'Survey view', 'Market view']
market_matrix_list = [theta_compare, r_compare, survey_view_compare, market_view_compare]
label_modes = ['complete market', 'short-sale constraint']
for i, market_matrix in enumerate(market_matrix_list):
    y_title = y_title_list[i]
    for j, index_Z_Y in enumerate(index_Z_Ys):
        Z = np.cumsum(dZ_matrix[index_Z_Y])
        red_case = red_cases[j]
        for k, index_Z_SI in enumerate(index_Z_SIs):
            Z_SI = np.cumsum(dZ_SI_matrix[index_Z_SI])
            yellow_case = yellow_cases[k]
            for l, phi in enumerate(phi_vector):
                y = market_matrix[:, j, k, l]
                fig, ax1 = plt.subplots(figsize=(15, 5))
                ax1.set_xlabel('Time in simulation, one random path')
                ax1.set_ylabel('Zt', color=color5)
                ax1.plot(t, Z, color=color5, linewidth=0.5, label='Z^Y_t')
                ax1.plot(t, Z_SI, color=color6, linewidth=0.5, label='Z^SI_t')
                ax1.tick_params(axis='y', labelcolor=color5)
                ax2 = ax1.twinx()
                ax2.set_ylabel(y_title, color=color2)
                lower = np.nanmin(market_matrix)
                upper = np.nanmax(market_matrix)
                ax2.set_ylim([lower, upper])
                for i in range(2):
                    yy = y[i]
                    ax2.plot(t, yy, color=colors[i], linewidth=0.4, label=label_modes[i])
                ax2.tick_params(axis='y', labelcolor=color2)
                plt.legend()
                # fig.suptitle('Zt and Market Price of Risk')
                fig.tight_layout()  # otherwise the right y-label is slightly clipped
                plt.savefig(red_case + yellow_case + 'vs complete' +' Zt and ' + y_title + str(round(phi, 2)) + '.png', dpi=500)
                plt.show()
                plt.close()



# for j, y_mat in enumerate(y_list):
#     y_title = y_title_list[j]
#     y = y_mat[0]
#
#     fig, ax1 = plt.subplots(figsize=(15, 5))
#     ax1.set_xlabel('Time in simulation, one random path')
#     ax1.set_ylabel('Zt', color=color5)
#     ax1.plot(t, y0, color=color5, linewidth=0.5, label='Z^Y_t')
#     ax1.plot(t, y01, color=color6, linewidth=0.5, label='Z^SI_t')
#     ax1.tick_params(axis='y', labelcolor=color5)
#     ax2 = ax1.twinx()
#     ax2.set_ylabel(y_title, color=color2)
#     scale = 1 if j <= 1 else 0.25
#     ax2.set_ylim([-scale, scale])
#
#     for i in range(n_phi):
#         label_i = 'phi = ' + str(round(phi_vector[i],2))
#         yy = y[i]
#         ax2.plot(t, yy, color=colors[i], linewidth=0.4, label=label_i)
#     ax2.tick_params(axis='y', labelcolor=color2)
#     plt.legend()
#     # fig.suptitle('Zt and Market Price of Risk')
#     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#     plt.savefig('Zt and ' + y_title + ' different phi' + '.png', dpi=500)
#     plt.show()
#     plt.close()




# illustrate how beliefs respond to shocks
age_group = 5
corrstep = int(age_group / dt)
age_max = 100
n_group = int(age_max / age_group - 1)
# todo: rename a, b
a0 = np.empty((N, n_phi, n_group))
b0 = np.empty((N, n_phi, n_group))

for i in range(N):  # N paths
    dZ_Y = dZ_matrix[i]
    dZ_SI = dZ_SI_matrix[i]
    for j in range(n_phi):
        beliefs = Delta_matrix[i, j] * sigma_Y + mu_Y
        for k in range(n_group):
            if k == 0:
                beliefs_group = np.mean(beliefs[:, -corrstep * (k + 1):],
                                        axis=1)
            else:
                beliefs_group = np.mean(beliefs[:, -corrstep * (k + 1):-corrstep * k],
                                        axis=1)

            a0[i, j, k] = np.corrcoef(dZ_Y, beliefs_group)[1, 0]
            b0[i, j, k] = np.corrcoef(dZ_SI, beliefs_group)[1, 0]

corr_a0 = np.average(a0, axis=0)
corr_b0 = np.average(b0, axis=0)
var_list = [corr_a0, corr_b0]
var_label = ['corr(dz^Y_t, mu_st)', 'corr(dz^SI_t, mu_st)']
x_age = np.linspace(age_group, age_max, n_group)
for i, var in enumerate(var_list):
    y_label = var_label[i]
    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax1.set_xlabel('Age')
    ax1.set_ylabel(y_label, color=color5)
    for j, phi in enumerate(phi_vector):
        y = var[j]
        phi_label = str(round(phi,2))
        ax1.plot(x_age, y, color=colors[j], label = phi_label, linewidth=0.5)
    plt.legend()
    # fig.suptitle('Zt and Market Price of Risk')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(y_label + '.png', dpi=500)
    plt.show()



horizons = [3, 6, 12, 24]
n_horizon = len(horizons)
report = ['coef', 't-stats', 'R-sqrd']
n_report = len(report)
var_names = ['Participation rate', 'Belief dispersion']
regression_results = np.empty((N, n_phi, n_horizon, len(var_names), n_report))
# predictive regression of stock returns on pariticipation rate
for i in range(N):
    for j in range(n_phi):
        excess_return_vector = np.cumsum(dR_matrix[i, j, 1: ] - r_matrix[i, j, :-1] * dt)
        popu_parti_vector = popu_parti_matrix[i, j, :-1]
        belief_dispersion_vector = belief_dispersion_matrix[i, j, :-1]
        x_list = [popu_parti_vector, belief_dispersion_vector]
        for k, horizon in enumerate(horizons):
            y_horizon1 = (excess_return_vector[horizon:] - excess_return_vector[: -horizon]) / (dt * horizon)  # make sure the timing alligns
            y_horizon = y_horizon1.reshape(-1, 1)
            for l, x_horizon_raw in enumerate(x_list):
                x_horizon1 = x_horizon_raw[: -horizon]
                x_horizon1 = x_horizon1 / np.std(x_horizon1)
                x_horizon1 = x_horizon1.reshape(-1, 1)
                x_horizon = sm.add_constant(x_horizon1)


                model = sm.OLS(y_horizon, x_horizon)
                est = model.fit()
                regression_results[i, j, k, l, 0] = est.params[1]
                regression_results[i, j, k, l, 1] = est.tvalues[1]
                regression_results[i, j, k, l, 2] = est.rsquared

mean_regression_results = np.mean(regression_results, axis=0)
header = ['(1) phi = 0', '(2) phi = 0.4', '(3) phi = 0.8']
# present the regression results in tables:

for j, var in enumerate(var_names):
    for i, horizon in enumerate(horizons):
        reg_data = np.empty((n_report, n_phi))
        for k in range(n_phi):
            reg_data[:, k] = mean_regression_results[k, i, j]
        report1 = [var, 't-stats', 'R-sqrd']
        print(var, ', ' + str(horizon) + ' months')
        print(tabulate.tabulate(reg_data, headers=header, showindex=report1, floatfmt=".4f", tablefmt='fancy_grid'))


# What predicts entry and exit?
# Entry:




# Replicate figure 1 in REStud
age_group = 5
corrstep = int(age_group / dt)
age_max = 100
n_group = int(age_max / age_group - 1)
mu_t_S_rt_matrix = theta_matrix * sigma_S
a = np.empty((N, n_phi, n_group))
b = np.empty((N, n_phi, n_group))
c = np.empty((N, n_phi, n_group))

for i in range(N):  # N paths
    dZ_Y = dZ_matrix[i]
    dZ_SI = dZ_SI_matrix[i]
    for j in range(n_phi):
        mu_t_S_rt = mu_t_S_rt_matrix[i, j]
        mu_st_S_rt = mu_st_rt_matrix[i, j]
        for k in range(n_group):
            if k == 0:
                mu_st_S_rt_group = np.mean(mu_st_S_rt[:, -corrstep * (k + 1):], axis=1)
            else:
                mu_st_S_rt_group = np.mean(mu_st_S_rt[:, -corrstep * (k + 1):-corrstep * k], axis=1)

            a[i, j, k] = np.corrcoef(mu_t_S_rt, mu_st_S_rt_group)[1, 0]
            b[i, j, k] = np.corrcoef(dZ_Y, mu_st_S_rt_group)[1, 0]
            c[i, j, k] = np.corrcoef(dZ_SI, mu_st_S_rt_group)[1, 0]

corr_a = np.average(a, axis=0)
corr_b = np.average(b, axis=0)
corr_c = np.average(c, axis=0)
var_list = [corr_a, corr_b, corr_c]
var_label = ['corr(mu^S-r_t, mu^S_{s,t}-r_t)', 'corr(dz^Y_t, mu^S_{s,t}-r_t)', 'corr(dz^SI_t, mu^S_{s,t}-r_t)']
x_age = np.linspace(age_group, age_max, n_group)
for i, var in enumerate(var_list):
    y_label = var_label[i]
    fig, ax1 = plt.subplots(figsize=(10, 10))
    ax1.set_xlabel('Age')
    ax1.set_ylabel(y_label, color=color5)
    for j, phi in enumerate(phi_vector):
        y = var[j]
        phi_label = str(round(phi,2))
        ax1.plot(x_age, y, color=colors[j], label = phi_label, linewidth=0.5)
    plt.legend()
    # fig.suptitle('Zt and Market Price of Risk')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(y_label + '.png', dpi=500)
    plt.show()


# #######################################
# ############# GRAPH TWO ###############
# #######################################
# # plot the market price of risk
# fig, ax1 = plt.subplots(figsize=(15, 5))
# ax1.set_xlabel('Time in simulation, one random path')
# ax1.set_ylabel('Zt', color=color5)
# ax1.plot(t, y0, color=color5, linewidth=0.5)
# ax1.tick_params(axis='y', labelcolor=color5)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Market price of risk', color=color2)
# ax2.set_ylim([-1, 1])
# ax2.plot(t, y21, color=color2, linewidth=0.4, label='Complete market')
# ax2.plot(t, y22, color=color3, linewidth=0.4, label='Short-sale constraint')
# ax2.hlines(sigma_Y, xmin=0, xmax=500, color='purple', linestyles='--', linewidth=0.8, label='Representative agent')
# ax2.tick_params(axis='y', labelcolor=color2)
# plt.legend()
# # fig.suptitle('Zt and Market Price of Risk')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('Zt and market price of risk' + '.png', dpi=500)
# plt.show()
#
# # plot the interest rate
# y33 = rho + mu_Y - sigma_Y ** 2
#
# fig, ax1 = plt.subplots(figsize=(15, 5))
# ax1.set_xlabel('Time in simulation, one random path')
# ax1.set_ylabel('Zt', color=color5)
# ax1.plot(t, y0, color=color5, linewidth=0.5)
# ax1.tick_params(axis='y', labelcolor=color5)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Interest rate (annual)', color=color2)
# ax2.set_ylim([0, 0.05])
# ax2.plot(t, y31, color=color2, linewidth=0.4, label='Complete market')
# ax2.plot(t, y32, color=color3, linewidth=0.4, label='Short-sale constraint')
# ax2.hlines(y33, xmin=0, xmax=500, color='purple', linestyles='--', linewidth=0.8, label='Representative agent')
# ax2.tick_params(axis='y', labelcolor=color2)
# plt.legend()
# # fig.suptitle('Zt and Market Price of Risk')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('Zt and interest rate' + '.png', dpi=500)
# plt.show()
#
# #######################################
# ############ GRAPH THREE ##############
# #######################################
#
# fig, ax1 = plt.subplots(figsize=(15, 5))
# ax1.set_xlabel('Time in simulation, one random path')
# ax1.set_ylabel('Zt', color=color5)
# ax1.plot(t, y0, color=color5, linewidth=0.5)
# ax1.tick_params(axis='y', labelcolor=color5)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Consumption weighted bias', color=color2)
# ax2.set_ylim([-0.5, 1])
# ax2.plot(t, y41, color='purple', linewidth=0.4, linestyle='--', label='Market view, complete market')
# ax2.plot(t, y42, color=color3, linewidth=0.4, label='Market view, no shorting')
# ax2.plot(t, y43, color=color4, linewidth=0.4, label='Participant consumption share')
# ax2.plot(t, y44, color=color2, linewidth=0.4, label='Participation rate')
# ax2.tick_params(axis='y', labelcolor=color2)
# plt.legend()
# # fig.suptitle('Zt and market view')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('Zt and market bias' + '.png', dpi=500)
# plt.show()
#
# # Regression
#
#
# #######################################
# ############ GRAPH  FOUR ##############
# #######################################
# # who are the investors
# y_label = 'participation composition of age groups'
# y51 = np.sum(y5[:, tau_cutoff1:], axis=1)
# y52 = np.sum(y5[:, tau_cutoff2:], axis=1)
# y53 = np.sum(y5[:, tau_cutoff3:], axis=1)
# y54 = np.sum(y5[:, ], axis=1)
# fig, ax1 = plt.subplots(figsize=(15, 5))
# ax1.set_ylabel('Participation composition of age groups', color=color2)
# ax1.set_ylim([0, 1])
# ax1.fill_between(t, y51, color='steelblue', linewidth=0.4, label='20 < Age <= 35, youngest quartile')
# ax1.fill_between(t, y52, y51, color='darkseagreen', linewidth=0.4, label='35 < Age <= 55')
# ax1.fill_between(t, y53, y52, color='moccasin', linewidth=0.4, label='55 < Age <= 89')
# ax1.fill_between(t, y54, y53, color='pink', linewidth=0.4, label='Age > 89, oldest quartile')
# ax1.tick_params(axis='y', labelcolor=color2)
# plt.legend()
# ax2 = ax1.twinx()
# ax2.set_xlabel('Time in simulation, one random path')
# ax2.set_ylabel('Zt', color=color5)
# ax2.plot(t, y0, color=color5, linewidth=0.5)
# ax2.tick_params(axis='y', labelcolor=color5)
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('Zt and participation composition' + '.png', dpi=500)
# plt.show()
#
# # illustrate who are investing and when do they quit
# nn = 10
# length = len(t)
# pi_time_series = np.zeros((nn, length))
# starts = np.zeros(nn)
# for i in range(nn):
#     start = int((i + 5) * 25 * (1 / dt))
#     starts[i] = start * dt
#     for j in range(length):
#         if j < start:
#             pi_time_series[i, j] = np.nan
#         else:
#             cohort_rank = length - (j - start) - 1
#             a = y6[j, cohort_rank]
#             pi_time_series[i, j] = a
#             if a == 0:
#                 pi_time_series[i, j + 1: j + 8] = 0
#                 pi_time_series[i, j + 8:] = np.nan
#                 break
#
# colors = ['darkmagenta', 'midnightblue', 'green', 'saddlebrown', 'darkgreen', 'firebrick', 'purple', 'blue',
#           'olivedrab', 'darkviolet']
# fig, ax1 = plt.subplots(figsize=(15, 5))
# ax1.set_xlabel('Time in simulation, one random path')
# ax1.set_ylabel('Zt', color=color5)
# ax1.plot(t, y0, color=color5, linewidth=0.5)
# ax1.tick_params(axis='y', labelcolor=color5)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Investment in stock market', color=color2)
# ax2.set_ylim([-0.1, 25])
# for i in range(nn):
#     y6i = pi_time_series[i]
#     plt.vlines(starts[i], ymax=25, ymin=0, color='grey', linestyle='--', linewidth=0.4)
#     ax2.plot(t, y6i, color=colors[i], linewidth=0.4)
# ax2.tick_params(axis='y', labelcolor=color2)
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('Zt and pi time series' + '.png', dpi=500)
# plt.show()
#
# #######################################
# ############ GRAPH  FIVE ##############
# #######################################
# # The rich can short
# # compare and plot theta
# fig, ax1 = plt.subplots(figsize=(15, 5))
# ax1.set_xlabel('Time in simulation, one random path')
# ax1.set_ylabel('Zt', color=color5)
# ax1.plot(t, y0, color=color5, linewidth=0.5)
# ax1.tick_params(axis='y', labelcolor=color5)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Market price of risk', color=color2)
# ax2.set_ylim([-1, 1])
# ax2.plot(t, y21, color=color2, linewidth=0.4, label='Complete market')
# ax2.plot(t, y22, color=color3, linewidth=0.4, label='Short-sale constraint')
# ax2.plot(t, y71, color='magenta', linewidth=0.4, label='Rich can short')
# # ax2.hlines(sigma_Y, xmin=0, xmax=500, color='purple', linestyles='--', linewidth=0.8, label='Representative agent')
# ax2.tick_params(axis='y', labelcolor=color2)
# plt.legend()
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('Zt and market price of risk, rich free' + '.png', dpi=500)
# plt.show()
#
# # Compare the bias between investors having long and short positions
# fig, ax1 = plt.subplots(figsize=(15, 5))
# ax1.set_xlabel('Time in simulation, one random path')
# ax1.set_ylabel('Zt', color=color5)
# ax1.plot(t, y0, color=color5, linewidth=0.5)
# ax1.tick_params(axis='y', labelcolor=color5)
# ax2 = ax1.twinx()
# ax2.set_ylabel('Market price of risk', color=color2)
# ax2.set_ylim([-0.5, 1])
# ax2.plot(t, y72, color='darkblue', linewidth=0.6, label='% investors')
# ax2.plot(t, y73, color='darkgreen', linewidth=0.6, label='% Short sellers')
# ax2.plot(t, y74, color='blue', linewidth=0.4, label='Average bias long')
# ax2.plot(t, y75, color='magenta', linewidth=0.4, label='Average bias short')
# # ax2.hlines(sigma_Y, xmin=0, xmax=500, color='purple', linestyles='--', linewidth=0.8, label='Representative agent')
# ax2.tick_params(axis='y', labelcolor=color2)
# plt.legend()
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig('Zt and participation, rich free' + '.png', dpi=500)
# plt.show()
#
# #######################################
# ############ GRAPH  SIX ###############
# #######################################
# #
# # test if subjective risk premia comove less with shocks / cyclicality of perceived risk premia
# # sensitivity of subjective vs. objective risk premia to business cycle indicator (dY/Y)
# # x: output growth from time t-T to time t
# # y: subjective and objective risk premia at tme t
# horizons = [1, 3, 6, 12, 24]
# m = len(horizons)
# results_obj_matrix = np.empty((Mpaths, m, 3))
# results_sub_matrix = np.empty((Mpaths, m, 3))
# header = []
# for i in range(Mpaths):
#     x_path = np.cumsum(dY_Y_matrix[i])
#
#     for j, horizon in enumerate(horizons):
#         if i == 0:
#             header_j = str(horizon) + '-month'
#             header.append(header_j)
#         x = (x_path[horizon:-horizon] - x_path[:-horizon * 2]) / (horizon * dt)
#         x = x / np.std(x)
#         x = x.reshape(-1, 1)
#         x = sm.add_constant(x)
#
#         y_sub_path = survey_view_parti_matrix[i, horizon:-horizon]
#         y_sub = y_sub_path.reshape(-1, 1)
#
#         y_obj_path = obj_rp_matrix[i, horizon:-horizon]
#         y_obj = y_obj_path.reshape(-1, 1)
#
#         # Objective risk premia:
#         model = sm.OLS(y_obj, x)
#
#         est = model.fit()
#         results_obj_matrix[i, j, 0] = est.params[1]
#         results_obj_matrix[i, j, 1] = est.tvalues[1]
#         results_obj_matrix[i, j, 2] = est.rsquared
#
#         # Subjective risk premia:
#         model = sm.OLS(y_sub, x)
#         est = model.fit()
#         results_sub_matrix[i, j, 0] = est.params[1]
#         results_sub_matrix[i, j, 1] = est.tvalues[1]
#         results_sub_matrix[i, j, 2] = est.rsquared
#
# result_obj = np.mean(results_obj_matrix, axis=0)
# result_sub = np.mean(results_sub_matrix, axis=0)
#
# # to table:
# index = ['coef', 't-stats', 'R2']
# n = len(index)
# for i in range(2):
#     reg_data = np.empty((n, m))
#     var = result_obj if i == 0 else result_sub
#     for j in range(n):
#         reg_data[j] = var[:,j]
#     print('result_obj' if i == 0 else 'result_sub')
#     print(tabulate.tabulate(reg_data, headers=header, showindex=index, floatfmt=".4f", tablefmt='fancy_grid'))
#
#






#######################################
########### GRAPH  SEVEN ##############
#######################################

# regressing difference in participarion on difference in experienced stock market returns
# over all simulated paths, and take average coefficient across all the paths

# young_age_cut = Nt - 20 / dt
# old_age_cut = Nt - 40 / dt
# young_prior = 20 / dt
# old_prior = 50 / dt
# diff_exprienced_growth = np.zeros((Mpaths, Nt - old_prior))
# diff_participation_rate = np.zeros((Mpaths, Nt - old_prior))
# for i in range(Nt - old_prior):
#     old_experienced_growth = np.average(dZ_matrix[:, i : i + old_prior])
#     young_experienced_growth = np.average(dZ_matrix[:, (i + old_prior - young_prior) : i + old_prior])
#     diff_exprienced_growth[:, i] = old_experienced_growth - young_experienced_growth
#
#     old_participation_rate = np.sum(y5[:, :old_age_cut], axis=1)  # not y5, but participation rate in matrix
#     young_participation_rate = np.sum(y5[:, young_age_cut:], axis=1)
#     diff_participation_rate[:, i] = old_participation_rate - young_participation_rate
#
# a = np.zeros(Mpaths)
#
# for j in range(Mpaths):
#     model = LinearRegression().fit(diff_exprienced_growth[j], diff_participation_rate[j])
#     a[j] = model.coef_

#######################################
########### GRAPH  EIGHT ##############
#######################################

# describe the mean and variance of beliefs (wealth-weighted) against belief of the marginal investor
# specific to one path
# relates to the information index. right now beliefs of non-participants make little sense

# marginal_belief = (-theta_drop) * sigma_Y + mu_Y


#######################################
############ GRAPH  NINE ##############
#######################################

# describe the predictive power of participation rate


start_t = 0
x_variables = [popu_parti_matrix, popu_parti_young_matrix, popu_parti_old_matrix, age_parti_matrix]
m = len(x_variables)
# np.cumsum(dR_matrix[i])
coeff_matrix1 = np.zeros((Mpaths, m, 3))
pvalue_matrix1 = np.zeros((Mpaths, m, 3))
tstats_matrix1 = np.zeros((Mpaths, m, 3))
rsqrd_matrix1 = np.zeros((Mpaths, m, 2))

for i in range(Mpaths):
    path_y = erp_S_matrix[i]
    path_x2 = survey_view_parti_matrix[i]
    path_x2 = path_x2 / np.std(path_x2)
    for j, var in enumerate(x_variables):
        path_x = var[i]
        y_raw = path_y[start_t:]  # equity risk premium at time t

        # univariate regressions
        x1_raw = path_x[start_t:]  # participation rate at time t
        x1_raw = x1_raw / np.std(x1_raw)
        x1_lag = x1_raw.reshape(-1, 1)
        x1_lag2 = sm.add_constant(x1_lag)
        y_predict = y_raw.reshape(-1, 1)

        model = sm.OLS(y_predict, x1_lag2)
        est = model.fit()
        coeff_matrix1[i, j, 0] = est.params[1]
        pvalue_matrix1[i, j, 0] = est.pvalues[1]
        tstats_matrix1[i, j, 0] = est.tvalues[1]
        rsqrd_matrix1[i, j, 0] = est.rsquared

        # bivariate regressions
        x2_lag = path_x2[start_t:]

        x_lag = np.append(x1_lag, x2_lag)
        x_lag = np.transpose(x_lag.reshape(2, -1))
        x_lag = sm.add_constant(x_lag)

        y_predict = y_raw.reshape(-1, 1)

        model = sm.OLS(y_predict, x_lag)
        est = model.fit()
        coeff_matrix1[i, j, 1] = est.params[1]
        coeff_matrix1[i, j, 2] = est.params[2]
        pvalue_matrix1[i, j, 1] = est.pvalues[1]
        pvalue_matrix1[i, j, 2] = est.pvalues[2]
        tstats_matrix1[i, j, 1] = est.tvalues[1]
        tstats_matrix1[i, j, 2] = est.tvalues[2]
        rsqrd_matrix1[i, j, 1] = est.rsquared

reg_coeffs1 = np.average(coeff_matrix1, axis=0)
reg_pvalues1 = np.average(pvalue_matrix1, axis=0)
reg_tstats1 = np.average(tstats_matrix1, axis=0)
reg_rsqrd1 = np.average(rsqrd_matrix1, axis=0)
reg_data = np.empty((5,8))
header = ['(1) parti rate', '(2)', '(3) parti rate, young', '(4)', '(5) parti rate, old', '(6)', '(7) parti age', '(8)']
index = ['coef', 't-stats', 'coef_x2', 't-stats_x2', 'R2']
for i in range(4):
    reg_data[0, i * 2] = reg_coeffs1[i, 0]
    reg_data[1, i * 2] = reg_tstats1[i, 0]
    reg_data[2, i * 2] = np.nan
    reg_data[3, i * 2] = np.nan
    reg_data[4, i * 2] = reg_rsqrd1[i, 0]

    reg_data[0, i * 2 + 1] = reg_coeffs1[i, 1]
    reg_data[1, i * 2 + 1] = reg_tstats1[i, 1]
    reg_data[2, i * 2 + 1] = reg_coeffs1[i, 2]
    reg_data[3, i * 2 + 1] = reg_tstats1[i, 2]
    reg_data[4, i * 2 + 1] = reg_rsqrd1[i, 1]

print(tabulate.tabulate(reg_data, headers=header, showindex = index,  floatfmt=".4f", tablefmt='fancy_grid'))



####
horizons = [1, 3, 6, 12, 36, 60, 120]
x_variables = [popu_parti_matrix, popu_parti_young_matrix, popu_parti_old_matrix, age_parti_matrix]
# x_variables = [popu_parti_matrix, popu_parti_young_matrix, popu_parti_old_matrix, age_parti_matrix]
m = len(x_variables)
n = len(horizons)
# np.cumsum(dR_matrix[i])
coeff_matrix2 = np.zeros((Mpaths, n, m, 3))
pvalue_matrix2 = np.zeros((Mpaths, n, m, 3))
tstats_matrix2 = np.zeros((Mpaths, n, m, 3))
rsqrd_matrix2 = np.zeros((Mpaths, n, m, 2))

for i in range(Mpaths):
    path_y = np.cumsum(dR_matrix[i])
    path_r = np.cumsum(r_matrix[i])
    path_x2 = survey_view_parti_matrix[i]
    path_x2 = path_x2 / np.std(path_x2)
    for j,horizon in enumerate(horizons):
        y_raw = (path_y[start_t + horizon + 1:] - path_y[start_t + 1: -horizon]) / (horizon * dt)
        #dR is the return from t-1 to t, and thus have to move 1 to have returns from t to t+1
        y_predict = y_raw.reshape(-1, 1)
        for k, var in enumerate(x_variables):
            path_x = var[i]
            # univariate regressions
            x1_raw = path_x[start_t: -1-horizon]  # participation rate at time t
            x1_raw = x1_raw / np.std(x1_raw)
            x1_lag = x1_raw.reshape(-1, 1)
            x1_lag2 = sm.add_constant(x1_lag)

            model = sm.OLS(y_predict, x1_lag2)
            est = model.fit()
            coeff_matrix2[i, j, k, 0] = est.params[1]
            pvalue_matrix2[i, j, k, 0] = est.pvalues[1]
            tstats_matrix2[i, j, k, 0] = est.tvalues[1]
            rsqrd_matrix2[i, j, k, 0] = est.rsquared

            # bivariate regressions
            x2_lag = path_x2[start_t: -1-horizon]  # average perceived risk premia at time t

            x_lag = np.append(x1_lag, x2_lag)
            x_lag = np.transpose(x_lag.reshape(2, -1))
            x_lag = sm.add_constant(x_lag)

            y_predict = y_raw.reshape(-1, 1)

            model = sm.OLS(y_predict, x_lag)
            est = model.fit()
            coeff_matrix2[i, j, k, 1] = est.params[1]
            coeff_matrix2[i, j, k, 2] = est.params[2]
            pvalue_matrix2[i, j, k, 1] = est.pvalues[1]
            pvalue_matrix2[i, j, k, 2] = est.pvalues[2]
            tstats_matrix2[i, j, k, 1] = est.tvalues[1]
            tstats_matrix2[i, j, k, 2] = est.tvalues[2]
            rsqrd_matrix2[i, j, k, 1] = est.rsquared

reg_coeffs2 = np.average(coeff_matrix2, axis=0)
reg_pvalues2 = np.average(pvalue_matrix2, axis=0)
reg_tstats2 = np.average(tstats_matrix2, axis=0)
reg_rsqrd2 = np.average(rsqrd_matrix2, axis=0)

for j in range(n):
    horizon = horizons[j]
    reg_data = np.empty((5, 8))
    header = ['(1) parti rate', '(2)', '(3) parti rate, young', '(4)', '(5) parti rate, old', '(6)', '(7) parti age',
              '(8)']
    index = ['coef', 't-stats', 'coef_x2', 't-stats_x2', 'R2']
    for i in range(4):
        reg_data[0, i * 2] = reg_coeffs2[j, i, 0]
        reg_data[1, i * 2] = reg_tstats2[j, i, 0]
        reg_data[2, i * 2] = np.nan
        reg_data[3, i * 2] = np.nan
        reg_data[4, i * 2] = reg_rsqrd2[j, i, 0]

        reg_data[0, i * 2 + 1] = reg_coeffs2[j, i, 1]
        reg_data[1, i * 2 + 1] = reg_tstats2[j, i, 1]
        reg_data[2, i * 2 + 1] = reg_coeffs2[j, i, 2]
        reg_data[3, i * 2 + 1] = reg_tstats2[j, i, 2]
        reg_data[4, i * 2 + 1] = reg_rsqrd2[j, i, 1]

    print(str(horizon) + '-month')
    print(tabulate.tabulate(reg_data, headers=header, showindex=index, floatfmt=".4f", tablefmt='fancy_grid'))




####
horizons = [1, 3, 6, 12, 36, 60, 120]
x_variables = [popu_parti_matrix, popu_parti_young_matrix, popu_parti_old_matrix, age_parti_matrix]
# x_variables = [popu_parti_matrix, popu_parti_young_matrix, popu_parti_old_matrix, age_parti_matrix]
m = len(x_variables)
n = len(horizons)
# np.cumsum(dR_matrix[i])
coeff_matrix2 = np.zeros((Mpaths, n, m, 3))
pvalue_matrix2 = np.zeros((Mpaths, n, m, 3))
tstats_matrix2 = np.zeros((Mpaths, n, m, 3))
rsqrd_matrix2 = np.zeros((Mpaths, n, m, 2))

for i in range(Mpaths):
    path_y = np.cumsum(dR_matrix[i])
    path_r = np.cumsum(r_matrix[i])
    path_x2 = survey_view_parti_matrix[i]
    path_x2 = path_x2 / np.std(path_x2)
    for j,horizon in enumerate(horizons):
        # y_raw = path_y[start_t + horizon + 1:] - path_y[start_t + 1: -horizon]  #dR is the return from t-1 to t, and thus have to move 1 to have returns from t to t+1
        y_raw = ((path_y[start_t + horizon + 1:] - path_y[start_t + 1: -horizon]) \
                - (path_r[start_t + horizon: -1] - path_r[start_t : -horizon - 1])) / (horizon * dt)
        y_predict = y_raw.reshape(-1, 1)
        for k, var in enumerate(x_variables):
            path_x = var[i]
            # univariate regressions
            x1_raw = path_x[start_t: -1-horizon]  # participation rate at time t
            x1_raw = x1_raw / np.std(x1_raw)
            x1_lag = x1_raw.reshape(-1, 1)
            x1_lag2 = sm.add_constant(x1_lag)

            model = sm.OLS(y_predict, x1_lag2)
            est = model.fit()
            coeff_matrix2[i, j, k, 0] = est.params[1]
            pvalue_matrix2[i, j, k, 0] = est.pvalues[1]
            tstats_matrix2[i, j, k, 0] = est.tvalues[1]
            rsqrd_matrix2[i, j, k, 0] = est.rsquared

            # bivariate regressions
            x2_lag = path_x2[start_t: -horizon - 1]

            x_lag = np.append(x1_lag, x2_lag)
            x_lag = np.transpose(x_lag.reshape(2, -1))
            x_lag = sm.add_constant(x_lag)

            y_predict = y_raw.reshape(-1, 1)

            model = sm.OLS(y_predict, x_lag)
            est = model.fit()
            coeff_matrix2[i, j, k, 1] = est.params[1]
            coeff_matrix2[i, j, k, 2] = est.params[2]
            pvalue_matrix2[i, j, k, 1] = est.pvalues[1]
            pvalue_matrix2[i, j, k, 2] = est.pvalues[2]
            tstats_matrix2[i, j, k, 1] = est.tvalues[1]
            tstats_matrix2[i, j, k, 2] = est.tvalues[2]
            rsqrd_matrix2[i, j, k, 1] = est.rsquared

reg_coeffs2 = np.average(coeff_matrix2, axis=0)
reg_pvalues2 = np.average(pvalue_matrix2, axis=0)
reg_tstats2 = np.average(tstats_matrix2, axis=0)
reg_rsqrd2 = np.average(rsqrd_matrix2, axis=0)

for j in range(n):
    horizon = horizons[j]
    reg_data = np.empty((5, 8))
    header = ['(1) parti rate', '(2)', '(3) parti rate, young', '(4)', '(5) parti rate, old', '(6)', '(7) parti age',
              '(8)']
    index = ['coef', 't-stats', 'coef_x2', 't-stats_x2', 'R2']
    for i in range(4):
        reg_data[0, i * 2] = reg_coeffs2[j, i, 0]
        reg_data[1, i * 2] = reg_tstats2[j, i, 0]
        reg_data[2, i * 2] = np.nan
        reg_data[3, i * 2] = np.nan
        reg_data[4, i * 2] = reg_rsqrd2[j, i, 0]

        reg_data[0, i * 2 + 1] = reg_coeffs2[j, i, 1]
        reg_data[1, i * 2 + 1] = reg_tstats2[j, i, 1]
        reg_data[2, i * 2 + 1] = reg_coeffs2[j, i, 2]
        reg_data[3, i * 2 + 1] = reg_tstats2[j, i, 2]
        reg_data[4, i * 2 + 1] = reg_rsqrd2[j, i, 1]

    print(str(horizon) + '-month')
    print(tabulate.tabulate(reg_data, headers=header, showindex=index, floatfmt=".4f", tablefmt='fancy_grid'))