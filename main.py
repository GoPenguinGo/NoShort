import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Callable, Tuple
from src.simulation import simulate_SI
from src.cohort_builder import build_cohorts_SI
from src.cohort_simulator import simulate_cohorts_SI
from src.param import rho, nu, mu_Y, sigma_Y, sigma_Y_sqr, sigma_S, v, tax, \
    beta, dt, T_hat, Npre, Vhat, Ninit, T_cohort, Nt, Nc, tau, cohort_size, \
    cutoffs, colors, modes_trade, modes_learn,\
    scenarios, dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    Z_Y_cases, Z_SI_cases
from src.stats import shocks, tau_calculator, good_times
from numba import jit
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tabulate as tabulate


# todo: to fill with patterns: https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_demo.html
# todo: remove variables that are never used, to save space
# different scenarios
N = 100  # for smaller number of paths

n_scenarios = 3
scenarios_short = scenarios[:n_scenarios]

phi_vector = np.arange(0,1,0.1)
n_phi = len(phi_vector)

age_cutoff = cutoffs[2]

theta_matrix = np.empty((N, n_scenarios, n_phi, Nt))
popu_parti_matrix = np.empty((N, n_scenarios, n_phi, Nt))
# market_view_matrix = np.empty((N, n_scenarios, n_phi, Nt))
survey_view_matrix = np.empty((N, n_scenarios, n_phi, Nt))
# Delta_matrix = np.empty((N, n_scenarios, n_phi, Nt, Nc))
# pi_matrix = np.empty((N, n_scenarios, n_phi, Nt, Nc))
# mu_st_rt_matrix = np.empty((N, n_scenarios, n_phi, Nt, Nc))
r_matrix = np.zeros((N, n_scenarios, n_phi, Nt))
belief_dispersion_matrix = np.zeros((N, n_scenarios, n_phi, Nt))
dR_matrix = np.zeros((N, n_scenarios, n_phi, Nt))
delta_bar_matrix = np.zeros((N, n_scenarios, n_phi, Nt))
Phi_parti_1_matrix = np.zeros((N, n_scenarios, n_phi, Nt))
parti_old_matrix = np.zeros((N, n_scenarios, n_phi, Nt))
parti_young_matrix = np.zeros((N, n_scenarios, n_phi, Nt))

# run the program for different values of phi, and store the results
for j in range(N):
    print(j)
    dZ = dZ_matrix[j]
    dZ_build = dZ_build_matrix[j]
    dZ_SI = dZ_SI_matrix[j]
    dZ_SI_build = dZ_SI_build_matrix[j]
    for k, scenario in enumerate(scenarios_short):
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        for l, phi in enumerate(phi_vector):
            (
                r,
                theta,
                f,
                Delta,
                pi,
                popu_parti,
                f_parti,
                Delta_bar_parti,
                dR,
                invest_tracker,
            ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta, phi,
                            Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                            top=0.05,
                            old_limit=100
                            )

            # Delta_matrix[j, k, l] = Delta
            # pi_matrix[j, k, l] = pi
            theta_matrix[j, k, l] = theta
            r_matrix[j, k, l] = r
            theta_mat = np.transpose(np.tile(theta, (Nc, 1)))
            # mu_st_rt_matrix[j, k, l] = theta_mat + Delta
            popu_parti_matrix[j, k, l] = popu_parti
            # market_view_matrix[j, k, l] = np.average(Delta, axis=1, weights=f)
            delta_bar_matrix[j, k, l] = Delta_bar_parti
            survey_view_matrix[j, k, l] = np.average(Delta, axis=1, weights=cohort_size)
            belief_dispersion_matrix[j, k, l] = np.std(Delta, axis=1)  # todo: maybe add weights
            dR_matrix[j, k, l] = dR
            # invest_tracker_matrix[j, k, l] = invest_tracker
            Phi_parti_1_matrix[j, k, l] = 1/f_parti
            parti_young_matrix[j, k, l] = np.average(invest_tracker[:, age_cutoff:], axis=1, weights = cohort_size[age_cutoff:])
            parti_old_matrix[j, k, l] = np.average(invest_tracker[:, :age_cutoff], axis=1, weights=cohort_size[:age_cutoff])
            # ( parti_young + parti_old )/2 = popu_parti

# ######################################
# ########## ONE RANDOM PATH ############
# ############ GRAPH ONE ###############
# ######################################

# ONE SPECIFIC PATH:
# generate data for the graphs:
phi_indexes = [0, 4, 8]
n_phi_short = len(phi_indexes)
phi_vector_short = phi_vector[phi_indexes]

dZ_build = dZ_build_matrix[0]
dZ_SI_build = dZ_SI_build_matrix[0]  # fix the shocks at the buildup stage

theta_compare = np.empty((n_scenarios, 2, 2, n_phi_short, Nt))
popu_parti_compare = np.empty((n_scenarios, 2, 2, n_phi_short, Nt))
market_view_compare = np.empty((n_scenarios, 2, 2, n_phi_short, Nt))
survey_view_compare = np.empty((n_scenarios, 2, 2, n_phi_short, Nt))
r_compare = np.zeros((n_scenarios, 2, 2, n_phi_short, Nt))
belief_dispersion_compare = np.zeros((n_scenarios, 2, 2, n_phi_short, Nt))
delta_bar_compare = np.zeros((n_scenarios, 2, 2, n_phi_short, Nt))
Phi_compare = np.zeros((n_scenarios, 2, 2, n_phi_short, Nt))
Delta_compare = np.empty((n_scenarios, 2, 2, n_phi_short, Nt, Nc))
pi_compare = np.empty((n_scenarios, 2, 2, n_phi_short, Nt, Nc))
cons_compare = np.zeros((n_scenarios, 2, 2, n_phi_short, Nt, Nc))
#cohort_size_mat = np.transpose(np.tile(cohort_size, (Nc, 1)))
cohort_size_mat = np.tile(cohort_size, (Nc, 1))

for g, scenario in enumerate(scenarios_short):
    mode_trade = scenario[0]
    mode_learn = scenario[1]
    for i in range(2):
        dZ = Z_Y_cases[i]
        log_Yt = np.cumsum((mu_Y - 0.5 * sigma_Y ** 2) * dt + sigma_Y * dZ)
        log_Yt_mat = np.transpose(np.tile(log_Yt, (Nc, 1)))
        for j in range(2):
            dZ_SI = Z_SI_cases[j]
            for k, phi in enumerate(phi_vector_short):
                (
                    r,
                    theta,
                    f,
                    Delta,
                    pi,
                    popu_parti,
                    Phi_parti,
                    Delta_bar_parti,
                    dR,
                    invest_tracker
                ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta,
                                phi,
                                Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                                top=0.05,
                                old_limit=100
                                )
                # invest_tracker = pi > 0
                Delta_compare[g, i, j, k] = Delta
                pi_compare[g, i, j, k] = pi
                theta_compare[g, i, j, k] = theta
                r_compare[g, i, j, k] = r
                theta_mat = np.transpose(np.tile(theta, (Nc, 1)))
                popu_parti_compare[g, i, j, k] = popu_parti
                market_view_compare[g, i, j, k] = np.average(Delta, axis=1, weights=f)
                delta_bar_compare[g, i, j, k] = Delta_bar_parti
                Phi_compare[g, i, j, k] = Phi_parti
                survey_view_compare[g, i, j, k] = np.average(Delta, axis=1, weights=cohort_size)
                belief_dispersion_compare[g, i, j, k] = np.std(Delta, axis=1)  # todo: maybe add weights
                cons_compare[g, i, j, k] = f / cohort_size_mat

# cohort_matrix_list = [pi_compare, Delta_compare, cons_compare]



red_cases = ['Good Z^Y ', 'Bad Z^Y ']
yellow_cases = ['Good Z^SI', 'Bad Z^SI']
nn = 3  # number of cohorts illustrated
t = np.arange(0, T_cohort, dt)
length = len(t)
starts = np.zeros(nn)
cohort_labels = ['cohort 1', 'cohort 2', 'cohort 3']
var_y_labels = ['Investment in stock market', 'Estimation error', 'Log consumption']
scenario_labels = ['Complete', 'Keep', 'Drop']
colors_short = ['midnightblue', 'darkgreen', 'darkviolet']
colors_short2 = ['mediumblue', 'saddlebrown', 'darkmagenta']
figure_labels = ['pi', 'Delta', 'Log cons']
label_phi = []
for i in range(n_phi_short):
    label_phi.append('phi = ' + str(phi_vector_short[i]))
labels = [scenario_labels, label_phi, label_phi]
lower = min(np.min(y1), np.min(y2))
upper = max(np.max(y1), np.max(y2))

# # plot the complicated figures...
# for i, cohort_matrix in enumerate(cohort_matrix_list):
#     for j, index_Z_Y in enumerate(index_Z_Ys):
#         Z = np.cumsum(dZ_matrix[index_Z_Y])
#         red_case = red_cases[j]
#         for k, index_Z_SI in enumerate(index_Z_SIs):
#             Z_SI = np.cumsum(dZ_SI_matrix[index_Z_SI])
#             yellow_case = yellow_cases[k]
#             for l, phi in enumerate(phi_vector):
#                 phi = phi_vector[l]
#                 y_interest = cohort_matrix[1, j, k, l]  # with short-sale constraint
#                 y_interest_time_series = np.zeros((nn, length))
#                 for m in range(nn):
#                     start = int((m + 1) * 100 * (1 / dt))
#                     starts[m] = start * dt
#                     for n in range(length):
#                         if n < start:
#                             y_interest_time_series[m, n] = np.nan
#                         else:
#                             cohort_rank = length - (n - start) - 1
#                             y_interest_time_series[m, n] = y_interest[n, cohort_rank]
#                 Delta_benchmarks = Delta_benchmark(post_var, sigma_Y, Nt, Vhat, phi, starts, dZ, dZ_SI, Npre, T_hat, dt)
#                 if i == 0:
#                     parti_time_series = (y_interest_time_series > 0)
#                     fig, ax1 = plt.subplots(figsize=(15, 5))
#                     ax1.set_xlabel('Time in simulation, one random path')
#                     ax1.set_ylabel('Zt', color=color5)
#                     ax1.plot(t, Z, color=color5, linewidth=0.5, label='Z^Y_t')
#                     ax1.plot(t, Z_SI, color=color6, linewidth=0.5, label='Z^SI_t')
#                     ax1.tick_params(axis='y', labelcolor=color5)
#                     ax2 = ax1.twinx()
#                     ax2.set_ylabel(var_y_labels[i], color=colors[0])
#                     ax2.set_ylim([0.01, 20])
#                     for m in range(nn):
#                         y_cohort = y_interest_time_series[m]
#                         plt.vlines(starts[m], ymax=20, ymin=0, color='grey', linestyle='--', linewidth=0.4)
#                         ax2.plot(t, y_cohort, label = cohort_labels[m], color=colors[m], linewidth=0.4)
#                     ax2.tick_params(axis='y', labelcolor=colors[0])
#                     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#                     plt.savefig(red_case + yellow_case + ' Zt and ' + figure_labels[i] + ' time series' + str(round(phi, 2)) + '.png', dpi=500)
#                     plt.show()
#                     plt.close()
#
#                 else:
#                     if i == 1:
#                         lower = -0.5
#                         upper = 0.5
#                     else:
#                         lower = 4
#                         upper = 15
#                     switch = abs(parti_time_series[:, 1:] ^ parti_time_series[:, :-1])
#                     col = np.reshape(switch[:, -1], (3, -1))
#                     switch = np.append(switch, col, axis=1)
#                     fig, ax1 = plt.subplots(figsize=(15, 5))
#                     ax1.set_xlabel('Time in simulation, one random path')
#                     ax1.set_ylabel('Zt', color=color5)
#                     ax1.plot(t, Z, color=color5, linewidth=0.5, label='Z^Y_t')
#                     ax1.plot(t, Z_SI, color=color6, linewidth=0.5, label='Z^SI_t')
#                     ax1.tick_params(axis='y', labelcolor=color5)
#                     ax2 = ax1.twinx()
#                     ax2.set_ylabel(var_y_labels[i], color=color2)
#                     ax2.set_ylim([lower, upper])
#                     for m in range(nn):
#                         # switch[m, starts[m]] = 1
#                         y_cohort = y_interest_time_series[m]
#                         y_cohort_N = np.ma.masked_where(parti_time_series[0] == 1, y_cohort)
#                         y_cohort_P = np.ma.masked_where(parti_time_series[0] == 0, y_cohort)
#                         y_cohort_switch = np.ma.masked_where(switch[0] == 0, y_cohort)
#                         plt.vlines(starts[m], ymax=upper, ymin=lower, color='grey', linestyle='--', linewidth=0.4)
#                         ax2.plot(t, y_cohort_P, label=cohort_labels[m], color=colors[m], linewidth=0.4)
#                         ax2.plot(t, y_cohort_N,  color=colors[m], linewidth=0.4, linestyle='dotted')
#                         if i == 1:
#                             ax2.fill_between(t, Delta_benchmarks[0], Delta_benchmarks[1], color=colors[m], alpha=0.4)
#                         if m == 0:
#                             ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o', label='state switch')
#                         else:
#                             ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o')
#                     ax2.tick_params(axis='y', labelcolor=color2)
#                     fig.tight_layout()  # otherwise the right y-label is slightly clipped
#                     plt.legend()
#                     plt.savefig(red_case + yellow_case + ' Zt and ' + figure_labels[i] + ' time series' + str(round(phi, 2)) + '.png', dpi=500)
#                     plt.show()
#                     plt.close()

# plot general + zoom-in figures:
# the general one:

pi_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
Delta_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
log_cons_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
switch_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
parti_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
for o in range(n_scenarios):
    for j, Z_Y_t in enumerate(Z_Y_cases):
        # Z = np.cumsum(dZ_matrix[index_Z_Y])
        # red_case = red_cases[j]
        for k, index_Z_SI in enumerate(Z_SI_cases):
            # Z_SI = np.cumsum(dZ_SI_matrix[index_Z_SI])
            # yellow_case = yellow_cases[k]

            for i in range(n_phi_short):
                Delta = Delta_compare[o, j, k, i]  # with short-sale constraint, keep mode
                pi = pi_compare[o, j, k, i]
                log_cons = cons_compare[o, j, k, i]
                for m in range(nn):
                    start = int((m + 1) * 100 * (1 / dt))
                    starts[m] = start * dt
                    for n in range(length):
                        if n < start:
                            Delta_time_series[o, j, k, i, m, n] = np.nan
                            pi_time_series[o, j, k, i, m, n] = np.nan
                            log_cons_time_series[o, j, k, i, m, n] = np.nan
                        else:
                            cohort_rank = length - (n - start) - 1
                            Delta_time_series[o, j, k, i, m, n] = Delta[n, cohort_rank]
                            pi_time_series[o, j, k, i, m, n] = pi[n, cohort_rank]
                            log_cons_time_series[o, j, k, i, m, n] = log_cons[n, cohort_rank]
                    parti = pi_time_series[o, j, k, i, m] > 0
                    switch = abs(parti[1:] ^ parti[:-1])
                    sw = np.append(switch, 0)
                    parti = np.where(sw == 1, 0.5, parti)
                    switch = np.insert(switch, 0, 0)
                    parti = np.where(switch == 1, 0.5, parti)
                    parti_time_series[o, j, k, i, m] = parti
                    switch_time_series[o, j, k, i, m] = sw



# ######################################
# ########### Figure 1 & 2 #############
# ######################################
# Delta (2 phi * 4 cases)
upper = 60  # todo: endogenize these parameters
lower = -60
scenario_index = 1
for i in range(1, n_phi_short, 1):
    phi = phi_vector_short[i]
    fig, axes = plt.subplots(nrows=4, ncols=1, sharey='all', sharex='all', figsize=(15, 20))
    for j, ax in enumerate(axes):
        red_index = 0 if j == 0 or j == 1 else 1
        yellow_index = 0 if j == 0 or j == 2 else 1
        red_case = red_cases[red_index]
        yellow_case = yellow_cases[yellow_index]
        Z = np.cumsum(dZ_matrix[index_Z_Ys[red_index]])
        Z_SI = np.cumsum(dZ_SI_matrix[index_Z_SIs[yellow_index]])
        if j == 3:
            ax.set_xlabel('Time in simulation, one random path')
        ax.set_ylabel('Zt', color=colors[4])
        ax.plot(t, Z, color=colors[8], linewidth=0.5, label='Z^Y_t')
        ax.plot(t, Z_SI, color=colors[6], linewidth=0.5, label='Z^SI_t')
        ax.set_ylim([lower, upper])
        ax.tick_params(axis='y', labelcolor=colors[4])
        if j == 0:
            ax.legend()
        ax.set_title(red_case + yellow_case)

        ax2 = ax.twinx()
        ax2.set_ylabel(var_y_labels[1], color=colors[4])
        ax2.set_ylim([-0.3, 0.7])
        for m in range(nn):
            # switch[m, starts[m]] = 1
            y_cohort = Delta_time_series[scenario_index, red_index, yellow_index, i, m]
            # y_cohort_N = np.ma.masked_where(parti_time_series[0] == 1, y_cohort)
            # y_cohort_P = np.ma.masked_where(parti_time_series[0] == 0, y_cohort)
            y_cohort_switch = np.ma.masked_where(switch_time_series[scenario_index, red_index, yellow_index, i, m] == 0, y_cohort)
            plt.vlines(starts[m], ymax=upper, ymin=lower, color='grey', linestyle='--', linewidth=0.4)
            ax2.plot(t, y_cohort, label=cohort_labels[m], color=colors_short[m], linewidth=0.4)
            # ax2.plot(t, y_cohort_P, label=cohort_labels[m], color=colors[m], linewidth=0.4)
            # ax2.plot(t, y_cohort_N, color=colors[m], linewidth=0.4, linestyle='dotted')
            # if i == 1:
            #     ax2.fill_between(t, Delta_benchmarks[0], Delta_benchmarks[1], color=colors[m], alpha=0.4)
            if m == 0:
                ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o', label='state switch')
            else:
                ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o')
        ax2.tick_params(axis='y', labelcolor=colors[4])
        if j == 0:
            ax2.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(
                'Shocks and Delta time series' + str(round(phi, 2)) + '.png',
                dpi=300)
    plt.show()
    # plt.close()


# ######################################
# ############# Figure 3 ###############
# ######################################
# the zoom-in ones
# bad & Bad, phi = 0.4
# cohort 1; cohort 2; cohort 1, complete; cohort 1, drop
year = 100
t_length = int(year/dt)
t_indexes = np.empty((2,2,2))
# t_indexes[0, 0, 0] = t_indexes[1, 0, 0] = t_indexes[1, 1, 0] = t_length
# t_indexes[0, 0, 1] = t_indexes[1, 0, 1] = t_indexes[1, 1, 1] = t_length * 2
# t_indexes[0, 1, 0] = t_length * 2
# t_indexes[0, 1, 1] = t_length * 3
# phi_where = [(1, 1), (1, 1)]
# cohort_indexes = [(0, 1), (0, 0)]
# scenario_indexs = [(1, 1), (0, 2)]
# titles_subfig = [('Cohort 1', 'Cohort 2'), ('Cohort 1, complete market', 'Cohort 1, drop')]
# y_interest = Delta_time_series[:,0,1]  # n_scenarios, n_phi_short, nn, length
# condition_matrix = parti_time_series[:,0,1]
# switch_matrix = switch_time_series[:,0,1]

red_case = 1
yellow_case = 0
t_indexes[0, 0, 0] = t_indexes[1, 0, 0] = t_indexes[1, 1, 0] = t_length * 3
t_indexes[0, 0, 1] = t_indexes[1, 0, 1] = t_indexes[1, 1, 1] = t_length * 4
t_indexes[0, 1, 0] = t_length
t_indexes[0, 1, 1] = t_length * 2
phi_where = [(1, 1), (1, 1)]
cohort_indexes = [(2, 0), (2, 2)]
scenario_indexs = [(1, 1), (0, 2)]
titles_subfig = [('Cohort 3', 'Cohort 1'), ('Cohort 3, complete market', 'Cohort 3, drop')]
y_interest = Delta_time_series[:, red_case, yellow_case]  # n_scenarios, n_phi_short, nn, length
condition_matrix = parti_time_series[:, red_case, yellow_case]
switch_matrix = switch_time_series[:, red_case, yellow_case]
theta = theta_compare[:, red_case, yellow_case]  # n_scenarios, 2, 2, n_phi_short, Nt
fig, axes = plt.subplots(nrows=2, ncols=2, sharey='all', figsize=(10, 10))
# fig.suptitle('Good Z^Y, Bad Z^SI')
for i, ax_row in enumerate(axes):
    for j, ax in enumerate(ax_row):
        left = int(t_indexes[i, j, 0])
        right = int(t_indexes[i, j, 1])
        x = t[left: right]
        scenario_index = scenario_indexs[i][j]
        phi_index = phi_where[i][j]
        cohort_index = cohort_indexes[i][j]
        y = y_interest[scenario_index, phi_index, cohort_index, left:right]
        condition = condition_matrix[scenario_index, phi_index, cohort_index, left:right]
        switch = switch_matrix[scenario_index, phi_index, cohort_index, left:right]
        cutoff_belief = -theta[scenario_index, phi_index, left:right]
        y_N = np.ma.masked_where(condition >= 0.8, y)
        y_P = np.ma.masked_where(condition <= 0.2, y)
        # y_N = np.ma.masked_where(condition >= 0.05, y)
        # y_P = np.ma.masked_where(condition <= 0.0001, y)
        y_switch = np.ma.masked_where(switch == 0, y)
        # ax.plot(x, y, label=cohort_labels[cohort_index], color='black', linewidth=0.4)
        if i == 1 and j == 0:
            ax.plot(x, y, color='black', linewidth=0.6)
            #ax.plot(x, cutoff_belief, label='cutoff Delta', color='pink', alpha=0.8, linewidth=0.4)
        else:
            ax.plot(x, cutoff_belief, label='cutoff Delta', color='blue', alpha=0.6, linewidth=0.4)
            ax.plot(x, y_P, label='P', color='black', linewidth=0.6)
            ax.plot(x, y_N, label='N', color='black', linewidth=0.6, linestyle='dotted')
            # ax2.fill_between(t, Delta_benchmarks[0], Delta_benchmarks[1], color=colors[m], alpha=0.4)
            ax.scatter(x, y_switch, color='red', s=10, marker='o', label='state switch')
        if i == j == 0:
            ax.legend()
        if j == 0:
            ax.set_ylabel('Standardized estimation error')
        if i == 1:
            ax.set_xlabel('Time in simulation')
        ax.set_title(titles_subfig[i][j])
        ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout(h_pad=2)
#plt.savefig('Shocks and Delta, zoom in' + str(red_case) + str(yellow_case) +'.png', dpi=500)
plt.show()
#plt.close()



# ######################################
# ############# Figure 4.1 #############
# ######################################
# time-series of interest rate and market price of risk, bad z^Y, bad z&SI
# 4.1.1 interest rate across different phi values, keep
# 4.1.2 interest rate across different scenarios, phi = 0
# 4.1.3 market price of risk across different phi values, keep
red_case = 1
yellow_case = 1
r_mat = r_compare[:, red_case, yellow_case]  # n_scenarios, n_phi_short, Nt
theta_mat = theta_compare[:, red_case, yellow_case]
y1 = r_mat[:,0]  # n_phi_short, Nt
y2 = r_mat[1]  # n_scenarios, Nt
y3 = theta_mat[1]  # n_phi_short, Nt
y_list = [y1, y2, y3]
Z = np.cumsum(Z_Y_cases[red_case])
Z_SI = np.cumsum(Z_SI_cases[yellow_case])
n_lines = [n_scenarios, n_phi_short, n_phi_short]
y_title_list = ['Interest rate', 'Interest rate', 'Market price of risk']

fig, axes = plt.subplots(nrows=3, ncols=1, sharex='all', figsize=(15, 15))
for j, ax in enumerate(axes):
    ax.set_ylabel('Zt', color=colors[4])
    ax.plot(t, Z, color=colors[8], linewidth=0.5, label='Z^Y_t')
    ax.plot(t, Z_SI, color=colors[6], linewidth=0.5, label='Z^SI_t')
    ax.tick_params(axis='y', labelcolor=colors[4])
    if j == 2:
        ax.set_xlabel('Time in simulation, one random path')

    y_vec = y_list[j]  # n_phi_short, Nt
    y_title = y_title_list[j]
    ax2 = ax.twinx()
    ax2.set_ylabel(y_title, color=colors[4])
    for i in range(n_lines[j]):
        y = y_vec[i]  # Nt
        color_i = colors_short2[i] if j == 0 else colors_short[i]
        ax2.plot(t, y, label=labels[j][i], color=color_i, linewidth=0.4)
    if i<2:
        ax2.set_ylim(lower, upper)
    if j==0:
        ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.set_title(y_title)
fig.tight_layout(h_pad=2)
plt.savefig('r and theta,' + str(red_case) + str(yellow_case)  +'.png', dpi=500)
plt.show()
plt.close()




# ######################################
# ############# Figure 4.2 #############
# ######################################
# time-series of Delta_bar, Phi, and participation rate, bad z^Y, bad z&SI
# 4.2.1 Delta_bar (keep + complete + drop) * (phi = 0, 0.8)
# 4.2.2 Phi (keep + drop) * (phi = 0, 0.8), also mark 1 for complete case
# 4.2.3 participation rate (keep + drop) * (phi = 0, 0.8)
red_case = 1
yellow_case = 1
titles_subfig = ['bar_Delta', 'Phi', 'Participation rate']
phi_indexes = [0, 2]
y1_case = delta_bar_compare[:, red_case, yellow_case]
y2_case = Phi_compare[:, red_case, yellow_case]
y3_case = popu_parti_compare[:, red_case, yellow_case]
left_t = 300
right_t = 400
Z = np.cumsum(Z_Y_cases[red_case])[int(left_t/dt):int(right_t/dt)]
Z_SI = np.cumsum(Z_SI_cases[yellow_case])[int(left_t/dt):int(right_t/dt)]
x = t[int(left_t/dt):int(right_t/dt)]

fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', figsize=(15, 15))
# for j, ax_row in enumerate(axes):
#     for k, ax in enumerate(ax_row):
#         ax.set_ylabel('Zt', color='black')
#         ax.plot(x, Z, color='black', linewidth=0.5, linestyle = 'solid', label='Z^Y_t')
#         ax.plot(x, Z_SI, color='gray', linewidth=0.5, linestyle = 'solid', label='Z^SI_t')
#         ax.tick_params(axis='y', labelcolor='black')
#         # ax.set_xlim(left_t, right_t)
#         if j == 2:
#             ax.set_xlabel('Time in simulation, one random path')
#         if j==0:
#             ax.legend(loc='upper left')
axes[0,0].set_title('The shocks')
axes[0,0].set_ylabel('Zt', color='black')
axes[0,0].plot(x, Z, color='black', linewidth=0.5, linestyle = 'solid', label='Z^Y_t')
axes[0,0].plot(x, Z_SI, color='gray', linewidth=0.5, linestyle = 'solid', label='Z^SI_t')
axes[0,0].tick_params(axis='y', labelcolor='black')
axes[0,0].legend(loc='upper left')

axes[0,1].set_title(titles_subfig[0])
axes[0,1].set_ylabel(titles_subfig[0], color='black')
# ax1 = axes[0].twinx()
for i in range(3):
    for j in range(2):
        y1 = y1_case[i, phi_indexes[j], int(left_t/dt):int(right_t/dt)]
        line_style = 'dotted' if j==0 else 'solid'
        label_i = scenario_labels[i]
        if j == 1:
            axes[0,1].plot(x, y1, label = label_i, linewidth=0.5, color=colors_short[i], linestyle=line_style)
        else:
            axes[0,1].plot(x, y1, linewidth=0.5, color=colors_short[i], linestyle=line_style)
axes[0,1].legend(loc='upper right')

axes[1, 0].set_title(titles_subfig[1])
axes[1, 0].set_ylabel(titles_subfig[1], color='black')
# ax2 = axes[1].twinx()
for i in range(1,3):
    for j in range(2):
        y2 = y2_case[i, phi_indexes[j], int(left_t/dt):int(right_t/dt)]
        line_style = 'dotted' if j==0 else 'solid'
        label_i = scenario_labels[i]
        if j == 1:
            axes[1, 0].plot(x, y2, label = label_i, linewidth=0.5, color=colors_short[i], linestyle=line_style)
        else:
            axes[1, 0].plot(x, y2, linewidth=0.5, color=colors_short[i], linestyle=line_style)
y21 = y2_case[0, 0,  int(left_t/dt):int(right_t/dt)]
# axes[1, 0].plot(x, y21, label = scenario_labels[0], linewidth=0.5, color=colors_short[0], linestyle='solid')
axes[1, 0].legend(loc='upper right')
axes[1,0].set_xlabel('Time in simulation, one random path')

axes[1,1].set_title(titles_subfig[2])
axes[1,1].set_ylabel(titles_subfig[2], color='black')
# ax3 = axes[2].twinx()
for i in range(1,3):
    for j in range(2):
        y3 = y3_case[i, phi_indexes[j], int(left_t/dt):int(right_t/dt)]
        line_style = 'dotted' if j==0 else 'solid'
        label_i = scenario_labels[i]
        if j == 1:
            axes[1,1].plot(x, y3, label=label_i, linewidth=0.5, color=colors_short[i], linestyle=line_style)
        else:
            axes[1,1].plot(x, y3, linewidth=0.5, color=colors_short[i], linestyle=line_style)
# y21 = y2_case[0, 0]
# ax3.plot(t, y1, label = label_scenarios[0], color=colors_short[0], linestyle='dashed')
axes[1,1].legend(loc='upper right')
axes[1,1].set_xlabel('Time in simulation, one random path')
fig.tight_layout(h_pad=2)
plt.savefig('Delta bar, Phi and parti rate,' + str(red_case) + str(yellow_case)  + '.png', dpi=500)
plt.show()
#plt.close()



# ######################################
# ############ Figure 4-3 ##############
# ######################################
# portfolio + consumption in time series
# bad & Bad, phi = 0.4
# portfolio, different phi
# portfolio, different scenarios
left_t = 300
right_t = 400
red_case = 1
yellow_case = 1
cohort_index = 2
phi_index = 1
scenario_index = 1
Z = np.cumsum(Z_Y_cases[red_case])[int(left_t/dt):int(right_t/dt)]
Z_SI = np.cumsum(Z_SI_cases[yellow_case])[int(left_t/dt):int(right_t/dt)]
x = t[int(left_t/dt):int(right_t/dt)]
y1_case = pi_time_series[:, red_case, yellow_case, phi_index, cohort_index]  # (n_scenarios, 2, 2, n_phi_short, nn, length) -> (n_scenarios, length)
y2_case = pi_time_series[scenario_index, red_case, yellow_case, :, cohort_index]  # (n_scenarios, 2, 2, n_phi_short, nn, length) -> (n_phi_short, length)
Delta_vec = Delta_time_series[:, red_case, yellow_case, phi_index, cohort_index]  # (n_scenarios, 2, 2, n_phi_short, nn, length) -> (n_scenarios, length)
Delta_bar_vec = delta_bar_compare[:, red_case, yellow_case, phi_index] # (n_scenarios, 2, 2, n_phi_short, Nt) -> (n_scenarios, Nt)
y3_case = (Delta_vec - Delta_bar_vec) / sigma_Y
y4_case = 1 / Phi_compare[:, red_case, yellow_case, phi_index]
y_cases = [y1_case, y2_case, y3_case, y4_case]
titles_subfig = ['Constraints', 'phi values', 'Belief element', 'Phi element']
fig, axes = plt.subplots(nrows=2, ncols=2, sharey='all', figsize=(15, 15))
for i, ax in enumerate(axes.flat):
    n_loop = n_phi_short if i == 0 else n_scenarios
    for j in range(n_loop):
        y_case = y_cases[i][j, int(left_t/dt):int(right_t/dt)]
        labels = label_phi if i == 1 else scenario_labels
        label_i = labels[j]
        ax.plot(x, y_case, label=label_i, color=colors_short[j], linewidth=0.4)
    ax.legend()
    ax.set_ylim(-8, 8)
    if i == 0 or i == 2:
        ax.set_ylabel('Investment in stock market')
    if i == 2 or i == 3:
        ax.set_xlabel('Time in simulation')
    ax.set_title(titles_subfig[i])
    ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout(h_pad=2)
plt.savefig('Shocks and Portfolio,' + str(red_case) + str(yellow_case)  + '.png', dpi=500)
plt.show()
#plt.close()

# ######################################
# ############ Figure 4-4 ##############
# ######################################
# consumption, different phi
# consumption, different scenarios
# build consumption data and time series over tau values
tax_vector = [0.005, 0.01, 0.015]
n_tax = len(tax_vector)
cons_compare_tau = np.zeros((n_scenarios, 2, 2, n_tax, n_phi_short, Nt, Nc))
pi_compare_tau = np.zeros((n_scenarios, 2, 2, n_tax, n_phi_short, Nt, Nc))
for g, scenario in enumerate(scenarios_short):
    mode_trade = scenario[0]
    mode_learn = scenario[1]
    for i in range(2):
        dZ = Z_Y_cases[i]
        log_Yt = np.cumsum((mu_Y - 0.5 * sigma_Y ** 2) * dt + sigma_Y * dZ)
        log_Yt_mat = np.transpose(np.tile(log_Yt, (Nc, 1)))
        for j in range(2):
            dZ_SI = Z_SI_cases[j]
            for k, tax_try in enumerate(tax_vector):
                beta_try = rho + nu - tax_try
                for l, phi_try in enumerate(phi_vector_short):
                    (
                        r,
                        theta,
                        f,
                        Delta,
                        pi,
                        popu_parti,
                        Phi_parti,
                        Delta_bar_parti,
                        dR,
                        invest_tracker
                    ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S,
                                    tax_try,
                                    beta_try,
                                    phi_try,
                                    Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                                    top=0.05,
                                    old_limit=100
                                    )
                    # invest_tracker = pi > 0
                    cons_compare_tau[g, i, j, k, l] = f * dt / cohort_size_mat
                    pi_compare_tau[g, i, j, k, l] = pi
pi_time_series_tau = np.zeros((n_scenarios, 2, 2, n_tax, n_phi_short, nn, length))
cons_share_time_series_tau = np.zeros((n_scenarios, 2, 2, n_tax, n_phi_short, nn, length))
switch_time_series_tau = np.zeros((n_scenarios, 2, 2, n_tax, n_phi_short, nn, length))
parti_time_series_tau = np.zeros((n_scenarios, 2, 2, n_tax, n_phi_short, nn, length))
for o in range(n_scenarios):
    for j in range(2):
        for k in range(2):
            for i in range(n_tax):
                for l in range(n_phi_short):
                    pi = pi_compare_tau[o, j, k, i, l]
                    cons = cons_compare_tau[o, j, k, i, l]
                    for m in range(nn):
                        start = int((m + 1) * 100 * (1 / dt))
                        starts[m] = start * dt
                        for n in range(length):
                            if n < start:
                                pi_time_series_tau[o, j, k, i, l, m, n] = np.nan
                                cons_share_time_series_tau[o, j, k, i, l, m, n] = np.nan
                            else:
                                cohort_rank = length - (n - start) - 1
                                pi_time_series_tau[o, j, k, i, l, m, n] = pi[n, cohort_rank]
                                cons_share_time_series_tau[o, j, k, i, l, m, n] = cons[n, cohort_rank]
                        parti = pi_time_series_tau[o, j, k, i, l, m] > 0
                        switch = abs(parti[1:] ^ parti[:-1])
                        sw = np.append(switch, 0)
                        parti = np.where(sw == 1, 0.5, parti)
                        switch = np.insert(switch, 0, 0)
                        parti = np.where(switch == 1, 0.5, parti)
                        parti_time_series_tau[o, j, k, i, l, m] = parti
                        switch_time_series_tau[o, j, k, i, l, m] = sw


right_t = 500
# phi_indexes = [1, 2, 1, 1]
phi_indexes = [0, 2, 0, 0]
scenario_indexs = [1, 1, 0, 2]
line_styles = [(0, (5, 10)), 'solid', (0, (1, 1))]
titles_subfig = ['Keep', 'Keep, phi = 0.8', 'Complete', 'Drop']
cases = ['Good ', 'Bad ']
for cohort_index in range(3):
    left_t = starts[cohort_index]
    for red_index in range(2):
        red_case = cases[red_index] + 'z^Y '
        Z = np.cumsum(Z_Y_cases[red_index])[int(left_t / dt):int(right_t / dt)]
        for yellow_index in range(2):
            yellow_case = cases[yellow_index] + 'z^SI '
            Z_SI = np.cumsum(Z_SI_cases[yellow_index])[int(left_t / dt):int(right_t / dt)]
            x = t[int(left_t / dt):int(right_t / dt)]
            fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all', figsize=(15, 15))
            for i, ax in enumerate(axes.flat):
                phi_index = phi_indexes[i]
                scenario_index = scenario_indexs[i]
                ax.set_title(titles_subfig[i])
                if i == 0 or i == 2:
                    ax.set_ylabel('Share of consumption', color='black')
                if i == 2 or i == 3:
                    ax.set_xlabel('Time in simulation, one random path')
                for k in range(n_tax):
                    y = cons_share_time_series_tau[scenario_index, red_index, yellow_index, k, phi_index, cohort_index,
                        int(left_t / dt):int(right_t / dt)]  # n_scenarios, 2, 2, n_tax, n_phi_short, nn, length
                    # condition = pi_time_series_tau[scenario_index, red_index, yellow_index, k, 2, phi_index, int(left_t/dt):int(right_t/dt)]
                    switch = switch_time_series_tau[scenario_index, red_index, yellow_index, k, phi_index, cohort_index,
                             int(left_t / dt):int(right_t / dt)]
                    y_switch = np.ma.masked_where(switch == 0, y)
                    # y_N = np.ma.masked_where(condition >= 0.8, y)
                    # y_P = np.ma.masked_where(condition <= 0.2, y)
                    label_i = 'tau = ' + str(tax_vector[k])
                    if k == 1:
                        ax.plot(x, y, color='black', linewidth=0.6, linestyle=line_styles[k], label=label_i)
                    else:
                        ax.plot(x, y, color='black', linewidth=0.4, linestyle=line_styles[k], label=label_i)
                    if scenario_index != 0:
                        if k == 0:
                            ax.scatter(x, y_switch, color='red', s=10, marker='o', label='state switch')
                        else:
                            ax.scatter(x, y_switch, color='red', s=10, marker='o')
                    # else:
                    #     ax.plot(x, y_P, label='P', color='black', linewidth=0.4)
                    #     ax.plot(x, y_N, label='N', color='black', linewidth=0.4, linestyle='dotted')
                    #     # ax2.fill_between(t, Delta_benchmarks[0], Delta_benchmarks[1], color=colors[m], alpha=0.4)
                ax.tick_params(axis='both', labelcolor='black')
                # ax.set_xlim(left_t, right_t)
                if i == 0:
                    ax.legend()
            fig.tight_layout(h_pad=2)
            plt.savefig('Individual consumption share,' + str(red_case) + str(yellow_case) + 'cohort' + str(cohort_index) + 'phi=' + str(phi_vector_short[phi_indexes[0]]) + '.png',
                        dpi=500)
            plt.show()
            plt.close()

#
# # ######################################
# # ############ GRAPH TWO ###############
# # ######################################
# market_matrix_list = [theta_compare, r_compare, popu_parti_compare, survey_view_compare, market_view_compare, delta_bar_compare, belief_dispersion_compare]
# y_title_list = ['Market price of risk', 'Interest rate', 'Participation rate', 'Survey view', 'Market view', 'Market view_parti', 'Belief dispersion']
#
# # Graphs for the short-sale constraint case, comparing different phi values
# for i, market_matrix in enumerate(market_matrix_list):
#     y_title = y_title_list[i]
#     for j, index_Z_Y in enumerate(index_Z_Ys):
#         Z = np.cumsum(dZ_matrix[index_Z_Y])
#         red_case = red_cases[j]
#         for k, index_Z_SI in enumerate(index_Z_SIs):
#             Z_SI = np.cumsum(dZ_SI_matrix[index_Z_SI])
#             yellow_case = yellow_cases[k]
#             y = market_matrix[1, j, k]
#
#             fig, ax1 = plt.subplots(figsize=(15, 5))
#             ax1.set_xlabel('Time in simulation, one random path')
#             ax1.set_ylabel('Zt', color=color5)
#             ax1.plot(t, Z, color=color5, linewidth=0.5, label='Z^Y_t')
#             ax1.plot(t, Z_SI, color=color6, linewidth=0.5, label='Z^SI_t')
#             ax1.tick_params(axis='y', labelcolor=color5)
#             ax2 = ax1.twinx()
#             ax2.set_ylabel(y_title, color=color2)
#             lower = np.nanmin(market_matrix)
#             upper = np.nanmax(market_matrix)
#             ax2.set_ylim([lower, upper])
#
#             for i in range(n_phi):
#                 label_i = 'phi = ' + str(round(phi_vector[i], 2))
#                 yy = y[i]
#                 ax2.plot(t, yy, color=colors[i], linewidth=0.4, label=label_i)
#             ax2.tick_params(axis='y', labelcolor=color2)
#             plt.legend()
#             # fig.suptitle('Zt and Market Price of Risk')
#             fig.tight_layout()  # otherwise the right y-label is slightly clipped
#             plt.savefig(red_case + yellow_case + ' Zt and ' + y_title + ' different phi'+ '.png', dpi=500)
#             plt.show()
#             plt.close()
#
# # ######################################
# # ########### GRAPH THREE ##############
# # ######################################
# # Comparing with the complete market case:
# y_title_list = ['Market price of risk', 'Interest rate', 'Survey view', 'Market view']
# market_matrix_list = [theta_compare, r_compare, survey_view_compare, market_view_compare]
# label_modes = ['complete market', 'short-sale constraint']
# for i, market_matrix in enumerate(market_matrix_list):
#     y_title = y_title_list[i]
#     for j, index_Z_Y in enumerate(index_Z_Ys):
#         Z = np.cumsum(dZ_matrix[index_Z_Y])
#         red_case = red_cases[j]
#         for k, index_Z_SI in enumerate(index_Z_SIs):
#             Z_SI = np.cumsum(dZ_SI_matrix[index_Z_SI])
#             yellow_case = yellow_cases[k]
#             for l, phi in enumerate(phi_vector):
#                 y = market_matrix[:, j, k, l]
#                 fig, ax1 = plt.subplots(figsize=(15, 5))
#                 ax1.set_xlabel('Time in simulation, one random path')
#                 ax1.set_ylabel('Zt', color=color5)
#                 ax1.plot(t, Z, color=color5, linewidth=0.5, label='Z^Y_t')
#                 ax1.plot(t, Z_SI, color=color6, linewidth=0.5, label='Z^SI_t')
#                 ax1.tick_params(axis='y', labelcolor=color5)
#                 ax2 = ax1.twinx()
#                 ax2.set_ylabel(y_title, color=color2)
#                 lower = np.nanmin(market_matrix)
#                 upper = np.nanmax(market_matrix)
#                 ax2.set_ylim([lower, upper])
#                 for i in range(2):
#                     yy = y[i]
#                     ax2.plot(t, yy, color=colors[i], linewidth=0.4, label=label_modes[i])
#                 ax2.tick_params(axis='y', labelcolor=color2)
#                 plt.legend()
#                 # fig.suptitle('Zt and Market Price of Risk')
#                 fig.tight_layout()  # otherwise the right y-label is slightly clipped
#                 plt.savefig(red_case + yellow_case + 'vs complete' +' Zt and ' + y_title + str(round(phi, 2)) + '.png', dpi=500)
#                 plt.show()
#                 plt.close()
#



# ######################################
# ########### ACROSS PATHS #############
# ############ GRAPH ONE ###############
# ######################################

# N, n_scenarios, n_phi, Nt
var_list = [r_matrix, theta_matrix, delta_bar_matrix, Phi_parti_1_matrix, popu_parti_matrix, belief_dispersion_matrix]
var_name_list = ['interest rate', 'market price of risk',
                 'consumption-weighted estimation error of participants', 'consumption share of participants',
                 'participation rate', 'belief dispersion']
type_list = ['mean', 'vola', 'scaled vola']
age_labels = ['20 < Age <= 35, youngest quartile', '35 < Age <= 55', '55 < Age <= 89', 'Age > 89, oldest quartile']

x = phi_vector

for i, variable in enumerate(var_list):
    var_name = var_name_list[i]
    y_mean_mat = np.mean(variable, axis=3)
    y_std_mat = np.std(variable, axis=3)  # shape = N * n_scenarios * n_phi
    y_mean = np.mean(y_mean_mat, axis=0)
    y_std = np.mean(y_std_mat, axis=0)  # shape = n_scenarios * n_phi
    #y_std_mean = y_std / abs(y_mean)
    if i<= 4:
        # y_list = [y_mean, y_std, y_std_mean]
        y_list = [y_mean, y_std]
    else:
        y_list = [y_mean]
    for j, y in enumerate(y_list):
        type = type_list[j]
        fig, ax = plt.subplots()  # consider include Vhat in the graphs on the LHS axis
        for k, scenario in enumerate(scenarios_short):
            mode_trade = scenario[0]
            mode_learn = scenario[1]
            y_vector = y[k]  # shape = n_phi
            label_i = mode_trade if mode_trade == 'complete' else mode_trade + mode_learn
            ax.plot(x, y_vector, label=label_i)
        ax.set_ylabel(var_name)
        ax.set_xlabel('phi')
        ax.legend()
        # plt.savefig(type + ' compare ' + var_name + '.png', dpi=500, format="png")
        plt.show()
        # plt.close()


# decomposition of variance for theta:
theta_vola = np.mean(np.var(theta_matrix, axis=3), axis=0)   # shape = n_scenarios * n_phi
delta_bar_vola = np.mean(np.var((-delta_bar_matrix), axis=3), axis=0)
Phi_parti_1_vola = np.mean(np.var(Phi_parti_1_matrix * sigma_Y, axis=3), axis=0)
cov_Phiparti_deltabar_matrix = np.empty((N, n_scenarios, n_phi))
for i in range(N):
    for j in range(n_scenarios):
        for k in range(n_phi):
            delta_bar = delta_bar_matrix[i, j, k]
            Phi_parti1 = Phi_parti_1_matrix[i, j, k]
            cova = np.cov(-delta_bar, Phi_parti1 * sigma_Y)
            cov_Phiparti_deltabar_matrix[i, j, k] = cova[0,1]
cov_Phiparti_deltabar = np.mean(cov_Phiparti_deltabar_matrix, axis=0)
line_styles = [(0, (5, 10)), 'solid', (0, (1, 1))]
titles_subfig = ['market price of risk', 'consumption-weighted estimation error of participants', 'consumption share of participants * sigmaY', 'covariance']
y_list = [theta_vola, delta_bar_vola, Phi_parti_1_vola, cov_Phiparti_deltabar]
fig, axs = plt.subplots(nrows=2, ncols=2, sharey='all', sharex='all', figsize=(10, 10))
for j, ax in enumerate(axs.flat):
    y = y_list[j]
    ax.set_title(titles_subfig[j])
    for k in range(n_scenarios):
        y_sce = y[k]
        label_i = scenario_labels[k]
        ax.plot(phi_vector, y_sce, color='black', linewidth=0.7, label=label_i, linestyle=line_styles[k])
    if j == 0:
        ax.legend()
    if j == 0 or j == 2:
        ax.set_ylabel('Variance')
    if j == 2 or j == 3:
        ax.set_xlabel('phi values')
fig.tight_layout()
# plt.suptitle('Variance decomposition, market price of risk', fontsize=16)
# fig.supxlabel('phi')
# fig.supylabel('variance')
plt.savefig('Variance decomposition, market price of risk.png', dpi=500, format="png")
plt.show()
# plt.close()


# ######################################
# ############ GRAPH TWO ###############
# ######################################
# regressions

horizons = [3, 6, 12, 24]
n_horizon = len(horizons)
report = ['coef', 't-stats', 'R-sqrd']
n_report = len(report)
# var_names = ['Participation rate', 'Belief dispersion', 'survey view', 'Participation rate, young', 'Participation rate, old']
var_names = ['Participation rate', 'Belief dispersion', 'survey view']
phi_indeces = [0, 4, 8]
n_phi_short = len(phi_indeces)
regression_results_uni = np.empty((N, n_scenarios, n_phi_short, n_horizon, len(var_names), n_report))
# var_list = [popu_parti_matrix, belief_dispersion_matrix, survey_view_matrix, parti_young_matrix, parti_old_matrix]
var_list = [popu_parti_matrix, belief_dispersion_matrix, survey_view_matrix]

# predictive regression of stock returns on pariticipation rate
for i in range(N):
    for j in range(n_scenarios):
        for k, phi_index in enumerate(phi_indeces):
            excess_return_vector = np.cumsum(dR_matrix[i, j, phi_index, 1:] - r_matrix[i, j, phi_index, :-1] * dt)
            x_list = []
            for var in var_list:
                var_vector = var[i, j, phi_index, :-1]
                x_list.append(var_vector)
            for l, horizon in enumerate(horizons):
                y_horizon1 = (excess_return_vector[horizon:] - excess_return_vector[: -horizon]) / (
                            dt * horizon)  # make sure the timings allign
                y_horizon = y_horizon1.reshape(-1, 1)
                for m, x_horizon_raw in enumerate(x_list):
                    if scenarios[j][0] == 'complete' and var == popu_parti_matrix:
                        regression_results_uni[i, j, k, l, m, 0] = est.params[1]
                        regression_results_uni[i, j, k, l, m, 1] = est.tvalues[1]
                        regression_results_uni[i, j, k, l, m, 2] = est.rsquared
                    x_horizon1 = x_horizon_raw[: -horizon]
                    x_horizon1 = x_horizon1 / np.std(x_horizon1)
                    x_horizon1 = x_horizon1.reshape(-1, 1)
                    x_horizon = sm.add_constant(x_horizon1)

                    model = sm.OLS(y_horizon, x_horizon)
                    est = model.fit()
                    regression_results_uni[i, j, k, l, m, 0] = est.params[1]
                    regression_results_uni[i, j, k, l, m, 1] = est.tvalues[1]
                    regression_results_uni[i, j, k, l, m, 2] = est.rsquared


mean_regression_results = np.mean(regression_results_uni, axis=0)
header = ['(1) phi = 0', '(2) phi = 0.4', '(3) phi = 0.8']
# present the regression results in tables:

for k, scenario in enumerate(scenarios):
    label_scenario = scenario[0] if scenario[0] == 'complete' else scenario[0] + scenario[1]
    print(label_scenario)
    for j, var in enumerate(var_names):
        for i, horizon in enumerate(horizons):
            reg_data = np.empty((n_report, n_phi))
            for l in range(n_phi):
                reg_data[:, l] = mean_regression_results[k, l, i, j]
            report1 = [var, 't-stats', 'R-sqrd']
            print(var, ', ' + str(horizon) + ' months')
            print(tabulate.tabulate(reg_data, headers=header, showindex=report1, floatfmt=".4f", tablefmt='fancy_grid'))




# ######################################
# ############ GRAPH THREE #############
# ######################################

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