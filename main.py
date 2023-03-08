import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from typing import Callable, Tuple
from src.simulation import simulate_SI
from src.cohort_builder import build_cohorts_SI
from src.cohort_simulator import simulate_cohorts_SI
from src.param import rho, nu, mu_Y, sigma_Y, sigma_Y_sqr, sigma_S, v, tax, \
    beta, dt, T_hat, Npre, Vhat, Ninit, T_cohort, Nt, Nc, tau, cohort_size, \
    n_age_groups, cutoffs, colors, modes_trade, modes_learn,\
    scenarios, dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    dZ_Y_cases, dZ_SI_cases, dZ_build_case, dZ_SI_build_case, t, red_labels, yellow_labels, cohort_labels, \
    scenario_labels, colors_short , colors_short2, PN_labels, age_labels
from src.stats import shocks, tau_calculator, good_times, Delta_st_compare, weighted_variance
from numba import jit
import matplotlib.pyplot as plt
import statsmodels.api as sm
# import tabulate as tabulate
from scipy.interpolate import make_interp_spline


# todo: remove variables that are never used, to save space
# todo: organize the code
# different scenarios
N = 100  # for smaller number of paths

n_scenarios = 5
scenarios_short = scenarios[:n_scenarios]

# n_scenarios = 1
# scenarios_short = scenarios[1:2]

phi_vector = np.arange(0,1,0.1)
n_phi = len(phi_vector)

phi_indexes = [0, 4, 8]
n_phi_short = len(phi_indexes)
phi_vector_short = phi_vector[phi_indexes]

age_cutoff = cutoffs[2]



###############################################
###############################################



# theta_matrix = np.empty((N, n_scenarios, n_phi, Nt))
# popu_parti_matrix = np.empty((N, n_scenarios, n_phi, Nt))
# # market_view_matrix = np.empty((N, n_scenarios, n_phi, Nt))
# survey_view_matrix = np.empty((N, n_scenarios, n_phi, Nt))
# #Delta_matrix = np.empty((N, n_scenarios, n_phi_short, Nt, Nc))
# #pi_matrix = np.empty((N, n_scenarios, n_phi_short, Nt, Nc))
# # mu_st_rt_matrix = np.empty((N, n_scenarios, n_phi, Nt, Nc))
# r_matrix = np.zeros((N, n_scenarios, n_phi, Nt))
# #belief_variance_matrix = np.zeros((N, n_scenarios, n_phi, Nt))
# #belief_skew_matrix = np.zeros((N, n_scenarios, n_phi, Nt))
# dR_matrix = np.zeros((N, n_scenarios, n_phi, Nt))
# delta_bar_matrix = np.zeros((N, n_scenarios, n_phi, Nt))
# Phi_parti_1_matrix = np.zeros((N, n_scenarios, n_phi, Nt))
# #parti_old_matrix = np.zeros((N, n_scenarios, n_phi, Nt))
# #parti_young_matrix = np.zeros((N, n_scenarios, n_phi, Nt))
#
# # run the program for different values of phi, and store the results
# for j in range(N):
#     print(j)
#     dZ = dZ_matrix[j]
#     dZ_build = dZ_build_matrix[j]
#     dZ_SI = dZ_SI_matrix[j]
#     dZ_SI_build = dZ_SI_build_matrix[j]
#     for k, scenario in enumerate(scenarios_short):
#         mode_trade = scenario[0]
#         mode_learn = scenario[1]
#         for l, phi_try in enumerate(phi_vector):
#             (
#                 r,
#                 theta,
#                 f,
#                 Delta,
#                 pi,
#                 popu_parti,
#                 f_parti,
#                 Delta_bar_parti,
#                 dR,
#                 invest_tracker,
#                 popu_short_mat,
#                 popu_can_short_mat,
#             ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta, phi_try,
#                             Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
#                             top=0.05,
#                             old_limit=100
#                             )
#
#             # Delta_matrix[j, k, l] = np.average(Delta, axis = 0)
#             # pi_matrix[j, k, l] = pi
#             theta_matrix[j, k, l] = theta
#             r_matrix[j, k, l] = r
#             theta_mat = np.transpose(np.tile(theta, (Nc, 1)))
#             # mu_st_rt_matrix[j, k, l] = theta_mat + Delta
#             popu_parti_matrix[j, k, l] = popu_parti
#             # market_view_matrix[j, k, l] = np.average(Delta, axis=1, weights=f)
#             delta_bar_matrix[j, k, l] = Delta_bar_parti
#             survey_view_matrix[j, k, l] = np.average(Delta, axis=1, weights=cohort_size)
#             #belief_variance_matrix[j, k, l] = weighted_variance(Delta, cohort_size, 1)  # I defined the two weighted functions myself
#             #belief_skew_matrix[j, k, l] = weighted_skew(Delta, cohort_size, 1)
#             dR_matrix[j, k, l] = dR
#             # invest_tracker_matrix[j, k, l] = invest_tracker
#             Phi_parti_1_matrix[j, k, l] = 1/f_parti
#             #parti_young_matrix[j, k, l] = np.average(invest_tracker[:, age_cutoff:], axis=1, weights = cohort_size[age_cutoff:])
#             #parti_old_matrix[j, k, l] = np.average(invest_tracker[:, :age_cutoff], axis=1, weights=cohort_size[:age_cutoff])
#             # ( parti_young + parti_old )/2 = popu_parti


# ######################################
# ########## ONE RANDOM PATH ############
# ############ GRAPH ONE ###############
# ######################################

# ONE SPECIFIC PATH:
print('Generating data for the graphs:')
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
invest_tracker_compare = np.zeros((n_scenarios, 2, 2, n_phi_short, Nt, Nc))
cohort_size_mat = np.tile(cohort_size, (Nc, 1))
for g, scenario in enumerate(scenarios_short):
    mode_trade = scenario[0]
    mode_learn = scenario[1]
    for i in range(2):
        dZ = dZ_Y_cases[i]
        log_Yt = np.cumsum((mu_Y - 0.5 * sigma_Y ** 2) * dt + sigma_Y * dZ)
        log_Yt_mat = np.transpose(np.tile(log_Yt, (Nc, 1)))
        for j in range(2):
            dZ_SI = dZ_SI_cases[j]
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
                    invest_tracker,
                    popu_can_short,
                    popu_short,
                    Phi_can_short,
                    Phi_short,
                ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta,
                                phi,
                                Npre, Ninit, T_hat, dZ_build_case, dZ, dZ_SI_build_case, dZ_SI, tau, cohort_size,
                                need_f='True',
                                need_Delta='True',
                                need_pi='True',
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
                invest_tracker_compare[g, i, j, k] = invest_tracker
# cohort_matrix_list = [pi_compare, Delta_compare, cons_compare]


nn = 3  # number of cohorts illustrated
length = len(t)
starts = np.zeros(nn)
Delta_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
pi_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
cons_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
switch_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
parti_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
for o in range(n_scenarios):
    for i in range(2):
        for j in range(2):
            for l in range(n_phi_short):
                pi = pi_compare[o, i, j, l]
                cons = cons_compare[o, i, j, l]
                Delta = Delta_compare[o, i, j, l]
                for m in range(nn):
                    start = int((m + 1) * 100 * (1 / dt))
                    starts[m] = start * dt
                    for n in range(length):
                        if n < start:
                            pi_time_series[o, i, j, l, m, n] = np.nan
                            cons_time_series[o, i, j, l, m, n] = np.nan
                            Delta_time_series[o, i, j, l, m, n] = np.nan
                        else:
                            cohort_rank = length - (n - start) - 1
                            Delta_time_series[o, i, j, l, m, n] = Delta[n, cohort_rank]
                            pi_time_series[o, i, j, l, m, n] = pi[n, cohort_rank]
                            cons_time_series[o, i, j, l, m, n] = cons[n, cohort_rank]
                    parti = pi_time_series[o, i, j, l, m] > 0
                    switch = abs(parti[1:] ^ parti[:-1])
                    sw = np.append(switch, 0)
                    parti = np.where(sw == 1, 0.5, parti)
                    switch = np.insert(switch, 0, 0)
                    parti = np.where(switch == 1, 0.5, parti)
                    parti_time_series[o, i, j, l, m] = parti
                    switch_time_series[o, i, j, l, m] = sw


label_phi = []
for i in range(n_phi_short):
    label_phi.append(r'$\phi$ = ' + str(phi_vector_short[i]))
labels = [scenario_labels, label_phi, label_phi]
# lower = min(np.min(y1), np.min(y2))
# upper = max(np.max(y1), np.max(y2))

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




# ######################################
# ########### Figure 1 & 2 #############
# ######################################
# Delta (2 phi * 4 cases)
print('Figure 1')
upper = 60  # todo: endogenize these parameters
lower = -60
scenario_index = 1
for i in range(1, n_phi_short, 1):
    phi = phi_vector_short[i]
    fig, axes = plt.subplots(nrows=4, ncols=1, sharey='all', sharex='all', figsize=(15, 20))
    for j, ax in enumerate(axes):
        red_index = 0 if j == 0 or j == 1 else 1
        yellow_index = 0 if j == 0 or j == 2 else 1
        red_case = red_labels[red_index]
        yellow_case = yellow_labels[yellow_index]
        Z = np.cumsum(Z_Y_cases[red_index])
        Z_SI = np.cumsum(Z_SI_cases[yellow_index])  # todo: plot SI (combination of Z_Y ad Z_SI) instead of Z_SI
        if j == 3:
            ax.set_xlabel('Time in simulation')
        ax.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
        ax.plot(t, Z, color='red', linewidth=0.5, label=r'$z^Y_t$')
        ax.plot(t, Z_SI, color='gold', linewidth=0.5, label=r'$z^{SI}_t$')
        ax.set_ylim([lower, upper])
        ax.tick_params(axis='y', labelcolor='black')
        if j == 0:
            ax.legend()
        ax.set_title(red_case + yellow_case)

        ax2 = ax.twinx()
        ax2.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
        ax2.set_ylim([-0.3, 0.4])
        for m in range(nn):
            # switch[m, starts[m]] = 1
            y_cohort = Delta_time_series[scenario_index, red_index, yellow_index, i, m]
            y_cohort_N = np.ma.masked_where(parti_time_series[scenario_index, red_index, yellow_index, i, m] == 1, y_cohort)
            y_cohort_P = np.ma.masked_where(parti_time_series[scenario_index, red_index, yellow_index, i, m] == 0, y_cohort)
            y_cohort_switch = np.ma.masked_where(switch_time_series[scenario_index, red_index, yellow_index, i, m] == 0, y_cohort)
            plt.vlines(starts[m], ymax=upper, ymin=lower, color='grey', linestyle='--', linewidth=0.4)
            #  ax2.plot(t, y_cohort, label=cohort_labels[m], color=colors_short[m], linewidth=0.4)
            if j == 0:
                ax2.plot(t, y_cohort_P, color=colors_short[m], linewidth=0.4, label=cohort_labels[m])
                ax2.plot(t, y_cohort_N, color=colors_short[m], linewidth=0.4, linestyle='dotted')
                ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o')
            elif j == 1 and m == 0:
                ax2.plot(t, y_cohort_P, color=colors_short[m], linewidth=0.4, label=PN_labels[0])
                ax2.plot(t, y_cohort_N, color=colors_short[m], linewidth=0.4, linestyle='dotted', label=PN_labels[1])
                ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o', label='switch')
            else:
                ax2.plot(t, y_cohort_P, color=colors_short[m], linewidth=0.4)
                ax2.plot(t, y_cohort_N, color=colors_short[m], linewidth=0.4, linestyle='dotted')
                ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o')

        ax2.tick_params(axis='y', labelcolor='black')
        if j <= 1:
            ax2.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(
                'Shocks and Delta time series' + str(round(phi, 2)) + '.png',
                dpi=60)
    plt.show()
    plt.close()


# ######################################
# ############# Figure 3 ###############
# ######################################
# the zoom-in ones
# bad & Bad, phi = 0.4
# cohort 1; cohort 2; cohort 1, complete; cohort 1, disappointment
year = 100
t_length = int(year/dt)
t_indexes = np.empty((2,2,2))
red_case = 1
yellow_case = 1
cohort_start = 3
t_indexes[0, 0, 0] = t_indexes[1, 0, 0] = t_indexes[1, 1, 0] = t_indexes[0, 1, 0] = t_length * cohort_start
t_indexes[0, 0, 1] = t_indexes[1, 0, 1] = t_indexes[1, 1, 1] = t_indexes[0, 1, 1] = t_length * (1 + cohort_start)
phi_where = [(1, 2), (1, 1)]
cohort_indexes = [(cohort_start-1, cohort_start-1), (cohort_start-1, cohort_start-1)]
scenario_indexs = [(1, 1), (0, 2)]
titles_subfig = [(r'Reentry, $\phi=0.4$', r'Reentry, $\phi=0.8$'), (r'Complete market, $\phi=0.4$', r'Disappointment, $\phi=0.4$')]
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
            ax.plot(x, cutoff_belief, label=r'Cutoff $\Delta_{s,t}$', color='blue', alpha=0.6, linewidth=0.4)
            ax.plot(x, y_P, label='Participant (P)', color='black', linewidth=0.6)
            ax.plot(x, y_N, label='Nonparticipant (N)', color='black', linewidth=0.6, linestyle='dotted')
            # ax2.fill_between(t, Delta_benchmarks[0], Delta_benchmarks[1], color=colors[m], alpha=0.4)
            ax.scatter(x, y_switch, color='red', s=10, marker='o', label='switch')
        if i == j == 0:
            ax.legend()
        if j == 0:
            ax.set_ylabel(r'Estimation error $\Delta_{s,t}$')
        if i == 1:
            ax.set_xlabel('Time in simulation')
        ax.set_title(titles_subfig[i][j])
        ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout(h_pad=2)
plt.savefig('Shocks and Delta, zoom in' + str(red_case) + str(yellow_case) +'.png', dpi=60)
plt.show()
plt.close()



# ######################################
# ############ Figure 3.1 ##############
# ######################################
N_1 = 1000
phi_5 = [0, 0.2, 0.4, 0.6, 0.8]
n_phi_5 = len(phi_5)
# tax_vector = [0.005]
tax_vector = [0.01]
# tax_vector = [0.015]
# tax_vector = [0.005, 0.01, 0.015]
n_tax = len(tax_vector)
Delta_reentry_matrix = np.empty((N_1, n_phi_5, Nc))
f_reentry_matrix = np.empty((N_1, n_phi_5, Nc))
distance_reentry_matrix = np.empty((N_1, n_phi_5, Nc))
invest_matrix = np.empty((N_1, n_phi_5, Nc))

Delta_complete_matrix = np.empty((N_1, n_phi_5, Nc))
f_complete_matrix = np.empty((N_1, n_phi_5, Nc))
distance_complete_matrix = np.empty((N_1, n_phi_5, Nc))
dt_root = np.sqrt(dt)
for j in range(N_1):
    print(j)
    dZ = np.random.randn(Nt) * dt_root
    dZ_build = np.random.randn(Nc) * dt_root
    dZ_SI = np.random.randn(Nt) * dt_root
    dZ_SI_build = np.random.randn(Nc) * dt_root
    for l, phi_try in enumerate(phi_5):
        (
            r_reentry,
            theta_reentry,
            f_reentry,
            Delta_reentry,
            pi_reentry,
            popu_parti_reentry,
            f_parti_reentry,
            Delta_bar_parti_reentry,
            dR_reentry,
            invest_tracker_reentry,
            popu_can_short_reentry,
            popu_short_reentry,
            Phi_can_short_reentry,
            Phi_short_reentry,
        ) = simulate_SI('w_constraint', 'reentry', Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S,
                        tax,
                        beta,
                        phi_try,
                        Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                        need_f='True',
                        need_Delta='True',
                        need_pi='False',
                        )
        Delta_bar_parti_mat = np.transpose(np.tile(Delta_bar_parti_reentry, (Nc, 1)))
        Delta_reentry_matrix[j, l] = np.average(np.abs(Delta_reentry), axis=0)
        f_reentry_matrix[j, l] = np.average(f_reentry, axis=0)
        distance_reentry_matrix[j, l] = np.average(np.abs(Delta_reentry - Delta_bar_parti_mat), axis=0)
        invest_matrix[j, l] = np.average(invest_tracker_reentry, axis=0)

        (
            r_complete,
            theta_complete,
            f_complete,
            Delta_complete,
            pi_complete,
            popu_complete,
            f_parti_complete,
            Delta_bar_complete,
            dR_complete,
            invest_tracker_complete,
            popu_can_short_complete,
            popu_short_complete,
            Phi_can_short_complete,
            Phi_short_complete,
        ) = simulate_SI('complete', 'reentry', Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S,
                        tax,
                        beta,
                        phi_try,
                        Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                        need_f='True',
                        need_Delta='True',
                        need_pi='False',
                        )
        Delta_bar_complte_mat = np.transpose(np.tile(Delta_bar_complete, (Nc, 1)))
        distance_complete_matrix[j, l] = np.average(np.abs(Delta_complete - Delta_bar_complte_mat), axis=0)
        Delta_complete_matrix[j, l] = np.average(np.abs(Delta_complete), axis=0)
        f_complete_matrix[j, l] = np.average(np.abs(f_complete), axis=0)

# # read the data:
# Delta_reentry_vector = np.empty((3, n_tax, Nc))
# f_reentry_vector = np.empty((3, n_tax, Nc))
# distance_reentry_vector = np.empty((3, n_tax, Nc))
# invest_vector = np.empty((3, n_tax, Nc))
# Delta_complete_vector = np.empty((3, n_tax, Nc))
# f_complete_vector = np.empty((3, n_tax, Nc))
# distance_complete_vector = np.empty((3, n_tax, Nc))
# var_list = [Delta_reentry_vector, f_reentry_vector, distance_reentry_vector, invest_vector,
#             Delta_complete_vector, f_complete_vector, distance_complete_vector]
# var_name_list = ['Delta_reentry_data', 'f_reentry_data', 'distance_reentry_data', 'invest_data',
#             'Delta_complete_data', 'f_complete_data', 'distance_complete_data']
# for i, var in enumerate(var_list):
#     var_name = var_name_list[i]
#     for j, tax_try in enumerate(tax_vector):
#         var_name_j = var_name  + 'taxrate' + str(tax_try) +'.npy'
#         y = np.load(var_name_j)
#         var[:, j] = np.flip(np.mean(np.mean(y, axis=0), axis=1), axis=1)

Delta_reentry_vector = np.flip(np.average(Delta_reentry_matrix, axis=0), axis=1)
f_reentry_vector = np.flip(np.average(f_reentry_matrix, axis=0), axis=1)
invest_vector = np.flip(np.average(invest_matrix, axis=0), axis=1)
Delta_complete_vector = np.flip(np.average(Delta_complete_matrix, axis=0), axis=1)
f_complete_vector = np.flip(np.average(f_complete_matrix, axis=0), axis=1)

# Graph:
t_cut = 100
N_cut = int(t_cut/dt)
x = t[:N_cut]
data_point = np.arange(0, N_cut, 15)
y_cases = [Delta_complete_vector, Delta_reentry_vector, invest_vector]
fig_titles = ['Complete market', 'Reentry', 'Reentry']
y_titles = [r'Average $\mid\Delta_{s,t}\mid$', r'Average $\mid\Delta_{s,t}\mid$', 'Average participation probability']
fig, axes = plt.subplots(nrows=1, ncols=3, sharex='all', figsize=(15, 5))
for j, ax in enumerate(axes):
    ax.set_xlabel('Age')
    y_case = y_cases[j]
    ax.set_ylabel(y_titles[j])
    for i in range(5):
        y = y_case[i, :N_cut]
        label_i = r'$\phi$=' + str('{0:.2f}'.format(phi_5[i]))
        ax.plot(x[data_point], y[data_point], color=colors[i], linewidth=0.5, label=label_i)
    if j < 2:
        ax.set_ylim(0.04, 0.18)
    if j == 0:
        ax.legend()
    ax.set_title(fig_titles[j], color='black')
    ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(str(N_1) + 'paths, ' + str(t_cut) + 'age, ' +'Average estimation error and age.png', dpi=100)
plt.show()
#plt.close()



# t_cut = 500
# N_cut = int(t_cut/dt)
# x = t[:N_cut]
# data_point = np.arange(0, N_cut, 15)
# y_cases = [Delta_reentry_vector, f_reentry_vector, distance_reentry_vector * sigma_Y, invest_vector]
# y_complete_cases = [Delta_complete_vector, f_complete_vector, distance_complete_vector * sigma_Y, invest_vector]
# fig_titles = [r'Average estimation error $\Delta_{s,t}$', r'Average cohort wealth share $f_{s,t}$',
#             r'Average distance to market view $\mu_{s,t}-\bar{\mu}_t$', 'Average participation probability']
# y_titles = [r'Average $\Delta_{s,t}$', r'Average $f_{s,t}$',
#             r'Average $\mu_{s,t}-\bar{\mu}_t$', 'Average participation probability']
# fig, axes = plt.subplots(nrows=4, ncols=2, sharex='all', sharey='row', figsize=(10, 15))
# for i, ax_row in enumerate(axes):
#     var = y_cases[i]
#     label_fig = fig_titles[i]
#     var_complete = y_complete_cases[i]
#     for j, ax in enumerate(ax_row):
#         ax.set_xlabel('Age')
#         if j == 0:  # tax = 0.01, different phi
#             y_case = var[:, 1]
#             y_complete_case = var_complete[:, 1]
#             label_j = label_fig + r', $\tau$=' + str('{0:.3f}'.format(tax_vector[1]))
#             ax.set_ylabel(y_titles[i])
#         if j == 1:  # phi = 0.4, different tax
#             y_case = var[1]
#             y_complete_case = var_complete[1]
#             label_j = label_fig + r', $\phi$=' + str('{0:.1f}'.format(phi_5[1]))
#         ax.set_title(label_j, color='black')
#         for k in range(3):
#             if j == 0:
#                 label_i = r'$\phi$=' + str('{0:.1f}'.format(phi_5[k]))
#             else:
#                 label_i = r'$\tau$=' + str('{0:.3f}'.format(tax_vector[k]))
#             y = y_case[k, :N_cut]
#             y_complete = y_complete_case[k, :N_cut]
#             X_Y_Spline = make_interp_spline(x[data_point], y[data_point])
#             X_ = np.linspace(x[data_point].min(), x[data_point].max(), 100)
#             Y_ = X_Y_Spline(X_)
#             if i <= 2:
#                 if i == 1 and j == 0 and k == 0:
#                     ax.plot(x, y_complete, color=colors_short[k], linewidth=0.6, linestyle='dashed', label='Complete market benchmark')
#                     ax.plot(X_, Y_, color=colors_short[k], linewidth=0.8, label='Reentry')
#                     ax.legend()
#                 else:
#                     ax.plot(x, y_complete, color=colors_short[k], linewidth=0.6, linestyle='dashed')
#                     ax.plot(X_, Y_, color=colors_short[k], linewidth=0.8, label=label_i)
#             else:
#                 ax.plot(X_, Y_, color=colors_short[k], linewidth=0.8, label=label_i)
#             if i == 2:
#                 x_minmax = np.argmin(y)
#                 if x_minmax * dt < np.max(x):
#                     ax.axvline(x_minmax * dt, 0.05, 0.95, color=colors_short[k], linewidth=0.4, linestyle='dotted')
#             if i == 3:
#                 x_minmax = np.argmax(y)
#                 if x_minmax * dt < np.max(x):
#                     ax.axvline(x_minmax * dt, 0.05, 0.95, color=colors_short[k], linewidth=0.4, linestyle='dotted')
#         # if j == 0 or j == 2:
#         #     ax.set_ylim(0.04, 0.18)
#         if i == 0:
#             ax.legend()
#         ax.tick_params(axis='y', labelcolor='black')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# # plt.savefig('Average participation prob over age, tau and phi.png', dpi=100)
# plt.show()
# # plt.close()



# ######################################
# ############ Figure 3.1 ##############
# ######################################
N_1 = 200
phi = 0
tax = 0.01
age_cutoff = 500
cohort_cutoff = int(age_cutoff / dt)
n_age_groups = 10
n_cohort_per_group = int(cohort_cutoff / n_age_groups)
cummu_popu = np.cumsum(cohort_size)
age_buckets = np.searchsorted(cummu_popu, 1-np.arange(0, 1.1, (1/n_age_groups)))
Delta_reentry_matrix = np.empty((N_1, Nt, n_age_groups))
f_reentry_matrix = np.empty((N_1, Nt, n_age_groups))
invest_matrix = np.empty((N_1, Nt, n_age_groups))
dt_root = np.sqrt(dt)
for j in range(N_1):
    print(j)
    dZ = np.random.randn(Nt) * dt_root
    dZ_build = np.random.randn(Nc) * dt_root
    dZ_SI = np.random.randn(Nt) * dt_root
    dZ_SI_build = np.random.randn(Nc) * dt_root
    (
        r_reentry,
        theta_reentry,
        f_reentry,
        Delta_reentry,
        pi_reentry,
        popu_parti_reentry,
        f_parti_reentry,
        Delta_bar_parti_reentry,
        dR_reentry,
        invest_tracker_reentry,
        popu_can_short_reentry,
        popu_short_reentry,
        Phi_can_short_reentry,
        Phi_short_reentry,
    ) = simulate_SI('w_constraint', 'reentry', Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S,
                    tax,
                    beta,
                    phi,
                    Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                    need_f='True',
                    need_Delta='True',
                    need_pi='False',
                    top=0.05,
                    old_limit=100
                    )
    # Delta_age_groups = np.reshape(Delta_reentry, (Nt, -1, n_cohort_per_group))
    # f_reentry_age_groups = np.reshape(f_reentry, (Nt, -1, n_cohort_per_group))
    # invest_age_groups = np.reshape(invest_tracker_reentry, (Nt, -1, n_cohort_per_group))
    #
    # Delta_reentry_matrix[j] = np.flip(
    #     np.average(
    #         Delta_age_groups,
    #         axis=2,
    #         weights=cohort_size[-n_cohort_per_group:]
    #     )[:, -n_age_groups:],
    #     axis=1)
    # f_reentry_matrix[j] = np.flip(
    #     np.sum(
    #         f_reentry_age_groups,
    #         axis=2
    #     )[:, -n_age_groups:] * dt,
    #     axis=1)
    # invest_matrix[j] = np.flip(
    #     np.average(
    #         invest_age_groups,
    #         axis=2,
    #         weights=cohort_size[-n_cohort_per_group:]
    #     )[:, -n_age_groups:],
    #     axis=1)
    for i in range(n_age_groups):
        below = int(age_buckets[i+1])
        above = int(age_buckets[i])
        Delta_reentry_matrix[j, :, i] = np.average(Delta_reentry[:, below:above], axis=1, weights=cohort_size[below:above])
        f_reentry_matrix[j, :, i] = np.sum(f_reentry[:, below:above], axis=1) * dt
        invest_matrix[j, :, i] = np.average(invest_tracker_reentry[:, below:above], axis=1,
                                                   weights=cohort_size[below:above])

var_list = [Delta_reentry_matrix, f_reentry_matrix, invest_matrix]
var_name_list = ['Delta_reentry_data', 'f_reentry_data', 'invest_data']

# Graph:
x = np.arange(1, 11, 1)
y_case = invest_matrix
# y_case = np.average(invest_matrix, axis=0)
y_title = 'Average participation probability'
conditions = [Delta_reentry_matrix, f_reentry_matrix]
# conditions = [np.average(Delta_reentry_matrix, axis=0), np.average(f_reentry_matrix, axis=0)]
condition_labels = [r'Estimation error $\Delta_{s,t}$ quartiles conditional on age',
                    r'Wealth share $f_{s,t}$ quartiles conditional on age']
n_tiles = 6
tiles = np.arange(0, 101, int(100/n_tiles))
label_tiles = 'Quartile '
fig, axes = plt.subplots(nrows=1, ncols=2, sharex='all', figsize=(15, 7))
for j, ax in enumerate(axes):
    # ax.set_xlabel('Age')
    ax.set_xlabel('Age dixile')
    condition = conditions[j]
    ax.set_ylabel(y_title)
    y_data = np.empty((n_tiles, n_age_groups, 3))
    for i in range(n_age_groups):
        # y_data_age_group = y_case[:, i]
        # condition_data_age_group = condition[:, i]
        y_data_age_group = y_case[:, :, i]
        condition_data_age_group = condition[:, :, i]
        condition_quartiles = np.percentile(condition_data_age_group, tiles)
        for k in range(n_tiles):
            below = condition_quartiles[k]
            above = condition_quartiles[k+1]
            a = condition_data_age_group >= below
            b = condition_data_age_group <= above
            data_where = np.where(a * b == 1)
            y_data_where = y_data_age_group[data_where]
            y_data[k, i, 0] = np.median(y_data_where)
            y_data[k, i, 1] = np.percentile(y_data_where, 25)
            y_data[k, i, 2] = np.percentile(y_data_where, 75)
    for i in range(n_tiles):
        if i == 0:
            label_i = label_tiles + str(i+1) + ', smallest'
        elif i == n_tiles - 1:
            label_i = label_tiles + str(i + 1) + ', largest'
        else:
            label_i = label_tiles + str(i + 1)
        ax.plot(x, y_data[i, :, 0], color=colors[i], linewidth=0.5, label=label_i)
        ax.fill_between(x, y_data[i, :, 1], y_data[i, :, 2], color=colors[i], linewidth=0., alpha=0.3)
        ax.legend()
    ax.set_title(condition_labels[j], color='black')
    ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig(str(N_1) + 'paths, ' + str(t_cut) + 'age, ' +'Average estimation error and age.png', dpi=100)
plt.show()
#plt.close()

n_tiles = 2   # above & below median for both belief and wealth
tiles = np.arange(0, 101, int(100/n_tiles))

fig, ax = plt.subplots(figsize=(15, 15))
# ax.set_xlabel('Age')
ax.set_xlabel('Age dixile')
condition1 = conditions[0]
label_tiles1 = r'$\Delta_{s,t}$'
condition2 = conditions[1]
label_tiles2 = r'$f_{s,t}$'
ax.set_ylabel(y_title)
y_data = np.empty((n_tiles, n_tiles, n_age_groups, 3))
for i in range(n_age_groups):
    # y_data_age_group = y_case[:, :, i]
    # condition_data_age_group = condition[:, :, i]
    # y_data_age_group = y_case[:, i]
    # condition_data_age_group1 = condition1[:, i]
    # condition_data_age_group2 = condition2[:, i]
    y_data_age_group = y_case[:, :, i]
    condition_data_age_group1 = condition1[:, :, i]
    condition_data_age_group2 = condition2[:, :, i]
    condition_median1 = np.median(condition_data_age_group1)
    condition_median2 = np.median(condition_data_age_group2)
    for k1 in range(n_tiles):
        a = condition_data_age_group1 <= condition_median1 if k1 == 0 \
            else condition_data_age_group1 >= condition_median1
        for k2 in range(n_tiles):
            b = condition_data_age_group2 <= condition_median2 if k1 == 0 \
                else condition_data_age_group2 >= condition_median2
            data_where = np.where(a * b == 1)
        y_data_where = y_data_age_group[data_where]
        y_data[k1, k2, i, 0] = np.median(y_data_where)
        y_data[k1, k2, i, 1] = np.percentile(y_data_where, 25)
        y_data[k1, k2, i, 2] = np.percentile(y_data_where, 75)
ii = 0
for k1 in range(n_tiles):
    label_k1 = 'Below median '+label_tiles1 if k1 == 0 else 'Above median '+label_tiles1
    for k2 in range(n_tiles):
        label_k2 = 'Below median ' + label_tiles2 if k2 == 0 else 'Above median ' + label_tiles2
        ax.plot(x, y_data[k1, k2, :, 0], color=colors[ii], linewidth=0.5, label=label_k1+label_k2)
        ax.fill_between(x, y_data[k1, k2, :, 1], y_data[k1, k2, :, 2], color=colors[ii], linewidth=0., alpha=0.3)
        ax.legend()
        ii += 1
ax.set_title('Two way sorts', color='black')
ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig(str(N_1) + 'paths, ' + str(t_cut) + 'age, ' +'Average estimation error and age.png', dpi=100)
plt.show()
#plt.close()


n_bins = 30
age_group_indexes = [0, n_age_groups-1]  # 0-5 years old vs. 95-100 years old
total_count = np.count_nonzero(Delta_reentry_matrix[:, :, 0])
fig, axes = plt.subplots(ncols=4, figsize=(15, 4))
for j, ax in enumerate(axes):
    if j == 0 or j == 2:
        age_index = age_group_indexes[0]
    else:
        age_index = age_group_indexes[1]
    if j == 0 or j == 1:
        condition = conditions[0][:, :, age_index]
    else:
        condition = conditions[1][:, :, age_index]

    condition_var_density = np.empty(n_bins)
    condition_quartiles = np.percentile(condition, tiles)
    min_condition = np.min(condition)
    max_condition = np.max(condition)
    width_bins = (max_condition - min_condition) / n_bins
    condition_var_x = np.linspace(min_condition + width_bins / 2, max_condition - width_bins / 2, n_bins)
    for i in range(n_bins):
        bin_left = min_condition + i * width_bins
        bin_right = bin_left + width_bins
        bin_1 = condition <= bin_right
        bin_2 = condition >= bin_left
        bin_where = np.where(bin_1 * bin_2 == 1)
        condition_var_density[i] = np.shape(bin_where)[1] / total_count
    X_Y_Spline = make_interp_spline(condition_var_x, condition_var_density)
    X_ = np.linspace(min_condition, max_condition, 1000)
    Y_ = X_Y_Spline(X_)
    for i in range(n_tiles):
        if i > 0:
            ax.axvline(condition_quartiles[i], 0.05, 0.95, linestyle='dashed', linewidth=0.8, color='gray')
        left_x = min_condition if i == 0 else condition_quartiles[i]
        right_x = max_condition if i == n_tiles - 1 else condition_quartiles[i + 1]
        a = X_ >= left_x
        b = right_x >= X_
        bin_where = np.where(a * b == 1)
        x = X_[bin_where]
        y = Y_[bin_where]
        ax.fill_between(x, 0, y, color=colors[i], linewidth=0., alpha=0.3)
    # ax.legend()
    # ax.set_xlim(0, 0.45) if tax > 0.01 else ax.set_xlim(0, max_condition)
    # ax.set_xlabel(condition_label)
    ax.set_ylabel('Density')
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
# plt.savefig(str(tax)+'Intuition wealth distribution.png', dpi=200)
plt.show()
# plt.close()


# ######################################
# #### Figure endogenous learning ######
# ######################################
N_1 = 2000
n_scenarios = 2  # complete vs. reentry
keep_when = int(200 / dt)
t_gap = int(2 / dt)  # 2-year non-overlapping rolling window
N_gap = int((Nt - keep_when) / t_gap)
t_rolling_pre = np.arange(keep_when - 1, Nt - 1, t_gap)  # pre
t_rolling_post = t_rolling_pre + t_gap
phi_vector = [0, 0.4, 0.8]
n_phi = len(phi_vector)
shocks_mat = np.empty((N_1, N_gap))
shocks_SI_mat = np.empty((N_1, N_gap))
parti_rate_pre_mat = np.empty((N_1, n_phi, N_gap))
parti_rate_post_mat = np.empty((N_1, n_phi, N_gap))
update_belief_mat = np.empty((N_1, n_scenarios, n_phi, N_gap))
# update_belief_parti_mat = np.empty((N_1, n_phi, N_gap, 2))
# update_belief_age_mat = np.empty((N_1, n_scenarios, n_phi, N_gap, 4))
cohort_size_mat = np.tile(cohort_size[1:], (Nt - 1, 1))
cohort_size_short = cohort_size[1:]

# run
for i in range(N_1):
    print(i)
    dZ = np.random.randn(Nt) * dt_root
    dZ_build = np.random.randn(Nc) * dt_root
    dZ_SI = np.random.randn(Nt) * dt_root
    dZ_SI_build = np.random.randn(Nc) * dt_root
    shocks = np.cumsum(dZ)
    shocks_mat[i] = shocks[t_rolling_post] - shocks[t_rolling_pre]
    shocks_SI = np.cumsum(dZ_SI)
    shocks_SI_mat[i] = shocks_SI[t_rolling_post] - shocks_SI[t_rolling_pre]
    for j in range(n_scenarios):
        scenario = scenarios[j]
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        for n, phi in enumerate(phi_vector):
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
                popu_can_short,
                popu_short,
                Phi_can_short,
                Phi_short,
            ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta,
                            phi,
                            Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                            need_f='False',
                            need_Delta='True',
                            need_pi='True',
                            top=0.05,
                            old_limit=100
                            )
            update_belief = Delta[1:, :-1] - Delta[:-1, 1:]  # change of belief incorporating the current shock t, t = 1:5999
            update_belief_t = np.average(
                update_belief, weights=cohort_size_mat, axis=1
            ) # everyone
            update_belief_cumsum = np.cumsum(update_belief_t)
            update_belief_mat[i, j, n] = update_belief_cumsum[t_rolling_post - 1] - update_belief_cumsum[t_rolling_pre - 1]

            if mode_trade == 'w_constraint':
                parti_rate_pre_mat[i, n] = popu_parti[t_rolling_pre]
                parti_rate_post_mat[i, n] = popu_parti[t_rolling_post]


x_var = parti_rate_pre_mat
y_var = update_belief_mat
n_bins = 8
y_percentiles = [50, 25, 75]
data_figure_y = np.zeros((2, 2, 3, n_bins - 1, len(y_percentiles)))
data_figure_x = np.zeros((2, 2, 3, n_bins - 1))
data_figure_complete = np.zeros((2, 2, 3, 3, len(y_percentiles)))
# reentry scenario
condition_var1 = shocks_mat
condition_var2 = shocks_SI_mat

for i1 in range(2):
    data_where1 = condition_var1 >= np.percentile(condition_var1, 75) if i1 == 0 else condition_var1 < np.percentile(condition_var1, 25)
    for i2 in range(2):
        data_where2 = condition_var2 >= np.percentile(condition_var2, 75) if i2 == 0 else condition_var2 < np.percentile(condition_var2, 25)
        data_where = np.where(data_where1 * data_where2 == 1)
        for k in range(n_phi):
            data_focus = y_var[:, 1, k][data_where]
            x_focus = x_var[:, k][data_where]
            below_dz = np.percentile(x_focus, 0)
            above_dz = np.percentile(x_focus, 100)
            bins = np.linspace(below_dz, above_dz, n_bins)
            bin_size = (above_dz - below_dz) / (n_bins - 1)
            data_figure_x[i1, i2, k] = np.linspace(below_dz + bin_size / 2, above_dz - bin_size / 2, n_bins - 1)
            data_complete = y_var[:, 0, k][data_where]
            data_figure_complete[i1, i2, k] = np.percentile(data_complete, y_percentiles)

            for j in range(n_bins - 1):
                bin_0 = bins[j]
                bin_1 = bins[j + 1]
                below_bin = bin_1 >= x_focus
                above_bin = x_focus >= bin_0
                bin_where = np.where(below_bin * above_bin == 1)
                data_focus_z = data_focus[bin_where]
                data_figure_y[i1, i2, k, j] = np.percentile(data_focus_z, y_percentiles)


# figure:
label_np = ['Good ', 'Bad ']
label_shock = [r'$dz^{Y}$, ', r'signal, $dz^{SI}$']
labels = [r'$\phi = 0.0$', r'$\phi = 0.4$', r'$\phi = 0.8$']
X_ = np.linspace(0.3, 0.8, 100)
X_complete = np.linspace(0.98, 1., 3)
X_gap = np.linspace(0.8, 0.98, 2)
fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10), sharey='all')
for j, row in enumerate(axes):
    for k, ax in enumerate(row):
        for l in range(3):
            y = data_figure_y[j, k, l]
            x = data_figure_x[j, k, l]
            y_complete = data_figure_complete[j, k, l]
            Y_mat = np.empty((3, 100))
            for m in range(3):
                X_Y_Spline = make_interp_spline(x, y[:, m])
                Y_mat[m] = X_Y_Spline(X_)
            if j == k == 0:
                ax.plot(X_, Y_mat[0], color=colors[l], linewidth=0.8, label=labels[l])
            else:
                ax.plot(X_, Y_mat[0], color=colors[l], linewidth=0.8)
            ax.plot(X_complete, y_complete[:, 0], color=colors[l], linewidth=0.8)
            ax.plot(X_gap, [Y_mat[0, -1], y_complete[0, 0]], color=colors[l], linewidth=0.8, linestyle ='dashed')
            ax.fill_between(X_, Y_mat[1], Y_mat[2], color=colors[l], linewidth=0., alpha=0.3)
            ax.fill_between(X_complete, y_complete[:, 1], y_complete[:, 2], color=colors[l], linewidth=0., alpha=0.3)
            ax.axhline(0, 0.05, 0.95, color='gray', linestyle='dashed', linewidth=0.6)
            if j == k == 0:
                ax.legend()
        ax.set_xlabel('Participation rate in the economy')
        ax.set_ylabel(r'Changes in average estimation error, $\Delta$')
        ax.set_title(label_np[j] + r'fundamental $dz^{Y}$, ' + label_np[k] + r'signal $dz^{SI}$')
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
# plt.savefig('Endogenous_learning.png', dpi=100)
plt.show()
# plt.close()


# ######################################
# ############ Figure 3.2 ##############
# ######################################
N_1 = 1000
age_cut = 150
Nc_cut = int(age_cut/dt)
# drift_N_matrix = np.empty((N_1, n_scenarios, n_phi_short, Nc_cut))
drift_P_matrix = np.empty((N_1, n_scenarios, n_phi_short, Nc_cut))
diffusion_P_matrix = np.empty((N_1, n_scenarios, n_phi_short, Nc_cut))
drift_pi_matrix = np.empty((N_1, n_scenarios, n_phi_short, Nc_cut))
diffusion_pi_matrix = np.empty((N_1, n_scenarios, n_phi_short, Nc_cut))
r_matrix = np.empty((N_1, n_scenarios, n_phi_short))
dt_root = np.sqrt(dt)
for j in range(N_1):
    print(j)
    dZ = np.random.randn(Nt) * dt_root
    dZ_build = np.random.randn(Nc) * dt_root
    dZ_SI = np.random.randn(Nt) * dt_root
    dZ_SI_build = np.random.randn(Nc) * dt_root
    for k, scenario in enumerate(scenarios_short):
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        for l, phi_try in enumerate(phi_vector_short):
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
                popu_can_short,
                popu_short,
                Phi_can_short,
                Phi_short,
            ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta, phi_try,
                            Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                            need_f='False',
                            need_Delta='True',
                            need_pi='True',
                            top=0.05,
                            old_limit=100
                            )
            r_matrix[j, k, l] = np.average(r)
            theta_mat = np.transpose(np.tile(theta, (Nc_cut, 1)))
            r_mat = np.transpose(np.tile(r, (Nc_cut, 1)))
            # drift_N = -rho + r_mat
            drift_P = -rho + r_mat + 0.5 * theta_mat**2 - 0.5 * Delta[:, -Nc_cut:]**2
            diffusion_P = np.abs(theta_mat + Delta[:, -Nc_cut:])
            # drift_N_matrix[j, k, l] = np.average(drift_N, weights=(1-invest_tracker[:, Nc_cut]), axis=0)
            weights_condition = np.sum(invest_tracker[:, -Nc_cut:], axis=0)
            empty = np.where(weights_condition==0)
            if np.sum(empty) != 0:
                m = np.max(empty)
                cons_mean = np.empty(Nc_cut)
                cons_mean[:m+1] = np.nan
                cons_mean[m+1:] = np.average(drift_P[:, -Nc_cut+m+1:], weights=invest_tracker[:, -Nc_cut+m+1:], axis=0)
                cons_vola = np.empty(Nc_cut)
                cons_vola[:m+1] = np.nan
                cons_vola[m+1:] = np.average(diffusion_P[:, -Nc_cut+m+1:], weights=invest_tracker[:, -Nc_cut+m+1:], axis=0)
                pi_mean = np.empty(Nc_cut)
                pi_mean[:m+1] = np.nan
                pi_mean[m+1:] = np.average(pi[:, -Nc_cut+m+1:], weights=invest_tracker[:, -Nc_cut+m+1:], axis=0)
                pi_vola = np.empty(Nc_cut)
                pi_vola[:m+1] = np.nan
                pi_vola[m+1:] = np.sqrt(weighted_variance(pi[:, -Nc_cut+m+1:], invest_tracker[:, -Nc_cut+m+1:], ax=0))
            else:
                cons_mean = np.average(drift_P, weights=invest_tracker[:, -Nc_cut:], axis=0)
                cons_vola = np.average(diffusion_P, weights=invest_tracker[:, -Nc_cut:], axis=0)
                pi_mean = np.average(pi[:, -Nc_cut:], weights=invest_tracker[:, -Nc_cut:], axis=0)
                pi_vola = np.sqrt(weighted_variance(pi[:, -Nc_cut:], invest_tracker[:, -Nc_cut:], ax=0))
            drift_P_matrix[j, k, l] = cons_mean
            diffusion_P_matrix[j, k, l] = cons_vola
            drift_pi_matrix[j, k, l] = pi_mean
            diffusion_pi_matrix[j, k, l] = pi_vola
drift_P_vector = np.flip(np.nanmean(np.append(drift_P_matrix, np.load('drift_P_data.npy'), axis=0), axis=0), axis=2) # (n_scenarios, n_phi_short, Nc_cut)
diffusion_P_vector = np.flip(np.nanmean(np.append(diffusion_P_matrix, np.load('diffusion_P_data.npy'), axis=0), axis=0), axis=2)
drift_pi_vector = np.flip(np.nanmean(np.append(drift_pi_matrix, np.load('drift_pi_data.npy'), axis=0), axis=0), axis=2)
diffusion_pi_vector = np.flip(np.nanmean(np.append(diffusion_pi_matrix, np.load('diffusion_pi_data.npy'), axis=0), axis=0), axis=2)
r_vector = np.mean(np.append(r_matrix, np.load('drift_N_data.npy'), axis=0), axis=0)
# drift_P_vector = np.flip(np.nanmean(drift_P_matrix, axis=0), axis=2)  # (n_scenarios, n_phi_short, Nc_cut)
# diffusion_P_vector = np.flip(np.nanmean(diffusion_P_matrix, axis=0), axis=2)
# drift_pi_vector = np.flip(np.nanmean(drift_pi_matrix, axis=0), axis=2)  # (n_scenarios, n_phi_short, Nc_cut)
# diffusion_pi_vector = np.flip(np.nanmean(diffusion_pi_matrix, axis=0), axis=2)
# r_vector = np.nanmean(r_matrix, axis=0)  # (n_scenarios, n_phi_short)
# np.save('drift_P_data', drift_P_matrix)
# np.save('diffusion_P_data', diffusion_P_matrix)
# np.save('drift_pi_data', drift_pi_matrix)
# np.save('diffusion_pi_data', diffusion_pi_matrix)
# np.save('drift_N_data', r_matrix)

# Graph:
scenario_mat = [scenario_labels[1:3], scenario_labels[3:5]]
y_cases_mat = [[drift_P_vector, diffusion_P_vector], [drift_pi_vector, diffusion_pi_vector]]
fig_titles_mat = [['log consumption and age.png', 'pi and age.png'],
                  ['log consumption and age partial shorting.png', 'pi and age partial shorting.png']]
var_names = [r'log$\left(c_{s,t}\right)$', r'$\pi_{s,t}/W_{s,t}$']
for aa in range(2):
    scenario_vec = scenario_mat[aa]
    age_cut = 100 if aa == 0 else 120
    Nc_cut = int(age_cut / dt)
    x = t[:Nc_cut]
    count = np.arange(0, Nc_cut, 12, dtype=int)
    for bb in range(2):
        y_cases = y_cases_mat[bb]
        var_name = var_names[bb]
        fig_title = fig_titles_mat[aa][bb]
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='col', figsize=(10, 8))
        for j, ax_row in enumerate(axes):
            for k, ax in enumerate(ax_row):
                ax.set_xlabel('Age')
                ax.set_title(scenario_vec[j])
                y_case = y_cases[k]
                sce_index = j + 1 if aa == 0 else j + 3
                y_sce = y_case[sce_index]
                for i in range(n_phi_short):
                    y = y_sce[i, :Nc_cut]  # (Nc_cut)
                    X_Y_Spline = make_interp_spline(x[count], y[count])
                    X_ = np.linspace(x.min(), x.max(), 100)
                    Y_ = X_Y_Spline(X_)
                    if aa == 1:
                        ax.axvline(100, 0.02, 0.98, color='gray', linestyle='dotted',
                                   linewidth=0.8)
                    if k == 0:
                        if i == 0:
                            if bb == 0:
                                ax.axhline(r_vector[sce_index, i] - rho, 0.05, 0.95, color=colors_short[i], linestyle='dashed',
                                       linewidth=0.8, label='Nonparticipants')
                                ax.plot(X_, Y_, color=colors_short[i], linewidth=0.8, label='Participants')
                                if j == 0:
                                    ax.legend(loc='upper right')
                            else:
                                ax.plot(X_, Y_, color=colors_short[i], linewidth=0.8)
                        else:
                            ax.plot(X_, Y_, color=colors_short[i], linewidth=0.8)
                            if bb == 0:
                                ax.axhline(r_vector[sce_index, i] - rho, 0.05, 0.95, color=colors_short[i], linestyle='dashed',
                                       linewidth=0.8)
                    if k == 1:
                        ax.plot(X_, Y_, color=colors_short[i], linewidth=0.8, label=label_phi[i])
                        if j == 0:
                            ax.legend(loc='upper right')
                if k == 0:
                    ax.set_ylabel('Average drift of ' + var_name, color='black')
                else:
                    ax.set_ylabel('Average volatility of ' + var_name, color='black')
                ax.tick_params(axis='y', labelcolor='black')
        fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
        plt.savefig(fig_title, dpi=60)
        plt.show()
        # plt.close()




# ######################################
# ############# Figure 4.1 #############
# ######################################
# time-series of interest rate and market price of risk, bad z^Y, bad z&SI
# 4.1.1 interest rate across different phi values, reentry
# 4.1.2 interest rate across different scenarios, phi = 0
# 4.1.3 market price of risk across different phi values, reentry
red_case = 1
yellow_case = 1
r_mat = r_compare[:, red_case, yellow_case]  # n_scenarios, n_phi_short, Nt
theta_mat = theta_compare[:, red_case, yellow_case]
y1 = r_mat[:,0]  # n_scenarios, Nt
y2 = r_mat[1]  # n_phi_short, Nt
y3 = theta_mat[1]  # n_phi_short, Nt
y_list = [y1, y2, y3]
Z = np.cumsum(Z_Y_cases[red_case])
Z_SI = np.cumsum(Z_SI_cases[yellow_case])
n_lines = [n_scenarios, n_phi_short, n_phi_short]
y_title_list = [r'Interest rate $r_t$, $\phi=0$', r'Interest rate $r_t$, Reentry', r'Market price of risk $\theta_t$, Reentry']
labels = [scenario_labels, label_phi, label_phi]
fig, axes = plt.subplots(nrows=3, ncols=1, sharex='all', figsize=(15, 15))
for j, ax in enumerate(axes):
    ax.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
    ax.plot(t, Z, color='red', linewidth=0.5, label=r'$z^Y_t$')
    ax.plot(t, Z_SI, color='gold', linewidth=0.5, label=r'$z^{SI}_t$')
    ax.tick_params(axis='y', labelcolor='black')
    if j == 2:
        ax.set_xlabel('Time in simulation')

    y_vec = y_list[j]  # n_phi_short, Nt
    y_title = y_title_list[j]
    ax2 = ax.twinx()
    ax2.set_ylabel(y_title, color='black')
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
plt.savefig('r and theta,' + str(red_case) + str(yellow_case)  +'.png', dpi=60)
plt.show()
plt.close()

############ IA other cases
red_cases = [0, 0, 1]
yellow_cases = [0, 1, 0]
phi_index = 0
scenario_index = 1
y_title_list = [r'Interest rate $r_t$, $\phi=0$', r'Interest rate $r_t$, Reentry']
fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', sharey='all', figsize=(20, 15))
for i, ax_row in enumerate(axes):
    red_case = red_cases[i]
    yellow_case = yellow_cases[i]
    r_mat = r_compare[:, red_case, yellow_case]  # n_scenarios, n_phi_short, Nt
    for j, ax in enumerate(ax_row):
        y_mat = r_mat[:, phi_index] if j == 0 else r_mat[scenario_index]  # n_scenarios, Nt
        labels = scenario_labels if j == 0 else label_phi
        y_title = y_title_list[j]
        Z = np.cumsum(Z_Y_cases[red_case])
        Z_SI = np.cumsum(Z_SI_cases[yellow_case])
        ax2 = ax.twinx()
        ax2.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
        ax2.plot(t, Z, color='red', linewidth=0.4, label=r'$z^Y_t$')
        ax2.plot(t, Z_SI, color='gold', linewidth=0.4, label=r'$z^{SI}_t$')
        ax2.tick_params(axis='y', labelcolor='black')
        if i == 2:
            ax.set_xlabel('Time in simulation')
        for k in range(3):
            y = y_mat[k]
            color_i = colors_short2[k] if j == 0 else colors_short[k]
            ax.plot(t, y, label=labels[k], color=color_i, linewidth=0.6)
            y_title = y_title_list[j]
        if i == 0:
            ax2.legend(loc='upper right')
            ax.legend(loc='upper left')
        ax.set_ylabel(r'Interest rate $r_t$')
        ax.set_title(y_title)
fig.tight_layout(h_pad=2)
plt.savefig('IA r and theta.png', dpi=60)
plt.show()
plt.close()


# ######################################
# ############# Figure 4.2 #############
# ######################################
# time-series of Delta_bar, Phi, and participation rate, bad z^Y, bad z&SI
# 4.2.1 Delta_bar (reentry + complete + disappointment) * (phi = 0, 0.8)
# 4.2.2 Phi (reentry + disappointment) * (phi = 0, 0.8), also mark 1 for complete case
# 4.2.3 participation rate (reentry + disappointment) * (phi = 0, 0.8)
red_case = 1
yellow_case = 1
titles_subfig = [r'Wealth weighted average estimation error conditional on participation $\bar{\Delta}_t$', r'Wealth share of participants $\Phi_t$', 'Participation rate']
yaxis_subfig = [r'$\bar{\Delta}_t$', r'$\Phi_t$', 'Participation rate']
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
axes[0,0].set_title('Shocks to output and signal')
axes[0,0].set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
axes[0,0].plot(x, Z, color='red', linewidth=0.5, linestyle = 'solid', label=r'$z^Y_t$')
axes[0,0].plot(x, Z_SI, color='gold', linewidth=0.5, linestyle = 'solid', label=r'$z^{SI}_t$')
axes[0,0].tick_params(axis='y', labelcolor='black')
axes[0,0].legend(loc='upper left')

axes[0,1].set_title(titles_subfig[0])
axes[0,1].set_ylabel(yaxis_subfig[0], color='black')
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
axes[1, 0].set_ylabel(yaxis_subfig[1], color='black')
# ax2 = axes[1].twinx()
for i in range(1,3):
    for j in range(2):
        y2 = y2_case[i, phi_indexes[j], int(left_t/dt):int(right_t/dt)]
        line_style = 'dotted' if j==0 else 'solid'
        if i == 1 and j == 0:
            axes[1, 0].plot(x, y2, label = r'$\phi=0.0$', linewidth=0.5, color=colors_short[i], linestyle=line_style)
        elif i == 1 and j == 1:
            axes[1, 0].plot(x, y2, label = r'$\phi=0.8$', linewidth=0.5, color=colors_short[i], linestyle=line_style)
        else:
            axes[1, 0].plot(x, y2, linewidth=0.5, color=colors_short[i], linestyle=line_style)
y21 = y2_case[0, 0, int(left_t/dt):int(right_t/dt)]
# axes[1, 0].plot(x, y21, label = scenario_labels[0], linewidth=0.5, color=colors_short[0], linestyle='solid')
axes[1,0].legend(loc='upper right')
axes[1,0].set_xlabel('Time in simulation')

axes[1,1].set_title(titles_subfig[2])
axes[1,1].set_ylabel(yaxis_subfig[2], color='black')
# ax3 = axes[2].twinx()
for i in range(1,3):
    for j in range(2):
        y3 = y3_case[i, phi_indexes[j], int(left_t/dt):int(right_t/dt)]
        line_style = 'dotted' if j==0 else 'solid'
        label_i = scenario_labels[i]
        axes[1,1].plot(x, y3, linewidth=0.5, color=colors_short[i], linestyle=line_style)
# y21 = y2_case[0, 0]
# ax3.plot(t, y1, label = label_scenarios[0], color=colors_short[0], linestyle='dashed')
#axes[1,1].legend(loc='upper right')
axes[1,1].set_xlabel('Time in simulation')
fig.tight_layout(h_pad=2)
plt.savefig('Delta bar, Phi and parti rate,' + str(red_case) + str(yellow_case)  + '.png', dpi=60)
plt.show()
# plt.close()

############ IA other cases

titles_subfig = [r'Participants wealth weighted average estimation error $\bar{\Delta}_t$', r'Wealth share of participants $\Phi_t$', 'Participation rate']
scenario_indexes = [1,2]
yaxis_subfig = [r'$\bar{\Delta}_t$', r'$\Phi_t$', 'Participation rate']
phi_indexes = [0, 2]
red_cases = [0, 0, 1]
yellow_cases = [0, 1, 0]
var_list = [delta_bar_compare, Phi_compare, popu_parti_compare]
left_t = 300
right_t = 400
x = t[int(left_t/dt):int(right_t/dt)]

fig, axes = plt.subplots(nrows=3, ncols=3, sharex='all', sharey='col', figsize=(15, 15))
for i, ax_row in enumerate(axes):
    red_case = red_cases[i]
    yellow_case = yellow_cases[i]
    Z = np.cumsum(Z_Y_cases[red_case])[int(left_t / dt):int(right_t / dt)]
    Z_SI = np.cumsum(Z_SI_cases[yellow_case])[int(left_t / dt):int(right_t / dt)]
    for j, ax in enumerate(ax_row):
        title_subfig = titles_subfig[j]
        var = var_list[j][:, red_case, yellow_case]  # n_sce * n_phi * nt
        ax.set_ylabel(yaxis_subfig[j], color='black')
        for scenario_index in scenario_indexes:
            color_use = colors_short[scenario_index]
            for phi_index in phi_indexes:
                y = var[scenario_index, phi_index, int(left_t/dt):int(right_t/dt)]
                line_style = 'dotted' if phi_index == 0 else 'solid'
                labels = scenario_labels if j == 0 else label_phi
                if i == j == 0 and phi_index != 0:
                    ax.plot(x, y, label=labels[scenario_index], linewidth=0.5,
                            color=colors_short[scenario_index], linestyle=line_style)
                    ax.legend(loc='upper right')
                if i == 0 and j == 1 and scenario_index == 1:
                    ax.plot(x, y, label=labels[phi_index], linewidth=0.5,
                            color=colors_short[scenario_index], linestyle=line_style)
                    ax.legend(loc='upper right')
                else:
                    ax.plot(x, y, linewidth=0.5, color=colors_short[scenario_index],
                            linestyle=line_style)
        # if j == 0:
        #     ax1 = ax.twinx()
        #     ax1.plot(x, Z, color='red', linewidth=0.5, linestyle='solid', label=r'$z^Y_t$')
        #     ax1.plot(x, Z_SI, color='gold', linewidth=0.5, linestyle='solid', label=r'$z^{SI}_t$')
        #     ax1.legend(loc='lower left')
        if i == 0:
            ax.set_title(title_subfig)

        if i == 2:
            ax.set_xlabel('Time in simulation')
fig.tight_layout(h_pad=2)
plt.savefig('IA Delta bar, Phi and parti rate.png', dpi=100)
plt.show()
# plt.close()

# ######################################
# ############ Figure 4-3 ##############
# ######################################
# portfolio in time series
# bad & Bad, phi = 0.4
# portfolio, different phi
# portfolio, different scenarios
cohort_index = 2
left_t = (cohort_index + 1) * 100
right_t = (cohort_index + 2) * 100
red_case = 1
yellow_case = 1
phi_index = 1
scenario_index = 1
Z = np.cumsum(Z_Y_cases[red_case])[int(left_t/dt):int(right_t/dt)]
Z_SI = np.cumsum(Z_SI_cases[yellow_case])[int(left_t/dt):int(right_t/dt)]
x = t[int(left_t/dt):int(right_t/dt)]
y1_pi = pi_time_series[:, red_case, yellow_case, phi_index, cohort_index]  # (n_scenarios, 2, 2, n_phi_short, nn, length) -> (n_scenarios, length)
y2_pi = pi_time_series[scenario_index, red_case, yellow_case, :, cohort_index]  # (n_scenarios, 2, 2, n_phi_short, nn, length) -> (n_phi_short, length)
y1_belief = (Delta_time_series[:, red_case, yellow_case, phi_index, cohort_index] -
             delta_bar_compare[:, red_case, yellow_case, phi_index])/ sigma_Y
y2_belief = (Delta_time_series[scenario_index, red_case, yellow_case, :, cohort_index] -
             delta_bar_compare[scenario_index, red_case, yellow_case, :])/ sigma_Y
y1_wealth = 1 / Phi_compare[:, red_case, yellow_case, phi_index]
y2_wealth = 1 / Phi_compare[scenario_index, red_case, yellow_case]
y_cases = [y1_pi, y1_belief, y1_wealth,
           y2_pi, y2_belief, y2_wealth]
# y_cases = [y1_pi, y2_pi, y1_belief, y1_wealth]
titles_subfig = [r'Portfolios, across scenarios, $\phi=0.4$', r'Belief component, across scenarios, $\phi=0.4$',
                 r'Wealth component, across scenarios, $\phi=0.4$',
                 r'Portfolios, reentry, across values of $\phi$', r'Belief component, reentry, across values of $\phi$',
                 r'Wealth component, reentry, across values of $\phi$']
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
for i, ax in enumerate(axes.flat):
    n_loop = n_phi_short if i >= 3 else n_scenarios
    labels = label_phi if i >= 3 else scenario_labels
    for j in range(n_loop):
        label_i = labels[j]
        if i == 2 and j == 0:
            length = int(right_t/dt) - int(left_t/dt)
            y_case = np.ones(length)
            ax.plot(x, y_case, label=label_i, color=colors_short[j], linewidth=0.4)
        else:
            y_case = y_cases[i][j, int(left_t/dt):int(right_t/dt)]
            labels = label_phi if i > 2 else scenario_labels
            ax.plot(x, y_case, label=label_i, color=colors_short[j], linewidth=0.4)
    if i == 1 or i == 4:
        if i == 1:
            ax.set_ylim(-6, 3)
        else:
            ax.set_ylim(-7, 4)
        ax.set_xlabel('Time in simulation')
        ax.set_ylabel(r'Investment in stock market, Belief component')
    else:
        ax.set_ylim(-1, 7)
    if i == 0 or i == 3:
        ax.legend(loc='upper left')
        ax.set_ylabel(r'Investment in stock market, $\pi_{s,t}/W_{s,t}$')
    if i == 2 or i == 5:
        ax.set_ylabel(r'Investment in stock market, Wealth component')
# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
# for i, ax in enumerate(axes.flat):
#     n_loop = n_phi_short if i == 0 else n_scenarios
#     for j in range(n_loop):
#         y_case = y_cases[i][j, int(left_t/dt):int(right_t/dt)]
#         labels = label_phi if i == 1 else scenario_labels
#         label_i = labels[j]
#         ax.plot(x, y_case, label=label_i, color=colors_short[j], linewidth=0.4)
#     ax.legend()
#     if i == 2:
#         ax.set_ylim(-6, 3)
#     else:
#         ax.set_ylim(-1, 7)
#     if i == 0 or i == 2:
#         ax.set_ylabel(r'Investment in stock market, $\pi_{s,t}/W_{s,t}$')
#     if i == 2 or i == 3:
#         ax.set_xlabel('Time in simulation')
    ax.set_title(titles_subfig[i])
    ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout(h_pad=2)
plt.savefig('Shocks and Portfolio,' + str(red_case) + str(yellow_case)  + '.png', dpi=60)
plt.show()
plt.close()

############ IA other cases

titles_subfig = [r'Portfolios, across scenarios', r'Belief component, across scenarios, $\phi=0.4$',
                 r'Wealth component, across scenarios, $\phi=0.4$']
yaxis_subfig = r'Investment in stock market, $\pi_{s,t}/W_{s,t}$'
phi_fix = 1
cohort_index = 2
left_t = (cohort_index + 1) * 100
right_t = (cohort_index + 2) * 100
red_cases = [0, 0, 1]
yellow_cases = [0, 1, 0]
scenario_indexes = [0, 1, 2]
phi_indexes = [0, 2]
pi_compare_cohort = pi_time_series[:, :, :, :, cohort_index]
Delta_compo_cohort = (Delta_time_series[:, :, :, :, cohort_index] - delta_bar_compare) / sigma_Y
var_list = [pi_compare_cohort, Delta_compo_cohort, 1 / Phi_compare]
left_t = 300
right_t = 400
x = t[int(left_t/dt):int(right_t/dt)]

fig, axes = plt.subplots(nrows=3, ncols=3, sharex='all', figsize=(15, 15))
for i, ax_row in enumerate(axes):
    red_case = red_cases[i]
    yellow_case = yellow_cases[i]
    for j, ax in enumerate(ax_row):
        title_subfig = titles_subfig[j]
        var = var_list[j][:, red_case, yellow_case]
        for scenario_index in scenario_indexes:
            color_use = colors_short[scenario_index]
            y = var[scenario_index, phi_fix, int(left_t/dt):int(right_t/dt)]
            labels = scenario_labels
            ax.plot(x, y, label=labels[scenario_index]+ ', ' + label_phi[phi_fix], linewidth=0.5,
                    color=colors_short[scenario_index], linestyle='solid')
            if j == 0:
                ax.set_ylabel(yaxis_subfig, color='black')
                if scenario_index == 1:
                    for phi_index in phi_indexes:
                        y0 = var[scenario_index, phi_index, int(left_t / dt):int(right_t / dt)]
                        line_style = 'dotted' if phi_index == 0 else 'dashed'
                        ax.plot(x, y0, label=labels[scenario_index] + ', ' + label_phi[phi_index], linewidth=0.5,
                                color=colors_short[scenario_index], linestyle='dotted')
                if i == 0:
                    ax.legend(loc='upper right')
        if i == 0:
            ax.set_title(title_subfig)
        if i == 2:
            ax.set_xlabel('Time in simulation')
fig.tight_layout(h_pad=2)
plt.savefig('IA Shocks and Portfolio.png', dpi=100)
plt.show()
plt.close()


# ######################################
# ############ Figure 4-4 ##############
# ######################################
# consumption, different tax rates
# consumption, different scenarios
# build consumption data and time series over tau values
tax_vector = [0.008, 0.010, 0.012]
n_tax = len(tax_vector)
dZ = Z_Y_cases[1]
dZ_SI = Z_SI_cases[1]
n_scenarios =  1
n_phi_short = 1
cohort_index = 2
left_t = int(starts[cohort_index]/dt)
right_t = int(left_t + 100/dt)
scenario_index = 1
phi_index = 0
line_styles = [(0, (5, 10)), 'solid', (0, (1, 1))]
titles_subfig = [r'Individual consumption share $c_{s,t}/Y_t$', r'Cohort consumption share $f_{s,t}$']
titles_subfig_IA = [r'Individual consumption share', r'Cohort consumption share']
yaxis_subfig = [r'$c_{s,t}/Y_t$', r'$f_{s,t}$']
cases = [0, 1]

for case_dzY in cases:
    for case_dzSI in cases:
        dZ_build = dZ_build_matrix[0]
        dZ_SI_build = dZ_SI_build_matrix[0]
        dZ = Z_Y_cases[case_dzY]  # bad
        dZ_SI = Z_SI_cases[case_dzSI]  # bad
        cons_compare_tau = np.zeros((n_scenarios, n_tax, n_phi_short, Nt, Nc))
        f_compare_tau = np.zeros((n_scenarios, n_tax, n_phi_short, Nt, Nc))
        pi_compare_tau = np.zeros((n_scenarios, n_tax, n_phi_short, Nt, Nc))
        for g, scenario in enumerate([scenarios_short[scenario_index]]):
            mode_trade = scenario[0]
            mode_learn = scenario[1]
            for k, tax_try in enumerate(tax_vector):
                beta_try = rho + nu - tax_try
                for l, phi_try in enumerate([phi_vector_short[phi_index]]):
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
                        invest_tracker,
                        popu_can_short,
                        popu_short,
                        Phi_can_short,
                        Phi_short,
                    ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S,
                                    tax_try,
                                    beta_try,
                                    phi_try,
                                    Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                                    need_f='True',
                                    need_Delta='False',
                                    need_pi='True',
                                    top=0.05,
                                    old_limit=100
                                    )
                    # invest_tracker = pi > 0
                    cons_compare_tau[g, k, l] = f * dt / cohort_size_mat
                    f_compare_tau[g, k, l] = f
                    pi_compare_tau[g, k, l] = pi

        length = len(t)
        pi_time_series_tau = np.zeros((n_scenarios, n_tax, n_phi_short, nn, length))
        cons_time_series_tau = np.zeros((n_scenarios, n_tax, n_phi_short, nn, length))
        f_time_series_tau = np.zeros((n_scenarios, n_tax, n_phi_short, nn, length))
        switch_time_series_tau = np.zeros((n_scenarios, n_tax, n_phi_short, nn, length))
        parti_time_series_tau = np.zeros((n_scenarios, n_tax, n_phi_short, nn, length))
        for o in range(n_scenarios):
            for i in range(n_tax):
                for l in range(n_phi_short):
                    pi = pi_compare_tau[o, i, l]
                    cons = cons_compare_tau[o, i, l]
                    f = f_compare_tau[o, i, l]
                    for m in range(nn):
                        start = int((m + 1) * 100 * (1 / dt))
                        starts[m] = start * dt
                        for n in range(length):
                            if n < start:
                                pi_time_series_tau[o, i, l, m, n] = np.nan
                                cons_time_series_tau[o, i, l, m, n] = np.nan
                                f_time_series_tau[o, i, l, m, n] = np.nan
                            else:
                                cohort_rank = length - (n - start) - 1
                                pi_time_series_tau[o, i, l, m, n] = pi[n, cohort_rank]
                                cons_time_series_tau[o, i, l, m, n] = cons[n, cohort_rank]
                                f_time_series_tau[o, i, l, m, n] = f[n, cohort_rank]
                        parti = pi_time_series_tau[o, i, l, m] > 0
                        switch = abs(parti[1:] ^ parti[:-1])
                        sw = np.append(switch, 0)
                        parti = np.where(sw == 1, 0.5, parti)
                        switch = np.insert(switch, 0, 0)
                        parti = np.where(switch == 1, 0.5, parti)
                        parti_time_series_tau[o, i, l, m] = parti
                        switch_time_series_tau[o, i, l, m] = sw

        fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
        for i, ax in enumerate(axes):
            if case_dzY == case_dzSI == scenario_index == 1:
                ax.set_title(titles_subfig[i] + ', ' + scenario_labels[scenario_index]+ ', ' + r'$\phi=0.0$')
            else:
                ax.set_title(titles_subfig_IA[i] + ', '  + red_labels[case_dzY] + scenario_labels[scenario_index]+ ', ' + r'$\phi=0.0$')

            var = cons_time_series_tau if i == 0 else f_time_series_tau
            for k in range(n_tax):
                y = var[0, k, 0,
                    cohort_index,
                    left_t:right_t]  # n_scenarios, 2, 2, n_tax, n_phi_short, nn, length
                # condition = pi_time_series_tau[scenario_index, red_index, yellow_index, k, 2, phi_index, int(left_t/dt):int(right_t/dt)]
                switch = switch_time_series_tau[0, k, 0,
                         cohort_index,
                         left_t:right_t]
                y_switch = np.ma.masked_where(switch == 0, y)
                # y_N = np.ma.masked_where(condition >= 0.8, y)
                # y_P = np.ma.masked_where(condition <= 0.2, y)
                label_i = r'$\tau$ = ' + str('{0:.3f}'.format(tax_vector[k]))
                if k == 1:
                    ax.plot(t[left_t:right_t], y, color='black', linewidth=0.6, linestyle=line_styles[k], label=label_i)
                else:
                    ax.plot(t[left_t:right_t], y, color='black', linewidth=0.4, linestyle=line_styles[k], label=label_i)
                if scenario_index != 0:
                    if k == 0:
                        ax.scatter(t[left_t:right_t], y_switch, color='red', s=10, marker='o', label='Switch')
                    else:
                        ax.scatter(t[left_t:right_t], y_switch, color='red', s=10, marker='o')
            ax.tick_params(axis='both', labelcolor='black')
            ax.set_ylabel(yaxis_subfig[i], color='black')
            # ax.set_xlim(left_t, right_t)
            if i == 0:
                ax.legend()
            ax.set_xlabel('Time in simulation')
        fig.tight_layout(h_pad=2)
        if case_dzY == case_dzSI == scenario_index == 1:
            plt.savefig('Consumption share tau.png', dpi=100)
        else:
            plt.savefig('IA Consumption share tau' + str(case_dzY) + str(case_dzSI) + str(scenario_index) + '.png', dpi=100)
        plt.show()
        plt.close()

### average across paths:
N_1 = 200
tax_vector = [0.008, 0.010, 0.012]
n_tax = len(tax_vector)
f_matrix = np.empty((N_1, 3, n_tax, Nc))
dt_root = np.sqrt(dt)
phi_try = 0.4

for j in range(N_1):
    print(j)
    dZ = np.random.randn(Nt) * dt_root
    dZ_build = np.random.randn(Nc) * dt_root
    dZ_SI = np.random.randn(Nt) * dt_root
    dZ_SI_build = np.random.randn(Nc) * dt_root
    for l in range(3):
        scenario = scenarios_short[l]
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        for m, tax_try in enumerate(tax_vector):
            beta_try = rho + nu - tax_try
            (
                r_reentry,
                theta_reentry,
                f_reentry,
                Delta_reentry,
                pi_reentry,
                popu_parti_reentry,
                f_parti_reentry,
                Delta_bar_parti_reentry,
                dR_reentry,
                invest_tracker_reentry,
                popu_can_short_reentry,
                popu_short_reentry,
                Phi_can_short_reentry,
                Phi_short_reentry,
            ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S,
                            tax_try,
                            beta_try,
                            phi_try,
                            Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                            need_f='True',
                            need_Delta='False',
                            need_pi='False',
                            top=0.05,
                            old_limit=100
                            )
            f_matrix[j, l, m] = np.average(f_reentry, axis=0)
c_vector = np.flip(np.average(f_matrix * dt / cohort_size, axis=0), axis=2)
f_vector = np.flip(np.average(f_matrix, axis=0), axis=2)

age_cut = 100
N_cut = int(age_cut / dt)
titles_subfig = [r'Average individual consumption share $c_{s,t}/Y_t$', r'Average cohort consumption share $f_{s,t}$']
yaxis_subfig = [r'$Average c_{s,t}/Y_t$', r'$Average f_{s,t}$']
line_styles = [(0, (5, 10)), 'solid', (0, (1, 1))]

fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
for i, ax in enumerate(axes):
    ax.set_title(titles_subfig[i]+ ', ' + r'$\phi=0.4$')
    var = c_vector if i == 0 else f_vector
    for k in range(n_tax):
        y = var[1, k,
            :N_cut]  # n_scenarios, 2, 2, n_tax, n_phi_short, nn, length
        label_i = r'$\tau$ = ' + str('{0:.3f}'.format(tax_vector[k]))
        if k == 1:
            ax.plot(t[:N_cut], y, color=colors_short[1], linewidth=0.8, linestyle=line_styles[k], label='Reentry, '+label_i)
            y_complete = var[0, k, :N_cut]
            y_disappintment = var[2, k, :N_cut]
            ax.plot(t[:N_cut], y_complete, color=colors_short[0], linewidth=0.6, linestyle=line_styles[k], label='Complete, '+label_i)
            ax.plot(t[:N_cut], y_disappintment, color=colors_short[2], linewidth=0.6, linestyle=line_styles[k], label='Disappointment, '+label_i)
        else:
            ax.plot(t[:N_cut], y, color=colors_short[1], linewidth=0.8, linestyle=line_styles[k], label='Reentry, '+label_i)
    ax.tick_params(axis='both', labelcolor='black')
    ax.set_ylabel(yaxis_subfig[i], color='black')
    # ax.set_xlim(left_t, right_t)
    if i == 0:
        ax.legend()
    ax.set_xlabel('Age')
fig.tight_layout(h_pad=2)
plt.savefig('Average consumption share tau.png', dpi=100)
plt.show()
plt.close()

# ######################################
# ########### ACROSS PATHS #############
# ############ GRAPH ONE ###############
# ######################################
# mean:
# N, n_scenarios, n_phi, Nt
var_list = [r_matrix, theta_matrix, delta_bar_matrix, Phi_parti_1_matrix, popu_parti_matrix]
var_name_list = [r'$r_t$', r'$\theta_t$',
                 r'$\bar{\Delta}_t$', r'$\frac{1}{\Phi_t}$',
                 'participation rate', r'$var(\Delta_{s,t})$']
# var_list = [r_matrix, theta_matrix, delta_bar_matrix, Phi_parti_1_matrix, popu_parti_matrix, belief_variance_matrix, belief_skew_matrix]
# var_name_list = ['interest rate', 'market price of risk',
#                  'consumption-weighted estimation error of participants', 'consumption share of participants',
#                  'participation rate', 'belief variance', 'belief skewness']
# age_labels = ['20 < Age <= 35, youngest quartile', '35 < Age <= 55', '55 < Age <= 89', 'Age > 89, oldest quartile']
x = phi_vector
# fig, axs = plt.subplots(nrows=3, ncols=2, sharex='all', figsize=(15, 20))
# for j, ax in enumerate(axs.flat):
#     y = var_list[j]
#     y_mean_mat = np.mean(y, axis=3)
#     y_mean = np.mean(y_mean_mat, axis=0)  # n_scenarios * n_phi
#     ax.set_title(var_name_list[j])
#     for k in range(n_scenarios):
#         y_sce = y_mean[k]
#         label_i = scenario_labels[k]
#         ax.plot(phi_vector, y_sce, color=colors_short[k], label=label_i)
#     if j == 0:
#         ax.legend()
#     if j == 0 or j == 2 or j == 4:
#         ax.set_ylabel('mean level')
#     if j == 4 or j == 5:
#         ax.set_xlabel(r'$\phi$ values')
# fig.tight_layout()
# # plt.suptitle('Variance decomposition, market price of risk', fontsize=16)
# # fig.supxlabel('phi')
# # fig.supylabel('variance')
# plt.savefig('mean values.png', dpi=500, format="png")
# plt.show()
# #plt.close()

for i, var in enumerate(var_list):
    var_name = var_name_list[i]
    fig, ax = plt.subplots(figsize=(5, 5))
    y_mean_mat = np.mean(var, axis=3)
    y_mean = np.mean(y_mean_mat, axis=0)  # n_scenarios * n_phi
    ax.set_title(var_name)
    for k in range(n_scenarios):
        y_sce = y_mean[k]
        label_i = scenario_labels[k]
        X_Y_Spline = make_interp_spline(phi_vector, y_sce)
        # Returns evenly spaced numbers
        # over a specified interval.
        X_ = np.linspace(phi_vector.min(), phi_vector.max(), 100)
        Y_ = X_Y_Spline(X_)
        ax.plot(X_, Y_, color=colors_short[k], label=label_i)
    ax.legend()
    ax.set_ylabel('mean level')
    ax.set_xlabel(r'$\phi$ values')
    fig.tight_layout()
    plt.savefig(str(i) +' smooth.png', dpi=500, format="png")
    plt.show()
    # plt.close()


# decomposition of variance for theta:
colors_short = ['midnightblue', 'darkgreen', 'darkviolet']
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
titles_subfig = [r'$\theta$', r'$-\bar{\Delta}$', r' $\sigma_Y\frac{1}{\Phi}$', r'$cov(-\bar{\Delta}, \sigma_Y\frac{1}{\Phi})$']
y_list = [theta_vola, delta_bar_vola, Phi_parti_1_vola, cov_Phiparti_deltabar]
fig, axs = plt.subplots(nrows=2, ncols=2, sharey='all', sharex='all', figsize=(10, 10))
for j, ax in enumerate(axs.flat):
    y = y_list[j]
    ax.set_title(titles_subfig[j])
    for k in range(n_scenarios):
        y_sce = y[k]
        label_i = scenario_labels[k]
        ax.plot(phi_vector, y_sce, color=colors_short[k], label=label_i)
    if j == 0:
        ax.legend()
    if j == 0 or j == 2:
        ax.set_ylabel('Variance')
    if j == 2 or j == 3:
        ax.set_xlabel(r'$\phi$ values')
fig.tight_layout()
# plt.suptitle('Variance decomposition, market price of risk', fontsize=16)
# fig.supxlabel('phi')
# fig.supylabel('variance')
plt.savefig('Variance decomposition, market price of risk.png', dpi=500, format="png")
plt.show()
plt.close()


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

# marginal_belief = (-theta_disappointment) * sigma_Y + mu_Y


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