import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from typing import Callable, Tuple
from src.simulation import simulate_SI, simulate_SI_mean_vola
from src.cohort_builder import build_cohorts_SI
from src.cohort_simulator import simulate_cohorts_SI
from src.param import rho, nu, mu_Y, sigma_Y, sigma_Y_sqr, sigma_S, v, tax, \
    beta, dt, T_hat, Npre, Vhat, Ninit, T_cohort, Nt, Nc, tau, cohort_size, \
    n_age_groups, cutoffs, colors, modes_trade, modes_learn,\
    scenarios, dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    Z_Y_cases, Z_SI_cases, t, red_cases, yellow_cases, cohort_labels, \
    scenario_labels, colors_short, colors_short2, PN_labels, age_labels, \
    top, old_limit, Npres
from src.stats import shocks, tau_calculator, good_times, Delta_st_compare
from numba import jit
import matplotlib.pyplot as plt
import statsmodels.api as sm
# import tabulate as tabulate
from scipy.interpolate import make_interp_spline

n_scenarios = 5
scenarios_all = scenarios[:n_scenarios]

n_scenarios_short = 2
scenarios_short = scenarios[3:3+n_scenarios_short]

phi_vector = np.arange(0,1,0.1)
n_phi = len(phi_vector)

phi_indexes = [0, 4, 8]
n_phi_short = len(phi_indexes)
phi_vector_short = phi_vector[phi_indexes]

# Individual paths as examples:
dZ_build = dZ_build_matrix[0]
dZ_SI_build = dZ_SI_build_matrix[0]  # fix the shocks at the buildup stage
theta_compare = np.empty((n_scenarios, 2, 2, n_phi_short, Nt))
popu_parti_compare = np.empty((n_scenarios, 2, 2, n_phi_short, Nt))
r_compare = np.zeros((n_scenarios, 2, 2, n_phi_short, Nt))
delta_bar_compare = np.zeros((n_scenarios, 2, 2, n_phi_short, Nt))
Phi_compare = np.zeros((n_scenarios, 2, 2, n_phi_short, Nt))
Delta_compare = np.empty((n_scenarios, 2, 2, n_phi_short, Nt, Nc))
pi_compare = np.empty((n_scenarios, 2, 2, n_phi_short, Nt, Nc))
cons_compare = np.zeros((n_scenarios, 2, 2, n_phi_short, Nt, Nc))
invest_tracker_compare = np.zeros((n_scenarios, 2, 2, n_phi_short, Nt, Nc))
popu_can_short_compare = np.empty((n_scenarios, 2, 2, n_phi_short, Nt))
popu_short_compare = np.empty((n_scenarios, 2, 2, n_phi_short, Nt))
Phi_can_short_compare = np.empty((n_scenarios, 2, 2, n_phi_short, Nt))
Phi_short_compare = np.empty((n_scenarios, 2, 2, n_phi_short, Nt))
cohort_size_mat = np.tile(cohort_size, (Nc, 1))
for g, scenario in enumerate(scenarios_all):
    mode_trade = scenario[0]
    mode_learn = scenario[1]
    for i in range(2):
        dZ = Z_Y_cases[i]
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
                    invest_tracker,
                    popu_can_short,
                    popu_short,
                    Phi_can_short,
                    Phi_short,
                ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta,
                                phi,
                                Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
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
                delta_bar_compare[g, i, j, k] = Delta_bar_parti
                Phi_compare[g, i, j, k] = Phi_parti
                cons_compare[g, i, j, k] = f / cohort_size_mat
                invest_tracker_compare[g, i, j, k] = invest_tracker
                popu_can_short_compare[g, i, j, k] = popu_can_short
                popu_short_compare[g, i, j, k] = popu_short
                Phi_can_short_compare[g, i, j, k] = Phi_can_short
                Phi_short_compare[g, i, j, k] = Phi_short

# cohort_matrix_list = [pi_compare, Delta_compare, cons_compare]
nn = 3  # number of cohorts illustrated
length = len(t)
starts = np.zeros(nn)
Delta_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
# invest_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
pi_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
cons_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
parti_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
short_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
switch_parti_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
switch_short_time_series = np.zeros((n_scenarios, 2, 2, n_phi_short, nn, length))
for o in range(n_scenarios):
    for i in range(2):
        for j in range(2):
            for l in range(n_phi_short):
                pi = pi_compare[o, i, j, l]
                cons = cons_compare[o, i, j, l]
                Delta = Delta_compare[o, i, j, l]
                invest = invest_tracker_compare[o, i, j, l]
                for m in range(nn):
                    start = int((m + 1) * 100 * (1 / dt))
                    starts[m] = start * dt
                    for n in range(length):
                        if n < start:
                            Delta_time_series[o, i, j, l, m, n] = np.nan
                            pi_time_series[o, i, j, l, m, n] = np.nan
                            cons_time_series[o, i, j, l, m, n] = np.nan
                            # invest_time_series[o, i, j, l, m, n] = np.nan
                        else:
                            cohort_rank = length - (n - start) - 1
                            Delta_time_series[o, i, j, l, m, n] = Delta[n, cohort_rank]
                            pi_time_series[o, i, j, l, m, n] = pi[n, cohort_rank]
                            cons_time_series[o, i, j, l, m, n] = cons[n, cohort_rank]
                            # invest_time_series[o, i, j, l, m, n] = invest[n, cohort_rank]
                    long = pi_time_series[o, i, j, l, m] > 0.000001
                    short = pi_time_series[o, i, j, l, m] < -0.000001
                    parti = 1 - (1 - long) * (1 - short)

                    switch_parti = abs(parti[1:] ^ parti[:-1])
                    sw = np.append(switch_parti, 0)
                    parti = np.where(sw == 1, 0.5, parti)
                    switch_parti = np.insert(switch_parti, 0, 0)
                    parti = np.where(switch_parti == 1, 0.5, parti)
                    parti_time_series[o, i, j, l, m] = parti
                    switch_parti_time_series[o, i, j, l, m] = sw

                    switch_short = abs(short[1:] ^ short[:-1])
                    sw_short = np.append(switch_short, 0)
                    short = np.where(sw == 1, 0.5, short)
                    switch_short = np.insert(switch_short, 0, 0)
                    short = np.where(switch_short == 1, 0.5, short)
                    short_time_series[o, i, j, l, m] = short
                    switch_short_time_series[o, i, j, l, m] = sw_short

# ######################################
# ############# Figure 1 ###############
# ######################################
# the zoom-in ones
# bad & Bad, phi = 0.4
# cohort 1; cohort 2; cohort 1, complete; cohort 1, disappointment
year = 100
t_length = int(year/dt)
t_indexes = np.empty((2,2,2))
red_case = 1
yellow_case = 1
phi_index = 1
scenario_indexs = [1, 2]
titles_subfig = [(r'Old Reentry, Cohort 1', r'Old Reentry, Cohort 2', r'Old Reentry, Cohort 3'), (r'Old Disappointment, Cohort 1', r'Old Disappointment, Cohort 2', r'Old Disappointment, Cohort 3')]
y_interest = Delta_time_series[:, red_case, yellow_case]  # n_scenarios, n_phi_short, nn, length
condition_parti_matrix = parti_time_series[:, red_case, yellow_case]
switch_parti_matrix = switch_parti_time_series[:, red_case, yellow_case]
switch_short_matrix = switch_short_time_series[:, red_case, yellow_case]
theta = theta_compare[:, red_case, yellow_case]  # n_scenarios, 2, 2, n_phi_short, Nt
fig, axes = plt.subplots(nrows=2, ncols=3, sharey='all', figsize=(15, 10))
# fig.suptitle('Good Z^Y, Bad Z^SI')
for i, ax_row in enumerate(axes):
    scenario_index = i + 3
    for j, ax in enumerate(ax_row):
        cohort_start = j + 1
        left = int(t_length * cohort_start)
        right = int(t_length * (2 + cohort_start))
        cohort_indexes = [(cohort_start - 1, cohort_start - 1), (cohort_start - 1, cohort_start - 1)]
        x = t[left: right]
        cohort_index = j
        y = y_interest[scenario_index, phi_index, cohort_index, left:right]
        condition_parti = condition_parti_matrix[scenario_index, phi_index, cohort_index, left:right]
        switch_parti = switch_parti_matrix[scenario_index, phi_index, cohort_index, left:right]
        switch_short = switch_short_matrix[scenario_index, phi_index, cohort_index, left:right]
        cutoff_belief = -theta[scenario_index, phi_index, left:right]
        y_N = np.ma.masked_where(condition_parti >= 0.8, y)
        y_P = np.ma.masked_where(condition_parti <= 0.2, y)
        # y_N = np.ma.masked_where(condition >= 0.05, y)
        # y_P = np.ma.masked_where(condition <= 0.0001, y)
        y_switch_parti = np.ma.masked_where(switch_parti == 0, y)
        y_switch_short = np.ma.masked_where(switch_short == 0, y)
        # ax.plot(x, y, label=cohort_labels[cohort_index], color='black', linewidth=0.4)
        ax.plot(x, cutoff_belief, label=r'Cutoff $\Delta_{s,t}$', color='blue', alpha=0.6, linewidth=0.4)
        ax.plot(x, y_P, label='Participant (P)', color='black', linewidth=0.6)
        ax.plot(x, y_N, label='Nonparticipant (N)', color='black', linewidth=0.6, linestyle='dotted')
        # ax2.fill_between(t, Delta_benchmarks[0], Delta_benchmarks[1], color=colors[m], alpha=0.4)
        ax.scatter(x, y_switch_parti, color='red', s=10, marker='o', label='switch')
        ax.scatter(x, y_switch_short, color='green', s=10, marker='o', label='short')
        if i == j == 0:
            ax.legend()
        if j == 0:
            ax.set_ylabel(r'Estimation error $\Delta_{s,t}$')
        if i == 1:
            ax.set_xlabel('Time in simulation')
        ax.set_title(titles_subfig[i][j])
        ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout(h_pad=2)
plt.savefig('Can short, Shocks and Delta, zoom in' + str(red_case) + str(yellow_case) +'.png', dpi=100)
plt.show()
plt.close()



# ######################################
# ############ Figure 2 ################
# ######################################
# portfolio in time series
# bad & Bad, phi = 0.4
# portfolio, different phi
# portfolio, different scenarios
label_phi = []
for i in range(n_phi_short):
    label_phi.append(r'$\phi$ = ' + str(phi_vector_short[i]))
year = 100
t_length = int(year/dt)
t_indexes = np.empty((2,2,2))
red_case = 1
yellow_case = 1
scenario_indexs = [1, 2]
titles_subfig = [(r'Old Reentry, Cohort 1', r'Old Reentry, Cohort 2', r'Old Reentry, Cohort 3'), (r'Old Disappointment, Cohort 1', r'Old Disappointment, Cohort 2', r'Old Disappointment, Cohort 3')]
y_interest = pi_time_series[:, red_case, yellow_case]  # n_scenarios, n_phi_short, nn, length
condition_parti_matrix = parti_time_series[:, red_case, yellow_case]
switch_parti_matrix = switch_parti_time_series[:, red_case, yellow_case]
switch_short_matrix = switch_short_time_series[:, red_case, yellow_case]
# theta = theta_compare[:, red_case, yellow_case]  # n_scenarios, 2, 2, n_phi_short, Nt
fig, axes = plt.subplots(nrows=2, ncols=3, sharey='all', figsize=(15, 10))
# fig.suptitle('Good Z^Y, Bad Z^SI')
for i, ax_row in enumerate(axes):
    scenario_index = i + 3
    for j, ax in enumerate(ax_row):
        cohort_start = j + 1
        left = int(t_length * cohort_start)
        right = int(t_length * (1.5 + cohort_start))
        cohort_indexes = [(cohort_start - 1, cohort_start - 1), (cohort_start - 1, cohort_start - 1)]
        x = t[left: right]
        cohort_index = j
        ax.axhline(0, 0.05, 0.95, color='gray', linestyle='--', linewidth=0.4)
        ax.axvline(left*dt+100, 0.05, 0.95, color='gray', linestyle='--', linewidth=0.4)
        for k in range(n_phi_short):
            phi_index = k
            y = y_interest[scenario_index, phi_index, cohort_index, left:right]
            condition_parti = condition_parti_matrix[scenario_index, phi_index, cohort_index, left:right]
            switch_parti = switch_parti_matrix[scenario_index, phi_index, cohort_index, left:right]
            switch_short = switch_short_matrix[scenario_index, phi_index, cohort_index, left:right]
            # cutoff_belief = -theta[scenario_index, phi_index, left:right]
            # y_N = np.ma.masked_where(condition_parti >= 0.8, y)
            # y_P = np.ma.masked_where(condition_parti <= 0.2, y)
            # y_N = np.ma.masked_where(condition >= 0.05, y)
            # y_P = np.ma.masked_where(condition <= 0.0001, y)
            y_switch_parti = np.ma.masked_where(switch_parti == 0, condition_parti-0.5)
            y_switch_short = np.ma.masked_where(switch_short == 0, y)
            ax.plot(x, y, color=colors_short[k], linewidth=0.6, label=label_phi[k])
            # ax.plot(x, cutoff_belief, label=r'Cutoff $\Delta_{s,t}$', color='blue', alpha=0.6, linewidth=0.4)
            # ax.plot(x, y_P, label='Participant (P)', color='black', linewidth=0.6)
            # ax.plot(x, y_N, label='Nonparticipant (N)', color='black', linewidth=0.6, linestyle='dotted')
            # ax2.fill_between(t, Delta_benchmarks[0], Delta_benchmarks[1], color=colors[m], alpha=0.4)
            if k == 0:
                ax.scatter(x, y_switch_parti, color='red', s=10, marker='o', label='switch')
            else:
                ax.scatter(x, y_switch_parti, color='red', s=10, marker='o')
            # if k == 0:
            #     ax.scatter(x, y_switch_parti, color=colors_short[k], s=10, marker='o', alpha=0.4, label='switch')
            # else:
            #     ax.scatter(x, y_switch_parti, color=colors_short[k], s=10, marker='o', alpha=0.4)
            # ax.scatter(x, y_switch_short, color='green', s=10, marker='o', label='short')
        if i == j == 0:
            ax.legend()
        if j == 0:
            ax.set_ylabel(r'Investment in stock market, $\pi_{s,t}/W_{s,t}$')
        if i == 1:
            ax.set_xlabel('Time in simulation')
        ax.set_title(titles_subfig[i][j])
        ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout(h_pad=2)
plt.savefig('Can short, Shocks and pi, zoom in' + str(red_case) + str(yellow_case) +'.png', dpi=100)
plt.show()
plt.close()


# ######################################
# ############ Figure 3 ################
# ######################################
# time-series of interest rate and market price of risk, bad z^Y, bad z&SI
# 4.1.1 interest rate across different phi values, reentry
# 4.1.2 interest rate across different scenarios, phi = 0
# 4.1.3 market price of risk across different phi values, reentry
red_case = 1
yellow_case = 1
theta_mat = theta_compare[:, red_case, yellow_case, 0]  # n_scenarios, Nt
Z = np.cumsum(Z_Y_cases[red_case])
Z_SI = np.cumsum(Z_SI_cases[yellow_case])
y_title_list = [r'Market price of risk $\theta_t$, reentry, partial shorting', r'Market price of risk $\theta_t$, disappointment, partial shorting']
fig, axes = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all', figsize=(15, 10))
for j, ax in enumerate(axes):
    ax.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
    ax.plot(t, Z, color='red', linewidth=0.5, label=r'$z^Y_t$')
    ax.plot(t, Z_SI, color='gold', linewidth=0.5, label=r'$z^{SI}_t$')
    ax.tick_params(axis='y', labelcolor='black')
    if j == 2:
        ax.set_xlabel('Time in simulation')
    ax_title = y_title_list[j]
    ax2 = ax.twinx()
    ax2.set_ylabel(r'$\theta_t$', color='black')
    ax2.plot(t, theta_mat[0], label=scenario_labels[0], color=colors_short[0], linestyle='dotted', linewidth=0.5)
    if j == 0:
        ax.legend(loc='upper left')
        ax2.plot(t, theta_mat[1], label=scenario_labels[1], color=colors_short[1], linestyle='dotted', linewidth=0.5)
        ax2.plot(t, theta_mat[3], label=scenario_labels[3], color=colors_short[2], linewidth=0.5)
    else:
        ax2.plot(t, theta_mat[2], label=scenario_labels[2], color=colors_short[1], linestyle='dotted', linewidth=0.5)
        ax2.plot(t, theta_mat[4], label=scenario_labels[4], color=colors_short[2], linewidth=0.5)
    ax2.legend(loc='upper right')
    ax2.set_ylim(-0.3,0.5)
    ax.set_title(ax_title)
fig.tight_layout(h_pad=2)
plt.savefig('partial shorting theta,' + str(red_case) + str(yellow_case)  +'.png', dpi=100)
plt.show()
plt.close()


# ######################################
# ############ Figure 4 ################
# ######################################
# time-series of Delta_bar, Phi, and participation rate, bad z^Y, bad z&SI
# 4.2.1 Delta_bar (reentry + complete + disappointment) * (phi = 0, 0.8)
# 4.2.2 Phi (reentry + disappointment) * (phi = 0, 0.8), also mark 1 for complete case
# 4.2.3 participation rate (reentry + disappointment) * (phi = 0, 0.8)
red_case = 1
yellow_case = 1
# titles_subfig = [r'Wealth weighted average estimation error conditional on participation $\bar{\Delta}_t$', r'Wealth share of participants $\Phi_t$', 'Participation rate']
yaxis_subfig = [r'$\bar{\Delta}_t$', r'$\Phi_t$', 'Participation rate']
fig_title_list = [r'Reentry, partial shorting, $\phi=0.4$', r'Disappointment, partial shorting, $\phi=0.4$']
left_t = 300
right_t = 400
y1_case = delta_bar_compare[:, red_case, yellow_case, 1, int(left_t/dt):int(right_t/dt)]
y2_case = Phi_compare[:, red_case, yellow_case, 1, int(left_t/dt):int(right_t/dt)]
y21_case = Phi_can_short_compare[:, red_case, yellow_case, 1, int(left_t/dt):int(right_t/dt)]
y22_case = Phi_short_compare[:, red_case, yellow_case, 1, int(left_t/dt):int(right_t/dt)]
y3_case = popu_parti_compare[:, red_case, yellow_case, 1, int(left_t/dt):int(right_t/dt)]
y31_case = popu_can_short_compare[:, red_case, yellow_case, 1, int(left_t/dt):int(right_t/dt)]
y32_case = popu_short_compare[:, red_case, yellow_case, 1, int(left_t/dt):int(right_t/dt)]
x = t[int(left_t/dt):int(right_t/dt)]
fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', figsize=(15, 20))
for j, ax_row in enumerate(axes):
    for k, ax in enumerate(ax_row):
        ax.set_ylabel(yaxis_subfig[j], color='black')
        if j == 0:
            if k == 0:
                y = [y1_case[0], y1_case[1], y1_case[3]]
                labels = [scenario_labels[0], scenario_labels[1], scenario_labels[3]]
            else:
                y = [y1_case[0], y1_case[2], y1_case[4]]
                labels = [scenario_labels[0], scenario_labels[2], scenario_labels[4]]
            ax.set_title(fig_title_list[k])
            line_styles = ['solid', 'solid', 'solid']
            colors_use = colors_short
        elif j == 1:
            if k == 0:
                y = [y2_case[1], y2_case[3], y21_case[3], y22_case[3]]
                labels = [scenario_labels[1], scenario_labels[3], 'Wealth share, can short', 'Wealth share, short']
            else:
                y = [y2_case[2], y2_case[4], y21_case[4], y22_case[4]]
                labels = [scenario_labels[2], scenario_labels[4], 'Wealth share, can short', 'Wealth share, short']
            line_styles = ['solid', 'solid', 'dashed', 'dotted']
            colors_use = [colors_short[1], colors_short[2], colors_short[2], colors_short[2]]
        else:
            if k == 0:
                y = [y3_case[1], y3_case[3], y31_case[3], y32_case[3]]
                labels = [scenario_labels[1], scenario_labels[3], 'Population, can short', 'Population, short']
            else:
                y = [y3_case[2], y3_case[4], y31_case[4], y32_case[4]]
                labels = [scenario_labels[2], scenario_labels[4], 'Population, can short', 'Population, short']
            line_styles = ['solid', 'solid', 'dashed', 'dotted']
            colors_use = [colors_short[1], colors_short[2], colors_short[2], colors_short[2]]
        for l, y_i in enumerate(y):
            label_i = labels[l]
            ax.plot(x, y_i, color=colors_use[l], linewidth=0.5, linestyle=line_styles[l], label=labels[l])
        ax.tick_params(axis='y', labelcolor='black')
        ax.legend(loc='upper left')
        if j == 2:
            ax.set_xlabel('Time in simulation')
fig.tight_layout(h_pad=2)
# plt.savefig('Delta bar, Phi and parti rate,' + str(red_case) + str(yellow_case)  + '.png', dpi=100)
plt.show()
# plt.close()



# Across paths, equilibirum values
# ######################################
# ############ Figure 3 ################
# ######################################
# over phi
N = 200  # for smaller number of paths
# tax_vector = [0.008]
# tax_vector = [0.01]
tax_vector = [0.012]
n_tax = len(tax_vector)
theta_matrix = np.empty((N, n_scenarios, n_tax, n_phi, 2))
delta_bar_matrix = np.zeros((N, n_scenarios, n_tax, n_phi, 2))
Phi_parti_1_matrix = np.zeros((N, n_scenarios, n_tax, n_phi, 2))
Phi_parti_matrix = np.zeros((N, n_scenarios, n_tax, n_phi, 2))
popu_age_matrix = np.zeros((N, n_scenarios, n_tax, n_phi, 2, n_age_groups))
wealthshare_age_matrix = np.zeros((N, n_scenarios, n_tax, n_phi, 2, n_age_groups))
popu_can_short_matrix = np.zeros((N, n_scenarios, n_tax, n_phi, 2))
popu_short_matrix = np.zeros((N, n_scenarios, n_tax, n_phi, 2))
Phi_can_short_matrix = np.zeros((N, n_scenarios, n_tax, n_phi, 2))
Phi_short_matrix = np.zeros((N, n_scenarios, n_tax, n_phi, 2))
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
        for l, tax_try in enumerate(tax_vector):
            beta_try = rho + nu - tax_try
            for m, phi_try in enumerate(phi_vector):
                (
                    r,
                    theta,
                    # popu_parti,
                    Delta_bar_parti,
                    Phi_parti,
                    Phi_parti_1,
                    popu_age,
                    # belief_age,
                    wealthshare_age,
                    popu_can_short,
                    popu_short,
                    Phi_can_short,
                    Phi_short,
                ) = simulate_SI_mean_vola(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S,
                                          tax_try,
                                          beta_try,
                                          phi_try,
                                          Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                                          top, old_limit, cutoffs, n_age_groups,
                                          )
                theta_matrix[j, k, l, m] = theta
                delta_bar_matrix[j, k, l, m] = Delta_bar_parti
                Phi_parti_matrix[j, k, l, m] = Phi_parti
                Phi_parti_1_matrix[j, k, l, m] = Phi_parti_1
                popu_age_matrix[j, k, l, m] = popu_age
                wealthshare_age_matrix[j, k, l, m] = wealthshare_age
                popu_can_short_matrix[j, k, l, m] = popu_can_short
                popu_short_matrix[j, k, l, m] = popu_short
                Phi_can_short_matrix[j, k, l, m] = Phi_can_short
                Phi_short_matrix[j, k, l, m] = Phi_short

var_list = [theta_matrix, Phi_parti_matrix, Phi_parti_1_matrix, delta_bar_matrix,
            popu_age_matrix, wealthshare_age_matrix,
            popu_can_short_matrix, popu_short_matrix, Phi_can_short_matrix, Phi_short_matrix]
var_name_list = ['market price of risk', 'consumption share of participants', 'consumption share 1 of participants',
                 'consumption-weighted estimation error of participants', 'popu age', 'wealth age',
                 'popu can short', 'popu short', 'wealth share can short', 'wealth share short']
type_list = ['mean', 'vola']
for i, var in enumerate(var_list):
    np.save('partial shorting' + var_name_list[i] + str(tax_vector[0]), var)


tax_vector = [0.008, 0.01, 0.012]
n_tax = len(tax_vector)
theta_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
Phi_parti_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
Phi_parti_1_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
Delta_bar_parti_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
popu_age_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2, n_age_groups))
wealthshare_age_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2, n_age_groups))
popu_can_short_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
popu_short_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
Phi_can_short_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
Phi_short_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
var_list = [theta_Mat, Phi_parti_Mat, Phi_parti_1_Mat, Delta_bar_parti_Mat,
            popu_age_Mat, wealthshare_age_Mat, popu_can_short_Mat, popu_short_Mat, Phi_short_Mat]
var_name_list = ['market price of risk', 'consumption share of participants', 'consumption share 1 of participants',
                 'consumption-weighted estimation error of participants', 'popu age', 'wealth age',
                 'popu can short', 'popu short', 'wealth share can short', 'wealth share short']
for i, var in enumerate(var_list):
    var_name = var_name_list[i]
    for j, tax_rate in enumerate(tax_vector):
        var_name_j = 'partial shorting' + var_name + str(tax_rate) +'.npy'
        y = np.load(var_name_j)
        var[:, j] = np.mean(np.mean(y[:200], axis=0), axis=1)

# theta and decompostition of theta over phi
x = phi_vector
x_start = 0
var_name_list = [r'market price of risk $\theta_t$', r'$\sigma_Y\frac{1}{\Phi_t}$', r'$\bar{\Delta}_t$']
# var_name_list = ['market price of risk', 'wealth share of participants', 'wealth-weighted estimation error of participants']
Phi_parti_1_sigma_Mat = Phi_parti_1_Mat * sigma_Y
var_list = [theta_Mat, Phi_parti_1_sigma_Mat, Delta_bar_parti_Mat]  # Shape((n_scenarios, n_tax, n_phi, 2))
scenario_list = ['Partial Shorting, Reentry', 'Partial Shorting, Disappointment']
tau_list = [r'$\tau=0.008$', r'$\tau=0.010$', r'$\tau=0.012$']
colors_short = ['midnightblue', 'darkgreen', 'darkviolet']
line_styles = ['dotted', 'solid', 'dashed']
fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', figsize=(15, 20))
for i, axes_row in enumerate(axes):
    row_name = var_name_list[i]
    var = var_list[i]  # Shape((n_scenarios, n_tax, n_phi, 2))
    for j, ax in enumerate(axes_row):
        y = var[:, :, :, j]  # Shape((n_scenarios, n_tax, n_phi))
        column_name = 'Mean' if j == 0 else 'Volatility'
        for k in range(0,3):
            # if i == 0 and j == 0:
            #     ax.axhline(sigma_Y, 0.05, 0.95, linestyle=line_styles[0], color=colors_short[0])
            line_style = line_styles[k]
            for l in range(n_scenarios):
                y_i = y[l, k]
                X_Y_Spline = make_interp_spline(x, y_i)
                # Returns evenly spaced numbers
                # over a specified interval.
                X_ = np.linspace(np.min(x), np.max(x), 100)
                Y_ = X_Y_Spline(X_)
                if i == 0:
                    if j == 0 and k == 1:
                        ax.plot(X_[x_start:], Y_[x_start:], linestyle=line_style, color=colors_short[l+1],
                                label=scenario_list[l])
                    elif j == 1 and l == 0:
                        ax.plot(X_[x_start:], Y_[x_start:], linestyle=line_style, color=colors_short[l+1],
                                label=tau_list[k])
                    else:
                        ax.plot(X_[x_start:], Y_[x_start:], linestyle=line_style, color=colors_short[l+1])
                else:
                    ax.plot(X_[x_start:], Y_[x_start:], linestyle=line_style, color=colors_short[l+1])
        if j == 0:
            ax.set_ylabel(row_name, rotation=90)
        if i == 2:
            ax.set_xlabel(r'$\phi$')
        if i == 0:
            ax.legend()
            ax.set_title(column_name)
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig('Partial Shorting, phi and values mean vola.png', dpi=200, format="png")
plt.savefig('initial window and values mean vola.png', dpi=500, format="png")
plt.show()
plt.close()

# Phi & participation rate over phi
x = phi_vector
x_start = 0
title_list = [r'Participate', r'Can short', r'Short']
var_name_list = [r'Wealth shares', r'Population']
popu_Mat = popu_age_Mat[:, :, :, :, 3]
var_list = [[Phi_parti_Mat, Phi_can_short_Mat, Phi_short_Mat],
            [popu_age_Mat, popu_can_short_Mat, popu_short_Mat]]  # Shape((n_scenarios, n_tax, n_phi, 2))
scenario_list = ['Partial Shorting, Reentry', 'Partial Shorting, Disappointment']
tax_index = 1
# tau_list = [r'$\tau=0.008$', r'$\tau=0.010$', r'$\tau=0.012$']
colors_short = ['midnightblue', 'darkgreen', 'darkviolet']
line_styles = ['dotted', 'solid', 'dashed']
fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', figsize=(15, 15))
for i, axes_row in enumerate(axes):
    row_name = var_name_list[i]
    var = var_list[i]  # Shape 3*((n_scenarios, n_tax, n_phi, 2))
    for j, ax in enumerate(axes_row):
        column_name = scenario_list[j]
        for k in range(3):
            y = var[k][j, tax_index, :, 0]  # Shape((n_phi))
            line_style = line_styles[k]
            ax.plot(x, y, linestyle=line_style, color=colors_short[k], label=title_list[k])
            # for l in range(n_scenarios):
            #     y_i = y[l, k]
            #     X_Y_Spline = make_interp_spline(x, y_i)
            #     # Returns evenly spaced numbers
            #     # over a specified interval.
            #     X_ = np.linspace(np.min(x), np.max(x), 100)
            #     Y_ = X_Y_Spline(X_)
            #     if i == 0:
            #         if j == 0 and k == 1:
            #             ax.plot(X_[x_start:], Y_[x_start:], linestyle=line_style, color=colors_short[l+1],
            #                     label=scenario_list[l])
            #         elif j == 1 and l == 0:
            #             ax.plot(X_[x_start:], Y_[x_start:], linestyle=line_style, color=colors_short[l+1],
            #                     label=tau_list[k])
            #         else:
            #             ax.plot(X_[x_start:], Y_[x_start:], linestyle=line_style, color=colors_short[l+1])
            #     else:
            #         ax.plot(X_[x_start:], Y_[x_start:], linestyle=line_style, color=colors_short[l+1])
        if j == 0:
            ax.set_ylabel(row_name, rotation=90)
        if i >= 2:
            ax.set_xlabel(r'$\phi$')
        if i == 0:
            ax.legend()
            ax.set_title(column_name)
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
# plt.savefig('Partial Shorting, phi and values mean vola.png', dpi=200, format="png")
# plt.savefig('initial window and values mean vola.png', dpi=500, format="png")
plt.show()
# plt.close()

###################################################################
################### Initial Training Window #######################
###################################################################
T_hats = dt * Npres
T_hat_dimension = len(T_hats)
# N = 30  # can choose a smaller number than Mpaths as the number of paths

n_scenarios_short = 1
a_sce = 3
# a_sce = 4

# Generate matrix to store the results
r_matrix = np.zeros((N, n_scenarios_short, n_phi_short, T_hat_dimension, 2))  # for mean and std
theta_matrix = np.zeros((N, n_scenarios_short, n_phi_short, T_hat_dimension, 2))
Phi_parti_matrix = np.zeros((N, n_scenarios_short, n_phi_short, T_hat_dimension, 2))
Phi_parti_1_matrix = np.zeros((N, n_scenarios_short, n_phi_short, T_hat_dimension, 2))
# popu_parti_matrix = np.zeros((N, n_scenarios_short, n_phi_short, T_hat_dimension, 2))
Delta_bar_parti_matrix = np.zeros((N, n_scenarios_short, n_phi_short, T_hat_dimension, 2))
popu_age_matrix = np.zeros((N, n_scenarios_short, n_phi_short, T_hat_dimension, 2, n_age_groups))
# belief_age_matrix = np.zeros((N, n_scenarios_short, n_phi_short, T_hat_dimension, 2, n_age_groups))
wealthshare_age_matrix = np.zeros((N, n_scenarios_short, n_phi_short, T_hat_dimension, 2, n_age_groups))

# write a lighter version of the simulation function that returns only the desired values (mean and std, instead of whole raw data)
for l in range(N):
    print(l)
    dZ = dZ_matrix[l]
    dZ_build = dZ_build_matrix[l]
    dZ_SI = dZ_SI_matrix[l]
    dZ_SI_build = dZ_SI_build_matrix[l]
    time_s = time.time()
    for m in range(n_scenarios):
        scenario = scenarios[m+a_sce]
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        for n, phi_try in enumerate(phi_vector):
            for o, T_hat_try in enumerate(T_hats):
                Npre_try = int(Npres[o])
                Vhat_try = (sigma_Y ** 2) / T_hat_try  # prior variance
                (
                    r,
                    theta,
                    # popu_parti,
                    Delta_bar_parti,
                    Phi_parti,
                    Phi_parti_1,
                    popu_age,
                    # belief_age,
                    wealthshare_age,
                    popu_can_short,
                    popu_short,
                    Phi_can_short,
                    Phi_short,
                ) = simulate_SI_mean_vola(mode_trade, mode_learn, Nc, Nt, dt, rho, nu,
                                          Vhat_try,
                                          mu_Y, sigma_Y, sigma_S,
                                          tax,
                                          beta,
                                          phi_try,
                                          Npre_try,
                                          Ninit,
                                          T_hat_try, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                                          top, old_limit, cutoffs, n_age_groups,
                                          )
                r_matrix[l, m, n, o] = r
                theta_matrix[l, m, n, o] = theta
                # popu_parti_matrix[l, m, n, o] = popu_parti
                Delta_bar_parti_matrix[l, m, n, o] = Delta_bar_parti
                Phi_parti_matrix[l, m, n, o] = Phi_parti
                Phi_parti_1_matrix[l, m, n, o] = Phi_parti_1
                popu_age_matrix[l, m, n, o] = popu_age
                # belief_age_matrix[l, m, n, o] = belief_age
                wealthshare_age_matrix[l, m, n, o] = wealthshare_age
    print(time.time() - time_s)


# save the data
var_list = [r_matrix, theta_matrix, Phi_parti_matrix, Phi_parti_1_matrix,
            Delta_bar_parti_matrix, popu_age_matrix,
            wealthshare_age_matrix]
var_name_list = ['interest rate', 'market price of risk', 'consumption share of participants', 'consumption share 1 of participants',
                 'consumption-weighted estimation error of participants', 'participation rate in age groups',
                 'wealth share in age groups']
type_list = ['mean', 'vola']
age_labels = ['20 < Age <= 35, youngest quartile', '35 < Age <= 55', '55 < Age <= 89', 'Age > 89, oldest quartile']

for i, var in enumerate(var_list):
    np.save('partial shorting' + var_name_list[i] + str(a_sce), var[:200])

# read the data:
r_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))  # for mean and std
theta_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))
Phi_parti_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))
Phi_parti_1_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))
#popu_parti_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))
Delta_bar_parti_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))
popu_age_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
#belief_age_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
wealthshare_age_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
var_list = [r_Mat, theta_Mat, Phi_parti_Mat, Phi_parti_1_Mat,
            Delta_bar_parti_Mat, popu_age_Mat, wealthshare_age_Mat]
var_name_list = ['interest rate', 'market price of risk', 'consumption share of participants', 'consumption share 1 of participants',
                 'consumption-weighted estimation error of participants', 'participation rate in age groups',
                 'wealth share in age groups']
for i, var in enumerate(var_list):
    var_name = var_name_list[i]
    for j in range(N_scenarios):
        var_name_j = 'partial shorting' + var_name + str(j) +'.npy'
        y = np.load(var_name_j)
        var[j] = np.mean(y[:200], axis=0)


# graphs:
######################################
######## OVER INITIAL WINDOW #########
############ GRAPH ONE ###############
######################################
# plot market price of risk, Phi, Delta_bar over Npre
x = Npres/12
x_start = 0
# x = Npres[1:]
var_name_list = [r'market price of risk $\theta_t$', r'$\sigma_Y\frac{1}{\Phi_t}$', r'$\bar{\Delta}_t$']
# var_name_list = ['market price of risk', 'consumption share of participants', 'consumption-weighted estimation error of participants']
Phi_parti_1_sigma_Mat = Phi_parti_1_Mat * sigma_Y
var_list = [theta_Mat, Phi_parti_1_sigma_Mat, Delta_bar_parti_Mat]  # Shape((N_scenarios, n_phi, T_hat_dimension, 2))
scenario_list = ['Complete', 'Reentry', 'Disappointment']
phi_list = [r'$\phi=0.0$', r'$\phi=0.4$', r'$\phi=0.8$']
colors_short = ['midnightblue', 'darkgreen', 'darkviolet']
line_styles = ['dotted', 'solid', 'dashed']
fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', figsize=(15, 20))
for i, axes_row in enumerate(axes):
    row_name = var_name_list[i]
    var = var_list[i]  # Shape((N_scenarios, 3, T_hat_dimension, 2))
    for j, ax in enumerate(axes_row):
        y = var[:, :, :, j]  # Shape((N_scenarios, 2, T_hat_dimension))
        column_name = 'Mean' if j == 0 else 'Volatility'
        for k in range(3):
            line_style = line_styles[k]
            for l in range(N_scenarios):
                y_i = y[l, k]
                if j == 0 and k == 1:
                    ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l], label=scenario_list[l])
                elif j == 1 and l == 0:
                    ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l], label=phi_list[k])
                else:
                    ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l])
        if j == 0:
            ax.set_ylabel(row_name, rotation=90)
        if i == 2:
            # ax.set_xlabel('initial window (months)')
            ax.set_xlabel('initial window (years)')
        if i == 0:
            ax.legend()
            ax.set_title(column_name)
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig('initial window and values mean vola years.png', dpi=100, format="png")
# plt.savefig('initial window and values mean vola.png', dpi=500, format="png")
plt.show()
# plt.close()

######################################
######## OVER INITIAL WINDOW #########
############ GRAPH ONE ###############
############ COVARIANCE ##############
######################################
theta_var = np.zeros((N_scenarios, n_phi, T_hat_dimension))
Phi_parti_1_var = np.zeros((N_scenarios, n_phi, T_hat_dimension))
Delta_bar_parti_var = np.zeros((N_scenarios, n_phi, T_hat_dimension))
var_list = [theta_var, Phi_parti_1_var, Delta_bar_parti_var]
var_name_list = ['market price of risk', 'consumption share 1 of participants', 'consumption-weighted estimation error of participants']
for i, var in enumerate(var_list):
    var_name = var_name_list[i]
    for j in range(N_scenarios):
        var_name_j = var_name + str(j) +'.npy'
        y = np.load(var_name_j)[:, :, :, :, 1]
        var[j] = np.mean(y ** 2, axis=0)
cov_Mat = theta_var - sigma_Y ** 2 * Phi_parti_1_var - Delta_bar_parti_var
x_start = 1
fig, ax = plt.subplots(figsize=(5, 5))
for k in range(3):
    line_style = line_styles[k]
    for l in range(N_scenarios):
        y_i = cov_Mat[l, k]
        if k == 1:
            ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l], label=scenario_list[l])
        elif l == 0:
            ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l])
        else:
            ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l])
ax.set_ylabel(r'Covariance, -2cov($\sigma_Y\frac{1}{\Phi_t}, \bar{\Delta}_t$)', rotation=90)
ax.set_xlabel('initial window (years)')
ax.legend()
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig('initial window and covariance years.png', dpi=500, format="png")
# plt.savefig('initial window and values mean vola.png', dpi=500, format="png")
plt.show()
# plt.close()

######################################
######## OVER INITIAL WINDOW #########
############ GRAPH TWO ###############
######################################
# plot participation rate and wealth share over Npre
x = Npres/12
# x = Npres[1:]
var_name_list = ['Participation rates in age groups', 'Wealth shares in age groups']
var_list = [popu_age_Mat, wealthshare_age_Mat]  # Shape((N_scenarios, n_phi, T_hat_dimension, 2))
phi_index = 1
scenario_list = ['Complete', 'Reentry', 'Disappointment']
phi_list = ['phi=0.0', 'phi=0.4']
colors_short = ['midnightblue', 'red', 'darkgreen', 'darkviolet']
age_labels = ['20 < Age <= 35, youngest quartile', '35 < Age <= 55', '55 < Age <= 89', 'Age > 89, oldest quartile']
hatches = ['/', 'o', '\\', '.']
low = np.zeros((T_hat_dimension))
fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', sharey='all', figsize=(15, 20))
for i, axes_row in enumerate(axes):
    row_name = scenario_list[i]
    for j, ax in enumerate(axes_row):
        var = var_list[j][i, phi_index, :, 0]  # Shape((N_scenarios, n_phi, T_hat_dimension, 2, n_age_groups)) -> Shape((T_hat_dimension, 4))
        if j == 0:
            for k in range(n_age_groups):
                y_i = var[:, k]
                if k == 0:
                    # ax.fill_between(x, y_i, color=colors_short[k], hatch=hatches[k], linewidth=0, label=age_labels[k])
                    ax.plot(x, y_i, color=colors_short[k], label=age_labels[k])
                    ax.fill_between(x, y_i, low, facecolor="none", edgecolor=colors_short[k], hatch=hatches[k], linewidth=0)
                    y_cumu = low + y_i
                # elif i == 0:
                #     y_plot = y_i + y_cumu
                #     # ax.fill_between(x, y_i, y_1, color=colors_short[k], linewidth=0, label=age_labels[k])
                #     ax.plot(x, y_plot, color=colors_short[k], label=age_labels[k])
                #     ax.fill_between(x, y_plot, y_cumu, facecolor="none", edgecolor=colors_short[k], hatch=hatches[k], linewidth=0)
                #     y_cumu = y_plot
                else:
                    y_1 = var[:, k - 1]
                    # ax.fill_between(x, y_i, y_1, color=colors_short[k], linewidth=0, label=age_labels[k])
                    ax.plot(x, y_i, color=colors_short[k], label=age_labels[k])
                    ax.fill_between(x, y_i, y_1, facecolor="none", edgecolor=colors_short[k], hatch=hatches[k], linewidth=0)
            if j == 0:
                ax.set_ylabel(row_name)
        else:
            for k in range(n_age_groups):
                y_i = var[:, k]
                if k == 0:
                    # ax.fill_between(x, y_i, color=colors_short[k], hatch=hatches[k], linewidth=0, label=age_labels[k])
                    ax.plot(x, y_i, color=colors_short[k], label=age_labels[k])
                else:
                    y_1 = var[:, k - 1]
                    # ax.fill_between(x, y_i, y_1, color=colors_short[k], linewidth=0, label=age_labels[k])
                    ax.plot(x, y_i, color=colors_short[k], label=age_labels[k])
                if i == 0:
                    ax.legend()
        if i == 0:
            # ax.legend()
            ax.set_title(var_name_list[j])
        if i == 2:
            ax.set_xlabel('initial window (years)')
            ax.tick_params(axis='x', labelcolor='black')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('initial window and values age groups.png', dpi=100, format="png")
plt.show()
#plt.close()
