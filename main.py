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
    n_age_groups, cutoffs, colors, modes_trade, modes_learn, Mpath, \
    scenarios, dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    dZ_Y_cases, dZ_SI_cases, dZ_build_case, dZ_SI_build_case, t, red_labels, yellow_labels, cohort_labels, \
    scenario_labels, colors_short, colors_short2, PN_labels, age_labels
from src.stats import shocks, tau_calculator, good_times, Delta_st_compare, weighted_variance
from numba import jit
import matplotlib.pyplot as plt
import statsmodels.api as sm
# import tabulate as tabulate
from scipy.interpolate import make_interp_spline
import pandas as pd

n_scenarios = 5
scenarios_short = scenarios[:n_scenarios]

phi_vector = np.arange(0, 1, 0.1)
n_phi = len(phi_vector)

phi_indexes = [0, 4, 8]
n_phi_short = len(phi_indexes)
phi_vector_short = phi_vector[phi_indexes]

label_phi = []
for i in range(n_phi_short):
    label_phi.append(r'$\phi$ = ' + str(phi_vector_short[i]))
labels = [scenario_labels, label_phi, label_phi]

age_cutoff = cutoffs[2]

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

# ######################################
# ########### Figure 1 & IA1 #############
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
            y_cohort_N = np.ma.masked_where(parti_time_series[scenario_index, red_index, yellow_index, i, m] == 1,
                                            y_cohort)
            y_cohort_P = np.ma.masked_where(parti_time_series[scenario_index, red_index, yellow_index, i, m] == 0,
                                            y_cohort)
            y_cohort_switch = np.ma.masked_where(switch_time_series[scenario_index, red_index, yellow_index, i, m] == 0,
                                                 y_cohort)
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
# ############# Figure 2 ###############
# ######################################
# the zoom-in ones
# bad & Bad, phi = 0.4
# cohort 1; cohort 2; cohort 1, complete; cohort 1, disappointment
print('Figure 2')
year = 100
t_length = int(year / dt)
t_indexes = np.empty((2, 2, 2))
red_case = 1
yellow_case = 1
cohort_start = 3
t_indexes[0, 0, 0] = t_indexes[1, 0, 0] = t_indexes[1, 1, 0] = t_indexes[0, 1, 0] = t_length * cohort_start
t_indexes[0, 0, 1] = t_indexes[1, 0, 1] = t_indexes[1, 1, 1] = t_indexes[0, 1, 1] = t_length * (1 + cohort_start)
phi_where = [(1, 2), (1, 1)]
cohort_indexes = [(cohort_start - 1, cohort_start - 1), (cohort_start - 1, cohort_start - 1)]
scenario_indexs = [(1, 1), (0, 2)]
titles_subfig = [(r'Reentry, $\phi=0.4$', r'Reentry, $\phi=0.8$'),
                 (r'Complete market, $\phi=0.4$', r'Disappointment, $\phi=0.4$')]
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
            # ax.plot(x, cutoff_belief, label='cutoff Delta', color='pink', alpha=0.8, linewidth=0.4)
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
plt.savefig('Shocks and Delta, zoom in' + str(red_case) + str(yellow_case) + '.png', dpi=60)
plt.show()
plt.close()

# ######################################
# ############## Figure 5 ##############
# ######################################
# time-series of interest rate and market price of risk, bad z^Y, bad z&SI
# 4.1.1 interest rate across different phi values, reentry
# 4.1.2 interest rate across different scenarios, phi = 0
# 4.1.3 market price of risk across different phi values, reentry
print('Figure 5')
red_case = 1
yellow_case = 1
r_mat = r_compare[:, red_case, yellow_case]  # n_scenarios, n_phi_short, Nt
theta_mat = theta_compare[:, red_case, yellow_case]
y1 = r_mat[:, 0]  # n_scenarios, Nt
y2 = r_mat[1]  # n_phi_short, Nt
y3 = theta_mat[1]  # n_phi_short, Nt
y_list = [y1, y2, y3]
Z = np.cumsum(Z_Y_cases[red_case])
Z_SI = np.cumsum(Z_SI_cases[yellow_case])
n_lines = [n_scenarios, n_phi_short, n_phi_short]
y_title_list = [r'Interest rate $r_t$, $\phi=0$', r'Interest rate $r_t$, Reentry',
                r'Market price of risk $\theta_t$, Reentry']
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
    if i < 2:
        ax2.set_ylim(lower, upper)
    if j == 0:
        ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.set_title(y_title)
fig.tight_layout(h_pad=2)
plt.savefig('r and theta,' + str(red_case) + str(yellow_case) + '.png', dpi=60)
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
# ############# Figure 7 #############
# ######################################
# time-series of Delta_bar, Phi, and participation rate, bad z^Y, bad z&SI
print('Figure 7')
red_case = 1
yellow_case = 1
titles_subfig = [r'Wealth weighted average estimation error conditional on participation $\bar{\Delta}_t$',
                 r'Wealth share of participants $\Phi_t$', 'Participation rate']
yaxis_subfig = [r'$\bar{\Delta}_t$', r'$\Phi_t$', 'Participation rate']
phi_indexes = [0, 2]
y1_case = delta_bar_compare[:, red_case, yellow_case]
y2_case = Phi_compare[:, red_case, yellow_case]
y3_case = popu_parti_compare[:, red_case, yellow_case]
left_t = 300
right_t = 400
Z = np.cumsum(Z_Y_cases[red_case])[int(left_t / dt):int(right_t / dt)]
Z_SI = np.cumsum(Z_SI_cases[yellow_case])[int(left_t / dt):int(right_t / dt)]
x = t[int(left_t / dt):int(right_t / dt)]

fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', figsize=(15, 15))
axes[0, 0].set_title('Shocks to output and signal')
axes[0, 0].set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
axes[0, 0].plot(x, Z, color='red', linewidth=0.5, linestyle='solid', label=r'$z^Y_t$')
axes[0, 0].plot(x, Z_SI, color='gold', linewidth=0.5, linestyle='solid', label=r'$z^{SI}_t$')
axes[0, 0].tick_params(axis='y', labelcolor='black')
axes[0, 0].legend(loc='upper left')

axes[0, 1].set_title(titles_subfig[0])
axes[0, 1].set_ylabel(yaxis_subfig[0], color='black')
for i in range(3):
    for j in range(2):
        y1 = y1_case[i, phi_indexes[j], int(left_t / dt):int(right_t / dt)]
        line_style = 'dotted' if j == 0 else 'solid'
        label_i = scenario_labels[i]
        if j == 1:
            axes[0, 1].plot(x, y1, label=label_i, linewidth=0.5, color=colors_short[i], linestyle=line_style)
        else:
            axes[0, 1].plot(x, y1, linewidth=0.5, color=colors_short[i], linestyle=line_style)
axes[0, 1].legend(loc='upper right')

axes[1, 0].set_title(titles_subfig[1])
axes[1, 0].set_ylabel(yaxis_subfig[1], color='black')
for i in range(1, 3):
    for j in range(2):
        y2 = y2_case[i, phi_indexes[j], int(left_t / dt):int(right_t / dt)]
        line_style = 'dotted' if j == 0 else 'solid'
        if i == 1 and j == 0:
            axes[1, 0].plot(x, y2, label=r'$\phi=0.0$', linewidth=0.5, color=colors_short[i], linestyle=line_style)
        elif i == 1 and j == 1:
            axes[1, 0].plot(x, y2, label=r'$\phi=0.8$', linewidth=0.5, color=colors_short[i], linestyle=line_style)
        else:
            axes[1, 0].plot(x, y2, linewidth=0.5, color=colors_short[i], linestyle=line_style)
y21 = y2_case[0, 0, int(left_t / dt):int(right_t / dt)]
axes[1, 0].legend(loc='upper right')
axes[1, 0].set_xlabel('Time in simulation')

axes[1, 1].set_title(titles_subfig[2])
axes[1, 1].set_ylabel(yaxis_subfig[2], color='black')
# ax3 = axes[2].twinx()
for i in range(1, 3):
    for j in range(2):
        y3 = y3_case[i, phi_indexes[j], int(left_t / dt):int(right_t / dt)]
        line_style = 'dotted' if j == 0 else 'solid'
        label_i = scenario_labels[i]
        axes[1, 1].plot(x, y3, linewidth=0.5, color=colors_short[i], linestyle=line_style)
axes[1, 1].set_xlabel('Time in simulation')
fig.tight_layout(h_pad=2)
plt.savefig('Delta bar, Phi and parti rate,' + str(red_case) + str(yellow_case) + '.png', dpi=60)
# plt.show()
# plt.close()

############ IA other cases

titles_subfig = [r'Participants wealth weighted average estimation error $\bar{\Delta}_t$',
                 r'Wealth share of participants $\Phi_t$', 'Participation rate']
scenario_indexes = [1, 2]
yaxis_subfig = [r'$\bar{\Delta}_t$', r'$\Phi_t$', 'Participation rate']
phi_indexes = [0, 2]
red_cases = [0, 0, 1]
yellow_cases = [0, 1, 0]
var_list = [delta_bar_compare, Phi_compare, popu_parti_compare]
left_t = 300
right_t = 400
x = t[int(left_t / dt):int(right_t / dt)]

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
                y = var[scenario_index, phi_index, int(left_t / dt):int(right_t / dt)]
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
        if i == 0:
            ax.set_title(title_subfig)

        if i == 2:
            ax.set_xlabel('Time in simulation')
fig.tight_layout(h_pad=2)
plt.savefig('IA Delta bar, Phi and parti rate.png', dpi=100)
plt.show()
plt.close()

# ######################################
# ############  Figure 8  ##############
# ######################################
# portfolio in time series
# bad & Bad, phi = 0.4
# portfolio, different phi
# portfolio, different scenarios
print('Figure 8')
cohort_index = 2
left_t = (cohort_index + 1) * 100
right_t = (cohort_index + 2) * 100
red_case = 1
yellow_case = 1
phi_index = 1
scenario_index = 1
Z = np.cumsum(Z_Y_cases[red_case])[int(left_t / dt):int(right_t / dt)]
Z_SI = np.cumsum(Z_SI_cases[yellow_case])[int(left_t / dt):int(right_t / dt)]
x = t[int(left_t / dt):int(right_t / dt)]
y1_pi = pi_time_series[:, red_case, yellow_case, phi_index,
        cohort_index]  # (n_scenarios, 2, 2, n_phi_short, nn, length) -> (n_scenarios, length)
y2_pi = pi_time_series[scenario_index, red_case, yellow_case, :,
        cohort_index]  # (n_scenarios, 2, 2, n_phi_short, nn, length) -> (n_phi_short, length)
y1_belief = (Delta_time_series[:, red_case, yellow_case, phi_index, cohort_index] -
             delta_bar_compare[:, red_case, yellow_case, phi_index]) / sigma_Y
y2_belief = (Delta_time_series[scenario_index, red_case, yellow_case, :, cohort_index] -
             delta_bar_compare[scenario_index, red_case, yellow_case, :]) / sigma_Y
y1_wealth = 1 / Phi_compare[:, red_case, yellow_case, phi_index]
y2_wealth = 1 / Phi_compare[scenario_index, red_case, yellow_case]
y_cases = [y1_pi, y1_belief, y1_wealth,
           y2_pi, y2_belief, y2_wealth]
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
            length = int(right_t / dt) - int(left_t / dt)
            y_case = np.ones(length)
            ax.plot(x, y_case, label=label_i, color=colors_short[j], linewidth=0.4)
        else:
            y_case = y_cases[i][j, int(left_t / dt):int(right_t / dt)]
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
    ax.set_title(titles_subfig[i])
    ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout(h_pad=2)
plt.savefig('Shocks and Portfolio,' + str(red_case) + str(yellow_case) + '.png', dpi=60)
# plt.show()
# plt.close()

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
x = t[int(left_t / dt):int(right_t / dt)]

fig, axes = plt.subplots(nrows=3, ncols=3, sharex='all', figsize=(15, 15))
for i, ax_row in enumerate(axes):
    red_case = red_cases[i]
    yellow_case = yellow_cases[i]
    for j, ax in enumerate(ax_row):
        title_subfig = titles_subfig[j]
        var = var_list[j][:, red_case, yellow_case]
        for scenario_index in scenario_indexes:
            color_use = colors_short[scenario_index]
            y = var[scenario_index, phi_fix, int(left_t / dt):int(right_t / dt)]
            labels = scenario_labels
            ax.plot(x, y, label=labels[scenario_index] + ', ' + label_phi[phi_fix], linewidth=0.5,
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
# plt.show()
# plt.close()


# ######################################
# ############ Figure  9.1 #############
# ######################################
# consumption, different tax rates
# build consumption data and time series over tau values
print('Figure 9')
tax_vector = [0.008, 0.010, 0.012]
n_tax = len(tax_vector)
dZ = Z_Y_cases[1]
dZ_SI = Z_SI_cases[1]
n_scenarios = 1
n_phi_short = 1
cohort_index = 2
left_t = int(starts[cohort_index] / dt)
right_t = int(left_t + 100 / dt)
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
                ax.set_title(titles_subfig[i] + ', ' + scenario_labels[scenario_index] + ', ' + r'$\phi=0.0$')
            else:
                ax.set_title(titles_subfig_IA[i] + ', ' + red_labels[case_dzY] + scenario_labels[
                    scenario_index] + ', ' + r'$\phi=0.0$')

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
            plt.savefig('IA Consumption share tau' + str(case_dzY) + str(case_dzSI) + str(scenario_index) + '.png',
                        dpi=100)
        plt.show()
        plt.close()

# ######################################
# ############ Fig 6 & 13 ##############
# ###### Distribution of Delta #########
# ######################################
print('Figure 6 & 13')
cases = [0, 1]
cohort_size_mat = np.tile(cohort_size, (Nc, 1))
Npres_try = [60, 240]
phi_fix = phi_vector[4]
for case_dzY in cases:
    for case_dzSI in cases:
        dZ = dZ_Y_cases[case_dzY]
        dZ_SI = dZ_SI_cases[case_dzSI]
        for j, Npre_try in enumerate(Npres_try):
            T_hat_try = dt * Npre_short
            Vhat_try = (sigma_Y ** 2) / T_hat_short
            Delta_compare = np.empty((len(scenarios_two), Nt, Nc))
            invest_tracker_compare = np.empty((len(scenarios_two), Nt, Nc))
            theta_compare = np.empty((len(scenarios_two), Nt))
            for g, scenario in enumerate(scenarios_two):
                mode_trade = scenario[0]
                mode_learn = scenario[1]
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
                ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu,
                                Vhat_try,
                                mu_Y, sigma_Y, sigma_S, tax, beta,
                                phi_fix,
                                Npre_try, Ninit,
                                T_hat_try,
                                dZ_build_case, dZ, dZ_SI_build_case, dZ_SI, tau, cohort_size,
                                need_f='False',
                                need_Delta='True',
                                need_pi='False',
                                top=0.05,
                                old_limit=100
                                )
                theta_compare[g] = theta
                Delta_compare[g] = Delta
                invest_tracker_compare[g] = invest_tracker
            y_overall = np.empty((len(scenarios_two), Nt, 5))  # overall
            y_P = np.empty((len(scenarios_two), Nt, 5))  # participants / long
            y_N = np.empty((len(scenarios_two), Nt, 5))  # non-participants / short
            y_min = np.empty((len(scenarios_two), Nt, n_age_groups))
            y_max = np.empty((len(scenarios_two), Nt, n_age_groups))
            y_cases = [y_overall, y_P, y_N]
            for i in range(len(scenarios_two)):
                for n in range(n_age_groups):
                    Delta_age_group = Delta_compare[i, :, cutoffs[n + 1]:cutoffs[n]]
                    y_min[i, :, n] = np.amin(Delta_age_group, axis=1)
                    y_max[i, :, n] = np.amax(Delta_age_group, axis=1)
                for m in range(Nt):
                    Delta = Delta_compare[i, m]  # ((Nt, Nc))
                    parti_cohorts = invest_tracker_compare[i, m]
                    if np.sum(parti_cohorts) == Nt:
                        cohort_sizes = [cohort_size]
                        Deltas = [Delta]
                        n_var = 1
                    else:
                        Delta1 = parti_cohorts * Delta
                        Delta2 = (1 - parti_cohorts) * Delta
                        cohort_size1 = parti_cohorts * cohort_size
                        cohort_size2 = (1 - parti_cohorts) * cohort_size
                        cohort_sizes = [cohort_size, cohort_size1, cohort_size2]
                        Deltas = [Delta, Delta1, Delta2]
                        n_var = 3
                    for n in range(n_var):
                        Del = Deltas[n]
                        cohort_siz = cohort_sizes[n]
                        Delta_rank = Del.argsort()
                        Delta_sorted = Del[Delta_rank[::-1]]
                        cohort_size_sorted = cohort_siz[Delta_rank[::-1]]
                        popu_cumsum = np.cumsum(cohort_size_sorted)
                        total_popu = popu_cumsum[-1]
                        Delta_cutoff = np.zeros(5)
                        cutoff = np.searchsorted(popu_cumsum, [0.25 * total_popu, 0.5 * total_popu, 0.75 * total_popu])
                        Delta_cutoff[1:4] = Delta_sorted[cutoff]  # highest to lowest
                        Delta_cutoff[0] = np.max(Del[np.nonzero(Del)])
                        Delta_cutoff[4] = np.min(Del[np.nonzero(Del)])
                        y_cases[n][i, m] = Delta_cutoff

            left_t = 200
            right_t = 400
            Z = np.cumsum(dZ)[int(left_t / dt):int(right_t / dt)]
            Z_SI = np.cumsum(dZ_SI)[int(left_t / dt):int(right_t / dt)]
            x = t[int(left_t / dt):int(right_t / dt)]
            scenario_indexes = [0, 1]
            fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all', figsize=(15, 15))
            for i, ax_row in enumerate(axes):
                scenario_index = scenario_indexes[i]
                y1 = y_overall[scenario_index, int(left_t / dt):int(right_t / dt)]
                y2 = y_P[scenario_index, int(left_t / dt):int(right_t / dt)]  # ((Nt, 5))
                y3 = y_N[scenario_index, int(left_t / dt):int(right_t / dt)]  # ((Nt, 5))
                y4 = y_min[scenario_index, int(left_t / dt):int(right_t / dt)]
                y5 = y_max[scenario_index, int(left_t / dt):int(right_t / dt)]
                belief_cutoff_case = -theta_compare[scenario_index, int(left_t / dt):int(right_t / dt)]
                for j, ax in enumerate(ax_row):
                    ax.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
                    ax.set_title(scenario_labels[scenario_index + 1] + r', $\phi=0.4$')
                    if j == 0:
                        if i == 0:
                            ax2 = ax.twinx()
                            ax2.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
                            ax2.plot(x, Z, color='red', linewidth=0.5, label=r'$z^Y_t$')
                            ax2.plot(x, Z_SI, color='gold', linewidth=0.5, label=r'$z^{SI}_t$')
                            ax2.tick_params(axis='y', labelcolor='black')
                            ax2.legend(loc='upper left')
                        y20 = y2[:, 0]
                        y21 = y2[:, 1]
                        y22 = y2[:, 2]
                        y23 = y2[:, 3]
                        y24 = np.maximum(belief_cutoff_case, y1[:, 4])
                        y30 = np.maximum(belief_cutoff_case, y3[:, 0])
                        y31 = y3[:, 1]
                        y32 = y3[:, 2]
                        y33 = y3[:, 3]
                        y34 = y3[:, 4]
                        ax.fill_between(x, y20, y24, color='blue', linewidth=0., alpha=0.3)
                        ax.fill_between(x, y21, y23, color='blue', linewidth=0., alpha=0.5, label=PN_labels[0])
                        ax.fill_between(x, y30, y34, color='green', linewidth=0., alpha=0.3)
                        ax.fill_between(x, y31, y33, color='green', linewidth=0., alpha=0.5, label=PN_labels[1])
                        # ax.plot(x, y22, color='blue', linewidth=0.4, label='P')
                        # ax.plot(x, y32, color='green', linewidth=0.4, label='N')
                        ax.plot(x, belief_cutoff_case, color='black', linewidth=0.4, label=r'Cutoff $\Delta_{s,t}$')
                    else:
                        ax.plot(x, belief_cutoff_case, color='black', linewidth=0.4, label=r'Cutoff $\Delta_{s,t}$')
                        for k in range(n_age_groups):
                            y40 = y4[:, k]
                            y50 = y5[:, k]
                            ax.fill_between(x, y40, y50, color=colors_short[k], linewidth=0., alpha=0.4,
                                            label=age_labels[k])
                    if i == 0:
                        ax.legend(loc='upper right')
                    else:
                        ax.set_xlabel('Time in simulation')
            fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
            if case_dzY == case_dzSI == 1:
                if j == 0:
                    plt.savefig(str(case_dzY) + str(case_dzSI) + 'Distribution of Delta.png', dpi=60)
                else:
                    plt.savefig(str(case_dzY) + str(case_dzSI) + 'Distribution of Delta, short window.png', dpi=60)
            else:
                if j == 0:
                    plt.savefig('IA ' + str(case_dzY) + str(case_dzSI) + 'Distribution of Delta.png', dpi=60)
            # plt.show()
            # plt.close()

# ######################################
# ############  Figure 3  ##############
# ###########  Figure 9.2  #############
# ########### & Figure 10 ##############
# ######################################
print('Data generation for Figure 3, 9.2 and 10')
# varying tau only when phi == 0.0
# storing drift, diffusion, average view when phi == 0.0 and tau == 0.01
n_scenarios_short = 3
scenarios_short = scenarios[n_scenarios_short]
age_cut = 150
Nc_cut = int(age_cut / dt)
data_keep = np.arange(0, Nc, 60)
Nt_short = len(data_keep)

Delta_matrix = np.empty((Mpath, n_scenarios_short, n_phi_5, Nc_cut), dtype=np.float32)
distance_matrix = np.empty((Mpath, n_scenarios_short, n_phi_5, Nc_cut), dtype=np.float32)
invest_matrix = np.empty((Mpath, n_scenarios_short, n_phi_5, Nc_cut), dtype=np.int)

popu_cummu = np.cumsum(cohort_size)
popus = np.array([0.1, 0.5])
popus_1 = 1 - popus
cutoff_young = np.searchsorted(popu_cummu, popus_1)
cutoff_old = np.searchsorted(popu_cummu, popus)
diffusion_P_matrix = np.empty((Mpath, n_scenarios, Nt_short, Nc_cut), dtype=np.float32)  # store data only when phi == 0
diffusion_matrix = np.empty((Mpath, n_scenarios, Nt_short, Nc_cut), dtype=np.float32)
drift_matrix = np.empty((Mpath, n_scenarios, Nt_short, Nc_cut), dtype=np.float32)
drift_P_matrix = np.empty((Mpath, n_scenarios, Nt_short, Nc_cut), dtype=np.float32)
r_matrix = np.empty((Mpath, n_scenarios, Nt_short), dtype=np.float32)
parti_old_matrix = np.empty((Mpath, n_scenarios, 2, Nt_short), dtype=np.float32)
parti_young_matrix = np.empty((Mpath, n_scenarios, 2, Nt_short), dtype=np.float32)
average_belief_matrix = np.empty((Mpath, n_scenarios, Nt_short), dtype=np.float32)
cohort_size_short = cohort_size[-Nc_cut:]

tax_vector = [0.008, 0.010, 0.012]
n_tax = len(tax_vector)
f_matrix_tax = np.empty((Mpath, n_scenarios_short, n_tax, Nc_cut), dtype=np.float32)  # store data only when phi == 0

for i in range(Mpath):
    if np.mod(i, 50) == 0:
        print(i)
    dZ = np.random.randn(Nt) * dt_root
    dZ_build = np.random.randn(Nc) * dt_root
    dZ_SI = np.random.randn(Nt) * dt_root
    dZ_SI_build = np.random.randn(Nc) * dt_root
    for j, scenario in enumerate(scenarios_short):
        scenario_trade = scenario[0]
        scenario_learn = scenario[1]
        for l, phi_try in enumerate(phi_5):
            if l == 0:
                for k, tax_try in enumerate(tax_vector):
                    beta_try = rho + nu - tax_try
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
                    ) = simulate_SI(scenario_trade, scenario_learn,
                                    Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S,
                                    tax_try,
                                    beta_try,
                                    phi_try,
                                    Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                                    need_f='True',
                                    need_Delta='True',
                                    need_pi='False',
                                    )
                    f_matrix_tax[i, j, k] = np.average(f, axis=0)[-Nc_cut:]
                    if tax_try == 0.01:
                        Delta_matrix[i, j, l] = np.average(np.abs(Delta), axis=0)[-Nc_cut:]  # across time
                        invest_matrix[i, j, l] = np.average(invest_tracker_reentry, axis=0)[-Nc_cut:]

                        theta_mat = np.transpose(np.tile(theta, (Nc_cut, 1)))
                        r_mat = np.transpose(np.tile(r, (Nc_cut, 1)))
                        drift_N = -rho + r_mat
                        Delta_short = Delta[:, -Nc_cut:]
                        drift_P = -rho + r_mat + 0.5 * theta_mat ** 2 - 0.5 * Delta_short ** 2
                        diffusion_P = np.abs(theta_mat + Delta_short)
                        r_matrix[i, j] = r[data_keep]
                        invest_tracker_short = invest_tracker[:, -Nc_cut:]
                        drift_c = drift_P * invest_tracker_short + drift_N * (1 - invest_tracker_short)
                        drift_matrix[i, j] = drift_c[data_keep]
                        drift_P_matrix[i, j] = (drift_P * invest_tracker_short)[data_keep]
                        diffusion_P_matrix[i, j] = np.abs((diffusion_P * invest_tracker_short))[data_keep]
                        market_view = np.average(Delta_short, weights=cohort_size_short, axis=1)  # across cohorts
                        average_belief_matrix[i, j] = market_view[data_keep]
                        for ii in range(2):
                            parti_rate_young = np.average(invest_tracker[:, cutoff_young[ii]:], axis=1)
                            parti_rate_old = np.average(invest_tracker[:, :cutoff_old[ii]], axis=1)
                            parti_old_matrix[i, j, ii] = parti_rate_old[data_keep]
                            parti_young_matrix[i, j, ii] = parti_rate_young[data_keep]

            else:
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
                ) = simulate_SI(scenario_trade, scenario_learn,
                                Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S,
                                tax,
                                beta,
                                phi_try,
                                Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                                need_f='True',
                                need_Delta='True',
                                need_pi='False',
                                )
                Delta_matrix[i, j, l] = np.average(np.abs(Delta), axis=0)[-Nc_cut:]
                invest_matrix[i, j, l] = np.average(invest_tracker, axis=0)[-Nc_cut:]

# Figure 3
t_cut = 100
N_cut = int(t_cut / dt)
x = t[:N_cut]
data_point = np.arange(0, N_cut, 15)
Delta_vector = np.flip(np.average(Delta_matrix, axis=0), axis=2)
invest_vector = np.flip(np.average(invest_matrix, axis=0), axis=2)
y_cases = [Delta_vector[0], Delta_vector[1], invest_vector[1]]
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
plt.savefig('Average estimation error and age.png', dpi=60)
# plt.show()
# plt.close()

# Figure 9.2
f_vector = np.flip(np.average(f_matrix_tax, axis=0), axis=2)
c_vector = f_vector / np.flip(cohort_size)
titles_subfig = [r'Average individual consumption share $c_{s,t}/Y_t$', r'Average cohort consumption share $f_{s,t}$']
yaxis_subfig = [r'$Average c_{s,t}/Y_t$', r'$Average f_{s,t}$']
line_styles = [(0, (5, 10)), 'solid', (0, (1, 1))]
fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
for i, ax in enumerate(axes):
    ax.set_title(titles_subfig[i] + ', ' + r'$\phi=0.4$')
    var = c_vector if i == 0 else f_vector
    for k in range(n_tax):
        y = var[1, k, :N_cut]  # n_scenarios, 2, 2, n_tax, n_phi_short, nn, length
        label_i = r'$\tau$ = ' + str('{0:.3f}'.format(tax_vector[k]))
        if k == 1:
            ax.plot(x, y, color=colors_short[1], linewidth=0.8, linestyle=line_styles[k], label='Reentry, ' + label_i)
            y_complete = var[0, k, :N_cut]
            y_disappintment = var[2, k, :N_cut]
            ax.plot(x, y_complete, color=colors_short[0], linewidth=0.6, linestyle=line_styles[k],
                    label='Complete, ' + label_i)
            ax.plot(x, y_disappintment, color=colors_short[2], linewidth=0.6, linestyle=line_styles[k],
                    label='Disappointment, ' + label_i)
        else:
            ax.plot(x, y, color=colors_short[1], linewidth=0.8, linestyle=line_styles[k], label='Reentry, ' + label_i)
    ax.tick_params(axis='both', labelcolor='black')
    ax.set_ylabel(yaxis_subfig[i], color='black')
    # ax.set_xlim(left_t, right_t)
    if i == 0:
        ax.legend()
    ax.set_xlabel('Age')
fig.tight_layout(h_pad=2)
plt.savefig('Average consumption share tau.png', dpi=100)
# plt.show()
# plt.close()


# Figure 10
parti_gap_mat = parti_young_matrix - parti_old_matrix
# condition = parti_gap_mat[:, :, 0]
condition = average_belief_matrix
interval = 60  # 5-year age bins
bins = np.arange(0, Nc_cut + 1, interval, dtype=int)
n_bins = len(bins)
x = (bins[:n_bins - 1] + interval / 2) * dt
quartiles = np.arange(0, 100 + 1, 25)
results_data = np.zeros((4, n_bins - 1, 6))
results_data_uncon = np.zeros((n_bins - 1, 6))
i = 1
r_mat = r_matrix[:, i]
drift_c_mat = np.flip(drift_matrix[:, i], axis=2)
drift_P_mat = np.flip(drift_P_matrix[:, i], axis=2)
drift_N_mat = drift_c_mat - drift_P_mat
drift_P_mat = np.ma.masked_equal(drift_P_mat, 0)
drift_N_mat = np.ma.masked_equal(drift_N_mat, 0)
diffusion_c_mat = np.flip(diffusion_P_matrix[:, i], axis=2)  # includes both N and P
diffusion_P_mat = np.ma.masked_equal(diffusion_c_mat, 0)  # includes both N and P
condition_mat = condition[:, i]
quartiles_condition = np.percentile(condition_mat, quartiles)
for k in range(4):
    condition_below = quartiles_condition[k]
    condition_above = quartiles_condition[k + 1]
    a = condition_mat >= condition_below
    b = condition_mat <= condition_above
    a_b = a * b
    r_focus = r_mat * np.ma.masked_equal(a_b, 0)
    a_b_mat = np.tile(np.reshape(a_b, (N_1, len(data_keep), 1)), (1, 1, Nc_cut))
    masked = np.ma.masked_equal(a_b_mat, 0)
    # condition_where = np.where(a * b == 1)
    drift_c_focus = drift_c_mat * masked
    diffusion_c_focus = diffusion_P_mat * masked
    # parti_focus = parti_mat * masked
    for l in range(n_bins - 1):
        age_below = bins[l]
        age_above = bins[l + 1]
        results_data[i, k, l, 0] = np.nanmean(drift_c_focus[:, :, age_below:age_above])
        results_data[i, k, l, 1] = np.nanmean(drift_P_focus[:, :, age_below:age_above])
        results_data[i, k, l, 2] = np.nanmean(drift_N_focus[:, :, age_below:age_above])
        # results_data[i, k, l, 1] = np.nanstd(drift_c_focus[:, :, age_below:age_above])
        results_data[i, k, l, 3] = np.nanmean(diffusion_c_focus[:, :, age_below:age_above])
        results_data[i, k, l, 4] = np.nanmean(diffusion_P_focus[:, :, age_below:age_above])
        results_data[i, k, l, 5] = np.nanmean(r_focus)
for l in range(n_bins - 1):
    age_below = bins[l]
    age_above = bins[l + 1]
    results_data_uncon[l, 0] = np.nanmean(drift_c_mat[:, :, age_below:age_above])
    results_data_uncon[l, 1] = np.nanmean(drift_P_mat[:, :, age_below:age_above])
    results_data_uncon[l, 2] = np.nanmean(drift_N_mat[:, :, age_below:age_above])
    results_data_uncon[l, 3] = np.nanmean(diffusion_c_mat[:, :, age_below:age_above])
    results_data_uncon[l, 4] = np.nanmean(diffusion_P_mat[:, :, age_below:age_above])
    results_data_uncon[l, 5] = np.nanmean(r_mat)

var_name = r'log$\left(c_{s,t}\right)$'
quartile_labels = [', 1st quartile', ', 2nd quartile', ', 3rd quartile', ', 4th quartile']
fig_titles = [r'Drift, $1^{st}$ quartile', r'Drift, $4^{th}$ quartile', 'Diffusion']
X_ = np.linspace(2.5, 100, 200)
scenario = scenarios[1]
fig, axes = plt.subplots(ncols=3, sharex='all', sharey='col', figsize=(20, 7))
for k, ax in enumerate(axes):
    ax.set_xlabel('Age')
    ax.set_title(fig_titles[k])
    for j in range(2):
        m = j if j == 0 else 3
        drift_focus = results_data[m]
        X_Y_Spline = make_interp_spline(x, drift_focus)
        Y_ = X_Y_Spline(X_)
        drift_c_focus = Y_[:, 0]
        drift_P_focus = Y_[:, 1]
        drift_N_focus = Y_[:, 2]
        diffusion_c_focus = Y_[:, 3]
        diffusion_P_focus = Y_[:, 4]
        r_rho_focus = Y_[:, 5] - rho
        if k == 0:
            if j == 0:
                ax.plot(X_, drift_c_focus, color=colors[m], label='Overall', alpha=0.8)
                ax.plot(X_, drift_P_focus, color=colors[m], label='Participants', alpha=0.8,
                        linestyle='dashdot')
                ax.plot(X_, drift_N_focus, color=colors[m], label='Nonparticipants', alpha=0.8,
                        linestyle='dashed')
                ax.axhline(r_rho_focus[0], 0.05, 0.95, color='gray', label=r'Average $r_t - \rho$',
                           alpha=0.4)
            ax.text(12, 0.005, 'Low', size=12,
                    bbox={'facecolor': 'w', 'alpha': 0.2, 'pad': 0.5, 'boxstyle': 'larrow'})
            ax.text(36, 0.005, 'Participation rate', size=12)
            ax.text(80, 0.005, 'High', size=12,
                    bbox={'facecolor': 'w', 'alpha': 0.2, 'pad': 0.5, 'boxstyle': 'rarrow'})
            ax.set_ylabel('Average drift of ' + var_name, color='black')
            ax.set_ylim([0.0, 0.04])
            ax.legend()
            ax.tick_params(axis='y', labelcolor='black')
        if k == 1:
            if j == 1:
                ax.plot(X_, drift_c_focus, color=colors[m], label='Overall', alpha=0.8)
                ax.plot(X_, drift_P_focus, color=colors[m], label='Participants', alpha=0.8,
                        linestyle='dashdot')
                ax.plot(X_, drift_N_focus, color=colors[m], label='Nonparticipants', alpha=0.8,
                        linestyle='dashed')
                ax.axhline(r_rho_focus[0], 0.05, 0.95, color='gray', label=r'Average $r_t - \rho$',
                           alpha=0.4)
            ax.text(12, 0.005, 'High', size=12,
                    bbox={'facecolor': 'w', 'alpha': 0.2, 'pad': 0.5, 'boxstyle': 'larrow'})
            ax.text(36, 0.005, 'Participation rate', size=12)
            ax.text(80, 0.005, 'Low', size=12,
                    bbox={'facecolor': 'w', 'alpha': 0.2, 'pad': 0.5, 'boxstyle': 'rarrow'})
            # ax.set_ylabel('Average drift of ' + var_name, color='black')
            # ax.legend()
            # ax.tick_params(axis='y', labelcolor='black')
            ax.set_ylim([0.0, 0.04])
        if k == 2:
            ax.plot(X_, diffusion_c_focus, color=colors[m], alpha=0.8, label='Overall' + quartile_labels[m])
            ax.plot(X_, diffusion_P_focus, color=colors[m], alpha=0.8, label='Participants' + quartile_labels[m],
                    linestyle='dashdot')
            ax.axhline(0, 0.05, 0.95, color=colors[m], label='Nonparticipants', alpha=0.8,
                       linestyle='dashed')
            ax.set_ylabel('Average volatility of ' + var_name, color='black')
            ax.legend()
            ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig('belief_log consumption and age.png', dpi=60)
# plt.savefig('10' + scenario[1] + 'log consumption and age.png', dpi=60)
# plt.show()
# plt.close()
fig, axes = plt.subplots(ncols=2, sharex='all', sharey='col', figsize=(15, 7))
for k, ax in enumerate(axes):
    ax.set_xlabel('Age')
    fig_title = 'Drift' if k == 0 else 'Diffusion'
    ax.set_title(fig_title)
    drift_focus = results_data_uncon
    X_Y_Spline = make_interp_spline(x, drift_focus)
    Y_ = X_Y_Spline(X_)
    drift_c_focus = Y_[:, 0]
    drift_P_focus = Y_[:, 1]
    drift_N_focus = Y_[:, 2]
    diffusion_c_focus = Y_[:, 3]
    diffusion_P_focus = Y_[:, 4]
    r_rho_focus = Y_[:, 5] - rho
    if k == 0:
        ax.plot(X_, drift_c_focus, color=colors[5], label='Overall', alpha=0.8)
        ax.plot(X_, drift_P_focus, color=colors[5], label='Participants', alpha=0.8,
                linestyle='dashdot')
        ax.plot(X_, drift_N_focus, color=colors[5], label='Nonparticipants', alpha=0.8,
                linestyle='dashed')
        ax.axhline(r_rho_focus[0], 0.05, 0.95, color='gray', label=r'Average $r_t - \rho$',
                   alpha=0.4)
        ax.set_ylabel('Average drift of ' + var_name, color='black')
        ax.set_ylim([0.0, 0.04])
        ax.legend()
        ax.tick_params(axis='y', labelcolor='black')
    if k == 1:
        ax.plot(X_, diffusion_c_focus, color=colors[5], alpha=0.8, label='Overall')
        ax.plot(X_, diffusion_P_focus, color=colors[5], alpha=0.8, label='Participants',
                linestyle='dashdot')
        ax.axhline(0, 0.05, 0.95, color=colors[5], label='Nonparticipants', alpha=0.8,
                   linestyle='dashed')
        ax.set_ylabel('Average volatility of ' + var_name, color='black')
        ax.legend()
        ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig('belief_uncon_log consumption and age.png', dpi=60)
# plt.savefig('10_uncon' + scenario[1] + 'log consumption and age.png', dpi=60)
# plt.show()
# plt.close()


# ######################################
# ############  Figure 4  ##############
# ############ Figure 11  ##############
# ########### & Figure 15  #############
# ######################################
print('Figure 4, 11, and 15')
# fig 4: phi = 0.0, 0.4, & 0.8, complete vs. reentry
# fig 11: tau = 0.01 and tau = 0.015, phi == 0, reentry
# fig 15: phi == 0, initial window = 60 and 240, complete vs. reentry
n_scenarios_short = 2  # complete vs. reentry
scenarios_short = scenarios[:n_scenarios_short]
t_gap = int(2 / dt)  # 2-year non-overlapping rolling window
t_rolling_pre = np.arange(0, Nt - 1, t_gap)[:-1]  # pre
t_rolling_post = t_rolling_pre + t_gap
n_gap = len(t_rolling_pre)
shocks = np.cumsum(dZ_matrix, axis=1)
shocks_mat = shocks[:, t_rolling_post] - shocks[:, t_rolling_pre]
shocks_SI = np.cumsum(dZ_SI_matrix, axis=1)
shocks_SI_mat = shocks_SI[:, t_rolling_post] - shocks_SI[:, t_rolling_pre]
parti_rate_pre_mat = np.empty((Mpath, n_phi_short, n_gap), dtype=np.float32)
update_belief_mat = np.empty((Mpath, n_scenarios_short, n_phi_short, n_gap), dtype=np.float32)
cohort_size_short = cohort_size[1:]
cohort_size_short_mat = np.tile(cohort_size_short, (Nt - 1, 1))
cohort_size_mat = np.tile(cohort_size, (Nt, 1))
tau_mat = np.tile(tau, (Nt, 1))
popu_fig11 = 0.1
cutoff_age_old_below_fig11 = np.searchsorted(cummu_popu, popu_fig11)
cutoff_age_young_fig11 = np.searchsorted(cummu_popu, 1 - popu_fig11)
tax_short = [0.01, 0.015]
n_tax_short = len(tax_short)
P_old_compare = np.empty((Mpath, n_tax_short, Nt), dtype=np.float32)
P_young_compare = np.empty((Mpath, n_tax_short, Nt), dtype=np.float32)
belief_popu_old_compare_fig11 = np.empty((Mpath, n_tax_short, Nt), dtype=np.float32)
belief_popu_young_compare_fig11 = np.empty((Mpath, n_tax_short, Nt), dtype=np.float32)
belief_popu_fig11 = np.empty((Mpath, n_tax_short, Nt), dtype=np.float32)
Wealthshare_old_compare = np.empty((Mpath, n_tax_short, Nt), dtype=np.float32)
Wealthshare_young_compare = np.empty((Mpath, n_tax_short, Nt), dtype=np.float32)
Npres_short = [60, 240]
n_Npres_short = len(Npres_short)
popu_fig15 = 0.5
cutoff_age_old_below_fig15 = np.searchsorted(cummu_popu, popu_fig15)
cutoff_age_young_fig15 = np.searchsorted(cummu_popu, 1 - popu_fig15)
belief_popu_fig15 = np.empty((Mpath, n_Npres_short, Nt), dtype=np.float32)
Phi_compare = np.empty((Mpath, n_scenarios_short, n_Npres_short, Nt), dtype=np.float32)
Delta_bar_compare = np.empty((Mpath, n_scenarios_short, n_Npres_short, Nt), dtype=np.float32)
belief_popu_fig15 = np.empty((Mpath, n_Npres_short, Nt), dtype=np.float32)
Phi_old_compare = np.empty((Mpath, n_scenarios_short, n_Npres_short, Nt), dtype=np.float32)
Phi_young_compare = np.empty((Mpath, n_scenarios_short, n_Npres_short, Nt), dtype=np.float32)
belief_f_old_compare = np.empty((Mpath, n_scenarios_short, n_Npres_short, Nt), dtype=np.float32)  # of participants
belief_f_young_compare = np.empty((Mpath, n_scenarios_short, n_Npres_short, Nt), dtype=np.float32)  # of participants

# run
for i in range(4000):
# for i in range(Mpath):
    if np.mod(i, 50) == 0:
        print(i)
    dZ = dZ_matrix[i]
    dZ_build = dZ_build_matrix[i]
    dZ_SI = dZ_SI_matrix[i]
    dZ_SI_build = dZ_SI_build_matrix[i]
    for j, scenario in enumerate(scenarios_short):
        scenario = scenarios[j]
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        for n, phi_try in enumerate(phi_vector_short):
            for k, tax_try in enumerate(tax_short):
                beta_try = rho + nu - tax_try
                for m, Npre_try in enumerate(Npres_short):
                    T_hat_try = Npre_try * dt
                    Vhat_try = (sigma_Y ** 2) / T_hat_try  # prior variance
                    if phi_try == 0:
                        # if (tax_try == 0.015 and mode_trade != 'w_constraint') or (tax_try == 0.015 and Npre_try == 60):
                        #     pass
                        if (tax_try == 0.015) or (Npre_try == 60):
                            pass
                        else:
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
                            ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu,
                                            Vhat_try,
                                            mu_Y, sigma_Y, sigma_S,
                                            tax_try,
                                            beta,
                                            phi_try,
                                            Npre_try, Ninit,
                                            T_hat,
                                            dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                                            need_f='True',
                                            need_Delta='True',
                                            need_pi='True',
                                            )
                            # if tax_try == 0.015 and mode_trade == 'w_constraint':
                                # save results for fig 11
                                # belief_popu_fig15[i, k] = np.average(Delta, weights=cohort_size_mat,
                                #                                      axis=1)  # same average belief as phi == 0
                                # P_old_compare[i, k] = np.sum(
                                #     invest_tracker[:, :cutoff_age_old_below_fig11] *
                                #     cohort_size_mat[:, :cutoff_age_old_below_fig11],
                                #     axis=1) / popu_fig11
                                # P_young_compare[i, k] = np.sum(invest_tracker[:, cutoff_age_young_fig11:] *
                                #                                cohort_size_mat[:, cutoff_age_young_fig11:],
                                #                                axis=1) / popu_fig11
                                # Wealthshare_old_compare[i, k] = np.sum(
                                #     f[:, :cutoff_age_old_below_fig11] * dt, axis=1)
                                # Wealthshare_young_compare[i, k] = np.sum(f[:, cutoff_age_young_fig11:] * dt, axis=1)
                                # belief_popu_old_compare_fig11[i, k] = np.average(
                                #     Delta[:, :cutoff_age_old_below_fig11],
                                #     weights=cohort_size_mat[:, :cutoff_age_old_below_fig11],
                                #     axis=1)
                                # belief_popu_young_compare_fig11[i, k] = np.average(Delta[:, cutoff_age_young_fig11:],
                                #                                                    weights=cohort_size_mat[:,
                                #                                                            cutoff_age_young_fig11:],
                                #                                                    axis=1)


                            # elif Npre_try == 60:
                                # save results for fig 15
                                # Phi_compare[i, j, m] = f_parti
                                # Delta_bar_compare[i, j, m] = Delta_bar_parti
                                # if j == 0:
                                #     belief_popu_fig15[i, m] = np.average(Delta, weights=cohort_size_mat,
                                #                                          axis=1)  # same average belief as phi == 0
                                #     belief_f_old_compare[i, j, m] = np.average(Delta[:, :cutoff_age_old_below_fig15],
                                #                                                weights=f[:,
                                #                                                        :cutoff_age_old_below_fig15] * dt,
                                #                                                axis=1)
                                #     belief_f_young_compare[i, j, m] = np.average(Delta[:, cutoff_age_young_fig15:],
                                #                                                  weights=f[:,
                                #                                                          cutoff_age_young_fig15:] * dt,
                                #                                                  axis=1)
                                #     Phi_old_compare[i, j, m] = np.sum(f[:, :cutoff_age_old_below_fig15] * dt, axis=1)
                                #     Phi_young_compare[i, j, m] = np.sum(f[:, cutoff_age_young_fig15:] * dt, axis=1)
                                # else:
                                #     parti = pi > 0
                                #     belief_f_old_compare[i, j, m] = np.ma.average(Delta[:, :cutoff_age_old_below_fig15],
                                #                                                   weights=f[:,
                                #                                                           :cutoff_age_old_below_fig15] * parti[
                                #                                                                                          :,
                                #                                                                                          :cutoff_age_old_below_fig15] * dt,
                                #                                                   axis=1)
                                #     belief_f_young_compare[i, j, m] = np.ma.average(Delta[:, cutoff_age_young_fig15:],
                                #                                                     weights=f[:,
                                #                                                             cutoff_age_young_fig15:] * parti[
                                #                                                                                        :,
                                #                                                                                        cutoff_age_young_fig15:] * dt,
                                #                                                     axis=1)
                                #     Phi_old_compare[i, j, m] = np.sum(parti[:, :cutoff_age_old_below_fig15]
                                #                                       * f[:, :cutoff_age_old_below_fig15] * dt,
                                #                                       axis=1)
                                #     Phi_young_compare[i, j, m] = np.sum(parti[:, cutoff_age_young_fig15:]
                                #                                         * f[:, cutoff_age_young_fig15:] * dt,
                                #                                         axis=1)
                            # else:
                                # save results for fig 4
                            update_belief = Delta[1:, :-1] - Delta[:-1, 1:]  # change of belief
                            update_belief_t = np.average(update_belief, weights=cohort_size_short, axis=1)  # everyone
                            update_belief_cumsum = np.cumsum(update_belief_t)
                            update_belief_mat[i, j, n] = update_belief_cumsum[t_rolling_post - 1] - \
                                                             update_belief_cumsum[
                                                                 t_rolling_pre - 1]
                            if mode_trade == 'w_constraint':
                                parti_rate_pre_mat[i, n] = popu_parti[t_rolling_pre]

                                # # save results for fig 11
                                # belief_popu_fig11[i, k] = np.average(Delta, weights=cohort_size_mat,
                                #                                      axis=1)  # same average belief as phi == 0
                                # P_old_compare[i, k] = np.sum(
                                #     invest_tracker[:, :cutoff_age_old_below_fig11] *
                                #     cohort_size_mat[:, :cutoff_age_old_below_fig11],
                                #     axis=1) / popu_fig11
                                # P_young_compare[i, k] = np.sum(invest_tracker[:, cutoff_age_young_fig11:] *
                                #                                cohort_size_mat[:, cutoff_age_young_fig11:],
                                #                                axis=1) / popu_fig11
                                # Wealthshare_old_compare[i, k] = np.sum(
                                #     f[:, :cutoff_age_old_below_fig11] * dt, axis=1)
                                # Wealthshare_young_compare[i, k] = np.sum(f[:, cutoff_age_young_fig11:] * dt, axis=1)
                                # belief_popu_old_compare_fig11[i, k] = np.average(
                                #     Delta[:, :cutoff_age_old_below_fig11],
                                #     weights=cohort_size_mat[:, :cutoff_age_old_below_fig11],
                                #     axis=1)
                                # belief_popu_young_compare_fig11[i, k] = np.average(Delta[:, cutoff_age_young_fig11:],
                                #                                                    weights=cohort_size_mat[:,
                                #                                                            cutoff_age_young_fig11:],
                                #                                                    axis=1)
                                #
                                # # save results for fig 15
                                # Phi_compare[i, j, m] = f_parti
                                # Delta_bar_compare[i, j, m] = Delta_bar_parti
                                # if j == 0:
                                #     belief_popu_fig15[i, m] = np.average(Delta, weights=cohort_size_mat,
                                #                                          axis=1)  # same average belief as phi == 0
                                #     belief_f_old_compare[i, j, m] = np.average(Delta[:, :cutoff_age_old_below_fig15],
                                #                                                weights=f[:,
                                #                                                        :cutoff_age_old_below_fig15] * dt,
                                #                                                axis=1)
                                #     belief_f_young_compare[i, j, m] = np.average(Delta[:, cutoff_age_young_fig15:],
                                #                                                  weights=f[:,
                                #                                                          cutoff_age_young_fig15:] * dt,
                                #                                                  axis=1)
                                #     Phi_old_compare[i, j, m] = np.sum(f[:, :cutoff_age_old_below_fig15] * dt, axis=1)
                                #     Phi_young_compare[i, j, m] = np.sum(f[:, cutoff_age_young_fig15:] * dt, axis=1)
                                # else:
                                #     parti = pi > 0
                                #     belief_f_old_compare[i, j, m] = np.ma.average(Delta[:, :cutoff_age_old_below_fig15],
                                #                                                   weights=f[:,
                                #                                                           :cutoff_age_old_below_fig15] * parti[
                                #                                                                                          :,
                                #                                                                                          :cutoff_age_old_below_fig15] * dt,
                                #                                                   axis=1)
                                #     belief_f_young_compare[i, j, m] = np.ma.average(Delta[:, cutoff_age_young_fig15:],
                                #                                                     weights=f[:,
                                #                                                             cutoff_age_young_fig15:] * parti[
                                #                                                                                        :,
                                #                                                                                        cutoff_age_young_fig15:] * dt,
                                #                                                     axis=1)
                                #     Phi_old_compare[i, j, m] = np.sum(parti[:, :cutoff_age_old_below_fig15]
                                #                                       * f[:, :cutoff_age_old_below_fig15] * dt,
                                #                                       axis=1)
                                #     Phi_young_compare[i, j, m] = np.sum(parti[:, cutoff_age_young_fig15:]
                                #                                         * f[:, cutoff_age_young_fig15:] * dt,
                                #                                         axis=1)


                    else:
                        if tax_try == 0.015 or Npre_try == 60:
                            pass
                        else:
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
                            ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu,
                                            Vhat_try,
                                            mu_Y, sigma_Y, sigma_S,
                                            tax_try,
                                            beta,
                                            phi_try,
                                            Npre_try, Ninit,
                                            T_hat,
                                            dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                                            need_f='False',
                                            need_Delta='True',
                                            need_pi='False',
                                            )
                            # save results for fig 4
                            update_belief = Delta[1:, :-1] - Delta[:-1, 1:]  # change of belief
                            update_belief_t = np.average(update_belief, weights=cohort_size_short, axis=1)  # everyone
                            update_belief_cumsum = np.cumsum(update_belief_t)
                            update_belief_mat[i, j, n] = update_belief_cumsum[t_rolling_post - 1] - \
                                                         update_belief_cumsum[
                                                             t_rolling_pre - 1]
                            if mode_trade == 'w_constraint':
                                parti_rate_pre_mat[i, n] = popu_parti[t_rolling_pre]

# Figure 4
# for each phi value:
    # bins of participation rate on the x axis
    # covariance between dZ^Y or dZ^SI and the change in belief conditional on bin
x_var = parti_rate_pre_mat[:keep_until]
y_var = update_belief_mat[:keep_until]
condition_var1 = shocks_mat[:keep_until]
condition_var2 = shocks_SI_mat[:keep_until]
n_bins = 10
data_figure_mean = np.zeros((n_phi_short, n_bins - 1))
data_figure_std = np.zeros((n_phi_short, n_bins - 1))
data_figure_cov = np.zeros((2, n_phi_short, n_bins - 1))
data_figure_x = np.zeros((n_phi_short, n_bins - 1))
data_figure_complete_cov = np.zeros((2, n_phi_short))
data_figure_complete_std = np.zeros((n_phi_short))
# reentry scenario
for k in range(n_phi_short):
    data_focus = y_var[:, 1, k]  # for the reentry scenario
    x_focus = x_var[:, k]
    below_dz = np.percentile(x_focus, 10)
    above_dz = np.percentile(x_focus, 90)
    bins = np.linspace(below_dz, above_dz, n_bins)
    bin_size = (above_dz - below_dz) / (n_bins - 1)
    data_figure_x[k] = np.linspace(below_dz + bin_size / 2, above_dz - bin_size / 2, n_bins - 1)
    data_complete = y_var[:, 0, k]
    data_figure_complete_std[k] = np.std(np.abs(data_complete))
    data_figure_complete_cov[0, k] = np.corrcoef(np.reshape(data_complete, (1, -1))
                                          , np.reshape(condition_var1, (1, -1)))[0, 1]
    data_figure_complete_cov[1, k] = np.corrcoef(np.reshape(data_complete, (1, -1))
                                          , np.reshape(condition_var2, (1, -1)))[0, 1]
    for j in range(n_bins - 1):
        bin_0 = bins[j]
        bin_1 = bins[j + 1]
        below_bin = bin_1 >= x_focus
        above_bin = x_focus >= bin_0
        bin_where = np.where(below_bin * above_bin == 1)
        data_focus_z = data_focus[bin_where]
        condition_var1_focus = condition_var1[bin_where]
        condition_var2_focus = condition_var2[bin_where]
        data_figure_mean[k, j] = np.mean(np.abs(data_focus_z))
        data_figure_cov[0, k, j] = np.corrcoef(data_focus_z, condition_var1_focus)[0, 1]
        data_figure_cov[1, k, j] = np.corrcoef(data_focus_z, condition_var2_focus)[0, 1]
        data_figure_std[k, j] = np.std(np.abs(data_focus_z))

label_shock = [r'Shocks to the output, $dz^{Y}$', r'Shocks to the signal, $dz^{SI}$']
labels = [r'$\phi = 0.0$', r'$\phi = 0.4$', r'$\phi = 0.8$']
X_ = np.linspace(data_figure_x[:, 0].max(), data_figure_x[:, -1].min(), 5)
X_gap = np.linspace(X_.max(), 0.99, 2)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey='all')
for i, ax in enumerate(axes):
    for l in range(n_phi_short):
        y = data_figure_cov[i, l]
        x = data_figure_x[l]
        y_complete = data_figure_complete[i, l]
        X_Y_Spline = make_interp_spline(x, y)
        Y_ = X_Y_Spline(X_)
        if i == 0:
            # ax.plot(x, y, color=colors_short[l], linewidth=0.8, label=labels[l])
            ax.plot(X_, Y_, color=colors_short[l], linewidth=0.8, label=labels[l])
        else:
            # ax.plot(x, y, color=colors_short[l], linewidth=0.8)
            ax.plot(X_, Y_, color=colors_short[l], linewidth=0.8)
        ax.scatter(1, y_complete, color=colors_short[l], marker='o')
        ax.plot(X_gap, [Y_[-1], y_complete], color=colors_short[l], linewidth=0.8, linestyle='dashed')
        ax.axhline(0, 0.05, 0.95, color='gray', linestyle='dashed', linewidth=0.6)
        if i == 0:
            ax.legend()
    ax.set_xlabel('Participation rate in the economy')
    ax.set_ylim(-1, 1)
    ax.set_ylabel(r'Covariance with change in average belief')
    ax.set_title(label_shock[i])
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
# plt.savefig('Endogenous_learning.png', dpi=100)
plt.show()
# plt.close()

# x_var = parti_rate_pre
# # y_var = update_belief
# y_var = np.empty((N_1 * 6, 2, n_phi, N_gap))
# for i in range(n_phi):
#     for j in range(2):
#         y_focus = update_belief[:, j, i]
#         y_var[:, j, i] = (y_focus - np.mean(y_focus)) / np.std(y_focus)
# n_bins = 5
# y_percentiles = [50, 25, 75]
# data_figure_y = np.zeros((2, 2, 3, n_bins - 1, len(y_percentiles)))
# data_figure_x = np.zeros((2, 2, 3, n_bins - 1))
# data_figure_complete = np.zeros((2, 2, 3, 3, len(y_percentiles)))
# # reentry scenario
# condition_var1 = shocks
# condition_var2 = shocks_SI
# for i1 in range(2):
#     data_where1 = condition_var1 >= np.percentile(condition_var1, 75) if i1 == 0 else condition_var1 < np.percentile(condition_var1, 25)
#     for i2 in range(2):
#         data_where2 = condition_var2 >= np.percentile(condition_var2, 75) if i2 == 0 else condition_var2 < np.percentile(condition_var2, 25)
#         data_where = np.where(data_where1 * data_where2 == 1)
#         for k in range(n_phi):
#             data_focus = y_var[:, 1, k][data_where]
#             x_focus = x_var[:, k][data_where]
#             below_dz = np.percentile(x_focus, 10)
#             above_dz = np.percentile(x_focus, 90)
#             bins = np.linspace(below_dz, above_dz, n_bins)
#             bin_size = (above_dz - below_dz) / (n_bins - 1)
#             data_figure_x[i1, i2, k] = np.linspace(below_dz + bin_size / 2, above_dz - bin_size / 2, n_bins - 1)
#             data_complete = y_var[:, 0, k][data_where]
#             data_figure_complete[i1, i2, k] = np.percentile(data_complete, y_percentiles)
#
#             for j in range(n_bins - 1):
#                 bin_0 = bins[j]
#                 bin_1 = bins[j + 1]
#                 below_bin = bin_1 >= x_focus
#                 above_bin = x_focus >= bin_0
#                 bin_where = np.where(below_bin * above_bin == 1)
#                 data_focus_z = data_focus[bin_where]
#                 data_figure_y[i1, i2, k, j] = np.percentile(data_focus_z, y_percentiles)
# label_np = ['Good ', 'Bad ']
# label_shock = [r'$dz^{Y}$, ', r'signal, $dz^{SI}$']
# labels = [r'$\phi = 0.0$', r'$\phi = 0.4$', r'$\phi = 0.8$']
# X_ = np.linspace(x.min(), x.max(), 20)
# X_complete = np.linspace(0.99, 1.01, 3)
# X_gap = np.linspace(x.max(), 0.99, 2)
# fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10), sharey='all')
# for j, row in enumerate(axes):
#     for k, ax in enumerate(row):
#         for l in range(3):
#             y = data_figure_y[j, k, l]
#             x = data_figure_x[j, k, l]
#             y_complete = data_figure_complete[j, k, l]
#             Y_mat = np.empty((3, 20))
#             for m in range(3):
#                 X_Y_Spline = make_interp_spline(x, y[:, m])
#                 Y_mat[m] = X_Y_Spline(X_)
#             if j == k == 0:
#                 ax.plot(X_, Y_mat[0], color=colors[l], linewidth=0.8, label=labels[l])
#             else:
#                 ax.plot(X_, Y_mat[0], color=colors[l], linewidth=0.8)
#             ax.plot(X_complete, y_complete[:, 0], color=colors[l], linewidth=0.8)
#             ax.plot(X_gap, [Y_mat[0, -1], y_complete[0, 0]], color=colors[l], linewidth=0.8, linestyle='dashed')
#             ax.fill_between(X_, Y_mat[1], Y_mat[2], color=colors[l], linewidth=0., alpha=0.3)
#             ax.fill_between(X_complete, y_complete[:, 1], y_complete[:, 2], color=colors[l], linewidth=0., alpha=0.3)
#             ax.axhline(0, 0.05, 0.95, color='gray', linestyle='dashed', linewidth=0.6)
#             if j == k == 0:
#                 ax.legend()
#         ax.set_xlabel('Participation rate in the economy')
#         ax.set_ylabel(r'Changes in average estimation error, $\Delta$')
#         ax.set_title(label_np[j] + r'fundamental $dz^{Y}$, ' + label_np[k] + r'signal $dz^{SI}$')
# fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
# # plt.savefig('Endogenous_learning.png', dpi=100)
# plt.show()
# # plt.close()

# figure 11
for i in range(N_paths):
    print(i)
    dZ_build = np.random.randn(Nt) * np.sqrt(dt)
    dZ = np.random.randn(Nt) * np.sqrt(dt)
    dZ_SI_build = np.random.randn(Nt) * np.sqrt(dt)
    dZ_SI = np.random.randn(Nt) * np.sqrt(dt)
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
                    phi_fix,
                    Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                    need_f='True',
                    need_Delta='True',
                    need_pi='False',
                    )
    theta_compare[i] = theta
    Phi_compare[i] = f_parti
    Delta_bar_compare[i] = Delta_bar_parti
    for j, popu in enumerate(popus):
        # cutoff_age_old = np.searchsorted(cummu_popu, popu)
        cutoff_age_old_top = np.searchsorted(cummu_popu, popu * 2)
        cutoff_age_old_below = np.searchsorted(cummu_popu, popu)
        cutoff_age_young = np.searchsorted(cummu_popu, 1 - popu)
        total_popu_old = np.sum(cohort_size[cutoff_age_old_below:cutoff_age_old_top])
        total_popu_young = np.sum(cohort_size[cutoff_age_young:])
        P_old_compare[i, j] = np.sum(invest_tracker[:, cutoff_age_old_below:cutoff_age_old_top] *
                                     cohort_size_mat[:, cutoff_age_old_below:cutoff_age_old_top],
                                  axis=1) / total_popu_old
        P_young_compare[i, j] = np.sum(invest_tracker[:, cutoff_age_young:] *
                                       cohort_size_mat[:, cutoff_age_young:],
                                    axis=1) / total_popu_young
        Phi_old_compare[i, j] = np.sum(f[:, cutoff_age_old_below:cutoff_age_old_top] *
                                       invest_tracker[:, cutoff_age_old_below:cutoff_age_old_top] * dt, axis=1)
        Phi_young_compare[i, j] = np.sum(f[:, cutoff_age_young:] * invest_tracker[:, cutoff_age_young:] * dt, axis=1)
        Wealthshare_old_compare[i, j] = np.sum(f[:, cutoff_age_old_below:cutoff_age_old_top] * dt, axis=1)
        Wealthshare_young_compare[i, j] = np.sum(f[:, cutoff_age_young:] * dt, axis=1)
        belief_f_old_compare[i, j] = np.average(Delta[:, cutoff_age_old_below:cutoff_age_old_top],
                                             weights=f[:, cutoff_age_old_below:cutoff_age_old_top] * dt,
                                             axis=1)
        belief_f_young_compare[i, j] = np.average(Delta[:, cutoff_age_young:],
                                               weights=f[:, cutoff_age_young:] * dt,
                                               axis=1)
        belief_popu_old_compare[i, j] = np.average(Delta[:, cutoff_age_old_below:cutoff_age_old_top],
                                                weights=cohort_size_mat[:, cutoff_age_old_below:cutoff_age_old_top],
                                                axis=1)
        belief_popu_young_compare[i, j] = np.average(Delta[:, cutoff_age_young:],
                                                  weights=cohort_size_mat[:, cutoff_age_young:],
                                                  axis=1)

# construct the condition:
n_tiles = 4
n_bins = 30
popu_index = 0
belief_f_gap_compare = belief_f_old_compare - belief_f_young_compare
belief_popu_gap_compare = belief_popu_old_compare - belief_popu_young_compare
cutoff_belief = -theta_compare
belief_f_distance_young = belief_f_young_compare - cutoff_belief
belief_f_distance_old = belief_f_old_compare - cutoff_belief
belief_popu_distance_young = belief_popu_young_compare - cutoff_belief
belief_popu_distance_old = belief_popu_old_compare - cutoff_belief
parti_gap = P_old_compare - P_young_compare
Phi_gap = Phi_old_compare - Phi_young_compare
wealth_gap = Wealthshare_old_compare - Wealthshare_young_compare
# y_variables = [parti_gap[:, popu_index, :], belief_f_distance_young[:, popu_index, :], belief_f_distance_old[:, popu_index, :]
# y_complete_variables = [theta_complete, Phi_complete, Delta_bar_complete]
# x_mat = belief_f_gap_compare[:, popu_index, :]
# x_varname = r'Wealth weighted $\Delta_{s,t}$, old minus young'
x_mat = belief_popu_gap_compare[:, popu_index, :]
x_varname = r'Average estimation error $\Delta_{s,t}$, old minus young'
# x_mat = wealth_gap[:, popu_index, :]
# x_varname =  r'Wealth share old minus young'
x_range = 0.25
x_range_left = np.percentile(x_mat, 5)
x_range_right = np.percentile(x_mat, 95)
width_bins = (x_range_right - x_range_left) / n_bins
a = x_range_left <= x_mat
b = x_mat <= x_range_right
where_within = np.where(a * b == 1)  # winsorize
x_mat_within = x_mat[where_within]
total_count = np.shape(where_within)[1]
# y_varnames = ['Participation rate, old - young', 'Distance to cutoff belief, young', 'Distance to cutoff belief, old']
# condition_var = Wealthshare_young_compare[:, popu_index, :]
# condition_label = r'Wealth share young, quartile '
# condition_var = Phi_gap[:, popu_index, :]
# condition_label = r'$\Phi$ old-young, quartile '
condition_var = wealth_gap[:, popu_index, :]
condition_label = r'Wealth share, old minus young'
# condition_var = belief_f_gap_compare[:, popu_index, :]
# condition_label = r'Wealth weighted $\Delta_{s,t}$, old - young, Quartile'

condition_var_within = condition_var[where_within]
condition = np.percentile(condition_var_within, np.arange(0, 101, (100/n_tiles)))
# condition = (np.max(condition_var_within) - np.min(condition_var_within)) \
#             * np.arange(0, 101, (100/n_tiles)) / 100 + np.min(condition_var_within)


y = np.empty((n_tiles, n_bins, 3))
x = np.linspace(x_range_left + width_bins / 2, x_range_right - width_bins / 2, n_bins)
X_ = np.linspace(x_range_left, x_range_right, 50)
y_mat_within = parti_gap[:, popu_index, :][where_within]
y_varname = 'Participation rate, old minus young'
fig, ax = plt.subplots(figsize=(10, 8))
for i in range(n_tiles):
    below = condition[i]
    above = condition[i + 1]
    a = below < condition_var_within
    b = condition_var_within < above
    data_where = np.where(a * b == 1)
    x_var = x_mat_within[data_where]
    y_var = y_mat_within[data_where]
    for n in range(n_bins):
        bin_left = x_range_left + n * width_bins
        bin_right = bin_left + width_bins
        bin_1 = x_var <= bin_right
        bin_2 = x_var >= bin_left
        bin_where = np.where(bin_1 * bin_2 == 1)
        y[i, n, 0] = np.median(y_var[bin_where])
        y[i, n, 1] = np.percentile(y_var[bin_where], 25)
        y[i, n, 2] = np.percentile(y_var[bin_where], 75)
    Y_ = np.empty((3, 50))
    for m in range(3):
        y_i = y[i, :, m]
        X_Y_Spline = make_interp_spline(x, y_i)
        Y_[m] = X_Y_Spline(X_)
    # ax.plot(x, y[i, :, 0], linewidth=0.6, color=colors[i], label=condition_label + str(i + 1))
    # ax.fill_between(x, y[i, :, 1], y[i, :, 2], color=colors[i], linewidth=0., alpha=0.3)
    ax.plot(X_, Y_[0], linewidth=0.6, color=colors[i], label=condition_label + ', Quartile ' + str(i + 1))
    ax.fill_between(X_, Y_[1], Y_[2], color=colors[i], linewidth=0., alpha=0.3)
ax.axvline(0, 0.05, 0.95, linestyle='dashed', color='gray', linewidth=0.8)
ax.axhline(0, 0.05, 0.95, linestyle='dashed', color='gray', linewidth=0.8)
ax.legend(loc='upper left')
ax.set_xlabel(x_varname)
ax.set_ylabel(y_varname)
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig(str(tax)+'Intuition'+ x_varname[:4] + 'belief two sorts.png', dpi=200)
# plt.savefig('85-115old'+str(tax)+'Intuition'+ x_varname[:4] + 'belief two sorts.png', dpi=200)
plt.show()
# plt.close()


# distribution of wealth gap given the belief gap in [-x_range, x_range]:
n_bins = 30
fig, axes = plt.subplots(ncols=2, sharey = 'all', figsize=(10, 4))
for j, ax in enumerate(axes):
    if j == 0:
        condition_var_density = np.empty(n_bins)
        min_condition = np.min(condition_var_within)
        max_condition = np.max(condition_var_within)
        width_bins = (max_condition - min_condition) / n_bins
        condition_var_x = np.linspace(min_condition + width_bins / 2, max_condition - width_bins / 2, n_bins)
        for i in range(n_bins):
            bin_left = min_condition + i * width_bins
            bin_right = bin_left + width_bins
            bin_1 = condition_var_within <= bin_right
            bin_2 = condition_var_within >= bin_left
            bin_where = np.where(bin_1 * bin_2 == 1)
            condition_var_density[i] = np.shape(bin_where)[1] / total_count
        X_Y_Spline = make_interp_spline(condition_var_x, condition_var_density)
        X_ = np.linspace(min_condition, max_condition, 1000)
        Y_ = X_Y_Spline(X_)
        for i in range(n_tiles):
            if i > 0:
                ax.axvline(condition[i], 0.05, 0.95, linestyle='dashed', linewidth=0.8, color='gray')
            left_x = min_condition if i == 0 else condition[i]
            right_x = max_condition if i == n_tiles - 1 else condition[i + 1]
            a = X_ >= left_x
            b = right_x >= X_
            bin_where = np.where(a * b == 1)
            x = X_[bin_where]
            y = Y_[bin_where]
            ax.fill_between(x, 0, y, color=colors[i], linewidth=0., alpha=0.3, label='Quartile ' + str(i + 1))
        ax.legend(loc='upper left')
        ax.set_xlim(0, 0.45) if tax > 0.01 else ax.set_xlim(0, max_condition)
        ax.set_xlabel(condition_label)
        ax.set_ylabel('Density')
    else:
        width_bins = (x_range_right - x_range_left) / n_bins
        y = np.empty((n_tiles, n_bins))
        x = np.linspace(x_range_left + width_bins / 2, x_range_right - width_bins / 2, n_bins)
        y_bottom = 0
        for i in range(n_tiles):
            below = condition[i]
            above = condition[i + 1]
            a = below < condition_var_within
            b = condition_var_within < above
            data_where = np.where(a * b == 1)
            x_var = x_mat_within[data_where]
            # min_gap = np.min(x_var)
            # max_gap = np.max(x_var)
            # width_bins = (max_gap - min_gap) / n_bins
            for n in range(n_bins):
                # bin_left = min_gap + n * width_bins
                # bin_right = bin_left + width_bins
                bin_left = x_range_left + n * width_bins
                bin_right = bin_left + width_bins
                bin_1 = x_var <= bin_right
                bin_2 = x_var >= bin_left
                bin_where = np.where(bin_1 * bin_2 == 1)
                y[i, n] = np.shape(bin_where)[1] / total_count
            X_ = np.linspace(x_range_left, x_range_right, 100)
            X_Y_Spline = make_interp_spline(x, y[i])
            Y_ = X_Y_Spline(X_)
            y_top = y_bottom + Y_
            ax.fill_between(X_, y_top, y_bottom, linewidth=0., color=colors[i],
                            alpha=0.3)
            y_bottom = y_top
        # ax.legend(loc='upper right')
        ax.axvline(0, 0.05, 0.95, linestyle='dashed', linewidth=0.8, color='gray')
        # ax.set_ylim(top = 0.08)
        ax.set_xlabel(x_varname)
        # ax.set_ylabel('Density')
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig(str(tax)+'Intuition wealth distribution.png', dpi=200)
# plt.savefig('85-115old'+str(tax)+'Intuition wealth distribution.png', dpi=200)
plt.show()
# plt.close()


# figure 15
######################################################################
#### why is Delta_bar more volatile with the shorting constraint? ####
######################################################################
for i in range(N_paths):
    print(i)
    dZ_build = np.random.randn(Nt) * np.sqrt(dt)
    dZ = np.random.randn(Nt) * np.sqrt(dt)
    dZ_SI_build = np.random.randn(Nt) * np.sqrt(dt)
    dZ_SI = np.random.randn(Nt) * np.sqrt(dt)
    for k in range(2):
        scenario = scenarios[k]
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        for o, Npre_try in enumerate(Npres):
            T_hat_try = Npre_try * dt
            Vhat_try = (sigma_Y ** 2) / T_hat_try  # prior variance
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
            ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat_try, mu_Y, sigma_Y, sigma_S, tax, beta,
                            phi_fix,
                            Npre_try, Ninit, T_hat_try, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                            need_f='True',
                            need_Delta='True',
                            need_pi='True',
                            )
            theta_compare[i, k, o] = theta
            Phi_compare[i, k, o] = f_parti
            Delta_bar_compare[i, k, o] = Delta_bar_parti
            if k == 0:
                belief_f_old_compare[i, k, o] = np.average(Delta[:, :cutoff_age_old_below],
                                                           weights=f[:, :cutoff_age_old_below] * dt,
                                                           axis=1)
                belief_f_young_compare[i, k, o] = np.average(Delta[:, cutoff_age_young:],
                                                             weights=f[:, cutoff_age_young:] * dt,
                                                             axis=1)
                phi_old_compare[i, k, o] = np.sum(f[:, :cutoff_age_old_below] * dt, axis=1)
                phi_young_compare[i, k, o] = np.sum(f[:, cutoff_age_young:] * dt, axis=1)
            if k == 1:
                parti = pi > 0
                belief_f_old_compare[i, k, o] = np.ma.average(Delta[:, :cutoff_age_old_below],
                                                              weights=f[:, :cutoff_age_old_below] * parti[:,
                                                                                                    :cutoff_age_old_below] * dt,
                                                              axis=1)
                belief_f_young_compare[i, k, o] = np.ma.average(Delta[:, cutoff_age_young:],
                                                                weights=f[:, cutoff_age_young:] * parti[:,
                                                                                                  cutoff_age_young:] * dt,
                                                                axis=1)
                belief_popu_old_compare[i, o] = np.average(Delta[:, :cutoff_age_old_below],
                                                           weights=cohort_size_mat[:,
                                                                   :cutoff_age_old_below],
                                                           axis=1)
                belief_popu_young_compare[i, o] = np.average(Delta[:, cutoff_age_young:],
                                                             weights=cohort_size_mat[:, cutoff_age_young:],
                                                             axis=1)
                parti_old_compare[i, o] = np.sum(parti[:, :cutoff_age_old_below]
                                                 * cohort_size_mat[:, :cutoff_age_old_below],
                                                 axis=1)
                parti_young_compare[i, o] = np.sum(parti[:, cutoff_age_young:]
                                                   * cohort_size_mat[:, cutoff_age_young:],
                                                   axis=1)
                phi_old_compare[i, k, o] = np.sum(parti[:, :cutoff_age_old_below]
                                                  * f[:, :cutoff_age_old_below] * dt,
                                                  axis=1)
                phi_young_compare[i, k, o] = np.sum(parti[:, cutoff_age_young:]
                                                    * f[:, cutoff_age_young:] * dt,
                                                    axis=1)


# winsorize extreme shocks
Npre_index = 0
average_Delta_bar = np.mean(np.mean(Delta_bar_compare[:, :, Npre_index], axis=0), axis=1)
x_index = (belief_popu_old_compare[:, Npre_index] + belief_popu_young_compare[:, Npre_index]) / 2
x_label = 'Average estimation error'
below_dz = np.percentile(x_index, 10)
below_data = x_index >= below_dz
above_dz = np.percentile(x_index, 90)
above_data = above_dz >= x_index
data_where = np.where(below_data * above_data == 1)

# prepare data for the figure:
n_bins = 10
bins = np.linspace(below_dz, above_dz, n_bins)
# quartile_var = (Wealthshare_old_compare - Wealthshare_young_compare)[:, :, Npre_index]
quartile_var = Wealthshare_old_compare[:, :, Npre_index]
# quartile_var = belief_popu_young_compare[:, Npre_index] - belief_popu_old_compare[:, Npre_index]
# quartile_var = parti_young_compare[:, :, Npre_index]
y_percentiles = [50, 25, 75]
phi_old = phi_old_compare / (phi_old_compare + phi_young_compare)
phi_young = phi_young_compare / (phi_old_compare + phi_young_compare)
belief_f_young = np.where(belief_f_young_compare == 0, np.nan, belief_f_young_compare)  # converting empty cells from 0 to nan
belief_f_old = np.where(belief_f_old_compare == 0, np.nan, belief_f_old_compare)
data_var = Delta_bar_compare[:, :, Npre_index]
data_figure = np.zeros((2, n_bins - 1, len(y_percentiles)))
parti_figure = np.zeros((n_bins - 1, 2, len(y_percentiles)))
wealthshare_figure = np.zeros((n_bins - 1, 2, len(y_percentiles)))
phi_figure = np.zeros((2, 2, n_bins - 1, len(y_percentiles)))
belief_figure = np.zeros((2, 2, n_bins - 1, len(y_percentiles)))
belief_phi_figure = np.zeros((2, 2, n_bins - 1, len(y_percentiles)))
for i in range(2):
    data_focus = data_var[:, i]
    # quartile_var_focus = quartile_var[:, i]
    # # quartile_var_focus = quartile_var
    # quartiles = np.percentile(quartile_var_focus[data_where], np.linspace(0, 100, 5))
    phi_var_old = phi_old
    phi_var_young = phi_young[:, Npre_index]
    for j in range(n_bins - 1):
        bin_0 = bins[j]
        bin_1 = bins[j + 1]
        below_bin = bin_1 >= x_index
        above_bin = x_index >= bin_0
        bin_where = np.where(below_bin * above_bin == 1)
        # condition_bin = quartile_var_focus[bin_where]
        data_focus_z = data_focus[bin_where]
        data_figure[i, j] = np.percentile(data_focus_z, y_percentiles)
        for l in range(2):
            parti_var = parti_old_compare[:, Npre_index] if l == 0 else parti_young_compare[:, Npre_index]
            # wealthshare_var = Wealthshare_old_compare[:, 1, Npre_index] if l == 0 else Wealthshare_young_compare[:, 1, Npre_index]
            phi_var = phi_old[:, i, Npre_index] if l == 0 else phi_young[:, i, Npre_index]
            belief_var_nan = belief_f_old[:, i, Npre_index] if l == 0 else belief_f_young[:, i, Npre_index]
            belief_var = belief_f_old_compare[:, i, Npre_index] if l == 0 else belief_f_young_compare[:, i, Npre_index]
            # parti_figure[j, l, 0] = np.mean(parti_var[bin_where])
            # parti_figure[j, l, 1] = np.std(parti_var[bin_where])
            # wealthshare_figure[j, l, 0] = np.mean(wealthshare_var[bin_where])
            # wealthshare_figure[j, l, 1] = np.std(wealthshare_var[bin_where])
            # phi_figure[i, l, j, 0] = np.mean(phi_var[bin_where])
            # phi_figure[i, l, j, 1] = np.std(phi_var[bin_where])
            parti_figure[j, l] = np.percentile(parti_var[bin_where], y_percentiles)
            # wealthshare_figure[j, l] = np.percentile(wealthshare_var[bin_where], y_percentiles)
            phi_figure[i, l, j] = np.percentile(phi_var[bin_where], y_percentiles)
            belief_figure[i, l, j] = np.nanpercentile(belief_var_nan[bin_where], y_percentiles)
            belief_phi = phi_var * belief_var
            belief_phi_figure[i, l, j] = np.percentile(belief_phi[bin_where], y_percentiles)

        # # quartiles = np.percentile(condition_bin, np.linspace(0, 100, 5))
        # for k in range(1, 5):
        #     quartile_below = quartiles[k - 1]
        #     quartile_above = quartiles[k]
        #     below_bin = condition_bin <= quartile_above
        #     above_bin = condition_bin >= quartile_below
        #     bin_focus = np.where(above_bin * below_bin == 1)
        #     data_focus_bin = data_focus_z[bin_focus]
        #
        #     # data_figure[i1, i2, j, k, 0] = np.mean(data_focus_bin)
        #     # data_figure[i1, i2, j, k, 1] = np.percentile(data_focus_bin, 5)
        #     # data_figure[i1, i2, j, k, 2] = np.percentile(data_focus_bin, 95)
        #     data_figure[i, j, k] = np.percentile(data_focus_bin, y_percentiles)
        #     data_figure_variance[i, j, k, 0] = np.mean(data_focus_bin)
        #     data_figure_variance[i, j, k, 1] = np.sqrt(np.mean((data_focus_bin - average_Delta_bar[i]) ** 2))
bin_size = (above_dz - below_dz) / (n_bins - 1)
x = np.linspace(below_dz + bin_size / 2, above_dz - bin_size / 2, n_bins - 1)
labels = [[r'$\bar{\Delta}_t^{old}$', r'$\bar{\Delta}_t^{young}$'],
          [r'$\Phi_t^{old} / (\Phi_t^{old} + \Phi_t^{young})$', r'$\Phi_t^{young} / (\Phi_t^{old} + \Phi_t^{young})$'],
          [r'$\Phi_t^{old}\bar{\Delta}_t^{old} / (\Phi_t^{old} + \Phi_t^{young})$', r'$\Phi_t^{young}\bar{\Delta}_t^{young} / (\Phi_t^{old} + \Phi_t^{young})$']]
# labels = [r'Wealth old minus young, ' + 'Lowest quartile', 'Second quartile', 'Third quartile', 'Highest quartile']
sub_titles = ['Complete market', 'Reentry']
y_labels = ['Estimation error of the participants',
             'Wealth share of the participants',
            r'Contribution to $\bar{\Delta}_t$']
# X_ = np.linspace(-0.2, 0.2, 100)
X_ = np.linspace(below_dz, above_dz, 100)
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(10, 15), sharex='all', sharey='row')
for j, rows in enumerate(axes):
    for i, ax in enumerate(rows):
        if j == 0:
            y_focus = belief_figure[i]
        elif j == 1:
            y_focus = phi_figure[i]
        else:
            y_focus = belief_phi_figure[i]
        for k in range(2):
            if j == 2:
                y_i = y_focus[k]
                Y_ = np.empty((3, 100))
                for l in range(3):
                    X_Y_Spline = make_interp_spline(x, y_i[:, l])
                    Y_[l] = X_Y_Spline(X_)
                ax.plot(X_, Y_[0], color=colors_short[k], linewidth=1, label=labels[j][k])
                ax.fill_between(X_, Y_[2], Y_[1], color=colors_short[k], linewidth=0., alpha=0.4)
                if k == 1:
                    y_i = data_figure[i]  # n_bin-1 * 3
                    Y_ = np.empty((3, 100))
                    for l in range(3):
                        X_Y_Spline = make_interp_spline(x, y_i[:, l])
                        Y_[l] = X_Y_Spline(X_)
                    ax.plot(X_, Y_[0], color='gray', linewidth=1,
                            label=r'$\bar{\Delta}_t$')
                    ax.fill_between(X_, Y_[2], Y_[1], color='gray', linewidth=0., alpha=0.2)
                    ax.axhline(average_Delta_bar[i], 0.05, 0.95, color='saddlebrown', linewidth=0.8, linestyle='dashed',
                           label=r'Unconditional mean $\bar{\Delta}_t$')
                    x_mean = [X_[np.searchsorted(Y_[0], average_Delta_bar[i])]]
                    ax.scatter(x_mean, [average_Delta_bar[i]], marker='o', color='saddlebrown')
            else:
                y_i = y_focus[k]
                Y_ = np.empty((3, 100))
                for l in range(3):
                    X_Y_Spline = make_interp_spline(x, y_i[:, l])
                    Y_[l] = X_Y_Spline(X_)
                ax.plot(X_, Y_[0], color=colors_short[k], linewidth=1, label=labels[j][k])
                ax.fill_between(X_, Y_[2], Y_[1], color=colors_short[k], linewidth=0., alpha=0.4)
            ax.set_ylabel(y_labels[j])
        if i == 0:
            ax.legend(loc='upper left')
        if j == 0:
            ax.set_title(sub_titles[i])
        if j == 2:
            ax.set_xlabel(x_label)
        else:
            ax.axvline(0, 0.05, 0.95, color='gray', linewidth=0.8, linestyle='dashed')
        # ax.set_xlim(-0.25, 0.25)
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig(str(Npres[Npre_index]) + 'DeltaVola.png', dpi=100)
plt.show()


# ######################################
# ############ Figure 3.2 ##############
# ######################################


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
reg_data = np.empty((5, 8))
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
    for j, horizon in enumerate(horizons):
        y_raw = (path_y[start_t + horizon + 1:] - path_y[start_t + 1: -horizon]) / (horizon * dt)
        # dR is the return from t-1 to t, and thus have to move 1 to have returns from t to t+1
        y_predict = y_raw.reshape(-1, 1)
        for k, var in enumerate(x_variables):
            path_x = var[i]
            # univariate regressions
            x1_raw = path_x[start_t: -1 - horizon]  # participation rate at time t
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
            x2_lag = path_x2[start_t: -1 - horizon]  # average perceived risk premia at time t

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
    for j, horizon in enumerate(horizons):
        # y_raw = path_y[start_t + horizon + 1:] - path_y[start_t + 1: -horizon]  #dR is the return from t-1 to t, and thus have to move 1 to have returns from t to t+1
        y_raw = ((path_y[start_t + horizon + 1:] - path_y[start_t + 1: -horizon]) \
                 - (path_r[start_t + horizon: -1] - path_r[start_t: -horizon - 1])) / (horizon * dt)
        y_predict = y_raw.reshape(-1, 1)
        for k, var in enumerate(x_variables):
            path_x = var[i]
            # univariate regressions
            x1_raw = path_x[start_t: -1 - horizon]  # participation rate at time t
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