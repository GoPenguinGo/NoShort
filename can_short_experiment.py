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
    scenario_labels, colors_short, colors_short2, PN_labels, age_labels, top, old_limit
from src.stats import shocks, tau_calculator, good_times, Delta_st_compare
from numba import jit
import matplotlib.pyplot as plt
import statsmodels.api as sm
# import tabulate as tabulate
from scipy.interpolate import make_interp_spline

n_scenarios = 2
scenarios_short = scenarios[3:3+n_scenarios]

# n_scenarios = 1
# scenarios_short = scenarios[1:2]

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
cohort_size_mat = np.tile(cohort_size, (Nc, 1))
for g, scenario in enumerate(scenarios_short):
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
                    popu_short,
                    popu_can_short,
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
                delta_bar_compare[g, i, j, k] = Delta_bar_parti
                Phi_compare[g, i, j, k] = Phi_parti
                cons_compare[g, i, j, k] = f / cohort_size_mat
                invest_tracker_compare[g, i, j, k] = invest_tracker

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
    scenario_index = i
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
    scenario_index = i
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


# Across paths, equilibirum values
# ######################################
# ############ Figure 3 ################
# ######################################
# over phi
N = 200  # for smaller number of paths
# tax_vector = [0.008]
tax_vector = [0.01]
# tax_vector = [0.012]
n_tax = len(tax_vector)
theta_matrix = np.empty((N, n_scenarios, n_tax, n_phi, 2))
delta_bar_matrix = np.zeros((N, n_scenarios, n_tax, n_phi, 2))
Phi_parti_1_matrix = np.zeros((N, n_scenarios, n_tax, n_phi, 2))
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
                Phi_parti_1_matrix[j, k, l, m] = Phi_parti_1
                popu_age_matrix[j, k, l, m] = popu_age
                wealthshare_age_matrix[j, k, l, m] = wealthshare_age
                popu_can_short_matrix[j, k, l, m] = popu_can_short
                popu_short_matrix[j, k, l, m] = popu_short
                Phi_can_short_matrix[j, k, l, m] = Phi_can_short
                Phi_short_matrix[j, k, l, m] = Phi_short

var_list = [theta_matrix, Phi_parti_1_matrix, delta_bar_matrix,
            popu_age_matrix, wealthshare_age_matrix, popu_can_short_matrix, popu_short_matrix, Phi_short_matrix]
var_name_list = ['market price of risk', 'consumption share 1 of participants',
                 'consumption-weighted estimation error of participants', 'popu age', 'wealth age',
                 'popu can short', 'popu short', 'cons share short']
type_list = ['mean', 'vola']
for i, var in enumerate(var_list):
    np.save(var_name_list[i] + str(tax_vector[0]), var)


tax_vector = [0.008, 0.01, 0.012]
n_tax = len(tax_vector)
theta_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
Phi_parti_1_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
Delta_bar_parti_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
popu_age_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2, n_age_groups))
wealthshare_age_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2, n_age_groups))
popu_can_short_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
popu_short_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
Phi_short_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
var_list = [theta_Mat, Phi_parti_1_Mat, Delta_bar_parti_Mat,
            popu_age_Mat, wealthshare_age_Mat, popu_can_short_Mat, popu_short_Mat, Phi_short_Mat]
var_name_list = ['market price of risk', 'consumption share 1 of participants',
                 'consumption-weighted estimation error of participants', 'popu age', 'wealth age',
                 'popu can short', 'popu short', 'cons share short']
for i, var in enumerate(var_list):
    var_name = var_name_list[i]
    for j, tax_rate in enumerate(tax_vector):
        var_name_j = var_name + str(tax_rate) +'.npy'
        y = np.load(var_name_j)
        var[:, j] = np.mean(np.mean(y[:200], axis=0), axis=1)


x = phi_vector
x_start = 0
var_name_list = [r'market price of risk $\theta_t$', r'$\sigma_Y\frac{1}{\Phi_t}$', r'$\bar{\Delta}_t$']
# var_name_list = ['market price of risk', 'consumption share of participants', 'consumption-weighted estimation error of participants']
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
            line_style = line_styles[k]
            for l in range(n_scenarios):
                y_i = y[l, k]
                X_Y_Spline = make_interp_spline(x, y_i)
                # Returns evenly spaced numbers
                # over a specified interval.
                X_ = np.linspace(x.min(), x.max(), 100)
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
# plt.savefig('initial window and values mean vola.png', dpi=500, format="png")
plt.show()
# plt.close()
