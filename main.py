import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from typing import Callable, Tuple
from src.simulation import simulate_SI, simulate_SI_mean_vola, simulate_mix_types
from src.param import rho, nu, mu_Y, sigma_Y, sigma_Y_sqr, v, tax, phi, \
    dt, T_hat, Npre, Vhat, Ninit, T_cohort, Nt, Nc, tau, cohort_size, \
    cutoffs_age, n_age_cutoffs, colors, modes_trade, modes_learn, Mpath, \
    scenarios, dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    dZ_Y_cases, dZ_SI_cases, dZ_build_case, dZ_SI_build_case, t, red_labels, yellow_labels, cohort_labels, \
    scenario_labels, colors_short, colors_short2, PN_labels, age_labels, cummu_popu, dt_root, \
    Ntype, rho_i, alpha_i, beta_i, beta0, beta_cohort_type, cohort_type_size
from src.param_mix import Nconstraint, alpha_i_mix, beta_i_mix, beta0_mix, beta_cohort_mix, beta_cohort_type_mix, \
    rho_i_mix, cohort_type_size_mix
from src.stats import shocks, tau_calculator, good_times, Delta_st_compare, weighted_variance
from numba import jit
import statsmodels.api as sm
import tabulate as tab
from scipy.interpolate import make_interp_spline
from multiprocessing import Pool
import pandas as pd

np.set_printoptions(precision=4, suppress=True)
results = np.load('results.npz')
results = np.load("results_mean_vola_alternative_new.npz")
file_list = results.files
#
# ######################################
# #############  Table 1  ##############
# ######################################
# results_mean_vola = np.load('results_mean_vola.npz')
# file_list_mean_vola = results_mean_vola.files
# for file in file_list_mean_vola:
#     print(file)
#     print(np.average(results_mean_vola[file], axis=0))
#
# Nsce = 3
# Ncolumn = int(Nsce * 2)
# # Panel 1: mean vola of asset pricing values
# table_output = np.zeros((5, Ncolumn))
# # var_list = [r_baseline_mat, theta_baseline_mat, Phi1_baseline_mat * sigma_Y, Delta_bar_baseline_mat]
# header = np.tile(['Mean', 'Std'], 2)
# # show_index = [r'$r_t$', r'$\theta_t$', r'$\sigma_Y\frac{1}{\Phi_t}$', r'$\bar{\Delta}_t$']
# for j, file in enumerate(file_list_mean_vola[1:6]):
#     var_average = np.average(results_mean_vola[file], axis=0)  # shape (n_scenarios, 2)
#     for i in range(Nsce):
#         row_index = j
#         col_index = i * 2
#         table_output[row_index, col_index:col_index + 2] = var_average[i]
# print(tab.tabulate(table_output, headers=header, showindex=file_list_mean_vola[1:6], floatfmt=".4f",
#                    tablefmt='latex_raw'))
#
# # Panel 2: mean vola of theta state variables
# file = file_list_mean_vola[7]
# var_average = np.average(results_mean_vola[file], axis=0)  # shape (n_scenarios, 2)
# N_row = np.shape(var_average)[1]
# table_output = np.zeros((N_row, Ncolumn))
# for j in range(N_row):
#     for i in range(Nsce):
#         row_index = j
#         col_index = i * 2
#         table_output[row_index, col_index:col_index + 2] = var_average[i, j]
# show_index = [r'$\sigma_Y\frac{1}{\Phi_t}$', r'$\bar{\Delta}_t$']
# print(tab.tabulate(table_output, showindex=show_index, floatfmt=".4f", tablefmt='latex_raw'))
#
# # Panel 3: mean vola of sigma_S state variables
# file = file_list_mean_vola[8]
# var_average = np.average(results_mean_vola[file], axis=0)  # shape (n_scenarios, 2)
# N_row = np.shape(var_average)[1]
# table_output = np.zeros((N_row, Ncolumn))
# for j in range(N_row):
#     for i in range(Nsce):
#         row_index = j
#         col_index = i * 2
#         table_output[row_index, col_index:col_index + 2] = var_average[i, j]
# show_index = [r'$\tilde{\Phi}_t$',
#               r'$\sigma_Y\tilde{\Phi}_t/\Bar{\Phi}_t$',
#               r'$\tilde{\Delta}_t - \Bar{\Delta}_t$',
#               r'$(\tilde{\Phi}_t\tilde{\Delta}_t - \Bar{\Delta}_t) $']
# print(tab.tabulate(table_output, showindex=show_index, floatfmt=".4f", tablefmt='latex_raw'))
#
# # Panel 4: covariance
# file = file_list_mean_vola[11]
# var_average = np.average(results_mean_vola[file], axis=0)  # shape (n_scenarios, 2)
# N_row = np.shape(var_average)[1]
# table_output = np.zeros((N_row, Ncolumn))
# for j in range(N_row):
#     for i in range(Nsce):
#         row_index = j
#         col_index = i * 2
#         table_output[row_index, col_index] = var_average[i, j]
# show_index = [r'$\text{Cov}(dz_t^Y, \theta_t)$',
#               r'$\text{Cov}(dz_t^Y, \mu_t^S)$',
#               r'$\text{Cov}(dz_t^Y, \sigma_t^S)$',
#               r'$\text{Cov}(dz_t^{SI}, \theta_t)$',
#               r'$\text{Cov}(\Bar{\Phi}_t, P_t)$',
#               r'$\text{Cov}(\tilde{\Phi}_t, P_t)$',
#               ]
# print(tab.tabulate(table_output, showindex=show_index, floatfmt=".4f", tablefmt='latex_raw'))
#
# # Panel 5: participation rate in age groups
# file = file_list_mean_vola[9]
# var_average = np.average(results_mean_vola[file], axis=0)  # shape (n_scenarios, 2)
# N_row = np.shape(var_average)[1]
# table_output = np.zeros((N_row, Ncolumn))
# for j in range(N_row):
#     for i in range(Nsce):
#         row_index = j
#         col_index = i * 2
#         table_output[row_index, col_index] = var_average[i, j]
# show_index = [r'$0 < \text{Age} \leq 15$',
#               r'$15 < \text{Age} \leq 35$',
#               r'$35 < \text{Age} \leq 69$',
#               r'$\text{Age} > 69$',
#               ]
# print(tab.tabulate(table_output,
#                    showindex=show_index,
#                    floatfmt=".4f", tablefmt='latex_raw'))
#
# # Panel 6: participation rate in wealth groups
# file = file_list_mean_vola[10]
# var_average = np.average(results_mean_vola[file], axis=0)  # shape (n_scenarios, 2)
# N_row = np.shape(var_average)[1]
# table_output = np.zeros((N_row, Ncolumn))
# for j in range(N_row):
#     for i in range(Nsce):
#         row_index = j
#         col_index = i * 2
#         table_output[row_index, col_index] = var_average[i, j]
# show_index = [r'$ \text{wealth} \leq 1$',
#               r'$1 <\text{wealth} \leq 10$',
#               r'$10 < \text{wealth} \leq 100$',
#               r'$\text{wealth} > 100$',
#               ]
# print(tab.tabulate(table_output,
#                    showindex=show_index,
#                    floatfmt=".4f", tablefmt='latex_raw'))
#
# # Panel 7: participation rate covariances
# file = file_list_mean_vola[13]
# var_average = np.average(results_mean_vola[file], axis=0)  # shape (n_scenarios, 2)
# N_row = np.shape(var_average)[1]
# table_output = np.zeros((N_row, Ncolumn))
# for j in range(N_row):
#     for i in range(Nsce):
#         row_index = j
#         col_index = i * 2
#         table_output[row_index, col_index] = var_average[i, j, 1]
# show_index = [r'$ \text{Cov}\left(R^S_{t,t+2}, P_t\right)$',
#               r'$ \text{Cov}\left(R_{t,t+2}, P_t\right)$',
#               r'$ \text{Cov}\left(R^S_{t,t+2}-R_{t,t+2}, P_t\right)$',
#               r'$ \text{Cov}\left(\text{average}\sigma_S, P_t\right) > 100$',
#               ]
# print(tab.tabulate(table_output,
#                    showindex=show_index,
#                    floatfmt=".4f", tablefmt='latex_raw'))
#

######################################
#############  Table 3  ##############
######################################


file_names_alternative = ['results_mean_vola_alternative1.npz',
                          'results_mean_vola_alternative2.npz',
                          # 'results_mean_vola_alternative3.npz',
                          # 'results_mean_vola_alternative4.npz',
                          'results_mean_vola_alternative5.npz']

files_need1 = ['theta_mean_vola', 'r_mean_vola', 'mu_S_mean_vola', 'sigma_S_mean_vola']
files_need2 = ['parti_age_group', 'parti_wealth_group']
alternative_results1 = np.zeros((5, 4, 4, 3, 2), dtype=np.float32)
alternative_results2 = np.zeros((5, 2, 4, 3, 4), dtype=np.float32)
for i, file in enumerate(file_names_alternative):
    results_al = np.load(file)
    for j, file_need in enumerate(files_need1):
        alternative_results1[i, j] = np.average(results_al[file_need], axis=0)
    for j, file_need in enumerate(files_need2):
        alternative_results2[i, j] = np.average(results_al[file_need], axis=0)


Nsce = 3
Ncolumn = int(Nsce * 2)
# Panel 1: mean vola of asset pricing values
table_output = np.zeros((5, Ncolumn))
# var_list = [r_baseline_mat, theta_baseline_mat, Phi1_baseline_mat * sigma_Y, Delta_bar_baseline_mat]
header = np.tile(['Mean', 'Std'], 2)
# show_index = [r'$r_t$', r'$\theta_t$', r'$\sigma_Y\frac{1}{\Phi_t}$', r'$\bar{\Delta}_t$']
for j, file in enumerate(file_list_mean_vola[1:6]):
    var_average = np.average(results_mean_vola[file], axis=0)  # shape (n_scenarios, 2)
    for i in range(Nsce):
        row_index = j
        col_index = i * 2
        table_output[row_index, col_index:col_index + 2] = var_average[i]
print(tab.tabulate(table_output, headers=header, showindex=file_list_mean_vola[1:6], floatfmt=".4f",
                   tablefmt='latex_raw'))


# Additional parameters for the figures
n_scenarios_short = 3
scenarios_short = scenarios[:n_scenarios_short]

phi_vector = np.arange(0, 1, 0.1)
n_phi = len(phi_vector)

phi_indexes = [0, 4, 8]
n_phi_short = len(phi_indexes)
phi_vector_short = phi_vector[phi_indexes]

phi_indexes_5 = [0, 2, 4, 6, 8]
n_phi_5 = 5
phi_5 = phi_vector[phi_indexes_5]

label_phi = []
for i in range(n_phi_short):
    label_phi.append(r'$phi$ = ' + str(phi_vector_short[i]))
labels = [scenario_labels, label_phi, label_phi]

age_cutoff = cutoffs_age[2]

scenarios_two = scenarios[1:3]
Npre_short = np.array([60, 240])
T_hat_short = dt * Npre_short

plt.rcParams["font.family"] = 'serif'

######################################
########## ONE RANDOM PATH ############F
############ GRAPH ONE ###############
######################################

# ONE SPECIFIC PATH:
print('Generating data for the graphs:')
theta_compare = np.empty((n_scenarios_short, 2, 2, Nt), dtype=np.float32)
popu_parti_compare = np.empty((n_scenarios_short, 2, 2, Nt), dtype=np.float32)
r_compare = np.zeros((n_scenarios_short, 2, 2, Nt), dtype=np.float32)
Delta_bar_compare = np.zeros((n_scenarios_short, 2, 2, Nt), dtype=np.float32)
Delta_tilde_compare = np.zeros((n_scenarios_short, 2, 2, Nt), dtype=np.float32)
Phi_compare = np.zeros((n_scenarios_short, 2, 2, Nt), dtype=np.float32)
dR_compare = np.zeros((n_scenarios_short, 2, 2, Nt), dtype=np.float32)
mu_S_compare = np.zeros((n_scenarios_short, 2, 2, Nt), dtype=np.float32)
sigma_S_compare = np.zeros((n_scenarios_short, 2, 2, Nt), dtype=np.float32)
beta_compare = np.zeros((n_scenarios_short, 2, 2, Nt), dtype=np.float32)
Delta_compare = np.empty((n_scenarios_short, 2, 2, Nt, Nconstraint, Nc), dtype=np.float16)
pi_compare = np.empty((n_scenarios_short, 2, 2, Nt, Nconstraint, Nc), dtype=np.float16)
cons_compare = np.zeros((n_scenarios_short, 2, 2, Nt, Ntype, Nconstraint, Nc), dtype=np.float16)
invest_tracker_compare = np.zeros((n_scenarios_short, 2, 2, Nt, Nconstraint, Nc), dtype=np.float16)
Delta_mix = np.empty((n_scenarios_short, 2, 2, Nt, Nconstraint, Nc), dtype=np.float16)
cohort_type_size_mix_mat = np.tile(cohort_type_size_mix, (Nt, 1, 1, 1))
cohort_type_size_mat = np.tile(cohort_type_size, (Nt, 1, 1))
for g, scenario in enumerate(scenarios_short):
    if g <= 1:
        mode_trade = scenario[0]
        mode_learn = scenario[1]
    for i in range(2):
        dZ = dZ_Y_cases[i]
        # log_Yt = np.cumsum((mu_Y - 0.5 * sigma_Y ** 2) * dt + sigma_Y * dZ)
        for j in range(2):
            dZ_SI = dZ_SI_cases[j]
            if g <= 1:
                (
                    r,
                    theta,
                    f_c,
                    Delta,
                    pi,
                    parti,
                    Phi_parti,
                    Delta_bar_parti,
                    Delta_tilde_parti,
                    dR,
                    mu_S,
                    sigma_S,
                    beta,
                    invest_tracker,
                    parti_age_group,
                    parti_wealth_group,
                ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax, beta0,
                                phi,
                                Npre, Ninit, T_hat, dZ_build_case, dZ, dZ_SI_build_case, dZ_SI, tau, cutoffs_age,
                                Ntype, rho_i, alpha_i, beta_i, beta_cohort_type, cohort_type_size,
                                need_f='True',
                                need_Delta='True',
                                need_pi='True',
                                )
                # invest_tracker = pi > 0
                Delta_compare[g, i, j, :, 0] = Delta
                pi_compare[g, i, j, :, 0] = pi
                theta_compare[g, i, j] = theta
                r_compare[g, i, j] = r
                popu_parti_compare[g, i, j] = parti
                Delta_bar_compare[g, i, j] = Delta_bar_parti
                Delta_tilde_compare[g, i, j] = Delta_tilde_parti
                Phi_compare[g, i, j] = Phi_parti
                cons_compare[g, i, j, :, :, 0] = f_c / cohort_type_size_mat
                invest_tracker_compare[g, i, j, :, 0] = invest_tracker
                dR_compare[g, i, j] = dR
                mu_S_compare[g, i, j] = mu_S
                sigma_S_compare[g, i, j] = sigma_S
                beta_compare[g, i, j] = beta
            else:
                (
                    r,
                    theta,
                    f_c,
                    Delta,
                    pi,
                    parti,
                    Phi_parti,
                    Delta_bar_parti,
                    Delta_tilde_parti,
                    dR,
                    mu_S,
                    sigma_S,
                    beta,
                    invest_tracker,
                    parti_age_group,
                    parti_wealth_group,
                ) = simulate_mix_types(Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax,
                                       beta0_mix,
                                       phi, Npre, Ninit, T_hat,
                                       dZ_build_case, dZ, dZ_SI_build_case, dZ_SI,
                                       tau, cutoffs_age, Ntype,
                                       Nconstraint, rho_i_mix, alpha_i_mix, beta_i_mix, beta_cohort_type_mix,
                                       cohort_type_size_mix,
                                       need_f='True',
                                       need_Delta='True',
                                       need_pi='True',
                                       )
                Delta_compare[g, i, j] = Delta
                pi_compare[g, i, j] = pi
                theta_compare[g, i, j] = theta
                r_compare[g, i, j] = r
                popu_parti_compare[g, i, j] = parti
                Delta_bar_compare[g, i, j] = Delta_bar_parti
                Delta_tilde_compare[g, i, j] = Delta_tilde_parti
                Phi_compare[g, i, j] = Phi_parti
                cons_compare[g, i, j] = f_c / cohort_type_size_mix_mat
                invest_tracker_compare[g, i, j] = invest_tracker
                dR_compare[g, i, j] = dR
                mu_S_compare[g, i, j] = mu_S
                sigma_S_compare[g, i, j] = sigma_S
                beta_compare[g, i, j] = beta

# cohort_matrix_list = [pi_compare, Delta_compare, cons_compare]
nn = 3  # number of cohorts illustrated
length = len(t)
starts = np.zeros(nn)
Delta_time_series = np.zeros((n_scenarios_short-1, 2, 2, nn, length), dtype=np.float32)
pi_time_series = np.zeros((n_scenarios_short-1, 2, 2, nn, length), dtype=np.float32)
cons_time_series = np.zeros((n_scenarios_short-1, 2, 2, nn, Ntype, length), dtype=np.float32)
switch_time_series = np.zeros((n_scenarios_short-1, 2, 2, nn, length), dtype=np.float32)
parti_time_series = np.zeros((n_scenarios_short-1, 2, 2, nn, length), dtype=np.float32)
Delta_time_series_mix = np.zeros((2, 2, nn, Nconstraint, length), dtype=np.float32)
pi_time_series_mix = np.zeros((2, 2, nn, Nconstraint, length), dtype=np.float32)
cons_time_series_mix = np.zeros((2, 2, nn, Ntype, Nconstraint, length), dtype=np.float32)
switch_time_series_mix = np.zeros((2, 2, nn, Nconstraint, length), dtype=np.float32)
parti_time_series_mix = np.zeros((2, 2, nn, Nconstraint, length), dtype=np.float32)
for o in range(n_scenarios_short-1):
    for i in range(2):
        for j in range(2):
            pi = pi_compare[o, i, j, :, 0]
            cons = cons_compare[o, i, j, :, :, 0]  # shape ((6000 * 2 * 6000))
            Delta = Delta_compare[o, i, j, :, 0]
            for m in range(nn):
                start = int((m + 1) * 100 * (1 / dt))
                starts[m] = start * dt
                for n in range(length):
                    if n < start:
                        pi_time_series[o, i, j, m, n] = np.nan
                        cons_time_series[o, i, j, m, :, n] = np.nan
                        Delta_time_series[o, i, j, m, n] = np.nan
                    else:
                        cohort_rank = length - (n - start) - 1
                        Delta_time_series[o, i, j, m, n] = Delta[n, cohort_rank]
                        pi_time_series[o, i, j, m, n] = pi[n, cohort_rank]
                        cons_time_series[o, i, j, m, :, n] = cons[n, :, cohort_rank]
                parti = pi_time_series[o, i, j, m] > 0
                switch = abs(parti[1:] ^ parti[:-1])
                sw = np.append(switch, 0)
                parti = np.where(sw == 1, 0.5, parti)
                switch = np.insert(switch, 0, 0)
                parti = np.where(switch == 1, 0.5, parti)
                parti_time_series[o, i, j, m] = parti
                switch_time_series[o, i, j, m] = sw
o = 2
for i in range(2):
    for j in range(2):
        for k in range(Nconstraint):
            pi = pi_compare[o, i, j, :, k]
            cons = cons_compare[o, i, j, :, :, k]  # shape ((6000 * 2 * 6000))
            Delta = Delta_compare[o, i, j, :, k]
            for m in range(nn):
                start = int((m + 1) * 100 * (1 / dt))
                starts[m] = start * dt
                for n in range(length):
                    if n < start:
                        pi_time_series_mix[i, j, m, k, n] = np.nan
                        cons_time_series_mix[i, j, m, :, k, n] = np.nan
                        Delta_time_series_mix[i, j, m, k, n] = np.nan
                    else:
                        cohort_rank = length - (n - start) - 1
                        Delta_time_series_mix[i, j, m, k, n] = Delta[n, cohort_rank]
                        pi_time_series_mix[i, j, m, k, n] = pi[n, cohort_rank]
                        cons_time_series_mix[i, j, m, :, k, n] = cons[n, :, cohort_rank]
                        if k == 0:  # the unconstrained
                            parti_time_series_mix[i, j, m, k, n] = 1
                            switch_time_series_mix[i, j, m, k, n] = 0
                        elif k == 1:  # the excluded
                            parti_time_series_mix[i, j, m, k, n] = 0
                            switch_time_series_mix[i, j, m, k, n] = 0
                        else:
                            parti_time_series_mix[i, j, m, k, n] = 1 if pi_time_series_mix[i, j, m, k, n] > 0 else 0
                            switch_PN = 1 if (pi_time_series_mix[i, j, m, k, n-1] > 0) and (
                                        pi_time_series_mix[i, j, m, k, n] == 0) else 0
                            switch_NP = 1 if (pi_time_series_mix[i, j, m, k, n] > 0) and (
                                    pi_time_series_mix[i, j, m, k, n-1] == 0) else 0
                            switch_time_series_mix[i, j, m, k, n] = 1 if switch_PN == 1 else 0
                            switch_time_series_mix[i, j, m, k, n] = 1 if switch_NP == 1 else 0
                            if switch_NP == 1:
                                parti_time_series_mix[i, j, m, k, n] = 0.5
                            if switch_PN == 1:
                                parti_time_series_mix[i, j, m, k, n - 1] = 0.5



######################################
########### Figure 1 & IA1 #############
######################################
print('Figure 1')
upper = 60  # todo: endogenize these parameters
lower = -60
scenario_index = 1
fig, axes = plt.subplots(nrows=2, ncols=1, sharey='all', sharex='all', figsize=(15, 9))
for j, ax in enumerate(axes):
    red_index = 1
    yellow_index = 0 if j == 0 else 1
    red_case = red_labels[red_index]
    yellow_case = yellow_labels[yellow_index]
    Z = np.cumsum(dZ_Y_cases[red_index])
    Z_SI = np.cumsum(dZ_SI_cases[yellow_index])  # todo: plot SI (combination of Z_Y ad Z_SI) instead of Z_SI
    # if j == 3:
    #     ax.set_xlabel('Time in simulation')
    ax.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
    ax.plot(t, Z, color='red', linewidth=0.4, label=r'$z^Y_t$')
    ax.plot(t, Z_SI, color='gold', linewidth=0.4, label=r'$z^{SI}_t$')
    ax.set_ylim([lower, upper])
    ax.tick_params(axis='y', labelcolor='black')
    ax.tick_params(axis='x', labelcolor='black')
    if j == 0:
        ax.legend()
    ax.set_title('Reentry scenario, ' + red_case + yellow_case)

    ax2 = ax.twinx()
    ax2.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
    ax2.set_ylim([-0.3, 0.4])
    for m in range(nn):
        # switch[m, starts[m]] = 1
        y_cohort = Delta_time_series[scenario_index, red_index, yellow_index, m]
        y_cohort_N = np.ma.masked_where(parti_time_series[scenario_index, red_index, yellow_index, m] == 1,
                                        y_cohort)
        y_cohort_P = np.ma.masked_where(parti_time_series[scenario_index, red_index, yellow_index, m] == 0,
                                        y_cohort)
        y_cohort_switch = np.ma.masked_where(switch_time_series[scenario_index, red_index, yellow_index, m] == 0,
                                             y_cohort)
        plt.vlines(starts[m], ymax=upper, ymin=lower, color='grey', linestyle='--', linewidth=0.4)
        #  ax2.plot(t, y_cohort, label=cohort_labels[m], color=colors_short[m], linewidth=0.4)
        if j == 0:
            ax2.plot(t, y_cohort_P, color=colors_short[m], linewidth=0.8, label=cohort_labels[m])
            ax2.plot(t, y_cohort_N, color=colors_short[m], linewidth=0.8, linestyle='dotted')
            ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o')
        elif j == 1 and m == 0:
            ax2.plot(t, y_cohort_P, color=colors_short[m], linewidth=0.8, label=PN_labels[0])
            ax2.plot(t, y_cohort_N, color=colors_short[m], linewidth=0.8, linestyle='dotted', label=PN_labels[1])
            ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o', label='switch')
        else:
            ax2.plot(t, y_cohort_P, color=colors_short[m], linewidth=0.8)
            ax2.plot(t, y_cohort_N, color=colors_short[m], linewidth=0.8, linestyle='dotted')
            ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o')

    ax2.tick_params(axis='y', labelcolor='black')
    if j <= 1:
        ax2.legend()
    # Save the subfigs for slides, etc.
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # Pad the saved area by 10% in the x-direction and 20% in the y-direction
    # fig.savefig(' Shocks and Delta time series subfig' + str(i) + str(j + 1) + '.png',
    #             bbox_inches=extent.expanded(1.2, 1.25), dpi=200)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(
    'f1.png',
    dpi=100)
plt.show()
# plt.close()


print('Figure 1-mix')
upper = 60  # todo: endogenize these parameters
lower = -60
scenario_index = 1
constraint_labels = ['Designated P', 'Designated N', 'Disappointment', 'Reentry']
colors_short3 = ['mediumblue', 'saddlebrown', 'purple', 'darkgreen']
color_dot = 'red'
fig, axes = plt.subplots(nrows=2, ncols=1, sharey='all', sharex='all', figsize=(15, 9))
for j, ax in enumerate(axes):
    red_index = 1
    yellow_index = 1
    red_case = red_labels[red_index]
    yellow_case = yellow_labels[yellow_index]
    Z = np.cumsum(dZ_Y_cases[red_index])
    Z_SI = np.cumsum(dZ_SI_cases[yellow_index])  # todo: plot SI (combination of Z_Y ad Z_SI) instead of Z_SI
    if j == 1:
        ax.set_xlabel('Time in simulation')

    ax.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
    ax.plot(t, Z, color='red', linewidth=0.4, label=r'$z^Y_t$')
    ax.plot(t, Z_SI, color='gold', linewidth=0.4, label=r'$z^{SI}_t$')
    ax.set_ylim([lower, upper])
    ax.tick_params(axis='y', labelcolor='black')
    ax.tick_params(axis='x', labelcolor='black')
    if j == 0:
        ax.legend(loc='upper left')
    ax.set_title('Mix scenario, ' + red_case + yellow_case)

    ax2 = ax.twinx()
    ax2.set_ylabel(r'Estimation error $\Delta_{j, s,t}$', color='black')
    ax2.set_ylim([-0.35, 0.4])
    m = 0 if j == 0 else 2
    for kk in range(Nconstraint-1):
        # switch[m, starts[m]] = 1
        k = kk + 1 if kk > 0 else kk
        y_cohort = Delta_time_series_mix[red_index, yellow_index, m, k]
        y_cohort_N = np.ma.masked_where(parti_time_series_mix[red_index, yellow_index, m, k] == 1,
                                        y_cohort)
        y_cohort_P = np.ma.masked_where(parti_time_series_mix[red_index, yellow_index, m, k] == 0,
                                        y_cohort)
        y_cohort_switch = np.ma.masked_where(switch_time_series_mix[red_index, yellow_index, m, k] == 0,
                                             y_cohort)
        plt.vlines(starts[m], ymax=upper, ymin=lower, color='grey', linestyle='--', linewidth=0.4)
        #  ax2.plot(t, y_cohort, label=cohort_labels[m], color=colors_short[m], linewidth=0.4)
        if j == 0:
            ax2.plot(t, y_cohort_P, color=colors_short3[k], linewidth=0.8, label=constraint_labels[k])
            ax2.plot(t, y_cohort_N, color=colors_short3[k], linewidth=0.8, linestyle='dotted')
            ax2.scatter(t, y_cohort_switch, color=color_dot, s=10, marker='o')
            ax2.legend()
        else:
            ax2.plot(t, y_cohort_P, color=colors_short3[k], linewidth=0.8, label=PN_labels[0])
            ax2.plot(t, y_cohort_N, color=colors_short3[k], linewidth=0.8, linestyle='dotted', label=PN_labels[1])
            ax2.scatter(t, y_cohort_switch, color=color_dot, s=10, marker='o', label='switch')
            # if kk == 0:
            #     ax2.legend()
    ax2.tick_params(axis='y', labelcolor='black')
    # # if j <= 1:
    # ax2.legend()
    # Save the subfigs for slides, etc.
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # Pad the saved area by 10% in the x-direction and 20% in the y-direction
    # fig.savefig(' Shocks and Delta time series subfig' + str(i) + str(j + 1) + '.png',
    #             bbox_inches=extent.expanded(1.2, 1.25), dpi=200)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(
    'f1_1.png',
    dpi=100)
plt.show()
# plt.close()



# IA:
# ONE SPECIFIC PATH:
# print('Generating data for the graphs:')
# phi_8 = 0.8
# scenario = scenarios[1]
# mode_trade = scenario[0]
# mode_learn = scenario[1]
# Delta_compare = np.empty((2, 2, Nt, Nc), dtype=np.float16)
# pi_compare = np.empty((2, 2, Nt, Nc), dtype=np.float16)
# cons_compare = np.zeros((2, 2, Nt, Ntype, Nc), dtype=np.float16)
# invest_tracker_compare = np.zeros((2, 2, Nt, Nc), dtype=np.float16)
# cohort_type_size_mat = np.tile(cohort_type_size, (Nt, 1, 1))
# for i in range(2):
#     dZ = dZ_Y_cases[i]
#     # log_Yt = np.cumsum((mu_Y - 0.5 * sigma_Y ** 2) * dt + sigma_Y * dZ)
#     for j in range(2):
#         dZ_SI = dZ_SI_cases[j]
#         (
#             r,
#             theta,
#             f_c,
#             Delta,
#             pi,
#             parti,
#             Phi_parti,
#             Delta_bar_parti,
#             Delta_tilde_parti,
#             dR,
#             mu_S,
#             sigma_S,
#             beta,
#             invest_tracker,
#             parti_age_group,
#             parti_wealth_group,
#         ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax, beta0,
#                         phi_8,
#                         Npre, Ninit, T_hat, dZ_build_case, dZ, dZ_SI_build_case, dZ_SI, tau, cutoffs_age,
#                         Ntype, rho_i, alpha_i, beta_i, beta_cohort_type, cohort_type_size,
#                         need_f='True',
#                         need_Delta='True',
#                         need_pi='True',
#                         )
#         # invest_tracker = pi > 0
#         Delta_compare[i, j] = Delta
#         pi_compare[i, j] = pi
#         cons_compare[i, j] = f_c / cohort_type_size_mat
#         invest_tracker_compare[i, j] = invest_tracker
#
# nn = 3  # number of cohorts illustrated
# length = len(t)
# starts = np.zeros(nn)
# Delta_time_series = np.zeros((2, 2, nn, length), dtype=np.float32)
# pi_time_series = np.zeros((2, 2, nn, length), dtype=np.float32)
# cons_time_series = np.zeros((2, 2, nn, Ntype, length), dtype=np.float32)
# switch_time_series = np.zeros((2, 2, nn, length), dtype=np.int8)
# parti_time_series = np.zeros((2, 2, nn, length), dtype=np.int8)
# for i in range(2):
#     for j in range(2):
#         pi = pi_compare[i, j]
#         cons = cons_compare[i, j]  # shape ((6000 * 2 * 6000))
#         Delta = Delta_compare[i, j]
#         for m in range(nn):
#             start = int((m + 1) * 100 * (1 / dt))
#             starts[m] = start * dt
#             for n in range(length):
#                 if n < start:
#                     pi_time_series[i, j, m, n] = np.nan
#                     cons_time_series[i, j, m, :, n] = np.nan
#                     Delta_time_series[i, j, m, n] = np.nan
#                 else:
#                     cohort_rank = length - (n - start) - 1
#                     Delta_time_series[i, j, m, n] = Delta[n, cohort_rank]
#                     pi_time_series[i, j, m, n] = pi[n, cohort_rank]
#                     cons_time_series[i, j, m, :, n] = cons[n, :, cohort_rank]
#             parti = pi_time_series[i, j, m] > 0
#             switch = abs(parti[1:] ^ parti[:-1])
#             sw = np.append(switch, 0)
#             parti = np.where(sw == 1, 0.5, parti)
#             switch = np.insert(switch, 0, 0)
#             parti = np.where(switch == 1, 0.5, parti)
#             parti_time_series[i, j, m] = parti
#             switch_time_series[i, j, m] = sw
# upper = 60  # todo: endogenize these parameters
# lower = -60
# scenario_index = 1
# fig, axes = plt.subplots(nrows=4, ncols=1, sharey='all', sharex='all', figsize=(15, 18))
# for j, ax in enumerate(axes):
#     red_index = 0 if j == 0 or j == 1 else 1
#     yellow_index = 0 if j == 0 or j == 2 else 1
#     red_case = red_labels[red_index]
#     yellow_case = yellow_labels[yellow_index]
#     Z = np.cumsum(dZ_Y_cases[red_index])
#     Z_SI = np.cumsum(dZ_SI_cases[yellow_index])  # todo: plot SI (combination of Z_Y ad Z_SI) instead of Z_SI
#     if j == 3:
#         ax.set_xlabel('Time in simulation')
#     ax.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
#     ax.plot(t, Z, color='red', linewidth=0.4, label=r'$z^Y_t$')
#     ax.plot(t, Z_SI, color='gold', linewidth=0.4, label=r'$z^{SI}_t$')
#     ax.set_ylim([lower, upper])
#     ax.tick_params(axis='y', labelcolor='black')
#     ax.tick_params(axis='x', labelcolor='black')
#     if j == 0:
#         ax.legend()
#     ax.set_title(red_case + yellow_case)
#
#     ax2 = ax.twinx()
#     ax2.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
#     ax2.set_ylim([-0.3, 0.4])
#     for m in range(nn):
#         # switch[m, starts[m]] = 1
#         y_cohort = Delta_time_series[red_index, yellow_index, m]
#         y_cohort_N = np.ma.masked_where(parti_time_series[red_index, yellow_index, m] == 1,
#                                         y_cohort)
#         y_cohort_P = np.ma.masked_where(parti_time_series[red_index, yellow_index, m] == 0,
#                                         y_cohort)
#         y_cohort_switch = np.ma.masked_where(switch_time_series[red_index, yellow_index, m] == 0,
#                                              y_cohort)
#         plt.vlines(starts[m], ymax=upper, ymin=lower, color='grey', linestyle='--', linewidth=0.4)
#         #  ax2.plot(t, y_cohort, label=cohort_labels[m], color=colors_short[m], linewidth=0.4)
#         if j == 0:
#             ax2.plot(t, y_cohort_P, color=colors_short[m], linewidth=0.8, label=cohort_labels[m])
#             ax2.plot(t, y_cohort_N, color=colors_short[m], linewidth=0.8, linestyle='dotted')
#             ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o')
#         elif j == 1 and m == 0:
#             ax2.plot(t, y_cohort_P, color=colors_short[m], linewidth=0.8, label=PN_labels[0])
#             ax2.plot(t, y_cohort_N, color=colors_short[m], linewidth=0.8, linestyle='dotted', label=PN_labels[1])
#             ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o', label='switch')
#         else:
#             ax2.plot(t, y_cohort_P, color=colors_short[m], linewidth=0.8)
#             ax2.plot(t, y_cohort_N, color=colors_short[m], linewidth=0.8, linestyle='dotted')
#             ax2.scatter(t, y_cohort_switch, color='red', s=10, marker='o')
#
#     ax2.tick_params(axis='y', labelcolor='black')
#     if j <= 1:
#         ax2.legend()
#     # Save the subfigs for slides, etc.
#     extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.savefig(
#     'IA_Shocks and Delta time series.png',
#     dpi=100)
# plt.show()
# # plt.close()


print('Figure 5')
# ######################################
# ############## Figure 5 ##############
# ######################################
# time-series of interest rate, market price of risk, stock volatility
# phi = 0.4, bad z^Y, bad z&SI
red_case = 1
yellow_case = 1
r_mat = r_compare[:, red_case, yellow_case]  # n_scenarios, n_phi_short, Nt
theta_mat = theta_compare[:, red_case, yellow_case]
sigma_S_mat = sigma_S_compare[:, red_case, yellow_case]
y_list = [r_mat, theta_mat, sigma_S_mat]
Z = np.cumsum(dZ_Y_cases[red_case])
Z_SI = np.cumsum(dZ_SI_cases[yellow_case])
y_title_list = [red_labels[1] + yellow_labels[1],
                r'Interest rate $r_t$',
                r'Market price of risk $\theta_t$',
                r'Stock volatility $\sigma^S_t$']
labels = scenario_labels
fig, axes = plt.subplots(nrows=4, ncols=1, sharex='all', figsize=(15, 20))
for j, ax in enumerate(axes):
    y_title = y_title_list[j]
    if j == 0:
        ax.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
        ax.plot(t, Z, color='red', linewidth=0.5, label=r'$z^Y_t$')
        ax.plot(t, Z_SI, color='gold', linewidth=0.5, label=r'$z^{SI}_t$')
        ax.tick_params(axis='both', labelcolor='black')
    else:
        y_vec = y_list[j - 1]  # n_phi_short, Nt
        ax.set_ylabel(y_title, color='black')
        for i in range(n_scenarios_short-1):
            y = y_vec[i]  # Nt
            color_i = colors_short[i]
            ax.plot(t, y, label=labels[i], color=color_i, linewidth=0.4)
            # ax2.set_ylim(lower, upper)
    if j <= 1:
        ax.legend(loc='upper left')
    if j == 3:
        ax.set_xlabel('Time in simulation')
    ax.set_title(y_title)
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig('r and theta, subfig ' + str(j + 1) + '.png', bbox_inches=extent.expanded(1.2, 1.3), dpi=200)
fig.tight_layout(h_pad=2)
plt.savefig('r and theta,' + str(red_case) + str(yellow_case) + '.png', dpi=60)
plt.savefig('r and theta,' + str(red_case) + str(yellow_case) + 'HD.png', dpi=200)
plt.show()
plt.close()

############ IA other cases
# red_cases = [0, 0, 1]
# yellow_cases = [0, 1, 0]
# phi_index = 0
# scenario_index = 1
# y_title_list = [r'Interest rate $r_t$, $\phi=0$', r'Interest rate $r_t$, Reentry']
# fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', sharey='all', figsize=(15, 20))
# for i, ax_row in enumerate(axes):
#     red_case = red_cases[i]
#     yellow_case = yellow_cases[i]
#     r_mat = r_compare[:, red_case, yellow_case]  # n_scenarios, n_phi_short, Nt
#     for j, ax in enumerate(ax_row):
#         y_mat = r_mat[:, phi_index] if j == 0 else r_mat[scenario_index]  # n_scenarios, Nt
#         labels = scenario_labels if j == 0 else label_phi
#         y_title = y_title_list[j]
#         Z = np.cumsum(dZ_Y_cases[red_case])
#         Z_SI = np.cumsum(dZ_SI_cases[yellow_case])
#         ax2 = ax.twinx()
#         ax2.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
#         ax2.plot(t, Z, color='red', linewidth=0.4, label=r'$z^Y_t$')
#         ax2.plot(t, Z_SI, color='gold', linewidth=0.4, label=r'$z^{SI}_t$')
#         ax2.tick_params(axis='y', labelcolor='black')
#         if i == 2:
#             ax.set_xlabel('Time in simulation')
#         for k in range(3):
#             y = y_mat[k]
#             color_i = colors_short2[k] if j == 0 else colors_short[k]
#             ax.plot(t, y, label=labels[k], color=color_i, linewidth=0.6)
#             y_title = y_title_list[j]
#         if i == 0:
#             ax2.legend(loc='upper right')
#             ax.legend(loc='upper left')
#         ax.set_ylabel(r'Interest rate $r_t$')
#         ax.set_title(y_title)
# fig.tight_layout(h_pad=2)
# plt.savefig('IA r and theta.png', dpi=60)
# plt.show()
# plt.close()

# ######################################
# ############# hetero rho #############
# ######################################
# comparing the complete market and the reentry scenario, with phi=0.4
# red_case = 1
# yellow_case = 1
# phi_case = 1
# r_mat = r_compare[:, red_case, yellow_case, phi_case]  # n_scenarios, n_phi_short, Nt
# theta_mat = theta_compare[:, red_case, yellow_case, phi_case]
# sigma_S = sigma_S_mat[:, red_case, yellow_case, phi_case]
# beta = beta_mat[:, red_case, yellow_case, phi_case]
# mu_S = mu_S_mat[:, red_case, yellow_case, phi_case]
# equi_premium = mu_S - r_mat
# dR = dR_mat[:, red_case, yellow_case, phi_case]
# window = 60
# R_cumu = np.cumsum(dR, axis=1)[:, window:] - np.cumsum(dR, axis=1)[:, :-window]
#
# y_list = [r_mat, theta_mat, sigma_S, beta, mu_S, equi_premium, R_cumu]
# Z = np.cumsum(dZ_Y_cases[red_case])
# Z_SI = np.cumsum(dZ_SI_cases[yellow_case])
# n_lines = 2
# y_title_list = [r'Interest rate $r_t$',
#                 r'Market price of risk $\theta_t$',
#                 r'Stock volatility $\sigma_t^S$',
#                 r'Average consumption wealth ratio $\bar{\beta}_t$',
#                 r'Expected returns $\mu_t^S$',
#                 r'Equity premium $\mu_t^S - r_t$',
#                 r'Realized stock returns $dR$']
# labels = scenario_labels
# for m, y_vec in enumerate(y_list):
#     fig, ax = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=(20, 7))
#     y_title = y_title_list[m]
#     ax.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
#     ax.plot(t, Z, color='red', linewidth=0.5, label=r'$z^Y_t$')
#     ax.plot(t, Z_SI, color='gold', linewidth=0.5, label=r'$z^{SI}_t$')
#     ax.tick_params(axis='both', labelcolor='black')
#     ax.set_xlabel('Time in simulation')
#
#     ax2 = ax.twinx()
#     ax2.set_ylabel(y_title, color='black')
#     for i in range(n_lines):
#         y = y_vec[i]  # Nt
#         color_i = colors_short2[i]
#         ax2.plot(t, y, label=labels[i], color=color_i, linewidth=0.4) if m != 6 else ax2.plot(t[window:], y,
#                                                                                               label=labels[i],
#                                                                                               color=color_i,
#                                                                                               linewidth=0.4)
#     ax.legend(loc='upper left')
#     ax2.legend(loc='upper right')
#     ax.set_title(y_title)
#     # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#     # fig.savefig('r and theta, subfig ' + str(j + 1) + '.png', bbox_inches=extent.expanded(1.2, 1.3), dpi=200)
#
#     fig.tight_layout(h_pad=2)
#     # plt.savefig('r and theta,' + str(red_case) + str(yellow_case) + '.png', dpi=60)
#     plt.savefig('HR' + str(m) + 'HD.png', dpi=200)
#     plt.show()
#     # plt.close()
#
# red_case = 1
# yellow_case = 1
# r_mat = r_compare[:, red_case, yellow_case]  # n_scenarios, n_phi_short, Nt
# theta_mat = theta_compare[:, red_case, yellow_case]
# sigma_S = sigma_S_mat[:, red_case, yellow_case]
# beta = beta_mat[:, red_case, yellow_case]
# mu_S = mu_S_mat[:, red_case, yellow_case]
# equi_premium = mu_S - r_mat
# dR = dR_mat[:, red_case, yellow_case]
# window = 60
# R_cumu = (np.cumsum(dR, axis=2)[:, :, window:] - np.cumsum(dR, axis=2)[:, :, :-window]) / (window * dt)
#
# y_list = [r_mat, theta_mat, sigma_S, beta, mu_S, equi_premium, R_cumu]
# Z = np.cumsum(dZ_Y_cases[red_case])
# Z_SI = np.cumsum(dZ_SI_cases[yellow_case])
# n_lines = 2
# y_title_list = [r'Interest rate $r_t$',
#                 r'Market price of risk $\theta_t$',
#                 r'Stock volatility $\sigma_t^S$',
#                 r'Average consumption wealth ratio $\bar{\beta}_t$',
#                 r'Expected returns $\mu_t^S$',
#                 r'Equity premium $\mu_t^S - r_t$',
#                 r'Realized stock returns $dR$, 5y moving window, annualized']
# labels = [scenario_labels, label_phi]
#
# for m, y_mat in enumerate(y_list):
#     fig, axes = plt.subplots(nrows=2, ncols=1, sharex='all', sharey='all', figsize=(15, 10))
#     y_title = y_title_list[m]
#
#     for j, ax in enumerate(axes):
#         ax.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
#         ax.plot(t, Z, color='red', linewidth=0.5, label=r'$z^Y_t$')
#         ax.plot(t, Z_SI, color='gold', linewidth=0.5, label=r'$z^{SI}_t$')
#         ax.tick_params(axis='both', labelcolor='black')
#         ax.set_xlabel('Time in simulation')
#
#         ax2 = ax.twinx()
#         ax2.set_ylabel(y_title, color='black')
#
#         y_vec = y_mat[:, 1] if j == 0 else y_mat[1]
#         n_lines = 3 if j == 0 else 3
#         for i in range(n_lines):
#             y = y_vec[i]  # Nt
#             color_i = colors_short2[i] if j == 0 else colors_short[i]
#             ax2.plot(t, y, label=labels[j][i], color=color_i, linewidth=0.4) if m != 6 \
#                 else ax2.plot(t[window:], y, label=labels[j][i], color=color_i, linewidth=0.4)
#         # if i < 2:
#         #     ax2.set_ylim(lower, upper)
#         if j == 0:
#             ax.legend(loc='upper left')
#         ax2.legend(loc='upper right')
#         # title_app = r', $\phi=0.4$' if j == 0 else ', reentry'
#         ax.set_title(y_title)
#         # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#         # fig.savefig('r and theta, subfig ' + str(j + 1) + '.png', bbox_inches=extent.expanded(1.2, 1.3), dpi=200)
#     fig.tight_layout(h_pad=2)
#     plt.savefig('Hrho' + str(m) + str(red_case) + str(yellow_case) + 'HD.png', dpi=200)
#     # plt.savefig('r and theta,' + str(red_case) + str(yellow_case) + 'HD.png', dpi=200)
#     plt.show()
#     plt.close()
#
# # regressions:
# # 2 different measures:
# #   (1) RCFS 2021 paper, sqrt of moving average of squared stock returns (~Integrated GARCH)
# #   (2) GJR-GARCH
# x_horizon1 = x_horizon_raw[: -horizon]
# x_horizon1 = x_horizon1 / np.std(x_horizon1)
# x_horizon1 = x_horizon1.reshape(-1, 1)
# x_horizon = sm.add_constant(x_horizon1)
#
# model = sm.OLS(y_horizon, x_horizon)
# est = model.fit()
#
# regression_results_uni[i, j, k, l, m, 0] = est.params[1]
# regression_results_uni[i, j, k, l, m, 1] = est.tvalues[1]
# regression_results_uni[i, j, k, l, m, 2] = est.rsquared
#
# # ~Integrated GARCH
# horizon = 3
# Inte_garch_vola = np.sqrt(
#     (np.cumsum(dR_mat ** 2, axis=4)[:, :, :, :, window:]
#      - np.cumsum(dR_mat ** 2, axis=4)[:, :, :, :, :-window])
#     / (window * dt))
# horizon_vola = np.sqrt(
#     (np.cumsum(sigma_S_mat ** 2, axis=4)[:, :, :, :, horizon + 1:]
#      - np.cumsum(sigma_S_mat ** 2, axis=4)[:, :, :, :, 1:-horizon])
#     / (horizon))  # transform monthly data to #horizon-monthly data
#
# # GJR-GARCH
# Garch_results_coef = np.zeros((3, 2, 2, 3, 4))
# Garch_results_tvar = np.zeros((3, 2, 2, 3, 4))
# # y_mat = sigma_S_mat ** 2
# y_mat = horizon_vola
# for i in range(n_scenarios_short):
#     for j in range(2):
#         dZ = np.reshape(dZ_Y_cases[j], (-1, 1))
#         Z = np.cumsum(dZ, axis=0)
#         dZ_horizon = (Z[horizon:] - Z[:-horizon])[np.arange(0, Nt - horizon, horizon)]
#         for k in range(2):
#             for l in range(n_phi_short):
#                 vola_raw = y_mat[i, j, k, l, np.arange(0, Nt - horizon, horizon)]
#                 y = vola_raw[1:]
#                 y = (y - np.average(y)) / np.std(y)
#                 reshape_sigma = np.reshape(vola_raw, (-1, 1))
#                 x_2 = reshape_sigma[:-1]
#                 x_2 = (x_2 - np.average(x_2)) / np.std(x_2)
#                 # x_1_raw = (reshape_sigma * dZ) ** 2
#                 x_1 = reshape_sigma[:-1] * dZ_horizon[1:]
#                 x_1 = (x_1 - np.average(x_1)) / np.std(x_1)
#                 # x_2_raw = reshape_sigma ** 2
#                 contraction = -mu_Y / sigma_Y * dt * horizon
#                 condi = (dZ_horizon[1:] < contraction) * (dZ_horizon[:-1] < contraction)
#                 x_3 = condi * x_1
#                 x_4 = condi * x_2
#                 x_3 = (x_3 - np.average(x_3)) / np.std(x_3)
#                 x_4 = (x_4 - np.average(x_4)) / np.std(x_4)
#                 # x_raw = np.concatenate((x_1, x_2, x_3, x_4), axis=1)
#                 x_raw = np.concatenate((x_1, x_2, x_4), axis=1)
#                 x = sm.add_constant(x_raw)
#                 model = sm.OLS(y, x)
#                 est = model.fit()
#                 Garch_results_coef[i, j, k, l] = est.params
#                 Garch_results_tvar[i, j, k, l] = est.tvalues
#
# # ######################################
# # ############  Figure 8  ##############
# # ######################################
# # portfolio in time series
# # bad & Bad, phi = 0.4
# # portfolio, different phi
# # portfolio, different scenarios
# print('Figure 8')
# cohort_index = 2
# left_t = (cohort_index + 1) * 100
# right_t = (cohort_index + 2) * 100
# red_case = 1
# yellow_case = 1
# phi_index = 1
# scenario_index = 1
# Z = np.cumsum(dZ_Y_cases[red_case])[int(left_t / dt):int(right_t / dt)]
# Z_SI = np.cumsum(dZ_SI_cases[yellow_case])[int(left_t / dt):int(right_t / dt)]
# x = t[int(left_t / dt):int(right_t / dt)]
# y1_pi = pi_time_series[:, red_case, yellow_case, phi_index,
#         cohort_index]  # (n_scenarios, 2, 2, n_phi_short, nn, length) -> (n_scenarios, length)
# y2_pi = pi_time_series[scenario_index, red_case, yellow_case, :,
#         cohort_index]  # (n_scenarios, 2, 2, n_phi_short, nn, length) -> (n_phi_short, length)
# y1_belief = (Delta_time_series[:, red_case, yellow_case, phi_index, cohort_index] -
#              Delta_bar_compare[:, red_case, yellow_case, phi_index]) / sigma_Y
# y2_belief = (Delta_time_series[scenario_index, red_case, yellow_case, :, cohort_index] -
#              Delta_bar_compare[scenario_index, red_case, yellow_case, :]) / sigma_Y
# y1_wealth = 1 / Phi_compare[:, red_case, yellow_case, phi_index]
# y2_wealth = 1 / Phi_compare[scenario_index, red_case, yellow_case]
# y_cases = [y1_pi, y1_belief, y1_wealth,
#            y2_pi, y2_belief, y2_wealth]
# titles_subfig = [r'Portfolios, across scenarios, $\phi=0.4$', r'Belief component, across scenarios, $\phi=0.4$',
#                  r'Wealth component, across scenarios, $\phi=0.4$',
#                  r'Portfolios, reentry, across values of $\phi$', r'Belief component, reentry, across values of $\phi$',
#                  r'Wealth component, reentry, across values of $\phi$']
# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
# for i, ax in enumerate(axes.flat):
#     n_loop = n_phi_short if i >= 3 else n_scenarios_short
#     labels = label_phi if i >= 3 else scenario_labels
#     for j in range(n_loop):
#         label_i = labels[j]
#         if i == 2 and j == 0:
#             length = int(right_t / dt) - int(left_t / dt)
#             y_case = np.ones(length)
#             ax.plot(x, y_case, label=label_i, color=colors_short[j], linewidth=0.4)
#         else:
#             y_case = y_cases[i][j, int(left_t / dt):int(right_t / dt)]
#             labels = label_phi if i > 2 else scenario_labels
#             ax.plot(x, y_case, label=label_i, color=colors_short[j], linewidth=0.4)
#     if i == 1 or i == 4:
#         if i == 1:
#             ax.set_ylim(-6, 3)
#         else:
#             ax.set_ylim(-7, 4)
#         ax.set_xlabel('Time in simulation')
#         ax.set_ylabel(r'Investment in stock market, Belief component')
#     else:
#         ax.set_ylim(-1, 7)
#     if i == 0 or i == 3:
#         ax.legend(loc='upper left')
#         ax.set_ylabel(r'Investment in stock market, $\pi_{s,t}/W_{s,t}$')
#     if i == 2 or i == 5:
#         ax.set_ylabel(r'Investment in stock market, Wealth component')
#     ax.set_title(titles_subfig[i])
#     ax.tick_params(axis='y', labelcolor='black')
#     fig.tight_layout(h_pad=3, w_pad=3)
#     extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#     fig.savefig('Shocks and Portfolio, subfig ' + str(i + 1) + str(j + 1) + '.png',
#                 bbox_inches=extent.expanded(1.25, 1.3),
#                 dpi=200)
# plt.savefig('Shocks and Portfolio,' + str(red_case) + str(yellow_case) + '.png', dpi=60)
# plt.savefig('Shocks and Portfolio,' + str(red_case) + str(yellow_case) + 'HD.png', dpi=200)
# # plt.show()
# # plt.close()
#
# ############ IA other cases
#
# titles_subfig = [r'Portfolios, across scenarios', r'Belief component, across scenarios, $\phi=0.4$',
#                  r'Wealth component, across scenarios, $\phi=0.4$']
# yaxis_subfig = r'Investment in stock market, $\pi_{s,t}/W_{s,t}$'
# phi_fix = 1
# cohort_index = 2
# left_t = (cohort_index + 1) * 100
# right_t = (cohort_index + 2) * 100
# red_cases = [0, 0, 1]
# yellow_cases = [0, 1, 0]
# scenario_indexes = [0, 1, 2]
# phi_indexes = [0, 2]
# pi_compare_cohort = pi_time_series[:, :, :, :, cohort_index]
# Delta_compo_cohort = (Delta_time_series[:, :, :, :, cohort_index] - Delta_bar_compare) / sigma_Y
# var_list = [pi_compare_cohort, Delta_compo_cohort, 1 / Phi_compare]
# left_t = 300
# right_t = 400
# x = t[int(left_t / dt):int(right_t / dt)]
#
# fig, axes = plt.subplots(nrows=3, ncols=3, sharex='all', figsize=(15, 15))
# for i, ax_row in enumerate(axes):
#     red_case = red_cases[i]
#     yellow_case = yellow_cases[i]
#     for j, ax in enumerate(ax_row):
#         title_subfig = titles_subfig[j]
#         var = var_list[j][:, red_case, yellow_case]
#         for scenario_index in scenario_indexes:
#             color_use = colors_short[scenario_index]
#             y = var[scenario_index, phi_fix, int(left_t / dt):int(right_t / dt)]
#             labels = scenario_labels
#             ax.plot(x, y, label=labels[scenario_index] + ', ' + label_phi[phi_fix], linewidth=0.5,
#                     color=colors_short[scenario_index], linestyle='solid')
#             if j == 0:
#                 ax.set_ylabel(yaxis_subfig, color='black')
#                 if scenario_index == 1:
#                     for phi_index in phi_indexes:
#                         y0 = var[scenario_index, phi_index, int(left_t / dt):int(right_t / dt)]
#                         line_style = 'dotted' if phi_index == 0 else 'dashed'
#                         ax.plot(x, y0, label=labels[scenario_index] + ', ' + label_phi[phi_index], linewidth=0.5,
#                                 color=colors_short[scenario_index], linestyle='dotted')
#                 if i == 0:
#                     ax.legend(loc='upper right')
#         if i == 0:
#             ax.set_title(title_subfig)
#         if i == 2:
#             ax.set_xlabel('Time in simulation')
# fig.tight_layout(h_pad=2)
# plt.savefig('IA Shocks and Portfolio.png', dpi=100)
# # plt.show()
# # plt.close()
#
#
# # ######################################
# # ############ Figure  9.1 #############
# # ######################################
# # consumption, different tax rates
# # build consumption data and time series over tau values
# print('Figure 9')
# tax_vector = [0.008, 0.010, 0.012]
# n_tax = len(tax_vector)
# dZ = dZ_Y_cases[1]
# dZ_SI = dZ_SI_cases[1]
# n_scenarios = 1
# n_phi_short = 1
# cohort_index = 2
# left_t = int(starts[cohort_index] / dt)
# right_t = int(left_t + 100 / dt)
# scenario_index = 1
# phi_index = 0
# line_styles = [(0, (5, 10)), 'solid', (0, (1, 1))]
# titles_subfig = [r'Individual consumption share $c_{s,t}/Y_t$', r'Cohort consumption share $f_{s,t}$']
# titles_subfig_IA = [r'Individual consumption share', r'Cohort consumption share']
# yaxis_subfig = [r'$c_{s,t}/Y_t$', r'$f_{s,t}$']
# cases = [0, 1]
#
# for case_dzY in cases:
#     for case_dzSI in cases:
#         dZ_build = dZ_build_matrix[0]
#         dZ_SI_build = dZ_SI_build_matrix[0]
#         dZ = dZ_Y_cases[case_dzY]  # bad
#         dZ_SI = dZ_SI_cases[case_dzSI]  # bad
#         cons_compare_tau = np.zeros((n_scenarios, n_tax, n_phi_short, Nt, Nc))
#         f_compare_tau = np.zeros((n_scenarios, n_tax, n_phi_short, Nt, Nc))
#         pi_compare_tau = np.zeros((n_scenarios, n_tax, n_phi_short, Nt, Nc))
#         for g, scenario in enumerate([scenarios_short[scenario_index]]):
#             mode_trade = scenario[0]
#             mode_learn = scenario[1]
#             for k, tax_try in enumerate(tax_vector):
#                 beta_try = rho + nu - tax_try
#                 for l, phi_try in enumerate([phi_vector_short[phi_index]]):
#                     (
#                         r,
#                         theta,
#                         f,
#                         Delta,
#                         pi,
#                         popu_parti,
#                         Phi_parti,
#                         Delta_bar_parti,
#                         dR,
#                         invest_tracker,
#                         popu_can_short,
#                         popu_short,
#                         Phi_can_short,
#                         Phi_short,
#                     ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S,
#                                     tax_try,
#                                     beta_try,
#                                     phi_try,
#                                     Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
#                                     need_f='True',
#                                     need_Delta='False',
#                                     need_pi='True',
#                                     top=0.05,
#                                     old_limit=100
#                                     )
#                     # invest_tracker = pi > 0
#                     cons_compare_tau[g, k, l] = f * dt / cohort_size_mat
#                     f_compare_tau[g, k, l] = f
#                     pi_compare_tau[g, k, l] = pi
#
#         length = len(t)
#         pi_time_series_tau = np.zeros((n_scenarios, n_tax, n_phi_short, nn, length))
#         cons_time_series_tau = np.zeros((n_scenarios, n_tax, n_phi_short, nn, length))
#         f_time_series_tau = np.zeros((n_scenarios, n_tax, n_phi_short, nn, length))
#         switch_time_series_tau = np.zeros((n_scenarios, n_tax, n_phi_short, nn, length))
#         parti_time_series_tau = np.zeros((n_scenarios, n_tax, n_phi_short, nn, length))
#         for o in range(n_scenarios):
#             for i in range(n_tax):
#                 for l in range(n_phi_short):
#                     pi = pi_compare_tau[o, i, l]
#                     cons = cons_compare_tau[o, i, l]
#                     f = f_compare_tau[o, i, l]
#                     for m in range(nn):
#                         start = int((m + 1) * 100 * (1 / dt))
#                         starts[m] = start * dt
#                         for n in range(length):
#                             if n < start:
#                                 pi_time_series_tau[o, i, l, m, n] = np.nan
#                                 cons_time_series_tau[o, i, l, m, n] = np.nan
#                                 f_time_series_tau[o, i, l, m, n] = np.nan
#                             else:
#                                 cohort_rank = length - (n - start) - 1
#                                 pi_time_series_tau[o, i, l, m, n] = pi[n, cohort_rank]
#                                 cons_time_series_tau[o, i, l, m, n] = cons[n, cohort_rank]
#                                 f_time_series_tau[o, i, l, m, n] = f[n, cohort_rank]
#                         parti = pi_time_series_tau[o, i, l, m] > 0
#                         switch = abs(parti[1:] ^ parti[:-1])
#                         sw = np.append(switch, 0)
#                         parti = np.where(sw == 1, 0.5, parti)
#                         switch = np.insert(switch, 0, 0)
#                         parti = np.where(switch == 1, 0.5, parti)
#                         parti_time_series_tau[o, i, l, m] = parti
#                         switch_time_series_tau[o, i, l, m] = sw
#
#         fig, axes = plt.subplots(ncols=2, figsize=(14, 5))
#         for i, ax in enumerate(axes):
#             if case_dzY == case_dzSI == scenario_index == 1:
#                 ax.set_title(titles_subfig[i] + ', ' + scenario_labels[scenario_index] + ', ' + r'$\phi=0.0$')
#             else:
#                 ax.set_title(titles_subfig_IA[i] + ', ' + red_labels[case_dzY] + scenario_labels[
#                     scenario_index] + ', ' + r'$\phi=0.0$')
#
#             var = cons_time_series_tau if i == 0 else f_time_series_tau
#             for k in range(n_tax):
#                 y = var[0, k, 0,
#                     cohort_index,
#                     left_t:right_t]  # n_scenarios, 2, 2, n_tax, n_phi_short, nn, length
#                 # condition = pi_time_series_tau[scenario_index, red_index, yellow_index, k, 2, phi_index, int(left_t/dt):int(right_t/dt)]
#                 switch = switch_time_series_tau[0, k, 0,
#                          cohort_index,
#                          left_t:right_t]
#                 y_switch = np.ma.masked_where(switch == 0, y)
#                 # y_N = np.ma.masked_where(condition >= 0.8, y)
#                 # y_P = np.ma.masked_where(condition <= 0.2, y)
#                 label_i = r'$\tau$ = ' + str('{0:.3f}'.format(tax_vector[k]))
#                 if k == 1:
#                     ax.plot(t[left_t:right_t], y, color='black', linewidth=0.6, linestyle=line_styles[k], label=label_i)
#                 else:
#                     ax.plot(t[left_t:right_t], y, color='black', linewidth=0.4, linestyle=line_styles[k], label=label_i)
#                 if scenario_index != 0:
#                     if k == 0:
#                         ax.scatter(t[left_t:right_t], y_switch, color='red', s=10, marker='o', label='Switch')
#                     else:
#                         ax.scatter(t[left_t:right_t], y_switch, color='red', s=10, marker='o')
#             ax.tick_params(axis='both', labelcolor='black')
#             ax.set_ylabel(yaxis_subfig[i], color='black')
#             # ax.set_xlim(left_t, right_t)
#             if i == 0:
#                 ax.legend()
#             ax.set_xlabel('Time in simulation')
#             fig.tight_layout(h_pad=2, w_pad=2)
#             if case_dzY == case_dzSI == scenario_index == 1:
#                 extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#                 fig.savefig('Consumption share tau, subfig ' + str(i + 1) + '.png',
#                             bbox_inches=extent.expanded(1.25, 1.3),
#                             dpi=200)
#
#         if case_dzY == case_dzSI == scenario_index == 1:
#             plt.savefig('Consumption share tau.png', dpi=100)
#             plt.savefig('Consumption share tau HD.png', dpi=200)
#         else:
#             plt.savefig('IA Consumption share tau' + str(case_dzY) + str(case_dzSI) + str(scenario_index) + '.png',
#                         dpi=100)
#         plt.show()
#         plt.close()

print('Figure 5 & 13')
# ######################################
# ############ Fig 5 & 13 ##############
# ###### Distribution of Delta #########
# ######################################
cases = [1]
# cohort_size_mat = np.tile(cohort_size, (Nc, 1))
cohort_size_flat = cohort_size[0]
Npres_try = [60, 240]
scenario = scenarios[1]
for case_dzY in cases:
    for case_dzSI in cases:
        dZ = dZ_Y_cases[case_dzY]
        dZ_SI = dZ_SI_cases[case_dzSI]
        for j, Npre_try in enumerate(Npres_try):
            T_hat_try = dt * Npre_try
            Vhat_try = (sigma_Y ** 2) / T_hat_try
            mode_trade = scenario[0]
            mode_learn = scenario[1]
            (
                r,
                theta,
                f_c,
                Delta,
                pi,
                parti,
                Phi_parti,
                Delta_bar_parti,
                Delta_tilde_parti,
                dR,
                mu_S,
                sigma_S,
                beta,
                invest_tracker,
                parti_age_group,
                parti_wealth_group,
            ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, nu,
                            Vhat_try,
                            mu_Y, sigma_Y, tax, beta0, phi,
                            Npre_try,
                            Ninit,
                            T_hat_try,
                            dZ_build_case, dZ, dZ_SI_build_case, dZ_SI, tau, cutoffs_age,
                            Ntype, rho_i, alpha_i, beta_i, beta_cohort_type, cohort_type_size,
                            need_f='False',
                            need_Delta='True',
                            need_pi='True',
                            )
            theta_compare = theta
            Delta_compare = Delta
            invest_tracker_compare = invest_tracker

            y_overall = np.empty((Nt, 5))  # overall
            y_P = np.empty((Nt, 5))  # participants / long
            y_N = np.empty((Nt, 5))  # non-participants / short
            y_min = np.empty((Nt, n_age_cutoffs))
            y_max = np.empty((Nt, n_age_cutoffs))
            y_cases = [y_overall, y_P, y_N]
            for n in range(n_age_cutoffs):
                Delta_age_group = Delta_compare[:, cutoffs_age[n + 1]:cutoffs_age[n]]
                y_min[:, n] = np.amin(Delta_age_group, axis=1)
                y_max[:, n] = np.amax(Delta_age_group, axis=1)
            for m in range(Nt):
                Delta = Delta_compare[m]  # ((Nt, Nc))
                parti_cohorts = invest_tracker_compare[m]
                if np.sum(parti_cohorts) == Nt:
                    cohort_sizes = [cohort_size_flat]
                    Deltas = [Delta]
                    n_var = 1
                else:
                    Delta1 = parti_cohorts * Delta
                    Delta2 = (1 - parti_cohorts) * Delta
                    cohort_size1 = parti_cohorts * cohort_size_flat
                    cohort_size2 = (1 - parti_cohorts) * cohort_size_flat
                    cohort_sizes = [cohort_size_flat, cohort_size1, cohort_size2]
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
                    y_cases[n][m] = Delta_cutoff
            left_t = 200
            right_t = 400
            Z = np.cumsum(dZ)[int(left_t / dt):int(right_t / dt)]
            Z_SI = np.cumsum(dZ_SI)[int(left_t / dt):int(right_t / dt)]
            x = t[int(left_t / dt):int(right_t / dt)]
            fig, axes = plt.subplots(ncols=2, sharex='all', sharey='all', figsize=(15, 7))
            y1 = y_overall[int(left_t / dt):int(right_t / dt)]
            y2 = y_P[int(left_t / dt):int(right_t / dt)]  # ((Nt, 5))
            y3 = y_N[int(left_t / dt):int(right_t / dt)]  # ((Nt, 5))
            y4 = y_min[int(left_t / dt):int(right_t / dt)]
            y5 = y_max[int(left_t / dt):int(right_t / dt)]
            belief_cutoff_case = -theta_compare[int(left_t / dt):int(right_t / dt)]
            for jj, ax in enumerate(axes):
                ax.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
                if j == 0:
                    ax.set_title(r'Distribution of estimation error, $5$-year initial window$')
                else:
                    ax.set_title('Distribution of estimation error')
                if jj == 0:
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
                    for k in range(n_age_cutoffs):
                        y40 = y4[:, k]
                        y50 = y5[:, k]
                        ax.fill_between(x, y40, y50, color=colors_short[k], linewidth=0., alpha=0.4,
                                        label=age_labels[k])
                ax.legend(loc='upper right')
                ax.set_xlabel('Time in simulation')
                fig.tight_layout(h_pad=2, w_pad=2)  # otherwise the right y-label is slightly clipped
                extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                if case_dzY == case_dzSI == 1:
                    fig.savefig('Distribution of Delta, subfig ' + str(i + 1) + str(j + 1) + '.png',
                                bbox_inches=extent.expanded(1.25, 1.3),
                                dpi=200)

            if case_dzY == case_dzSI == 1:
                if j == 0:
                    plt.savefig('f13.png', dpi=60)
                    plt.savefig('f13 HD.png', dpi=200)
                if j == 1:
                    plt.savefig('f5.png', dpi=60)
                    plt.savefig('f5 HD.png', dpi=200)
            else:
                if j == 1:
                    plt.savefig('IA ' + str(case_dzY) + str(case_dzSI) + 'Distribution of Delta.png', dpi=60)
            plt.show()
            plt.close()

print('Figure 4')
# ######################################
# ############  Figure 4  ##############
# ######################################
phi_vector = np.array([0.4, 0.0, 0.8])
t_cut = 100
N_cut = int(t_cut / dt)
data_point = np.arange(0, N_cut, 15)
x = t[:N_cut][data_point]
Delta_matrix = np.zeros((2, 3, len(data_point)))
invest_matrix = np.zeros((3, len(data_point)))
results_fig4 = np.load('results_fig4.npz')
Delta_matrix[:, 0] = np.average(results['fig4_abs_Delta'], axis=0)
invest_matrix[0] = np.average(results['fig4_parti_prob'], axis=0)
Delta_matrix[:, 1:] = np.average(results_fig4['fig4_abs_Delta'], axis=0)
invest_matrix[1:] = np.average(results_fig4['fig4_parti_prob'], axis=0)
Delta_vector = Delta_matrix
invest_vector = invest_matrix
fig_titles = [r'Reentry and complete market, average $\mid\Delta_{s,t}\mid$',
              'Reentry, average participation probability']
y_titles = [r'Average $\mid\Delta_{s,t}\mid$', 'Average participation probability']
fig, axes = plt.subplots(nrows=1, ncols=2, sharex='all', figsize=(15, 7.5))
for j, ax in enumerate(axes):
    ax.set_xlabel('Age')
    y_case = Delta_vector if j == 0 else invest_vector
    ax.set_ylabel(y_titles[j])
    for i in range(3):
        if j == 0:
            ax.set_ylim(0.04, 0.18)
            y_reentry = y_case[1, i, :N_cut]
            label_i = r'$\phi$=' + str('{0:.2f}'.format(phi_vector[i]))
            ax.plot(x, y_reentry, color=colors[i], linewidth=0.8, label=label_i)
            ax.legend()
            if phi_vector[i] != 0:
                y_complete = y_case[0, i, :N_cut]
                ax.plot(x, y_complete, color=colors[i], linewidth=0.8, linestyle='dashed')
        else:
            y = y_case[i, :N_cut]
            ax.plot(x, y, color=colors[i], linewidth=0.8)
    ax.set_title(fig_titles[j], color='black')
    ax.tick_params(axis='y', labelcolor='black')
    ax.set_xlim(-1, 100)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Average estimation error and age.png', dpi=60)
plt.show()
# plt.close()

# Figure 4
percentiles_x = np.linspace(10, 90, 9)
n_bins = len(percentiles_x)
x_var = results_fig4['fig4_belief_pre']
condition_var = results_fig4['fig4_parti_pre']
y_var = results_fig4['fig4_belief_post'] - x_var
data_figure_x = np.zeros((2, n_age_cutoffs, n_bins - 1))
data_figure_parti = np.zeros((n_age_cutoffs, n_bins - 1))
data_average_y = np.zeros((2, n_age_cutoffs, n_bins - 1))
x_all_var = np.average(x_var, axis=2)
condition_all_var = np.average(condition_var, axis=1)
y_all_var = np.average(y_var, axis=2)
data_all_x = np.zeros((2, n_bins - 1))
data_all_condition = np.zeros((n_bins - 1))
data_all_y = np.zeros((2, n_bins - 1))
for sce_index in range(2):
    x_all_focus = x_all_var[:, sce_index]
    y_all_focus = y_all_var[:, sce_index]  # for the reentry scenario
    bins = np.percentile(x_all_focus, percentiles_x)
    for i in range(n_bins - 1):
        bin_0 = bins[i]
        bin_1 = bins[i + 1]
        below_bin = bin_1 > x_all_focus
        above_bin = x_all_focus >= bin_0
        data_all_y[sce_index, i] = np.ma.average(
            y_all_focus[np.where(below_bin * above_bin == 1)])
        data_all_x[sce_index, i] = np.percentile(
            x_all_focus[np.where(below_bin * above_bin == 1)], 50)
        if sce_index == 1:
            condi_all_focus = condition_all_var
            condi_bin = condi_all_focus[np.where(below_bin * above_bin == 1)]
            data_all_condition[i] = np.average(condi_bin)
    for age_index in range(n_age_cutoffs):
        x_focus = x_var[:, sce_index, age_index]
        y_focus = y_var[:, sce_index, age_index]  # for the reentry scenario
        bins = np.percentile(x_focus, percentiles_x)
        for i in range(n_bins - 1):
            bin_0 = bins[i]
            bin_1 = bins[i + 1]
            below_bin = bin_1 > x_focus
            above_bin = x_focus >= bin_0
            data_average_y[sce_index, age_index, i] = np.ma.average(
                y_focus[np.where(below_bin * above_bin == 1)])
            data_figure_x[sce_index, age_index, i] = np.percentile(
                x_focus[np.where(below_bin * above_bin == 1)], 50)
            if sce_index == 1:
                condi_focus = condition_var[:, age_index]
                condi_bin = condi_focus[np.where(below_bin * above_bin == 1)]
                data_figure_parti[age_index, i] = np.average(condi_bin)

X_1 = np.linspace(data_all_x.min(), data_all_x.max(), 200)
X_2 = np.linspace(data_figure_x[:, 0].min(), data_figure_x[:, 0].max(), 200)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7.5))
labels = scenario_labels
for j, ax in enumerate(axes):
    for i in range(n_scenarios_short-1):
        if j == 0:
            y = data_all_y[i]  # average updates given prior belief, reentry
            x = data_all_x[i]
        else:
            y = data_average_y[i, 0]  # average updates given prior belief, reentry, youngest group
            x = data_figure_x[i, 0]
        X_Y_Spline = make_interp_spline(x, y, k=3)
        # X_ = np.linspace(x.min(), x.max(), 200)
        X_ = X_1 if j == 0 else X_2
        Y_ = X_Y_Spline(X_)
        line_style_i = 'solid' if i == 1 else 'dashed'
        if i == 0:
            complete = np.copy(Y_)
        if j == 0:
            ax.plot(X_, Y_, color=colors_short[i], linewidth=0.8, label=labels[i], linestyle=line_style_i)
            ax.set_xlim(-0.075, 0.075)
            ax.legend()
        else:
            ax.plot(X_, Y_, color=colors_short[i], linewidth=0.8, linestyle=line_style_i)
            ax.set_xlim(-0.15, 0.15)
        if i == 1:
            x0 = np.searchsorted(-Y_, 0)
            # ax.fill_between(X_ - X_[x0], Y_, complete, color=colors_short[i], alpha=0.4)
            ax.plot(X_ - X_[x0], Y_, color=colors_short[i], linewidth=0.8, linestyle='dashed')
            y_parti = data_figure_parti[0] if j == 1 else data_all_condition
            X_Y_parti_Spline = make_interp_spline(x, y_parti, k=3)
            Y_parti = X_Y_parti_Spline(X_)
            ax2 = ax.twinx()
            if j == 0:
                ax2.plot(X_, Y_parti, color='red', linewidth=0.8, linestyle='solid')
            else:
                ax2.plot(X_, Y_parti, color='red', linewidth=0.8, label='Participation rate', linestyle='solid')
                ax2.legend()
            ax2.set_ylim(0, 1)
            ax.plot(0, 0, color=colors_short[0], marker='o')
            ax.plot(X_[x0], 0, color=colors_short[1], marker='o')
    ax.axhline(0, 0.01, 0.99, color='gray', linestyle='dashed', linewidth=0.6, alpha=0.6)
    ax.axvline(0, 0.01, 0.99, color='gray', linestyle='dashed', linewidth=0.6, alpha=0.6)
    ax.set_xlabel(r'Average estimation error, prior')
    # ax.set_xlim(x[0], x[3])
    # ax.set_ylim(y[3], y[0])
    title_i = 'Overall population, ' if j == 0 else 'Youngest quartile, '
    ax.set_title(title_i + 'change in average estimation error over 2 years')
    if j == 0:
        ax.set_ylabel(r'Change in average estimation error, post - prior')
    else:
        ax2.set_ylabel(r'Participation rate, prior')
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig('f2_2.png', dpi=100)
plt.show()
# plt.close()


print('Figure 8')
######################################
############  Figure 8  ##############
######################################
# winsorize extreme shocks
Npre_index = 0

Phi_compare = results['fig8_Phi']
Delta_bar_compare = results['fig8_Delta_bar']
belief_popu = results['fig8_belief']
Phi_old_compare = results['fig8_Phi_old']
Phi_young_compare = results['fig8_Phi_young']
belief_f_old = results['fig8_old_belief_fc']
belief_f_young = results['fig8_young_belief_fc']
belief_f_old_compare = np.copy(belief_f_old)
belief_f_old_compare[np.isnan(belief_f_old_compare)] = 0
belief_f_young_compare = np.copy(belief_f_young)
belief_f_young_compare[np.isnan(belief_f_young_compare)] = 0
average_Delta_bar = np.mean(np.mean(Delta_bar_compare, axis=2), axis=0)
x_index = belief_popu
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
# quartile_var = belief_popu_young_compare[:, Npre_index] - belief_popu_old_compare[:, Npre_index]
# quartile_var = parti_young_compare[:, :, Npre_index]
y_percentiles = [50, 25, 75]
phi_old = Phi_old_compare / Phi_compare
phi_young = Phi_young_compare / Phi_compare
data_var = Delta_bar_compare
data_figure = np.zeros((n_scenarios_short, n_bins - 1, len(y_percentiles)))
phi_figure = np.zeros((n_scenarios_short, 2, n_bins - 1, len(y_percentiles)))
belief_figure = np.zeros((n_scenarios_short, 2, n_bins - 1, len(y_percentiles)))
belief_phi_figure = np.zeros((n_scenarios_short, 2, n_bins - 1, len(y_percentiles)))
for i in range(n_scenarios_short):
    data_focus = data_var[:, i]
    phi_var_old = phi_old
    phi_var_young = phi_young
    for j in range(n_bins - 1):
        bin_0 = bins[j]
        bin_1 = bins[j + 1]
        below_bin = bin_1 >= x_index
        above_bin = x_index >= bin_0
        bin_where = np.where(below_bin * above_bin == 1)
        data_focus_z = data_focus[bin_where]
        data_figure[i, j] = np.percentile(data_focus_z, y_percentiles)
        for l in range(2):
            phi_var = phi_old[:, i] if l == 0 else phi_young[:, i]
            belief_var_nan = belief_f_old[:, i] if l == 0 else belief_f_young[:, i]
            belief_var = belief_f_old_compare[:, i] if l == 0 else belief_f_young_compare[:, i]
            phi_figure[i, l, j] = np.percentile(phi_var[bin_where], y_percentiles)
            belief_figure[i, l, j] = np.nanpercentile(belief_var_nan[bin_where], y_percentiles)
            belief_phi = phi_var * belief_var
            belief_phi_figure[i, l, j] = np.percentile(belief_phi[bin_where], y_percentiles)

bin_size = (above_dz - below_dz) / (n_bins - 1)
x = np.linspace(below_dz + bin_size / 2, above_dz - bin_size / 2, n_bins - 1)
labels = [[r'$\bar{\Delta}_t^{old}$', r'$\bar{\Delta}_t^{young}$'],
          [r'$\bar{\Phi}_t^{old} / (\bar{\Phi}_t^{old} + \bar{\Phi}_t^{young})$', r'$\bar{\Phi}_t^{young} / (\bar{\Phi}_t^{old} + \bar{\Phi}_t^{young})$'],
          [r'$\bar{\Phi}_t^{old}\bar{\Delta}_t^{old} / (\bar{\Phi}_t^{old} + \bar{\Phi}_t^{young})$',
           r'$\bar{\Phi}_t^{young}\bar{\Delta}_t^{young} / (\bar{\Phi}_t^{old} + \bar{\Phi}_t^{young})$']]
# labels = [r'Wealth old minus young, ' + 'Lowest quartile', 'Second quartile', 'Third quartile', 'Highest quartile']
sub_titles = ['Complete market', 'Reentry', 'Disappointment']
y_labels = ['Estimation error of the participants',
            'Consumption share of the participants',
            r'Contribution to $\bar{\Delta}_t$']
# X_ = np.linspace(-0.2, 0.2, 100)
X_ = np.linspace(below_dz, above_dz, 100)
fig, axes = plt.subplots(ncols=2, nrows=3, figsize=(15, 12), sharex='all', sharey='row')
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
                X_Y_Spline = make_interp_spline(x, y_i)
                Y_ = X_Y_Spline(X_)
                ax.plot(X_, Y_[:, 0], color=colors_short[k], linewidth=1, label=labels[j][k])
                ax.fill_between(X_, Y_[:, 2], Y_[:, 1], color=colors_short[k], linewidth=0., alpha=0.4)
                if k == 1:
                    y_i = data_figure[i]  # n_bin-1 * 3
                    X_Y_Spline = make_interp_spline(x, y_i)
                    Y_ = X_Y_Spline(X_)
                    ax.plot(X_, Y_[:, 0], color='gray', linewidth=1,
                            label=r'$\bar{\Delta}_t$')
                    ax.fill_between(X_, Y_[:, 2], Y_[:, 1], color='gray', linewidth=0., alpha=0.2)
                    ax.axhline(average_Delta_bar[i], 0.05, 0.95, color='saddlebrown', linewidth=0.8, linestyle='dashed',
                               label=r'Unconditional mean $\bar{\Delta}_t$')
                    x_mean = [X_[np.searchsorted(Y_[:, 0], average_Delta_bar[i])]]
                    ax.scatter(x_mean, [average_Delta_bar[i]], marker='o', color='saddlebrown')
            else:
                y_i = y_focus[k]
                X_Y_Spline = make_interp_spline(x, y_i)
                Y_ = X_Y_Spline(X_)
                ax.plot(X_, Y_[:, 0], color=colors_short[k], linewidth=1, label=labels[j][k])
                ax.fill_between(X_, Y_[:, 2], Y_[:, 1], color=colors_short[k], linewidth=0., alpha=0.4)
        if i == 0:
            ax.legend(loc='upper left')
            ax.set_ylabel(y_labels[j])
        if j == 0:
            ax.set_title(sub_titles[i])
        if j == 2:
            ax.set_xlabel(x_label)
        else:
            ax.axvline(0, 0.05, 0.95, color='gray', linewidth=0.8, linestyle='dashed')
        # ax.set_xlim(-0.25, 0.25)
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slisghtly clipped
plt.savefig('240DeltaVolaSQ.png', dpi=100)
plt.savefig('240DeltaVola HDSQ.png', dpi=200)
plt.show()

# # ######################################
# # ############  Figure 3  ##############
# # ###########  Figure 9.2  #############
# # ########### & Figure 10 ##############
# # ######################################
# print('Data generation for Figure 3 and 10')
# # varying tau only when phi == 0.0
# # storing drift, diffusion, average view when phi == 0.0 and tau == 0.01
# age_cut = 100
# Nc_cut = int(age_cut / dt)
# cohort_keep = np.arange(-Nc_cut, 1, 60)
# Nc_short = len(cohort_keep)
# data_keep = np.arange(0, Nc, 60)
# Nt_short = len(data_keep)
#
# popu_cummu = np.cumsum(cohort_size)
# popus = np.array([0.1, 0.5])
# popus_1 = 1 - popus
# cutoff_young = np.searchsorted(popu_cummu, popus_1)
# cutoff_old = np.searchsorted(popu_cummu, popus)
# diffusion_P_matrix = np.empty((Mpath, n_scenarios, Nt_short, Nc_short),
#                               dtype=np.float32)  # store data only when phi == 0
# diffusion_matrix = np.empty((Mpath, n_scenarios, Nt_short, Nc_short), dtype=np.float32)
# drift_matrix = np.empty((Mpath, n_scenarios, Nt_short, Nc_short), dtype=np.float32)
# drift_P_matrix = np.empty((Mpath, n_scenarios, Nt_short, Nc_short), dtype=np.float32)
# r_matrix = np.empty((Mpath, n_scenarios, Nt_short), dtype=np.float32)
# parti_old_matrix = np.empty((Mpath, n_scenarios, 2, Nt_short), dtype=np.float32)
# parti_young_matrix = np.empty((Mpath, n_scenarios, 2, Nt_short), dtype=np.float32)
# average_belief_matrix = np.empty((Mpath, n_scenarios, Nt_short), dtype=np.float32)
# cohort_size_short = cohort_size[-Nc_cut:]
#
# # Figure 3
# t_cut = 100
# N_cut = int(t_cut / dt)
# x = t[:N_cut]
# data_point = np.arange(0, N_cut, 15)
# Delta_vector = np.flip(np.average(Delta_matrix, axis=0), axis=2)
# invest_vector = np.flip(np.average(invest_matrix, axis=0), axis=2)
# y_cases = [Delta_vector[0], Delta_vector[1], invest_vector[1]]
# fig_titles = ['Complete market', 'Reentry', 'Reentry']
# y_titles = [r'Average $\mid\Delta_{s,t}\mid$', r'Average $\mid\Delta_{s,t}\mid$', 'Average participation probability']
# fig, axes = plt.subplots(nrows=1, ncols=3, sharex='all', figsize=(15, 5))
# for j, ax in enumerate(axes):
#     ax.set_xlabel('Age')
#     y_case = y_cases[j]
#     ax.set_ylabel(y_titles[j])
#     for i in range(5):
#         y = y_case[i, :N_cut]
#         label_i = r'$\phi$=' + str('{0:.2f}'.format(phi_5[i]))
#         ax.plot(x[data_point], y[data_point], color=colors[i], linewidth=0.5, label=label_i)
#     if j < 2:
#         ax.set_ylim(0.04, 0.18)
#     if j == 0:
#         ax.legend()
#     ax.set_title(fig_titles[j], color='black')
#     ax.tick_params(axis='y', labelcolor='black')
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# # plt.savefig('Average estimation error and age.png', dpi=60)
# # plt.show()
# # plt.close()
#
#
# # Figure 10
#
# parti_gap_mat = parti_young_matrix - parti_old_matrix
# # condition = parti_gap_mat[:, :, 0]
# condition = average_belief_matrix
# x = -np.flip(cohort_keep) * dt
# quartiles = np.arange(0, 100 + 1, 25)
# results_data = np.zeros((4, Nc_short, 6))
# results_data_uncon = np.zeros((Nc_short, 6))
# i = 0
# r_mat = r_matrix[:, i]
# drift_c_mat = np.flip(drift_matrix[:, i], axis=2)
# drift_P_mat = np.flip(drift_P_matrix[:, i], axis=2)
# drift_N_mat = drift_c_mat - drift_P_mat
# drift_P_mat = np.ma.masked_equal(drift_P_mat, 0)
# drift_N_mat = np.ma.masked_equal(drift_N_mat, 0)
# diffusion_c_mat = np.flip(diffusion_P_matrix[:, i], axis=2)  # includes both N and P
# diffusion_P_mat = np.ma.masked_equal(diffusion_c_mat, 0)  # includes only P
# condition_mat = condition[:, i]
# quartiles_condition = np.percentile(condition_mat, quartiles)
# for k in range(4):
#     condition_below = quartiles_condition[k]
#     condition_above = quartiles_condition[k + 1]
#     a = condition_mat >= condition_below
#     b = condition_mat <= condition_above
#     a_b = a * b
#     r_focus = r_mat * np.ma.masked_equal(a_b, 0)
#     a_b_mat = np.tile(np.reshape(a_b, (2800, len(data_keep), 1)), (1, 1, Nc_short))
#     masked = np.ma.masked_equal(a_b_mat, 0)
#     # condition_where = np.where(a * b == 1)
#     drift_c_focus = drift_c_mat * masked
#     diffusion_c_focus = diffusion_c_mat * masked
#     drift_P_focus = drift_P_mat * masked
#     diffusion_P_focus = diffusion_P_mat * masked
#     drift_N_focus = drift_N_mat * masked
#     # parti_focus = parti_mat * masked
#     results_data[k, :, 0] = np.nanmean(np.nanmean(drift_c_focus, axis=0), axis=0)
#     results_data[k, :, 1] = np.nanmean(np.nanmean(drift_P_focus, axis=0), axis=0)
#     results_data[k, :, 2] = np.nanmean(np.nanmean(drift_N_focus, axis=0), axis=0)
#     # results_data[k, l, 1] = np.nanstd(drift_c_focus[:, :, age_below:age_above])
#     results_data[k, :, 3] = np.nanmean(np.nanmean(diffusion_c_focus, axis=0), axis=0)
#     results_data[k, :, 4] = np.nanmean(np.nanmean(diffusion_P_focus, axis=0), axis=0)
#     results_data[k, :, 5] = np.nanmean(np.nanmean(r_focus, axis=0), axis=0) - rho
# results_data_uncon[:, 0] = np.nanmean(np.nanmean(drift_c_mat, axis=0), axis=0)
# results_data_uncon[:, 1] = np.nanmean(np.nanmean(drift_P_mat, axis=0), axis=0)
# results_data_uncon[:, 2] = np.nanmean(np.nanmean(drift_N_mat, axis=0), axis=0)
# results_data_uncon[:, 3] = np.nanmean(np.nanmean(diffusion_c_mat, axis=0), axis=0)
# results_data_uncon[:, 4] = np.nanmean(np.nanmean(diffusion_P_mat, axis=0), axis=0)
# results_data_uncon[:, 5] = np.nanmean(np.nanmean(r_mat, axis=0), axis=0) - rho
#
# # make 3 * 2 figures
# var_name = r'log$\left(c_{s,t}\right)$'
# quartile_labels = [', average belief 1st quartile', ', average belief 2nd quartile',
#                    ', average belief 3rd quartile', ', average belief 4th quartile']
# fig_titles = [r', average belief $1^{st}$ quartile', r', average belief $4^{th}$ quartile', ', overall']
# X_ = np.linspace(5, 100, 200)
# scenario = scenarios[1]
# fig, axes = plt.subplots(ncols=2, nrows=3, sharex='all', sharey='col', figsize=(15, 20))
# for k, ax_row in enumerate(axes):  # 3
#     for i, ax in enumerate(ax_row):  # 2
#         ax.set_xlabel('Age')
#         title_i = 'Drift' if i == 0 else 'Diffusion'
#         ax.set_title(title_i + fig_titles[k])
#         if k <= 1:
#             m = 0 if k == 0 else 3
#             data_focus = results_data[m]
#         else:
#             data_focus = results_data_uncon
#         X_Y_Spline = make_interp_spline(x[1:], data_focus[1:])
#         Y_ = X_Y_Spline(X_)
#         drift_c_focus = Y_[:, 0]
#         drift_P_focus = Y_[:, 1]
#         drift_N_focus = Y_[:, 2]
#         diffusion_c_focus = Y_[:, 3]
#         diffusion_P_focus = Y_[:, 4]
#         r_rho_focus = Y_[:, 5]
#         if i == 0:  # drift
#             ax.plot(X_, drift_c_focus, color='black', label='Average', alpha=0.8)
#             ax.plot(X_, drift_P_focus, color='red', label='Participants', alpha=0.8,
#                     linestyle='dashdot')
#             ax.plot(X_, drift_N_focus, color='mediumblue', label='Nonparticipants', alpha=0.8,
#                     linestyle='dashed')
#             ax.axhline(r_rho_focus[0], 0.05, 0.95, color='gray', label=r'Average $r_t - \rho$',
#                        alpha=0.4)
#             if k == 0:
#                 ax.text(12, 0.01, 'Low', size=12,
#                         bbox={'facecolor': 'w', 'alpha': 0.2, 'pad': 0.5, 'boxstyle': 'larrow'})
#                 ax.text(36, 0.01, 'Participation rate', size=12)
#                 ax.text(80, 0.01, 'High', size=12,
#                         bbox={'facecolor': 'w', 'alpha': 0.2, 'pad': 0.5, 'boxstyle': 'rarrow'})
#                 ax.legend()
#             if k == 1:
#                 ax.text(12, 0.01, 'High', size=12,
#                         bbox={'facecolor': 'w', 'alpha': 0.2, 'pad': 0.5, 'boxstyle': 'larrow'})
#                 ax.text(36, 0.01, 'Participation rate', size=12)
#                 ax.text(80, 0.01, 'Low', size=12,
#                         bbox={'facecolor': 'w', 'alpha': 0.2, 'pad': 0.5, 'boxstyle': 'rarrow'})
#             ax.set_ylabel('Average drift of ' + var_name, color='black')
#             ax.tick_params(axis='y', labelcolor='black')
#         else:  # diffusion
#             ax.plot(X_, diffusion_c_focus, color='black', alpha=0.8, label='Overall')
#             ax.plot(X_, diffusion_P_focus, color='red', alpha=0.8, label='Participants',
#                     linestyle='dashdot')
#             ax.axhline(0, 0.05, 0.95, color='mediumblue', label='Nonparticipants', alpha=0.8,
#                        linestyle='dashed')
#             ax.set_ylabel('Average volatility of ' + var_name, color='black')
#             ax.tick_params(axis='y', labelcolor='black')
# fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
# plt.savefig('log consumption and age.png', dpi=60)
# plt.savefig('log consumption and age HD.png', dpi=200)
# plt.show()
# # plt.close()

# ######################################
# ############  Figure 4  ##############
# ############ Figure 11  ##############
# ########### & Figure 15  #############
# ######################################
print('Figure 9')
belief_popu_old_compare = results["fig9_old_belief"]
belief_popu_young_compare = results["fig9_young_belief"]
P_old_compare = results["fig9_old_parti"]
P_young_compare = results["fig9_young_parti"]
Wealthshare_old_compare = results["fig9_old_fw"]
Wealthshare_young_compare = results["fig9_young_fw"]
n_tiles = 4
n_bins = 30
popu_index = 0
belief_popu_gap_compare = belief_popu_old_compare - belief_popu_young_compare
parti_gap = P_old_compare - P_young_compare
Phi_gap = Phi_old_compare - Phi_young_compare
wealth_gap = Wealthshare_old_compare - Wealthshare_young_compare
# y_variables = [parti_gap[:, popu_index, :], belief_f_distance_young[:, popu_index, :], belief_f_distance_old[:, popu_index, :]
# y_complete_variables = [theta_complete, Phi_complete, Delta_bar_complete]
# x_mat = belief_f_gap_compare[:, popu_index, :]
# x_varname = r'Wealth weighted $\Delta_{s,t}$, old minus young'
x_mat = belief_popu_gap_compare
x_varname = r'Average estimation error $\Delta_{s,t}$, old minus young'
x_range = 0.25
x_range_left = np.percentile(x_mat, 5)
x_range_right = np.percentile(x_mat, 95)
width_bins = (x_range_right - x_range_left) / n_bins
a = x_range_left <= x_mat
b = x_mat <= x_range_right
where_within = np.where(a * b == 1)  # winsorize
x_mat_within = x_mat[where_within]
total_count = np.shape(where_within)[1]
condition_var = wealth_gap
condition_label = r'Consumption share, old minus young'
condition_var_within = condition_var[where_within]
condition = np.percentile(condition_var_within, np.arange(0, 101, (100 / n_tiles)))
y = np.empty((n_tiles, n_bins, 3))
x = np.linspace(x_range_left + width_bins / 2, x_range_right - width_bins / 2, n_bins)
X_ = np.linspace(x_range_left, x_range_right, 50)
y_mat_within = parti_gap[where_within]
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
scenario_label = 'Reentry'
plt.savefig(scenario_label + 'belief two sorts.png', dpi=100)
plt.savefig(scenario_label + 'belief two sorts HD.png', dpi=200)
# plt.savefig('85-115old'+str(tax)+'Intuition'+ x_varname[:4] + 'belief two sorts.png', dpi=200)
plt.show()
# plt.close()


# distribution of wealth gap given the belief gap in [-x_range, x_range]:
n_bins = 30
fig, axes = plt.subplots(ncols=2, sharey='all', figsize=(10, 4))
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
scenario_label = 'Reentry'
plt.savefig(scenario_label + 'Intuition wealth distribution.png', dpi=100)
# plt.savefig('85-115old'+str(tax)+'Intuition wealth distribution.png', dpi=200)
plt.show()
# plt.close()


print('Figure 10')
Z_mat = np.cumsum(dZ_matrix, axis=1)
dZ_mat = Z_mat[:, 24:] - Z_mat[:, :-24]
parti_pre_mat = results["fig4_parti_pre"]
parti_post_mat = results["fig10_parti_post"]
leverage_pre_mat = results["fig10_leverage_pre"]
leverage_post_mat = results["fig10_leverage_post"]

change_parti_rate_mat = (parti_post_mat - parti_pre_mat) / 4
change_leverage_mat = leverage_post_mat - leverage_pre_mat
change_parti_rate_overall = np.sum(change_parti_rate_mat / 4, axis=1)
change_leverage_overall = 1 / np.sum(parti_post_mat / 4, axis=1) - 1 / np.sum(parti_pre_mat / 4, axis=1)
n_bins = 15
data_figure_median_parti = np.zeros(
    (n_bins - 1, n_age_cutoffs, 3))  # x: dZ^Y & dZ^SI, y: parti_rate & leverage
data_figure_median_lev = np.zeros(
    (n_bins - 1, n_age_cutoffs, 3))  # x: dZ^Y & dZ^SI, y: parti_rate & leverage
data_figure_mean_parti = np.zeros(
    (n_bins - 1, n_age_cutoffs))  # x: dZ^Y & dZ^SI, y: parti_rate & leverage
data_figure_mean_lev = np.zeros(
    (n_bins - 1, n_age_cutoffs))  # x: dZ^Y & dZ^SI, y: parti_rate & leverage
data_figure_overall_lev = np.zeros((n_bins - 1))
data_figure_overall_parti = np.zeros((n_bins - 1))
x_var = dZ_mat
x_max = np.percentile(x_var, 90)
x_min = np.percentile(x_var, 10)
x_width = (x_max - x_min) / (n_bins - 1)
x_bins = np.linspace(x_min, x_max, n_bins)
data_figure_x = (x_bins[1:] + x_bins[:-1]) / 2
for j in range(n_bins - 1):
    bin_below = x_var >= x_bins[j]
    bin_above = x_bins[j + 1] >= x_var
    data_where = np.where(bin_above * bin_below == 1)
    for m in range(n_age_cutoffs):
        if m == 0:
            data_figure_overall_lev[j] = np.average(change_leverage_overall[data_where])
            data_figure_overall_parti[j] = np.average(change_parti_rate_overall[data_where])
        y_parti_bin = change_parti_rate_mat[:, m][data_where]
        y_lev_bin = change_leverage_mat[:, m][data_where]
        data_figure_median_parti[j, m] = np.percentile(y_parti_bin, np.array([25, 50, 75]))
        data_figure_median_lev[j, m] = np.percentile(y_lev_bin, np.array([25, 50, 75]))
        data_figure_mean_parti[j, m] = np.average(y_parti_bin)
        data_figure_mean_lev[j, m] = np.average(y_lev_bin)

label_shock = r'Shocks to the output, $dz^{Y}$'
label_scenario = r'Reentry'
label_title = [r'Entry and exit in the stock market', r'Changes in participants portfolio leverage']
labels = r'$\phi = 0.4$'
age_labels = ['0 < Age <= 15, youngest quartile', '15 < Age <= 35', '35 < Age <= 69', 'Age > 69, oldest quartile']
X_ = np.linspace(-1.5, 1.5, 200)
plt.rcParams["font.family"] = "serif"
x = data_figure_x
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 7.5), sharex='all')
for i, ax in enumerate(axes):
    y_focus = data_figure_mean_parti if i == 0 else data_figure_mean_lev
    y_overall = data_figure_overall_parti if i == 0 else data_figure_overall_lev
    ax.set_xlabel(label_shock)
    ax.set_ylabel(label_title[i])
    title_i = 'Response to shocks, participation' if i == 0 else 'Response to shocks, leverage'
    ax.set_title(title_i)
    if i == 0:
        ax.set_ylim(-0.03, 0.03)
    else:
        ax.set_ylim(-0.6, 0.6)
    for age_index in range(n_age_cutoffs):
        y = y_focus[:, age_index]
        X_Y_Spline = make_interp_spline(x, y, k=5)
        Y_ = X_Y_Spline(X_)
        # ax.plot(x, y[:, 1], color=colors_short[age_index], linewidth=0.8, linestyle='solid', label=age_labels[age_index])
        # ax.fill_between(x, y[:, 0], y[:, 2], color=colors_short[age_index], linewidth=0, alpha=0.25)
        y_focus_overall = y_overall
        X_Y_overall_Spline = make_interp_spline(x, y_focus_overall, k=5)
        Y_overall = X_Y_overall_Spline(X_)
        ax.plot(X_, Y_, color=colors_short[age_index], linewidth=0.8, linestyle='solid',
                label=age_labels[age_index])
        ax.plot(X_, Y_overall, color='black', linewidth=0.8, linestyle='dashed')
        if i == 0:
            ax.legend()
        ax.axhline(0, 0.05, 0.95, color='gray', linestyle='dotted', linewidth=0.6, alpha=0.6)
        ax.axvline(0, 0.05, 0.95, color='gray', linestyle='dotted', linewidth=0.6, alpha=0.6)
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig('Reaction to shocks.png', dpi=100)
plt.savefig('Reaction to shocksHD.png', dpi=200)
plt.show()
# plt.close()


#
# print('Figure 4, 11, and 15')
# # fig 4: phi = 0.0, 0.4, & 0.8, complete vs. reentry
# # fig 11: tau = 0.01 and tau = 0.015, phi == 0, reentry & disappointment
# # fig 15: phi == 0, initial window = 60 and 240, complete, reentry & disappointment
# need_fig4 = 'false'
# need_fig11 = 'true'
# n_scenarios_short = 3  # complete vs. reentry
# scenarios_short = scenarios[:n_scenarios_short]
# if need_fig4 == 'true':
#     # ages_focus = np.array([10, 50, 100], dtype=int)
#     # ts_focus = (ages_focus / dt).astype(int)
#     # n_ages = len(ages_focus)
#     # t_begin = np.max(ts_focus)
#     t_gap = int(2 / dt)  # 2-year window
#     N_cut = int(Nc - t_gap)
#     # t_rolling = np.arange(0, Nt - 1, t_gap).astype(int)
#     # n_cut = len(t_rolling)
#     # t_rolling_pre = t_rolling[:-1]  # pre
#     # t_rolling_post = t_rolling[1:]
#     # n_gap = len(t_rolling)
#     # shocks = np.cumsum(dZ_matrix, axis=1)
#     # shocks_mat = shocks[:Mpath_short, t_rolling_post] - shocks[:Mpath_short, t_rolling_pre]
#     # shocks_SI = np.cumsum(dZ_SI_matrix, axis=1)
#     # shocks_SI_mat = shocks_SI[:Mpath_short, t_rolling_post] - shocks_SI[:Mpath_short, t_rolling_pre]
#     parti_rate_mat = np.zeros((Mpath, n_phi_short, n_age_cutoffs, N_cut), dtype=np.float32)
#     parti_rate_post_mat = np.zeros((Mpath, n_phi_short, n_age_cutoffs, N_cut), dtype=np.float32)
#     belief_pre_mat = np.zeros((Mpath, n_scenarios_short, n_phi_short, n_age_cutoffs, N_cut), dtype=np.float32)
#     belief_post_mat = np.zeros((Mpath, n_scenarios_short, n_phi_short, n_age_cutoffs, N_cut), dtype=np.float32)
#     cohort_size_short = cohort_size[1:]
#     cohort_size_short_mat = np.tile(cohort_size_short, (Nt - 1, 1))
#     cohort_size_mat = np.tile(cohort_size, (Nt, 1))
#     tau_mat = np.tile(tau, (Nt, 1))
# if need_fig11 == 'true':
#     cohort_size_mat = np.tile(cohort_size, (Nt, 1))
#     tau_mat = np.tile(tau, (Nt, 1))
#     popu_fig11 = 0.1
#     cutoff_age_old_below_fig11 = np.searchsorted(cummu_popu, popu_fig11)
#     cutoff_age_young_fig11 = np.searchsorted(cummu_popu, 1 - popu_fig11)
#     tax_short = [0.01, 0.015]
#     n_tax_short = len(tax_short)
#     P_old_compare = np.empty((Mpath, n_scenarios_short, n_tax_short, Nt), dtype=np.float32)
#     P_young_compare = np.empty((Mpath, n_scenarios_short, n_tax_short, Nt), dtype=np.float32)
#     belief_popu_old_compare_fig11 = np.empty((Mpath, Nt), dtype=np.float32)
#     belief_popu_young_compare_fig11 = np.empty((Mpath, Nt), dtype=np.float32)
#     belief_popu_fig11 = np.empty((Mpath, Nt), dtype=np.float32)
#     Wealthshare_old_compare = np.empty((Mpath, n_scenarios_short, n_tax_short, Nt), dtype=np.float32)
#     Wealthshare_young_compare = np.empty((Mpath, n_scenarios_short, n_tax_short, Nt), dtype=np.float32)
#
# # run
# for i in range(Mpath):
#     # for i in range(Mpath):
#     print(i)
#     ii = i if i < 1000 else i + 4000
#     dZ = dZ_matrix[ii]
#     dZ_build = dZ_build_matrix[ii]
#     dZ_SI = dZ_SI_matrix[ii]
#     dZ_SI_build = dZ_SI_build_matrix[ii]
#     for j, scenario in enumerate(scenarios_short):
#         mode_trade = scenario[0]
#         mode_learn = scenario[1]
#         for n, phi_try in enumerate(phi_vector_short):
#             if (need_fig4 == 'true') and (need_fig11 == 'true'):
#                 for k, tax_try in enumerate(tax_short):
#                     beta_try = rho + nu - tax_try
#                     (
#                         r,
#                         theta,
#                         f,
#                         Delta,
#                         pi,
#                         popu_parti,
#                         f_parti,
#                         Delta_bar_parti,
#                         dR,
#                         invest_tracker,
#                         popu_can_short,
#                         popu_short,
#                         Phi_can_short,
#                         Phi_short,
#                     ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu,
#                                     Vhat,
#                                     mu_Y, sigma_Y, sigma_S,
#                                     tax_try,
#                                     beta_try,
#                                     phi_try,
#                                     Npre, Ninit,
#                                     T_hat,
#                                     dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
#                                     need_f='True',
#                                     need_Delta='True',
#                                     need_pi='True',
#                                     )
#
#                     if tax_try == 0.01:
#                         for mm in range(n_age_cutoffs):
#                             age_bottom = cutoffs_age[mm + 1] if mm <= 2 else -N_cut
#                             age_top = cutoffs_age[mm]
#                             weights_group = cohort_size[age_bottom:age_top]
#                             belief_pre_mat[i, j, n, mm] = np.average(Delta[:-t_gap, age_bottom:age_top],
#                                                                      weights=weights_group,
#                                                                      axis=1)
#                             belief_post_mat[i, j, n, mm] = np.average(
#                                 Delta[t_gap:, age_bottom - t_gap:age_top - t_gap],
#                                 weights=weights_group,
#                                 axis=1)
#                             if mode_trade == 'w_constraint':
#                                 parti_rate_mat[i, n, mm] = np.average(invest_tracker[:-t_gap, age_bottom:age_top],
#                                                                       weights=weights_group, axis=1)
#                                 parti_rate_post_mat[i, n, mm] = np.average(
#                                     invest_tracker[t_gap:, age_bottom - t_gap:age_top - t_gap],
#                                     weights=weights_group, axis=1)
#                     if phi_try == 0:
#                         # save results for fig 11
#                         invest = pi > 0
#                         if j == 0 and k == 0:
#                             belief_popu_fig11[i] = np.average(Delta, weights=cohort_size,
#                                                               axis=1)  # same average belief bc phi == 0
#                             belief_popu_old_compare_fig11[i] = np.average(
#                                 Delta[:, :cutoff_age_old_below_fig11],
#                                 weights=cohort_size[:cutoff_age_old_below_fig11],
#                                 axis=1)
#                             belief_popu_young_compare_fig11[i] = np.average(Delta[:, cutoff_age_young_fig11:],
#                                                                             weights=cohort_size[
#                                                                                     cutoff_age_young_fig11:],
#                                                                             axis=1)
#                         P_old_compare[i, j, k] = np.sum(
#                             invest[:, :cutoff_age_old_below_fig11] *
#                             cohort_size_mat[:, :cutoff_age_old_below_fig11],
#                             axis=1) / popu_fig11
#                         P_young_compare[i, j, k] = np.sum(invest[:, cutoff_age_young_fig11:] *
#                                                           cohort_size_mat[:, cutoff_age_young_fig11:],
#                                                           axis=1) / popu_fig11
#                         Wealthshare_old_compare[i, j, k] = np.sum(
#                             f[:, :cutoff_age_old_below_fig11] * dt,
#                             axis=1)
#                         Wealthshare_young_compare[i, j, k] = np.sum(f[:, cutoff_age_young_fig11:] * dt,
#                                                                     axis=1)
#
#             elif need_fig11 == 'false':
#                 tax_try = tax
#                 beta_try = beta
#                 (
#                     r,
#                     theta,
#                     f,
#                     Delta,
#                     pi,
#                     popu_parti,
#                     f_parti,
#                     Delta_bar_parti,
#                     dR,
#                     invest_tracker,
#                     popu_can_short,
#                     popu_short,
#                     Phi_can_short,
#                     Phi_short,
#                 ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu,
#                                 Vhat,
#                                 mu_Y, sigma_Y, sigma_S,
#                                 tax_try,
#                                 beta_try,
#                                 phi_try,
#                                 Npre, Ninit,
#                                 T_hat,
#                                 dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
#                                 need_f='False',
#                                 need_Delta='True',
#                                 need_pi='True',
#                                 )
#
#                 for mm in range(n_age_cutoffs):
#                     age_bottom = cutoffs_age[mm + 1] if mm <= 2 else -N_cut
#                     age_top = cutoffs_age[mm]
#                     weights_group = cohort_size[age_bottom:age_top]
#                     belief_pre_mat[i, j, n, mm] = np.average(Delta[:-t_gap, age_bottom:age_top],
#                                                              weights=weights_group,
#                                                              axis=1)
#                     belief_post_mat[i, j, n, mm] = np.average(
#                         Delta[t_gap:, age_bottom - t_gap:age_top - t_gap],
#                         weights=weights_group,
#                         axis=1)
#                     if mode_trade == 'w_constraint':
#                         parti_rate_mat[i, n, mm] = np.average(invest_tracker[:-t_gap, age_bottom:age_top],
#                                                               weights=weights_group, axis=1)
#                         parti_rate_post_mat[i, n, mm] = np.average(
#                             invest_tracker[t_gap:, age_bottom - t_gap:age_top - t_gap],
#                             weights=weights_group, axis=1)
#             else:
#                 if phi_try == 0:
#                     for k, tax_try in enumerate(tax_short):
#                         beta_try = rho + nu - tax_try
#                         (
#                             r,
#                             theta,
#                             f,
#                             Delta,
#                             pi,
#                             popu_parti,
#                             f_parti,
#                             Delta_bar_parti,
#                             dR,
#                             invest_tracker,
#                             popu_can_short,
#                             popu_short,
#                             Phi_can_short,
#                             Phi_short,
#                         ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu,
#                                         Vhat,
#                                         mu_Y, sigma_Y, sigma_S,
#                                         tax_try,
#                                         beta_try,
#                                         phi_try,
#                                         Npre, Ninit,
#                                         T_hat,
#                                         dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
#                                         need_f='True',
#                                         need_Delta='True',
#                                         need_pi='True',
#                                         )
#
#                         # save results for fig 11
#                         invest = pi > 0
#                         if j == 0 and k == 0:
#                             belief_popu_fig11[i] = np.average(Delta, weights=cohort_size,
#                                                               axis=1)  # same average belief bc phi == 0
#                             belief_popu_old_compare_fig11[i] = np.average(
#                                 Delta[:, :cutoff_age_old_below_fig11],
#                                 weights=cohort_size[:cutoff_age_old_below_fig11],
#                                 axis=1)
#                             belief_popu_young_compare_fig11[i] = np.average(Delta[:, cutoff_age_young_fig11:],
#                                                                             weights=cohort_size[
#                                                                                     cutoff_age_young_fig11:],
#                                                                             axis=1)
#                         P_old_compare[i, j, k] = np.sum(
#                             invest[:, :cutoff_age_old_below_fig11] *
#                             cohort_size_mat[:, :cutoff_age_old_below_fig11],
#                             axis=1) / popu_fig11
#                         P_young_compare[i, j, k] = np.sum(invest[:, cutoff_age_young_fig11:] *
#                                                           cohort_size_mat[:, cutoff_age_young_fig11:],
#                                                           axis=1) / popu_fig11
#                         Wealthshare_old_compare[i, j, k] = np.sum(
#                             f[:, :cutoff_age_old_below_fig11] * dt,
#                             axis=1)
#                         Wealthshare_young_compare[i, j, k] = np.sum(f[:, cutoff_age_young_fig11:] * dt,
#                                                                     axis=1)
#                 else:
#                     pass
#
# # figure 11
# for i in range(Mpath):
#     print(i)
#     dZ = dZ_matrix[i]
#     dZ_build = dZ_build_matrix[i]
#     dZ_SI = dZ_SI_matrix[i]
#     dZ_SI_build = dZ_SI_build_matrix[i]
#     (
#         r,
#         theta,
#         f,
#         Delta,
#         pi,
#         popu_parti,
#         f_parti,
#         Delta_bar_parti,
#         dR,
#         invest_tracker,
#         popu_can_short,
#         popu_short,
#         Phi_can_short,
#         Phi_short,
#     ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta,
#                     phi_fix,
#                     Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
#                     need_f='True',
#                     need_Delta='True',
#                     need_pi='False',
#                     )
#     theta_compare[i] = theta
#     Phi_compare[i] = f_parti
#     Delta_bar_compare[i] = Delta_bar_parti
#     for j, popu in enumerate(popus):
#         # cutoff_age_old = np.searchsorted(cummu_popu, popu)
#         cutoff_age_old_top = np.searchsorted(cummu_popu, popu * 2)
#         cutoff_age_old_below = np.searchsorted(cummu_popu, popu)
#         cutoff_age_young = np.searchsorted(cummu_popu, 1 - popu)
#         total_popu_old = np.sum(cohort_size[cutoff_age_old_below:cutoff_age_old_top])
#         total_popu_young = np.sum(cohort_size[cutoff_age_young:])
#         P_old_compare[i, j] = np.sum(invest_tracker[:, cutoff_age_old_below:cutoff_age_old_top] *
#                                      cohort_size_mat[:, cutoff_age_old_below:cutoff_age_old_top],
#                                      axis=1) / total_popu_old
#         P_young_compare[i, j] = np.sum(invest_tracker[:, cutoff_age_young:] *
#                                        cohort_size_mat[:, cutoff_age_young:],
#                                        axis=1) / total_popu_young
#         Phi_old_compare[i, j] = np.sum(f[:, cutoff_age_old_below:cutoff_age_old_top] *
#                                        invest_tracker[:, cutoff_age_old_below:cutoff_age_old_top] * dt, axis=1)
#         Phi_young_compare[i, j] = np.sum(f[:, cutoff_age_young:] * invest_tracker[:, cutoff_age_young:] * dt, axis=1)
#         Wealthshare_old_compare[i, j] = np.sum(f[:, cutoff_age_old_below:cutoff_age_old_top] * dt, axis=1)
#         Wealthshare_young_compare[i, j] = np.sum(f[:, cutoff_age_young:] * dt, axis=1)
#         belief_f_old_compare[i, j] = np.average(Delta[:, cutoff_age_old_below:cutoff_age_old_top],
#                                                 weights=f[:, cutoff_age_old_below:cutoff_age_old_top] * dt,
#                                                 axis=1)
#         belief_f_young_compare[i, j] = np.average(Delta[:, cutoff_age_young:],
#                                                   weights=f[:, cutoff_age_young:] * dt,
#                                                   axis=1)
#         belief_popu_old_compare_fig11[i, j] = np.average(Delta[:, cutoff_age_old_below:cutoff_age_old_top],
#                                                          weights=cohort_size_mat[:,
#                                                                  cutoff_age_old_below:cutoff_age_old_top],
#                                                          axis=1)
#         belief_popu_young_compare_fig11[i, j] = np.average(Delta[:, cutoff_age_young:],
#                                                            weights=cohort_size_mat[:, cutoff_age_young:],
#                                                            axis=1)
#
# # construct the condition:
# n_tiles = 4
# n_bins = 30
# popu_index = 0
# belief_f_gap_compare = belief_f_old_compare - belief_f_young_compare
# belief_popu_gap_compare = belief_popu_old_compare_fig11 - belief_popu_young_compare_fig11
# cutoff_belief = -theta_compare
# belief_f_distance_young = belief_f_young_compare - cutoff_belief
# belief_f_distance_old = belief_f_old_compare - cutoff_belief
# belief_popu_distance_young = belief_popu_young_compare_fig11 - cutoff_belief
# belief_popu_distance_old = belief_popu_old_compare_fig11 - cutoff_belief
# parti_gap = P_old_compare - P_young_compare
# Phi_gap = Phi_old_compare - Phi_young_compare
# wealth_gap = Wealthshare_old_compare - Wealthshare_young_compare
# # y_variables = [parti_gap[:, popu_index, :], belief_f_distance_young[:, popu_index, :], belief_f_distance_old[:, popu_index, :]
# # y_complete_variables = [theta_complete, Phi_complete, Delta_bar_complete]
# # x_mat = belief_f_gap_compare[:, popu_index, :]
# # x_varname = r'Wealth weighted $\Delta_{s,t}$, old minus young'
# x_mat = belief_popu_gap_compare[:, popu_index, :]
# x_varname = r'Average estimation error $\Delta_{s,t}$, old minus young'
# x_range = 0.25
# x_range_left = np.percentile(x_mat, 5)
# x_range_right = np.percentile(x_mat, 95)
# width_bins = (x_range_right - x_range_left) / n_bins
# a = x_range_left <= x_mat
# b = x_mat <= x_range_right
# where_within = np.where(a * b == 1)  # winsorize
# x_mat_within = x_mat[where_within]
# total_count = np.shape(where_within)[1]
# condition_var = wealth_gap
# condition_label = r'Wealth share, old minus young'
# condition_var_within = condition_var[where_within]
# condition = np.percentile(condition_var_within, np.arange(0, 101, (100 / n_tiles)))
# y = np.empty((n_tiles, n_bins, 3))
# x = np.linspace(x_range_left + width_bins / 2, x_range_right - width_bins / 2, n_bins)
# X_ = np.linspace(x_range_left, x_range_right, 50)
# y_mat_within = parti_gap[where_within]
# y_varname = 'Participation rate, old minus young'
# fig, ax = plt.subplots(figsize=(10, 8))
# for i in range(n_tiles):
#     below = condition[i]
#     above = condition[i + 1]
#     a = below < condition_var_within
#     b = condition_var_within < above
#     data_where = np.where(a * b == 1)
#     x_var = x_mat_within[data_where]
#     y_var = y_mat_within[data_where]
#     for n in range(n_bins):
#         bin_left = x_range_left + n * width_bins
#         bin_right = bin_left + width_bins
#         bin_1 = x_var <= bin_right
#         bin_2 = x_var >= bin_left
#         bin_where = np.where(bin_1 * bin_2 == 1)
#         y[i, n, 0] = np.median(y_var[bin_where])
#         y[i, n, 1] = np.percentile(y_var[bin_where], 25)
#         y[i, n, 2] = np.percentile(y_var[bin_where], 75)
#     Y_ = np.empty((3, 50))
#     for m in range(3):
#         y_i = y[i, :, m]
#         X_Y_Spline = make_interp_spline(x, y_i)
#         Y_[m] = X_Y_Spline(X_)
#     # ax.plot(x, y[i, :, 0], linewidth=0.6, color=colors[i], label=condition_label + str(i + 1))
#     # ax.fill_between(x, y[i, :, 1], y[i, :, 2], color=colors[i], linewidth=0., alpha=0.3)
#     ax.plot(X_, Y_[0], linewidth=0.6, color=colors[i], label=condition_label + ', Quartile ' + str(i + 1))
#     ax.fill_between(X_, Y_[1], Y_[2], color=colors[i], linewidth=0., alpha=0.3)
# ax.axvline(0, 0.05, 0.95, linestyle='dashed', color='gray', linewidth=0.8)
# ax.axhline(0, 0.05, 0.95, linestyle='dashed', color='gray', linewidth=0.8)
# ax.legend(loc='upper left')
# ax.set_xlabel(x_varname)
# ax.set_ylabel(y_varname)
# fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
# scenario_label = 'Complete' if scenario_index == 0 else scenarios_short[scenario_index][1]
# plt.savefig(scenario_label + str(tax_index) + 'belief two sorts.png', dpi=100)
# plt.savefig(scenario_label + str(tax_index) + 'belief two sorts HD.png', dpi=200)
# # plt.savefig('85-115old'+str(tax)+'Intuition'+ x_varname[:4] + 'belief two sorts.png', dpi=200)
# plt.show()
# # plt.close()
#
#
# # distribution of wealth gap given the belief gap in [-x_range, x_range]:
# n_bins = 30
# fig, axes = plt.subplots(ncols=2, sharey='all', figsize=(10, 4))
# for j, ax in enumerate(axes):
#     if j == 0:
#         condition_var_density = np.empty(n_bins)
#         min_condition = np.min(condition_var_within)
#         max_condition = np.max(condition_var_within)
#         width_bins = (max_condition - min_condition) / n_bins
#         condition_var_x = np.linspace(min_condition + width_bins / 2, max_condition - width_bins / 2, n_bins)
#         for i in range(n_bins):
#             bin_left = min_condition + i * width_bins
#             bin_right = bin_left + width_bins
#             bin_1 = condition_var_within <= bin_right
#             bin_2 = condition_var_within >= bin_left
#             bin_where = np.where(bin_1 * bin_2 == 1)
#             condition_var_density[i] = np.shape(bin_where)[1] / total_count
#         X_Y_Spline = make_interp_spline(condition_var_x, condition_var_density)
#         X_ = np.linspace(min_condition, max_condition, 1000)
#         Y_ = X_Y_Spline(X_)
#         for i in range(n_tiles):
#             if i > 0:
#                 ax.axvline(condition[i], 0.05, 0.95, linestyle='dashed', linewidth=0.8, color='gray')
#             left_x = min_condition if i == 0 else condition[i]
#             right_x = max_condition if i == n_tiles - 1 else condition[i + 1]
#             a = X_ >= left_x
#             b = right_x >= X_
#             bin_where = np.where(a * b == 1)
#             x = X_[bin_where]
#             y = Y_[bin_where]
#             ax.fill_between(x, 0, y, color=colors[i], linewidth=0., alpha=0.3, label='Quartile ' + str(i + 1))
#         ax.legend(loc='upper left')
#         ax.set_xlim(0, 0.45) if tax > 0.01 else ax.set_xlim(0, max_condition)
#         ax.set_xlabel(condition_label)
#         ax.set_ylabel('Density')
#     else:
#         width_bins = (x_range_right - x_range_left) / n_bins
#         y = np.empty((n_tiles, n_bins))
#         x = np.linspace(x_range_left + width_bins / 2, x_range_right - width_bins / 2, n_bins)
#         y_bottom = 0
#         for i in range(n_tiles):
#             below = condition[i]
#             above = condition[i + 1]
#             a = below < condition_var_within
#             b = condition_var_within < above
#             data_where = np.where(a * b == 1)
#             x_var = x_mat_within[data_where]
#             # min_gap = np.min(x_var)
#             # max_gap = np.max(x_var)
#             # width_bins = (max_gap - min_gap) / n_bins
#             for n in range(n_bins):
#                 # bin_left = min_gap + n * width_bins
#                 # bin_right = bin_left + width_bins
#                 bin_left = x_range_left + n * width_bins
#                 bin_right = bin_left + width_bins
#                 bin_1 = x_var <= bin_right
#                 bin_2 = x_var >= bin_left
#                 bin_where = np.where(bin_1 * bin_2 == 1)
#                 y[i, n] = np.shape(bin_where)[1] / total_count
#             X_ = np.linspace(x_range_left, x_range_right, 100)
#             X_Y_Spline = make_interp_spline(x, y[i])
#             Y_ = X_Y_Spline(X_)
#             y_top = y_bottom + Y_
#             ax.fill_between(X_, y_top, y_bottom, linewidth=0., color=colors[i],
#                             alpha=0.3)
#             y_bottom = y_top
#         # ax.legend(loc='upper right')
#         ax.axvline(0, 0.05, 0.95, linestyle='dashed', linewidth=0.8, color='gray')
#         # ax.set_ylim(top = 0.08)
#         ax.set_xlabel(x_varname)
#         # ax.set_ylabel('Density')
# fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
# scenario_label = 'Complete' if scenario_index == 0 else scenarios_short[scenario_index][1]
# plt.savefig(scenario_label + str(tax_index) + 'Intuition wealth distribution.png', dpi=100)
# # plt.savefig('85-115old'+str(tax)+'Intuition wealth distribution.png', dpi=200)
# plt.show()
# # plt.close()
#
#
# # figure 15
# ######################################################################
# #### why is Delta_bar more volatile with the shorting constraint? ####
# ######################################################################
# # scenarios: complete, reentry, & disappointment
# # phi = 0
# # Npres: 60 & 240
# Npres_short = [60, 240]
# n_Npres_short = len(Npres_short)
# popu_fig15 = 0.5
# n_scenarios_short = 3
# phi_try = 0.0
# scenarios_short = scenarios[:n_scenarios_short]
# cutoff_age_old_below_fig15 = np.searchsorted(cummu_popu, popu_fig15)
# cutoff_age_young_fig15 = np.searchsorted(cummu_popu, 1 - popu_fig15)
# Phi_compare = np.empty((Mpath, n_scenarios_short, n_Npres_short, Nt), dtype=np.float32)
# Delta_bar_compare = np.empty((Mpath, n_scenarios_short, n_Npres_short, Nt), dtype=np.float32)
# belief_popu_fig15 = np.empty((Mpath, n_Npres_short, Nt), dtype=np.float32)
# Phi_old_compare = np.empty((Mpath, n_scenarios_short, n_Npres_short, Nt), dtype=np.float32)
# Phi_young_compare = np.empty((Mpath, n_scenarios_short, n_Npres_short, Nt), dtype=np.float32)
# belief_f_old_compare = np.empty((Mpath, n_scenarios_short, n_Npres_short, Nt), dtype=np.float32)  # of participants
# belief_f_young_compare = np.empty((Mpath, n_scenarios_short, n_Npres_short, Nt), dtype=np.float32)  # of participants
# for i in range(Mpath):
#     # for i in range(Mpath):
#     print(i)
#     ii = i if i < 1000 else i + 4000
#     dZ = dZ_matrix[ii]
#     dZ_build = dZ_build_matrix[ii]
#     dZ_SI = dZ_SI_matrix[ii]
#     dZ_SI_build = dZ_SI_build_matrix[ii]
#     for j, scenario in enumerate(scenarios_short):
#         mode_trade = scenario[0]
#         mode_learn = scenario[1]
#         for m, Npre_try in enumerate(Npres_short):
#             T_hat_try = Npre_try * dt
#             Vhat_try = (sigma_Y ** 2) / T_hat_try  # prior variance
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
#                 popu_can_short,
#                 popu_short,
#                 Phi_can_short,
#                 Phi_short,
#             ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu,
#                             Vhat_try,
#                             mu_Y, sigma_Y, sigma_S,
#                             tax,
#                             beta,
#                             phi_try,
#                             Npre_try, Ninit,
#                             T_hat_try,
#                             dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
#                             need_f='True',
#                             need_Delta='True',
#                             need_pi='True',
#                             )
#             # save results for fig 15
#             Phi_compare[i, j, m] = f_parti
#             Delta_bar_compare[i, j, m] = Delta_bar_parti
#             if j == 0:
#                 belief_popu_fig15[i, m] = np.average(Delta, weights=cohort_size,
#                                                      axis=1)  # same average belief as phi == 0
#                 belief_f_old_compare[i, j, m] = np.average(Delta[:, :cutoff_age_old_below_fig15],
#                                                            weights=f[:, :cutoff_age_old_below_fig15] * dt,
#                                                            axis=1)
#                 belief_f_young_compare[i, j, m] = np.average(Delta[:, cutoff_age_young_fig15:],
#                                                              weights=f[:, cutoff_age_young_fig15:] * dt,
#                                                              axis=1)
#                 Phi_old_compare[i, j, m] = np.sum(f[:, :cutoff_age_old_below_fig15] * dt, axis=1)
#                 Phi_young_compare[i, j, m] = np.sum(f[:, cutoff_age_young_fig15:] * dt, axis=1)
#             else:
#                 parti = pi > 0
#                 belief_f_old_compare[i, j, m] = np.ma.average(Delta[:, :cutoff_age_old_below_fig15],
#                                                               weights=f[:, : cutoff_age_old_below_fig15] *
#                                                                       parti[:, : cutoff_age_old_below_fig15] * dt,
#                                                               axis=1)
#                 belief_f_young_compare[i, j, m] = np.ma.average(Delta[:, cutoff_age_young_fig15:],
#                                                                 weights=f[:, cutoff_age_young_fig15:] *
#                                                                         parti[:, cutoff_age_young_fig15:] * dt,
#                                                                 axis=1)
#                 Phi_old_compare[i, j, m] = np.sum(parti[:, :cutoff_age_old_below_fig15]
#                                                   * f[:, :cutoff_age_old_below_fig15] * dt,
#                                                   axis=1)
#                 Phi_young_compare[i, j, m] = np.sum(parti[:, cutoff_age_young_fig15:]
#                                                     * f[:, cutoff_age_young_fig15:] * dt,
#                                                     axis=1)
#
# # winsorize extreme shocks
# Npre_index = 0
# average_Delta_bar = np.mean(np.mean(Delta_bar_compare[:, :, Npre_index], axis=0), axis=1)
# x_index = belief_popu_fig15[:, Npre_index]
# x_label = 'Average estimation error'
# below_dz = np.percentile(x_index, 10)
# below_data = x_index >= below_dz
# above_dz = np.percentile(x_index, 90)
# above_data = above_dz >= x_index
# data_where = np.where(below_data * above_data == 1)
#
# # prepare data for the figure:
# n_bins = 10
# bins = np.linspace(below_dz, above_dz, n_bins)
# # quartile_var = (Wealthshare_old_compare - Wealthshare_young_compare)[:, :, Npre_index]
# # quartile_var = belief_popu_young_compare[:, Npre_index] - belief_popu_old_compare[:, Npre_index]
# # quartile_var = parti_young_compare[:, :, Npre_index]
# y_percentiles = [50, 25, 75]
# phi_old = Phi_old_compare / Phi_compare
# phi_young = Phi_young_compare / Phi_compare
# belief_f_young = np.where(belief_f_young_compare == 0, np.nan,
#                           belief_f_young_compare)  # converting empty cells from 0 to nan
# belief_f_old = np.where(belief_f_old_compare == 0, np.nan, belief_f_old_compare)
# data_var = Delta_bar_compare[:, :, Npre_index]
# data_figure = np.zeros((n_scenarios_short, n_bins - 1, len(y_percentiles)))
# phi_figure = np.zeros((n_scenarios_short, 2, n_bins - 1, len(y_percentiles)))
# belief_figure = np.zeros((n_scenarios_short, 2, n_bins - 1, len(y_percentiles)))
# belief_phi_figure = np.zeros((n_scenarios_short, 2, n_bins - 1, len(y_percentiles)))
# for i in range(n_scenarios_short):
#     data_focus = data_var[:, i]
#     phi_var_old = phi_old
#     phi_var_young = phi_young[:, Npre_index]
#     for j in range(n_bins - 1):
#         bin_0 = bins[j]
#         bin_1 = bins[j + 1]
#         below_bin = bin_1 >= x_index
#         above_bin = x_index >= bin_0
#         bin_where = np.where(below_bin * above_bin == 1)
#         data_focus_z = data_focus[bin_where]
#         data_figure[i, j] = np.percentile(data_focus_z, y_percentiles)
#         for l in range(2):
#             phi_var = phi_old[:, i, Npre_index] if l == 0 else phi_young[:, i, Npre_index]
#             belief_var_nan = belief_f_old[:, i, Npre_index] if l == 0 else belief_f_young[:, i, Npre_index]
#             belief_var = belief_f_old_compare[:, i, Npre_index] if l == 0 else belief_f_young_compare[:, i, Npre_index]
#             phi_figure[i, l, j] = np.percentile(phi_var[bin_where], y_percentiles)
#             belief_figure[i, l, j] = np.nanpercentile(belief_var_nan[bin_where], y_percentiles)
#             belief_phi = phi_var * belief_var
#             belief_phi_figure[i, l, j] = np.percentile(belief_phi[bin_where], y_percentiles)
#
# bin_size = (above_dz - below_dz) / (n_bins - 1)
# x = np.linspace(below_dz + bin_size / 2, above_dz - bin_size / 2, n_bins - 1)
# labels = [[r'$\bar{\Delta}_t^{old}$', r'$\bar{\Delta}_t^{young}$'],
#           [r'$\Phi_t^{old} / (\Phi_t^{old} + \Phi_t^{young})$', r'$\Phi_t^{young} / (\Phi_t^{old} + \Phi_t^{young})$'],
#           [r'$\Phi_t^{old}\bar{\Delta}_t^{old} / (\Phi_t^{old} + \Phi_t^{young})$',
#            r'$\Phi_t^{young}\bar{\Delta}_t^{young} / (\Phi_t^{old} + \Phi_t^{young})$']]
# # labels = [r'Wealth old minus young, ' + 'Lowest quartile', 'Second quartile', 'Third quartile', 'Highest quartile']
# sub_titles = ['Complete market', 'Reentry', 'Disappointment']
# y_labels = ['Estimation error of the participants',
#             'Wealth share of the participants',
#             r'Contribution to $\bar{\Delta}_t$']
# # X_ = np.linspace(-0.2, 0.2, 100)
# X_ = np.linspace(below_dz, above_dz, 100)
# fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(15, 15), sharex='all', sharey='row')
# for j, rows in enumerate(axes):
#     for i, ax in enumerate(rows):
#         if j == 0:
#             y_focus = belief_figure[i]
#         elif j == 1:
#             y_focus = phi_figure[i]
#         else:
#             y_focus = belief_phi_figure[i]
#         for k in range(2):
#             if j == 2:
#                 y_i = y_focus[k]
#                 X_Y_Spline = make_interp_spline(x, y_i)
#                 Y_ = X_Y_Spline(X_)
#                 ax.plot(X_, Y_[:, 0], color=colors_short[k], linewidth=1, label=labels[j][k])
#                 ax.fill_between(X_, Y_[:, 2], Y_[:, 1], color=colors_short[k], linewidth=0., alpha=0.4)
#                 if k == 1:
#                     y_i = data_figure[i]  # n_bin-1 * 3
#                     X_Y_Spline = make_interp_spline(x, y_i)
#                     Y_ = X_Y_Spline(X_)
#                     ax.plot(X_, Y_[:, 0], color='gray', linewidth=1,
#                             label=r'$\bar{\Delta}_t$')
#                     ax.fill_between(X_, Y_[:, 2], Y_[:, 1], color='gray', linewidth=0., alpha=0.2)
#                     ax.axhline(average_Delta_bar[i], 0.05, 0.95, color='saddlebrown', linewidth=0.8, linestyle='dashed',
#                                label=r'Unconditional mean $\bar{\Delta}_t$')
#                     x_mean = [X_[np.searchsorted(Y_[:, 0], average_Delta_bar[i])]]
#                     ax.scatter(x_mean, [average_Delta_bar[i]], marker='o', color='saddlebrown')
#             else:
#                 y_i = y_focus[k]
#                 X_Y_Spline = make_interp_spline(x, y_i)
#                 Y_ = X_Y_Spline(X_)
#                 ax.plot(X_, Y_[:, 0], color=colors_short[k], linewidth=1, label=labels[j][k])
#                 ax.fill_between(X_, Y_[:, 2], Y_[:, 1], color=colors_short[k], linewidth=0., alpha=0.4)
#         if i == 0:
#             ax.legend(loc='upper left')
#             ax.set_ylabel(y_labels[j])
#         if j == 0:
#             ax.set_title(sub_titles[i])
#         if j == 2:
#             ax.set_xlabel(x_label)
#         else:
#             ax.axvline(0, 0.05, 0.95, color='gray', linewidth=0.8, linestyle='dashed')
#         # ax.set_xlim(-0.25, 0.25)
# fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
# plt.savefig(str(Npres_short[Npre_index]) + 'DeltaVola.png', dpi=100)
# plt.savefig(str(Npres_short[Npre_index]) + 'DeltaVola HD.png', dpi=200)
# plt.show()
#
# # ######################################
# # ############ Figure 3.2 ##############
# # ######################################
#
#
# # ######################################
# # ############ GRAPH TWO ###############
# # ######################################
# # regressions
#
# horizons = [3, 6, 12, 24]
# n_horizon = len(horizons)
# report = ['coef', 't-stats', 'R-sqrd']
# n_report = len(report)
# # var_names = ['Participation rate', 'Belief dispersion', 'survey view', 'Participation rate, young', 'Participation rate, old']
# var_names = ['Participation rate', 'Belief dispersion', 'survey view']
# phi_indeces = [0, 4, 8]
# n_phi_short = len(phi_indeces)
# regression_results_uni = np.empty((N, n_scenarios, n_phi_short, n_horizon, len(var_names), n_report))
# # var_list = [popu_parti_matrix, belief_dispersion_matrix, survey_view_matrix, parti_young_matrix, parti_old_matrix]
# var_list = [popu_parti_matrix, belief_dispersion_matrix, survey_view_matrix]
#
# # predictive regression of stock returns on pariticipation rate
# for i in range(N):
#     for j in range(n_scenarios):
#         for k, phi_index in enumerate(phi_indeces):
#             excess_return_vector = np.cumsum(dR_matrix[i, j, phi_index, 1:] - r_matrix[i, j, phi_index, :-1] * dt)
#             x_list = []
#             for var in var_list:
#                 var_vector = var[i, j, phi_index, :-1]
#                 x_list.append(var_vector)
#             for l, horizon in enumerate(horizons):
#                 y_horizon1 = (excess_return_vector[horizon:] - excess_return_vector[: -horizon]) / (
#                         dt * horizon)  # make sure the timings allign
#                 y_horizon = y_horizon1.reshape(-1, 1)
#                 for m, x_horizon_raw in enumerate(x_list):
#                     if scenarios[j][0] == 'complete' and var == popu_parti_matrix:
#                         regression_results_uni[i, j, k, l, m, 0] = est.params[1]
#                         regression_results_uni[i, j, k, l, m, 1] = est.tvalues[1]
#                         regression_results_uni[i, j, k, l, m, 2] = est.rsquared
#                     x_horizon1 = x_horizon_raw[: -horizon]
#                     x_horizon1 = x_horizon1 / np.std(x_horizon1)
#                     x_horizon1 = x_horizon1.reshape(-1, 1)
#                     x_horizon = sm.add_constant(x_horizon1)
#
#                     model = sm.OLS(y_horizon, x_horizon)
#                     est = model.fit()
#                     regression_results_uni[i, j, k, l, m, 0] = est.params[1]
#                     regression_results_uni[i, j, k, l, m, 1] = est.tvalues[1]
#                     regression_results_uni[i, j, k, l, m, 2] = est.rsquared
#
# mean_regression_results = np.mean(regression_results_uni, axis=0)
# header = ['(1) phi = 0', '(2) phi = 0.4', '(3) phi = 0.8']
# # present the regression results in tables:
#
# for k, scenario in enumerate(scenarios):
#     label_scenario = scenario[0] if scenario[0] == 'complete' else scenario[0] + scenario[1]
#     print(label_scenario)
#     for j, var in enumerate(var_names):
#         for i, horizon in enumerate(horizons):
#             reg_data = np.empty((n_report, n_phi))
#             for l in range(n_phi):
#                 reg_data[:, l] = mean_regression_results[k, l, i, j]
#             report1 = [var, 't-stats', 'R-sqrd']
#             print(var, ', ' + str(horizon) + ' months')
#             print(tabulate.tabulate(reg_data, headers=header, showindex=report1, floatfmt=".4f", tablefmt='fancy_grid'))
#
# # #######################################
# # ############ GRAPH  FIVE ##############
# # #######################################
# # # The rich can short
# # # compare and plot theta
# # fig, ax1 = plt.subplots(figsize=(15, 5))
# # ax1.set_xlabel('Time in simulation, one random path')
# # ax1.set_ylabel('Zt', color=color5)
# # ax1.plot(t, y0, color=color5, linewidth=0.5)
# # ax1.tick_params(axis='y', labelcolor=color5)
# # ax2 = ax1.twinx()
# # ax2.set_ylabel('Market price of risk', color=color2)
# # ax2.set_ylim([-1, 1])
# # ax2.plot(t, y21, color=color2, linewidth=0.4, label='Complete market')
# # ax2.plot(t, y22, color=color3, linewidth=0.4, label='Short-sale constraint')
# # ax2.plot(t, y71, color='magenta', linewidth=0.4, label='Rich can short')
# # # ax2.hlines(sigma_Y, xmin=0, xmax=500, color='purple', linestyles='--', linewidth=0.8, label='Representative agent')
# # ax2.tick_params(axis='y', labelcolor=color2)
# # plt.legend()
# # fig.tight_layout()  # otherwise the right y-label is slightly clipped
# # plt.savefig('Zt and market price of risk, rich free' + '.png', dpi=500)
# # plt.show()
# #
# # # Compare the bias between investors having long and short positions
# # fig, ax1 = plt.subplots(figsize=(15, 5))
# # ax1.set_xlabel('Time in simulation, one random path')
# # ax1.set_ylabel('Zt', color=color5)
# # ax1.plot(t, y0, color=color5, linewidth=0.5)
# # ax1.tick_params(axis='y', labelcolor=color5)
# # ax2 = ax1.twinx()
# # ax2.set_ylabel('Market price of risk', color=color2)
# # ax2.set_ylim([-0.5, 1])
# # ax2.plot(t, y72, color='darkblue', linewidth=0.6, label='% investors')
# # ax2.plot(t, y73, color='darkgreen', linewidth=0.6, label='% Short sellers')
# # ax2.plot(t, y74, color='blue', linewidth=0.4, label='Average bias long')
# # ax2.plot(t, y75, color='magenta', linewidth=0.4, label='Average bias short')
# # # ax2.hlines(sigma_Y, xmin=0, xmax=500, color='purple', linestyles='--', linewidth=0.8, label='Representative agent')
# # ax2.tick_params(axis='y', labelcolor=color2)
# # plt.legend()
# # fig.tight_layout()  # otherwise the right y-label is slightly clipped
# # plt.savefig('Zt and participation, rich free' + '.png', dpi=500)
# # plt.show()
# #
# # #######################################
# # ############ GRAPH  SIX ###############
# # #######################################
# # #
# # # test if subjective risk premia comove less with shocks / cyclicality of perceived risk premia
# # # sensitivity of subjective vs. objective risk premia to business cycle indicator (dY/Y)
# # # x: output growth from time t-T to time t
# # # y: subjective and objective risk premia at tme t
# # horizons = [1, 3, 6, 12, 24]
# # m = len(horizons)
# # results_obj_matrix = np.empty((Mpaths, m, 3))
# # results_sub_matrix = np.empty((Mpaths, m, 3))
# # header = []
# # for i in range(Mpaths):
# #     x_path = np.cumsum(dY_Y_matrix[i])
# #
# #     for j, horizon in enumerate(horizons):
# #         if i == 0:
# #             header_j = str(horizon) + '-month'
# #             header.append(header_j)
# #         x = (x_path[horizon:-horizon] - x_path[:-horizon * 2]) / (horizon * dt)
# #         x = x / np.std(x)
# #         x = x.reshape(-1, 1)
# #         x = sm.add_constant(x)
# #
# #         y_sub_path = survey_view_parti_matrix[i, horizon:-horizon]
# #         y_sub = y_sub_path.reshape(-1, 1)
# #
# #         y_obj_path = obj_rp_matrix[i, horizon:-horizon]
# #         y_obj = y_obj_path.reshape(-1, 1)
# #
# #         # Objective risk premia:
# #         model = sm.OLS(y_obj, x)
# #
# #         est = model.fit()
# #         results_obj_matrix[i, j, 0] = est.params[1]
# #         results_obj_matrix[i, j, 1] = est.tvalues[1]
# #         results_obj_matrix[i, j, 2] = est.rsquared
# #
# #         # Subjective risk premia:
# #         model = sm.OLS(y_sub, x)
# #         est = model.fit()
# #         results_sub_matrix[i, j, 0] = est.params[1]
# #         results_sub_matrix[i, j, 1] = est.tvalues[1]
# #         results_sub_matrix[i, j, 2] = est.rsquared
# #
# # result_obj = np.mean(results_obj_matrix, axis=0)
# # result_sub = np.mean(results_sub_matrix, axis=0)
# #
# # # to table:
# # index = ['coef', 't-stats', 'R2']
# # n = len(index)
# # for i in range(2):
# #     reg_data = np.empty((n, m))
# #     var = result_obj if i == 0 else result_sub
# #     for j in range(n):
# #         reg_data[j] = var[:,j]
# #     print('result_obj' if i == 0 else 'result_sub')
# #     print(tabulate.tabulate(reg_data, headers=header, showindex=index, floatfmt=".4f", tablefmt='fancy_grid'))
# #
# #
#
#
# #######################################
# ########### GRAPH  SEVEN ##############
# #######################################
#
# # regressing difference in participarion on difference in experienced stock market returns
# # over all simulated paths, and take average coefficient across all the paths
#
# # young_age_cut = Nt - 20 / dt
# # old_age_cut = Nt - 40 / dt
# # young_prior = 20 / dt
# # old_prior = 50 / dt
# # diff_exprienced_growth = np.zeros((Mpaths, Nt - old_prior))
# # diff_participation_rate = np.zeros((Mpaths, Nt - old_prior))
# # for i in range(Nt - old_prior):
# #     old_experienced_growth = np.average(dZ_matrix[:, i : i + old_prior])
# #     young_experienced_growth = np.average(dZ_matrix[:, (i + old_prior - young_prior) : i + old_prior])
# #     diff_exprienced_growth[:, i] = old_experienced_growth - young_experienced_growth
# #
# #     old_participation_rate = np.sum(y5[:, :old_age_cut], axis=1)  # not y5, but participation rate in matrix
# #     young_participation_rate = np.sum(y5[:, young_age_cut:], axis=1)
# #     diff_participation_rate[:, i] = old_participation_rate - young_participation_rate
# #
# # a = np.zeros(Mpaths)
# #
# # for j in range(Mpaths):
# #     model = LinearRegression().fit(diff_exprienced_growth[j], diff_participation_rate[j])
# #     a[j] = model.coef_
#
# #######################################
# ########### GRAPH  EIGHT ##############
# #######################################
#
# # describe the mean and variance of beliefs (wealth-weighted) against belief of the marginal investor
# # specific to one path
# # relates to the information index. right now beliefs of non-participants make little sense
#
# # marginal_belief = (-theta_disappointment) * sigma_Y + mu_Y
#
#
# #######################################
# ############ GRAPH  NINE ##############
# #######################################
#
# # # describe the predictive power of participation rate
# #
# #
# # start_t = 0
# # x_variables = [popu_parti_matrix, popu_parti_young_matrix, popu_parti_old_matrix, age_parti_matrix]
# # m = len(x_variables)
# # # np.cumsum(dR_matrix[i])
# # coeff_matrix1 = np.zeros((Mpaths, m, 3))
# # pvalue_matrix1 = np.zeros((Mpaths, m, 3))
# # tstats_matrix1 = np.zeros((Mpaths, m, 3))
# # rsqrd_matrix1 = np.zeros((Mpaths, m, 2))
# #
# # for i in range(Mpaths):
# #     path_y = erp_S_matrix[i]
# #     path_x2 = survey_view_parti_matrix[i]
# #     path_x2 = path_x2 / np.std(path_x2)
# #     for j, var in enumerate(x_variables):
# #         path_x = var[i]
# #         y_raw = path_y[start_t:]  # equity risk premium at time t
# #
# #         # univariate regressions
# #         x1_raw = path_x[start_t:]  # participation rate at time t
# #         x1_raw = x1_raw / np.std(x1_raw)
# #         x1_lag = x1_raw.reshape(-1, 1)
# #         x1_lag2 = sm.add_constant(x1_lag)
# #         y_predict = y_raw.reshape(-1, 1)
# #
# #         model = sm.OLS(y_predict, x1_lag2)
# #         est = model.fit()
# #         coeff_matrix1[i, j, 0] = est.params[1]
# #         pvalue_matrix1[i, j, 0] = est.pvalues[1]
# #         tstats_matrix1[i, j, 0] = est.tvalues[1]
# #         rsqrd_matrix1[i, j, 0] = est.rsquared
# #
# #         # bivariate regressions
# #         x2_lag = path_x2[start_t:]
# #
# #         x_lag = np.append(x1_lag, x2_lag)
# #         x_lag = np.transpose(x_lag.reshape(2, -1))
# #         x_lag = sm.add_constant(x_lag)
# #
# #         y_predict = y_raw.reshape(-1, 1)
# #
# #         model = sm.OLS(y_predict, x_lag)
# #         est = model.fit()
# #         coeff_matrix1[i, j, 1] = est.params[1]
# #         coeff_matrix1[i, j, 2] = est.params[2]
# #         pvalue_matrix1[i, j, 1] = est.pvalues[1]
# #         pvalue_matrix1[i, j, 2] = est.pvalues[2]
# #         tstats_matrix1[i, j, 1] = est.tvalues[1]
# #         tstats_matrix1[i, j, 2] = est.tvalues[2]
# #         rsqrd_matrix1[i, j, 1] = est.rsquared
# #
# # reg_coeffs1 = np.average(coeff_matrix1, axis=0)
# # reg_pvalues1 = np.average(pvalue_matrix1, axis=0)
# # reg_tstats1 = np.average(tstats_matrix1, axis=0)
# # reg_rsqrd1 = np.average(rsqrd_matrix1, axis=0)
# # reg_data = np.empty((5, 8))
# # header = ['(1) parti rate', '(2)', '(3) parti rate, young', '(4)', '(5) parti rate, old', '(6)', '(7) parti age', '(8)']
# # index = ['coef', 't-stats', 'coef_x2', 't-stats_x2', 'R2']
# # for i in range(4):
# #     reg_data[0, i * 2] = reg_coeffs1[i, 0]
# #     reg_data[1, i * 2] = reg_tstats1[i, 0]
# #     reg_data[2, i * 2] = np.nan
# #     reg_data[3, i * 2] = np.nan
# #     reg_data[4, i * 2] = reg_rsqrd1[i, 0]
# #
# #     reg_data[0, i * 2 + 1] = reg_coeffs1[i, 1]
# #     reg_data[1, i * 2 + 1] = reg_tstats1[i, 1]
# #     reg_data[2, i * 2 + 1] = reg_coeffs1[i, 2]
# #     reg_data[3, i * 2 + 1] = reg_tstats1[i, 2]
# #     reg_data[4, i * 2 + 1] = reg_rsqrd1[i, 1]
# #
# # print(tabulate.tabulate(reg_data, headers=header, showindex=index, floatfmt=".4f", tablefmt='fancy_grid'))
# #
# # ####
# # horizons = [1, 3, 6, 12, 36, 60, 120]
# # x_variables = [popu_parti_matrix, popu_parti_young_matrix, popu_parti_old_matrix, age_parti_matrix]
# # # x_variables = [popu_parti_matrix, popu_parti_young_matrix, popu_parti_old_matrix, age_parti_matrix]
# # m = len(x_variables)
# # n = len(horizons)
# # # np.cumsum(dR_matrix[i])
# # coeff_matrix2 = np.zeros((Mpath, n, m, 3))
# # pvalue_matrix2 = np.zeros((Mpath, n, m, 3))
# # tstats_matrix2 = np.zeros((Mpath, n, m, 3))
# # rsqrd_matrix2 = np.zeros((Mpath, n, m, 2))
# #
# # for i in range(Mpath):
# #     path_y = np.cumsum(dR_matrix[i])
# #     path_r = np.cumsum(r_matrix[i])
# #     path_x2 = survey_view_parti_matrix[i]
# #     path_x2 = path_x2 / np.std(path_x2)
# #     for j, horizon in enumerate(horizons):
# #         y_raw = (path_y[start_t + horizon + 1:] - path_y[start_t + 1: -horizon]) / (horizon * dt)
# #         # dR is the return from t-1 to t, and thus have to move 1 to have returns from t to t+1
# #         y_predict = y_raw.reshape(-1, 1)
# #         for k, var in enumerate(x_variables):
# #             path_x = var[i]
# #             # univariate regressions
# #             x1_raw = path_x[start_t: -1 - horizon]  # participation rate at time t
# #             x1_raw = x1_raw / np.std(x1_raw)
# #             x1_lag = x1_raw.reshape(-1, 1)
# #             x1_lag2 = sm.add_constant(x1_lag)
# #
# #             model = sm.OLS(y_predict, x1_lag2)
# #             est = model.fit()
# #             coeff_matrix2[i, j, k, 0] = est.params[1]
# #             pvalue_matrix2[i, j, k, 0] = est.pvalues[1]
# #             tstats_matrix2[i, j, k, 0] = est.tvalues[1]
# #             rsqrd_matrix2[i, j, k, 0] = est.rsquared
# #
# #             # bivariate regressions
# #             x2_lag = path_x2[start_t: -1 - horizon]  # average perceived risk premia at time t
# #
# #             x_lag = np.append(x1_lag, x2_lag)
# #             x_lag = np.transpose(x_lag.reshape(2, -1))
# #             x_lag = sm.add_constant(x_lag)
# #
# #             y_predict = y_raw.reshape(-1, 1)
# #
# #             model = sm.OLS(y_predict, x_lag)
# #             est = model.fit()
# #             coeff_matrix2[i, j, k, 1] = est.params[1]
# #             coeff_matrix2[i, j, k, 2] = est.params[2]
# #             pvalue_matrix2[i, j, k, 1] = est.pvalues[1]
# #             pvalue_matrix2[i, j, k, 2] = est.pvalues[2]
# #             tstats_matrix2[i, j, k, 1] = est.tvalues[1]
# #             tstats_matrix2[i, j, k, 2] = est.tvalues[2]
# #             rsqrd_matrix2[i, j, k, 1] = est.rsquared
# #
# # reg_coeffs2 = np.average(coeff_matrix2, axis=0)
# # reg_pvalues2 = np.average(pvalue_matrix2, axis=0)
# # reg_tstats2 = np.average(tstats_matrix2, axis=0)
# # reg_rsqrd2 = np.average(rsqrd_matrix2, axis=0)
# #
# # for j in range(n):
# #     horizon = horizons[j]
# #     reg_data = np.empty((5, 8))
# #     header = ['(1) parti rate', '(2)', '(3) parti rate, young', '(4)', '(5) parti rate, old', '(6)', '(7) parti age',
# #               '(8)']
# #     index = ['coef', 't-stats', 'coef_x2', 't-stats_x2', 'R2']
# #     for i in range(4):
# #         reg_data[0, i * 2] = reg_coeffs2[j, i, 0]
# #         reg_data[1, i * 2] = reg_tstats2[j, i, 0]
# #         reg_data[2, i * 2] = np.nan
# #         reg_data[3, i * 2] = np.nan
# #         reg_data[4, i * 2] = reg_rsqrd2[j, i, 0]
# #
# #         reg_data[0, i * 2 + 1] = reg_coeffs2[j, i, 1]
# #         reg_data[1, i * 2 + 1] = reg_tstats2[j, i, 1]
# #         reg_data[2, i * 2 + 1] = reg_coeffs2[j, i, 2]
# #         reg_data[3, i * 2 + 1] = reg_tstats2[j, i, 2]
# #         reg_data[4, i * 2 + 1] = reg_rsqrd2[j, i, 1]
# #
# #     print(str(horizon) + '-month')
# #     print(tabulate.tabulate(reg_data, headers=header, showindex=index, floatfmt=".4f", tablefmt='fancy_grid'))
# #
# # ####
# # horizons = [1, 3, 6, 12, 36, 60, 120]
# # x_variables = [popu_parti_matrix, popu_parti_young_matrix, popu_parti_old_matrix, age_parti_matrix]
# # # x_variables = [popu_parti_matrix, popu_parti_young_matrix, popu_parti_old_matrix, age_parti_matrix]
# # m = len(x_variables)
# # n = len(horizons)
# # # np.cumsum(dR_matrix[i])
# # coeff_matrix2 = np.zeros((Mpaths, n, m, 3))
# # pvalue_matrix2 = np.zeros((Mpaths, n, m, 3))
# # tstats_matrix2 = np.zeros((Mpaths, n, m, 3))
# # rsqrd_matrix2 = np.zeros((Mpaths, n, m, 2))
# #
# # for i in range(Mpaths):
# #     path_y = np.cumsum(dR_matrix[i])
# #     path_r = np.cumsum(r_matrix[i])
# #     path_x2 = survey_view_parti_matrix[i]
# #     path_x2 = path_x2 / np.std(path_x2)
# #     for j, horizon in enumerate(horizons):
# #         # y_raw = path_y[start_t + horizon + 1:] - path_y[start_t + 1: -horizon]  #dR is the return from t-1 to t, and thus have to move 1 to have returns from t to t+1
# #         y_raw = ((path_y[start_t + horizon + 1:] - path_y[start_t + 1: -horizon]) \
# #                  - (path_r[start_t + horizon: -1] - path_r[start_t: -horizon - 1])) / (horizon * dt)
# #         y_predict = y_raw.reshape(-1, 1)
# #         for k, var in enumerate(x_variables):
# #             path_x = var[i]
# #             # univariate regressions
# #             x1_raw = path_x[start_t: -1 - horizon]  # participation rate at time t
# #             x1_raw = x1_raw / np.std(x1_raw)
# #             x1_lag = x1_raw.reshape(-1, 1)
# #             x1_lag2 = sm.add_constant(x1_lag)
# #
# #             model = sm.OLS(y_predict, x1_lag2)
# #             est = model.fit()
# #             coeff_matrix2[i, j, k, 0] = est.params[1]
# #             pvalue_matrix2[i, j, k, 0] = est.pvalues[1]
# #             tstats_matrix2[i, j, k, 0] = est.tvalues[1]
# #             rsqrd_matrix2[i, j, k, 0] = est.rsquared
# #
# #             # bivariate regressions
# #             x2_lag = path_x2[start_t: -horizon - 1]
# #
# #             x_lag = np.append(x1_lag, x2_lag)
# #             x_lag = np.transpose(x_lag.reshape(2, -1))
# #             x_lag = sm.add_constant(x_lag)
# #
# #             y_predict = y_raw.reshape(-1, 1)
# #
# #             model = sm.OLS(y_predict, x_lag)
# #             est = model.fit()
# #             coeff_matrix2[i, j, k, 1] = est.params[1]
# #             coeff_matrix2[i, j, k, 2] = est.params[2]
# #             pvalue_matrix2[i, j, k, 1] = est.pvalues[1]
# #             pvalue_matrix2[i, j, k, 2] = est.pvalues[2]
# #             tstats_matrix2[i, j, k, 1] = est.tvalues[1]
# #             tstats_matrix2[i, j, k, 2] = est.tvalues[2]
# #             rsqrd_matrix2[i, j, k, 1] = est.rsquared
# #
# # reg_coeffs2 = np.average(coeff_matrix2, axis=0)
# # reg_pvalues2 = np.average(pvalue_matrix2, axis=0)
# # reg_tstats2 = np.average(tstats_matrix2, axis=0)
# # reg_rsqrd2 = np.average(rsqrd_matrix2, axis=0)
# #
# # for j in range(n):
# #     horizon = horizons[j]
# #     reg_data = np.empty((5, 8))
# #     header = ['(1) parti rate', '(2)', '(3) parti rate, young', '(4)', '(5) parti rate, old', '(6)', '(7) parti age',
# #               '(8)']
# #     index = ['coef', 't-stats', 'coef_x2', 't-stats_x2', 'R2']
# #     for i in range(4):
# #         reg_data[0, i * 2] = reg_coeffs2[j, i, 0]
# #         reg_data[1, i * 2] = reg_tstats2[j, i, 0]
# #         reg_data[2, i * 2] = np.nan
# #         reg_data[3, i * 2] = np.nan
# #         reg_data[4, i * 2] = reg_rsqrd2[j, i, 0]
# #
# #         reg_data[0, i * 2 + 1] = reg_coeffs2[j, i, 1]
# #         reg_data[1, i * 2 + 1] = reg_tstats2[j, i, 1]
# #         reg_data[2, i * 2 + 1] = reg_coeffs2[j, i, 2]
# #         reg_data[3, i * 2 + 1] = reg_tstats2[j, i, 2]
# #         reg_data[4, i * 2 + 1] = reg_rsqrd2[j, i, 1]
# #
# #     print(str(horizon) + '-month')
# #     print(tabulate.tabulate(reg_data, headers=header, showindex=index, floatfmt=".4f", tablefmt='fancy_grid'))
