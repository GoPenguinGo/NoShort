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
    Z_Y_cases, Z_SI_cases, t, top, old_limit
from src.stats import shocks, tau_calculator, good_times, Delta_st_compare
from numba import jit
import matplotlib.pyplot as plt
import statsmodels.api as sm
# import tabulate as tabulate
from scipy.interpolate import make_interp_spline

# todo: to fill with patterns: https://matplotlib.org/stable/gallery/shapes_and_collections/hatch_demo.html
# todo: remove variables that are never used, to save space
# todo: organize the code
# different scenarios
N = 200  # for smaller number of paths

n_scenarios = 3
scenarios_short = scenarios[:n_scenarios]

# n_scenarios = 1
# scenarios_short = scenarios[1:2]

phi_vector = np.arange(0, 1, 0.1)
n_phi = len(phi_vector)
# phi_indexes = [0, 4, 8]
# n_phi_short = len(phi_indexes)
# phi_vector_short = phi_vector[phi_indexes]
age_cutoff = cutoffs[2]
# tax_vector = [0.008]
# tax_vector = [0.01]
tax_vector = [0.012]
n_tax = len(tax_vector)
theta_matrix = np.empty((N, n_scenarios, n_tax, n_phi, 2))
delta_bar_matrix = np.zeros((N, n_scenarios, n_tax, n_phi, 2))
Phi_parti_1_matrix = np.zeros((N, n_scenarios, n_tax, n_phi, 2))
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

# var_list = [theta_matrix, Phi_parti_1_matrix, delta_bar_matrix]
# var_name_list = ['market price of risk', 'consumption share 1 of participants',
#                  'consumption-weighted estimation error of participants']
# type_list = ['mean', 'vola']
# for i, var in enumerate(var_list):
#     np.save(var_name_list[i] + str(tax_vector[0]), var)


tax_vector = [0.008, 0.01, 0.012]
n_tax = len(tax_vector)
theta_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
Phi_parti_1_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
Delta_bar_parti_Mat = np.zeros((n_scenarios, n_tax, n_phi, 2))
var_list = [theta_Mat, Phi_parti_1_Mat, Delta_bar_parti_Mat]
var_name_list = ['market price of risk', 'consumption share 1 of participants',
                 'consumption-weighted estimation error of participants']
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
scenario_list = ['Complete', 'Reentry', 'Disappointment']
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
                        ax.plot(X_[x_start:], Y_[x_start:], linestyle=line_style, color=colors_short[l],
                                label=scenario_list[l])
                    elif j == 1 and l == 0:
                        ax.plot(X_[x_start:], Y_[x_start:], linestyle=line_style, color=colors_short[l],
                                label=tau_list[k])
                    else:
                        ax.plot(X_[x_start:], Y_[x_start:], linestyle=line_style, color=colors_short[l])
                else:
                    if l != 0:
                        ax.plot(X_[x_start:], Y_[x_start:], linestyle=line_style, color=colors_short[l])
        if j == 0:
            ax.set_ylabel(row_name, rotation=90)
        if i == 2:
            ax.set_xlabel(r'$\phi$')
        if i == 0:
            ax.legend()
            ax.set_title(column_name)
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig('phi and values mean vola.png', dpi=200, format="png")
# plt.savefig('initial window and values mean vola.png', dpi=500, format="png")
plt.show()
# plt.close()
