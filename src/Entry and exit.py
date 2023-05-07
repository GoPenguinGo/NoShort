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
    scenario_labels, colors_short, colors_short2, PN_labels, age_labels, cummu_popu
from src.stats import shocks, tau_calculator, good_times, Delta_st_compare, weighted_variance
from numba import jit
import matplotlib.pyplot as plt
import statsmodels.api as sm
# import tabulate as tabulate
from scipy.interpolate import make_interp_spline
import pandas as pd

# what predicts entry and exit?
# with a negative shock (participants -> nonparticipants), positive shock (nonparticipants -> participants)
Mpath = 2000
n_scenarios_short = 3
phi_fix = 0.4
scenarios_short = scenarios[:n_scenarios_short]
parti_rate_mat = np.zeros((Mpath, n_scenarios_short, Nt, n_age_groups), dtype=np.float32)
leverage_popu_mat = np.zeros((Mpath, n_scenarios_short, Nt, n_age_groups), dtype=np.float32)
leverage_wealth_mat = np.zeros((Mpath, n_scenarios_short, Nt, n_age_groups), dtype=np.float32)
leverage_popu_condi_mat = np.zeros((Mpath, n_scenarios_short, Nt, n_age_groups), dtype=np.float32)
leverage_wealth_condi_mat = np.zeros((Mpath, n_scenarios_short, Nt, n_age_groups), dtype=np.float32)
for i in range(Mpath):
    print(i)
    ii = int(i * 5) +1
    dZ = dZ_matrix[ii]
    dZ_build = dZ_build_matrix[ii]
    dZ_SI = dZ_SI_matrix[ii]
    dZ_SI_build = dZ_SI_build_matrix[ii]
    for j, scenario in enumerate(scenarios_short):
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
                        Vhat,
                        mu_Y, sigma_Y, sigma_S,
                        tax,
                        beta,
                        phi_fix,
                        Npre, Ninit,
                        T_hat,
                        dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                        need_f='True',
                        need_Delta='True',
                        need_pi='True',
                        )
        theta_mat = np.transpose(np.tile(theta, (Nc, 1)))
        leverage = pi
        leverage_wealth = leverage * f
        leverage_condi = np.ma.masked_where(leverage == 0, leverage)
        leverage_wealth_condi = np.ma.masked_where(leverage_wealth == 0, leverage_wealth)
        for age_n in range(n_age_groups):
            if mode_trade == 'w_constraint':
                parti_rate_mat[i, j, :, age_n] = np.average(invest_tracker[:, cutoffs[age_n + 1]:cutoffs[age_n]], axis=1,
                                                            weights=cohort_size[cutoffs[age_n + 1]:cutoffs[age_n]]) / 4
            leverage_popu_mat[i, j, :, age_n] = np.average(leverage[:, cutoffs[age_n + 1]:cutoffs[age_n]], axis=1,
                                                           weights=cohort_size[cutoffs[age_n + 1]:cutoffs[age_n]])
            leverage_wealth_mat[i, j, :, age_n] = np.sum(leverage_wealth[:, cutoffs[age_n + 1]:cutoffs[age_n]], axis=1) \
                                                  / np.sum(f[:, cutoffs[age_n + 1]:cutoffs[age_n]], axis=1)
            leverage_popu_condi_mat[i, j, :, age_n] = np.ma.average(leverage_condi[:, cutoffs[age_n + 1]:cutoffs[age_n]], axis=1,
                                                           weights=cohort_size[cutoffs[age_n + 1]:cutoffs[age_n]])
            leverage_wealth_condi_mat[i, j, :, age_n] = np.ma.sum(leverage_wealth_condi[:, cutoffs[age_n + 1]:cutoffs[age_n]], axis=1) \
                                                  / np.sum(f[:, cutoffs[age_n + 1]:cutoffs[age_n]], axis=1)


# Figure 4
paths = 5 * np.arange(0, 2000, 1)
Z_mat = np.cumsum(dZ_matrix[paths], axis=1)
Z_SI_mat = np.cumsum(dZ_SI_matrix[paths], axis=1)
dZ_mat = Z_mat[:, 24+1200:] - Z_mat[:, 1200:-24]
dZ_SI_mat = Z_SI_mat[:, 24+1200:] - Z_SI_mat[:, 1200:-24]
# save_var_list = [dZ_mat, dZ_SI_mat, parti_rate_mat, leverage_popu_condi_mat]
# for i, var in enumerate(save_var_list):
#     np.save('entry_exit'+str(i), var)
dZ_mat1 = np.copy(dZ_mat)
dZ_SI_mat1 = np.copy(dZ_SI_mat)
parti_rate_mat1 = np.copy(parti_rate_mat)
leverage_popu_condi_mat1 = np.copy(leverage_popu_condi_mat)
dZ_mat2 = np.load('entry_exit'+str(0)+'.npy')
dZ_SI_mat2 = np.load('entry_exit'+str(1)+'.npy')
parti_rate_mat2 = np.load('entry_exit'+str(2)+'.npy')
leverage_popu_condi_mat2 = np.load('entry_exit'+str(3)+'.npy')
dZ_mat = np.append(dZ_mat1, dZ_mat2, axis=0)
dZ_SI_mat = np.append(dZ_SI_mat1, dZ_SI_mat2, axis=0)
parti_rate_mat = np.append(parti_rate_mat1, parti_rate_mat2, axis=0)
leverage_popu_condi_mat = np.append(leverage_popu_condi_mat1, leverage_popu_condi_mat2, axis=0)

change_parti_rate_mat = (parti_rate_mat[:, :, 24+1200:] - parti_rate_mat[:, :, 1200:-24])
# change_leverage_mat = leverage_wealth_mat[:, :, 1:] - leverage_wealth_mat[:, :, :-1]
change_leverage_mat = leverage_popu_condi_mat[:, :, 24+1200:] - leverage_popu_condi_mat[:, :, 1200:-24]
parti_rate_overall = np.sum(parti_rate_mat[:, 1:], axis=3)
leverage_overall = 1/parti_rate_overall
change_parti_rate_overall = parti_rate_overall[:, :, 24+1200:] - parti_rate_overall[:, :, 1200:-24]
change_leverage_overall = leverage_overall[:, :, 24+1200:] - leverage_overall[:, :, 1200:-24]
n_bins = 15
data_figure_median_parti = np.zeros((2, n_scenarios_short, n_bins - 1, n_age_groups, 3))  # x: dZ^Y & dZ^SI, y: parti_rate & leverage
data_figure_median_lev = np.zeros((2, n_scenarios_short, n_bins - 1, n_age_groups, 3))  # x: dZ^Y & dZ^SI, y: parti_rate & leverage
data_figure_mean_parti = np.zeros((2, n_scenarios_short, n_bins - 1, n_age_groups))  # x: dZ^Y & dZ^SI, y: parti_rate & leverage
data_figure_mean_lev = np.zeros((2, n_scenarios_short, n_bins - 1, n_age_groups))  # x: dZ^Y & dZ^SI, y: parti_rate & leverage
# data_figure_cov = np.zeros((2, 2, n_scenarios_short, n_bins - 1, n_age_groups))
data_figure_overall_lev = np.zeros((2, n_scenarios_short - 1, n_bins - 1))
data_figure_overall_parti = np.zeros((2, n_scenarios_short - 1, n_bins - 1))
data_figure_x = np.zeros((2, n_bins - 1))
for i in range(2):
    x_var = dZ_mat if i == 0 else dZ_SI_mat
    x_max = np.percentile(x_var, 90)
    x_min = np.percentile(x_var, 10)
    x_width = (x_max - x_min) / (n_bins - 1)
    x_bins = np.linspace(x_min, x_max, n_bins)
    data_figure_x[i] = (x_bins[1:] + x_bins[:-1]) / 2
    for j in range(n_bins - 1):
        bin_below = x_var >= x_bins[j]
        bin_above = x_bins[j + 1] >= x_var
        data_where = np.where(bin_above * bin_below == 1)
        for l in range(n_scenarios_short):
            for m in range(n_age_groups):
                if l > 0 and m == 0:
                    data_figure_overall_lev[i, l-1, j] = np.average(change_leverage_overall[:, l-1][data_where])
                    data_figure_overall_parti[i, l - 1, j] = np.average(change_parti_rate_overall[:, l - 1][data_where])
                y_parti_bin = change_parti_rate_mat[:, l, :, m][data_where]
                y_lev_bin = change_leverage_mat[:, l, :, m][data_where]
                data_figure_median_parti[i, l, j, m] = np.percentile(y_parti_bin, np.array([25,50,75]))
                data_figure_median_lev[i, l, j, m] = np.percentile(y_lev_bin, np.array([25,50,75]))
                data_figure_mean_parti[i, l, j, m] = np.average(y_parti_bin)
                data_figure_mean_lev[i, l, j, m] = np.average(y_lev_bin)


save_var_list = [data_figure_median_parti, data_figure_median_lev,
                 data_figure_mean_parti, data_figure_mean_lev,
                 data_figure_overall_parti, data_figure_overall_lev]
for i, var in enumerate(save_var_list):
    np.save('entry_exit'+str(i), var)

label_shock = [r'Shocks to the output, $dz^{Y}$', r'Shocks to the signal, $dz^{SI}$']
label_scenario = [r'Complete Market', r'Reentry', r'Disappoitment']
label_title = [r'Entry and exit in the stock market', r'Changes in participants portfolio leverage']
labels = [r'$\phi = 0.0$', r'$\phi = 0.4$', r'$\phi = 0.8$']
age_labels = ['0 < Age <= 15, youngest quartile', '15 < Age <= 35', '35 < Age <= 69', 'Age > 69, oldest quartile']
X_ = np.linspace(-1.5, 1.5, 200)
plt.rcParams["font.family"] = "serif"
k = 1
for nn in range(2):
    shock_index = nn
    label_sh = label_shock[shock_index]
    x = data_figure_x[shock_index]
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15), sharey='row', sharex='all')
    for i, ax_row in enumerate(axes):
        if k == 0:
            y_focus = data_figure_median_parti if i == 0 else data_figure_median_lev
        else:
            y_focus = data_figure_mean_parti if i == 0 else data_figure_mean_lev
            y_overall = data_figure_overall_parti if i == 0 else data_figure_overall_lev
        for j, ax in enumerate(ax_row):
            scenario_index = j + 1
            label_sce = label_scenario[scenario_index]
            ax.set_xlabel(label_sh)
            if j == 0:
                ax.set_ylabel(label_title[i])
            ax.set_title(label_sce + r', $\phi = 0.4$')
            for age_index in range(n_age_groups):
                y = y_focus[shock_index, scenario_index, :, age_index]
                X_Y_Spline = make_interp_spline(x, y, k=5)
                Y_ = X_Y_Spline(X_)
                # ax.plot(x, y[:, 1], color=colors_short[age_index], linewidth=0.8, linestyle='solid', label=age_labels[age_index])
                # ax.fill_between(x, y[:, 0], y[:, 2], color=colors_short[age_index], linewidth=0, alpha=0.25)
                if k == 0:
                    ax.plot(X_, Y_[:, 1], color=colors_short[age_index], linewidth=0.8, linestyle='solid', label=age_labels[age_index])
                    ax.fill_between(X_, Y_[:, 0], Y_[:, 2], color=colors_short[age_index], linewidth=0, alpha=0.25)
                else:
                    y_focus_overall = y_overall[shock_index, scenario_index-1]
                    X_Y_overall_Spline = make_interp_spline(x, y_focus_overall, k=5)
                    Y_overall = X_Y_overall_Spline(X_)
                    ax.plot(X_, Y_, color=colors_short[age_index], linewidth=0.8, linestyle='solid',
                            label=age_labels[age_index])
                    ax.plot(X_, Y_overall, color='black', linewidth=0.8, linestyle='dashed')
                if i == j == 0:
                    ax.legend()
                ax.axhline(0, 0.05, 0.95, color='gray', linestyle='dotted', linewidth=0.6, alpha=0.6)
                ax.axvline(0, 0.05, 0.95, color='gray', linestyle='dotted', linewidth=0.6, alpha=0.6)
            # ax.set_ylim(-1, 1)
            # ax.set_ylabel(r'Covariance with change in average belief')
    fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
    plt.savefig('Reaction to shocks' + str(nn) + '.png', dpi=100)
    plt.savefig('Reaction to shocks' + str(nn) + 'HD.png', dpi=200)
    plt.show()
    # plt.close()








