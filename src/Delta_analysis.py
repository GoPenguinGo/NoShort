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
    Z_Y_cases, Z_SI_cases, scenario_labels, colors_short, age_labels
from src.stats import shocks, tau_calculator, good_times, Delta_st_compare, dDelta_st_calculator, post_var
from numba import jit
import matplotlib.pyplot as plt
import statsmodels.api as sm
# import tabulate as tabulate
from scipy.interpolate import make_interp_spline

t = np.arange(0, T_cohort, dt)
n_scenarios = 3
scenarios_short = scenarios[:n_scenarios]

scenarios_two = scenarios[1:3]
# n_scenarios = 1
# scenarios_short = scenarios[1:2]

phi_vector = np.arange(0,1,0.1)
n_phi = len(phi_vector)

phi_indexes = [0, 4, 8]
n_phi_short = len(phi_indexes)
phi_vector_short = phi_vector[phi_indexes]


# ######################################
# ############ The signal ##############
# ###### P of better estimate ##########
# ######################################
n_paths = 10000
Delta_init = 0
age_vector = [5, 20, 50]
n_age = len(age_vector)
data_delta_compare = np.empty((n_age, n_phi, 3))
for i, age in enumerate(age_vector):
    for j, phi_try in enumerate(phi_vector):
        t_s = time.time()
        data_delta_compare[i, j] = Delta_st_compare(Delta_init, Vhat, age, dt, sigma_Y_sqr, phi_try, n_paths)
        print(time.time() - t_s)
legend_condition = ['unconditional', 'positive corr', 'negative corr']
fig, axes = plt.subplots(nrows=1, ncols=3, sharey='all', figsize=(10, 5))
# fig.suptitle('Good Z^Y, Bad Z^SI')
for i, ax in enumerate(axes):
    for j in range(3):
        y = data_delta_compare[i, 1:, j]
        ax.plot(phi_vector[1:], y, color=colors_short[j], linewidth=0.6, label=legend_condition[j])
        if i == 0:
            ax.legend()
            ax.set_ylabel('P(better estimate with signal)')
        ax.set_xlabel(r'$\phi$ values')
        ax.set_title('age = ' + str(age_vector[i]))
        ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout(h_pad=2)
plt.savefig('Probability of better estimate with signal.png', dpi=500)
plt.show()
#plt.close()


# ######################################
# ############ The signal ##############
# ###### Corr(shocks, dDelta) ##########
# ######################################

max_age = 100
n_paths = 10000
step_age = 2
age = np.arange(0, max_age, step_age)
n_age = len(age)
step_N = np.arange(0, int(max_age / dt), int(step_age / dt), dtype=int)

long_length = int(max_age / dt)
shocks_z_Y = np.random.randn(long_length, n_paths) * np.sqrt(dt)  # shape: (n_paths)
z_Y = np.cumsum(shocks_z_Y, axis=0)
shocks_z_SI = np.random.randn(long_length, n_paths) * np.sqrt(dt)
z_SI = np.cumsum(shocks_z_SI, axis=0)
z_Y_init = np.sum(np.random.randn(Npre, n_paths) * np.sqrt(dt), axis=0)
Delta_init = z_Y_init / T_hat
dz_Y = np.random.randn(n_paths) * np.sqrt(dt)
dz_SI = np.random.randn(n_paths) * np.sqrt(dt)
dz_Y_mat = np.tile(dz_Y, (n_age, 1))
dz_SI_mat = np.tile(dz_SI, (n_age, 1))
corr_dDelta_dz_Y = np.empty((n_phi, n_age))
corr_dDelta_dz_SI = np.empty((n_phi, n_age))
for i, phi_try in enumerate(phi_vector):
    a_phi = (1 - phi_try ** 2)
    Vhat_st = sigma_Y_sqr * a_phi * Vhat / (sigma_Y_sqr * a_phi + Vhat * age)
    phi_factor = phi_try / np.sqrt(a_phi)
    factor_P = 1 / (sigma_Y_sqr * a_phi + Vhat * age)
    Vhat_st_mat = np.transpose(np.tile(Vhat_st, (n_paths, 1)))
    factor_P_mat = np.transpose(np.tile(factor_P, (n_paths, 1)))
    Delta_P = sigma_Y_sqr * a_phi * factor_P_mat * Delta_init + Vhat_st_mat * a_phi * factor_P_mat * (
            z_Y[step_N, :] - phi_factor * z_SI[step_N, :])  # shape: (n_paths)
    dDelta_st = Vhat_st_mat / sigma_Y ** 2 * (-1/a_phi * Delta_P * dt + dz_Y_mat - phi_try / np.sqrt(a_phi) * dz_SI_mat)
    for j in range(n_age):
        corr_dDelta_dz_Y[i, j] = np.corrcoef(dDelta_st[j,:], dz_Y_mat[j, :])[0,1]
        corr_dDelta_dz_SI[i, j] = np.corrcoef(dDelta_st[j, :], dz_SI_mat[j, :])[0, 1]


# ######################################
# ############## Delta #################
# ###### Distribution of Delta #########
# ######################################
dZ_build = dZ_build_matrix[0]
dZ_SI_build = dZ_SI_build_matrix[0]
dZ = Z_Y_cases[1]  # bad
dZ_SI = Z_SI_cases[1]  # bad
phi_fix = phi_vector[4]
Delta_compare = np.empty((len(scenarios_two), Nt, Nc))
invest_tracker_compare = np.empty((len(scenarios_two), Nt, Nc))
theta_compare = np.empty((len(scenarios_two), Nt))
cohort_size_mat = np.tile(cohort_size, (Nc, 1))
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
        Phi_parti,
        Delta_bar_parti,
        dR,
        invest_tracker
    ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta,
                    phi_fix,
                    Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
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
Z = np.cumsum(Z_Y_cases[1])[int(left_t/dt):int(right_t/dt)]
Z_SI = np.cumsum(Z_SI_cases[1])[int(left_t/dt):int(right_t/dt)]
x = t[int(left_t/dt):int(right_t/dt)]
scenario_indexes = [0, 1]
fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all', figsize=(15, 15))
for i, ax_row in enumerate(axes):
    scenario_index = scenario_indexes[i]
    y1 = y_overall[scenario_index, int(left_t/dt):int(right_t/dt)]
    y2 = y_P[scenario_index, int(left_t/dt):int(right_t/dt)]  # ((Nt, 5))
    y3 = y_N[scenario_index, int(left_t/dt):int(right_t/dt)]  # ((Nt, 5))
    y4 = y_min[scenario_index, int(left_t/dt):int(right_t/dt)]
    y5 = y_max[scenario_index, int(left_t/dt):int(right_t/dt)]
    belief_cutoff_case = -theta_compare[scenario_index, int(left_t/dt):int(right_t/dt)]
    for j, ax in enumerate(ax_row):
        ax.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
        ax.set_title(scenario_labels[scenario_index+1] + r', $\phi=0.4$')
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
            ax.fill_between(x, y21, y23, color='blue', linewidth=0., alpha=0.5, label='P')
            ax.fill_between(x, y30, y34, color='green', linewidth=0., alpha=0.3)
            ax.fill_between(x, y31, y33, color='green', linewidth=0., alpha=0.5, label='N')
            # ax.plot(x, y22, color='blue', linewidth=0.4, label='P')
            # ax.plot(x, y32, color='green', linewidth=0.4, label='N')
            ax.plot(x, belief_cutoff_case, color='black', linewidth=0.4, label='Cutoff belief')
        else:
            ax.plot(x, belief_cutoff_case, color='black', linewidth=0.4, label='Cutoff belief')
            for k in range(n_age_groups):
                y40 = y4[:, k]
                y50 = y5[:, k]
                ax.fill_between(x, y40, y50, color=colors_short[k], linewidth=0., alpha=0.4, label=age_labels[k])
        if i == 0:
            ax.legend(loc='upper right')
        else:
            ax.set_xlabel('Time in simulation, one random path')
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig('Distribution of Delta.png', dpi=500)
plt.show()
# plt.close()


# ######################################
# ############## Delta #################
# ###### Distribution of Delta #########
# ######## with small window ###########
# ######################################
dZ_build = dZ_build_matrix[0]
dZ_SI_build = dZ_SI_build_matrix[0]
dZ = Z_Y_cases[1]  # bad
dZ_SI = Z_SI_cases[1]  # bad
phi_fix = phi_vector[4]
Delta_compare = np.empty((len(scenarios_two), Nt, Nc))
invest_tracker_compare = np.empty((len(scenarios_two), Nt, Nc))
theta_compare = np.empty((len(scenarios_two), Nt))
cohort_size_mat = np.tile(cohort_size, (Nc, 1))
Npre_short = 60
T_hat_short = dt * Npre_short
Vhat_short = (sigma_Y ** 2) / T_hat_short
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
        Phi_parti,
        Delta_bar_parti,
        dR,
        invest_tracker
    ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu,
                    Vhat_short,
                    mu_Y, sigma_Y, sigma_S, tax, beta,
                    phi_fix,
                    Npre_short,
                    Ninit,
                    T_hat_short, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
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
Z = np.cumsum(Z_Y_cases[1])[int(left_t/dt):int(right_t/dt)]
Z_SI = np.cumsum(Z_SI_cases[1])[int(left_t/dt):int(right_t/dt)]
x = t[int(left_t/dt):int(right_t/dt)]
scenario_indexes = [0, 1]
fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all', figsize=(15, 15))
for i, ax_row in enumerate(axes):
    scenario_index = scenario_indexes[i]
    y1 = y_overall[scenario_index, int(left_t/dt):int(right_t/dt)]
    y2 = y_P[scenario_index, int(left_t/dt):int(right_t/dt)]  # ((Nt, 5))
    y3 = y_N[scenario_index, int(left_t/dt):int(right_t/dt)]  # ((Nt, 5))
    y4 = y_min[scenario_index, int(left_t/dt):int(right_t/dt)]
    y5 = y_max[scenario_index, int(left_t/dt):int(right_t/dt)]
    belief_cutoff_case = -theta_compare[scenario_index, int(left_t/dt):int(right_t/dt)]
    for j, ax in enumerate(ax_row):
        ax.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
        ax.set_title(scenario_labels[scenario_index+1] + r', $\phi=0.4$')
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
            ax.fill_between(x, y21, y23, color='blue', linewidth=0., alpha=0.5, label='P')
            ax.fill_between(x, y30, y34, color='green', linewidth=0., alpha=0.3)
            ax.fill_between(x, y31, y33, color='green', linewidth=0., alpha=0.5, label='N')
            # ax.plot(x, y22, color='blue', linewidth=0.4, label='P')
            # ax.plot(x, y32, color='green', linewidth=0.4, label='N')
            ax.plot(x, belief_cutoff_case, color='black', linewidth=0.4, label='Cutoff belief')
        else:
            ax.plot(x, belief_cutoff_case, color='black', linewidth=0.4, label='Cutoff belief')
            for k in range(n_age_groups):
                y40 = y4[:, k]
                y50 = y5[:, k]
                ax.fill_between(x, y40, y50, color=colors_short[k], linewidth=0., alpha=0.4, label=age_labels[k])
        if i == 0:
            ax.legend(loc='upper right')
        else:
            ax.set_xlabel('Time in simulation, one random path')
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig('Distribution of Delta window 5 years.png', dpi=500)
plt.show()
# plt.close()