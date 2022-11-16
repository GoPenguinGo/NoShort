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
    n_age_groups, cutoffs, colors, modes_trade, modes_learn, \
    scenarios, dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    Z_Y_cases, Z_SI_cases, scenario_labels, colors_short, age_labels, PN_labels
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

phi_vector = np.arange(0, 1, 0.1)
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
plt.savefig('Probability of better estimate with signal.png', dpi=200)
plt.show()
# plt.close()


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
    dDelta_st = Vhat_st_mat / sigma_Y ** 2 * (
            -1 / a_phi * Delta_P * dt + dz_Y_mat - phi_try / np.sqrt(a_phi) * dz_SI_mat)
    for j in range(n_age):
        corr_dDelta_dz_Y[i, j] = np.corrcoef(dDelta_st[j, :], dz_Y_mat[j, :])[0, 1]
        corr_dDelta_dz_SI[i, j] = np.corrcoef(dDelta_st[j, :], dz_SI_mat[j, :])[0, 1]

# ######################################
# ############## Delta #################
# ###### Distribution of Delta #########
# ######################################
cases = [0, 1]
for case_dzY in cases:
    for case_dzSI in cases:
        dZ_build = dZ_build_matrix[0]
        dZ_SI_build = dZ_SI_build_matrix[0]
        dZ = Z_Y_cases[case_dzY]  # bad
        dZ_SI = Z_SI_cases[case_dzSI]  # bad
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
            plt.savefig(str(case_dzY) + str(case_dzSI) + 'Distribution of Delta.png', dpi=200)
        else:
            plt.savefig('IA ' + str(case_dzY) + str(case_dzSI) + 'Distribution of Delta.png', dpi=200)
        plt.show()
        plt.close()


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
Z = np.cumsum(Z_Y_cases[1])[int(left_t / dt):int(right_t / dt)]
Z_SI = np.cumsum(Z_SI_cases[1])[int(left_t / dt):int(right_t / dt)]
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
        ax.set_title(scenario_labels[scenario_index + 1] + r', $\phi=0.4$' + r', 5 years initial training window')
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
                ax.fill_between(x, y40, y50, color=colors_short[k], linewidth=0., alpha=0.4, label=age_labels[k])
        if i == 0:
            ax.legend(loc='upper right')
        else:
            ax.set_xlabel('Time in simulation')
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig('Distribution of Delta window 5 years.png', dpi=100)
plt.show()
plt.close()

# ######################################
# ######## Age of Participants #########
# ######## and Asset Pricing  ##########
# ######################################
# N_paths = 2000
# phi_fix = 0
# scenario = scenarios[1]
# mode_trade = scenario[0]
# mode_learn = scenario[1]
# cohort_size_mat = np.tile(cohort_size, (Nc, 1))
# tau_mat = np.tile(tau, (Nc, 1))
# cummu_popu = np.cumsum(cohort_size)
# popu = 0.5
# cutoff_age_old = np.searchsorted(cummu_popu, 1 - popu)
# cutoff_age_young = np.searchsorted(cummu_popu, popu)
# theta_compare = np.empty((N_paths, Nt))
# Phi_compare = np.empty((N_paths, Nt))
# Delta_bar_compare = np.empty((N_paths, Nt))
# P_old_compare = np.empty((N_paths, Nt))
# P_young_compare = np.empty((N_paths, Nt))
# P_average_age_compare = np.empty((N_paths, Nt))
# belief_f_old_compare = np.empty((N_paths, Nt))
# belief_f_young_compare = np.empty((N_paths, Nt))
# belief_popu_old_compare = np.empty((N_paths, Nt))
# belief_popu_young_compare = np.empty((N_paths, Nt))
# Phi_old_compare = np.empty((N_paths, Nt))
# Phi_young_compare = np.empty((N_paths, Nt))
# Wealthshare_old_compare = np.empty((N_paths, Nt))
# Wealthshare_young_compare = np.empty((N_paths, Nt))
# # theta_complete = np.empty((N_paths, Nt))
# # Phi_complete = np.empty((N_paths, Nt))
# # Delta_bar_complete = np.empty((N_paths, Nt))
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
                    top=0.05,
                    old_limit=100,
                    )
    theta_compare[i] = theta
    Phi_compare[i] = f_parti
    Delta_bar_compare[i] = Delta_bar_parti
    P_old_compare[i] = np.sum(invest_tracker[:, :cutoff_age_old] * cohort_size_mat[:, :cutoff_age_old], axis=1) / popu
    P_young_compare[i] = np.sum(invest_tracker[:, cutoff_age_young:] * cohort_size_mat[:, cutoff_age_young:],
                                axis=1) / popu
    Phi_old_compare[i] = np.sum(f[:, :cutoff_age_old] * invest_tracker[:, :cutoff_age_old] * dt, axis=1)
    Phi_young_compare[i] = np.sum(f[:, cutoff_age_young:] * invest_tracker[:, cutoff_age_young:] * dt, axis=1)
    Wealthshare_old_compare[i] = np.sum(f[:, :cutoff_age_old] * dt, axis=1)
    Wealthshare_young_compare[i] = np.sum(f[:, cutoff_age_young:] * dt, axis=1)
    # P_average_age_compare[i] = np.average(tau_mat, axis=1, weights=invest_tracker * cohort_size_mat)
    belief_f_old_compare[i] = np.average(Delta[:, :cutoff_age_old],
                                       weights=f[:, :cutoff_age_old] * dt,
                                       axis=1)
    belief_f_young_compare[i] = np.average(Delta[:, cutoff_age_young:],
                                       weights=f[:, cutoff_age_young:] * dt,
                                       axis=1)
    belief_popu_old_compare[i] = np.average(Delta[:, :cutoff_age_old],
                                       weights=cohort_size_mat[:, :cutoff_age_old],
                                       axis=1)
    belief_popu_young_compare[i] = np.average(Delta[:, cutoff_age_young:],
                                       weights=cohort_size_mat[:, cutoff_age_young:],
                                       axis=1)
    # (
    #     r_complete,
    #     theta_complete_t,
    #     f_complete,
    #     Delta_complete,
    #     pi_complete,
    #     popu_parti_complete,
    #     Phi_parti_complete_t,
    #     Delta_bar_parti_complete_t,
    #     dR_complete,
    #     invest_tracker_complete,
    #     popu_short_complete,
    #     popu_can_short_complete,
    #     Phi_can_short_complete,
    #     Phi_short_complete,
    # ) = simulate_SI('complete', 'reentry', Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta,
    #                 phi_fix,
    #                 Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
    #                 need_f='False',
    #                 need_Delta='False',
    #                 need_pi='False',
    #                 top=0.05,
    #                 old_limit=100,
    #                 )
    # theta_complete[i] = theta_complete_t
    # Phi_complete[i] = Phi_parti_complete_t
    # Delta_bar_complete[i] = Delta_bar_parti_complete_t

# y_variables = [theta_compare, Phi_compare, Delta_bar_compare]
# x_variables = [P_old_compare, P_young_compare, P_average_age_compare, belief_gap_compare]
# y_varnames = [r'Market price of risk $\theta$', r'Consumption share of participants $\Phi$', r'Estimation error of participants $\bar{\Delta}$']
# x_varnames = ['Participation rate, old', 'Participation rate, young', 'Average age of participants', r'$\Delta_{s,t}$, old - young']
# for i, var_y in enumerate(y_variables):
#     for j, var_x in enumerate(x_variables):
#         # y = np.mean(var_y, axis=1)
#         # x = np.mean(var_x, axis=1)
#         # y = np.mean(var_y, axis=0)
#         # x = np.mean(var_x, axis=0)
#         y = var_y
#         x = var_x
#         fig, ax = plt.subplots(figsize=(5, 5))
#         ax.scatter(x, y, s=0.1, alpha=0.5)
#         ax.set_xlabel(x_varnames[j])
#         ax.set_ylabel(y_varnames[i])
#         fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
#         plt.savefig('Intuition ' + str(i) + str(j) +'.png', dpi=200)
#         plt.show()
#         plt.close()

# max_gap = np.max(belief_gap_compare)
# min_gap = np.min(belief_gap_compare)
# n_bins = 40
# width_bins = (max_gap - min_gap) / n_bins
# # histogram:
# counts, bins = np.histogram(belief_gap_compare, bins=n_bins)
# total_counts = np.sum(counts)
# plt.hist(bins[:-1], bins, weights=counts / total_counts)
# plt.grid(True)
# # plt.savefig('Histogram_belief_gap' + str(N_paths) + '.png', dpi=200)
# plt.show()

# y on x
# parti_gap = P_old_compare - P_young_compare
# Phi_gap = Phi_old_compare - Phi_young_compare
# y_variables = [theta_compare, Phi_compare, Delta_bar_compare, parti_gap]
# y_complete_variables = [theta_complete, Phi_complete, Delta_bar_complete]
# x_var = belief_gap_compare
# y_varnames = [r'Market price of risk $\theta$', r'Consumption share of participants $\Phi$',
#               r'Estimation error of participants $\bar{\Delta}$', 'Diff in participation rate']
# x_varname = r'$\Delta_{s,t}$, old - young'
# x = np.linspace(min_gap, max_gap, n_bins)
# y_varnames = [r'Market price of risk $\theta$', r'Consumption share of participants $\Phi$',
#               r'Estimation error of participants $\bar{\Delta}$', 'Diff in participation rate']
# x_varname = r'$\Delta_{s,t}$, old - young'
# for i, var_y in enumerate(y_variables):
#     y = np.empty((n_bins))
#     if i < 3:
#         y_complete_variable = y_complete_variables[i]
#         y_complete = np.empty((n_bins))
#         for j in range(n_bins):
#             bin_left = min_gap + j * width_bins
#             bin_right = bin_left + width_bins
#             bin_1 = x_var <= bin_right
#             bin_2 = x_var >= bin_left
#             bin_where = np.where(bin_1 * bin_2 == 1)
#             y[j] = np.mean(var_y[bin_where])
#             y_complete[j] = np.mean(y_complete_variable[bin_where])
#         fig, ax = plt.subplots(figsize=(5, 5))
#         X_Y_Spline = make_interp_spline(x, y)
#         X_Y_comp_Spline = make_interp_spline(x, y_complete)
#         # Returns evenly spaced numbers
#         # over a specified interval.
#         X_ = np.linspace(x.min(), x.max(), 100)
#         Y_ = X_Y_Spline(X_)
#         Y_comp = X_Y_comp_Spline(X_)
#         ax.plot(X_, Y_, color=colors_short[0], label='Reentry', linewidth=0.6)
#         ax.plot(X_, Y_comp, color=colors_short[1], label='Complete', linewidth=0.6)
#         if i == 0:
#             ax.axhline(sigma_Y, 0.05, 0.95, linestyle='dotted', color='black', linewidth=0.4)
#             ax.axvline(0, 0.05, 0.95, linestyle='dotted', color='black', linewidth=0.4)
#     else:
#         for j in range(n_bins):
#             bin_left = min_gap + j * width_bins
#             bin_right = bin_left + width_bins
#             bin_1 = x_var <= bin_right
#             bin_2 = x_var >= bin_left
#             bin_where = np.where(bin_1 * bin_2 == 1)
#             y[j] = np.mean(var_y[bin_where])
#         fig, ax = plt.subplots(figsize=(5, 5))
#         X_Y_Spline = make_interp_spline(x, y)
#         # Returns evenly spaced numbers
#         # over a specified interval.
#         X_ = np.linspace(x.min(), x.max(), 100)
#         Y_ = X_Y_Spline(X_)
#         ax.plot(X_, Y_, color=colors_short[0], label='Reentry', linewidth=0.6)
#         Y0 = X_Y_Spline(0)
#         ax.axhline(Y0, 0.05, 0.95, linestyle='dotted', color='black', linewidth=0.4)
#         ax.axvline(0, 0.05, 0.95, linestyle='dotted', color='black', linewidth=0.4)
#     ax.set_title(y_varnames[i])
#     ax.set_xlabel(x_varname)
#     ax.set_ylabel(y_varnames[i])
#     ax.legend()
#     fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
#     # plt.savefig('Intuition ' + str(i) + str(N_paths) + '.png', dpi=200)
#     plt.show()
#     # plt.close()

# construct the condition:
n_tiles = 5
n_bins = 10
belief_f_gap_compare = belief_f_old_compare - belief_f_young_compare
belief_popu_gap_compare = belief_popu_old_compare - belief_popu_young_compare
parti_gap = P_old_compare - P_young_compare
Phi_gap = Phi_old_compare - Phi_young_compare
wealth_gap = Wealthshare_old_compare - Wealthshare_young_compare
y_variables = [theta_compare, Phi_compare, Delta_bar_compare, parti_gap]
# y_complete_variables = [theta_complete, Phi_complete, Delta_bar_complete]
x_mat = belief_popu_gap_compare
y_varnames = [r'Market price of risk $\theta$', r'Consumption share of participants $\Phi$',
              r'Estimation error of participants $\bar{\Delta}$', 'Participation rate, old - young']
x_varname = r'$\Delta_{s,t}$, old - young'
# condition_var = Wealthshare_young_compare
# condition_label = r'Wealth share young, quartile '
condition_var = Wealthshare_young_compare
condition_label = r'$\Phi$ young, quartile '
# condition_var = Phi_gap
# condition_label = r'$\Phi$ old-young, quartile '
# condition_var = wealth_gap
# condition_label = r'Wealth share old-young, quartile '
condition = np.percentile(condition_var, np.arange(0, 101, (100/n_tiles)))
y = np.empty((n_tiles, n_bins))
for l, y_mat in enumerate(y_variables):
    y_varname = y_varnames[l]
    fig, ax = plt.subplots(figsize=(5, 5))
    for i in range(n_tiles):
        below = condition[i]
        above = condition[i + 1]
        a = below < condition_var
        b = condition_var < above
        data_where = np.where(a * b == 1)
        x_var = x_mat[data_where]
        y_var = y_mat[data_where]
        min_gap = np.min(x_var)
        max_gap = np.max(x_var)
        width_bins = (max_gap - min_gap) / n_bins
        for j in range(n_bins):
            bin_left = min_gap + j * width_bins
            bin_right = bin_left + width_bins
            bin_1 = x_var <= bin_right
            bin_2 = x_var >= bin_left
            bin_where = np.where(bin_1 * bin_2 == 1)
            y[i, j] = np.mean(y_var[bin_where])
        x = np.linspace(min_gap + width_bins / 2, max_gap - width_bins / 2, n_bins)
        y_i = y[i]
        X_Y_Spline = make_interp_spline(x, y_i)
        X_ = np.linspace(x.min(), x.max(), 100)
        Y_ = X_Y_Spline(X_)
        ax.plot(X_, Y_, linewidth=0.6, color=colors[i], label=condition_label + str(i))
    ax.axvline(0, 0.05, 0.95, linestyle='dashed', color='gray')
    ax.axhline(0, 0.05, 0.95, linestyle='dashed', color='gray')
    ax.legend()
    # ax.set_xlim(-0.2, 0.2)
    ax.set_xlabel(x_varname)
    ax.set_ylabel(y_varname)
    fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
    # plt.savefig('Intuition dixiles.png', dpi=200)
    plt.show()
    # plt.close()



# # ######################################
# # ######## Age of Participants #########
# # ######## and Asset Pricing  ##########
# # ######################################
# ## how long they stay in the stock market upon entry
# # & how long they stay out of the stock market upon exit
# scenario_index = 2  # reentry
# N_1 = 1000
# phi_5 = np.linspace(0, 0.8, 5)
# dt_root = np.sqrt(dt)
# age_cut = 120
# N_cut = int(age_cut / dt)
# s_cohorts = int(Nt - N_cut + 1)
# switch_exit = np.empty((N_1, 5, N_cut - 1))
# switch_entry = np.empty((N_1, 5, N_cut - 1))
#
# for j in range(N_1):
#     print(j)
#     dZ = np.random.randn(Nt) * dt_root
#     dZ_build = np.random.randn(Nc) * dt_root
#     dZ_SI = np.random.randn(Nt) * dt_root
#     dZ_SI_build = np.random.randn(Nc) * dt_root
#     for l, phi_try in enumerate(phi_5):
#         (
#             r_reentry,
#             theta_reentry,
#             f_reentry,
#             Delta_reentry,
#             pi_reentry,
#             popu_parti_reentry,
#             f_parti_reentry,
#             Delta_bar_parti_reentry,
#             dR_reentry,
#             invest_tracker_reentry,
#             popu_short_reentry,
#             popu_can_short_reentry,
#             Phi_can_short_reentry,
#             Phi_short_reentry,
#         ) = simulate_SI('w_constraint', 'reentry', Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta,
#                         phi_try,
#                         Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
#                         need_f='False',
#                         need_Delta='False',
#                         need_pi='False',
#                         top=0.05,
#                         old_limit=100
#                         )
#         parti_tracker = np.rot90(invest_tracker_reentry)
#         parti_time_series = np.zeros((s_cohorts, N_cut))
#         for m in range(s_cohorts):
#             parti_time_series[m] = np.diag(parti_tracker, k=-m)[:N_cut]
#         exit_time_series = ((parti_time_series[:, 1:] - parti_time_series[:, :-1]) == -1)
#         entry_time_series = ((parti_time_series[:, 1:] - parti_time_series[:, :-1]) == 1)
#         # exit_time_series = np.insert(exit_time_series, 0, (parti_time_series[:, 0] == 0), axis=1)
#         switch_exit[j, l] = np.mean(exit_time_series, axis=0)
#         switch_entry[j, l] = np.mean(entry_time_series, axis=0)
#
# np.save('switch_exit_data', switch_exit)
# np.save('switch_entry_data', switch_entry)
#
# switch_exit_age = np.average(switch_exit, axis=0)
# switch_entry_age = np.average(switch_entry, axis=0)
#
# switch_exit_year = np.empty((5, 20))
# switch_entry_year = np.empty((5, 20))
# for i in range(20):
#     for j in range(5):
#         switch_exit_year[j, i] = np.sum(switch_exit_age[j, int(i / dt):int((i + 5) / dt)])
#         switch_entry_year[j, i] = np.sum(switch_entry_age[j, int(i / dt):int((i + 5) / dt)])
# switch_year = switch_entry_year + switch_exit_year
#
# y_variables = [switch_year, switch_entry_year, switch_exit_year]
# length = 20
# x = np.linspace(0, 95, 20)
# y_varnames = [r'switch', r'entry', r'exit']
# x_varname = r'Age'
# for i, var_y in enumerate(y_variables):
#     fig, ax = plt.subplots(figsize=(5, 5))
#     for j in range(5):
#         y = var_y[j][:length]
#         X_Y_Spline = make_interp_spline(x, y)
#         # Returns evenly spaced numbers
#         # over a specified interval.
#         X_ = np.linspace(x.min(), x.max(), 100)
#         Y_ = X_Y_Spline(X_)
#         ax.plot(X_, Y_, color=colors[j], label=r'$\phi=$'+str(phi_5[j]), linewidth=0.6)
#     ax.set_title(y_varnames[i])
#     ax.set_xlabel(x_varname)
#     ax.set_ylabel(y_varnames[i])
#     ax.legend()
#     fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
#     # plt.savefig('Intuition ' + str(i) + str(N_paths) + '.png', dpi=200)
#     plt.show()
#     # plt.close()