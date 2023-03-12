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
    dZ_Y_cases, dZ_SI_cases, scenario_labels, colors_short, age_labels, PN_labels
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
        dZ = dZ_Y_cases[case_dzY]  # bad
        dZ_SI = dZ_SI_cases[case_dzSI]  # bad
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
            plt.savefig(str(case_dzY) + str(case_dzSI) + 'Distribution of Delta.png', dpi=60)
        else:
            plt.savefig('IA ' + str(case_dzY) + str(case_dzSI) + 'Distribution of Delta.png', dpi=60)
        plt.show()
        plt.close()


# ######################################
# ############## Delta #################
# ###### Distribution of Delta #########
# ######## with small window ###########
# ######################################
dZ_build = dZ_build_matrix[0]
dZ_SI_build = dZ_SI_build_matrix[0]
dZ = dZ_Y_cases[1]  # bad
dZ_SI = dZ_SI_cases[1]  # bad
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
        f_parti,
        Delta_bar_parti,
        dR,
        invest_tracker,
        popu_can_short,
        popu_short,
        Phi_can_short,
        Phi_short,
    ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu,
                    Vhat_short,
                    mu_Y, sigma_Y, sigma_S, tax, beta,
                    phi_fix,
                    Npre_short,
                    Ninit,
                    T_hat_short,
                    dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
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
Z = np.cumsum(dZ_Y_cases[1])[int(left_t / dt):int(right_t / dt)]
Z_SI = np.cumsum(dZ_SI_cases[1])[int(left_t / dt):int(right_t / dt)]
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
plt.savefig('Distribution of Delta window 5 years.png', dpi=60)
plt.show()
plt.close()

# ######################################
# ######## Age of Participants #########
# ######## and Asset Pricing  ##########
# ######################################
N_paths = 1000
phi_fix = 0
scenario = scenarios[1]
mode_trade = scenario[0]
mode_learn = scenario[1]
cohort_size_mat = np.tile(cohort_size, (Nc, 1))
tau_mat = np.tile(tau, (Nc, 1))
cummu_popu = np.cumsum(cohort_size)
# popus = [0.1, 0.5]
popus = [0.1]
n_popu = len(popus)
theta_compare = np.empty((N_paths, n_popu, Nt))
Phi_compare = np.empty((N_paths, n_popu, Nt))
Delta_bar_compare = np.empty((N_paths, n_popu, Nt))
P_old_compare = np.empty((N_paths, n_popu, Nt))
P_young_compare = np.empty((N_paths, n_popu, Nt))
P_average_age_compare = np.empty((N_paths, n_popu, Nt))
belief_f_old_compare = np.empty((N_paths, n_popu, Nt))
belief_f_young_compare = np.empty((N_paths, n_popu, Nt))
belief_popu_old_compare = np.empty((N_paths, n_popu, Nt))
belief_popu_young_compare = np.empty((N_paths, n_popu, Nt))
Phi_old_compare = np.empty((N_paths, n_popu, Nt))
Phi_young_compare = np.empty((N_paths, n_popu, Nt))
Wealthshare_old_compare = np.empty((N_paths, n_popu, Nt))
Wealthshare_young_compare = np.empty((N_paths, n_popu, Nt))
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
        # total_popu_old = np.sum(cohort_size[:cutoff_age_old])
        # total_popu_young = np.sum(cohort_size[cutoff_age_young:])
        # P_old_compare[i, j] = np.sum(invest_tracker[:, :cutoff_age_old] * cohort_size_mat[:, :cutoff_age_old],
        #                           axis=1) / popu
        # P_young_compare[i, j] = np.sum(invest_tracker[:, cutoff_age_young:] * cohort_size_mat[:, cutoff_age_young:],
        #                             axis=1) / popu
        # Phi_old_compare[i, j] = np.sum(f[:, :cutoff_age_old] * invest_tracker[:, :cutoff_age_old] * dt, axis=1)
        # Phi_young_compare[i, j] = np.sum(f[:, cutoff_age_young:] * invest_tracker[:, cutoff_age_young:] * dt, axis=1)
        # Wealthshare_old_compare[i, j] = np.sum(f[:, :cutoff_age_old] * dt, axis=1)
        # Wealthshare_young_compare[i, j] = np.sum(f[:, cutoff_age_young:] * dt, axis=1)
        # belief_f_old_compare[i, j] = np.average(Delta[:, :cutoff_age_old],
        #                                      weights=f[:, :cutoff_age_old] * dt,
        #                                      axis=1)
        # belief_f_young_compare[i, j] = np.average(Delta[:, cutoff_age_young:],
        #                                        weights=f[:, cutoff_age_young:] * dt,
        #                                        axis=1)
        # belief_popu_old_compare[i, j] = np.average(Delta[:, :cutoff_age_old],
        #                                         weights=cohort_size_mat[:, :cutoff_age_old],
        #                                         axis=1)
        # belief_popu_young_compare[i, j] = np.average(Delta[:, cutoff_age_young:],
        #                                           weights=cohort_size_mat[:, cutoff_age_young:],
        #                                           axis=1)



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

# # distance to cutoff belief over wealth gap
# n_bins = 50
# # y_var = [belief_f_distance_young[:, popu_index, :], belief_f_distance_old[:, popu_index, :]]
# y_var = [belief_popu_distance_young[:, popu_index, :], belief_popu_distance_old[:, popu_index, :]]
# x_var = [Wealthshare_young_compare[:, popu_index, :], Wealthshare_old_compare[:, popu_index, :]]
# # x_var = [wealth_gap[:, popu_index, :], -wealth_gap[:, popu_index, :]]
# fig, axes = plt.subplots(ncols=2, sharey='all', figsize=(10, 4))
# for j, ax in enumerate(axes):
#     y_mat = y_var[j][where_within]
#     x_mat = x_var[j][where_within]
#     x_min = np.min(x_mat)
#     x_max = np.max(x_mat)
#     width_bins = (x_max - x_min) / n_bins
#     y = np.empty((n_bins, 3))
#     for i in range(n_bins):
#         bin_left = x_min + i * width_bins
#         bin_right = bin_left + width_bins
#         bin_1 = x_mat <= bin_right
#         bin_2 = x_mat >= bin_left
#         bin_where = np.where(bin_1 * bin_2 == 1)
#         y[i, 0] = np.median(y_mat[bin_where])
#         y[i, 1] = np.percentile(y_mat[bin_where], 5)
#         y[i, 2] = np.percentile(y_mat[bin_where], 95)
#     x = np.linspace(x_min + width_bins / 2, x_max - width_bins / 2, n_bins)
#     # X_ = np.linspace(x_min, x_max, 50)
#     # Y_ = np.empty((3, 50))
#     # for m in range(3):
#     #     y_i = y[:, m]
#     #     X_Y_Spline = make_interp_spline(x, y_i)
#     #     Y_[m] = X_Y_Spline(X_)
#     # ax.plot(X_, Y_[0], linewidth=0.6)
#     # ax.fill_between(X_, Y_[1], Y_[2], linewidth=0., alpha=0.3)
#     ax.plot(x, y[:, 0], linewidth=0.6)
#     ax.fill_between(x, y[:, 1], y[:, 2], linewidth=0., alpha=0.3)
#     # ax.axvline(0, 0.05, 0.95, linestyle='dashed', color='gray', linewidth=0.8)
#     ax.axhline(0, 0.05, 0.95, linestyle='dashed', color='gray', linewidth=0.8)
#     # ax.legend(loc='upper left')
#     # ax.set_xlabel(x_varname)
#     # ax.set_ylabel(y_varname)
# fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
# # plt.savefig(str(tax)+'Intuition wealth belief two sorts.png', dpi=200)
# plt.show()
# # plt.close()


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


# # ######################################
# # ######## Age of Participants #########
# # ######## and Asset Pricing  ##########
# # ######################################
# ## dynamic, asymmetric impact of the young and the old on the cutoff belief between N and P
# # sort on recent realized shocks, fix the parameters'
N_paths = 2000
t_gaps = [3]  # time span for accumulated shocks
N_t_gaps = len(t_gaps)
scenario_index = 2  # reentry
phi_fix = 0
scenario = scenarios[1]
mode_trade = scenario[0]
mode_learn = scenario[1]
cohort_size_mat = np.tile(cohort_size, (Nc, 1))
tau_mat = np.tile(tau, (Nc, 1))
cummu_popu = np.cumsum(cohort_size)
popu = 0.1  # oldest vs. youngest 10%
cutoff_age_old = np.searchsorted(cummu_popu, popu)
cutoff_age_young = np.searchsorted(cummu_popu, 1 - popu)
total_popu_old = np.sum(cohort_size[:cutoff_age_old])
total_popu_young = np.sum(cohort_size[cutoff_age_young:])
shocks_Z = np.empty((N_paths, Nt))
parti_old = np.empty((N_paths, Nt))  # participation rate old
parti_young = np.empty((N_paths, Nt))  # participation rate young
# belief_f_old = np.empty((N_paths, Nt))  # wealth weighted belief
# belief_f_young = np.empty((N_paths, Nt))
belief_popu_old = np.empty((N_paths, Nt))  # population weighted belief
belief_popu_young = np.empty((N_paths, Nt))
# Phi_old = np.empty((N_paths, Nt))
# Phi_young = np.empty((N_paths, Nt))
wealthshare_old = np.empty((N_paths, Nt))  # wealth share
wealthshare_young = np.empty((N_paths, Nt))
# theta_complete = np.empty((N_paths, Nt))
# Phi_complete = np.empty((N_paths, Nt))
# Delta_bar_complete = np.empty((N_paths, Nt))
for i in range(N_paths):
    print(i)
    dZ_build = np.random.randn(Nt) * np.sqrt(dt)
    dZ = np.random.randn(Nt) * np.sqrt(dt)
    dZ_SI_build = np.random.randn(Nt) * np.sqrt(dt)
    dZ_SI = np.random.randn(Nt) * np.sqrt(dt)
    shocks_Z[i] = dZ
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
    parti_old[i] = np.sum(invest_tracker[:, :cutoff_age_old] * cohort_size_mat[:, :cutoff_age_old],
                             axis=1) / total_popu_old
    parti_young[i] = np.sum(invest_tracker[:, cutoff_age_young:] * cohort_size_mat[:, cutoff_age_young:],
                               axis=1) / total_popu_young
    wealthshare_old[i] = np.sum(f[:, :cutoff_age_old] * dt, axis=1)
    wealthshare_young[i] = np.sum(f[:, cutoff_age_young:] * dt, axis=1)
    belief_popu_old[i] = np.average(Delta[:, :cutoff_age_old],
                                       weights=cohort_size_mat[:, :cutoff_age_old],
                                       axis=1)
    belief_popu_young[i] = np.average(Delta[:, cutoff_age_young:],
                                       weights=cohort_size_mat[:, cutoff_age_young:],
                                       axis=1)

# graphs:
for j in range(N_t_gaps):
    t_gap = t_gaps[j]
    t_gap_vector = np.arange(0, Nt, t_gap)
    Z_reshape = np.sum(np.reshape(shocks_Z[:, 1:t_gap_vector[-1]+1], (N_paths, -1, t_gap)), axis=2)
    n = len(Z_reshape)
    parti_old_change = parti_old[:, t_gap_vector[1:]] - parti_old[:, t_gap_vector[:-1]]  # change
    parti_young_change = parti_young[:, t_gap_vector[1:]] - parti_young[:, t_gap_vector[:-1]]
    wealthshare_old_reshape = wealthshare_old[:, t_gap_vector[:-1]]
    wealthshare_young_reshape = wealthshare_young[:, t_gap_vector[:-1]]
    belief_popu_old_reshape = belief_popu_old[:, t_gap_vector[:-1]]
    belief_popu_young_reshape = belief_popu_young[:, t_gap_vector[:-1]]
    # construct the condition:
    n_bins = 20
    n_tiles = 4
    n_bins_left = int(n_bins/10)
    n_bins_right = n_bins - n_bins_left
    y_mat_all = [parti_old_change, parti_young_change]
    x_mat = Z_reshape
    y_varname = [r'Change in participation rate, oldest $10\%$',  r'Change in participation rate, youngest $10\%$']
    x_varname = r'Realized shocks, $z^{SI}_{t+\frac{1}{4}}-z^{SI}_t$'
    column_name = ['Overall', r'Conditional on wealth share quartiles at $t$']
    # condition_vars = [wealthshare_old_reshape, wealthshare_young_reshape]
    # condition_labels = [[r'Overall', r'Wealth share old, quartile '], [r'Overall', r'Wealth share young, quartile ']]
    condition_vars = [wealthshare_young_reshape, wealthshare_young_reshape]
    condition_labels = [[r'Overall', r'Wealth share young, quartile '], [r'Overall', r'Wealth share young, quartile ']]
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='all', figsize=(15, 15))
    for k1, axes_row in enumerate(axes):
        condition_var = condition_vars[k1]
        y_mat = y_mat_all[k1]
        for k2, ax in enumerate(axes_row):
            if k2 == 0:
                condition = [np.min(condition_var), np.max(condition_var)]
                n_condition = 1
            else:
                condition = np.percentile(condition_var, np.arange(0, 101, (100 / n_tiles)))
                n_condition = n_tiles
            y = np.empty((n_condition, n_bins_right - n_bins_left, 3))

            for i in range(n_condition):
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
                for l in range(n_bins_right - n_bins_left):
                    bin_left = min_gap + (l + n_bins_left) * width_bins
                    bin_right = bin_left + width_bins
                    bin_1 = x_var <= bin_right
                    bin_2 = x_var >= bin_left
                    bin_where = np.where(bin_1 * bin_2 == 1)
                    y[i, l, 0] = np.median(y_var[bin_where])
                    y[i, l, 1] = np.percentile(y_var[bin_where], 25)
                    y[i, l, 2] = np.percentile(y_var[bin_where], 75)
                x = np.linspace(min_gap + n_bins_left * width_bins + width_bins / 2,
                                max_gap - n_bins_left * width_bins - width_bins / 2, n_bins_right - n_bins_left)
                X_ = np.linspace(-1, 1, 30)
                Y_ = np.empty((3, 30))
                for m in range(3):
                    y_i = y[i, :, m]
                    X_Y_Spline = make_interp_spline(x, y_i)
                    Y_[m] = X_Y_Spline(X_)
                label_condi = condition_labels[k1][k2] + str(i + 1) if k2 > 0 else condition_labels[k1][k2]
                ax.plot(X_, Y_[0], linewidth=0.8, color=colors[i], label=label_condi)
                if k2 == 0:
                    ax.fill_between(
                        X_, Y_[1], Y_[2], color=colors[i], linewidth=0., alpha=0.2,
                        label=r'$25^{th}$ to $75^{th}$ percentile')
                else:
                    ax.fill_between(
                        X_, Y_[1], Y_[2], color=colors[i], linewidth=0., alpha=0.2)
            ax.axvline(0, 0.05, 0.95, linewidth=0.4,  linestyle='dashed', color='gray')
            ax.axhline(0, 0.05, 0.95, linewidth=0.4, linestyle='dashed', color='gray')
            ax.legend()
            # ax.set_xlim(-1, 1)
            ax.set_xlabel(x_varname)
            ax.set_ylabel(y_varname[k1])
            ax.set_title(column_name[k2])
            fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
            # plt.savefig('Intuition quartiles ' + str(t_gap) + ' months.png', dpi=200)
            plt.show()
            # plt.close()



######################################################################
#### why is Delta_bar more volatile with the shorting constraint? ####
######################################################################
N_paths = 5000
phi_fix = 0
# look at smaller Npre?
Npres = [60, 240]
n_Npres = len(Npres)
cohort_size_mat = np.tile(cohort_size, (Nc, 1))
tau_mat = np.tile(tau, (Nc, 1))
cummu_popu = np.cumsum(cohort_size)
popus = [0.1, 0.5]
popu = 0.5
n_popu = len(popus)
cutoff_age_old_below = np.searchsorted(cummu_popu, popu)
cutoff_age_young = np.searchsorted(cummu_popu, 1 - popu)
theta_compare = np.empty((N_paths, 2, n_Npres, Nt))
Phi_compare = np.empty((N_paths, 2, n_Npres, Nt))
Delta_bar_compare = np.empty((N_paths, 2, n_Npres, Nt))
belief_popu_old_compare = np.empty((N_paths, n_Npres, Nt))
belief_popu_young_compare = np.empty((N_paths, n_Npres, Nt))
parti_old_compare = np.empty((N_paths, n_Npres, Nt))
parti_young_compare = np.empty((N_paths, n_Npres, Nt))
phi_old_compare = np.empty((N_paths, 2, n_Npres, Nt))
phi_young_compare = np.empty((N_paths, 2, n_Npres, Nt))
belief_f_old_compare = np.empty((N_paths, 2, n_Npres, Nt))  # of participants
belief_f_young_compare = np.empty((N_paths, 2, n_Npres, Nt))  # of participants

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
# x_index = belief_popu_young_compare[:, Npre_index]
# x_index = belief_popu_old_compare[:, Npre_index]
x_index = (belief_popu_old_compare[:, Npre_index] + belief_popu_young_compare[:, Npre_index]) / 2
# x_label = 'Average estimation error, younger half'
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

# figure:
# bin_size = (above_dz - below_dz) / (n_bins - 1)
# x = np.linspace(below_dz + bin_size / 2, above_dz - bin_size / 2, n_bins - 1)
# labels = [r'Wealth share old, ' + 'Lowest quartile', 'Second quartile', 'Third quartile', 'Highest quartile']
# # labels = [r'Wealth old minus young, ' + 'Lowest quartile', 'Second quartile', 'Third quartile', 'Highest quartile']
# sub_titles = ['Comparison', 'Complete market', 'Reentry']
#
# fig, axes = plt.subplots(ncols=3, figsize=(15, 5), sharey='all', sharex='all')
# for j, ax in enumerate(axes):
#     if j == 0:
#         # y_focus = data_figure[:, :, 0]
#         y_focus = data_figure_variance[:, :, 0]
#         for k in range(2):
#             # y_i = y_focus[k]
#             # X_Y_Spline = make_interp_spline(x, y_i[:, 0])
#             # X_ = np.linspace(x.min(), x.max(), 100)
#             # Y_ = X_Y_Spline(X_)
#             # label_j = 'Complete market' if k == 0 else 'Reentry'
#             # x_mean = [X_[np.searchsorted(Y_, average_Delta_bar[k])]]
#             # ax.plot(X_, Y_, color=colors_short[k], linewidth=1, label=label_j)
#             # # ax.fill_between(x, y_i[:, 2], y_i[:, 1], color=colors_short[k], linewidth=0., alpha=0.3)
#             # ax.axhline(average_Delta_bar[k], 0.05, 0.95, color=colors_short[k], linewidth=1, linestyle='dashed', label='Unconditional mean, ' + label_j)
#             # # ax.axvline(0, 0.05, 0.95, color='gray', linewidth=0.8, linestyle='dashed')
#             # ax.scatter(x_mean, [average_Delta_bar[k]], marker='o', color=colors_short[k])
#             # ax.legend(loc='upper left')
#
#             y_i = y_focus[k]
#             X_Y_Spline = make_interp_spline(x, y_i[:, 0])
#             X_Y_Spline_std_below = make_interp_spline(x, y_i[:, 0] - 1.65 * y_i[:, 1])
#             X_Y_Spline_std_above = make_interp_spline(x, y_i[:, 0] + 1.65 * y_i[:, 1])
#             X_ = np.linspace(x.min(), x.max(), 100)
#             Y_ = X_Y_Spline(X_)
#             Y_below = X_Y_Spline_std_below(X_)
#             Y_above = X_Y_Spline_std_above(X_)
#             label_j = 'Complete market' if k == 0 else 'Reentry'
#             x_mean = [X_[np.searchsorted(Y_, average_Delta_bar[k])]]
#             ax.plot(X_, Y_, color=colors_short[k], linewidth=1, label=label_j)
#             ax.fill_between(X_, Y_below, Y_above, color=colors_short[k], linewidth=0., alpha=0.3)
#             ax.axhline(average_Delta_bar[k], 0.05, 0.95, color=colors_short[k], linewidth=1, linestyle='dashed', label='Unconditional mean, ' + label_j)
#             # ax.axvline(0, 0.05, 0.95, color='gray', linewidth=0.8, linestyle='dashed')
#             ax.scatter(x_mean, [average_Delta_bar[k]], marker='o', color=colors_short[k])
#             ax.legend(loc='upper left')
#
#     else:
#         # if j == 1:
#         #     y_focus = data_figure[0, :, 1:]
#         # else:
#         #     y_focus = data_figure[1, :, 1:]
#         # for k in range(4):
#         #     y_i = y_focus[:, k]
#         # for k in range(4):
#         #     y_i = y_focus[:, k]
#         #     Y_ = np.empty((3, 100))
#         #     X_ = np.linspace(x.min(), x.max(), 100)
#         #     for l in range(3):
#         #         X_Y_Spline = make_interp_spline(x, y_i[:, l])
#         #         Y_[l] = X_Y_Spline(X_)
#         #     if j == 1:
#         #         ax.plot(X_, Y_[0], color=colors[k], linewidth=0.8, label=labels[k])
#         #     else:
#         #         ax.plot(X_, Y_[0], color=colors[k], linewidth=0.8)
#         #     ax.fill_between(X_, Y_[2], Y_[1], color=colors[k], linewidth=0., alpha=0.2)
#         #     if j == 1:
#         #         ax.legend(loc='upper left')
#         if j == 1:
#             y_focus = data_figure_variance[0, :, 1:]
#         else:
#             y_focus = data_figure_variance[1, :, 1:]
#         for k in range(4):
#             y_i = y_focus[:, k]
#             Y_ = np.empty((3, 100))
#             X_ = np.linspace(x.min(), x.max(), 100)
#             for l in range(3):
#                 if l == 0:
#                     X_Y_Spline = make_interp_spline(x, y_i[:, l])
#                 elif l == 1:
#                     X_Y_Spline = make_interp_spline(x, y_i[:, 0] - 1.65 * y_i[:, 1])
#                 else:
#                     X_Y_Spline = make_interp_spline(x, y_i[:, 0] + 1.65 * y_i[:, 1])
#                 Y_[l] = X_Y_Spline(X_)
#             if j == 1:
#                 ax.plot(X_, Y_[0], color=colors[k], linewidth=0.8, label=labels[k])
#             else:
#                 ax.plot(X_, Y_[0], color=colors[k], linewidth=0.8)
#             ax.fill_between(X_, Y_[2], Y_[1], color=colors[k], linewidth=0., alpha=0.2)
#             if j == 1:
#                 ax.legend(loc='upper left')
#     ax.set_xlabel(x_label)
#     ax.set_ylabel(r'Estimation error of participants, $\bar{\Delta_t}$')
#     # ax.set_xlim(-0.25, 0.25)
#     # ax.set_title(sub_titles[j][k])
# fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
# # plt.savefig(x_label[:10] + 'DeltaVola.png', dpi=100)
# plt.show()
# # plt.close()


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
# plt.close()