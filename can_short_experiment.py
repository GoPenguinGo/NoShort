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
    Z_Y_cases, Z_SI_cases, t, red_cases, yellow_cases, cohort_labels, \
    scenario_labels, colors_short , colors_short2, PN_labels, age_labels
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

# Individual paths:
dZ_build = dZ_build_matrix[0]
dZ_SI_build = dZ_SI_build_matrix[0]  # fix the shocks at the buildup stage
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
        dZ = Z_Y_cases[i]
        log_Yt = np.cumsum((mu_Y - 0.5 * sigma_Y ** 2) * dt + sigma_Y * dZ)
        log_Yt_mat = np.transpose(np.tile(log_Yt, (Nc, 1)))
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
                    invest_tracker
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
                # market_view_compare[g, i, j, k] = np.average(Delta, axis=1, weights=f)
                delta_bar_compare[g, i, j, k] = Delta_bar_parti
                Phi_compare[g, i, j, k] = Phi_parti
                # survey_view_compare[g, i, j, k] = np.average(Delta, axis=1, weights=cohort_size)
                # belief_dispersion_compare[g, i, j, k] = np.std(Delta, axis=1)  # todo: maybe add weights
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
                    parti = pi_time_series[o, i, j, l, m] > 0.00001 or pi_time_series[o, i, j, l, m] < -0.00001
                    switch = abs(parti[1:] ^ parti[:-1])
                    sw = np.append(switch, 0)
                    parti = np.where(sw == 1, 0.5, parti)
                    switch = np.insert(switch, 0, 0)
                    parti = np.where(switch == 1, 0.5, parti)
                    parti_time_series[o, i, j, l, m] = parti
                    switch_time_series[o, i, j, l, m] = sw