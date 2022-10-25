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
phi_indexes = [0, 4, 8]
n_phi_short = len(phi_indexes)
phi_vector_short = phi_vector[phi_indexes]
age_cutoff = cutoffs[2]
tax_vector = [0.005, 0.01, 0.015]
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