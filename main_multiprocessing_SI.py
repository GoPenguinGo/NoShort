import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from typing import Callable, Tuple
from src.simulation import simulate_SI, simulate_SI_mean_vola
from src.param import rho, nu, mu_Y, sigma_Y, sigma_Y_sqr, v, tax, phi, \
    dt, T_hat, Npre, Vhat, Ninit, T_cohort, Nt, Nc, tau, cohort_size, \
    cutoffs_age, n_age_cutoffs, colors, modes_trade, modes_learn, Mpath, \
    scenarios, dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    dZ_Y_cases, dZ_SI_cases, dZ_build_case, dZ_SI_build_case, t, red_labels, yellow_labels, cohort_labels, \
    scenario_labels, colors_short, colors_short2, PN_labels, age_labels, cummu_popu, dt_root, \
    Ntype, rho_i, alpha_i, beta_i, beta0, beta_cohort_type, cohort_type_size
from src.stats import shocks, tau_calculator, good_times, Delta_st_compare, weighted_variance
from numba import jit
import matplotlib.pyplot as plt
import statsmodels.api as sm
import tabulate as tab
from scipy.interpolate import make_interp_spline
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

phi_vector = np.arange(0, 1, 0.1)
n_phi = len(phi_vector)

phi_indexes = [0, 4, 8]
n_phi_short = len(phi_indexes)
phi_vector_short = phi_vector[phi_indexes]


# Define the simulate_scenario function as shown in the previous answer
def simulate_mpath(i: int,
                   Nscenario=2,
                   ):
    print(i)
    age_cut = 100
    Nc_cut = int(age_cut / dt)
    t_gap = int(2 / dt)  # 2-year window
    N_cut = int(Nc - t_gap)
    gap = 15
    data_point = np.arange(0, N_cut, gap)
    N_data = len(data_point)

    # Initialize results for the current Mpath
    parti_rate_mat = np.zeros((Nscenario, n_phi_short, N_cut), dtype=np.float32)
    belief_pre_mat = np.zeros((Nscenario, n_phi_short, N_cut), dtype=np.float32)
    belief_post_mat = np.zeros((Nscenario, n_phi_short, N_cut), dtype=np.float32)
    Delta_matrix = np.empty((Nscenario, n_phi_short, N_data), dtype=np.float32)
    invest_matrix = np.empty((Nscenario, n_phi_short, N_data), dtype=np.float32)

    dZ_build = dZ_build_matrix[i]
    dZ = dZ_matrix[i]
    dZ_SI_build = dZ_SI_build_matrix[i]
    dZ_SI = dZ_SI_matrix[i]
    for g, scenario in enumerate(scenarios[:Nscenario]):
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        for l, phi in enumerate(phi_vector_short):
            (
                r,
                theta,
                f_c,
                # f_w,
                Delta,
                pi,
                popu_parti,
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
                # w_indiv,
            ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax, beta0,
                            phi,
                            Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cutoffs_age,
                            Ntype, rho_i, alpha_i, beta_i, beta_cohort_type, cohort_type_size,
                            need_f='False',
                            need_Delta='True',
                            need_pi='False',
                            )
            parti_rate_mat[g, l] = np.average(np.average(invest_tracker[:-t_gap] * cohort_type_size, axis=2),
                                              axis=1)  # population weighted
            belief_pre_mat[g, l] = np.average(np.average(Delta[:-t_gap] * cohort_type_size, axis=2), axis=1)
            belief_post_mat[g, l] = np.average(np.average(Delta[t_gap:] * cohort_type_size, axis=2), axis=1)
            Delta_matrix[g, l] = np.flip(np.average(np.abs(Delta), axis=0), axis=1)[0, data_point]
            invest_matrix[g, l] = np.flip(np.average(invest_tracker, axis=0), axis=1)[0, data_point]

    return (
        i,
        parti_rate_mat,
        belief_pre_mat,
        belief_post_mat,
        Delta_matrix,
        invest_matrix,
    )


# Create a Pool of processes for parallel execution
# Create a ProcessPoolExecutor for parallel execution
Mpath = 32


def main():
    # Create a ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=16) as executor:  # Adjust the number of workers as needed
        results = [executor.submit(simulate_mpath, i) for i in range(Mpath)]

    # Initialize a list to store the results
    results_list = []

    # Retrieve results from parallel processes
    for result in results:
        i, \
            parti_rate_results, \
            belief_pre_results, \
            belief_post_results, \
            Delta_results, \
            invest_results, \
            = result.result()

        data = {
            "i": i,
            "parti_rate": parti_rate_results,
            "belief_pre": belief_pre_results,
            "belief_post": belief_post_results,
            "Delta": Delta_results,
            "invest": invest_results,
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a .npz file
    results_dict = results_df.to_dict(orient='list')
    np.savez("results.npz", **results_dict)


if __name__ == '__main__':
    main()
