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

# Define the simulate_scenario function as shown in the previous answer
Mpath = 100

def simulate_mpath(i: int,
                   Nscenario=2,
                   ):
    print(i)
    # Initialize results for the current Mpath
    dR_mean_vola_results = np.zeros((Nscenario, 2))
    theta_mean_vola_results = np.zeros((Nscenario, 2))
    r_mean_vola_results = np.zeros((Nscenario, 2))
    mu_S_mean_vola_results = np.zeros((Nscenario, 2))
    sigma_S_mean_vola_results = np.zeros((Nscenario, 2))
    beta_mean_vola_results = np.zeros((Nscenario, 2))
    theta_save_mean_vola_results = np.zeros((Nscenario, 2, 2))
    sigma_S_save_mean_vola_results = np.zeros((Nscenario, 4, 2))
    # parti_group_mean_vola_results = np.zeros((Nscenario, 2, 4))
    parti_age_group_mean_vola_results = np.zeros((Nscenario, 4))
    parti_wealth_group_mean_vola_results = np.zeros((Nscenario, 4))
    cov_save_mean_vola_results = np.zeros((Nscenario, 6))

    dZ_build = dZ_build_matrix[i]
    dZ = dZ_matrix[i]
    dZ_SI_build = dZ_SI_build_matrix[i]
    dZ_SI = dZ_SI_matrix[i]

    for g, scenario in enumerate(scenarios[:Nscenario]):
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        (
            dR_mean_vola,
            theta_mean_vola,
            r_mean_vola,
            mu_S_mean_vola,
            sigma_S_mean_vola,
            beta_mean_vola,
            theta_save_mean_vola,
            sigma_S_save_mean_vola,
            # parti_group_mean_vola,
            parti_age_group_mean_vola,
            parti_wealth_group_mean_vola,
            cov_save_mean_vola,
        ) = simulate_SI_mean_vola(
            mode_trade, mode_learn, Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax, beta0,
            phi, Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau,
            Ntype, rho_i, alpha_i, beta_i, beta_cohort_type, cohort_type_size
        )

        dR_mean_vola_results[g] = dR_mean_vola
        theta_mean_vola_results[g] = theta_mean_vola
        r_mean_vola_results[g] = r_mean_vola
        mu_S_mean_vola_results[g] = mu_S_mean_vola
        sigma_S_mean_vola_results[g] = sigma_S_mean_vola
        beta_mean_vola_results[g] = beta_mean_vola
        theta_save_mean_vola_results[g] = theta_save_mean_vola
        sigma_S_save_mean_vola_results[g] = sigma_S_save_mean_vola
        # parti_group_mean_vola_results[g] = parti_group_mean_vola
        parti_age_group_mean_vola_results[g] = parti_age_group_mean_vola
        parti_wealth_group_mean_vola_results[g] = parti_wealth_group_mean_vola
        cov_save_mean_vola_results[g] = cov_save_mean_vola

    return (
        i,
        dR_mean_vola_results,
        theta_mean_vola_results,
        r_mean_vola_results,
        mu_S_mean_vola_results,
        sigma_S_mean_vola_results,
        beta_mean_vola_results,
        theta_save_mean_vola_results,
        sigma_S_save_mean_vola_results,
        # parti_group_mean_vola_results,
        parti_age_group_mean_vola_results,
        parti_wealth_group_mean_vola_results,
        cov_save_mean_vola_results,
    )

# Create a Pool of processes for parallel execution
# Create a ProcessPoolExecutor for parallel execution
def main():
    # Create a ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=16) as executor:  # Adjust the number of workers as needed
        results = [executor.submit(simulate_mpath, i) for i in range(Mpath)]
    # Initialize a list to store the results
    results_list = []

    # Retrieve results from parallel processes
    for result in results:
        i, dR_mean_vola_results, \
            theta_mean_vola_results, \
            r_mean_vola_results, \
            mu_S_mean_vola_results, \
            sigma_S_mean_vola_results, \
            beta_mean_vola_results, \
            theta_save_mean_vola_results, \
            sigma_S_save_mean_vola_results, \
            parti_age_group_mean_vola_results,\
            parti_wealth_group_mean_vola_results,\
            cov_save_mean_vola_results = result.result()

        data = {
            "i": i,
            "dR_mean_vola": dR_mean_vola_results,
            "theta_mean_vola": theta_mean_vola_results,
            "r_mean_vola": r_mean_vola_results,
            "mu_S_mean_vola": mu_S_mean_vola_results,
            "sigma_S_mean_vola": sigma_S_mean_vola_results,
            "beta_mean_vola": beta_mean_vola_results,
            "theta_save_mean_vola": theta_save_mean_vola_results,
            "sigma_S_save_mean_vola": sigma_S_save_mean_vola_results,
            "parti_age_group_mean_vola":  parti_age_group_mean_vola_results,
            "parti_wealth_group_mean_vola":  parti_wealth_group_mean_vola_results,
            "cov_save_mean_vola": cov_save_mean_vola_results
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a .npz file
    results_dict = results_df.to_dict(orient='list')
    np.savez("results.npz", **results_dict)

if __name__ == '__main__':
    main()
