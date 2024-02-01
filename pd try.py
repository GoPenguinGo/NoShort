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
np.seterr(invalid='ignore')


# noinspection PyTypeChecker
def simulate_path(i: int,
                  Nscenario=2,
                  ):
    print(i)
    # Initialize results for the current Mpath

    # shocks
    dZ_build = dZ_build_matrix[i]
    dZ = dZ_matrix[i]
    dZ_SI_build = dZ_SI_build_matrix[i]
    dZ_SI = dZ_SI_matrix[i]

    pd_path = np.empty((Nscenario, Nt))
    vola_path = np.empty((Nscenario, Nt))
    ave_belief = np.empty((Nscenario, Nt))
    ave_c_belief = np.empty((Nscenario, Nt))
    r_path = np.empty((Nscenario, Nt))
    parti_path = np.empty((Nscenario, Nt))
    Phi_parti_path = np.empty((Nscenario, Nt))
    parti_age_path = np.empty((Nscenario, Nt, 4))

    for g, scenario in enumerate(scenarios[:Nscenario]):
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
        ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax, beta0,
                        phi,
                        Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cutoffs_age,
                        Ntype, rho_i, alpha_i, beta_i, beta_cohort_type, cohort_type_size,
                        need_f='True',
                        need_Delta='True',
                        need_pi='True',
                        )

        pd_path[g] = beta
        vola_path[g] = sigma_S
        ave_belief[g] = np.average(Delta, weights=cohort_size[0], axis=1)
        r_path[g] = r
        parti_path[g] = parti
        Phi_parti_path[g] = Phi_parti
        parti_age_path[g] = parti_age_group
        cons_weights = np.sum(f_c, axis=1)
        ave_c_belief[g] = np.average(Delta, weights=cons_weights, axis=1)

    return (
        i,
        pd_path,
        vola_path,
        ave_belief,
        ave_c_belief,
        r_path,
        parti_path,
        Phi_parti_path,
        parti_age_path
    )


# Create a Pool of processes for parallel execution
# Create a ProcessPoolExecutor for parallel execution
def main():
    # Create a ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=16) as executor:  # Adjust the number of workers as needed
        results = [executor.submit(simulate_path, i) for i in range(Mpath)]
    # Initialize a list to store the results
    results_list = []

    # Retrieve results from parallel processes
    for result in results:
        i, \
        pd_path, \
        vola_path, \
        ave_belief, \
        ave_c_belief, \
        r_path, \
        parti_path, \
        Phi_parti_path, \
        parti_age_path = result.result()

        data = {
            "i": i,
            "price_dividend": pd_path,
            "stock_vola": vola_path,
            "average_belief": ave_belief,
            "average_c_belief": ave_c_belief,
            "interest_rate": r_path,
            "parti_rate": parti_path,
            "Phi_bar": Phi_parti_path,
            "parti_rate_age": parti_age_path,
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a .npz file
    results_dict = results_df.to_dict(orient='list')
    np.savez("results_pd_try.npz", **results_dict)


if __name__ == '__main__':
    main()
