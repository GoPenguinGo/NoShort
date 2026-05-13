from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.param import N_workers, Mpath
from src.param import (mu_Y, sigma_Y, \
                       dt, Ninit, Nt, Nc, tau, tax, \
                       Ntype, alpha_i, \
                       dZ_matrix, dZ_build_matrix, cohort_size, T_hat, Npre, Vhat, nu, beta0, phi,
                       window_bell, entry_bound, exit_bound)
from src.param_mix import Nconstraint, rho_i_mix, density
from src.simulation import simulate_mix_mean_vola

plt.rcParams["font.family"] = 'serif'
window = 12  # 1-year non-overlapping windows
sample = np.arange(600, Nt - 600, window)
N_sample = len(sample)
# age_cut = 100
# Nc_cut = int(age_cut / dt)
age_sample = np.arange(1, int(200 / dt), 12)
cohort_sample = np.arange(Nc, Nc - 1200, -60) - 1

alpha_constraint = np.ones(
    (1, Nconstraint)) * density
alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
cohort_type_size_mix = cohort_size * alpha_i_mix
# cohort_size_mix = np.sum(cohort_type_size_mix, axis=0)
beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio
rho_cohort_type_mix = alpha_i_mix * beta_i_mix * np.exp(
    -(rho_i_mix + nu) * tau)  # shape(2, 6000)

np.seterr(invalid='ignore')


def simulate_path(
        i: int,
):
    print(i)

    dZ_build = dZ_build_matrix[i]
    dZ = dZ_matrix[i]

    if np.mod(i, 10) == 0:
        # reentry_time_compare = np.zeros((14, 2, Nt - int(window_bell / dt) - 12), dtype=np.int8)
        # exit_time_compare = np.zeros((14, 2, Nt - int(window_bell / dt)), dtype=np.int8)
        need_invest_matrix = 'True'
    else:
        # reentry_time_compare = 0
        # exit_time_compare = 0
        need_invest_matrix = 'False'
    # need_invest_matrix = 'False'
    # reentry_time_compare = 0
    # reentry_time_compare = np.zeros((n_scenarios - 1, 14, 2, Nt - int(window_bell / dt) - 12), dtype=np.int8)
    # need_invest_matrix = 'True'

    (
        table_mean_vola,
        table_parti,
        table_parti_cov,
        reentry_time,
        regression_table1_b,
        regression_table2_b,
        Delta_diff,
    ) = simulate_mix_mean_vola(
        Nc,
        Nt,
        dt,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        tax,
        beta0,
        phi,
        Npre,
        Ninit,
        T_hat,
        entry_bound,
        exit_bound,
        dZ_build,
        dZ,
        Ntype,
        Nconstraint,
        rho_i_mix,
        alpha_i_mix,
        beta_i_mix,
        rho_cohort_type_mix,
        cohort_type_size_mix,
        window_bell,
        need_invest_matrix
    )

    if need_invest_matrix == 'True':
        np.save(r'simu_results/' + str(i) + 'reentry_time', reentry_time[:, 2:])
    # np.save(folder_address + str(i) + str(phi_i) + 'parti_age', parti_age_compare)

    return (
        i,
        table_mean_vola,
        table_parti,
        table_parti_cov,
        regression_table1_b,
        regression_table2_b,
        Delta_diff,
    )


def main():
    # Create a ProcessPoolExecutor for parallel execution
    for j in range(int(Mpath / N_workers)):
        per_path = N_workers
        paths_j = j * per_path

        with ProcessPoolExecutor(max_workers=N_workers) as executor:  # Adjust the number of workers as needed
            results = [executor.submit(simulate_path, i) for i in range(paths_j, paths_j + per_path)]
        # Initialize a list to store the results
        results_list = []

        # Retrieve results from parallel processes
        for result in results:
            i, \
            table_mean_vola, \
            table_parti, \
            table_parti_cov, \
            regression_table1_b, \
            regression_table2_b,\
            Delta_diff, = result.result()

            data = {
                "i": i,
                "table_mean_vola": table_mean_vola,
                "table_parti": table_parti,
                "table_parti_cov": table_parti_cov,
                "regression_table1_b": regression_table1_b,
                "regression_table2_b": regression_table2_b,
                "Delta_diff": Delta_diff
            }
            results_list.append(data)

        # Create a DataFrame from the list of dictionaries
        results_df = pd.DataFrame(results_list)
        results_dict = results_df.to_dict(orient='list')
        np.savez(r'simu_results/' + str(j) + "simulation_new.npz", **results_dict)


if __name__ == '__main__':
    main()

