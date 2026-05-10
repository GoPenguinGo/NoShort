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

# mode_trade = "w_constraint"
# mode_learn = 'reentry'
#
#
# def simulate_path(
#         i: int,
#         rho_i,
#         nu,
#         tax,
#         T_hat,
# ):
#     print(i)
#     # shocks
#     dZ_build = dZ_build_matrix[i]
#     dZ = dZ_matrix[i]
#     dZ_SI_build = dZ_SI_build_matrix[i]
#     dZ_SI = dZ_SI_matrix[i]
#
#     Npre = int(T_hat / dt)
#     Vhat = (sigma_Y ** 2) / T_hat  # prior variance
#
#     beta_i = (nu + rho_i) / (1 + tax)  # consumption wealth ratio
#     beta0 = np.sum(alpha_i * beta_i).astype(float)
#     rho_cohort_type = alpha_i * beta_i * np.exp(-(rho_i + nu) * tau)  # shape(2, 6000)
#
#     (
#         r,
#         theta,
#         f_c,
#         Delta,
#         pi,
#         parti,
#         Phi_bar_parti,
#         Phi_tilde_parti,
#         Delta_bar_parti,
#         Delta_tilde_parti,
#         dR,
#         mu_S,
#         sigma_S,
#         beta,
#         invest_tracker,
#         parti_age_group,
#         parti_wealth_group,
#         entry_mat,
#         exit_mat
#     ) = simulate_SI(mode_trade,
#                     mode_learn,
#                     Nc,
#                     Nt,
#                     dt,
#                     nu,
#                     Vhat,
#                     mu_Y,
#                     sigma_Y,
#                     tax,
#                     beta0,
#                     phi,
#                     Npre,
#                     Ninit,
#                     T_hat,
#                     dZ_build,
#                     dZ,
#                     dZ_SI_build,
#                     dZ_SI,
#                     tau,
#                     cutoffs_age,
#                     Ntype,
#                     rho_i,
#                     alpha_i,
#                     beta_i,
#                     rho_cohort_type,
#                     cohort_type_size,
#                     need_f='False',
#                     need_Delta='False',
#                     need_pi='False',
#                     )
#     past_annual_return = np.zeros(Nt, dtype=np.float32)
#     gap = 12
#     past_annual_return[gap:] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
#
#     # run regressions and save results instead of saving the data:
#     x = past_annual_return[sample]
#     y = parti[sample]
#     x_regress = sm.add_constant(x)
#     model = sm.OLS(y, x_regress)
#     est = model.fit()
#     est_b = est.params[1]
#     np.save(folder_address + str(i) + ".npy", est_b)
#     return (
#         i,
#         est_b
#     )
#
#
# def main():
#     # Create a ProcessPoolExecutor for parallel execution
#     parti_df = pd.DataFrame()
#     for T_hat in T_hat_set:
#         for rho_i in rho_i_set:
#             for nu in nu_set:
#                 for tax in tax_set:
#                     with ProcessPoolExecutor(max_workers=25) as executor:  # Adjust the number of workers as needed
#                         col_name = str(T_hat) + str(int(rho_i[1, 0] * 1000)) + str(int(nu * 1000)) + str(int(tax * 100))
#                         results = [executor.submit(simulate_path, i,
#                                                    rho_i,
#                                                    nu,
#                                                    tax,
#                                                    T_hat, ) for i in range(Mpath)]
#                         # Initialize a list to store the results
#                         results_list = []
#                         # Retrieve results from parallel processes
#                         for result in results:
#                             i, regression_table1 = result.result()
#
#                             data = {
#                                 "i": i,
#                                 "return_parti_reg": regression_table1,
#                             }
#                             results_list.append(data)
#
#                         # Create a DataFrame from the list of dictionaries
#                         results_df = pd.DataFrame(results_list)
#
#                         # Save the DataFrame to a .npz file
#                         results_dict = results_df.to_dict(orient='list')
#                         np.savez(col_name+".npz", **results_dict)
#
#                         reg_results1 = np.empty((100, 1))
#                         for i in range(100):
#                             reg_results1[i] = np.load(folder_address + str(i) + ".npy")
#                         ave_reg1 = np.average(reg_results1, axis=0)
#                         parti_df[col_name] = ave_reg1.astype(np.float32)
#     parti_df.to_stata('stata_dataset/all_corr.dta')
