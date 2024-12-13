import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI_mean_vola, simulate_mix_mean_vola
from src.param import mu_Y, sigma_Y, \
    dt, Ninit, Nt, Nc, tau, tax, \
    cutoffs_age, Ntype, alpha_i, \
    dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    cohort_type_size, cohort_size, T_hat, Npre, Vhat
from src.param import phi
from src.param import nu, rho_i, beta_i, beta0, rho_cohort_type, beta_cohort
from src.param_mix import Nconstraint
from src.param_mix import rho_i_mix
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import statsmodels.api as sm

plt.rcParams["font.family"] = 'serif'
# (complete, excluded, disappointment, reentry)
density_set = [
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.25, 0.25, 0.25, 0.25),
]
n_scenarios = len(density_set)
# T_hat_set = [2, 3]
# rho_i_set = [
#     np.array([[0.001], [-0.003]]),
#     np.array([[0.001], [-0.001]]),
#     np.array([[0.001], [0.003]]),
#     np.array([[0.001], [0.005]]),
#
# ]
# # for testing:
# Mpath = 10
# Mpath = 10
window = 12  # 1-year non-overlapping windows
sample = np.arange(600, Nt - 600, window)
N_sample = len(sample)
# age_cut = 100
# Nc_cut = int(age_cut / dt)
age_sample = np.arange(1, int(200 / dt), 12)
cohort_sample = np.arange(Nc, Nc - 1200, -60) - 1
window_bell = 20

np.seterr(invalid='ignore')
# folder_address = r'E:\Users\A2010290\Documents\GitHub\NoShort/reg_results2/'


# folder_address = r'C:\Users\A2010290\OneDrive - BI Norwegian Business School (BIEDU)\Documents\GitHub computer 2\NoShort/reg_results2/'


def simulate_path(
        i: int,
):
    print(i)
    # shocks
    i = 0

    dZ_build = dZ_build_matrix[i]
    dZ = dZ_matrix[i]
    dZ_SI_build = dZ_SI_build_matrix[i]
    dZ_SI = dZ_SI_matrix[i]

    # sample: in time-series
    theta_compare = np.zeros(n_scenarios, dtype=np.float32)
    r_compare = np.zeros(n_scenarios, dtype=np.float32)
    mu_S_compare = np.zeros(n_scenarios, dtype=np.float32)
    sigma_S_compare = np.zeros(n_scenarios, dtype=np.float32)
    parti_compare = np.zeros((n_scenarios - 1, N_sample), dtype=np.float32)
    entry_compare = np.zeros(n_scenarios - 1, dtype=np.float32)
    exit_compare = np.zeros(n_scenarios - 1, dtype=np.float32)
    Delta_age_compare = np.zeros((2, len(age_sample)), dtype=np.float32)
    parti_age_compare = 0
    regression_table1 = np.zeros((n_scenarios - 1, 3, 3), dtype=np.float32)
    regression_table2 = np.zeros((n_scenarios - 1, 3, 3), dtype=np.float32)
    cov_compare = np.zeros((n_scenarios, 5), dtype=np.float32)
    if np.mod(i, 10) == 0:
        reentry_time_compare = np.zeros((n_scenarios - 1, 14, 2, Nt - int(window_bell / dt) - 12), dtype=np.int8)
        need_invest_matrix = 'True'
    else:
        reentry_time_compare = 0
        need_invest_matrix = 'False'

    for g, type_density in enumerate(density_set):
        if type_density[0] == 1:
            mode_trade = 'complete'
            mode_learn = 'reentry'
            (
                theta_ave,
                r_ave,
                mu_S_ave,
                sigma_S_ave,
                parti_age_ave,
                Delta_age_ave,
                reentry_time,
                entry_ave,
                exit_ave,
                cov_matrix,
                parti,
                regression_table1_b,
                regression_table2_b
            ) = simulate_SI_mean_vola(
                mode_trade,
                mode_learn,
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
                dZ_build,
                dZ,
                dZ_SI_build,
                dZ_SI,
                tau,
                Ntype,
                rho_i,
                alpha_i,
                beta_i,
                rho_cohort_type,
                cohort_type_size,
                need_invest_matrix
            )
            theta_compare[g] = theta_ave
            r_compare[g] = r_ave
            mu_S_compare[g] = mu_S_ave
            sigma_S_compare[g] = sigma_S_ave
            Delta_age_compare[g] = Delta_age_ave
            cov_compare[g] = cov_matrix

        elif type_density[3] == 1:
            mode_trade = 'w_constraint'
            mode_learn = 'reentry'

            (
                theta_ave,
                r_ave,
                mu_S_ave,
                sigma_S_ave,
                parti_age_ave,
                Delta_age_ave,
                reentry_time,
                entry_ave,
                exit_ave,
                cov_matrix,
                parti,
                regression_table1_b,
                regression_table2_b
            ) = simulate_SI_mean_vola(
                mode_trade,
                mode_learn,
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
                dZ_build,
                dZ,
                dZ_SI_build,
                dZ_SI,
                tau,
                Ntype,
                rho_i,
                alpha_i,
                beta_i,
                rho_cohort_type,
                cohort_type_size,
                need_invest_matrix
            )
            theta_compare[g] = theta_ave
            r_compare[g] = r_ave
            mu_S_compare[g] = mu_S_ave
            sigma_S_compare[g] = sigma_S_ave
            Delta_age_compare[g] = Delta_age_ave
            cov_compare[g] = cov_matrix
            parti_compare[g - 1] = parti
            entry_compare[g - 1] = entry_ave
            exit_compare[g - 1] = exit_ave
            parti_age_compare = parti_age_ave
            regression_table1[g - 1] = regression_table1_b
            regression_table2[g - 1] = regression_table2_b
            reentry_time_compare[g - 1] = reentry_time if need_invest_matrix == 'True' else 0

        else:
            alpha_constraint = np.ones(
                (1, Nconstraint)) * type_density
            alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
            cohort_type_size_mix = cohort_size * alpha_i_mix
            # cohort_size_mix = np.sum(cohort_type_size_mix, axis=0)
            beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio
            rho_cohort_type_mix = alpha_i_mix * beta_i_mix * np.exp(
                -(rho_i_mix + nu) * tau)  # shape(2, 6000)

            (
                theta_ave,
                r_ave,
                mu_S_ave,
                sigma_S_ave,
                reentry_time,
                entry_ave,
                exit_ave,
                cov_matrix,
                parti,
                regression_table1_b,
                regression_table2_b
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
                dZ_build,
                dZ,
                dZ_SI_build,
                dZ_SI,
                Ntype,
                Nconstraint,
                rho_i_mix,
                alpha_i_mix,
                beta_i_mix,
                rho_cohort_type_mix,
                cohort_type_size_mix,
                need_invest_matrix
            )

            theta_compare[g] = theta_ave
            r_compare[g] = r_ave
            mu_S_compare[g] = mu_S_ave
            sigma_S_compare[g] = sigma_S_ave
            cov_compare[g] = cov_matrix
            parti_compare[g - 1] = parti
            entry_compare[g - 1] = entry_ave
            exit_compare[g - 1] = exit_ave
            regression_table1[g - 1] = regression_table1_b
            regression_table2[g - 1] = regression_table2_b
            reentry_time_compare[g - 1] = reentry_time[:, 2:] if need_invest_matrix == 'True' else 0

    return (
        theta_compare,
        r_compare,
        mu_S_compare,
        sigma_S_compare,
        parti_compare,
        entry_compare,
        exit_compare,
        Delta_age_compare,
        parti_age_compare,
        regression_table1,
        regression_table2,
        cov_compare,
        reentry_time_compare,
    )


def main():
    # Create a ProcessPoolExecutor for parallel execution
    for j in range(1):
        per_path = 20
        paths_j = j * per_path
        with ProcessPoolExecutor(max_workers=10) as executor:  # Adjust the number of workers as needed
            results = [executor.submit(simulate_path, i) for i in range(paths_j, paths_j + per_path)]
        # Initialize a list to store the results
        results_list = []
        # Retrieve results from parallel processes
        for result in results:
            i, \
                theta_compare, \
                r_compare, \
                mu_S_compare, \
                sigma_S_compare, \
                parti_compare, \
                entry_compare, \
                exit_compare, \
                Delta_age_compare, \
                parti_age_compare, \
                regression_table1, \
                regression_table2, \
                cov_compare, \
                reentry_time_compare = result.result()
            data = {
                "i": i,
                "theta": theta_compare,
                "r": r_compare,
                "mu_S": mu_S_compare,
                "sigma_S": sigma_S_compare,
                "parti": parti_compare,
                "entry": entry_compare,
                "exit": exit_compare,
                "Delta_age": Delta_age_compare,
                "parti_age": parti_age_compare,
                "reg1": regression_table1,
                "reg2": regression_table2,
                "cov_mat": cov_compare,
                "reentry_time": reentry_time_compare
            }
            results_list.append(data)
        # Create a DataFrame from the list of dictionaries
        results_df = pd.DataFrame(results_list)
        results_dict = results_df.to_dict(orient='list')
        np.savez(str(j)+"simulation_new.npz", **results_dict)



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
