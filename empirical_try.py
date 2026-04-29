import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_mix_types
from src.param import mu_Y, sigma_Y, \
    dt, Ninit, Nc, Nt, tau, \
    cutoffs_age, Ntype, alpha_i, \
    dZ_build_matrix, dZ_SI_build_matrix, dZ_SI_matrix, dZ_matrix, \
    cohort_size
# from src.param_mix import rho_i_mix
from concurrent.futures import ProcessPoolExecutor
# from src.param import rho_i, beta_i, beta0, rho_cohort_type, beta_cohort
from src.param_mix import Nconstraint
# from cupyx.scipy.interpolate import RBFInterpolator

country_names = [
    # 'US',
    # 'Finland',
    # 'Germany',
    'Norway'
]
folder_address = r'C:\Users\zeshu\BI Norwegian Business School Dropbox\Zeshu XU\to sync\between_computers\entry_exit/empirical/'
# folder_address = r'C:/Users\A2010290\OneDrive - BI Norwegian Business School (BIEDU)/Documents\GitHub computer 2/NoShort/empirical/'
plt.rcParams["font.family"] = 'serif'

# (complete, excluded, reentry)
density_types_set = [
    # (0.3, 0.3, 0.4),*
    # (0.4, 0.3, 0.3),
    # (0.5, 0.3, 0.2),
    # (0.6, 0.3, 0.1),
    # (0.2, 0.4, 0.4)*,
    # (0.3, 0.4, 0.3),
    # (0.4, 0.4, 0.2),*
    # (0.5, 0.4, 0.1),
    # (0.1, 0.5, 0.4),
    # (0.2, 0.5, 0.3),
    (0.3, 0.5, 0.2), #norway
    # (0.4, 0.5, 0.1), #Germany
]
T_hat_set = [
    # 2,
    # 3,
    # 4,
    5,
    # 10,
]
rho_i = np.array([[0.001], [0.005]])
# rho_i = np.array([[0.01], [0.01]])
nu = 0.02
# nu = 0.05
tax = 0.35
tax_set = [
    0.35,
    # 0.3,
    # 0.25
]

phi_set = [
    # 0.0,
    # 0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    # 0.7,
]

# phi = 0.5

entry_boundary_set = [
    # 0.0,
    0.01,
    0.02,
    0.03,
    # 0.04,
    # 0.05,
    # 0.06,
    # 0.07,
]

exit_bound_set = [
    0.0,
    0.01,
    # 0.02,
    # 0.03,
]


n_entry_boundary = len(entry_boundary_set)

Mpath = 10
np.seterr(invalid='ignore')
age_cutoffs = [int(Nt-1), int(Nt-1-12*20), int(Nt-1-12*40), 0]


# noinspection PyTypeChecker
def simulate_path(
        i: int,
        data_shocks,
        country: str,
):
    print(i)
    # shocks
    dZ_build = dZ_build_matrix[i]
    dZ_SI_build = dZ_SI_build_matrix[i]
    dZ = dZ_matrix[i]
    dZ_SI = dZ_SI_matrix[i]
    filler = np.random.randn(data_shocks['dZ_SI'].isna().sum()) * np.sqrt(dt)
    data_shocks.loc[data_shocks['dZ_SI'].isna(), 'dZ_SI'] = filler
    dZ_actual = data_shocks.to_numpy()[:, 0]
    dZ_SI_actual = data_shocks.to_numpy()[:, 1]
    Nt_data = dZ_actual.size
    dZ[-Nt_data:] = dZ_actual
    dZ_SI[-Nt_data:] = dZ_SI_actual
    parti_df = pd.DataFrame(data_shocks.index.astype(str), columns=['yyyymm'])
    for T_hat in T_hat_set:
        for density_n, density_types in enumerate(density_types_set):
            for phi in phi_set:
                for entry_bound in entry_boundary_set:
                    for exit_bound in exit_bound_set:
                        # exit_bound = np.copy(entry_bound)
                        Npre = int(T_hat / dt)
                        Vhat = (sigma_Y ** 2) / T_hat  # prior variance

                        beta_i = (nu + rho_i) / (1 + tax)  # consumption wealth ratio
                        beta0 = np.sum(alpha_i * beta_i).astype(float)
                        alpha_constraint = np.ones(
                            (1, Nconstraint)) * density_types
                        alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
                        cohort_type_size_mix = cohort_size * alpha_i_mix

                        rho_i_mix = np.tile(np.reshape(rho_i, (-1, 1, 1)), (1, Nconstraint, 1))
                        beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio
                        rho_cohort_type_mix = alpha_i_mix * beta_i_mix * np.exp(
                            -(rho_i_mix + nu) * tau)  # shape(2, 6000)

                        col_name = f'{int(T_hat)}_{int(phi * 10)}_{int(density_n)}_{int(entry_bound * 100)}_{int(exit_bound * 100)}'

                        (
                            r,
                            theta,
                            f_c,
                            Delta,
                            pi,
                            parti,
                            Phi_bar_parti,
                            Phi_tilde_parti,
                            Delta_bar_parti,
                            Delta_tilde_parti,
                            dR,
                            mu_S,
                            sigma_S,
                            beta,
                            invest_tracker,
                            parti_age_group,
                            portf_age_group,
                            entry_mat,
                            exit_mat
                        ) = simulate_mix_types(Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax,
                                               beta0,
                                               phi, Npre, Ninit, T_hat,
                                               entry_bound,
                                               exit_bound,
                                               dZ_build, dZ, dZ_SI_build, dZ_SI,
                                               age_cutoffs, Ntype,
                                               Nconstraint, rho_i_mix, alpha_i_mix, beta_i_mix,
                                               rho_cohort_type_mix,
                                               cohort_type_size_mix,
                                               need_f='False',
                                               need_Delta='True',
                                               need_pi='True',
                                               )

                        parti_df['parti' + col_name] = parti[-Nt_data:].astype(np.float32)
                        # age_belief = np.zeros((len(age_cutoffs) - 1, Nt_data))
                        # for n in range(len(age_cutoffs) - 1):
                        #     age_belief[n] = np.average(
                        #         np.average(Delta[-Nt_data:, :, age_cutoffs[n + 1]:age_cutoffs[n]],
                        #                    weights=cohort_size[0, age_cutoffs[n + 1]:age_cutoffs[n]],
                        #                    axis=2),
                        #         weights=density_types,
                        #         axis=1)
                        # parti_df['belief_old' + col_name] = age_belief[-1].astype(np.float32)
                        # parti_df['belief_young' + col_name] = age_belief[0].astype(np.float32)
                        # parti_df['parti_old' + col_name] = parti_age_group[-Nt_data:, -1].astype(np.float32)
                        # parti_df['parti_young' + col_name] = parti_age_group[-Nt_data:, 0].astype(np.float32)
                        # parti_df['portf_old' + col_name] = portf_age_group[-Nt_data:, -1].astype(np.float32)
                        # parti_df['portf_young' + col_name] = portf_age_group[-Nt_data:, 0].astype(np.float32)
                        parti_df['entry' + col_name] = entry_mat[-Nt_data:, 0].astype(np.float32)
                        parti_df['exit' + col_name] = exit_mat[-Nt_data:, 0].astype(np.float32)
                        parti_df.to_stata(f'stata_dataset/{country}/{i}_phi1.dta')

    return (
        i,
        # popu_parti_compare,
        # dR_compare,
    )


def main():
    for country in country_names:
        # Create a ProcessPoolExecutor for parallel execution
        # run this on a grid of parameters & type densities & signal
        data_shocks = pd.read_excel(
            f'{folder_address}realized_shocks_{country}.xlsx',
            sheet_name='Sheet1',
            index_col=0
        )
        with ProcessPoolExecutor(max_workers=20) as executor:  # Adjust the number of workers as needed
            results = [executor.submit(simulate_path, i, data_shocks, country) for i in range(Mpath)]
        # Initialize a list to store the results
        results_list = []

        # Retrieve results from parallel processes
        for result in results:
            i = result.result()

            data = {
                "i": i,
            }
            results_list.append(data)

        # Create a DataFrame from the list of dictionaries
        # results_df = pd.DataFrame(results_list)

if __name__ == '__main__':
    main()



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from src.simulation import simulate_mix_types
# from src.param import mu_Y, sigma_Y, \
#     dt, Ninit, Nc, Nt, tau, \
#     cutoffs_age, Ntype, alpha_i, \
#     dZ_build_matrix, dZ_SI_build_matrix, dZ_SI_matrix, dZ_matrix, \
#     cohort_size
# # from src.param_mix import rho_i_mix
# from concurrent.futures import ProcessPoolExecutor
# # from src.param import rho_i, beta_i, beta0, rho_cohort_type, beta_cohort
# from src.param_mix import Nconstraint
# # from cupyx.scipy.interpolate import RBFInterpolator
#
# country_names = [
#     'US',
#     'Finland',
#     'Germany',
#     'Norway'
# ]
# folder_address = r'E:\Users\A2010290\Documents\GitHub\NoShort/empirical/'
# # folder_address = r'C:/Users\A2010290\OneDrive - BI Norwegian Business School (BIEDU)/Documents\GitHub computer 2/NoShort/empirical/'
# plt.rcParams["font.family"] = 'serif'
#
# # (complete, excluded, disappointment, reentry)
# density_types_set = [
#     (0.25, 0.25, 0.25, 0.25),
#     # (0.25, 0.25, 0.3, 0.2),
#     # (0.25, 0.25, 0.2, 0.3),
#     # (0.25, 0.2, 0.3, 0.25),
#     # (0.25, 0.3, 0.2, 0.25),
#     # (0.25, 0.2, 0.25, 0.3),
#     # (0.25, 0.3, 0.25, 0.2),
#     # (0.2, 0.3, 0.25, 0.25),
#     # (0.3, 0.2, 0.25, 0.25),
#     # (0.2, 0.25, 0.3, 0.25),
#     # (0.3, 0.25, 0.2, 0.25),
#     # (0.2, 0.25, 0.25, 0.3),
#     # (0.3, 0.25, 0.25, 0.2),
# ]
# T_hat_set = [
#     2,
#     5,
#     10,
# ]
# rho_i = np.array([[0.001], [0.005]])
# # rho_i = np.array([[0.01], [0.01]])
# nu = 0.02
# tax = 0.35
#
# phi_set = [0.0, 0.5]
#
# Mpath = 30
# np.seterr(invalid='ignore')
# age_cutoffs_SCF = [int(Nt-1), int(Nt-1-12*15), int(Nt-1-12*35), int(Nt-1-12*55), 0]
#
#
# # noinspection PyTypeChecker
# def simulate_path(
#         i: int,
#         data_shocks,
#         country: str,
# ):
#     print(i)
#     # shocks
#     dZ_build = dZ_build_matrix[i]
#     dZ_SI_build = dZ_SI_build_matrix[i]
#     dZ = dZ_matrix[i]
#     filler = np.random.randn(data_shocks['dZ_SI'].isna().sum()) * np.sqrt(dt)
#     data_shocks.loc[data_shocks['dZ_SI'].isna(), 'dZ_SI'] = filler
#     dZ_actual = data_shocks.to_numpy()[:, 0]
#     dZ_SI_actual = data_shocks.to_numpy()[:, 1]
#     Nt_data = dZ_actual.size
#     dZ[-Nt_data:] = dZ_actual
#     dZ_SI = dZ_SI_matrix[i]
#     dZ_SI[-Nt_data:] = dZ_SI_actual
#     parti_df = pd.DataFrame(data_shocks.index.astype(str), columns=['yyyymm'])
#     for T_hat in T_hat_set:
#         for density_n, density_types in enumerate(density_types_set):
#             for phi in phi_set:
#                 Npre = int(T_hat / dt)
#                 Vhat = (sigma_Y ** 2) / T_hat  # prior variance
#
#                 beta_i = (nu + rho_i) / (1 + tax)  # consumption wealth ratio
#                 beta0 = np.sum(alpha_i * beta_i).astype(float)
#                 alpha_constraint = np.ones(
#                     (1, Nconstraint)) * density_types
#                 alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
#                 cohort_type_size_mix = cohort_size * alpha_i_mix
#
#                 rho_i_mix = np.tile(np.reshape(rho_i, (-1, 1, 1)), (1, Nconstraint, 1))
#                 beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio
#                 rho_cohort_type_mix = alpha_i_mix * beta_i_mix * np.exp(
#                     -(rho_i_mix + nu) * tau)  # shape(2, 6000)
#
#                 need_Delta_country = 'True' if country == 'US' else 'False'
#                 need_pi_country = 'True' if country == 'US' else 'False'
#
#                 col_name = f'{int(T_hat)}_{int(phi * 10)}_{int(density_n)}'
#
#                 (
#                     r,
#                     theta,
#                     f_c,
#                     Delta,
#                     pi,
#                     parti,
#                     Phi_bar_parti,
#                     Phi_tilde_parti,
#                     Delta_bar_parti,
#                     Delta_tilde_parti,
#                     dR,
#                     mu_S,
#                     sigma_S,
#                     beta,
#                     invest_tracker,
#                     parti_age_group,
#                     portf_age_group,
#                     entry_mat,
#                     exit_mat
#                 ) = simulate_mix_types(Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax,
#                                        beta0,
#                                        phi, Npre, Ninit, T_hat,
#                                        dZ_build, dZ, dZ_SI_build, dZ_SI,
#                                        age_cutoffs_SCF, Ntype,
#                                        Nconstraint, rho_i_mix, alpha_i_mix, beta_i_mix,
#                                        rho_cohort_type_mix,
#                                        cohort_type_size_mix,
#                                        need_f='False',
#                                        need_Delta=need_Delta_country,
#                                        need_pi=need_pi_country,
#                                        )
#
#                 parti_df['parti' + col_name] = parti[-Nt_data:].astype(np.float32)
#                 parti_df['beta' + col_name] = beta[-Nt_data:].astype(np.float32)
#                 if country == 'US':
#                     age_belief = np.zeros((4, Nt_data))
#                     for n in range(len(age_cutoffs_SCF) - 1):
#                         age_belief[n] = np.average(
#                             np.average(Delta[-Nt_data:, :, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]],
#                                        weights=cohort_size[0, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]],
#                                        axis=2),
#                             weights=density_types,
#                             axis=1)
#                     parti_df['belief_old' + col_name] = age_belief[3].astype(np.float32)
#                     parti_df['belief_young' + col_name] = age_belief[0].astype(np.float32)
#                     parti_df['parti_old' + col_name] = parti_age_group[-Nt_data:, 3].astype(np.float32)
#                     parti_df['parti_young' + col_name] = parti_age_group[-Nt_data:, 0].astype(np.float32)
#                     parti_df['portf_old' + col_name] = portf_age_group[-Nt_data:, 3].astype(np.float32)
#                     parti_df['portf_young' + col_name] = portf_age_group[-Nt_data:, 0].astype(np.float32)
#
#                     parti_dividend = np.zeros(Nt_data).astype(np.float32)
#                     for t in range(Nt_data):
#                         parti_dividend_t = np.zeros((12, 4, Nc - 12)).astype(np.float32)
#                         for tc in range(12):
#                             parti_dividend_t[tc] = pi[Nt - Nt_data + t - tc, :, 12 - tc: -tc] != 0 if tc != 0 \
#                                 else pi[Nt - Nt_data + t - tc, :, 12 - tc:] != 0
#                         parti_dividend[t] = np.average(np.sum(parti_dividend_t, axis=0) != 0,
#                                                        weights=cohort_type_size_mix[0, :, 12:])
#                     parti_df['parti_dividend' + col_name] = parti_dividend.astype(np.float32)
#
#                 if country == 'Finland' or country == 'Norway':
#                     parti_df['entry' + col_name] = entry_mat[-Nt_data:, 0].astype(np.float32)
#                     parti_df['exit' + col_name] = exit_mat[-Nt_data:, 0].astype(np.float32)
#                 parti_df.to_stata('stata_dataset/' + country + '/' + str(i) + '.dta')
#
#     return (
#         i,
#         # popu_parti_compare,
#         # dR_compare,
#     )
#
#
# def main():
#     for country in country_names:
#         # Create a ProcessPoolExecutor for parallel execution
#         # run this on a grid of parameters & type densities & signal
#         data_shocks = pd.read_excel(
#             folder_address + r'realized_shocks_' + country + '.xlsx',
#             sheet_name='Sheet1',
#             index_col=0
#         )
#         with ProcessPoolExecutor(max_workers=30) as executor:  # Adjust the number of workers as needed
#             results = [executor.submit(simulate_path, i, data_shocks, country) for i in range(Mpath)]
#         # Initialize a list to store the results
#         results_list = []
#
#         # Retrieve results from parallel processes
#         for result in results:
#             i = result.result()
#
#             data = {
#                 "i": i,
#             }
#             results_list.append(data)
#
#         # Create a DataFrame from the list of dictionaries
#         # results_df = pd.DataFrame(results_list)
#
# if __name__ == '__main__':
#     main()
