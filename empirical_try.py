from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.param import mu_Y, sigma_Y, \
    dt, Ninit, Nc, Nt, tau, \
    Ntype, alpha_i, \
    dZ_build_matrix, dZ_matrix, \
    cohort_size, cutoffs_age
from src.param_mix import Nconstraint
from src.simulation import simulate_mix_types


country_names = [
    'US',
    'Finland',
    'Germany',
    'Norway'
]

plt.rcParams["font.family"] = 'serif'

# (complete, excluded, reentry)
density_types_set = [
    # (0.3, 0.3, 0.4),*
    # (0.4, 0.3, 0.3),
    # (0.5, 0.3, 0.2),
    # (0.6, 0.3, 0.1),
    # (0.2, 0.4, 0.4)*,
    # (0.3, 0.4, 0.3),
    # (0.4, 0.4, 0.2),
    # (0.5, 0.4, 0.1),
    (0.25, 0.5, 0.25),
    (0.2, 0.5, 0.3),
    (0.3, 0.5, 0.2), #norway
    # (0.4, 0.5, 0.1), #Germany
]
# T_hat_set = [
#     # 2,
#     # 3,
#     # 4,
#     5,
#     # 10,
# ]
T_hat = 5
rho_i = np.array([[0.001], [0.005]])
# rho_i = np.array([[0.01], [0.01]])
nu = 0.02
# nu = 0.05
tax = 0.35
# tax_set = [
#     0.35,
#     # 0.3,
#     # 0.25
# ]

# phi_set = [
#     # 0.0,
#     # 0.2,
#     # 0.3,
#     0.4,
#     0.5,
#     0.6,
#     # 0.7,
# ]

phi = 0.5

entry_boundary_set = [
    # 0.0,
    0.01,
    0.02,
    0.03,
    0.04,
    # 0.05,
    # 0.06,
    # 0.07,
]

exit_bound_set = [
    # 0.0,
    0.01,
    0.02,
    0.03,
    0.04,
    # 0.05,
    # 0.06,
    # 0.07,
]

mode_learn_set = [
    # 'theta',
    'invest'
]


n_entry_boundary = len(entry_boundary_set)

Mpath = 10
np.seterr(invalid='ignore')


# noinspection PyTypeChecker
def simulate_path(
        i: int,
        data_shocks,
        country: str,
):
    print(i)
    # shocks
    dZ_build = dZ_build_matrix[i]
    dZ = dZ_matrix[i]
    dZ_actual = data_shocks.to_numpy()[:, 0]
    Nt_data = dZ_actual.size
    dZ[-Nt_data:] = dZ_actual
    parti_df = pd.DataFrame(data_shocks.index.astype(str), columns=['yyyymm'])
    if country == "US":
        parti_df_Michigan = pd.DataFrame(data_shocks.index.astype(str), columns=['yyyymm'])
    for mode in mode_learn_set:
        for density_n, density_types in enumerate(density_types_set):
            for entry_bound in entry_boundary_set:
                for exit_bound in exit_bound_set:

                    if entry_bound >= exit_bound:
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

                        col_name = f'{int(density_n)}_{int(entry_bound * 100)}_{int(exit_bound * 100)}'

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
                            parti_age_group,
                            # Delta_popu,
                            # portf_age_group,
                            entry_mat,
                            exit_mat
                        ) = simulate_mix_types(Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax,
                                               beta0,
                                               phi, Npre, Ninit, T_hat,
                                               entry_bound,
                                               exit_bound,
                                               dZ_build, dZ,
                                               age_cutoffs, Ntype,
                                               Nconstraint, rho_i_mix, alpha_i_mix, beta_i_mix,
                                               rho_cohort_type_mix,
                                               cohort_type_size_mix,
                                               need_f='False',
                                               need_Delta='True',
                                               need_pi='True',
                                               mode_learn=mode,
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
                        # parti_df['pd' + col_name] = (1 / beta[-Nt_data:]).astype(np.float32)
                        # parti_df['vola' + col_name] = sigma_S[-Nt_data:].astype(np.float32)
                        parti_df.to_stata(f'stata_dataset/{country}/{i}_phi0.dta')

                        if country == "US":
                            age_belief = np.zeros((len(cutoffs_age) - 1, Nt_data))
                            for n in range(len(cutoffs_age) - 1):
                                age_belief[n] = np.average(
                                    np.average(Delta[-Nt_data:, :, cutoffs_age[n + 1]:cutoffs_age[n]],
                                               weights=cohort_size[0, cutoffs_age[n + 1]:cutoffs_age[n]],
                                               axis=2),
                                    weights=density_types,
                                    axis=1)
                            parti_df_Michigan['belief_old' + col_name] = age_belief[-2].astype(np.float32)
                            parti_df_Michigan['belief_young' + col_name] = age_belief[0].astype(np.float32)
                            parti_df_Michigan['parti_old' + col_name] = parti_age_group[-Nt_data:, -2].astype(np.float32)
                            parti_df_Michigan['parti_young' + col_name] = parti_age_group[-Nt_data:, 0].astype(np.float32)
                            parti_df_Michigan.to_stata(f'stata_dataset/{country}/{i}_Michigan.dta')

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
            f'empirical/realized_shocks_{country}.xlsx',
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
