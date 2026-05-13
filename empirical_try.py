from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.param import mu_Y, sigma_Y, \
    dt, Ninit, Nc, Nt, tau, \
    Ntype, alpha_i, \
    dZ_build_matrix, dZ_matrix, \
    cohort_size, cutoffs_age, rho_i, nu, tax, exit_bound, entry_bound, T_hat, phi
from src.param_mix import Nconstraint, density
from src.simulation import simulate_mix_types


country_names = [
    'US',
    'Finland',
    'Germany',
    'Norway'
]

plt.rcParams["font.family"] = 'serif'

# sensitivities:
# exit_bound, entry_bound, T_hat, phi, density
total_param = 5
vary_param = 2
diff_param = [0.01, 0.01, 1, 0.1, 0.05]
column_param = ['exitbound', 'entrybound', 'That', 'phi', 'density']
column_vary = ['up', 'down']


Mpath = 10
np.seterr(invalid='ignore')


# noinspection PyTypeChecker
def simulate_path(
        i: int,
        data_shocks,
        country: str,
        entry_bound_i,
):
    print(i)
    # shocks
    dZ_build = dZ_build_matrix[i]
    dZ = dZ_matrix[i]
    dZ_actual = data_shocks.to_numpy()[:, 0]
    Nt_data = dZ_actual.size
    dZ[-Nt_data:] = dZ_actual
    parti_df = pd.DataFrame(data_shocks.index.astype(str), columns=['yyyymm'])

    params = [exit_bound, entry_bound_i, T_hat, phi, density]
    exit_bound_use, entry_bound_use, T_hat_use, phi_use, density_use = params

    Npre = int(T_hat_use / dt)
    Vhat = (sigma_Y ** 2) / T_hat_use  # prior variance

    beta_i = (nu + rho_i) / (1 + tax)  # consumption wealth ratio
    beta0 = np.sum(alpha_i * beta_i).astype(float)
    alpha_constraint = np.ones(
        (1, Nconstraint)) * density_use
    alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
    cohort_type_size_mix = cohort_size * alpha_i_mix

    rho_i_mix = np.tile(np.reshape(rho_i, (-1, 1, 1)), (1, Nconstraint, 1))
    beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio
    rho_cohort_type_mix = alpha_i_mix * beta_i_mix * np.exp(
        -(rho_i_mix + nu) * tau)  # shape(2, 6000)

    col_name = f'baseline'

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
    ) = simulate_mix_types(Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax, beta0,
                           phi_use,
                           Npre, Ninit,
                           T_hat_use,
                           entry_bound_use,
                           exit_bound_use,
                           dZ_build, dZ,
                           cutoffs_age, Ntype,
                           Nconstraint, rho_i_mix, alpha_i_mix, beta_i_mix,
                           rho_cohort_type_mix,
                           cohort_type_size_mix,
                           need_f='False',
                           need_Delta='True',
                           need_pi='True',
                           mode_learn='invest',
                           )

    parti_df['parti' + col_name] = parti[-Nt_data:].astype(np.float32)
    parti_df['entry' + col_name] = entry_mat[-Nt_data:, 0].astype(np.float32)
    parti_df['exit' + col_name] = exit_mat[-Nt_data:, 0].astype(np.float32)

    if country == "US":
        age_belief = np.zeros((len(cutoffs_age) - 1, Nt_data))
        for n in range(len(cutoffs_age) - 1):
            age_belief[n] = np.average(
                np.average(Delta[-Nt_data:, :, cutoffs_age[n + 1]:cutoffs_age[n]],
                           weights=cohort_size[0, cutoffs_age[n + 1]:cutoffs_age[n]],
                           axis=2),
                weights=density_use,
                axis=1)
        parti_df_Michigan = pd.DataFrame(data_shocks.index.astype(str), columns=['yyyymm'])
        parti_df_Michigan['belief_old'] = age_belief[-2].astype(np.float32)
        parti_df_Michigan['belief_young'] = age_belief[0].astype(np.float32)
        parti_df_Michigan['parti_old'] = parti_age_group[-Nt_data:, -2].astype(np.float32)
        parti_df_Michigan['parti_young'] = parti_age_group[-Nt_data:, 0].astype(np.float32)
        parti_df_Michigan.to_stata(f'stata_dataset/{country}/{i}_Michigan.dta')

    for ii in range(total_param):
        for jj in range(vary_param):
            diff_n = diff_param[ii] if jj == 0 else -diff_param[ii]
            if ii == 0:
                params = [exit_bound + diff_n, entry_bound_i, T_hat, phi, density]
            elif ii == 1:
                params = [exit_bound, entry_bound_i + diff_n, T_hat, phi, density]
            elif ii == 2:
                params = [exit_bound, entry_bound_i, T_hat + diff_n, phi, density]
            elif ii == 3:
                params = [exit_bound, entry_bound_i, T_hat, phi + diff_n, density]
            else:
                diff_density = (diff_n, 0, -diff_n)
                params = [exit_bound, entry_bound_i, T_hat, phi, density + diff_density]
            exit_bound_use, entry_bound_use, T_hat_use, phi_use, density_use = params

            if entry_bound_use >= exit_bound_use:
                Npre = int(T_hat_use / dt)
                Vhat = (sigma_Y ** 2) / T_hat_use  # prior variance

                beta_i = (nu + rho_i) / (1 + tax)  # consumption wealth ratio
                beta0 = np.sum(alpha_i * beta_i).astype(float)
                alpha_constraint = np.ones(
                    (1, Nconstraint)) * density_use
                alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
                cohort_type_size_mix = cohort_size * alpha_i_mix

                rho_i_mix = np.tile(np.reshape(rho_i, (-1, 1, 1)), (1, Nconstraint, 1))
                beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio
                rho_cohort_type_mix = alpha_i_mix * beta_i_mix * np.exp(
                    -(rho_i_mix + nu) * tau)  # shape(2, 6000)

                col_name = f'{column_param[ii]}_{column_vary[jj]}'

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
                ) = simulate_mix_types(Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax, beta0,
                                       phi_use,
                                       Npre, Ninit,
                                       T_hat_use,
                                       entry_bound_use,
                                       exit_bound_use,
                                       dZ_build, dZ,
                                       cutoffs_age, Ntype,
                                       Nconstraint, rho_i_mix, alpha_i_mix, beta_i_mix,
                                       rho_cohort_type_mix,
                                       cohort_type_size_mix,
                                       need_f='False',
                                       need_Delta='True',
                                       need_pi='True',
                                       mode_learn='invest',
                                       )

                parti_df['parti' + col_name] = parti[-Nt_data:].astype(np.float32)
                parti_df['entry' + col_name] = entry_mat[-Nt_data:, 0].astype(np.float32)
                parti_df['exit' + col_name] = exit_mat[-Nt_data:, 0].astype(np.float32)

    parti_df.to_stata(f'stata_dataset/{country}/{i}.dta')

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
        entry_bound_i = entry_bound if country != "Finland" else 0.01
        with ProcessPoolExecutor(max_workers=20) as executor:  # Adjust the number of workers as needed
            results = [executor.submit(simulate_path, i, data_shocks, country, entry_bound_i) for i in range(Mpath)]
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
