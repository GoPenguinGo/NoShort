import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_mix_types, simulate_SI
from src.param import mu_Y, sigma_Y, \
    dt, Ninit, Nc, Nt, tau, \
    cutoffs_age, Ntype, alpha_i, \
    dZ_build_matrix, dZ_SI_build_matrix, dZ_SI_matrix, dZ_matrix, \
    cohort_size, T_hat, rho_i, nu, tax, phi, Vhat, Npre, beta_i, beta0, \
    cohort_type_size, rho_cohort_type
from src.param_mix import rho_i_mix, beta_i_mix
from concurrent.futures import ProcessPoolExecutor
# from src.param import rho_i, beta_i, beta0, rho_cohort_type, beta_cohort
from src.param_mix import Nconstraint



# (complete, excluded, disappointment, reentry)
Mpath = 2000
np.seterr(invalid='ignore')
age_cutoffs = [int(Nt-1), int(Nt-1-12*20), int(Nt-1-12*40), 0]
exp_old = 50
exp_young = 20

data_sample = np.arange(int(exp_old / dt), Nt, 60)
t_sample = np.arange(int(200 / dt), Nt, 12)
c_sample = np.arange(-1, -Nc + 12, -12)


# noinspection PyTypeChecker
def simulate_path(
        i: int,
):
    print(i)
    # shocks
    dZ_build = dZ_build_matrix[i]
    dZ_SI_build = dZ_SI_build_matrix[i]
    dZ = dZ_matrix[i]
    dZ_SI = dZ_SI_matrix[i]

    cov_matrix = np.zeros((2, 3))
    cov_matrix2 = np.zeros((2, 5))

    for density_n in range(2):
        if density_n == 0:
            mode_trade = 'w_constraint'
            mode_learn = 'reentry'
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
            ) = simulate_SI(mode_trade,
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
                            age_cutoffs,
                            Ntype,
                            rho_i,
                            alpha_i,
                            beta_i,
                            rho_cohort_type,
                            cohort_type_size,
                            need_f='True',
                            need_Delta='True',
                            need_pi='True',
                            )

            portf_age_group[np.isnan(portf_age_group)] = 0

            age_belief = np.zeros((3, Nt))
            for n in range(len(age_cutoffs) - 1):
                age_belief[n] = np.average(Delta[:, age_cutoffs[n + 1]:age_cutoffs[n]],
                               weights=cohort_type_size[0, age_cutoffs[n + 1]:age_cutoffs[n]], axis=1)

            total_returns = np.cumsum(dR)
            experience_old = np.zeros(Nt)
            experience_old[exp_old * 12:] = total_returns[exp_old * 12:] - total_returns[:-exp_old * 12]
            experience_young = np.zeros(Nt)
            experience_young[exp_young * 12:] = total_returns[exp_young * 12:] - total_returns[:-exp_young * 12]

            experience_gap = (experience_old - experience_young)[data_sample]
            belief_gap = (age_belief[2] - age_belief[0])[data_sample]
            parti_gap = (parti_age_group[:, 2] - parti_age_group[:, 0])[data_sample]
            portf_gap = (portf_age_group[:, 2] - portf_age_group[:, 0])[data_sample]

            cov_matrix[density_n, 0] = np.corrcoef(experience_gap, belief_gap)[0, 1]
            cov_matrix[density_n, 1] = np.corrcoef(belief_gap, parti_gap)[0, 1]
            cov_matrix[density_n, 2] = np.corrcoef(belief_gap, portf_gap)[0, 1]

            invest = np.zeros((Nt, Nc))
            invest[np.where(pi>0)] = 1.0
            invest_annual = invest[t_sample]
            belief_annual = Delta[t_sample]
            cumu_returns_annual = total_returns[t_sample]
            del invest
            portf = pi * np.sum(f_c, axis=1)
            portf_annual = portf[t_sample]
            del portf
            del pi
            del f_c
            del Delta
            change_invest_annual = (invest_annual[1:, :-12] - invest_annual[:-1, 12:])[:, c_sample]
            change_belief_annual = (belief_annual[1:, :-12] - belief_annual[:-1, 12:])[:, c_sample]
            returns_annual = cumu_returns_annual[1:] - cumu_returns_annual[:-1]
            change_portf_annual = (portf_annual[1:, :-12] - portf_annual[:-1, 12:])[:, c_sample]

            num_samples_row = 2000
            num_samples_col = 300
            weight_ave = cohort_size[0, c_sample]
            row_index = np.random.choice(np.arange(weight_ave.shape[0]),
                                         p=weight_ave / np.sum(weight_ave),
                                         size=num_samples_row
                                         )
            col_index = np.random.choice(np.arange(change_portf_annual.shape[0]),
                                         size=num_samples_col
                                         )

            returns_annual_flat = np.reshape(
                np.tile(
                    np.reshape(
                        returns_annual[col_index], (-1, 1)),
                    (1, num_samples_row)
                ), (-1))

            cov_matrix2[density_n, 0] = np.corrcoef(
                np.reshape(change_belief_annual[col_index][:, row_index], (-1)),
                np.reshape(change_invest_annual[col_index][:, row_index], (-1))
            )[0, 1]
            cov_matrix2[density_n, 1] = np.corrcoef(
                np.reshape(change_belief_annual[col_index][:, row_index], (-1)),
                np.reshape(change_portf_annual[col_index][:, row_index], (-1))
            )[0, 1]
            cov_matrix2[density_n, 2] = np.corrcoef(
                returns_annual_flat,
                np.reshape(change_belief_annual[col_index][:, row_index], (-1)),
            )[0, 1]
            cov_matrix2[density_n, 3] = np.corrcoef(
                returns_annual_flat,
                np.reshape(change_invest_annual[col_index][:, row_index], (-1)),
            )[0, 1]
            cov_matrix2[density_n, 4] = np.corrcoef(
                returns_annual_flat,
                np.reshape(change_portf_annual[col_index][:, row_index], (-1)),
            )[0, 1]

        else:
            density_types = (0.25, 0.25, 0.25, 0.25)
            alpha_constraint = np.ones(
                (1, Nconstraint)) * density_types
            alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
            cohort_type_size_mix = cohort_size * alpha_i_mix
            rho_cohort_type_mix = alpha_i_mix * beta_i_mix * np.exp(
                -(rho_i_mix + nu) * tau)  # shape(2, 6000)

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
                                   dZ_build, dZ, dZ_SI_build, dZ_SI,
                                   age_cutoffs, Ntype,
                                   Nconstraint, rho_i_mix, alpha_i_mix, beta_i_mix,
                                   rho_cohort_type_mix,
                                   cohort_type_size_mix,
                                   need_f='True',
                                   need_Delta='True',
                                   need_pi='True',
                                   )

            age_belief = np.zeros((3, Nt))
            for n in range(len(age_cutoffs) - 1):
                age_belief[n] = np.average(
                    np.average(Delta[:, :, age_cutoffs[n + 1]:age_cutoffs[n]],
                               weights=cohort_size[0, age_cutoffs[n + 1]:age_cutoffs[n]],
                               axis=2),
                    weights=density_types,
                    axis=1)

            total_returns = np.cumsum(dR)
            experience_old = np.zeros(Nt)
            experience_old[exp_old * 12:] = total_returns[exp_old * 12:] - total_returns[:-exp_old * 12]
            experience_young = np.zeros(Nt)
            experience_young[exp_young * 12:] = total_returns[exp_young * 12:] - total_returns[:-exp_young * 12]

            experience_gap = (experience_old - experience_young)[data_sample]
            belief_gap = (age_belief[2] - age_belief[0])[data_sample]
            parti_gap = (parti_age_group[:, 2] - parti_age_group[:, 0])[data_sample]
            portf_gap = (portf_age_group[:, 2] - portf_age_group[:, 0])[data_sample]

            cov_matrix[density_n, 0] = np.corrcoef(experience_gap, belief_gap)[0, 1]
            cov_matrix[density_n, 1] = np.corrcoef(belief_gap, parti_gap)[0, 1]
            cov_matrix[density_n, 2] = np.corrcoef(belief_gap, portf_gap)[0, 1]

            invest = np.zeros((Nt, 4, Nc))
            invest[np.where(pi!=0)] = 1.0
            invest_annual = invest[t_sample]
            belief_annual = Delta[t_sample]
            cumu_returns_annual = total_returns[t_sample]
            del invest
            portf = pi * np.sum(f_c, axis=1)
            portf_annual = portf[t_sample]
            del portf
            del pi
            del f_c
            del Delta
            change_invest_annual = (invest_annual[1:, :, :-12] - invest_annual[:-1, :, 12:])[:, :, c_sample]
            change_belief_annual = (belief_annual[1:, :, :-12] - belief_annual[:-1, :, 12:])[:, :, c_sample]
            returns_annual = cumu_returns_annual[1:] - cumu_returns_annual[:-1]
            change_portf_annual = (portf_annual[1:, :, :-12] - portf_annual[:-1, :, 12:])[:, :, c_sample]

            num_samples_row = 1000
            num_samples_col = 300
            weight_ave = cohort_type_size_mix[0, 0, c_sample]
            row_index = np.random.choice(np.arange(weight_ave.shape[0]),
                                         p=weight_ave / np.sum(weight_ave),
                                         size=num_samples_row
                                         )
            col_index = np.random.choice(np.arange(change_portf_annual.shape[0]),
                                         size=num_samples_col
                                         )

            returns_annual_flat = np.reshape(
                np.tile(
                    np.reshape(
                        returns_annual[col_index], (-1, 1, 1)),
                    (1, 4, num_samples_row)
                ), (-1))

            cov_matrix2[density_n, 0] = np.corrcoef(
                np.reshape(change_belief_annual[col_index][:, :, row_index], (-1)),
                np.reshape(change_invest_annual[col_index][:, :, row_index], (-1))
            )[0, 1]
            cov_matrix2[density_n, 1] = np.corrcoef(
                np.reshape(change_belief_annual[col_index][:, :, row_index], (-1)),
                np.reshape(change_portf_annual[col_index][:, :, row_index], (-1))
            )[0, 1]
            cov_matrix2[density_n, 2] = np.corrcoef(
                returns_annual_flat,
                np.reshape(change_belief_annual[col_index][:, :, row_index], (-1)),
            )[0, 1]
            cov_matrix2[density_n, 3] = np.corrcoef(
                returns_annual_flat,
                np.reshape(change_invest_annual[col_index][:, :, row_index], (-1)),
            )[0, 1]
            cov_matrix2[density_n, 4] = np.corrcoef(
                returns_annual_flat,
                np.reshape(change_portf_annual[col_index][:, :, row_index], (-1)),
            )[0, 1]

    return (
        i,
        cov_matrix,
        cov_matrix2
    )


def main():
    with ProcessPoolExecutor(max_workers=30) as executor:  # Adjust the number of workers as needed
        results = [executor.submit(simulate_path, i) for i in range(Mpath)]
    # Initialize a list to store the results
    results_list = []

    # Retrieve results from parallel processes
    for result in results:
        i, \
        cov_matrix, \
        cov_matrix2, = result.result()
        data = {
            "i": i,
            "cov_matrix": cov_matrix,
            "cov_matrix2": cov_matrix2,
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a .npz file
    results_dict = results_df.to_dict(orient='list')
    np.savez("Michigan_model.npz", **results_dict)


if __name__ == '__main__':
    main()
