import numpy as np
from src.simulation import simulate_SI, simulate_SI_mean_vola
from src.param import rho, nu, mu_Y, sigma_Y, sigma_Y_sqr, v, tax, \
    dt, T_hat, Npre, Vhat, Ninit, T_cohort, Nt, Nc, tau, cohort_size, \
    cutoffs_age, Mpath, n_age_cutoffs, \
    scenarios, dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    Ntype, rho_i, alpha_i, beta_i, beta0, beta_cohort_type, cohort_type_size
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

# Define the simulate_scenario function as shown in the previous answer
Mpath = 10
np.seterr(invalid='ignore')


# noinspection PyTypeChecker
def simulate_path_fig4(i: int,
                       Nscenario=2,
                       Nphi=2):
    print(i)
    # Initialize results for the current Mpath
    # for fig 4:
    age_cut = 100
    Nc_cut = int(age_cut / dt)
    data_point = np.arange(0, Nc_cut, 15)
    invest_results = np.zeros((Nphi, len(data_point)), dtype=np.float32)
    Delta_results = np.zeros((Nscenario, Nphi, len(data_point)), dtype=np.float32)
    t_gap = int(2 / dt)  # 2-year window
    N_cut = int(Nc - t_gap)
    parti_pre_results = np.zeros((n_age_cutoffs, N_cut), dtype=np.float32)
    belief_pre_results = np.zeros((Nscenario, n_age_cutoffs, N_cut), dtype=np.float32)
    belief_post_results = np.zeros((Nscenario, n_age_cutoffs, N_cut), dtype=np.float32)

    phi_vector = np.array([0, 0.8])

    # shocks
    dZ_build = dZ_build_matrix[i]
    dZ = dZ_matrix[i]
    dZ_SI_build = dZ_SI_build_matrix[i]
    dZ_SI = dZ_SI_matrix[i]

    # dZ_build = np.random.randn(Nc) * dt_root
    # dZ = np.random.randn(Nt) * dt_root
    # dZ_SI_build = np.random.randn(Nc) * dt_root
    # dZ_SI = np.random.randn(Nt) * dt_root

    for g, scenario in enumerate(scenarios[:Nscenario]):
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        for h, phi in enumerate(phi_vector):
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

            Delta_results[g, h] = np.average(np.abs(Delta[:, 0]), axis=0)[-Nc_cut:][data_point]
            if g == 1:
                invest_results[h] = np.average(invest_tracker[:, 0], axis=0)[-Nc_cut:][data_point]
            if h == 1:
                for mm in range(n_age_cutoffs):
                    age_bottom = cutoffs_age[mm + 1] if mm <= 2 else -N_cut
                    age_top = cutoffs_age[mm]
                    weights_group = cohort_type_size[0, age_bottom:age_top]
                    belief_pre_results[g, mm] = np.average(Delta[:-t_gap, 0, age_bottom:age_top],
                                                           weights=weights_group,
                                                           axis=1)
                    belief_post_results[g, mm] = np.average(
                        Delta[t_gap:, 0, age_bottom - t_gap:age_top - t_gap],
                        weights=weights_group,
                        axis=1)
                    if g == 1:
                        parti_pre_results[mm] = np.average(invest_tracker[:-t_gap, 0, age_bottom:age_top],
                                                           weights=weights_group, axis=1)
    return (
        i,
        Delta_results,
        invest_results,
        parti_pre_results,
        belief_pre_results,
        belief_post_results,
    )


# Create a Pool of processes for parallel execution
# Create a ProcessPoolExecutor for parallel execution
def main():
    # Create a ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=16) as executor:  # Adjust the number of workers as needed
        results = [executor.submit(simulate_path_fig4, i) for i in range(Mpath)]
    # Initialize a list to store the results
    results_list = []

    # Retrieve results from parallel processes
    for result in results:
        i, \
            Delta_results, \
            invest_results, \
            parti_pre_results, \
            belief_pre_results, \
            belief_post_results = result.result()

        data = {
            "i": i,
            "fig4_abs_Delta": Delta_results,
            "fig4_parti_prob": invest_results,
            "fig4_parti_pre": parti_pre_results,
            "fig4_belief_pre": belief_pre_results,
            "fig4_belief_post": belief_post_results
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a .npz file
    results_dict = results_df.to_dict(orient='list')
    np.savez("results_fig4.npz", **results_dict)


if __name__ == '__main__':
    main()
