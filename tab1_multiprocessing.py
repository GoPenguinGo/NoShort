import numpy as np
from src.simulation import simulate_SI_mean_vola, simulate_mix_mean_vola
from src.param import nu, mu_Y, sigma_Y, dt, \
    Nt, Nc, tau, Mpath, \
    scenarios, dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    Ntype, alpha_i, cohort_type_size, rho_cohort_type, Vhat, tax, beta0, \
    phi, Npre, Ninit, T_hat, rho_i, beta_i
from src.param_mix import Nconstraint, alpha_i_mix, rho_i_mix, beta_i_mix, cohort_type_size_mix, rho_cohort_type_mix
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

keep_data = int(Nt - 200 / dt)
np.seterr(invalid='ignore')

Mpath = 10

def simulate_mean_vola_path(i: int,
                            Nscenario=3,
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
    parti_age_group_mean_vola_results = np.zeros((Nscenario, 4))
    parti_wealth_group_mean_vola_results = np.zeros((Nscenario, 4))
    cov_save_mean_vola_results = np.zeros((Nscenario, 6))
    parti_results = np.zeros((Nscenario, keep_data))
    cov_parti_results = np.zeros((Nscenario, 4, 3))

    dZ_build = dZ_build_matrix[i]
    dZ = dZ_matrix[i]
    dZ_SI_build = dZ_SI_build_matrix[i]
    dZ_SI = dZ_SI_matrix[i]

    # dZ_build = np.random.randn(Nc) * dt_root
    # dZ = np.random.randn(Nt) * dt_root
    # dZ_SI_build = np.random.randn(Nc) * dt_root
    # dZ_SI = np.random.randn(Nt) * dt_root

    for g in range(Nscenario):
        if g <= 1:
            scenario = scenarios[g]
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
                parti_age_group_mean_vola,
                parti_wealth_group_mean_vola,
                cov_save_mean_vola,
                parti_mean_vola,
                cov_parti_mean_vola,
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
                cohort_type_size
            )
        else:
            (
                dR_mean_vola,
                theta_mean_vola,
                r_mean_vola,
                mu_S_mean_vola,
                sigma_S_mean_vola,
                beta_mean_vola,
                theta_save_mean_vola,
                sigma_S_save_mean_vola,
                parti_age_group_mean_vola,
                parti_wealth_group_mean_vola,
                cov_save_mean_vola,
                parti_mean_vola,
                cov_parti_mean_vola
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
            )

        dR_mean_vola_results[g] = dR_mean_vola
        theta_mean_vola_results[g] = theta_mean_vola
        r_mean_vola_results[g] = r_mean_vola
        mu_S_mean_vola_results[g] = mu_S_mean_vola
        sigma_S_mean_vola_results[g] = sigma_S_mean_vola
        beta_mean_vola_results[g] = beta_mean_vola
        theta_save_mean_vola_results[g] = theta_save_mean_vola
        sigma_S_save_mean_vola_results[g] = sigma_S_save_mean_vola
        parti_age_group_mean_vola_results[g] = parti_age_group_mean_vola
        parti_wealth_group_mean_vola_results[g] = parti_wealth_group_mean_vola
        cov_save_mean_vola_results[g] = cov_save_mean_vola
        parti_results[g] = parti_mean_vola
        cov_parti_results[g] = cov_parti_mean_vola
    covariance_parti = np.corrcoef(parti_results[1], parti_results[2])[0, 1]

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
        parti_age_group_mean_vola_results,
        parti_wealth_group_mean_vola_results,
        cov_save_mean_vola_results,
        covariance_parti,
        cov_parti_results,
    )


def main():
    # Create a ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=16) as executor:  # Adjust the number of workers as needed
        results = [executor.submit(simulate_mean_vola_path, i) for i in range(Mpath)]
    # Initialize a list to store the results
    results_list = []

    # Retrieve results from parallel processes
    for result in results:
        i, \
        dR_mean_vola_results, \
        theta_mean_vola_results, \
        r_mean_vola_results, \
        mu_S_mean_vola_results, \
        sigma_S_mean_vola_results, \
        beta_mean_vola_results, \
        theta_save_mean_vola_results, \
        sigma_S_save_mean_vola_results, \
        parti_age_group_mean_vola_results, \
        parti_wealth_group_mean_vola_results, \
        cov_save_mean_vola_results, \
        covariance_parti, \
        cov_parti_results = result.result()

        data = {
            "i": i,
            "dR_mean_vola": dR_mean_vola_results,
            "theta_mean_vola": theta_mean_vola_results,
            "r_mean_vola": r_mean_vola_results,
            "mu_S_mean_vola": mu_S_mean_vola_results,
            "sigma_S_mean_vola": sigma_S_mean_vola_results,
            "beta_mean_vola": beta_mean_vola_results,
            "theta_compo_mean_vola": theta_save_mean_vola_results,
            "sigma_S_save_mean_vola": sigma_S_save_mean_vola_results,
            "parti_age_group": parti_age_group_mean_vola_results,
            "parti_wealth_group": parti_wealth_group_mean_vola_results,
            "cov_list": cov_save_mean_vola_results,
            "cov_parti_between": covariance_parti,
            "cov_parti": cov_parti_results
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a .npz file
    results_dict = results_df.to_dict(orient='list')
    np.savez("results_mean_vola.npz", **results_dict)


if __name__ == '__main__':
    main()
