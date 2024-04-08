import numpy as np
from src.simulation import simulate_SI, simulate_SI_mean_vola
from src.param import nu, mu_Y, sigma_Y, tax, phi, \
    dt, T_hat, Npre, Vhat, Ninit, Nt, Nc, tau, cohort_size, \
    cutoffs_age, n_age_cutoffs, Mpath, \
    scenarios, dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    cummu_popu, Ntype, rho_i, alpha_i, beta_i, beta0, rho_cohort_type, cohort_type_size
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

# # for testing:
# Mpath = 10
np.seterr(invalid='ignore')


# noinspection PyTypeChecker
def simulate_path(
        i: int,
        Nscenario=2,
):
    print(i)
    # Initialize results for the current Mpath
    # for fig 4:
    age_cut = 100
    Nc_cut = int(age_cut / dt)
    data_point = np.arange(0, Nc_cut, 15)
    invest_results = 0
    Delta_results = np.zeros((Nscenario, len(data_point)), dtype=np.float32)
    t_gap = int(2 / dt)  # 2-year window
    N_cut = int(Nc - t_gap)
    parti_pre_results = np.zeros((n_age_cutoffs, N_cut), dtype=np.float32)

    # for figure 9
    popu_cummu = np.cumsum(cohort_size)
    popu = 0.1
    cutoff_age_old_below = np.searchsorted(popu_cummu, popu)
    cutoff_age_young = np.searchsorted(popu_cummu, 1 - popu)
    belief_popu_old_results = 0
    belief_popu_young_results = 0
    P_old_results = 0
    P_young_results = 0
    Wealthshare_old_results = 0
    Wealthshare_young_results = 0

    # for figure 8
    popu_5 = 0.5
    cutoff_age_old_below_5 = np.searchsorted(popu_cummu, popu_5)
    cutoff_age_young_5 = np.searchsorted(popu_cummu, 1 - popu_5)
    Phi_results = np.zeros((Nscenario, Nt), dtype=np.float32)
    Delta_bar_results = np.zeros((Nscenario, Nt), dtype=np.float32)
    belief_popu_results = 0
    belief_f_old_results = np.zeros((Nscenario, Nt), dtype=np.float32)
    belief_f_young_results = np.zeros((Nscenario, Nt), dtype=np.float32)
    Phi_old_results = np.zeros((Nscenario, Nt), dtype=np.float32)
    Phi_young_results = np.zeros((Nscenario, Nt), dtype=np.float32)

    # for figure 10:
    # can use parti_pre from fig4
    parti_post_results = np.zeros((n_age_cutoffs, N_cut), dtype=np.float32)
    leverage_parti_pre_results = np.zeros((n_age_cutoffs, N_cut), dtype=np.float32)
    leverage_parti_post_results = np.zeros((n_age_cutoffs, N_cut), dtype=np.float32)

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
            parti_wealth_group,
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
                        cutoffs_age,
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

        Delta_results[g] = np.flip(np.average(np.abs(Delta), axis=0)[-Nc_cut:])[data_point]
        if g == 1:
            invest_results = np.flip(np.average(invest_tracker, axis=0)[-Nc_cut:])[data_point]

            belief_popu_old_results = np.average(
                Delta[:, :cutoff_age_old_below],
                weights=cohort_type_size[0, :cutoff_age_old_below],
                axis=1)
            belief_popu_young_results = np.average(
                Delta[:, cutoff_age_young:],
                weights=cohort_type_size[0, cutoff_age_young:],
                axis=1)
            P_old_results = np.average(
                invest_tracker[:, :cutoff_age_old_below],
                weights=cohort_type_size[0, :cutoff_age_old_below],
                axis=1) / popu
            P_young_results = np.average(
                invest_tracker[:, cutoff_age_young:],
                weights=cohort_type_size[0, cutoff_age_young:],
                axis=1) / popu
            Wealthshare_old_results = np.sum(
                np.sum(
                    f_c[:, :, :cutoff_age_old_below] * dt,
                    axis=2), axis=1
            )
            Wealthshare_young_results = np.sum(
                np.sum(
                    f_c[:, :, cutoff_age_young:] * dt,
                    axis=2), axis=1
            )

        for mm in range(n_age_cutoffs):
            age_bottom = cutoffs_age[mm + 1] if mm <= 2 else -N_cut
            age_top = cutoffs_age[mm]
            weights_group = cohort_type_size[0, age_bottom:age_top]
            if g == 1:
                parti_pre_results[mm] = np.average(invest_tracker[:-t_gap, age_bottom:age_top],
                                                   weights=weights_group, axis=1)
                parti_post_results[mm] = np.average(invest_tracker[t_gap:, age_bottom - t_gap:age_top - t_gap],
                                                    weights=weights_group, axis=1)
                pi = np.ma.masked_where(pi == 0, pi)
                leverage_parti_pre_results[mm] = np.ma.average(
                    pi[:-t_gap, age_bottom:age_top], weights=weights_group, axis=1)
                leverage_parti_post_results[mm] = np.ma.average(
                    pi[t_gap:, age_bottom - t_gap:age_top - t_gap], weights=weights_group, axis=1)

        Phi_results[g] = Phi_bar_parti
        Delta_bar_results[g] = Delta_bar_parti
        if g == 0:
            belief_popu_results = np.average(Delta, weights=cohort_type_size[0],
                                             axis=1)  # same average belief as phi == 0
            belief_f_old_results[g] = np.sum(
                Delta[:, :cutoff_age_old_below_5] * np.sum(
                    f_c[:, :, :cutoff_age_old_below_5], axis=1
                ), axis=1
            ) / np.sum(
                np.sum(f_c[:, :, :cutoff_age_old_below_5], axis=2
                       ), axis=1
            )
            belief_f_young_results[g] = np.sum(
                Delta[:, cutoff_age_young_5:] * np.sum(f_c[:, :, cutoff_age_young_5:], axis=1
                                                       ), axis=1
            ) / np.sum(
                np.sum(f_c[:, :, cutoff_age_young_5:], axis=2
                       ), axis=1
            )
            Phi_old_results[g] = np.sum(
                np.sum(
                    f_c[:, :, :cutoff_age_old_below_5] * dt,
                    axis=2
                ),
                axis=1
            )
            Phi_young_results[g] = np.sum(
                np.sum(
                    f_c[:, :, cutoff_age_young_5:] * dt,
                    axis=2
                ),
                axis=1
            )
        else:
            belief_f_old_results[g] = np.sum(
                Delta[:, :cutoff_age_old_below_5] * invest_tracker[:, :cutoff_age_old_below_5] * np.sum(
                    f_c[:, :, :cutoff_age_old_below_5],
                    axis=1
                ), axis=1
            ) / np.sum(
                invest_tracker[:, :cutoff_age_old_below_5] * np.sum(f_c[:, :, :cutoff_age_old_below_5], axis=1
                                                                    ), axis=1
            )
            belief_f_young_results[g] = np.sum(
                Delta[:, cutoff_age_young_5:] * invest_tracker[:, cutoff_age_young_5:] * np.sum(
                    f_c[:, :, cutoff_age_young_5:],
                    axis=1
                ), axis=1
            ) / np.sum(
                invest_tracker[:, cutoff_age_young_5:] * np.sum(f_c[:, :, cutoff_age_young_5:], axis=1
                                                                ), axis=1
            )
            Phi_old_results[g] = np.sum(
                np.sum(
                    f_c[:, :, :cutoff_age_old_below_5], axis=1
                ) * invest_tracker[:, :cutoff_age_old_below_5] * dt,
                axis=1
            )
            Phi_young_results[g] = np.sum(
                np.sum(
                    f_c[:, :, cutoff_age_young_5:],
                    axis=1
                ) * invest_tracker[:, cutoff_age_young_5:] * dt,
                axis=1
            )

    return (
        i,
        Delta_results,
        invest_results,
        parti_pre_results,
        belief_popu_old_results,
        belief_popu_young_results,
        P_old_results,
        P_young_results,
        Wealthshare_old_results,
        Wealthshare_young_results,
        Phi_results,
        Delta_bar_results,
        belief_popu_results,
        belief_f_old_results,
        belief_f_young_results,
        Phi_old_results,
        Phi_young_results,
        parti_post_results,
        leverage_parti_pre_results,
        leverage_parti_post_results
    )


def main():
    # Create a ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=32) as executor:  # Adjust the number of workers as needed
        results = [executor.submit(simulate_path, i) for i in range(Mpath)]
    # Initialize a list to store the results
    results_list = []

    # Retrieve results from parallel processes
    for result in results:
        i, \
        Delta_results, \
        invest_results, \
        parti_pre_results, \
        belief_popu_old_results, \
        belief_popu_young_results, \
        P_old_results, \
        P_young_results, \
        Wealthshare_old_results, \
        Wealthshare_young_results, \
        Phi_results, \
        Delta_bar_results, \
        belief_popu_results, \
        belief_f_old_results, \
        belief_f_young_results, \
        Phi_old_results, \
        Phi_young_results, \
        parti_post_results, \
        leverage_parti_pre_results, \
        leverage_parti_post_results = result.result()

        data = {
            "i": i,
            "fig4_abs_Delta": Delta_results,
            "fig4_parti_prob": invest_results,
            "fig4_parti_pre": parti_pre_results,
            "fig9_old_belief": belief_popu_old_results,
            "fig9_young_belief": belief_popu_young_results,
            "fig9_old_parti": P_old_results,
            "fig9_young_parti": P_young_results,
            "fig9_old_fw": Wealthshare_old_results,
            "fig9_young_fw": Wealthshare_young_results,
            "fig8_Phi": Phi_results,
            "fig8_Delta_bar": Delta_bar_results,
            "fig8_belief": belief_popu_results,
            "fig8_old_belief_fc": belief_f_old_results,
            "fig8_young_belief_fc": belief_f_young_results,
            "fig8_Phi_old": Phi_old_results,
            "fig8_Phi_young": Phi_young_results,
            "fig10_parti_post": parti_post_results,
            "fig10_leverage_pre": leverage_parti_pre_results,
            "fig10_leverage_post": leverage_parti_post_results
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a .npz file
    results_dict = results_df.to_dict(orient='list')
    np.savez("results.npz", **results_dict)


if __name__ == '__main__':
    main()
