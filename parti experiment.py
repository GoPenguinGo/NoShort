import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI, simulate_mix_types
from src.param import nu, mu_Y, sigma_Y, \
    dt, Ninit, Nt, Nc, tau, tax, Vhat, T_hat, beta_i, beta0, rho_cohort_type, Npre, \
    cutoffs_age, Ntype, alpha_i, \
    dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    cohort_type_size, cohort_size, rho_i
from src.param_mix import Nconstraint, rho_i_mix
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import statsmodels.api as sm


plt.rcParams["font.family"] = 'serif'
# (complete, excluded, disappointment, reentry)
density_set = [
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    (0.25, 0.25, 0.25, 0.25),
    # (0.25, 0.25, 0.0, 0.5),
    # (0.25, 0.25, 0.5, 0.0),
    # (0.5, 0.25, 0.0, 0.25),
    # (0.1, 0.1, 0.0, 0.8),
    # (0.1, 0.1, 0.4, 0.4),
]
n_scenarios = len(density_set)
phi_set = [0.0, 0.4, 0.8]
n_phi = len(phi_set)

# # for testing:
# Mpath = 10
Mpath = 1000
window = 12  # 1-year non-overlapping windows
sample = np.arange(600, Nt - 600, window)
N_sample = len(sample)
window_bell = 240
sample_bell = np.arange(600, Nt - 600, window_bell)
N_sample_bell = len(sample_bell)
age_cutoffs_SCF = [int(Nt - 1), int(Nt - 1 - 12 * 15), int(Nt - 1 - 12 * 35), int(Nt - 1 - 12 * 55), 0]
age_cut = 100
Nc_cut = int(age_cut / dt)
age_sample = np.arange(1, Nc_cut, 12)
age_sample2 = np.arange(1, int(200 / dt), 12)
cohort_sample = np.arange(Nc, Nc - 1200, -60) - 1
np.seterr(invalid='ignore')


# noinspection PyTypeChecker
def simulate_path(
        i: int,
):
    print(i)
    # shocks
    dZ_build = dZ_build_matrix[i]
    dZ = dZ_matrix[i]
    dZ_SI_build = dZ_SI_build_matrix[i]
    dZ_SI = dZ_SI_matrix[i]

    # sample: in time-series
    # parti_compare = np.zeros((n_scenarios, n_phi, N_sample), dtype=np.float32)
    # parti_age_group_compare = np.zeros((n_scenarios, n_phi, 4, N_sample), dtype=np.float32)
    # parti_wealth_group_compare = np.zeros((n_scenarios, n_phi, N_sample, 4), dtype=np.float32)
    annual_return_compare = np.zeros((n_scenarios, n_phi, 6, N_sample), dtype=np.float32)
    pd_compare = np.zeros((n_scenarios, n_phi, N_sample), dtype=np.float32)
    future_exc_R_compare = np.zeros((n_scenarios, n_phi, N_sample), dtype=np.float32)
    entry_compare = np.zeros((n_scenarios, n_phi, N_sample), dtype=np.float32)
    exit_compare = np.zeros((n_scenarios, n_phi, N_sample), dtype=np.float32)
    bell_length_compare = np.zeros((n_scenarios, n_phi, N_sample_bell, 4, Nc - window_bell), dtype=int)
    bell_length_reentry_compare = np.zeros((n_scenarios, n_phi, N_sample_bell, 4, Nc - window_bell - 12), dtype=int)
    # age_sample: in cross-section
    entry_cumu_compare = np.zeros((n_scenarios, n_phi, age_cut), dtype=np.float32)
    Delta_age_compare = np.zeros((n_scenarios, n_phi, 4, len(age_sample2)), dtype=np.float32)
    parti_age_compare = np.zeros((n_scenarios, n_phi, 4, len(age_sample2)), dtype=np.float32)
    # mean_vola_compare = np.zeros((n_scenarios, n_phi, 5, 2), dtype=np.float32)
    coef_age_compare = np.zeros((n_scenarios, n_phi, len(cohort_sample), 4), dtype=np.float32)
    for g, density in enumerate(density_set):
        if g <= 1:
            for h, phi in enumerate(phi_set):
                mode_trade = "w_constraint" if g == 1 else "complete"
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
                    parti_wealth_group,
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

                # save data relevant for regressions
                # non-overlapping data, take a sample every 5 years
                past_annual_return = np.zeros((3, Nt))
                future_annual_return = np.zeros((3, Nt))
                for n, gap in enumerate([12, 24, 36]):
                    past_annual_return[n, gap:] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
                    past_annual_return[n, :gap] = np.cumsum(dR[:gap]) / (gap / 12)
                    future_annual_return[n, :-gap] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
                    future_annual_return[n, -gap:] = (np.cumsum(dR)[-gap:]) / (gap / 12)
                annual_return_compare[g, h, :3] = past_annual_return[:, sample]
                annual_return_compare[g, h, 3:] = future_annual_return[:, sample]
                pd_compare[g, h] = np.copy(1 / beta)[sample]
                # parti_compare[g, h] = parti[sample]
                future_exc_R_compare[g, h] = (future_annual_return[0] - r)[sample]
                entry_compare[g, h] = entry_mat[sample]
                exit_compare[g, h] = exit_mat[sample]
                # age_parti = np.zeros((4, Nt))  # participation rate in age groups in time-series
                # for n in range(len(age_cutoffs_SCF) - 1):
                #     age_parti[n] = np.average(invest_tracker[:, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]],
                #                    weights=cohort_size[0, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]], axis=1)
                # parti_age_group_compare[g, h] = age_parti[:, sample]
                # parti_wealth_group_compare[g, h] = parti_wealth_group[sample]

                cohort_actions = np.zeros((1200, Nc - 1200), dtype=int)
                for n in range(1, 1200 + 1):
                    cohort_actions[n - 1] = invest_tracker[n - 1:Nc - 1200 + n - 1, -n]

                annual_sample = np.arange(0, 1200, 12)
                cohort_invest_annual = np.zeros((101, Nc - 1200), dtype=int)
                cohort_invest_annual[1:] = cohort_actions[annual_sample]
                cohort_entry_annual = (cohort_invest_annual[1:] - cohort_invest_annual[:-1]) > 0
                entry_cumu = np.cumsum(np.average(cohort_entry_annual, axis=1), axis=0)
                entry_cumu_compare[g, h] = entry_cumu

                # calculate the length of bell:  only look at the re-entry type
                for n, entry_n in enumerate(sample_bell):  # 20 year non-overlapping windows
                    following_cohorts = (invest_tracker[entry_n, :-12] - invest_tracker[entry_n - 12, 12:] > 0)[
                                        window_bell:]
                    following_cohorts = np.append(following_cohorts, invest_tracker[entry_n, -12:])

                    following_cohorts_exit = (invest_tracker[entry_n, :-12] - invest_tracker[entry_n - 12,
                                                                              12:] < 0)[
                                             window_bell:]  # ignoring the cohorts born during the "year"

                    parti_bell = np.zeros((window_bell, Nc - window_bell))
                    parti_bell[0] = following_cohorts
                    exit_bell = np.zeros(
                        (Nc - window_bell))  # following the entering cohorts until they exit the first time

                    parti_bell_exit = np.zeros((window_bell, Nc - window_bell - 12))
                    parti_bell_exit[0] = following_cohorts_exit
                    reentry_bell = np.zeros((Nc - window_bell - 12))

                    for nn in range(1, window_bell):
                        cohorts_in = invest_tracker[entry_n + nn, window_bell - nn:-nn]
                        cohorts_out = (1 - cohorts_in)[:-12]
                        exit_nn = (
                                invest_tracker[entry_n + nn, window_bell - nn:-nn]
                                - invest_tracker[entry_n + nn - 1, window_bell - nn + 1:-nn + 1] < 0
                        ) if nn != 1 else (
                                invest_tracker[entry_n + nn, window_bell - nn:-nn]
                                - invest_tracker[entry_n + nn - 1, window_bell - nn + 1:] < 0
                        )
                        reentry_nn = (
                                invest_tracker[entry_n + nn, window_bell - nn:-nn]
                                - invest_tracker[entry_n + nn - 1, window_bell - nn + 1:-nn + 1] > 0
                        ) if nn != 1 else (
                                invest_tracker[entry_n + nn, window_bell - nn:-nn]
                                - invest_tracker[entry_n + nn - 1, window_bell - nn + 1:] > 0
                        )
                        exit_bell = exit_bell + exit_nn > 0
                        reentry_bell = reentry_bell + reentry_nn[:-12] > 0
                        parti_bell[nn] = cohorts_in * following_cohorts * (1 - exit_bell)
                        parti_bell_exit[nn] = cohorts_out * following_cohorts_exit * (1 - reentry_bell)
                        bell_length_compare[g, h, n] = list(map(int, np.sum(parti_bell, axis=0) * dt)) + (
                                np.sum(parti_bell, axis=0) * dt > 0)
                        bell_length_reentry_compare[g, h, n] = list(
                            map(int, np.sum(parti_bell_exit, axis=0) * dt)) + (
                                                                       np.sum(parti_bell_exit, axis=0) * dt > 0)

                # stock return alpha:
                for n, cohort in enumerate(cohort_sample):
                    R_st = (mu_S[:-1] * pi[:-1, cohort] + (1 - pi[:-1, cohort]) * r[:-1]) * dt + sigma_S[
                                                                                                 :-1] * dZ[1:]
                    x_regress = sm.add_constant(dR[1:])
                    model = sm.OLS(R_st, x_regress)
                    est = model.fit()
                    coef_age_compare[g, h, n] = est.params[0]

                Delta_age_compare[g, h] = np.average(np.abs(Delta), axis=0)[-age_sample2]
                parti_age_compare[g, h] = np.average(invest_tracker, axis=0)[-age_sample2]

                # mean_list = [r, theta, sigma_S, mu_S, parti]
                # for n, mean_var in enumerate(mean_list):
                #     mean_vola_compare[g, h, n, 0] = np.average(mean_var)
                #     mean_vola_compare[g, h, n, 1] = np.std(mean_var)
                # correlation_compare[g, h, 0] = np.corrcoef(dZ, theta)[0, 1]
                # correlation_compare[g, h, 1] = np.corrcoef(dZ, sigma_S)[0, 1]
                # correlation_compare[g, h, 2] = np.corrcoef(dZ_SI, sigma_S)[0, 1]
                # correlation_compare[g, h, 3] = np.corrcoef(Phi_bar_parti, parti)[0, 1]
                # correlation_compare[g, h, 4] = np.corrcoef(Phi_tilde_parti, parti)[0, 1]

        else:
            alpha_constraint = np.ones(
                (1, Nconstraint)) * density
            alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
            cohort_type_size_mix = cohort_size * alpha_i_mix
            # cohort_size_mix = np.sum(cohort_type_size_mix, axis=0)
            beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio
            rho_cohort_type_mix = alpha_i_mix * beta_i_mix * np.exp(
                -(rho_i_mix + nu) * tau)  # shape(2, 6000)

            phi = 0.4
            h = 1

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
                entry_mat,
                exit_mat,
            ) = simulate_mix_types(Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax,
                                   beta0,
                                   phi, Npre, Ninit, T_hat,
                                   dZ_build, dZ, dZ_SI_build, dZ_SI,
                                   cutoffs_age, Ntype,
                                   Nconstraint, rho_i_mix, alpha_i_mix, beta_i_mix,
                                   rho_cohort_type_mix,
                                   cohort_type_size_mix,
                                   need_f='True',
                                   need_Delta='True',
                                   need_pi='True',
                                   )

            # save data relevant for regressions
            # non-overlapping data, take a sample every 5 years
            past_annual_return = np.zeros((3, Nt))
            future_annual_return = np.zeros((3, Nt))
            for n, gap in enumerate([12, 24, 36]):
                past_annual_return[n, gap:] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
                past_annual_return[n, :gap] = np.cumsum(dR[:gap]) / (gap / 12)
                future_annual_return[n, :-gap] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
                future_annual_return[n, -gap:] = (np.cumsum(dR)[-gap:]) / (gap / 12)
            annual_return_compare[g, h, :3] = past_annual_return[:, sample]
            annual_return_compare[g, h, 3:] = future_annual_return[:, sample]
            pd_compare[g, h] = np.copy(1 / beta)[sample]
            # parti_compare[g, h] = parti[sample]
            future_exc_R_compare[g, h] = (future_annual_return[0] - r)[sample]
            entry_compare[g, h] = entry_mat[sample]
            exit_compare[g, h] = exit_mat[sample]
            # age_parti = np.zeros((4, Nt))
            # for n in range(len(age_cutoffs_SCF) - 1):
            #     age_parti[n] = np.average(
            #         np.average(invest_tracker[:, :, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]],
            #                    weights=cohort_size[0, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]], axis=2),
            #         weights=density,
            #         axis=1)
            # parti_age_group_compare[g, h] = age_parti[:, sample]
            # parti_wealth_group_compare[g, h] = parti_wealth_group[sample]

            cohort_actions = np.zeros((1200, Nc - 1200, 4), dtype=int)
            for n in range(1, 1200 + 1):
                cohort_actions[n - 1] = invest_tracker[n - 1:Nc - 1200 + n - 1, :, -n]

            annual_sample = np.arange(0, 1200, 12)
            cohort_invest_annual = np.zeros((101, Nc - 1200, 4), dtype=int)
            cohort_invest_annual[1:] = cohort_actions[annual_sample]
            cohort_entry_annual = (cohort_invest_annual[1:] - cohort_invest_annual[:-1]) > 0
            entry_cumu = np.average(np.cumsum(np.average(cohort_entry_annual, axis=1), axis=0), weights=density,
                                    axis=1)
            entry_cumu_compare[g, h] = entry_cumu

            # # Calculate the fraction of investors re-entering stock market
            # # follow the exiting cohorts for 5 years and calculate the fraction of them re-entering the stock market
            # # compare to Samuli's paper
            # for n in range(12, Nt - 60):
            #     following_cohorts = (invest_tracker[n, 3, :-12] - invest_tracker[n - 12, 3, 12:] < 0)[60:]
            #     disappointed_cohorts = (invest_tracker[n, 2, :-12] - invest_tracker[n - 12, 2, 12:] < 0)[60:]
            #     parti_bell_reenter = np.zeros((5, Nc - 60 - 12))
            #     cohorts_in_cumu = np.zeros((Nc - 60 - 12))
            #     for nn in range(5):
            #         nn_index = int((nn + 1) * 12)
            #         cohorts_in = invest_tracker[n + nn_index, 3, 60 - nn_index:-nn_index - 12]
            #         cohorts_in_cumu = (cohorts_in_cumu + cohorts_in > 0)
            #         parti_bell_reenter[nn] = following_cohorts * cohorts_in_cumu
            #     popu_reenter = np.sum(parti_bell_reenter * cohort_size[:, 60 + 12:], axis=1) * density_set[g][3]
            #     # popu_exit = (np.sum(following_cohorts * cohort_size[0, 60 + 12:]) * density_set[g][3] +
            #     #              np.sum(disappointed_cohorts * cohort_size[0, 60 + 12:]) * density_set[g][2])
            #     popu_reenter_compare[a, b, c, g, :, n] = popu_reenter
            # popu_exit_compare[a, b, c, g] = exit_mat

            # calculate the length of bell:  only look at the re-entry type
            for n, entry_n in enumerate(sample_bell):  # 20 year non-overlapping windows
                following_cohorts = (invest_tracker[entry_n, :, :-12] - invest_tracker[entry_n - 12, :, 12:] > 0)[:,
                                    window_bell:]
                following_cohorts = np.append(following_cohorts, invest_tracker[entry_n, :, -12:], axis=1)

                following_cohorts_exit = (invest_tracker[entry_n, :, :-12] - invest_tracker[entry_n - 12, :,
                                                                             12:] < 0)[
                                         :, window_bell:]  # ignoring the cohorts born during the "year"

                parti_bell = np.zeros((window_bell, 4, Nc - window_bell))
                parti_bell[0] = following_cohorts
                exit_bell = np.zeros(
                    (4, Nc - window_bell))  # following the entering cohorts until they exit the first time

                parti_bell_exit = np.zeros((window_bell, 4, Nc - window_bell - 12))
                parti_bell_exit[0] = following_cohorts_exit
                reentry_bell = np.zeros((4, Nc - window_bell - 12))

                for nn in range(1, window_bell):
                    cohorts_in = invest_tracker[entry_n + nn, :, window_bell - nn:-nn]
                    cohorts_out = (1 - cohorts_in)[:, :-12]
                    exit_nn = (
                            invest_tracker[entry_n + nn, :, window_bell - nn:-nn]
                            - invest_tracker[entry_n + nn - 1, :, window_bell - nn + 1:-nn + 1] < 0
                    ) if nn != 1 else (
                            invest_tracker[entry_n + nn, :, window_bell - nn:-nn]
                            - invest_tracker[entry_n + nn - 1, :, window_bell - nn + 1:] < 0
                    )
                    reentry_nn = (
                            invest_tracker[entry_n + nn, :, window_bell - nn:-nn]
                            - invest_tracker[entry_n + nn - 1, :, window_bell - nn + 1:-nn + 1] > 0
                    ) if nn != 1 else (
                            invest_tracker[entry_n + nn, :, window_bell - nn:-nn]
                            - invest_tracker[entry_n + nn - 1, :, window_bell - nn + 1:] > 0
                    )
                    exit_bell = exit_bell + exit_nn > 0
                    reentry_bell = reentry_bell + reentry_nn[:, :-12] > 0
                    parti_bell[nn] = cohorts_in * following_cohorts * (1 - exit_bell)
                    parti_bell_exit[nn] = cohorts_out * following_cohorts_exit * (1 - reentry_bell)
                for m in range(4):
                    bell_length_compare[g, h, n, m] = list(map(int, np.sum(parti_bell[:, m], axis=0) * dt)) + (
                            np.sum(parti_bell[:, m], axis=0) * dt > 0)
                    bell_length_reentry_compare[g, h, n, m] = list(
                        map(int, np.sum(parti_bell_exit[:, m], axis=0) * dt)) + (
                                                                      np.sum(parti_bell_exit[:, m], axis=0) * dt > 0)
                # todo: this is based on the number of cohorts, not affected by the cohort population density;
                #  do agents exiting upon more negative shocks take longer to re-enter?
                #  reentry vs. disappointment type

            # stock return alpha:
            for n, cohort in enumerate(cohort_sample):
                for m in range(4):
                    R_st = (mu_S[:-1] * pi[:-1, m, cohort] + (1 - pi[:-1, m, cohort]) * r[:-1]) * dt + sigma_S[
                                                                                                       :-1] * dZ[1:]
                    x_regress = sm.add_constant(dR[1:])
                    model = sm.OLS(R_st, x_regress)
                    est = model.fit()
                    coef_age_compare[g, h, n, m] = est.params[0]

            Delta_age_compare[g, h] = np.average(np.abs(Delta), axis=0)[:, -age_sample]
            parti_age_compare[g, h] = np.average(invest_tracker, axis=0)[:, -age_sample]

            # mean_list = [r, theta, sigma_S, mu_S, parti]
            # for n, mean_var in enumerate(mean_list):
            #     mean_vola_compare[g, h, n, 0] = np.average(mean_var)
            #     mean_vola_compare[g, h, n, 1] = np.std(mean_var)
            # correlation_compare[g, h, 0] = np.corrcoef(dZ, theta)[0, 1]
            # correlation_compare[g, h, 1] = np.corrcoef(dZ, sigma_S)[0, 1]
            # correlation_compare[g, h, 2] = np.corrcoef(dZ_SI, sigma_S)[0, 1]
            # correlation_compare[g, h, 3] = np.corrcoef(Phi_bar_parti, parti)[0, 1]
            # correlation_compare[g, h, 4] = np.corrcoef(Phi_tilde_parti, parti)[0, 1]

    return (
        i,
        # parti_compare,
        # parti_age_group_compare,
        # parti_wealth_group_compare,
        annual_return_compare,
        pd_compare,
        future_exc_R_compare,
        entry_compare,
        exit_compare,
        bell_length_compare,
        bell_length_reentry_compare,
        entry_cumu_compare,
        coef_age_compare,
        Delta_age_compare,
        parti_age_compare,
        # mean_vola_compare,
        # correlation_compare
    )


def main():
    # Create a ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=20) as executor:  # Adjust the number of workers as needed
        results = [executor.submit(simulate_path, i) for i in range(Mpath)]
    # Initialize a list to store the results
    results_list = []

    # Retrieve results from parallel processes
    for result in results:
        i, \
        annual_return_result, \
        pd_result, \
        future_exc_R_result, \
        entry_result, \
        exit_result, \
        bell_length_result, \
        bell_length_reentry_result, \
        entry_cumu_result, \
        coef_age_result, \
        Delta_age_result, \
        parti_age_result = result.result()

        data = {
            "i": i,
            "annual stock return": annual_return_result,
            "pd ratio": pd_result,
            "future excess return": future_exc_R_result,
            "entry rate": entry_result,
            "exit rate": exit_result,
            "bell length": bell_length_result,
            "bell length reentry": bell_length_reentry_result,
            "nr of entry": entry_cumu_result,
            "age alpha": coef_age_result,
            "age Delta": Delta_age_result,
            "age parti": parti_age_result
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a .npz file
    results_dict = results_df.to_dict(orient='list')
    np.savez("parti_rate_regressions.npz", **results_dict)
    # np.savez("parti_Ch.npz", **results_dict)


if __name__ == '__main__':
    main()
