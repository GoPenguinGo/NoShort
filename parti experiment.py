import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI, simulate_mix_types
from src.param import nu, mu_Y, sigma_Y, \
    dt, Ninit, Nt, Nc, tau, tax, Vhat, T_hat, beta_i, beta0, rho_cohort_type, Npre, \
    cutoffs_age,  Ntype, alpha_i, \
    dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    cohort_type_size, cohort_size, rho_i
from src.param_mix import Nconstraint, rho_i_mix
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import statsmodels.api as sm
# import tabulate


plt.rcParams["font.family"] = 'serif'
# (complete, excluded, disappointment, reentry)
density_set = [
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
Mpath = 500
window = 12  # 1-year non-overlapping windows
sample = np.arange(600, Nt - 600, window)
N_sample = len(sample)
window_bell = 240
sample_bell = np.arange(600, Nt - 600, window_bell)
N_sample_bell = len(sample_bell)
age_cutoffs_SCF = [int(Nt-1), int(Nt-1-12*15), int(Nt-1-12*35), int(Nt-1-12*55), 0]
age_cut = 100
Nc_cut = int(age_cut / dt)
age_sample = np.arange(0, Nc_cut, 12)
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
    parti_compare = np.zeros((n_scenarios, n_phi, N_sample), dtype=np.float32)
    parti_age_group_compare = np.zeros((n_scenarios, n_phi, 4, N_sample), dtype=np.float32)
    parti_wealth_group_compare = np.zeros((n_scenarios, n_phi, N_sample, 4), dtype=np.float32)
    annual_return_compare = np.zeros((n_scenarios, n_phi, 6, N_sample), dtype=np.float32)
    pd_compare = np.zeros((n_scenarios, n_phi, N_sample), dtype=np.float32)
    future_exc_R_compare = np.zeros((n_scenarios, n_phi, N_sample), dtype=np.float32)
    entry_compare = np.zeros((n_scenarios, n_phi, N_sample), dtype=np.float32)
    exit_compare = np.zeros((n_scenarios, n_phi, N_sample), dtype=np.float32)
    bell_length_compare = np.zeros((n_scenarios, n_phi, N_sample_bell, 4, Nc - window_bell), dtype=int)
    bell_length_reentry_compare = np.zeros((n_scenarios, n_phi, N_sample_bell, 4, Nc - window_bell - 12), dtype=int)
    # age_sample: in cross-section
    entry_cumu_compare = np.zeros((n_scenarios, n_phi, age_cut), dtype=np.float32)
    Delta_age_compare = np.zeros((n_scenarios, n_phi, 4, len(age_sample)), dtype=np.float32)
    parti_age_compare = np.zeros((n_scenarios, n_phi, 4, len(age_sample)), dtype=np.float32)
    mean_vola_compare = np.zeros((n_scenarios, n_phi, 5, 2), dtype=np.float32)
    coef_age_compare = np.zeros((n_scenarios, n_phi, len(cohort_sample), 4, 2), dtype=np.float32)
    # average
    correlation_compare = np.zeros((n_scenarios, n_phi, 6), dtype=np.float32)

    for g, density in enumerate(density_set):
        if g == 0:
            for h, phi in enumerate(phi_set):
                mode_trade = "w_constraint"
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
                parti_compare[g, h] = parti[sample]
                future_exc_R_compare[g, h] = (future_annual_return[0] - r)[sample]
                entry_compare[g, h] = entry_mat[sample]
                exit_compare[g, h] = exit_mat[sample]
                age_parti = np.zeros((4, Nt))  # participation rate in age groups in time-series
                for n in range(len(age_cutoffs_SCF) - 1):
                    age_parti[n] = np.average(invest_tracker[:, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]],
                                   weights=cohort_size[0, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]], axis=1)
                parti_age_group_compare[g, h] = age_parti[:, sample]
                parti_wealth_group_compare[g, h] = parti_wealth_group[sample]

                cohort_actions = np.zeros((1200, Nc - 1200), dtype=int)
                for n in range(1, 1200 + 1):
                    cohort_actions[n - 1] = invest_tracker[n - 1:Nc - 1200 + n - 1, -n]

                annual_sample = np.arange(0, 1200, 12)
                cohort_invest_annual = np.zeros((101, Nc - 1200), dtype=int)
                cohort_invest_annual[1:] = cohort_actions[annual_sample]
                cohort_entry_annual = (cohort_invest_annual[1:] - cohort_invest_annual[:-1]) > 0
                entry_cumu = np.cumsum(np.average(cohort_entry_annual, axis=1), axis=0)
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
                    following_cohorts = (invest_tracker[entry_n, :-12] - invest_tracker[entry_n - 12, 12:] > 0)[window_bell:]
                    following_cohorts = np.append(following_cohorts, invest_tracker[entry_n, -12:])

                    following_cohorts_exit = (invest_tracker[entry_n, :-12] - invest_tracker[entry_n - 12,
                                                                                 12:] < 0)[window_bell:]  # ignoring the cohorts born during the "year"

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
                    # todo: do agents exiting upon more negative shocks take longer to re-enter?
                    #  reentry vs. disappointment type

                # stock return alpha:
                for n, cohort in enumerate(cohort_sample):
                    R_st = (mu_S[:-1] * pi[:-1, cohort] + (1 - pi[:-1, cohort]) * r[:-1]) * dt + sigma_S[
                                                                                                           :-1] * dZ[1:]
                    x_regress = sm.add_constant(dR[1:])
                    model = sm.OLS(R_st, x_regress)
                    est = model.fit()
                    coef_age_compare[g, h, n] = est.params

                Delta_age_compare[g, h] = np.average(np.abs(Delta), axis=0)[age_sample]
                parti_age_compare[g, h] = np.average(invest_tracker, axis=0)[age_sample]

                mean_list = [r, theta, sigma_S, mu_S, parti]
                for n, mean_var in enumerate(mean_list):
                    mean_vola_compare[g, h, n, 0] = np.average(mean_var)
                    mean_vola_compare[g, h, n, 1] = np.std(mean_var)
                correlation_compare[g, h, 0] = np.corrcoef(dZ, theta)[0, 1]
                correlation_compare[g, h, 1] = np.corrcoef(dZ, sigma_S)[0, 1]
                correlation_compare[g, h, 2] = np.corrcoef(dZ_SI, sigma_S)[0, 1]
                correlation_compare[g, h, 3] = np.corrcoef(Phi_bar_parti, parti)[0, 1]
                correlation_compare[g, h, 4] = np.corrcoef(Phi_tilde_parti, parti)[0, 1]

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
            parti_compare[g, h] = parti[sample]
            future_exc_R_compare[g, h] = (future_annual_return[0] - r)[sample]
            entry_compare[g, h] = entry_mat[sample]
            exit_compare[g, h] = exit_mat[sample]
            age_parti = np.zeros((4, Nt))
            for n in range(len(age_cutoffs_SCF) - 1):
                age_parti[n] = np.average(
                    np.average(invest_tracker[:, :, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]],
                               weights=cohort_size[0, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]], axis=2),
                    weights=density,
                    axis=1)
            parti_age_group_compare[g, h] = age_parti[:, sample]
            parti_wealth_group_compare[g, h] = parti_wealth_group[sample]

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
                    coef_age_compare[g, h, n, m] = est.params

            Delta_age_compare[g, h] = np.average(np.abs(Delta), axis=0)[:, age_sample]
            parti_age_compare[g, h] = np.average(invest_tracker, axis=0)[:, age_sample]

            mean_list = [r, theta, sigma_S, mu_S, parti]
            for n, mean_var in enumerate(mean_list):
                mean_vola_compare[g, h, n, 0] = np.average(mean_var)
                mean_vola_compare[g, h, n, 1] = np.std(mean_var)
            correlation_compare[g, h, 0] = np.corrcoef(dZ, theta)[0, 1]
            correlation_compare[g, h, 1] = np.corrcoef(dZ, sigma_S)[0, 1]
            correlation_compare[g, h, 2] = np.corrcoef(dZ_SI, sigma_S)[0, 1]
            correlation_compare[g, h, 3] = np.corrcoef(Phi_bar_parti, parti)[0, 1]
            correlation_compare[g, h, 4] = np.corrcoef(Phi_tilde_parti, parti)[0, 1]

    return (
        i,
        parti_compare,
        parti_age_group_compare,
        parti_wealth_group_compare,
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
        mean_vola_compare,
        correlation_compare
    )


def main():
    # Create a ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=12) as executor:  # Adjust the number of workers as needed
        results = [executor.submit(simulate_path, i) for i in range(Mpath)]
    # Initialize a list to store the results
    results_list = []

    # Retrieve results from parallel processes
    for result in results:
        i, \
        parti_result, \
        parti_age_group_result, \
        parti_wealth_group_result, \
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
        parti_age_result, \
        mean_vola_result, \
        correlation_result  = result.result()

        data = {
            "i": i,
            "participation rate": parti_result,
            "participation rate in age groups": parti_age_group_result,
            "participation rate in wealth groups": parti_wealth_group_result,
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
            "age parti": parti_age_result,
            "mean vola": mean_vola_result,
            "correlations": correlation_result
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a .npz file
    results_dict = results_df.to_dict(orient='list')
    np.savez("parti_rate_regressions.npz", **results_dict)
    # np.savez("parti_Ch.npz", **results_dict)

    #### Finland: how many re-enter the stock market in a given year
    # results_df = np.load('parti_rate_fin.npz')
    # data_shocks = pd.read_excel(r'E:/Users/A2010290/Documents/GitHub/NoShort/finland_realized_shocks.xlsx',
    #                             sheet_name='Sheet1',
    #                             index_col=0)
    # dZ_actual = data_shocks.to_numpy()[:, 0]
    # Nt_data = dZ_actual.size
    # parti_df = pd.DataFrame(data_shocks.index.astype(str), columns=['Yearmon'])
    # reentry_mat = np.average(results_df['popu_reentry'][:, 1, 0, 0, 0], axis=0)
    # exit_mat = np.average(results_df['popu_exit'][:, 1, 0, 0, 0], axis=0)
    # parti_df['exit'] = exit_mat[-Nt_data:].astype(np.float32)
    # for j in range(5):
    #     parti_df['reentry' + str(j)] = reentry_mat[j, -Nt_data:].astype(np.float32)
    # parti_df.to_stata('stata_dataset/fin_reentry.dta')

    # # Analysis of the bell length: Distribution of participation bells, ignoring 0
    # results_df = np.load('parti_rate_regressions.npz')
    # bell_length_mat = results_df['bell length']
    # bell_length_reentry_mat = results_df['bell length reentry']
    # unique, counts = np.unique(bell_length_mat, return_counts=True)
    # counts_percentage = counts[1:] / np.sum(counts[1:])
    # unique_reentry, counts_reentry = np.unique(bell_length_reentry_mat, return_counts=True)
    # counts_percentage_reentry = counts_reentry[1:] / np.sum(counts_reentry[1:])
    #
    # y_list = [counts_percentage, counts_percentage_reentry]
    # y_titles = ['Years in the stock market before first exit', 'Years out of the stock market before first re-entry']
    # x = np.arange(1, len(counts_percentage)+1, 1)
    # fig, axes = plt.subplots(nrows=2, ncols=1, sharey='all', sharex='all', figsize=(10, 15))
    # for i, ax in enumerate(axes):
    #     ax.plot(x, y_list[i], linewidth=2)
    #     ax.set_title(y_titles[i])
    #     ax.set_xlabel('Years')
    #     ax.set_ylabel('Proportion of re-entry observations')
    # plt.savefig('Reentry_bell_length.png', dpi=200)
    # plt.show()
    # plt.close()
    #
    # y = np.average(results_df['nr of entry'], axis=0)[0]
    # x = np.arange(1, len(y)+1, 1)
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharey='all', sharex='all', figsize=(10, 8))
    # ax.plot(x, y, linewidth=2)
    # ax.set_title('Number of entry to the stock market given age')
    # ax.set_xlabel('Age')
    # ax.set_ylabel('Number of reentry (based on annual obs)')
    # plt.savefig('Reentry_bell_number.png', dpi=200)
    # plt.show()
    # plt.close()

    # age_alpha_mat = results_df['age alpha']
    # y = np.average(age_alpha_mat, axis=0)[:, 0] / dt * 100
    # x = np.arange(0, len(y) * 5, 5)
    # fig, ax = plt.subplots(nrows=1, ncols=1, sharey='all', sharex='all', figsize=(10, 8))
    # ax.plot(x, y, linewidth=2)
    # ax.set_title(r'Average annual alpha given age, $dR_{s,t} = \alpha_{t-s} + \beta_{t-s} dR_t^S + \epsilon_{t-s, t}$')
    # ax.set_xlabel('Age')
    # ax.set_ylabel(r'Average annual alpha, $\%$')
    # ax.axhline(0, 0.05, 0.95, linestyle='dashed', color='gray')
    # plt.savefig('Reentry_age_alpha.png', dpi=200)
    # plt.show()
    # plt.close()

    # # # analysis:
    # results_df = np.load('parti_Ch.npz')
    # # regression 1: participation rate on returns and pd
    # parti = np.copy(results_df["participation rate"])
    # parti_age = np.copy(results_df["participation rate in age groups"])
    # annual_return = np.copy(results_df["annual stock return"])
    # f_return = np.copy(results_df["future excess return"])
    # l_one_year_return = annual_return[:, :, 0]
    # l_two_year_return = annual_return[:, :, 1]
    # l_thr_year_return = annual_return[:, :, 2]
    # pd_ratio = np.copy(results_df["pd ratio"])
    # x_set = [
    #     l_one_year_return,
    #     l_two_year_return,
    #     l_thr_year_return,
    # ]
    # y = np.copy(parti_age)
    # for sce in range(n_scenarios_short):
    #     label_scenario = 'Reentry' if sce == 0 else 'Mix - 4'
    #     reg_data = np.zeros((3, 4, Mpath))
    #     for i, x in enumerate(x_set):
    #         for j in range(4):
    #             for path in range(Mpath):
    #                 x_regress = sm.add_constant(x[path, sce])
    #                 model = sm.OLS(y[path, sce, j], x_regress)
    #                 est = model.fit()
    #                 reg_data[i, j, path] = est.params[1]
    #     print(label_scenario)
    #     print(tabulate.tabulate(np.average(reg_data, axis=2), floatfmt=".3f", tablefmt='latex_raw'))
    #
    # y = np.copy(f_return)
    # x = parti_age
    # for sce in range(n_scenarios_short):
    #     label_scenario = 'Reentry' if sce == 0 else 'Mix - 4'
    #     reg_data_uni = np.zeros((4, Mpath))
    #     reg_data_multi = np.zeros((4, Mpath))
    #     for path in range(Mpath):
    #         # corr_here = np.corrcoef(x[path, sce])[0, 1:]
    #         for age_group in range(4):
    #             x0 = x[path, sce, age_group]
    #             x_regress = sm.add_constant(x0)
    #             model = sm.OLS(y[path, sce], x_regress)
    #             est = model.fit()
    #             reg_data_uni[age_group, path] = est.params[1]
    #         x0 = x[path, sce, 0]
    #         x1 = x[path, sce, 1]
    #         x2 = x[path, sce, 2]
    #         x3 = x[path, sce, 3]
    #         x_multi = np.column_stack((x0, x1, x2, x3))
    #         x_regress = sm.add_constant(x_multi)
    #         model = sm.OLS(y[path, sce], x_regress)
    #         est = model.fit()
    #         reg_data_multi[:, path] = est.params[1:]
    #
    #         # reg_data[:, path] = corr_here
    #     print(label_scenario)
    #     print(np.average(reg_data_uni, axis=1))
    #     print(np.average(reg_data_multi, axis=1))



    # regression_table1_b = np.zeros((n_scenarios_short, len(x_set), Mpath, 2))
    # regression_table1_se = np.zeros((n_scenarios_short, len(x_set), Mpath, 2))
    # for sce in range(n_scenarios_short):
    #     for i, x in enumerate(x_set):
    #         for path in range(Mpath):
    #             if i < len(x_set) - 1:
    #                 x_regress = sm.add_constant(x[path, sce])
    #                 model = sm.OLS(y[path, sce], x_regress)
    #                 est = model.fit()
    #                 # b0 = est.params[0]
    #                 regression_table1_b[sce, i, path, :1] = est.params[1:]
    #                 regression_table1_se[sce, i, path, :1] = est.bse[1:]
    #
    #             else:
    #                 x_multi = np.column_stack((x[path, sce], pd_ratio[path, sce]))
    #                 x_regress = sm.add_constant(x_multi)
    #                 model = sm.OLS(y[path, sce], x_regress)
    #                 est = model.fit()
    #                 # b0 = est.params[0]
    #                 regression_table1_b[sce, i, path] = est.params[1:]
    #                 regression_table1_se[sce, i, path] = est.bse[1:]
    #
    # table1_b = np.average(regression_table1_b, axis=2)
    # table1_se = np.average(regression_table1_se, axis=2)
    #
    # for k in range(n_scenarios_short):
    #     label_scenario = 'Reentry' if k == 0 else 'Mix - 4'
    #     reg_data = np.zeros(((len(x_set) - 1) * 2, len(x_set)))
    #     for i in range(len(x_set) - 1):
    #         reg_data[i * 2, i] = table1_b[k, i, 0]
    #         reg_data[i * 2 + 1, i] = table1_se[k, i, 0]
    #     reg_data[0, len(x_set) - 1] = table1_b[k, len(x_set) - 1, 0]
    #     reg_data[1, len(x_set) - 1] = table1_se[k, len(x_set) - 1, 0]
    #     reg_data[6, len(x_set) - 1] = table1_b[k, len(x_set) - 1, 1]
    #     reg_data[7, len(x_set) - 1] = table1_se[k, len(x_set) - 1, 1]
    #     print(label_scenario)
    #     print(tabulate.tabulate(reg_data, floatfmt=".3f", tablefmt='latex_raw'))
    #
    # # regression 2: participation rate predicts returns
    # f_one_year_return = annual_return[:, :, 3]
    # f_two_year_return = annual_return[:, :, 4]
    # f_thr_year_return = annual_return[:, :, 5]
    # y_set = [
    #     f_one_year_return,
    #     f_two_year_return,
    #     f_thr_year_return
    # ]
    # x = np.copy(results_df["participation rate"])
    # regression_table2_b = np.zeros((n_scenarios_short, len(y_set), Mpath))
    # regression_table2_se = np.zeros((n_scenarios_short, len(y_set), Mpath))
    # for sce in range(n_scenarios_short):
    #     for i, y in enumerate(y_set):
    #         for path in range(Mpath):
    #             x_regress = sm.add_constant(x[path, sce])
    #             model = sm.OLS(y[path, sce], x_regress)
    #             est = model.fit()
    #             # b0 = est.params[0]
    #             regression_table2_b[sce, i, path] = est.params[1]
    #             regression_table2_se[sce, i, path] = est.bse[1]
    # table2_b = np.average(regression_table2_b, axis=2)
    # table2_se = np.average(regression_table2_se, axis=2)
    # for k in range(n_scenarios_short):
    #     label_scenario = 'Reentry' if k == 0 else 'Mix - 4'
    #     reg_data = np.zeros((2, len(y_set)))
    #     for i in range(len(y_set)):
    #         reg_data[0, i] = table2_b[k, i]
    #         reg_data[1, i] = table2_se[k, i]
    #     print(label_scenario)
    #     print(tabulate.tabulate(reg_data, floatfmt=".3f", tablefmt='latex_raw'))


    # results_df = np.load('parti_rate_regressions.npz')
    # # regression 1: participation rate on returns and pd
    # parti = np.copy(results_df["participation rate"])
    # annual_return = np.copy(results_df["annual stock return"])
    # l_one_year_return = annual_return[:, :, 0]
    # l_two_year_return = annual_return[:, :, 1]
    # l_thr_year_return = annual_return[:, :, 2]
    # min_return = np.copy(results_df["monthly min max return"])[:, :, 0]
    # max_return = np.copy(results_df["monthly min max return"])[:, :, 1]
    # pd_ratio = np.copy(results_df["pd ratio"])
    # x_set = [
    #     l_one_year_return,
    #     l_two_year_return,
    #     l_thr_year_return,
    #     # min_return,
    #     # max_return,
    #     pd_ratio,
    #     l_one_year_return,
    # ]
    # # regression_table1_b = np.zeros((n_scenarios_short, len(x_set), Mpath, 2))
    # # regression_table1_se = np.zeros((n_scenarios_short, len(x_set), Mpath, 2))
    # # for sce in range(n_scenarios_short):
    # #     for i, x in enumerate(x_set):
    # #         for path in range(Mpath):
    # #             if i < len(x_set) - 1:
    # #                 x_regress = sm.add_constant(x[path, sce])
    # #                 model = sm.OLS(y[path, sce], x_regress)
    # #                 est = model.fit()
    # #                 # b0 = est.params[0]
    # #                 regression_table1_b[sce, i, path, :1] = est.params[1:]
    # #                 regression_table1_se[sce, i, path, :1] = est.bse[1:]
    # #
    # #             else:
    # #                 x_multi = np.column_stack((x[path, sce], pd_ratio[path, sce]))
    # #                 x_regress = sm.add_constant(x_multi)
    # #                 model = sm.OLS(y[path, sce], x_regress)
    # #                 est = model.fit()
    # #                 # b0 = est.params[0]
    # #                 regression_table1_b[sce, i, path] = est.params[1:]
    # #                 regression_table1_se[sce, i, path] = est.bse[1:]
    # #
    # # table1_b = np.average(regression_table1_b, axis=2)
    # # table1_se = np.average(regression_table1_se, axis=2)
    # #
    # # for k in range(n_scenarios_short):
    # #     label_scenario = 'Reentry' if k == 0 else 'Mix - 4'
    # #     reg_data = np.zeros(((len(x_set) - 1) * 2, len(x_set)))
    # #     for i in range(len(x_set) - 1):
    # #         reg_data[i * 2, i] = table1_b[k, i, 0]
    # #         reg_data[i * 2 + 1, i] = table1_se[k, i, 0]
    # #     reg_data[0, len(x_set) - 1] = table1_b[k, len(x_set) - 1, 0]
    # #     reg_data[1, len(x_set) - 1] = table1_se[k, len(x_set) - 1, 0]
    # #     reg_data[6, len(x_set) - 1] = table1_b[k, len(x_set) - 1, 1]
    # #     reg_data[7, len(x_set) - 1] = table1_se[k, len(x_set) - 1, 1]
    # #     print(label_scenario)
    # #     print(tabulate.tabulate(reg_data, floatfmt=".3f", tablefmt='latex_raw'))
    # #
    # # # regression 2: participation rate predicts returns
    # # f_one_year_return = annual_return[:, :, 3]
    # # f_two_year_return = annual_return[:, :, 4]
    # # f_thr_year_return = annual_return[:, :, 5]
    # # y_set = [
    # #     f_one_year_return,
    # #     f_two_year_return,
    # #     f_thr_year_return
    # # ]
    # # x = np.copy(results_df["participation rate"])
    # # regression_table2_b = np.zeros((n_scenarios_short, len(y_set), Mpath))
    # # regression_table2_se = np.zeros((n_scenarios_short, len(y_set), Mpath))
    # # for sce in range(n_scenarios_short):
    # #     for i, y in enumerate(y_set):
    # #         for path in range(Mpath):
    # #             x_regress = sm.add_constant(x[path, sce])
    # #             model = sm.OLS(y[path, sce], x_regress)
    # #             est = model.fit()
    # #             # b0 = est.params[0]
    # #             regression_table2_b[sce, i, path] = est.params[1]
    # #             regression_table2_se[sce, i, path] = est.bse[1]
    # # table2_b = np.average(regression_table2_b, axis=2)
    # # table2_se = np.average(regression_table2_se, axis=2)
    # # for k in range(n_scenarios_short):
    # #     label_scenario = 'Reentry' if k == 0 else 'Mix - 4'
    # #     reg_data = np.zeros((2, len(y_set)))
    # #     for i in range(len(y_set)):
    # #         reg_data[0, i] = table2_b[k, i]
    # #         reg_data[1, i] = table2_se[k, i]
    # #     print(label_scenario)
    # #     print(tabulate.tabulate(reg_data, floatfmt=".3f", tablefmt='latex_raw'))
    # x_set = [
    #     l_one_year_return,
    #     l_two_year_return,
    #     l_thr_year_return,
    #     parti,
    # ]
    # x_titles = [
    #     r'$R_{t-1,t}$',
    #     r'$R_{t-2,t}$',
    #     r'$R_{t-3,t}$',
    #     r'participation rate$_t$',
    # ]
    # y = pd_ratio
    # for n_sce in range(n_scenarios_short):
    #     fig, axes = plt.subplots(nrows=len(x_set), ncols=1, sharey='all', figsize=(7, 30))
    #     for i, ax in enumerate(axes):
    #         x_use = x_set[i]
    #         ax.scatter(x_use[:, n_sce], y[:, n_sce], s=0.1, c='navy')
    #         ax.set_title(x_titles[i])
    #         ax.set_xlabel(x_titles[i])
    #         ax.set_ylabel('Price-dividend ratio')
    #     save_fig = 'reentry' if n_sce == 0 else 'mix'
    #     plt.savefig('PD_'+save_fig+'.png', dpi=200)
    #     plt.show()
    #     plt.close()




    # #
    # # plot:
    # results_df = np.load('parti rate.npz')
    # y_set = [[
    #     results_df["interest rate"],
    #     results_df["dR"],
    #     results_df["expected return"]
    # ],
    #     [results_df["expected vola"],
    #      results_df["stock vola"],
    #      results_df["price dividend ratio"],
    #      ]]
    # x_set = [
    #     results_df["parti rate"],
    #     # popu_parti_old / popu_parti_compare,
    #     # popu_parti_young / popu_parti_compare,
    # ]
    # label_x_set = [
    #     r'Participation rate, $P_t$',
    #     # r'Participation rate among the oldest quartile',
    #     # r'Oldest quartile / total',
    #     # # r'Participation rate among the youngest quartile',
    #     # r'Youngest quartile / total',
    # ]
    # label_set = [[
    #     r'Cumulated interest rate, $r_{t,t+2}$',
    #     r'Realized stock returns, $dR_{t,t+2}$',
    #     r'Expected returns, $\mu^S_{t,t+2}$'
    # ], [
    #     r'Average diffusion term, $\sigma^S_{t,t+2}$',
    #     r'Average volatility, $E[dR^2_{t,t+2}]$',
    #     r'Price dividend ratio, $pd_{t+2}$'
    # ],
    # ]
    # condi = results_df["parti rate young"] / results_df["parti rate"] / 4
    # labels = ['Mostly old', 'Mostly young']
    # # condi = results_df["average belief"]
    # # labels = ['Overall pessimism', 'Overall optimism']
    # # condi = results_df["recent shocks"]
    # # labels = ['Negative shocks', 'Positive shocks']
    #
    # for ii, x in enumerate(x_set):
    #     fig, axes = plt.subplots(nrows=n_scenarios_short, ncols=1,
    #                              # sharex='all', sharey='all',
    #                              figsize=(4, 20))
    #     j = 1
    #     i = 0
    #     for jj, ax in enumerate(axes):
    #         condi_sce = condi[:, jj]
    #         # condi_sce = condi
    #         # condi_thres = np.percentile(condi_sce, [0, 10, 90, 100])
    #         condi_thres = np.percentile(condi_sce, [0, 25, 50, 75, 100])
    #         ax.grid(True)
    #         for k in range(int(len(condi_thres) - 1)):
    #             condi_ab = condi_sce >= condi_thres[k]
    #             condi_bl = condi_sce < condi_thres[k + 1]
    #             condi_in = np.where(condi_bl * condi_ab == 1)
    #             y = y_set[j][i][:, jj][condi_in]
    #             if k == 0:
    #                 label_k = labels[0]
    #                 ax.scatter(x[:, jj][condi_in], y, s=1, marker='.', c=colors_short[k], label=label_k)
    #             elif k == int(len(condi_thres) - 2):
    #                 label_k = labels[1]
    #                 ax.scatter(x[:, jj][condi_in], y, s=1, marker='.', c=colors_short[k], label=label_k)
    #             else:
    #                 ax.scatter(x[:, jj][condi_in], y, s=1, marker='.', c=colors_short[k])
    #         ax.set_xlabel(label_x_set[ii])
    #         ax.set_ylabel(r'$\sigma^S_{t,t+2}$')
    #         ax.set_title(str(density_set[jj]))
    #         ax.legend()
    #     fig.tight_layout(h_pad=2)
    #     plt.savefig('1quartiles_' + 'learn' + str(T_hat) + '_tax' + str(tax) + '_scatter.png', dpi=200)
        # plt.show()
        # plt.close()

    # for ii, x in enumerate(x_set):
    #     for jj in range(3):
    #         condi_sce = condi[:, jj]
    #         condi_thres = np.percentile(condi_sce, [0, 25, 50, 75, 100])
    #         fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', figsize=(10, 15))
    #         for i, row in enumerate(axes):
    #             for j, ax in enumerate(row):
    #                 ax.grid(True)
    #                 y_title = label_set[j][i]
    #                 for k in range(4):
    #                     condi_ab = condi_sce >= condi_thres[k]
    #                     condi_bl = condi_sce < condi_thres[k + 1]
    #                     condi_in = np.where(condi_bl * condi_ab == 1)
    #                     y = y_set[j][i][:, jj][condi_in]
    #                     if k == 0:
    #                         label_k = labels[0]
    #                         ax.scatter(x[:, jj][condi_in], y, s=1, marker='.', c=colors_short[k], label=label_k)
    #                     elif k == 3:
    #                         label_k = labels[1]
    #                         ax.scatter(x[:, jj][condi_in], y, s=1, marker='.', c=colors_short[k], label=label_k)
    #                     else:
    #                         ax.scatter(x[:, jj][condi_in], y, s=1, marker='.', c=colors_short[k])
    #                 ax.set_xlabel(label_x_set[ii])
    #                 ax.set_title(y_title)
    #                 if i == j == 0:
    #                     ax.legend()
    #                 # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #                 # fig.savefig('r and theta, subfig ' + str(j + 1) + '.png', bbox_inches=extent.expanded(1.2, 1.3), dpi=200)
    #         fig.tight_layout(h_pad=2)
    #         plt.savefig(str(jj)+'quartiles_' + 'learn' + str(T_hat) + '_tax' + str(tax) + '_scatter.png', dpi=200)
    #         # plt.show()
    #         # plt.close()



if __name__ == '__main__':
    main()
