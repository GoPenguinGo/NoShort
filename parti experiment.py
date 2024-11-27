import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI, simulate_mix_types
from src.param import mu_Y, sigma_Y, \
    dt, Ninit, Nt, Nc, tau, tax, \
    cutoffs_age, Ntype, alpha_i, \
    dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    cohort_type_size, cohort_size
from src.param import phi
# from src.param import rho_i, beta_i, beta0, rho_cohort_type, beta_cohort
from src.param_mix import Nconstraint
# from src.param_mix import rho_i_mix
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import statsmodels.api as sm


plt.rcParams["font.family"] = 'serif'
# (complete, excluded, disappointment, reentry)
density_set = [
    # (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    # (0.25, 0.25, 0.25, 0.25),
    # (0.25, 0.25, 0.0, 0.5),
    # (0.25, 0.25, 0.5, 0.0),
    # (0.5, 0.25, 0.0, 0.25),
    # (0.1, 0.1, 0.0, 0.8),
    # (0.1, 0.1, 0.4, 0.4),
]
n_scenarios = len(density_set)
T_hat_set = [2]
n_T_hat = len(T_hat_set)
rho_i = np.array([[0.001], [-0.02]])
nu = 0.03
# phi_set = [0.0, 0.4, 0.8]
# n_phi = len(phi_set)
beta_i = (nu + rho_i) / (1 + tax)  # consumption wealth ratio
beta0 = np.sum(alpha_i * beta_i).astype(float)
rho_cohort_type = alpha_i * beta_i * np.exp(-(rho_i + nu) * tau)  # shape(2, 6000)
beta_cohort = np.sum(np.exp(-beta_i * tau) * alpha_i, axis=0)
rho_i_mix = np.tile(np.reshape(rho_i, (-1, 1, 1)), (1, Nconstraint, 1))

# # for testing:
# Mpath = 10
Mpath = 100
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
folder_address = r'C:\Users\A2010290\OneDrive - BI Norwegian Business School (BIEDU)\Documents\GitHub computer 2\NoShort/reg_results2/'


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
    # parti_compare = np.zeros((n_scenarios-1, N_sample), dtype=np.float32)
    # parti_age_group_compare = np.zeros((n_scenarios, n_phi, 4, N_sample), dtype=np.float32)
    # parti_wealth_group_compare = np.zeros((n_scenarios, n_phi, N_sample, 4), dtype=np.float32)
    # annual_return_compare = np.zeros((n_scenarios, 6, N_sample), dtype=np.float32)
    # pd_compare = np.zeros((n_scenarios, N_sample), dtype=np.float32)
    # future_exc_R_compare = np.zeros((n_scenarios, N_sample), dtype=np.float32)
    # entry_compare = np.zeros((n_scenarios-1, N_sample), dtype=np.float32)
    # exit_compare = np.zeros((n_scenarios-1, N_sample), dtype=np.float32)
    # bell_length_compare = np.zeros((n_scenarios-1, N_sample_bell, 4, Nc - window_bell), dtype=int)
    # bell_length_reentry_compare = np.zeros((n_scenarios-1, N_sample_bell, 4, Nc - window_bell - 12), dtype=int)
    # age_sample: in cross-section
    # entry_cumu_compare = np.zeros((n_scenarios-1, age_cut), dtype=np.float32)
    # Delta_age_compare = np.zeros((n_scenarios, n_T_hat, 4, len(age_sample2)), dtype=np.float32)
    # parti_age_compare = np.zeros((n_scenarios, n_T_hat,  4, len(age_sample2)), dtype=np.float32)
    # coef_age_compare = np.zeros((n_scenarios, n_T_hat, len(cohort_sample), 4), dtype=np.float32)
    regression_table1 = np.zeros((n_scenarios, n_T_hat, 3, 3), dtype=np.float32)
    regression_table2 = np.zeros((n_scenarios, n_T_hat, 3, 3), dtype=np.float32)

    for g, density in enumerate(density_set):
        for h, T_hat in enumerate(T_hat_set):
            Npre = int(T_hat / dt)
            Vhat = (sigma_Y ** 2) / T_hat  # prior variance
            if g < 1:
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
                                need_f='False',
                                need_Delta='False',
                                need_pi='False',
                                )


                # annual_return_compare[g, :3] = past_annual_return[:, sample]
                # annual_return_compare[g, 3:] = future_annual_return[:, sample]
                # pd_compare[g] = np.copy(1 / beta)[sample]
                # future_exc_R_compare[g] = (future_annual_return[0] - r)[sample]

                # parti_compare[g] = parti[sample]
                # parti_age_compare[g, h] = np.average(invest_tracker, axis=0)[-age_sample2]

                # cohort_actions = np.zeros((1200, Nc - 1200), dtype=int)
                # for n in range(1, 1200 + 1):
                #     cohort_actions[n - 1] = invest_tracker[n - 1:Nc - 1200 + n - 1, -n]
                #
                # annual_sample = np.arange(0, 1200, 12)
                # cohort_invest_annual = np.zeros((101, Nc - 1200), dtype=int)
                # cohort_invest_annual[1:] = cohort_actions[annual_sample]
                # cohort_entry_annual = (cohort_invest_annual[1:] - cohort_invest_annual[:-1]) > 0
                # entry_cumu = np.cumsum(np.average(cohort_entry_annual, axis=1), axis=0)
                # entry_cumu_compare[g] = entry_cumu

                # for n, cohort in enumerate(cohort_sample):
                #     R_st = (mu_S[:-1] * pi[:-1, cohort] + (1 - pi[:-1, cohort]) * r[:-1]) * dt + sigma_S[
                #                                                                                  :-1] * dZ[1:]
                #     x_regress = sm.add_constant(dR[1:])
                #     model = sm.OLS(R_st, x_regress)
                #     est = model.fit()
                #     coef_age_compare[g, h, n] = est.params[0]
                # Delta_age_compare[g, h] = np.average(np.abs(Delta), axis=0)[-age_sample2]

            else:
                alpha_constraint = np.ones(
                    (1, Nconstraint)) * density
                alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
                cohort_type_size_mix = cohort_size * alpha_i_mix
                # cohort_size_mix = np.sum(cohort_type_size_mix, axis=0)
                beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio
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
                                       need_f='False',
                                       need_Delta='False',
                                       need_pi='False',
                                       )

            # save data relevant for regressions
            # non-overlapping data, take a sample every 5 years
            past_annual_return = np.zeros((3, Nt), dtype=np.float32)
            future_annual_return = np.zeros((3, Nt), dtype=np.float32)
            change_parti = np.zeros((3, len(sample)), dtype=np.float32)
            for n, gap in enumerate([12, 24, 36]):
                past_annual_return[n, gap:] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
                past_annual_return[n, :gap] = np.cumsum(dR[:gap]) / (gap / 12)
                change_parti[n] = np.log(parti[sample] / parti[sample - gap])
                future_annual_return[n, :-gap] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
                future_annual_return[n, -gap:] = (np.cumsum(dR)[-gap:]) / (gap / 12)

            # run regressions and save results instead of saving the data:
            x_set = np.copy(past_annual_return[:, sample])

            y_set = [change_parti,
                     entry_mat[sample],
                     exit_mat[sample]]
            regression_table1_b = np.zeros((len(x_set), len(y_set)), dtype=np.float32)
            for ii in range(3):
                x = x_set[ii]
                for jj, y_mat in enumerate(y_set):
                    if jj == 1:  # entry on high return
                        y = y_mat[:, ii]
                        x_condi = (x > np.percentile(x, 75)) + 0
                        x_regress = sm.add_constant(x_condi)
                        model = sm.OLS(y, x_regress)
                    elif jj == 2:  # exit on low return
                        y = y_mat[:, ii]
                        x_condi = (x < np.percentile(x, 25)) + 0
                        x_regress = sm.add_constant(x_condi)
                        model = sm.OLS(y, x_regress)
                    else:
                        y = y_mat[ii]
                        x_regress = sm.add_constant(x)
                        model = sm.OLS(y, x_regress)
                    est = model.fit()
                    regression_table1_b[ii, jj] = est.params[1]

            x_set = [parti[sample],
                     entry_mat[sample, 0],
                     exit_mat[sample, 0]]
            y_set = future_annual_return[:, sample]
            regression_table2_b = np.zeros((len(x_set), len(y_set)), dtype=np.float32)
            for ii in range(3):
                x = x_set[ii]
                for jj in range(3):
                    y = y_set[jj]
                    x_regress = sm.add_constant(x)
                    model = sm.OLS(y, x_regress)
                    est = model.fit()
                    regression_table2_b[ii, jj] = est.params[1]
            regression_table1[g, h] = regression_table1_b
            regression_table2[g, h] = regression_table2_b

    np.save(folder_address + str(i) + "reg1.npy", regression_table1)
    np.save(folder_address + str(i) + "reg2.npy", regression_table2)

    # parti_compare = np.zeros((n_scenarios-1, n_phi, N_sample), dtype=np.float32)
    # # parti_age_group_compare = np.zeros((n_scenarios, n_phi, 4, N_sample), dtype=np.float32)
    # # parti_wealth_group_compare = np.zeros((n_scenarios, n_phi, N_sample, 4), dtype=np.float32)
    # annual_return_compare = np.zeros((n_scenarios, n_phi, 6, N_sample), dtype=np.float32)
    # pd_compare = np.zeros((n_scenarios, n_phi, N_sample), dtype=np.float32)
    # future_exc_R_compare = np.zeros((n_scenarios, n_phi, N_sample), dtype=np.float32)
    # entry_compare = np.zeros((n_scenarios-1, N_sample), dtype=np.float32)
    # exit_compare = np.zeros((n_scenarios-1, N_sample), dtype=np.float32)
    # # bell_length_compare = np.zeros((n_scenarios-1, N_sample_bell, 4, Nc - window_bell), dtype=int)
    # # bell_length_reentry_compare = np.zeros((n_scenarios-1, N_sample_bell, 4, Nc - window_bell - 12), dtype=int)
    # # age_sample: in cross-section
    # entry_cumu_compare = np.zeros((n_scenarios-1, n_phi, age_cut), dtype=np.float32)
    # Delta_age_compare = np.zeros((n_scenarios, n_phi, 4, len(age_sample2)), dtype=np.float32)
    # parti_age_compare = np.zeros((n_scenarios-1, n_phi, 4, len(age_sample2)), dtype=np.float32)
    # # mean_vola_compare = np.zeros((n_scenarios, n_phi, 5, 2), dtype=np.float32)
    # coef_age_compare = np.zeros((n_scenarios, n_phi, len(cohort_sample), 4), dtype=np.float32)

    # for g, density in enumerate(density_set):
    #     if g <= 1:
    #         for h, phi in enumerate(phi_set):
    #             mode_trade = "w_constraint" if g == 1 else "complete"
    #             mode_learn = 'reentry'
    #             (
    #                 r,
    #                 theta,
    #                 f_c,
    #                 Delta,
    #                 pi,
    #                 parti,
    #                 Phi_bar_parti,
    #                 Phi_tilde_parti,
    #                 Delta_bar_parti,
    #                 Delta_tilde_parti,
    #                 dR,
    #                 mu_S,
    #                 sigma_S,
    #                 beta,
    #                 invest_tracker,
    #                 parti_age_group,
    #                 parti_wealth_group,
    #                 entry_mat,
    #                 exit_mat
    #             ) = simulate_SI(mode_trade,
    #                             mode_learn,
    #                             Nc,
    #                             Nt,
    #                             dt,
    #                             nu,
    #                             Vhat,
    #                             mu_Y,
    #                             sigma_Y,
    #                             tax,
    #                             beta0,
    #                             phi,
    #                             Npre,
    #                             Ninit,
    #                             T_hat,
    #                             dZ_build,
    #                             dZ,
    #                             dZ_SI_build,
    #                             dZ_SI,
    #                             tau,
    #                             cutoffs_age,
    #                             Ntype,
    #                             rho_i,
    #                             alpha_i,
    #                             beta_i,
    #                             rho_cohort_type,
    #                             cohort_type_size,
    #                             need_f='True',
    #                             need_Delta='True',
    #                             need_pi='True',
    #                             )
    #
    #             # save data relevant for regressions
    #             # non-overlapping data, take a sample every 5 years
    #             past_annual_return = np.zeros((3, Nt))
    #             future_annual_return = np.zeros((3, Nt))
    #             for n, gap in enumerate([12, 24, 36]):
    #                 past_annual_return[n, gap:] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
    #                 past_annual_return[n, :gap] = np.cumsum(dR[:gap]) / (gap / 12)
    #                 future_annual_return[n, :-gap] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
    #                 future_annual_return[n, -gap:] = (np.cumsum(dR)[-gap:]) / (gap / 12)
    #             annual_return_compare[g, h, :3] = past_annual_return[:, sample]
    #             annual_return_compare[g, h, 3:] = future_annual_return[:, sample]
    #             pd_compare[g, h] = np.copy(1 / beta)[sample]
    #             future_exc_R_compare[g, h] = (future_annual_return[0] - r)[sample]
    #             if g > 0:
    #                 parti_compare[g-1, h] = parti[sample]
    #                 parti_age_compare[g - 1, h] = np.average(invest_tracker, axis=0)[-age_sample2]
    #
    #                 cohort_actions = np.zeros((1200, Nc - 1200), dtype=int)
    #                 for n in range(1, 1200 + 1):
    #                     cohort_actions[n - 1] = invest_tracker[n - 1:Nc - 1200 + n - 1, -n]
    #
    #                 annual_sample = np.arange(0, 1200, 12)
    #                 cohort_invest_annual = np.zeros((101, Nc - 1200), dtype=int)
    #                 cohort_invest_annual[1:] = cohort_actions[annual_sample]
    #                 cohort_entry_annual = (cohort_invest_annual[1:] - cohort_invest_annual[:-1]) > 0
    #                 entry_cumu = np.cumsum(np.average(cohort_entry_annual, axis=1), axis=0)
    #                 entry_cumu_compare[g-1, h] = entry_cumu
    #
    #                 if h == 1:
    #                     entry_compare[g-1] = entry_mat[sample]
    #                     exit_compare[g-1] = exit_mat[sample]
    #
    #                     # # calculate the length of bell:  only look at the re-entry type
    #                     # for n, entry_n in enumerate(sample_bell):  # 20 year non-overlapping windows
    #                     #     following_cohorts = (invest_tracker[entry_n, :-12] - invest_tracker[entry_n - 12, 12:] > 0)[
    #                     #                         window_bell:]
    #                     #     following_cohorts = np.append(following_cohorts, invest_tracker[entry_n, -12:])
    #                     #
    #                     #     following_cohorts_exit = (invest_tracker[entry_n, :-12] - invest_tracker[entry_n - 12,
    #                     #                                                               12:] < 0)[
    #                     #                              window_bell:]  # ignoring the cohorts born during the "year"
    #                     #
    #                     #     parti_bell = np.zeros((window_bell, Nc - window_bell))
    #                     #     parti_bell[0] = following_cohorts
    #                     #     exit_bell = np.zeros(
    #                     #         (Nc - window_bell))  # following the entering cohorts until they exit the first time
    #                     #
    #                     #     parti_bell_exit = np.zeros((window_bell, Nc - window_bell - 12))
    #                     #     parti_bell_exit[0] = following_cohorts_exit
    #                     #     reentry_bell = np.zeros((Nc - window_bell - 12))
    #                     #
    #                     #     for nn in range(1, window_bell):
    #                     #         cohorts_in = invest_tracker[entry_n + nn, window_bell - nn:-nn]
    #                     #         cohorts_out = (1 - cohorts_in)[:-12]
    #                     #         exit_nn = (
    #                     #                 invest_tracker[entry_n + nn, window_bell - nn:-nn]
    #                     #                 - invest_tracker[entry_n + nn - 1, window_bell - nn + 1:-nn + 1] < 0
    #                     #         ) if nn != 1 else (
    #                     #                 invest_tracker[entry_n + nn, window_bell - nn:-nn]
    #                     #                 - invest_tracker[entry_n + nn - 1, window_bell - nn + 1:] < 0
    #                     #         )
    #                     #         reentry_nn = (
    #                     #                 invest_tracker[entry_n + nn, window_bell - nn:-nn]
    #                     #                 - invest_tracker[entry_n + nn - 1, window_bell - nn + 1:-nn + 1] > 0
    #                     #         ) if nn != 1 else (
    #                     #                 invest_tracker[entry_n + nn, window_bell - nn:-nn]
    #                     #                 - invest_tracker[entry_n + nn - 1, window_bell - nn + 1:] > 0
    #                     #         )
    #                     #         exit_bell = exit_bell + exit_nn > 0
    #                     #         reentry_bell = reentry_bell + reentry_nn[:-12] > 0
    #                     #         parti_bell[nn] = cohorts_in * following_cohorts * (1 - exit_bell)
    #                     #         parti_bell_exit[nn] = cohorts_out * following_cohorts_exit * (1 - reentry_bell)
    #                     #         bell_length_compare[g - 1, n] = list(map(int, np.sum(parti_bell, axis=0) * dt)) + (
    #                     #                 np.sum(parti_bell, axis=0) * dt > 0)
    #                     #         bell_length_reentry_compare[g - 1, n] = list(
    #                     #             map(int, np.sum(parti_bell_exit, axis=0) * dt)) + (
    #                     #                                                            np.sum(parti_bell_exit,
    #                     #                                                                   axis=0) * dt > 0)
    #
    #             # stock return alpha:
    #             for n, cohort in enumerate(cohort_sample):
    #                 R_st = (mu_S[:-1] * pi[:-1, cohort] + (1 - pi[:-1, cohort]) * r[:-1]) * dt + sigma_S[
    #                                                                                              :-1] * dZ[1:]
    #                 x_regress = sm.add_constant(dR[1:])
    #                 model = sm.OLS(R_st, x_regress)
    #                 est = model.fit()
    #                 coef_age_compare[g, h, n] = est.params[0]
    #             Delta_age_compare[g, h] = np.average(np.abs(Delta), axis=0)[-age_sample2]
    #
    #     else:
    #         alpha_constraint = np.ones(
    #             (1, Nconstraint)) * density
    #         alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
    #         cohort_type_size_mix = cohort_size * alpha_i_mix
    #         # cohort_size_mix = np.sum(cohort_type_size_mix, axis=0)
    #         beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio
    #         rho_cohort_type_mix = alpha_i_mix * beta_i_mix * np.exp(
    #             -(rho_i_mix + nu) * tau)  # shape(2, 6000)
    #
    #         phi = 0.4
    #         h = 1
    #
    #         (
    #             r,
    #             theta,
    #             f_c,
    #             Delta,
    #             pi,
    #             parti,
    #             Phi_bar_parti,
    #             Phi_tilde_parti,
    #             Delta_bar_parti,
    #             Delta_tilde_parti,
    #             dR,
    #             mu_S,
    #             sigma_S,
    #             beta,
    #             invest_tracker,
    #             parti_age_group,
    #             parti_wealth_group,
    #             entry_mat,
    #             exit_mat,
    #         ) = simulate_mix_types(Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax,
    #                                beta0,
    #                                phi, Npre, Ninit, T_hat,
    #                                dZ_build, dZ, dZ_SI_build, dZ_SI,
    #                                cutoffs_age, Ntype,
    #                                Nconstraint, rho_i_mix, alpha_i_mix, beta_i_mix,
    #                                rho_cohort_type_mix,
    #                                cohort_type_size_mix,
    #                                need_f='True',
    #                                need_Delta='True',
    #                                need_pi='True',
    #                                )
    #
    #         # save data relevant for regressions
    #         # non-overlapping data, take a sample every 5 years
    #         past_annual_return = np.zeros((3, Nt))
    #         future_annual_return = np.zeros((3, Nt))
    #         for n, gap in enumerate([12, 24, 36]):
    #             past_annual_return[n, gap:] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
    #             past_annual_return[n, :gap] = np.cumsum(dR[:gap]) / (gap / 12)
    #             future_annual_return[n, :-gap] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
    #             future_annual_return[n, -gap:] = (np.cumsum(dR)[-gap:]) / (gap / 12)
    #         annual_return_compare[g, h, :3] = past_annual_return[:, sample]
    #         annual_return_compare[g, h, 3:] = future_annual_return[:, sample]
    #         pd_compare[g, h] = np.copy(1 / beta)[sample]
    #         future_exc_R_compare[g, h] = (future_annual_return[0] - r)[sample]
    #
    #         parti_compare[g-1, h] = parti[sample]
    #         entry_compare[g-1] = entry_mat[sample]
    #         exit_compare[g-1] = exit_mat[sample]
    #         parti_age_compare[g-1, h] = np.average(invest_tracker, axis=0)[:, -age_sample2]
    #
    #         cohort_actions = np.zeros((1200, Nc - 1200, 4), dtype=int)
    #         for n in range(1, 1200 + 1):
    #             cohort_actions[n - 1] = invest_tracker[n - 1:Nc - 1200 + n - 1, :, -n]
    #
    #         annual_sample = np.arange(0, 1200, 12)
    #         cohort_invest_annual = np.zeros((101, Nc - 1200, 4), dtype=int)
    #         cohort_invest_annual[1:] = cohort_actions[annual_sample]
    #         cohort_entry_annual = (cohort_invest_annual[1:] - cohort_invest_annual[:-1]) > 0
    #         entry_cumu = np.average(np.cumsum(np.average(cohort_entry_annual, axis=1), axis=0), weights=density,
    #                                 axis=1)
    #         entry_cumu_compare[g-1, h] = entry_cumu
    #         # calculate the length of bell:  only look at the re-entry type
    #         # for n, entry_n in enumerate(sample_bell):  # 20 year non-overlapping windows
    #         #     following_cohorts = (invest_tracker[entry_n, :, :-12] - invest_tracker[entry_n - 12, :, 12:] > 0)[:,
    #         #                         window_bell:]
    #         #     following_cohorts = np.append(following_cohorts, invest_tracker[entry_n, :, -12:], axis=1)
    #         #
    #         #     following_cohorts_exit = (invest_tracker[entry_n, :, :-12] - invest_tracker[entry_n - 12, :,
    #         #                                                                  12:] < 0)[
    #         #                              :, window_bell:]  # ignoring the cohorts born during the "year"
    #         #
    #         #     parti_bell = np.zeros((window_bell, 4, Nc - window_bell))
    #         #     parti_bell[0] = following_cohorts
    #         #     exit_bell = np.zeros(
    #         #         (4, Nc - window_bell))  # following the entering cohorts until they exit the first time
    #         #
    #         #     parti_bell_exit = np.zeros((window_bell, 4, Nc - window_bell - 12), dtype=int)
    #         #     parti_bell_exit[0] = following_cohorts_exit
    #         #     reentry_bell = np.zeros((4, Nc - window_bell - 12), dtype=int)
    #         #
    #         #     for nn in range(1, window_bell):
    #         #         cohorts_in = invest_tracker[entry_n + nn, :, window_bell - nn:-nn]
    #         #         cohorts_out = (1 - cohorts_in)[:, :-12]
    #         #         exit_nn = (
    #         #                 invest_tracker[entry_n + nn, :, window_bell - nn:-nn]
    #         #                 - invest_tracker[entry_n + nn - 1, :, window_bell - nn + 1:-nn + 1] < 0
    #         #         ) if nn != 1 else (
    #         #                 invest_tracker[entry_n + nn, :, window_bell - nn:-nn]
    #         #                 - invest_tracker[entry_n + nn - 1, :, window_bell - nn + 1:] < 0
    #         #         )
    #         #         reentry_nn = (
    #         #                 invest_tracker[entry_n + nn, :, window_bell - nn:-nn]
    #         #                 - invest_tracker[entry_n + nn - 1, :, window_bell - nn + 1:-nn + 1] > 0
    #         #         ) if nn != 1 else (
    #         #                 invest_tracker[entry_n + nn, :, window_bell - nn:-nn]
    #         #                 - invest_tracker[entry_n + nn - 1, :, window_bell - nn + 1:] > 0
    #         #         )
    #         #         exit_bell = exit_bell + exit_nn > 0
    #         #         reentry_bell = reentry_bell + reentry_nn[:, :-12] > 0
    #         #         parti_bell[nn] = cohorts_in * following_cohorts * (1 - exit_bell)
    #         #         parti_bell_exit[nn] = cohorts_out * following_cohorts_exit * (1 - reentry_bell)
    #         #     for m in range(4):
    #         #         bell_length_compare[g-1, n, m] = list(map(int, np.sum(parti_bell[:, m], axis=0) * dt)) + (
    #         #                 np.sum(parti_bell[:, m], axis=0) * dt > 0)
    #         #         bell_length_reentry_compare[g-1, n, m] = list(
    #         #             map(int, np.sum(parti_bell_exit[:, m], axis=0) * dt)) + (
    #         #                                                           np.sum(parti_bell_exit[:, m], axis=0) * dt > 0)
    #         # stock return alpha:
    #         for n, cohort in enumerate(cohort_sample):
    #             for m in range(4):
    #                 R_st = (mu_S[:-1] * pi[:-1, m, cohort] + (1 - pi[:-1, m, cohort]) * r[:-1]) * dt + sigma_S[
    #                                                                                                    :-1] * dZ[1:]
    #                 x_regress = sm.add_constant(dR[1:])
    #                 model = sm.OLS(R_st, x_regress)
    #                 est = model.fit()
    #                 coef_age_compare[g, h, n, m] = est.params[0]
    #         Delta_age_compare[g, h] = np.average(np.abs(Delta), axis=0)[:, -age_sample2]

    return (
        i,
        # parti_compare,
        # parti_age_group_compare,
        # parti_wealth_group_compare,
        # annual_return_compare,
        # pd_compare,
        # future_exc_R_compare,
        # entry_compare,
        # exit_compare,
        # bell_length_compare,
        # bell_length_reentry_compare,
        # entry_cumu_compare,
        # coef_age_compare,
        # Delta_age_compare,
        # parti_age_compare,
        regression_table1,
        regression_table2,
        # mean_vola_compare,
        # correlation_compare
    )


def main():
    # Create a ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=10) as executor:  # Adjust the number of workers as needed
        results = [executor.submit(simulate_path, i) for i in range(Mpath)]
    # Initialize a list to store the results
    results_list = []

    # Retrieve results from parallel processes
    for result in results:
        # i, \
        # parti_result, \
        # annual_return_result, \
        # pd_result, \
        # future_exc_R_result, \
        # entry_result, \
        # exit_result, \
        # entry_cumu_result, \
        # coef_age_result, \
        # Delta_age_result, \
        # parti_age_result = result.result()
        i, \
        regression_table1, \
        regression_table2 = result.result()

        data = {
            "i": i,
            # "participation rate": parti_result,
            # "annual stock return": annual_return_result,
            # "pd ratio": pd_result,
            # "future excess return": future_exc_R_result,
            # "entry rate": entry_result,
            # "exit rate": exit_result,
            # # "bell length": bell_length_result,
            # # "bell length reentry": bell_length_reentry_result,
            # "nr of entry": entry_cumu_result,
            # "age alpha": coef_age_result,
            # "age Delta": Delta_age_result,
            # "age parti": parti_age_result,
            "return_parti_reg": regression_table1,
            "parti_return_reg": regression_table2,
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a .npz file
    results_dict = results_df.to_dict(orient='list')
    np.savez("parti_rate_regressions2.npz", **results_dict)
    # np.savez("parti_Ch.npz", **results_dict)


if __name__ == '__main__':
    main()
