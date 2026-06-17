import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_mix_types
from src.param import mu_Y, sigma_Y, \
    dt, Ninit, Nc, Nt, tau, \
    cutoffs_age, Ntype, alpha_i, \
    dZ_build_matrix, dZ_matrix, \
    cohort_size, rho_i, tax, beta0, beta_i, nu, \
    T_hat, entry_bound, exit_bound, phi
from concurrent.futures import ProcessPoolExecutor
from src.param_mix import Nconstraint, rho_i_mix, density


plt.rcParams["font.family"] = 'serif'

T_hat_vec = [2, 5, 10, 15, 20]

a_rho_bar = 1
b_rho_bar = -(tax * beta0 + rho_i[0, 0] + rho_i[1, 0])
c_rho_bar = alpha_i[0, 0] * tax * beta_i[0, 0] * (rho_i[0, 0] - rho_i[1, 0]) + rho_i[1, 0] * rho_i[0, 0] + tax * beta0 * \
            rho_i[1, 0]
rho_bar = (-b_rho_bar - np.sqrt(b_rho_bar ** 2 - 4 * a_rho_bar * c_rho_bar)) / (2 * a_rho_bar)
T = 400
N_T = int(T / dt)
Mpath = 400  # is enough as the results are averages already
c_sample = np.arange(-1, -int(200/dt), -24)

fc_init = tax / (1 + tax) * (1 + rho_i / nu)
growth_c_i = nu - tax * beta0 + rho_bar - rho_i
f_c_benchmark = (fc_init * np.exp(growth_c_i * tau))[:, c_sample]
keep = int(200 / dt)
flow_sample = np.arange(keep, Nt, 12)
sample = -np.arange(1, Nt + 1, 12)
t_s_mat = np.tile(np.reshape(np.cumsum(np.ones(N_T) * dt) - dt, (-1, 1)), (1, 2))
rho_i_mat = np.reshape(rho_i, (1, -1))
discount_rate_mat = np.exp(-(nu + rho_i_mat) * t_s_mat)[:N_T]



def utility_mu(mu_Y_use, sigma_Y_use):
    t_s = np.cumsum(np.ones(N_T) * dt) - dt
    E_log_C = (mu_Y_use + rho_bar - rho_i + nu - beta0 * nu - 1 / 2 * sigma_Y_use ** 2) * t_s * dt
    discount_rate = np.exp(-(nu + rho_i) * t_s)
    E_util_t = E_log_C * discount_rate
    E_util = np.sum(E_util_t, axis=1)
    return E_util


def E_log_C_growth_calculator(
        pi_W,
        r,
        mu_S,
        sigma_S,
):
    return r + pi_W * (mu_S - r) - 1 / 2 * (pi_W * sigma_S) ** 2



def simulate_path(
        i: int,
        density,
        T_hat_use=T_hat,
        phi_use=phi,
):
    print(i)
    # shocks
    dZ_build = dZ_build_matrix[i]
    dZ = dZ_matrix[i]

    Npre_use = int(T_hat_use / dt)
    Vhat_use = (sigma_Y ** 2) / T_hat_use  # prior variance

    alpha_constraint = np.ones(
        (1, Nconstraint)) * density
    alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
    cohort_type_size_mix = cohort_size * alpha_i_mix
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
        parti_age_group,
        # Delta_popu,
        # parti_wealth_group,
        entry_mat,
        exit_mat
    ) = simulate_mix_types(Nc, Nt, dt, nu,
                           Vhat_use,
                           mu_Y, sigma_Y, tax, beta0,
                           phi_use,
                           Npre_use,
                           Ninit,
                           T_hat_use,
                           entry_bound,
                           exit_bound,
                           dZ_build, dZ,
                           cutoffs_age, Ntype,
                           Nconstraint, rho_i_mix, alpha_i_mix, beta_i_mix,
                           rho_cohort_type_mix,
                           cohort_type_size_mix,
                           need_f='True',
                           need_Delta='True',
                           need_pi='True',
                           mode_learn='invest',
                           )

    ave_r = np.average(r[keep:])
    ave_mu_S = np.average(mu_S[keep:])
    ave_sigma_S = np.average(sigma_S[keep:])

    benchmark_log_C_growth1 = E_log_C_growth_calculator(1.0,
                                                        ave_r,
                                                        ave_mu_S,
                                                        ave_sigma_S) - rho_i[0]

    benchmark_log_C_growth2 = E_log_C_growth_calculator(0.0,
                                                        ave_r,
                                                        ave_mu_S,
                                                        ave_sigma_S) - rho_i[0]

    pi_focus = pi[keep:, -1]

    # beliefs-driven entry and exit in GE
    g_log_C_mat = (np.reshape(r - rho_i[0] + 1 / 2 * theta ** 2, (-1, 1)) - 1 / 2 * Delta[:, -1] ** 2)[keep:]
    g_log_C_mat_N = np.tile(np.reshape(r - rho_i[0], (-1, 1)), (1, Nc))[keep:]
    nonparti = np.where(pi_focus <= 0)
    g_log_C_mat[nonparti] = g_log_C_mat_N[nonparti]

    # beliefs-driven entry and exit, using unconditional average returns
    g_log_C_experience = E_log_C_growth_calculator(pi_focus,
                                                   ave_r,
                                                   ave_mu_S,
                                                   ave_sigma_S) - rho_i[0]

    # average entry and exit
    uncon_pi = np.average(
        pi_focus[np.where(pi_focus > 0)],
        weights=np.tile(cohort_size, (Nt - keep, 1))[np.where(pi_focus > 0)]
    )
    uncon_parti = np.average(
        pi_focus > 0,
        weights=np.tile(cohort_size, (Nt - keep, 1))
    )
    g_log_C_ave_ee = (
                             E_log_C_growth_calculator(
                                 uncon_pi,
                                 ave_r,
                                 ave_mu_S,
                                 ave_sigma_S) - rho_i[0]) * uncon_parti + (
                             E_log_C_growth_calculator(
                                 0.0,
                                 ave_r,
                                 ave_mu_S,
                                 ave_sigma_S) - rho_i[0]) * (1 - uncon_parti)

    g_log_C_ave_ee_endo = np.tile(
        np.reshape(r + uncon_pi * (mu_S - r) - 1 / 2 * (uncon_pi * sigma_S) ** 2, (-1, 1)), (1, Nc)
    )[keep:] - rho_i[0]
    g_log_C_ave_ee_endo[nonparti] = g_log_C_mat_N[nonparti]

    g_log_C_ave_ee_nolev = (
                             E_log_C_growth_calculator(
                                 1.0,
                                 ave_r,
                                 ave_mu_S,
                                 ave_sigma_S) - rho_i[0]) * uncon_parti + (
                             E_log_C_growth_calculator(
                                 0.0,
                                 ave_r,
                                 ave_mu_S,
                                 ave_sigma_S) - rho_i[0]) * (1 - uncon_parti)

    g_log_C_ave_ee_nolev_endo = np.tile(
        np.reshape(r + 1.0 * (mu_S - r) - 1 / 2 * (1.0 * sigma_S) ** 2, (-1, 1)), (1, Nc)
    )[keep:] - rho_i[0]
    g_log_C_ave_ee_nolev_endo[nonparti] = g_log_C_mat_N[nonparti]

    ave_pi = np.average(
        pi_focus,
        weights=np.tile(cohort_size, (Nt - keep, 1))
    )
    g_log_C_no_ee = E_log_C_growth_calculator(ave_pi,
                                              ave_r,
                                              ave_mu_S,
                                              ave_sigma_S) - rho_i[0]

    return (
        i,
        # flow[flow_sample],
        np.average(g_log_C_mat, axis=0)[sample],
        np.average(g_log_C_experience, axis=0)[sample],
        # np.average(g_log_C_ave_ee_endo, axis=0)[sample],
        np.array([
            np.average(np.average(g_log_C_mat, axis=0), weights=cohort_size[0]),
            np.average(np.average(g_log_C_experience, axis=0), weights=cohort_size[0]),
            np.average(np.average(g_log_C_ave_ee_endo, axis=0), weights=cohort_size[0]),
            g_log_C_ave_ee[0],
            np.average(np.average(g_log_C_ave_ee_nolev_endo, axis=0), weights=cohort_size[0]),
            g_log_C_ave_ee_nolev[0],
            g_log_C_no_ee[0],
        ]),
        np.array([benchmark_log_C_growth1[0], benchmark_log_C_growth2[0]])
    )


def main():
    for T_hat_use in T_hat_vec:
        with ProcessPoolExecutor(max_workers=20) as executor:  # Adjust the number of workers as needed
            results = [executor.submit(simulate_path, i, density, T_hat_use) for i in range(Mpath)]
        results_list = []

        for result in results:
            i, \
                g_log_C_mat, \
                g_log_C_experience, \
                g_log_C_ave, \
                g_log_C_benchmark = result.result()

            data = {
                'i': i,
                'g_log_C_mat': g_log_C_mat,
                'g_log_C_experience': g_log_C_experience,
                'g_log_C_ave': g_log_C_ave,
                'g_log_C_benchmark': g_log_C_benchmark,
            }
            results_list.append(data)

        results_df = pd.DataFrame(results_list)
        results_dict = results_df.to_dict(orient='list')
        np.savez(f"simu_results/welfare{T_hat_use}.npz", **results_dict)


if __name__ == '__main__':
    main()





