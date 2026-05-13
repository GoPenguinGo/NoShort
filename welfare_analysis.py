import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_mix_types
from src.param import mu_Y, sigma_Y, \
    dt, Ninit, Nc, Nt, tau, \
    cutoffs_age, Ntype, alpha_i, \
    dZ_build_matrix, dZ_SI_build_matrix, dZ_SI_matrix, dZ_matrix, \
    cohort_size, rho_i, tax, beta0, beta_i, nu, Vhat, phi, Npre, \
    T_hat, rho_cohort_type, cohort_type_size, entry_bound, exit_bound, phi_vector
from concurrent.futures import ProcessPoolExecutor
from src.param_mix import Nconstraint, rho_i_mix, density
from scipy.interpolate import make_interp_spline


a_rho_bar = 1
b_rho_bar = -(tax * beta0 + rho_i[0, 0] + rho_i[1, 0])
c_rho_bar = alpha_i[0, 0] * tax * beta_i[0, 0] * (rho_i[0, 0] - rho_i[1, 0]) + rho_i[1, 0] * rho_i[0, 0] + tax * beta0 * \
            rho_i[1, 0]
rho_bar = (-b_rho_bar - np.sqrt(b_rho_bar ** 2 - 4 * a_rho_bar * c_rho_bar)) / (2 * a_rho_bar)
T = 400
N_T = int(T / dt)
Mpath = 2000
# Nt_long = 8400
Nt_long = 6000
c_sample = np.arange(-1, -int(200/dt), -24)

fc_init = tax / (1 + tax) * (1 + rho_i / nu)
growth_c_i = nu - tax * beta0 + rho_bar - rho_i
f_c_benchmark = (fc_init * np.exp(growth_c_i * tau))[:, c_sample]
keep = int(200 / dt)
sample = np.arange(0, N_T, 1)
# T_hat_vec = np.append(
#     np.arange(1, 5, 2),
#     np.arange(5, 30, 5),
# )
T_hat_vec = [2, 5, 10, 20]
t_s_mat = np.tile(np.reshape(np.cumsum(np.ones(N_T) * dt) - dt, (-1, 1)), (1, 2))
rho_i_mat = np.reshape(rho_i, (1, -1))
discount_rate_mat = np.exp(-(nu + rho_i_mat) * t_s_mat)[:N_T]
folder_address = r'E:\Users\A2010290\Documents\GitHub\NoShort/welfare/'
plt.rcParams["font.family"] = 'serif'



def utility_mu(mu_Y_use, sigma_Y_use):
    t_s = np.cumsum(np.ones(N_T) * dt) - dt
    E_log_C = (mu_Y_use + rho_bar - rho_i + nu - beta0 * nu - 1 / 2 * sigma_Y_use ** 2) * t_s * dt
    discount_rate = np.exp(-(nu + rho_i) * t_s)
    E_util_t = E_log_C * discount_rate
    E_util = np.sum(E_util_t, axis=1)
    return E_util


# def simulate_path(
#         i: int,
#         T_hat_use,
# ):
#     print(i)
#     # shocks
#     dZ_build = dZ_build_matrix[i]
#     dZ = np.random.randn(Nt_long) * np.sqrt(dt)
#     dZ_SI_build = dZ_SI_build_matrix[i]
#     dZ_SI = np.random.randn(Nt_long) * np.sqrt(dt)
#
#     Npre_use = int(T_hat_use / dt)
#     Vhat_use = (sigma_Y ** 2) / T_hat_use  # prior variance
#
#     log_C_mat = np.zeros((10, int(T / dt), 2), dtype=np.float32)
#     (
#         r,
#         theta,
#         f_c,
#         Delta,
#         pi,
#         parti,
#         Phi_bar_parti,
#         Phi_tilde_parti,
#         Delta_bar_parti,
#         Delta_tilde_parti,
#         dR,
#         mu_S,
#         sigma_S,
#         beta,
#         invest_tracker,
#         parti_age_group,
#         parti_wealth_group,
#         entry_mat,
#         exit_mat
#     ) = simulate_SI(mode_trade,
#                     mode_learn,
#                     Nc,
#                     Nt_long,
#                     dt,
#                     nu,
#                     Vhat_use,
#                     mu_Y,
#                     sigma_Y,
#                     tax,
#                     beta0,
#                     phi,
#                     Npre_use,
#                     Ninit,
#                     T_hat_use,
#                     dZ_build,
#                     dZ,
#                     dZ_SI_build,
#                     dZ_SI,
#                     tau,
#                     cutoffs_age,
#                     Ntype,
#                     rho_i,
#                     alpha_i,
#                     beta_i,
#                     rho_cohort_type,
#                     cohort_type_size,
#                     need_f='True',
#                     need_Delta='False',
#                     need_pi='False',
#                     )
#     C_mat = np.exp((mu_Y - 1/2 * sigma_Y ** 2) * np.arange(0, Nt_long) * dt + sigma_Y * np.cumsum(dZ))
#     C_mat_reshape = np.tile(np.reshape(C_mat, (-1, 1, 1)), (1, 2, 6000))
#     C_matrix = f_c * C_mat_reshape / cohort_type_size * dt
#     sample = np.arange(0, N_T, 1)
#
#     del C_mat_reshape
#
#     for j in range(10):
#         t_start = int(100 / dt) + int(10 / dt) * j
#         log_C_mat[j] = np.log(C_matrix[t_start + sample, :, Nt - 1 - sample] / C_matrix[t_start, :, -1])
#
#     # np.save(folder_address + str(i) + ".npy", np.average(log_C_mat, axis=0))
#     E_util_path_t = np.average(log_C_mat, axis=0) * dt * discount_rate_mat
#     E_util_path = np.sum(E_util_path_t, axis=0)
#
#     f_c_indi = np.average(np.ma.masked_invalid(f_c) / cohort_type_size * dt, axis=0)[:, c_sample]
#     return (
#         i,
#         # E_util_path,
#         f_c_indi,
#     )


def simulate_path(
        i: int,
        T_hat_use,
        phi_use,
        density,
):
    print(i)
    # shocks
    dZ_build = dZ_build_matrix[i]
    dZ = np.random.randn(Nt_long) * np.sqrt(dt)

    Npre_use = int(T_hat_use / dt)
    Vhat_use = (sigma_Y ** 2) / T_hat_use  # prior variance

    log_C_mat = np.zeros((10, int(T / dt), Ntype, Nconstraint), dtype=np.float32)
    cohort_size = cohort_type_size_mix[0, 0]

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

    if density[-1] == 0:
        C_share = f_c[:, :, 0] / cohort_type_size_mix[:, 0] * dt
        flow = np.average(np.average(np.log(C_share), weights=cohort_size, axis=2), weights=alpha_i[:, 0], axis=1)

    else:
        C_share = np.maximum(f_c / cohort_type_size_mix * dt, 1e-5)  # per unit of agents
        flow = np.average(np.average(np.log(C_share), weights=cohort_size, axis=3), weights=alpha_i[:, 0], axis=1)

        C_mat = np.exp((mu_Y - 1 / 2 * sigma_Y ** 2) * np.arange(0, Nt_long) * dt + sigma_Y * np.cumsum(dZ))

        for j in range(10):
            t_start = int(10 / dt) * j
            log_C_mat[j] = np.log(
                C_share[t_start + sample, :, :, Nt - 1 - sample] / C_share[t_start, :, :, -1]
            ) * np.reshape(np.log(
                C_mat[t_start + sample] / C_mat[t_start]
            ), (-1, 1, 1))

        # np.save(folder_address + str(i) + ".npy", np.average(log_C_mat, axis=0))
        E_util_path_t = np.average(log_C_mat, axis=0) * np.reshape(discount_rate_mat, (-1, Ntype, 1))
        E_util_path = np.sum(E_util_path_t * dt, axis=0)

    return (
        i,
        flow[-keep:],
        E_util_path,
    )



def main():
    for T_hat_try in T_hat_vec:
        for phi in phi_vector:
            with ProcessPoolExecutor(max_workers=20) as executor:  # Adjust the number of workers as needed
                results = [executor.submit(simulate_path, i, int(T_hat_try), density, phi) for i in range(Mpath)]
            results_list = []

            for result in results:
                i, \
                    flow_welfare, \
                    E_util = result.result()

                data = {
                    'i': i,
                    'flow_welfare': flow_welfare,
                    'E_util': E_util,
                }
                results_list.append(data)

            results_df = pd.DataFrame(results_list)
            results_dict = results_df.to_dict(orient='list')
            np.savez(folder_address + f"{int(T_hat_try)}_{phi}_wel_flow.npz", **results_dict)

            # with ProcessPoolExecutor(max_workers=20) as executor:  # Adjust the number of workers as needed
            #     results = [executor.submit(simulate_path, i, int(T_hat_try)) for i in range(Mpath)]
            # results_list = []
            #
            # for result in results:
            #     i, \
            #     f_c_result,  = result.result()
            #
            #     data = {
            #         'i': i,
            #         'f_c_indi': f_c_result,
            #     }
            #     results_list.append(data)
            #
            # results_df = pd.DataFrame(results_list)
            # results_dict = results_df.to_dict(orient='list')
            # np.savez(folder_address + str(int(T_hat_try)) + "_indi.npz", **results_dict)

    ### the complete one
    with ProcessPoolExecutor(max_workers=20) as executor:  # Adjust the number of workers as needed
        results = [executor.submit(simulate_path, i, 10, (1, 0, 0)) for i in range(Mpath)]
    results_list = []

    for result in results:
        i, \
            flow_welfare, \
            E_util = result.result()

        data = {
            'i': i,
            'flow_welfare': flow_welfare,

        }
        results_list.append(data)

    results_df = pd.DataFrame(results_list)
    results_dict = results_df.to_dict(orient='list')
    np.savez(folder_address + "benchmark_wel_flow.npz", **results_dict)


if __name__ == '__main__':
    main()
    #
    # N_points = 1000
    # mu_vec = np.linspace(0, mu_Y, N_points)
    # E_util_mu = np.zeros((N_points, 2))
    # E_util_t_mu = np.zeros((N_points, 2, len(c_sample)))
    # for i, mu_try in enumerate(mu_vec):
    #     E_util_mu[i] = utility_mu(mu_try, sigma_Y)
    #
    # sigma_vec = np.flip(np.linspace(sigma_Y, 0.5, N_points))
    # E_util_sigma = np.zeros((N_points, 2))
    # for i, sigma_try in enumerate(sigma_vec):
    #     E_util_sigma[i] = utility_mu(mu_Y, sigma_try)
    #
    # t_s_mat = np.tile(np.reshape(np.cumsum(np.ones(N_T) * dt) - dt, (-1, 1)), (1, 2))
    # rho_i_mat = np.reshape(rho_i, (1, -1))
    # discount_rate_mat = np.exp(-(nu + rho_i_mat) * t_s_mat)
    #
    # equiv_mu = np.zeros((len(T_hat_vec), 2))
    # equiv_sigma = np.zeros((len(T_hat_vec), 2))
    #
    # for i, T_hat_try in enumerate(T_hat_vec):
    #     E_util_learn_path = np.load(folder_address + str(int(T_hat_try)) + ".npz")['E_util']
    #     E_util_learn = np.average(np.ma.masked_invalid(E_util_learn_path), axis=0)
    #     for j in range(2):
    #         equiv_mu[i, j] = mu_vec[np.searchsorted(E_util_mu[:, j], E_util_learn[j])]
    #         equiv_sigma[i, j] = sigma_vec[np.searchsorted(E_util_sigma[:, j], E_util_learn[j])]
    #
    #
    # # E_util_vola10 = utility_mu(mu_Y, 0.15)
    # # equiv_mu_vola10 = np.zeros(2)
    # # for i in range(2):
    # #     equiv_mu_vola10[i] = mu_vec[np.searchsorted(E_util_mu[:, i], E_util_vola10[i])]
    #
    # X_Y_Spline = make_interp_spline(T_hat_vec, equiv_mu, k=3)
    # X_ = np.linspace(1, 25, 100)
    # Y_ = X_Y_Spline(X_)
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    # ax.set_xlabel('Pre-entry learning window')
    # ax.set_ylabel(r'Equivalent $\mu^Y$')
    # # ax.set_ylim(0, 0.02)
    # ax.plot(X_, Y_[:, 0], color='navy', linewidth=1.5, label=r'Type a, $\rho=0.1\%$')
    # ax.plot(X_, Y_[:, 1], color='red', linewidth=1.5, label=r'Type b, $\rho=0.5\%$')
    # plt.axhline(y=0.02, color='gray', linestyle='dashed', label=r'Actual $\mu^Y$')
    # plt.legend(loc='lower right')
    # ax.tick_params(axis='y', labelcolor='black')
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.savefig('Welfare.png', dpi=100)
    # plt.show()
    # plt.close()

    # line_styles = ['dashed', 'solid', 'dotted', 'dashdot']
    # x = np.arange(0, int(200/dt), 24) * dt
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    # ax.set_xlabel('Age')
    # ax.set_ylabel(r'Consumption relative to the benchmark economy')
    # for i, T_hat_try in enumerate(T_hat_vec):
    #     f_c_indi_mat = np.load(folder_address + str(int(T_hat_try)) + ".npz")['f_c_indi']
    #     f_c_indi = np.average(f_c_indi_mat, axis=0)
    #     ax.plot(x, f_c_indi[0] / f_c_benchmark[0], color='navy', linewidth=2, linestyle= line_styles[i], label=f'Learning window = {T_hat_try}')
    # ax.plot(x, f_c_benchmark[0] / f_c_benchmark[0], color='gray', linewidth=2, linestyle='dashed', label=r'Benchmark OLG economy')
    # plt.legend(loc='lower right')
    # ax.tick_params(axis='y', labelcolor='black')
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.savefig('Welfare_1.png', dpi=100)
    # plt.show()
    # plt.close()







