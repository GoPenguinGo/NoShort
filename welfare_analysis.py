import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI
from src.param import mu_Y, sigma_Y, \
    dt, Ninit, Nc, Nt, tau, \
    cutoffs_age, Ntype, alpha_i, \
    dZ_build_matrix, dZ_SI_build_matrix, dZ_SI_matrix, dZ_matrix, \
    cohort_size, rho_i, tax, beta0, beta_i, nu, Vhat, phi, Npre, \
    T_hat, rho_cohort_type, cohort_type_size
from concurrent.futures import ProcessPoolExecutor

a_rho_bar = 1
b_rho_bar = -(tax * beta0 + rho_i[0, 0] + rho_i[1, 0])
c_rho_bar = alpha_i[0, 0] * tax * beta_i[0, 0] * (rho_i[0, 0] - rho_i[1, 0]) + rho_i[1, 0] * rho_i[0, 0] + tax * beta0 * \
            rho_i[1, 0]
rho_bar = (-b_rho_bar - np.sqrt(b_rho_bar ** 2 - 4 * a_rho_bar * c_rho_bar)) / (2 * a_rho_bar)
T = 500
N_T = int(T / dt)
mode_trade = "w_constraint"
# mode_trade = "complete"
mode_learn = 'reentry'
Mpath = 2000
Nt_long = 8400
T_hat_vec = np.append(
    np.arange(1, 4, 1),
    np.arange(4, 30, 3),
)
t_s_mat = np.tile(np.reshape(np.cumsum(np.ones(N_T) * dt) - dt, (-1, 1)), (1, 2))
rho_i_mat = np.reshape(rho_i, (1, -1))
discount_rate_mat = np.exp(-(nu + rho_i_mat) * t_s_mat)
folder_address = r'E:\Users\A2010290\Documents\GitHub\NoShort/welfare/'
plt.rcParams["font.family"] = 'serif'

def utility_mu(mu_Y_use):
    t_s = np.cumsum(np.ones(N_T) * dt) - dt
    E_log_C = (mu_Y_use + rho_bar - rho_i + nu - beta0 * nu - 1 / 2 * sigma_Y ** 2) * t_s * dt
    discount_rate = np.exp(-(nu + rho_i) * t_s)
    E_util = np.sum(E_log_C * discount_rate, axis=1)
    return E_util


def simulate_path(
        i: int,
        T_hat_use,
):
    print(i)
    # shocks
    dZ_build = dZ_build_matrix[i]
    dZ = np.random.randn(Nt_long) * np.sqrt(dt)
    dZ_SI_build = dZ_SI_build_matrix[i]
    dZ_SI = np.random.randn(Nt_long) * np.sqrt(dt)

    Npre_use = int(T_hat_use / dt)
    Vhat_use = (sigma_Y ** 2) / T_hat_use  # prior variance

    log_C_mat = np.zeros((10, int(T / dt), 2), dtype=np.float32)
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
                    Nt_long,
                    dt,
                    nu,
                    Vhat_use,
                    mu_Y,
                    sigma_Y,
                    tax,
                    beta0,
                    phi,
                    Npre_use,
                    Ninit,
                    T_hat_use,
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
                    need_Delta='False',
                    need_pi='False',
                    )
    C_mat = np.exp((mu_Y - 1/2 * sigma_Y ** 2) * np.arange(0, Nt_long) * dt + sigma_Y * np.cumsum(dZ))
    C_mat_reshape = np.tile(np.reshape(C_mat, (-1, 1, 1)), (1, 2, 6000))
    C_matrix = f_c * C_mat_reshape / cohort_type_size * dt
    sample = np.arange(0, N_T, 1)

    del C_mat_reshape

    for j in range(10):
        t_start = int(100 / dt) + int(10 / dt) * j
        log_C_mat[j] = np.log(C_matrix[t_start + sample, :, Nt - 1 - sample] / C_matrix[t_start, :, -1])

    # np.save(folder_address + str(i) + ".npy", np.average(log_C_mat, axis=0))
    E_util_path = np.sum(np.average(log_C_mat, axis=0) * dt * discount_rate_mat, axis=0)

    return (
        i,
        E_util_path
    )


def main():
    for T_hat_try in T_hat_vec[1:3]:
        with ProcessPoolExecutor(max_workers=25) as executor:  # Adjust the number of workers as needed
            results = [executor.submit(simulate_path, i, int(T_hat_try)) for i in range(Mpath)]
            # Initialize a list to store the results
        results_list = []

        # Retrieve results from parallel processes
        for result in results:
            i, \
            E_util_path_result = result.result()

            data = {
                "i": i,
                'E_util': E_util_path_result

            }
            results_list.append(data)

        # Create a DataFrame from the list of dictionaries
        results_df = pd.DataFrame(results_list)
        results_dict = results_df.to_dict(orient='list')
        np.savez(folder_address + str(int(T_hat_try)) + ".npz", **results_dict)


if __name__ == '__main__':
    main()

    # N_points = 200
    # mu_vec = np.linspace(0, mu_Y, N_points)
    # E_util_mu = np.zeros((N_points, 2))
    # for i, mu_try in enumerate(mu_vec):
    #     E_util_mu[i] = utility_mu(mu_try)
    #
    # t_s_mat = np.tile(np.reshape(np.cumsum(np.ones(N_T) * dt) - dt, (-1, 1)), (1, 2))
    # rho_i_mat = np.reshape(rho_i, (1, -1))
    # discount_rate_mat = np.exp(-(nu + rho_i_mat) * t_s_mat)
    #
    # equiv_mu = np.zeros((len(T_hat_vec), 2))
    #
    # for i, T_hat_try in enumerate(T_hat_vec):
    #     E_util_learn_path = np.load(folder_address + str(int(T_hat_try)) + ".npz")['E_util']
    #     E_util_learn = np.average(np.ma.masked_invalid(E_util_learn_path), axis=0)
    #     for j in range(2):
    #         equiv_mu[i, j] = mu_vec[np.searchsorted(E_util_mu[:, j], E_util_learn[j])]
    #
    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 4))
    # ax.set_xlabel('Pre-entry learning window')
    # ax.set_ylabel(r'Equivalent $\mu^Y$')
    # ax.set_ylim(0, 0.025)
    # ax.plot(T_hat_vec, equiv_mu[:, 0], color='navy', linewidth=1.5, label=r'type 1, $\rho=0.001$')
    # ax.plot(T_hat_vec, equiv_mu[:, 1], color='red', linewidth=1.5, label=r'type 2, $\rho=-0.001$')
    # plt.axhline(y=0.02, color='gray', linestyle='dashed', label=r'Actual $\mu^Y$')
    # plt.legend()
    # ax.tick_params(axis='y', labelcolor='black')
    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.savefig('Welfare.png', dpi=100)
    # plt.show()
    # plt.close()





