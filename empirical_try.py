import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI, simulate_mix_types
from src.param import nu, mu_Y, sigma_Y, \
    dt, Ninit, Nc, Nt, tau, \
    cutoffs_age, scenarios, Ntype, alpha_i, \
    dZ_build_matrix, dZ_SI_build_matrix, dZ_SI_matrix, dZ_matrix, \
    cohort_type_size, cohort_size, rho_i
from src.param_mix import Nconstraint, rho_i_mix
from concurrent.futures import ProcessPoolExecutor

# run this on a grid of parameters & type densities & signal
# data_shocks = pd.read_excel(r'E:/Users/A2010290/Documents/GitHub/NoShort/realized_shocks.xlsx', sheet_name='Sheet1',
#                             index_col=0)
# dZ_SI_raw = data_shocks.to_numpy()[:, 1]
# Nt = dZ_SI_raw.size - np.count_nonzero(np.isnan(dZ_SI_raw))
# Nall = dZ_SI_raw.size
# dZ_SI = dZ_SI_raw[-Nt:]
# dZ = data_shocks.to_numpy()[:, 0][-Nt:]
data_shocks = pd.read_excel(r'E:/Users/A2010290/Documents/GitHub/NoShort/germany_realized_shocks.xlsx',
                            sheet_name='Sheet1',
                            index_col=0)
# data_shocks = pd.read_excel(r'E:/Users/A2010290/Documents/GitHub/NoShort/finland_realized_shocks.xlsx',
#                             sheet_name='Sheet1',
#                             index_col=0)
# data_shocks = pd.read_excel(r'E:/Users/A2010290/Documents/GitHub/NoShort/norway_realized_shocks.xlsx', sheet_name='Sheet1',
#                             index_col=0)
dZ_actual = data_shocks.to_numpy()[:, 0]
# Nt = dZ_actual.size
Nt_data = dZ_actual.size

plt.rcParams["font.family"] = 'serif'

# (complete, excluded, disappointment, reentry)
density_set = [
    # (0.0, 0.0, 0.0, 1.0),
    # (0.0, 0.0, 1.0, 0.0),
    (0.25, 0.25, 0.25, 0.25),
    # (0.25, 0.25, 0.0, 0.5),
    # (0.25, 0.25, 0.5, 0.0),
    # (0.5, 0.25, 0.0, 0.25),
    # (0.1, 0.1, 0.0, 0.8),
    # (0.1, 0.1, 0.4, 0.4),
]
n_scenarios_short = len(density_set)
scenarios_short = scenarios[1:3]

# T_hat_set = [1, 2, 3,
#              5, 10, 20
#              ]  # Pre-learning window
T_hat_set = [1, 2, 5, 10
             ]  # Pre-learning window
# phi_set = [0.0, 0.2, 0.4,
#            0.6, 0.8
#            ]
# phi_set = [0.0, 0.4, 0.8
#            ]
# phi_set = [0.0, 0.4]
phi_set = [0.0]
tax_set = [0.3, 0.5, 0.7]
# tax_set = [0.2, 0.3,
#            0.5, 0.6, 0.8
#            ]
n_T_hat = len(T_hat_set)
n_phi = len(phi_set)
n_tax = len(tax_set)

# # for testing:
# Mpath = 160
Mpath = 32
np.seterr(invalid='ignore')


# noinspection PyTypeChecker
def simulate_path(
        i: int,
):
    print(i)
    # shocks
    dZ_build = dZ_build_matrix[i]
    dZ = dZ_matrix[i]
    dZ[-Nt_data:] = dZ_actual
    dZ_SI_build = dZ_SI_build_matrix[i]
    dZ_SI = dZ_SI_matrix[i]

    popu_parti_compare = np.empty((n_T_hat, n_phi, n_tax, n_scenarios_short, Nt), dtype=np.float32)
    # popu_parti_old = np.empty((n_scenarios_short, Nt), dtype=np.float32)
    # popu_parti_young = np.empty((n_scenarios_short, Nt), dtype=np.float32)
    # r_compare = np.zeros((n_scenarios_short, Nt), dtype=np.float32)
    dR_compare = np.empty((n_T_hat, n_phi, n_tax, n_scenarios_short, Nt), dtype=np.float32)
    # mu_S_compare = np.zeros((n_scenarios_short, Nt), dtype=np.float32)
    # sigma_S_compare = np.zeros((n_scenarios_short, Nt), dtype=np.float32)
    # pd_compare = np.zeros((n_scenarios_short, Nt), dtype=np.float32)
    # average_belief_compare = np.zeros((n_scenarios_short, Nt), dtype=np.float32)
    popu_reenter_compare = np.empty((n_T_hat, n_phi, n_tax, n_scenarios_short, 5, Nt), dtype=np.float32)
    popu_exit_compare = np.empty((n_T_hat, n_phi, n_tax, n_scenarios_short, Nt), dtype=np.float32)

    # parti_df = pd.DataFrame(data_shocks.index.astype(str), columns=['yyyymm'])
    parti_df = pd.DataFrame(data_shocks.index.astype(str), columns=['Yearmon'])

    for a, T_hat in enumerate(T_hat_set):
        print(a)
        Npre = int(T_hat / dt)
        Vhat = (sigma_Y ** 2) / T_hat  # prior variance
        for b, phi in enumerate(phi_set):
            for c, tax in enumerate(tax_set):
                beta_i = (nu + rho_i) / (1 + tax)  # consumption wealth ratio
                beta0 = np.sum(alpha_i * beta_i).astype(float)
                rho_cohort_type = alpha_i * beta_i * np.exp(-(rho_i + nu) * tau)  # shape(2, 6000)
                for g in range(n_scenarios_short):
                    col_name = str(T_hat) + '_' + str(phi) + '_' + str(tax) + '_' + str(g)
                    # if g <= 1:
                    if g < 0:
                        entry = exit = 0
                        scenario = scenarios_short[g]
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
                        # average_belief = np.average(Delta, axis=1, weights=cohort_size[0])

                    else:
                        alpha_constraint = np.ones(
                            (1, Nconstraint)) * density_set[g]
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
                            entry,
                            exit
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
                        # average_belief = np.average(np.average(Delta, axis=2, weights=cohort_size[0]), axis=1,
                        #                             weights=alpha_constraint[0])
                    # r_compare[g] = r
                    popu_parti_compare[a, b, c, g] = parti
                    # popu_parti_old[g] = parti_age_group[:, 3]
                    # popu_parti_young[g] = parti_age_group[:, 0]
                    # average_belief_compare[g] = average_belief
                    dR_compare[a, b, c, g] = dR
                    # mu_S_compare[g] = mu_S
                    # sigma_S_compare[g] = sigma_S
                    # pd_compare[g] = 1 / beta

                    # parti_all = np.zeros(Nall)
                    # entry_all = np.zeros(Nall)
                    # exit_all = np.zeros(Nall)

                    # parti_all[-Nt:] = parti
                    # entry_all[-Nt:] = entry
                    # exit_all[-Nt:] = exit

                    # old vs. young: experience gap, belief gap, and participation rate gap
                    # cohort_belief = np.average(Delta, axis=1, weights=alpha_constraint[0])
                    # average_belief_old = np.average(cohort_belief[:, :cutoffs_age[3]], axis=1,
                    #                                 weights=cohort_size[0, :cutoffs_age[3]])
                    # average_belief_young = np.average(cohort_belief[:, cutoffs_age[1]:], axis=1,
                    #                                   weights=cohort_size[0, cutoffs_age[1]:])
                    parti_old = parti_age_group[:, 3]
                    parti_young = parti_age_group[:, 0]
                    # parti_age_compare = np.zeros((2, N_sample))

                    # Calculate the fraction of investors re-entering stock market
                    # follow the exiting cohorts for 5 years and calculate the fraction of them re-entering the stock market
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
                    # popu_exit_compare[a, b, c, g] = exit

                    parti_df['parti' + col_name] = parti[-Nt_data:].astype(np.float32)
                    # parti_df['belief_old' + col_name] = average_belief_old[-Nt_data:].astype(np.float32)
                    # parti_df['belief_young' + col_name] = average_belief_young[-Nt_data:].astype(np.float32)
                    parti_df['parti_old' + col_name] = parti_old[-Nt_data:].astype(np.float32)
                    parti_df['parti_young' + col_name] = parti_young[-Nt_data:].astype(np.float32)

                    # parti_df['parti' + col_name] = parti_all.astype(np.float32)
                    parti_df['entry' + col_name] = entry[-Nt_data:].astype(np.float32)
                    parti_df['exit' + col_name] = exit[-Nt_data:].astype(np.float32)

    parti_df.to_stata('stata_dataset/' + str(i) + 'deu_parti.dta')
    # parti_df.to_stata('stata_dataset/' + str(i) + 'fin_parti.dta')
    # parti_df.to_stata('stata_dataset/' + str(i) + '_parti.dta')
    # parti_df.to_stata('stata_dataset/' + str(i) + 'nor_parti.dta')
    return (
        i,
        popu_parti_compare,
        # popu_parti_old,
        # popu_parti_young,
        # r_compare,
        dR_compare,
        # mu_S_compare,
        # sigma_S_compare,
        # pd_compare,
        # average_belief_compare,
        # popu_reenter_compare,
        # popu_exit_compare,
    )


def main():
    # Create a ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=16) as executor:  # Adjust the number of workers as needed
        results = [executor.submit(simulate_path, i) for i in range(Mpath)]
    # Initialize a list to store the results
    results_list = []

    # Retrieve results from parallel processes
    for result in results:
        i, \
            popu_parti_result, \
            dR_result = result.result()
            # popu_reentry_result, \
            # popu_exit_result
        # popu_parti_old_result, \
        # popu_parti_young_result, \
        # r_result, \
        # dR_result, \
        # mu_S_result, \
        # sigma_S_result, \
        # pd_result, \
        # average_belief_result,

        data = {
            "i": i,
            "parti rate": popu_parti_result,
            # "parti rate old": popu_parti_old_result,
            # "parti rate young": popu_parti_young_result,
            # "interest rate": r_result,
            "dR": dR_result,
            # "expected return": mu_S_result,
            # "expected vola": sigma_S_result,
            # "price dividend ratio": pd_result,
            # "average belief": average_belief_result,
            # "popu_reentry": popu_reentry_result,
            # "popu_exit": popu_exit_result,
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a .npz file
    results_dict = results_df.to_dict(orient='list')
    np.savez("parti_rate_deu.npz", **results_dict)

    #### Finland: how many re-enter the stock market in a given year
    results_df = np.load('parti_rate_fin.npz')
    data_shocks = pd.read_excel(r'E:/Users/A2010290/Documents/GitHub/NoShort/finland_realized_shocks.xlsx',
                                sheet_name='Sheet1',
                                index_col=0)
    dZ_actual = data_shocks.to_numpy()[:, 0]
    Nt_data = dZ_actual.size
    parti_df = pd.DataFrame(data_shocks.index.astype(str), columns=['Yearmon'])
    reentry_mat = np.average(results_df['popu_reentry'][:, 1, 0, 0, 0], axis=0)
    exit_mat = np.average(results_df['popu_exit'][:, 1, 0, 0, 0], axis=0)
    parti_df['exit'] = exit_mat[-Nt_data:].astype(np.float32)
    for j in range(5):
        parti_df['reentry'+str(j)] = reentry_mat[j, -Nt_data:].astype(np.float32)
    parti_df.to_stata('stata_dataset/fin_reentry.dta')

    # # plot:
    # results_df = np.load('parti rate real shocks.npz')
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
    #     # plt.show()
    #     # plt.close()


# results_df = np.load('parti rate.npz')
# parti_rate = results_df["parti rate"]
# dR = results_df["dR"]
# phi = 0.4
# dSI = phi * dZ + np.sqrt(1 - phi ** 2) * dZ_SI
# shocks_list = [dZ, dZ_SI, dSI]
# corr_matrix_dZ = np.zeros((Mpath, n_T_hat, n_tax, n_scenarios_short, 3))
# corr_matrix_dZ_SI = np.zeros((Mpath, n_T_hat, n_tax, n_scenarios_short, 3))
# corr_matrix_dSI = np.zeros((Mpath, n_T_hat, n_tax, n_scenarios_short, 3))
# corr_list = [corr_matrix_dZ, corr_matrix_dZ_SI, corr_matrix_dSI]
# for a in range(Mpath):
#     for b in range(n_T_hat):
#         for c in range(n_tax):
#             for d in range(n_scenarios_short):
#                 for e, shocks in enumerate(shocks_list):
#                     corr_list[e][a, b, c, d, 1] = np.corrcoef(shocks[:-24], dR[a, b, 1, c, d, 24:])[0, 1]
#                     corr_list[e][a, b, c, d, 2] = np.corrcoef(shocks[:-60], dR[a, b, 1, c, d, 60:])[0, 1]
#                     corr_list[e][a, b, c, d, 0] = np.corrcoef(shocks[:-12], dR[a, b, 1, c, d, 12:])[0, 1]
#                     # corr_matrix1[a, b, c, d] = corr1
#
# corr_dZ = np.average(corr_matrix_dZ, axis=0)
# corr_dZ_SI = np.average(corr_matrix_dZ_SI, axis=0)
# corr_dSI = np.average(corr_matrix_dSI, axis=0)

if __name__ == '__main__':
    main()
