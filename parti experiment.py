import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI, simulate_mix_types
from src.param import nu, mu_Y, sigma_Y, phi, \
    dt, Ninit, Nt, Nc, tau, \
    cutoffs_age, scenarios, dZ_Y_cases, dZ_SI_cases, dZ_build_case, \
    dZ_SI_build_case, t, scenario_labels, colors_short, Ntype, alpha_i, \
    dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    cohort_type_size, cohort_size, rho_i
from src.param_mix import Nconstraint, beta_i_mix, rho_i_mix
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

## scatter plots for expected returns, vola, excess returns and r as function of participation
## maybe no linear relation between parti and vola, but parti and r

plt.rcParams["font.family"] = 'serif'

T_hat = 5  # Pre-trading period
Npre = int(T_hat / dt)
Vhat = (sigma_Y ** 2) / T_hat  # prior variance

# (complete, excluded, disappointment, reentry)
density_set = [
    (0.0, 0.0, 0.0, 1.0),
    (0.0, 0.0, 1.0, 0.0),
    (0.25, 0.25, 0.25, 0.25),
    (0.25, 0.25, 0.0, 0.5),
    (0.25, 0.25, 0.5, 0.0),
    (0.1, 0.1, 0.0, 0.8),
    (0.1, 0.1, 0.8, 0.0),
]
n_scenarios_short = len(density_set)
scenarios_short = scenarios[1:3]
# tax = 0.3    # marginal rate of consumption tax
tax = 0.5
beta_i = (nu + rho_i) / (1 + tax)  # consumption wealth ratio
beta0 = np.sum(alpha_i * beta_i).astype(float)

rho_cohort_type = alpha_i * beta_i * np.exp(-(rho_i + nu) * tau)  # shape(2, 6000)
beta_cohort = np.sum(np.exp(-beta_i * tau) * alpha_i, axis=0)

Mpath = 30
# print('Generating data for the graphs:')
window = 24  # 2-year non-overlapping windows
N_data_point = int(Nt / window - 1)

# # for testing:
# Mpath = 10
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

    popu_parti_compare = np.empty((n_scenarios_short, N_data_point), dtype=np.float32)
    popu_parti_old = np.empty((n_scenarios_short, N_data_point), dtype=np.float32)
    popu_parti_young = np.empty((n_scenarios_short, N_data_point), dtype=np.float32)
    r_compare = np.zeros((n_scenarios_short, N_data_point), dtype=np.float32)
    dR_compare = np.zeros((n_scenarios_short, N_data_point), dtype=np.float32)
    mu_S_compare = np.zeros((n_scenarios_short, N_data_point), dtype=np.float32)
    sigma_S_compare = np.zeros((n_scenarios_short, N_data_point), dtype=np.float32)
    vola_S_compare = np.zeros((n_scenarios_short, N_data_point), dtype=np.float32)
    pd_compare = np.zeros((n_scenarios_short, N_data_point), dtype=np.float32)
    average_belief_compare = np.zeros((n_scenarios_short, N_data_point), dtype=np.float32)

    for g in range(n_scenarios_short):
        if g <= 1:
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
            average_belief = np.average(Delta, axis=1, weights=cohort_size[0])

        else:
            alpha_constraint = np.ones(
                (1, Nconstraint)) * density_set[g]
            alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
            cohort_type_size_mix = cohort_size * alpha_i_mix
            # cohort_size_mix = np.sum(cohort_type_size_mix, axis=0)
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
            average_belief = np.average(np.average(Delta, axis=2, weights=cohort_size[0]), axis=1, weights=alpha_constraint[0])
        R_cumu = np.cumsum(dR)
        R2_cumu = np.cumsum(dR ** 2)
        vola_cumu = np.cumsum(sigma_S ** 2)
        r_cumu = np.cumsum(r)
        mu_S_cumu = np.cumsum(mu_S)
        r_compare[g] = np.reshape((r_cumu[window:] - r_cumu[:-window]) / window, (-1, window))[:, 0]
        popu_parti_compare[g] = np.reshape(parti, (-1, window))[:-1, 0]
        popu_parti_old[g] = np.reshape(parti_age_group[:, 3], (-1, window))[:-1, 0]
        popu_parti_young[g] = np.reshape(parti_age_group[:, 0], (-1, window))[:-1, 0]
        # average_belief = np.average(np.average(Delta, axis=2, weights=cohort_size[0]), axis=1,
        #                             weights=alpha_constraint[0])
        average_belief_compare[g] = np.reshape(average_belief, (-1, window))[:-1, 0]
        dR_compare[g] = np.reshape((R_cumu[window:] - R_cumu[:-window]) / window, (-1, window))[:, 0]
        vola_S_compare[g] = np.reshape((R2_cumu[window:] - R2_cumu[:-window]) / window, (-1, window))[:, 0]
        mu_S_compare[g] = np.reshape((mu_S_cumu[window:] - mu_S_cumu[:-window]) / window, (-1, window))[:, 0]
        sigma_S_compare[g] = np.reshape(np.sqrt((vola_cumu[window:] - vola_cumu[:-window]) / window),
                                           (-1, window))[
                                :, 0]
        pd_compare[g] = np.reshape(1 / beta, (-1, window))[1:, 0]
    window_Z = 60
    dZ_total = np.empty(int(Nt + window_Z))
    dZ_total[:window_Z] = dZ_build[-window_Z:]
    dZ_total[window_Z:] = dZ
    Z_cumu = np.cumsum(dZ_total)
    recent_shocks_compare = np.reshape((Z_cumu[window_Z:] - Z_cumu[:-window_Z]) / window_Z, (-1, window))[:-1, 0]
    ## todo: when is the slope upward and when is downward?

    return (
        i,
        popu_parti_compare,
        popu_parti_old,
        popu_parti_young,
        r_compare,
        dR_compare,
        mu_S_compare,
        sigma_S_compare,
        vola_S_compare,
        pd_compare,
        average_belief_compare,
        recent_shocks_compare,
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
        popu_parti_old_result, \
        popu_parti_young_result, \
        r_result, \
        dR_result, \
        mu_S_result, \
        sigma_S_result, \
        vola_S_result, \
        pd_result, \
        average_belief_result, \
        recent_shocks_result, = result.result()

        data = {
            "i": i,
            "parti rate": popu_parti_result,
            "parti rate old": popu_parti_old_result,
            "parti rate young": popu_parti_young_result,
            "interest rate": r_result,
            "dR": dR_result,
            "expected return": mu_S_result,
            "expected vola": sigma_S_result,
            "stock vola": vola_S_result,
            "price dividend ratio": pd_result,
            "average belief": average_belief_result,
            "recent shocks": recent_shocks_result,
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a .npz file
    results_dict = results_df.to_dict(orient='list')
    np.savez("parti rate.npz", **results_dict)

    # plot:
    results_df = np.load('parti rate.npz')
    y_set = [[
        results_df["interest rate"],
        results_df["dR"],
        results_df["expected return"]
    ],
        [results_df["expected vola"],
         results_df["stock vola"],
         results_df["price dividend ratio"],
         ]]
    x_set = [
        results_df["parti rate"],
        # popu_parti_old / popu_parti_compare,
        # popu_parti_young / popu_parti_compare,
    ]
    label_x_set = [
        r'Participation rate, $P_t$',
        # r'Participation rate among the oldest quartile',
        # r'Oldest quartile / total',
        # # r'Participation rate among the youngest quartile',
        # r'Youngest quartile / total',
    ]
    label_set = [[
        r'Cumulated interest rate, $r_{t,t+2}$',
        r'Realized stock returns, $dR_{t,t+2}$',
        r'Expected returns, $\mu^S_{t,t+2}$'
    ], [
        r'Average diffusion term, $\sigma^S_{t,t+2}$',
        r'Average volatility, $E[dR^2_{t,t+2}]$',
        r'Price dividend ratio, $pd_{t+2}$'
    ],
    ]
    condi = results_df["parti rate young"] / results_df["parti rate"] / 4
    labels = ['Mostly old', 'Mostly young']
    # condi = results_df["average belief"]
    # labels = ['Overall pessimism', 'Overall optimism']
    # condi = results_df["recent shocks"]
    # labels = ['Negative shocks', 'Positive shocks']

    for ii, x in enumerate(x_set):
        fig, axes = plt.subplots(nrows=n_scenarios_short, ncols=1,
                                 # sharex='all', sharey='all',
                                 figsize=(4, 20))
        j = 1
        i = 0
        for jj, ax in enumerate(axes):
            condi_sce = condi[:, jj]
            # condi_sce = condi
            # condi_thres = np.percentile(condi_sce, [0, 10, 90, 100])
            condi_thres = np.percentile(condi_sce, [0, 25, 50, 75, 100])
            ax.grid(True)
            for k in range(int(len(condi_thres) - 1)):
                condi_ab = condi_sce >= condi_thres[k]
                condi_bl = condi_sce < condi_thres[k + 1]
                condi_in = np.where(condi_bl * condi_ab == 1)
                y = y_set[j][i][:, jj][condi_in]
                if k == 0:
                    label_k = labels[0]
                    ax.scatter(x[:, jj][condi_in], y, s=1, marker='.', c=colors_short[k], label=label_k)
                elif k == int(len(condi_thres) - 2):
                    label_k = labels[1]
                    ax.scatter(x[:, jj][condi_in], y, s=1, marker='.', c=colors_short[k], label=label_k)
                else:
                    ax.scatter(x[:, jj][condi_in], y, s=1, marker='.', c=colors_short[k])
            ax.set_xlabel(label_x_set[ii])
            ax.set_ylabel(r'$\sigma^S_{t,t+2}$')
            ax.set_title(str(density_set[jj]))
            ax.legend()
        fig.tight_layout(h_pad=2)
        plt.savefig('1quartiles_' + 'learn' + str(T_hat) + '_tax' + str(tax) + '_scatter.png', dpi=200)
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


