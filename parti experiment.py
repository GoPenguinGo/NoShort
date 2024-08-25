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
import statsmodels.api as sm
import tabulate

## scatter plots for expected returns, vola, excess returns and r as function of participation
## maybe no linear relation between parti and vola, but parti and r

plt.rcParams["font.family"] = 'serif'

T_hat = 20  # Pre-trading period
Npre = int(T_hat / dt)
Vhat = (sigma_Y ** 2) / T_hat  # prior variance

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
n_scenarios_short = len(density_set)
# scenarios_short = scenarios[1:3]
tax = 0.3    # marginal rate of consumption tax
# tax = 0.5
beta_i = (nu + rho_i) / (1 + tax)  # consumption wealth ratio
beta0 = np.sum(alpha_i * beta_i).astype(float)

rho_cohort_type = alpha_i * beta_i * np.exp(-(rho_i + nu) * tau)  # shape(2, 6000)
beta_cohort = np.sum(np.exp(-beta_i * tau) * alpha_i, axis=0)

Mpath = 100
# print('Generating data for the graphs:')
window = 12  # 1-year non-overlapping windows
sample = np.arange(600, Nt - 600, window)
N_sample = len(sample)
# N_data_point = int(Nt / window - 1)

age_cutoffs_SCF = [int(Nt-1), int(Nt-1-12*15), int(Nt-1-12*35), int(Nt-1-12*55), 0]
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

    # popu_parti_compare = np.empty((n_scenarios_short, N_data_point), dtype=np.float32)
    # popu_parti_old = np.empty((n_scenarios_short, N_data_point), dtype=np.float32)
    # popu_parti_young = np.empty((n_scenarios_short, N_data_point), dtype=np.float32)
    # r_compare = np.zeros((n_scenarios_short, N_data_point), dtype=np.float32)
    # dR_compare = np.zeros((n_scenarios_short, N_data_point), dtype=np.float32)
    # mu_S_compare = np.zeros((n_scenarios_short, N_data_point), dtype=np.float32)
    # sigma_S_compare = np.zeros((n_scenarios_short, N_data_point), dtype=np.float32)
    # vola_S_compare = np.zeros((n_scenarios_short, N_data_point), dtype=np.float32)
    # pd_compare = np.zeros((n_scenarios_short, N_data_point), dtype=np.float32)
    # average_belief_compare = np.zeros((n_scenarios_short, N_data_point), dtype=np.float32)
    parti_compare = np.zeros((n_scenarios_short, N_sample), dtype=np.float32)
    parti_age_group_compare = np.zeros((n_scenarios_short, 5, N_sample), dtype=np.float32)
    annual_return_compare = np.zeros((n_scenarios_short, 6, N_sample), dtype=np.float32)
    pd_compare = np.zeros((n_scenarios_short, N_sample), dtype=np.float32)
    future_exc_R_compare = np.zeros((n_scenarios_short, N_sample), dtype=np.float32)
    entry_compare = np.zeros((n_scenarios_short, N_sample), dtype=np.float32)
    exit_compare = np.zeros((n_scenarios_short, N_sample), dtype=np.float32)
    # experience_age_compare = np.zeros((n_scenarios_short, 2, N_sample), dtype=np.float32)
    # average_belief_compare = np.zeros((n_scenarios_short, 2, N_sample), dtype=np.float32)
    # parti_age_compare = np.zeros((n_scenarios_short, 2, N_sample), dtype=np.float32)

    for g, density in enumerate(density_set):
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

        # save data relevant for regressions
        # non-overlapping data, take a sample every 5 years
        past_annual_return = np.zeros((3, Nt))
        future_annual_return = np.zeros((3, Nt))
        for n, gap in enumerate([12, 24, 36]):
            past_annual_return[n, gap:] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
            past_annual_return[n, :gap] = np.cumsum(dR[:gap]) / (gap / 12)
            future_annual_return[n, :-gap] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
            future_annual_return[n, -gap:] = (np.cumsum(dR)[-gap:]) / (gap / 12)
        annual_return_compare[g, :3] = past_annual_return[:, sample]
        annual_return_compare[g, 3:] = future_annual_return[:, sample]
        pd_compare[g] = np.copy(1 / beta)[sample]
        parti_compare[g] = parti[sample]
        future_exc_R_compare[g] = (future_annual_return[0] - r)[sample]
        entry_compare[g] = entry[sample]
        exit_compare[g] = exit[sample]
        age_parti = np.zeros((5, Nt))
        for n in range(len(age_cutoffs_SCF) - 1):
            age_parti[n] = np.average(
                np.average(invest_tracker[:, :, age_cutoffs_SCF[n+1]:age_cutoffs_SCF[n]], weights=cohort_size[0, age_cutoffs_SCF[n+1]:age_cutoffs_SCF[n]], axis=2),
            weights=density,
            axis=1)
        parti_age_group_compare[g] = age_parti[:, sample]

    return (
        i,
        parti_compare,
        parti_age_group_compare,
        annual_return_compare,
        pd_compare,
        future_exc_R_compare,
        entry_compare,
        exit_compare,
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
        parti_result, \
        parti_age_group_result, \
        annual_return_result, \
        pd_result, \
        future_exc_R_result, \
        entry_result, \
        exit_result = result.result()
        # parti_result, \
        # annual_return_result, \
        # monthly_minmax_result, \
        # pd_result = result.result()
        # i, \
        # popu_parti_result, \
        # popu_parti_old_result, \
        # popu_parti_young_result, \
        # r_result, \
        # dR_result, \
        # mu_S_result, \
        # sigma_S_result, \
        # vola_S_result, \
        # pd_result, \
        # average_belief_result, \
        # recent_shocks_result, = result.result()

        data = {
            "i": i,
            "participation rate": parti_result,
            "participation rate in age groups": parti_age_group_result,
            "annual stock return": annual_return_result,
            "pd ratio": pd_result,
            "future excess return": future_exc_R_result,
            "entry rate": entry_result,
            "exit rate": exit_result,

            # "participation rate": parti_result,
            # "annual stock return": annual_return_result,
            # "monthly min max return": monthly_minmax_result,
            # "pd ratio": pd_result,

            # "parti rate": popu_parti_result,
            # "parti rate old": popu_parti_old_result,
            # "parti rate young": popu_parti_young_result,
            # "interest rate": r_result,
            # "dR": dR_result,
            # "expected return": mu_S_result,
            # "expected vola": sigma_S_result,
            # "stock vola": vola_S_result,
            # "price dividend ratio": pd_result,
            # "average belief": average_belief_result,
            # "recent shocks": recent_shocks_result,
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)

    # Save the DataFrame to a .npz file
    results_dict = results_df.to_dict(orient='list')
    # np.savez("parti_rate_regressions.npz", **results_dict)
    np.savez("parti_Ch.npz", **results_dict)

    # # analysis:
    results_df = np.load('parti_Ch.npz')
    # regression 1: participation rate on returns and pd
    parti = np.copy(results_df["participation rate"])
    parti_age = np.copy(results_df["participation rate in age groups"])
    annual_return = np.copy(results_df["annual stock return"])
    f_return = np.copy(results_df["future excess return"])
    l_one_year_return = annual_return[:, :, 0]
    l_two_year_return = annual_return[:, :, 1]
    l_thr_year_return = annual_return[:, :, 2]
    pd_ratio = np.copy(results_df["pd ratio"])
    x_set = [
        l_one_year_return,
        l_two_year_return,
        l_thr_year_return,
    ]
    y = np.copy(parti_age)
    for sce in range(n_scenarios_short):
        label_scenario = 'Reentry' if sce == 0 else 'Mix - 4'
        reg_data = np.zeros((3, 4, Mpath))
        for i, x in enumerate(x_set):
            for j in range(4):
                for path in range(Mpath):
                    x_regress = sm.add_constant(x[path, sce])
                    model = sm.OLS(y[path, sce, j], x_regress)
                    est = model.fit()
                    reg_data[i, j, path] = est.params[1]
        print(label_scenario)
        print(tabulate.tabulate(np.average(reg_data, axis=2), floatfmt=".3f", tablefmt='latex_raw'))

    y = np.copy(f_return)
    x = parti_age
    for sce in range(n_scenarios_short):
        label_scenario = 'Reentry' if sce == 0 else 'Mix - 4'
        reg_data_uni = np.zeros((4, Mpath))
        reg_data_multi = np.zeros((4, Mpath))
        for path in range(Mpath):
            # corr_here = np.corrcoef(x[path, sce])[0, 1:]
            for age_group in range(4):
                x0 = x[path, sce, age_group]
                x_regress = sm.add_constant(x0)
                model = sm.OLS(y[path, sce], x_regress)
                est = model.fit()
                reg_data_uni[age_group, path] = est.params[1]
            x0 = x[path, sce, 0]
            x1 = x[path, sce, 1]
            x2 = x[path, sce, 2]
            x3 = x[path, sce, 3]
            x_multi = np.column_stack((x0, x1, x2, x3))
            x_regress = sm.add_constant(x_multi)
            model = sm.OLS(y[path, sce], x_regress)
            est = model.fit()
            reg_data_multi[:, path] = est.params[1:]

            # reg_data[:, path] = corr_here
        print(label_scenario)
        print(np.average(reg_data_uni, axis=1))
        print(np.average(reg_data_multi, axis=1))



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
