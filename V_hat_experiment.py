import time
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI
from src.param import Npres, rho, nu, mu_Y, sigma_Y, sigma_Y_sqr, sigma_S, v, tax, \
    beta, dt, Ninit, Nt, Nc, tau, cohort_size, n_age_groups,\
    cutoffs, phi_vector, n_phi, colors, modes_trade, modes_learn, scenarios, \
    dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix

# todo: the connection between belief and wealth?
#  Learning from repeated negative economic shocks: lead to both worse wealth condition and pessimism

T_hats = dt * Npres
T_hat_dimension = len(T_hats)
N = 30  # can choose a smaller number than Mpaths as the number of paths
# n_scenarios = 3
# scenarios_short = scenarios[:n_scenarios]

# scenarios_short = scenarios[3:5]
scenarios_short = scenarios[5:]
n_scenarios = len(scenarios_short)

# # Generate matrix to store the results
# r_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))  # for mean and std
# theta_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
# f_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
# popu_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
# Delta_bar_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
# popu_age_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, n_age_groups, 2))
# belief_age_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, n_age_groups, 2))
# wealthshare_age_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, n_age_groups, 2))
#
#
# for l in range(N):
#     print(l)
#     dZ = dZ_matrix[l]
#     dZ_build = dZ_build_matrix[l]
#     dZ_SI = dZ_SI_matrix[l]
#     dZ_SI_build = dZ_SI_build_matrix[l]
#     time_s = time.time()
#     for m, scenario in enumerate(scenarios_short):
#         mode_trade = scenario[0]
#         mode_learn = scenario[1]
#         for n, phi in enumerate(phi_vector):
#             for o, T_hat in enumerate(T_hats):
#                 Npre = int(Npres[o])
#                 Vhat = (sigma_Y ** 2) / T_hat  # prior variance
#                 print(T_hat, Npre, Vhat)
#
#                 (
#                     r,
#                     theta,
#                     f,
#                     Delta,
#                     max,
#                     pi,
#                     popu_parti,
#                     f_parti,
#                     Delta_bar_parti,
#                     dR,
#                     w,
#                     age_parti,
#                     n_parti,
#                 ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta,
#                                 phi,
#                                 Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
#                                 top=0.05,
#                                 old_limit=100
#                                 )
#
#                 r_matrix[l, m, n, o, 0] = np.mean(r)
#                 r_matrix[l, m, n, o, 1] = np.std(r)  # todo: maybe adjust the std level against mean later
#                 theta_matrix[l, m, n, o, 0] = np.mean(theta)
#                 theta_matrix[l, m, n, o, 1] = np.std(theta)
#                 popu_parti_matrix[l, m, n, o, 0] = np.mean(popu_parti)
#                 popu_parti_matrix[l, m, n, o, 1] = np.std(popu_parti)
#                 Delta_bar_parti_matrix[l, m, n, o, 0] = np.mean(Delta_bar_parti)
#                 Delta_bar_parti_matrix[l, m, n, o, 1] = np.std(Delta_bar_parti)
#                 f_parti_matrix[l, m, n, o, 0] = np.mean(f_parti)
#                 f_parti_matrix[l, m, n, o, 1] = np.std(f_parti)
#                 invest = (pi > 0)
#                 parti_rate = invest * cohort_size
#
#                 belief = (Delta * sigma_Y + mu_Y)
#                 belief_weights = f * dt
#
#                 for i in range(4):
#                     popu_age_matrix[l, m, n, o, i, 0] = np.mean(np.sum(parti_rate[:, cutoffs[i + 1]:], axis=1))
#                     popu_age_matrix[l, m, n, o, i, 1] = np.std(np.sum(parti_rate[:, cutoffs[i + 1]:], axis=1))
#
#                     belief_age_matrix[l, m, n, o, i, 0] = np.mean(
#                         np.average(
#                             belief[:, cutoffs[i + 1]:cutoffs[i]], weights=cohort_size[cutoffs[i + 1]:cutoffs[i]], axis=1
#                         ))
#                     belief_age_matrix[l, m, n, o, i, 1] = np.std(
#                         np.average(
#                             belief[:, cutoffs[i + 1]:cutoffs[i]], weights=cohort_size[cutoffs[i + 1]:cutoffs[i]], axis=1
#                         ))
#
#                     wealthshare_age_matrix[l, m, n, o, i, 0] = np.mean(
#                         np.sum(f[:, cutoffs[i + 1]:cutoffs[i]] * dt, axis=1)
#                     )
#                     wealthshare_age_matrix[l, m, n, o, i, 1] = np.std(
#                         np.sum(f[:, cutoffs[i + 1]:cutoffs[i]] * dt, axis=1)
#                     )
#
#     print(time.time() - time_s)


# graphs:
var_list = [r_matrix, theta_matrix, f_parti_matrix,
            Delta_bar_parti_matrix, popu_age_matrix,
            belief_age_matrix, wealthshare_age_matrix]
var_name_list = ['interest rate', 'market price of risk', 'consumption share of participants',
                 'consumption-weighted estimation error of participants', 'participation rate',
                 'average belief in age groups', 'wealth share in age groups']
type_list = ['mean', 'vola']
age_labels = ['20 < Age <= 35, youngest quartile', '35 < Age <= 55', '55 < Age <= 89', 'Age > 89, oldest quartile']

x = Npres

for i, var in enumerate(var_list):
    y_mat = np.mean(var, axis=1)
    var_name = var_name_list[i]
    for j, scenario in enumerate(scenarios_short):
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        if i <= 3:
            for k, type in enumerate(type_list):
                fig, ax = plt.subplots()  # consider include Vhat in the graphs on the LHS axis
                for l, phi in enumerate(phi_vector):
                    y = y_mat[j, l, :, k]
                    ax.plot(x, y, label = str(round(phi, 2)))
                ax.set_ylabel(var_name)
                ax.set_xlabel('initial window (months)')
                ax.legend()
                plt.savefig(type + ' initial window and ' + var_name + '_' + mode_learn + '_' + mode_trade + '.png', dpi=500, format="png")
                plt.show()
                plt.close()
        elif i == 4:
            for k, type in enumerate(type_list):
                for l, phi in enumerate(phi_vector):
                    fig, ax = plt.subplots()  # consider include Vhat in the graphs on the LHS axis
                    y = y_mat[j, l, :, :, k]
                    for m in range(n_age_groups):
                        y_age_group = y[:, m]
                        if m == 0:
                            ax.fill_between(x, y_age_group, color=colors[m], linewidth=0.4,
                                        label=age_labels[m])
                        else:
                            y_age_group_1 = y[:, m-1]
                            ax.fill_between(x, y_age_group, y_age_group_1, color=colors[m], linewidth=0.4,
                                        label=age_labels[m])
                    plt.legend()
                    ax.set_ylabel(var_name)
                    ax.set_xlabel('initial window (months)')
                    ax.legend()
                    plt.savefig(type + ' initial window and ' + var_name + '_' + mode_learn + '_' + mode_trade + '.png',
                                dpi=500, format="png")
                    plt.show()
                    plt.close()
        else:
            for k, type in enumerate(type_list):
                for l, phi in enumerate(phi_vector):
                    fig, ax = plt.subplots()  # consider include Vhat in the graphs on the LHS axis
                    y = y_mat[j, l, :, :, k]
                    for m in range(n_age_groups):
                        y_age_group = y[:, m]
                        if m == 0:
                            ax.plot(x, y_age_group, color=colors[m], linewidth=0.4,
                                            label=age_labels[m])
                        else:
                            y_age_group_1 = y[:, m - 1]
                            ax.plot(x, y_age_group, color=colors[m], linewidth=0.4,
                                            label=age_labels[m])
                    plt.legend()
                    ax.set_ylabel(var_name)
                    ax.set_xlabel('initial window (months)')
                    ax.legend()
                    plt.savefig(type + ' initial window and ' + var_name + '_' + mode_learn + '_' + mode_trade + '.png',
                                dpi=500, format="png")
                    plt.show()
                    plt.close()

