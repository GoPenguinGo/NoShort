import time
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI_mean_vola
from src.param import Npres, rho, nu, mu_Y, sigma_Y, sigma_Y_sqr, sigma_S, v, tax, \
    beta, dt, Ninit, Nt, Nc, tau, cohort_size, n_age_groups, \
    cutoffs, phi_vector, n_phi, colors, modes_trade, modes_learn, scenarios, \
    dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, top, old_limit

# todo: the connection between belief and wealth?
#  Learning from repeated negative economic shocks: lead to both worse wealth condition and pessimism

T_hats = dt * Npres
T_hat_dimension = len(T_hats)
# N = 30  # can choose a smaller number than Mpaths as the number of paths

n_scenarios = 1
a_sce = 2
N = 200

# scenarios_short = scenarios[1]
# n_scenarios = len(scenarios_short)

# scenarios_short = scenarios[2]
# n_scenarios = len(scenarios_short)

# scenarios_short = scenarios[3:5]
# n_scenarios = len(scenarios_short)

# scenarios_short = scenarios[5:]
# n_scenarios = len(scenarios_short)

# Generate matrix to store the results
r_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))  # for mean and std
theta_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
Phi_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
popu_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
Delta_bar_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
popu_age_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
belief_age_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
wealthshare_age_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))

# write a lighter version of the simulation function that returns only the desired values (mean and std, instead of whole raw data)
for l in range(N):
    print(l)
    dZ = dZ_matrix[l]
    dZ_build = dZ_build_matrix[l]
    dZ_SI = dZ_SI_matrix[l]
    dZ_SI_build = dZ_SI_build_matrix[l]
    time_s = time.time()
    for m in range(n_scenarios):
        scenario = scenarios[m+a_sce]
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        for n, phi_try in enumerate(phi_vector):
            for o, T_hat_try in enumerate(T_hats):
                Npre_try = int(Npres[o])
                Vhat_try = (sigma_Y ** 2) / T_hat_try  # prior variance
                (
                    r,
                    theta,
                    popu_parti,
                    Delta_bar_parti,
                    Phi_parti,
                    popu_age,
                    belief_age,
                    wealthshare_age,
                ) = simulate_SI_mean_vola(mode_trade, mode_learn, Nc, Nt, dt, rho, nu,
                                          Vhat_try,
                                          mu_Y, sigma_Y, sigma_S, tax, beta,
                                          phi_try,
                                          Npre_try,
                                          Ninit,
                                          T_hat_try,
                                          dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size, top, old_limit, cutoffs, n_age_groups,
                                          )
                r_matrix[l, m, n, o] = r
                theta_matrix[l, m, n, o] = theta
                popu_parti_matrix[l, m, n, o] = popu_parti
                Delta_bar_parti_matrix[l, m, n, o] = Delta_bar_parti
                Phi_parti_matrix[l, m, n, o] = Phi_parti
                popu_age_matrix[l, m, n, o] = popu_age
                belief_age_matrix[l, m, n, o] = belief_age
                wealthshare_age_matrix[l, m, n, o] = wealthshare_age
    print(time.time() - time_s)


# graphs:
var_list = [r_matrix, theta_matrix, Phi_parti_matrix, popu_parti_matrix,
            Delta_bar_parti_matrix, popu_age_matrix,
            belief_age_matrix, wealthshare_age_matrix]
var_name_list = ['interest rate', 'market price of risk', 'consumption share of participants', 'participation rate',
                 'consumption-weighted estimation error of participants', 'participation rate in age groups',
                 'average belief in age groups', 'wealth share in age groups']
type_list = ['mean', 'vola']
age_labels = ['20 < Age <= 35, youngest quartile', '35 < Age <= 55', '55 < Age <= 89', 'Age > 89, oldest quartile']

for i, var in enumerate(var_list):
    np.save(var_name_list[i] + str(a_sce), var)

# x = Npres
x = Npres[1:]
for i, var in enumerate(var_list):  # shape var: N * n_scenarios * n_phi * T_hat_dimension * 2
    y_mat = np.mean(var, axis=0)  # shape y_mat: n_scenarios * n_phi * T_hat_dimension * 2
    var_name = var_name_list[i]
    for j in range(n_scenarios):
        scenario = scenarios[j + a_sce]
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        if i <= 3:
            for k, type in enumerate(type_list):
                fig, ax = plt.subplots()  # consider include Vhat in the graphs on the LHS axis
                for l, phi in enumerate(phi_vector):
                    y = y_mat[j, l, :, k]  # shape y: T_hat_dimension
                    ax.plot(x, y[1:], label=str(round(phi, 2)))
                ax.set_ylabel(var_name)
                ax.set_xlabel('initial window (months)')
                ax.legend()
                mode_label = mode_trade if mode_trade == 'complete' else (mode_learn + '_' + mode_trade)
                #plt.savefig(type + ' initial window and ' + var_name + '_' + mode_label + '.png',
                            # dpi=500, format="png")
                plt.show()
                # plt.close()

        elif i == 4:
            for l, phi in enumerate(phi_vector):
                fig, ax = plt.subplots()  # consider include Vhat in the graphs on the LHS axis
                y = y_mat[j, l, :, :, 0]
                for m in range(n_age_groups):
                    y_age_group = y[:, m]
                    if m == 0:
                        ax.fill_between(x, y_age_group, color=colors[m], linewidth=0.4,
                                        label=age_labels[m])
                    else:
                        y_age_group_1 = y[:, m - 1]
                        ax.fill_between(x, y_age_group, y_age_group_1, color=colors[m], linewidth=0.4,
                                        label=age_labels[m])
                plt.legend()
                ax.set_ylabel(var_name)
                ax.set_xlabel('initial window (months)')
                ax.legend()
                mode_label = mode_trade if mode_trade == 'complete' else (mode_learn + '_' + mode_trade)
                #plt.savefig('mean initial window and ' + var_name + '_' + mode_label + '_' + str(
                    #round(phi, 2)) + '.png',
                            #dpi=500, format="png")
                plt.show()
                # plt.close()

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
                    mode_label = mode_trade if mode_trade == 'complete' else (mode_learn + '_' + mode_trade)
                    # plt.savefig(
                    #     type + ' initial window and ' + var_name + '_' + mode_label + '_' + str(round(phi, 2)) + '.png',
                    #     dpi=500, format="png")
                    plt.show()
                    # plt.close()
