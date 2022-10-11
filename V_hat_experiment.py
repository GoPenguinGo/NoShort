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
N_scenarios = 3
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


# # save the data
# var_list = [r_matrix, theta_matrix, Phi_parti_matrix, popu_parti_matrix,
#             Delta_bar_parti_matrix, popu_age_matrix,
#             belief_age_matrix, wealthshare_age_matrix]
# var_name_list = ['interest rate', 'market price of risk', 'consumption share of participants', 'participation rate',
#                  'consumption-weighted estimation error of participants', 'participation rate in age groups',
#                  'average belief in age groups', 'wealth share in age groups']
# type_list = ['mean', 'vola']
# age_labels = ['20 < Age <= 35, youngest quartile', '35 < Age <= 55', '55 < Age <= 89', 'Age > 89, oldest quartile']
#
# for i, var in enumerate(var_list):
#     np.save(var_name_list[i] + str(a_sce), var)

# read the data:
r_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))  # for mean and std
theta_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))
Phi_parti_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))
popu_parti_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))
Delta_bar_parti_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))
popu_age_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
belief_age_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
wealthshare_age_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
var_list = [r_Mat, theta_Mat, Phi_parti_Mat, popu_parti_Mat,
            Delta_bar_parti_Mat, popu_age_Mat,
            belief_age_Mat, wealthshare_age_Mat]
var_name_list = ['interest rate', 'market price of risk', 'consumption share of participants', 'participation rate',
                 'consumption-weighted estimation error of participants', 'participation rate in age groups',
                 'average belief in age groups', 'wealth share in age groups']
for i, var in enumerate(var_list):
    var_name = var_name_list[i]
    for j in range(N_scenarios):
        var_name_j = var_name + str(j) +'.npy'
        y = np.load(var_name_j)
        var[j] = np.mean(y, axis=0)


# graphs:
######################################
######## OVER INITIAL WINDOW #########
############ GRAPH ONE ###############
######################################
# plot market price of risk, Phi, Delta_bar over Npre
x = Npres/12
# x = Npres[1:]
var_name_list = ['market price of risk', 'consumption share of participants', 'consumption-weighted estimation error of participants']
var_list = [theta_Mat, Phi_parti_Mat, Delta_bar_parti_Mat]  # Shape((N_scenarios, n_phi, T_hat_dimension, 2))
phi_index = [0, 2]
scenario_list = ['Complete', 'Keep', 'Drop']
phi_list = ['phi=0.0', 'phi=0.4']
colors_short = ['midnightblue', 'darkgreen', 'darkviolet']
fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', figsize=(15, 20))
for i, axes_row in enumerate(axes):
    row_name = var_name_list[i]
    var = var_list[i][:, phi_index]  # Shape((N_scenarios, 2, T_hat_dimension, 2))
    for j, ax in enumerate(axes_row):
        y = var[:, :, :, j]  # Shape((N_scenarios, 2, T_hat_dimension))
        column_name = 'Mean' if j == 0 else 'Volatility'
        for k in range(2):
            line_style = 'dotted' if k == 0 else 'solid'
            for l in range(N_scenarios):
                y_i = y[l, k]
                if j == 0 and k == 1:
                    ax.plot(x, y_i, linestyle=line_style, color=colors_short[l], label=scenario_list[l])
                elif j == 1 and l == 0:
                    ax.plot(x, y_i, linestyle=line_style, color=colors_short[l], label=phi_list[k])
                else:
                    ax.plot(x, y_i, linestyle=line_style, color=colors_short[l])
        if j == 0:
            ax.set_ylabel(row_name)
        if i == 2:
            # ax.set_xlabel('initial window (months)')
            ax.set_xlabel('initial window (years)')
        if i == 0:
            ax.legend()
            ax.set_title(column_name)
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
# plt.savefig('initial window and values mean vola years.png', dpi=500, format="png")
# plt.savefig('initial window and values mean vola.png', dpi=500, format="png")
plt.show()
# plt.close()


######################################
######## OVER INITIAL WINDOW #########
############ GRAPH TWO ###############
######################################
# plot market price of risk, Phi, Delta_bar over Npre
x = Npres/12
# x = Npres[1:]
var_name_list = ['Participation rates in age groups', 'Wealth shares in age groups']
var_list = [popu_age_Mat, wealthshare_age_Mat]  # Shape((N_scenarios, n_phi, T_hat_dimension, 2))
phi_index = 1
scenario_list = ['Complete', 'Keep', 'Drop']
phi_list = ['phi=0.0', 'phi=0.4']
colors_short = ['midnightblue', 'red', 'darkgreen', 'darkviolet']
age_labels = ['20 < Age <= 35, youngest quartile', '35 < Age <= 55', '55 < Age <= 89', 'Age > 89, oldest quartile']
hatches = ['/', 'o', '\\', '.']
low = np.zeros((T_hat_dimension))
fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', sharey='all', figsize=(15, 20))
for i, axes_row in enumerate(axes):
    row_name = scenario_list[i]
    for j, ax in enumerate(axes_row):
        var = var_list[j][i, phi_index, :, 0]  # Shape((N_scenarios, n_phi, T_hat_dimension, 2, n_age_groups)) -> Shape((T_hat_dimension, 4))
        if j == 0:
            for k in range(n_age_groups):
                y_i = var[:, k]
                if k == 0:
                    # ax.fill_between(x, y_i, color=colors_short[k], hatch=hatches[k], linewidth=0, label=age_labels[k])
                    ax.plot(x, y_i, color=colors_short[k], label=age_labels[k])
                    ax.fill_between(x, y_i, low, facecolor="none", edgecolor=colors_short[k], hatch=hatches[k], linewidth=0)
                    y_cumu = low + y_i
                elif i == 0:
                    y_plot = y_i + y_cumu
                    # ax.fill_between(x, y_i, y_1, color=colors_short[k], linewidth=0, label=age_labels[k])
                    ax.plot(x, y_plot, color=colors_short[k], label=age_labels[k])
                    ax.fill_between(x, y_plot, y_cumu, facecolor="none", edgecolor=colors_short[k], hatch=hatches[k], linewidth=0)
                    y_cumu = y_plot
                else:
                    y_1 = var[:, k - 1]
                    # ax.fill_between(x, y_i, y_1, color=colors_short[k], linewidth=0, label=age_labels[k])
                    ax.plot(x, y_i, color=colors_short[k], label=age_labels[k])
                    ax.fill_between(x, y_i, y_1, facecolor="none", edgecolor=colors_short[k], hatch=hatches[k], linewidth=0)
            if j == 0:
                ax.set_ylabel(row_name)
        else:
            for k in range(n_age_groups):
                y_i = var[:, k]
                if k == 0:
                    # ax.fill_between(x, y_i, color=colors_short[k], hatch=hatches[k], linewidth=0, label=age_labels[k])
                    ax.plot(x, y_i, color=colors_short[k], label=age_labels[k])
                else:
                    y_1 = var[:, k - 1]
                    # ax.fill_between(x, y_i, y_1, color=colors_short[k], linewidth=0, label=age_labels[k])
                    ax.plot(x, y_i, color=colors_short[k], label=age_labels[k])
                if i == 0:
                    ax.legend()
        if i == 0:
            # ax.legend()
            ax.set_title(var_name_list[j])
        if i == 2:
            ax.set_xlabel('initial window (years)')
            ax.tick_params(axis='x', labelcolor='black')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('initial window and values age groups.png', dpi=500, format="png")
plt.show()
plt.close()
