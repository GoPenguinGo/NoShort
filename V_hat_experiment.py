import time
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI_mean_vola, simulate_SI
from src.param import Npres, rho, nu, mu_Y, sigma_Y, sigma_Y_sqr, sigma_S, v, tax, \
    beta, dt, Ninit, Nt, Nc, tau, cohort_size, n_age_groups, \
    cutoffs, phi_vector, n_phi, colors, modes_trade, modes_learn, scenarios, \
    dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, top, old_limit, \
    dZ_Y_cases, dZ_SI_cases, T_cohort, age_labels
import tabulate as tab

# todo: the connection between belief and wealth?
#  Learning from repeated negative economic shocks: lead to both worse wealth condition and pessimism

T_hats = dt * Npres
T_hat_dimension = len(T_hats)

# n_scenarios = 1
n_scenarios = 1
a_sce = 0
N = 5000

phi = 0.4

# Generate matrix to store the results
r_matrix = np.zeros((N, n_scenarios, T_hat_dimension, 2), dtype=np.float32)  # for mean and std
theta_matrix = np.zeros((N, n_scenarios, T_hat_dimension, 2), dtype=np.float32)
Phi_parti_matrix = np.zeros((N, n_scenarios, T_hat_dimension, 2), dtype=np.float32)
Phi_parti_1_matrix = np.zeros((N, n_scenarios, T_hat_dimension, 2), dtype=np.float32)
# popu_parti_matrix = np.zeros((N, n_scenarios, T_hat_dimension, 2))
Delta_bar_parti_matrix = np.zeros((N, n_scenarios, T_hat_dimension, 2), dtype=np.float32)
popu_age_matrix = np.zeros((N, n_scenarios, T_hat_dimension, 2, n_age_groups), dtype=np.float32)
# belief_age_matrix = np.zeros((N, n_scenarios, T_hat_dimension, 2, n_age_groups))
wealthshare_age_matrix = np.zeros((N, n_scenarios, T_hat_dimension, 2, n_age_groups), dtype=np.float32)
Delta_popu_parti_matrix = np.zeros((N, n_scenarios, T_hat_dimension, 2), dtype=np.float32)
variance_matrix = np.zeros((N, n_scenarios, T_hat_dimension, 4), dtype=np.float32)
# r_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))  # for mean and std
# theta_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
# Phi_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
# Phi_parti_1_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
# # popu_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
# Delta_bar_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
# popu_age_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
# # belief_age_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
# wealthshare_age_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
# Delta_popu_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
# variance_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 4))

# write a lighter version of the simulation function that returns only the desired values (mean and std, instead of whole raw data)
for l in range(N):
    print(l)
    # dZ = -dZ_matrix[l]
    # dZ_build = -dZ_build_matrix[l]
    # dZ_SI = -dZ_SI_matrix[l]
    # dZ_SI_build = -dZ_SI_build_matrix[l]
    dZ = dZ_matrix[int(l * 2)]
    dZ_build = dZ_build_matrix[int(l * 2)]
    dZ_SI = dZ_SI_matrix[int(l * 2)]
    dZ_SI_build = dZ_SI_build_matrix[int(l * 2)]
    time_s = time.time()
    for m in range(n_scenarios):
        scenario = scenarios[m + a_sce]
        # scenario = scenarios[m]
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        for o, T_hat_try in enumerate(T_hats):
            Npre_try = int(Npres[o])
            Vhat_try = (sigma_Y ** 2) / T_hat_try  # prior variance
            (
                r,
                theta,
                # popu_parti,
                Delta_bar_parti,
                Phi_parti,
                Phi_parti_1,
                popu_age,
                # belief_age,
                wealthshare_age,
                popu_can_short,
                popu_short,
                Phi_can_short,
                Phi_short,
                variances,
                Delta_popu_parti,
            ) = simulate_SI_mean_vola(mode_trade, mode_learn, Nc, Nt, dt, rho, nu,
                                      Vhat_try,
                                      mu_Y, sigma_Y, sigma_S, tax, beta,
                                      phi,
                                      Npre_try,
                                      Ninit,
                                      T_hat_try,
                                      dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size, top, old_limit, cutoffs,
                                      n_age_groups,
                                      )
            r_matrix[l, m, o] = r
            theta_matrix[l, m, o] = theta
            # popu_parti_matrix[l, m, o] = popu_parti
            Delta_bar_parti_matrix[l, m, o] = Delta_bar_parti
            Phi_parti_matrix[l, m, o] = Phi_parti
            Phi_parti_1_matrix[l, m, o] = Phi_parti_1
            popu_age_matrix[l, m, o] = popu_age
            # belief_age_matrix[l, m, o] = belief_age
            wealthshare_age_matrix[l, m, o] = wealthshare_age
            Delta_popu_parti_matrix[l, m, o] = Delta_popu_parti
            variance_matrix[l, m, o] = variances
    print(time.time() - time_s)

# table:
table_output = np.zeros((6, 6))
Npre_index = np.array([np.searchsorted(60, Npres), np.searchsorted(240, Npres)])
var_list = [theta_matrix, Phi_parti_1_matrix * sigma_Y, Delta_bar_parti_matrix]
header = np.tile(['Mean', 'Volatility'], 3)
show_index = np.tile(['5-year', '20-year'], 3)
for j, var in enumerate(var_list):
    var_average = np.average(var, axis=0)  # shape (n_scenarios, T_hat_dimension)
    for i in range(n_scenarios):
        for k in range(2):
            row_index = j * 2 + k
            col_index = i * 2
            table_output[row_index, col_index:col_index + 2] = var_average[i, k]
print(tab.tabulate(table_output, headers=header, floatfmt=".3f", tablefmt='latex'))

# save the data
var_list = [r_matrix, theta_matrix, Phi_parti_matrix, Phi_parti_1_matrix,
            Delta_bar_parti_matrix, popu_age_matrix,
            wealthshare_age_matrix, Delta_popu_parti_matrix, variance_matrix]
var_name_list = ['interest rate', 'market price of risk', 'consumption share of participants',
                 'consumption share 1 of participants',
                 'consumption-weighted estimation error of participants', 'participation rate in age groups',
                 'wealth share in age groups', 'population-weighted estimation error of participants',
                 'variance of participants']
type_list = ['mean', 'vola']
age_labels = ['20 < Age <= 35, youngest quartile', '35 < Age <= 55', '55 < Age <= 89', 'Age > 89, oldest quartile']

for i, var in enumerate(var_list):
    # np.save(var_name_list[i] + str(a_sce) + 'neg', var)
    np.save(var_name_list[i] + str(a_sce) + 'pos', var)

# read the data:
r_Mat = np.zeros((N_scenarios, T_hat_dimension, 2))  # for mean and std
theta_Mat = np.zeros((N_scenarios, T_hat_dimension, 2))
Phi_parti_Mat = np.zeros((N_scenarios, T_hat_dimension, 2))
Phi_parti_1_Mat = np.zeros((N_scenarios, T_hat_dimension, 2))
# popu_parti_Mat = np.zeros((N_scenarios, T_hat_dimension, 2))
Delta_bar_parti_Mat = np.zeros((N_scenarios, T_hat_dimension, 2))
popu_age_Mat = np.zeros((N_scenarios, T_hat_dimension, 2, n_age_groups))
# belief_age_Mat = np.zeros((N_scenarios, T_hat_dimension, 2, n_age_groups))
wealthshare_age_Mat = np.zeros((N_scenarios, T_hat_dimension, 2, n_age_groups))
Delta_popu_parti_Mat = np.zeros((N_scenarios, T_hat_dimension, 2))
variance_Mat = np.zeros((N_scenarios, T_hat_dimension, 4))
var_list = [r_Mat, theta_Mat, Phi_parti_Mat, Phi_parti_1_Mat,
            Delta_bar_parti_Mat, popu_age_Mat, wealthshare_age_Mat,
            Delta_popu_parti_Mat, variance_Mat]
var_name_list = ['interest rate', 'market price of risk', 'consumption share of participants',
                 'consumption share 1 of participants',
                 'consumption-weighted estimation error of participants', 'participation rate in age groups',
                 'wealth share in age groups', 'population-weighted estimation error of participants',
                 'variance of participants']
for i, var in enumerate(var_list):
    var_name = var_name_list[i]
    for j in range(N_scenarios):
        var_name_j_pos = var_name + str(j) + 'pos' + '.npy'
        var_name_j_neg = var_name + str(j) + 'neg' + '.npy'
        y_pos = np.load(var_name_j_pos)
        y_neg = np.load(var_name_j_neg)
        var[j] = (np.mean(y_pos, axis=0) + np.mean(y_neg, axis=0)) / 2

# graphs:
######################################
######## OVER INITIAL WINDOW #########
############ GRAPH ONE ###############
######################################
# plot market price of risk, Phi, Delta_bar over Npre
x = Npres / 12
x_start = 0
# x = Npres[1:]
var_name_list = [r'market price of risk $\theta_t$', r'$\sigma_Y\frac{1}{\Phi_t}$', r'$\bar{\Delta}_t$']
# var_name_list = ['market price of risk', 'consumption share of participants', 'consumption-weighted estimation error of participants']
Phi_parti_1_sigma_Mat = Phi_parti_1_Mat * sigma_Y
var_list = [theta_Mat, Phi_parti_1_sigma_Mat, Delta_bar_parti_Mat]  # Shape((N_scenarios, n_phi, T_hat_dimension, 2))
scenario_list = ['Complete', 'Reentry', 'Disappointment']
phi_list = [r'$\phi=0.0$', r'$\phi=0.4$', r'$\phi=0.8$']
colors_short = ['midnightblue', 'darkgreen', 'darkviolet']
line_styles = ['dotted', 'solid', 'dashed']
x_label_vhat = r'Initial window (years), $n$'
fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', figsize=(15, 20))
for i, axes_row in enumerate(axes):
    row_name = var_name_list[i]
    var = var_list[i]  # Shape((N_scenarios, 3, T_hat_dimension, 2))
    for j, ax in enumerate(axes_row):
        y = var[:, :, j]  # Shape((N_scenarios, 2, T_hat_dimension))
        column_name = 'Mean' if j == 0 else 'Volatility'
        # for k in range(3):
        #     line_style = line_styles[k]
        #     for l in range(N_scenarios):
        #         y_i = y[l, k]
        #         if j == 0 and k == 1:
        #             ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l],
        #                     label=scenario_list[l])
        #         elif j == 1 and l == 0:
        #             ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l], label=phi_list[k])
        #         else:
        #             ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l])
        line_style = 'solid'
        for l in range(N_scenarios):
            y_i = y[l]
            if j == 0:
                ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l],
                            label=scenario_list[l])
            else:
                ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l])
        if j == 0:
            ax.set_ylabel(row_name, rotation=90)
        if i == 2:
            # ax.set_xlabel('initial window (months)')
            ax.set_xlabel(x_label_vhat)
        if i == 0:
            ax.legend()
            ax.set_title(column_name)
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
# plt.savefig('initial window and values mean vola years.png', dpi=100, format="png")
plt.show()
# plt.close()

######################################
######## OVER INITIAL WINDOW #########
############ GRAPH ONE ###############
############ COVARIANCE ##############
######################################
theta_var = np.zeros((N_scenarios, n_phi, T_hat_dimension))
Phi_parti_1_var = np.zeros((N_scenarios, n_phi, T_hat_dimension))
Delta_bar_parti_var = np.zeros((N_scenarios, n_phi, T_hat_dimension))
var_list = [theta_var, Phi_parti_1_var, Delta_bar_parti_var]
var_name_list = ['market price of risk', 'consumption share 1 of participants',
                 'consumption-weighted estimation error of participants']
for i, var in enumerate(var_list):
    var_name = var_name_list[i]
    for j in range(N_scenarios):
        var_name_j = var_name + str(j) + '.npy'
        y = np.load(var_name_j)[:, :, :, :, 1]
        var[j] = np.mean(y ** 2, axis=0)
cov_Mat = theta_var - sigma_Y ** 2 * Phi_parti_1_var - Delta_bar_parti_var
x_start = 1
fig, ax = plt.subplots(figsize=(5, 5))
for k in range(3):
    line_style = line_styles[k]
    for l in range(N_scenarios):
        y_i = cov_Mat[l, k]
        if k == 1:
            ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l], label=scenario_list[l])
        elif l == 0:
            ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l])
        else:
            ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l])
ax.set_ylabel(r'Covariance, -2cov($\sigma_Y\frac{1}{\Phi_t}, \bar{\Delta}_t$)', rotation=90)
ax.set_xlabel(x_label_vhat)
ax.legend()
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig('initial window and covariance years.png', dpi=500, format="png")
# plt.savefig('initial window and values mean vola.png', dpi=500, format="png")
plt.show()
# plt.close()

######################################
######## OVER INITIAL WINDOW #########
############ GRAPH TWO ###############
######################################
# plot participation rate and wealth share over Npre
x = Npres / 12
# x = Npres[1:]
var_name_list = ['Participation rates in age groups', 'Wealth shares in age groups']
var_list = [popu_age_Mat, wealthshare_age_Mat]  # Shape((N_scenarios, n_phi, T_hat_dimension, 2))
phi_index = 1
scenario_list = ['Complete', 'Reentry', 'Disappointment']
phi_list = ['phi=0.0', 'phi=0.4']
colors_short = ['midnightblue', 'red', 'darkgreen', 'darkviolet']
hatches = ['/', 'o', '\\', '.']
low = np.zeros((T_hat_dimension))
fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', sharey='all', figsize=(15, 20))
for i, axes_row in enumerate(axes):
    row_name = scenario_list[i]
    for j, ax in enumerate(axes_row):
        var = var_list[j][i, phi_index, :,
              0]  # Shape((N_scenarios, n_phi, T_hat_dimension, 2, n_age_groups)) -> Shape((T_hat_dimension, 4))
        if j == 0:
            for k in range(n_age_groups):
                y_i = var[:, k]
                if k == 0:
                    # ax.fill_between(x, y_i, color=colors_short[k], hatch=hatches[k], linewidth=0, label=age_labels[k])
                    ax.plot(x, y_i, color=colors_short[k], label=age_labels[k])
                    ax.fill_between(x, y_i, low, facecolor="none", edgecolor=colors_short[k], hatch=hatches[k],
                                    linewidth=0)
                    y_cumu = low + y_i
                # elif i == 0:
                #     y_plot = y_i + y_cumu
                #     # ax.fill_between(x, y_i, y_1, color=colors_short[k], linewidth=0, label=age_labels[k])
                #     ax.plot(x, y_plot, color=colors_short[k], label=age_labels[k])
                #     ax.fill_between(x, y_plot, y_cumu, facecolor="none", edgecolor=colors_short[k], hatch=hatches[k], linewidth=0)
                #     y_cumu = y_plot
                else:
                    y_1 = var[:, k - 1]
                    # ax.fill_between(x, y_i, y_1, color=colors_short[k], linewidth=0, label=age_labels[k])
                    ax.plot(x, y_i, color=colors_short[k], label=age_labels[k])
                    ax.fill_between(x, y_i, y_1, facecolor="none", edgecolor=colors_short[k], hatch=hatches[k],
                                    linewidth=0)
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
            ax.set_xlabel(x_label_vhat)
            ax.tick_params(axis='x', labelcolor='black')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('initial window and values age groups.png', dpi=100, format="png")
plt.show()
plt.close()

######################################
######## OVER INITIAL WINDOW #########
############ GRAPH x ###############
# plot Delta_bar and average Delta over Npre
x = Npres / 12
x_start = 0
# x = Npres[1:]
var_name_list = [r'$\bar{\Delta}_t$', r'Average $\Delta_{st}$']
var_list = [Delta_bar_parti_Mat, Delta_popu_parti_Mat]  # Shape((N_scenarios, n_phi, T_hat_dimension, 2))
scenario_list = ['Complete', 'Reentry', 'Disappointment']
phi_list = [r'$\phi=0.4$']
colors_short = ['midnightblue', 'darkgreen', 'darkviolet']
line_styles = ['dotted', 'solid', 'dashed']
x_label_vhat = r'Initial window (years), $n$'
fig, axes = plt.subplots(nrows=2, ncols=2, sharex='all', sharey='col', figsize=(15, 15))
for i, axes_row in enumerate(axes):
    row_name = var_name_list[i]
    var = var_list[i]  # Shape((N_scenarios, T_hat_dimension, 2))
    for j, ax in enumerate(axes_row):
        y = var[:, :, j]  # Shape((N_scenarios, 2, T_hat_dimension))
        column_name = 'Mean' if j == 0 else 'Volatility'
        # for k in range(3):
        #     line_style = line_styles[k]
        #     for l in range(N_scenarios):
        #         y_i = y[l, k]
        #         if j == 0 and k == 1:
        #             ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l],
        #                     label=scenario_list[l])
        #         elif j == 1 and l == 0:
        #             ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l], label=phi_list[k])
        #         else:
        #             ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l])
        line_style = 'solid'
        for l in range(N_scenarios):
            y_i = y[l]
            if j == 0:
                ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l],
                            label=scenario_list[l])
            else:
                ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l])
        if j == 0:
            ax.set_ylabel(row_name, rotation=90)
        if i == 2:
            # ax.set_xlabel('initial window (months)')
            ax.set_xlabel(x_label_vhat)
        if i == 0:
            ax.legend()
            ax.set_title(column_name)
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
# plt.savefig('initial window and values mean vola years.png', dpi=100, format="png")
plt.show()
# plt.close()