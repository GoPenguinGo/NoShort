import time
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI_mean_vola, simulate_SI
from src.param import Npres, rho, nu, mu_Y, sigma_Y, sigma_Y_sqr, sigma_S, v, tax, \
    beta, dt, Ninit, Nt, Nc, tau, cohort_size, n_age_groups, \
    cutoffs, phi_vector, n_phi, colors, modes_trade, modes_learn, scenarios, \
    dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, top, old_limit, Z_Y_cases, Z_SI_cases, T_cohort

# todo: the connection between belief and wealth?
#  Learning from repeated negative economic shocks: lead to both worse wealth condition and pessimism

T_hats = dt * Npres
T_hat_dimension = len(T_hats)
# N = 30  # can choose a smaller number than Mpaths as the number of paths

n_scenarios = 1
a_sce = 2
N = 2
N_scenarios = 3

# Generate matrix to store the results
r_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))  # for mean and std
theta_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
Phi_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
Phi_parti_1_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
# popu_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
Delta_bar_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
popu_age_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
# belief_age_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
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
                    # popu_parti,
                    Delta_bar_parti,
                    Phi_parti,
                    Phi_parti_1,
                    popu_age,
                    # belief_age,
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
                # popu_parti_matrix[l, m, n, o] = popu_parti
                Delta_bar_parti_matrix[l, m, n, o] = Delta_bar_parti
                Phi_parti_matrix[l, m, n, o] = Phi_parti
                Phi_parti_1_matrix[l, m, n, o] = Phi_parti_1
                popu_age_matrix[l, m, n, o] = popu_age
                # belief_age_matrix[l, m, n, o] = belief_age
                wealthshare_age_matrix[l, m, n, o] = wealthshare_age
    print(time.time() - time_s)


# save the data
var_list = [r_matrix, theta_matrix, Phi_parti_matrix, Phi_parti_1_matrix,
            Delta_bar_parti_matrix, popu_age_matrix,
            wealthshare_age_matrix]
var_name_list = ['interest rate', 'market price of risk', 'consumption share of participants', 'consumption share 1 of participants',
                 'consumption-weighted estimation error of participants', 'participation rate in age groups',
                 'wealth share in age groups']
type_list = ['mean', 'vola']
age_labels = ['20 < Age <= 35, youngest quartile', '35 < Age <= 55', '55 < Age <= 89', 'Age > 89, oldest quartile']

for i, var in enumerate(var_list):
    np.save(var_name_list[i] + str(a_sce), var)




# read the data:
r_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))  # for mean and std
theta_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))
Phi_parti_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))
Phi_parti_1_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))
#popu_parti_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))
Delta_bar_parti_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2))
popu_age_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
#belief_age_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
wealthshare_age_Mat = np.zeros((N_scenarios, n_phi, T_hat_dimension, 2, n_age_groups))
var_list = [r_Mat, theta_Mat, Phi_parti_Mat, Phi_parti_1_Mat,
            Delta_bar_parti_Mat, popu_age_Mat, wealthshare_age_Mat]
var_name_list = ['interest rate', 'market price of risk', 'consumption share of participants', 'consumption share 1 of participants',
                 'consumption-weighted estimation error of participants', 'participation rate in age groups',
                 'wealth share in age groups']
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
x_start = 0
# x = Npres[1:]
var_name_list = [r'market price of risk $\theta_t$', r'$\sigma_Y\frac{1}{\Phi_t}$', r'$\bar{\Delta}_t$']
# var_name_list = ['market price of risk', 'consumption share of participants', 'consumption-weighted estimation error of participants']
Phi_parti_1_sigma_Mat = Phi_parti_1_Mat * sigma_Y
var_list = [theta_Mat, Phi_parti_1_sigma_Mat, Delta_bar_parti_Mat]  # Shape((N_scenarios, n_phi, T_hat_dimension, 2))
scenario_list = ['Complete', 'Keep', 'Drop']
phi_list = ['phi=0.0', 'phi=0.4', 'phi=0.8']
colors_short = ['midnightblue', 'darkgreen', 'darkviolet']
line_styles = ['dotted', 'solid', 'dashed']
fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', figsize=(15, 20))
for i, axes_row in enumerate(axes):
    row_name = var_name_list[i]
    var = var_list[i]  # Shape((N_scenarios, 3, T_hat_dimension, 2))
    for j, ax in enumerate(axes_row):
        y = var[:, :, :, j]  # Shape((N_scenarios, 2, T_hat_dimension))
        column_name = 'Mean' if j == 0 else 'Volatility'
        for k in range(3):
            line_style = line_styles[k]
            for l in range(N_scenarios):
                y_i = y[l, k]
                if j == 0 and k == 1:
                    ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l], label=scenario_list[l])
                elif j == 1 and l == 0:
                    ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l], label=phi_list[k])
                else:
                    ax.plot(x[x_start:], y_i[x_start:], linestyle=line_style, color=colors_short[l])
        if j == 0:
            ax.set_ylabel(row_name, rotation=90)
        if i == 2:
            # ax.set_xlabel('initial window (months)')
            ax.set_xlabel('initial window (years)')
        if i == 0:
            ax.legend()
            ax.set_title(column_name)
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig('initial window and values mean vola years.png', dpi=500, format="png")
# plt.savefig('initial window and values mean vola.png', dpi=500, format="png")
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
var_name_list = ['market price of risk', 'consumption share 1 of participants', 'consumption-weighted estimation error of participants']
for i, var in enumerate(var_list):
    var_name = var_name_list[i]
    for j in range(N_scenarios):
        var_name_j = var_name + str(j) +'.npy'
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
ax.set_xlabel('initial window (years)')
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


######################################
######## INITIAL WINDOW  AND #########
######## BELIEF DISTRIBUTION #########
########### GRAPH THREE ##############
######################################
phi_vector = np.arange(0,1,0.1)
n_phi = len(phi_vector)
phi_indexes = [0, 4, 8]
n_phi_short = len(phi_indexes)
phi_vector_short = phi_vector[phi_indexes]

Npre_short = [3, 60, 240]
n_Npre_short = len(Npre_short)
T_hat_short = Npre_short * dt

dZ_build = dZ_build_matrix[0]
dZ_SI_build = dZ_SI_build_matrix[0]  # fix the shocks at the buildup stage

theta_mat = np.empty((N_scenarios, 2, 2, n_Npre_short, n_phi_short, Nt))
Delta_mat = np.empty((N_scenarios, 2, 2, n_Npre_short, n_phi_short, Nt, Nc))
#invest_tracker_mat = np.empty((N_scenarios, 2, 2, n_Npre_short, n_phi_short, Nt, Nc))
for g in range(N_scenarios):
    scenario = scenarios[g]
    mode_trade = scenario[0]
    mode_learn = scenario[1]
    for i in range(2):
        dZ = Z_Y_cases[i]
        for j in range(2):
            dZ_SI = Z_SI_cases[j]
            for k, Npre_try in enumerate(Npre_short):
                T_hat_try = Npre_try * dt
                Vhat_try = (sigma_Y ** 2) / T_hat_try  # prior variance
                for l, phi_try in enumerate(phi_vector_short):
                    (
                        r,
                        theta,
                        f,
                        Delta,
                        pi,
                        popu_parti,
                        Phi_parti,
                        Delta_bar_parti,
                        dR,
                        invest_tracker
                    ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu,
                            Vhat_try,
                            mu_Y, sigma_Y, sigma_S, tax, beta,
                            phi_try,
                            Npre_try,
                            Ninit,
                            T_hat_try, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                            top = 0.05,
                            old_limit = 100
                            )
                    Delta_mat[g, i, j, k, l] = Delta
                    theta_mat[g, i, j, k, l] = theta
                    #invest_tracker_mat[g, i, j, k, l] = invest_tracker

red_cases = [r'Good $z^Y$ ', r'Bad $z^Y$ ']
yellow_cases = [r'Good $z^{SI}$ ', r'Bad $z^{SI}$ ']
t = np.arange(0, T_cohort, dt)
length = len(t)
scenario_labels = ['Complete', 'Reentry', 'Disappointment']
colors_short = ['midnightblue', 'darkgreen', 'darkviolet', 'red']
colors_short2 = ['mediumblue', 'saddlebrown', 'darkmagenta']
label_phi = []
for i in range(n_phi_short):
    label_phi.append(r'$\phi$ = ' + str(phi_vector_short[i]))

y_case = np.empty((N_scenarios, 2, 2, n_Npre_short, n_phi_short, Nt, n_age_groups, 2))  # min, max for each age group
for i in range(N_scenarios):
    for j in range(2):
        for k in range(2):
            for l in range(n_Npre_short):
                for m in range(n_phi_short):
                    Delta = Delta_mat[i, j, k, l, m]  # ((Nt, Nc))
                    for n in range(n_age_groups):
                        Delta_age_group = Delta[:, cutoffs[n+1]:cutoffs[n]]
                        y_case[i, j, k, l, m, :, n, 0] = np.amin(Delta_age_group, axis=1)
                        y_case[i, j, k, l, m, :, n, 1] = np.amax(Delta_age_group, axis=1)

age_labels = ['20 < Age <= 35, youngest quartile', '35 < Age <= 55', '55 < Age <= 89', 'Age > 89, oldest quartile']
scenario_indexes = [1, 2, 0, 1]
phi_indexes = [1, 1, 1, 2]
Npre_indexes = [1, 2]
cases = [1, 1]
red_case = cases[1]
yellow_case = cases[1]
Z = np.cumsum(Z_Y_cases[red_case])
Z_SI = np.cumsum(Z_SI_cases[yellow_case])
for h, Npre_index in enumerate(Npre_indexes):
    Npre = Npre_short[int(Npre_index)]
    fig, axes = plt.subplots(nrows=4, sharex='all', sharey='all', figsize=(15, 20))
    for i, ax in enumerate(axes):
        scenario_index = scenario_indexes[i]
        phi_index = phi_indexes[i]
        y = y_case[scenario_index, red_case, yellow_case, Npre_index, phi_index]
        ycutoff = -theta_mat[scenario_index, red_case, yellow_case, Npre_index, phi_index]
        ax.set_xlabel('Time in simulation, one random path')
        ax.set_ylabel('Zt', color='black')
        if i == 0:
            ax.plot(t, Z, color='red', linewidth=0.5, label=r'$z^Y_t$')
            ax.plot(t, Z_SI, color='gold', linewidth=0.5, label=r'$z^{SI}_t$')
            ax.tick_params(axis='y', labelcolor='black')
            ax.legend(loc='upper left')
        ax.set_title(scenario_labels[scenario_index] + ', ' + label_phi[phi_index])
        ax2 = ax.twinx()
        ax2.set_ylabel('Estimation error', color='black')
        for j in range(n_age_groups):
            ymin = y[:, j, 0]
            ymax = y[:, j, 1]
            ax2.fill_between(t, ymax, ymin, color=colors_short[j], linewidth=0., alpha=0.3, label=age_labels[j])
        if i == 0:
            ax2.plot(t, ycutoff, color='blue', linewidth=0.5, label='cutoff')
            ax2.legend(loc='upper right')
        else:
            ax2.plot(t, ycutoff, color='blue', linewidth=0.5)
        # ax2.plot(t, yy, color=colors[i], linewidth=0.4, label=label_i)
        ax2.tick_params(axis='y', labelcolor='black')
        # ax2.set_ylim()
        # ax2.grid()
    # fig.suptitle('Zt and Market Price of Risk')
    fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
    # plt.savefig(str(Npre) + 'Distribution of Delta age groups.png', dpi=300)
    plt.show()
    # plt.close()
