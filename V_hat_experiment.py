import time
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate, simulate_partial_constraint
from src.param import *


# modes = ['drop', 'ric_free']
# mode = 'rich_free'
# mode = 'drop'
mode = 'comp'
zoom_in = 'small'
# zoom_in = 'large'
T_hats = dt * np.arange(1, 13, 1) if zoom_in == 'small' else np.arange(1, 31, 2)
T_hat_dimension = len(T_hats)
nus = [0.01, 0.02, 0.03]
nu_dimension = len(nus)

# for graphs:
Tkeep = 100
Nkeep = int(Tkeep / dt)
Tsample = int(T_cohort / 100)
Nsamples = 500

# Generate matrix to store the results
dZ_matrix = np.zeros((Mpaths, Nt))
dZ_build_matrix = np.zeros((Mpaths, Nc-1))
# mu_C_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))
# sigma_C_matrix = np.zeros((T_hat_dimension, Mpaths))
# delta_matrix = np.zeros((T_hat_dimension, Mpaths, Nt, Nc))
r_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))
# f_matrix = np.zeros((T_hat_dimension, Mpaths, Nt, Nc))
theta_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))
pi_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))
f_parti_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))
popu_parti_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))
# Delta_bar_parti_matrix = np.zeros((T_hat_dimension, Mpaths, Nt))
# w_matrix = np.zeros((T_hat_dimension, Mpaths, Nt, Nc))
# w_cohort_matrix = np.zeros((T_hat_dimension, Mpaths, Nt, Nc))
age_parti_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))
n_parti_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))
popu_age_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths, 4))
belief_age_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths, 4))
wealthshare_age_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths, 4))

# invest_tracker_matrix = np.zeros((T_hat_dimension, Mpaths, Nt, Nc))

# Et_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))
# Vt_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))
dR_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))

for l in range(Mpaths):
    s = time.time()
    # same shocks for the different T_hats
    dZ_build = dt ** 0.5 * np.random.randn(int(Nc - 1))  # dZt for the build function
    dZ = dt ** 0.5 * np.random.randn(Nt)  # dZt for the simulate function
    dZ_matrix[l, :] = dZ
    dZ_build_matrix[l, :] = dZ_build

# The main loop builds up the economy with a large number of cohorts, and simulates the stationary economy forward
for l in range(Mpaths):
    dZ = dZ_matrix[l, :]
    dZ_build = dZ_build_matrix[l, :]

    for k, T_hat in enumerate(T_hats):
        Npre = int(T_hat / dt)
        Vhat = (sigma_Y ** 2) / T_hat  # prior variance

        for m, nu in enumerate(nus):
            if mode == 'drop' or mode == 'keep' or mode == 'comp':
                (
                    r,
                    theta,
                    f,
                    Delta,
                    max,
                    pi,
                    popu_parti,
                    f_parti,
                    Delta_bar_parti,
                    dR,
                    w_cohort,
                    age_parti,
                    n_parti,
                ) = simulate(mode, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, tax, beta, Npre, Ninit, T_hat,
                             dZ_build, dZ, tau,
                             cohort_size)
                invest_tracker = pi > 0

            else:
                (
                    r,
                    theta,
                    f,
                    Delta,
                    d_eta,
                    pi,
                    dR,
                    w_cohort,
                    popu_parti,
                    popu_can_short,
                    popu_short,
                    popu_long,
                    f_parti,
                    f_short,
                    f_long,
                    age_parti,
                    age_short,
                    age_long,
                    n_parti,
                    invest_tracker,
                    can_short_tracker,
                    long,
                    short,
                    Delta_bar_parti,
                    Delta_bar_long,
                    Delta_bar_short,
                ) = simulate_partial_constraint(mode, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, tax, beta, Npre,
                                                Ninit,
                                                T_hat, dZ_build, dZ, tau, cohort_size)
            dR_matrix[k, m, l] = np.mean(dR[1200:])
            r_matrix[k, m, l] = np.mean(r[1200:])
            theta_matrix[k, m, l] = np.mean(theta[1200:])
            popu_parti_matrix[k, m, l] = np.mean(popu_parti[1200:])
            parti_rate = invest_tracker * cohort_size

            belief = (Delta * sigma_Y + mu_Y)
            belief_weights = f * dt

            for i in range(4):
                popu_age_matrix[k, m, l, i] = np.mean(np.sum(parti_rate[1200:, cutoffs[i + 1]:], axis=1))

                weights_zero = (np.sum(invest_tracker[:, cutoffs[i + 1]:cutoffs[i]],
                                       axis=1) == 0)  # no one from the age group is participating
                belief_copy = belief.copy()
                if np.sum(weights_zero[1200:]) == Nt - 1200:
                    belief_age_matrix[k, m, l, i] = np.nan
                else:
                    # weights_zero = np.transpose(np.tile(weights_zero, (Nc, 1)))
                    a = np.where(weights_zero == 1)
                    belief_copy[a, :] = np.nan
                    belief_age_matrix[k, m, l, i] = np.nanmean(
                        np.average(belief_copy[1200:, cutoffs[i + 1]:cutoffs[i]],
                                   weights=belief_weights[1200:, cutoffs[i + 1]:cutoffs[i]], axis=1)
                    )

                wealthshare_age_matrix[k, m, l, i] = np.mean(
                    np.sum(f[1200:, cutoffs[i + 1]:cutoffs[i]] * dt, axis=1)
                )

            age_parti_matrix[k, m, l] = np.mean(age_parti[1200:])
            n_parti_matrix[k, m, l] = np.mean(n_parti[1200:])

        # covariance:
    print(l)


# graphs:
x = T_hats
y0 = (np.ones(len(T_hats)) * sigma_Y ** 2) / x
y1 = np.mean(r_matrix, axis=2)
y2 = np.mean(theta_matrix, axis=2)
y3 = np.mean(popu_age_matrix, axis=2)
y4 = np.mean(age_parti_matrix, axis=2)
y5 = np.mean(n_parti_matrix, axis=2)
y6 = -y2 * sigma_Y + mu_Y
y7 = np.nanmean(belief_age_matrix, axis=2)  # consumption-weighted beliefs for participants from each age group
y8 = np.nanmean(wealthshare_age_matrix, axis=2)  # wealth share each age group

xlabels = ['V_hat', 'interest rate', 'market price of risk', 'participation rate', 'age of participants',
           'number of cohorts', 'cutoff belief', 'cst belief of participants in age groups',
           'wealth share in age groups']
ys = [y0, y1, y2, y3, y4, y5, y6, y7, y8]

for i in range(len(ys)):
    for j in range(nu_dimension):
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        y_nu = ys[i]
        y = y_nu[j]
        if i == 3:
            ax.fill_between(x, y[:, 0], color='steelblue', linewidth=0.4, label='20 < Age <= 35, youngest quartile')
            ax.fill_between(x, y[:, 1], y[:, 0], color='darkseagreen', linewidth=0.4, label='35 < Age <= 55')
            ax.fill_between(x, y[:, 2], y[:, 1], color='moccasin', linewidth=0.4, label='55 < Age <= 89')
            ax.fill_between(x, y[:, 3], y[:, 2], color='pink', linewidth=0.4, label='Age > 89, oldest quartile')
            plt.legend()
        elif i == 7 or i == 8:
            ax.plot(x, y[:, 0], color='steelblue', linewidth=0.4, label='20 < Age <= 35, youngest quartile')
            ax.plot(x, y[:, 1], color='darkseagreen', linewidth=0.4, label='35 < Age <= 55')
            ax.plot(x, y[:, 2], color='moccasin', linewidth=0.4, label='55 < Age <= 89')
            ax.plot(x, y[:, 3], color='pink', linewidth=0.4, label='Age > 89, oldest quartile')
            plt.legend()
        else:
            ax.plot(x, y)
        ax.set_xlabel('initial window')
        if i == 0 or i == 6:
            ax.set_ylabel(xlabels[i])
        else:
            ax.set_ylabel('mean ' + xlabels[i])

        plt.savefig('initial window and ' + xlabels[i] + '_' + mode + '_' + zoom_in + str(nu) + '.png', dpi=500,
                    format="png")
        # plt.savefig('initial window and ' + xlabels[i] + '_' + mode + '.png', dpi=500, format="png")
        plt.show()
        plt.close()


