import time
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate, simulate_partial_constraint
from src.param import *


# modes = ['drop', 'ric_free']
modes = ['rich_free']
# modes = ['drop']
# zoom_in = 'small'
zoom_in = 'large'
T_hats = dt * np.arange(1, 13, 1) if zoom_in == 'small' else np.arange(40, 51, 1)
T_hat_dimension = len(T_hats)

# for graphs:
Tkeep = 100
Nkeep = int(Tkeep / dt)
Tsample = int(T_cohort / 100)
Nsamples = 500

# Generate matrix to store the results
dZ_matrix = np.zeros((Mpaths, Nt))
Z_matrix = np.zeros((Mpaths, Nt))
mu_C_matrix = np.zeros((T_hat_dimension, Mpaths))
sigma_C_matrix = np.zeros((T_hat_dimension, Mpaths))
# delta_matrix = np.zeros((T_hat_dimension, Mpaths, Nt, Nc))
r_matrix = np.zeros((T_hat_dimension, Mpaths))
# f_matrix = np.zeros((T_hat_dimension, Mpaths, Nt, Nc))
theta_matrix = np.zeros((T_hat_dimension, Mpaths))
pi_matrix = np.zeros((T_hat_dimension, Mpaths))
f_parti_matrix = np.zeros((T_hat_dimension, Mpaths))
popu_parti_matrix = np.zeros((T_hat_dimension, Mpaths))
# Delta_bar_parti_matrix = np.zeros((T_hat_dimension, Mpaths, Nt))
# w_matrix = np.zeros((T_hat_dimension, Mpaths, Nt, Nc))
# w_cohort_matrix = np.zeros((T_hat_dimension, Mpaths, Nt, Nc))
age_parti_matrix = np.zeros((T_hat_dimension, Mpaths))
n_parti_matrix = np.zeros((T_hat_dimension, Mpaths))
popu_age1_matrix = np.zeros((T_hat_dimension, Mpaths))
popu_age2_matrix = np.zeros((T_hat_dimension, Mpaths))
popu_age3_matrix = np.zeros((T_hat_dimension, Mpaths))


# invest_tracker_matrix = np.zeros((T_hat_dimension, Mpaths, Nt, Nc))

# Expected returns
# mu_S_matrix = np.zeros((T_hat_dimension, Mpaths))  # Expected returns under the true measure
# mu_S_s_matrix = np.zeros((T_hat_dimension, Mpaths, Nt, Nc))  # Expected returns under the measure of the agent we track
# mu_hat_S_matrix = np.zeros((T_hat_dimension, Mpaths, Nt))  # Simple average of expected returns, or consensus belief

# Equity risk premium
# erp_S_matrix = np.zeros((T_hat_dimension, Mpaths, Nt))
# erp_S_s_matrix = np.zeros((T_hat_dimension, Mpaths, Nt, Nc))
# erp_hat_S_matrix = np.zeros((T_hat_dimension, Mpaths, Nt))

Et_matrix = np.zeros((T_hat_dimension, Mpaths))
Vt_matrix = np.zeros((T_hat_dimension, Mpaths))
dR_matrix = np.zeros((T_hat_dimension, Mpaths))

# The main loop builds up the economy with a large number of cohorts, and simulates the stationary economy forward
for mode in modes:

    for l in range(Mpaths):
        s = time.time()
        # same shocks for the different T_hats
        dZ_build = dt ** 0.5 * np.random.randn(int(Nc - 1))  # dZt for the build function
        dZ = dt ** 0.5 * np.random.randn(Nt)  # dZt for the simulate function
        dZ_matrix[l, :] = dZ
        Z = np.cumsum(dZ)
        Z_matrix[l, :] = Z

        for k, T_hat in enumerate(T_hats):
            Npre = int(T_hat / dt)
            Vhat = (sigma_Y ** 2) / T_hat  # prior variance
            biasvec = dZ_build[-Npre:]  # dZt used in the build_cohorts function

            if mode == 'drop':
                (
                    mu_S,
                    mu_S_s,
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
                    w,
                    w_cohort,
                    age_parti,
                    n_parti,
                ) = simulate(mode, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, beta, Npre, T_hat, dZ_build, dZ, tau,
                             cohort_size)

                dR_matrix[k, l] = np.average(dR[1200:])
                r_matrix[k, l] = np.average(r[1200:])
                theta_matrix[k, l] = np.average(theta[1200:])
                popu_parti_matrix[k, l] = np.average(popu_parti[1200:])
                invest = pi > 0
                parti_rate = invest * cohort_size
                popu_age1_matrix[k, l] = np.average(np.sum(parti_rate[1200:, tau_cutoff1:], axis=1))
                popu_age2_matrix[k, l] = np.average(np.sum(parti_rate[1200:, tau_cutoff2:], axis=1))
                popu_age3_matrix[k, l] = np.average(np.sum(parti_rate[1200:, tau_cutoff3:], axis=1))
                age_parti_matrix[k, l] = np.average(age_parti[1200:])
                n_parti_matrix[k, l] = np.average(n_parti[1200:])

            else:
                (
                    mu_S,
                    mu_S_s,
                    r,
                    theta,
                    f,
                    Delta,
                    d_eta,
                    pi,
                    dR,
                    w,
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
                ) = simulate_partial_constraint(mode, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, beta, omega, Npre,
                                                T_hat, dZ_build, dZ, tau, cohort_size)

                dR_matrix[k, l] = np.average(dR[1200:])
                r_matrix[k, l] = np.average(r[1200:])
                theta_matrix[k, l] = np.average(theta[1200:])
                popu_parti_matrix[k, l] = np.average(popu_parti[1200:])
                invest = pi > 0
                parti_rate = invest * cohort_size
                popu_age1_matrix[k, l] = np.average(np.sum(parti_rate[1200:, tau_cutoff1:], axis=1))
                popu_age2_matrix[k, l] = np.average(np.sum(parti_rate[1200:, tau_cutoff2:], axis=1))
                popu_age3_matrix[k, l] = np.average(np.sum(parti_rate[1200:, tau_cutoff3:], axis=1))
                age_parti_matrix[k, l] = np.average(age_parti[1200:])
                n_parti_matrix[k, l] = np.average(n_parti[1200:])

            # covariance:

    # graphs:
    x = T_hats
    y0 = (np.ones(len(T_hats)) * sigma_Y ** 2) / x
    y1 = np.average(r_matrix, axis=1)
    y2 = np.average(theta_matrix, axis=1)
    y3 = np.average(popu_parti_matrix, axis=1)
    y31 = np.average(popu_age1_matrix, axis=1)
    y32 = np.average(popu_age2_matrix, axis=1)
    y33 = np.average(popu_age3_matrix, axis=1)
    y4 = np.average(age_parti_matrix, axis=1)
    y5 = np.average(n_parti_matrix, axis=1)
    y6 = -y2 * sigma_Y + mu_Y

    xlabels = ['V_hat', 'interest rate', 'market price of risk', 'participation rate', 'age of participants',
               'number of cohorts', 'cutoff belief']
    ys = [y0, y1, y2, y3, y4, y5, y6]

    for i in range(7):
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        y = ys[i]
        if i == 3:
            ax.fill_between(x, y31, color='steelblue', linewidth=0.4, label='20 < Age <= 35, youngest quartile')
            ax.fill_between(x, y32, y31, color='darkseagreen', linewidth=0.4, label='35 < Age <= 55')
            ax.fill_between(x, y33, y32, color='moccasin', linewidth=0.4, label='55 < Age <= 89')
            ax.fill_between(x, y, y33, color='pink', linewidth=0.4, label='Age > 89, oldest quartile')
        else:
            ax.plot(x, y)
        ax.set_xlabel('initial window')
        if i == 0 or i == 6:
            ax.set_ylabel(xlabels[i])
        else:
            ax.set_ylabel('mean ' + xlabels[i])
        plt.savefig('initial window and ' + xlabels[i] + '_' + mode + '_' + zoom_in + '.png', dpi=500)
        # plt.savefig('initial window and ' + xlabels[i] + '_' + mode + '.png', dpi=500)







