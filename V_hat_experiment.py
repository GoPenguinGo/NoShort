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
n_scenarios = 3
scenarios_short = scenarios[:n_scenarios]

# Generate matrix to store the results
r_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))  # for mean and std
theta_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
f_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
popu_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
Delta_bar_parti_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, 2))
popu_age_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, n_age_groups, 2))
belief_age_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, n_age_groups, 2))
wealthshare_age_matrix = np.zeros((N, n_scenarios, n_phi, T_hat_dimension, n_age_groups, 2))


for l in range(N):
    print(l)
    dZ = dZ_matrix[l]
    dZ_build = dZ_build_matrix[l]
    dZ_SI = dZ_SI_matrix[l]
    dZ_SI_build = dZ_SI_build_matrix[l]
    time_s = time.time()
    for m, scenario in enumerate(scenarios_short):
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        for n, phi in enumerate(phi_vector):
            for o, T_hat in enumerate(T_hats):
                Npre = int(Npres[o])
                Vhat = (sigma_Y ** 2) / T_hat  # prior variance
                print(T_hat, Npre, Vhat)

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
                    w,
                    age_parti,
                    n_parti,
                ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta,
                                phi,
                                Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                                top=0.05,
                                old_limit=100
                                )

                r_matrix[l, m, n, o, 0] = np.mean(r)
                r_matrix[l, m, n, o, 1] = np.std(r)  # todo: maybe adjust the std level against mean later
                theta_matrix[l, m, n, o, 0] = np.mean(theta)
                theta_matrix[l, m, n, o, 1] = np.std(theta)
                popu_parti_matrix[l, m, n, o, 0] = np.mean(popu_parti)
                popu_parti_matrix[l, m, n, o, 1] = np.std(popu_parti)
                Delta_bar_parti_matrix[l, m, n, o, 0] = np.mean(Delta_bar_parti)
                Delta_bar_parti_matrix[l, m, n, o, 1] = np.std(Delta_bar_parti)
                f_parti_matrix[l, m, n, o, 0] = np.mean(f_parti)
                f_parti_matrix[l, m, n, o, 1] = np.std(f_parti)
                invest = (pi > 0)
                parti_rate = invest * cohort_size

                belief = (Delta * sigma_Y + mu_Y)
                belief_weights = f * dt

                for i in range(4):
                    popu_age_matrix[l, m, n, o, i, 0] = np.mean(np.sum(parti_rate[:, cutoffs[i + 1]:], axis=1))
                    popu_age_matrix[l, m, n, o, i, 1] = np.std(np.sum(parti_rate[:, cutoffs[i + 1]:], axis=1))

                    belief_age_matrix[l, m, n, o, i, 0] = np.mean(
                        np.average(
                            belief[:, cutoffs[i + 1]:cutoffs[i]], weights=cohort_size[cutoffs[i + 1]:cutoffs[i]], axis=1
                        ))
                    belief_age_matrix[l, m, n, o, i, 1] = np.std(
                        np.average(
                            belief[:, cutoffs[i + 1]:cutoffs[i]], weights=cohort_size[cutoffs[i + 1]:cutoffs[i]], axis=1
                        ))

                    wealthshare_age_matrix[l, m, n, o, i, 0] = np.mean(
                        np.sum(f[:, cutoffs[i + 1]:cutoffs[i]] * dt, axis=1)
                    )
                    wealthshare_age_matrix[l, m, n, o, i, 1] = np.std(
                        np.sum(f[:, cutoffs[i + 1]:cutoffs[i]] * dt, axis=1)
                    )

    print(time.time() - time_s)




# The main loop builds up the economy with a large number of cohorts, and simulates the stationary economy forward
# for l in range(Mpaths):
#     dZ = dZ_matrix[l]
#     dZ_build = dZ_build_matrix[l]
#
#     for k, T_hat in enumerate(T_hats):
#         Npre = int(Npres[k])
#         Vhat = (sigma_Y ** 2) / T_hat  # prior variance
#         print(T_hat, Npre, Vhat)
#
#         for m, nu in enumerate(nus):
#             # this part is repetitive when there is only one value of nu
#             beta = rho + nu - tax
#
#             tau = np.arange(T_cohort, 0, -dt)  # age from 500 to 0
#             cohort_size = nu * np.exp(-nu * (tau - dt)) * dt  # cohort size when a new cohort is just born
#
#             # create age quartiles for analysis
#             cummu_popu = np.cumsum(cohort_size)
#             tau_cutoff1 = np.searchsorted(cummu_popu, 0.75)
#             tau_cutoff2 = np.searchsorted(cummu_popu, 0.5)
#             tau_cutoff3 = np.searchsorted(cummu_popu, 0.25)
#             cutoffs = [Nc, tau_cutoff1, tau_cutoff2, tau_cutoff3, 0]
#
#             if mode == 'drop' or mode == 'keep' or mode == 'comp':
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
#                     w_cohort,
#                     age_parti,
#                     n_parti,
#                 ) = simulate(mode, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta, Npre, Ninit, T_hat,
#                              dZ_build, dZ, tau,
#                              cohort_size)
#                 invest_tracker = pi > 0
#
#             elif mode == 'rich_free' or mode == 'back_collect' or mode == 'back_renew':
#                 (
#                     r,
#                     theta,
#                     f,
#                     Delta,
#                     d_eta,
#                     pi,
#                     dR,
#                     w_cohort,
#                     popu_parti,
#                     popu_can_short,
#                     popu_short,
#                     popu_long,
#                     f_parti,
#                     f_short,
#                     f_long,
#                     age_parti,
#                     age_short,
#                     age_long,
#                     n_parti,
#                     invest_tracker,
#                     can_short_tracker,
#                     long,
#                     short,
#                     Delta_bar_parti,
#                     Delta_bar_long,
#                     Delta_bar_short,
#                 ) = simulate_partial_constraint(mode, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta, Npre,
#                                                 Ninit, T_hat, dZ_build, dZ, tau, cohort_size)
#             else:
#                 print('Error! Mode not defined')
#                 break
#             dR_matrix[k, m, l] = np.mean(dR)
#             r_matrix[k, m, l] = np.mean(r)
#             theta_matrix[k, m, l] = np.mean(theta)
#             popu_parti_matrix[k, m, l] = np.mean(popu_parti)
#             Delta_bar_parti_matrix[k, m, l] = np.mean(Delta_bar_parti)
#             f_parti_matrix[k, m, l] = np.mean(f_parti)
#             parti_rate = invest_tracker * cohort_size
#
#             belief = (Delta * sigma_Y + mu_Y)
#             belief_weights = f * dt
#
#             for i in range(4):
#                 popu_age_matrix[k, m, l, i] = np.mean(np.sum(parti_rate[:, cutoffs[i + 1]:], axis=1))
#
#                 # weights_zero = (np.sum(invest_tracker[:, cutoffs[i + 1]:cutoffs[i]],
#                 #                        axis=1) == 0)  # no one from the age group is participating
#                 # belief_copy = belief.copy()
#                 # if np.sum(weights_zero) == Nt:
#                 #     belief_age_matrix[k, m, l, i] = np.nan
#                 # else:
#                 #     # weights_zero = np.transpose(np.tile(weights_zero, (Nc, 1)))
#                 #     a = np.where(weights_zero == 1)
#                 #     belief_copy[a, :] = np.nan
#                 #     belief_age_matrix[k, m, l, i] = np.nanmean(
#                 #         np.average(belief_copy[:, cutoffs[i + 1]:cutoffs[i]],
#                 #                    weights=belief_weights[:, cutoffs[i + 1]:cutoffs[i]], axis=1)
#                 #     )
#                 belief_age_matrix[k, m, l, i] = np.mean(
#                     belief[:, cutoffs[i + 1]:cutoffs[i]]
#                 )
#
#                 wealthshare_age_matrix[k, m, l, i] = np.mean(
#                     np.sum(f[:, cutoffs[i + 1]:cutoffs[i]] * dt, axis=1)
#                 )
#
#             age_parti_matrix[k, m, l] = np.mean(age_parti)
#             n_parti_matrix[k, m, l] = np.mean(n_parti)
#
#         # covariance:
#     print(l)

# graphs:
x = Npres
y0 = (np.ones(len(Npres)) * sigma_Y ** 2) / x
y1 = np.mean(r_matrix, axis=1)
y2 = np.mean(theta_matrix, axis=1)
y3 = np.mean(popu_age_matrix, axis=1)
y4 = np.mean(age_parti_matrix, axis=1)
y5 = np.mean(n_parti_matrix, axis=1)
y6 = -y2 * sigma_Y + mu_Y
y7 = np.nanmean(belief_age_matrix, axis=1)  # consumption-weighted beliefs for participants from each age group
y8 = np.nanmean(wealthshare_age_matrix, axis=1)  # wealth share each age group
y9 = np.mean(f_parti_matrix, axis=1)
y10 = np.mean(Delta_bar_parti_matrix, axis=1)

xlabels = ['V_hat', 'interest rate', 'market price of risk', 'participation rate', 'average age of participants',
           'number of cohorts', 'cutoff belief to participate', 'average belief in age groups',
           'wealth share in age groups', 'consumption share of participants', 'estimation error of participants']
ys = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9]

for i in range(len(ys)):
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    y = ys[i]
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
    ax.set_xlabel('initial window (months)')
    if i == 0 or i == 6:
        ax.set_ylabel(xlabels[i])
    else:
        ax.set_ylabel('mean ' + xlabels[i])

    # plt.savefig('initial window and ' + xlabels[i] + '_' + mode_learn + '_' + mode_trade + str(nu) + '.png', dpi=200, format="png")
    # plt.savefig('initial window and ' + xlabels[i] + '_' + mode + '.png', dpi=500, format="png")
    plt.show()
    # plt.close()

