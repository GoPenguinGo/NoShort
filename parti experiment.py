import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI, simulate_mix_types
from src.param import nu, mu_Y, sigma_Y, tax, phi, \
    dt, T_hat, Npre, Vhat, Ninit, Nt, Nc, tau, \
    cutoffs_age, scenarios, dZ_Y_cases, dZ_SI_cases, dZ_build_case, \
    dZ_SI_build_case, t, scenario_labels, colors_short, Ntype, alpha_i, \
    dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    cohort_type_size, cohort_size, beta_i, beta0, rho_i, rho_cohort_type
from src.param_mix import Nconstraint, beta_i_mix

## scatter plots for expected returns, vola, excess returns and r as function of participation
## maybe no linear relation between parti and vola, but parti and r

plt.rcParams["font.family"] = 'serif'

Mpaths = 50
print('Generating data for the graphs:')
n_scenarios_short = 1
window = 24  # 2-year non-overlapping windows
N_data_point = int(Nt / window - 1)
scenarios_short = scenarios[1:1 + n_scenarios_short]
popu_parti_compare = np.empty((Mpaths, n_scenarios_short, N_data_point), dtype=np.float32)
popu_parti_old = np.empty((Mpaths, n_scenarios_short, N_data_point), dtype=np.float32)
popu_parti_young = np.empty((Mpaths, n_scenarios_short, N_data_point), dtype=np.float32)
r_compare = np.zeros((Mpaths, n_scenarios_short, N_data_point), dtype=np.float32)
dR_compare = np.zeros((Mpaths, n_scenarios_short, N_data_point), dtype=np.float32)
mu_S_compare = np.zeros((Mpaths, n_scenarios_short, N_data_point), dtype=np.float32)
sigma_S_compare = np.zeros((Mpaths, n_scenarios_short, N_data_point), dtype=np.float32)
vola_S_compare = np.zeros((Mpaths, n_scenarios_short, N_data_point), dtype=np.float32)
pd_compare = np.zeros((Mpaths, n_scenarios_short, N_data_point), dtype=np.float32)
for i in range(Mpaths):
    dZ_build = dZ_build_matrix[i]
    dZ = dZ_matrix[i]
    dZ_SI_build = dZ_SI_build_matrix[i]
    dZ_SI = dZ_SI_matrix[i]
    for g, scenario in enumerate(scenarios_short):
        if g == 0:
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

        else:
            alpha_constraint = np.ones((1, Nconstraint)) * 1 / Nconstraint if g == 2 else np.ones(
                (1, Nconstraint)) * (0.5, 0.5, 0, 0)
            alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
            cohort_type_size_mix = cohort_size * alpha_i_mix
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
                                   dZ_build_case, dZ, dZ_SI_build_case, dZ_SI,
                                   cutoffs_age, Ntype,
                                   Nconstraint, rho_i_mix, alpha_i_mix, beta_i_mix,
                                   rho_cohort_type_mix,
                                   cohort_type_size_mix,
                                   need_f='True',
                                   need_Delta='True',
                                   need_pi='True',
                                   )
        R_cumu = np.cumsum(dR)
        R2_cumu = np.cumsum(dR ** 2)
        vola_cumu = np.cumsum(sigma_S ** 2)
        r_cumu = np.cumsum(r)
        mu_S_cumu = np.cumsum(mu_S)
        r_compare[i, g] = np.reshape((r_cumu[window:] - r_cumu[:-window]) / window, (-1, window))[:, 0]
        popu_parti_compare[i, g] = np.reshape(parti, (-1, window))[:-1, 0]
        popu_parti_old[i, g] = np.reshape(parti_age_group[:, 0], (-1, window))[:-1, 0]
        popu_parti_young[i, g] = np.reshape(parti_age_group[:, 3], (-1, window))[:-1, 0]
        dR_compare[i, g] = np.reshape((R_cumu[window:] - R_cumu[:-window]) / window, (-1, window))[:, 0]
        vola_S_compare[i, g] = np.reshape((R2_cumu[window:] - R2_cumu[:-window]) / window, (-1, window))[:, 0]
        mu_S_compare[i, g] = np.reshape((mu_S_cumu[window:] - mu_S_cumu[:-window]) / window, (-1, window))[:, 0]
        sigma_S_compare[i, g] = np.reshape(np.sqrt((vola_cumu[window:] - vola_cumu[:-window]) / window), (-1, window))[
                                :, 0]
        pd_compare[i, g] = np.reshape(1 / beta, (-1, window))[1:, 0]

## todo: when is the slope upward and when is downward?

# plot:
y_set = [[
    r_compare,
    dR_compare,
    mu_S_compare
],
    [sigma_S_compare,
     vola_S_compare,
     pd_compare
     ]]
x_set = [
    popu_parti_compare,
    popu_parti_old / popu_parti_compare,
    popu_parti_young / popu_parti_compare,
]
label_x_set = [
    r'Participation rate, $P_t$',
    # r'Participation rate among the oldest quartile',
    r'Oldest quartile / total',
    # r'Participation rate among the youngest quartile',
    r'Youngest quartile / total',
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
for ii, x in enumerate(x_set):
    fig, axes = plt.subplots(nrows=3, ncols=2, sharex='all', figsize=(10, 15))
    for i, row in enumerate(axes):
        for j, ax in enumerate(row):
            ax.grid(True)
            y_title = label_set[j][i]
            y = y_set[j][i]
            ax.scatter(x, y, s=1, marker='.', c='red')
            ax.set_xlabel(label_x_set[ii])
            ax.set_title(y_title)
            # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            # fig.savefig('r and theta, subfig ' + str(j + 1) + '.png', bbox_inches=extent.expanded(1.2, 1.3), dpi=200)
    fig.tight_layout(h_pad=2)
    # plt.savefig(str(ii)+'scatter.png', dpi=200)
    plt.show()
    # plt.close()

