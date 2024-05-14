import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI, simulate_mix_types
from src.param import nu, mu_Y, sigma_Y, tax, phi, \
    dt, T_hat, Npre, Vhat, Ninit, Nt, Nc, tau, \
    cutoffs_age, scenarios, dZ_Y_cases, dZ_SI_cases, dZ_build_case, \
    dZ_SI_build_case, t, scenario_labels, colors_short, Ntype, alpha_i, \
    cohort_type_size, cohort_size
from src.param_mix import Nconstraint, cohort_type_size_mix

#############################################
### Additional parameters for the figures ###
#############################################
phi_vector = np.arange(0, 1, 0.1)
n_phi = len(phi_vector)

phi_indexes = [0, 4, 8]
n_phi_short = len(phi_indexes)
phi_vector_short = phi_vector[phi_indexes]

phi_indexes_5 = [0, 2, 4, 6, 8]
n_phi_5 = 5
phi_5 = phi_vector[phi_indexes_5]

label_phi = []
for i in range(n_phi_short):
    label_phi.append(r'$phi$ = ' + str(phi_vector_short[i]))
labels = [scenario_labels, label_phi, label_phi]

age_cutoff = cutoffs_age[2]

scenarios_two = scenarios[1:3]
Npre_short = np.array([60, 240])
T_hat_short = dt * Npre_short

plt.rcParams["font.family"] = 'serif'


print('Generating data for the graphs:')
n_scenarios_short = 4
scenarios_short = scenarios[:n_scenarios_short]
theta_compare = np.empty((n_scenarios_short, 3, 2, 2, Nt), dtype=np.float32)
popu_parti_compare = np.empty((n_scenarios_short, 3, 2, 2, Nt), dtype=np.float32)
r_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt), dtype=np.float32)
Delta_bar_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt), dtype=np.float32)
Delta_tilde_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt), dtype=np.float32)
Phi_bar_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt), dtype=np.float32)
Phi_tilde_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt), dtype=np.float32)
dR_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt), dtype=np.float32)
mu_S_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt), dtype=np.float32)
sigma_S_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt), dtype=np.float32)
beta_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt), dtype=np.float32)
Delta_compare = np.empty((n_scenarios_short, 3, 2, 2, Nt, Nconstraint, Nc), dtype=np.float16)
pi_compare = np.empty((n_scenarios_short, 3, 2, 2, Nt, Nconstraint, Nc), dtype=np.float16)
cons_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt, Ntype, Nconstraint, Nc), dtype=np.float16)
invest_tracker_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt, Nconstraint, Nc), dtype=np.float16)
Delta_mix = np.empty((n_scenarios_short, 3, 2, 2, Nt, Nconstraint, Nc), dtype=np.float16)
f_c_type_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt, Ntype), dtype=np.float16)
Delta_bar_type_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt, Ntype), dtype=np.float16)
f_w_type_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt, Ntype), dtype=np.float16)
Delta_tilde_type_compare = np.zeros((n_scenarios_short, 3, 2, 2, Nt, Ntype), dtype=np.float16)
cohort_type_size_mix_mat = np.tile(cohort_type_size_mix, (Nt, 1, 1, 1))
cohort_type_size_mat = np.tile(cohort_type_size, (Nt, 1, 1))
for g, scenario in enumerate(scenarios_short):
    print(g)
    for h in range(3):
        if h == 0:
            rho_i = np.array([[0.001], [0.05]])
        elif h == 1:
            rho_i = np.array([[-0.005], [0.045]])
        else:
            rho_i = np.array([[-0.015], [0.035]])
        beta_i = (nu + rho_i) / (1 + tax)  # consumption wealth ratio
        beta0 = np.sum(alpha_i * beta_i).astype(float)

        rho_i_mix = np.tile(np.reshape(rho_i, (-1, 1, 1)), (1, Nconstraint, 1))
        beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio
        rho_cohort_type = alpha_i * beta_i * np.exp(-(rho_i + nu) * tau)  # shape(2, 6000)

        for i in range(2):
            dZ = dZ_Y_cases[i]
            # log_Yt = np.cumsum((mu_Y - 0.5 * sigma_Y ** 2) * dt + sigma_Y * dZ)
            for j in range(2):
                dZ_SI = dZ_SI_cases[j]
                if i + j != 2:
                    print('skip')
                else:
                    if g <= 1:
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
                        ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax, beta0,
                                        phi,
                                        Npre, Ninit, T_hat, dZ_build_case, dZ, dZ_SI_build_case, dZ_SI, tau,
                                        cutoffs_age,
                                        Ntype, rho_i, alpha_i, beta_i, rho_cohort_type, cohort_type_size,
                                        need_f='True',
                                        need_Delta='True',
                                        need_pi='True',
                                        )
                        # invest_tracker = pi > 0
                        Delta_compare[g, h, i, j, :, 0] = Delta
                        pi_compare[g, h, i, j, :, 0] = pi
                        theta_compare[g, h, i, j] = theta
                        r_compare[g, h, i, j] = r
                        popu_parti_compare[g, h, i, j] = parti
                        Delta_bar_compare[g, h, i, j] = Delta_bar_parti
                        Delta_tilde_compare[g, h, i, j] = Delta_tilde_parti
                        Phi_bar_compare[g, h, i, j] = Phi_bar_parti
                        Phi_tilde_compare[g, h, i, j] = Phi_tilde_parti
                        cons_compare[g, h, i, j, :, :, 0] = f_c / cohort_type_size_mat
                        invest_tracker_compare[g, h, i, j, :, 0] = invest_tracker
                        dR_compare[g, h, i, j] = dR
                        mu_S_compare[g, h, i, j] = mu_S
                        sigma_S_compare[g, h, i, j] = sigma_S
                        beta_compare[g, h, i, j] = beta

                        Delta_type = np.tile(np.reshape(Delta, (Nt, 1, Nc)), (1, 2, 1))
                        f_c_type_compare[g, h, i, j] = np.sum(f_c, axis=2) * dt
                        Delta_bar_type_compare[g, h, i, j] = np.average(Delta_type, axis=2, weights=f_c)
                        beta_mat = np.tile(np.reshape(beta, (-1, 1, 1)), (1, 2, Nc))
                        beta_i_mat = np.tile(np.reshape(beta_i, (1, -1, 1)), (Nt, 1, Nc))
                        f_w = f_c * beta_mat / beta_i_mat
                        f_w_type_compare[g, h, i, j] = np.sum(f_w, axis=2) * dt
                        Delta_tilde_type_compare[g, h, i, j] = np.average(Delta_type, axis=2, weights=f_w)

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
                        Delta_compare[g, h, i, j] = Delta
                        pi_compare[g, h, i, j] = pi
                        theta_compare[g, h, i, j] = theta
                        r_compare[g, h, i, j] = r
                        popu_parti_compare[g, h, i, j] = parti
                        Delta_bar_compare[g, h, i, j] = Delta_bar_parti
                        Delta_tilde_compare[g, h, i, j] = Delta_tilde_parti
                        Phi_bar_compare[g, h, i, j] = Phi_bar_parti
                        Phi_tilde_compare[g, h, i, j] = Phi_tilde_parti
                        cons_compare[g, h, i, j] = f_c / cohort_type_size_mix_mat
                        invest_tracker_compare[g, h, i, j] = invest_tracker
                        dR_compare[g, h, i, j] = dR
                        mu_S_compare[g, h, i, j] = mu_S
                        sigma_S_compare[g, h, i, j] = sigma_S
                        beta_compare[g, h, i, j] = beta

pd_compare = 1/beta_compare  # I think right now the function returns pd itself

# ######################################
# ############## Figure ##############
# ######################################
# time-series of volatility, pd ratio
# phi = 0.4, bad z^Y, bad z&SI
red_case = 1
yellow_case = 1
sigma_S_mat = sigma_S_compare[:, :, red_case, yellow_case]
pd_mat = pd_compare[:, :, red_case, yellow_case]
y_list = [sigma_S_mat, pd_mat]
Z = np.cumsum(dZ_Y_cases[red_case])
Z_SI = np.cumsum(dZ_SI_cases[yellow_case])
y_title_list = [r'Stock volatility $\sigma^S_t$',
                r'Price dividend ratio']

fig_title_list = ['vola_', 'pd_']
# alternative_var_list = [r'$\rho=\{0.1\%, 0.5\%\}$',
#                         r'$\rho=\{0.1\%, 10\%\}$',
#                         r'$\rho=\{-1.5\%, 8.5\%\}$',]

alternative_var_list = [r'$\rho=\{0.1\%, 5\%\}$',
                        r'$\rho=\{-0.5\%, 4.5\%\}$',
                        r'$\rho=\{-1.5\%, 3.5\%\}$',]

scenario_labels = ['Shocks', 'Complete', 'Reentry', 'Mix-4', 'Mix-BC']
n_rows = 5
for jj in range(2):  # variable, sigma_S if jj == 0, pd if jj == 1
    fig, axes = plt.subplots(nrows=n_rows, ncols=1, sharex='all', figsize=(8, 12))
    y_mat = y_list[jj]
    for j, ax in enumerate(axes):
        y_title = scenario_labels[j]  # scenarios, subfigs in a fig
        if j == 0:
            ax.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
            ax.plot(t[1:], Z[1:], color='red', linewidth=0.5, label=r'$z^Y_t$')
            ax.plot(t[1:], Z_SI[1:], color='gold', linewidth=0.5, label=r'$z^{SI}_t$')
            ax.tick_params(axis='both', labelcolor='black')
        else:
            y_vec = y_mat[j - 1]
            ax.set_ylabel(y_title_list[jj], color='black')
            for i in range(3):
                y = y_vec[i]  # Nt
                color_i = colors_short[i]
                ax.plot(t[1:], y[1:], label=alternative_var_list[i], color=color_i, linewidth=0.4)
                # ax2.set_ylim(lower, upper)
        if j <= 1:
            ax.legend(loc='upper left')
        if j == n_rows:
            ax.set_xlabel('Time in simulation')
        ax.set_title(y_title)
    fig.tight_layout(h_pad=2)
    plt.savefig(fig_title_list[jj]+'HD.png', dpi=200)
    plt.show()
    # plt.close()


# Time-series of consumption-weighted beliefs and consumption share of type A
red_case = 1
yellow_case = 1
Delta_bar_mat = Delta_bar_compare[:, :, red_case, yellow_case]  # shape (3 * 3 * 6000)
Delta_bar_type_mat = Delta_bar_type_compare[:, :, red_case, yellow_case]  # shape (3 * 3 * 6000 * 2)
f_c_type_mat = f_c_type_compare[:, :, red_case, yellow_case, :, 0]   # shape (3 * 3 * 6000)
Delta_tilde_type_mat = Delta_tilde_type_compare[:, :, red_case, yellow_case]  # shape (3 * 3 * 6000 * 2 )
f_w_type_mat = f_w_type_compare[:, :, red_case, yellow_case, :, 0]   # shape (3 * 3 * 6000)
y_list = [Delta_bar_mat, Delta_bar_type_mat, f_c_type_mat, Delta_tilde_type_mat, f_w_type_mat]
Z = np.cumsum(dZ_Y_cases[red_case])
Z_SI = np.cumsum(dZ_SI_cases[yellow_case])
y_title_list = ['Shocks',
                r'Consumption weighted belief $\bar{\Delta}_t$',
                r'Cons average belief $\bar{\Delta}_{a, t}$, and $\bar{\Delta}_{b, t}$',
                r'Cons share $f^c_{a, t}$',
                r'Wealth average belief $\tilde{\Delta}_{a, t}$, and $\tilde{\Delta}_{b, t}$',
                r'Wealth share $f^w_{a, t}$',
                ]
fig_title_list = ['Complete', 'Reentry', 'Mix-4', 'Mix-BC']

alternative_var_list = [r'$\rho=\{0.1\%, 5\%\}$',
                        r'$\rho=\{-0.5\%, 4.5\%\}$',
                        r'$\rho=\{-1.5\%, 3.5\%\}$',]
# alternative_var_list = [r'$\rho=\{0.1\%, 0.5\%\}$',
#                         r'$\rho=\{0.1\%, 10\%\}$',
#                         r'$\rho=\{-1.5\%, 8.5\%\}$']

n_rows = len(y_title_list)
for jj in range(4):
    fig, axes = plt.subplots(nrows=n_rows, ncols=1, sharex='all', figsize=(8, 15))
    for j, ax in enumerate(axes):
        y_title = y_title_list[j]  # scenarios, subfigs in a fig
        if j == 0:
            ax.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
            ax.plot(t[1:], Z[1:], color='red', linewidth=0.5, label=r'$z^Y_t$')
            ax.plot(t[1:], Z_SI[1:], color='gold', linewidth=0.5, label=r'$z^{SI}_t$')
            ax.tick_params(axis='both', labelcolor='black')
        else:
            y_vec = y_list[j - 1][jj]
            ax.set_ylabel(y_title_list[j], color='black')
            if j == 2 or j == 4:
                for i in range(3):
                    for ii in range(2):
                        y = y_vec[i, :, ii]  # Nt
                        color_i = colors_short[i]
                        line_style = 'solid' if ii == 0 else 'dotted'
                        ax.plot(t[1:], y[1:], label=alternative_var_list[i], color=color_i, linewidth=0.4, linestyle=line_style)
                        # ax2.set_ylim(lower, upper)
            else:
                for i in range(3):
                    y = y_vec[i]  # Nt
                    color_i = colors_short[i]
                    ax.plot(t[1:], y[1:], label=alternative_var_list[i], color=color_i, linewidth=0.8)
                    # ax2.set_ylim(lower, upper)
        if j <= 1:
            ax.legend(loc='upper left')
        if j == n_rows:
            ax.set_xlabel('Time in simulation')
        ax.set_title(y_title)
    fig.tight_layout(h_pad=2)
    plt.savefig(fig_title_list[jj]+'HD.png', dpi=200)
    plt.show()
    # plt.close()
