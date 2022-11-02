# ######################################
# ############ Figure 3.1 ##############
# ######################################
N_1 = 1000
phi_5 = np.linspace(0, 0.8, 5)
Delta_reentry_matrix = np.empty((N_1, 5, Nc))
Delta_complete_matrix = np.empty((N_1, 5, Nc))
invest_matrix = np.empty((N_1, 5, Nc))
dt_root = np.sqrt(dt)
for j in range(N_1):
    print(j)
    dZ = np.random.randn(Nt) * dt_root
    dZ_build = np.random.randn(Nc) * dt_root
    dZ_SI = np.random.randn(Nt) * dt_root
    dZ_SI_build = np.random.randn(Nc) * dt_root
    for l, phi_try in enumerate(phi_5):
        (
            r_reentry,
            theta_reentry,
            f_reentry,
            Delta_reentry,
            pi_reentry,
            popu_parti_reentry,
            f_parti_reentry,
            Delta_bar_parti_reentry,
            dR_reentry,
            invest_tracker_reentry,
        ) = simulate_SI('w_constraint', 'keep', Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta, phi_try,
                        Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                        top=0.05,
                        old_limit=100
                        )
        Delta_reentry_matrix[j, l] = np.average(np.abs(Delta_reentry), axis=0)
        invest_matrix[j, l] = np.average(invest_tracker_reentry, axis=0)
        (
            r_complete,
            theta_complete,
            f_complete,
            Delta_complete,
            pi_complete,
            popu_complete,
            f_parti_complete,
            Delta_bar_complete,
            dR_complete,
            invest_tracker_complete,
        ) = simulate_SI('complete', 'keep', Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta, phi_try,
                        Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                        top=0.05,
                        old_limit=100
                        )
        Delta_complete_matrix[j, l] = np.average(np.abs(Delta_complete), axis=0)
Delta_reentry_vector = np.flip(np.average(Delta_reentry_matrix, axis=0), axis=1)
invest_vector = np.flip(np.average(invest_matrix, axis=0), axis=1)
Delta_complete_vector = np.flip(np.average(Delta_complete_matrix, axis=0), axis=1)

# Graph:
t_cut = 100
N_cut = int(t_cut/dt)
x = t[:N_cut]
data_point = np.arange(0, N_cut, 15)
y_cases = [Delta_complete_vector, invest_vector, Delta_reentry_vector]
fig_titles = ['Complete market', 'Reentry', 'Reentry']
y_titles = [r'Average $\mid\Delta_{s,t}\mid$', 'Average participation probability', r'Average $\mid\Delta_{s,t}\mid$']
fig, axes = plt.subplots(nrows=1, ncols=3, sharex='all', figsize=(15, 5))
for j, ax in enumerate(axes):
    ax.set_xlabel('Age')
    y_case = y_cases[j]
    ax.set_ylabel(y_titles[j])
    for i in range(5):
        y = y_case[i, :N_cut]
        label_i = r'$\phi$=' + str('{0:.2f}'.format(phi_5[i]))
        ax.plot(x[data_point], y[data_point], color=colors[i], linewidth=0.5, label=label_i)
    if j == 0 or j == 2:
        ax.set_ylim(0.04, 0.18)
    if j == 0:
        ax.legend()
    ax.set_title(fig_titles[j], color='black')
    ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(str(N_1) + 'paths, ' + str(t_cut) + 'age, ' +'Average estimation error and age.png', dpi=100)
plt.show()
#plt.close()


# ######################################
# ############ Figure 3.2 ##############
# ######################################
N_1 = 500
age_cut = 100
Nc_cut = int(age_cut/dt)
# drift_N_matrix = np.empty((N_1, n_scenarios, n_phi_short, Nc_cut))
drift_P_matrix = np.empty((N_1, n_scenarios, n_phi_short, Nc_cut))
diffusion_P_matrix = np.empty((N_1, n_scenarios, n_phi_short, Nc_cut))
r_matrix = np.empty((N_1, n_scenarios, n_phi_short))
dt_root = np.sqrt(dt)
for j in range(N_1):
    print(j)
    dZ = np.random.randn(Nt) * dt_root
    dZ_build = np.random.randn(Nc) * dt_root
    dZ_SI = np.random.randn(Nt) * dt_root
    dZ_SI_build = np.random.randn(Nc) * dt_root
    for k, scenario in enumerate(scenarios_short):
        mode_trade = scenario[0]
        mode_learn = scenario[1]
        for l, phi_try in enumerate(phi_vector_short):
            (
                r,
                theta,
                f,
                Delta,
                pi,
                popu_parti,
                f_parti,
                Delta_bar_parti,
                dR,
                invest_tracker,
            ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta, phi_try,
                            Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size,
                            top=0.05,
                            old_limit=100
                            )
            r_matrix[j, k, l] = np.average(r)
            theta_mat = np.transpose(np.tile(theta, (Nc_cut, 1)))
            r_mat = np.transpose(np.tile(r, (Nc_cut, 1)))
            # drift_N = -rho + r_mat
            drift_P = -rho + r_mat + 0.5 * theta_mat**2 - 0.5 * Delta[:, -Nc_cut:]**2
            diffusion_P = np.abs(theta_mat + Delta[:, -Nc_cut:])
            # drift_N_matrix[j, k, l] = np.average(drift_N, weights=(1-invest_tracker[:, Nc_cut]), axis=0)
            drift_P_matrix[j, k, l] = np.average(drift_P, weights=invest_tracker[:, -Nc_cut:], axis=0)
            diffusion_P_matrix[j, k, l] = np.average(diffusion_P, weights=invest_tracker[:, -Nc_cut:], axis=0)
# drift_N_vector = np.flip(np.average(drift_N_matrix, axis=0), axis=2)
drift_P_vector = np.flip(np.average(drift_P_matrix, axis=0), axis=2)  # (n_scenarios, n_phi_short, Nc_cut)
diffusion_P_vector = np.flip(np.average(diffusion_P_matrix, axis=0), axis=2)
r_vector = np.average(r_matrix, axis=0)  # (n_scenarios, n_phi_short)
np.save('drift_P_data', drift_P_matrix)
np.save('diffusion_P_data', diffusion_P_matrix)
np.save('drift_N_data', r_matrix)

# Graph:
x = t[:Nc_cut]
fig, axes = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='col', figsize=(15, 8))
for j, ax in enumerate(axes.flat):
    if j > 1:
        ax.set_xlabel('Age')
        ax.set_title('Complete market')
    else:
        ax.set_xlabel('Age')
        ax.set_title('Reentry')
    y_case = drift_P_vector if j == 0 or j == 2 else diffusion_P_vector
    y_sce = y_case[1] if j <= 1 else y_case[0]  # (n_phi_short, Nc_cut)
    for i in range(n_phi_short):
        y = y_sce[i]  # (Nc_cut)
        if j == 0:
            ax.plot(x, y, color=colors_short[i], linewidth=0.8, label='Participants')
            if i == 0:
                ax.axhline(r_vector[1, i] - rho, 0.05, 0.95, color=colors_short[i], linestyle= 'dashed', linewidth=0.8, label='Nonparticipants')
                ax.legend()
            else:
                ax.axhline(r_vector[1, i] - rho, 0.05, 0.95, color=colors_short[i], linestyle= 'dashed', linewidth=0.8)
        else:
            ax.plot(x, y, color=colors_short[i], linewidth=0.8, label=label_phi[i])
            if j == 1:
                ax.legend()
    if j == 0 or j == 2:
        ax.set_ylabel('Drift of log consumption', color='black')
    else:
        ax.set_ylabel('Volatility of log consumption', color='black')
    ax.tick_params(axis='y', labelcolor='black')
fig.tight_layout(h_pad=2)  # otherwise the right y-label is slightly clipped
plt.savefig(str(N_1) + 'paths, ' + 'log consumption and age.png', dpi=200)
plt.show()
#plt.close()
