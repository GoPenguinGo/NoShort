import numpy as np

theta_matrix1 = np.empty((n_phi, Nt))
popu_parti_matrix1 = np.empty((n_phi, Nt))
market_view_matrix1 = np.empty((n_phi, Nt))
survey_view_matrix1 = np.empty((n_phi, Nt))
Delta_matrix1 = np.empty((n_phi, Nt, Nc))
pi_matrix1 = np.empty((n_phi, Nt, Nc))
mu_st_rt_matrix1 = np.empty((n_phi, Nt, Nc))
f_matrix1 = np.empty((n_phi, Nt, Nc))



dZ = dZ_SI_matrix[0]
dZ_build = dZ_build_matrix[0]
dZ_SI = dZ_matrix[0]
dZ_SI_build = dZ_SI_build_matrix[0]

for i, phi in enumerate(phi_vector):
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
    ) = simulate_SI(mode_trade, mode_learn, Nc, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, sigma_S, tax, beta, phi,
                    Npre, Ninit, T_hat, dZ_build, dZ, dZ_SI_build, dZ_SI, tau, cohort_size)
    # invest_tracker = pi > 0
    # Delta_matrix1[i] = Delta
    pi_matrix1[i] = pi
    f_matrix1[i] = f
    # theta_matrix1[i] = theta
    # theta_mat = np.transpose(np.tile(theta, (Nc, 1)))
    # mu_st_rt_matrix1[i] = theta_mat + Delta
    # popu_parti_matrix1[i] = popu_parti
    # market_view_matrix1[i] = np.average(Delta, axis=1, weights=f)
    # # market_view_parti_matrix[j,i] = Delta_bar_parti  # consumption weighted estimation error of the participants
    # # market_view_nonparti_matrix[j,i] = np.average(Delta, axis=1, weights=f*(1-invest_tracker))
    #
    # survey_view_matrix1[i] = np.average(Delta, axis=1, weights=cohort_size)



dZ = np.cumsum(dZ_SI_matrix[0])
dZ_SI = np.cumsum(dZ_matrix[0])

# Delta and pi
colors = ['darkmagenta', 'midnightblue', 'green', 'saddlebrown', 'darkgreen', 'firebrick', 'purple', 'blue',
          'olivedrab', 'darkviolet']



for k in range(n_phi):
    y1 = Delta_matrix1[k]
    y6 = pi_matrix1[k]
    phi = phi_vector[k]

    # Delta:
    nn = 3
    length = len(t)
    Delta_time_series = np.zeros((nn, length))
    parti_time_series = np.zeros((nn, length))
    for i in range(nn):
        start = int((i + 1) * 100 * (1 / dt))
        for j in range(length):
            if j < start:
                Delta_time_series[i, j] = np.nan
                parti_time_series[i, j] = np.nan
            else:
                cohort_rank = length - (j - start) - 1
                Delta_time_series[i, j] = y1[j, cohort_rank]
                parti_time_series[i, j] = (y6[j, cohort_rank] > 0)
    switch = abs(parti_time_series[:, 1:] - parti_time_series[:,:-1])
    col = np.reshape(switch[:, -1], (3, -1))
    switch = np.append(switch, col, axis=1)

    y11 = Delta_time_series[0]
    y11_N = np.ma.masked_where(parti_time_series[0] == 1, y11)
    y11_P = np.ma.masked_where(parti_time_series[0] == 0, y11)
    y11_switch = np.ma.masked_where(switch[0] == 0, y11)
    y12 = Delta_time_series[1]
    y12_N = np.ma.masked_where(parti_time_series[1] == 1, y12)
    y12_P = np.ma.masked_where(parti_time_series[1] == 0, y12)
    y12_switch = np.ma.masked_where(switch[1] == 0, y12)
    y13 = Delta_time_series[2]
    y13_N = np.ma.masked_where(parti_time_series[2] == 1, y13)
    y13_P = np.ma.masked_where(parti_time_series[2] == 0, y13)
    y13_switch = np.ma.masked_where(switch[2] == 0, y13)

    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.set_xlabel('Time in simulation, one random path')
    ax1.set_ylabel('Zt', color=color5)
    ax1.plot(t, dZ, color=color5, linewidth=0.5, label = 'Z^Y_t')
    ax1.plot(t, dZ_SI, color=color6, linewidth=0.5, label = 'Z^SI_t')
    ax1.tick_params(axis='y', labelcolor=color5)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Estimation error', color=color2)
    ax2.set_ylim([-0.5, 0.5])
    ax2.plot(t, y11_P, color=color2, linewidth=0.4, label = 'cohort 1')
    ax2.plot(t, y11_N, color=color2, linewidth=0.4, linestyle = 'dotted')
    ax2.scatter(t, y11_switch, color='red', s=10, marker = 'o', label = 'state switch')
    ax2.plot(t, y12_P, color=color3, linewidth=0.4, label = 'cohort 2')
    ax2.plot(t, y12_N, color=color3, linewidth=0.4, linestyle = 'dotted')
    ax2.scatter(t, y12_switch, color='red', s=10, marker = 'o')
    ax2.plot(t, y13_P, color=color4, linewidth=0.4, label = 'cohort 3')
    ax2.plot(t, y13_N, color=color4, linewidth=0.4, linestyle = 'dotted')
    ax2.scatter(t, y13_switch, color='red', s=10, marker = 'o')
    ax2.tick_params(axis='y', labelcolor=color2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()
    plt.savefig('Zt and Delta time series' + str(round(phi, 2)) + '.png', dpi=500)
    plt.show()
    plt.close()

# pi: illustrate who are investing and when do they quit
nn = 10
length = len(t)
pi_time_series = np.zeros((nn, length))
starts = np.zeros(nn)
for k in range(n_phi):
    phi = phi_vector[k]
    y6 = pi_matrix1[k]
    for i in range(nn):
        start = int((i + 5) * 25 * (1 / dt))
        starts[i] = start * dt
        for j in range(length):
            if j < start:
                pi_time_series[i, j] = np.nan
            else:
                cohort_rank = length - (j - start) - 1
                a = y6[j, cohort_rank]
                pi_time_series[i, j] = a
                # if a == 0:
                #     pi_time_series[i, j + 1: j + 8] = 0
                #     pi_time_series[i, j + 8:] = np.nan
                #     break

    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.set_xlabel('Time in simulation, one random path')
    ax1.set_ylabel('Zt', color=color5)
    ax1.plot(t, dZ, color=color5, linewidth=0.5, label = 'Z^Y_t')
    ax1.plot(t, dZ_SI, color=color6, linewidth=0.5, label = 'Z^SI_t')
    ax1.tick_params(axis='y', labelcolor=color5)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Investment in stock market', color=color2)
    ax2.set_ylim([0.01, 25])
    for i in range(nn):
        y6i = pi_time_series[i]
        plt.vlines(starts[i], ymax=25, ymin=0, color='grey', linestyle='--', linewidth=0.4)
        ax2.plot(t, y6i, color=colors[i], linewidth=0.4)
    ax2.tick_params(axis='y', labelcolor=color2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('Zt and pi time series' + str(round(phi, 2)) + '.png', dpi=500)
    plt.show()
    # plt.close()

# popu, theta, market view and survey view:
y_list = [theta_matrix1, popu_parti_matrix1, survey_view_matrix1, market_view_matrix1]
y_title_list = ['Market price of risk', 'Participation rate', 'Market view', 'Survey view']
for j, y_mat in enumerate(y_list):
    y_title = y_title_list[j]
    y = y_mat

    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.set_xlabel('Time in simulation, one random path')
    ax1.set_ylabel('Zt', color=color5)
    ax1.plot(t, dZ, color=color5, linewidth=0.5, label = 'Z^Y_t')
    ax1.plot(t, dZ_SI, color=color6, linewidth=0.5, label = 'Z^SI_t')
    ax1.tick_params(axis='y', labelcolor=color5)
    ax2 = ax1.twinx()
    ax2.set_ylabel(y_title, color=color2)
    scale = 1 if j <= 1 else 0.25
    ax2.set_ylim([-scale, scale])

    for i in range(n_phi):
        label_i = 'phi = ' + str(round(phi_vector[i],2))
        yy = y[i]
        ax2.plot(t, yy, color=colors[i], linewidth=0.4, label=label_i)
    ax2.tick_params(axis='y', labelcolor=color2)
    plt.legend()
    # fig.suptitle('Zt and Market Price of Risk')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('Zt and ' + y_title + ' different phi' + '.png', dpi=500)
    plt.show()
    plt.close()


y_log_Yt = np.cumsum((mu_Y - 0.5 * sigma_Y ** 2) * dt + sigma_Y * dZ_SI_matrix[0])
y_log_Yt_mat = np.transpose(np.tile(y_log_Yt, (Nc, 1)))
cohort_size_mat = np.transpose(np.tile(cohort_size, (Nc, 1)))
for k in range(n_phi):
    y6 = pi_matrix1[k]
    phi = phi_vector[k]
    y_fst = f_matrix1[k]
    # y_fst = f_matrix[0, k]
    y1 = y_log_Yt_mat + np.log(y_fst / cohort_size_mat)
    # y1 = np.log(y_fst / cohort_size_mat)

    # Delta:
    nn = 3
    length = len(t)
    Delta_time_series = np.zeros((nn, length))
    parti_time_series = np.zeros((nn, length))
    for i in range(nn):
        start = int((i + 1) * 100 * (1 / dt))
        for j in range(length):
            if j < start:
                Delta_time_series[i, j] = np.nan
                parti_time_series[i, j] = np.nan
            else:
                cohort_rank = length - (j - start) - 1
                Delta_time_series[i, j] = y1[j, cohort_rank]
                parti_time_series[i, j] = (y6[j, cohort_rank] > 0)
    switch = abs(parti_time_series[:, 1:] - parti_time_series[:,:-1])
    col = np.reshape(switch[:, -1], (3, -1))
    switch = np.append(switch, col, axis=1)

    y11 = Delta_time_series[0]
    y11_N = np.ma.masked_where(parti_time_series[0] == 1, y11)
    y11_P = np.ma.masked_where(parti_time_series[0] == 0, y11)
    y11_switch = np.ma.masked_where(switch[0] == 0, y11)
    y12 = Delta_time_series[1]
    y12_N = np.ma.masked_where(parti_time_series[1] == 1, y12)
    y12_P = np.ma.masked_where(parti_time_series[1] == 0, y12)
    y12_switch = np.ma.masked_where(switch[1] == 0, y12)
    y13 = Delta_time_series[2]
    y13_N = np.ma.masked_where(parti_time_series[2] == 1, y13)
    y13_P = np.ma.masked_where(parti_time_series[2] == 0, y13)
    y13_switch = np.ma.masked_where(switch[2] == 0, y13)

    fig, ax1 = plt.subplots(figsize=(15, 5))
    ax1.set_xlabel('Time in simulation, one random path')
    ax1.set_ylabel('Zt', color=color5)
    ax1.plot(t, dZ, color=color5, linewidth=0.5, label = 'Z^Y_t')
    ax1.plot(t, dZ_SI, color=color6, linewidth=0.5, label = 'Z^SI_t')
    ax1.tick_params(axis='y', labelcolor=color5)
    ax2 = ax1.twinx()
    ax2.set_ylabel('Log consumption', color=color2)
    ax2.set_ylim([0, 20])
    ax2.plot(t, y11_P, color=color2, linewidth=0.4, label = 'cohort 1')
    ax2.plot(t, y11_N, color=color2, linewidth=0.4, linestyle = 'dotted')
    ax2.scatter(t, y11_switch, color='red', s=10, marker = 'o', label = 'state switch')
    ax2.plot(t, y12_P, color=color3, linewidth=0.4, label = 'cohort 2')
    ax2.plot(t, y12_N, color=color3, linewidth=0.4, linestyle = 'dotted')
    ax2.scatter(t, y12_switch, color='red', s=10, marker = 'o')
    ax2.plot(t, y13_P, color=color4, linewidth=0.4, label = 'cohort 3')
    ax2.plot(t, y13_N, color=color4, linewidth=0.4, linestyle = 'dotted')
    ax2.scatter(t, y13_switch, color='red', s=10, marker = 'o')
    ax2.tick_params(axis='y', labelcolor=color2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.legend()
    plt.savefig('Zt and Consumption time series' + str(round(phi, 2)) + '.png', dpi=500)
    plt.show()
    # plt.close()
