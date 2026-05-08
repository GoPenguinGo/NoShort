import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_mix_types
from src.param import (rho_i, nu, mu_Y, sigma_Y, tax, phi, \
    dt, T_hat, Npre, Vhat, Ninit, Nt, Nc, tau, cohort_size, \
    cutoffs_age, colors, Mpath, \
    dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    cohort_labels, colors_short, PN_labels, age_labels, \
    Ntype, alpha_i, beta_i, beta0, rho_cohort_type, cohort_type_size,
                       popu_age_groups, entry_bound, exit_bound)
from src.param_mix import Nconstraint, rho_i_mix
import statsmodels.api as sm
import pandas as pd
import tabulate as tab
from scipy.interpolate import make_interp_spline

# Fig1: Shocks and beliefs
# Fig2: Distribution of beliefs
folder_address = r'C:\Users\zeshu\BI Norwegian Business School Dropbox\Zeshu XU\to sync\between_computers\entry_exit/empirical/'
# folder_address = r'C:/Users/A2010290/OneDrive - BI Norwegian Business School (BIEDU)/Documents/GitHub computer 2/NoShort/empirical/'
data_shocks = pd.read_excel(
    folder_address + r'realized_shocks_US.xlsx',
    # sheet_name='Sheet1',
    index_col=0
)

plt.rcParams["font.family"] = 'serif'

# (complete, excluded, disappointment, reentry)
density = (0.3, 0.5, 0.2)
phi_set = [
    # 0.0,
    0.5,
    1.0
]
n_scenarios = 2
n_phi = len(phi_set)
alpha_constraint = np.ones(
    (1, Nconstraint)) * density
alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
cohort_type_size_mix = cohort_size * alpha_i_mix
beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio
rho_cohort_type_mix = alpha_i_mix * beta_i_mix * np.exp(
    -(rho_i_mix + nu) * tau)  # shape(2, 6000)

dZ_build = dZ_build_matrix[0]
dZ = dZ_matrix[0]
dZ_actual = data_shocks.to_numpy()[:, 0]
Nt_data = dZ_actual.size
dZ[-dZ_actual.size:] = dZ_actual
theta_compare = np.empty((n_scenarios, n_phi, Nt_data), dtype=np.float32)
Delta_bar_compare = np.zeros((n_scenarios, n_phi, Nt_data), dtype=np.float32)
Delta_compare = np.empty((n_scenarios, n_phi, Nt_data, Nconstraint, Nc), dtype=np.float16)
pi_compare = np.empty((n_scenarios, n_phi, Nt_data, Nconstraint, Nc), dtype=np.float16)
invest_tracker_compare = np.zeros((n_scenarios, n_phi, Nt_data, Nconstraint, Nc), dtype=int)
parti_compare = np.zeros((n_scenarios, n_phi, Nt_data), dtype=np.float16)
for i in range(n_scenarios):
    if i == 0:
        entry_bound_use = np.copy(entry_bound)
        exit_bound_use = np.copy(exit_bound)
    else:
        entry_bound_use = 0.0
        exit_bound_use = 0.0
    for j, phi in enumerate(phi_set):
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
            parti_age_group,
            # Delta_popu,
            # parti_wealth_group,
            entry_mat,
            exit_mat
        ) = simulate_mix_types(Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax,
                               beta0,
                               phi, Npre, Ninit, T_hat,
                               entry_bound_use,
                               exit_bound_use,
                               dZ_build, dZ,
                               cutoffs_age, Ntype,
                               Nconstraint, rho_i_mix, alpha_i_mix, beta_i_mix,
                               rho_cohort_type_mix,
                               cohort_type_size_mix,
                               need_f='True',
                               need_Delta='True',
                               need_pi='True',
                               mode_learn='invest',
                               )
        Delta_compare[i, j] = Delta[-Nt_data:]
        pi_compare[i, j] = pi[-Nt_data:]
        invest_tracker_compare[i, j] = (pi[-Nt_data:] != 0)
        theta_compare[i, j] = theta[-Nt_data:]
        Delta_bar_compare[i, j] = Delta_bar_parti[-Nt_data:]
        parti_compare[i, j] = parti[-Nt_data:]


#####################################
#######  individual cohorts  #######
#####################################
nn = 3  # number of cohorts illustrated
starts = Nt_data - 43 * 12 + 1 - np.arange(nn) * 240
# starts = np.arange(nn) * 240 + 24 * 12
Delta_time_series = np.zeros((n_scenarios, n_phi, nn, Nconstraint, Nt_data), dtype=np.float32)
pi_time_series = np.zeros((n_scenarios, n_phi, nn, Nconstraint, Nt_data), dtype=np.float32)
entry_time_series = np.zeros((n_scenarios, n_phi, nn, Nconstraint, Nt_data), dtype=np.float32)
exit_time_series = np.zeros((n_scenarios, n_phi, nn, Nconstraint, Nt_data), dtype=np.float32)
parti_time_series = np.zeros((n_scenarios, n_phi, nn, Nconstraint, Nt_data), dtype=np.float32)
for i in range(n_scenarios):
    for j in range(n_phi):
        for k in range(Nconstraint):
            pi = pi_compare[i, j, :, k]
            Delta = Delta_compare[i, j, :, k]
            for m in range(nn):
                start = starts[m]
                for n in range(Nt_data):
                    if n < start:
                        pi_time_series[i, j, m, k, n] = np.nan
                        Delta_time_series[i, j, m, k, n] = np.nan
                    else:
                        cohort_rank = Nt - (n - start) - 1
                        Delta_time_series[i, j, m, k, n] = Delta[n, cohort_rank]
                        pi_time_series[i, j, m, k, n] = pi[n, cohort_rank]

                        if k == 0:  # the unconstrained
                            parti_time_series[i, j, m, k, n] = 1
                            entry_time_series[i, j, m, k, n] = 0
                            exit_time_series[i, j, m, k, n] = 0
                        elif k == 1:  # the excluded
                            parti_time_series[i, j, m, k, n] = 0
                            entry_time_series[i, j, m, k, n] = 0
                            exit_time_series[i, j, m, k, n] = 0
                        else:
                            # Using updated indexing for logic checks
                            parti_time_series[i, j, m, k, n] = 1 if pi_time_series[i, j, m, k, n] != 0 else 0

                            switch_PN = 1 if (pi_time_series[i, j, m, k, n - 1] != 0) and (
                                    pi_time_series[i, j, m, k, n] == 0) else 0

                            switch_NP = 1 if (pi_time_series[i, j, m, k, n] != 0) and (
                                    pi_time_series[i, j, m, k, n - 1] == 0) else 0

                            entry_time_series[i, j, m, k, n] = 1 if switch_NP == 1 else 0
                            exit_time_series[i, j, m, k, n] = 1 if switch_PN == 1 else 0
                            exit_time_series[i, j, m, k, start] = 0

                            if switch_NP == 1:
                                parti_time_series[i, j, m, k, n] = 0.5
                            if switch_PN == 1:
                                parti_time_series[i, j, m, k, n - 1] = 0.5



#####################################
#######  belief distribution  #######
#####################################
# phi = 0.5, reentry scenario
y_overall = np.empty((n_scenarios, n_phi, Nt_data, 6))  # overall
y_P = np.empty((n_scenarios, n_phi, Nt_data, 6))  # participants / long
y_N = np.empty((n_scenarios, n_phi, Nt_data, 6))  # non-participants / short
cohort_size_flat = np.sum(cohort_type_size_mix, axis=0)[-1]
age_cutoffs = [int(Nt-1), int(Nt-1-12*15), int(Nt-1-12*40), int(Nt-1-12*55), 0]  # Michigan
n_age_cutoffs = len(age_cutoffs) - 1
y_min = np.empty((n_scenarios, n_phi, Nt_data, n_age_cutoffs))
y_max = np.empty((n_scenarios, n_phi, Nt_data, n_age_cutoffs))
belief_cutoff = np.empty((n_scenarios, n_phi, Nt_data))
for i in range(n_scenarios):
    for j in range(n_phi):
        y_cases = [y_overall[i, j], y_P[i, j], y_N[i, j]]

        Delta_focus = Delta_compare[i, j, :, -1]
        invest_focus = invest_tracker_compare[i, j, :, -1]
        belief_cutoff[i, j] = -theta_compare[i, j]

        for n in range(n_age_cutoffs):
            Delta_age_group = Delta_focus[:, age_cutoffs[n + 1]:age_cutoffs[n]]
            y_min[i, j, :, n] = np.amin(Delta_age_group, axis=1)
            y_max[i, j, :, n] = np.amax(Delta_age_group, axis=1)
        for m in range(Nt_data):
            Delta = Delta_focus[m]  # ((Nt, Nc))
            parti_cohorts = invest_focus[m]
            if np.sum(parti_cohorts) == Nc:
                cohort_sizes = [cohort_size_flat]
                Deltas = [Delta]
                n_var = 1
            else:
                Delta1 = parti_cohorts * Delta
                Delta2 = (1 - parti_cohorts) * Delta
                cohort_size1 = parti_cohorts * cohort_size_flat
                cohort_size2 = (1 - parti_cohorts) * cohort_size_flat
                cohort_sizes = [cohort_size_flat, cohort_size1, cohort_size2]
                Deltas = [Delta, Delta1, Delta2]
                n_var = 3
            for n in range(n_var):
                Del = Deltas[n]
                cohort_siz = cohort_sizes[n]
                Delta_rank = Del.argsort()
                Delta_sorted = Del[Delta_rank[::-1]]
                cohort_size_sorted = cohort_siz[Delta_rank[::-1]]
                popu_cumsum = np.cumsum(cohort_size_sorted)
                total_popu = popu_cumsum[-1]
                Delta_cutoff = np.zeros(6)
                cutoff = np.searchsorted(popu_cumsum, [0.25 * total_popu, 0.5 * total_popu, 0.75 * total_popu])
                Delta_cutoff[1:4] = Delta_sorted[cutoff]  # highest to lowest
                Delta_cutoff[0] = np.max(Del[np.nonzero(Del)])
                Delta_cutoff[4] = np.min(Del[np.nonzero(Del)])
                Delta_cutoff[5] = np.average(Delta_sorted, weights=cohort_size_sorted)
                y_cases[n][m] = Delta_cutoff


######################################
###########   Figure 1   #############
######################################
print('Figure 1')
N_years = 100
x = 2023 - N_years + np.arange(int(N_years / dt)) * dt
colors_short = ['midnightblue', 'darkgreen', 'darkviolet', 'red']
fig, axes = plt.subplots(nrows=3, ncols=1, sharex='all', figsize=(10, 8))
for j, ax in enumerate(axes):
    if j == 0:
        ax.set_title(r'(a) Cohort estimation error')
        # ax.set_title(r'(a) Shocks to fundamental $z^Y$, and cohort estimation error')
        Delta_focus = Delta_time_series[0, 0, :, -1]
        parti_focus = parti_time_series[0, 0, :, -1]
        entry_focus = entry_time_series[0, 0, :, -1]
        exit_focus = exit_time_series[0, 0, :, -1]
        y_min_raw = np.nanmin(Delta_focus)  # only the re-entry type
        y_max_raw = np.nanmax(Delta_focus)
        ax.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
        ax.set_ylim([
            - (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2,
            (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2
        ])
        for m in range(nn):
            # switch[m, starts[m]] = 1
            y_cohort = Delta_focus[m]
            y_cohort_N = np.ma.masked_where(parti_focus[m] == 1,
                                            y_cohort)
            y_cohort_P = np.ma.masked_where(parti_focus[m] == 0,
                                            y_cohort)
            y_cohort_entry = np.ma.masked_where(
                entry_focus[m] == 0,
                y_cohort)
            y_cohort_exit = np.ma.masked_where(
                exit_focus[m] == 0,
                y_cohort)
            ax.vlines(
                x[starts[m] - (Nt_data - N_years * 12)],
                ymax=(y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2,
                ymin=- (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2,
                color='grey', linestyle='--', linewidth=0.6
            )
            ax.plot(x, y_cohort_P[-int(N_years / dt):], color=colors_short[m], linewidth=1, label=cohort_labels[m])
            ax.plot(x, y_cohort_N[-int(N_years / dt):], color=colors_short[m], linewidth=1, linestyle='dotted',
                    )
            if m == 2:
                ax.scatter(x, y_cohort_entry[-int(N_years / dt):], color='red', s=25, marker='o', label='Entry')
                ax.scatter(x, y_cohort_exit[-int(N_years / dt):], color='orange', s=25, marker='o', label='Exit')
            else:
                ax.scatter(x, y_cohort_entry[-int(N_years / dt):], color='red', s=25, marker='o')
                ax.scatter(x, y_cohort_exit[-int(N_years / dt):], color='orange', s=25, marker='o')

        ax.tick_params(axis='y', labelcolor='black')
        ax.legend(loc='lower left')

    elif j == 1:
        ax.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
        ax.set_title('(b) Average estimation error, participants vs. non-participants')
        ax.plot(
            x,
            y_P[0, 0, -int(N_years / dt):, -1],
            color='navy', linewidth=1, label=r'Participants'
        )
        ax.plot(
            x,
            y_N[0, 0, -int(N_years / dt):, -1],
            color='maroon', linewidth=1, label=r'Nonparticipants'
        )
        ax.plot(
            x,
            belief_cutoff[0, 0, -int(N_years / dt):] + entry_bound,
            color='black', linewidth=1, label=r'Cutoff $\Delta$ for entry', linestyle='dotted'
        )
        ax.legend(loc='lower left')

    else:
        ax.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
        ax.set_ylim(-1.4, 1.3)
        ax.set_title('(c) Distribution of estimation error, experience groups')
        ax.plot(x,
                belief_cutoff[0, 0, -int(N_years / dt):],
                color='black', linewidth=1,
                # label=r'Cutoff $\Delta_{s,t}$'
                )
        for k in range(n_age_cutoffs):
            color_age_group = colors_short[k]
            ax.fill_between(
                x,
                y_min[0, 0, -int(N_years / dt):, k],
                y_max[0, 0, -int(N_years / dt):, k],
                color=color_age_group, linewidth=0., alpha=0.4,
                label=age_labels[k]
            )
            ax.legend(loc='lower left')
        ax.tick_params(axis='y', labelcolor='black')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(
    'f1_merged.pdf',
    dpi=200)
plt.show()
plt.close()



######################################
######## Figure 1.Appendix   #########
######################################
print('Figure 1')
N_years = 100
x = 2023 - N_years + np.arange(int(N_years / dt)) * dt
colors_short = ['midnightblue', 'darkgreen', 'darkviolet', 'red']
fig, axes = plt.subplots(nrows=2, ncols=1, sharex='all', figsize=(10, 6))
for j, ax in enumerate(axes):
    if j == 0:  # entry_bound = exit_bound = 0
        ax.set_title(r'(a) Cohort estimation error, $\vartheta^h = \vartheta^l = 0$')
        # ax.set_title(r'(a) Shocks to fundamental $z^Y$, and cohort estimation error')
        Delta_focus = Delta_time_series[1, 0, :, -1]
        parti_focus = parti_time_series[1, 0, :, -1]
        entry_focus = entry_time_series[1, 0, :, -1]
        exit_focus = exit_time_series[1, 0, :, -1]
        y_min_raw = np.nanmin(Delta_focus)  # only the re-entry type
        y_max_raw = np.nanmax(Delta_focus)
        ax.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
        ax.set_ylim([
            - (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2,
            (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2
        ])
        for m in range(nn):
            # switch[m, starts[m]] = 1
            y_cohort = Delta_focus[m]
            y_cohort_N = np.ma.masked_where(parti_focus[m] == 1,
                                            y_cohort)
            y_cohort_P = np.ma.masked_where(parti_focus[m] == 0,
                                            y_cohort)
            y_cohort_entry = np.ma.masked_where(
                entry_focus[m] == 0,
                y_cohort)
            y_cohort_exit = np.ma.masked_where(
                exit_focus[m] == 0,
                y_cohort)
            ax.vlines(
                x[starts[m] - (Nt_data - N_years * 12)],
                ymax=(y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2,
                ymin=- (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2,
                color='grey', linestyle='--', linewidth=0.6
            )
            ax.plot(x, y_cohort_P[-int(N_years / dt):], color=colors_short[m], linewidth=1, label=cohort_labels[m])
            ax.plot(x, y_cohort_N[-int(N_years / dt):], color=colors_short[m], linewidth=1, linestyle='dotted',
                    )
            if m == 2:
                ax.scatter(x, y_cohort_entry[-int(N_years / dt):], color='red', s=25, marker='o', label='Entry')
                ax.scatter(x, y_cohort_exit[-int(N_years / dt):], color='orange', s=25, marker='o', label='Exit')
            else:
                ax.scatter(x, y_cohort_entry[-int(N_years / dt):], color='red', s=25, marker='o')
                ax.scatter(x, y_cohort_exit[-int(N_years / dt):], color='orange', s=25, marker='o')

        ax.tick_params(axis='y', labelcolor='black')
        ax.legend(loc='lower left')

    else:  # varphi = 1 (no reduced attention from nonparticipants)
        ax.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
        ax.set_title(r'(b) Average estimation error, participants vs. non-participants, $\varphi = 1$')
        ax.plot(
            x,
            y_P[0, 1, -int(N_years / dt):, -1],
            color='navy', linewidth=1, label=r'Participants'
        )
        ax.plot(
            x,
            y_N[0, 1, -int(N_years / dt):, -1],
            color='maroon', linewidth=1, label=r'Nonparticipants'
        )
        ax.plot(
            x,
            belief_cutoff[0, 1, -int(N_years / dt):],
            color='black', linewidth=1, label=r'Cutoff $\Delta$ for entry', linestyle='dotted'
        )
        ax.legend(loc='lower left')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(
    'OA_f1_merged.pdf',
    dpi=200)
plt.show()
plt.close()


######################################
###########   Figure 4   #############
######################################
# # counts of entry
folder_address = r'E:\Users\A2010290\Documents\GitHub\NoShort/simu_results/'
Mpath = 2000
n_files = int(Mpath / 25)

# how long before exiting upon entry &
# how long before entering upon exit
# Analysis of the bell length: Distribution of participation bells, ignoring 0
sample_shocks = np.arange(2400 + 240, Nt, int(20/dt))
spell_mat = np.zeros((int(Mpath / 10), 2, len(sample_shocks), 2, 5748), dtype=int)
stock_returns_mat = np.zeros((int(Mpath / 10), len(sample_shocks)))
gap = 12
for i in range(Mpath):
    if np.mod(i, 10) == 0:
        j = int(i / 10)
        spell_mat[j] = np.load(folder_address + str(i) + "reentry_time.npy")
        cumu_returns = np.zeros(Nt)
        cumu_returns[gap:] = (np.cumsum(dZ_matrix[i])[gap:] - np.cumsum(dZ_matrix[i])[:-gap]) / (gap / 12)
        stock_returns_mat[j] = cumu_returns[sample_shocks]

# ax.set_title('Simulation')
for i in range(2):
    fig, ax = plt.subplots(nrows=1, ncols=1, sharey='all', sharex='all', figsize=(5, 5))
    cutoffs_return = np.percentile(stock_returns_mat, [10])
    data_mat = spell_mat[:, i]
    label_i = 'reentry' if i == 0 else 'mix'
    for j in range(2):
        if j == 0:
            counts_mat = np.zeros((5748, 21))
            for ii in range(5748):
                unique, counts = np.unique(data_mat[:, :, :, ii], return_counts=True)
                for jj, uni_jj in enumerate(unique):
                    counts_mat[ii, uni_jj] = counts[jj]
        else:
            data_where = np.reshape(stock_returns_mat <= cutoffs_return, (int(Mpath / 10), -1, 1))
            counts_mat = np.zeros((5748, 21))
            for ii in range(5748):
                unique, counts = np.unique(data_mat[:, :, :, ii] * data_where, return_counts=True)
                for jj, uni_jj in enumerate(unique):
                    counts_mat[ii, uni_jj] = counts[jj]
        popu_average_counts = np.average(counts_mat, axis=0, weights=cohort_size[0, -5748:])
        counts_percentage = popu_average_counts[1:] / np.sum(popu_average_counts[1:])
        counts_percentage_cum = np.cumsum(counts_percentage)
        x = np.arange(1, 20)
        data_inter = np.arange(0, len(counts_percentage) - 1, 2)
        X_Y_Spline = make_interp_spline(x[data_inter], counts_percentage_cum[data_inter], k=3)
        X_ = np.linspace(1, 19, 100)
        Y_ = X_Y_Spline(X_)
        color_j = 'red' if j == 1 else 'navy'
        label_j = r'$dR_{t-1,t}$ bottom decile' if j == 1 else 'Unconditional'
        line_style_i = 'solid'
        ax.plot(X_, Y_, linewidth=2,
                linestyle=line_style_i,
                label=label_j, color=color_j)
        ax.legend(loc='lower right')
    ax.set_ylim(0.1, 1.0)
    ax.set_xlim(1, 10)
    ax.set_xlabel('Years since exiting')
    ax.set_ylabel('Fraction of individuals re-entering')
    plt.savefig(f'f5_2{label_i}.pdf', dpi=200)
    plt.show()
    plt.close()


######################################
##############  Tables  ##############
######################################
## Table 10 in the Online Appendix: asset pricing moments
## Panel A
Nsce = 3
Ncolumn = int(Nsce * 2)
table1_var = ['theta', 'r', 'mu_S', 'sigma_S']
Nrow = len(table1_var)
table1_mat = np.zeros((Nrow, n_files, Nsce, 2))
for i, var in enumerate(table1_var):
    for j in range(n_files):
        table1_mat[i, j] = np.average(np.load(folder_address + str(j) + str(0.5) + "simulation_new.npz")[var], axis=0)
table1_mat_ave = np.average(table1_mat, axis=1)
# Panel 1.1: mean vola of asset pricing values
table_output = np.zeros((Nrow, Ncolumn))
header = np.tile(['Mean', 'Std'], Nsce)
for j in range(Nrow):
    for i in range(Nsce):
        row_index = j
        col_index = i * 2
        table_output[row_index, col_index:col_index + 2] = table1_mat_ave[j, i]

show_index = [
    # r'$dR_t$',
              r'$\theta_t$',
              r'$\r_t$',
              r'$\mu^S_t$',
              r'$\sigma^S_t$',
              ]
print(tab.tabulate(table_output, headers=header, showindex=show_index, floatfmt=".4f",
                   tablefmt='latex_raw'))

## Table 10 in the Online Appendix: asset pricing moments
## Panel B
Nrow = 3
table12_mat = np.zeros((n_files, Nsce, Nrow))
for j in range(n_files):
    table12_mat[j] = np.average(np.load(folder_address + str(j) + str(0.5) + "simulation_new.npz")['cov_mat'], axis=0)[:, :3]
table12_mat_ave = np.average(table12_mat, axis=0)
table_output = np.zeros((Nrow, Ncolumn))
for j in range(Nrow):
    for i in range(Nsce):
        row_index = j
        col_index = i * 2
        table_output[row_index, col_index] = table12_mat_ave[i, j]
show_index = [r'$\text{Corr}(dz_t^Y, \theta_t)$',
              r'$\text{Corr}(dz_t^Y, \sigma_t^S)$',
              r'$\text{Corr}(dz_t^{SI}, \theta_t)$',
              ]
print(tab.tabulate(table_output, showindex=show_index, floatfmt=".4f", tablefmt='latex_raw'))

# Table 1: Participation, entry and exit
# Panel A: mean
table21_var = ['parti', 'entry', 'exit']
Nrow = len(table21_var)
Ncolumn = 2
table21_mat = np.zeros((Nrow, n_files, Ncolumn))
for i, var in enumerate(table21_var):
    for j in range(n_files):
        table21_mat[i, j] = np.average(np.load(folder_address + str(j) + str(0.5) + "simulation_new.npz")[var], axis=0)
table21_mat_ave = np.average(table21_mat, axis=1)
header = ['Re-entry', 'Mix-4']
print(tab.tabulate(table21_mat_ave, headers=header,
                   floatfmt=".4f", tablefmt='latex_raw'))

# Table 1: Participation, entry and exit
# Panel B: correlations between participation rate and asset pricing moments
# todo: beliefs as well?
Nrow = 2
table22_mat = np.zeros((n_files, Nsce, Nrow))
for j in range(n_files):
    table22_mat[j] = np.average(np.load(folder_address + str(j) + str(0.5) + "simulation_new.npz")['cov_mat'], axis=0)[:, 3:]
table22_mat_ave = np.average(table22_mat, axis=0)
table_output = np.zeros((Nrow, Ncolumn))
for j in range(Nrow):
    for i in range(Nsce - 1):
        table_output[j, i] = table22_mat_ave[i+1, j]
show_index = [
    r'$\text{Corr}(\Bar{\Phi}_t, P_t)$',
    r'$\text{Corr}(\tilde{\Phi}_t, P_t)$',
]
print(tab.tabulate(table_output, showindex=show_index, floatfmt=".4f", tablefmt='latex_raw'))

# Panel 3: participation rate, entry and exit on asset returns
table3_mat = np.zeros((n_files, Nsce - 1, 3, 3))
for j in range(n_files):
    table3_mat[j] = np.average(np.load(folder_address + str(j) + str(0.5) + "simulation_new.npz")['reg1'], axis=0)
table3_mat_ave = np.average(table3_mat, axis=0)
table_output = np.zeros((3, 6))
table_output[:, :3] = table3_mat_ave[0]
table_output[:, 3:] = table3_mat_ave[1]
header = np.tile(['P_t', r'$\text{Entry}_{t-n, t}$', r'$\text{Exit}_{t-n, t}$'], 2)
show_index = [
    r'$R_{t-1, t}$',
    r'$R_{t-2, t}$',
    r'$R_{t-3, t}$',
]
print(tab.tabulate(table_output, showindex=show_index, headers=header, floatfmt=".4f", tablefmt='latex_raw'))

table_output = np.zeros((3, 2))
table_output[:, 0] = table3_mat_ave[0, 1]
table_output[:, 1] = table3_mat_ave[1, 1]
show_index = [
    r'$\text{Corr}(R_{t-2, t}, P_t)$',
    r'$\text{Corr}(R_{t-2, t}^{high}, \text{Entry}_t)$',
    r'$\text{Corr}(R_{t-2, t}^{low}, \text{Exit}_t)$',
]
print(tab.tabulate(table_output, showindex=show_index, floatfmt=".4f", tablefmt='latex_raw'))

# Table 4: Predictive regressions: future asset returns on participation rate, entry and exit
table4_mat = np.zeros((n_files, Nsce - 1, 3, 2))
for j in range(n_files):
    table4_mat[j] = np.average(np.load(folder_address + str(j) + str(0.5) + "simulation_new.npz")['reg2'], axis=0)
table4_mat_ave = np.average(table4_mat, axis=0)
table4_mat_std = np.std(table4_mat, axis=0)
show_index = [
    r'$R^S_{t-n, t}/n$',
    r'$R^S_{t-n, t}/n - r_{t-n}$',
]
header = ['12m', '24m', '36m']
print(tab.tabulate(np.transpose(table4_mat_ave[1]), showindex=show_index, headers=header, floatfmt=".3f", tablefmt='latex_raw'))
print(tab.tabulate(np.transpose(table4_mat_std[1]), showindex=show_index, headers=header, floatfmt=".3f", tablefmt='latex_raw'))
# # Panel 4: participation rate in wealth groups
# file = file_list_mean_vola[10]
# var_average = np.average(results_mean_vola[file], axis=0)  # shape (n_scenarios, 2)
# N_row = np.shape(var_average)[1]
# table_output = np.zeros((N_row, int(Ncolumn/2)))
# for j in range(N_row):
#     for i in range(Nsce):
#         row_index = j
#         col_index = i
#         table_output[row_index, col_index] = var_average[i, j]
# show_index = [r'$ \text{wealth} \leq 1$',
#               r'$1 <\text{wealth} \leq 10$',
#               r'$10 < \text{wealth} \leq 100$',
#               r'$\text{wealth} > 100$',
#               ]
# print(tab.tabulate(table_output,
#                    showindex=show_index,
#                    floatfmt=".4f", tablefmt='latex_raw'))
#
# entry_rate = np.average(np.average(results_df1['entry rate'], axis=3), axis=0)
# exit_rate = np.average(np.average(results_df1['exit rate'], axis=3), axis=0)
# # Panel 7: participation rate covariances
# # todo: participation rate and vola: conditional on some state variables?
# file = file_list_mean_vola[13]
# var_average = np.average(results_mean_vola[file], axis=0)  # shape (n_scenarios, 2)
# N_row = np.shape(var_average)[1]
# table_output = np.zeros((N_row, int(Ncolumn/2)))
# for j in range(N_row):
#     for i in range(Nsce):
#         row_index = j
#         col_index = i
#         table_output[row_index, col_index] = var_average[i, j, 1]
# show_index = [r'$ \text{Cov}\left(R^S_{t,t+2}, P_t\right)$',
#               r'$ \text{Cov}\left(R_{t,t+2}, P_t\right)$',
#               r'$ \text{Cov}\left(R^S_{t,t+2}-R_{t,t+2}, P_t\right)$',
#               r'$ \text{Cov}\left(\text{average}\sigma_S, P_t\right)$',
#               r'$ \text{Cov}\left(R_{t,t+2}^2, P_t\right)$']
# print(tab.tabulate(table_output,
#                    showindex=show_index,
#                    floatfmt=".4f", tablefmt='latex_raw'))


######################################
##########  Regressions  #############
######################################
results_df2 = np.load("parti_rate_regressions2.npz")
age_alpha_mat = results_df2['age alpha']
y = np.average(age_alpha_mat[:, 0, :, 3], axis=0) / dt * 100
x = np.arange(0, len(y) * 5, 5)
constraint_labels = ['Designated P', 'Designated N', 'Disappointment', 'Reentry']
fig, axes = plt.subplots(nrows=1, ncols=2, sharey='all', sharex='all', figsize=(10, 4.5))
for i, ax in enumerate(axes):
    if i == 0:
        ax.plot(x, y, linewidth=2, label='Reentry', color=colors_short[3])
        ax.set_title(
            r'Average annual alpha given age, reentry')
    else:
        for j in range(4):
            y_j = np.average(age_alpha_mat[:, 1, :, j], axis=0) / dt * 100
            ax.plot(x, y_j, linewidth=2, label=constraint_labels[j], color=colors_short[j])
            ax.set_title(
                r'Average annual alpha given age, mix')
    # ax.set_title(r'Average annual alpha given age, $dR_{s,t} = \alpha_{t-s} + \beta_{t-s} dR_t^S + \epsilon_{t-s, t}$')
    ax.set_xlabel('Years since entering the economy')
    ax.set_ylabel(r'Average annual alpha, $\%$')
    ax.axhline(0, 0.05, 0.95, linestyle='dashed', color='gray')
    ax.legend()
plt.savefig('Reentry_age_alpha.pdf', dpi=200)
plt.show()
plt.close()

# regression 1: participation rate on returns and pd
reg_table1 = np.average(results_df2['return_parti_reg'], axis=0)
reg_table3 = np.average(results_df2['parti_return_reg'], axis=0)



# parti = np.copy(results_df1["participation rate"])
# annual_return = np.copy(results_df1["annual stock return"])
# l_one_year_return = annual_return[:, :, :, 0]
# l_two_year_return = annual_return[:, :, :, 1]
# l_thr_year_return = annual_return[:, :, :, 2]
# pd_ratio = np.copy(results_df1["pd ratio"])
# x_set = [
#     l_one_year_return,
#     l_two_year_return,
#     l_thr_year_return,
#     pd_ratio,
# ]
# Mpath = np.shape(parti)[0]
# regression_table1_b = np.zeros((2, len(x_set), Mpath))
# # regression_table1_se = np.zeros((2, len(x_set), Mpath, 2))
# for sce in range(2):
#     for i, x in enumerate(x_set):
#         for path in range(Mpath):
#             x_regress = sm.add_constant(x[path, sce+1, 1])
#             model = sm.OLS(parti[path, sce+1, 1], x_regress)
#             est = model.fit()
#             # b0 = est.params[0]
#             regression_table1_b[sce, i, path] = est.params[1]
#             # regression_table1_se[sce, i, path, :1] = est.bse[1:]
# table1_b = np.average(regression_table1_b, axis=2)
# # table1_se = np.average(regression_table1_se, axis=2)
#
# for k in range(2):
#     label_scenario = 'Reentry' if k == 0 else 'Mix - 4'
#     reg_data = np.zeros((len(x_set), len(x_set)))
#     for i in range(len(x_set)):
#         reg_data[i, i] = table1_b[k, i]
#     print(label_scenario)
#     print(tab.tabulate(reg_data, floatfmt=".3f", tablefmt='latex_raw'))
#
# # regression 2: participation rate predicts returns
# f_one_year_return = annual_return[:, :, :, 3]
# f_two_year_return = annual_return[:, :, :, 4]
# f_thr_year_return = annual_return[:, :, :, 5]
# y_set = [
#     f_one_year_return,
#     f_two_year_return,
#     f_thr_year_return
# ]
# x = np.copy(parti)
# regression_table2_b = np.zeros((2, len(y_set), Mpath))
# for sce in range(2):
#     for i, y in enumerate(y_set):
#         for path in range(Mpath):
#             x_regress = sm.add_constant(x[path, sce+1, 1])
#             model = sm.OLS(y[path, sce+1, 1], x_regress)
#             est = model.fit()
#             # b0 = est.params[0]
#             regression_table2_b[sce, i, path] = est.params[1]
# table2_b = np.average(regression_table2_b, axis=2)
# for k in range(2):
#     label_scenario = 'Reentry' if k == 0 else 'Mix - 4'
#     reg_data = np.zeros((1, len(y_set)))
#     for i in range(len(y_set)):
#         reg_data[0, i] = table2_b[k, i]
#     print(label_scenario)
#     print(tab.tabulate(reg_data, floatfmt=".3f", tablefmt='latex_raw'))
# folder_address = r'C:\Users\A2010290\OneDrive - BI Norwegian Business School (BIEDU)\Documents\GitHub computer 2\NoShort/reg_results/'
folder_address = r'E:\Users\A2010290\Documents\GitHub\NoShort/reg_results2/'
reg_results1 = np.empty((25, 1, 2, 2, 2, 3, 3))
# reg_results2 = np.empty((100, 1, 1, 3, 3))
for i in range(25):
    reg_results1[i] = np.load(folder_address + str(i) + "reg1.npy")
    # reg_results2[i] = np.load(folder_address + str(i) + "reg2.npy")
ave_reg1 = np.average(reg_results1, axis=0)

