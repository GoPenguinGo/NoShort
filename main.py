import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tabulate as tab
from scipy.interpolate import make_interp_spline
from src.param import Mpath, N_workers
from src.param import (nu, mu_Y, sigma_Y, tax, dt, T_hat, Npre, Vhat, Ninit, Nt, Nc, tau, cohort_size, \
                       cutoffs_age, dZ_matrix, dZ_build_matrix, cohort_labels, age_labels, \
                       Ntype, alpha_i, beta0, entry_bound, exit_bound)
from src.param_mix import Nconstraint, rho_i_mix, density
from src.simulation import simulate_mix_types

# Fig1: Shocks and beliefs
# Fig2: Distribution of beliefs
data_shocks = pd.read_excel(
    r'empirical/realized_shocks_US.xlsx',
    # sheet_name='Sheet1',
    index_col=0
)

plt.rcParams["font.family"] = 'serif'

n_files = int(Mpath / N_workers)

# (complete, excluded, reentry)
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
    'figures/f1_merged.pdf',
    dpi=200)
plt.show()
plt.close()




table_var = 'Delta_diff'
table_data = np.load(r'simu_results/' + str(0) + "simulation_new.npz")[table_var]
for j in range(1, n_files):
    table_data = np.append(table_data, np.load(r'simu_results/' + str(j) + "simulation_new.npz")[table_var], axis=0)

table_diff = pd.DataFrame(
    np.average(table_data, axis=0),
    index=[r'unconditional mean', r'annual return<10percentile'],
    columns=['Mean']
)
print(tab.tabulate(table_diff.transpose(), headers=[r'unconditional mean', r'annual return<10percentile'], floatfmt=".4f",
                   tablefmt='latex_raw'))

######################################
######## Figure 1.Appendix   #########
######################################
print('Figure 1 - OA')
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
        ax.set_title(r'(b) Average estimation error, participants vs. non-participants, $\varphi = 1$ vs. $\varphi = 0.5$')
        ax.plot(
            x,
            y_P[0, 1, -int(N_years / dt):, -1],
            color='navy', linewidth=1, label=r'Participants, $\varphi = 1$'
        )
        ax.plot(
            x,
            y_N[0, 1, -int(N_years / dt):, -1],
            color='maroon', linewidth=1, label=r'Nonparticipants, $\varphi = 1$'
        )
        ax.plot(
            x,
            y_P[0, 0, -int(N_years / dt):, -1],
            color='navy', linewidth=1, label=r'Participants, $\varphi = 0.5$', linestyle='dotted'
        )
        ax.plot(
            x,
            y_N[0, 0, -int(N_years / dt):, -1],
            color='maroon', linewidth=1, label=r'Nonparticipants, $\varphi = 0.5$', linestyle='dotted'
        )
        # ax.plot(
        #     x,
        #     belief_cutoff[0, 1, -int(N_years / dt):] + entry_bound,
        #     color='black', linewidth=1, label=r'Cutoff $\Delta$ for entry', linestyle='dotted'
        # )
        ax.legend(loc='lower left')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(
    'figures/OA_f1_merged.pdf',
    dpi=200)
plt.show()
plt.close()


######################################
###########   Figure 4   #############
######################################
# # counts of entry

# how long before exiting upon entry &
# how long before entering upon exit
# Analysis of the bell length: Distribution of participation bells, ignoring 0
sample_shocks = np.arange(2400 + 240, Nt, int(20/dt))
spell_mat = np.zeros((int(Mpath / 10), len(sample_shocks), 1, 5748), dtype=int)
stock_returns_mat = np.zeros((int(Mpath / 10), len(sample_shocks)))
# todo: save stock return mat from the runs directly
gap = 12
for i in range(Mpath):
    if np.mod(i, 10) == 0:
        j = int(i / 10)
        spell_mat[j] = np.load(r'simu_results/' + str(i) + 'reentry_time.npy')
        cumu_returns = np.zeros(Nt)
        cumu_returns[gap:] = (np.cumsum(dZ_matrix[i])[gap:] - np.cumsum(dZ_matrix[i])[:-gap]) / (gap / 12)
        stock_returns_mat[j] = cumu_returns[sample_shocks]

# ax.set_title('Simulation')
fig, ax = plt.subplots(nrows=1, ncols=1, sharey='all', sharex='all', figsize=(5, 5))
cutoffs_return = np.percentile(stock_returns_mat, [10])
for j in range(2):
    if j == 0:
        counts_mat = np.zeros((5748, 21))
        for ii in range(5748):
            unique, counts = np.unique(spell_mat[:, :, :, ii], return_counts=True)
            for jj, uni_jj in enumerate(unique):
                counts_mat[ii, uni_jj] = counts[jj]
    else:
        data_where = np.reshape(stock_returns_mat <= cutoffs_return, (int(Mpath / 10), -1, 1))
        counts_mat = np.zeros((5748, 21))
        for ii in range(5748):
            unique, counts = np.unique(spell_mat[:, :, :, ii] * data_where, return_counts=True)
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
plt.savefig(f'figures/f5_2.pdf', dpi=200)
plt.show()
plt.close()



######################################
##############  Tables  ##############
######################################
## Table 10 in the Online Appendix: asset pricing moments
## Panel A
table_var = 'table_mean_vola'
table1_data = np.load(r'simu_results/' + str(0) + "simulation_new.npz")[table_var]
for j in range(1, n_files):
    table1_data = np.append(table1_data, np.load(r'simu_results/' + str(j) + "simulation_new.npz")[table_var], axis=0)

table_mean_vola = pd.DataFrame(
    np.average(table1_data, axis=0),
    index=['Mean', 'Std_Dev'],
    columns=[r'$\theta$', r'$r$', r'$\mu_S$', r'$\sigma_S$', r'$\bar{\Delta}$', r'$\bar{\Phi}$']
)

print(tab.tabulate(table_mean_vola.transpose(), headers=['Mean', 'Std_Dev'], floatfmt=".4f",
                   tablefmt='latex_raw'))


table_var = 'table_parti'
table2_data = np.load(r'simu_results/' + str(0) + "simulation_new.npz")[table_var]
for j in range(1, n_files):
    table2_data = np.append(table2_data, np.load(r'simu_results/' + str(j) + "simulation_new.npz")[table_var], axis=0)
table_parti = pd.DataFrame(
    np.average(table2_data, axis=0),
    index=['Mean', 'Std_Dev', r'$10^{th}$', r'$50^{th}$', r'$90^{th}$'],
    columns=['Participation rate ', 'Entry rate', 'Exit rate']
)
print(tab.tabulate(table_parti.transpose(), headers=['Mean', 'Std_Dev', r'$10^{th}$', r'$50^{th}$', r'$90^{th}$'], floatfmt=".4f",
                   tablefmt='latex_raw'))

table_var = 'table_parti_cov'
table3_data = np.load(r'simu_results/' + str(0) + "simulation_new.npz")[table_var]
for j in range(1, n_files):
    table3_data = np.append(table3_data, np.load(r'simu_results/' + str(j) + "simulation_new.npz")[table_var], axis=0)
table_parti_cov = pd.DataFrame(
    np.average(table3_data, axis=0),
    index=['Cov'],
    columns=['cons_share', 'wealth_share', 'mu_S', 'theta']
)
print(tab.tabulate(table_parti_cov.transpose(), headers=['Cov'], floatfmt=".4f",
                   tablefmt='latex_raw'))



table_var = 'regression_table1_b'
table4_data = np.load(r'simu_results/' + str(0) + "simulation_new.npz")[table_var]
for j in range(1, n_files):
    table4_data = np.append(table4_data, np.load(r'simu_results/' + str(j) + "simulation_new.npz")[table_var], axis=0)
table_reg1 = pd.DataFrame(
    np.average(table4_data, axis=0),
    index=[r'$dR_{t-1, t}$', r'$dR_{t-2, t}$', r'$dR_{t-3, t}$'],
    columns=['parti', 'entry', 'exit']
)
print(tab.tabulate(table_reg1.transpose(), headers=[r'$dR_{t-1, t}$', r'$dR_{t-2, t}$', r'$dR_{t-3, t}$'], floatfmt=".4f",
                   tablefmt='latex_raw'))


table_var = 'regression_table2_b'
table5_data = np.load(r'simu_results/' + str(0) + "simulation_new.npz")[table_var]
for j in range(1, n_files):
    table5_data = np.append(table5_data, np.load(r'simu_results/' + str(j) + "simulation_new.npz")[table_var], axis=0)
table_reg2 = pd.DataFrame(
    np.average(table5_data, axis=0),
    index=['12m', '24m', '36m'],
    columns=[r'Total returns', r'Excess returns']
)
print(tab.tabulate(table_reg2.transpose(), headers=['12m', '24m', '36m'], floatfmt=".4f",
                   tablefmt='latex_raw'))

