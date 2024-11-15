import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI, simulate_mix_types
from src.param import rho, nu, mu_Y, sigma_Y, sigma_Y_sqr, tax, phi, \
    dt, T_hat, Npre, Vhat, Ninit, T_cohort, Nt, Nc, tau, cohort_size, \
    cutoffs_age, n_age_cutoffs, colors, modes_trade, modes_learn, Mpath, \
    scenarios, dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    dZ_Y_cases, dZ_SI_cases, dZ_build_case, dZ_SI_build_case, t, red_labels, yellow_labels, cohort_labels, \
    scenario_labels, colors_short, PN_labels, age_labels, \
    Ntype, rho_i, alpha_i, beta_i, beta0, rho_cohort_type, cohort_type_size
from src.param_mix import Nconstraint, alpha_i_mix, beta_i_mix, rho_cohort_type_mix, \
    rho_i_mix, cohort_type_size_mix
import statsmodels.api as sm
import pandas as pd
# import tabulate as tab
from scipy.interpolate import make_interp_spline

# Fig1: Shocks and beliefs
# Fig2: Distribution of beliefs
folder_address = r'C:/Users/A2010290/OneDrive - BI Norwegian Business School (BIEDU)/Documents/GitHub computer 2/NoShort/empirical/'
data_shocks = pd.read_excel(
    folder_address + r'realized_shocks_US.xlsx',
    sheet_name='Sheet1',
    index_col=0
)
results_df1 = np.load("parti_rate_regressions.npz")

plt.rcParams["font.family"] = 'serif'

# (complete, excluded, disappointment, reentry)
density_set = [
    (0.0, 0.0, 0.0, 1.0),
    (0.25, 0.25, 0.25, 0.25),
]
phi_set = [0.0, 0.4]
n_scenarios = len(density_set)
n_phi = len(phi_set)
dZ_build = dZ_build_matrix[0]
dZ_SI_build = dZ_SI_build_matrix[0]
dZ = dZ_matrix[0]
filler = np.random.randn(data_shocks['dZ_SI'].isna().sum()) * np.sqrt(dt)
data_shocks.loc[data_shocks['dZ_SI'].isna(), 'dZ_SI'] = filler
dZ_actual = data_shocks.to_numpy()[:, 0]
dZ_SI_actual = data_shocks.to_numpy()[:, 1]
Nt_data = dZ_actual.size
dZ[-Nt_data:] = dZ_actual
dZ_SI = dZ_SI_matrix[0]
dZ_SI[-Nt_data:] = dZ_SI_actual
theta_compare = np.empty((n_scenarios, n_phi, Nt_data), dtype=np.float32)
Delta_bar_compare = np.zeros((n_scenarios, n_phi, Nt_data), dtype=np.float32)
Delta_compare = np.empty((n_scenarios, n_phi, Nt_data, Nconstraint, Nc), dtype=np.float16)
pi_compare = np.empty((n_scenarios, n_phi, Nt_data, Nconstraint, Nc), dtype=np.float16)
invest_tracker_compare = np.zeros((n_scenarios, n_phi, Nt_data, Nconstraint, Nc), dtype=np.float16)
for i, phi in enumerate(phi_set):
    for j in range(n_scenarios):
        alpha_constraint = np.ones(
            (1, Nconstraint)) * density_set[j]
        alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
        cohort_type_size_mix = cohort_size * alpha_i_mix
        beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio
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
            entry_mat,
            exit_mat
        ) = simulate_mix_types(Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax,
                               beta0,
                               phi, Npre, Ninit, T_hat,
                               dZ_build, dZ, dZ_SI_build, dZ_SI,
                               cutoffs_age, Ntype,
                               Nconstraint, rho_i_mix, alpha_i_mix, beta_i_mix,
                               rho_cohort_type_mix,
                               cohort_type_size_mix,
                               need_f='True',
                               need_Delta='True',
                               need_pi='True',
                               )
        theta_compare[i, j] = theta[-Nt_data:]
        Delta_bar_compare[i, j] = Delta_bar_parti[-Nt_data:]
        Delta_compare[i, j] = Delta[-Nt_data:]
        pi_compare[i, j] = pi[-Nt_data:]
        invest_tracker_compare[i, j] = invest_tracker[-Nt_data:]

nn = 3  # number of cohorts illustrated
starts = np.arange(nn) * 240 + 14 * 12
Delta_time_series = np.zeros((n_scenarios, n_phi, nn, Nconstraint, Nt_data), dtype=np.float32)
pi_time_series = np.zeros((n_scenarios, n_phi, nn, Nconstraint, Nt_data), dtype=np.float32)
switch_time_series = np.zeros((n_scenarios, n_phi, nn, Nconstraint, Nt_data), dtype=np.float32)
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
                            switch_time_series[i, j, m, k, n] = 0
                        elif k == 1:  # the excluded
                            parti_time_series[i, j, m, k, n] = 0
                            switch_time_series[i, j, m, k, n] = 0
                        else:
                            parti_time_series[i, j, m, k, n] = 1 if pi_time_series[i, j, m, k, n] > 0 else 0
                            switch_PN = 1 if (pi_time_series[i, j, m, k, n - 1] > 0) and (
                                    pi_time_series[i, j, m, k, n] == 0) else 0
                            switch_NP = 1 if (pi_time_series[i, j, m, k, n] > 0) and (
                                    pi_time_series[i, j, m, k, n - 1] == 0) else 0
                            switch_time_series[i, j, m, k, n] = 1 if switch_PN == 1 else 0
                            switch_time_series[i, j, m, k, n] = 1 if switch_NP == 1 else 0
                            if switch_NP == 1:
                                parti_time_series[i, j, m, k, n] = 0.5
                            if switch_PN == 1:
                                parti_time_series[i, j, m, k, n - 1] = 0.5

######################################
###########   Figure 1   #############
######################################
print('Figure 1')
x = 1926 + np.arange(Nt_data) * dt
fig, axes = plt.subplots(nrows=4, ncols=1, sharex='all', figsize=(10, 12))
for j, ax in enumerate(axes):
    if j == 0:
        Z = np.cumsum(dZ_actual)
        Z_SI = np.cumsum(
            dZ_SI_actual)
        ax.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
        ax.plot(x, Z, color='black', linewidth=1.2, label=r'$z^Y_t$')
        ax.plot(x, Z_SI, color='gray', linewidth=1.2, label=r'$z^{SI}_t$')
        ax.tick_params(axis='y', labelcolor='black')
        ax.tick_params(axis='x', labelcolor='black')
        ax.legend(loc='upper left')
        ax.set_title(r'Shocks to fundamental $z^Y$, and shocks to the signal $z^{SI}$')
        ax.legend(loc='lower right')

    elif j <= 2:
        Delta_focus = Delta_time_series[0, j - 1, :, 3]
        parti_focus = parti_time_series[0, j - 1, :, 3]
        switch_focus = switch_time_series[0, j - 1, :, 3]
        y_min_raw = np.nanmin(Delta_focus)  # only the re-entry type
        y_max_raw = np.nanmax(Delta_focus)
        y_max = (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2
        y_min = - (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2
        ax.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
        ax.set_ylim([y_min, y_max])
        if j == 1:
            ax.set_title(r'Re-entry scenario, $\phi=0.0$')
        else:
            ax.set_title(r'Re-entry scenario, $\phi=0.4$')
        for m in range(nn):
            # switch[m, starts[m]] = 1
            y_cohort = Delta_focus[m]
            y_cohort_N = np.ma.masked_where(parti_focus[m] == 1,
                                            y_cohort)
            y_cohort_P = np.ma.masked_where(parti_focus[m] == 0,
                                            y_cohort)
            y_cohort_switch = np.ma.masked_where(
                switch_focus[m] == 0,
                y_cohort)
            ax.vlines(starts[m]*dt+1926, ymax=y_max, ymin=y_min, color='grey', linestyle='--', linewidth=0.6)
            #  ax2.plot(t, y_cohort, label=cohort_labels[m], color=colors_short[m], linewidth=0.4)
            if j == 1:
                ax.plot(x, y_cohort_P, color=colors_short[m], linewidth=1.5, label=cohort_labels[m])
                ax.plot(x, y_cohort_N, color=colors_short[m], linewidth=1.5, linestyle='dotted',
                        )
                ax.scatter(x, y_cohort_switch, color='red', s=10, marker='o')
            elif m == 0:
                ax.plot(x, y_cohort_P, color=colors_short[m], linewidth=1.5, label=PN_labels[0])
                ax.plot(x, y_cohort_N, color=colors_short[m], linewidth=1.5, linestyle='dotted',
                        label=PN_labels[1])
                ax.scatter(x, y_cohort_switch, color='red', s=10, marker='o', label='switch')
            else:
                ax.plot(x, y_cohort_P, color=colors_short[m], linewidth=1.5)
                ax.plot(x, y_cohort_N, color=colors_short[m], linewidth=1.5, linestyle='dotted')
                ax.scatter(x, y_cohort_switch, color='red', s=10, marker='o')
        ax.tick_params(axis='y', labelcolor='black')
        ax.legend(loc='lower right')

    else:
        constraint_labels = ['Designated P', 'Designated N', 'Disappointment', 'Reentry']
        colors_short3 = ['midnightblue', 'red', 'darkgreen', 'darkviolet']
        color_dot = 'red'
        ax.set_title(r'Mix scenario, $\phi=0.4$')
        ax.set_ylabel(r'Estimation error $\Delta_{j, s,t}$', color='black')
        m = 0
        Delta_focus = Delta_time_series[1, 1, m]
        parti_focus = parti_time_series[1, 1, m]
        switch_focus = switch_time_series[1, 1, m]
        y_min_raw = np.nanmin(Delta_focus)  # only the re-entry type
        y_max_raw = np.nanmax(Delta_focus)
        y_max = (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2
        y_min = - (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2
        for k in range(Nconstraint):
            if k != 1:  # not showing the excluded type
                y_cohort = Delta_focus[k]
                y_cohort_N = np.ma.masked_where(parti_focus[k] == 1,
                                                y_cohort)
                y_cohort_P = np.ma.masked_where(parti_focus[k] == 0,
                                                y_cohort)
                y_cohort_switch = np.ma.masked_where(switch_focus[k] == 0,
                                                     y_cohort)
                ax.vlines(starts[m]*dt+1926, ymax=y_max, ymin=y_min, color='grey', linestyle='--', linewidth=0.6)
                ax.plot(x, y_cohort_P, color=colors_short3[k], linewidth=1.5, label=constraint_labels[k])
                ax.plot(x, y_cohort_N, color=colors_short3[k], linewidth=1.5, linestyle='dotted')
                ax.scatter(x, y_cohort_switch, color=color_dot, s=10, marker='o')
            ax.legend(loc='lower right')
        ax.tick_params(axis='y', labelcolor='black')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(
    'f1.png',
    dpi=100)
plt.show()
plt.close()

######################################
###########   Figure 2   #############
######################################
# phi = 0.4, reentry scenario
y_overall = np.empty((Nt_data, 5))  # overall
y_P = np.empty((Nt_data, 5))  # participants / long
y_N = np.empty((Nt_data, 5))  # non-participants / short
y_min = np.empty((Nt_data, n_age_cutoffs))
y_max = np.empty((Nt_data, n_age_cutoffs))
y_cases = [y_overall, y_P, y_N]
alpha_constraint = np.ones(
    (1, Nconstraint)) * density_set[0]
alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
cohort_type_size_mix = cohort_size * alpha_i_mix
Delta_focus = Delta_compare[0, 1, :, 3]
invest_focus = invest_tracker_compare[0, 1, :, 3]
cohort_size_flat = np.sum(cohort_type_size_mix, axis=0)[3]
age_cutoffs_SCF = [int(Nt-1), int(Nt-1-12*15), int(Nt-1-12*35), int(Nt-1-12*55), 0]
n_age_cutoffs = len(age_cutoffs_SCF) - 1
for n in range(n_age_cutoffs):
    Delta_age_group = Delta_focus[:, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]]
    y_min[:, n] = np.amin(Delta_age_group, axis=1)
    y_max[:, n] = np.amax(Delta_age_group, axis=1)
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
        Delta_cutoff = np.zeros(5)
        cutoff = np.searchsorted(popu_cumsum, [0.25 * total_popu, 0.5 * total_popu, 0.75 * total_popu])
        Delta_cutoff[1:4] = Delta_sorted[cutoff]  # highest to lowest
        Delta_cutoff[0] = np.max(Del[np.nonzero(Del)])
        Delta_cutoff[4] = np.min(Del[np.nonzero(Del)])
        y_cases[n][m] = Delta_cutoff

x = 1926 + np.arange(Nt_data) * dt
y1 = np.copy(y_overall)
y2 = np.copy(y_P)
y3 = np.copy(y_N)
y4 = np.copy(y_min)
y5 = np.copy(y_max)
belief_cutoff_case = -theta_compare[0, 1]

fig, axes = plt.subplots(nrows=2, sharex='all', sharey='all', figsize=(10, 8))
for jj, ax in enumerate(axes):
    ax.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
    ax.set_ylim(-5, 3)
    if jj == 0:
        ax.set_title('Distribution of estimation error, participants vs. non-participants')
        y20 = y2[:, 0]
        y21 = y2[:, 1]
        y22 = y2[:, 2]
        y23 = y2[:, 3]
        y24 = np.maximum(belief_cutoff_case, y1[:, 4])
        y30 = np.maximum(belief_cutoff_case, y3[:, 0])
        y31 = y3[:, 1]
        y32 = y3[:, 2]
        y33 = y3[:, 3]
        y34 = y3[:, 4]
        ax.fill_between(x, y20, y24, color='blue', linewidth=0., alpha=0.4)
        ax.fill_between(x, y21, y23, color='blue', linewidth=0., alpha=0.7, label=PN_labels[0])
        ax.fill_between(x, y30, y34, color='green', linewidth=0., alpha=0.4)
        ax.fill_between(x, y31, y33, color='green', linewidth=0., alpha=0.7, label=PN_labels[1])
        ax.plot(x, belief_cutoff_case, color='black', linewidth=2, label=r'Cutoff $\Delta_{s,t}$')
    else:
        ax.set_title('Distribution of estimation error, age groups')
        ax.plot(x, belief_cutoff_case, color='black', linewidth=2, label=r'Cutoff $\Delta_{s,t}$')
        for k in range(n_age_cutoffs):
            y40 = y4[:, k]
            y50 = y5[:, k]
            ax.fill_between(x, y40, y50, color=colors_short[k], linewidth=0., alpha=0.4,
                            label=age_labels[k])
    ax.legend(loc='lower right')
    ax.set_xlabel('Time in simulation')
    fig.tight_layout(h_pad=2, w_pad=2)  # otherwise the right y-label is slightly clipped
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig('f2.png', dpi=200)
plt.show()
plt.close()


######################################
###########   Figure 3   #############
######################################
## Estimation error and participation rate given age
age_Delta = results_df1['age Delta']  # paths, scenario, phi, type, age
age_parti = results_df1['age parti']
ave_age_Delta = np.average(age_Delta, axis=0)
ave_age_parti = np.average(age_parti, axis=0)

phi_set = [0.0, 0.4, 0.8]
n_phi = len(phi_set)
age_cut = 100
Nc_cut = int(age_cut / dt)
x_age = np.arange(1, Nc_cut, 12) * dt
Delta_focus = ave_age_Delta[0, :, 3]
parti_focus = ave_age_parti[0, :, 3]
fig_titles = [r'Reentry and complete market, average $\mid\Delta_{s,t}\mid$',
              'Reentry, average participation probability']
y_titles = [r'Average $\mid\Delta_{s,t}\mid$', 'Average participation probability']
fig, axes = plt.subplots(nrows=1, ncols=2, sharex='all', figsize=(15, 7.5))
for j, ax in enumerate(axes):
    ax.set_xlabel('Age')
    y_case = Delta_vector if j == 0 else invest_vector
    ax.set_ylabel(y_titles[j])
    for i in range(3):
        if j == 0:
            ax.set_ylim(0.04, 0.18)
            y_reentry = y_case[1, i, :N_cut]
            label_i = r'$\phi$=' + str('{0:.2f}'.format(phi_vector[i]))
            ax.plot(x, y_reentry, color=colors[i], linewidth=0.8, label=label_i)
            ax.legend()
            if phi_vector[i] != 0:
                y_complete = y_case[0, i, :N_cut]
                ax.plot(x, y_complete, color=colors[i], linewidth=0.8, linestyle='dashed')
        else:
            y = y_case[i, :N_cut]
            ax.plot(x, y, color=colors[i], linewidth=0.8)
    ax.set_title(fig_titles[j], color='black')
    ax.tick_params(axis='y', labelcolor='black')
    ax.set_xlim(-1, 100)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Average estimation error and age.png', dpi=60)
plt.show()
# plt.close()