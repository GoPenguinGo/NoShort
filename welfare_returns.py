import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI, simulate_mix_types
from src.param import rho_i, nu, mu_Y, sigma_Y, tax, phi, \
    dt, T_hat, Npre, Vhat, Ninit, Nt, Nc, tau, cohort_size, \
    cutoffs_age, n_age_cutoffs, colors, Mpath, \
    dZ_matrix, dZ_SI_matrix, dZ_build_matrix, dZ_SI_build_matrix, \
    cohort_labels, colors_short, PN_labels, age_labels, \
    Ntype, alpha_i, beta_i, beta0, rho_cohort_type, cohort_type_size, popu_age_groups
from src.param_mix import Nconstraint, rho_i_mix
import statsmodels.api as sm
import pandas as pd
import tabulate as tab
from scipy.interpolate import make_interp_spline

# Fig1: Shocks and beliefs
# Fig2: Distribution of beliefs
folder_address = r'E:\Users\A2010290\Documents\GitHub\NoShort/empirical/'
# folder_address = r'C:/Users/A2010290/OneDrive - BI Norwegian Business School (BIEDU)/Documents/GitHub computer 2/NoShort/empirical/'
data_shocks = pd.read_excel(
    folder_address + r'realized_shocks_US.xlsx',
    sheet_name='Sheet1',
    index_col=0
)

plt.rcParams["font.family"] = 'serif'
a_rho_bar = 1
b_rho_bar = -(tax * beta0 + rho_i[0, 0] + rho_i[1, 0])
c_rho_bar = alpha_i[0, 0] * tax * beta_i[0, 0] * (rho_i[0, 0] - rho_i[1, 0]) + rho_i[1, 0] * rho_i[0, 0] + tax * beta0 * \
            rho_i[1, 0]
rho_bar = (-b_rho_bar - np.sqrt(b_rho_bar ** 2 - 4 * a_rho_bar * c_rho_bar)) / (2 * a_rho_bar)

r_benchmark = nu - tax * beta0 + rho_bar + mu_Y - sigma_Y ** 2
sigma_benchmark = sigma_Y


# (complete, excluded, disappointment, reentry)
density_set = [
    (0.0, 0.0, 0.0, 1.0),
    (0.25, 0.25, 0.25, 0.25),
]
phi_set = [0.0, 0.5]
n_scenarios = len(density_set)
n_phi = len(phi_set)
dZ_build = dZ_build_matrix[0]
dZ_SI_build = dZ_SI_build_matrix[0]
dZ = dZ_matrix[0]
dZ_SI = dZ_SI_matrix[0]
dZ_actual = data_shocks.to_numpy()[:, 0]
Nt_data = dZ_actual.size - data_shocks['dZ_SI'].isna().sum()
filler = np.random.randn(data_shocks['dZ_SI'].isna().sum()) * np.sqrt(dt)
data_shocks.loc[data_shocks['dZ_SI'].isna(), 'dZ_SI'] = filler
dZ_SI_actual = data_shocks.to_numpy()[:, 1]
dZ[-dZ_actual.size:] = dZ_actual
dZ_SI[-Nt_data:] = dZ_SI_actual[-Nt_data:]
theta_compare = np.empty((n_scenarios, n_phi, Nt_data), dtype=np.float32)
r_compare = np.empty((n_scenarios, n_phi, Nt_data), dtype=np.float32)
Delta_compare = np.empty((n_scenarios, n_phi, Nt_data, Nconstraint, Nc), dtype=np.float16)
pi_compare = np.empty((n_scenarios, n_phi, Nt_data, Nconstraint, Nc), dtype=np.float16)
invest_tracker_compare = np.zeros((n_scenarios, n_phi, Nt_data, Nconstraint, Nc), dtype=int)

for i in range(n_scenarios):
    for j, phi in enumerate(phi_set):
        if i == 0:
            mode_trade = 'w_constraint'
            mode_learn = 'reentry'
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
            Delta_compare[i, j, :, 3] = Delta[-Nt_data:]
            pi_compare[i, j, :, 3] = pi[-Nt_data:]
            invest_tracker_compare[i, j, :, 3] = (pi[-Nt_data:] != 0)
        else:
            alpha_constraint = np.ones(
                (1, Nconstraint)) * density_set[i]
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
                portf_age_group,
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
            Delta_compare[i, j] = Delta[-Nt_data:]
            pi_compare[i, j] = pi[-Nt_data:]
            invest_tracker_compare[i, j] = (pi[-Nt_data:] != 0)
        theta_compare[i, j] = theta[-Nt_data:]
        r_compare[i, j] = r[-Nt_data:]


nn = 3  # number of cohorts illustrated
starts = np.arange(nn) * 240 + 24 * 12
cons_growth_time_series = np.zeros(
    (n_scenarios, n_phi, nn, Nconstraint, Nt_data),
    dtype=np.float32
)
cons_growth_benchmark = np.zeros(
    (nn, Nt_data),
    dtype=np.float32
)
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
            theta = theta_compare[i, j]
            r = r_compare[i, j]
            for m in range(nn):
                start = starts[m]
                a = 0
                b = 0
                for n in range(Nt_data):
                    if n < start:
                        pi_time_series[i, j, m, k, n] = np.nan
                        Delta_time_series[i, j, m, k, n] = np.nan
                        cons_growth_time_series[i, j, m, k, n] = np.nan
                        cons_growth_benchmark[m, n] = np.nan
                    else:
                        cohort_rank = Nt - (n - start) - 1
                        Delta_time_series[i, j, m, k, n] = Delta[n, cohort_rank]
                        pi_time_series[i, j, m, k, n] = pi[n, cohort_rank]
                        theta_t = theta[n]
                        theta_st = theta_t + Delta[n, cohort_rank]
                        r_t = r[n]
                        dZ_t = dZ_actual[n]
                        a += (-rho_i[0, 0] + r_benchmark + 1/2 * sigma_benchmark ** 2) * dt + sigma_benchmark * dZ_t
                        cons_growth_benchmark[m, n] = np.copy(a)
                        if k == 0:  # the unconstrained
                            parti_time_series[i, j, m, k, n] = 1
                            entry_time_series[i, j, m, k, n] = 0
                            exit_time_series[i, j, m, k, n] = 0
                            theta_st_plus = np.copy(theta_st)
                        elif k == 1:  # the excluded
                            parti_time_series[i, j, m, k, n] = 0
                            entry_time_series[i, j, m, k, n] = 0
                            exit_time_series[i, j, m, k, n] = 0
                            theta_st_plus = 0
                        else:
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
                            theta_st_plus = 0 if parti_time_series[i, j, m, k, n] == 0 \
                                else np.copy(theta_st)
                        b += (-rho_i[0, 0] + r_t + theta_st_plus * theta_t - 1/2 * theta_st_plus ** 2) * dt + theta_st_plus * dZ_t
                        cons_growth_time_series[i, j, m, k, n] = np.copy(b)



print('Figure 1')
x = 1926 + np.arange(Nt_data) * dt
fig, axes = plt.subplots(nrows=4, ncols=1, sharex='all', figsize=(10, 12))
for j, ax in enumerate(axes):
    if j == 0:
        Z = np.cumsum(dZ_actual[-Nt_data:])
        Z_SI = np.cumsum(
            dZ_SI_actual[-Nt_data:])
        ax.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
        ax.plot(x, Z, color='black', linewidth=1.5, label=r'$z^Y_t$')
        ax.plot(x, Z_SI, color='gray', linewidth=1.5, label=r'$z^{SI}_t$')
        ax.tick_params(axis='y', labelcolor='black')
        ax.tick_params(axis='x', labelcolor='black')
        ax.set_title(r'(a) Shocks to fundamental $z^Y$, and shocks to the signal $z^{SI}$')
        ax.legend(loc='lower left')

    elif j <= 2:
        cons_focus = cons_growth_time_series[0, j - 1, :, 3]
        parti_focus = parti_time_series[0, j - 1, :, 3]
        entry_focus = entry_time_series[0, j - 1, :, 3]
        exit_focus = exit_time_series[0, j - 1, :, 3]
        y_min_raw = np.nanmin(cons_focus)  # only the re-entry type
        y_max_raw = np.nanmax(cons_focus)
        y_max = (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2
        y_min = - (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2
        ax.set_ylabel(r'Log consumption $log(c_{s,t})$', color='black')
        # ax.set_ylim([y_min, y_max])
        if j == 1:
            ax.set_title(r'(b) Log consumption, reentry scenario, $\phi=0.0$')
        else:
            ax.set_title(r'(c) Log consumption, reentry scenario, $\phi=0.5$')
        for m in range(nn):
            # switch[m, starts[m]] = 1
            y_cohort = cons_focus[m]
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
            ax.vlines(starts[m]*dt+1926, ymax=y_max, ymin=y_min, color='grey', linestyle='--', linewidth=0.6)
            #  ax2.plot(t, y_cohort, label=cohort_labels[m], color=colors_short[m], linewidth=0.4)
            # if j == 1:
            #     ax.plot(x, y_cohort_P, color=colors_short[m], linewidth=1.5, label=cohort_labels[m])
            #     ax.plot(x, y_cohort_N, color=colors_short[m], linewidth=1.5, linestyle='dashed',
            #             )
            #     ax.scatter(x, y_cohort_entry, color='red', s=25, marker='o')
            #     ax.scatter(x, y_cohort_exit, color='orange', s=25, marker='o')
            # elif m == 0:
            #     ax.plot(x, y_cohort_P, color=colors_short[m], linewidth=1.5, label=PN_labels[0])
            #     ax.plot(x, y_cohort_N, color=colors_short[m], linewidth=1.5, linestyle='dashed',
            #             label=PN_labels[1])
            #     ax.scatter(x, y_cohort_entry, color='red', s=25, marker='o', label='Entry')
            #     ax.scatter(x, y_cohort_exit, color='orange', s=25, marker='o', label='Exit')
            # else:
            #     ax.plot(x, y_cohort_P, color=colors_short[m], linewidth=1.5)
            #     ax.plot(x, y_cohort_N, color=colors_short[m], linewidth=1.5, linestyle='dashed')
            #     ax.scatter(x, y_cohort_entry, color='red', s=25, marker='o')
            #     ax.scatter(x, y_cohort_exit, color='orange', s=25, marker='o')
            ax.plot(x, y_cohort, color=colors_short[m], linewidth=1.5, label=cohort_labels[m])
            ax.plot(x, cons_growth_benchmark[m], color=colors_short[m], linewidth=1.5, linestyle='dotted')
            if m == 2:
                ax.scatter(x, y_cohort_entry, color='red', s=25, marker='o', label='Entry')
                ax.scatter(x, y_cohort_exit, color='orange', s=25, marker='o', label='Exit')
            else:
                ax.scatter(x, y_cohort_entry, color='red', s=25, marker='o')
                ax.scatter(x, y_cohort_exit, color='orange', s=25, marker='o')
        ax.tick_params(axis='y', labelcolor='black')
        ax.legend(loc='lower left')

    else:
        constraint_labels = ['Designated P', 'Designated N', 'Disappointment', 'Reentry']
        colors_short3 = ['midnightblue', 'red', 'darkgreen', 'darkviolet']
        color_dot = 'red'
        ax.set_title(r'(d) Log consumption, mix scenario, $\phi=0.5$')
        ax.set_ylabel(r'Log consumption $log(c_{j, s,t})$', color='black')
        m = 0
        cons_focus = cons_growth_time_series[1, 1, m]
        parti_focus = parti_time_series[1, 1, m]
        entry_focus = entry_time_series[1, 1, m]
        exit_focus = exit_time_series[1, 1, m]
        y_min_raw = np.nanmin(cons_focus)  # only the re-entry type
        y_max_raw = np.nanmax(cons_focus)
        y_max = (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2
        y_min = - (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2
        for k in range(Nconstraint):
            if k != 1:  # not showing the excluded type
                y_cohort = cons_focus[k]
                y_cohort_N = np.ma.masked_where(parti_focus[k] == 1,
                                                y_cohort)
                y_cohort_P = np.ma.masked_where(parti_focus[k] == 0,
                                                y_cohort)
                y_cohort_entry = np.ma.masked_where(entry_focus[k] == 0,
                                                     y_cohort)
                y_cohort_exit = np.ma.masked_where(exit_focus[k] == 0,
                                                     y_cohort)
                ax.vlines(starts[m]*dt+1926, ymax=y_max, ymin=y_min, color='grey', linestyle='--', linewidth=0.6)
                # ax.plot(x, y_cohort_P, color=colors_short3[k], linewidth=1.5, label=constraint_labels[k])
                # ax.plot(x, y_cohort_N, color=colors_short3[k], linewidth=1.5, linestyle='dashed')
                ax.plot(x, y_cohort, color=colors_short3[k], linewidth=1.5, label=constraint_labels[k])
                if k == 3:
                    ax.scatter(x, y_cohort_entry, color='red', s=25, marker='o', label='Entry')
                    ax.scatter(x, y_cohort_exit, color='orange', s=25, marker='o', label='Exit')
                else:
                    ax.scatter(x, y_cohort_entry, color='red', s=25, marker='o')
                    ax.scatter(x, y_cohort_exit, color='orange', s=25, marker='o')
            ax.plot(x, cons_growth_benchmark[m], color='black', linewidth=1.5, linestyle='dotted')
            ax.legend(loc='lower left')
        ax.tick_params(axis='y', labelcolor='black')
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig(
    'f1_welfare.png',
    dpi=200)
plt.show()
plt.close()





############################################

# y_overall = np.empty((Nt_data, 5))  # overall
# y_P = np.empty((Nt_data, 5))  # participants / long
# y_N = np.empty((Nt_data, 5))  # non-participants / short
# y_min = np.empty((Nt_data, n_age_cutoffs))
# y_max = np.empty((Nt_data, n_age_cutoffs))
# y_cases = [y_overall, y_P, y_N]
# alpha_constraint = np.ones(
#     (1, Nconstraint)) * density_set[0]
# alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
# cohort_type_size_mix = cohort_size * alpha_i_mix
# Delta_focus = Delta_compare[0, 1, :, 3]
# invest_focus = invest_tracker_compare[0, 1, :, 3]
# cohort_size_flat = np.sum(cohort_type_size_mix, axis=0)[3]
# age_cutoffs_SCF = [int(Nt), int(Nt-1-12*15), int(Nt-1-12*35), int(Nt-1-12*55), 0]
# n_age_cutoffs = len(age_cutoffs_SCF) - 1
# for n in range(n_age_cutoffs):
#     Delta_age_group = Delta_focus[:, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]]
#     y_min[:, n] = np.amin(Delta_age_group, axis=1)
#     y_max[:, n] = np.amax(Delta_age_group, axis=1)
# for m in range(Nt_data):
#     Delta = Delta_focus[m]  # ((Nt, Nc))
#     parti_cohorts = invest_focus[m]
#     if np.sum(parti_cohorts) == Nc:
#         cohort_sizes = [cohort_size_flat]
#         Deltas = [Delta]
#         n_var = 1
#     else:
#         Delta1 = parti_cohorts * Delta
#         Delta2 = (1 - parti_cohorts) * Delta
#         cohort_size1 = parti_cohorts * cohort_size_flat
#         cohort_size2 = (1 - parti_cohorts) * cohort_size_flat
#         cohort_sizes = [cohort_size_flat, cohort_size1, cohort_size2]
#         Deltas = [Delta, Delta1, Delta2]
#         n_var = 3
#     for n in range(n_var):
#         Del = Deltas[n]
#         cohort_siz = cohort_sizes[n]
#         Delta_rank = Del.argsort()
#         Delta_sorted = Del[Delta_rank[::-1]]
#         cohort_size_sorted = cohort_siz[Delta_rank[::-1]]
#         popu_cumsum = np.cumsum(cohort_size_sorted)
#         total_popu = popu_cumsum[-1]
#         Delta_cutoff = np.zeros(5)
#         cutoff = np.searchsorted(popu_cumsum, [0.25 * total_popu, 0.5 * total_popu, 0.75 * total_popu])
#         Delta_cutoff[1:4] = Delta_sorted[cutoff]  # highest to lowest
#         Delta_cutoff[0] = np.max(Del[np.nonzero(Del)])
#         Delta_cutoff[4] = np.min(Del[np.nonzero(Del)])
#         y_cases[n][m] = Delta_cutoff
#
# x = 1926 + np.arange(Nt_data) * dt
# y1 = np.copy(y_overall)
# y2 = np.copy(y_P)
# y3 = np.copy(y_N)
# y4 = np.copy(y_min)
# y5 = np.copy(y_max)
# belief_cutoff_case = -theta_compare[0, 1]
#
# fig, axes = plt.subplots(nrows=2, sharex='all', figsize=(12, 7))
# for jj, ax in enumerate(axes):
#     ax.set_ylabel(r'Estimation error $\Delta_{s,t}$', color='black')
#     if jj == 0:
#         ax.set_ylim(-10, 8)
#         Z = np.cumsum(dZ_actual[-Nt_data:])
#         Z_SI = np.cumsum(
#             dZ_SI_actual[-Nt_data:])
#         ax.set_ylabel(r'$z^Y_t$ and $z^{SI}_t$', color='black')
#         ax.plot(x, Z, color='black', linewidth=2, label=r'$z^Y_t$')
#         ax.plot(x, Z_SI, color='gray', linewidth=2, label=r'$z^{SI}_t$')
#         ax.tick_params(axis='y', labelcolor='black')
#         ax.tick_params(axis='x', labelcolor='black')
#         ax.set_title(r'(a) Shocks to fundamental $z^Y$, and shocks to the signal $z^{SI}$')
#         ax.legend(loc='lower right')
#         ax.set_xlabel('Time in simulation')
#     else:
#         ax.set_ylim(-1.2, 1.2)
#         ax.set_title('(b) Distribution of estimation error, age groups')
#         ax.plot(x, belief_cutoff_case, color='black', linewidth=2,
#                 label=r'Cutoff $\Delta_{s,t}$'
#                 )
#         ax.set_xlabel('Time in simulation')
#         for k in range(n_age_cutoffs):
#             y40 = y4[:, k]
#             y50 = y5[:, k]
#             ax.fill_between(x, y40, y50, color=colors_short[k], linewidth=0., alpha=0.4,
#                             label=age_labels[k])
#         ax.legend(loc='lower right')
#
#     fig.tight_layout(h_pad=2, w_pad=2)  # otherwise the right y-label is slightly clipped
#     extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
# fig.savefig('f2_slides.png', dpi=200)
# plt.show()
# plt.close()