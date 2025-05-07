import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI, simulate_mix_types
import cProfile
import pstats

from src.param import (
    rho_i,
    nu,
    mu_Y,
    sigma_Y,
    tax,
    phi,
    dt,
    T_hat,
    Npre,
    Vhat,
    Ninit,
    Nt,
    Nc,
    tau,
    cohort_size,
    cutoffs_age,
    n_age_cutoffs,
    colors,
    Mpath,
    dZ_matrix,
    dZ_SI_matrix,
    dZ_build_matrix,
    dZ_SI_build_matrix,
    cohort_labels,
    colors_short,
    PN_labels,
    age_labels,
    Ntype,
    alpha_i,
    beta_i,
    beta0,
    rho_cohort_type,
    cohort_type_size,
    popu_age_groups,
)
from src.param_mix import Nconstraint, rho_i_mix
import statsmodels.api as sm
import pandas as pd
import tabulate as tab
from scipy.interpolate import make_interp_spline
import os

# Define tic and toc globally
import time


def tic():
    global start_time
    start_time = time.perf_counter()


def toc():
    elapsed = time.perf_counter() - start_time
    print(f"Elapsed time: {elapsed:.4f} seconds")


# Fig1: Shocks and beliefs
# Fig2: Distribution of beliefs
# folder_address = r'E:\Users\A2010290\Documents\GitHub\NoShort/empirical/'
folder_address = os.path.join("empirical", "")
# folder_address = r'C:/Users/A2010290/OneDrive - BI Norwegian Business School (BIEDU)/Documents/GitHub computer 2/NoShort/empirical/'
data_shocks = pd.read_excel(
    os.path.join(folder_address, "realized_shocks_US.xlsx"),
    sheet_name="Sheet1",
    index_col=0,
)
# results_df1 = np.load("parti_rate_regressions.npz")
# file_list_regression = results_df1.files
# results_mean_vola = np.load('results_mean_vola.npz')
# file_list_mean_vola = results_mean_vola.files

plt.rcParams["font.family"] = "serif"

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
Nt_data = dZ_actual.size - data_shocks["dZ_SI"].isna().sum()
filler = np.random.randn(data_shocks["dZ_SI"].isna().sum()) * np.sqrt(dt)
data_shocks.loc[data_shocks["dZ_SI"].isna(), "dZ_SI"] = filler
dZ_SI_actual = data_shocks.to_numpy()[:, 1]
dZ[-dZ_actual.size :] = dZ_actual
dZ_SI[-Nt_data:] = dZ_SI_actual[-Nt_data:]
theta_compare = np.empty((n_scenarios, n_phi, Nt_data), dtype=np.float32)
Delta_bar_compare = np.zeros((n_scenarios, n_phi, Nt_data), dtype=np.float32)
Delta_compare = np.empty(
    (n_scenarios, n_phi, Nt_data, Nconstraint, Nc), dtype=np.float16
)
pi_compare = np.empty((n_scenarios, n_phi, Nt_data, Nconstraint, Nc), dtype=np.float16)
invest_tracker_compare = np.zeros(
    (n_scenarios, n_phi, Nt_data, Nconstraint, Nc), dtype=int
)
parti_age_group_compare = np.zeros((n_scenarios, n_phi, Nt_data, 4), dtype=np.float16)


def run_simulations():
    for i in range(n_scenarios):
        for j, phi in enumerate(phi_set):
            if i == 0:
                mode_trade = "w_constraint"
                mode_learn = "reentry"
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
                    exit_mat,
                ) = simulate_SI(
                    mode_trade,
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
                    need_f="True",
                    need_Delta="True",
                    need_pi="True",
                )
                Delta_compare[i, j, :, 3] = Delta[-Nt_data:]
                pi_compare[i, j, :, 3] = pi[-Nt_data:]
                invest_tracker_compare[i, j, :, 3] = pi[-Nt_data:] != 0
            else:
                alpha_constraint = np.ones((1, Nconstraint)) * density_set[i]
                alpha_i_mix = np.reshape(
                    alpha_i * alpha_constraint, (Ntype, Nconstraint, 1)
                )
                cohort_type_size_mix = cohort_size * alpha_i_mix
                beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio
                rho_cohort_type_mix = (
                    alpha_i_mix * beta_i_mix * np.exp(-(rho_i_mix + nu) * tau)
                )  # shape(2, 6000)
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
                    exit_mat,
                ) = simulate_mix_types(
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
                    cutoffs_age,
                    Ntype,
                    Nconstraint,
                    rho_i_mix,
                    alpha_i_mix,
                    beta_i_mix,
                    rho_cohort_type_mix,
                    cohort_type_size_mix,
                    need_f="True",
                    need_Delta="True",
                    need_pi="True",
                )
                Delta_compare[i, j] = Delta[-Nt_data:]
                pi_compare[i, j] = pi[-Nt_data:]
                invest_tracker_compare[i, j] = pi[-Nt_data:] != 0
            theta_compare[i, j] = theta[-Nt_data:]
            Delta_bar_compare[i, j] = Delta_bar_parti[-Nt_data:]
            parti_age_group_compare[i, j] = parti_age_group[-Nt_data:]
    return (
        theta_compare,
        Delta_compare,
        pi_compare,
        invest_tracker_compare,
        parti_age_group_compare,
    )


tic()
results = {}


def profiled_run():
    return run_simulations()


results = cProfile.runctx("profiled_run()", globals(), locals(), "profile_output")

toc()

nn = 3  # number of cohorts illustrated
starts = np.arange(nn) * 240 + 24 * 12
Delta_time_series = np.zeros(
    (n_scenarios, n_phi, nn, Nconstraint, Nt_data), dtype=np.float32
)
pi_time_series = np.zeros(
    (n_scenarios, n_phi, nn, Nconstraint, Nt_data), dtype=np.float32
)
entry_time_series = np.zeros(
    (n_scenarios, n_phi, nn, Nconstraint, Nt_data), dtype=np.float32
)
exit_time_series = np.zeros(
    (n_scenarios, n_phi, nn, Nconstraint, Nt_data), dtype=np.float32
)
parti_time_series = np.zeros(
    (n_scenarios, n_phi, nn, Nconstraint, Nt_data), dtype=np.float32
)
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
                            parti_time_series[i, j, m, k, n] = (
                                1 if pi_time_series[i, j, m, k, n] != 0 else 0
                            )
                            switch_PN = (
                                1
                                if (pi_time_series[i, j, m, k, n - 1] != 0)
                                and (pi_time_series[i, j, m, k, n] == 0)
                                else 0
                            )
                            switch_NP = (
                                1
                                if (pi_time_series[i, j, m, k, n] != 0)
                                and (pi_time_series[i, j, m, k, n - 1] == 0)
                                else 0
                            )
                            entry_time_series[i, j, m, k, n] = (
                                1 if switch_NP == 1 else 0
                            )
                            exit_time_series[i, j, m, k, n] = 1 if switch_PN == 1 else 0
                            exit_time_series[i, j, m, k, start] = 0
                            if switch_NP == 1:
                                parti_time_series[i, j, m, k, n] = 0.5
                            if switch_PN == 1:
                                parti_time_series[i, j, m, k, n - 1] = 0.5

######################################
###########   Figure 1   #############  #todo: figure 2 in the paper
######################################
print("Figure 1")
x = 1926 + np.arange(Nt_data) * dt
fig, axes = plt.subplots(nrows=4, ncols=1, sharex="all", figsize=(10, 12))
for j, ax in enumerate(axes):
    if j == 0:
        Z = np.cumsum(dZ_actual[-Nt_data:])
        Z_SI = np.cumsum(dZ_SI_actual[-Nt_data:])
        ax.set_ylabel(r"$z^Y_t$ and $z^{SI}_t$", color="black")
        ax.plot(x, Z, color="black", linewidth=1.5, label=r"$z^Y_t$")
        ax.plot(x, Z_SI, color="gray", linewidth=1.5, label=r"$z^{SI}_t$")
        ax.tick_params(axis="y", labelcolor="black")
        ax.tick_params(axis="x", labelcolor="black")
        ax.set_title(
            r"(a) Shocks to fundamental $z^Y$, and shocks to the signal $z^{SI}$"
        )
        ax.legend(loc="lower left")

    elif j <= 2:
        Delta_focus = Delta_time_series[0, j - 1, :, 3]
        parti_focus = parti_time_series[0, j - 1, :, 3]
        entry_focus = entry_time_series[0, j - 1, :, 3]
        exit_focus = exit_time_series[0, j - 1, :, 3]
        y_min_raw = np.nanmin(Delta_focus)  # only the re-entry type
        y_max_raw = np.nanmax(Delta_focus)
        y_max = (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2
        y_min = -(y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2
        ax.set_ylabel(r"Estimation error $\Delta_{s,t}$", color="black")
        ax.set_ylim([y_min, y_max])
        if j == 1:
            ax.set_title(r"(b) Estimation error, reentry scenario, $\phi=0.0$")
        else:
            ax.set_title(r"(c) Estimation error, reentry scenario, $\phi=0.5$")
        for m in range(nn):
            # switch[m, starts[m]] = 1
            y_cohort = Delta_focus[m]
            y_cohort_N = np.ma.masked_where(parti_focus[m] == 1, y_cohort)
            y_cohort_P = np.ma.masked_where(parti_focus[m] == 0, y_cohort)
            y_cohort_entry = np.ma.masked_where(entry_focus[m] == 0, y_cohort)
            y_cohort_exit = np.ma.masked_where(exit_focus[m] == 0, y_cohort)
            ax.vlines(
                starts[m] * dt + 1926,
                ymax=y_max,
                ymin=y_min,
                color="grey",
                linestyle="--",
                linewidth=0.6,
            )
            #  ax2.plot(t, y_cohort, label=cohort_labels[m], color=colors_short[m], linewidth=0.4)
            if j == 1:
                ax.plot(
                    x,
                    y_cohort_P,
                    color=colors_short[m],
                    linewidth=1.5,
                    label=cohort_labels[m],
                )
                ax.plot(
                    x,
                    y_cohort_N,
                    color=colors_short[m],
                    linewidth=1.5,
                    linestyle="dashed",
                )
                ax.scatter(x, y_cohort_entry, color="red", s=25, marker="o")
                ax.scatter(x, y_cohort_exit, color="orange", s=25, marker="o")
            elif m == 0:
                ax.plot(
                    x,
                    y_cohort_P,
                    color=colors_short[m],
                    linewidth=1.5,
                    label=PN_labels[0],
                )
                ax.plot(
                    x,
                    y_cohort_N,
                    color=colors_short[m],
                    linewidth=1.5,
                    linestyle="dashed",
                    label=PN_labels[1],
                )
                ax.scatter(
                    x, y_cohort_entry, color="red", s=25, marker="o", label="Entry"
                )
                ax.scatter(
                    x, y_cohort_exit, color="orange", s=25, marker="o", label="Exit"
                )
            else:
                ax.plot(x, y_cohort_P, color=colors_short[m], linewidth=1.5)
                ax.plot(
                    x,
                    y_cohort_N,
                    color=colors_short[m],
                    linewidth=1.5,
                    linestyle="dashed",
                )
                ax.scatter(x, y_cohort_entry, color="red", s=25, marker="o")
                ax.scatter(x, y_cohort_exit, color="orange", s=25, marker="o")
        ax.tick_params(axis="y", labelcolor="black")
        ax.legend(loc="lower left")

    else:
        constraint_labels = [
            "Designated P",
            "Designated N",
            "Disappointment",
            "Reentry",
        ]
        colors_short3 = ["midnightblue", "red", "darkgreen", "darkviolet"]
        color_dot = "red"
        ax.set_title(r"(d) Estimation error, mix scenario, $\phi=0.5$")
        ax.set_ylabel(r"Estimation error $\Delta_{j, s,t}$", color="black")
        m = 0
        Delta_focus = Delta_time_series[1, 1, m]
        parti_focus = parti_time_series[1, 1, m]
        entry_focus = entry_time_series[1, 1, m]
        exit_focus = exit_time_series[1, 1, m]
        y_min_raw = np.nanmin(Delta_focus)  # only the re-entry type
        y_max_raw = np.nanmax(Delta_focus)
        y_max = (y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2
        y_min = -(y_max_raw - y_min_raw) * 0.6 + (y_max_raw + y_min_raw) / 2
        for k in range(Nconstraint):
            if k != 1:  # not showing the excluded type
                y_cohort = Delta_focus[k]
                y_cohort_N = np.ma.masked_where(parti_focus[k] == 1, y_cohort)
                y_cohort_P = np.ma.masked_where(parti_focus[k] == 0, y_cohort)
                y_cohort_entry = np.ma.masked_where(entry_focus[k] == 0, y_cohort)
                y_cohort_exit = np.ma.masked_where(exit_focus[k] == 0, y_cohort)
                ax.vlines(
                    starts[m] * dt + 1926,
                    ymax=y_max,
                    ymin=y_min,
                    color="grey",
                    linestyle="--",
                    linewidth=0.6,
                )
                ax.plot(
                    x,
                    y_cohort_P,
                    color=colors_short3[k],
                    linewidth=1.5,
                    label=constraint_labels[k],
                )
                ax.plot(
                    x,
                    y_cohort_N,
                    color=colors_short3[k],
                    linewidth=1.5,
                    linestyle="dashed",
                )
                ax.scatter(x, y_cohort_entry, color="red", s=25, marker="o")
                ax.scatter(x, y_cohort_exit, color="orange", s=25, marker="o")
            ax.legend(loc="lower left")
        ax.tick_params(axis="y", labelcolor="black")
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("f1.png", dpi=200)
plt.show()
plt.close()

######################################
###########   Figure 2   #############  # todo: figure 4 in the paper
######################################
# phi = 0.5, reentry scenario
y_overall = np.empty((Nt_data, 5))  # overall
y_P = np.empty((Nt_data, 5))  # participants / long
y_N = np.empty((Nt_data, 5))  # non-participants / short
y_min = np.empty((Nt_data, n_age_cutoffs))
y_max = np.empty((Nt_data, n_age_cutoffs))
y_cases = [y_overall, y_P, y_N]
alpha_constraint = np.ones((1, Nconstraint)) * density_set[0]
alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
cohort_type_size_mix = cohort_size * alpha_i_mix
Delta_focus = Delta_compare[0, 1, :, 3]
invest_focus = invest_tracker_compare[0, 1, :, 3]
cohort_size_flat = np.sum(cohort_type_size_mix, axis=0)[3]
age_cutoffs_SCF = [
    int(Nt),
    int(Nt - 1 - 12 * 15),
    int(Nt - 1 - 12 * 35),
    int(Nt - 1 - 12 * 55),
    0,
]
n_age_cutoffs = len(age_cutoffs_SCF) - 1
for n in range(n_age_cutoffs):
    Delta_age_group = Delta_focus[:, age_cutoffs_SCF[n + 1] : age_cutoffs_SCF[n]]
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
        cutoff = np.searchsorted(
            popu_cumsum, [0.25 * total_popu, 0.5 * total_popu, 0.75 * total_popu]
        )
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

fig, axes = plt.subplots(nrows=2, sharex="all", sharey="all", figsize=(10, 8))
for jj, ax in enumerate(axes):
    ax.set_ylabel(r"Estimation error $\Delta_{s,t}$", color="black")
    ax.set_ylim(-1.5, 1.5)
    if jj == 0:
        ax.set_title(
            "(a) Distribution of estimation error, participants vs. non-participants"
        )
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
        ax.fill_between(
            x, y20, y24, color="blue", linewidth=0.0, alpha=0.4, label=PN_labels[0]
        )
        ax.fill_between(x, y21, y23, color="blue", linewidth=0.0, alpha=0.7)
        ax.fill_between(
            x, y30, y34, color="green", linewidth=0.0, alpha=0.4, label=PN_labels[1]
        )
        ax.fill_between(x, y31, y33, color="green", linewidth=0.0, alpha=0.7)
        ax.plot(
            x,
            belief_cutoff_case,
            color="black",
            linewidth=2,
            label=r"Cutoff $\Delta_{s,t}$",
        )
    else:
        ax.set_title("(b) Distribution of estimation error, age groups")
        ax.plot(
            x,
            belief_cutoff_case,
            color="black",
            linewidth=2,
            # label=r'Cutoff $\Delta_{s,t}$'
        )
        for k in range(n_age_cutoffs):
            y40 = y4[:, k]
            y50 = y5[:, k]
            ax.fill_between(
                x,
                y40,
                y50,
                color=colors_short[k],
                linewidth=0.0,
                alpha=0.4,
                label=age_labels[k],
            )
    ax.legend(loc="lower right")
    ax.set_xlabel("Time in simulation")
    fig.tight_layout(
        h_pad=2, w_pad=2
    )  # otherwise the right y-label is slightly clipped
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
fig.savefig("f2.png", dpi=200)
plt.show()
plt.close()

######################################
##########   Figure 2.1   ############
######################################
for i in range(2):
    parti_age_group_focus = parti_age_group_compare[i, 1]  # reentry, phi = 0.5
    y_focus = np.cumsum(parti_age_group_focus * popu_age_groups, axis=1)
    # y_focus = np.copy(parti_age_group_focus)
    fig, ax = plt.subplots(nrows=1, sharex="all", sharey="all", figsize=(10, 4))
    ax.set_ylabel(r"Participation rate", color="black")
    title_i = "Re-entry" if i == 0 else "Mix"
    ax.set_title("Participation rate in age groups, " + title_i)
    ax2 = ax.twinx()
    ax2.plot(x, Z, color="black", linewidth=1)
    ax2.set_ylabel(r"Shocks, $Z_t^Y$", color="black")
    for ii in range(4):
        # ax.plot(x, y_focus[:, ii], color=colors_short[ii], linewidth=2, label=age_labels[ii])
        y_focus_bottom = y_focus[:, ii - 1] if ii > 0 else y_focus[:, ii] * 0.0
        ax.fill_between(
            x,
            y_focus_bottom,
            y_focus[:, ii],
            color=colors_short[ii],
            linewidth=0.0,
            alpha=0.4,
            label=age_labels[ii],
        )
    ax.legend(loc="lower right")
    ax.set_xlabel("Time in simulation")
    fig.tight_layout(
        h_pad=2, w_pad=2
    )  # otherwise the right y-label is slightly clipped
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())

    fig.savefig(str(i) + "f2_1.png", dpi=200)
    plt.show()
    plt.close()


######################################
###########   Figure 3   #############  # todo: figure 3 in the paper
######################################
## Estimation error and participation rate given age
folder_address = r'E:\Users\A2010290\Documents\GitHub\NoShort/reg_results2/'
n_files = 40

Delta_age_all = np.zeros((3, n_files, 2, 200))
for i in range(n_files):  # todo: from "parti experiment.py"
    Delta_age_all[0, i] = np.average(np.load(folder_address + str(i) + str(0.0) + "simulation_new.npz")['Delta_age'], axis=0)
    Delta_age_all[1, i] = np.average(np.load(folder_address + str(i) + str(0.5) + "simulation_new.npz")['Delta_age'], axis=0)
    Delta_age_all[2, i] = np.average(np.load(folder_address + str(i) + str(0.8) + "simulation_new.npz")['Delta_age'],
                                     axis=0)
Delta_age_ave = np.average(Delta_age_all, axis=1)

phi_set = [0.0, 0.5, 0.8]
parti_age_all = np.zeros((100, 3, 200))
for i in range(100):
    for j, phi_j in enumerate(phi_set):
        parti_age_all[i, j] = np.load(folder_address + str(i) + str(phi_j) + 'parti_age.npy')
parti_age_ave = np.average(parti_age_all, axis=0)

n_phi = len(phi_set)
age_cut = 100
x_age = np.arange(0, age_cut)
fig_titles = [r'Reentry and complete market, average $\mid\Delta_{s,t}\mid$',
              'Reentry, average participation probability']
y_titles = [r'Average $\mid\Delta_{s,t}\mid$', 'Average participation probability']
fig, axes = plt.subplots(nrows=1, ncols=2, sharex='all', figsize=(10, 5))
for j, ax in enumerate(axes):
    ax.set_xlabel('Years since entering the economy')
    y_case = np.copy(Delta_age_ave) if j == 0 else np.copy(parti_age_ave)
    ax.set_ylabel(y_titles[j])
    for i in range(3):
        if j == 0:
            y_reentry = y_case[i, 1, :age_cut]
            y_complete = y_case[i, 0, :age_cut]
            if phi_set[i] == 0.5:
                ax.plot(x_age, y_reentry, color=colors[i], linewidth=1.5, label="Re-entry")
                ax.plot(x_age, y_complete, color=colors[i], linewidth=1.5, linestyle='dashed', label="Complete Market")
                ax.legend()
            elif phi_set[i] == 0:
                ax.plot(x_age, y_reentry, color=colors[i], linewidth=1.5)
            else:
                ax.plot(x_age, y_reentry, color=colors[i], linewidth=1.5)
                ax.plot(x_age, y_complete, color=colors[i], linewidth=1.5, linestyle='dashed')
        else:
            y = y_case[i, :age_cut]
            label_i = r'$\phi$=' + str('{0:.2f}'.format(phi_set[i]))
            ax.plot(x_age, y, color=colors[i], linewidth=1.5, label=label_i)
            ax.legend()
    ax.set_title(fig_titles[j], color='black')
    ax.tick_params(axis='y', labelcolor='black')
    # ax.set_xlim(-1, 100)
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('f3_1.png', dpi=200)
plt.show()
plt.close()

######################################
###########   Figure 4   #############
######################################
# how long before exiting upon entry &
# how long before entering upon exit
# Analysis of the bell length: Distribution of participation bells, ignoring 0

#todo: figure 8 in the paper;
# I for some reason had the wrong version of the code in the branch; now it should work

folder_address = r'E:\Users\A2010290\Documents\GitHub\NoShort/simu_results/'
n_files = int(Mpath / 25)
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

# unconditional
# y_titles = ['Years in the stock market before first exit', 'Years out of the stock market before first re-entry']
# fig, axes = plt.subplots(nrows=1, ncols=2, sharey='all', sharex='all', figsize=(10, 5))
# for i, ax in enumerate(axes):
#     for j in range(2):
#         data_mat = bell_length_mat if i == 0 else bell_length_reentry_mat
#         unique, counts = np.unique(data_mat[:, j+1, 1, :, 2:], return_counts=True)
#         counts_percentage = counts[1:-1] / np.sum(counts[1:-1])
#         # data_inter = np.arange(0, len(counts_percentage), 2)
#         x = np.arange(1, len(counts_percentage)+1)
#         # X_Y_Spline = make_interp_spline(x[data_inter], counts_percentage[data_inter], k=3)
#         # X_ = np.linspace(1, 20, 100)
#         # Y_ = X_Y_Spline(X_)
#         line_style_i = 'solid' if j == 0 else 'dashed'
#         label_i = 'Re-entry' if j == 0 else 'Mix'
#         ax.plot(x, counts_percentage, linewidth=2, linestyle=line_style_i, label=label_i, color='navy')
#         ax.legend()
#     ax.set_title(y_titles[i])
#     ax.set_xlabel('Years')
#     ax.set_ylabel('Proportion of observations')
# plt.savefig('f5_1.png', dpi=200)
# plt.show()
# plt.close()

# conditional on stock returns at the point of exit
fig, axes = plt.subplots(nrows=1, ncols=2, sharey='all', sharex='all', figsize=(10, 4.5))
for i, ax in enumerate(axes):
    cutoffs_return = np.percentile(stock_returns_mat, [10])
    title_i = 'Re-entry scenario' if i == 0 else 'Mix scenario'
    ax.set_title(title_i)
    data_mat = spell_mat[:, i]
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
        color_i = 'red' if j == 1 else 'navy'
        label_i = '1-year return bottom decile' if j == 1 else 'Unconditional'
        ax.plot(X_, Y_, linewidth=2,
                # linestyle=line_style_i,
                label=label_i, color=color_i)
        ax.legend(loc='lower right')
    ax.set_ylim(0.1, 1.0)
    ax.set_xlim(0, 20)
    ax.set_xlabel('Years since exiting')
    ax.set_ylabel('Fraction of individuals re-entering')
plt.savefig('f5_2.png', dpi=200)
plt.show()
plt.close()


######################################
#############  Table 1  ##############
######################################
Nsce = 3
Ncolumn = int(Nsce * 2)
table1_var = ['theta', 'r', 'mu_S', 'sigma_S']
Nrow = len(table1_var)
table1_mat = np.zeros((Nrow, n_files, Nsce, 2))
for i, var in enumerate(table1_var):
    for j in range(n_files):
        table1_mat[i, j] = np.average(np.load(folder_address + str(j) + "simulation_new.npz")[var], axis=0)
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
              r'$\theta_t$',
              r'$\r_t$',
              r'$\mu^S_t$',
              r'$\sigma^S_t$',
              ]
print(tab.tabulate(table_output, headers=header, showindex=show_index, floatfmt=".4f",
                   tablefmt='latex_raw'))

# Panel 1.2: correlations
Nrow = 3
table12_mat = np.zeros((n_files, Nsce, Nrow))
for j in range(n_files):
    table12_mat[j] = np.average(np.load(folder_address + str(j) + "simulation_new.npz")['cov_mat'], axis=0)[:, :3]
table12_mat_ave = np.average(table12_mat, axis=0)
# file = file_list_mean_vola[11]
# var_average = np.average(results_mean_vola[file], axis=0)  # shape (n_scenarios, 2)
# Nrow = np.shape(var_average)[1]
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

# Panel 21: participation rate, entry and exit
table21_var = ['parti', 'entry', 'exit']
Nrow = len(table21_var)
Ncolumn = 2
table21_mat = np.zeros((Nrow, n_files, Ncolumn))
for i, var in enumerate(table21_var):
    for j in range(n_files):
        table21_mat[i, j] = np.average(np.load(folder_address + str(j) + "simulation_new.npz")[var], axis=0)
table21_mat_ave = np.average(table21_mat, axis=1)
# file = file_list_mean_vola[9]
# var_average = np.average(results_mean_vola[file], axis=0)  # shape (n_scenarios, 2)
# N_row = np.shape(var_average)[1]
header = ['Re-entry', 'Mix-4']
print(tab.tabulate(table21_mat_ave, headers=header,
                   floatfmt=".4f", tablefmt='latex_raw'))
# Panel 2.2: correlations
Nrow = 2
table22_mat = np.zeros((n_files, Nsce, Nrow))
for j in range(n_files):
    table22_mat[j] = np.average(np.load(folder_address + str(j) + "simulation_new.npz")['cov_mat'], axis=0)[:, 3:]
table22_mat_ave = np.average(table22_mat, axis=0)
# file = file_list_mean_vola[11]
# var_average = np.average(results_mean_vola[file], axis=0)  # shape (n_scenarios, 2)
# Nrow = np.shape(var_average)[1]
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
    table3_mat[j] = np.average(np.load(folder_address + str(j) + "simulation_new.npz")['reg1'], axis=0)
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

# Panel 4: future asset returns on participation rate, entry and exit
table4_mat = np.zeros((n_files, Nsce - 1, 3, 3))
for j in range(n_files):
    table4_mat[j] = np.average(np.load(folder_address + str(j) + "simulation_new.npz")['reg2'], axis=0)
table4_mat_ave = np.average(table4_mat, axis=0)
table_output = np.zeros((3, 6))
table_output[:, :3] = table4_mat_ave[0]
table_output[:, 3:] = table4_mat_ave[1]
header = np.tile([
    r'$R^S_{t-1, t}$',
    r'$R^S_{t-2, t}$',
    r'$R^S_{t-3, t}$',
], 2)
show_index = [
    'P_t',
    r'$\text{Entry}_{t-1, t}$',
    r'$\text{Exit}_{t-1, t}$'
]
print(tab.tabulate(table_output, showindex=show_index, headers=header, floatfmt=".4f", tablefmt='latex_raw'))


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
plt.savefig('Reentry_age_alpha.png', dpi=200)
plt.show()
plt.close()

# regression 1: participation rate on past returns and pd
reg_table1 = np.average(results_df2['return_parti_reg'], axis=0)
reg_table3 = np.average(results_df2['parti_return_reg'], axis=0)


# # regression 2: participation rate predicts future returns
folder_address = r'E:\Users\A2010290\Documents\GitHub\NoShort/reg_results2/'
reg_results1 = np.empty((25, 1, 2, 2, 2, 3, 3))
# reg_results2 = np.empty((100, 1, 1, 3, 3))
for i in range(25):
    reg_results1[i] = np.load(folder_address + str(i) + "reg1.npy")
    # reg_results2[i] = np.load(folder_address + str(i) + "reg2.npy")
ave_reg1 = np.average(reg_results1, axis=0)



p = pstats.Stats("profile_output")
p.strip_dirs().sort_stats("cumulative").print_stats(30)
