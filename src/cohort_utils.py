import numpy as np
import statsmodels.api as sm
from typing import Tuple, Dict, Any

from numpy import ndarray, dtype, floating, signedinteger
from numpy._typing import _64Bit, _16Bit, _8Bit

from src.solver import (
    bisection,
    solve_theta,
    bisection_partial_constraint,
    solve_theta_partial_constraint,
)


def initialize_simulation_arrays(
    Nt: int,
    Ntype: int,
    Nc: int,
    dt: float,
    need_f: str,
    need_Delta: str,
    need_pi: str,
    use_constraints: bool = False,
    Nconstraint: int = 1,
) -> tuple[
    ndarray[Any, dtype[floating[_64Bit]]], ndarray[Any, dtype[floating[_64Bit]]], ndarray[Any, dtype[floating[_64Bit]]],
    ndarray[Any, dtype[floating[_64Bit]]], ndarray[Any, dtype[floating[_64Bit]]], ndarray[Any, dtype[floating[_64Bit]]],
    ndarray[Any, dtype[floating[_64Bit]]] | ndarray[Any, dtype[Any]], ndarray[Any, dtype[floating[_64Bit]]] | ndarray[
        Any, dtype[Any]], ndarray[Any, dtype[floating[_64Bit]]] | ndarray[Any, dtype[Any]], ndarray[
        Any, dtype[floating[_64Bit]]] | ndarray[Any, dtype[Any]], ndarray[Any, dtype[floating[_64Bit]]] | ndarray[
        Any, dtype[Any]], ndarray[Any, dtype[floating[_64Bit]]] | ndarray[Any, dtype[Any]], ndarray[
        Any, dtype[floating[_64Bit]]] | ndarray[Any, dtype[Any]], ndarray[Any, dtype[floating[_64Bit]]] | ndarray[
        Any, dtype[Any]], ndarray[Any, dtype[floating[_64Bit]]] | ndarray[Any, dtype[Any]], ndarray[
        Any, dtype[floating[_64Bit]]] | ndarray[Any, dtype[Any]] | int, ndarray[Any, dtype[floating[_64Bit]]] | ndarray[
        Any, dtype[Any]] | int, ndarray[Any, dtype[floating[_64Bit]]] | ndarray[Any, dtype[Any]] | int]:
    """
    Initialize the main simulation arrays.

    Args:
        Nt (int): Number of time steps.
        Ntype (int): Number of agent types.
        Nc (int): Number of cohorts.
        dt (float): Time step size.
        need_f (str): If "True", initializes the consumption share matrix.
        need_Delta (str): If "True", initializes the belief bias matrix.
        need_pi (str): If "True", initializes the portfolio matrix.
        use_constraints (bool): If True, enables constraint handling.
        Nconstraint (int): Number of constraint types (only relevant if use_constraints=True).

    Returns:
        Tuple of initialized arrays.
    """
    # Shared arrays
    # Equilibrium terms: (see Section 2.3, Eq. (12)–(14))
    dR = np.zeros(Nt)
    r = np.zeros(Nt)
    theta = np.zeros(Nt)
    mu_S = np.zeros(Nt)
    sigma_S = np.zeros(Nt)
    beta = np.zeros(Nt)
    Phi_bar_parti = np.ones(Nt, dtype=np.float16)
    Phi_tilde_parti = np.ones(Nt, dtype=np.float16)
    parti = np.ones(Nt, dtype=np.float16)
    parti_age_group = np.zeros((Nt, 4), dtype=np.float16)
    portf_age_group = np.zeros((Nt, 4), dtype=np.float16)
    parti_wealth_group = np.zeros((Nt, 4), dtype=np.float16)
    entry_mat = np.zeros((Nt, 3), dtype=np.float16)
    exit_mat = np.zeros((Nt, 3), dtype=np.float16)

    # Constraint handling
    if use_constraints:
        invest_mat = np.zeros((36, Nconstraint, Nc), dtype=np.int8)
        if need_f == "True":
            f_c = np.zeros((Nt, Ntype, Nconstraint, Nc), dtype=np.float16)
        else:
            f_c = 0

        if need_Delta == "True":
            Delta = np.zeros((Nt, Nconstraint, Nc), dtype=np.float16)
        else:
            Delta = 0

        if need_pi == "True":
            pi = np.zeros((Nt, Nconstraint, Nc), dtype=np.float16)
        else:
            pi = 0
    else:
        invest_mat = np.zeros((36, Nc), dtype=np.int8)
        if need_f == "True":
            f_c = np.zeros((Nt, Ntype, Nc), dtype=np.float16)
        else:
            f_c = 0

        if need_Delta == "True":
            Delta = np.zeros((Nt, Nc), dtype=np.float16)
        else:
            Delta = 0

        if need_pi == "True":
            pi = np.zeros((Nt, Nc), dtype=np.float16)
        else:
            pi = 0

    return (
        dR,
        r,
        theta,
        mu_S,
        sigma_S,
        beta,
        Phi_bar_parti,
        Phi_tilde_parti,
        parti,
        parti_age_group,
        portf_age_group,
        parti_wealth_group,
        entry_mat,
        exit_mat,
        invest_mat,
        f_c,
        Delta,
        pi,
    )


def compute_regression_tables(dR, r, sample, entry_mat, exit_mat, parti):
    """
    Compute regression tables 1b and 2b used in simulate_cohorts_mean_vola and simulate_mean_vola_mix_type.

    Returns:
        regression_table1_b (np.ndarray): shape (3, 3)
        regression_table2_b (np.ndarray): shape (3, 2)
    """
    Nt = len(dR)
    past_annual_return = np.zeros((3, Nt))
    future_annual_return = np.zeros((3, Nt))

    for n, gap in enumerate([12, 24, 36]):
        past_annual_return[n, gap:] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (
            gap / 12
        )
        past_annual_return[n, :gap] = np.cumsum(dR[:gap]) / (gap / 12)
        future_annual_return[n, :-gap] = (
            np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]
        ) / (gap / 12)
        future_annual_return[n, -gap:] = (np.cumsum(dR)[-gap:]) / (gap / 12)

    x_set = np.copy(past_annual_return[:, sample])
    y_set = [parti, entry_mat[sample], exit_mat[sample]]
    regression_table1_b = np.zeros((3, 3), dtype=np.float32)

    for ii in range(3):  # for 12, 24, 36 month gaps
        x = (x_set[ii] - np.average(x_set[ii])) / np.std(x_set[ii])
        for jj, y_mat in enumerate(y_set):
            if jj == 1:  # entry on high return
                y = (y_mat[:, ii] - np.average(y_mat[:, ii])) / np.std(y_mat[:, ii])
                x_condi = (x > np.percentile(x, 75)) + 0
                x_regress = sm.add_constant(x_condi)
            elif jj == 2:  # exit on low return
                y = (y_mat[:, ii] - np.average(y_mat[:, ii])) / np.std(y_mat[:, ii])
                x_condi = (x < np.percentile(x, 25)) + 0
                x_regress = sm.add_constant(x_condi)
            else:
                y = (y_mat[sample] - np.average(y_mat[sample])) / np.std(y_mat[sample])
                x_regress = sm.add_constant(x)
            model = sm.OLS(y, x_regress)
            est = model.fit()
            regression_table1_b[ii, jj] = est.params[1]

    x = np.reshape(
        (parti[sample] - np.average(parti[sample])) / np.std(parti[sample]), (-1, 1)
    )
    x_regress = sm.add_constant(x)
    y_set = [
        future_annual_return[:, sample],
        future_annual_return[:, sample] - r[sample],
    ]
    regression_table2_b = np.zeros((3, 2), dtype=np.float32)
    for ii in range(3):
        for jj, y_mat in enumerate(y_set):
            y = (y_mat[ii] - np.average(y_mat[ii])) / np.std(y_mat[ii])
            model = sm.OLS(y, x_regress)
            est = model.fit()
            regression_table2_b[ii, jj] = est.params[1]

    return regression_table1_b, regression_table2_b


def compute_reentry_exit_times(
    invest_matrix: np.ndarray, Nt: int, dt: float, window_bell: int = 20
):
    """
    Computes the reentry and exit timing for agents in the stock market based on investment matrix data.

    Args:
        invest_matrix (np.ndarray): Matrix tracking cohort investment, shape (T_cohorts, 4, Nt)
        Nt (int): Number of time periods
        dt (float): Time step size
        window_bell (int): Size of non-overlapping window to track (default is 20 years)

    Returns:
        reentry_time (np.ndarray): Binary matrix marking reentry events, shape (n_windows, 4, T_valid)
        exit_time (np.ndarray): Binary matrix marking exit events, shape (n_windows, 4, T_valid)
    """
    sample_bell = np.arange(0, np.shape(invest_matrix)[0], window_bell)
    reentry_time = np.zeros(
        (len(sample_bell) - 1, 4, Nt - int(window_bell / dt) - 12), dtype=int
    )
    exit_time = np.zeros(
        (len(sample_bell) - 1, 4, Nt - int(window_bell / dt)), dtype=int
    )

    for n, entry_n in enumerate(sample_bell[1:]):
        following_cohorts_entry = (
            invest_matrix[entry_n, :, :-12] - invest_matrix[entry_n - 1, :, 12:] > 0
        )[:, int(window_bell / dt) :]
        following_cohorts_entry = np.append(
            following_cohorts_entry, invest_matrix[entry_n, :, -12:], axis=1
        )
        parti_bell_entry = np.zeros((window_bell, 4, Nt - int(window_bell / dt)))
        parti_bell_entry[0] = following_cohorts_entry
        exit_bell = np.zeros((4, Nt - int(window_bell / dt)))

        following_cohorts_exit = (
            invest_matrix[entry_n, :, :-12] - invest_matrix[entry_n - 1, :, 12:] < 0
        )[:, int(window_bell / dt) :]
        parti_bell_exit = np.zeros((window_bell, 4, Nt - int(window_bell / dt) - 12))
        parti_bell_exit[0] = following_cohorts_exit
        reentry_bell = np.zeros((4, Nt - int(window_bell / dt) - 12))

        for nn in range(1, window_bell):
            cohorts_in = invest_matrix[
                entry_n + nn, :, int((window_bell - nn) / dt) : int(-nn / dt)
            ]
            cohorts_out = (1 - cohorts_in)[:, :-12]

            if nn != 1:
                exit_nn = (
                    invest_matrix[
                        entry_n + nn, :, int((window_bell - nn) / dt) : int(-nn / dt)
                    ]
                    - invest_matrix[
                        entry_n + nn - 1,
                        :,
                        int((window_bell - nn + 1) / dt) : int((-nn + 1) / dt),
                    ]
                ) < 0

                reentry_nn = (
                    invest_matrix[
                        entry_n + nn, :, int((window_bell - nn) / dt) : int(-nn / dt)
                    ]
                    - invest_matrix[
                        entry_n + nn - 1,
                        :,
                        int((window_bell - nn + 1) / dt) : int((-nn + 1) / dt),
                    ]
                ) > 0
            else:
                exit_nn = (
                    invest_matrix[
                        entry_n + nn, :, int(window_bell / dt - nn / dt) : int(-nn / dt)
                    ]
                    - invest_matrix[
                        entry_n + nn - 1, :, int((window_bell - nn + 1) / dt) :
                    ]
                ) < 0

                reentry_nn = (
                    invest_matrix[
                        entry_n + nn, :, int(window_bell / dt - nn / dt) : int(-nn / dt)
                    ]
                    - invest_matrix[
                        entry_n + nn - 1, :, int((window_bell - nn + 1) / dt) :
                    ]
                ) > 0

            exit_bell = exit_bell + exit_nn > 0
            reentry_bell = reentry_bell + reentry_nn[:, :-12] > 0
            parti_bell_entry[nn] = (
                cohorts_in * following_cohorts_entry * (1 - exit_bell)
            )
            parti_bell_exit[nn] = (
                cohorts_out * following_cohorts_exit * (1 - reentry_bell)
            )

        for m in range(4):
            reentry_time[n, m] = list(map(int, np.sum(parti_bell_exit[:, m], axis=0)))
            exit_time[n, m] = list(map(int, np.sum(parti_bell_entry[:, m], axis=0)))

    return reentry_time, exit_time


def update_wealth_and_beliefs(
    eta_st_eta_ss: np.ndarray,
    X: np.ndarray,
    d_eta_st: np.ndarray,
    dZ_t: float,
    dt: float,
    rho_cohort_type: np.ndarray,
    tax: float,
    beta0: float,
):
    """
    Updates eta_st_eta_ss and X based on signal shock and wealth tax adjustment.
    """

    # Update beliefs: Eq (15)
    eta_st_eta_ss = eta_st_eta_ss * np.exp((-0.5 * d_eta_st**2) * dt + d_eta_st * dZ_t)

    # Update wealth X: Eq (20)
    X_parts = tax * X * eta_st_eta_ss * rho_cohort_type * dt
    X_t = np.sum(X_parts) / (1 - tax * beta0 * dt)

    # Append newborns
    if eta_st_eta_ss.ndim == 2:
        # 2D case
        eta_st_eta_ss = np.append(
            eta_st_eta_ss[:, 1:], np.ones((eta_st_eta_ss.shape[0], 1)), axis=1
        )
        X_newborn = X_t * np.ones((X.shape[0], 1))
        X_new = np.append(X[:, 1:], X_newborn, axis=1)
    elif eta_st_eta_ss.ndim == 3:
        # 3D case
        eta_st_eta_ss = np.append(
            eta_st_eta_ss[:, :, 1:],
            np.ones((eta_st_eta_ss.shape[0], eta_st_eta_ss.shape[1], 1)),
            axis=2,
        )
        X_newborn = X_t * np.ones((X.shape[0], X.shape[1], 1))
        X_new = np.append(X[:, :, 1:], X_newborn, axis=2)
    else:
        raise ValueError("eta_st_eta_ss must be 2D or 3D array.")

    X_new = X_new / X_t  # normalize

    return X_parts, eta_st_eta_ss, X_new, X_t


def update_Delta_s_t(
    Delta_s_t: np.ndarray,
    dDelta_s_t: np.ndarray,
    init_bias: float,
):
    """
    Update Delta_s_t array with new belief innovations and initial bias.

    Args:
        Delta_s_t (np.ndarray): Current Delta_s_t
        dDelta_s_t (np.ndarray): Innovations
        init_bias (float): Initialization bias for newborns

    Returns:
        np.ndarray: Updated Delta_s_t
    """
    if Delta_s_t.ndim == 2:
        Delta_s_t = Delta_s_t[:, 1:] + dDelta_s_t[:, 1:]
        Delta_s_t = np.append(
            Delta_s_t, init_bias * np.ones((Delta_s_t.shape[0], 1)), axis=1
        )
    elif Delta_s_t.ndim == 3:
        Delta_s_t = Delta_s_t[:, :, 1:] + dDelta_s_t[:, :, 1:]
        Delta_s_t = np.append(
            Delta_s_t,
            init_bias * np.ones((Delta_s_t.shape[0], Delta_s_t.shape[1], 1)),
            axis=2,
        )
    else:
        raise ValueError("Delta_s_t must be 2D or 3D array.")

    return Delta_s_t


def find_market_clearing_theta_complete(
    possible_cons_share: np.ndarray,
    possible_delta_st: np.ndarray,
    sigma_Y: float,
) -> float:
    """
    Find market-clearing theta for complete or simple constrained markets.
    Calls solve_theta (no shorting constraint).
    """

    lowest_bound = -np.max(possible_delta_st)
    theta_t = bisection(
        solve_theta,
        lowest_bound,
        50,
        possible_cons_share,
        possible_delta_st,
        sigma_Y,
    )
    return theta_t


def find_market_clearing_theta_partial(
    possible_cons_share: np.ndarray,
    possible_delta_st: np.ndarray,
    can_short_tracker: np.ndarray,
    sigma_Y: float,
) -> float:
    """
    Find market-clearing theta for partial constraint markets (rich/old).
    Calls solve_theta_partial_constraint (with shorting constraint tracking).
    """

    lowest_bound = -np.max(possible_delta_st)
    theta_t = bisection_partial_constraint(
        solve_theta_partial_constraint,
        lowest_bound,
        50,
        possible_cons_share,
        possible_delta_st,
        can_short_tracker,
        sigma_Y,
    )
    return theta_t
