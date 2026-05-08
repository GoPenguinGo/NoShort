import numpy as np
from typing import Tuple, Any
from numpy import ndarray, dtype, float64
from pandas import DataFrame
from src.param_mix import cohort_type_size_mix, alpha_i_mix
from src.stats import post_var, dDelta_st_calculator
from src.solver import bisection_partial_constraint, solve_theta_partial_constraint
from tqdm import tqdm
import statsmodels.api as sm
import pandas as pd



def simulate_cohorts_mix_type(
        biasvec: np.ndarray,
        dZ: np.ndarray,
        Nt: int,
        Nc: int,
        dt: float,
        Ntype: int,
        Nconstraint: int,
        rho_i: np.ndarray,
        alpha_i: np.ndarray,
        beta_i: np.ndarray,
        rho_cohort_type: np.ndarray,
        beta0: float,
        nu: float,
        Vhat: float,
        mu_Y: float,
        sigma_Y: float,
        tax: float,
        phi: float,
        T_hat: float,
        Npre: float,
        entry_bound: float,
        exit_bound: float,
        cohort_type_size: np.ndarray,
        cutoffs_age: np.ndarray,
        Delta_s_t: np.ndarray,
        eta_st_eta_ss: np.ndarray,
        X: np.ndarray,
        d_eta_st: np.ndarray,
        invest_tracker: np.ndarray,
        information_tracker: np.ndarray,
        can_short_tracker: np.ndarray,
        tau_info: np.ndarray,
        Vhat_vector: np.ndarray,
        need_f: str,
        need_Delta: str,
        need_pi: str,
        mode_learn: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    # np.ndarray,
]:
    """ Simulate the economy forward
        a mixture of 4 different types of agents in each cohort:
        unconstrained (designated participants);
        excluded (designated nonparticipants);
        disappointment (leave the stock market for good upon binding constraint);
        & reentry.

    Args:
        biasvec (np.ndarray): shocks to the output in the build stage for the cohorts born between s=[0, Npre]
        dZ (np.ndarray): shocks to the output for each period, shape (Nt, )
        dZ_SI(np.ndarray): shocks to the signal for each period, shape (Nt, )
        Nt (int): number of periods in the simulation
        Nc (int): number of cohorts in the economy
        tau (np.ndarray): t-s, shape(Nt)
        dt (float): per unit of time
        Ntype (int): number of types for time preference
        rho_i (np.ndarray): time preference of each type
        alpha_i (np.ndarray): density of each type
        beta_i (np.ndarray): consumption wealth ratio of each type
        rho_cohort_type (np.ndarray): alpha_i * exp(-(rho_i + nu)*(t-s))
        beta0 (float): average consumption wealth ratio of a new born cohort
        nu (float): rate of birth and death
        Vhat (float): initial variance
        mu_Y (float): as in eq(1), drift of aggregate output growth
        sigma_Y (float): as in eq(1), diffusion of aggregate output growth
        tax (float): marginal rate of wealth tax
        phi (float): correlation between the signal and the output growth rate
        T_hat (float): pre-trading years
        Npre (float): pre-trading number of observations
        cohort_type_size (np.ndarray): cohort_size * alpha_i
        cutoffs_age (np.ndarray): cutoff ages for the age quartiles
        -*-*-*- from the cohort_builder function -*-*-*-
        Delta_s_t (np.ndarray): estimation bias, shape(Ntype, Nc, )
        eta_st_eta_ss(np.ndarray): shape(Ntype, Nc, )
        X(np.ndarray):W_s * Xi_s, shape(Ntype, Nc, )
        d_eta_st (np.ndarray): max(delta_st, -theta_t), shape(Ntype, Nc, )
        invest_tracker (np.ndarray): track if a cohort is still in the stock market, shape(Ntype, Nc, )
        tau_info (np.ndarray): t-t', time since the last time a cohort switch, shape(Ntype, Nc, )
        Vhat_vector (np.ndarray): variance at t', shape(Ntype, Nc, )
        -*-*-*- avoid saving results unnecessary for particular analysis -*-*-*-
        need_f (str): {'True', 'False'}
        need_Delta (str): {'True', 'False'}
        need_pi (str): {'True', 'False'}

    Returns:
        r (np.ndarray): interest rate, shape(Nt, )
        theta (np.ndarray): market price of risk, shape(Nt, )
        f_c (np.ndarray): consumption share, shape(Nt, Ntype, Nc)
        Delta (np.ndarray): standardized estimation error, shape(Nt, Ntype, Nc)
        pi (np.ndarray): portfolio, shape(Nt, Ntype, Nc)
        parti (np.ndarray): participation rate, shape(Nt)
        Phi_bar_parti (np.ndarray): consumption share of participants, shape(Nt)
        Phi_tilde_parti (np.ndarray): wealth share of participants, shape(Nt)
        Delta_bar_parti (np.ndarray): consumption weighted average estimation error of participants, shape(Nt)
        Delta_tilde_parti (np.ndarray): wealth weighted average estimation error of participants, shape(Nt)
        dR (np.ndarray): realized stock returns, shape(Nt)
        mu_S (np.ndarray): expected equity risk premia, shape(Nt)
        sigma_S (np.ndarray): conditional volatility of stock returns, shape(Nt)
        beta (np.ndarray): aggregate consumption wealth ratio, shape(Nt)
        invest_mat (np.ndarray): whether a cohort is in the stock market, shape(Nt, Nc)
        parti_age_group: participation rate in age groups, shape(Nt, 4)
        parti_wealth_group: participation rate in wealth groups, shape(Nt, 4)

    """
    # Initializing variables
    invest_newborn = np.array([[[1], [0], [1]]]) * np.ones((Ntype, Nconstraint, 1), dtype=np.int8)
    # top = np.array([1, 0.75, 0.5, 0.25, 0])

    Phi_bar_parti = np.ones(Nt, dtype=np.float16)  # consumption share of the stock market participants
    Phi_tilde_parti = np.ones(Nt, dtype=np.float16)

    # equilibrium terms:
    dR = np.zeros(Nt)  # stores stock returns
    r = np.zeros(Nt)  # interest rate
    theta = np.zeros(Nt)  # market price of risk
    mu_S = np.zeros(Nt)
    sigma_S = np.zeros(Nt)
    beta = np.zeros(Nt)
    Delta_popu = np.zeros((Nt, 2))  # participants and nonparticipants, population average
    Delta_bar_parti = np.zeros(Nt,
                               dtype=np.float16)  # consumption weighted estimation error of the stock market participants
    Delta_tilde_parti = np.zeros(Nt,
                                 dtype=np.float16)  # wealth weighted estimation error of the stock market participants
    parti = np.ones(Nt, dtype=np.float16)  # participation rate
    entry_mat = np.ones((Nt, 3), dtype=np.float16)
    exit_mat = np.ones((Nt, 3), dtype=np.float16)
    # parti_wealth_group = np.zeros((Nt, 4), dtype=np.float16)
    parti_age_group = np.zeros((Nt, len(cutoffs_age) - 1), dtype=np.float16)
    sigma_Y_sq = sigma_Y ** 2

    append_init = np.ones((Ntype, Nconstraint, 1), dtype=np.int8)
    # wealth_cutoffs = np.array([0, 1, 10, 100, 100000])

    if need_f == 'True':
        f_c = np.zeros((Nt, Ntype, Nconstraint, Nc), dtype=np.float16)  # evolution of cohort consumption share
    else:
        f_c = 0

    if need_Delta == 'True':
        Delta = np.zeros((Nt, Nconstraint, Nc), dtype=np.float16)  # stores bias in beliefs
    else:
        Delta = 0

    if need_pi == 'True':
        pi = np.zeros((Nt, Nconstraint, Nc), dtype=np.float16)  # portfolio choices
    else:
        pi = 0
    invest_mat = np.zeros((36, Nconstraint, Nc), dtype=np.int8)

    for i in tqdm(range(Nt)):
        dZ_t = dZ[i]

        # new cohort born (age 0), get wealth transfer, observe, invest
        eta_st_eta_ss = eta_st_eta_ss * np.exp(
            (-0.5 * d_eta_st ** 2) * dt
            + d_eta_st * dZ_t
        )  # equation (15)

        X_parts = tax * X * eta_st_eta_ss * rho_cohort_type * dt
        X_t = np.sum(X_parts) / (1 - tax * beta0 * dt)

        eta_st_eta_ss = np.append(eta_st_eta_ss[:, :, 1:], append_init, axis=2)
        X = np.append(X[:, :, 1:], X_t * np.ones((1, 1, 1)), axis=2)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort
        # eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
        #  eta_bar_t is the denominator; it creates issues if too close to 0
        #  so we rescale eta_bar to keep it away from 0, without changing f_st

        f_c_ist = X_parts / X_t / dt
        f_c_ist = np.append(f_c_ist[:, :, 1:], tax * alpha_i * beta_i, axis=2)

        beta_t = 1 / np.sum(f_c_ist / beta_i * dt)
        f_w_ist = f_c_ist / beta_i * beta_t
        if i > 0:
            dR_t = mu_S_t * dt + sigma_S_t * dZ_t
        else:
            dR_t = 0

        # update beliefs
        V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, phi, 'N')  # from eq(6)
        dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, phi, dt, V_st_N, Delta_s_t, dZ_t,
                                            'N')  # from eq(9)
        V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, phi, 'P')
        dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, phi, dt, V_st_P, Delta_s_t, dZ_t, 'P')
        dDelta_s_t = information_tracker * dDelta_s_t_P + (
                1 - information_tracker) * dDelta_s_t_N  # the participation decision of last time affects the updating pattern
        tau_info = np.append(tau_info[:, :, 1:], 0 * append_init, axis=2) + dt
        Vhat_vector = np.append(Vhat_vector[:, :, 1:], Vhat * append_init, axis=2)
        Vhat_vector[:, 0] = 0.0

        if i < Npre - 1:
            init_bias = (np.sum(biasvec[i + 1:]) + np.sum(dZ[:i + 1])) / T_hat * append_init
        else:
            init_bias = np.sum(dZ[i + 1 - Npre:i + 1]) / T_hat * append_init
        init_bias[:, 0] = 0.0

        Delta_s_t = Delta_s_t[:, :, 1:] + dDelta_s_t[:, :, 1:]
        Delta_s_t = np.append(Delta_s_t, init_bias, axis=2)

        # find the market clearing theta, given beliefs and consumption shares of cohorts in the economy
        invest_tracker = np.append(invest_tracker[:, :, 1:], invest_newborn,
                                   axis=2)  # all cohorts that are still in the market, 1 by default
        information_tracker = np.append(information_tracker[:, :, 1:], invest_newborn,
                                   axis=2)

        possible_cons_share = f_c_ist * dt * invest_tracker
        possible_cons_share[:, 2] = f_c_ist[:, 2] * dt
        possible_delta_st = Delta_s_t * invest_tracker
        possible_delta_st[:, 2] = Delta_s_t[:, 2]

        lowest_bound = -np.max(
            possible_delta_st[np.nonzero(possible_delta_st)]).astype(float)  # absolute lower bound where no agent holds the stock
        theta_t = bisection_partial_constraint(
            solve_theta_partial_constraint,
            lowest_bound,
            50,
            invest_tracker,
            can_short_tracker,
            possible_delta_st,
            possible_cons_share,
            sigma_Y,
            entry_bound,
            exit_bound,
        )

        theta_st = Delta_s_t + theta_t
        invest = (
                         theta_st >= exit_bound
                 ) * invest_tracker + (
                         theta_st >= entry_bound
                 ) * (1 - invest_tracker)

        invest = 1 - (invest != 1)  # not invest if a<0 and can not short
        invest[:, 1] = 0  # exclusion type
        invest[:, 0] = 1  # complete type
        # switch_P_to_N = invest_tracker * (1 - invest) * (
        #         can_short_tracker < 1)  # switch to nonparti if type R&E & not investing this period
        # switch_N_to_P = np.maximum(invest - invest_tracker,
        #                            0)  # switch to parti if not investing before & investing this period
        # switch_N_to_P[:, :2] = 0  # only applicable to the E type
        # switch = switch_N_to_P + switch_P_to_N
        # invest_tracker = invest_tracker + switch_N_to_P - switch_P_to_N
        invest_tracker = np.copy(invest)
        d_eta_st = (Delta_s_t + theta_t) * invest_tracker - theta_t

        # tau_info and V_hat has to change for the agents who switch (either P to N or vice versa)
        # the switches are specific to passing the exit_boundary
        # information = theta_st[:, 2] >= exit_bound
        if mode_learn == 'theta':
            information = theta_st[:, 2] >= exit_bound
        elif mode_learn == 'invest':
            information = invest[:, -1]
        else:
            print("mode learn not found")
            exit()
        switch_P_to_N = information_tracker * 0.0
        switch_N_to_P = information_tracker * 0.0
        switch_P_to_N[:, -1] = (information_tracker[:, -1] - information == 1)  # switch to nonparti if type R&E & not investing this period
        switch_N_to_P[:, -1] = (information_tracker[:, -1] - information == -1)  # only applicable to the E type
        switch = switch_N_to_P + switch_P_to_N
        information_tracker[:, -1] = np.copy(information)

        Vhat_vector = np.append(V_st_P[:, :, 1:], Vhat * append_init, axis=2) * switch_P_to_N + \
                      np.append(V_st_N[:, :, 1:], Vhat * append_init, axis=2) * switch_N_to_P + \
                      Vhat_vector * (1 - switch)  # reset V'
        tau_info = dt * switch + tau_info * (1 - switch)  # reset t'

        invest_fc_st = invest_tracker * f_c_ist * dt
        invest_fw_st = invest_tracker * f_w_ist * dt
        popu_parti_t = np.sum(cohort_type_size * invest_tracker)
        fc_parti_t = np.sum(invest_fc_st)
        fw_parti_t = np.sum(invest_fw_st)
        Delta_bar_parti_t = np.sum(Delta_s_t * invest_fc_st) / fc_parti_t
        Delta_tilde_parti_t = np.sum(Delta_s_t * invest_fw_st) / fw_parti_t
        sigma_S_t = fw_parti_t * (theta_t + Delta_tilde_parti_t)
        pi_st = (d_eta_st + theta_t) / sigma_S_t
        rho_bar_t = np.sum(rho_i * f_c_ist) / np.sum(f_c_ist)

        r_t = (
                nu - tax * beta0
                + rho_bar_t
                + mu_Y
                - sigma_Y * theta_t
        )

        mu_S_t = sigma_S_t * theta_t + r_t

        # store the results
        dR[i] = dR_t  # realized return from t-1 to t
        theta[i] = theta_t
        r[i] = r_t
        Delta_bar_parti[i] = Delta_bar_parti_t
        Delta_tilde_parti[i] = Delta_tilde_parti_t
        mu_S[i] = mu_S_t
        sigma_S[i] = np.abs(sigma_S_t)  # stock vola = absolute value of sigma
        beta[i] = beta_t
        if need_f == 'True':
            f_c[i, :] = f_c_ist
        if need_Delta == 'True':
            Delta[i, :] = Delta_s_t[0]
        if need_pi == 'True':
            pi[i, :] = pi_st[0]

        Phi_bar_parti[i] = fc_parti_t
        Phi_tilde_parti[i] = fw_parti_t
        parti[i] = popu_parti_t

        for j in range(3):
            entry_i = np.copy(invest_tracker[0])
            entry_i[:, :-12 * (j + 1)] = invest_tracker[0, :, :-12 * (j + 1)] > invest_mat[-12 * (j + 1), :, 12 * (
                    j + 1):]  # entry including the newborns
            # entry_i = invest_tracker[0, :, :-12 * (j + 1)] > invest_mat[-12 * (j + 1), :, 12 * (
            #         j + 1):]  # entry excluding the newborns
            exit_i = invest_tracker[0, :, :-12 * (j + 1)] < invest_mat[-12 * (j + 1), :, 12 * (j + 1):]
            entry_mat[i, j] = np.average(entry_i, weights=np.sum(cohort_type_size_mix, axis=0))
            exit_mat[i, j] = (np.average(exit_i, weights=np.sum(cohort_type_size_mix[:, :, :-12 * (j + 1)], axis=0)) 
                                     + np.average(invest_mat[-12 * (j + 1), :, 12 * (j + 1):], weights=np.sum(cohort_type_size_mix[:, :, :-12 * (j + 1)], axis=0)) * nu * j
                                     )
        invest_mat = np.copy(np.append(invest_mat[1:], np.reshape(invest_tracker[0], (1, Nconstraint, -1)), axis=0))

        for j in range(len(cutoffs_age) - 1):
            parti_age_group[i, j] = np.ma.average(
                invest_tracker[:, :, cutoffs_age[j + 1]:cutoffs_age[j]],
                weights=cohort_type_size[:, :, cutoffs_age[j + 1]:cutoffs_age[j]])
        # for j in range(len(wealth_cutoffs) - 1):
        #     within_group = np.where(
        #         (invest_fw_st >= wealth_cutoffs[j] * cohort_type_size / dt)
        #         * (f_w_ist < wealth_cutoffs[j + 1] * cohort_type_size / dt)
        #     )
        #     parti_wealth_group[i, j] = np.ma.average(invest_tracker[within_group],
        #                                              weights=cohort_type_size[within_group])


    return (
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
        # invest_mat,
        parti_age_group,
        entry_mat,
        exit_mat
    )


def simulate_mean_vola_mix_type(
        biasvec: np.ndarray,
        dZ: np.ndarray,
        Nt: int,
        dt: float,
        Ntype: int,
        Nconstraint: int,
        rho_i: np.ndarray,
        alpha_i: np.ndarray,
        beta_i: np.ndarray,
        rho_cohort_type: np.ndarray,
        beta0: float,
        nu: float,
        Vhat: float,
        mu_Y: float,
        sigma_Y: float,
        tax: float,
        phi: float,
        T_hat: float,
        Npre: float,
        entry_bound: float,
        exit_bound: float,
        cohort_type_size: np.ndarray,
        window_bell: int,
        Delta_s_t: np.ndarray,
        eta_st_eta_ss: np.ndarray,
        X: np.ndarray,
        d_eta_st: np.ndarray,
        invest_tracker: np.ndarray,
        information_tracker: np.ndarray,
        can_short_tracker: np.ndarray,
        tau_info: np.ndarray,
        Vhat_vector: np.ndarray,
        need_invest_matrix: str
) -> tuple[
    DataFrame, DataFrame, DataFrame, ndarray[Any, dtype[Any]] | int, ndarray[Any, dtype[float64]] | ndarray[Any, Any] |
                                     ndarray[Any, dtype[Any]] | ndarray[tuple[int, ...], dtype[float64]] | ndarray[
                                         tuple[int, ...], dtype[Any]], ndarray[Any, dtype[float64]] | ndarray[
                                         Any, Any] | ndarray[Any, dtype[Any]] | ndarray[
                                         tuple[int, ...], dtype[float64]] | ndarray[tuple[int, ...], dtype[Any]]]:
    """
    Simulate the economy forward, saving only the mean & std of the results
        a mixture of 4 different types of agents in each cohort:
        unconstrained (designated participants);
        excluded (designated nonparticipants);
        disappointment (leave the stock market for good upon binding constraint);
        & reentry.

    Args:
        biasvec (np.ndarray): shocks to the output in the build stage for the cohorts born between s=[0, Npre]
        dZ (np.ndarray): shocks to the output for each period, shape (Nt, )
        dZ_SI(np.ndarray): shocks to the signal for each period, shape (Nt, )
        Nt (int): number of periods in the simulation
        tau (np.ndarray): t-s, shape(Nt)
        dt (float): per unit of time
        Ntype (int): number of types for time preference
        rho_i (np.ndarray): time preference of each type
        alpha_i (np.ndarray): density of each type
        beta_i (np.ndarray): consumption wealth ratio of each type
        rho_cohort_type (np.ndarray): alpha_i * exp(-(rho_i + nu)*(t-s))
        beta0 (float): average consumption wealth ratio of a new born cohort
        nu (float): rate of birth and death
        Vhat (float): initial variance
        mu_Y (float): as in eq(1), drift of aggregate output growth
        sigma_Y (float): as in eq(1), diffusion of aggregate output growth
        tax (float): marginal rate of wealth tax
        phi (float): correlation between the signal and the output growth rate
        T_hat (float): pre-trading years
        Npre (float): pre-trading number of observations
        mode_trade (str): {'complete', 'w_constraint', 'partial_constraint_old'}
        mode_learn (str): {'reentry', 'disappointment'}
        cohort_type_size (np.ndarray): cohort_size * alpha_i
        cutoffs_age (np.ndarray): cutoff ages for the age quartiles
        -*-*-*- from the cohort_builder function -*-*-*-
        Delta_s_t (np.ndarray): estimation bias, shape(Ntype, Nc, )
        eta_st_eta_ss(np.ndarray): shape(Ntype, Nc, )
        X(np.ndarray):W_s * Xi_s, shape(Ntype, Nc, )
        d_eta_st (np.ndarray): max(delta_st, -theta_t), shape(Ntype, Nc, )
        invest_tracker (np.ndarray): track if a cohort is still in the stock market, shape(Ntype, Nc, )
        tau_info (np.ndarray): t-t', time since the last time a cohort switch, shape(Ntype, Nc, )
        Vhat_vector (np.ndarray): variance at t', shape(Ntype, Nc, )

    Returns:
        -*-*-*- time-series mean and standard deviation of the quantities -*-*-*-
        dR_matrix (np.ndarray): shape(2)
        theta_matrix (np.ndarray): shape(2)
        r_matrix (np.ndarray): shape(2)
        mu_S_matrix (np.ndarray): shape(2)
        sigma_S_matrix (np.ndarray): shape(2)
        beta_matrix (np.ndarray): shape(2)
        -*-*-*- time-series mean and standard deviation of the state variables determining the quantities -*-*-*-
        theta_save_matrix (np.ndarray): shape(2, 2)
        sigma_S_save_matrix (np.ndarray): shape(4, 2)
        parti_age_group_matrix (np.ndarray): shape(4), mean only
        parti_wealth_group_matrix (np.ndarray): shape(4), mean only
        cov_save_matrix (np.ndarray): shape(6, 2)
        parti (np.ndarray): shape(4), mean only
        cov_parti_matrix (np.ndarray): shape(4), mean only

    """
    # Initializing variables
    keep_when = int(200 / dt)
    sigma_Y_sq = sigma_Y ** 2
    mu_S_t = 0
    sigma_S_t = 0
    window = 12  # 1-year non-overlapping windows
    sample = np.arange(600, Nt - keep_when - 600, window)
    age_sample = np.arange(1, int(200 / dt), 12)

    dR = np.zeros(Nt - keep_when)  # stores stock returns
    r = np.zeros(Nt - keep_when)  # interest rate
    theta = np.zeros(Nt - keep_when)  # market price of risk
    mu_S = np.zeros(Nt - keep_when)
    sigma_S = np.zeros(Nt - keep_when)
    beta = np.zeros(Nt - keep_when)
    # Delta_popu = np.zeros((Nt - keep_when, 2))  # participants and nonparticipants, population average
    Delta_bar_parti = np.zeros(
        (Nt - keep_when))  # consumption weighted estimation error of the stock market participants
    Delta_tilde_parti = np.zeros((Nt - keep_when))  # wealth weighted estimation error of the stock market participants
    parti = np.ones((Nt - keep_when))  # participation rate
    Phi_bar_parti_1 = np.ones((Nt - keep_when))
    Phi_tilde_parti = np.ones((Nt - keep_when))

    # parti_age_group = np.ones((Nt - keep_when, 4))
    # N_wealth_group = 4
    # wealth_cutoffs = np.array([0, 1, 10, 100, 100000])
    # parti_wealth_group = np.ones((Nt - keep_when, N_wealth_group))

    entry_mat = np.ones((Nt - keep_when, Nconstraint))
    exit_mat = np.ones((Nt - keep_when, Nconstraint))
    invest_mat = np.ones((36, Nconstraint, Nt), dtype=np.int8)
    # Delta_matrix = np.empty((int((Nt - keep_when) / 12), len(age_sample)), dtype=np.float16)
    invest_matrix = np.ones((int((Nt - keep_when) / 12), Nconstraint, Nt), dtype=np.int8)

    append_init = np.ones((Ntype, Nconstraint, 1))
    invest_newborn = np.array([[[1], [0], [1]]]) * np.ones((Ntype, Nconstraint, 1))

    for i in tqdm(range(Nt)):
        dZ_t = dZ[i]

        # new cohort born (age 0), get wealth transfer, observe, invest
        eta_st_eta_ss = eta_st_eta_ss * np.exp(
            (-0.5 * d_eta_st ** 2) * dt
            + d_eta_st * dZ_t
        )  # equation (15)

        X_parts = tax * X * eta_st_eta_ss * rho_cohort_type * dt
        X_t = np.sum(X_parts) / (1 - tax * beta0 * dt)

        eta_st_eta_ss = np.append(eta_st_eta_ss[:, :, 1:], append_init, axis=2)
        X = np.append(X[:, :, 1:], X_t * np.ones((1, 1, 1)), axis=2)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort
        # eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
        #  eta_bar_t is the denominator; it creates issues if too close to 0
        #  so we rescale eta_bar to keep it away from 0, without changing f_st

        f_c_ist = X_parts / X_t / dt
        f_c_ist = np.append(f_c_ist[:, :, 1:], tax * alpha_i * beta_i, axis=2)

        beta_t = 1 / np.sum(f_c_ist / beta_i * dt)
        f_w_ist = f_c_ist / beta_i * beta_t
        if i > 0:
            dR_t = mu_S_t * dt + sigma_S_t * dZ_t
        else:
            dR_t = 0

        # update beliefs
        V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, phi, 'N')  # from eq(6)
        dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, phi, dt, V_st_N, Delta_s_t, dZ_t,
                                            'N')  # from eq(9)
        V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, phi, 'P')
        dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, phi, dt, V_st_P, Delta_s_t, dZ_t, 'P')
        dDelta_s_t = information_tracker * dDelta_s_t_P + (
                1 - information_tracker) * dDelta_s_t_N  # the participation decision of last time affects the updating pattern
        tau_info = np.append(tau_info[:, :, 1:], 0 * append_init, axis=2) + dt
        Vhat_vector = np.append(Vhat_vector[:, :, 1:], Vhat * append_init, axis=2)
        Vhat_vector[:, 0] = 0.0

        if i < Npre - 1:
            init_bias = (np.sum(biasvec[i + 1:]) + np.sum(dZ[:i + 1])) / T_hat * append_init
        else:
            init_bias = np.sum(dZ[i + 1 - Npre:i + 1]) / T_hat * append_init
        init_bias[:, 0] = 0.0

        Delta_s_t = Delta_s_t[:, :, 1:] + dDelta_s_t[:, :, 1:]
        Delta_s_t = np.append(Delta_s_t, init_bias, axis=2)

        # find the market clearing theta, given beliefs and consumption shares of cohorts in the economy
        invest_tracker = np.append(invest_tracker[:, :, 1:], invest_newborn,
                                   axis=2)  # all cohorts that are still in the market, 1 by default
        information_tracker = np.append(information_tracker[:, :, 1:], invest_newborn,
                                   axis=2)

        possible_cons_share = f_c_ist * dt * invest_tracker
        possible_cons_share[:, -1] = f_c_ist[:, -1] * dt
        possible_delta_st = Delta_s_t * invest_tracker
        possible_delta_st[:, -1] = Delta_s_t[:, -1]

        lowest_bound = -np.max(
            possible_delta_st[np.nonzero(possible_delta_st)]).astype(float)  # absolute lower bound where no agent holds the stock
        theta_t = bisection_partial_constraint(
            solve_theta_partial_constraint,
            lowest_bound,
            50,
            can_short_tracker,
            invest_tracker,
            possible_delta_st,
            possible_cons_share,
            sigma_Y,
            entry_bound,
            exit_bound,
        )

        theta_st = Delta_s_t + theta_t
        invest = (
                         theta_st >= exit_bound
                 ) * invest_tracker + (
                         theta_st >= entry_bound
                 ) * (1 - invest_tracker)

        invest = 1 - (invest != 1)  # not invest if a<0 and can not short
        invest[:, 1] = 0  # exclusion type
        invest[:, 0] = 1  # complete type
        invest_tracker = np.copy(invest)
        d_eta_st = (Delta_s_t + theta_t) * invest_tracker - theta_t

        # tau_info and V_hat has to change for the agents who switch (either P to N or vice versa)
        # the switches are specific to passing the exit_boundary
        # information = theta_st[:, 2] >= exit_bound
        information = invest[:, -1]
        switch_P_to_N = information_tracker * 0.0
        switch_N_to_P = information_tracker * 0.0
        switch_P_to_N[:, -1] = (information_tracker[:, -1] - information == 1)  # switch to nonparti if type R&E & not investing this period
        switch_N_to_P[:, -1] = (information_tracker[:, -1] - information == -1)  # only applicable to the E type
        switch = switch_N_to_P + switch_P_to_N
        information_tracker[:, -1] = np.copy(information)

        Vhat_vector = np.append(V_st_P[:, :, 1:], Vhat * append_init, axis=2) * switch_P_to_N + \
                      np.append(V_st_N[:, :, 1:], Vhat * append_init, axis=2) * switch_N_to_P + \
                      Vhat_vector * (1 - switch)  # reset V'
        tau_info = dt * switch + tau_info * (1 - switch)  # reset t'

        invest_fc_st = invest_tracker * f_c_ist * dt
        invest_fw_st = invest_tracker * f_w_ist * dt
        popu_parti_t = np.sum(cohort_type_size * invest_tracker)
        fc_parti_t = np.sum(invest_fc_st)
        fw_parti_t = np.sum(invest_fw_st)
        Delta_bar_parti_t = np.sum(Delta_s_t * invest_fc_st) / fc_parti_t
        Delta_tilde_parti_t = np.sum(Delta_s_t * invest_fw_st) / fw_parti_t
        sigma_S_t = fw_parti_t * (theta_t + Delta_tilde_parti_t)
        rho_bar_t = np.sum(rho_i * f_c_ist) / np.sum(f_c_ist)

        r_t = (
                nu - tax * beta0
                + rho_bar_t
                + mu_Y
                - sigma_Y * theta_t
        )

        mu_S_t = sigma_S_t * theta_t + r_t

        # store the results, only the aggregate values
        if i >= keep_when:  # only keeping the data after 200 years in the simulation
            ii = i - keep_when
            dR[ii] = dR_t  # realized return from t-1 to t
            theta[ii] = theta_t
            r[ii] = r_t
            # Delta_bar[ii] = np.average(Delta_s_t, weights=cohort_type_size)
            Delta_bar_parti[ii] = Delta_bar_parti_t
            Delta_tilde_parti[ii] = Delta_tilde_parti_t
            mu_S[ii] = mu_S_t
            sigma_S[ii] = np.abs(sigma_S_t)  # stock vola = absolute value of sigma
            beta[ii] = beta_t
            parti[ii] = popu_parti_t
            Phi_bar_parti_1[ii] = 1 / fc_parti_t
            Phi_tilde_parti[ii] = fw_parti_t
            if np.mod(ii, 12) == 0:
                invest_matrix[int(ii / 12)] = invest_tracker[0]
            # for l in range(N_wealth_group):
            #     within_group = np.where(
            #     (invest_fw_st >= wealth_cutoffs[l] * cohort_type_size / dt)
            #       * (w_indiv_ist < wealth_cutoffs[l + 1] * cohort_type_size / dt)
            #       )
            # parti_wealth_group[i, j] = np.ma.average(invest_tracker[within_group],
            #                                          weights=cohort_type_size[within_group])
            # for j in range(4):
            #     invest_age = invest_tracker[:, :, cutoffs_age[j + 1]:cutoffs_age[j]]
            #     cohort_type_age = cohort_type_size[:, :, cutoffs_age[j + 1]:cutoffs_age[j]]
            #     parti_age_group[ii, j] = np.ma.average(invest_age, weights=cohort_type_age)
            # turnover = invest_tracker[0, :, 12:] - invest_mat[i - 12, :, :-12]
            for j in range(3):
                entry_i = np.copy(invest_tracker[0])
                entry_i[:, :-12 * (j + 1)] = invest_tracker[0, :, :-12 * (j + 1)] > invest_mat[-12 * (j + 1), :, 12 * (
                        j + 1):]  # entry including the newborns
                # entry_i = invest_tracker[0, :, :-12 * (j + 1)] > invest_mat[-12 * (j + 1), :, 12 * (
                #         j + 1):]  # entry excluding the newborns
                exit_i = invest_tracker[0, :, :-12 * (j + 1)] < invest_mat[-12 * (j + 1), :, 12 * (j + 1):]
                entry_mat[ii, j] = np.average(entry_i, weights=np.sum(cohort_type_size_mix, axis=0))
                exit_mat[ii, j] = (np.average(exit_i, weights=np.sum(cohort_type_size_mix[:, :, :-12 * (j + 1)], axis=0))
                                  + np.average(invest_mat[-12 * (j + 1), :, 12 * (j + 1):],
                                               weights=np.sum(cohort_type_size_mix[:, :, :-12 * (j + 1)],
                                                              axis=0)) * nu * j
                                  )
        invest_mat = np.copy(np.append(invest_mat[1:], np.reshape(invest_tracker[0], (1, Nconstraint, -1)), axis=0))

    # save the mean and standard deviation
    data_mean_vola = {
        'theta': np.array([np.mean(theta), np.std(theta)]),
        'r': np.array([np.mean(r), np.std(r)]),
        'mu_S': np.array([np.mean(mu_S), np.std(mu_S)]),
        'sigma_S': np.array([np.mean(sigma_S), np.std(sigma_S)]),
        'Delta_bar': np.array([np.mean(Delta_bar_parti), np.std(Delta_bar_parti)]),
        'Phi_bar': np.array([np.mean(1/Phi_bar_parti_1), np.std(1/Phi_bar_parti_1)])
    }
    table_mean_vola = pd.DataFrame(data_mean_vola, index=['Mean', 'Std_Dev'])

    data_parti = {
        'entry': np.array([np.mean(entry_mat)] + list(np.percentile(entry_mat, [10, 50, 90]))),
        'exit': np.array([np.mean(exit_mat)] + list(np.percentile(exit_mat, [10, 50, 90]))),
        'parti': np.array([np.mean(parti)] + list(np.percentile(parti, [10, 50, 90]))),
    }
    table_parti = pd.DataFrame(data_parti, index=['Mean', '10th', '50th', '90th'])

    data_parti_cov = {
        'cons_share': np.corrcoef(parti, 1 / Phi_bar_parti_1)[0, 1],
        'wealth_share': np.corrcoef(parti, Phi_tilde_parti)[0, 1],
        'mu_S': np.corrcoef(parti, mu_S)[0, 1],
        'theta': np.corrcoef(parti, theta)[0, 1],
    }
    table_parti_cov = pd.DataFrame(data_parti_cov, index=['Cov'])

    past_annual_return = np.zeros((3, Nt - keep_when))
    future_annual_return = np.zeros((3, Nt - keep_when))
    for n, gap in enumerate([12, 24, 36]):
        past_annual_return[n, gap:] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
        past_annual_return[n, :gap] = np.cumsum(dR[:gap]) / (gap / 12)
        future_annual_return[n, :-gap] = (np.cumsum(dR)[gap:] - np.cumsum(dR)[:-gap]) / (gap / 12)
        future_annual_return[n, -gap:] = (np.cumsum(dR)[-gap:]) / (gap / 12)
    # run regressions and save results instead of saving the data:
    x_set = np.copy(past_annual_return[:, sample])
    y_set = [
        entry_mat[sample] - exit_mat[sample],
        entry_mat[sample],
        exit_mat[sample]
    ]
    regression_table1_b = np.zeros((len(x_set), len(y_set)), dtype=np.float32)
    for ii in range(3):
        x = (x_set[ii] - np.average(x_set[ii])) / np.std(x_set[ii])
        for jj, y_mat in enumerate(y_set):
            if jj == 1:  # entry on high return
                y = (y_mat[:, ii] - np.average(y_mat[:, ii])) / np.std(y_mat[:, ii])
                x_regress = sm.add_constant(x)
                # x_condi = (x > np.percentile(x, 75)) + 0
                # x_regress = sm.add_constant(x_condi)
                model = sm.OLS(y, x_regress)
            elif jj == 2:  # exit on low return
                y = (y_mat[:, ii] - np.average(y_mat[:, ii])) / np.std(y_mat[:, ii])
                x_regress = sm.add_constant(x)
                # x_condi = (x < np.percentile(x, 25)) + 0
                # x_regress = sm.add_constant(x_condi)
                model = sm.OLS(y, x_regress)
            else:
                y = (y_mat[:, ii] - np.average(y_mat[:, ii])) / np.std(y_mat[:, ii])
                x_regress = sm.add_constant(x)
                model = sm.OLS(y, x_regress)
            est = model.fit()
            regression_table1_b[ii, jj] = est.params[1]


    x = np.reshape((parti[sample] - np.average(parti[sample])) / np.std(parti[sample]), (-1, 1))
    y_set = [
        future_annual_return[:, sample],
        future_annual_return[:, sample] - r[sample]
    ]
    regression_table2_b = np.zeros((3, len(y_set)), dtype=np.float32)
    for ii in range(3):
        for jj, y_mat in enumerate(y_set):
            y = (y_mat[ii] - np.average(y_mat[ii])) / np.std(y_mat[ii])
            x_regress = sm.add_constant(x)
            model = sm.OLS(y, x_regress)
            est = model.fit()
            regression_table2_b[ii, jj] = est.params[1]

    # cohort_entry_annual = (invest_matrix[1:, :, :-12] - invest_matrix[:-1, :, 12:]) > 0
    # entry_cumu = np.cumsum(
    #     np.flip(np.average(
    #         np.average(cohort_entry_annual * dt, axis=0), axis=0, weights=alpha_i_mix[0, :, 0]
    #     ))
    # )[age_sample]

    # fraction of agents re-entering after exiting the stock market
    if need_invest_matrix == 'True':
        # window_bell = 20
        sample_bell = np.arange(0, np.shape(invest_matrix)[0], window_bell)
        reentry_time = np.zeros((len(sample_bell) - 1, Nconstraint, Nt - int(window_bell / dt) - 12), dtype=int)
        # exit_time = np.zeros((len(sample_bell) - 1, Nconstraint, Nt - int(window_bell / dt)), dtype=int)
        for n, entry_n in enumerate(sample_bell[1:]):  # 20 year non-overlapping windows
            following_cohorts_entry = (invest_matrix[entry_n, :, :-12] - invest_matrix[entry_n - 1, :, 12:] > 0)[
                                      :, int(window_bell / dt):]
            following_cohorts_entry = np.append(following_cohorts_entry, invest_matrix[entry_n, :, -12:], axis=1)
            parti_bell_entry = np.zeros((window_bell, Nconstraint, Nt - int(window_bell / dt)))
            parti_bell_entry[0] = following_cohorts_entry
            # exit_bell = np.zeros(
            #     (4, Nt - int(window_bell / dt)))  # following the entering cohorts until they exit the first time

            following_cohorts_exit = (invest_matrix[entry_n, :, :-12] - invest_matrix[entry_n - 1, :, 12:] < 0)[
                                     :, int(window_bell / dt):]  # ignoring the cohorts born during the "year"
            parti_bell_exit = np.zeros((window_bell, Nconstraint, Nt - int(window_bell / dt) - 12))
            parti_bell_exit[0] = following_cohorts_exit
            reentry_bell = np.zeros((Nconstraint, Nt - int(window_bell / dt) - 12))

            for nn in range(1, window_bell):
                cohorts_in = invest_matrix[entry_n + nn, :, int((window_bell - nn) / dt):int(-nn / dt)]
                cohorts_out = (1 - cohorts_in)[:, :-12]

                # exit_nn = (
                #         invest_matrix[entry_n + nn, :, int((window_bell - nn) / dt):int(-nn / dt)]
                #         - invest_matrix[entry_n + nn - 1, :, int((window_bell - nn + 1) / dt):int((-nn + 1) / dt)] < 0
                # ) if nn != 1 else (
                #         invest_matrix[entry_n + nn, :, int(window_bell / dt - nn / dt):int(-nn / dt)]
                #         - invest_matrix[entry_n + nn - 1, :, int((window_bell - nn + 1) / dt):] < 0
                # )

                reentry_nn = (
                        invest_matrix[entry_n + nn, :, int((window_bell - nn) / dt):int(-nn / dt)]
                        - invest_matrix[entry_n + nn - 1, :, int((window_bell - nn + 1) / dt):int((-nn + 1) / dt)] > 0
                ) if nn != 1 else (
                        invest_matrix[entry_n + nn, :, int(window_bell / dt - nn / dt):int(-nn / dt)]
                        - invest_matrix[entry_n + nn - 1, :, int((window_bell - nn + 1) / dt):] > 0
                )
                # exit_bell = exit_bell + exit_nn > 0
                # parti_bell_entry[nn] = cohorts_in * following_cohorts_entry * (1 - exit_bell)

                reentry_bell = reentry_bell + reentry_nn[:, :-12] > 0
                parti_bell_exit[nn] = cohorts_out * following_cohorts_exit * (1 - reentry_bell)

            for m in range(Nconstraint):
                reentry_time[n, m] = list(map(int, np.sum(parti_bell_exit[:, m], axis=0)))
                # exit_time[n, m] = list(
                #     map(int, np.sum(parti_bell_entry[:, m], axis=0))
                # )

    else:
        reentry_time = 0
        # exit_time = 0

    return (
        table_mean_vola,
        table_parti,
        table_parti_cov,
        reentry_time,
        regression_table1_b,
        regression_table2_b
    )