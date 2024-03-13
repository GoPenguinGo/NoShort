import numpy as np
from typing import Tuple
from src.stats import post_var, dDelta_st_calculator
from src.solver import bisection, solve_theta, bisection_partial_constraint, \
    solve_theta_partial_constraint
from tqdm import tqdm


def simulate_cohorts_SI(
        biasvec: np.ndarray,
        dZ: np.ndarray,
        dZ_SI: np.ndarray,
        Nt: int,
        Nc: int,
        tau: np.ndarray,
        dt: float,
        Ntype: int,
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
        mode_trade: str,
        mode_learn: str,
        cohort_type_size: np.ndarray,
        cutoffs_age: np.ndarray,
        Delta_s_t: np.ndarray,
        eta_st_eta_ss: np.ndarray,
        X: np.ndarray,
        d_eta_st: np.ndarray,
        invest_tracker: np.ndarray,
        tau_info: np.ndarray,
        Vhat_vector: np.ndarray,
        need_f: str,
        need_Delta: str,
        need_pi: str,
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
]:
    """ Simulate the economy forward

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
        Phi_parti (np.ndarray): consumption share of participants, shape(Nt)
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
    if need_f == 'True':
        f_c = np.zeros((Nt, Ntype, Nc), dtype=np.float16)  # evolution of cohort consumption share
    else:
        f_c = f_w = w_indiv_mat = 0

    if need_Delta == 'True':
        Delta = np.zeros((Nt, Nc), dtype=np.float16)  # stores bias in beliefs
    else:
        Delta = 0

    if need_pi == 'True':
        pi = np.zeros((Nt, Nc), dtype=np.float16)  # portfolio choices
    else:
        pi = 0

    Phi_bar_parti = np.ones(Nt, dtype=np.float16)
    Phi_tilde_parti = np.ones(Nt, dtype=np.float16)
    invest_mat = np.ones((Nt, Nc), dtype=np.int8)
    parti_wealth_group = np.ones((Nt, 4), dtype=np.float16)
    parti_age_group = np.ones((Nt, 4), dtype=np.float16)

    # equilibrium terms:
    dR = np.zeros(Nt)  # stores stock returns
    r = np.zeros(Nt)  # interest rate
    theta = np.zeros(Nt)  # market price of risk
    mu_S = np.zeros(Nt)
    sigma_S = np.zeros(Nt)
    beta = np.zeros(Nt)
    Delta_bar_parti = np.zeros(Nt,
                               dtype=np.float16)  # consumption weighted estimation error of the stock market participants
    Delta_tilde_parti = np.zeros(Nt,
                                 dtype=np.float16)  # wealth weighted estimation error of the stock market participants
    parti = np.ones(Nt, dtype=np.float16)  # participation rate

    dR_t = 0

    a_phi = (1 - phi ** 2)
    phi_sqr_a_phi = phi / np.sqrt(a_phi)
    a_phi_1 = 1 / a_phi
    sigma_Y_sq = sigma_Y ** 2

    mu_S_t = 0
    sigma_S_t = 0
    r_t = 0
    pi_st = 0
    w_indiv_ist = 0

    wealth_cutoffs = np.array([0, 1, 10, 100, 100000])

    for i in tqdm(range(Nt)):
        dZ_t = dZ[i]
        dZ_SI_t = dZ_SI[i]

        # new cohort born (age 0), get wealth transfer, observe, invest
        eta_st_eta_ss = eta_st_eta_ss * np.exp(
            (-0.5 * d_eta_st ** 2) * dt
            + d_eta_st * dZ_t
        )  # equation (15)

        # from equation (20) and the description below it
        # X_t = W_t * xi_t, is the sum of tax * X_s * eta_st_eta_ss * rho_cohort_type_short * dt, s<t;
        # X is the collection of all X_s, s<t.
        X_parts = tax * X * eta_st_eta_ss * rho_cohort_type * dt
        X_t = np.sum(X_parts) / (1 - tax * beta0 * dt)  # dividing by (1-tax*dt) keeps sum(f_st*dt) at 1

        eta_st_eta_ss = np.append(eta_st_eta_ss[:, 1:], np.ones((Ntype, 1)), axis=1)
        X = np.append(X[:, 1:], X_t * np.ones((1, 1)), axis=1)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort
        # eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
        #  eta_bar_t is the denominator; it creates issues if too close to 0
        #  so we rescale eta_bar to keep it away from 0, without changing f_st

        f_c_ist = X_parts / X_t / dt
        f_c_ist = np.append(f_c_ist[:, 1:], tax * alpha_i * beta_i, axis=1)

        beta_t = 1 / np.sum(f_c_ist / beta_i * dt)
        f_w_ist = f_c_ist / beta_i * beta_t

        w_indiv_ist = f_w_ist / cohort_type_size * dt
        dR_t = mu_S_t * dt + sigma_S_t * dZ_t  # realized stock return, mu_t^Sdt + sigma_t^Sdz_t

        # update beliefs
        if mode_trade == 'complete':  # everyone is P
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t, dZ_SI_t,
                                              'P')
            tau_info = tau  # au_info is the same with age; no switch between N and P for complete market

        elif mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
            V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')  # from eq(6)
            dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_t,
                                                dZ_SI_t,
                                                'N')  # from eq(9)
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t,
                                                dZ_SI_t,
                                                'P')
            dDelta_s_t = invest_tracker * dDelta_s_t_P + (
                    1 - invest_tracker) * dDelta_s_t_N  # the participation decision of last time affects the updating pattern
            tau_info = np.append(tau_info[:, 1:], np.zeros((Ntype, 1)),
                                 axis=1) + dt  # tau_info is t-t', where t' is either the last time a cohort switch, or the birth time

        else:
            print('mode_trade not found')
            exit()

        Vhat_vector = np.append(Vhat_vector[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1)

        if i < Npre - 1:
            init_bias = (np.sum(biasvec[i + 1:]) + np.sum(dZ[:i + 1])) / T_hat
        else:
            init_bias = np.sum(dZ[i + 1 - Npre: i + 1]) / T_hat

        Delta_s_t = Delta_s_t[:, 1:] + dDelta_s_t[:, 1:]
        Delta_s_t = np.append(Delta_s_t, init_bias * np.ones((Ntype, 1)), axis=1)

        # find the market clearing theta, given beliefs and consumption shares of cohorts in the economy
        invest_tracker = np.append(invest_tracker[:, 1:], np.ones((Ntype, 1)),
                                   axis=1)  # all cohorts that are still in the market, new born cohort can participate by default
        if mode_trade == 'w_constraint':
            if mode_learn == 'disappointment':
                possible_cons_share = f_c_ist * dt * invest_tracker
                possible_delta_st = Delta_s_t * invest_tracker
                lowest_bound = -np.max(possible_delta_st[np.nonzero(
                    possible_delta_st)])  # absolute lower bound for theta among active investors
                theta_t = bisection(
                    solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
                )  # solve for theta
                a = Delta_s_t + theta_t
                invest = (a > 0)
                switch_P_to_N = invest_tracker * (1 - invest)
                invest_tracker = invest * invest_tracker  # update invest tracker

                # tau_info and V_hat has to change for the agents who switched to N
                Vhat_vector = np.append(V_st_P[:, 1:], Vhat * np.ones((Ntype, 1)),
                                        axis=1) * switch_P_to_N + Vhat_vector * (
                                      1 - switch_P_to_N)  # reset prior variance V'
                tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock t'

            elif mode_learn == 'reentry':  # agents switch between type P and type N
                possible_cons_share = f_c_ist * dt
                possible_delta_st = Delta_s_t
                lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
                theta_t = bisection(
                    solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
                )  # solve for theta
                a = Delta_s_t + theta_t
                invest = (a > 0)
                switch_P_to_N = invest_tracker * (1 - invest)
                switch_N_to_P = np.maximum(invest - invest_tracker, 0)

                # tau_info and V_hat has to change for the agents who switch
                switch = switch_N_to_P + switch_P_to_N
                invest_tracker = invest
                Vhat_vector = np.append(V_st_P[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1) * switch_P_to_N + np.append(
                    V_st_N[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1) * switch_N_to_P + Vhat_vector * (
                                      1 - switch)  # reset prior variance V'
                tau_info = dt * switch + tau_info * (1 - switch)  # reset clock t'

            else:
                print('mode_learn not found')
                exit()

            d_eta_st = a * invest_tracker - theta_t
            invest_fc_st = invest_tracker * f_c_ist * dt
            invest_fw_st = invest_tracker * f_w_ist * dt
            popu_parti_t = np.sum(cohort_type_size * invest_tracker)
            fc_parti_t = np.sum(invest_fc_st)
            fw_parti_t = np.sum(invest_fw_st)
            Delta_bar_parti_t = np.sum(Delta_s_t * invest_fc_st) / fc_parti_t
            Delta_tilde_parti_t = np.sum(Delta_s_t * invest_fw_st) / fw_parti_t
            sigma_S_t = fw_parti_t * (theta_t + Delta_tilde_parti_t)
            pi_st = (d_eta_st + theta_t) / sigma_S_t

        elif mode_trade == 'complete':
            fc_ist_standard = f_c_ist * dt
            Delta_bar_parti_t = np.sum(fc_ist_standard * Delta_s_t)
            theta_t = sigma_Y - Delta_bar_parti_t
            d_eta_st = Delta_s_t
            popu_parti_t = 1

            fc_parti_t = fw_parti_t = 1
            Delta_bar_parti_t = np.sum(Delta_s_t * f_c_ist * dt)
            Delta_tilde_parti_t = np.sum(Delta_s_t * f_w_ist * dt)
            sigma_S_t = fw_parti_t * (theta_t + Delta_tilde_parti_t)
            pi_st = (d_eta_st + theta_t) / sigma_S_t
        else:
            print('mode_trade not found')
            exit()

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
        sigma_S[i] = np.abs(sigma_S_t)
        beta[i] = beta_t
        if need_f == 'True':
            f_c[i] = f_c_ist
        if need_Delta == 'True':
            Delta[i] = Delta_s_t[0]
        if need_pi == 'True':
            pi[i] = pi_st[0]
        if mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
            Phi_bar_parti[i] = fc_parti_t
            Phi_tilde_parti[i] = fw_parti_t
            parti[i] = popu_parti_t
            invest_mat[i] = invest_tracker[0]
            for j in range(4):
                parti_age_group[i, j] = np.ma.average(invest_tracker[:, cutoffs_age[j + 1]:cutoffs_age[j]],
                                                      weights=cohort_type_size[:, cutoffs_age[j + 1]:cutoffs_age[j]])
                within_group = np.where((w_indiv_ist >= wealth_cutoffs[j]) * (w_indiv_ist < wealth_cutoffs[j + 1]))
                parti_wealth_group[i, j] = np.ma.average(invest_tracker[within_group],
                                                         weights=cohort_type_size[within_group])
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
        invest_mat,
        parti_age_group,
        parti_wealth_group,
    )


def simulate_cohorts_mean_vola(
        biasvec: np.ndarray,
        dZ: np.ndarray,
        dZ_SI: np.ndarray,
        Nt: int,
        tau: np.ndarray,
        dt: float,
        Ntype: int,
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
        mode_trade: str,
        mode_learn: str,
        cohort_type_size: np.ndarray,
        cutoffs_age: np.ndarray,
        Delta_s_t: np.ndarray,
        eta_st_eta_ss: np.ndarray,
        X: np.ndarray,
        d_eta_st: np.ndarray,
        invest_tracker: np.ndarray,
        tau_info: np.ndarray,
        Vhat_vector: np.ndarray,
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
]:
    """ Simulate the economy forward, saving only the mean & std of the results

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

    dR = np.zeros(Nt - keep_when)  # stores stock returns
    r = np.zeros(Nt - keep_when)  # interest rate
    theta = np.zeros(Nt - keep_when)  # market price of risk
    mu_S = np.zeros(Nt - keep_when)
    sigma_S = np.zeros(Nt - keep_when)
    beta = np.zeros(Nt - keep_when)
    Delta_bar = np.zeros(Nt - keep_when)
    Delta_bar_parti = np.zeros(
        (Nt - keep_when))  # consumption weighted estimation error of the stock market participants
    Delta_tilde_parti = np.zeros((Nt - keep_when))  # wealth weighted estimation error of the stock market participants
    parti = np.ones((Nt - keep_when))  # participation rate
    Phi_bar_parti_1 = np.ones((Nt - keep_when))
    Phi_tilde_parti = np.ones((Nt - keep_when))
    parti_age_group = np.ones((Nt - keep_when, 4))
    N_wealth_group = 4
    parti_wealth_group = np.ones((Nt - keep_when, N_wealth_group))

    a_phi = (1 - phi ** 2)
    phi_sqr_a_phi = phi / np.sqrt(a_phi)
    a_phi_1 = 1 / a_phi
    sigma_Y_sq = sigma_Y ** 2
    dR_t = 0
    mu_S_t = 0
    sigma_S_t = 0
    wealth_cutoffs = np.array([0, 1, 10, 100, 100000])

    for i in tqdm(range(Nt)):
        dZ_t = dZ[i]
        dZ_SI_t = dZ_SI[i]

        # new cohort born (age 0), get wealth transfer, observe, invest
        eta_st_eta_ss = eta_st_eta_ss * np.exp(
            (-0.5 * d_eta_st ** 2) * dt
            + d_eta_st * dZ_t
        )  # equation (15)

        # from equation (20) and the description below it
        # X_t = W_t * xi_t, is the sum of tax * X_s * eta_st_eta_ss * rho_cohort_type_short * dt, s<t;
        # X is the collection of all X_s, s<t.
        X_parts = tax * X * eta_st_eta_ss * rho_cohort_type * dt
        X_t = np.sum(X_parts) / (1 - tax * beta0 * dt)

        eta_st_eta_ss = np.append(eta_st_eta_ss[:, 1:], np.ones((Ntype, 1)), axis=1)
        X = np.append(X[:, 1:], X_t * np.ones((1, 1)), axis=1)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort

        f_c_ist = X_parts / X_t / dt
        f_c_ist = np.append(f_c_ist[:, 1:], tax * alpha_i * beta_i, axis=1)

        beta_t = 1 / np.sum(f_c_ist / beta_i * dt)
        f_w_ist = f_c_ist / beta_i * beta_t

        w_indiv_ist = f_w_ist / cohort_type_size * dt

        dR_t = mu_S_t * dt + sigma_S_t * dZ_t  # realized stock return

        # update beliefs
        if mode_trade == 'complete':  # everyone is P
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t, dZ_SI_t,
                                              'P')
            tau_info = tau

        elif mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
            V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')  # from eq(6)
            dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_t,
                                                dZ_SI_t,
                                                'N')  # from eq(9)
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t,
                                                dZ_SI_t,
                                                'P')
            dDelta_s_t = invest_tracker * dDelta_s_t_P + (
                    1 - invest_tracker) * dDelta_s_t_N  # the participation decision of last time affects the
            tau_info = np.append(tau_info[:, 1:], np.zeros((Ntype, 1)), axis=1) + dt

        else:
            print('mode_trade not found')
            exit()

        Vhat_vector = np.append(Vhat_vector[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1)

        if i < Npre - 1:
            init_bias = (np.sum(biasvec[i + 1:]) + np.sum(dZ[:i + 1])) / T_hat
        else:
            init_bias = np.sum(dZ[i + 1 - Npre: i + 1]) / T_hat

        Delta_s_t = Delta_s_t[:, 1:] + dDelta_s_t[:, 1:]
        Delta_s_t = np.append(Delta_s_t, init_bias * np.ones((Ntype, 1)), axis=1)

        # find the market clearing theta, given beliefs and consumption shares of cohorts in the economy
        invest_tracker = np.append(invest_tracker[:, 1:], np.ones((Ntype, 1)),
                                   axis=1)

        if mode_trade == 'w_constraint':
            if mode_learn == 'disappointment':
                possible_cons_share = f_c_ist * dt * invest_tracker
                possible_delta_st = Delta_s_t * invest_tracker
                lowest_bound = -np.max(possible_delta_st[np.nonzero(
                    possible_delta_st)])  # absolute lower bound for theta among active investors
                theta_t = bisection(
                    solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
                )  # solve for theta
                a = Delta_s_t + theta_t
                invest = (a > 0)
                switch_P_to_N = invest_tracker * (1 - invest)
                invest_tracker = invest * invest_tracker  # update invest tracker

                # tau_info and V_hat has to change for the agents who switched to N
                Vhat_vector = np.append(V_st_P[:, 1:], Vhat * np.ones((Ntype, 1)),
                                        axis=1) * switch_P_to_N + Vhat_vector * (
                                      1 - switch_P_to_N)  # reset V'
                tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset t'

            elif mode_learn == 'reentry':  # agents switch between type P and type N
                possible_cons_share = f_c_ist * dt
                possible_delta_st = Delta_s_t
                lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
                theta_t = bisection(
                    solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
                )  # solve for theta
                a = Delta_s_t + theta_t
                invest = (a > 0)
                switch_P_to_N = invest_tracker * (1 - invest)
                switch_N_to_P = np.maximum(invest - invest_tracker, 0)
                # tau_info and V_hat has to change for the agents who switch (either P to N or vice versa)
                switch = switch_N_to_P + switch_P_to_N
                invest_tracker = invest

                Vhat_vector = np.append(V_st_P[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1) * switch_P_to_N + np.append(
                    V_st_N[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1) * switch_N_to_P + Vhat_vector * (
                                      1 - switch)  # reset V'
                tau_info = dt * switch + tau_info * (1 - switch)  # reset t'

            else:
                print('mode_learn not found')
                exit()

            d_eta_st = a * invest_tracker - theta_t
            invest_fc_st = invest_tracker * f_c_ist * dt
            invest_fw_st = invest_tracker * f_w_ist * dt
            popu_parti_t = np.sum(cohort_type_size * invest_tracker)
            fc_parti_t = np.sum(invest_fc_st)
            fw_parti_t = np.sum(invest_fw_st)
            Delta_bar_parti_t = np.sum(Delta_s_t * invest_fc_st) / fc_parti_t
            Delta_tilde_parti_t = np.sum(Delta_s_t * invest_fw_st) / fw_parti_t
            sigma_S_t = fw_parti_t * (theta_t + Delta_tilde_parti_t)
            pi_st = (d_eta_st + theta_t) / sigma_S_t

        elif mode_trade == 'complete':
            fc_ist_standard = f_c_ist * dt
            Delta_bar_parti_t = np.sum(fc_ist_standard * Delta_s_t)
            theta_t = sigma_Y - Delta_bar_parti_t
            d_eta_st = Delta_s_t
            invest = Delta_s_t >= -theta_t  # long stock
            popu_parti_t = 1

            fc_parti_t = fw_parti_t = 1
            Delta_bar_parti_t = np.sum(Delta_s_t * f_c_ist * dt)
            Delta_tilde_parti_t = np.sum(Delta_s_t * f_w_ist * dt)
            sigma_S_t = fw_parti_t * (theta_t + Delta_tilde_parti_t)
        else:
            print('mode_trade not found')
            exit()

        rho_bar_t = np.sum(rho_i * f_c_ist) / np.sum(f_c_ist)

        r_t = (
                nu - tax * beta0
                + rho_bar_t
                + mu_Y
                - sigma_Y * theta_t
        )

        mu_S_t = sigma_S_t * theta_t + r_t

        # store the results
        if i >= keep_when:  # only keeping the data after 200 years in the simulation
            ii = i - keep_when
            dR[ii] = dR_t  # realized return from t-1 to t
            theta[ii] = theta_t
            r[ii] = r_t
            Delta_bar[ii] = np.average(Delta_s_t, weights=cohort_type_size)
            Delta_bar_parti[ii] = Delta_bar_parti_t
            Delta_tilde_parti[ii] = Delta_tilde_parti_t
            mu_S[ii] = mu_S_t
            sigma_S[ii] = np.abs(sigma_S_t)  # stock vola = absolute value of sigma
            beta[ii] = beta_t
            parti[ii] = popu_parti_t
            if mode_trade == 'w_constraint':
                Phi_bar_parti_1[ii] = 1 / fc_parti_t
                Phi_tilde_parti[ii] = fw_parti_t
                for l in range(N_wealth_group):
                    within_group = np.where((w_indiv_ist >= wealth_cutoffs[l]) * (w_indiv_ist < wealth_cutoffs[l + 1]))
                    parti_wealth_group[ii, l] = np.ma.average(invest_tracker[within_group],
                                                              weights=cohort_type_size[within_group]
                                                              )
                for j in range(4):
                    invest_age = invest_tracker[:, cutoffs_age[j + 1]:cutoffs_age[j]]
                    cohort_type_age = cohort_type_size[:, cutoffs_age[j + 1]:cutoffs_age[j]]
                    parti_age_group[ii, j] = np.ma.average(invest_age,
                                                           weights=cohort_type_age)

    # save the mean and standard deviation
    dR_matrix = np.array([np.mean(dR / dt), np.std(dR / dt)])
    theta_matrix = np.array([np.mean(theta), np.std(theta)])
    r_matrix = np.array([np.mean(r), np.std(r)])
    mu_S_matrix = np.array([np.mean(mu_S), np.std(mu_S)])
    sigma_S_matrix = np.array([np.mean(sigma_S), np.std(sigma_S)])
    beta_matrix = np.array([np.mean(beta), np.std(beta)])

    Delta_bar_parti_matrix = np.array([np.mean(Delta_bar_parti), np.std(Delta_bar_parti)])
    Delta_tilde_bar_parti = Delta_tilde_parti - Delta_bar_parti
    Delta_tilde_bar_parti_matrix = np.array([np.mean(Delta_tilde_bar_parti), np.std(Delta_tilde_bar_parti)])
    Phi_tilde_bar_parti = Phi_tilde_parti * Phi_bar_parti_1
    Phi_bar_parti_1_matrix = np.array([np.mean(Phi_bar_parti_1 * sigma_Y), np.std(Phi_bar_parti_1 * sigma_Y)])
    Phi_tilde_parti_matrix = np.array([np.mean(Phi_tilde_parti), np.std(Phi_tilde_parti)])
    Phi_tilde_bar_parti_matrix = np.array(
        [np.mean(Phi_tilde_bar_parti * sigma_Y), np.std(Phi_tilde_bar_parti * sigma_Y)])
    Delta_Phi_tilde = Delta_tilde_bar_parti * Phi_tilde_parti
    Delta_Phi_tilde_matrix = np.array([np.mean(Delta_Phi_tilde), np.std(Delta_Phi_tilde)])
    parti_age_group_matrix = np.array(np.mean(parti_age_group, axis=0))
    Delta_bar_mat = np.tile(np.reshape(Delta_bar, (-1, 1)), (1, N_wealth_group))
    parti_wealth_group_mask = parti_wealth_group
    parti_wealth_group_matrix = np.array(np.nanmean(parti_wealth_group_mask, axis=0))
    cov_theta_z_Y = np.corrcoef(dZ[keep_when:], theta)[0, 1]
    cov_muS_z_Y = np.corrcoef(dZ[keep_when:], mu_S)[0, 1]
    cov_sigmaS_z_Y = np.corrcoef(dZ[keep_when:], sigma_S)[0, 1]
    cov_theta_z_SI = np.corrcoef(dZ_SI[keep_when:], theta)[0, 1]
    cov_parti_cons_share = np.corrcoef(parti, 1 / Phi_bar_parti_1)[0, 1]
    cov_parti_wealth_share = np.corrcoef(parti, Phi_tilde_parti)[0, 1]

    # save results that are closely related to each other together

    # results related to theta
    theta_save_matrix = np.array([
        Phi_bar_parti_1_matrix,
        Delta_bar_parti_matrix,
    ])

    # results related to sigma_S
    sigma_S_save_matrix = np.array([
        Phi_tilde_parti_matrix,
        Phi_tilde_bar_parti_matrix,
        Delta_tilde_bar_parti_matrix,
        Delta_Phi_tilde_matrix,
    ])

    # results about covariance
    cov_save_matrix = np.array([
        cov_theta_z_Y,
        cov_muS_z_Y,
        cov_sigmaS_z_Y,
        cov_theta_z_SI,
        cov_parti_cons_share,
        cov_parti_wealth_share,
    ])

    # results about covariance, different window
    cov_parti_matrix1 = np.array([])
    cov_parti_matrix2 = np.array([])
    cov_parti_matrix3 = np.array([])
    cov_parti_matrix4 = np.array([])
    windows = np.array([6, 24, 60])
    R_cumu = np.cumsum(dR)
    vola_cumu = np.cumsum(sigma_S ** 2)
    r_cumu = np.cumsum(r)
    for j, window in enumerate(windows):
        R_window = R_cumu[window:] - R_cumu[:-window]
        vola_window = np.sqrt((vola_cumu[window:] - vola_cumu[:-window]) / window)
        r_window = r_cumu[window:] - r_cumu[:-window]
        R_window_gap = np.reshape(R_window, (-1, window))[:, 0]
        vola_window_gap = np.reshape(vola_window, (-1, window))[:, 0]
        r_window_gap = np.reshape(r_window, (-1, window))[:, 0]
        parti_window = np.reshape(parti, (-1, window))[:, 0]
        r_reshape = np.reshape(r, (-1, window))
        erp_window_gap = R_window_gap - np.sum(r_reshape, axis=1)[:-1]
        corr1 = np.corrcoef(R_window_gap, parti_window[:-1])[0, 1]
        corr2 = np.corrcoef(r_window_gap, parti_window[:-1])[0, 1]
        corr3 = np.corrcoef(erp_window_gap, parti_window[:-1])[0, 1]
        corr4 = np.corrcoef(vola_window_gap, parti_window[:-1])[0, 1]
        cov_parti_matrix1 = np.append(cov_parti_matrix1, corr1)
        cov_parti_matrix2 = np.append(cov_parti_matrix2, corr2)
        cov_parti_matrix3 = np.append(cov_parti_matrix3, corr3)
        cov_parti_matrix4 = np.append(cov_parti_matrix4, corr4)
    cov_parti_matrix = np.array([
        cov_parti_matrix1,
        cov_parti_matrix2,
        cov_parti_matrix3,
        cov_parti_matrix4,
    ])

    return (
        dR_matrix,
        theta_matrix,
        r_matrix,
        mu_S_matrix,
        sigma_S_matrix,
        beta_matrix,
        theta_save_matrix,
        sigma_S_save_matrix,
        parti_age_group_matrix,
        parti_wealth_group_matrix,
        cov_save_matrix,
        parti,
        cov_parti_matrix,
    )


def simulate_cohorts_mix_type(
        biasvec: np.ndarray,
        dZ: np.ndarray,
        dZ_SI: np.ndarray,
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
        cohort_type_size: np.ndarray,
        cutoffs_age: np.ndarray,
        Delta_s_t: np.ndarray,
        eta_st_eta_ss: np.ndarray,
        X: np.ndarray,
        d_eta_st: np.ndarray,
        invest_tracker: np.ndarray,
        can_short_tracker: np.ndarray,
        tau_info: np.ndarray,
        Vhat_vector: np.ndarray,
        need_f: str,
        need_Delta: str,
        need_pi: str,
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
    invest_newborn = np.array([[[1], [0], [1], [1]]]) * np.ones((Ntype, Nconstraint, 1), dtype=np.int8)
    can_short_newborn = np.array([[[1], [0], [0], [0]]]) * np.ones((Ntype, Nconstraint, 1), dtype=np.int8)
    # top = np.array([1, 0.75, 0.5, 0.25, 0])

    Phi_bar_parti = np.ones(Nt, dtype=np.float16)  # consumption share of the stock market participants
    Phi_tilde_parti = np.ones(Nt, dtype=np.float16)

    popu_short = np.zeros(Nt, dtype=np.float16)
    Phi_can_short = np.zeros(Nt, dtype=np.float16)

    # equilibrium terms:
    dR = np.zeros(Nt)  # stores stock returns
    r = np.zeros(Nt)  # interest rate
    theta = np.zeros(Nt)  # market price of risk
    mu_S = np.zeros(Nt)
    sigma_S = np.zeros(Nt)
    beta = np.zeros(Nt)
    Delta_bar_parti = np.zeros(Nt,
                               dtype=np.float16)  # consumption weighted estimation error of the stock market participants
    Delta_tilde_parti = np.zeros(Nt,
                                 dtype=np.float16)  # wealth weighted estimation error of the stock market participants
    parti = np.ones(Nt, dtype=np.float16)  # participation rate
    parti_wealth_group = np.zeros((Nt, 4), dtype=np.float16)
    parti_age_group = np.zeros((Nt, 4), dtype=np.float16)
    dR_t = 0
    a_phi = (1 - phi ** 2)
    phi_sqr_a_phi = phi / np.sqrt(a_phi)
    a_phi_1 = 1 / a_phi
    sigma_Y_sq = sigma_Y ** 2

    append_init = np.ones((Ntype, Nconstraint, 1), dtype=np.int8)
    wealth_cutoffs = np.array([0, 1, 10, 100, 100000])

    if need_f == 'True':
        f_c = np.zeros((Nt, Ntype, Nconstraint, Nc), dtype=np.float16)  # evolution of cohort consumption share
        # f_w = np.zeros((Nt, Ntype, Nc))  # evolution of cohort wealth share
        # w_indiv_mat = np.zeros((Nt, Ntype, Nc))
    else:
        f_c = f_w = w_indiv_mat = 0

    if need_Delta == 'True':
        Delta = np.zeros((Nt, Nconstraint, Nc), dtype=np.float16)  # stores bias in beliefs
    else:
        Delta = 0

    if need_pi == 'True':
        pi = np.zeros((Nt, Nconstraint, Nc), dtype=np.float16)  # portfolio choices
    else:
        pi = 0
    invest_mat = np.zeros((Nt, Nconstraint, Nc), dtype=np.int8)

    for i in tqdm(range(Nt)):
        dZ_t = dZ[i]
        dZ_SI_t = dZ_SI[i]

        # todo: eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
        #  eta_bar_t is the denominator; it creates issues if too close to 0
        #  so I rescale eta_bar to keep it away from 0, without changing f_st

        # new cohort born (age 0), get wealth transfer, observe, invest
        eta_st_eta_ss = eta_st_eta_ss * np.exp(
            (-0.5 * d_eta_st ** 2) * dt
            + d_eta_st * dZ_t
        )  # equation (15)

        X_parts = tax * X * eta_st_eta_ss * rho_cohort_type * dt
        X_t = np.sum(X_parts) / (1 - tax * beta0 * dt)
        # eta_bar_t = np.sum(eta_bar_parts)

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

        w_indiv_ist = f_w_ist / cohort_type_size * dt
        if i > 0:
            dR_t = mu_S_t * dt + sigma_S_t * dZ_t

        # update beliefs
        V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')  # from eq(6)
        dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_t, dZ_SI_t,
                                            'N')  # from eq(9)
        V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
        dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t, dZ_SI_t,
                                            'P')
        dDelta_s_t = invest_tracker * dDelta_s_t_P + (
                1 - invest_tracker) * dDelta_s_t_N  # the participation decision of last time affects the updating pattern
        tau_info = np.append(tau_info[:, :, 1:], 0 * append_init, axis=2) + dt
        Vhat_vector = np.append(Vhat_vector[:, :, 1:], Vhat * append_init, axis=2)

        if i < Npre - 1:
            init_bias = (np.sum(biasvec[i + 1:]) + np.sum(dZ[:i + 1])) / T_hat
        else:
            init_bias = np.sum(dZ[i + 1 - Npre:i + 1]) / T_hat

        Delta_s_t = Delta_s_t[:, :, 1:] + dDelta_s_t[:, :, 1:]
        Delta_s_t = np.append(Delta_s_t, init_bias * append_init, axis=2)

        # find the market clearing theta, given beliefs and consumption shares of cohorts in the economy
        invest_tracker = np.append(invest_tracker[:, :, 1:], invest_newborn,
                                   axis=2)  # all cohorts that are still in the market, 1 by default
        can_short_tracker = np.append(can_short_tracker[:, :, 1:], can_short_newborn, axis=2)

        possible_cons_share = f_c_ist * dt * invest_tracker
        possible_delta_st = Delta_s_t * invest_tracker

        lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound
        theta_t = bisection_partial_constraint(
            solve_theta_partial_constraint, lowest_bound, 50, can_short_tracker, possible_delta_st, possible_cons_share,
            sigma_Y
        )
        a = Delta_s_t + theta_t
        invest = 1 - (a < 0) * (can_short_tracker < 1)  # not invest if a<0 and can not short
        invest[:, 1] = 0  # exclusion type
        switch_P_to_N = invest_tracker * (1 - invest) * (
                can_short_tracker < 1)  # switch to nonparti if type R&E & not investing this period
        switch_N_to_P = np.maximum(invest - invest_tracker,
                                   0)  # switch to parti if not investing before & investing this period
        switch_N_to_P[:, :3] = 0  # only applicable to the E type
        switch = switch_N_to_P + switch_P_to_N
        invest_tracker = invest_tracker + switch_N_to_P - switch_P_to_N
        d_eta_st = a * invest_tracker - theta_t

        # tau_info and V_hat has to change for the agents who switch (either P to N or vice versa)
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
        popu_can_short_t = np.sum(cohort_type_size * can_short_tracker)
        Phi_can_short_t = np.sum(can_short_tracker * f_c_ist * dt)
        short = pi_st < 0
        Phi_short_t = np.sum(short * f_c_ist * dt)
        popu_short_t = np.sum(cohort_type_size * short)

        rho_bar_t = np.sum(rho_i * f_c_ist) / np.sum(f_c_ist)
        rho_tilde_t = np.sum(rho_i * f_w_ist) / np.sum(f_w_ist)

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
        invest_mat[i] = invest_tracker[0]
        popu_short[i] = popu_short_t
        Phi_can_short[i] = Phi_can_short_t
        parti[i] = popu_parti_t

        for j in range(4):
            parti_age_group[i, j] = np.ma.average(invest_tracker[:, :, cutoffs_age[j + 1]:cutoffs_age[j]],
                                                  weights=cohort_type_size[:, :, cutoffs_age[j + 1]:cutoffs_age[j]])
            within_group = np.where((w_indiv_ist >= wealth_cutoffs[j]) * (w_indiv_ist < wealth_cutoffs[j + 1]))
            parti_wealth_group[i, j] = np.ma.average(invest_tracker[within_group],
                                                     weights=cohort_type_size[within_group])

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
        invest_mat,
        parti_age_group,
        parti_wealth_group,
    )


def simulate_mean_vola_mix_type(
        biasvec: np.ndarray,
        dZ: np.ndarray,
        dZ_SI: np.ndarray,
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
        cohort_type_size: np.ndarray,
        cutoffs_age: np.ndarray,
        Delta_s_t: np.ndarray,
        eta_st_eta_ss: np.ndarray,
        X: np.ndarray,
        d_eta_st: np.ndarray,
        invest_tracker: np.ndarray,
        can_short_tracker: np.ndarray,
        tau_info: np.ndarray,
        Vhat_vector: np.ndarray,
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
]:
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
    dR = np.zeros(Nt - keep_when)  # stores stock returns
    r = np.zeros(Nt - keep_when)  # interest rate
    theta = np.zeros(Nt - keep_when)  # market price of risk
    mu_S = np.zeros(Nt - keep_when)
    sigma_S = np.zeros(Nt - keep_when)
    beta = np.zeros(Nt - keep_when)
    Delta_bar = np.zeros(Nt - keep_when)
    Delta_bar_parti = np.zeros(
        (Nt - keep_when))  # consumption weighted estimation error of the stock market participants
    Delta_tilde_parti = np.zeros((Nt - keep_when))  # wealth weighted estimation error of the stock market participants
    parti = np.ones((Nt - keep_when))  # participation rate
    Phi_bar_parti_1 = np.ones((Nt - keep_when))
    Phi_tilde_parti = np.ones((Nt - keep_when))
    parti_age_group = np.ones((Nt - keep_when, 4))
    N_wealth_group = 4
    parti_wealth_group = np.ones((Nt - keep_when, N_wealth_group))
    wealth_groups = np.linspace(1, 0, N_wealth_group + 1)
    parti_age_wealth_group = np.ones((Nt - keep_when, 4, N_wealth_group))

    a_phi = (1 - phi ** 2)
    phi_sqr_a_phi = phi / np.sqrt(a_phi)
    a_phi_1 = 1 / a_phi
    sigma_Y_sq = sigma_Y ** 2

    append_init = np.ones((Ntype, Nconstraint, 1))
    invest_newborn = np.array([[[1], [0], [1], [1]]]) * np.ones((Ntype, Nconstraint, 1))
    can_short_newborn = np.array([[[1], [0], [0], [0]]]) * np.ones((Ntype, Nconstraint, 1))
    wealth_cutoffs = np.array([0, 1, 10, 100, 100000])

    for i in tqdm(range(Nt)):
        dZ_t = dZ[i]
        dZ_SI_t = dZ_SI[i]

        # new cohort born (age 0), get wealth transfer, observe, invest
        eta_st_eta_ss = eta_st_eta_ss * np.exp(
            (-0.5 * d_eta_st ** 2) * dt
            + d_eta_st * dZ_t
        )  # equation (15)

        # from equation (20) and the description below it
        # X_t = W_t * xi_t, is the sum of tax * X_s * eta_st_eta_ss * rho_cohort_type_short * dt, s<t;
        # X is the collection of all X_s, s<t.
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

        w_indiv_ist = f_w_ist / cohort_type_size * dt
        if i > 0:
            dR_t = mu_S_t * dt + sigma_S_t * dZ_t
        else:
            dR_t = 0

        # update beliefs
        V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')  # from eq(6)
        dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_t, dZ_SI_t,
                                            'N')  # from eq(9)
        V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
        dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t, dZ_SI_t,
                                            'P')
        dDelta_s_t = invest_tracker * dDelta_s_t_P + (
                1 - invest_tracker) * dDelta_s_t_N  # the participation decision of last time affects the updating pattern
        tau_info = np.append(tau_info[:, :, 1:], 0 * append_init, axis=2) + dt
        Vhat_vector = np.append(Vhat_vector[:, :, 1:], Vhat * append_init, axis=2)

        if i < Npre - 1:
            init_bias = (np.sum(biasvec[i + 1:]) + np.sum(dZ[:i + 1])) / T_hat
        else:
            init_bias = np.sum(dZ[i + 1 - Npre:i + 1]) / T_hat

        Delta_s_t = Delta_s_t[:, :, 1:] + dDelta_s_t[:, :, 1:]
        Delta_s_t = np.append(Delta_s_t, init_bias * append_init, axis=2)

        # find the market clearing theta, given beliefs and consumption shares of cohorts in the economy
        invest_tracker = np.append(invest_tracker[:, :, 1:], invest_newborn,
                                   axis=2)  # all cohorts that are still in the market, new born agents can participate by default
        can_short_tracker = np.append(can_short_tracker[:, :, 1:], can_short_newborn, axis=2)

        possible_cons_share = f_c_ist * dt * invest_tracker
        possible_delta_st = Delta_s_t * invest_tracker

        lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound
        theta_t = bisection_partial_constraint(
            solve_theta_partial_constraint, lowest_bound, 50, can_short_tracker, possible_delta_st, possible_cons_share,
            sigma_Y
        )
        a = Delta_s_t + theta_t
        invest = 1 - (a < 0) * (can_short_tracker < 1)  # not invest if a<0 and can not short
        invest[:, 1] = 0  # exclusion type
        switch_P_to_N = invest_tracker * (1 - invest) * (
                can_short_tracker < 1)  # switch to nonparti if type R&E & not investing this period
        switch_N_to_P = np.maximum(invest - invest_tracker,
                                   0)  # switch to parti if not investing before & investing this period
        switch_N_to_P[:, :3] = 0  # only applicable to the E type
        switch = switch_N_to_P + switch_P_to_N
        invest_tracker = invest_tracker + switch_N_to_P - switch_P_to_N
        d_eta_st = a * invest_tracker - theta_t

        # tau_info and V_hat has to change for the agents who switched to N
        Vhat_vector = np.append(V_st_P[:, :, 1:], Vhat * append_init, axis=2) * switch_P_to_N + \
                      np.append(V_st_N[:, :, 1:], Vhat * append_init, axis=2) * switch_N_to_P + \
                      Vhat_vector * (1 - switch)  # reset initial variance
        tau_info = dt * switch + tau_info * (1 - switch)  # reset clock

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
            Delta_bar[ii] = np.average(Delta_s_t, weights=cohort_type_size)
            Delta_bar_parti[ii] = Delta_bar_parti_t
            Delta_tilde_parti[ii] = Delta_tilde_parti_t
            mu_S[ii] = mu_S_t
            sigma_S[ii] = np.abs(sigma_S_t)  # stock vola = absolute value of sigma
            beta[ii] = beta_t
            parti[ii] = popu_parti_t
            Phi_bar_parti_1[ii] = 1 / fc_parti_t
            Phi_tilde_parti[ii] = fw_parti_t
            for l in range(N_wealth_group):
                within_group = np.where((w_indiv_ist >= wealth_cutoffs[l]) * (w_indiv_ist < wealth_cutoffs[l + 1]))
                parti_wealth_group[ii, l] = np.ma.average(invest_tracker[within_group],
                                                          weights=cohort_type_size[within_group])
            for j in range(4):
                invest_age = invest_tracker[:, :, cutoffs_age[j + 1]:cutoffs_age[j]]
                cohort_type_age = cohort_type_size[:, :, cutoffs_age[j + 1]:cutoffs_age[j]]
                parti_age_group[ii, j] = np.ma.average(invest_age, weights=cohort_type_age)
    dR_matrix = np.array([np.mean(dR / dt), np.std(dR / dt)])
    theta_matrix = np.array([np.mean(theta), np.std(theta)])
    r_matrix = np.array([np.mean(r), np.std(r)])
    mu_S_matrix = np.array([np.mean(mu_S), np.std(mu_S)])
    sigma_S_matrix = np.array([np.mean(sigma_S), np.std(sigma_S)])
    beta_matrix = np.array([np.mean(beta), np.std(beta)])

    Delta_bar_parti_matrix = np.array([np.mean(Delta_bar_parti), np.std(Delta_bar_parti)])
    Delta_tilde_bar_parti = Delta_tilde_parti - Delta_bar_parti
    Delta_tilde_bar_parti_matrix = np.array([np.mean(Delta_tilde_bar_parti), np.std(Delta_tilde_bar_parti)])
    Phi_tilde_bar_parti = Phi_tilde_parti * Phi_bar_parti_1
    Phi_bar_parti_1_matrix = np.array([np.mean(Phi_bar_parti_1 * sigma_Y), np.std(Phi_bar_parti_1 * sigma_Y)])
    Phi_tilde_parti_matrix = np.array([np.mean(Phi_tilde_parti), np.std(Phi_tilde_parti)])
    Phi_tilde_bar_parti_matrix = np.array(
        [np.mean(Phi_tilde_bar_parti * sigma_Y), np.std(Phi_tilde_bar_parti * sigma_Y)])
    Delta_Phi_tilde = Delta_tilde_bar_parti * Phi_tilde_parti
    Delta_Phi_tilde_matrix = np.array([np.mean(Delta_Phi_tilde), np.std(Delta_Phi_tilde)])
    parti_age_group_matrix = np.array(np.mean(parti_age_group, axis=0))
    Delta_bar_mat = np.tile(np.reshape(Delta_bar, (-1, 1)), (1, N_wealth_group))
    parti_wealth_group_mask = parti_wealth_group
    parti_wealth_group_matrix = np.array(np.nanmean(parti_wealth_group_mask, axis=0))
    cov_theta_z_Y = np.corrcoef(dZ[keep_when:], theta)[0, 1]
    cov_muS_z_Y = np.corrcoef(dZ[keep_when:], mu_S)[0, 1]
    cov_sigmaS_z_Y = np.corrcoef(dZ[keep_when:], sigma_S)[0, 1]
    cov_theta_z_SI = np.corrcoef(dZ_SI[keep_when:], theta)[0, 1]
    cov_parti_cons_share = np.corrcoef(parti, 1 / Phi_bar_parti_1)[0, 1]
    cov_parti_wealth_share = np.corrcoef(parti, Phi_tilde_parti)[0, 1]

    theta_save_matrix = np.array([
        Phi_bar_parti_1_matrix,
        Delta_bar_parti_matrix,
    ])

    sigma_S_save_matrix = np.array([
        Phi_tilde_parti_matrix,
        Phi_tilde_bar_parti_matrix,
        Delta_tilde_bar_parti_matrix,
        Delta_Phi_tilde_matrix,
    ])

    cov_save_matrix = np.array([
        cov_theta_z_Y,
        cov_muS_z_Y,
        cov_sigmaS_z_Y,
        cov_theta_z_SI,
        cov_parti_cons_share,
        cov_parti_wealth_share,
    ])

    cov_parti_matrix1 = np.array([])
    cov_parti_matrix2 = np.array([])
    cov_parti_matrix3 = np.array([])
    cov_parti_matrix4 = np.array([])
    windows = np.array([6, 24, 60])
    R_cumu = np.cumsum(dR)
    vola_cumu = np.cumsum(sigma_S ** 2)
    r_cumu = np.cumsum(r)
    for j, window in enumerate(windows):
        R_window = R_cumu[window:] - R_cumu[:-window]
        vola_window = np.sqrt((vola_cumu[window:] - vola_cumu[:-window]) / window)
        r_window = r_cumu[window:] - r_cumu[:-window]
        R_window_gap = np.reshape(R_window, (-1, window))[:, 0]
        vola_window_gap = np.reshape(vola_window, (-1, window))[:, 0]
        r_window_gap = np.reshape(r_window, (-1, window))[:, 0]
        parti_window = np.reshape(parti, (-1, window))[:, 0]
        r_reshape = np.reshape(r, (-1, window))
        erp_window_gap = R_window_gap - np.sum(r_reshape, axis=1)[:-1]
        corr1 = np.corrcoef(R_window_gap, parti_window[:-1])[0, 1]
        corr2 = np.corrcoef(r_window_gap, parti_window[:-1])[0, 1]
        corr3 = np.corrcoef(erp_window_gap, parti_window[:-1])[0, 1]
        corr4 = np.corrcoef(vola_window_gap, parti_window[:-1])[0, 1]
        cov_parti_matrix1 = np.append(cov_parti_matrix1, corr1)
        cov_parti_matrix2 = np.append(cov_parti_matrix2, corr2)
        cov_parti_matrix3 = np.append(cov_parti_matrix3, corr3)
        cov_parti_matrix4 = np.append(cov_parti_matrix4, corr4)
    cov_parti_matrix = np.array([
        cov_parti_matrix1,
        cov_parti_matrix2,
        cov_parti_matrix3,
        cov_parti_matrix4,
    ])

    return (
        dR_matrix,
        theta_matrix,
        r_matrix,
        mu_S_matrix,
        sigma_S_matrix,
        beta_matrix,
        theta_save_matrix,
        sigma_S_save_matrix,
        parti_age_group_matrix,
        parti_wealth_group_matrix,
        cov_save_matrix,
        parti,
        cov_parti_matrix,
    )
