import numpy as np
from typing import Tuple, List, Any
from src.stats import post_var, fadingmemo, dDelta_st_calculator, weighted_variance
from src.solver import bisection, solve_theta, find_the_rich, bisection_partial_constraint, \
    solve_theta_partial_constraint, find_the_rich_mix
from tqdm import tqdm
from numba import jit


# todo: use matrix calculation in the functions, calculate multiple paths per run
def simulate_cohorts_SI(
        # Y: np.ndarray,
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
        beta_cohort_type: np.ndarray,
        beta0: float,
        nu: float,
        Vhat: float,
        mu_Y: float,
        sigma_Y: float,
        tax: float,
        phi: float,
        T_hat: float,
        Npre: float,
        # Ninit: int,
        mode_trade: str,
        mode_learn: str,
        # cohort_size: np.ndarray,
        cohort_type_size: np.ndarray,
        cutoffs_age: np.ndarray,
        Delta_s_t: np.ndarray,
        eta_st_eta_ss: np.ndarray,
        X: np.ndarray,
        d_eta_st: np.ndarray,
        invest_tracker: np.ndarray,
        # can_short_tracker: np.ndarray,
        tau_info: np.ndarray,
        Vhat_vector: np.ndarray,
        need_f: str,
        need_Delta: str,
        need_pi: str,
        # top: float,
        # old_limit: float,
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
]:
    """"" Simulate the economy forward

    Args:
        biasvec (np.ndarray): pre-trading period shocks to form beliefs of the initial cohort, shape(Npre,)
        dZt (np.ndarray): random shocks, shape (Nt)
        Nt (int): number of periods in the simulation
        Nc (int): number of cohorts in the economy
        tau (np.ndarray): t-s, shape(Nt)
        # IntVec (np.ndarray): ~similar to consumption share, shape(Nc)
        dt (float): per unit of time
        rho (float): discount factor
        nu (float): rate of birth and death
        Vhat (float): initial variance
        mu_Y (float): as in eq(1), drift of aggregate output growth
        sigma_Y (float): as in eq(1), diffusion of aggregate output growth
        sigma_S (float): as in eq(26), diffusion of stock price 
        tax (float): as in eq(18), consumption share of the newborn cohort
        T_hat (float): pre-trading years
        Npre (float): pre-trading number of obs
        mode (str): versions of the model
        - from the cohort_builder function -
        f_st (np.ndarray): consumption share input
        eta_st_ss (np.ndarray): disagreement input
        eta_bar (np.ndarray): average disagreement input
        Delta_s_t (np.ndarray): bias for each cohort, shape(Nc)
        d_eta_st (np.ndarray): max(Delta_), shape(Nc)
        invest_tracker (np.ndarry): shape(Nt)
        tau_info: np.ndarray,
        good_time_simulate: np.ndarray,

    Returns:
        mu_S (np.ndarray): expected return under true measure at t, shape(Nt, )
        mu_S_s (np.ndarray): expected stock return each cohort, shape(Nt, Nc, )
        r (np.ndarray): interest rate, shape(Nt, )
        theta (np.ndarray): market price of risk, shape(Nt, )
        f (np.ndarray): consumption share over time, shape(Nt, Nc, )
        Delta (np.ndarray): bias over time, shape(Nt, Nc, )
        max (np.ndarray): max(-theta, delta_s_t) over time, shape(Nt, Nc, )
        pi (np.ndarray): portfolio choice over time, shape(Nt, Nc, )
        parti (np.ndarray): population that invest in stocks over time, shape(Nt, )
        f_parti (np.ndarray): aggregate consumption share over time conditional on invest in stocks, shape(Nt, )
        Delta_bar_parti (np.ndarray): aggregate consumption weighted bias over time conditional on invest in stocks, shape(Nt, )
        dR (np.ndarray): change of stock returns over time, shape(Nt, )
        w (np.ndarray): individual wealth, shape(Nt, Nc, )
        w_cohort (np.ndarray): cohort wealth = individual wealth * cohort size, shape(Nt, Nc, )
        age (np.ndarray):  average age participating in the stock market, shape(Nt, )
        n_parti (np.ndarray): number of cohorts participating in the stock market, shape(Nt, )
    """ ""
    # Initializing variables
    # cohort-type-specific terms:
    # max = np.zeros((Nt, Ntype, Nc))  # stores max(delta, -theta)
    # invest_mat = np.ones((Nt, Ntype, Nc))

    if need_f == 'True':
        f_c = np.zeros((Nt, Ntype, Nc), dtype=np.float16)  # evolution of cohort consumption share
        # f_w = np.zeros((Nt, Ntype, Nc))  # evolution of cohort wealth share
        # w_indiv_mat = np.zeros((Nt, Ntype, Nc))
    else:
        f_c = f_w = w_indiv_mat = 0

    if need_Delta == 'True':
        Delta = np.zeros((Nt, Ntype, Nc), dtype=np.float16)  # stores bias in beliefs
    else:
        Delta = 0

    if need_pi == 'True':
        pi = np.zeros((Nt, Ntype, Nc), dtype=np.float16)  # portfolio choices
    else:
        pi = 0

    quartiles = np.array([1, 0.75, 0.5, 0.25, 0])
    Phi_parti = np.ones((Nt))
    invest_mat = np.ones((Nt, Ntype, Nc), dtype=int)
    parti_wealth_group = np.ones((Nt, 4))
    parti_age_group = np.ones((Nt, 4))

    # if mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
    #     popu_can_short= np.zeros((Nt))
    #     popu_short = np.zeros((Nt))
    #     Phi_can_short = np.zeros((Nt))
    #     Phi_short = np.zeros((Nt))
    # else:
    #     popu_can_short = 0
    #     popu_short = 0
    #     Phi_can_short = 0
    #     Phi_short = 0

    # equilibrium terms:
    dR = np.zeros(Nt)  # stores stock returns
    r = np.zeros(Nt)  # interest rate
    theta = np.zeros(Nt)  # market price of risk
    mu_S = np.zeros(Nt)
    sigma_S = np.zeros(Nt)
    beta = np.zeros(Nt)
    Delta_bar_parti = np.zeros((Nt))  # consumption weighted estimation error of the stock market participants
    Delta_tilde_parti = np.zeros((Nt))  # wealth weighted estimation error of the stock market participants
    parti = np.ones((Nt))  # participation rate

    # upperbound = np.arange(10,55,5)
    # theta_t_matrix = np.zeros((Nt, len(upperbound)))

    dR_t = 0
    # age = np.zeros(Nt)
    # n_parti = np.zeros(Nt)

    a_phi = (1 - phi ** 2)
    phi_sqr_a_phi = phi / np.sqrt(a_phi)
    a_phi_1 = 1 / a_phi
    sigma_Y_sq = sigma_Y ** 2

    # popu_can_short = 0
    # popu_short = 0

    mu_S_t = 0
    sigma_S_t = 0
    r_t = 0
    pi_st = 0
    w_indiv_ist = 0

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
        )  # equation (11)

        X_parts = tax * np.exp(-tax * tau) * X * beta_cohort_type * eta_st_eta_ss * dt  # equation (18)
        X_t = np.sum(X_parts) / (1 - tax * dt)  # equation (18)  # dividing by (1-tax*dt) keeps sum(f_st*dt) at 1
        # eta_bar_t = np.sum(eta_bar_parts)

        eta_st_eta_ss = np.append(eta_st_eta_ss[:, 1:], np.ones((Ntype, 1)), axis=1)
        X = np.append(X[:, 1:], X_t * alpha_i, axis=1)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort
        # todo: eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
        #  eta_bar_t is the denominator; it creates issues if too close to 0
        #  so I rescale eta_bar to keep it away from 0, without changing f_st

        f_w_ist = X_parts / X_t / dt
        f_w_ist = np.append(f_w_ist[:, 1:], tax * alpha_i, axis=1)

        beta_t = np.sum(f_w_ist * beta_i) * dt
        f_c_ist = f_w_ist * beta_i / beta_t

        # Wealth
        # w_t = Y[i] / beta_t  # total wealth at time t
        w_indiv_ist = f_w_ist / cohort_type_size
        # if i == 0:
        #     w_ist = w_t * f_w_ist  # cohort
        #     w_indiv_ist = w_ist / cohort_type_size * dt  # indiv
        # else:
        dR_t = mu_S_t * dt + sigma_S_t * dZ_t  # realized stock return, mu_t^Sdt + sigma_t^Sdz_t
        #     dw_indiv_ist = ((r_t - rho_i) + pi_st * (
        #             mu_S_t - r_t)) * w_indiv_ist * dt + w_indiv_ist * pi_st * sigma_S_t * dZ_t  # r_t, theta_t, pi_st from last loop, dZ_t just realized
        #     w_indiv_ist = w_indiv_ist + dw_indiv_ist
        #     w_ist = w_indiv_ist * cohort_type_size / dt
        #     adjust_scale = w_t * (1 / dt - tax) / np.sum(w_ist[:, 1:])
        #     w_ist = np.append(w_ist[:, 1:] * adjust_scale, np.ones((2, 1)) * w_t * tax, axis=1)
        #     w_indiv_ist = np.append(w_indiv_ist[:, 1:] * adjust_scale, np.ones((2, 1)) * w_t * tax / nu, axis=1)
        # # w_indiv_mat[i] = w_indiv_ist

        # update beliefs
        if mode_trade == 'complete':  # everyone is P
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t, dZ_SI_t,
                                              'P')
            tau_info = tau

        elif mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
            V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')
            dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_t,
                                                dZ_SI_t,
                                                'N')  # from eq(5)
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t,
                                                dZ_SI_t,
                                                'P')
            dDelta_s_t = invest_tracker * dDelta_s_t_P + (
                    1 - invest_tracker) * dDelta_s_t_N  # the participation decision of last time affects the updating pattern
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
                                   axis=1)  # all cohorts that are still in the market, 1 by default

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
                                      1 - switch_P_to_N)  # reset initial variance
                tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock

            elif mode_learn == 'reentry':  # agents stay as type P even if constrained
                possible_cons_share = f_c_ist * dt
                possible_delta_st = Delta_s_t
                lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
                theta_t = bisection(
                    solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
                )  # solve for theta
                # theta_t_vector = bisection_test_multi_equi(
                #     solve_theta, lowest_bound, upperbound, possible_cons_share, possible_delta_st, sigma_Y
                # )  # solve for theta
                # theta_t = theta_t_vector[-1]
                a = Delta_s_t + theta_t
                invest = (a > 0)
                switch_P_to_N = invest_tracker * (1 - invest)
                switch_N_to_P = np.maximum(invest - invest_tracker, 0)
                # tau_info and V_hat has to change for the agents who switched to N
                switch = switch_N_to_P + switch_P_to_N
                invest_tracker = invest

                Vhat_vector = np.append(V_st_P[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1) * switch_P_to_N + np.append(
                    V_st_N[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1) * switch_N_to_P + Vhat_vector * (
                                      1 - switch)  # reset initial variance
                tau_info = dt * switch + tau_info * (1 - switch)  # reset clock

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
            # age_t = np.sum(cohort_size * tau * invest_tracker)
            # n_parti_t = np.sum(invest_tracker) / Nc

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
            pi_st = (d_eta_st + theta_t) / sigma_S_t
            # age_t = np.sum(cohort_size * tau * invest)
            # n_parti_t = np.sum(invest) / Nc

        # elif mode_trade == 'partial_constraint_rich':  #todo: edit the cases, think hard about invest_tracker, participation rate, etc.
        #     can_short_tracker = np.append(can_short_tracker[1:], 0)
        #     if mode_learn == 'disappointment':
        #         possible_cons_share = f_st * dt * invest_tracker
        #         possible_delta_st = Delta_s_t * invest_tracker
        #         indiv_w_possible = w_indiv_st * invest_tracker
        #         cohort_size_possible = cohort_size * invest_tracker
        #         wealth_cutoff = find_the_rich(indiv_w_possible, cohort_size_possible,
        #                                       top)  # find the cohorts that make the richest 1% pupolation in the current period that are still in the market
        #         can_short = indiv_w_possible >= wealth_cutoff  # these cohorts can short in this period
        #         can_short_tracker = (can_short_tracker + can_short >= 1)
        #
        #         lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
        #         theta_t = bisection_partial_constraint(
        #             solve_theta_partial_constraint, -50, 50, can_short_tracker, possible_delta_st, possible_cons_share,
        #             sigma_Y
        #         )
        #         a = Delta_s_t + theta_t
        #         invest = 1 - (a < 0) * (can_short_tracker < 1)
        #         switch_P_to_N = invest_tracker * (1 - invest)
        #         invest_tracker = invest * invest_tracker  # update invest tracker
        #
        #         # tau_info and V_hat has to change for the agents who switched to N
        #         Vhat_vector = np.append(V_st_P[1:], Vhat) * switch_P_to_N + Vhat_vector * (1 - switch_P_to_N)  # reset initial variance
        #         tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock
        #
        #     elif mode_learn == 'reentry':
        #         possible_cons_share = f_st * dt
        #         possible_delta_st = Delta_s_t
        #         indiv_w_possible = w_indiv_st
        #         cohort_size_possible = cohort_size
        #         wealth_cutoff = find_the_rich(indiv_w_possible, cohort_size_possible,
        #                                       top)  # find the cohorts that make the richest 1% pupolation in the current period that are still in the market
        #         can_short = indiv_w_possible >= wealth_cutoff  # these cohorts can short in this period
        #         can_short_tracker = (can_short_tracker + can_short >= 1)
        #
        #         lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
        #         theta_t = bisection_partial_constraint(
        #             solve_theta_partial_constraint, -50, 50, can_short_tracker, possible_delta_st, possible_cons_share,
        #             sigma_Y
        #         )
        #         a = Delta_s_t + theta_t
        #         invest = 1 - (a < 0) * (can_short_tracker < 1)
        #         switch_P_to_N = invest_tracker * (1 - invest)
        #         switch_N_to_P = np.maximum(invest - invest_tracker, 0)
        #         # tau_info and V_hat has to change for the agents who switched to N
        #         switch = switch_N_to_P + switch_P_to_N
        #         invest_tracker = invest
        #
        #         Vhat_vector = np.append(V_st_P[1:], Vhat) * switch_P_to_N + np.append(V_st_N[1:], Vhat) * switch_N_to_P + Vhat_vector * (1 - switch)  # reset initial variance
        #         tau_info = dt * switch + tau_info * (1 - switch)  # reset clock
        #
        #     else:
        #         print('mode_learn not found')
        #         exit()
        #     d_eta_st = a * invest_tracker - theta_t
        #     invest_fst = invest_tracker * f_st * dt
        #     popu_parti_t = np.sum(cohort_size * invest_tracker)
        #     f_parti_t = np.sum(invest_fst)
        #     Delta_bar_parti_t = np.sum(Delta_s_t * invest_fst) / f_parti_t
        #     pi_st = (d_eta_st + theta_t) / sigma_S
        #     age_t = np.sum(cohort_size * tau * invest_tracker)
        #     n_parti_t = np.sum(invest_tracker) / Nc
        #
        # elif mode_trade == 'partial_constraint_old':
        #     can_short_tracker = np.append(can_short_tracker[:, 1:], np.zeros((Ntype, 1)), axis=1)
        #
        #     if mode_learn == 'disappointment':
        #         possible_cons_share = f_c_ist * dt * invest_tracker
        #         possible_delta_st = Delta_s_t * invest_tracker
        #         can_short_possible = (tau >= old_limit)
        #         can_short_tracker = can_short_possible * invest_tracker
        #
        #         lowest_bound = lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound for theta among active investors
        #         theta_t = bisection_partial_constraint(
        #             solve_theta_partial_constraint, lowest_bound, 50, can_short_tracker, possible_delta_st, possible_cons_share,
        #             sigma_Y
        #         )
        #         a = Delta_s_t + theta_t
        #         invest = 1 - (a < 0) * (can_short_tracker < 1)
        #         switch_P_to_N = invest_tracker * (1 - invest)
        #         invest_tracker = invest * invest_tracker  # update invest tracker
        #
        #         # tau_info and V_hat has to change for the agents who switched to N
        #         Vhat_vector = np.append(V_st_P[1:], Vhat) * switch_P_to_N + Vhat_vector * (1 - switch_P_to_N)  # reset initial variance
        #         tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock
        #
        #     elif mode_learn == 'reentry':
        #         possible_cons_share = f_c_ist * dt
        #         possible_delta_st = Delta_s_t
        #         can_short_tracker = (tau >= old_limit)
        #
        #         lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
        #         theta_t = bisection_partial_constraint(
        #             solve_theta_partial_constraint, lowest_bound, 50, can_short_tracker, possible_delta_st, possible_cons_share,
        #             sigma_Y
        #         )
        #         a = Delta_s_t + theta_t
        #         invest = 1 - (a < 0) * (can_short_tracker < 1)
        #         switch_P_to_N = invest_tracker * (1 - invest)
        #         switch_N_to_P = np.maximum(invest - invest_tracker, 0)
        #         # tau_info and V_hat has to change for the agents who switched to N
        #         switch = switch_N_to_P + switch_P_to_N
        #         invest_tracker = invest
        #
        #         Vhat_vector = np.append(V_st_P[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1) * switch_P_to_N + np.append(V_st_N[:, 1:], Vhat* np.ones((Ntype, 1)), axis=1) * switch_N_to_P + Vhat_vector * (1 - switch)  # reset initial variance
        #         tau_info = dt * switch + tau_info * (1 - switch)  # reset clock
        #     else:
        #         print('mode_learn not found')
        #         exit()
        #
        #     d_eta_st = a * invest_tracker - theta_t
        #     invest_fc_st = invest_tracker * f_c_ist * dt
        #     invest_fw_st = invest_tracker * f_w_ist * dt
        #     popu_parti_t = np.sum(cohort_type_size * invest_tracker)
        #     fc_parti_t = np.sum(invest_fc_st)
        #     fw_parti_t = np.sum(invest_fw_st)
        #     Delta_bar_parti_t = np.sum(Delta_s_t * invest_fc_st) / fc_parti_t
        #     Delta_tilde_parti_t = np.sum(Delta_s_t * invest_fw_st) / fw_parti_t
        #     sigma_S_t = fw_parti_t * (theta_t + Delta_tilde_parti_t)
        #     pi_st = (d_eta_st + theta_t) / sigma_S_t
        #     # age_t = np.sum(cohort_size * tau * invest_tracker)
        #     # n_parti_t = np.sum(invest_tracker) / Nc
        #     popu_can_short_t = np.sum(cohort_type_size * can_short_tracker)
        #     Phi_can_short_t = np.sum(can_short_tracker * f_c_ist * dt)
        #     short = pi_st < 0
        #     Phi_short_t = np.sum(short * f_c_ist * dt)
        #     popu_short_t = np.sum(cohort_type_size * short)
        else:
            print('mode_trade not found')
            exit()

        rho_bar_t = np.sum(rho_i * f_c_ist) / np.sum(f_c_ist)
        rho_tilde_t = np.sum(rho_i * f_w_ist) / np.sum(f_w_ist)

        r_t = (
                nu - tax * beta0 / beta_t
                + rho_bar_t
                + mu_Y
                - sigma_Y * theta_t
        )

        mu_S_t = tax - nu - rho_tilde_t + beta_t + sigma_S_t * theta_t + r_t

        # store the results
        # storing the values takes a lot of time
        dR[i] = dR_t  # realized return from t-1 to t
        theta[i] = theta_t
        r[i] = r_t
        Delta_bar_parti[i] = Delta_bar_parti_t
        Delta_tilde_parti[i] = Delta_tilde_parti_t
        mu_S[i] = mu_S_t
        sigma_S[i] = sigma_S_t
        if sigma_S_t < 0:
            print('negative vola')
        beta[i] = beta_t
        # w[i, :] = w_st
        # age[i] = age_t
        # n_parti[i] = n_parti_t
        if need_f == 'True':
            f_c[i, :] = f_c_ist
            # f_w[i, :] = f_w_ist
        if need_Delta == 'True':
            Delta[i, :] = Delta_s_t
        if need_pi == 'True':
            pi[i, :] = pi_st
        if mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
            Phi_parti[i] = fc_parti_t
            parti[i] = popu_parti_t
            invest_mat[i] = invest_tracker
            wealth_cutoffs = find_the_rich_mix(
                w_indiv_ist,
                cohort_type_size,
                quartiles,
            )
            for j in range(4):
                parti_age_group[i, j] = np.average(invest_tracker[:, cutoffs_age[j + 1]:cutoffs_age[j]],
                                                   weights=cohort_type_size[:, cutoffs_age[j + 1]:cutoffs_age[j]])
                within_group = (w_indiv_ist >= wealth_cutoffs[j]) * (w_indiv_ist < wealth_cutoffs[j + 1])
                parti_wealth_group[i, j] = np.sum(invest_tracker * within_group * cohort_type_size) / \
                                           np.sum(within_group * cohort_type_size)
        # if mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
        #     popu_can_short[i] = popu_can_short_t
        #     popu_short[i] = popu_short_t
        #     Phi_can_short[i] = Phi_can_short_t
        #     Phi_short[i] = Phi_short_t
        # switch_P_to_N_ts[i] = np.sum(switch_P_to_N)
        # switch_N_to_P_ts[i] = np.sum(switch_N_to_P)

    return (
        r,
        theta,
        f_c,
        # f_w,
        Delta,
        # max,
        pi,
        parti,
        Phi_parti,
        Delta_bar_parti,
        Delta_tilde_parti,
        dR,
        mu_S,
        sigma_S,
        beta,
        # w,
        # age,
        # n_parti,
        invest_mat,
        parti_age_group,
        parti_wealth_group,
        # popu_can_short,
        # popu_short,
        # Phi_can_short,
        # Phi_short,
        # w_indiv_mat,
    )


def simulate_cohorts_mean_vola(
        # Y: np.ndarray,
        biasvec: np.ndarray,
        dZ: np.ndarray,
        dZ_SI: np.ndarray,
        Nt: int,
        # Nc: int,
        tau: np.ndarray,
        dt: float,
        Ntype: int,
        rho_i: np.ndarray,
        alpha_i: np.ndarray,
        beta_i: np.ndarray,
        beta_cohort_type: np.ndarray,
        beta0: float,
        nu: float,
        Vhat: float,
        mu_Y: float,
        sigma_Y: float,
        tax: float,
        phi: float,
        T_hat: float,
        Npre: float,
        # Ninit: int,
        mode_trade: str,
        mode_learn: str,
        # cohort_size: np.ndarray,
        cohort_type_size: np.ndarray,
        cutoffs_age: np.ndarray,
        Delta_s_t: np.ndarray,
        eta_st_eta_ss: np.ndarray,
        X: np.ndarray,
        d_eta_st: np.ndarray,
        invest_tracker: np.ndarray,
        # can_short_tracker: np.ndarray,
        tau_info: np.ndarray,
        Vhat_vector: np.ndarray,
        # top: float,
        # old_limit: float,
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
    """"" Simulate the economy forward

    Args:
        biasvec (np.ndarray): pre-trading period shocks to form beliefs of the initial cohort, shape(Npre,)
        dZt (np.ndarray): random shocks, shape (Nt)
        Nt (int): number of periods in the simulation
        Nc (int): number of cohorts in the economy
        tau (np.ndarray): t-s, shape(Nt)
        # IntVec (np.ndarray): ~similar to consumption share, shape(Nc)
        dt (float): per unit of time
        rho (float): discount factor
        nu (float): rate of birth and death
        Vhat (float): initial variance
        mu_Y (float): as in eq(1), drift of aggregate output growth
        sigma_Y (float): as in eq(1), diffusion of aggregate output growth
        sigma_S (float): as in eq(26), diffusion of stock price 
        tax (float): as in eq(18), consumption share of the newborn cohort
        T_hat (float): pre-trading years
        Npre (float): pre-trading number of obs
        mode (str): versions of the model
        - from the cohort_builder function -
        f_st (np.ndarray): consumption share input
        eta_st_ss (np.ndarray): disagreement input
        eta_bar (np.ndarray): average disagreement input
        Delta_s_t (np.ndarray): bias for each cohort, shape(Nc)
        d_eta_st (np.ndarray): max(Delta_), shape(Nc)
        invest_tracker (np.ndarry): shape(Nt)
        tau_info: np.ndarray,
        good_time_simulate: np.ndarray,

    Returns:
        mu_S (np.ndarray): expected return under true measure at t, shape(Nt, )
        mu_S_s (np.ndarray): expected stock return each cohort, shape(Nt, Nc, )
        r (np.ndarray): interest rate, shape(Nt, )
        theta (np.ndarray): market price of risk, shape(Nt, )
        f (np.ndarray): consumption share over time, shape(Nt, Nc, )
        Delta (np.ndarray): bias over time, shape(Nt, Nc, )
        max (np.ndarray): max(-theta, delta_s_t) over time, shape(Nt, Nc, )
        pi (np.ndarray): portfolio choice over time, shape(Nt, Nc, )
        parti (np.ndarray): population that invest in stocks over time, shape(Nt, )
        f_parti (np.ndarray): aggregate consumption share over time conditional on invest in stocks, shape(Nt, )
        Delta_bar_parti (np.ndarray): aggregate consumption weighted bias over time conditional on invest in stocks, shape(Nt, )
        dR (np.ndarray): change of stock returns over time, shape(Nt, )
        w (np.ndarray): individual wealth, shape(Nt, Nc, )
        w_cohort (np.ndarray): cohort wealth = individual wealth * cohort size, shape(Nt, Nc, )
        age (np.ndarray):  average age participating in the stock market, shape(Nt, )
        n_parti (np.ndarray): number of cohorts participating in the stock market, shape(Nt, )
    """ ""
    # Initializing variables
    keep_when = int(200 / dt)
    # Phi_parti = np.zeros((Nt - keep_when))  # consumption share of the stock market participants
    # Phi_parti_1_matrix = np.zeros((Nt - keep_when))
    # # parti = np.zeros((Nt - keep_when))  # participation rate
    # popu_age = np.zeros((Nt - keep_when, n_age_groups))
    # # belief_age = np.zeros((Nt - keep_when, n_age_groups))
    # wealthshare_age = np.zeros((Nt - keep_when, n_age_groups))
    # cov_save = np.zeros((Nt - keep_when, 3))
    # # var_save = np.zeros((Nt - keep_when, 4))
    # Delta_popu_parti = np.zeros((Nt - keep_when))

    # if mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
    #     popu_can_short = np.zeros((Nt - keep_when))
    #     popu_short = np.zeros((Nt - keep_when))
    #     Phi_short = np.zeros((Nt - keep_when))
    #     Phi_can_short = np.zeros((Nt - keep_when))
    # else:
    #     popu_can_short = 0
    #     popu_short = 0
    #     Phi_can_short = 0
    #     Phi_short = 0

    # equilibrium terms:
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
    ave_age_wealth_group = np.ones((Nt - keep_when, N_wealth_group))
    ave_belief_wealth_group = np.ones((Nt - keep_when, N_wealth_group))
    wealth_groups = np.linspace(1, 0, N_wealth_group+1)
    parti_age_wealth_group = np.ones((Nt - keep_when, 4, N_wealth_group))
    # upperbound = np.arange(10,55,5)
    # theta_t_matrix = np.zeros((Nt, len(upperbound)))
    # age = np.zeros(Nt)
    # n_parti = np.zeros(Nt)

    a_phi = (1 - phi ** 2)
    phi_sqr_a_phi = phi / np.sqrt(a_phi)
    a_phi_1 = 1 / a_phi
    sigma_Y_sq = sigma_Y ** 2
    #
    # popu_can_short = 0
    # popu_short = 0
    dR_t = 0
    mu_S_t = 0
    sigma_S_t = 0
    r_t = 0
    pi_st = 0
    w_indiv_ist = 0
    parti_window = int(3 / dt)
    wealth_cutoffs = np.array([0, 1, 10, 100, 100000])

    for i in tqdm(range(Nt)):
        dZ_t = dZ[i]
        dZ_SI_t = dZ_SI[i]

        # new cohort born (age 0), get wealth transfer, observe, invest
        eta_st_eta_ss = eta_st_eta_ss * np.exp(
            (-0.5 * d_eta_st ** 2) * dt
            + d_eta_st * dZ_t
        )  # equation (11)

        X_parts = tax * np.exp(-tax * tau) * X * beta_cohort_type * eta_st_eta_ss * dt  # equation (18)
        X_t = np.sum(X_parts) / (1 - tax * dt)  # equation (18)  # dividing by (1-tax*dt) keeps sum(f_st*dt) at 1
        # eta_bar_t = np.sum(eta_bar_parts)

        eta_st_eta_ss = np.append(eta_st_eta_ss[:, 1:], np.ones((Ntype, 1)), axis=1)
        X = np.append(X[:, 1:], X_t * np.ones((1, 1)), axis=1)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort

        f_w_ist = X_parts / X_t / dt
        f_w_ist = np.append(f_w_ist[:, 1:], tax * alpha_i, axis=1)

        beta_t = np.sum(f_w_ist * beta_i) * dt
        f_c_ist = f_w_ist * beta_i / beta_t

        w_indiv_ist = f_w_ist / cohort_type_size * dt
        c_indiv_ist = f_c_ist / cohort_type_size * dt

        # Wealth
        # w_t = Y[i] / beta_t  # total wealth at time t
        # if i == 0:
        #     w_ist = w_t * f_w_ist  # cohort
        #     w_indiv_ist = w_ist / cohort_type_size * dt  # indiv
        # else:
        dR_t = mu_S_t * dt + sigma_S_t * dZ_t  # realized stock return, mu_t^Sdt + sigma_t^Sdz_t
        #     dw_indiv_ist = ((r_t - rho_i) + pi_st * (
        #             mu_S_t - r_t)) * w_indiv_ist * dt + w_indiv_ist * pi_st * sigma_S_t * dZ_t  # r_t, theta_t, pi_st from last loop, dZ_t just realized
        #     w_indiv_ist = w_indiv_ist + dw_indiv_ist
        #     w_ist = w_indiv_ist * cohort_type_size / dt
        #     adjust_scale = w_t * (1 / dt - tax) / np.sum(w_ist[:, 1:])
        #     w_ist = np.append(w_ist[:, 1:] * adjust_scale, np.ones((2, 1)) * w_t * tax, axis=1)
        #     w_indiv_ist = np.append(w_indiv_ist[:, 1:] * adjust_scale, np.ones((2, 1)) * w_t * tax / nu, axis=1)

        # update beliefs
        if mode_trade == 'complete':  # everyone is P
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t, dZ_SI_t,
                                              'P')
            tau_info = tau

        elif mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
            V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')
            dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_t,
                                                dZ_SI_t,
                                                'N')  # from eq(5)
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t,
                                                dZ_SI_t,
                                                'P')
            dDelta_s_t = invest_tracker * dDelta_s_t_P + (
                    1 - invest_tracker) * dDelta_s_t_N  # the participation decision of last time affects the
            # updating pattern
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
                                   axis=1)  # all cohorts that are still in the market, 1 by default

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
                                      1 - switch_P_to_N)  # reset initial variance
                tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock

            elif mode_learn == 'reentry':  # agents stay as type P even if constrained
                possible_cons_share = f_c_ist * dt
                possible_delta_st = Delta_s_t
                lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
                theta_t = bisection(
                    solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
                )  # solve for theta
                # theta_t_vector = bisection_test_multi_equi(
                #     solve_theta, lowest_bound, upperbound, possible_cons_share, possible_delta_st, sigma_Y
                # )  # solve for theta
                # theta_t = theta_t_vector[-1]
                a = Delta_s_t + theta_t
                invest = (a > 0)
                switch_P_to_N = invest_tracker * (1 - invest)
                switch_N_to_P = np.maximum(invest - invest_tracker, 0)
                # tau_info and V_hat has to change for the agents who switched to N
                switch = switch_N_to_P + switch_P_to_N
                invest_tracker = invest

                Vhat_vector = np.append(V_st_P[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1) * switch_P_to_N + np.append(
                    V_st_N[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1) * switch_N_to_P + Vhat_vector * (
                                      1 - switch)  # reset initial variance
                tau_info = dt * switch + tau_info * (1 - switch)  # reset clock

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
            # age_t = np.sum(cohort_size * tau * invest_tracker)
            # n_parti_t = np.sum(invest_tracker) / Nc

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
        rho_tilde_t = np.sum(rho_i * f_w_ist) / np.sum(f_w_ist)
        r_t = (
                nu - tax * beta0 / beta_t
                + rho_bar_t
                + mu_Y
                - sigma_Y * theta_t
        )
        mu_S_t = tax - nu - rho_tilde_t + beta_t + sigma_S_t * theta_t + r_t

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
            sigma_S[ii] = sigma_S_t
            beta[ii] = beta_t
            parti[ii] = popu_parti_t
            if sigma_S_t < 0:
                print('negative vola')
            if mode_trade == 'w_constraint':
                Phi_bar_parti_1[ii] = 1 / fc_parti_t
                Phi_tilde_parti[ii] = fw_parti_t
                # if np.mod(ii, parti_window) == 0:
                #     wealth_cutoffs = find_the_rich_mix(
                #         w_indiv_ist[:, parti_window:],
                #         # c_indiv_ist,
                #         cohort_type_size[:, parti_window:],
                #         wealth_groups,
                #     )
                #     print(wealth_cutoffs)
                #     mark_where = np.zeros((Ntype, Nt-parti_window))
                #     mark = ii
                #     for l in range(N_wealth_group):
                #         within_group = np.where((w_indiv_ist[:, parti_window:] >= wealth_cutoffs[l]) * (w_indiv_ist[:, parti_window:] < wealth_cutoffs[l + 1]))
                #         mark_where[within_group] = l + 1
                #
                # for l in range(N_wealth_group):
                #     within_group = np.where(
                #         mark_where == l + 1
                #     )
                #     if mark == ii:
                #         invest_tracker_move = invest_tracker[:, parti_window:]
                #         cohort_type_size_move = cohort_type_size[:, parti_window:]
                #     else:
                #         invest_tracker_move = invest_tracker[:, parti_window - (ii - mark):-(ii - mark)]
                #         cohort_type_size_move = cohort_type_size[:, parti_window - (ii - mark):-(ii - mark)]
                #     parti_wealth_group[ii, l] = np.average(invest_tracker_move[within_group],
                #                                            weights=cohort_type_size_move[within_group]
                #                                            )
                for l in range(N_wealth_group):
                    within_group = np.where((w_indiv_ist >= wealth_cutoffs[l]) * (w_indiv_ist < wealth_cutoffs[l + 1]))
                    # within_group = np.where((c_indiv_ist >= wealth_cutoffs[l]) * (c_indiv_ist < wealth_cutoffs[l + 1]))
                    # parti_wealth_group[ii, l] = np.sum(invest_tracker * within_group * cohort_type_size) / \
                    #                             np.sum(within_group * cohort_type_size)
                    parti_wealth_group[ii, l] = np.average(invest_tracker[within_group],
                                                           weights=cohort_type_size[within_group]
                                                           )
                    # ave_age_wealth_group[ii, l] = np.average(tau_mat[within_group],
                    #                                        weights=cohort_type_size[within_group]
                    #                                        )
                    # ave_belief_wealth_group[ii, l] = np.average(Delta_s_t[within_group],
                    #                                        weights=cohort_type_size[within_group]
                    #                                        )
                for j in range(4):
                    invest_age = invest_tracker[:, cutoffs_age[j + 1]:cutoffs_age[j]]
                    # w_indiv_ist_age = w_indiv_ist[:, cutoffs_age[j + 1]:cutoffs_age[j]]
                    cohort_type_age = cohort_type_size[:, cutoffs_age[j + 1]:cutoffs_age[j]]
                    parti_age_group[ii, j] = np.average(invest_age,
                                                        weights=cohort_type_age)
                    # wealth_cutoffs_age_group = find_the_rich_mix(
                    #     w_indiv_ist_age,
                    #     cohort_type_age,
                    #     wealth_groups,
                    # )
                    # for jj in range(N_wealth_group):
                    #     within_group = (w_indiv_ist_age >= wealth_cutoffs_age_group[jj]) * (w_indiv_ist_age < wealth_cutoffs_age_group[jj + 1])
                    #     total = within_group * cohort_type_age
                    #     parti_age_wealth_group[ii, j, jj] = np.sum(invest_age * total) / np.sum(total)
            # if mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
            #     Phi_parti[ii] = fc_parti_t
    #         if mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
    #             popu_can_short[ii] = popu_can_short_t
    #             popu_short[ii] = popu_short_t
    #             Phi_can_short[ii] = Phi_can_short_t
    #             Phi_short[ii] = Phi_short_t
    #         # parti_rate = invest_tracker * cohort_size
    #         # belief = (Delta_s_t * sigma_Y + mu_Y)
    #         for j in range(4):
    #             if mode_trade != 'complete':
    #                 popu_age[ii, j] = np.average(invest_tracker[cutoffs[j + 1]:cutoffs[j]], weights=cohort_size[cutoffs[j + 1]:cutoffs[j]])
    #             wealthshare_age[ii, j] = np.sum(f_st[cutoffs[j + 1]:cutoffs[j]] * dt)
    #         # var_save[ii] = [var_cons_cohort_parti_t, var_cons_indiv_parti_t,
    #         #                var_Delta_cohort_parti_t, var_Delta_indiv_parti_t]
    # # var_save_matrix = np.mean(var_save, axis=0)

    dR_matrix = np.array([np.mean(dR/dt), np.std(dR/dt)])
    theta_matrix = np.array([np.mean(theta), np.std(theta)])
    r_matrix = np.array([np.mean(r), np.std(r)])
    mu_S_matrix = np.array([np.mean(mu_S), np.std(mu_S)])
    sigma_S_matrix = np.array([np.mean(sigma_S), np.std(sigma_S)])
    beta_matrix = np.array([np.mean(beta), np.std(beta)])

    sigma_Delta_bar_parti_matrix = np.array([np.mean(sigma_Y * Delta_bar_parti), np.std(sigma_Y * Delta_bar_parti)])
    Delta_tilde_bar_parti = Delta_tilde_parti - Delta_bar_parti
    Delta_tilde_bar_parti_matrix = np.array([np.mean(Delta_tilde_bar_parti), np.std(Delta_tilde_bar_parti)])
    Phi_tilde_bar_parti = Phi_tilde_parti * Phi_bar_parti_1
    Phi_bar_parti_1_matrix = np.array([np.mean(Phi_bar_parti_1), np.std(Phi_bar_parti_1)])
    Phi_tilde_parti_matrix = np.array([np.mean(Phi_tilde_parti), np.std(Phi_tilde_parti)])
    Phi_tilde_bar_parti_matrix = np.array([np.mean(Phi_tilde_bar_parti), np.std(Phi_tilde_bar_parti)])
    Delta_Phi_tilde = Delta_tilde_bar_parti * Phi_tilde_parti
    Delta_Phi_tilde_matrix = np.array([np.mean(Delta_Phi_tilde), np.std(Delta_Phi_tilde)])
    parti_age_group_matrix = np.array(np.mean(parti_age_group, axis=0))
    Delta_bar_mat = np.tile(np.reshape(Delta_bar, (-1, 1)), (1, N_wealth_group))
    # parti_wealth_group_mask = np.ma.masked_where((Delta_bar_mat >= 0.05) | (Delta_bar_mat <= -0.05), parti_wealth_group)
    parti_wealth_group_mask = parti_wealth_group
    parti_wealth_group_matrix = np.array(np.nanmean(parti_wealth_group_mask, axis=0))
    # parti_age_wealth_group_matrix = np.array(np.nanmean(parti_age_wealth_group, axis=0))
    cov_theta_z_Y = np.corrcoef(dZ[keep_when:], theta)[0, 1]
    cov_muS_z_Y = np.corrcoef(dZ[keep_when:], mu_S)[0, 1]
    cov_sigmaS_z_Y = np.corrcoef(dZ[keep_when:], sigma_S)[0, 1]
    cov_theta_z_SI = np.corrcoef(dZ_SI[keep_when:], theta)[0, 1]
    cov_parti_cons_share = np.corrcoef(parti, 1 / Phi_bar_parti_1)[0, 1]
    cov_parti_wealth_share = np.corrcoef(parti, Phi_tilde_parti)[0, 1]

    cov_parti_dR = np.corrcoef(parti, dR)[0, 1]
    cov_parti_zY = np.corrcoef(parti, dZ[keep_when:])[0, 1]

    theta_save_matrix = np.array([
        Phi_bar_parti_1_matrix,
        sigma_Delta_bar_parti_matrix,
    ])

    sigma_S_save_matrix = np.array([
        Phi_tilde_parti_matrix,
        Phi_tilde_bar_parti_matrix,
        Delta_tilde_bar_parti_matrix,
        Delta_Phi_tilde_matrix,
    ])

    # parti_group_matrix = np.array([
    #     parti_age_group_matrix,
    #     parti_wealth_group_matrix,
    # ])

    cov_save_matrix = np.array([
        cov_theta_z_Y,
        cov_muS_z_Y,
        cov_sigmaS_z_Y,
        cov_theta_z_SI,
        cov_parti_cons_share,
        cov_parti_wealth_share,
    ])

    cov_parti_matrix = np.array([
        cov_parti_dR,
        cov_parti_zY
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
        # parti_group_matrix,
        parti_age_group_matrix,
        parti_wealth_group_matrix,
        # parti_age_wealth_group_matrix,
        cov_save_matrix,
        parti,
        cov_parti_matrix,
    )



def simulate_cohorts_mix_type(
        # Y: np.ndarray,
        biasvec: np.ndarray,
        dZ: np.ndarray,
        dZ_SI: np.ndarray,
        Nt: int,
        # Nc: int,
        tau: np.ndarray,
        dt: float,
        Ntype: int,
        Nconstraint: int,
        rho_i: np.ndarray,
        alpha_i: np.ndarray,
        beta_i: np.ndarray,
        beta_cohort_type: np.ndarray,
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
]:
    # Initializing variables
    invest_newborn = np.array([[[1], [0], [1], [1]]]) * np.ones((Ntype, Nconstraint, 1))
    can_short_newborn = np.array([[[1], [0], [0], [0]]]) * np.ones((Ntype, Nconstraint, 1))
    top = np.array([1, 0.75, 0.5, 0.25, 0])

    Phi_parti = np.ones((Nt))  # consumption share of the stock market participants

    popu_short = np.zeros((Nt))
    Phi_can_short = np.zeros((Nt))

    # equilibrium terms:
    dR = np.zeros(Nt)  # stores stock returns
    r = np.zeros(Nt)  # interest rate
    theta = np.zeros(Nt)  # market price of risk
    mu_S = np.zeros(Nt)
    sigma_S = np.zeros(Nt)
    beta = np.zeros(Nt)
    Delta_bar_parti = np.zeros((Nt))  # consumption weighted estimation error of the stock market participants
    Delta_tilde_parti = np.zeros((Nt))  # wealth weighted estimation error of the stock market participants
    parti = np.ones((Nt))  # participation rate
    parti_wealth_group = np.zeros((Nt, 4))
    parti_age_group = np.zeros((Nt, 4))
    dR_t = 0
    a_phi = (1 - phi ** 2)
    phi_sqr_a_phi = phi / np.sqrt(a_phi)
    a_phi_1 = 1 / a_phi
    sigma_Y_sq = sigma_Y ** 2

    append_init = np.ones((Ntype, Nconstraint, 1))

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
        )  # equation (11)


        X_parts = tax * np.exp(-tax * tau) * X * beta_cohort_type * eta_st_eta_ss * dt    # equation (18)
        X_t = np.sum(X_parts) / ( 1 - tax * dt)  # equation (18)  # dividing by (1-tax*dt) keeps sum(f_st*dt) at 1
        # eta_bar_t = np.sum(eta_bar_parts)

        eta_st_eta_ss = np.append(eta_st_eta_ss[:, :, 1:], append_init, axis=2)
        X = np.append(X[:, :, 1:], X_t * alpha_i, axis=2)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort
        # todo: eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
        #  eta_bar_t is the denominator; it creates issues if too close to 0
        #  so I rescale eta_bar to keep it away from 0, without changing f_st

        f_w_ist = X_parts / X_t / dt
        f_w_ist = np.append(f_w_ist[:, :, 1:], tax * alpha_i, axis=2)

        beta_t = np.sum(f_w_ist * beta_i) * dt
        f_c_ist = f_w_ist * beta_i / beta_t
        w_indiv_ist = f_w_ist / cohort_type_size
        if i > 0:
            dR_t = mu_S_t * dt + sigma_S_t * dZ_t

        # update beliefs
        V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')
        dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_t, dZ_SI_t,
                      'N')  # from eq(5)
        V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
        dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t, dZ_SI_t,
                      'P')
        dDelta_s_t = invest_tracker * dDelta_s_t_P + (
                        1 - invest_tracker) * dDelta_s_t_N  # the participation decision of last time affects the updating pattern
        tau_info = np.append(tau_info[:, :, 1:], 0 * append_init, axis=2) + dt
        Vhat_vector = np.append(Vhat_vector[:, :, 1:], Vhat * append_init, axis=2)

        if i < Npre-1:
            init_bias = (np.sum(biasvec[i+1:]) + np.sum(dZ[:i+1])) / T_hat
        else:
            init_bias = np.sum(dZ[i+1 - Npre:i+1]) / T_hat

        Delta_s_t = Delta_s_t[:, :, 1:] + dDelta_s_t[:, :, 1:]
        Delta_s_t = np.append(Delta_s_t, init_bias * append_init, axis=2)

        # find the market clearing theta, given beliefs and consumption shares of cohorts in the economy
        invest_tracker = np.append(invest_tracker[:, :, 1:],  invest_newborn, axis=2)  # all cohorts that are still in the market, 1 by default
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
        # invest_tracker = invest * invest_tracker
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
        pi_st = (d_eta_st + theta_t) / sigma_S_t
        # age_t = np.sum(cohort_size * tau * invest_tracker)
        # n_parti_t = np.sum(invest_tracker) / Nc
        popu_can_short_t = np.sum(cohort_type_size * can_short_tracker)
        Phi_can_short_t = np.sum(can_short_tracker * f_c_ist * dt)
        short = pi_st < 0
        Phi_short_t = np.sum(short * f_c_ist * dt)
        popu_short_t = np.sum(cohort_type_size * short)

        rho_bar_t = np.sum(rho_i * f_c_ist) / np.sum(f_c_ist)
        rho_tilde_t = np.sum(rho_i * f_w_ist) / np.sum(f_w_ist)

        r_t = (
                nu - tax * beta0 / beta_t
                + rho_bar_t
                + mu_Y
                - sigma_Y * theta_t
        )

        mu_S_t = tax - nu - rho_tilde_t + beta_t + sigma_S_t * theta_t + r_t

        # store the results
        # storing the values takes a lot of time
        dR[i] = dR_t  # realized return from t-1 to t
        theta[i] = theta_t
        r[i] = r_t
        Delta_bar_parti[i] = Delta_bar_parti_t
        Delta_tilde_parti[i] = Delta_tilde_parti_t
        mu_S[i] = mu_S_t
        sigma_S[i] = sigma_S_t
        if sigma_S_t<0:
            print('negative vola')
        beta[i] = beta_t
        # if need_f == 'True':
        #     f_c[i, :] = f_c_ist
        #     f_w[i, :] = f_w_ist
        # if need_Delta == 'True':
        #     Delta[i, :] = Delta_s_t
        # if need_pi == 'True':
        #     # pi[i, :] = pi_st

        Phi_parti[i] = fc_parti_t
        # parti[i] = popu_parti_t
        # invest_mat[i] = invest_tracker
        popu_short[i] = popu_short_t
        Phi_can_short[i] = Phi_can_short_t

        wealth_cutoffs = find_the_rich_mix(
            w_indiv_ist,
            cohort_type_size,
            top
        )
        for j in range(4):
            parti_age_group[i, j] = np.average(invest_tracker[:, :, cutoffs_age[j + 1]:cutoffs_age[j]],
                                               weights=cohort_type_size[:, :, cutoffs_age[j + 1]:cutoffs_age[j]])
            within_group = (w_indiv_ist >= wealth_cutoffs[j]) * (w_indiv_ist < wealth_cutoffs[j + 1])
            parti_wealth_group[i, j] = np.sum(invest_tracker * within_group * cohort_type_size) / \
                                       np.sum(within_group * cohort_type_size)

    return (
        r,
        theta,
        parti,
        Phi_parti,
        Delta_bar_parti,
        Delta_tilde_parti,
        dR,
        mu_S,
        sigma_S,
        beta,
        parti_age_group,
        parti_wealth_group,
    )


def simulate_mean_vola_mix_type(
        # Y: np.ndarray,
        biasvec: np.ndarray,
        dZ: np.ndarray,
        dZ_SI: np.ndarray,
        Nt: int,
        # Nc: int,
        tau: np.ndarray,
        dt: float,
        Ntype: int,
        Nconstraint: int,
        rho_i: np.ndarray,
        alpha_i: np.ndarray,
        beta_i: np.ndarray,
        beta_cohort_type: np.ndarray,
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
    wealth_groups = np.linspace(1, 0, N_wealth_group+1)
    parti_age_wealth_group = np.ones((Nt - keep_when, 4, N_wealth_group))

    a_phi = (1 - phi ** 2)
    phi_sqr_a_phi = phi / np.sqrt(a_phi)
    a_phi_1 = 1 / a_phi
    sigma_Y_sq = sigma_Y ** 2

    append_init = np.ones((Ntype, Nconstraint, 1))
    invest_newborn = np.array([[[1], [0], [1], [1]]]) * np.ones((Ntype, Nconstraint, 1))
    can_short_newborn = np.array([[[1], [0], [0], [0]]]) * np.ones((Ntype, Nconstraint, 1))
    top = np.array([1, 0.75, 0.5, 0.25, 0])

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
        )  # equation (11)

        X_parts = tax * np.exp(-tax * tau) * X * beta_cohort_type * eta_st_eta_ss * dt  # equation (18)
        X_t = np.sum(X_parts) / (1 - tax * dt)  # equation (18)  # dividing by (1-tax*dt) keeps sum(f_st*dt) at 1
        # eta_bar_t = np.sum(eta_bar_parts)

        eta_st_eta_ss = np.append(eta_st_eta_ss[:, :, 1:], append_init, axis=2)
        X = np.append(X[:, :, 1:], X_t * np.ones((1, 1, 1)), axis=2)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort
        # todo: eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
        #  eta_bar_t is the denominator; it creates issues if too close to 0
        #  so I rescale eta_bar to keep it away from 0, without changing f_st

        f_w_ist = X_parts / X_t / dt
        f_w_ist = np.append(f_w_ist[:, :, 1:], tax * alpha_i, axis=2)

        beta_t = np.sum(f_w_ist * beta_i) * dt
        f_c_ist = f_w_ist * beta_i / beta_t
        w_indiv_ist = f_w_ist / cohort_type_size
        c_indiv_ist = f_c_ist / cohort_type_size
        if i > 0:
            dR_t = mu_S_t * dt + sigma_S_t * dZ_t
        else:
            dR_t = 0

        # update beliefs
        V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')
        dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_t, dZ_SI_t,
                                            'N')  # from eq(5)
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
        # invest_tracker = invest * invest_tracker
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
        pi_st = (d_eta_st + theta_t) / sigma_S_t
        # age_t = np.sum(cohort_size * tau * invest_tracker)
        # n_parti_t = np.sum(invest_tracker) / Nc
        # popu_can_short_t = np.sum(cohort_type_size * can_short_tracker)
        # Phi_can_short_t = np.sum(can_short_tracker * f_c_ist * dt)
        # short = pi_st < 0
        # Phi_short_t = np.sum(short * f_c_ist * dt)
        # popu_short_t = np.sum(cohort_type_size * short)

        rho_bar_t = np.sum(rho_i * f_c_ist) / np.sum(f_c_ist)
        rho_tilde_t = np.sum(rho_i * f_w_ist) / np.sum(f_w_ist)

        r_t = (
                nu - tax * beta0 / beta_t
                + rho_bar_t
                + mu_Y
                - sigma_Y * theta_t
        )

        mu_S_t = tax - nu - rho_tilde_t + beta_t + sigma_S_t * theta_t + r_t

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
            sigma_S[ii] = sigma_S_t
            beta[ii] = beta_t
            parti[ii] = popu_parti_t
            if sigma_S_t < 0:
                print('negative vola')
            Phi_bar_parti_1[ii] = 1 / fc_parti_t
            Phi_tilde_parti[ii] = fw_parti_t
            wealth_cutoffs = find_the_rich_mix(
                # w_indiv_ist,
                c_indiv_ist,
                cohort_type_size,
                wealth_groups,
            )
            for l in range(N_wealth_group):
                within_group = (w_indiv_ist >= wealth_cutoffs[l]) * (w_indiv_ist < wealth_cutoffs[l + 1])
                parti_wealth_group[ii, l] = np.sum(invest_tracker * within_group * cohort_type_size) / \
                                            np.sum(within_group * cohort_type_size)
            for j in range(4):
                invest_age = invest_tracker[:, :, cutoffs_age[j + 1]:cutoffs_age[j]]
                # w_indiv_ist_age = w_indiv_ist[:, cutoffs_age[j + 1]:cutoffs_age[j]]
                cohort_type_age = cohort_type_size[:, :, cutoffs_age[j + 1]:cutoffs_age[j]]
                parti_age_group[ii, j] = np.average(invest_age, weights=cohort_type_age)
                # wealth_cutoffs_age_group = find_the_rich_mix(
                #     w_indiv_ist_age,
                #     cohort_type_age,
                #     wealth_groups,
                # )
                # for jj in range(N_wealth_group):
                #     within_group = (w_indiv_ist_age >= wealth_cutoffs_age_group[jj]) * (
                #             w_indiv_ist_age < wealth_cutoffs_age_group[jj + 1])
                #     total = within_group * cohort_type_age
                #     parti_age_wealth_group[ii, j, jj] = np.sum(invest_age * total) / np.sum(total)
    dR_matrix = np.array([np.mean(dR / dt), np.std(dR / dt)])
    theta_matrix = np.array([np.mean(theta), np.std(theta)])
    r_matrix = np.array([np.mean(r), np.std(r)])
    mu_S_matrix = np.array([np.mean(mu_S), np.std(mu_S)])
    sigma_S_matrix = np.array([np.mean(sigma_S), np.std(sigma_S)])
    beta_matrix = np.array([np.mean(beta), np.std(beta)])

    sigma_Delta_bar_parti_matrix = np.array([np.mean(sigma_Y * Delta_bar_parti), np.std(sigma_Y * Delta_bar_parti)])
    Delta_tilde_bar_parti = Delta_tilde_parti - Delta_bar_parti
    Delta_tilde_bar_parti_matrix = np.array([np.mean(Delta_tilde_bar_parti), np.std(Delta_tilde_bar_parti)])
    Phi_tilde_bar_parti = Phi_tilde_parti * Phi_bar_parti_1
    Phi_bar_parti_1_matrix = np.array([np.mean(Phi_bar_parti_1), np.std(Phi_bar_parti_1)])
    Phi_tilde_parti_matrix = np.array([np.mean(Phi_tilde_parti), np.std(Phi_tilde_parti)])
    Phi_tilde_bar_parti_matrix = np.array([np.mean(Phi_tilde_bar_parti), np.std(Phi_tilde_bar_parti)])
    Delta_Phi_tilde = Delta_tilde_bar_parti * Phi_tilde_parti
    Delta_Phi_tilde_matrix = np.array([np.mean(Delta_Phi_tilde), np.std(Delta_Phi_tilde)])
    parti_age_group_matrix = np.array(np.mean(parti_age_group, axis=0))
    Delta_bar_mat = np.tile(np.reshape(Delta_bar, (-1, 1)), (1, N_wealth_group))
    # parti_wealth_group_mask = np.ma.masked_where((Delta_bar_mat >= 0.05) | (Delta_bar_mat <= -0.05), parti_wealth_group)
    parti_wealth_group_mask = parti_wealth_group
    parti_wealth_group_matrix = np.array(np.nanmean(parti_wealth_group_mask, axis=0))
    # parti_age_wealth_group_matrix = np.array(np.nanmean(parti_age_wealth_group, axis=0))
    cov_theta_z_Y = np.corrcoef(dZ[keep_when:], theta)[0, 1]
    cov_muS_z_Y = np.corrcoef(dZ[keep_when:], mu_S)[0, 1]
    cov_sigmaS_z_Y = np.corrcoef(dZ[keep_when:], sigma_S)[0, 1]
    cov_theta_z_SI = np.corrcoef(dZ_SI[keep_when:], theta)[0, 1]
    cov_parti_cons_share = np.corrcoef(parti, 1 / Phi_bar_parti_1)[0, 1]
    cov_parti_wealth_share = np.corrcoef(parti, Phi_tilde_parti)[0, 1]

    cov_parti_dR = np.corrcoef(parti, dR)[0, 1]
    cov_parti_zY = np.corrcoef(parti, dZ[keep_when:])[0, 1]

    theta_save_matrix = np.array([
        Phi_bar_parti_1_matrix,
        sigma_Delta_bar_parti_matrix,
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

    cov_parti_matrix = np.array([
        cov_parti_dR,
        cov_parti_zY
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
        # parti_group_matrix,
        parti_age_group_matrix,
        parti_wealth_group_matrix,
        # parti_age_wealth_group_matrix,
        cov_save_matrix,
        parti,
        cov_parti_matrix,
    )

