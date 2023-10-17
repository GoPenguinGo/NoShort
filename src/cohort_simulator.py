import numpy as np
from typing import Tuple, List, Any
from src.stats import post_var, fadingmemo, dDelta_st_calculator, weighted_variance
from src.solver import bisection, solve_theta, find_the_rich, bisection_partial_constraint, solve_theta_partial_constraint
from tqdm import tqdm
from numba import jit

# todo: use matrix calculation in the functions, calculate multiple paths per run
def simulate_cohorts_SI(
        Y: np.ndarray,
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
        Ninit: int,
        mode_trade: str,
        mode_learn: str,
        cohort_size: np.ndarray,
        cohort_type_size: np.ndarray,
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
        top: float,
        old_limit: float,
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
    invest_mat = np.ones((Nt, Ntype, Nc))

    if need_f == 'True':
        f_c = np.zeros((Nt, Ntype, Nc))    # evolution of cohort consumption share
        f_w = np.zeros((Nt, Ntype, Nc))   # evolution of cohort wealth share
    else:
        f_c = f_w = 0

    if need_Delta == 'True':
        Delta = np.zeros((Nt, Ntype, Nc))  # stores bias in beliefs
    else:
        Delta = 0

    if need_pi == 'True':
        pi = np.zeros((Nt, Ntype, Nc))  # portfolio choices
    else:
        pi = 0

    if mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
        Phi_parti = np.ones((Nt))  # consumption share of the stock market participants
        parti = np.ones((Nt, Ntype, Nc))
        invest_mat = np.ones((Nt, Ntype, Nc))
    else:
        Phi_parti = 0
        parti = 0
        invest_mat = 0

    if mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
        popu_can_short= np.zeros((Nt))
        popu_short = np.zeros((Nt))
        Phi_can_short = np.zeros((Nt))
        Phi_short = np.zeros((Nt))
    else:
        popu_can_short = 0
        popu_short = 0
        Phi_can_short = 0
        Phi_short = 0

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

    popu_can_short = 0
    popu_short = 0

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

        eta_st_eta_ss = np.append(eta_st_eta_ss[:, 1:], eta_st_eta_ss_init, axis=1)
        X = np.append(X[:, 1:], np.ones((Ntype, 1)) * X_t, axis=1)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort
        # todo: eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
        #  eta_bar_t is the denominator; it creates issues if too close to 0
        #  so I rescale eta_bar to keep it away from 0, without changing f_st

        f_w_ist = X_parts / X_t / dt
        f_w_ist = np.append(f_w_ist[:, 1:], tax * np.ones((Ntype, 1)), axis=1)

        beta_t = np.sum(f_w_ist * beta_i) * dt
        f_c_ist = f_w_ist * beta_i / beta_t

        # Wealth
        w_t = Y[i] / beta_t  # total wealth at time t
        if i == 0:
            w_ist = w_t * f_w_ist  # cohort
            w_indiv_ist = w_ist / cohort_type_size * dt  # indiv
        else:
            dR_t = mu_S_t * dt + sigma_S_t * dZ_t  # realized stock return, mu_t^Sdt + sigma_t^Sdz_t
            dw_indiv_ist = ((r_t - rho_i) + pi_st * (mu_S_t - r_t)) * w_indiv_ist * dt + w_indiv_ist * pi_st * sigma_S_t * dZ_t  # r_t, theta_t, pi_st from last loop, dZ_t just realized
            w_indiv_ist = w_indiv_ist + dw_indiv_ist
            w_ist = w_indiv_ist * cohort_type_size / dt
            adjust_scale = w_t * (1 / dt - tax) / np.sum(w_ist[:, 1:])
            w_ist = np.append(w_ist[:, 1:] * adjust_scale, np.ones((2, 1)) * w_t * tax, axis=1)
            w_indiv_ist = np.append(w_indiv_ist[:, 1:] * adjust_scale, np.ones((2, 1)) * w_t * tax / nu, axis=1)

        # update beliefs
        if mode_trade == 'complete':  # everyone is P
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t, dZ_SI_t,
                      'P')
            tau_info = tau

        elif mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
            V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')
            dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_t, dZ_SI_t,
                      'N')  # from eq(5)
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t, dZ_SI_t,
                      'P')
            dDelta_s_t = invest_tracker * dDelta_s_t_P + (
                        1 - invest_tracker) * dDelta_s_t_N  # the participation decision of last time affects the updating pattern
            tau_info = np.append(tau_info[:, 1:], np.zeros((Ntype, 1)), axis=1) + dt

        else:
            print('mode_trade not found')

        Vhat_vector = np.append(Vhat_vector[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1)

        if i < Npre-1:
            init_bias = (np.sum(biasvec[i+1:]) + np.sum(dZ[:i+1])) / T_hat
        else:
            init_bias = np.sum(dZ[i+1 - Npre: i+1]) / T_hat

        Delta_s_t = Delta_s_t[:, 1:] + dDelta_s_t[:, 1:]
        Delta_s_t = np.append(Delta_s_t, init_bias * np.ones((Ntype, 1)), axis=1)

        # find the market clearing theta, given beliefs and consumption shares of cohorts in the economy
        invest_tracker = np.append(invest_tracker[:, 1:], np.ones((Ntype, 1)), axis=1)  # all cohorts that are still in the market, 1 by default

        if mode_trade == 'w_constraint':
            if mode_learn == 'disappointment':
                possible_cons_share = f_c_ist * dt * invest_tracker
                possible_delta_st = Delta_s_t * invest_tracker
                lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound for theta among active investors
                theta_t = bisection(
                    solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
                )  # solve for theta
                a = Delta_s_t + theta_t
                invest = (a > 0)
                switch_P_to_N = invest_tracker * (1 - invest)
                invest_tracker = invest * invest_tracker  # update invest tracker

                # tau_info and V_hat has to change for the agents who switched to N
                Vhat_vector = np.append(V_st_P[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1) * switch_P_to_N + Vhat_vector * (1 - switch_P_to_N)  # reset initial variance
                tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock

            elif mode_learn == 'reentry':   # agents stay as type P even if constrained
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

                Vhat_vector = np.append(V_st_P[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1) * switch_P_to_N + np.append(V_st_N[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1) * switch_N_to_P + Vhat_vector * (1 - switch)  # reset initial variance
                tau_info = dt * switch + tau_info * (1 - switch)  # reset clock

            else:
                print('mode_learn not found')
                exit()

            d_eta_st = a * invest_tracker - theta_t
            invest_fc_st = invest_tracker * f_c_ist * dt
            invest_fw_st = invest_tracker * f_w_ist * dt
            popu_parti_t = np.sum(cohort_size * invest_tracker)
            fc_parti_t = np.sum(invest_fc_st)
            fw_parti_t = np.sum(invest_fw_st)
            Delta_bar_parti_t = np.sum(Delta_s_t * invest_fc_st) / fc_parti_t
            Delta_tilde_parti_t = np.sum(Delta_s_t * invest_fw_st) / fw_parti_t
            sigma_S_t = sigma_Y - Delta_bar_parti_t + Delta_tilde_parti_t
            pi_st = (d_eta_st + theta_t) / sigma_S_t
            age_t = np.sum(cohort_size * tau * invest_tracker)
            n_parti_t = np.sum(invest_tracker) / Nc

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
            sigma_S_t = sigma_Y - Delta_bar_parti_t + Delta_tilde_parti_t
            pi_ist = (d_eta_st + theta_t) / sigma_S_t
            age_t = np.sum(cohort_size * tau * invest)
            n_parti_t = np.sum(invest) / Nc

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
        elif mode_trade == 'partial_constraint_old':
            can_short_tracker = np.append(can_short_tracker[:, 1:], np.zeros((Ntype, 1)), axis=1)

            if mode_learn == 'disappointment':
                possible_cons_share = f_c_ist * dt * invest_tracker
                possible_delta_st = Delta_s_t * invest_tracker
                can_short_possible = (tau >= old_limit)
                can_short_tracker = can_short_possible * invest_tracker

                lowest_bound = lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound for theta among active investors
                theta_t = bisection_partial_constraint(
                    solve_theta_partial_constraint, lowest_bound, 50, can_short_tracker, possible_delta_st, possible_cons_share,
                    sigma_Y
                )
                a = Delta_s_t + theta_t
                invest = 1 - (a < 0) * (can_short_tracker < 1)
                switch_P_to_N = invest_tracker * (1 - invest)
                invest_tracker = invest * invest_tracker  # update invest tracker

                # tau_info and V_hat has to change for the agents who switched to N
                Vhat_vector = np.append(V_st_P[1:], Vhat) * switch_P_to_N + Vhat_vector * (1 - switch_P_to_N)  # reset initial variance
                tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock

            elif mode_learn == 'reentry':
                possible_cons_share = f_c_ist * dt
                possible_delta_st = Delta_s_t
                can_short_tracker = (tau >= old_limit)

                lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
                theta_t = bisection_partial_constraint(
                    solve_theta_partial_constraint, lowest_bound, 50, can_short_tracker, possible_delta_st, possible_cons_share,
                    sigma_Y
                )
                a = Delta_s_t + theta_t
                invest = 1 - (a < 0) * (can_short_tracker < 1)
                switch_P_to_N = invest_tracker * (1 - invest)
                switch_N_to_P = np.maximum(invest - invest_tracker, 0)
                # tau_info and V_hat has to change for the agents who switched to N
                switch = switch_N_to_P + switch_P_to_N
                invest_tracker = invest

                Vhat_vector = np.append(V_st_P[:, 1:], Vhat * np.ones((Ntype, 1)), axis=1) * switch_P_to_N + np.append(V_st_N[:, 1:], Vhat* np.ones((Ntype, 1)), axis=1) * switch_N_to_P + Vhat_vector * (1 - switch)  # reset initial variance
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
            sigma_S_t = sigma_Y - Delta_bar_parti_t + Delta_tilde_parti_t
            pi_st = (d_eta_st + theta_t) / sigma_S_t
            # age_t = np.sum(cohort_size * tau * invest_tracker)
            # n_parti_t = np.sum(invest_tracker) / Nc
            popu_can_short_t = np.sum(cohort_type_size * can_short_tracker)
            Phi_can_short_t = np.sum(can_short_tracker * f_c_ist * dt)
            short = pi_st < 0
            Phi_short_t = np.sum(short * f_c_ist * dt)
            popu_short_t = np.sum(cohort_type_size * short)


        else:
            print('mode_trade not found')
            exit()

        rho_bar_t = np.sum(rho_i * f_c_ist)
        rho_tilde_t = np.sum(rho_i * f_w_ist)

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
        beta[i] = beta_t
        w[i, :] = w_st
        age[i] = age_t
        n_parti[i] = n_parti_t
        if need_f == 'True':
            f_c[i, :] = f_c_ist
            f_w[i, :] = f_w_ist
        if need_Delta == 'True':
            Delta[i, :] = Delta_s_t
        if need_pi == 'True':
            pi[i, :] = pi_st
        if mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
            Phi_parti[i] = fc_parti_t
            parti[i] = popu_parti_t
            invest_mat[i] = invest_tracker
        if mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
            popu_can_short[i] = popu_can_short_t
            popu_short[i] = popu_short_t
            Phi_can_short[i] = Phi_can_short_t
            Phi_short[i] = Phi_short_t
        # switch_P_to_N_ts[i] = np.sum(switch_P_to_N)
        # switch_N_to_P_ts[i] = np.sum(switch_N_to_P)

    return (
        r,
        theta,
        f_c,
        f_w,
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
        popu_can_short,
        popu_short,
        Phi_can_short,
        Phi_short,
    )




def simulate_cohorts_mean_vola(
        Y: np.ndarray,
        biasvec: np.ndarray,
        dZ: np.ndarray,
        dZ_SI: np.ndarray,
        Nt: int,
        Nc: int,
        tau: np.ndarray,
        dt: float,
        rho: float,
        nu: float,
        Vhat: float,
        mu_Y: float,
        sigma_Y: float,
        sigma_S: float,
        tax: float,
        beta: float,
        phi: float,
        T_hat: float,
        Npre: float,
        Ninit: int,
        mode_trade: str,
        mode_learn: str,
        cohort_size: np.ndarray,
        Delta_s_t: np.ndarray,
        eta_st_eta_ss: np.ndarray,
        eta_bar: np.ndarray,
        d_eta_st: np.ndarray,
        invest_tracker: np.ndarray,
        can_short_tracker: np.ndarray,
        tau_info: np.ndarray,
        Vhat_vector: np.ndarray,
        top: float,
        old_limit: float,
        cutoffs: np.ndarray,
        n_age_groups: int,
):
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
    # cohort-specific terms:
    #Delta = np.zeros((Nt, Nc))  # stores bias in beliefs
    #f = np.zeros((Nt, Nc))  # evolution of cohort consumption share
    #pi = np.zeros((Nt, Nc))  # portfolio choices
    #w = np.zeros((Nt, Nc))  # evolution of wealth for cohorts
    #invest_mat = np.zeros((Nt, Nc))

    # aggregate terms:
    keep_when = int(200 / dt)
    #dR = np.zeros(Nt - keep_when)
    r = np.zeros(Nt - keep_when)  # interest rate
    theta = np.zeros(Nt - keep_when)  # market price of risk
    Phi_parti = np.zeros((Nt - keep_when))  # consumption share of the stock market participants
    Phi_parti_1_matrix = np.zeros((Nt - keep_when))
    Delta_bar_parti = np.zeros((Nt - keep_when))  # consumption weighted estimation error of the stock market participants
    # parti = np.zeros((Nt - keep_when))  # participation rate
    popu_age = np.zeros((Nt - keep_when, n_age_groups))
    # belief_age = np.zeros((Nt - keep_when, n_age_groups))
    wealthshare_age = np.zeros((Nt - keep_when, n_age_groups))
    cov_save = np.zeros((Nt - keep_when, 3))
    popu_can_short = np.zeros((Nt - keep_when))
    popu_short = np.zeros((Nt - keep_when))
    Phi_short = np.zeros((Nt - keep_when))
    Phi_can_short = np.zeros((Nt - keep_when))
    # var_save = np.zeros((Nt - keep_when, 4))
    Delta_popu_parti = np.zeros((Nt - keep_when))

    if mode_trade == 'complete':
        invest_tracker = np.ones(Nc)

    dR_t = 0

    a_phi = (1 - phi ** 2)
    phi_sqr_a_phi = phi / np.sqrt(a_phi)
    a_phi_1 = 1 / a_phi
    sigma_Y_sq = sigma_Y ** 2

    for i in tqdm(range(Nt)):
    # for i in tqdm(range(keep_when)):
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

        eta_bar_parts = tax * np.exp(-tax * tau) * eta_bar * eta_st_eta_ss * dt  # equation (18)
        eta_bar_t = np.sum(eta_bar_parts) / (1 - tax * dt)  # equation (18)

        eta_st_eta_ss = np.append(eta_st_eta_ss[1:], 1)
        eta_bar = np.append(eta_bar[1:], eta_bar_t)
        eta_bar = eta_bar / eta_bar_t  # rescale, does not change the relative magnitude of each cohort

        f_st = eta_bar_parts / eta_bar_t / dt
        f_st = np.append(f_st[1:], tax)

        # Wealth
        if i == 0:
            w_st = Y[i] / beta * f_st  # cohort
            w_indiv_st = w_st / cohort_size * dt  # indiv
        else:
            dR_t = mu_S_t * dt + sigma_S * dZ_t  # realized stock return, mu_t^Sdt + sigma_t^Sdz_t
            w_t = Y[i] / beta  # total wealth at time t
            dw_indiv_st = ((r_t + nu - tax - beta) + pi_st * (mu_S_t - r_t)) * w_indiv_st * dt + w_indiv_st * pi_st * sigma_S * dZ_t  # r_t, theta_t, pi_st from last loop, dZ_t just realized
            w_indiv_st = w_indiv_st + dw_indiv_st
            w_st = w_indiv_st * cohort_size / dt
            adjust_scale = w_t * (1 / dt - tax) / np.sum(w_st[1:])
            w_st = np.append(w_st[1:] * adjust_scale, w_t * tax)
            w_indiv_st = np.append(w_indiv_st[1:] * adjust_scale, w_t * tax / nu)

        # update beliefs
        if mode_trade == 'complete':  # everyone is P
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t, dZ_SI_t,
                      'P')
            tau_info = tau

        elif mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
            V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')
            dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_t, dZ_SI_t,
                      'N')  # from eq(5)
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t, dZ_SI_t,
                      'P')
            dDelta_s_t = invest_tracker * dDelta_s_t_P + (
                        1 - invest_tracker) * dDelta_s_t_N  # the participation decision of last time affects the updating pattern
            tau_info = np.append(tau_info[1:], 0) + dt

        else:
            print('mode_trade not found')

        Vhat_vector = np.append(Vhat_vector[1:], Vhat)

        if i < Npre-1:
            init_bias = (np.sum(biasvec[i+1:]) + np.sum(dZ[:i+1])) / T_hat
        else:
            init_bias = np.sum(dZ[i+1 - Npre: i+1]) / T_hat

        Delta_s_t = Delta_s_t[1:] + dDelta_s_t[1:]
        Delta_s_t = np.append(Delta_s_t, init_bias)

        # find the market clearing theta, given beliefs and consumption shares of cohorts in the economy
        if mode_trade != 'complete':
            invest_tracker = np.append(invest_tracker[1:], 1)  # all cohorts that are still in the market, 1 by default

        if mode_trade == 'w_constraint':
            if mode_learn == 'disappointment':
                possible_cons_share = f_st * dt * invest_tracker
                possible_delta_st = Delta_s_t * invest_tracker
                lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound for theta among active investors
                theta_t = bisection(
                    solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
                )  # solve for theta
                a = Delta_s_t + theta_t
                invest = (a > 0)
                switch_P_to_N = invest_tracker * (1 - invest)
                invest_tracker = invest * invest_tracker  # update invest tracker

                # tau_info and V_hat has to change for the agents who switched to N
                Vhat_vector = np.append(V_st_P[1:], Vhat) * switch_P_to_N + Vhat_vector * (1 - switch_P_to_N)  # reset variance
                tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock

            elif mode_learn == 'reentry':   # agents stay as type P even if constrained
                possible_cons_share = f_st * dt
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

                Vhat_vector = np.append(V_st_P[1:], Vhat) * switch_P_to_N + np.append(V_st_N[1:], Vhat) * switch_N_to_P + Vhat_vector * (1 - switch)  # reset initial variance
                tau_info = dt * switch + tau_info * (1 - switch)  # reset clock

            else:
                print('mode_learn not found')
                exit()

            d_eta_st = a * invest_tracker - theta_t
            pi_st = (d_eta_st + theta_t) / sigma_S
            if i >= keep_when:
                invest_fst = invest_tracker * f_st * dt
                popu_parti_t = np.sum(cohort_size * invest_tracker)
                f_parti_t = np.sum(invest_fst)
                Delta_bar_parti_t = np.sum(Delta_s_t * invest_fst) / f_parti_t

                # age_t = np.sum(cohort_size * tau * invest_tracker)
                # n_parti_t = np.sum(invest_tracker) / Nc
                Delta_popu_parti_t = np.sum(Delta_s_t * invest_tracker * cohort_size) / popu_parti_t
                var_cons_cohort_parti_t = np.var(invest_fst[np.nonzero(invest_fst)])
                indiv_fst = invest_fst / cohort_size
                var_cons_indiv_parti_t = np.var(indiv_fst[np.nonzero(indiv_fst)])
                Delta_parti = Delta_s_t * invest_tracker
                cohort_size_parti = cohort_size * invest_tracker
                var_Delta_cohort_parti_t = np.var(Delta_parti[np.nonzero(Delta_parti)])
                var_Delta_indiv_parti_t = weighted_variance(Delta_parti[np.nonzero(Delta_parti)], cohort_size_parti[np.nonzero(invest_tracker)])


        elif mode_trade == 'complete':
            f_st_standard = f_st * dt
            Delta_bar_parti_t = np.sum(f_st_standard * Delta_s_t)
            theta_t = sigma_Y - Delta_bar_parti_t
            d_eta_st = Delta_s_t
            invest = Delta_s_t >= -theta_t  # long stock
            popu_parti_t = 1

            f_parti_t = 1
            pi_st = (d_eta_st + theta_t) / sigma_S
            if i >= keep_when:
                # age_t = np.sum(cohort_size * tau * invest)
                # n_parti_t = np.sum(invest) / Nc
                Delta_popu_parti_t = np.sum(Delta_s_t * cohort_size)
                var_cons_cohort_parti_t = np.var(f_st_standard)
                var_cons_indiv_parti_t = np.var(f_st_standard / cohort_size)
                var_Delta_cohort_parti_t = np.var(Delta_s_t)
                var_Delta_indiv_parti_t = weighted_variance(Delta_s_t, cohort_size)


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

        elif mode_trade == 'partial_constraint_old':
            can_short_tracker = np.append(can_short_tracker[1:], 0)

            if mode_learn == 'disappointment':
                possible_cons_share = f_st * dt * invest_tracker
                possible_delta_st = Delta_s_t * invest_tracker
                can_short_possible = (tau >= old_limit)
                can_short_tracker = can_short_possible * invest_tracker

                lowest_bound = lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound for theta among active investors
                theta_t = bisection_partial_constraint(
                    solve_theta_partial_constraint, lowest_bound, 50, can_short_tracker, possible_delta_st, possible_cons_share,
                    sigma_Y
                )
                a = Delta_s_t + theta_t
                invest = 1 - (a < 0) * (can_short_tracker < 1)
                switch_P_to_N = invest_tracker * (1 - invest)
                invest_tracker = invest * invest_tracker  # update invest tracker

                # tau_info and V_hat has to change for the agents who switched to N
                Vhat_vector = np.append(V_st_P[1:], Vhat) * switch_P_to_N + Vhat_vector * (1 - switch_P_to_N)  # reset initial variance
                tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock

            elif mode_learn == 'reentry':
                possible_cons_share = f_st * dt
                possible_delta_st = Delta_s_t
                can_short_tracker = (tau >= old_limit)

                lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
                theta_t = bisection_partial_constraint(
                    solve_theta_partial_constraint, lowest_bound, 50, can_short_tracker, possible_delta_st, possible_cons_share,
                    sigma_Y
                )
                a = Delta_s_t + theta_t
                invest = 1 - (a < 0) * (can_short_tracker < 1)
                switch_P_to_N = invest_tracker * (1 - invest)
                switch_N_to_P = np.maximum(invest - invest_tracker, 0)
                # tau_info and V_hat has to change for the agents who switched to N
                switch = switch_N_to_P + switch_P_to_N
                invest_tracker = invest

                Vhat_vector = np.append(V_st_P[1:], Vhat) * switch_P_to_N + np.append(V_st_N[1:], Vhat) * switch_N_to_P + Vhat_vector * (1 - switch)  # reset initial variance
                tau_info = dt * switch + tau_info * (1 - switch)  # reset clock
            else:
                print('mode_learn not found')
                exit()

            d_eta_st = a * invest_tracker - theta_t
            pi_st = (d_eta_st + theta_t) / sigma_S

            if i >= keep_when:
                invest_fst = invest_tracker * f_st * dt
                popu_parti_t = np.sum(cohort_size * invest_tracker)
                f_parti_t = np.sum(invest_fst)
                Delta_bar_parti_t = np.sum(Delta_s_t * invest_fst) / f_parti_t
                Delta_popu_parti_t = np.sum(Delta_s_t * invest_tracker * cohort_size) / popu_parti_t
                var_cons_cohort_parti_t = np.var(invest_fst[np.nonzero(invest_fst)])
                indiv_fst = invest_fst / cohort_size
                var_cons_indiv_parti_t = np.var(indiv_fst[np.nonzero(indiv_fst)])
                Delta_parti = Delta_s_t * invest_tracker
                cohort_size_parti = cohort_size * invest_tracker
                var_Delta_cohort_parti_t = np.var(Delta_parti[np.nonzero(Delta_parti)])
                var_Delta_indiv_parti_t = weighted_variance(Delta_parti[np.nonzero(Delta_parti)], cohort_size_parti[np.nonzero(invest_tracker)])

                short = pi_st < 0
                # age_t = np.sum(cohort_size * tau * invest_tracker)
                # n_parti_t = np.sum(invest_tracker) / Nc
                popu_can_short_t = np.sum(cohort_size * can_short_tracker)
                popu_short_t = np.sum(cohort_size * short)
                Phi_can_short_t = np.sum(can_short_tracker * f_st * dt)
                Phi_short_t = np.sum(short * f_st * dt)


        else:
            print('mode_trade not found')
            exit()

        r_t = (
                rho
                + mu_Y
                + nu - tax
                - sigma_Y * theta_t
        )

        mu_S_t = sigma_S * theta_t + r_t

        # store the results, only the aggregate values
        if i >= keep_when:  # only keeping the data after 200 years in the simulation
            ii = i - keep_when
            theta[ii] = theta_t
            r[ii] = r_t
            Phi_parti[ii] = f_parti_t
            Delta_bar_parti[ii] = Delta_bar_parti_t
            Delta_popu_parti[ii] = Delta_popu_parti_t
            if mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
                Phi_parti[ii] = f_parti_t
            if mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
                popu_can_short[ii] = popu_can_short_t
                popu_short[ii] = popu_short_t
                Phi_can_short[ii] = Phi_can_short_t
                Phi_short[ii] = Phi_short_t
            # parti_rate = invest_tracker * cohort_size
            # belief = (Delta_s_t * sigma_Y + mu_Y)
            for j in range(4):
                if mode_trade != 'complete':
                    popu_age[ii, j] = np.average(invest_tracker[cutoffs[j + 1]:cutoffs[j]], weights=cohort_size[cutoffs[j + 1]:cutoffs[j]])
                wealthshare_age[ii, j] = np.sum(f_st[cutoffs[j + 1]:cutoffs[j]] * dt)
            # var_save[ii] = [var_cons_cohort_parti_t, var_cons_indiv_parti_t,
            #                var_Delta_cohort_parti_t, var_Delta_indiv_parti_t]
    # var_save_matrix = np.mean(var_save, axis=0)

    r_matrix = [np.mean(r), np.std(r)]
    theta_matrix = [np.mean(theta), np.std(theta)]
    Delta_bar_parti_matrix = [np.mean(Delta_bar_parti), np.std(Delta_bar_parti)]
    Delta_popu_parti_matrix = [np.mean(Delta_popu_parti), np.std(Delta_popu_parti)]
    wealthshare_age_matrix = [np.mean(wealthshare_age, axis=0), np.std(wealthshare_age, axis=0)]
    if mode_trade == 'complete':
        Phi_parti_matrix = [1, 0]
        Phi_parti_1_matrix = [1, 0]
        popu_age_matrix = [(0.25, 0.25, 0.25, 0.25), (0, 0, 0, 0)]
    else:
        Phi_parti_matrix = [np.mean(Phi_parti), np.std(Phi_parti)]
        Phi_parti_1_matrix = [np.mean(1/Phi_parti), np.std(1/Phi_parti)]
        popu_age_matrix = [np.mean(popu_age, axis=0), np.std(popu_age, axis=0)]
    if mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
        popu_can_short_matrix = [np.mean(popu_can_short), np.std(popu_can_short)]
        popu_short_matrix = [np.mean(popu_short), np.std(popu_short)]
        Phi_can_short_matrix = [np.mean(Phi_can_short), np.std(Phi_can_short)]
        Phi_short_matrix = [np.mean(Phi_short), np.std(Phi_short)]
    else:
        popu_can_short_matrix = [0,0]
        popu_short_matrix = [0,0]
        Phi_can_short_matrix = [0,0]
        Phi_short_matrix = [0,0]
    cov_theta_z_Y = np.corrcoef(dZ[keep_when:], theta)[0, 1]
    cov_theta_z_SI = np.corrcoef(dZ_SI[keep_when:], theta)[0, 1]
    cov_parti_wealth_share = np.corrcoef(np.sum(popu_age, axis=1) / 4, Phi_parti)[0, 1]
    cov_save_matrix = np.array([cov_theta_z_Y, cov_theta_z_SI, cov_parti_wealth_share])
    short_save_matrix = np.array([popu_can_short_matrix,
        popu_short_matrix,
        Phi_can_short_matrix,
        Phi_short_matrix])

    return (
        r_matrix,
        theta_matrix,
        Delta_bar_parti_matrix,
        Phi_parti_matrix,
        Phi_parti_1_matrix,
        popu_age_matrix,
        wealthshare_age_matrix,
        Delta_popu_parti_matrix,
        short_save_matrix,
        cov_save_matrix,
    )





#
#
# def simulate_cohorts(
#         Y: np.ndarray,
#         biasvec: np.ndarray,
#         dZ: np.ndarray,
#         Nt: int,
#         Nc: int,
#         tau: np.ndarray,
#         dt: float,
#         rho: float,
#         nu: float,
#         Vhat: float,
#         mu_Y: float,
#         sigma_Y: float,
#         sigma_S: float,
#         tax: float,
#         beta: float,
#         T_hat: float,
#         Npre: float,
#         Ninit: int,
#         mode_trade: str,
#         mode_learn: str,
#         cohort_size: np.ndarray,
#         Delta_s_t: np.ndarray,
#         eta_st_eta_ss: np.ndarray,
#         eta_bar: np.ndarray,
#         d_eta_st: np.ndarray,
#         invest_tracker: np.ndarray,
#         tau_info: np.ndarray,
#         good_time_simulate: np.ndarray,
# ) -> Tuple[
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
# ]:
#     """"" Simulate the economy forward
#
#     Args:
#         biasvec (np.ndarray): pre-trading period shocks to form beliefs of the initial cohort, shape(Npre,)
#         dZt (np.ndarray): random shocks, shape (Nt)
#         Nt (int): number of periods in the simulation
#         Nc (int): number of cohorts in the economy
#         tau (np.ndarray): t-s, shape(Nt)
#         # IntVec (np.ndarray): ~similar to consumption share, shape(Nc)
#         dt (float): per unit of time
#         rho (float): discount factor
#         nu (float): rate of birth and death
#         Vhat (float): initial variance
#         mu_Y (float): as in eq(1), drift of aggregate output growth
#         sigma_Y (float): as in eq(1), diffusion of aggregate output growth
#         sigma_S (float): as in eq(26), diffusion of stock price
#         tax (float): as in eq(18), consumption share of the newborn cohort
#         T_hat (float): pre-trading years
#         Npre (float): pre-trading number of obs
#         mode (str): versions of the model
#         - from the cohort_builder function -
#         f_st (np.ndarray): consumption share input
#         eta_st_ss (np.ndarray): disagreement input
#         eta_bar (np.ndarray): average disagreement input
#         Delta_s_t (np.ndarray): bias for each cohort, shape(Nc)
#         d_eta_st (np.ndarray): max(Delta_), shape(Nc)
#         invest_tracker (np.ndarry): shape(Nt)
#         tau_info: np.ndarray,
#         good_time_simulate: np.ndarray,
#
#     Returns:
#         mu_S (np.ndarray): expected return under true measure at t, shape(Nt, )
#         mu_S_s (np.ndarray): expected stock return each cohort, shape(Nt, Nc, )
#         r (np.ndarray): interest rate, shape(Nt, )
#         theta (np.ndarray): market price of risk, shape(Nt, )
#         f (np.ndarray): consumption share over time, shape(Nt, Nc, )
#         Delta (np.ndarray): bias over time, shape(Nt, Nc, )
#         max (np.ndarray): max(-theta, delta_s_t) over time, shape(Nt, Nc, )
#         pi (np.ndarray): portfolio choice over time, shape(Nt, Nc, )
#         parti (np.ndarray): population that invest in stocks over time, shape(Nt, )
#         f_parti (np.ndarray): aggregate consumption share over time conditional on invest in stocks, shape(Nt, )
#         Delta_bar_parti (np.ndarray): aggregate consumption weighted bias over time conditional on invest in stocks, shape(Nt, )
#         dR (np.ndarray): change of stock returns over time, shape(Nt, )
#         w (np.ndarray): individual wealth, shape(Nt, Nc, )
#         w_cohort (np.ndarray): cohort wealth = individual wealth * cohort size, shape(Nt, Nc, )
#         age (np.ndarray):  average age participating in the stock market, shape(Nt, )
#         n_parti (np.ndarray): number of cohorts participating in the stock market, shape(Nt, )
#     """ ""
#     # Initializing variables
#     # cohort-specific terms:
#     Delta = np.zeros((Nt, Nc))  # stores bias in beliefs
#     max = np.zeros((Nt, Nc))  # stores max(delta, -theta)
#     f = np.zeros((Nt, Nc))  # evolution of cohort consumption share
#     pi = np.zeros((Nt, Nc))  # portfolio choices
#     w_cohort = np.zeros((Nt, Nc))  # evolution of wealth for cohorts
#     short = np.zeros((Nt, Nc))
#
#     # aggregate terms:
#     dR = np.zeros(Nt)  # stores stock returns
#     r = np.zeros(Nt)  # interest rate
#     theta = np.zeros(Nt)  # market price of risk
#     f_parti = np.zeros((Nt))  # consumption share of the stock market participants
#     Delta_bar_parti = np.zeros((Nt))  # disagreement of the stock market participants
#     parti = np.zeros((Nt))  # participation rate
#
#     dR_t = 0
#     age = np.zeros(Nt)
#     n_parti = np.zeros(Nt)
#
#     for i in tqdm(range(Nt)):
#         dZ_t = dZ[i]
#
#         # todo: eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
#         #  eta_bar_t is the denominator; it creates issues if too close to 0
#         #  so I rescale eta_bar to keep it away from 0, without changing f_st
#
#         # new cohort born (age 0), get wealth transfer, observe, invest
#         eta_st_eta_ss = eta_st_eta_ss * np.exp(
#             (-0.5 * d_eta_st ** 2) * dt
#             + d_eta_st * dZ_t
#         )  # equation (11)
#
#         eta_bar_parts = tax * np.exp(-tax * tau) * eta_bar * eta_st_eta_ss * dt  # equation (18)
#         eta_bar_t = np.sum(eta_bar_parts) / (1 - tax * dt)  # equation (18)
#
#         eta_st_eta_ss = np.append(eta_st_eta_ss[1:], 1)
#         eta_bar = np.append(eta_bar[1:], eta_bar_t)
#         eta_bar = eta_bar / eta_bar_t  # rescale, does not change the relative magnitude of each cohort
#
#         f_st = eta_bar_parts / eta_bar_t / dt
#         f_st = np.append(f_st[1:], tax)
#
#         # Wealth
#         if i == 0:
#             w_cohort_st = Y[i] / beta * f_st
#             w_st = w_cohort_st / cohort_size * dt
#         else:
#             dR_t = mu_S_t * dt + sigma_S * dZ_t  # realized stock return, mu_t^Sdt + sigma_t^Sdz_t
#             w_t = Y[i] / beta
#             dw_st = ((r_t + nu - tax - beta) + pi_st * (mu_S_t - r_t)) * w_st * dt + w_st * pi_st * sigma_S * dZ_t  # r_t, theta_t, pi_st from last loop, dZ_t just realized
#             w_st = w_st[1:] + dw_st[1:]
#             w_st = np.append(w_st, w_t * tax / nu)
#             w_cohort_st = w_st * cohort_size / dt
#
#         # update beliefs
#         dDelta_s_t = (
#             post_var(sigma_Y, Vhat, tau_info) / sigma_Y**2
#                      ) * (
#             -Delta_s_t * dt + dZ_t
#         )  # from eq(5)
#
#         if mode_learn == 'back_renew' and mode_trade == 'disappointment':
#             tau_info = np.append(tau_info[1:], 0) + dt
#
#         if i < Npre-1:
#             init_bias = (np.sum(biasvec[i+1:]) + np.sum(dZ[:i+1])) / T_hat
#         else:
#             init_bias = np.sum(dZ[i+1 - Npre: i+1]) / T_hat
#
#         Delta_s_t = Delta_s_t[1:] + dDelta_s_t[1:]
#         Delta_s_t = np.append(Delta_s_t, init_bias)
#
#         # find the market clearing theta, given beliefs and consumption shares of cohorts in the economy
#         invest_tracker = np.append(invest_tracker[1:], 1)  # all cohorts that are still in the market, 1 by default
#
#         if mode_trade == 'disappointment':
#             if good_time_simulate[i] == 1:
#
#                 if mode_learn == 'back_collect':
#                     # agents who have left the market respond to the recent positive shocks
#                     # they collect all the information they missed during the drop period
#                     invest_tracker = np.ones(Nc)  # all can invest
#
#                 if mode_learn == 'back_renew':
#                     if i < Npre - 1:
#                         renew_bias = (np.sum(biasvec[i + 1:]) + np.sum(dZ[:i + 1])) / (Npre * dt)
#                     else:
#                         renew_bias = np.sum(dZ[i + 1 - Npre: i + 1]) / (Npre * dt)
#                     Delta_s_t = Delta_s_t * invest_tracker + renew_bias * (1 - invest_tracker)
#                     tau_info = invest_tracker * tau_info + (1 - invest_tracker) * dt
#                     invest_tracker = np.ones(Nc)  # all can invest
#
#             possible_cons_share = f_st * dt * invest_tracker
#             possible_delta_st = Delta_s_t * invest_tracker
#             lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
#             theta_t = bisection(
#                     solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
#             )  # solve for theta
#             a = Delta_s_t + theta_t
#             invest = (a >= 0)
#             # want_to_short_t = invest_tracker * (1 - invest)
#             invest_tracker = invest * invest_tracker
#             d_eta_st = a * invest_tracker - theta_t
#             invest_fst = invest_tracker * f_st * dt
#             popu_parti_t = np.sum(cohort_size * invest_tracker)
#             Delta_bar_parti_t = np.sum(Delta_s_t * invest_fst)
#             f_parti_t = np.sum(invest_fst)
#             pi_st = (d_eta_st + theta_t) / sigma_S
#             age_t = np.sum(cohort_size * tau * invest_tracker)
#             n_parti_t = np.sum(invest_tracker) / Nc
#
#         elif mode_trade == 'reentry':
#             lowest_bound = -np.max(Delta_s_t)  # absolute lower bound for theta
#             f_st_standard = f_st * dt
#             theta_t = bisection(
#                 solve_theta, lowest_bound, 50, f_st_standard, Delta_s_t, sigma_Y
#             )  # solve for theta
#             d_eta_st = np.maximum(
#                 -theta_t, Delta_s_t
#             )  # update max(Delta_s_t, -theta)
#             invest = Delta_s_t >= -theta_t
#             invest_fst = invest * f_st_standard
#             popu_parti_t = np.sum(cohort_size * invest)
#             Delta_bar_parti_t = np.sum(Delta_s_t * invest_fst)
#             f_parti_t = np.sum(invest_fst)
#             pi_st = (d_eta_st + theta_t) / sigma_S
#             age_t = np.sum(cohort_size * tau * invest)
#             n_parti_t = np.sum(invest) / Nc
#
#         elif mode_trade == 'comp':
#             f_st_standard = f_st * dt
#             Delta_bar_parti_t = np.sum(f_st_standard * Delta_s_t)
#             theta_t = sigma_Y - Delta_bar_parti_t
#             d_eta_st = Delta_s_t
#             invest = Delta_s_t >= -theta_t  # long stock
#             popu_parti_t = 1
#
#             f_parti_t = 1
#             pi_st = (d_eta_st + theta_t) / sigma_S
#             age_t = np.sum(cohort_size * tau * invest)
#             n_parti_t = np.sum(invest) / Nc
#
#         else:
#             print('Warning! Mode trade not defined')
#             exit()
#
#         r_t = (
#                 rho
#                 + mu_Y
#                 + nu - tax
#                 - sigma_Y * theta_t
#         )
#
#         mu_S_t = sigma_S * theta_t + r_t
#
#         # store the results
#         dR[i] = dR_t  # realized return from t-1 to t
#         theta[i] = theta_t
#         r[i] = r_t
#         f[i, :] = f_st
#         Delta[i, :] = Delta_s_t
#         max[i, :] = d_eta_st
#         f_parti[i] = f_parti_t
#         Delta_bar_parti[i] = Delta_bar_parti_t
#         pi[i, :] = pi_st
#         parti[i] = popu_parti_t
#         w_cohort[i, :] = w_cohort_st
#         age[i] = age_t
#         n_parti[i] = n_parti_t
#
#     return (
#         r,
#         theta,
#         f,
#         Delta,
#         max,
#         pi,
#         parti,
#         f_parti,
#         Delta_bar_parti,
#         dR,
#         w_cohort,
#         age,
#         n_parti,
#     )
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# def simulate_cohorts_partial_constraint(
#         Y: np.ndarray,
#         biasvec: np.ndarray,
#         dZ: np.ndarray,
#         Nt: int,
#         Nc: int,
#         tau: np.ndarray,
#         dt: float,
#         rho: float,
#         nu: float,
#         Vhat: float,
#         mu_Y: float,
#         sigma_Y: float,
#         sigma_S: float,
#         tax: float,
#         beta: float,
#         T_hat: float,
#         Npre: float,
#         Ninit: int,
#         mode_trade: str,
#         mode_learn: str,
#         cohort_size: np.ndarray,
#         Delta_s_t: np.ndarray,
#         eta_st_eta_ss: np.ndarray,
#         eta_bar: np.ndarray,
#         d_eta_st: np.ndarray,
#         invest_tracker_t: np.ndarray,
#         can_short_tracker_t: np.ndarray,
#         tau_info: np.ndarray,
#         good_time_simulate: np.ndarray,
# ) -> Tuple[
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
# ]:
#     """"" Simulate the economy forward, partial constraint case
#     Args:
#         biasvec (np.ndarray): pre-trading period shocks to form beliefs of the initial cohort, shape(Npre,)
#         dZt (np.ndarray): random shocks, shape (Nt)
#         Nt (int): number of periods in the simulation
#         Nc (int): number of cohorts in the economy
#         tau (np.ndarray): t-s, shape(Nt)
#         dt (float): per unit of time
#         rho (float): discount factor
#         nu (float): rate of birth and death
#         Vhat (float): initial variance
#         mu_Y (float): as in eq(1), drift of aggregate output growth
#         sigma_Y (float): as in eq(1), diffusion of aggregate output growth
#         sigma_S (float): as in eq(26), diffusion of stock price
#         tax (float): as in eq(18), consumption share of the newborn cohort
#         T_hat (float): pre-trading years
#         Npre (float): pre-trading number of obs
#         mode (str): versions of the model
#         - from the cohort_builder function -
#         f_st (np.ndarray): consumption share input
#         eta_st_ss (np.ndarray): disagreement input
#         eta_bar (np.ndarray): average disagreement input
#         Delta_s_t (np.ndarray): bias for each cohort, shape(Nc)
#         d_eta_st (np.ndarray): max(Delta_), shape(Nc)
#         invest_tracker (np.ndarry): shape(Nt)
#
#     Returns:
#         mu_S (np.ndarray): expected return under true measure at t, shape(Nt, )
#         mu_S_s (np.ndarray): expected stock return each cohort, shape(Nt, Nc, )
#         r (np.ndarray): interest rate, shape(Nt, )
#         theta (np.ndarray): market price of risk, shape(Nt, )
#         f (np.ndarray): consumption share over time, shape(Nt, Nc, )
#         Delta (np.ndarray): bias over time, shape(Nt, Nc, )
#         d_eta (np.ndarray): change of eta, shape(Nt, Nc, )
#         pi (np.ndarray): portfolio choice over time, shape(Nt, Nc, )
#         dR (np.ndarray): objective stock returns over time, shape(Nt, )
#         w (np.ndarray): individual wealth over time, shape(Nt, Nc, )
#         w_cohort (np.ndarray): cohort wealth over time, shape(Nt, Nc, )
#         popu_parti (np.ndarray): population that invest in stocks over time, shape(Nt, )
#         popu_can_short (np.ndarray): population that can short over time, shape(Nt, )
#         popu_short (np.ndarray): population that actually short in stocks over time, shape(Nt, )
#         popu_long (np.ndarray): population that actually long in stocks over time, shape(Nt, )
#         f_parti (np.ndarray): aggregate consumption share over time conditional on investing in stocks, shape(Nt, )
#         f_short (np.ndarray): aggregate consumption share over time conditional on shorting stocks, shape(Nt, )
#         f_long (np.ndarray): aggregate consumption share over time conditional on longing stocks, shape(Nt, )
#         age_parti (np.ndarray):  average age participating in the stock market, shape(Nt, )
#         age_short (np.ndarray):  average age shorting the stock market, shape(Nt, )
#         age_long (np.ndarray):  average age longing the stock market, shape(Nt, )
#         n_parti (np.ndarray): number of cohorts participating in the stock market, shape(Nt, )
#         invest_tracker (np.ndarray): tracks if a cohort is still in the market, shape(Nt, Nc, )
#         can_short_tracker (np.ndarray): tracks if a cohort can short, shape(Nt, Nc, )
#         long (np.ndarray): tracks all the cohorts over time, shape(Nt, Nc, )
#         short(np.ndarray): tracks all the cohorts over time, shape(Nt, Nc, )
#         Delta_bar_parti (np.ndarray): aggregate consumption weighted bias over time conditional on invest in stocks, shape(Nt, )
#         Delta_bar_long (np.ndarray): aggregate consumption weighted bias over time conditional on invest in stocks, shape(Nt, )
#         Delta_bar_short (np.ndarray): aggregate consumption weighted bias over time conditional on invest in stocks, shape(Nt, )
#     """ ""
#     # Initializing variables
#     # cohort-specific terms:
#     Delta = np.zeros((Nt, Nc))  # stores bias in beliefs
#     d_eta = np.zeros((Nt, Nc))  # stores max(delta, -theta)
#     f = np.zeros((Nt, Nc))  # evolution of cohort consumption share
#     pi = np.zeros((Nt, Nc))  # portfolio choices
#     w_cohort = np.zeros((Nt, Nc))  # evolution of wealth for cohorts
#     invest_tracker = np.zeros((Nt, Nc))
#     can_short_tracker = np.zeros((Nt, Nc))
#     long = np.zeros((Nt, Nc))
#     short = np.zeros((Nt, Nc))
#
#     # aggregate terms:
#     dR = np.zeros(Nt)  # stores stock returns
#     r = np.zeros(Nt)  # interest rate
#     theta = np.zeros(Nt)  # market price of risk
#     f_parti = np.zeros((Nt))  # consumption share of the stock market participants
#     Delta_bar_parti = np.zeros((Nt))  # average belief of all investors
#     Delta_bar_long = np.zeros((Nt))  # average belief of the long investors
#     Delta_bar_short = np.zeros((Nt))  # average belief of the short-sellers
#
#     dR_t = 0
#     age = np.zeros(Nt)
#     n_parti = np.zeros(Nt)
#
#     for i in tqdm(range(Nt)):
#         dZ_t = dZ[i]
#
#         # todo: eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
#         #  eta_bar_t is the denominator; it creates issues if too close to 0
#         #  so I rescale eta_bar to keep it away from 0, without changing f_st
#
#         # new cohort born (age 0), get wealth transfer, observe, invest
#         eta_st_eta_ss = eta_st_eta_ss * np.exp(
#             (-0.5 * d_eta_st ** 2) * dt
#             + d_eta_st * dZ_t
#         )  # equation (11)
#
#         eta_bar_parts = tax * np.exp(-tax * tau) * eta_bar * eta_st_eta_ss * dt  # equation (18)
#         eta_bar_t = np.sum(eta_bar_parts) / (1 - tax * dt)  # equation (18)
#
#         eta_st_eta_ss = np.append(eta_st_eta_ss[1:], 1)
#         eta_bar = np.append(eta_bar[1:], eta_bar_t)
#         eta_bar = eta_bar / eta_bar_t  # rescale, does not change the relative magnitude of each cohort
#
#         f_st = eta_bar_parts / eta_bar_t / dt
#         f_st = np.append(f_st[1:], tax)
#
#         # intvec = intvec * np.exp(
#         #     (-0.5 * d_eta_st ** 2 - tax) * dt
#         #     + d_eta_st * dZ_t
#         # )  # eq(18), intvec = tau * exp() * eta_bar_s * eta_st / eta_ss
#         #
#         # # add a new cohort
#         # # Cohort consumption (wealth) share:
#         # intvec = intvec[1:]
#         # eta_t = np.sum(intvec * dt) / (1 - tax * dt)
#         # intvec = np.append(intvec, tax * eta_t)
#         # intvec = intvec / eta_t
#         # f_st = intvec
#
#         # Wealth
#         if i == 0:
#             w_cohort_st = Y[i] / beta * f_st
#             w_st = w_cohort_st / cohort_size * dt
#         else:
#             dR_t = mu_S_t * dt + sigma_S * dZ_t  # realized stock return, mu_t^Sdt + sigma_t^Sdz_t
#             w_t = Y[i] / beta
#             dw_st = ((r_t + nu - tax - beta) + pi_st * (mu_S_t - r_t)) * w_st * dt + w_st * pi_st * sigma_S * dZ_t  # r_t, theta_t, pi_st from last loop, dZ_t just realized
#             w_st = w_st[1:] + dw_st[1:]
#             w_st = np.append(w_st, w_t * tax / nu)
#             w_cohort_st = w_st * cohort_size / dt
#
#         # update beliefs
#         dDelta_s_t = (
#             post_var(sigma_Y, Vhat, tau_info) / sigma_Y**2
#                      ) * (
#             -Delta_s_t * dt + dZ_t
#         )  # from eq(5)
#         if mode == 'back_renew':
#             tau_info = np.append(tau_info[1:], 0) + dt
#
#         if i < Npre - 1:
#             init_bias = (np.sum(biasvec[i+1:]) + np.sum(dZ[:i+1])) / T_hat
#         else:
#             init_bias = np.sum(dZ[i+1 - Npre: i+1]) / T_hat
#
#         Delta_s_t = Delta_s_t[1:] + dDelta_s_t[1:]
#         Delta_s_t = np.append(Delta_s_t, init_bias)
#
#         invest_tracker_t = np.append(invest_tracker_t[1:], 1)  # all cohorts that are still in the market, 1 by default
#         can_short_tracker_t = np.append(can_short_tracker_t[1:], 0)  # the cohorts that are allowed to short, 0 by default
#
#         if good_time_simulate[i] == 1:
#
#             if mode == 'back_collect':
#                 # agents who have left the market respond to the recent positive shocks
#                 # they collect all the information they missed during the drop period
#                 invest_tracker_t = np.ones(Nc)  # all can invest
#
#             if mode == 'back_renew':
#                 if i < Npre - 1:
#                     renew_bias = (np.sum(biasvec[i + 1:]) + np.sum(dZ[:i + 1])) / (Npre * dt)
#                 else:
#                     renew_bias = np.sum(dZ[i + 1 - Npre: i + 1]) / (Npre * dt)
#                 Delta_s_t = Delta_s_t * invest_tracker_t + renew_bias * (1 - invest_tracker_t)
#                 tau_info = invest_tracker_t * tau_info + (1 - invest_tracker_t) * dt
#                 invest_tracker_t = np.ones(Nc)  # all can invest
#
#         # find the market clearing theta, given beliefs and consumption shares of cohorts in the economy
#         f_st_possible = f_st * dt * invest_tracker_t
#         indiv_w_possible = f_st_possible / cohort_size
#         cohort_size_possible = cohort_size * invest_tracker_t
#         Delta_s_t_possible = Delta_s_t * invest_tracker_t
#         wealth_cutoff = find_the_rich(
#             indiv_w_possible, cohort_size_possible, top=0.05
#         )  # find the cohorts that make the richest 1% pupolation in the current period that are still in the market
#         can_short = indiv_w_possible >= wealth_cutoff  # these cohorts can short in this period
#         can_short_tracker_t = (
#             can_short_tracker_t + can_short >= 1
#         )  # once rich, always can short
#
#         theta_t = bisection_partial_constraint(
#             solve_theta_partial_constraint, -50, 50, can_short_tracker_t, Delta_s_t_possible, f_st_possible, sigma_Y
#         )
#         want_to_short = (Delta_s_t + theta_t) < 0
#         constrained = invest_tracker_t * want_to_short * (
#             1 - can_short_tracker_t
#         )  # in the market * want to short * can't short
#         short_t = want_to_short * can_short_tracker_t
#         long_t = (1 - want_to_short) * invest_tracker_t
#         invest_tracker_t = invest_tracker_t - constrained  # constrained people drop, update invest_tracker
#         pi_st = (Delta_s_t + theta_t) / sigma_S * invest_tracker_t
#         d_eta_st = Delta_s_t * invest_tracker_t - theta_t * (1 - invest_tracker_t)
#
#         r_t = (
#                 rho
#                 + mu_Y
#                 + nu - tax
#                 - sigma_Y * theta_t
#         )
#
#         mu_S_t = sigma_S * theta_t + r_t
#
#         # store the results
#         d_eta[i, :] = d_eta_st
#         dR[i] = dR_t  # realized return from dZt
#         theta[i] = theta_t
#         r[i] = r_t
#         f[i, :] = f_st
#         Delta[i, :] = Delta_s_t
#         invest_tracker[i, :] = invest_tracker_t
#         can_short_tracker[i, :] = can_short_tracker_t
#         long[i, :] = long_t
#         short[i, :] = short_t
#         w_cohort[i, :] = w_cohort_st
#         pi[i, :] = pi_st
#         Delta_bar_parti[i] = np.sum(Delta_s_t * invest_tracker_t * f_st) / np.sum(invest_tracker_t * f_st)
#         Delta_bar_long[i] = np.sum(Delta_s_t * long_t * f_st) / np.sum(long_t * f_st)
#         total_c_short = np.sum(short_t * f_st)
#         if total_c_short == 0:
#             Delta_bar_short[i] = np.nan
#         else:
#             Delta_bar_short[i] = np.sum(Delta_s_t * short_t * f_st) / total_c_short
#
#     popu_parti = np.sum(cohort_size * invest_tracker, 1)
#     popu_can_short = np.sum(cohort_size * can_short_tracker, 1)
#     popu_short = np.sum(cohort_size * short, 1)
#     popu_long = np.sum(cohort_size * long, 1)
#     f_parti = np.sum(invest_tracker * f_st * dt, 1)
#     f_short = np.sum(short * f_st * dt, 1)
#     f_long = np.sum(long * f_st * dt, 1)
#     age_parti = np.sum(cohort_size * tau * invest_tracker, 1)
#     age_short = np.sum(cohort_size * tau * short, 1)
#     age_long = np.sum(cohort_size * tau * long, 1)
#     n_parti = np.sum(invest_tracker, 1) / Nc
#
#     return (
#         r,
#         theta,
#         f,
#         Delta,
#         d_eta,
#         pi,
#         dR,
#         w_cohort,
#         popu_parti,
#         popu_can_short,
#         popu_short,
#         popu_long,
#         f_parti,
#         f_short,
#         f_long,
#         age_parti,
#         age_short,
#         age_long,
#         n_parti,
#         invest_tracker,
#         can_short_tracker,
#         long,
#         short,
#         Delta_bar_parti,
#         Delta_bar_long,
#         Delta_bar_short,
#     )






#
# def simulate_cohorts_fading(
#         Y: np.ndarray,
#         biasvec: np.ndarray,
#         dZ: np.ndarray,
#         Nt: int,
#         Nc: int,
#         tau: np.ndarray,
#         dt: float,
#         rho: float,
#         nu: float,
#         Vhat: float,
#         mu_Y: float,
#         sigma_Y: float,
#         sigma_S: float,
#         tax: float,
#         beta: float,
#         T_hat: float,
#         Npre: float,
#         Ninit: int,
#         mode: str,
#         cohort_size: np.ndarray,
#         intvec: np.ndarray,
#         Delta_s_t: np.ndarray,
#         d_eta_st: np.ndarray,
#         invest_tracker: np.ndarray,
#         int_zt: np.ndarray,
#         delta_ss: np.ndarray,
# ) -> Tuple[
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
# ]:
#     """"" Simulate the economy forward
#
#     Args:
#         biasvec (np.ndarray): pre-trading period shocks to form beliefs of the initial cohort, shape(Npre,)
#         dZt (np.ndarray): random shocks, shape (Nt)
#         Nt (int): number of periods in the simulation
#         Nc (int): number of cohorts in the economy
#         tau (np.ndarray): t-s, shape(Nt)
#         # IntVec (np.ndarray): ~similar to consumption share, shape(Nc)
#         dt (float): per unit of time
#         rho (float): discount factor
#         nu (float): rate of birth and death
#         Vhat (float): initial variance
#         mu_Y (float): as in eq(1), drift of aggregate output growth
#         sigma_Y (float): as in eq(1), diffusion of aggregate output growth
#         sigma_S (float): as in eq(26), diffusion of stock price
#         tax (float): as in eq(18), consumption share of the newborn cohort
#         T_hat (float): pre-trading years
#         Npre (float): pre-trading number of obs
#         mode (str): versions of the model
#         - from the cohort_builder function -
#         f_st (np.ndarray): consumption share input
#         eta_st_ss (np.ndarray): disagreement input
#         eta_bar (np.ndarray): average disagreement input
#         Delta_s_t (np.ndarray): bias for each cohort, shape(Nc)
#         d_eta_st (np.ndarray): max(Delta_), shape(Nc)
#         invest_tracker (np.ndarry): shape(Nt)
#
#     Returns:
#         mu_S (np.ndarray): expected return under true measure at t, shape(Nt, )
#         mu_S_s (np.ndarray): expected stock return each cohort, shape(Nt, Nc, )
#         r (np.ndarray): interest rate, shape(Nt, )
#         theta (np.ndarray): market price of risk, shape(Nt, )
#         f (np.ndarray): consumption share over time, shape(Nt, Nc, )
#         Delta (np.ndarray): bias over time, shape(Nt, Nc, )
#         max (np.ndarray): max(-theta, delta_s_t) over time, shape(Nt, Nc, )
#         pi (np.ndarray): portfolio choice over time, shape(Nt, Nc, )
#         parti (np.ndarray): population that invest in stocks over time, shape(Nt, )
#         f_parti (np.ndarray): aggregate consumption share over time conditional on invest in stocks, shape(Nt, )
#         Delta_bar_parti (np.ndarray): aggregate consumption weighted bias over time conditional on invest in stocks, shape(Nt, )
#         dR (np.ndarray): change of stock returns over time, shape(Nt, )
#         w (np.ndarray): individual wealth, shape(Nt, Nc, )
#         w_cohort (np.ndarray): cohort wealth = individual wealth * cohort size, shape(Nt, Nc, )
#         age (np.ndarray):  average age participating in the stock market, shape(Nt, )
#         n_parti (np.ndarray): number of cohorts participating in the stock market, shape(Nt, )
#     """ ""
#     # Initializing variables
#     # cohort-specific terms:
#     Delta = np.zeros((Nt, Nc))  # stores bias in beliefs
#     max = np.zeros((Nt, Nc))  # stores max(delta, -theta)
#     f = np.zeros((Nt, Nc))  # evolution of cohort consumption share
#     pi = np.zeros((Nt, Nc))  # portfolio choices
#     w_cohort = np.zeros((Nt, Nc))  # evolution of wealth for cohorts
#     short = np.zeros((Nt, Nc))
#
#     # aggregate terms:
#     dR = np.zeros(Nt)  # stores stock returns
#     r = np.zeros(Nt)  # interest rate
#     theta = np.zeros(Nt)  # market price of risk
#     f_parti = np.zeros((Nt))  # consumption share of the stock market participants
#     Delta_bar_parti = np.zeros((Nt))  # disagreement of the stock market participants
#     parti = np.zeros((Nt))  # participation rate
#
#     dR_t = 0
#     age = np.zeros(Nt)
#     n_parti = np.zeros(Nt)
#
#     for i in tqdm(range(Nt)):
#         dZ_t = dZ[i]
#
#         intvec = intvec * np.exp(
#             (-0.5 * d_eta_st ** 2 - tax) * dt
#             + d_eta_st * dZ_t
#         )  # eq(18), intvec = tau * exp() * eta_bar_s * eta_st / eta_ss
#
#         # add a new cohort
#         # Cohort consumption (wealth) share:
#         intvec = intvec[1:]
#         eta_t = np.sum(intvec * dt) / (1 - tax * dt)
#         intvec = np.append(intvec, tax * eta_t)
#         intvec = intvec / eta_t
#         f_st = intvec
#
#         # Wealth
#         if i == 0:
#             w_cohort_st = Y[i] / beta * f_st
#             w_st = w_cohort_st / cohort_size * dt
#         else:
#             dR_t = mu_S_t * dt + sigma_S * dZ_t  # realized stock return, mu_t^Sdt + sigma_t^Sdz_t
#             w_t = Y[i] / beta
#             dw_st = ((r_t + nu - tax - beta) + pi_st * (mu_S_t - r_t)) * w_st * dt + w_st * pi_st * sigma_S * dZ_t  # r_t, theta_t, pi_st from last loop, dZ_t just realized
#             w_st = w_st[1:] + dw_st[1:]
#             w_st = np.append(w_st, w_t * tax / nu)
#             w_cohort_st = w_st * cohort_size / dt
#
#         # update beliefs
#         Delta_s_t = fadingmemo(v, tau, sigma_Y, Vhat, int_zt, delta_ss)
#         if i < Npre-1:
#            init_bias = (np.sum(biasvec[i+1:]) + np.sum(dZ[:i+1])) / T_hat
#         else:
#            init_bias = np.sum(dZ[i+1 - Npre: i+1]) / T_hat
#         delta_ss = np.append(delta_ss[1:], init_bias)
#         int_zt = np.append(int_zt[1:], 0)
#         int_zt = int_zt * (1 - v) ** dt + dZ_t
#         Delta_s_t = np.append(Delta_s_t[1:], init_bias)
#
#
#         # find the market clearing theta, given beliefs and consumption shares of cohorts in the economy
#         if mode == 'disappointment':
#             invest_tracker = invest_tracker[1:]
#             invest_tracker = np.append(invest_tracker, 1)
#             possible_cons_share = f_st * dt * invest_tracker
#             possible_delta_st = Delta_s_t * invest_tracker
#             lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
#             theta_t = bisection(
#                 solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
#             )  # solve for theta
#             a = Delta_s_t + theta_t
#             invest = (a >= 0)
#             #want_to_short_t = invest_tracker * (1 - invest)
#             invest_tracker = invest * invest_tracker
#             d_eta_st = a * invest_tracker - theta_t
#             invest_fst = invest_tracker * f_st * dt
#             popu_parti_t = np.sum(cohort_size * invest_tracker)
#             Delta_bar_parti_t = np.sum(Delta_s_t * invest_fst)
#             f_parti_t = np.sum(invest_fst)
#             pi_st = (d_eta_st + theta_t) / sigma_S
#             age_t = np.sum(cohort_size * tau * invest_tracker)
#             n_parti_t = np.sum(invest_tracker) / Nc
#
#         elif mode == 'reentry':
#             lowest_bound = -np.max(Delta_s_t)  # absolute lower bound for theta
#             f_st_standard = f_st * dt
#             theta_t = bisection(
#                 solve_theta, lowest_bound, 50, f_st_standard, Delta_s_t, sigma_Y
#             )  # solve for theta
#             d_eta_st = np.maximum(
#                 -theta_t, Delta_s_t
#             )  # update max(Delta_s_t, -theta)
#             invest = Delta_s_t >= -theta_t
#             invest_fst = invest * f_st_standard
#             popu_parti_t = np.sum(cohort_size * invest)
#             Delta_bar_parti_t = np.sum(Delta_s_t * invest_fst)
#             f_parti_t = np.sum(invest_fst)
#             pi_st = (d_eta_st + theta_t) / sigma_S
#             age_t = np.sum(cohort_size * tau * invest)
#             n_parti_t = np.sum(invest) / Nc
#
#         elif mode == 'comp':
#             f_st_standard = f_st * dt
#             Delta_bar_parti_t = np.sum(f_st_standard * Delta_s_t)
#             theta_t = sigma_Y - Delta_bar_parti_t
#             d_eta_st = Delta_s_t
#             invest = Delta_s_t >= -theta_t  # long stock
#             popu_parti_t = 1
#
#             f_parti_t = 1
#             pi_st = (d_eta_st + theta_t) / sigma_S
#             age_t = np.sum(cohort_size * tau * invest)
#             n_parti_t = np.sum(invest) / Nc
#
#         else:
#             print('Warning! Mode not defined')
#             exit()
#
#         r_t = (
#                 rho
#                 + mu_Y
#                 + nu - tax
#                 - sigma_Y * theta_t
#         )
#
#         mu_S_t = sigma_S * theta_t + r_t
#
#         # store the results
#         dR[i] = dR_t  # realized return from t-1 to t
#         theta[i] = theta_t
#         r[i] = r_t
#         f[i, :] = f_st
#         Delta[i, :] = Delta_s_t
#         max[i, :] = d_eta_st
#         f_parti[i] = f_parti_t
#         Delta_bar_parti[i] = Delta_bar_parti_t
#         pi[i, :] = pi_st
#         parti[i] = popu_parti_t
#         w_cohort[i, :] = w_cohort_st
#         age[i] = age_t
#         n_parti[i] = n_parti_t
#
#     return (
#         r,
#         theta,
#         f,
#         Delta,
#         max,
#         pi,
#         parti,
#         f_parti,
#         Delta_bar_parti,
#         dR,
#         w_cohort,
#         age,
#         n_parti,
#     )