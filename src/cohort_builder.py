import numpy as np
from src.solver import bisection, solve_theta, find_the_rich, bisection_partial_constraint, solve_theta_partial_constraint
from tqdm import tqdm
from typing import Tuple
from src.stats import post_var, dDelta_st_calculator
from numba import jit

def build_cohorts_SI(
    dZ_build: np.ndarray,
    dZ_SI_build: np.ndarray,
    Nc: int,
    dt: float,
    tau: np.ndarray,
    rho: float,
    nu: float,
    Vhat: float,
    mu_Y: float,
    sigma_Y: float,
    tax: float,
    phi: float,
    Npre: int,
    Ninit: int,
    T_hat: float,
    mode_trade: str,
    mode_learn: str,
    top = 0.05,
    old_limit = 100,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """builds up a sufficiently large set of cohorts in the economy, view each cohort as one agent with a constantly shrinking size

    Args:
        dZ_build (np.ndarray): random shocks of aggregate output for each period, shape (Nc-1, )
        Nc (int): number of periods  = number of cohorts in the economy
        dt (float): unit of time
        rho (float): rho, discount factor
        nu (float): birth / death rate, each cohort starts at size nu and shrinks at speed of nu
        Vhat (float): initial variance of beliefs
        mu_Y (float): mean of aggregate output growth
        sigma_Y (float): sd of aggregate output growth
        tax (float): marginal rate of wealth tax
        Npre (int): pre-trading periods
        T_hat (float): pre-trading years
        mode (str): describes the mode

    Returns:
        f_st (np.ndarray): consumption shares
        Delta_s_t (np.ndarray): bias, shape(Nc, )
        # eta_st_ss (np.ndarray): consumption change process, shape(Nc, )
        # eta_bar (np.ndarray): consumption weighted disagreement, shape(Nc, )
        d_eta_st (np.ndarray): max(delta_s_t, -theta_t), shape(Nc, )
        invest_tracker (np.ndarray): track if a cohort is still in the risky market
    """
    Delta_s_t = np.zeros(1)  # belief bias, eq(3)
    d_eta_st = np.zeros(1)  # disagreement, eq(11)
    eta_bar = np.ones(1)
    eta_st_eta_ss = np.ones(1)
    invest_tracker = np.ones(Ninit) if mode_trade != 'complete' else np.ones(Nc)
    can_short_tracker = np.zeros(Ninit) if mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old' else np.zeros(Nc)
    tau_info = np.ones(1) * dt
    Vhat_vector = np.ones(1) * Vhat
    a_phi = (1 - phi ** 2)
    phi_sqr_a_phi = phi / np.sqrt(a_phi)
    a_phi_1 = 1 / a_phi
    sigma_Y_sq = sigma_Y ** 2
    cohort_size = nu * np.exp(-nu * (tau - dt)) * dt
    theta_mat = np.empty(Nc)

    for i in tqdm(range(1, Nc)):
    #for i in tqdm(range(1, Ninit)):
        # new cohort born (age 0), get wealth transfer, observe, invest
        tau_short = tau[-i:]
        dZ_build_t = dZ_build[i - 1]
        dZ_SI_build_t = dZ_SI_build[i - 1]

        eta_st_eta_ss = eta_st_eta_ss * np.exp(
            (-0.5 * d_eta_st ** 2) * dt
            + d_eta_st * dZ_build_t
        )  # equation (11)

        eta_bar_parts = tax * np.exp(-tax * tau_short) * eta_bar * eta_st_eta_ss * dt    # equation (18)
        eta_bar_t = np.sum(eta_bar_parts) / ( 1 - tax * dt)  # equation (18)  # dividing by (1-tax*dt) keeps sum(f_st*dt) at 1
        # eta_bar_t = np.sum(eta_bar_parts)

        eta_st_eta_ss = np.append(eta_st_eta_ss, 1)
        eta_bar = np.append(eta_bar, eta_bar_t)
        eta_bar = eta_bar / eta_bar_t  # rescale, does not change the relative magnitude of each cohort
        # todo: eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
        #  eta_bar_t is the denominator; it creates issues if too close to 0
        #  so I rescale eta_bar to keep it away from 0, without changing f_st

        f_st = eta_bar_parts / eta_bar_t / dt
        f_st = np.append(f_st, tax)

        # update beliefs
        if mode_trade == 'complete':
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'P')
        elif mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
            if i < Ninit:
                V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
                dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'P')
            else:
                V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')
                dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'N')  # from eq(5)
                V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
                dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'P')  # from eq(8)
                dDelta_s_t = invest_tracker * dDelta_s_t_P + (1 - invest_tracker) * dDelta_s_t_N

        else:
            print('mode_trade not found')
            exit()


        # add a new cohort to Vhat_vector and tau_info
        Vhat_vector = np.append(Vhat_vector, Vhat)
        if mode_trade == 'complete':  # where tau_info is the same with age; no switch between N and P for complete market
            tau_info = tau[-i - 1:]
        else:  # where tau_info is the time distance from the latest state switch
            tau_info = np.append(tau_info, 0) + dt

        if i < Npre:
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(Delta_s_t, 0)  # newborns begin with 0 bias when there are not enough earlier observations
        else:
            init_bias = np.average(dZ_build[int(i - Npre) : i]) / dt
            Delta_s_t+= dDelta_s_t
            Delta_s_t = np.append(
                Delta_s_t, init_bias
            )  # newborns begin with Npre earlier observations of the dividend process

        # find the market clearing theta, given beliefs and consumption shares
        if i < Ninit or mode_trade == 'complete':  # Ninit: initial rounds where the short-sale constraint is relaxed
            d_eta_st = (
                Delta_s_t  # relax the short-sale constraint in the beginning
            )

        elif mode_trade == 'w_constraint':
            invest_tracker = np.append(invest_tracker, 1)  # indicator of current type, =1 for a cohort if type == P

            if mode_learn == 'disappointment':  # agents switch from type P to type N once constrained, and stay as type N
                possible_cons_share = f_st * dt * invest_tracker
                possible_delta_st = Delta_s_t * invest_tracker
                lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound
                theta_t = bisection(
                    solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
                )
                a = Delta_s_t + theta_t
                invest = (a > 0)
                switch_P_to_N = invest_tracker * (1 - invest)
                invest_tracker = invest * invest_tracker
                d_eta_st = a * invest_tracker - theta_t
                # tau_info and V_hat has to change for the agents who switched to N
                Vhat_vector = np.append(V_st_P, Vhat) * switch_P_to_N + Vhat_vector * (1 - switch_P_to_N)  # reset initial variance
                tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock

            elif mode_learn == 'reentry':   # agents switch between type P and type N
                possible_cons_share = f_st * dt
                possible_delta_st = Delta_s_t
                lowest_bound = -np.max(possible_delta_st)  # absolute lower bound
                theta_t = bisection(
                    solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
                )
                a = Delta_s_t + theta_t
                invest = (a > 0)
                switch_P_to_N = invest_tracker * (1 - invest)
                switch_N_to_P = np.maximum(invest - invest_tracker, 0)
                switch = switch_N_to_P + switch_P_to_N
                invest_tracker = invest
                d_eta_st = a * invest_tracker - theta_t

                # tau_info and V_hat has to change for the agents who switched to N
                Vhat_vector = np.append(V_st_P, Vhat) * switch_P_to_N + np.append(V_st_N, Vhat) * switch_N_to_P + Vhat_vector * (1 - switch)  # reset initial variance
                tau_info = dt * switch + tau_info * (1 - switch)  # reset clock

            else:
                print('mode_learn not found')
                break

        # elif mode_trade == 'partial_constraint_rich':
        #     invest_tracker = np.append(invest_tracker, 1)  # indicator of current type, =1 for a cohort if type == P, by default type == P as newborn
        #     can_short_tracker = np.append(can_short_tracker, 0)
        #     cohort_size_short = cohort_size[-i-1:]  # todo: think about cohort size in the building function
        #
        #     if mode_learn == 'disappointment':  # agents switch from type P to type N once constrained, and stay as type N
        #         possible_cons_share = f_st * dt * invest_tracker
        #         possible_delta_st = Delta_s_t * invest_tracker
        #         indiv_w_possible = possible_cons_share / cohort_size_short
        #         cohort_size_possible = cohort_size_short * invest_tracker
        #         wealth_cutoff = find_the_rich(indiv_w_possible, cohort_size_possible,
        #                                       top)  # find the cohorts that make the richest 1% pupolation in the current period that are still in the market
        #         can_short = indiv_w_possible >= wealth_cutoff  # these cohorts can short in this period
        #         can_short_tracker = (can_short_tracker + can_short >= 1)
        #
        #         lowest_bound = -np.max(possible_delta_st)  # absolute lower bound
        #         theta_t = bisection_partial_constraint(
        #             solve_theta_partial_constraint, -50, 50, can_short_tracker, possible_delta_st, possible_cons_share,
        #             sigma_Y
        #         )
        #         a = Delta_s_t + theta_t
        #         invest = 1 - (a < 0) * (can_short_tracker < 1)
        #         switch_P_to_N = invest_tracker * (1 - invest)
        #         invest_tracker = invest * invest_tracker
        #         d_eta_st = a * invest_tracker - theta_t
        #
        #         # tau_info and V_hat has to change for the agents who switched to N
        #         Vhat_vector = np.append(V_st_P, Vhat) * switch_P_to_N + Vhat_vector * (
        #                 1 - switch_P_to_N)  # reset initial variance
        #         tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock
        #         # print(np.sum(can_short_tracker * cohort_size_short))
        #
        #     elif mode_learn == 'reentry':  # agents switch between type P and type N
        #         possible_cons_share = f_st * dt
        #         possible_delta_st = Delta_s_t
        #         indiv_w_possible = possible_cons_share / cohort_size_short
        #         cohort_size_possible = cohort_size_short
        #         wealth_cutoff = find_the_rich(indiv_w_possible, cohort_size_possible, top)  # find the cohorts that make the richest 1% pupolation in the current period that are still in the market
        #         can_short = indiv_w_possible >= wealth_cutoff  # these cohorts can short in this period
        #         can_short_tracker = (can_short_tracker + can_short >= 1)
        #
        #         lowest_bound = -np.max(possible_delta_st)  # absolute lower bound
        #         theta_t = bisection_partial_constraint(
        #             solve_theta_partial_constraint, -50, 50, can_short_tracker, possible_delta_st, possible_cons_share,
        #             sigma_Y
        #         )
        #         a = Delta_s_t + theta_t
        #         invest = 1 - (a < 0) * (can_short_tracker < 1)
        #         switch_P_to_N = invest_tracker * (1 - invest)
        #         switch_N_to_P = np.maximum(invest - invest_tracker, 0)
        #         switch = switch_N_to_P + switch_P_to_N
        #         invest_tracker = invest
        #         d_eta_st = a * invest_tracker - theta_t
        #
        #         # tau_info and V_hat has to change for the agents who switched to N
        #         Vhat_vector = np.append(V_st_P, Vhat) * switch_P_to_N + np.append(V_st_N,
        #                           Vhat) * switch_N_to_P + Vhat_vector * (
        #                           1 - switch)  # reset initial variance
        #         tau_info = dt * switch + tau_info * (1 - switch)  # reset clock
        #         # print(np.sum(can_short_tracker * cohort_size_short))
        #
        #     else:
        #         print('mode_learn not found')
        #         break

        elif mode_trade == 'partial_constraint_old':
            invest_tracker = np.append(invest_tracker, 1)  # indicator of current type, =1 for a cohort if type == P, by default type == P as newborn
            # can_short_tracker = np.append(can_short_tracker, 0)
            # cohort_size_short = cohort_size[-i-1:]
            tau_short_1 = tau[-i-1:]

            if mode_learn == 'disappointment':  # agents switch from type P to type N once constrained, and stay as type N
                # cohorts older than 100 years can short, if they are still type P by the time
                possible_cons_share = f_st * dt * invest_tracker
                possible_delta_st = Delta_s_t * invest_tracker
                can_short_possible = (tau_short_1 >= old_limit)
                can_short_tracker = can_short_possible * invest_tracker

                lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound
                theta_t = bisection_partial_constraint(
                    solve_theta_partial_constraint, lowest_bound, 50, can_short_tracker, possible_delta_st, possible_cons_share,
                    sigma_Y
                )
                a = Delta_s_t + theta_t
                invest = 1 - (a < 0) * (can_short_tracker < 1)
                switch_P_to_N = invest_tracker * (1 - invest)
                invest_tracker = invest * invest_tracker
                d_eta_st = a * invest_tracker - theta_t

                # tau_info and V_hat has to change for the agents who switched to N
                Vhat_vector = np.append(V_st_P, Vhat) * switch_P_to_N + Vhat_vector * (
                        1 - switch_P_to_N)  # reset initial variance
                tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock
                # print(np.sum(can_short_tracker * cohort_size_short))

            elif mode_learn == 'reentry':  # agents switch between type P and type N
                possible_cons_share = f_st * dt
                possible_delta_st = Delta_s_t
                can_short_tracker = (tau_short_1 >= old_limit)

                lowest_bound = -np.max(possible_delta_st)  # absolute lower bound
                theta_t = bisection_partial_constraint(
                    solve_theta_partial_constraint, lowest_bound, 50, can_short_tracker, possible_delta_st, possible_cons_share,
                    sigma_Y
                )
                a = Delta_s_t + theta_t
                invest = 1 - (a < 0) * (can_short_tracker < 1)
                switch_P_to_N = invest_tracker * (1 - invest)
                switch_N_to_P = np.maximum(invest - invest_tracker, 0)
                switch = switch_N_to_P + switch_P_to_N
                invest_tracker = invest
                d_eta_st = a * invest_tracker - theta_t

                # tau_info and V_hat has to change for the agents who switched to N
                Vhat_vector = np.append(V_st_P, Vhat) * switch_P_to_N + np.append(V_st_N,
                                  Vhat) * switch_N_to_P + Vhat_vector * (
                                  1 - switch)  # reset initial variance
                tau_info = dt * switch + tau_info * (1 - switch)  # reset clock
                # print(np.sum(can_short_tracker * cohort_size_short))

            else:
                print('mode_learn not found')
                break

        else:
            print('mode_trade not found')
            break
        #theta_mat[i] = theta_t

    #print(theta_mat)

    return (
        Delta_s_t,
        eta_st_eta_ss,
        eta_bar,
        d_eta_st,
        invest_tracker,
        tau_info,
        Vhat_vector,
        can_short_tracker,
    )






# def build_cohorts_SI(
#     dZ_build: np.ndarray,
#     dZ_SI_build: np.ndarray,
#     Nc: int,
#     dt: float,
#     tau: np.ndarray,
#     rho: float,
#     nu: float,
#     Vhat: float,
#     mu_Y: float,
#     sigma_Y: float,
#     tax: float,
#     phi: float,
#     Npre: int,
#     Ninit: int,
#     T_hat: float,
#     good_time_build: np.ndarray,
#     mode_trade: str,
#     mode_learn: str,
# ) -> Tuple[
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
# ]:
#     """builds up a sufficiently large set of cohorts in the economy, view each cohort as one agent with a constantly shrinking size
#
#     Args:
#         dZ_build (np.ndarray): random shocks of aggregate output for each period, shape (Nc-1, )
#         Nc (int): number of periods  = number of cohorts in the economy
#         dt (float): unit of time
#         rho (float): rho, discount factor
#         nu (float): birth / death rate, each cohort starts at size nu and shrinks at speed of nu
#         Vhat (float): initial variance of beliefs
#         mu_Y (float): mean of aggregate output growth
#         sigma_Y (float): sd of aggregate output growth
#         tax (float): marginal rate of wealth tax
#         Npre (int): pre-trading periods
#         T_hat (float): pre-trading years
#         mode (str): describes the mode
#
#     Returns:
#         f_st (np.ndarray): consumption shares
#         Delta_s_t (np.ndarray): bias, shape(Nc, )
#         # eta_st_ss (np.ndarray): consumption change process, shape(Nc, )
#         # eta_bar (np.ndarray): consumption weighted disagreement, shape(Nc, )
#         d_eta_st (np.ndarray): max(delta_s_t, -theta_t), shape(Nc, )
#         invest_tracker (np.ndarray): track if a cohort is still in the risky market
#     """
#     Delta_s_t = np.zeros(1)  # belief bias, eq(3)
#     d_eta_st = np.zeros(1)  # disagreement, eq(11)
#     eta_bar = np.ones(1)
#     eta_st_eta_ss = np.ones(1)
#     invest_tracker = np.ones(Ninit) if mode_trade != 'complete' else np.ones(Nc)
#     tau_info = np.ones(1) * dt
#     Vhat_vector = np.ones(1) * Vhat
#
#     for i in tqdm(range(1, Nc)):
#     #for i in tqdm(range(1, Ninit)):
#         # new cohort born (age 0), get wealth transfer, observe, invest
#         tau_short = tau[-i:]
#
#         eta_st_eta_ss = eta_st_eta_ss * np.exp(
#             (-0.5 * d_eta_st ** 2) * dt
#             + d_eta_st * dZ_build[i - 1]
#         )  # equation (11)
#
#         eta_bar_parts = tax * np.exp(-tax * tau_short) * eta_bar * eta_st_eta_ss * dt    # equation (18)
#         eta_bar_t = np.sum(eta_bar_parts) / ( 1 - tax * dt)  # equation (18)  # dividing by (1-tax*dt) keeps sum(f_st*dt) at 1
#         # eta_bar_t = np.sum(eta_bar_parts)
#
#         eta_st_eta_ss = np.append(eta_st_eta_ss, 1)
#         eta_bar = np.append(eta_bar, eta_bar_t)
#         eta_bar = eta_bar / eta_bar_t  # rescale, does not change the relative magnitude of each cohort
#         # todo: eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
#         #  eta_bar_t is the denominator; it creates issues if too close to 0
#         #  so I rescale eta_bar to keep it away from 0, without changing f_st
#
#         f_st = eta_bar_parts / eta_bar_t / dt
#         f_st = np.append(f_st, tax)
#
#         # update beliefs
#
#
#         if i < Ninit:
#             V_st_P = post_var(sigma_Y, Vhat_vector, tau_info, phi, 'P')
#             dDelta_s_t = V_st_P / sigma_Y ** 2 * (
#                         1 / (1 - phi ** 2)
#                     ) * (
#                         -Delta_s_t * dt + dZ_build[i - 1] + phi * dZ_SI_build[i - 1]
#                            )
#         else:
#             V_st_N = post_var(sigma_Y, Vhat_vector, tau_info, phi, 'N')
#             dDelta_s_t_N = (V_st_N / sigma_Y ** 2
#                             ) * (
#                                    -Delta_s_t * dt + dZ_build[i - 1]
#                            )  # from eq(5)
#             V_st_P = post_var(sigma_Y, Vhat_vector, tau_info, phi, 'P')
#             dDelta_s_t_P = V_st_P / sigma_Y ** 2 * (
#                         1 / (1 - phi ** 2)
#                     ) * (
#                         -Delta_s_t * dt + dZ_build[i - 1] + phi * dZ_SI_build[i - 1]
#                            )  # from eq(8)
#             dDelta_s_t = invest_tracker * dDelta_s_t_P + (1 - invest_tracker) * dDelta_s_t_N
#
#         Vhat_vector = np.append(Vhat_vector, Vhat)  # iterate Vhat (Vhat is either the initial variance or the starting point after state switch)
#
#         if mode_learn == 'reentry' or mode_trade == 'complete':  # where tau_info is the same with age
#             tau_info = tau[-i - 1:]
#         else:  # where tau_info is the distance from state switch
#             tau_info = np.append(tau_info, 0) + dt
#
#         if i < Npre:
#             Delta_s_t += dDelta_s_t
#             Delta_s_t = np.append(Delta_s_t, 0)  # newborns begin with 0 bias when there are not enough earlier observations
#         else:
#             init_bias = np.average(dZ_build[int(i - Npre) : i]) / dt
#             Delta_s_t+= dDelta_s_t
#             Delta_s_t = np.append(
#                 Delta_s_t, init_bias
#             )  # newborns begin with Npre earlier observations
#
#         # find the market clearing theta, given beliefs and consumption shares
#         if i < Ninit or mode_trade == 'complete':  # Ninit: initial rounds where the short-sale constraint is relaxed
#             d_eta_st = (
#                 Delta_s_t  # relax the short-sale constraint in the beginning
#             )
#         elif mode_trade == 'w_constraint':
#             invest_tracker = np.append(invest_tracker, 1)  # indicator of the participating cohorts, new cohort by default can invest
#             possible_cons_share = f_st * dt * invest_tracker
#             possible_delta_st = Delta_s_t * invest_tracker
#             lowest_bound = -np.max(possible_delta_st[np.nonzero(
#                 possible_delta_st)])  # absolute lower bound for theta among active investors
#             theta_t = bisection(
#                 solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
#             )
#             a = Delta_s_t + theta_t
#             invest = (a >= 0)
#             if mode_learn == 'disappointment':  # agents switch from type P to type N once constrained
#                 switch_P_to_N = invest_tracker * (1 - invest)
#                 invest_tracker = invest * invest_tracker
#                 d_eta_st = a * invest_tracker - theta_t
#                 # tau_info and V_hat has to change for the agents who switched to N
#                 Vhat_vector = np.append(V_st_P, Vhat) * switch_P_to_N + Vhat_vector * (1 - switch_P_to_N)  # reset initial variance
#                 tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock
#
#             # elif mode_learn == 'reentry':   # agents stay as type P even if constrained
#             #     invest_tracker = np.append(invest_tracker[:-1], invest[-1])
#             #     d_eta_st = a * invest_tracker - theta_t
#
#             else:
#                 print('mode_learn not found')
#                 break
#
#         else:
#             print('mode_trade not found')
#             break
#
#     return (
#         Delta_s_t,
#         eta_st_eta_ss,
#         eta_bar,
#         d_eta_st,
#         invest_tracker,
#         tau_info,
#         Vhat_vector,
#     )




# def build_cohorts(
#     dZ_build: np.ndarray,
#     Nc: int,
#     dt: float,
#     tau: np.ndarray,
#     rho: float,
#     nu: float,
#     Vhat: float,
#     mu_Y: float,
#     sigma_Y: float,
#     tax: float,
#     Npre: int,
#     Ninit: int,
#     T_hat: float,
#     good_time_build: np.ndarray,
#     mode_trade: str,
#     mode_learn: str,
# ) -> Tuple[
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
# ]:
#     """builds up a sufficiently large set of cohorts in the economy, view each cohort as one agent with a constantly shrinking size
#
#     Args:
#         dZ_build (np.ndarray): random shocks of aggregate output for each period, shape (Nc-1, )
#         Nc (int): number of periods  = number of cohorts in the economy
#         dt (float): unit of time
#         rho (float): rho, discount factor
#         nu (float): birth / death rate, each cohort starts at size nu and shrinks at speed of nu
#         Vhat (float): initial variance of beliefs
#         mu_Y (float): mean of aggregate output growth
#         sigma_Y (float): sd of aggregate output growth
#         tax (float): marginal rate of wealth tax
#         Npre (int): pre-trading periods
#         T_hat (float): pre-trading years
#         mode (str): describes the mode
#
#     Returns:
#         f_st (np.ndarray): consumption shares
#         Delta_s_t (np.ndarray): bias, shape(Nc, )
#         # eta_st_ss (np.ndarray): consumption change process, shape(Nc, )
#         # eta_bar (np.ndarray): consumption weighted disagreement, shape(Nc, )
#         d_eta_st (np.ndarray): max(delta_s_t, -theta_t), shape(Nc, )
#         invest_tracker (np.ndarray): track if a cohort is still in the risky market
#     """
#     Delta_s_t = np.zeros(1)  # belief bias, eq(3)
#     d_eta_st = np.zeros(1)  # disagreement, eq(11)
#     eta_bar = np.ones(1)
#     eta_st_eta_ss = np.ones(1)
#     invest_tracker = np.ones(Ninit) if mode_trade == 'disappointment' else np.ones(Nc)
#     tau_info = np.ones(1) * dt
#
#     for i in tqdm(range(1, Nc)):
#
#         # new cohort born (age 0), get wealth transfer, observe, invest
#         tau_short = tau[-i:]
#
#         eta_st_eta_ss = eta_st_eta_ss * np.exp(
#             (-0.5 * d_eta_st ** 2) * dt
#             + d_eta_st * dZ_build[i - 1]
#         )  # equation (11)
#
#         eta_bar_parts = tax * np.exp(-tax * tau_short) * eta_bar * eta_st_eta_ss * dt    # equation (18)
#         eta_bar_t = np.sum(eta_bar_parts) / ( 1 - tax * dt)  # equation (18)  # dividing by (1-tax*dt) keeps sum(f_st*dt) at 1
#         # eta_bar_t = np.sum(eta_bar_parts)
#
#         eta_st_eta_ss = np.append(eta_st_eta_ss, 1)
#         eta_bar = np.append(eta_bar, eta_bar_t)
#         eta_bar = eta_bar / eta_bar_t  # rescale, does not change the relative magnitude of each cohort
#         # todo: eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
#         #  eta_bar_t is the denominator; it creates issues if too close to 0
#         #  so I rescale eta_bar to keep it away from 0, without changing f_st
#
#         f_st = eta_bar_parts / eta_bar_t / dt
#         f_st = np.append(f_st, tax)
#
#         # update beliefs
#         dDelta_s_t = (post_var(sigma_Y, Vhat, tau_info) / sigma_Y**2
#                       ) * (
#                 -Delta_s_t * dt + dZ_build[i - 1]
#         )  # from eq(5)
#
#         if mode_learn == 'back_renew' and mode_trade == 'disappointment':
#             tau_info = np.append(tau_info, 0) + dt
#         else:
#             tau_info = tau[-i - 1:]
#
#         if i < Npre:
#             Delta_s_t += dDelta_s_t
#             Delta_s_t = np.append(Delta_s_t, 0)  # newborns begin with 0 bias when there are not enough earlier observations
#         else:
#             init_bias = np.average(dZ_build[int(i - Npre) : i]) / dt
#             Delta_s_t+= dDelta_s_t
#             Delta_s_t = np.append(
#                 Delta_s_t, init_bias
#             )  # newborns begin with Npre earlier observations
#
#         # find the market clearing theta, given beliefs and consumption shares
#         if i < Ninit or mode_trade == 'comp':  # Ninit: initial rounds where the short-sale constraint is relaxed
#             d_eta_st = (
#                 Delta_s_t  # relax the short-sale constraint in the beginning
#             )
#         else:
#             invest_tracker = np.append(invest_tracker, 1)  # all cohorts that are still in the market, new cohort by default can invest
#             if mode_trade == 'disappointment':
#                 if good_time_build[i - 1] == 1:
#                     if mode_learn == 'back_collect':
#                         # agents who have left the market respond to the recent positive shocks
#                         # they collect all the information they missed during the drop period
#                         invest_tracker = np.ones(i + 1)  # all can invest
#
#                     if mode_learn == 'back_renew':
#                         renew_bias = np.sum(dZ_build[int(i - Npre): i]) / (Npre * dt)
#                         Delta_s_t = Delta_s_t * invest_tracker + renew_bias * (1 - invest_tracker)
#                         tau_info = invest_tracker * tau_info + (1 - invest_tracker) * dt
#                         invest_tracker = np.ones(i + 1)
#
#                 possible_cons_share = f_st * dt * invest_tracker
#                 possible_delta_st = Delta_s_t * invest_tracker
#                 lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound for theta among active investors
#                 theta_t = bisection(
#                     solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
#                 )  # solve for theta, 10 is a far away upper bound for theta
#                 a = Delta_s_t + theta_t
#                 invest = (a >= 0)
#                 invest_tracker = invest * invest_tracker
#                 d_eta_st = a * invest_tracker - theta_t
#
#             if mode_trade == 'reentry':
#                 lowest_bound = -np.max(Delta_s_t[np.nonzero(Delta_s_t)]) # absolute lower bound for theta
#                 f_st_standard = f_st * dt
#                 theta_t = bisection(
#                     solve_theta, lowest_bound, 50, f_st_standard, Delta_s_t, sigma_Y
#                 )  # solve for theta
#                 d_eta_st = np.maximum(
#                     -theta_t, Delta_s_t
#                 )  # update max(Delta_s_t, -theta)
#             # print(theta_t)
#
#     return (
#         Delta_s_t,
#         eta_st_eta_ss,
#         eta_bar,
#         d_eta_st,
#         invest_tracker,
#         tau_info,
#     )
#
#
#
#
#
# def build_cohorts_partial_constraint(
#         dZ_build: np.ndarray,
#         Nc: int,
#         dt: float,
#         tau: np.ndarray,
#         cohort_size: np.ndarray,
#         rho: float,
#         nu: float,
#         Vhat: float,
#         mu_Y: float,
#         sigma_Y: float,
#         tax: float,
#         Npre: int,
#         Ninit: int,
#         T_hat: float,
#         good_time_build: np.ndarray,
#         mode_trade: str,
#         mode_learn: str,
# ) -> Tuple[
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
# ]:
#     """builds up a sufficiently large set of cohorts in the economy, view each cohort as one agent with a constantly shrinking size
#
#     Args:
#         dZ_build (np.ndarray): random shocks of aggregate output for each period, shape (Nc-1, )
#         Nc (int): number of periods  = number of cohorts in the economy
#         dt (float): unit of time
#         rho (float): rho, discount factor
#         nu (float): birth / death rate, each cohort starts at size nu and shrinks at speed of nu
#         Vhat (float): initial variance of beliefs
#         mu_Y (float): mean of aggregate output growth
#         sigma_Y (float): sd of aggregate output growth
#         tax (float): marginal rate of wealth tax
#         Npre (int): pre-trading periods
#         T_hat (float): pre-trading years
#         good_time_build (np.ndarray): an information indicator that attracts attention of all agents
#         mode (str): describes the mode
#
#     Returns:
#         f_st (np.ndarray): consumption shares
#         Delta_s_t (np.ndarray): bias, shape(Nc, )
#         eta_st_ss (np.ndarray): consumption change process, shape(Nc, )
#         eta_bar (np.ndarray): consumption weighted disagreement, shape(Nc, )
#         d_eta_st (np.ndarray): summarizes xi_st, shape(Nc, )
#         invest_tracker (np.ndarray): track if a cohort is still in the risky market
#         can_short_tracker (np.ndarray): track if a cohort can short
#     """
#     Delta_s_t = np.zeros(1)  # belief bias, eq(3)
#     d_eta_st = np.zeros(1)  # disagreement, eq(11)
#     eta_bar = np.ones(1)
#     eta_st_eta_ss = np.ones(1)
#     theta = np.zeros(Nc)
#     invest_tracker = np.ones(Ninit)
#     can_short_tracker = np.ones(Ninit)
#     tau_info = np.ones(1) * dt
#
#     if mode_trade in ['rich_free'] and mode_learn in ['give_up', 'back_collect', 'back_renew']:
#         for i in tqdm(range(1, Nc)):
#             # for i in tqdm(range(1, Ninit)):
#             # new cohort born (age 0), get wealth transfer, observe, invest
#             tau_short = tau[-i:]
#
#             eta_st_eta_ss = eta_st_eta_ss * np.exp(
#                 (-0.5 * d_eta_st ** 2) * dt
#                 + d_eta_st * dZ_build[i - 1]
#             )  # equation (11)
#
#             eta_bar_parts = tax * np.exp(-tax * tau_short) * eta_bar * eta_st_eta_ss * dt  # equation (18)
#             eta_bar_t = np.sum(eta_bar_parts) / (
#                         1 - tax * dt)  # equation (18)  # dividing by (1-tax*dt) keeps sum(f_st*dt) at 1
#             # eta_bar_t = np.sum(eta_bar_parts)
#
#             eta_st_eta_ss = np.append(eta_st_eta_ss, 1)
#             eta_bar = np.append(eta_bar, eta_bar_t)
#             eta_bar = eta_bar / eta_bar_t  # rescale, does not change the relative magnitude of each cohort
#             # todo: eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
#             #  eta_bar_t is the denominator; it creates issues if too close to 0
#             #  so I rescale eta_bar to keep it away from 0, without changing f_st
#
#             f_st = eta_bar_parts / eta_bar_t / dt
#             f_st = np.append(f_st, tax)
#
#             # update beliefs
#             dDelta_s_t = post_var(sigma_Y, Vhat, tau_info, phi, 'N') / sigma_Y ** 2 * (
#                                  -Delta_s_t * dt + dZ_build[i - 1]
#                          )  # from eq(7)
#
#             if mode_learn == 'back_renew':
#                 tau_info = np.append(tau_info, 0) + dt
#             else:
#                 tau_info = tau[-i - 1:]
#
#             if i < Npre:
#                 Delta_s_t += dDelta_s_t
#                 Delta_s_t = np.append(Delta_s_t,
#                                       0)  # newborns begin with 0 bias when there are not enough earlier observations
#             else:
#                 init_bias = np.average(dZ_build[int(i - Npre): i]) / dt
#                 Delta_s_t += dDelta_s_t
#                 Delta_s_t = np.append(
#                     Delta_s_t, init_bias
#                 )  # newborns begin with Npre earlier observations
#
#             # find the market clearing theta, given beliefs and consumption shares
#             if i < Ninit:
#                 d_eta_st = (
#                     Delta_s_t  # relax the short-sale constraint in the beginning
#                 )
#
#             else:
#                 # add a new cohort in the trackers
#                 invest_tracker = np.append(invest_tracker,
#                                            1)  # all cohorts that are still in the market, new cohort by default can invest
#                 can_short_tracker = np.append(can_short_tracker,
#                                                 0)  # some cohorts that are allowed to short, new cohort by default can't short
#                 cohort_size_short = cohort_size[-i - 1:]
#
#                 if good_time_build[i - 1] == 1:
#
#                     if mode_learn == 'back_collect':
#                         # agents who have left the market respond to the recent positive shocks
#                         # they collect all the information they missed during the drop period
#                         invest_tracker = np.ones(i + 1)  # all can invest
#
#                     if mode_learn == 'back_renew':
#                         renew_bias = np.sum(dZ_build[int(i - Npre): i]) / (Npre * dt)
#                         Delta_s_t = Delta_s_t * invest_tracker + renew_bias * (1 - invest_tracker)
#                         tau_info = invest_tracker * tau_info + (1 - invest_tracker) * dt
#                         invest_tracker = np.ones(i + 1)
#
#                 f_st_possible = f_st * dt * invest_tracker
#                 indiv_w_possible = f_st_possible / cohort_size_short
#                 cohort_size_possible = cohort_size_short * invest_tracker
#                 Delta_s_t_possible = Delta_s_t * invest_tracker
#                 wealth_cutoff = find_the_rich(
#                     indiv_w_possible, cohort_size_possible, top=0.05
#                 )  # find the cohorts that make the richest 1% pupolation in the current period that are still in the market
#                 can_short = indiv_w_possible >= wealth_cutoff  # these cohorts can short in this period
#                 can_short_tracker = (
#                         can_short_tracker + can_short >= 1
#                 )  # once rich, always can short
#
#                 theta_t = bisection_partial_constraint(
#                     solve_theta_partial_constraint, -50, 50, can_short_tracker, Delta_s_t_possible, f_st_possible,
#                     sigma_Y
#                 )
#                 want_to_short = (Delta_s_t + theta_t) < 0
#                 constrained = invest_tracker * want_to_short * (
#                         1 - can_short_tracker
#                 )  # in the market * want to short * can't short
#                 short_t = want_to_short * can_short_tracker
#                 long_t = (1 - want_to_short) * invest_tracker
#                 invest_tracker = invest_tracker - constrained  # constrained people drop, update invest_tracker
#                 pi_st = (Delta_s_t + theta_t) / sigma_S * invest_tracker
#                 d_eta_st = Delta_s_t * invest_tracker - theta_t * (1 - invest_tracker)
#     else:
#         print('mode not found')
#
#     return (
#         Delta_s_t,
#         eta_st_eta_ss,
#         eta_bar,
#         d_eta_st,
#         invest_tracker,
#         can_short_tracker,
#         tau_info,
#     )
#



#
# def build_cohorts_fading(
#     dZ_build: np.ndarray,
#     Nc: int,
#     dt: float,
#     tau: np.ndarray,
#     rho: float,
#     nu: float,
#     Vhat: float,
#     mu_Y: float,
#     sigma_Y: float,
#     tax: float,
#     Npre: int,
#     Ninit: int,
#     T_hat: float,
#     v: float,
#     mode: str
# ) -> Tuple[
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
#     np.ndarray,
# ]:
#     """builds up a sufficiently large set of cohorts in the economy, view each cohort as one agent with a constantly shrinking size
#
#     Args:
#         dZ_build (np.ndarray): random shocks of aggregate output for each period, shape (Nc-1, )
#         Nc (int): number of periods  = number of cohorts in the economy
#         dt (float): unit of time
#         rho (float): rho, discount factor
#         nu (float): birth / death rate, each cohort starts at size nu and shrinks at speed of nu
#         Vhat (float): initial variance of beliefs
#         mu_Y (float): mean of aggregate output growth
#         sigma_Y (float): sd of aggregate output growth
#         tax (float): marginal rate of wealth tax
#         Npre (int): pre-trading periods
#         T_hat (float): pre-trading years
#         mode (str): describes the mode
#
#     Returns:
#         f_st (np.ndarray): consumption shares
#         Delta_s_t (np.ndarray): bias, shape(Nc, )
#         # eta_st_ss (np.ndarray): consumption change process, shape(Nc, )
#         # eta_bar (np.ndarray): consumption weighted disagreement, shape(Nc, )
#         d_eta_st (np.ndarray): max(delta_s_t, -theta_t), shape(Nc, )
#         invest_tracker (np.ndarray): track if a cohort is still in the risky market
#     """
#     Delta_s_t = np.zeros(1)  # belief bias, eq(3)
#     d_eta_st = np.zeros(1)  # disagreement, eq(11)
#     eta_bar = np.ones(1)
#     eta_st_ss = np.ones(1)
#     f_st = np.ones(1)
#     theta = np.zeros(Nc)
#     invest_tracker = np.ones(Nc) if mode == 'comp' else np.ones(Ninit)
#     reduction = np.exp(-tax * dt)
#     intvec = 1 / dt
#
#     int_zt = dZ_build[0]
#     delta_ss = np.zeros(1)
#
#     for i in tqdm(range(1, Nc)):
#         tau_short = tau[-i:]
#
#         intvec = intvec * np.exp(
#             (-0.5 * d_eta_st ** 2 - tax) * dt
#             + d_eta_st * dZ_build[i - 1]
#         )  # eq(18), intvec = tau * exp() * eta_bar_s * eta_st / eta_ss
#
#         # add a new cohort
#         # Cohort consumption (wealth) share:
#         eta_t = np.sum(intvec * dt) / (1 - tax * dt)
#         intvec = np.append(intvec, tax * eta_t)
#         intvec = intvec / eta_t
#         f_st = intvec
#
#         # alternatively:
#         Delta_s_t = fadingmemo(v, tau_short, sigma_Y, Vhat, int_zt, delta_ss)
#         if i < Npre:
#             DELbias = 0
#         else:
#             DELbias = np.sum(dZ_build[int(i - Npre) : i]) / T_hat  # newborns begin with Npre earlier observations
#         delta_ss = np.append(delta_ss, DELbias)
#         int_zt = np.append(int_zt, 0)
#         int_zt = int_zt * (1 - v) ** dt + dZ_build[i - 1]
#         Delta_s_t = np.append(Delta_s_t, DELbias)
#
#         # find the market clearing theta, given beliefs and consumption shares
#         if i < Ninit or mode == 'comp':  # Ninit: initial rounds where the short-sale constraint is relaxed
#             d_eta_st = (
#                 Delta_s_t  # relax the short-sale constraint in the beginning
#             )
#         else:
#             if mode == 'disappointment':
#                 invest_tracker = np.append(invest_tracker, 1)
#                 possible_cons_share = f_st * dt * invest_tracker
#                 possible_delta_st = Delta_s_t * invest_tracker
#                 lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound for theta among active investors
#                 theta_t = bisection(
#                     solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
#                 )  # solve for theta, 10 is a far away upper bound for theta
#                 a = Delta_s_t + theta_t
#                 invest = (a >= 0)
#                 invest_tracker = invest * invest_tracker
#                 d_eta_st = a * invest_tracker - theta_t
#
#                 theta[i] = theta_t
#
#             if mode == 'reentry':
#                 lowest_bound = -np.max(Delta_s_t[np.nonzero(Delta_s_t)]) # absolute lower bound for theta
#                 f_st_standard = f_st * dt
#                 theta_t = bisection(
#                     solve_theta, lowest_bound, 50, f_st_standard, Delta_s_t, sigma_Y
#                 )  # solve for theta
#                 d_eta_st = np.maximum(
#                     -theta_t, Delta_s_t
#                 )  # update max(Delta_s_t, -theta)
#             # print(theta_t)
#
#     if mode == 'reentry':
#         invest_tracker = (Delta_s_t >= -theta_t)
#
#     return (
#         Delta_s_t,
#         d_eta_st,
#         invest_tracker,
#         intvec,
#         int_zt,
#         delta_ss,
#     )