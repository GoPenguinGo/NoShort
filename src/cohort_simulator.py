import numpy as np
from typing import Tuple
from src.stats import post_var, fadingmemo
from src.solver import bisection, solve_theta, find_the_rich, bisection_partial_constraint, solve_theta_partial_constraint
from tqdm import tqdm
from numba import jit


def simulate_cohorts_SI(
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
        tau_info: np.ndarray,
        Vhat_vector: np.ndarray,
        good_time_simulate: np.ndarray,
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
    # cohort-specific terms:
    Delta = np.zeros((Nt, Nc))  # stores bias in beliefs
    max = np.zeros((Nt, Nc))  # stores max(delta, -theta)
    f = np.zeros((Nt, Nc))  # evolution of cohort consumption share
    pi = np.zeros((Nt, Nc))  # portfolio choices
    w_cohort = np.zeros((Nt, Nc))  # evolution of wealth for cohorts
    short = np.zeros((Nt, Nc))

    # aggregate terms:
    dR = np.zeros(Nt)  # stores stock returns
    r = np.zeros(Nt)  # interest rate
    theta = np.zeros(Nt)  # market price of risk
    f_parti = np.zeros((Nt))  # consumption share of the stock market participants
    Delta_bar_parti = np.zeros((Nt))  # disagreement of the stock market participants
    parti = np.zeros((Nt))  # participation rate

    dR_t = 0
    age = np.zeros(Nt)
    n_parti = np.zeros(Nt)

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

        eta_bar_parts = tax * np.exp(-tax * tau) * eta_bar * eta_st_eta_ss * dt  # equation (18)
        eta_bar_t = np.sum(eta_bar_parts) / (1 - tax * dt)  # equation (18)

        eta_st_eta_ss = np.append(eta_st_eta_ss[1:], 1)
        eta_bar = np.append(eta_bar[1:], eta_bar_t)
        eta_bar = eta_bar / eta_bar_t  # rescale, does not change the relative magnitude of each cohort

        f_st = eta_bar_parts / eta_bar_t / dt
        f_st = np.append(f_st[1:], tax)

        # Wealth
        if i == 0:
            w_cohort_st = Y[i] / beta * f_st
            w_st = w_cohort_st / cohort_size * dt
        else:
            dR_t = mu_S_t * dt + sigma_S * dZ_t  # realized stock return, mu_t^Sdt + sigma_t^Sdz_t
            w_t = Y[i] / beta
            dw_st = ((r_t + nu - tax - beta) + pi_st * (mu_S_t - r_t)) * w_st * dt + w_st * pi_st * sigma_S * dZ_t  # r_t, theta_t, pi_st from last loop, dZ_t just realized
            w_st = w_st[1:] + dw_st[1:]
            w_st = np.append(w_st, w_t * tax / nu)
            w_cohort_st = w_st * cohort_size / dt

        # update beliefs
        V_st_N = post_var(sigma_Y, Vhat_vector, tau_info, phi, 'N')
        dDelta_s_t_N = (V_st_N / sigma_Y ** 2
                        ) * (
                               -Delta_s_t * dt + dZ_t
                       )  # from eq(5)
        V_st_P = post_var(sigma_Y, Vhat_vector, tau_info, phi, 'P')
        dDelta_s_t_P = V_st_P / sigma_Y ** 2 * (
                        1 / (1 - phi ** 2)
                    ) * (
                        -Delta_s_t * dt + dZ_t + phi * dZ_SI_t
                           )
        dDelta_s_t = invest_tracker * dDelta_s_t_P + (1 - invest_tracker) * dDelta_s_t_N

        Vhat_vector = np.append(Vhat_vector[1:], Vhat)

        if mode_learn == 'keep' or mode_trade == 'complete':  # where tau_info is the same with age
            tau_info = tau
        else:  # where tau_info is the distance from state switch
            tau_info = np.append(tau_info[1:], 0) + dt

        if i < Npre-1:
            init_bias = (np.sum(biasvec[i+1:]) + np.sum(dZ[:i+1])) / T_hat
        else:
            init_bias = np.sum(dZ[i+1 - Npre: i+1]) / T_hat

        Delta_s_t = Delta_s_t[1:] + dDelta_s_t[1:]
        Delta_s_t = np.append(Delta_s_t, init_bias)

        # find the market clearing theta, given beliefs and consumption shares of cohorts in the economy
        invest_tracker = np.append(invest_tracker[1:], 1)  # all cohorts that are still in the market, 1 by default

        if mode_trade == 'w_constraint':
            possible_cons_share = f_st * dt * invest_tracker
            possible_delta_st = Delta_s_t * invest_tracker
            lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
            theta_t = bisection(
                solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
            )  # solve for theta
            a = Delta_s_t + theta_t
            invest = (a >= 0)

            if mode_learn == 'drop':
                switch_P_to_N = invest_tracker * (1 - invest)
                # tau_info and V_hat has to change for the agents who switched to N
                Vhat_vector = np.append(V_st_P[1:], Vhat) * switch_P_to_N + Vhat_vector * (1 - switch_P_to_N)  # reset initial variance
                tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock

                invest_tracker = invest * invest_tracker  # update invest tracker
                d_eta_st = a * invest_tracker - theta_t
                invest_fst = invest_tracker * f_st * dt
                popu_parti_t = np.sum(cohort_size * invest_tracker)
                Delta_bar_parti_t = np.sum(Delta_s_t * invest_fst)
                f_parti_t = np.sum(invest_fst)
                pi_st = (d_eta_st + theta_t) / sigma_S
                age_t = np.sum(cohort_size * tau * invest_tracker)
                n_parti_t = np.sum(invest_tracker) / Nc

            # elif mode_learn == 'keep':   # agents stay as type P even if constrained
            #     invest_tracker = np.append(invest_tracker[:-1], invest[-1])
            #     d_eta_st = a * invest_tracker - theta_t

            else:
                print('mode_learn not found')
                break

        elif mode_trade == 'complete':
            f_st_standard = f_st * dt
            Delta_bar_parti_t = np.sum(f_st_standard * Delta_s_t)
            theta_t = sigma_Y - Delta_bar_parti_t
            d_eta_st = Delta_s_t
            invest = Delta_s_t >= -theta_t  # long stock
            popu_parti_t = 1

            f_parti_t = 1
            pi_st = (d_eta_st + theta_t) / sigma_S
            age_t = np.sum(cohort_size * tau * invest)
            n_parti_t = np.sum(invest) / Nc

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

        # store the results
        dR[i] = dR_t  # realized return from t-1 to t
        theta[i] = theta_t
        r[i] = r_t
        f[i, :] = f_st
        Delta[i, :] = Delta_s_t
        max[i, :] = d_eta_st
        f_parti[i] = f_parti_t
        Delta_bar_parti[i] = Delta_bar_parti_t
        pi[i, :] = pi_st
        parti[i] = popu_parti_t
        w_cohort[i, :] = w_cohort_st
        age[i] = age_t
        n_parti[i] = n_parti_t

    return (
        r,
        theta,
        f,
        Delta,
        max,
        pi,
        parti,
        f_parti,
        Delta_bar_parti,
        dR,
        w_cohort,
        age,
        n_parti,
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
#         if mode_learn == 'back_renew' and mode_trade == 'drop':
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
#         if mode_trade == 'drop':
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
#         elif mode_trade == 'keep':
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
#         if mode == 'drop':
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
#         elif mode == 'keep':
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