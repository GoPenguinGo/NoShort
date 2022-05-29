import numpy as np
from src.solver import bisection, solve_theta, bisection_partial_constraint, solve_theta_partial_constraint
from tqdm import tqdm
from typing import Tuple
from src.stats import post_var
from numba import jit


def build_cohorts(
    dZt: np.ndarray,
    Nc: int,
    dt: float,
    tau: np.ndarray,
    rho: float,
    nu: float,
    Vhat: float,
    mu_Y: float,
    sigma_Y: float,
    beta: float,
    Npre: int,
    T_hat: float,
    mode: str
) -> Tuple[
    # np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    # np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """builds up a sufficiently large set of cohorts in the economy, view each cohort as one agent with a constantly shrinking size

    Args:
        dZt (np.ndarray): random shocks of aggregate output for each period, shape (Nc-1, )
        Nc (int): number of periods  = number of cohorts in the economy
        dt (float): unit of time
        rho (float): rho, discount factor
        nu (float): birth / death rate, each cohort starts at size nu and shrinks at speed of nu
        Vhat (float): initial variance of beliefs
        mu_Y (float): mean of aggregate output growth
        sigma_Y (float): sd of aggregate output growth
        beta (float): marginal rate of wealth tax
        # omega (float): marginal propensity to consume
        T_hat (float): pre-trading years
        mode (str): describes the mode

    Returns:
        f_st (np.ndarray): consumption shares
        Delta_s_t (np.ndarray): bias, shape(Nc, )
        eta_st_ss (np.ndarray): consumption change process, shape(Nc, )
        eta_bar (np.ndarray): consumption weighted disagreement, shape(Nc, )
        MaxThetaDelta_s_t (np.ndarray): max(delta_s_t, -theta_t), shape(Nc, )
        invest_tracker (np.ndarray): track if a cohort is still in the risky market
    """
    Delta_s_t = np.zeros(1)  # belief bias, eq(3)
    MaxThetaDelta_s_t = np.zeros(1)  # disagreement, eq(11)
    eta_bar = np.ones(1)
    eta_st_ss = np.ones(1)
    f_st = np.ones(1)
    invest_tracker = np.ones(Nc) if mode == 'complete' else np.ones(Npre)
    if mode == 'learn_hard':
        bad_times = dZt <= np.percentile(dZt, 1)  # bottom 1%
        invest = np.ones(Npre)
        invest_when_bad = np.ones(Npre)

    for i in tqdm(range(1, Nc)):
        tau_short = tau[-i:]

        eta_st_ss = eta_st_ss * np.exp(
            -0.5 * MaxThetaDelta_s_t ** 2 * dt
            + MaxThetaDelta_s_t * dZt[i - 1]
        )

        Part = beta * np.exp(-beta * tau_short) * eta_bar * eta_st_ss
        eta_bar_t = np.sum(Part * dt) / (1 - beta * dt)
        eta_bar = np.append(eta_bar, eta_bar_t)
        eta_st_ss = np.append(eta_st_ss, 1)

        f_st = Part / eta_bar_t
        f_st = np.append(f_st, beta)    # cohort consumption share

        # update beliefs
        dDelta_s_t = (post_var(sigma_Y, Vhat, tau_short) / sigma_Y**2
                      ) * (
            -Delta_s_t * dt + dZt[i - 1]
        )  # from eq(5)
        if i < Npre:
            Delta_s_t = Delta_s_t + dDelta_s_t
            Delta_s_t = np.append(Delta_s_t, 0)  # newborns begin with 0 bias when there are not enough earlier observations
        else:
            DELbias = np.sum(dZt[int(i - Npre) : i]) / T_hat
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(
                Delta_s_t, DELbias
            )  # newborns begin with Npre earlier observations

        # find the market clearing theta, given beliefs and consumption shares
        if i < Npre or mode == 'complete':
            MaxThetaDelta_s_t = (
                Delta_s_t  # relax the short-sale constraint in the beginning
            )
        else:
            if mode == 'drop':
                invest_tracker = np.append(invest_tracker, 1)
                possible_cons_share = f_st * dt * invest_tracker
                possible_delta_st = Delta_s_t * invest_tracker
                lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
                theta_t = bisection(
                    solve_theta, lowest_bound, 10, possible_cons_share, possible_delta_st, sigma_Y
                )  # solve for theta
                a = Delta_s_t + theta_t
                invest = (a >= 0)
                invest_tracker = invest * invest_tracker
                MaxThetaDelta_s_t = a * invest_tracker - theta_t

            if mode == 'keep':
                lowest_bound = -np.max(Delta_s_t)  # absolute lower bound for theta
                f_st_standard = f_st * dt
                theta_t = bisection(
                    solve_theta, lowest_bound, 10, f_st_standard, Delta_s_t, sigma_Y
                )  # solve for theta
                MaxThetaDelta_s_t = np.maximum(
                    -theta_t, Delta_s_t
                )  # update max(Delta_s_t, -theta)

    if mode == 'keep':
        invest_tracker = (Delta_s_t >= -theta_t)

    return (
        f_st,
        Delta_s_t,
        eta_st_ss,
        eta_bar,
        MaxThetaDelta_s_t,
        invest_tracker,
    )




def build_cohorts_partial_constraint(
        dZt: np.ndarray,
        Nc: int,
        dt: float,
        tau: np.ndarray,
        cohort_size: np.ndarray,
        rho: float,
        nu: float,
        Vhat: float,
        mu_Y: float,
        sigma_Y: float,
        beta: float,
        Npre: int,
        T_hat: float,
        good_time_build: np.ndarray,
        mode: str
) -> Tuple[
    # np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    # np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """builds up a sufficiently large set of cohorts in the economy, view each cohort as one agent with a constantly shrinking size

    Args:
        # todo: edit the description
        dZt (np.ndarray): random shocks of aggregate output for each period, shape (Nc-1, )
        Nc (int): number of periods  = number of cohorts in the economy
        dt (float): unit of time
        rho (float): rho, discount factor
        nu (float): birth / death rate, each cohort starts at size nu and shrinks at speed of nu
        Vhat (float): initial variance of beliefs
        mu_Y (float): mean of aggregate output growth
        sigma_Y (float): sd of aggregate output growth
        beta (float): marginal rate of wealth tax
        # omega (float): marginal propensity to consume
        T_hat (float): pre-trading years
        mode (str): describes the mode

    Returns:
        # DeltaConditional (np.ndarray): consumption weighted aggregate max(delta_s_t, -theta_t), as in eq(19), shape(Nc, )
        # IntVec (np.ndarray): ~similar to consumption share, shape(Nc, )
        # Xt (np.ndarray): xi_t * Yt, shape(Nc, )
        Delta_s_t (np.ndarray): bias, shape(Nc, )
        # Yt (np.ndarray): aggregate output, shape(Nc, )
        # Zt (np.ndarray): cumulated shocks, shape(Nc, )
        # consumptionshare (np.ndarray): shape(Nc, )
        tau (np.ndarray): t-s, shape(Nc, )
        MaxThetaDelta_s_t (np.ndarray): max(delta_s_t, -theta_t), shape(Nc, )
        # TODO: @chingyulin: use NamedTuple for the return
    """
    Delta_s_t = np.zeros(1)  # belief bias, eq(3)
    d_eta_st_ss = np.zeros(1)  # disagreement, eq(11)
    eta_bar = np.ones(1)
    eta_st_ss = np.ones(1)
    f_st = np.ones(1)
    invest_tracker = np.ones(Npre)
    can_short_tracker = np.zeros(Npre)
    for i in tqdm(range(1, Nc)):
        tau_short = tau[-i:]

        eta_st_ss = eta_st_ss * np.exp(
            -0.5 * d_eta_st_ss ** 2 * dt
            + d_eta_st_ss * dZt[i - 1]
        )

        Part = beta * np.exp(-beta * tau_short) * eta_bar * eta_st_ss
        eta_bar_t = np.sum(Part * dt) / (1 - beta * dt)
        eta_bar = np.append(eta_bar, eta_bar_t)
        eta_st_ss = np.append(eta_st_ss, 1)

        f_st = Part / eta_bar_t
        f_st = np.append(f_st, beta)  # cohort consumption share

        # update beliefs
        dDelta_s_t = (post_var(sigma_Y, Vhat, tau_short) / sigma_Y ** 2
                      ) * (
                             -Delta_s_t * dt + dZt[i - 1]
                     )  # from eq(5)
        if i < Npre:
            Delta_s_t = Delta_s_t + dDelta_s_t
            Delta_s_t = np.append(Delta_s_t,
                                  0)  # newborns begin with 0 bias when there are not enough earlier observations
        else:
            DELbias = np.sum(dZt[int(i - Npre): i]) / T_hat  # todo: should this be different for agents coming back in the renew case?
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(
                Delta_s_t, DELbias
            )  # newborns begin with Npre earlier observations

        # find the market clearing theta, given beliefs and consumption shares
        if i < Npre:
            d_eta_st_ss = (
                Delta_s_t  # relax the short-sale constraint in the beginning
            )
        else:
            # add a new cohort in the trackers
            invest_tracker = np.append(invest_tracker, 1)  # all cohorts that are still in the market, new cohort by default can invest
            can_short_tracker = np.append(can_short_tracker, 0)  # all cohorts that are allowed to short, new cohort by default can't short
            cohort_size_short = cohort_size[-i - 1:]

            if mode == 'back_collect' and good_time_build[i - 1] == 1:
                # agents who have left the market respond to the recent positive shocks
                # they collect all the information they missed during the drop period
                invest_tracker_t = np.ones(i + 1)  # all can invest

            if mode == 'back_renew' and good_time_build[i - 1] == 1:
                return_bias = np.sum(dZt[int(i - window): i]) / (window * dt)
                Delta_s_t = Delta_s_t * invest_tracker_t + return_bias * (1 - invest_tracker_t)
                invest_tracker_t = np.ones(i + 1)

            f_st_possible = f_st * dt * invest_tracker
            indiv_w_possible = f_st_possible / cohort_size_short
            cohort_size_possible = cohort_size_short * invest_tracker
            Delta_s_t_possible = Delta_s_t * invest_tracker
            wealth_cutoff = find_the_rich(
                    indiv_w_possible, cohort_size_possible, top=0.05
                )  # find the cohorts that make the richest 5% pupolation in the current period that are still in the market
            can_short = indiv_w_possible >= wealth_cutoff  # these cohorts can start shorting in this period if they couldn't before
            can_short_tracker = (can_short_tracker + can_short >= 1)   # once rich, always can short

            theta_t = bisection_partial_constraint(
                    solve_theta_partial_constraint, -10, 10, can_short_tracker, Delta_s_t_possible, f_st_possible, sigma_Y
            )
            want_to_short = (Delta_s_t + theta_t) < 0
            constrained = invest_tracker * want_to_short * (1 - can_short_tracker)  # in the market * want to short * can't short
            invest_tracker = invest_tracker - constrained  # constrained people drop, update invest_tracker
            d_eta_st_ss = Delta_s_t * invest_tracker - theta_t * (1 - invest_tracker)

    return (
        f_st,
        Delta_s_t,
        eta_st_ss,
        eta_bar,
        d_eta_st_ss,
        invest_tracker,
        can_short_tracker,
    )