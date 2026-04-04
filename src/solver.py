import numpy as np
from typing import Callable


# def bisection(
#         optimfun: Callable[[float, np.ndarray, np.ndarray, float], np.float64],
#         xlow: np.float64,
#         xhigh: np.float64,
#         arg1: np.ndarray,
#         arg2: np.ndarray,
#         arg3: float,
#         eps: float = 1e-9,
# ) -> np.float64:
#     """Bisection method to solve x (theta)
#
#     Args:
#         optimfun (Callable[[float, np.ndarray, np.ndarray], float]): the function we want to find the root for
#         xlow (float): lower bound for x
#         xhigh (float): upper bound for x
#         arg1 (np.ndarray): second input for optimfun
#         arg2 (np.ndarray): third input for optimfun
#         eps (float, optional): converging criteria. Precision of estimation. Defaults to 1e-6.
#
#     Returns:
#         xmid: the estimated value that makes the optimfun close to 0
#
#     """
#     flow = optimfun(xlow, arg1, arg2, arg3)
#     fhigh = optimfun(xhigh, arg1, arg2, arg3)
#     diff = 1
#     iter = 0
#     xmid = 100000
#
#     while diff > eps:
#         xmid = (xlow + xhigh) / 2
#         fmid = optimfun(xmid, arg1, arg2, arg3)
#         if flow * fmid < 0:  # root between flow and fmid
#             xhigh = xmid
#             fhigh = fmid
#         elif fmid * fhigh < 0:  # root between fmid and fhigh
#             xlow = xmid
#             flow = fmid
#         diff = abs(fhigh - flow)
#         iter += 1
#         if iter > 50:
#             print("Warning! It takes more than 50 iteration to converge.")
#             break
#     return xmid
#
#
#
# def solve_theta(
#         theta_guess: np.float64,
#         consumption_share: np.ndarray,
#         Delta_s_t: np.ndarray,
#         sigma_Y: float,
# ) -> np.float64:
#     """RHS - LHS of the eq(22), used to iteratively solve theta
#
#     Args:
#         theta_guess (np.float64): any potential value of theta
#         consumption_share (np.ndarray): shape (T, ), fst as in the def for the experience component of eq(24)
#         Delta_s_t (np.ndarray): shape (T, ), delta_s_t as in the def for the experience component of eq(24)
#         sigma_Y (float): sigma_Y in eq(1)
#
#     Returns:
#         np.float64: RHS - LHS
#     """
#     invest = (
#             Delta_s_t >= -theta_guess
#     )  # eq(10) and eq(11), invest if theta_s_t >= -theta, constrained if otherwise
#     invest_consumption_share = invest * consumption_share
#     Delta_bar_parti = np.sum(
#         Delta_s_t * invest_consumption_share
#     )  # Experience component, as defined below eq(24)
#     total_invest_c_share = np.sum(
#         invest_consumption_share
#     )  # Constraint component, as defined below eq(24)
#     if total_invest_c_share == 0:
#         diff = 10000
#     else:
#         diff = (
#                        sigma_Y - Delta_bar_parti
#                ) / total_invest_c_share - theta_guess  # RHS - LHS, equals to 0 if find the right theta
#     return diff


def find_the_rich(
        indiv_w: np.ndarray,
        cohort_size: np.ndarray,
        top: float = 0.05) -> np.float64:
    '''
    :param indiv_w (np.ndarray): individual wealth of the agents, shape (Nc,)
    :param cohort_size (np.ndarray): shape (Nc,)
    :param top (float): can short criteria
    :return: a cutoff individual wealth level above which agents are then able to short
    '''
    wealth_rank = indiv_w.argsort()
    indiv_w_sorted = indiv_w[wealth_rank[::-1]]
    cohort_size_sorted = cohort_size[wealth_rank[::-1]]
    popu_cumsum = np.cumsum(cohort_size_sorted)
    popu_cum = popu_cumsum / popu_cumsum[-1]
    cutoff = np.searchsorted(popu_cum, top)
    wealth_cutoff = indiv_w_sorted[cutoff]
    return wealth_cutoff



def find_the_rich_mix(
        indiv_w: np.ndarray,
        cohort_type_size: np.ndarray,
        top: np.ndarray) -> np.float64:
    '''
    :param indiv_w (np.ndarray): individual wealth of the agents, shape (Nc,)
    :param cohort_size (np.ndarray): shape (Nc,)
    :param top (float): can short criteria
    :return: a cutoff individual wealth level above which agents are then able to short
    '''
    indiv_w_flat = indiv_w.flatten()
    wealth_rank = indiv_w_flat.argsort()
    indiv_w_sorted = indiv_w_flat[wealth_rank[::-1]]
    cohort_type_size_flat = cohort_type_size.flatten()
    cohort_size_sorted = cohort_type_size_flat[wealth_rank[::-1]]
    popu_cumsum = np.cumsum(cohort_size_sorted)
    popu_cum = popu_cumsum / popu_cumsum[-1]
    cutoff = np.searchsorted(popu_cum, top)
    wealth_cutoff = indiv_w_sorted[cutoff]
    return wealth_cutoff



# def solve_theta_partial_constraint(
#         theta_guess: float,
#         unconstrained: np.ndarray,
#         Delta_s_t: np.ndarray,
#         consumption_share: np.ndarray,
#         sigma_Y: float,
# ) -> np.float64:
#     '''
#     solve for theta in the conditionally constrained case, with the goal of market clearing in the stock market
#     :param theta_guess (float): any guess of theta
#     :param unconstrained (np.ndarray): the cohorts that can short
#     :param Delta_s_t (np.ndarray): estimation error, shape (Nc,)
#     :param consumption_share (np.ndarray): shape (Nc,)
#     :param sigma_Y (float): volatility of aggregate output
#     :return: the distance to converge
#     '''
#     constrained = 1 - unconstrained
#     pi_constrained = np.maximum(Delta_s_t + theta_guess, 0)  # investment if a cohort can't short
#     part_constrained = np.sum(pi_constrained * consumption_share * constrained)  # for those cohorts that can't short
#     part_unconstrained = np.sum((Delta_s_t + theta_guess) * consumption_share * unconstrained)  # for those cohorts that can short
#
#     diff = (part_constrained + part_unconstrained) / sigma_Y - 1
#
#     return diff
#
#
#
# def bisection_partial_constraint(
#         optimfun: Callable[[float, np.ndarray, np.ndarray, np.ndarray, float], np.float64],
#         xlow: np.float64,
#         xhigh: np.float64,
#         arg1: np.ndarray,
#         arg2: np.ndarray,
#         arg3: np.ndarray,
#         arg4: float,
#         eps: float = 1e-6,
# ) -> np.float64:
#     """Bisection method to solve x (theta)
#
#     Args:
#         optimfun (Callable[[float, np.ndarray, np.ndarray], float]): the function we want to find the root for
#         xlow (float): lower bound for x
#         xhigh (float): upper bound for x
#         arg1 (np.ndarray): second input for optimfun
#         arg2 (np.ndarray): third input for optimfun
#         arg3 (np.ndarray): fourth input for optimfun
#         arg4 (np.ndarray): fifth input for optimfun
#         eps (float, optional): converging criteria. Defaults to 1e-6.
#
#     Returns:
#         xmid: the estimated value that makes the optimfun close to 0
#
#     """
#     flow = optimfun(xlow, arg1, arg2, arg3, arg4)
#     fhigh = optimfun(xhigh, arg1, arg2, arg3, arg4)
#     diff = 1
#     iter = 0
#     xmid = 10000
#
#     while diff > eps:
#         xmid = (xlow + xhigh) / 2
#         fmid = optimfun(xmid, arg1, arg2, arg3, arg4)
#         if flow * fmid < 0:  # root between flow and fmid
#             xhigh = xmid
#             fhigh = fmid
#         elif fmid * fhigh < 0:  # root between fmid and fhigh
#             xlow = xmid
#             flow = fmid
#         diff = abs(fhigh - flow)
#         iter += 1
#         if iter > 50:
#             print("Warning! It takes more than 50 iteration to converge.")
#             break
#     return xmid



def bisection(
        optimfun: Callable[[float, np.ndarray, np.ndarray, np.ndarray, float, float, float], np.float64],
        xlow: np.float64,
        xhigh: np.float64,
        arg1: np.ndarray,
        arg2: np.ndarray,
        arg3: np.ndarray,
        arg4: float,
        arg5: float,
        arg6: float,
        eps: float = 1e-9,
) -> np.float64:
    """Bisection method to solve x (theta)

    Args:
        optimfun (Callable[[float, np.ndarray, np.ndarray], float]): the function we want to find the root for
        xlow (float): lower bound for x
        xhigh (float): upper bound for x
        arg1 (np.ndarray): second input for optimfun
        arg2 (np.ndarray): third input for optimfun
        eps (float, optional): converging criteria. Precision of estimation. Defaults to 1e-6.

    Returns:
        xmid: the estimated value that makes the optimfun close to 0

    """
    flow = optimfun(xlow, arg1, arg2, arg3, arg4, arg5, arg6)
    fhigh = optimfun(xhigh, arg1, arg2, arg3, arg4, arg5, arg6)
    diff = 1
    iter = 0
    xmid = 100000

    while diff > eps:
        xmid = (xlow + xhigh) / 2
        fmid = optimfun(xmid, arg1, arg2, arg3, arg4, arg5, arg6)
        if flow * fmid < 0:  # root between flow and fmid
            xhigh = xmid
            fhigh = fmid
        elif fmid * fhigh < 0:  # root between fmid and fhigh
            xlow = xmid
            flow = fmid
        diff = abs(fhigh - flow)
        iter += 1
        # if iter > 30:
            # print("Warning! It takes more than 50 iteration to converge.")
            # print(xhigh - xlow)
        if iter > 50:
            break
    return xmid




def solve_theta(
        theta_guess: np.float64,
        consumption_share: np.ndarray,
        invest_tracker: np.ndarray,
        Delta_s_t: np.ndarray,
        sigma_Y: float,
        entry_bound: float,
        exit_bound: float,
) -> np.float64:
    """RHS - LHS of the eq(22), used to iteratively solve theta

    Args:
        theta_guess (np.float64): any potential value of theta
        consumption_share (np.ndarray): shape (T, ), fst as in the def for the experience component of eq(24)
        Delta_s_t (np.ndarray): shape (T, ), delta_s_t as in the def for the experience component of eq(24)
        sigma_Y (float): sigma_Y in eq(1)

    Returns:
        np.float64: RHS - LHS
    """
    theta_st = Delta_s_t + theta_guess
    invest = (
                     theta_st >= exit_bound
             ) * invest_tracker + (
                     theta_st >= entry_bound
             ) * (1 - invest_tracker)
    invest_consumption_share = invest * consumption_share
    Delta_bar_parti = np.sum(
        Delta_s_t * invest_consumption_share
    )  # Experience component, as defined below eq(24)
    total_invest_c_share = np.sum(
        invest_consumption_share
    )  # Constraint component, as defined below eq(24)
    if total_invest_c_share == 0:
        diff = 10000
    else:
        diff = (
                       sigma_Y - Delta_bar_parti
               ) / total_invest_c_share - theta_guess  # RHS - LHS, equals to 0 if find the right theta
    return diff



def solve_theta_partial_constraint(
        theta_guess: float,
        invest_tracker: np.ndarray,
        unconstrained: np.ndarray,
        Delta_s_t: np.ndarray,
        consumption_share: np.ndarray,
        sigma_Y: float,
        entry_bound: float,
        exit_bound: float,
) -> np.float64:
    '''
    solve for theta in the conditionally constrained case, with the goal of market clearing in the stock market
    :param theta_guess (float): any guess of theta
    :param unconstrained (np.ndarray): the cohorts that can short
    :param Delta_s_t (np.ndarray): estimation error, shape (Nc,)
    :param consumption_share (np.ndarray): shape (Nc,)
    :param sigma_Y (float): volatility of aggregate output
    :return: the distance to converge
    '''
    constrained = 1 - unconstrained

    theta_st = Delta_s_t + theta_guess
    invest = (
                     theta_st >= exit_bound
             ) * invest_tracker + (
                     theta_st >= entry_bound
             ) * (1 - invest_tracker)

    f_sum = np.sum(invest * consumption_share * constrained) + np.sum(consumption_share * unconstrained)
    Delta_bar_f = (np.sum(
        Delta_s_t * invest * consumption_share * constrained
    ) + np.sum(
        Delta_s_t * consumption_share * unconstrained
    ))

    if f_sum != 0:
        diff = (sigma_Y - Delta_bar_f) / f_sum - theta_guess
    else:
        diff = 1000

    return diff



def bisection_partial_constraint(
        optimfun: Callable[[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float], np.float64],
        xlow: float,
        xhigh: float,
        arg1: np.ndarray,
        arg2: np.ndarray,
        arg3: np.ndarray,
        arg4: np.ndarray,
        arg5: float,
        arg6: float,
        arg7: float,
        eps: float = 1e-6,
) -> np.float64:
    """Bisection method to solve x (theta)

    Args:
        optimfun (Callable[[float, np.ndarray, np.ndarray], float]): the function we want to find the root for
        xlow (float): lower bound for x
        xhigh (float): upper bound for x
        arg1 (np.ndarray): second input for optimfun
        arg2 (np.ndarray): third input for optimfun
        arg3 (np.ndarray): fourth input for optimfun
        arg4 (np.ndarray): fifth input for optimfun
        eps (float, optional): converging criteria. Defaults to 1e-6.

    Returns:
        xmid: the estimated value that makes the optimfun close to 0

    """
    flow = optimfun(xlow, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    fhigh = optimfun(xhigh, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
    diff = 1
    iter = 0
    xmid = 10000

    while diff > eps:
        xmid = (xlow + xhigh) / 2
        fmid = optimfun(xmid, arg1, arg2, arg3, arg4, arg5, arg6, arg7)
        if flow * fmid < 0:  # root between flow and fmid
            xhigh = xmid
            fhigh = fmid
        elif fmid * fhigh < 0:  # root between fmid and fhigh
            xlow = xmid
            flow = fmid
        diff = abs(fhigh - flow)
        iter += 1
        if iter > 50:
            # print("Warning! It takes more than 50 iteration to converge.")
            # print(diff)
            break
    return xmid