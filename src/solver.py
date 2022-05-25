import numpy as np
from typing import Callable
from numba import jit



# TODO: @chingyulin: use *args for optimfun
# todo: use numba to improve the speed of these functions

@jit(nopython=True)
def bisection(
        optimfun: Callable[[float, np.ndarray, np.ndarray, float], np.float64],
        xlow: np.float64,
        xhigh: np.float64,
        arg1: np.ndarray,
        arg2: np.ndarray,
        arg3: float,
        eps: float = 1e-6,
) -> np.float64:
    """Bisection method to solve x (theta)

    Args:
        optimfun (Callable[[float, np.ndarray, np.ndarray], float]): the function we want to find the root for
        xlow (float): lower bound for x
        xhigh (float): upper bound for x
        arg1 (np.ndarray): second input for optimfun
        arg2 (np.ndarray): third input for optimfun
        eps (float, optional): converging criteria. Defaults to 1e-6.

    Returns:
        xmid: the estimated value that makes the optimfun close to 0

    """
    flow = optimfun(xlow, arg1, arg2, arg3)
    fhigh = optimfun(xhigh, arg1, arg2, arg3)
    diff = 1
    iter = 0

    while diff > eps:
        xmid = (xlow + xhigh) / 2
        fmid = optimfun(xmid, arg1, arg2, arg3)
        if flow * fmid < 0:  # root between flow and fmid
            xhigh = xmid
            fhigh = fmid
        elif fmid * fhigh < 0:  # root between fmid and fhigh
            xlow = xmid
            flow = fmid
        diff = abs(fhigh - flow)
        iter += 1
        if iter > 50:
            print("Warning! It takes more than 50 iteration to converge.")
            break
    return xmid


@jit(nopython=True)
def solve_theta(
        thetaguess: np.float64,
        consumptionshare: np.ndarray,
        Delta_s_t: np.ndarray,
        sigma_Y: float,
) -> np.float64:
    """RHS - LHS of the eq(24), used to iteratively solve theta

    Args:
        thetaguess (np.float64): any potential value of theta
        consumptionshare (np.ndarray): shape (T, ), fst as in the def for the experience component of eq(24)
        Delta_s_t (np.ndarray): shape (T, ), delta_s_t as in the def for the experience component of eq(24)
        sigma_Y (float): sigma_Y in eq(1)

    Returns:
        np.float64: RHS - LHS
    """
    invest = (
            Delta_s_t >= -thetaguess
    )  # eq(10) and eq(11), invest if theta_s_t >= -theta, constrained if otherwise
    invest_consumptionshare = invest * consumptionshare
    DeltabarCondi = np.sum(
        Delta_s_t * invest_consumptionshare
    )  # Experience component, as defined below eq(24)
    InvestCons = np.sum(
        invest_consumptionshare
    )  # Constraint component, as defined below eq(24)
    if InvestCons == 0:
        diff = 10000
    else:
        diff = (
                       sigma_Y - DeltabarCondi
               ) / InvestCons - thetaguess  # RHS - LHS, equals to 0 if find the right theta
    return diff

@jit(nopython = True)
def find_the_rich(
        indiv_w: np.ndarray,
        cohort_size: np.ndarray,
        top: float = 0.01) -> np.float64:
    wealth_rank = indiv_w.argsort()
    indiv_w_sorted = indiv_w[wealth_rank[::-1]]
    cohort_size_sorted = cohort_size[wealth_rank[::-1]]
    popu_cum = np.cumsum(cohort_size_sorted)
    cutoff = np.searchsorted(popu_cum, top)
    wealth_cutoff = indiv_w_sorted[cutoff]
    return wealth_cutoff

@jit(nopython = True)
def solve_theta_partial_constraint(
        theta_guess: float,
        unconstrained: np.ndarray,
        Delta_s_t: np.ndarray,
        consumption_share: np.ndarray,
        sigma_Y: float,
) -> np.float64:
    '''
    solve for theta in the conditionally constrained case, with the goal of market clearing in the stock market
    :param theta_guess:
    :param unconstrained:
    :param Delta_s_t:
    :param consumption_share:
    :param sigma_Y:
    :return:
    '''
    pi_constrained = np.maximum(Delta_s_t + theta_guess, 0)
    part_constrained = np.sum(pi_constrained * consumption_share * (1-unconstrained))
    part_unconstrained = np.sum((Delta_s_t + theta_guess) * consumption_share * unconstrained)
    return part_constrained + part_unconstrained - sigma_Y

@jit(nopython=True)
def bisection_partial_constraint(
        optimfun: Callable[[float, np.ndarray, np.ndarray, np.ndarray, float], np.float64],
        xlow: np.float64,
        xhigh: np.float64,
        arg1: np.ndarray,
        arg2: np.ndarray,
        arg3: np.ndarray,
        arg4: float,
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
    flow = optimfun(xlow, arg1, arg2, arg3, arg4)
    fhigh = optimfun(xhigh, arg1, arg2, arg3, arg4)
    diff = 1
    iter = 0

    while diff > eps:
        xmid = (xlow + xhigh) / 2
        fmid = optimfun(xmid, arg1, arg2, arg3, arg4)
        if flow * fmid < 0:  # root between flow and fmid
            xhigh = xmid
            fhigh = fmid
        elif fmid * fhigh < 0:  # root between fmid and fhigh
            xlow = xmid
            flow = fmid
        diff = abs(fhigh - flow)
        iter += 1
        if iter > 50:
            print("Warning! It takes more than 50 iteration to converge.")
            break
    return xmid
