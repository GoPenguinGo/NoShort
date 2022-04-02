import numpy as np
from typing import Callable

# TODO: @chingyulin: use *args for optimfun
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
        optimfun (Callable[[float, np.ndarray, np.ndarray], float]): _description_
        xlow (float): lower bound for x
        xhigh (float): upper bound for x
        arg1 (np.ndarray): second input for optimfun
        arg2 (np.ndarray): third input for optimfun
        eps (float, optional): converging criteria. Defaults to 1e-6.

    Returns:
        np.float64: _description_ #TODO: GoPenguinGo
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
    return xmid


def solve_theta(
    thetaguess: np.float64,
    consumptionshare: np.ndarray,
    Delta_s_t: np.ndarray,
    sigma_Y: float,
) -> np.float64:
    """RHS - LHS of the eq(24), used to iteratively solve theta

    Args:
        thetaguess (np.float64): _description_ #TODO: GoPenguinGo
        consumptionshare (np.ndarray): shape (T, )
        Delta_s_t (np.ndarray): shape (T, )

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
    diff = (
        sigma_Y
        - DeltabarCondi  # TODO: GoPenguinGo: make sigma_Y a argument of the function
    ) / InvestCons - thetaguess  # RHS - LHS, equals to 0 if find the right theta
    return diff
