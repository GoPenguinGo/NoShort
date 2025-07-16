import numpy as np
# from numba import jit
from typing import Tuple, Callable


def post_var(sigma_Y_sq: float, V_hat: float, tau: np.ndarray, a_phi: float, type: str) -> np.ndarray:
    """Calculate the posterior variance, correspond to eq(6)

    Args:
        sigma_Y_sq (float): sigma_Y squared
        V_hat (float): initial variance or variance at the last time when switch occurred
        tau (np.ndarray): (t - t') in eq(2), shape (T, )
        a_phi (float): (1 - phi^2)
        type (str): "P" or "N"

    Returns:
        np.ndarray: shape (T, )
    """
    if type == 'N':
        V = sigma_Y_sq * V_hat / (sigma_Y_sq + V_hat * tau)
    elif type == 'P':
        V = sigma_Y_sq * a_phi * V_hat / (sigma_Y_sq * a_phi + V_hat * tau)
    else:
        print('Error: type not found')
        V = V_hat
    return V.astype(np.float32)


# @jit(nopython=True)
def dDelta_st_calculator(sigma_Y_sq: float,
                         a1: float,
                         a2: float,
                         dt: float,
                         V_st: np.ndarray,
                         Delta_s_t: np.ndarray,
                         dZ_t: float,
                         dZ_SI_t: float,
                         type: str) -> np.ndarray:
    """Calculate change in beliefs, as in eq(9)

    Args:
        sigma_Y_sq (float): sigma_Y squared
        a1 (float): 1/(1-phi^2)
        a2 (float): phi/sqrt(1-phi^2)
        dt (float): dt
        V_st (np.ndarray): posterior variance
        Delta_s_t (np.ndarray): prior estimation error
        dZ_t (float): shocks to the fundamental
        dZ_SI_t (float): shocks to the signal
        type (str): "P" or "N"

    Returns:
        np.ndarray: shape (T, )
    """
    if type == 'P':
        dDelta_s_t = V_st / sigma_Y_sq * (
                - a1 * Delta_s_t * dt + dZ_t - a2 * dZ_SI_t
        )
    elif type == 'N':
        dDelta_s_t = V_st / sigma_Y_sq * (
                -Delta_s_t * dt + dZ_t
        )
    else:
        print('Error: type not found')
        dDelta_s_t = 0
    return dDelta_s_t.astype(np.float32)



def shocks(
        dZt: np.ndarray,
        mu_Y: float,
        sigma_Y: float,
        dt: float
) -> np.ndarray:
    """Calculate Zt and Yt from shocks dZt

Args:
    dZt (np.ndarray): shocks for each period
    mu_Y (float): mu_Y in eq(1)
    sigma_Y (float): sigma_Y in eq(1)
    dt (float): per unit of time

Returns:
    Zt (np.float64): cumulated shocks over the period
    Yt (np.float64): output over the period
"""
    # Zt = np.cumsum(dZt)  # cumulated shocks, Nc * 1
    # Zt = np.insert(Zt, 0, 0)
    yg = (
                 mu_Y - 0.5 * sigma_Y ** 2
         ) * dt + sigma_Y * dZt  # output in log, (Nc - 1) *1, eq(1)
    Yt = np.exp(np.cumsum(yg))
    # Yt = np.insert(Yt, 0, 1)
    return Yt


def tau_calculator(
        dt: float,
        T_cohort: int
) -> np.ndarray:
    """ Calculate tau
Args:
    dt (float): per unit of time
    T_cohort (int): number of periods
returns:
    tau
    """
    tau = np.arange(T_cohort, 0, -dt)
    return tau
