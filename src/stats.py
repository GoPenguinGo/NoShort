import numpy as np
from numba import jit
from typing import Tuple


@jit(nopython=True)
def post_var(sigma_Y: float, V_hat: float, tau: np.ndarray) -> np.ndarray:
    """Calculate the posterior variance, correspond to eq(2)

    Args:
        sigma_Y (float): sigma_Y in eq(1), sd of Yt growth
        V_hat (float): V_hat in eq(2)
        tau (np.ndarray): (t - s) in eq(2), shape (T, )

    Returns:
        np.ndarray: shape (T, )
    """
    sigma_Y_sq = sigma_Y**2
    V = sigma_Y_sq * V_hat / (sigma_Y_sq + V_hat * tau)
    return V


@jit(nopython=True)
def shocks(
        dZt: np.ndarray,
        mu_Y: float,
        sigma_Y: float,
        dt: float
) -> Tuple[
    np.ndarray,
    np.ndarray
]:
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
    Zt = np.cumsum(dZt)  # cumulated shocks, Nc * 1
    # Zt = np.insert(Zt, 0, 0)
    yg = (
                 mu_Y - 0.5 * sigma_Y ** 2
         ) * dt + sigma_Y * dZt  # output in log, (Nc - 1) *1, eq(1)
    Yt = np.exp(np.cumsum(yg))
    # Yt = np.insert(Yt, 0, 1)
    return (
        Zt,
        Yt
    )


@jit(nopython=True)
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

