import numpy as np
from numba import jit
from typing import Tuple


@jit(nopython=True)
def post_var(sigma_Y: float, V_hat: float, tau: np.ndarray, phi, type) -> np.ndarray:
    """Calculate the posterior variance, correspond to eq(2)

    Args:
        sigma_Y (float): sigma_Y in eq(1), sd of Yt growth
        V_hat (float): V_hat in eq(2)
        tau (np.ndarray): (t - s) in eq(2), shape (T, )

    Returns:
        np.ndarray: shape (T, )
    """
    sigma_Y_sq = sigma_Y ** 2
    if type == 'N':
        V = sigma_Y_sq * V_hat / (sigma_Y_sq + V_hat * tau)
    elif type == 'P':
        a_phi = 1 - phi ** 2
        V = sigma_Y_sq * a_phi * V_hat / (sigma_Y_sq * a_phi + V_hat * tau)
    else:
        print('type not defined')
        V = V_hat
    return V


@jit(nopython=True)
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


@jit(nopython=True)
def good_times(
        dZt_build: np.ndarray,
        dZt: np.ndarray,
        dt: float,
        Nt: int,
        Nc: int,
        window: int,
        z: float,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    returns the indicator for good times when agents previously dropped out from the stock market might return
    :param dZt_build:
    :param dZt:
    :param dt:
    :param Nt:
    :param window:
    :param z:
    :return:
    """
    cummu_dZt_build = np.zeros(Nc)
    cummu_dZt = np.zeros(Nt)
    for j in range(Nc):
        if j < window:
            cummu_dZt_build[j] = 0
        else:
            cummu_dZt_build[j] = np.sum(dZt_build[j + 1 - window: j + 1])
    for i in range(Nt):
        if i < window:
            cummu_dZt[i] = np.sum(dZt_build[i + 1 - window:]) + np.sum(dZt[: i + 1])
        else:
            cummu_dZt[i] = np.sum(dZt[i + 1 - window: i + 1])
    sigma_cummu = (dt * window) ** 0.5
    good_time_build = cummu_dZt_build >= z * sigma_cummu
    good_time_simulate = cummu_dZt >= z * sigma_cummu
    return good_time_build, good_time_simulate



def fadingmemo(v, tau, sigma_Y, V_hat, int_zt, delta_ss):
    v_st = np.log(1-v) / (
        (1-v) ** tau - 1
    )
    coef = v_st / (v_st * sigma_Y ** 2 + V_hat)
    delta_st = coef * (sigma_Y ** 2 * delta_ss + V_hat * int_zt)
    return delta_st