import numpy as np
from numba import jit
from typing import Tuple, Callable


@jit(nopython=True)
def post_var(sigma_Y_sq: float, V_hat: float, tau: np.ndarray, a_phi: float, type: str) -> np.ndarray:
    """Calculate the posterior variance, correspond to eq(2)

    Args:
        sigma_Y (float): sigma_Y in eq(1), sd of Yt growth
        V_hat (float): V_hat in eq(2)
        tau (np.ndarray): (t - s) in eq(2), shape (T, )

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
    return V


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
    """Calculate change in beliefs

    Args:

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
    return dDelta_s_t


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


def Delta_st_compare(
        Delta_init: float,
        Vhat_init: float,
        tau: float,
        dt: float,
        sigma_Y_sq: float,
        phi: float,
        n_paths: int,
) -> Tuple[
    float,
    float,
    float,
]:
    length = int(tau / dt)
    shocks_z_Y = np.random.randn(length, n_paths) * np.sqrt(dt)  # shape: (n_paths)
    z_Y = np.sum(shocks_z_Y, axis=0)
    shocks_z_SI = np.random.randn(length, n_paths) * np.sqrt(dt)
    z_SI = np.sum(shocks_z_SI, axis=0)
    phi_factor = phi / np.sqrt(1 - phi ** 2)
    # for N:
    factor_N = 1 / (sigma_Y_sq + Vhat_init * tau)
    Delta_N = sigma_Y_sq * factor_N * Delta_init + Vhat_init * factor_N * z_Y  # shape: (n_paths)
    # for P:
    factor_P = 1 / (sigma_Y_sq * (1 - phi ** 2) + Vhat_init * tau)
    Delta_P = sigma_Y_sq * (1 - phi ** 2) * factor_P * Delta_init + Vhat_init * (1 - phi ** 2) * factor_P * (
                z_Y - phi_factor * z_SI)  # shape: (n_paths)
    ratio_P_better = np.sum(np.abs(Delta_P) - np.abs(Delta_N) <= 0) / n_paths
    positive_corr = z_SI * z_Y >= 0
    condi_positive = np.where(positive_corr > 0)
    condi_negative = np.where(positive_corr == 0)
    ratio_P_better_positive = np.sum(np.abs(Delta_P[condi_positive]) - np.abs(Delta_N[condi_positive]) <= 0) / np.sum(
        positive_corr)
    ratio_P_better_negative = np.sum(np.abs(Delta_P[condi_negative]) - np.abs(Delta_N[condi_negative]) <= 0) / \
                              (n_paths - np.sum(
                                  positive_corr))
    return ratio_P_better, ratio_P_better_positive, ratio_P_better_negative



# def Delta_benchmark(post_var: Callable[[float, float, float, float, str], np.float64],
#                     dDelta_st_calculator: Callable[[float, float, float, float, np.ndarray, np.ndarray, float, float, str], np.ndarray],
#                     sigma_Y, Nt, Vhat, phi, s_vector, dZ, dZ_SI, Npre, T_hat, dt):
#     sigma_Y_sq = sigma_Y ** 2
#     n_cohorts = len(s_vector)
#     Delta_N = np.empty(Nt, n_cohorts)
#     Delta_P = np.empty(Nt, n_cohorts)
#     a_phi_1 = 1/(1 - phi ** 2)
#
#     for n, s in enumerate(s_vector):
#         init_bias = np.sum(dZ[s + 1 - Npre: s + 1]) / T_hat
#         for t in range(Nt, n_cohorts):
#             tau = t - s
#             dZ_t = dZ[t]
#             dZ_SI_t = dZ_SI[t]
#             if tau < 0:
#                 Delta_N[t, n] = np.nan
#                 Delta_P[t, n] = np.nan
#             elif tau == 0:
#                 Delta_N[t, n] = init_bias
#                 Delta_P[t, n] = init_bias
#             else:
#                 Delta_N_1 = Delta_N[t - 1, n]
#                 Delta_P_1 = Delta_P[t - 1, n]
#                 V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')
#                 dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_t,
#                                                     dZ_SI_t,
#                                                     'N')  # from eq(5)
#                 V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
#                 dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_t,
#                                                     dZ_SI_t,
#                                                     'P')
#
#                 Delta_N[t, n] = Delta_N_1 + dDelta_s_t_N
#                 Delta_P[t, n] = Delta_P_1 + dDelta_s_t_P
#
#     return Delta_N, Delta_P


def fadingmemo(v, tau, sigma_Y, V_hat, int_zt, delta_ss):
    v_st = np.log(1 - v) / (
            (1 - v) ** tau - 1
    )
    coef = v_st / (v_st * sigma_Y ** 2 + V_hat)
    delta_st = coef * (sigma_Y ** 2 * delta_ss + V_hat * int_zt)
    return delta_st


# def weighted_mean(var, wts, ax):
#     """Calculates the weighted mean"""
#     return np.average(var, weights=wts, axis=ax)

def weighted_variance(var, wts, ax):
    """Calculates the weighted variance"""
    return np.average((var - np.average(var, weights=wts, axis=ax)) ** 2, weights=wts, axis=ax)


def weighted_skew(var, wts, ax):
    """Calculates the weighted skewness"""
    return (np.average((var - np.average(var, weights=wts, axis=ax)) ** 3, weights=wts, axis=ax) /
            weighted_variance(var, wts, ax) ** (1.5))

# def weighted_kurtosis(var, wts):
#     """Calculates the weighted skewness"""
#     return (np.average((var - weighted_mean(var, wts))**4, weights=wts) /
#             weighted_variance(var, wts)**(2))
