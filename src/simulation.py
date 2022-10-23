import time
import numpy as np
from typing import Tuple
from src.cohort_builder import build_cohorts, build_cohorts_partial_constraint
from src.cohort_simulator import simulate_cohorts, simulate_cohorts_partial_constraint
from src.stats import shocks, good_times


def simulate(
    mode_trade: str,
    mode_learn: str,
    Nc: int,
    Nt: int,
    dt: float,
    rho: float,
    nu: float,
    Vhat: float,
    mu_Y: float,
    sigma_Y: float,
    sigma_S: float,
    tax: float,
    beta: float,
    Npre: int,
    Ninit: int,
    T_hat: int,
    dZ_build: np.ndarray,
    dZ: np.ndarray,
    tau: np.ndarray,
    cohort_size: np.ndarray,
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
]:
    """A program that combines cohort_builder and cohort_simulator, and finishes one whole simulation path
    :param mode: scenario of the function, see param for scenario names
    :param Nc: number of cohorts
    :param dt: time increment
    :param rho: time preference
    :param nu: birth and death rate
    :param Vhat: initial variance
    :param mu_Y: mean of aggregate output growth rate
    :param sigma_Y: volatility of aggregate output growth rate
    :param tax: initial consumption ratio
    :param Npre: pre-trading periods
    :param T_hat: pre-trading years
    :return:
    """

    biasvec = dZ_build[-Npre:]  # dZt used in the build_cohorts function

    Y = shocks(
        dZ,
        mu_Y,
        sigma_Y,
        dt,
    )

    (good_time_build, good_time_simulate,) = good_times(
        dZ_build,
        dZ,
        dt,
        Nt,
        Nc,
        window=12,
        z=1.28,
    )

    (
        Delta_s_t,
        eta_st_eta_ss,
        eta_bar,
        d_eta_st,
        invest_tracker,
        tau_info_build,
    ) = build_cohorts(
        dZ_build,
        Nc,
        dt,
        tau,
        rho,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        tax,
        Npre,
        Ninit,
        T_hat,
        good_time_build,
        mode_trade,
        mode_learn,
    )

    (
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
    ) = simulate_cohorts(
        Y,
        biasvec,
        dZ,
        Nt,
        Nc,
        tau,
        dt,
        rho,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        sigma_S,
        tax,
        beta,
        T_hat,
        Npre,
        Ninit,
        mode_trade,
        mode_learn,
        cohort_size,
        Delta_s_t,
        eta_st_eta_ss,
        eta_bar,
        d_eta_st,
        invest_tracker,
        tau_info_build,
        good_time_simulate,
    )

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


def simulate_partial_constraint(
    mode_trade: str,
    mode_learn: str,
    Nc: int,
    Nt: int,
    dt: float,
    rho: float,
    nu: float,
    Vhat: float,
    mu_Y: float,
    sigma_Y: float,
    sigma_S: float,
    tax: float,
    beta: float,
    Npre: int,
    Ninit: int,
    T_hat: int,
    dZ_build: np.ndarray,
    dZ: np.ndarray,
    tau: np.ndarray,
    cohort_size: np.ndarray,
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
    """
    :param mode: scenario of the function, see param for scenario names
    :param Nc: number of cohorts
    :param Nt: number of periods
    :param dt: time increment
    :param rho: time preference
    :param nu: birth and death rate
    :param Vhat: initial variance
    :param mu_Y: mean of aggregate output growth rate
    :param sigma_Y: volatility of aggregate output growth rate
    :param tax: initial consumption share
    :param beta: marginal propensity to consume
    :param Npre: pre-trading periods
    :param T_hat: pre-trading years
    :param dZ_build: shocks for the building cohorts part
    :param dZ: shocks for the simulation part
    :param tau: age of each cohort
    :param cohort_size: size of each cohort
    :return:
    """

    # dZ_build = dt ** 0.5 * np.random.randn(int(Nc - 1))  # dZt for the build function
    biasvec = dZ_build[-Npre:]  # dZt used in the build_cohorts function
    # dZ = dt ** 0.5 * np.random.randn(Nt)  # dZt for the simulate function

    Y = shocks(
        dZ,
        mu_Y,
        sigma_Y,
        dt,
    )

    (good_time_build, good_time_simulate,) = good_times(
        dZ_build,
        dZ,
        dt,
        Nt,
        Nc,
        window=12,
        z=1.28,
    )

    (
        Delta_s_t,
        eta_st_eta_ss,
        eta_bar,
        d_eta_st,
        invest_tracker_build,
        can_short_tracker_build,
        tau_info_build,
    ) = build_cohorts_partial_constraint(
        dZ_build,
        Nc,
        dt,
        tau,
        cohort_size,
        rho,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        tax,
        Npre,
        Ninit,
        T_hat,
        good_time_build,
        mode_trade,
        mode_learn,
    )

    (
        r,
        theta,
        f,
        Delta,
        d_eta,
        pi,
        dR,
        w_cohort,
        popu_parti,
        popu_can_short,
        popu_short,
        popu_long,
        f_parti,
        f_short,
        f_long,
        age_parti,
        age_short,
        age_long,
        n_parti,
        invest_tracker,
        can_short_tracker,
        long,
        short,
        Delta_bar_parti,
        Delta_bar_long,
        Delta_bar_short,
    ) = simulate_cohorts_partial_constraint(
        Y,
        biasvec,
        dZ,
        Nt,
        Nc,
        tau,
        dt,
        rho,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        sigma_S,
        tax,
        beta,
        T_hat,
        Npre,
        Ninit,
        mode_trade,
        mode_learn,
        cohort_size,
        Delta_s_t,
        eta_st_eta_ss,
        eta_bar,
        d_eta_st,
        invest_tracker_build,
        can_short_tracker_build,
        tau_info_build,
        good_time_simulate,
    )

    return (
        r,
        theta,
        f,
        Delta,
        d_eta,
        pi,
        dR,
        w_cohort,
        popu_parti,
        popu_can_short,
        popu_short,
        popu_long,
        f_parti,
        f_short,
        f_long,
        age_parti,
        age_short,
        age_long,
        n_parti,
        invest_tracker,
        can_short_tracker,
        long,
        short,
        Delta_bar_parti,
        Delta_bar_long,
        Delta_bar_short,
    )
