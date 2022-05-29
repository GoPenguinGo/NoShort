import time
import numpy as np
from typing import Tuple
from src.cohort_builder import build_cohorts, build_cohorts_partial_constraint
from src.cohort_simulator import simulate_cohorts, simulate_cohorts_partial_constraint
from src.param import *
from src.stats import shocks, good_times



def simulate(
    mode: str,
    Nc: int,
    dt: float,
    rho: float,
    nu: float,
    Vhat: float,
    mu_Y: float,
    sigma_Y: float,
    beta: float,
    Npre: int,
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
]:
    '''
    :param mode: scenario of the function, see param for scenario names
    :param Nc: number of cohorts
    :param dt: time increment
    :param rho: time preference
    :param nu: birth and death rate
    :param Vhat: initial variance
    :param mu_Y: mean of aggregate output growth rate
    :param sigma_Y: volatility of aggregate output growth rate
    :param beta: initial consumption ratio
    :param Npre: pre-trading periods
    :param T_hat: pre-trading years
    :return:
    '''


    time_s = time.time()
    # dZ_build = dt ** 0.5 * np.random.randn(int(Nc - 1))  # dZt for the build function
    biasvec = dZ_build[-Npre:]  # dZt used in the build_cohorts function
    # dZ = dt ** 0.5 * np.random.randn(Nt)  # dZt for the simulate function

    (
        Z,
        Y,
    ) = shocks(
        dZ,
        mu_Y,
        sigma_Y,
        dt,
    )
    #
    # (
    #     good_time_build,
    #     good_time_simulate,
    # ) = good_times(
    #     dZ_build,
    #     dZ,
    #     dt,
    #     Nt,
    #     Nc,
    #     window=12,
    #     z=1.28,
    # )

    (
        f_st,
        Delta_s_t,
        eta_st_ss,
        eta_bar,
        MaxThetaDelta_s_t,
        invest_tracker,
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
        beta,
        Npre,
        T_hat,
        mode,
    )

    (
        mu_S,
        mu_S_s,
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
        w,
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
        beta,
        omega,
        T_hat,
        Npre,
        mode,
        cohort_size,
        f_st,
        Delta_s_t,
        eta_st_ss,
        eta_bar,
        MaxThetaDelta_s_t,
        invest_tracker,
    )
    if time.time() - time_s >= 10:
        print('takes more than 10s')
    return (
        mu_S,
        mu_S_s,
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
        w,
        w_cohort,
        age,
        n_parti,
    )


def simulate_partial_constraint(
    mode: str,
    Nc: int,
    Nt: int,
    dt: float,
    rho: float,
    nu: float,
    Vhat: float,
    mu_Y: float,
    sigma_Y: float,
    beta: float,
    omega: float,
    Npre: int,
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
        np.ndarray,
        np.ndarray,
        np.ndarray,
]:
    '''
    :param mode: scenario of the function, see param for scenario names
    :param Nc: number of cohorts
    :param Nt: number of periods
    :param dt: time increment
    :param rho: time preference
    :param nu: birth and death rate
    :param Vhat: initial variance
    :param mu_Y: mean of aggregate output growth rate
    :param sigma_Y: volatility of aggregate output growth rate
    :param beta: initial consumption share
    :param omega: marginal propensity to consume
    :param Npre: pre-trading periods
    :param T_hat: pre-trading years
    :param dZ_build: shocks for the building cohorts part
    :param dZ: shocks for the simulation part
    :param tau: age of each cohort
    :param cohort_size: size of each cohort
    :return:
    '''


    time_s = time.time()
    # dZ_build = dt ** 0.5 * np.random.randn(int(Nc - 1))  # dZt for the build function
    biasvec = dZ_build[-Npre:]  # dZt used in the build_cohorts function
    # dZ = dt ** 0.5 * np.random.randn(Nt)  # dZt for the simulate function

    (
        Z,
        Y,
    ) = shocks(
        dZ,
        mu_Y,
        sigma_Y,
        dt,
    )

    (
        good_time_build,
        good_time_simulate,
    ) = good_times(
        dZ_build,
        dZ,
        dt,
        Nt,
        Nc,
        window=12,
        z=1.28,
    )


    (
        f_st,
        Delta_s_t,
        eta_st_ss,
        eta_bar,
        d_eta_st_ss,
        invest_tracker_build,
        can_short_tracker_build,
    ) = build_cohorts_partial_constraint(dZ_build, Nc, dt, tau, cohort_size, rho, nu, Vhat, mu_Y, sigma_Y, beta, Npre, T_hat, good_time_build, mode)


    (
        mu_S,
        mu_S_s,
        r,
        theta,
        f,
        Delta,
        d_eta,
        pi,
        dR,
        w,
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
        beta,
        omega,
        T_hat,
        Npre,
        mode,
        cohort_size,
        f_st,
        Delta_s_t,
        eta_st_ss,
        eta_bar,
        d_eta_st_ss,
        invest_tracker_build,
        can_short_tracker_build,
        good_time_simulate,
    )


    if time.time() - time_s >= 15:
        print('takes more than 15s')

    return (
        mu_S,
        mu_S_s,
        r,
        theta,
        f,
        Delta,
        d_eta,
        pi,
        dR,
        w,
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


