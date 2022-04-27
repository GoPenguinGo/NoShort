import numpy as np
from typing import Tuple
from src.stats import post_var
from src.solver import bisection, solve_theta
from tqdm import tqdm
from numba import jit

@jit(nopython=True)
def get_part(IntVec, MaxThetaDelta_s_t, dt, dZt, i):
    return IntVec * np.exp(
            -(0.5 * MaxThetaDelta_s_t**2) * dt + MaxThetaDelta_s_t * dZt[i]
        )

@jit(nopython=True)
def simulate_cohorts(
    biasvec: np.ndarray,
    dZt: np.ndarray,
    Nt: int,
    Nc: int,
    tau: np.ndarray,
    IntVec: np.ndarray,
    Delta_s_t: np.ndarray,
    MaxThetaDelta_s_t: np.ndarray,
    dt: float,
    rho: float,
    nu: float,
    Vhat: float,
    mu_Y: float,
    sigma_Y: float,
    sigma_S: float,
    beta: float,
    T_hat: float,
    Npre: float,
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
]:
    """"" Simulate the economy forward
    
    Args:
        biasvec (np.ndarray): pre-trading period shocks to form beliefs of the initial cohort, shape(Npre,)
        dZt (np.ndarray): random shocks, shape (Nt)
        Nt (int): number of periods in the simulation
        Nc (int): number of cohorts in the economy
        tau (np.ndarray): t-s, shape(Nt)
        IntVec (np.ndarray): ~similar to consumption share, shape(Nc)
        Delta_s_t (np.ndarray): bias for each cohort, shape(Nc)
        MaxThetaDelta_s_t (np.ndarray): max(Delta_), shape(Nc)
        dt (float): per unit of time
        rho (float): discount factor
        nu (float): rate of birth and death
        Vhat (float): initial variance
        mu_Y (float): as in eq(1), drift of aggregate output growth
        sigma_Y (float): as in eq(1), diffusion of aggregate output growth
        sigma_S (float): as in eq(26), diffusion of stock price 
        beta (float): as in eq(18), consumption share of the newborn cohort
        T_hat (float): pre-trading years
        Npre (float): pre-trading number of obs
    
    Returns:
        Xt2 (np.ndarray): xi_t * Yt, shape(Nt, )
        part1 (np.ndarray): Consumption of each cohort, eq(16), where eta_s_t / eta_s_s follows eq(11), shape(Nc, )
        mu_S (np.ndarray): expected return under true measure at t, shape(Nt, )
        mu_S_s (np.ndarray): expected stock return each cohort, shape(Nt, Nc, )
        mu_hat_S (np.ndarray): survey belief in the economy, shape(Nt, )
        r (np.ndarray): interest rate, shape(Nt, )
        theta (np.ndarray): market price of risk, shape(Nt, )
        # muC_s_t (np.ndarray): drift of individual consumption, shape(Nc, )
        # sigmaC_s_t (np.ndarray): diffusion of individual consumption, shape(Nc, )
        BIGF (np.ndarray): consumption share over time, shape(Nt, Nc, )
        BIGDELTA (np.ndarray): bias over time, shape(Nt, Nc, )
        BIGMAX (np.ndarray): max(-theta, delta_s_t) over time, shape(Nt, Nc, )
        BIGPORT (np.ndarray): portfolio choice over time, shape(Nt, Nc, )
        BIGPOPU (np.ndarray): population that invest in stocks over time, shape(Nt, )
        BIGFCONDI (np.ndarray): aggregate consumption share over time conditional on invest in stocks, shape(Nt, )
        BIGDELTABARCONDI (np.ndarray): aggregate consumption weighted bias over time conditional on invest in stocks, shape(Nt, )
        dR (np.ndarray): change of stock returns over time, shape(Nt, )
        
    """ ""
    # Initializing variables
    Xt2 = np.ones(Nt)
    # Deltabar2Conditional = np.ones(Nt)
    # Et = np.ones(Nt)
    # Vt = np.ones(Nt)

    part1 = np.zeros(Nc)
    dR = np.zeros(Nt)
    reduction = np.exp(-nu * dt)
    BIGDELTA = np.zeros((Nt, Nc))
    BIGMAX = np.zeros((Nt, Nc))  # stores max(delta, -theta)
    BIGF = np.zeros((Nt, Nc))
    BIGPORT = np.zeros((Nt, Nc))
    BIGFCONDI = np.zeros((Nt))
    BIGDELTABARCONDI = np.zeros((Nt))
    BIGPOPU = np.zeros((Nt))

    cohort = nu * np.exp(-nu * tau) * dt
    population = np.sum(cohort)  # ~1
    cohort_size = cohort / population

    # Expected returns
    mu_S = np.zeros(Nt)  # expected return under true measure
    mu_S_s = np.zeros((Nt, Nc))  # expected return under agent-measure
    mu_hat_S = np.zeros(Nt)  # average belief in the economy

    # Consumption
    # muC_s_t = np.zeros(Nt)  # drift log consumption
    # sigmaC_s_t = np.zeros(Nt)  # diffusion log consumption

    # Aggregate quantities
    r = np.zeros(Nt)  # interest rate
    theta = np.zeros(Nt)  # market price of risk

    for i in range(0, Nt):
        Part = get_part(IntVec, MaxThetaDelta_s_t, dt, dZt, i)


        sumPart = np.sum(Part)
        # Deltabar2Conditional[i] = np.sum(Part * MaxThetaDelta_s_t) / sumPart
        f = Part / sumPart  # consumption share

        # find theta

        A = -np.max(Delta_s_t)
        theta_t = bisection(solve_theta, A, 10, f, Delta_s_t, sigma_Y)
        MaxThetaDelta_s_t = np.maximum(-theta_t, Delta_s_t)
        invest = Delta_s_t >= -theta_t
        invest_f = invest * f
        popuCondi = np.sum(cohort_size * invest)
        DeltabarCondi = np.sum(Delta_s_t * invest_f)
        fCondi = np.sum(invest_f)

        r_t = (
            rho
            + mu_Y
            + nu * (1 - beta)
            - (sigma_Y**2 - sigma_Y * DeltabarCondi) / fCondi
        )

        mu_S_t = sigma_S * theta_t + r_t
        mu_S_st = (
            mu_S_t + sigma_S * Delta_s_t
        )  # expected stock return for agent born at t
        muhat_S_t = mu_S_t + sigma_S * np.sum(
            cohort_size * Delta_s_t
        )  # survey average forecast

        dR[i] = mu_S_t * dt + sigma_S * dZt[i]  # mu_t^Sdt + sigma_t^Sdz_t

        # Et[i] = sum(f * (Vbar / (1 + (Vbar / sigma_Y ** 2) * dt * RevNt)) * (1 / sigma_Y))
        # Vt[i] = (sum(f * Delta_s_t ** 2) - Deltabar2[i] ** 2) * sigma_Y

        Port = np.maximum(Delta_s_t + theta_t, 0) / sigma_S

        theta[i] = theta_t
        r[i] = r_t
        Xt2[i] = sumPart
        mu_S[i] = mu_S_t
        mu_S_s[i, :] = mu_S_st
        mu_hat_S[i] = muhat_S_t
        BIGF[i, :] = f
        BIGDELTA[i, :] = Delta_s_t
        BIGMAX[i, :] = MaxThetaDelta_s_t
        BIGFCONDI[i] = fCondi
        BIGDELTABARCONDI[i] = DeltabarCondi
        BIGPORT[i] = Port
        BIGPOPU[i] = popuCondi

        # Updating: (exogenous, for next period)
        dDelta_s_t = (post_var(sigma_Y, Vhat, tau) / sigma_Y**2) * (
            -Delta_s_t * dt + dZt[i]
        )
        if i < Npre:
            DELbias = (np.sum(biasvec[i:]) + np.sum(dZt[:i])) / T_hat
        else:
            DELbias = np.sum(dZt[i - Npre : i]) / T_hat

        Delta_s_t = Delta_s_t[1:] + dDelta_s_t[1:]
        Delta_s_t = np.append(Delta_s_t, DELbias)
        IntVec = reduction * Part[1:]
        IntVec = np.append(IntVec, beta * (1 - reduction) * sumPart)

    return (
        Xt2,
        part1,
        mu_S,
        mu_S_s,
        mu_hat_S,
        r,
        theta,
        BIGF,
        BIGDELTA,
        BIGMAX,
        BIGPORT,
        BIGPOPU,
        BIGFCONDI,
        BIGDELTABARCONDI,
        dR,
    )
