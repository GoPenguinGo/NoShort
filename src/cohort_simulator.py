import numpy as np
from typing import Tuple
from src.stats import post_var
from src.solver import bisection, solve_theta

# TODO: @GoPenguinGo: type and comment this function
def simulate_cohorts(
    biasvec: np.ndarray,
    dZt: np.ndarray,
    Nt: int,
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
    bet: float,
    That: float,
    Npre: float,
    DeltabarCondi: np.float64,
    fCondi: np.float64,
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
        Nt (int): number of observations
        tau (np.ndarray): t-s, shape(Nt)
        IntVec (np.ndarray): ~similar to consumption share, shape(Nt)
        Delta_s_t (np.ndarray): bias for each cohort, shape(Nt)
        MaxThetaDelta_s_t (np.ndarray): max(Delta_), shape(Nt)
        dt (float): per unit of time
        rho (float): discount factor
        nu (float): rate of birth and death
        Vhat (float): initial variance
        mu_Y (float): as in eq(1), drift of aggregate output growth
        sigma_Y (float): as in eq(1), diffusion of aggregate output growth
        sigma_S (float): as in eq(26), diffusion of stock price 
        bet (float): as in eq(18), consumption share of the newborn cohort
        That (float): pre-trading years
        Npre (float): pre-trading number of obs
        DeltabarCondi (np.float64): the experience component in eq(24)
        fCondi (np.float64): the constraint component in eq(24)
    
    Returns:
        Xt2 (np.ndarray): xi_t * Yt, shape(Nt, )
        MaxThetaDelta_s_t (np.ndarray): max(delta_s_t, -theta_t), shape(Nt, )
        part1 (np.ndarray): Consumption of each cohort, eq(16), where eta_s_t / eta_s_s follows eq(11), shape(Nt, )
        mu_S (np.ndarray): expected return under true measure at t, shape(Nt, )
        mu_S_t (np.ndarray): expected stock return for agent born at t, shape(Nt, )
        muhat_S_t (np.ndarray): average belief in the economy, shape(Nt, )
        r_t (np.ndarray): interest rate, shape(Nt, )
        theta_t (np.ndarray): market price of risk, shape(Nt, )
        Port (np.ndarray): portfolio choice, shape(Nt, )
        muC_s_t (np.ndarray): drift of individual consumption, shape(Nt, )
        sigmaC_s_t (np.ndarray): diffusion of individual consumption, shape(Nt, )
        BIGF (np.ndarray): consumption share over time, shape(Nt, Nt, )
        BIGDELTA (np.ndarray): bias over time, shape(Nt, Nt, )
        # Et
        # Vt
        dR (np.ndarray): change of stock returns over time, shape(Nt, )
    """""
    # Initializing variables
    Xt2 = np.ones(Nt)
    Deltabar2Conditional = np.ones(Nt)
    Et = np.ones(Nt)
    Vt = np.ones(Nt)

    part1 = np.zeros(Nt)
    dR = np.zeros(Nt)
    reduction = np.exp(-nu * dt)
    BIGDELTA = np.zeros((Nt, Nt))
    BIGMAX = np.zeros((Nt, Nt))  # stores max(delta, -theta)
    BIGF = np.zeros((Nt, Nt))
    BIGFCONDI = np.zeros((Nt))
    BIGDELTABARCONDI = np.zeros((Nt))  # different from max(delta, -theta)

    RevNt = np.flip(range(Nt))
    fhat = reduction**RevNt * nu * dt
    fhat = fhat / sum(fhat)

    # Expected returns
    mu_S = np.zeros(Nt)  # expected return under true measure
    mu_S_t = np.zeros(Nt)  # expected return under agent-measure
    muhat_S_t = np.zeros(Nt)  # average belief in the economy

    muC_s_t = np.zeros(Nt)  # drift log consumption
    sigmaC_s_t = np.zeros(Nt)  # diffusion log consumption

    # Aggregate quantities
    r_t = np.zeros(Nt)  # interest rate
    theta_t = np.zeros(Nt)  # market price of risk
    fst = np.zeros(Nt)  # consumption share
    for i in range(Nt):
        BIGDELTA[i, :] = Delta_s_t
        BIGMAX[i, :] = MaxThetaDelta_s_t
        BIGFCONDI[i] = fCondi
        BIGDELTABARCONDI[i] = DeltabarCondi

        if i < Nt - 1:  # information & belief
            if i == 0:
                part1[i] = sum(biasvec) / That
            else:
                part1[i] = part1[i - 1] + (
                    post_var(sigma_Y, Vhat, (i * dt)) / sigma_Y**2
                ) * (-part1[i - 1] * dt + dZt[i - 1])

        Part = IntVec * np.exp(
            -(0.5 * MaxThetaDelta_s_t**2) * dt + MaxThetaDelta_s_t * dZt[i]
        )

        Xt2[i] = sum(Part)
        Deltabar2Conditional[i] = sum(Part * MaxThetaDelta_s_t) / Xt2[i]
        f = Part / Xt2[i]  # consumption share
        BIGF[i, :] = f

        # find theta
        A = -max(Delta_s_t)
        theta_t[i] = bisection(solve_theta, A, 10, f, Delta_s_t)
        MaxThetaDelta_s_t = np.maximum(-theta_t[i], Delta_s_t)
        invest = Delta_s_t >= -theta_t[i]
        DeltabarCondi = sum(Delta_s_t * invest * f)
        fCondi = sum(invest * f)

        r_t[i] = (
            rho
            + mu_Y
            + nu * (1 - bet)
            - (sigma_Y**2 - sigma_Y * DeltabarCondi) / fCondi
        )

        mu_S[i] = sigma_S * (sigma_Y - DeltabarCondi) / fCondi + r_t[i]
        mu_S_t[i] = (
            mu_S[i] + sigma_S * part1[i]
        )  # expected stock return for agent born at t
        muhat_S_t[i] = mu_S[i] + sigma_S * sum(fhat * Delta_s_t)

        dR[i] = mu_S[i] * dt + sigma_S * dZt[i]  # mu_t^Sdt + sigma_t^Sdz_t

        # Et[i] = sum(f * (Vbar / (1 + (Vbar / sigma_Y ** 2) * dt * RevNt)) * (1 / sigma_Y))
        # Vt[i] = (sum(f * Delta_s_t ** 2) - Deltabar2[i] ** 2) * sigma_Y

        # Updating:
        dDelta_s_t = (post_var(sigma_Y, Vhat, tau) / sigma_Y**2) * (
            -Delta_s_t * dt + np.ones(len(Delta_s_t)) * dZt[i]
        )
        if i < Npre:
            DELbias = (sum(biasvec[i:]) + sum(dZt[:i])) / That
        else:
            DELbias = sum(dZt[i - Npre : i]) / That

        Delta_s_t = Delta_s_t[1:] + dDelta_s_t[1:]
        Delta_s_t = np.append(Delta_s_t, DELbias)
        IntVec = reduction * Part[1:]
        IntVec = np.append(IntVec, bet * (1 - reduction) * Xt2[i])

    Port = np.maximum(Delta_s_t + theta_t[i], 0) / sigma_S

    return (
        Xt2,
        MaxThetaDelta_s_t,
        part1,
        mu_S,
        mu_S_t,
        muhat_S_t,
        r_t,
        theta_t,
        Port,
        muC_s_t,
        sigmaC_s_t,
        BIGF,
        BIGDELTA,
        #Et,
        #Vt,
        dR,
    )


