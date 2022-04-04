import numpy as np
from src.solver import bisection, solve_theta
from tqdm import tqdm
from typing import Tuple

from src.stats import post_var


def build_cohorts(
    dZt: np.ndarray,
    Nt: int,
    dt: float,
    rho: float,
    nu: float,
    Vhat: float,
    mu_Y: float,
    sigma_Y: float,
    beta: float,
    T_hat: float,
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
    np.float64,
    np.float64,
]:
    """builds up a sufficiently large set of cohorts in the economy, view each cohort as one agent with a constantly shrinking size

    Args:
        dZt (np.ndarray): random shocks of aggregate output for each period, shape (Nt-1, )
        Nt (int): number of periods
        dt (float): unit of time
        rho (float): rho, discount factor
        nu (float): birth / death rate, each cohort starts at size nu and shrinks at speed of nu
        Vhat (float): initial variance of beliefs
        mu_Y (float): mean of aggregate output growth
        sigma_Y (float): sd of aggregate output growth
        beta (float): initial consumption of the newborn agents
        T_hat (float): pre-trading years

    Returns:
        DeltaConditional (np.ndarray): consumption weighted aggregate max(delta_s_t, -theta_t), as in eq(19), shape(Nt, )
        IntVec (np.ndarray): ~similar to consumption share, shape(Nt, )
        Xt (np.ndarray): xi_t * Yt, shape(Nt, )
        Delta_s_t (np.ndarray): bias, shape(Nt, )
        Yt (np.ndarray): aggregate output, shape(Nt, )
        Zt (np.ndarray): cumulated shocks, shape(Nt, )
        consumptionshare (np.ndarray): shape(Nt, )
        tau (np.ndarray): t-s, shape(Nt, )
        MaxThetaDelta_s_t (np.ndarray): max(delta_s_t, -theta_t), shape(Nt, )
        DeltabarCondi (np.float64): experience component in (24)
        fCondishape (np.float64): constraint component in (24)
        MaxDeltaTheta_s_t (np.ndarray): max(delta_s_t, -theta_t), shape(Nt, )
        DeltabarCondi (np.float64): experience component in (24)
        fCondishape (np.float64): constraint component in (24)
        #TODO: @chingyulin: use NamedTuple for the return
    """

    Npre: int = int(T_hat / dt)  # Number of pre-trading observations
    Zt = np.insert(np.cumsum(dZt), 0, 0)  # cumulated shocks, Nt * 1
    yg = (mu_Y - 0.5 * sigma_Y**2) * dt * np.ones(
        int(Nt - 1)
    ) + sigma_Y * dZt  # output in log, (Nt - 1) *1, eq(1)
    Yt = np.insert(np.exp(np.cumsum(yg)), 0, 1)  # output, Nt *1
    DeltaConditional = np.zeros(Nt)
    Delta_s_t = np.zeros(1)  # belief bias, eq(3)
    MaxThetaDelta_s_t = np.zeros(1)  # disagreement, eq(11)
    Xt = np.ones(Nt) * nu * beta  # similar to consumption share, similar to eq(18)
    IntVec = nu * beta  # consumption share of a newborn cohort
    # TODO: @chingyulin: tau can allocate the memory
    tau = np.zeros(1)  # t-s
    tau[0] = dt
    reduction = np.exp(-nu * dt)  # cohort size shrink at this rate
    theta_t = np.zeros(Nt)  # market price of risk
    for i in tqdm(range(1, Nt)):
        Part = IntVec * np.exp(
            -(rho + 0.5 * MaxThetaDelta_s_t * MaxThetaDelta_s_t) * dt
            + MaxThetaDelta_s_t * dZt[i - 1]
        )  # Consumption of each cohort, eq(16), where eta_s_t / eta_s_s follows eq(11)
        if i == 1:  # only one cohort in the economy
            Xt[i] = Part
            DeltaConditional[i] = Part * MaxThetaDelta_s_t
        else:  # more cohorts
            Xt[i] = np.sum(Part)  # total consumption
            DeltaConditional[i] = (
                np.sum(Part * MaxThetaDelta_s_t) / Xt[i]
            )  # eq(19), consumption weighted max(Delta_s_t, -theta)

        IntVec = reduction * Part
        IntVec = np.append(
            IntVec, beta * (1 - reduction) * Xt[i]
        )  # updated consumption, add a newborn cohort
        consumptionshare = IntVec / Xt[i]  # consumption share

        # update beliefs
        dDelta_s_t = (post_var(sigma_Y, Vhat, tau) / sigma_Y**2) * (
            -Delta_s_t * dt + np.ones(len(Delta_s_t)) * dZt[i - 1]
        )  # from eq(5)
        if i < Npre:
            # TODO: @chingyulin: this can be optimized
            Delta_s_t = Delta_s_t + dDelta_s_t
            Delta_s_t = np.append(Delta_s_t, 0)  # newborns begin with 0
        else:
            DELbias = np.sum(dZt[int(i - Npre) : i]) / T_hat

            Delta_s_t += dDelta_s_t
            # TODO: @chingyulin: this can be optimized
            Delta_s_t = np.append(
                Delta_s_t, DELbias
            )  # newborns begin with available earlier observations

        # update tau
        tau += dt
        tau = np.append(tau, 0)  # TODO: @chingyulin: this can be optimized

        # find the market clearing theta, given beliefs and consumption shares
        # need a large enough number of cohorts to make the distribution of beliefs reasonably continuous
        if i < Npre:
            MaxThetaDelta_s_t = (
                Delta_s_t  # relax the short-sale constraint in the beginning
            )
        else:
            lowest_bound = -np.max(Delta_s_t)  # absolute lower bound for theta
            # TODO: @GoPenguinGo: is `10` in the argument of the bisection a hard-coded value?
            # Should it's put as a configurable parameter?
            theta_t[i] = bisection(
                solve_theta, lowest_bound, 1e2, consumptionshare, Delta_s_t, sigma_Y
            )  # solve for theta
            MaxDeltaTheta_s_t = np.maximum(
                -theta_t[i], Delta_s_t
            )  # update max(Delta_s_t, -theta)

    # similar to LookForTheta function, store the final value of elements in eq(24)
    invest = Delta_s_t >= -theta_t[Nt - 1]
    invest_f = invest * consumptionshare
    DeltabarCondi = np.sum(Delta_s_t * invest_f)  # eq(24) experience component
    fCondi = np.sum(invest_f)  # eq(24) constraint component

    return (
        DeltaConditional,
        IntVec,
        Xt,
        Delta_s_t,
        Yt,
        Zt,
        consumptionshare,
        tau,
        MaxThetaDelta_s_t,
        DeltabarCondi,
        fCondi,
    )
