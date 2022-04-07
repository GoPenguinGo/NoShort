import numpy as np
from src.solver import bisection, solve_theta
from tqdm import tqdm
from typing import Tuple
from src.stats import post_var


def build_cohorts(
    dZt: np.ndarray,
    Nc: int,
    dt: float,
    rho: float,
    nu: float,
    Vhat: float,
    mu_Y: float,
    sigma_Y: float,
    beta: float,
    T_hat: float,
    mode: str
) -> Tuple[
    # np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    # np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """builds up a sufficiently large set of cohorts in the economy, view each cohort as one agent with a constantly shrinking size

    Args:
        dZt (np.ndarray): random shocks of aggregate output for each period, shape (Nc-1, )
        Nc (int): number of periods  = number of cohorts in the economy
        dt (float): unit of time
        rho (float): rho, discount factor
        nu (float): birth / death rate, each cohort starts at size nu and shrinks at speed of nu
        Vhat (float): initial variance of beliefs
        mu_Y (float): mean of aggregate output growth
        sigma_Y (float): sd of aggregate output growth
        beta (float): initial consumption of the newborn agents
        T_hat (float): pre-trading years
        mode (str): describes the mode

    Returns:
        # DeltaConditional (np.ndarray): consumption weighted aggregate max(delta_s_t, -theta_t), as in eq(19), shape(Nc, )
        IntVec (np.ndarray): ~similar to consumption share, shape(Nc, )
        Xt (np.ndarray): xi_t * Yt, shape(Nc, )
        Delta_s_t (np.ndarray): bias, shape(Nc, )
        Yt (np.ndarray): aggregate output, shape(Nc, )
        Zt (np.ndarray): cumulated shocks, shape(Nc, )
        # consumptionshare (np.ndarray): shape(Nc, )
        tau (np.ndarray): t-s, shape(Nc, )
        MaxThetaDelta_s_t (np.ndarray): max(delta_s_t, -theta_t), shape(Nc, )
        #TODO: @chingyulin: use NamedTuple for the return
    """

    Npre: int = int(T_hat / dt)  # Number of pre-trading observations
    Zt = np.insert(np.cumsum(dZt), 0, 0)  # cumulated shocks, Nc * 1
    yg = (mu_Y - 0.5 * sigma_Y**2) * dt + sigma_Y * dZt  # output in log, (Nc - 1) *1, eq(1)
    Yt = np.insert(np.exp(np.cumsum(yg)), 0, 1)  # output, Nc *1
    # DeltaConditional = np.zeros(Nc)
    Delta_s_t = np.zeros(1)  # belief bias, eq(3)
    MaxThetaDelta_s_t = np.zeros(1)  # disagreement, eq(11)
    Xt = np.ones(Nc) * nu * beta  # similar to consumption share, similar to eq(18)
    IntVec = nu * beta  # consumption share of a newborn cohort
    # TODO: @chingyulin: tau can allocate the memory
    tau = np.zeros(1)  # t-s
    tau[0] = dt
    reduction = np.exp(-nu * dt)  # cohort size shrink at this rate
    theta = np.zeros(Nc)  # market price of risk
    invest_tracker = np.ones(Npre)
    for i in tqdm(range(1, Nc)):
    #for i in tqdm(range(1, Npre)):
        Part = IntVec * np.exp(
            -(rho + 0.5 * MaxThetaDelta_s_t * MaxThetaDelta_s_t) * dt
            + MaxThetaDelta_s_t * dZt[i - 1]
        )  # Consumption of each cohort, eq(16), where eta_s_t / eta_s_s follows eq(11)
        if i == 1:  # only one cohort in the economy
            Xt[i] = Part
            # DeltaConditional[i] = Part * MaxThetaDelta_s_t
        else:  # more cohorts
            Xt[i] = np.sum(Part)  # total consumption
            #DeltaConditional[i] = (
                #np.sum(Part * MaxThetaDelta_s_t) / Xt[i]
            #)  # eq(19), consumption weighted max(Delta_s_t, -theta)

        IntVec = reduction * Part
        IntVec = np.append(
            IntVec, beta * (1 - reduction) * Xt[i]
        )  # updated consumption, add a newborn cohort
        consumptionshare = IntVec / Xt[i]  # consumption share

        # update beliefs
        dDelta_s_t = (post_var(sigma_Y, Vhat, tau) / sigma_Y**2) * (
            -Delta_s_t * dt + dZt[i - 1]
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
            if mode == 'drop':
                invest_tracker = np.append(invest_tracker, 1)
                possible_cons_share = consumptionshare * invest_tracker
                possible_delta_st = Delta_s_t * invest_tracker
                theta_t = bisection(
                    solve_theta, lowest_bound, 10, possible_cons_share, possible_delta_st, sigma_Y
                )  # solve for theta
                invest = Delta_s_t >= -theta_t
                invest_tracker = invest * invest_tracker
                MaxThetaDelta_s_t = Delta_s_t * invest_tracker + (1 - invest_tracker) * (-theta_t)
                theta[i] = theta_t

            else:
                lowest_bound = -np.max(Delta_s_t)  # absolute lower bound for theta
                theta[i] = bisection(
                    solve_theta, lowest_bound, 10, consumptionshare, Delta_s_t, sigma_Y
                )  # solve for theta
                MaxThetaDelta_s_t = np.maximum(
                    -theta[i], Delta_s_t
                )  # update max(Delta_s_t, -theta)

    return (
        # DeltaConditional,
        IntVec,
        Xt,
        Delta_s_t,
        Yt,
        Zt,
        # consumptionshare,
        tau,
        MaxThetaDelta_s_t,
        invest_tracker,
    )


#####################################################################################

def build_cohorts_complete_market(
    dZt: np.ndarray,
    Nc: int,
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
    # np.ndarray,
    # np.ndarray,
    # np.ndarray,
    # np.ndarray,
    # np.ndarray,
    # np.ndarray,
]:
    """builds up a sufficiently large set of cohorts in the economy, view each cohort as one agent with a constantly shrinking size
    complete market version
    run this function along with the incomplete version above,
    as this function returns results that are used in comparison with the previous results

    Args:
        dZt (np.ndarray): random shocks of aggregate output for each period, shape (Nc-1, )
        Nc (int): number of periods  = number of cohorts in the economy
        dt (float): unit of time
        rho (float): rho, discount factor
        nu (float): birth / death rate, each cohort starts at size nu and shrinks at speed of nu
        Vhat (float): initial variance of beliefs
        mu_Y (float): mean of aggregate output growth
        sigma_Y (float): sd of aggregate output growth
        beta (float): initial consumption of the newborn agents
        T_hat (float): pre-trading years

    Returns:
        # DeltaConditional (np.ndarray): consumption weighted aggregate max(delta_s_t, -theta_t), as in eq(19), shape(Nc, )
        IntVec (np.ndarray): ~similar to consumption share, shape(Nc, )
        Xt (np.ndarray): xi_t * Yt, shape(Nc, )
        # Delta_s_t (np.ndarray): bias, shape(Nc, )
        # Yt (np.ndarray): aggregate output, shape(Nc, )
        # Zt (np.ndarray): cumulated shocks, shape(Nc, )
        # consumptionshare (np.ndarray): shape(Nc, )
        # tau (np.ndarray): t-s, shape(Nc, )
        # MaxThetaDelta_s_t (np.ndarray): max(delta_s_t, -theta_t), shape(Nc, )
    """

    Npre: int = int(T_hat / dt)  # Number of pre-trading observations

    # Zt = np.insert(np.cumsum(dZt), 0, 0)  # cumulated shocks, Nc * 1
    # yg = (mu_Y - 0.5 * sigma_Y**2) * dt + sigma_Y * dZt  # output in log, (Nc - 1) *1, eq(1)
    # Yt = np.insert(np.exp(np.cumsum(yg)), 0, 1)  # output, Nc *1
    # DeltaConditional = np.zeros(Nc)
    Delta_s_t = np.zeros(1)  # belief bias, eq(3)
    # MaxThetaDelta_s_t = np.zeros(1)  # disagreement, eq(11)
    Xt = np.ones(Nc) * nu * beta  # similar to consumption share, similar to eq(18)
    IntVec = nu * beta  # consumption share of a newborn cohort

    tau = np.zeros(1)  # t-s
    tau[0] = dt
    reduction = np.exp(-nu * dt)  # cohort size shrink at this rate
    theta_t = np.zeros(Nc)  # market price of risk
    for i in tqdm(range(1, Nc)):
        Part = IntVec * np.exp(
            -(rho + 0.5 * Delta_s_t * Delta_s_t) * dt
            + Delta_s_t * dZt[i - 1]
        )  # Consumption of each cohort, eq(16), where eta_s_t / eta_s_s follows eq(11)
        if i == 1:  # only one cohort in the economy
            Xt[i] = Part
            # DeltaConditional[i] = Part * MaxThetaDelta_s_t
        else:  # more cohorts
            Xt[i] = np.sum(Part)  # total consumption
            #DeltaConditional[i] = (
                #np.sum(Part * MaxThetaDelta_s_t) / Xt[i]
            #)  # eq(19), consumption weighted max(Delta_s_t, -theta)

        IntVec = reduction * Part
        IntVec = np.append(
            IntVec, beta * (1 - reduction) * Xt[i]
        )  # updated consumption, add a newborn cohort
        consumptionshare = IntVec / Xt[i]  # consumption share

        # update beliefs
        dDelta_s_t = (post_var(sigma_Y, Vhat, tau) / sigma_Y**2) * (
            -Delta_s_t * dt + dZt[i - 1]
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
        # if i < Npre:
        #     MaxThetaDelta_s_t = (
        #         Delta_s_t  # relax the short-sale constraint in the beginning
        #     )
        # else:
        #     lowest_bound = -np.max(Delta_s_t)  # absolute lower bound for theta
        #     # Should it's put as a configurable parameter?
        #     theta_t[i] = bisection(
        #         solve_theta, lowest_bound, 10, consumptionshare, Delta_s_t, sigma_Y
        #     )  # solve for theta
        #     MaxThetaDelta_s_t = np.maximum(
        #         -theta_t[i], Delta_s_t
        #     )  # update max(Delta_s_t, -theta)

    return (
        # DeltaConditional,
        IntVec,
        Xt,
        # Delta_s_t,
        # Yt,
        # Zt,
        # consumptionshare,
        # tau,
        # MaxThetaDelta_s_t,
    )
