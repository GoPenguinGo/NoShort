import numpy as np
from typing import Tuple
from src.stats import post_var
from src.solver import bisection, solve_theta
from tqdm import tqdm
from numba import jit


# todo: trace output, consumption, and wealth
def simulate_cohorts(
        biasvec: np.ndarray,
        dZt: np.ndarray,
        Nt: int,
        Nc: int,
        tau: np.ndarray,
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
        mode: str,
        cohort_size: np.ndarray,
        f_st: np.ndarray,
        Delta_s_t: np.ndarray,
        eta_st_ss: np.ndarray,
        eta_bar: np.ndarray,
        MaxThetaDelta_s_t: np.ndarray,
        invest_tracker: np.ndarray
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
        # IntVec (np.ndarray): ~similar to consumption share, shape(Nc)
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
        mode (str): versions of the model
        - from the cohort_builder function -
        f_st (np.ndarray): consumption share input
        eta_st_ss (np.ndarray): disagreement input
        eta_bar (np.ndarray): average disagreement input
        Delta_s_t (np.ndarray): bias for each cohort, shape(Nc)
        MaxThetaDelta_s_t (np.ndarray): max(Delta_), shape(Nc)
        invest_tracker (np.ndarry): shape(Nt)

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
    # Et = np.ones(Nt)
    # Vt = np.ones(Nt)

    dR = np.zeros(Nt)
    BIGDELTA = np.zeros((Nt, Nc))
    BIGMAX = np.zeros((Nt, Nc))  # stores max(delta, -theta)
    BIGF = np.zeros((Nt, Nc))
    BIGPORT = np.zeros((Nt, Nc))
    BIGFCONDI = np.zeros((Nt))
    BIGDELTABARCONDI = np.zeros((Nt))
    BIGPOPU = np.zeros((Nt))

    # Expected returns
    mu_S = np.zeros(Nt)  # expected return under true measure
    mu_S_s = np.zeros((Nt, Nc))  # expected return under agent-measure
    mu_hat_S = np.zeros(Nt)  # average belief in the economy

    # Consumption
    # Wealth todo: write the code to track the evolution of wealth

    # Aggregate quantities
    r = np.zeros(Nt)  # interest rate
    theta = np.zeros(Nt)  # market price of risk

    for i in tqdm(range(0, Nt)):
        # todo:
        #  (1) think about the time sequence of how things happen (done)
        #  (2) think about the drop case: once an agent is out of the stock market, stop updating and stop investing for good
        #       currently I let them update, but keep their "opinions" muted
        #       in this case, the survey data type of analysis wouldn't work
        eta_st_ss = eta_st_ss * np.exp(
            -0.5 * MaxThetaDelta_s_t ** 2 * dt
            + MaxThetaDelta_s_t * dZt[i]
        )

        Part = beta * np.exp(-beta * tau) * eta_bar * eta_st_ss
        eta_st_ss = eta_st_ss[1:]
        eta_st_ss = np.append(eta_st_ss, 1)
        Part = Part[1:]
        eta_bar_t = np.sum(Part * dt) / (1 - beta * dt)
        eta_bar = eta_bar[1:]
        eta_bar = np.append(eta_bar, eta_bar_t)

        # update beliefs
        dDelta_s_t = (post_var(sigma_Y, Vhat, tau) / sigma_Y**2) * (
            -Delta_s_t * dt + dZt[i]
        )  # from eq(5)
        if i < Npre:
            DELbias = (np.sum(biasvec[i:]) + np.sum(dZt[:i])) / T_hat
        else:
            DELbias = np.sum(dZt[i - Npre: i]) / T_hat

        Delta_s_t = Delta_s_t[1:] + dDelta_s_t[1:]
        Delta_s_t = np.append(Delta_s_t, DELbias)

        # consumption or wealth share:
        f_st = Part / eta_bar_t * dt
        f_st = np.append(f_st, beta * dt)  # cohort consumption share

        # Wealth todo: write the code to track evolution of wealth; test if sum(f_st * dt) == 1
        # stock_wealth = Yt[i] / w
        # w_st = stock_wealth * f_st / (nu * np.exp(-nu * tau))

        # find the market clearing theta, given beliefs and consumption shares
        # todo: participation rate seems to low
        if mode == 'drop':
            invest_tracker = invest_tracker[1:]
            invest_tracker = np.append(invest_tracker, 1)
            possible_cons_share = f_st * invest_tracker
            possible_delta_st = Delta_s_t * invest_tracker
            lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
            theta_t = bisection(
                solve_theta, lowest_bound, 10, possible_cons_share, possible_delta_st, sigma_Y
            )  # solve for theta
            invest = Delta_s_t >= -theta_t
            invest_tracker = invest * invest_tracker
            MaxThetaDelta_s_t = Delta_s_t * invest_tracker + (1 - invest_tracker) * (-theta_t)
            invest_fst = invest_tracker * f_st
            popuCondi = np.sum(cohort_size * invest_tracker)
            DeltabarCondi = np.sum(Delta_s_t * invest_fst)
            fCondi = np.sum(invest_fst)
            Port = (MaxThetaDelta_s_t + theta_t) / sigma_S  # todo: *fst, Wst

            # prepare invest_tracker for the next period
            invest_tracker = invest_tracker[1:]
            invest_tracker = np.append(invest_tracker, 1)

        else:
            lowest_bound = -np.max(Delta_s_t)  # absolute lower bound for theta
            theta_t = bisection(
                solve_theta, lowest_bound, 10, f_st, Delta_s_t, sigma_Y
            )  # solve for theta
            MaxThetaDelta_s_t = np.maximum(
                -theta_t, Delta_s_t
            )  # update max(Delta_s_t, -theta)
            invest = Delta_s_t >= -theta_t
            invest_fst = invest * f_st
            popuCondi = np.sum(cohort_size * invest)
            DeltabarCondi = np.sum(Delta_s_t * invest_fst)
            fCondi = np.sum(invest_fst)
            Port = (MaxThetaDelta_s_t + theta_t) / sigma_S

        r_t = (
                rho
                + mu_Y
                + nu - beta
                - sigma_Y * theta_t
        )

        mu_S_t = sigma_S * theta_t + r_t
        # todo: need to change the updating mechanism of Delta_s_t once a cohort quits the stock market
        mu_S_st = (
                mu_S_t + sigma_S * Delta_s_t
        )  # expected stock return for agent born at t
        muhat_S_t = mu_S_t + sigma_S * np.sum(
            cohort_size * Delta_s_t
        )  # survey average forecast

        dR[i] = mu_S_t * dt + sigma_S * dZt[i]  # mu_t^Sdt + sigma_t^Sdz_t

        # Et[i] = sum(f * (Vbar / (1 + (Vbar / sigma_Y ** 2) * dt * RevNt)) * (1 / sigma_Y))
        # Vt[i] = (sum(f * Delta_s_t ** 2) - Deltabar2[i] ** 2) * sigma_Y

        # todo: * Wst, and
        #  (1) track wealth and consumption
        #  (2) track realized stock return
        #  (3) test if the sum of wealth in the economy equals to stock value

        theta[i] = theta_t
        r[i] = r_t
        mu_S[i] = mu_S_t
        mu_S_s[i, :] = mu_S_st
        mu_hat_S[i] = muhat_S_t
        BIGF[i, :] = f_st
        BIGDELTA[i, :] = Delta_s_t
        BIGMAX[i, :] = MaxThetaDelta_s_t
        BIGFCONDI[i] = fCondi
        BIGDELTABARCONDI[i] = DeltabarCondi
        BIGPORT[i] = Port
        BIGPOPU[i] = popuCondi

    return (
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



# Part = IntVec * np.exp(
#     -(0.5 * MaxThetaDelta_s_t ** 2) * dt + MaxThetaDelta_s_t * dZt[i]
# )
#
# sumPart = np.sum(Part)
# # Deltabar2Conditional[i] = np.sum(Part * MaxThetaDelta_s_t) / sumPart
# IntVec = reduction * Part[1:]
# # IntVec = np.append(IntVec, beta * (1 - reduction) * sumPart)
# IntVec = np.append(IntVec, beta / nu * (1 - reduction) * sumPart)
#
# f = IntVec / sumPart  # consumption share
#
# # Updating: (exogenous, for next period)
# dDelta_s_t = (post_var(sigma_Y, Vhat, tau) / sigma_Y ** 2) * (
#         -Delta_s_t * dt + dZt[i]
# )
# if i < Npre:
#     DELbias = (np.sum(biasvec[i:]) + np.sum(dZt[:i])) / T_hat
# else:
#     DELbias = np.sum(dZt[i - Npre: i]) / T_hat
#
# Delta_s_t = Delta_s_t[1:] + dDelta_s_t[1:]
# Delta_s_t = np.append(Delta_s_t, DELbias)



#####################################################################################

def simulate_cohorts_complete_market(
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
        mode (str): describes the mode of the func

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
    # BIGDELTA = np.zeros((Nt, Nc))
    # BIGMAX = np.zeros((Nt, Nc))  # stores max(delta, -theta)
    BIGF = np.zeros((Nt, Nc))
    BIGPORT = np.zeros((Nt, Nc))
    # BIGFCONDI = np.zeros((Nt))
    # BIGDELTABARCONDI = np.zeros((Nt))
    # BIGPOPU = np.zeros((Nt))

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

    for i in tqdm(range(0, Nt)):

        Part = IntVec * np.exp(
            -(0.5 * MaxThetaDelta_s_t ** 2) * dt + MaxThetaDelta_s_t * dZt[i]
        )

        sumPart = np.sum(Part)
        # Deltabar2Conditional[i] = np.sum(Part * MaxThetaDelta_s_t) / sumPart
        f = Part / sumPart  # consumption share

        # find theta
        A = -np.max(Delta_s_t)
        theta_t = bisection(
            solve_theta, A, 10, f, Delta_s_t, sigma_Y
        )
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
                - (sigma_Y ** 2 - sigma_Y * DeltabarCondi) / fCondi
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
        dDelta_s_t = (post_var(sigma_Y, Vhat, tau) / sigma_Y ** 2) * (
                -Delta_s_t * dt + dZt[i]
        )
        if i < Npre:
            DELbias = (np.sum(biasvec[i:]) + np.sum(dZt[:i])) / T_hat
        else:
            DELbias = np.sum(dZt[i - Npre: i]) / T_hat

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
