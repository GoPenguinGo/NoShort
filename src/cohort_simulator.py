import numpy as np
from typing import Tuple
from src.stats import post_var
from src.solver import bisection, solve_theta
from tqdm import tqdm
from numba import jit


# todo: trace output, consumption, and wealth
def simulate_cohorts(
        Y: np.ndarray,
        biasvec: np.ndarray,
        dZ: np.ndarray,
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
        omega: float,
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
    # cohort-specific terms:
    Delta = np.zeros((Nt, Nc))  # stores bias in beliefs
    max = np.zeros((Nt, Nc))  # stores max(delta, -theta)
    f = np.zeros((Nt, Nc))  # evolution of cohort consumption share
    pi = np.zeros((Nt, Nc))  # portfolio choices
    w_cohort = np.zeros((Nt, Nc))  # evolution of wealth for cohorts
    w = np.zeros((Nt, Nc))  # evolution of wealth for individual agents alive

    # aggregate terms:
    dR = np.zeros(Nt)  # stores stock returns
    r = np.zeros(Nt)  # interest rate
    theta = np.zeros(Nt)  # market price of risk
    f_parti = np.zeros((Nt))  # consumption share of the stock market participants
    Delta_bar_parti = np.zeros((Nt))  # disagreement of the stock market participants
    parti = np.zeros((Nt))  # participation rate

    # Expected returns
    mu_S = np.zeros(Nt)  # expected return under true measure
    mu_S_s = np.zeros((Nt, Nc))  # expected return under agent-measure
    mu_hat_S = np.zeros(Nt)  # average belief in the economy

    dR_t = 0
    age = np.zeros(Nt)

    for i in tqdm(range(Nt)):
        # todo: think about the drop case: once an agent is out of the stock market, stop updating and stop investing for good
        #       currently I let them update, but keep their "opinions" muted
        #       in this case, the survey data type of analysis wouldn't work

        # realization of shocks
        dZ_t = dZ[i]

        eta_st_ss = eta_st_ss * np.exp(
            -0.5 * MaxThetaDelta_s_t ** 2 * dt
            + MaxThetaDelta_s_t * dZ_t
        )
        part = beta * np.exp(-beta * tau) * eta_bar * eta_st_ss
        # add a new cohort
        eta_st_ss = eta_st_ss[1 : ]
        eta_st_ss = np.append(eta_st_ss, 1)
        part = part[1 : ]
        eta_bar_t = np.sum(part * dt) / (1 - beta * dt)
        eta_bar = eta_bar[1 : ]
        eta_bar = np.append(eta_bar, eta_bar_t)

        # Cohort consumption (wealth) share:
        # sum(f_st * dt) == 1 for any t
        f_st = part / eta_bar_t
        f_st = np.append(f_st, beta)  # add a new cohort who consumes beta proportion of the total output

        # Wealth
        if i == 0:
            w_cohort_st = Y[i] / omega * f_st
            w_st = w_cohort_st / cohort_size * dt
        else:
            dR_t = mu_S_t * dt + sigma_S * dZ_t  # realized stock return, mu_t^Sdt + sigma_t^Sdz_t
            w_t = Y[i] / omega
            dw_st = ((r_t + nu - beta - omega) * w_st + pi_st * (mu_S_t - r_t)) * dt + pi_st * sigma_S * dZ_t  # r_t, theta_t, pi_st from last loop, dZ_t just realized
            w_st = w_st[1:] + dw_st[1:]
            w_st = np.append(w_st, w_t * beta / nu)
            w_cohort_st = w_st * cohort_size / dt

        # update beliefs
        # todo: need to change the updating mechanism of Delta_s_t once a cohort quits the stock market
        dDelta_s_t = (
            post_var(sigma_Y, Vhat, tau) / sigma_Y**2
                     ) * (
            -Delta_s_t * dt + dZ_t
        )  # from eq(5)
        if i < Npre-1:
            init_bias = (np.sum(biasvec[i+1:]) + np.sum(dZ[:i+1])) / T_hat
        else:
            init_bias = np.sum(dZ[i+1 - Npre: i+1]) / T_hat

        Delta_s_t = Delta_s_t[1:] + dDelta_s_t[1:]
        Delta_s_t = np.append(Delta_s_t, init_bias)

        # find the market clearing theta, given beliefs and consumption shares of cohorts in the economy
        if mode == 'drop':
            invest_tracker = invest_tracker[1:]
            invest_tracker = np.append(invest_tracker, 1)
            possible_cons_share = f_st * dt * invest_tracker
            possible_delta_st = Delta_s_t * invest_tracker
            lowest_bound = -np.max(possible_delta_st)  # absolute lower bound for theta among active investors
            theta_t = bisection(
                solve_theta, lowest_bound, 10, possible_cons_share, possible_delta_st, sigma_Y
            )  # solve for theta
            a = Delta_s_t + theta_t
            invest = (a >= 0)
            invest_tracker = invest * invest_tracker
            MaxThetaDelta_s_t = a * invest_tracker - theta_t
            invest_fst = invest_tracker * f_st * dt
            popu_parti_t = np.sum(cohort_size * invest_tracker)
            Delta_bar_parti_t = np.sum(Delta_s_t * invest_fst)
            f_parti_t = np.sum(invest_fst)
            pi_st = (MaxThetaDelta_s_t + theta_t) / sigma_S * w_st
            age_t = np.sum(cohort_size * tau * invest_tracker)

        else:
            lowest_bound = -np.max(Delta_s_t)  # absolute lower bound for theta
            f_st_standard = f_st * dt
            theta_t = bisection(
                solve_theta, lowest_bound, 10, f_st_standard, Delta_s_t, sigma_Y
            )  # solve for theta
            MaxThetaDelta_s_t = np.maximum(
                -theta_t, Delta_s_t
            )  # update max(Delta_s_t, -theta)
            invest = Delta_s_t >= -theta_t
            invest_fst = invest * f_st_standard
            popu_parti_t = np.sum(cohort_size * invest)
            Delta_bar_parti_t = np.sum(Delta_s_t * invest_fst)
            age_t = np.sum(cohort_size * tau * invest)
            f_parti_t = np.sum(invest_fst)
            pi_st = (MaxThetaDelta_s_t + theta_t) / sigma_S * w_st

        r_t = (
                rho
                + mu_Y
                + nu - beta
                - sigma_Y * theta_t
        )

        mu_S_t = sigma_S * theta_t + r_t
        mu_S_st = (
                mu_S_t + sigma_S * Delta_s_t
        )  # expected stock return for agent cohorts
        # muhat_S_t = mu_S_t + sigma_S * np.sum(
        #     cohort_size * Delta_s_t
        # )  # survey average forecast

        # store the results
        dR[i] = dR_t  # realized return from dZt
        theta[i] = theta_t
        r[i] = r_t
        mu_S[i] = mu_S_t
        mu_S_s[i, :] = mu_S_st
        # mu_hat_S[i] = muhat_S_t
        f[i, :] = f_st
        Delta[i, :] = Delta_s_t
        max[i, :] = MaxThetaDelta_s_t
        f_parti[i] = f_parti_t
        Delta_bar_parti[i] = Delta_bar_parti_t
        pi[i, :] = pi_st
        parti[i] = popu_parti_t
        w[i, :] = w_st
        w_cohort[i, :] = w_cohort_st
        age[i] = age_t

    return (
        mu_S,
        mu_S_s,
        # mu_hat_S,
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
