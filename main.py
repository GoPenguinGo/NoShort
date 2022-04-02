import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from tqdm import tqdm

# TODO: move the function and parameters to different files


# @GoPenguinGo: it seems tau is always np.ndarray right?
def post_var(sigY: float, Vbar: float, tau: np.ndarray) -> np.ndarray:
    """Calculate the posterior variance, correspond to eq(2)

    Args:
        sigY (float): _description_ #TODO: @GoPenguinGo
        Vbar (float): _description_
        tau (np.ndarray): (t - s) in eq(2), shape (T, )

    Returns:
        np.ndarray: shape (T, )
    """
    V = sigY**2 * Vbar / (sigY**2 * np.ones(len(tau)) + Vbar * tau)
    return V


def search_for_theta(
    thetaguess: np.float64, consumptionshare: np.ndarray, Delta_s_t: np.ndarray
) -> np.float64:
    """RHS - LHS of the eq(24), used to iteratively solve theta 

    Args:
        thetaguess (np.float64): _description_ #TODO: GoPenguinGo
        consumptionshare (np.ndarray): shape (T, )
        Delta_s_t (np.ndarray): shape (T, )

    Returns:
        np.float64: RHS - LHS
    """
    invest = (
        Delta_s_t >= -thetaguess
    )  # eq(10) and eq(11), invest if theta_s_t >= -theta, constrained if otherwise
    invest_consumptionshare = invest * consumptionshare
    DeltabarCondi = np.sum(
        Delta_s_t * invest_consumptionshare
    )  # Experience component, as defined below eq(24)
    InvestCons = np.sum(
        invest_consumptionshare
    )  # Constraint component, as defined below eq(24)
    g = (
        sigma_Y - DeltabarCondi
    ) / InvestCons - thetaguess  # RHS - LHS, equals to 0 if find the right theta
    return g


def bisection(
    optimfun: Callable[[float, np.ndarray, np.ndarray], float],
    xlow: float,
    xhigh: float,
    arg1: np.ndarray,
    arg2: np.ndarray,
    convcrit: float=1e-6,
) -> np.float64:
    """Bisection method to solve x (theta)

    Args:
        optimfun (Callable[[float, np.ndarray, np.ndarray], float]): _description_
        xlow (float): lower bound for x
        xhigh (float): upper bound for x
        arg1 (np.ndarray): second input for optimfun
        arg2 (np.ndarray): third input for optimfun
        convcrit (float, optional): converging criteria. Defaults to 1e-6.

    Returns:
        np.float64: _description_ #TODO: GoPenguinGo
    """
    flow = optimfun(xlow, arg1, arg2)
    fhigh = optimfun(xhigh, arg1, arg2)
    diff = 1
    iter = 0

    while diff > convcrit:
        xmid = (xlow + xhigh) / 2
        fmid = optimfun(xmid, arg1, arg2)
        if flow * fmid < 0:  # root between flow and fmid
            xhigh = xmid
            fhigh = fmid
        elif fmid * fhigh < 0:  # root between fmid and fhigh
            xlow = xmid
            flow = fmid
        diff = abs(fhigh - flow)
        iter += 1
    return xmid


def BuildUpCohortsMAIN(
    dZt: np.ndarray,
    Nt: int,
    dt: float,
    rho: float,
    nu: float,
    Vbar: float,
    mu_Y: float,
    sigma_Y: float,
    beta: float,
    That: float,
):
    """builds up a sufficiently large set of cohorts in the economy, view each cohort as one agent with a constantly shrinking size

    Args:
        dZt (np.ndarray): random shocks of aggregate output for each period, (Nt-1)*1
        Nt (int): number of periods
        dt (float): unit of time
        rho (float): rho, discount factor
        nu (float): birth / death rate, each cohort starts at size nu and shrinks at speed of nu
        Vbar (float): initial variance of beliefs
        mu_Y (float): mean and of aggregate output growth
        sigma_Y (float): mean and sd of aggregate output growth
        beta (float): initial consumption of the newborn agents
        That (float): pre-trading years

    Returns:
        _type_: _description_ #TODO: @GoPenguinGo: add the return type and the description
    """

    Npre = int(That / dt)  # Number of pre-trading observations
    Zt = np.insert(np.cumsum(dZt), 0, 0)  # cumulated shocks, Nt * 1
    yg = (mu_Y - 0.5 * sigma_Y**2) * dt * np.ones(
        int(Nt - 1)
    ) + sigma_Y * dZt  # output in log, (Nt - 1) *1, equation (1)
    Yt = np.insert(np.exp(np.cumsum(yg)), 0, 1)  # output, Nt *1
    DeltaConditional = np.zeros(Nt)
    Delta_s_t = np.zeros(1)  # belief bias, Eq(3)
    MaxDeltaTheta_s_t = np.zeros(1)  # disagreement, Eq(11)
    Xt = np.ones(Nt) * nu * beta  # similar to consumption share, similar to Eq(18)
    IntVec = 1 * nu * beta  # consumption share of a newborn cohort
    # TODO: @chingyulin: tau can allocate the memory
    tau = np.zeros(1)  # t-s
    tau[0] = dt
    reduction = np.exp(-nu * dt)  # cohort size shrink at this rate
    theta_t = np.zeros(Nt)  # market price of risk
    for i in tqdm(range(1, Nt)):
        Part = IntVec * np.exp(
            -(rho + 0.5 * MaxDeltaTheta_s_t * MaxDeltaTheta_s_t) * dt
            + MaxDeltaTheta_s_t * dZt[i - 1]
        )  # Consumption of each cohort, Eq(16), where eta_s_t / eta_s_s follows Eq(11)
        if i == 1:  # only one cohort in the economy
            Xt[i] = Part
            DeltaConditional[i] = Part * MaxDeltaTheta_s_t
        else:  # more cohorts
            Xt[i] = np.sum(Part)  # total consumption
            DeltaConditional[i] = (
                np.sum(Part * MaxDeltaTheta_s_t) / Xt[i]
            )  # Eq(19), consumption weighted max(Delta_s_t, -theta)

        IntVec = reduction * Part
        IntVec = np.append(
            IntVec, beta * (1 - reduction) * Xt[i]
        )  # updated consumption, add a newborn cohort
        consumptionshare = IntVec / Xt[i]  # consumption share

        # update beliefs
        dDelta_s_t = (post_var(sigma_Y, Vbar, tau) / sigma_Y**2) * (
            -Delta_s_t * dt + np.ones(len(Delta_s_t)) * dZt[i - 1]
        )  # from Eq(5)
        if i < Npre:
            #TODO: @chingyulin: this can be optimized
            Delta_s_t = Delta_s_t + dDelta_s_t
            Delta_s_t = np.append(Delta_s_t, 0)  # newborns begin with 0
        else:
            DELbias = np.sum(dZt[int(i - Npre) : i]) / That

            Delta_s_t += dDelta_s_t
            #TODO: @chingyulin: this can be optimized
            Delta_s_t = np.append(
                Delta_s_t, DELbias
            )  # newborns begin with available earlier observations

        # update tau
        tau += dt
        tau = np.append(tau, 0) #TODO: @chingyulin: this can be optimized

        # find the market clearing theta, given beliefs and consumption shares
        # need a large enough number of cohorts to make the distribution of beliefs reasonably continuous
        if i < Npre:
            MaxDeltaTheta_s_t = (
                Delta_s_t  # relax the short-sale constraint in the beginning
            )
        else:
            lowest_bound = -np.max(Delta_s_t)  # absolute lower bound for theta
            # TODO: @GoPenguinGo: is `10` in the argument of the bisection a hard-coded value?
            # Should it's put as a configurable parameter?
            theta_t[i] = bisection(search_for_theta, lowest_bound, 10, consumptionshare, Delta_s_t)  # solve for theta
            MaxDeltaTheta_s_t = np.maximum(
                -theta_t[i], Delta_s_t
            )  # update max(Delta_s_t, -theta)

    # similar to LookForTheta function, store the final value of elements in Eq(24)
    invest = Delta_s_t >= -theta_t[Nt - 1]
    invest_f = invest * consumptionshare
    DeltabarCondi = np.sum(Delta_s_t * invest_f)  # Eq(24) experience component
    fCondi = np.sum(invest_f)  # Eq(24) constraint component

    return (
        DeltaConditional,
        IntVec,
        Xt,
        Delta_s_t,
        Yt,
        Zt,
        consumptionshare,
        tau,
        MaxDeltaTheta_s_t,
        DeltabarCondi,
        fCondi,
    )


def SimCohortsMAIN(
    biasvec,
    dZt,
    Nt,
    tau,
    IntVec,
    Delta_s_t,
    MaxDeltaTheta_s_t,
    dt,
    rho,
    nu,
    Vbar,
    mu_Y,
    sigma_Y,
    sigma_S,
    bet,
    That,
    Npre,
    DeltabarCondi,
    fCondi,
):
    # Initializing variables
    Xt2 = np.ones(Nt)
    Deltabar2Conditional = np.ones(Nt)
    Et = np.ones(Nt)
    Vt = np.ones(Nt)

    part1 = np.zeros(Nt)
    dR = np.zeros(Nt)
    # counter = 0
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
        BIGMAX[i, :] = MaxDeltaTheta_s_t
        BIGFCONDI[i] = fCondi
        BIGDELTABARCONDI[i] = DeltabarCondi

        if i < Nt - 1:  # information & belief
            if i == 0:
                part1[i] = sum(biasvec) / That
            else:
                part1[i] = part1[i - 1] + (
                    post_var(sigma_Y, Vbar, (i * dt)) / sigma_Y**2
                ) * (-part1[i - 1] * dt + dZt[i - 1])

        Part = IntVec * np.exp(
            -(0.5 * MaxDeltaTheta_s_t**2) * dt + MaxDeltaTheta_s_t * dZt[i]
        )

        Xt2[i] = sum(Part)
        Deltabar2Conditional[i] = sum(Part * MaxDeltaTheta_s_t) / Xt2[i]
        f = Part / Xt2[i]  # consumption share
        BIGF[i, :] = f

        # find theta
        A = -max(Delta_s_t)
        theta_t[i] = bisection(search_for_theta, A, 10, f, Delta_s_t)
        MaxDeltaTheta_s_t = np.maximum(-theta_t[i], Delta_s_t)
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
        dDelta_s_t = (post_var(sigma_Y, Vbar, tau) / sigma_Y**2) * (
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
        Deltabar2,
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
        Et,
        Vt,
        dR,
    )

#############################################################################################################
# TODO: move the paramters to a separate config file
# Parameters
rho = 0.001  # Time discount factor
nu = 0.02  # Death rate
mu_Y = 0.02  # Growth rate of output
sigma_Y = 0.033  # Standard deviation of output
sigma_S = (
    sigma_Y  # In equilibrium the stock price diffusion is the same as output diffusion
)
w = 0.92  # Fraction of total output paid out as endowment

# Some pre-calculations
# D = rho ** 2 + 4 * (rho * nu + nu ** 2) * (1 - w)
D = (rho + nu) * (rho + nu - 4 * nu**2)
bet = (rho + nu - D**0.5) / (2 * nu)
rlog = rho + mu_Y - sigma_Y**2

# Setting prior variance
That = 20  # Pre-trading period
# dt = 1 / 4
# Tcohort = 100
dt = 1 / 12  # time incremental
Npre = int(That / dt)
Vbar = (sigma_Y**2) / That  # prior variance
Tcohort = 500  # time horizon to keep track of cohorts
Nt = int(Tcohort / dt)

MC = 1
fMAT = np.zeros((MC, Nt))

time_tolerance = 4

time_s = time.time()
for i in tqdm(range(MC)):
    dZt = np.sqrt(dt) * np.random.randn(int(Nt - 1))
    (
        Deltabar,
        IntVec,
        Xt,
        Delta_s_t,
        Yt,
        Zt,
        f,
        tau,
        MaxDeltaTheta,
        DeltabarCondi,
        fCondi,
    ) = BuildUpCohortsMAIN(dZt, Nt, dt, rho, nu, Vbar, mu_Y, sigma_Y, bet, That)
    fMAT[i, :] = f
assert time.time() - time_s < time_tolerance; print(f"It takes more than {time_tolerance} to build up the cohorts")

# Initializing some variables
Mpaths = 100
Tsample = int(Tcohort / 100)
Nsamples = 100
stepcorr = int(Tsample / dt)
corrZport = np.zeros((Mpaths, Nsamples))
corrZMUs_t = np.zeros((Mpaths, Nsamples))
corrMU_sMUs_t = np.zeros((Mpaths, Nsamples))
corrMuSmuHat = np.zeros((Mpaths, 1))
fMAT = np.zeros((Mpaths, Nt))
mC = np.zeros((Mpaths, Nt))
sC = np.zeros((Mpaths, Nt))
DeltaHatMAT = np.zeros((Mpaths, Nt))
rMAT = np.zeros((Mpaths, Nt))
thetaMAT = np.zeros((Mpaths, Nt))
portMAT = np.zeros((Mpaths, Nt))
Zmat = np.zeros((Mpaths, Nt))

# Expected returns
muSMAT = np.zeros((Mpaths, Nt))  # Expected returns under the true measure
muSsMat = np.zeros(
    (Mpaths, Nt)
)  # Expected returns under the measure of the agent we track
muShatMAT = np.zeros(
    (Mpaths, Nt)
)  # Simple average of expected returns, or consensus belief
EtMAT = np.zeros((Mpaths, Nt))
VtMAT = np.zeros((Mpaths, Nt))
RxMAT = np.zeros((Mpaths, Nt))
muCst = np.zeros((Mpaths, Nsamples))
logmuCst = np.zeros((Mpaths, Nsamples))
sigCst = np.zeros((Mpaths, Nsamples))
stdCst = np.zeros((Mpaths, Nsamples))

# The main loop builds up the economy with a large number of cohorts, and simulates the stationary economy forward
for k in range(Mpaths):
    s = time.time()
    if k % 10 == 0:
        dZt = dt**0.5 * np.random.randn(int(Nt - 1))
        (
            Deltabar,
            IntVec,
            Xt,
            Delta_s_t,
            Yt,
            Zt,
            f,
            tau,
            MaxDeltaTheta,
            DeltabarCondi,
            fCondi,
        ) = BuildUpCohortsMAIN(dZt, Nt, dt, rho, nu, Vbar, mu_Y, sigma_Y, bet, That)
    dZforbias = np.diff(Zt)
    biasvec = dZforbias[-Npre:]
    dZt = dt**0.5 * np.random.randn(Nt)
    Zt = np.cumsum(dZt)

    (
        Xt2,
        Deltabar2,
        Part1,
        mu_S,
        mu_S_t,
        muhat_S_t,
        r_t,
        theta_t,
        Port,
        muC_s_t,
        sigmaC_s_t,
        BIGf,
        BIGDELTA,
        Et,
        Vt,
        dR,
    ) = SimCohortsMAIN(
        biasvec,
        dZt,
        Nt,
        tau,
        IntVec,
        Delta_s_t,
        dt,
        rho,
        nu,
        Vbar,
        mu_Y,
        sigma_Y,
        sigma_S,
        bet,
        That,
        Npre,
        DeltabarCondi,
        fCondi,
    )

    RxMAT[k, :] = np.transpose(dR)
    EtMAT[k, :] = np.transpose(Et)
    VtMAT[k, :] = np.transpose(Et)
    DeltaHatMAT[k, :] = np.transpose(Deltabar2)
    rMAT[k, :] = np.transpose(r_t)
    thetaMAT[k, :] = np.transpose(theta_t)
    Zmat[k, :] = np.transpose(Zt)

    portMAT[k, :] = np.transpose(Port)

    muSMAT[k, :] = np.transpose(mu_S + rlog - r_t)
    muSsMat[k, :] = np.transpose(mu_S_t + rlog - r_t)
    muShatMAT[k, :] = np.transpose(muhat_S_t + rlog - r_t)

    mu_S = mu_S + rlog - r_t
    muhat_S_t = muhat_S_t + rlog - r_t
    mu_S_t = mu_S_t + rlog - r_t
    mC[k, :] = np.transpose(muC_s_t)
    sC[k, :] = np.transpose(sigmaC_s_t)
    corrMuSmuHat[k] = np.corrcoef(muhat_S_t, mu_S)[0, 1]
    fMAT[k, :] = np.mean(BIGf, axis=0)

    for l in range(Nsamples):
        a = int(l * stepcorr)
        b = int((l + 1) * stepcorr)
        corrZMUs_t[k, l] = np.corrcoef(Zt[a:b], mu_S_t[a:b])[0, 1]
        corrZport[k, l] = np.corrcoef(Zt[a:b], Port[a:b])[0, 1]
        corrMU_sMUs_t[k, l] = np.corrcoef(mu_S[a:b], mu_S_t[a:b])[0, 1]
        muCst[k, l] = np.mean(muC_s_t[a:b])
        logmuCst[k, l] = np.mean(muC_s_t[a:b]) - 0.5 * sum((sigmaC_s_t[a:b]) ** 2)
        sigCst[k, l] = np.mean(sigmaC_s_t[a:b])
        stdCst[k, l] = np.mean(abs(sigmaC_s_t[a:b]))
    print(time.time() - s)

MaxAge = 100
MaxAgeN = int(MaxAge / Tsample)
tperiod = range(Tsample, 100 + Tsample, Tsample)
meanZport = np.mean(corrZport, axis=0)
meanZmus_t = np.mean(corrZMUs_t, axis=0)

# Compute the mean values from the simulation
meanMus = np.mean(corrMU_sMUs_t, axis=0)
meanMuCst = np.mean(muCst, axis=0)
meanSCst = np.mean(sigCst, axis=0)
meanStdCst = np.mean(stdCst, axis=0)
meanLogMuCst = np.mean(logmuCst, axis=0)

# Figures
# Figure 1 in the paper
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 10))
ax1.plot(tperiod, meanMus[:MaxAgeN])
ax2.plot(tperiod, meanZmus_t[:MaxAgeN])

# Figure 2 in the paper
