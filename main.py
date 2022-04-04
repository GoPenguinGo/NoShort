import time
import numpy as np
import matplotlib.pyplot as plt
<<<<<<< Updated upstream
from tqdm import tqdm
from src.cohort_builder import build_cohorts
from src.cohort_simulator import simulate_cohorts

# TODO: @chingyulin: make cohort a class
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
beta = (rho + nu - D**0.5) / (2 * nu)
rlog = rho + mu_Y - sigma_Y**2

# Setting prior variance
T_hat = 20  # Pre-trading period
# dt = 1 / 4
# Tcohort = 100
dt = 1 / 12  # time incremental
Npre = int(T_hat / dt)
Vbar = (sigma_Y**2) / T_hat  # prior variance
T_cohort = 500  # time horizon to keep track of cohorts
Nt = int(T_cohort / dt)

MC = 1
fMAT = np.zeros((MC, Nt))

time_tolerance = 5
=======
from typing import Callable, Tuple
from tqdm import tqdm
from src.cohort_builder import build_cohorts
from src.cohort_simulator import simulate_cohorts
from src.param import *

#############################################################################################################

# TODO: @chingyulin: make cohort a class
>>>>>>> Stashed changes

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
<<<<<<< Updated upstream
        fCondi,
=======
        fCondi

>>>>>>> Stashed changes
    ) = build_cohorts(
        dZt=dZt,
        Nt=Nt,
        dt=dt,
        rho=rho,
        Vbar=Vbar,
        mu_Y=mu_Y,
        sigma_Y=sigma_Y,
        beta=beta,
        T_hat=T_hat,
        nu=nu,
    )
    fMAT[i, :] = f
if time.time() - time_s > time_tolerance:
    print(f"It takes more than {time_tolerance}s to build up the cohorts")


# The main loop builds up the economy with a large number of cohorts, and simulates the stationary economy forward
for k in range(Mpaths):
    s = time.time()
    if k % 10 == 0:
        dZt = dt**0.5 * np.random.randn(int(Nt - 1))
        (
            DeltaConditional,
            IntVec,
            Xt,
            Delta_s_t,
            Yt,
            Zt,
            f,
            tau,
            MaxThetaDelta_s_t,
            DeltabarCondi,
            fCondi,
<<<<<<< Updated upstream
        ) = build_cohorts(dZt, Nt, dt, rho, nu, Vbar, mu_Y, sigma_Y, beta, T_hat)
    dZforbias = np.diff(Zt)
=======
        ) = build_cohorts(dZt, Nt, dt, rho, nu, Vhat, mu_Y, sigma_Y, beta, T_hat)

    dZforbias = np.diff(Zt)  # dZt used in the build_cohorts function
>>>>>>> Stashed changes
    biasvec = dZforbias[-Npre:]
    dZt = dt**0.5 * np.random.randn(Nt)  # dZt forward
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
    ) = simulate_cohorts(  # TODO: missing fCondi
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
        beta,
        T_hat,
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
