import numpy as np

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
D = (rho + nu) * (rho + nu - 4 * nu**2)
beta = (rho + nu - D**0.5) / (2 * nu)
rlog = rho + mu_Y - sigma_Y**2

# Setting prior variance
T_hat = 20  # Pre-trading period
dt = 1 / 12  # time incremental
Npre = int(T_hat / dt)
Vhat = (sigma_Y**2) / T_hat  # prior variance
T_cohort = 500  # time horizon to keep track of cohorts
Nt = int(T_cohort / dt)

MC = 1
fMAT = np.zeros((MC, Nt))

time_tolerance = 5

# Initializing some variables
Mpaths = 100
Tsample = int(T_cohort / 100)
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