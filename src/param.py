import numpy as np

# Parameters
rho = 0.001  # Time discount factor
nu = 0.02  # Death rate
mu_Y = 0.02  # Growth rate of output
sigma_Y = 0.033  # Standard deviation of output
sigma_Y_sqr = sigma_Y ** 2
sigma_S = (
    sigma_Y  # In equilibrium the stock price diffusion is the same as output diffusion
)

beta = 0.015  # marginal rate of wealth tax
# beta = 0.02
# beta = 0.01
omega = rho + nu - beta  # marginal propensity to consume

# Setting prior variance
dt = 1 / 12  # time incremental
# T_hat = 20  # Pre-trading period
# Npre = int(T_hat / dt)
# Vhat = (sigma_Y ** 2) / T_hat  # prior variance
T_cohort = 500  # time horizon to keep track of cohorts
Nt = int(T_cohort / dt)  # number of periods
Nc = int(T_cohort / dt)  # number of cohorts


# generate values that are fixed in the main loop
tau = np.arange(T_cohort, 0, -dt)  # age from 500 to 0
cohort_size = nu * np.exp(-nu * (tau - dt)) * dt  # cohort size when a new cohort is just born

# create age quartiles for analysis
cummu_popu = np.cumsum(cohort_size)
tau_cutoff1 = np.searchsorted(cummu_popu, 0.75)
tau_cutoff2 = np.searchsorted(cummu_popu, 0.5)
tau_cutoff3 = np.searchsorted(cummu_popu, 0.25)

Mpaths = 50

# for graphs:
Tkeep = 100
Nkeep = int(Tkeep / dt)
Tsample = int(T_cohort / 100)
Nsamples = 500
stepcorr = int(Tsample / dt)
corrZport = np.zeros((Mpaths, Nsamples))
corrZMUs_t = np.zeros((Mpaths, Nsamples))
corrMU_sMUs_t = np.zeros((Mpaths, Nsamples))
corrMuSmuHat = np.zeros((Mpaths, 1))

mu_C_s_t = np.zeros((Mpaths, Nsamples))
log_mu_C_s_t = np.zeros((Mpaths, Nsamples))
sigma_C_s_t = np.zeros((Mpaths, Nsamples))
std_C_s_t = np.zeros((Mpaths, Nsamples))

####################################################################################
# Store the values from the main loop
dZ_matrix = np.zeros((Mpaths, Nt))
Z_matrix = np.zeros((Mpaths, Nt))
mu_C_matrix = np.zeros((Mpaths, Nt))
sigma_C_matrix = np.zeros((Mpaths, Nt))
delta_matrix = np.zeros((Mpaths, Nt, Nc))
r_matrix = np.zeros((Mpaths, Nt))
f_matrix = np.zeros((Mpaths, Nt, Nc))
f_short_matrix = np.zeros((Mpaths, Nt, Nc))
f_long_matrix = np.zeros((Mpaths, Nt, Nc))
theta_matrix = np.zeros((Mpaths, Nt))
pi_matrix = np.zeros((Mpaths, Nt, Nc))
f_parti_matrix = np.zeros((Mpaths, Nt))
popu_parti_matrix = np.zeros((Mpaths, Nt))
popu_can_short_matrix = np.zeros((Mpaths, Nt))
popu_short_matrix = np.zeros((Mpaths, Nt))
popu_long_matrix = np.zeros((Mpaths, Nt))
Delta_bar_parti_matrix = np.zeros((Mpaths, Nt))
Delta_bar_long_matrix = np.zeros((Mpaths, Nt))
Delta_bar_short_matrix = np.zeros((Mpaths, Nt))
w_matrix = np.zeros((Mpaths, Nt, Nc))
w_cohort_matrix = np.zeros((Mpaths, Nt, Nc))
age_parti_matrix = np.zeros((Mpaths, Nt))
age_short_matrix = np.zeros((Mpaths, Nt))
age_long_matrix = np.zeros((Mpaths, Nt))
n_parti_matrix = np.zeros((Mpaths, Nt))

invest_tracker_matrix = np.zeros((Mpaths, Nt, Nc))
can_short_tracker_matrix = np.zeros((Mpaths, Nt, Nc))
long_indicator_matrix = np.zeros((Mpaths, Nt, Nc))
short_indicator_matrix = np.zeros((Mpaths, Nt, Nc))

# Expected returns
mu_S_matrix = np.zeros((Mpaths, Nt))  # Expected returns under the true measure
mu_S_s_matrix = np.zeros(
    (Mpaths, Nt, Nc)
)  # Expected returns under the measure of the agent we track
mu_hat_S_matrix = np.zeros(
    (Mpaths, Nt)
)  # Simple average of expected returns, or consensus belief

# Equity risk premium
erp_S_matrix = np.zeros(
    (Mpaths, Nt)
)

erp_S_s_matrix = np.zeros(
    (Mpaths, Nt, Nc)
)
erp_hat_S_matrix = np.zeros(
    (Mpaths, Nt)
)


Et_matrix = np.zeros((Mpaths, Nt))
Vt_matrix = np.zeros((Mpaths, Nt))
dR_matrix = np.zeros((Mpaths, Nt))