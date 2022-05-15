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

# Some pre-calculations
# D = (rho + nu) * (rho + nu - 4 * nu ** 2)
# beta = (rho + nu - D ** 0.5) / (2 * nu)

# Setting prior variance
T_hat = 20  # Pre-trading period
dt = 1 / 12  # time incremental
Npre = int(T_hat / dt)
Vhat = (sigma_Y ** 2) / T_hat  # prior variance
T_cohort = 500  # time horizon to keep track of cohorts
Nt = int(T_cohort / dt)  # number of periods
Nc = int(T_cohort / dt)  # number of cohorts

mode1 = 'keep'
mode2 = 'drop'
mode3 = 'complete'
mode4 = 'rich_free'
mode5 = 'back_collect'
mode6 = 'back_renew'
time_tolerance = 5

Mpaths = 20

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
theta_matrix = np.zeros((Mpaths, Nt))
pi_matrix = np.zeros((Mpaths, Nt, Nc))
f_parti_matrix = np.zeros((Mpaths, Nt))
parti_matrix = np.zeros((Mpaths, Nt))
Delta_bar_parti_matrix = np.zeros((Mpaths, Nt))
w_matrix = np.zeros((Mpaths, Nt, Nc))
w_cohort_matrix = np.zeros((Mpaths, Nt, Nc))
age_matrix = np.zeros((Mpaths, Nt))

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


######################## for the drop case
# Store the values from the main loop
mu_C_drop_matrix = np.zeros((Mpaths, Nt))
sigma_C_drop_matrix = np.zeros((Mpaths, Nt))
delta_drop_matrix = np.zeros((Mpaths, Nt, Nc))
r_drop_matrix = np.zeros((Mpaths, Nt))
f_drop_matrix = np.zeros((Mpaths, Nt, Nc))
theta_drop_matrix = np.zeros((Mpaths, Nt))
pi_drop_matrix = np.zeros((Mpaths, Nt, Nc))
f_parti_drop_matrix = np.zeros((Mpaths, Nt))
parti_drop_matrix = np.zeros((Mpaths, Nt))
Delta_bar_parti_drop_matrix = np.zeros((Mpaths, Nt))
w_drop_matrix = np.zeros((Mpaths, Nt, Nc))
w_cohort_drop_matrix = np.zeros((Mpaths, Nt, Nc))
age_drop_matrix = np.zeros((Mpaths, Nt))

# Expected returns
mu_S_drop_matrix = np.zeros((Mpaths, Nt))  # Expected returns under the true measure
mu_S_s_drop_matrix = np.zeros(
    (Mpaths, Nt, Nc)
)  # Expected returns under the measure of the agent we track
mu_hat_S_drop_matrix = np.zeros(
    (Mpaths, Nt)
)  # Simple average of expected returns, or consensus belief

# Equity risk premium
erp_S_drop_matrix = np.zeros(
    (Mpaths, Nt)
)

erp_S_s_drop_matrix = np.zeros(
    (Mpaths, Nt, Nc)
)
erp_hat_S_drop_matrix = np.zeros(
    (Mpaths, Nt)
)


Et_drop_matrix = np.zeros((Mpaths, Nt))
Vt_drop_matrix= np.zeros((Mpaths, Nt))
dR_drop_matrix = np.zeros((Mpaths, Nt))