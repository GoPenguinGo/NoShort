import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple
from tqdm import tqdm
from src.cohort_builder import build_cohorts
from src.cohort_simulator import simulate_cohorts


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
Vhat = (sigma_Y**2) / T_hat  # prior variance
T_cohort = 500  # time horizon to keep track of cohorts
Nt = int(T_cohort / dt)

MC = 1
fMAT = np.zeros((MC, Nt))

time_tolerance = 5