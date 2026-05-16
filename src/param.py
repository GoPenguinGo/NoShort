import numpy as np


# Parameters
Ntype = 2
rho_i = np.array([[0.001], [0.005]])  # baseline
alpha_i = np.ones((Ntype, 1)) * 1 / Ntype
nu = 0.02  # Death rate
mu_Y = 0.02  # Growth rate of output
sigma_Y = 0.033  # Standard deviation of output
sigma_Y_sqr = sigma_Y ** 2
sigma_SI = 0.3
phi = 0.5
tax = 0.35   # marginal rate of consumption tax
beta_i = (nu + rho_i) / (1 + tax)  # consumption wealth ratio
beta0 = np.sum(alpha_i * beta_i).astype(float)

entry_bound = 0.04
exit_bound = 0.01

# Setting prior variance
dt = 1 / 12  # time incremental
dt_root = np.sqrt(dt)
T_hat = 8  # Pre-trading period
Npre = int(T_hat / dt)
Vhat = (sigma_Y ** 2) / T_hat  # prior variance
Ninit = int(20 / dt)
T_cohort = 500  # time horizon to keep track of cohorts
Nt = int(T_cohort / dt)  # number of periods
Nc = int(T_cohort / dt)  # number of cohorts

# generate values that are fixed in the main loop
tau = np.reshape(np.arange(T_cohort, 0, -dt), (1, -1))  # age from 500 to 0; shape(1, 6000)
cohort_size = nu * np.exp(-nu * (tau - dt)) * dt  # cohort size when a new cohort is just born
cohort_type_size = cohort_size * alpha_i
rho_cohort_type = alpha_i * beta_i * np.exp(-(rho_i + nu) * tau)  # shape(2, 6000)
beta_cohort = np.sum(np.exp(-beta_i * tau) * alpha_i, axis=0)

# create age quartiles for analysis
cummu_popu = np.cumsum(cohort_size)
cutoffs_age_SCF = [int(Nt-1), int(Nt-1-15/dt), int(Nt-1-35/dt), int(Nt-1-50/dt), 0]  # SCF
# popu_age_groups = cummu_popu[cutoffs_age[:-1]] - cummu_popu[cutoffs_age[1:]]
cutoffs_age = [int(Nt-1), int(Nt-1-15/dt), int(Nt-1-35/dt), int(Nt-1-50/dt), 0]  # Michigan
popu_age_groups = cummu_popu[cutoffs_age[:-1]] - cummu_popu[cutoffs_age[1:]]

Mpath = 2000
N_workers = 20
t = np.arange(0, T_cohort, dt)

phi_vector = np.array([0.5, 1.0])
n_phi = len(phi_vector)

window_bell = 20

# labels:
cohort_labels = ['Cohort 1', 'Cohort 2', 'Cohort 3']
scenario_labels = ['Complete', 'Reentry', 'Mix']
colors_short = ['midnightblue', 'darkgreen', 'darkviolet', 'red']
colors_short2 = ['mediumblue', 'saddlebrown', 'darkmagenta']
PN_labels = ['Participant (P)', 'Nonparticipant (N)']
age_labels = [r'Experience $\leq$ 15', r'15 < Experience $\leq$ 35', r'35 < Experience $\leq$ 50', r'Experience > 50']

# dZ_mat1 = np.random.randn(int(Mpath / 2 * Nt)).astype(np.float16)
# dZ_mat = np.reshape(np.append(dZ_mat1, -dZ_mat1), (-1, Nt)) * dt_root
# dZ_SI_mat1 = np.random.randn(int(Mpath / 2 * Nt)).astype(np.float16)
# dZ_SI_mat = np.reshape(np.append(dZ_SI_mat1, -dZ_SI_mat1), (-1, Nt)) * dt_root
# dZ_build_mat1 = np.random.randn(int(Mpath / 2 * Nt)).astype(np.float16)
# dZ_build_mat = np.reshape(np.append(dZ_build_mat1, -dZ_build_mat1), (-1, Nt)) * dt_root
# dZ_SI_build_mat1 = np.random.randn(int(Mpath / 2 * Nt)).astype(np.float16)
# dZ_SI_build_mat = np.reshape(np.append(dZ_SI_build_mat1, -dZ_SI_build_mat1), (-1, Nt)) * dt_root
# np.save('shocks/dZ_matrix', dZ_mat)
# np.save('shocks/dZ_SI_matrix', dZ_SI_mat)
# np.save('shocks/dZ_build_matrix', dZ_build_mat)
# np.save('shocks/dZ_SI_build_matrix', dZ_SI_build_mat)

dZ_matrix = np.load('shocks/dZ_matrix.npy')
dZ_build_matrix = np.load('shocks/dZ_build_matrix.npy')
dZ_SI_matrix = np.load('shocks/dZ_SI_matrix.npy')
dZ_SI_build_matrix = np.load('shocks/dZ_SI_build_matrix.npy')

# the shocks in the time-series
dZ_build_case = np.load('shocks/dZ_build_case.npy')
dZ_SI_build_case = np.load('shocks/dZ_SI_build_case.npy')
dZ_Y_cases = np.load('shocks/Z_Y_cases.npy')
dZ_SI_cases = np.load('shocks/Z_SI_cases.npy')

# top_wealth = 0.05
# old_age_limit = 100

colors = ['mediumblue', 'orange', 'darkmagenta', 'red', 'gold', 'midnightblue', 'green', 'saddlebrown', 'darkgreen', 'firebrick', 'purple', 'blue',
          'olivedrab', 'darkviolet', 'pink', 'black', ]

modes_trade = ['complete', 'w_constraint', 'partial_constraint_old']
modes_learn = ['reentry', 'disappointment']
scenarios = []
for mode_trade in modes_trade:
    if mode_trade == 'complete':
        scenarios.append([mode_trade, 'reentry'])
    else:
        for mode_learn in modes_learn:
            scenarios.append([mode_trade, mode_learn])