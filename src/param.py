import numpy as np

# Parameters
rho = 0.001  # Time discount factor
nu = 0.02  # Death rate
# nu = 0.01
# nu = 0.03
mu_Y = 0.02  # Growth rate of output
sigma_Y = 0.033  # Standard deviation of output
sigma_Y_sqr = sigma_Y ** 2
sigma_S = (
    sigma_Y  # In equilibrium the stock price diffusion is the same as output diffusion
)
# for the SI signal:
sigma_SI = 0.3
# phi = 0.3

v = 0.018  # from Nagel and Xu (2021 RFS)

# tax = 0.015  # marginal rate of wealth tax
# tax = 0.02
tax = 0.01
beta = rho + nu - tax  # marginal propensity to consume

# Setting prior variance
dt = 1 / 12  # time incremental
dt_root = np.sqrt(dt)
T_hat = 20  # Pre-trading period
Npre = int(T_hat / dt)
Vhat = (sigma_Y ** 2) / T_hat  # prior variance
Ninit = int(20 / dt)
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
cutoffs = [Nc, tau_cutoff1, tau_cutoff2, tau_cutoff3, 0]
n_age_groups = 4

Mpath = 10000

# for graphs:
t = np.arange(0, T_cohort, dt)
Tkeep = 100
Nkeep = int(Tkeep / dt)
Tsample = int(T_cohort / 100)
Nsamples = 500
stepcorr = int(Tsample / dt)

phi_vector = [0, 0.4, 0.8]
n_phi = len(phi_vector)

phi_5 = [0, 0.2, 0.4, 0.6, 0.8]
n_phi_5 = len(phi_5)

tax_vector = [0.008, 0.01, 0.012]
n_tax = len(tax_vector)


# for V_hat_experiment:
Npres_a = np.arange(1, 6, 1)
Npres_b = np.arange(6, 13, 3)
Npres_c = np.arange(24, 61, 12)
Npres_d = np.arange(84, 181, 24)
Npres_e = np.arange(204, 361, 48)
Npre_list = [Npres_b, Npres_c, Npres_d, Npres_e]
Npres = Npres_a
for i in Npre_list:
    Npres = np.append(Npres, i)


# labels:
red_labels = [r'Positive local trend in $z^Y$, ', r'Negative local trend in $z^Y$, ']
yellow_labels = [r'Positive local trend in $z^{SI}$ ', r'Negative local trend in $z^{SI}$ ']
cohort_labels = ['cohort 1', 'cohort 2', 'cohort 3']
scenario_labels = ['Complete', 'Reentry', 'Disappointment', 'Reentry, partial shorting', 'Disappointment, partial shorting']
colors_short = ['midnightblue', 'darkgreen', 'darkviolet', 'red']
colors_short2 = ['mediumblue', 'saddlebrown', 'darkmagenta']
PN_labels = ['Participant (P)', 'Nonparticipant (N)']
# age_labels = ['20 < Age <= 35, youngest quartile', '35 < Age <= 55', '55 < Age <= 89', 'Age > 89, oldest quartile']
age_labels = ['0 < Age <= 15, youngest quartile', '15 < Age <= 35', '35 < Age <= 69', 'Age > 69, oldest quartile']
# label_phi = []
# for i in range(n_phi_short):
#     label_phi.append(r'$\phi$ = ' + str(phi_vector_short[i]))

dZ_mat = np.random.randn(int(Mpath / 2 * Nt))
dZ_mat = np.reshape(np.append(dZ_mat, -dZ_mat), (-1, Nt)) * dt_root
dZ_SI_mat = np.random.randn(int(Mpath / 2 * Nt))
dZ_SI_mat = np.reshape(np.append(dZ_SI_mat, -dZ_mat), (-1, Nt)) * dt_root
dZ_build_mat = np.random.randn(int(Mpath / 2 * Nt))
dZ_build_mat = np.reshape(np.append(dZ_build_mat, -dZ_mat), (-1, Nt)) * dt_root
dZ_SI_build_mat = np.random.randn(int(Mpath / 2 * Nt))
dZ_SI_build_mat = np.reshape(np.append(dZ_SI_build_mat, -dZ_mat), (-1, Nt)) * dt_root
np.save('dZ_matrix', dZ_mat)
np.save('dZ_SI_matrix', dZ_SI_mat)
np.save('dZ_build_matrix', dZ_build_mat)
np.save('dZ_SI_build_matrix', dZ_SI_build_mat)

dZ_matrix = np.load('dZ_matrix.npy')
dZ_build_matrix = np.load('dZ_build_matrix.npy')
dZ_SI_matrix = np.load('dZ_SI_matrix.npy')
dZ_SI_build_matrix = np.load('dZ_SI_build_matrix.npy')

# the shocks in the time-series
dZ_build_case = np.load('dZ_build_case.npy')
dZ_SI_build_case = np.load('dZ_SI_build_case.npy')
dZ_Y_cases = np.load('Z_Y_cases.npy')
dZ_SI_cases = np.load('Z_SI_cases.npy')

top = 0.05
old_limit = 100

colors = ['mediumblue', 'orange', 'darkmagenta', 'red', 'gold', 'midnightblue', 'green', 'saddlebrown', 'darkgreen', 'firebrick', 'purple', 'blue',
          'olivedrab', 'darkviolet', 'pink', 'black', ]

# modes_trade = ['complete', 'w_constraint', 'partial_constraint_rich', 'partial_constraint_old']
modes_trade = ['complete', 'w_constraint', 'partial_constraint_old']
modes_learn = ['reentry', 'disappointment']
scenarios = []
for mode_trade in modes_trade:
    if mode_trade == 'complete':
        scenarios.append([mode_trade, 'reentry'])
    else:
        for mode_learn in modes_learn:
            scenarios.append([mode_trade, mode_learn])


