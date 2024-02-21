import numpy as np
from src.param import Ntype, rho_i, alpha_i, nu, tax, cohort_size, tau

# Parameters
Nconstraint = 4
alpha_constraint = np.ones((1, Nconstraint)) * 1 / Nconstraint
alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
rho_i_mix = np.tile(np.reshape(rho_i, (-1, 1, 1)), (1, Nconstraint, 1))
beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio

# generate values that are fixed in the main loop
cohort_type_size_mix = cohort_size * alpha_i_mix
beta_cohort_type_mix = alpha_i_mix * np.exp(-beta_i_mix * tau)  # shape(2, 6000)
rho_cohort_type_mix = alpha_i_mix * np.exp(-(rho_i_mix + nu) * tau)  # shape(2, 6000)
beta_cohort_mix = np.sum(np.exp(-beta_i_mix * tau) * alpha_i_mix, axis=0)
