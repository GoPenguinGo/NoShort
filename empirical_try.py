import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_mix_types
from src.param import nu, mu_Y, sigma_Y, \
    dt, Ninit, Nc, Nt, tau, \
    cutoffs_age, Ntype, alpha_i, \
    dZ_build_matrix, dZ_SI_build_matrix, dZ_SI_matrix, dZ_matrix, \
    cohort_size, rho_i
from src.param_mix import Nconstraint, rho_i_mix
from concurrent.futures import ProcessPoolExecutor
# from cupyx.scipy.interpolate import RBFInterpolator

# country_names = ['US', 'Finland', 'Germany', 'Norway']
country_names = [ 'US']
folder_address = r'C:/Users/A2010290/OneDrive - BI Norwegian Business School (BIEDU)/Documents/GitHub computer 2/NoShort/empirical/'
plt.rcParams["font.family"] = 'serif'

# (complete, excluded, disappointment, reentry)
density_set = [
    # (0.0, 0.0, 0.0, 1.0),
    # (0.0, 0.0, 1.0, 0.0),
    (0.25, 0.25, 0.25, 0.25),
    # (0.25, 0.25, 0.0, 0.5),
    # (0.25, 0.25, 0.5, 0.0),
    # (0.5, 0.25, 0.0, 0.25),
    # (0.1, 0.1, 0.0, 0.8),
    # (0.1, 0.1, 0.4, 0.4),
]
n_scenarios = len(density_set)
# T_hat_set = [1, 2, 5]  # Pre-learning window
# phi_set = [0.0, 0.2, 0.4]
# tax_set = [0.3, 0.4, 0.5]
T_hat_set = [2]  # Pre-learning window
phi_set = [0.0, 0.4]
tax_set = [0.5]
n_T_hat = len(T_hat_set)
n_phi = len(phi_set)
n_tax = len(tax_set)

# # for testing:
Mpath = 50
# Mpath = 1
np.seterr(invalid='ignore')
age_cutoffs_SCF = [int(Nt-1), int(Nt-1-12*15), int(Nt-1-12*35), int(Nt-1-12*55), 0]

# noinspection PyTypeChecker
def simulate_path(
        i: int,
        data_shocks,
        country: str,
):
    print(i)
    # shocks
    dZ_build = dZ_build_matrix[i]
    dZ_SI_build = dZ_SI_build_matrix[i]
    dZ = dZ_matrix[i]
    filler = np.random.randn(data_shocks['dZ_SI'].isna().sum()) * np.sqrt(dt)
    data_shocks.loc[data_shocks['dZ_SI'].isna(), 'dZ_SI'] = filler
    dZ_actual = data_shocks.to_numpy()[:, 0]
    dZ_SI_actual = data_shocks.to_numpy()[:, 1]
    Nt_data = dZ_actual.size
    dZ[-Nt_data:] = dZ_actual
    dZ_SI = dZ_SI_matrix[i]
    dZ_SI[-Nt_data:] = dZ_SI_actual

    popu_parti_compare = np.empty((n_T_hat, n_phi, n_tax, n_scenarios, Nt), dtype=np.float32)
    # popu_parti_old = np.empty((n_scenarios_short, Nt), dtype=np.float32)
    # popu_parti_young = np.empty((n_scenarios_short, Nt), dtype=np.float32)
    # r_compare = np.zeros((n_scenarios_short, Nt), dtype=np.float32)
    dR_compare = np.empty((n_T_hat, n_phi, n_tax, n_scenarios, Nt), dtype=np.float32)
    # mu_S_compare = np.zeros((n_scenarios_short, Nt), dtype=np.float32)
    # sigma_S_compare = np.zeros((n_scenarios_short, Nt), dtype=np.float32)
    # pd_compare = np.zeros((n_scenarios_short, Nt), dtype=np.float32)
    # average_belief_compare = np.zeros((n_scenarios_short, Nt), dtype=np.float32)
    # popu_reenter_compare = np.empty((n_T_hat, n_phi, n_tax, n_scenarios_short, 5, Nt), dtype=np.float32)
    # popu_exit_compare = np.empty((n_T_hat, n_phi, n_tax, n_scenarios_short, Nt), dtype=np.float32)

    parti_df = pd.DataFrame(data_shocks.index.astype(str), columns=['yyyymm'])
    for a, T_hat in enumerate(T_hat_set):
        # print(a)
        Npre = int(T_hat / dt)
        Vhat = (sigma_Y ** 2) / T_hat  # prior variance
        for b, phi in enumerate(phi_set):
            for c, tax in enumerate(tax_set):
                beta_i = (nu + rho_i) / (1 + tax)  # consumption wealth ratio
                beta0 = np.sum(alpha_i * beta_i).astype(float)
                for g, density in enumerate(density_set):
                    col_name = str(T_hat) + '_' + str(phi) + '_' + str(tax) + '_' + str(g)
                    alpha_constraint = np.ones(
                        (1, Nconstraint)) * density
                    alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
                    cohort_type_size_mix = cohort_size * alpha_i_mix
                    beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio
                    rho_cohort_type_mix = alpha_i_mix * beta_i_mix * np.exp(
                        -(rho_i_mix + nu) * tau)  # shape(2, 6000)
                    (
                        r,
                        theta,
                        f_c,
                        Delta,
                        pi,
                        parti,
                        Phi_bar_parti,
                        Phi_tilde_parti,
                        Delta_bar_parti,
                        Delta_tilde_parti,
                        dR,
                        mu_S,
                        sigma_S,
                        beta,
                        invest_tracker,
                        parti_age_group,
                        parti_wealth_group,
                        entry_mat,
                        exit_mat
                    ) = simulate_mix_types(Nc, Nt, dt, nu, Vhat, mu_Y, sigma_Y, tax,
                                           beta0,
                                           phi, Npre, Ninit, T_hat,
                                           dZ_build, dZ, dZ_SI_build, dZ_SI,
                                           cutoffs_age, Ntype,
                                           Nconstraint, rho_i_mix, alpha_i_mix, beta_i_mix,
                                           rho_cohort_type_mix,
                                           cohort_type_size_mix,
                                           need_f='True',
                                           need_Delta='True',
                                           need_pi='True',
                                           )
                    popu_parti_compare[a, b, c, g] = parti
                    dR_compare[a, b, c, g] = dR

                    parti_df['parti' + col_name] = parti[-Nt_data:].astype(np.float32)
                    if country == 'US' or 'Germany' or 'Norway':
                        age_parti = np.zeros((4, Nt_data))
                        age_belief = np.zeros((4, Nt_data))
                        for n in range(len(age_cutoffs_SCF) - 1):
                            age_parti[n] = np.average(
                                np.average(invest_tracker[-Nt_data:, :, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]],
                                           weights=cohort_size[0, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]], axis=2),
                                weights=density,
                                axis=1)
                            age_belief[n] = np.average(
                                np.average(Delta[-Nt_data:, :, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]],
                                           weights=cohort_size[0, age_cutoffs_SCF[n + 1]:age_cutoffs_SCF[n]], axis=2),
                                weights=density,
                                axis=1)
                        parti_df['belief_old' + col_name] = age_belief[3].astype(np.float32)
                        parti_df['belief_young' + col_name] = age_belief[0].astype(np.float32)
                        parti_df['parti_old' + col_name] = age_parti[3].astype(np.float32)
                        parti_df['parti_young' + col_name] = age_parti[0].astype(np.float32)

                    # parti_df['parti' + col_name] = parti_all.astype(np.float32)
                    if country == 'Norway' or 'Finland':
                        parti_df['entry' + col_name] = entry_mat[-Nt_data:].astype(np.float32)
                        parti_df['exit' + col_name] = exit_mat[-Nt_data:].astype(np.float32)

    parti_df.to_stata('stata_dataset/' + country + '/' + str(i) + '.dta')
    return (
        i,
        popu_parti_compare,
        dR_compare,
    )


def main():
    for country in country_names:
        # Create a ProcessPoolExecutor for parallel execution
        # run this on a grid of parameters & type densities & signal
        data_shocks = pd.read_excel(
            folder_address + r'realized_shocks_' + country + '.xlsx',
            sheet_name='Sheet1',
            index_col=0
        )
        with ProcessPoolExecutor(max_workers=12) as executor:  # Adjust the number of workers as needed
            results = [executor.submit(simulate_path, i, data_shocks, country) for i in range(Mpath)]
        # Initialize a list to store the results
        results_list = []

        # Retrieve results from parallel processes
        for result in results:
            i, \
            popu_parti_result, \
            dR_result = result.result()

            data = {
                "i": i,
                "parti rate": popu_parti_result,
                "dR": dR_result,
            }
            results_list.append(data)

        # Create a DataFrame from the list of dictionaries
        results_df = pd.DataFrame(results_list)

if __name__ == '__main__':
    main()
