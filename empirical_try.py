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
country_names = [ 'Germany']
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
T_hat_set = [1, 2, 5]  # Pre-learning window
phi_set = [0.0, 0.2, 0.4]
tax_set = [0.3, 0.4, 0.5]
n_T_hat = len(T_hat_set)
n_phi = len(phi_set)
n_tax = len(tax_set)

# # for testing:
# Mpath = 160
Mpath = 1
np.seterr(invalid='ignore')


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
                for g in range(n_scenarios):
                    col_name = str(T_hat) + '_' + str(phi) + '_' + str(tax) + '_' + str(g)
                    alpha_constraint = np.ones(
                        (1, Nconstraint)) * density_set[g]
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
                        entry,
                        exit
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
                    # average_belief = np.average(np.average(Delta, axis=2, weights=cohort_size[0]), axis=1,
                    #                             weights=alpha_constraint[0])
                    # r_compare[g] = r
                    popu_parti_compare[a, b, c, g] = parti
                    # popu_parti_old[g] = parti_age_group[:, 3]
                    # popu_parti_young[g] = parti_age_group[:, 0]
                    # average_belief_compare[g] = average_belief
                    dR_compare[a, b, c, g] = dR
                    # mu_S_compare[g] = mu_S
                    # sigma_S_compare[g] = sigma_S
                    # pd_compare[g] = 1 / beta

                    # parti_all = np.zeros(Nall)
                    # entry_all = np.zeros(Nall)
                    # exit_all = np.zeros(Nall)

                    # parti_all[-Nt:] = parti
                    # entry_all[-Nt:] = entry
                    # exit_all[-Nt:] = exit

                    # old vs. young: experience gap, belief gap, and participation rate gap
                    cohort_belief = np.average(Delta, axis=1, weights=alpha_constraint[0])
                    average_belief_old = np.average(cohort_belief[:, :cutoffs_age[3]], axis=1,
                                                    weights=cohort_size[0, :cutoffs_age[3]])
                    average_belief_young = np.average(cohort_belief[:, cutoffs_age[1]:], axis=1,
                                                      weights=cohort_size[0, cutoffs_age[1]:])
                    parti_old = parti_age_group[:, 3]
                    parti_young = parti_age_group[:, 0]
                    # parti_age_compare = np.zeros((2, N_sample))

                    parti_df['parti' + col_name] = parti[-Nt_data:].astype(np.float32)

                    if country == 'US' or 'Germany' or 'Norway':
                        parti_df['belief_old' + col_name] = average_belief_old[-Nt_data:].astype(np.float32)
                        parti_df['belief_young' + col_name] = average_belief_young[-Nt_data:].astype(np.float32)
                        parti_df['parti_old' + col_name] = parti_old[-Nt_data:].astype(np.float32)
                        parti_df['parti_young' + col_name] = parti_young[-Nt_data:].astype(np.float32)

                    # parti_df['parti' + col_name] = parti_all.astype(np.float32)
                    if country == 'Norway' or 'Finland':
                        parti_df['entry' + col_name] = entry[-Nt_data:].astype(np.float32)
                        parti_df['exit' + col_name] = exit[-Nt_data:].astype(np.float32)

    parti_df.to_stata('stata_dataset/' + country + '/' + str(i) + '.dta')
    return (
        i,
        popu_parti_compare,
        # popu_parti_old,
        # popu_parti_young,
        # r_compare,
        dR_compare,
        # mu_S_compare,
        # sigma_S_compare,
        # pd_compare,
        # average_belief_compare,
        # popu_reenter_compare,
        # popu_exit_compare,
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
            # popu_reentry_result, \
            # popu_exit_result
            # popu_parti_old_result, \
            # popu_parti_young_result, \
            # r_result, \
            # dR_result, \
            # mu_S_result, \
            # sigma_S_result, \
            # pd_result, \
            # average_belief_result,

            data = {
                "i": i,
                "parti rate": popu_parti_result,
                # "parti rate old": popu_parti_old_result,
                # "parti rate young": popu_parti_young_result,
                # "interest rate": r_result,
                "dR": dR_result,
                # "expected return": mu_S_result,
                # "expected vola": sigma_S_result,
                # "price dividend ratio": pd_result,
                # "average belief": average_belief_result,
                # "popu_reentry": popu_reentry_result,
                # "popu_exit": popu_exit_result,
            }
            results_list.append(data)

        # Create a DataFrame from the list of dictionaries
        results_df = pd.DataFrame(results_list)

if __name__ == '__main__':
    main()
