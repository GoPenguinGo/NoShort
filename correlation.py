import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.simulation import simulate_SI
from src.param import mu_Y, sigma_Y, \
    dt, Ninit, Nc, Nt, tau, \
    cutoffs_age, Ntype, alpha_i, \
    dZ_build_matrix, dZ_SI_build_matrix, dZ_SI_matrix, dZ_matrix, \
    cohort_size, rho_i, tax, beta0, beta_i, nu, Vhat, phi, Npre, \
    T_hat, rho_cohort_type, cohort_type_size
from concurrent.futures import ProcessPoolExecutor
from scipy.interpolate import make_interp_spline
import statsmodels.api as sm


mode_trade = "w_constraint"
mode_learn = 'reentry'

folder_address = r'C:\Users\A2010290\OneDrive - BI Norwegian Business School (BIEDU)\Documents\GitHub computer 2\NoShort/'
plt.rcParams["font.family"] = 'serif'



def simulate_path(
        i: int,
):
    print(i)
    # shocks
    dZ_build = dZ_build_matrix[i]
    dZ = dZ_matrix[i]
    dZ_SI_build = dZ_SI_build_matrix[i]
    dZ_SI = dZ_SI_matrix[i]

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
    ) = simulate_SI(mode_trade,
                    mode_learn,
                    Nc,
                    Nt,
                    dt,
                    nu,
                    Vhat,
                    mu_Y,
                    sigma_Y,
                    tax,
                    beta0,
                    phi,
                    Npre,
                    Ninit,
                    T_hat,
                    dZ_build,
                    dZ,
                    dZ_SI_build,
                    dZ_SI,
                    tau,
                    cutoffs_age,
                    Ntype,
                    rho_i,
                    alpha_i,
                    beta_i,
                    rho_cohort_type,
                    cohort_type_size,
                    need_f='False',
                    need_Delta='True',
                    need_pi='True',
                    )

    # regression: regressing change of belief on shocks, with a dummy about time and age
    gap = 12

    t_sample = np.arange(0, Nt, gap)
    c_sample = np.arange(0, Nc, 60)

    parti = pi > 0

    Delta_prior = Delta[:-gap, gap:]
    d_Delta = Delta[gap:, :-gap] - Delta_prior
    Parti_prior = parti[:-gap, gap:]
    d_parti = parti[gap:, :-gap] - parti_prior
    shocks = dZ[gap:]

    past_annual_return = np.zeros(Nt)

    past_annual_return[n, gap:] = (np.cumsum(dZ)[gap:] - np.cumsum(dZ)[:-gap]) / (gap / 12)
    past_annual_return[n, :gap] = np.cumsum(dR[:gap]) / (gap / 12)

    regression_table = np.zeros((len(c_sample), 4))
    for j in range(len(c_sample)):
        y = d_Delta[t_sample][:, c_sample][:, j]
        x0 = shocks[t_sample]
        x1 = Parti_prior[t_sample][:, c_sample][:, j]
        x0_x1 = x0 * x1
        x2 = Delta_prior[t_sample][:, c_sample][:, j]
        x = np.column_stack((x0, x1, x0_x1, x2))
        x_regress = sm.add_constant(x)
        model = sm.OLS(y, x_regress)
        est = model.fit()
        regression_table[j] = est.params[1:]

    print(np.average(regression_table[:, 2]))

    return (
        i,
        regression_table
    )


def main():
    with ProcessPoolExecutor(max_workers=12) as executor:  # Adjust the number of workers as needed
        results = [executor.submit(simulate_path, i) for i in range(36)]
        # Initialize a list to store the results
    results_list = []

    # Retrieve results from parallel processes
    for result in results:
        i, \
            regression_table = result.result()

        data = {
            "i": i,
            "regression_table": regression_table
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)
    results_dict = results_df.to_dict(orient='list')
    np.savez(folder_address +"corr.npz", **results_dict)


if __name__ == '__main__':
    main()




