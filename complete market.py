import numpy as np
from tqdm import tqdm
from typing import Tuple
from src.param import (dZ_matrix, dZ_build_matrix, Nt, Nc, dt, rho_i,
                       alpha_i, beta0, nu, Vhat, mu_Y, sigma_Y, tax, phi, T_hat, Npre,
                       Ninit, Ntype, cohort_size, tau, Mpath)
from src.stats import post_var, dDelta_st_calculator
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


Nconstraint = 1
alpha_constraint = np.ones((1, Nconstraint)) * 1 / Nconstraint
# alpha_constraint = np.ones((1, Nconstraint)) * (0.5, 0.5, 0, 0)
alpha_i_mix = np.reshape(alpha_i * alpha_constraint, (Ntype, Nconstraint, 1))
rho_i_mix = np.tile(np.reshape(rho_i, (-1, 1, 1)), (1, Nconstraint, 1))
beta_i_mix = (nu + rho_i_mix) / (1 + tax)  # consumption wealth ratio

# generate values that are fixed in the main loop
cohort_type_size_mix = cohort_size * alpha_i_mix
beta_cohort_type_mix = alpha_i_mix * np.exp(-beta_i_mix * tau)  # shape(2, 6000)
rho_cohort_type = alpha_i_mix * beta_i_mix * np.exp(-(rho_i_mix + nu) * tau)  # shape(2, 6000)
beta_cohort_mix = np.sum(np.exp(-beta_i_mix * tau) * alpha_i_mix, axis=0)



def build_cohorts_mix_type(
    i_path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    dZ_build = dZ_build_matrix[i_path]

    Delta_s_t = np.zeros((Ntype, Nconstraint, 1), dtype=np.float32)
    d_eta_st = np.zeros((Ntype, Nconstraint, 1), dtype=np.float32)
    X = np.ones((1, 1, 1))
    eta_st_eta_ss_init = np.ones((Ntype, Nconstraint, 1))
    eta_st_eta_ss = eta_st_eta_ss_init
    tau_info = np.ones((Ntype, Nconstraint, 1)) * dt
    Vhat_init = np.ones((Ntype, Nconstraint, 1)) * Vhat
    Vhat_init[:, 0] = 0.0
    Vhat_vector = np.copy(Vhat_init)
    sigma_Y_sq = sigma_Y ** 2

    for i in tqdm(range(1, Nc)):
        # new cohort born (age 0), get wealth transfer, observe, invest
        rho_cohort_type_short = rho_cohort_type[:, :, -i:]
        dZ_build_t = dZ_build[i - 1]

        eta_st_eta_ss = eta_st_eta_ss * np.exp(
            (-0.5 * d_eta_st ** 2) * dt
            + d_eta_st * dZ_build_t
        )  # equation (15)

        # from equation (20) and the description below it
        # X_t = W_t * xi_t, is the sum of tax * X_s * eta_st_eta_ss * rho_cohort_type_short * dt, s<t;
        # X is the collection of all X_s, s<t.
        X_parts = tax * X * eta_st_eta_ss * rho_cohort_type_short * dt
        X_t = np.sum(X_parts) / (1 - tax * beta0 * dt)  # dividing by (1-tax*dt) keeps sum(f_st*dt) at 1

        eta_st_eta_ss = np.append(eta_st_eta_ss, eta_st_eta_ss_init, axis=2)
        X = np.append(X, np.ones((1, 1, 1)) * X_t, axis=2)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort
        # eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
        #  eta_bar_t is the denominator; it creates issues if too close to 0
        #  so we rescale eta_bar to keep it away from 0, without changing f_st

        f_c_ist = X_parts / X_t / dt
        f_c_ist = np.append(f_c_ist, tax * alpha_i_mix * beta_i_mix, axis=2)

        # update beliefs
        V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, phi, 'P')
        dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, phi, dt, V_st_P, Delta_s_t, dZ_build_t, 'P')

        # add a new cohort to Vhat_vector and tau_info
        Vhat_vector = np.append(Vhat_vector, Vhat_init, axis=2)
        tau_info = np.append(tau_info, np.zeros((Ntype, Nconstraint, 1)), axis=2) + dt

        if i < Npre:
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(Delta_s_t, np.zeros((Ntype, Nconstraint, 1)), axis=2)  # newborns begin with 0 bias when there are not enough observations
        else:
            init_bias = np.average(dZ_build[int(i - Npre): i]) / dt * np.ones((Ntype, Nconstraint, 1))
            init_bias[:, 0] = 0.0
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(
                Delta_s_t, init_bias, axis=2
            )  # newborns begin with Npre observations of the dividend process

        # find the market clearing theta, given beliefs and consumption shares
        if i < Ninit:  # Ninit: initial rounds where the short-sale constraint is relaxed
            d_eta_st = (
                Delta_s_t  # relax the short-sale constraint in the beginning
            )

        else:
            theta_t = sigma_Y - np.average(Delta_s_t, weights=f_c_ist)
            d_eta_st = Delta_s_t + theta_t


    return (
        Delta_s_t,
        eta_st_eta_ss,
        X,
        d_eta_st,
        tau_info,
        Vhat_vector,
    )



def simulate_path_complete(
        i_path: int
) -> tuple:

    dZ = dZ_matrix[i_path]

    # Initializing variables
    (Delta_s_t,
    eta_st_eta_ss,
    X,
    d_eta_st,
    tau_info,
    Vhat_vector
     ) = build_cohorts_mix_type(
        i_path
    )

    biasvec = dZ_build_matrix[i_path]

    keep_when = int(200 / dt)
    sigma_Y_sq = sigma_Y ** 2
    mu_S_t = 0
    sigma_S_t = 0

    dR = np.zeros(Nt - keep_when)  # stores stock returns
    r = np.zeros(Nt - keep_when)  # interest rate
    theta = np.zeros(Nt - keep_when)  # market price of risk
    mu_S = np.zeros(Nt - keep_when)
    sigma_S = np.zeros(Nt - keep_when)
    beta = np.zeros(Nt - keep_when)
    Delta_bar_parti = np.zeros(
        (Nt - keep_when))  # consumption weighted estimation error of the stock market participants
    Delta_tilde_parti = np.zeros((Nt - keep_when))  # wealth weighted estimation error of the stock market participants

    append_init = np.ones((Ntype, Nconstraint, 1))

    for i in tqdm(range(Nt)):
        dZ_t = dZ[i]

        # new cohort born (age 0), get wealth transfer, observe, invest
        eta_st_eta_ss = eta_st_eta_ss * np.exp(
            (-0.5 * d_eta_st ** 2) * dt
            + d_eta_st * dZ_t
        )  # equation (15)

        X_parts = tax * X * eta_st_eta_ss * rho_cohort_type * dt
        X_t = np.sum(X_parts) / (1 - tax * beta0 * dt)

        eta_st_eta_ss = np.append(eta_st_eta_ss[:, :, 1:], append_init, axis=2)
        X = np.append(X[:, :, 1:], X_t * np.ones((1, 1, 1)), axis=2)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort
        # eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
        #  eta_bar_t is the denominator; it creates issues if too close to 0
        #  so we rescale eta_bar to keep it away from 0, without changing f_st

        f_c_ist = X_parts / X_t / dt
        f_c_ist = np.append(f_c_ist[:, :, 1:], tax * alpha_i_mix * beta_i_mix, axis=2)

        beta_t = 1 / np.sum(f_c_ist / beta_i_mix * dt)
        f_w_ist = f_c_ist / beta_i_mix * beta_t
        if i > 0:
            dR_t = mu_S_t * dt + sigma_S_t * dZ_t
        else:
            dR_t = 0

        # update beliefs
        V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, phi, 'P')
        dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, phi, dt, V_st_P, Delta_s_t, dZ_t, 'P')
        Vhat_vector = np.append(Vhat_vector[:, :, 1:], Vhat * append_init, axis=2)

        if i < Npre - 1:
            init_bias = (np.sum(biasvec[i + 1:]) + np.sum(dZ[:i + 1])) / T_hat * append_init
        else:
            init_bias = np.sum(dZ[i + 1 - Npre:i + 1]) / T_hat * append_init
        init_bias[:, 0] = 0.0

        Delta_s_t = Delta_s_t[:, :, 1:] + dDelta_s_t[:, :, 1:]
        Delta_s_t = np.append(Delta_s_t, init_bias, axis=2)

        theta_t = sigma_Y - np.average(Delta_s_t, weights=f_c_ist)
        d_eta_st = Delta_s_t + theta_t

        Delta_bar_parti_t = np.average(Delta_s_t, weights=f_c_ist)
        Delta_tilde_parti_t = np.average(Delta_s_t, weights=f_w_ist)
        sigma_S_t = (theta_t + Delta_tilde_parti_t)
        rho_bar_t = np.sum(rho_i_mix * f_c_ist) / np.sum(f_c_ist)

        r_t = (
                nu - tax * beta0
                + rho_bar_t
                + mu_Y
                - sigma_Y * theta_t
        )

        mu_S_t = sigma_S_t * theta_t + r_t

        # store the results, only the aggregate values
        if i >= keep_when:  # only keeping the data after 200 years in the simulation
            ii = i - keep_when
            dR[ii] = dR_t  # realized return from t-1 to t
            theta[ii] = theta_t
            r[ii] = r_t
            Delta_bar_parti[ii] = Delta_bar_parti_t
            Delta_tilde_parti[ii] = Delta_tilde_parti_t
            mu_S[ii] = mu_S_t
            sigma_S[ii] = np.abs(sigma_S_t)  # stock vola = absolute value of sigma
            beta[ii] = beta_t

    # save the mean and standard deviation
    data_mean_vola = {
        'theta': np.array([np.mean(theta), np.std(theta)]),
        'r': np.array([np.mean(r), np.std(r)]),
        'mu_S': np.array([np.mean(mu_S), np.std(mu_S)]),
        'sigma_S': np.array([np.mean(sigma_S), np.std(sigma_S)]),
        'Delta_bar': np.array([np.mean(Delta_bar_parti), np.std(Delta_bar_parti)]),
        'Delta_tilde': np.array([np.mean(Delta_tilde_parti), np.std(Delta_tilde_parti)]),
    }
    table_mean_vola = pd.DataFrame(data_mean_vola, index=['Mean', 'Std_Dev'])


    return (
        i_path,
        table_mean_vola,
    )


def main():
    # Create a ProcessPoolExecutor for parallel execution
    with ProcessPoolExecutor(max_workers=20) as executor:  # Adjust the number of workers as needed
        results = [executor.submit(simulate_path_complete, i) for i in range(Mpath)]
    # Initialize a list to store the results
    results_list = []

    # Retrieve results from parallel processes
    for result in results:
        i, \
            table_mean_vola = result.result()

        data = {
            "i": i,
            "table_mean_vola": table_mean_vola,
        }
        results_list.append(data)

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)
    results_dict = results_df.to_dict(orient='list')
    np.savez(r'simu_results/simulation_complete_market.npz', **results_dict)


if __name__ == '__main__':
    main()
