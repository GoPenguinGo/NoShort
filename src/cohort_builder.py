import numpy as np
from src.solver import bisection, solve_theta, bisection_partial_constraint, solve_theta_partial_constraint
from tqdm import tqdm
from typing import Tuple
from src.stats import post_var, dDelta_st_calculator

def build_cohorts_SI(
    dZ_build: np.ndarray,
    dZ_SI_build: np.ndarray,
    Nc: int,
    dt: float,
    tau: np.ndarray,
    Ntype: int,
    beta_i: np.ndarray,
    alpha_i: np.ndarray,
    beta_cohort_type: np.ndarray,
    Vhat: float,
    sigma_Y: float,
    tax: float,
    phi: float,
    Npre: int,
    Ninit: int,
    mode_trade: str,
    mode_learn: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """builds up a sufficiently large set of cohorts in the economy, view each cohort as one agent with a constantly shrinking size

    Args:
        dZ_build (np.ndarray): random shocks of aggregate output for each period, shape (Nc-1, )
        Nc (int): number of periods  = number of cohorts in the economy
        dt (float): unit of time
        rho (float): rho, discount factor
        nu (float): birth / death rate, each cohort starts at size nu and shrinks at speed of nu
        Vhat (float): initial variance of beliefs
        mu_Y (float): mean of aggregate output growth
        sigma_Y (float): sd of aggregate output growth
        tax (float): marginal rate of wealth tax
        Npre (int): pre-trading periods
        T_hat (float): pre-trading years
        mode (str): describes the mode

    Returns:
        f_st (np.ndarray): consumption shares
        Delta_s_t (np.ndarray): bias, shape(Nc, )
        # eta_st_ss (np.ndarray): consumption change process, shape(Nc, )
        # eta_bar (np.ndarray): consumption weighted disagreement, shape(Nc, )
        d_eta_st (np.ndarray): max(delta_s_t, -theta_t), shape(Nc, )
        invest_tracker (np.ndarray): track if a cohort is still in the risky market
    """

    # size of matrix: type * cohort; or type * 1; or 1 * cohort
    Delta_s_t = np.zeros((Ntype, 1))  # belief bias, eq(3)
    d_eta_st = np.zeros((Ntype, 1))  # disagreement, eq(11)
    X = np.ones((Ntype, 1))
    eta_st_eta_ss_init = np.ones((Ntype, 1))
    eta_st_eta_ss = eta_st_eta_ss_init
    invest_tracker = np.ones((Ntype, Ninit)) if mode_trade != 'complete' else np.ones((Ntype, Nc))
    can_short_tracker = np.zeros((Ntype, Ninit)) if mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old' else np.zeros((Ntype, Nc))
    tau_info = np.ones((Ntype, 1)) * dt
    Vhat_init = np.ones((Ntype, 1)) * Vhat
    Vhat_vector = Vhat_init
    a_phi = (1 - phi ** 2)
    phi_sqr_a_phi = phi / np.sqrt(a_phi)
    a_phi_1 = 1 / a_phi
    sigma_Y_sq = sigma_Y ** 2
    # theta_mat = np.empty(Nc)

    for i in tqdm(range(1, Nc)):
    #for i in tqdm(range(1, Ninit)):
        # new cohort born (age 0), get wealth transfer, observe, invest
        tau_short = tau[:, -i:]  # shape(1, i)
        beta_cohort_type_short = beta_cohort_type[:, -i:]
        dZ_build_t = dZ_build[i - 1]
        dZ_SI_build_t = dZ_SI_build[i - 1]

        eta_st_eta_ss = eta_st_eta_ss * np.exp(
            (-0.5 * d_eta_st ** 2) * dt
            + d_eta_st * dZ_build_t
        )  # equation (11)

        X_parts = tax * np.exp(-tax * tau_short) * X * beta_cohort_type_short * eta_st_eta_ss * dt    # equation (18)
        X_t = np.sum(X_parts) / (1 - tax * dt)  # equation (18)  # dividing by (1-tax*dt) keeps sum(f_st*dt) at 1
        # eta_bar_t = np.sum(eta_bar_parts)

        eta_st_eta_ss = np.append(eta_st_eta_ss, eta_st_eta_ss_init, axis=1)
        X = np.append(X, np.ones((Ntype, 1)) * X_t, axis=1)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort
        # todo: eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
        #  eta_bar_t is the denominator; it creates issues if too close to 0
        #  so I rescale eta_bar to keep it away from 0, without changing f_st

        f_w_ist = X_parts / X_t / dt
        f_w_ist = np.append(f_w_ist, tax * alpha_i, axis=1)

        beta_t = np.sum(f_w_ist * beta_i) * dt
        f_c_ist = f_w_ist * beta_i / beta_t

        # update beliefs
        # todo: rewrite the functions to allow matrix calculation
        if mode_trade == 'complete':
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'P')
        elif mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
            if i < Ninit:
                V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
                dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'P')
            else:
                V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')
                dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'N')  # from eq(5)
                V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
                dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'P')  # from eq(8)
                dDelta_s_t = invest_tracker * dDelta_s_t_P + (1 - invest_tracker) * dDelta_s_t_N

        else:
            print('mode_trade not found')
            exit()


        # add a new cohort to Vhat_vector and tau_info
        Vhat_vector = np.append(Vhat_vector, Vhat_init, axis=1)
        if mode_trade == 'complete':  # where tau_info is the same with age; no switch between N and P for complete market
            tau_info = tau[:, -i - 1:]
        else:  # where tau_info is the time distance from the latest state switch
            tau_info = np.append(tau_info, np.zeros((Ntype, 1)), axis=1) + dt

        if i < Npre:
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(Delta_s_t, np.zeros((Ntype, 1)), axis=1)  # newborns begin with 0 bias when there are not enough earlier observations
        else:
            init_bias = np.average(dZ_build[int(i - Npre):i]) / dt
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(
                Delta_s_t, init_bias * np.zeros((Ntype, 1)), axis=1
            )  # newborns begin with Npre earlier observations of the dividend process

        # find the market clearing theta, given beliefs and consumption shares
        if i < Ninit or mode_trade == 'complete':  # Ninit: initial rounds where the short-sale constraint is relaxed
            d_eta_st = (
                Delta_s_t  # relax the short-sale constraint in the beginning
            )

        elif mode_trade == 'w_constraint':
            invest_tracker = np.append(invest_tracker, np.ones((Ntype, 1)), axis=1)  # indicator of current type, =1 for a cohort if type == P

            if mode_learn == 'disappointment':  # agents switch from type P to type N once constrained, and stay as type N
                possible_cons_share = f_c_ist * dt * invest_tracker
                possible_delta_st = Delta_s_t * invest_tracker
                lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound
                theta_t = bisection(
                    solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
                )
                a = Delta_s_t + theta_t
                invest = (a > 0)
                switch_P_to_N = invest_tracker * (1 - invest)
                invest_tracker = invest * invest_tracker
                d_eta_st = a * invest_tracker - theta_t
                # tau_info and V_hat has to change for the agents who switched to N
                Vhat_vector = np.append(V_st_P, Vhat * np.ones((Ntype, 1)), axis=1) * switch_P_to_N + Vhat_vector * (1 - switch_P_to_N)  # reset initial variance
                tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock

            elif mode_learn == 'reentry':   # agents switch between type P and type N
                possible_cons_share = f_c_ist * dt
                possible_delta_st = Delta_s_t
                lowest_bound = -np.max(possible_delta_st)  # absolute lower bound
                theta_t = bisection(
                    solve_theta, lowest_bound, 50, possible_cons_share, possible_delta_st, sigma_Y
                )
                a = Delta_s_t + theta_t
                invest = (a > 0)
                switch_P_to_N = invest_tracker * (1 - invest)
                switch_N_to_P = np.maximum(invest - invest_tracker, 0)
                switch = switch_N_to_P + switch_P_to_N
                invest_tracker = invest
                d_eta_st = a * invest_tracker - theta_t

                # tau_info and V_hat has to change for the agents who switched to N
                Vhat_vector = np.append(V_st_P, Vhat * np.ones((Ntype, 1)), axis=1) * switch_P_to_N + np.append(V_st_N, Vhat * np.ones((Ntype, 1)), axis=1) * switch_N_to_P + Vhat_vector * (1 - switch)  # reset initial variance
                tau_info = dt * switch + tau_info * (1 - switch)  # reset clock

            else:
                print('mode_learn not found')
                break

        # elif mode_trade == 'partial_constraint_rich':
        #     invest_tracker = np.append(invest_tracker, 1)  # indicator of current type, =1 for a cohort if type == P, by default type == P as newborn
        #     can_short_tracker = np.append(can_short_tracker, 0)
        #     cohort_size_short = cohort_size[-i-1:]  # todo: think about cohort size in the building function
        #
        #     if mode_learn == 'disappointment':  # agents switch from type P to type N once constrained, and stay as type N
        #         possible_cons_share = f_st * dt * invest_tracker
        #         possible_delta_st = Delta_s_t * invest_tracker
        #         indiv_w_possible = possible_cons_share / cohort_size_short
        #         cohort_size_possible = cohort_size_short * invest_tracker
        #         wealth_cutoff = find_the_rich(indiv_w_possible, cohort_size_possible,
        #                                       top)  # find the cohorts that make the richest 1% pupolation in the current period that are still in the market
        #         can_short = indiv_w_possible >= wealth_cutoff  # these cohorts can short in this period
        #         can_short_tracker = (can_short_tracker + can_short >= 1)
        #
        #         lowest_bound = -np.max(possible_delta_st)  # absolute lower bound
        #         theta_t = bisection_partial_constraint(
        #             solve_theta_partial_constraint, -50, 50, can_short_tracker, possible_delta_st, possible_cons_share,
        #             sigma_Y
        #         )
        #         a = Delta_s_t + theta_t
        #         invest = 1 - (a < 0) * (can_short_tracker < 1)
        #         switch_P_to_N = invest_tracker * (1 - invest)
        #         invest_tracker = invest * invest_tracker
        #         d_eta_st = a * invest_tracker - theta_t
        #
        #         # tau_info and V_hat has to change for the agents who switched to N
        #         Vhat_vector = np.append(V_st_P, Vhat) * switch_P_to_N + Vhat_vector * (
        #                 1 - switch_P_to_N)  # reset initial variance
        #         tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock
        #         # print(np.sum(can_short_tracker * cohort_size_short))
        #
        #     elif mode_learn == 'reentry':  # agents switch between type P and type N
        #         possible_cons_share = f_st * dt
        #         possible_delta_st = Delta_s_t
        #         indiv_w_possible = possible_cons_share / cohort_size_short
        #         cohort_size_possible = cohort_size_short
        #         wealth_cutoff = find_the_rich(indiv_w_possible, cohort_size_possible, top)  # find the cohorts that make the richest 1% pupolation in the current period that are still in the market
        #         can_short = indiv_w_possible >= wealth_cutoff  # these cohorts can short in this period
        #         can_short_tracker = (can_short_tracker + can_short >= 1)
        #
        #         lowest_bound = -np.max(possible_delta_st)  # absolute lower bound
        #         theta_t = bisection_partial_constraint(
        #             solve_theta_partial_constraint, -50, 50, can_short_tracker, possible_delta_st, possible_cons_share,
        #             sigma_Y
        #         )
        #         a = Delta_s_t + theta_t
        #         invest = 1 - (a < 0) * (can_short_tracker < 1)
        #         switch_P_to_N = invest_tracker * (1 - invest)
        #         switch_N_to_P = np.maximum(invest - invest_tracker, 0)
        #         switch = switch_N_to_P + switch_P_to_N
        #         invest_tracker = invest
        #         d_eta_st = a * invest_tracker - theta_t
        #
        #         # tau_info and V_hat has to change for the agents who switched to N
        #         Vhat_vector = np.append(V_st_P, Vhat) * switch_P_to_N + np.append(V_st_N,
        #                           Vhat) * switch_N_to_P + Vhat_vector * (
        #                           1 - switch)  # reset initial variance
        #         tau_info = dt * switch + tau_info * (1 - switch)  # reset clock
        #         # print(np.sum(can_short_tracker * cohort_size_short))
        #
        #     else:
        #         print('mode_learn not found')
        #         break

        # elif mode_trade == 'partial_constraint_old':
        #     invest_tracker = np.append(invest_tracker, 1)  # indicator of current type, =1 for a cohort if type == P, by default type == P as newborn
        #     # can_short_tracker = np.append(can_short_tracker, 0)
        #     # cohort_size_short = cohort_size[-i-1:]
        #     tau_short_1 = tau[-i-1:]
        #
        #     if mode_learn == 'disappointment':  # agents switch from type P to type N once constrained, and stay as type N
        #         # cohorts older than 100 years can short, if they are still type P by the time
        #         possible_cons_share = f_st * dt * invest_tracker
        #         possible_delta_st = Delta_s_t * invest_tracker
        #         can_short_possible = (tau_short_1 >= old_limit)
        #         can_short_tracker = can_short_possible * invest_tracker
        #
        #         lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound
        #         theta_t = bisection_partial_constraint(
        #             solve_theta_partial_constraint, lowest_bound, 50, can_short_tracker, possible_delta_st, possible_cons_share,
        #             sigma_Y
        #         )
        #         a = Delta_s_t + theta_t
        #         invest = 1 - (a < 0) * (can_short_tracker < 1)
        #         switch_P_to_N = invest_tracker * (1 - invest)
        #         invest_tracker = invest * invest_tracker
        #         d_eta_st = a * invest_tracker - theta_t
        #
        #         # tau_info and V_hat has to change for the agents who switched to N
        #         Vhat_vector = np.append(V_st_P, Vhat) * switch_P_to_N + Vhat_vector * (
        #                 1 - switch_P_to_N)  # reset initial variance
        #         tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock
        #         # print(np.sum(can_short_tracker * cohort_size_short))
        #
        #     elif mode_learn == 'reentry':  # agents switch between type P and type N
        #         possible_cons_share = f_st * dt
        #         possible_delta_st = Delta_s_t
        #         can_short_tracker = (tau_short_1 >= old_limit)
        #
        #         lowest_bound = -np.max(possible_delta_st)  # absolute lower bound
        #         theta_t = bisection_partial_constraint(
        #             solve_theta_partial_constraint, lowest_bound, 50, can_short_tracker, possible_delta_st, possible_cons_share,
        #             sigma_Y
        #         )
        #         a = Delta_s_t + theta_t
        #         invest = 1 - (a < 0) * (can_short_tracker < 1)
        #         switch_P_to_N = invest_tracker * (1 - invest)
        #         switch_N_to_P = np.maximum(invest - invest_tracker, 0)
        #         switch = switch_N_to_P + switch_P_to_N
        #         invest_tracker = invest
        #         d_eta_st = a * invest_tracker - theta_t
        #
        #         # tau_info and V_hat has to change for the agents who switched to N
        #         Vhat_vector = np.append(V_st_P, Vhat) * switch_P_to_N + np.append(V_st_N,
        #                           Vhat) * switch_N_to_P + Vhat_vector * (
        #                           1 - switch)  # reset initial variance
        #         tau_info = dt * switch + tau_info * (1 - switch)  # reset clock
        #         # print(np.sum(can_short_tracker * cohort_size_short))
        #
        #     else:
        #         print('mode_learn not found')
        #         break

        else:
            print('mode_trade not found')
            break
        # theta_mat[i] = theta_t

    #print(theta_mat)

    return (
        Delta_s_t,
        eta_st_eta_ss,
        X,
        d_eta_st,
        invest_tracker,
        tau_info,
        Vhat_vector,
        can_short_tracker,
    )


def build_cohorts_mix_type(
    dZ_build: np.ndarray,
    dZ_SI_build: np.ndarray,
    Nc: int,
    dt: float,
    tau: np.ndarray,
    Ntype: int,
    Nconstraint: int,
    beta_i: np.ndarray,
    alpha_i: np.ndarray,
    beta_cohort_type: np.ndarray,
    Vhat: float,
    sigma_Y: float,
    tax: float,
    phi: float,
    Npre: int,
    Ninit: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """builds up a sufficiently large set of cohorts in the economy, view each cohort as one agent with a constantly shrinking size
    a mixture of 4 different types of agents in each cohort:
        unconstrained;
        excluded;
        disappointment;
        & reentry.

    Args:
        dZ_build (np.ndarray): random shocks of aggregate output for each period, shape (Nc-1, )
        Nc (int): number of periods  = number of cohorts in the economy
        dt (float): unit of time
        rho (float): rho, discount factor
        nu (float): birth / death rate, each cohort starts at size nu and shrinks at speed of nu
        Vhat (float): initial variance of beliefs
        mu_Y (float): mean of aggregate output growth
        sigma_Y (float): sd of aggregate output growth
        tax (float): marginal rate of wealth tax
        Npre (int): pre-trading periods
        T_hat (float): pre-trading years
        mode (str): describes the mode

    Returns:
        f_st (np.ndarray): consumption shares
        Delta_s_t (np.ndarray): bias, shape(Nc, )
        # eta_st_ss (np.ndarray): consumption change process, shape(Nc, )
        # eta_bar (np.ndarray): consumption weighted disagreement, shape(Nc, )
        d_eta_st (np.ndarray): max(delta_s_t, -theta_t), shape(Nc, )
        invest_tracker (np.ndarray): track if a cohort is still in the risky market
    """

    # size of matrix: type * cohort; or type * 1; or 1 * cohort
    Delta_s_t = np.zeros((Ntype, Nconstraint, 1))  # belief bias, eq(3)
    d_eta_st = np.zeros((Ntype, Nconstraint, 1))  # disagreement, eq(11)
    X = np.ones((Ntype, Nconstraint, 1))
    eta_st_eta_ss_init = np.ones((Ntype, Nconstraint, 1))
    eta_st_eta_ss = eta_st_eta_ss_init
    invest_tracker = np.ones((Ntype, Nconstraint, Ninit))
    invest_newborn = np.array([[[1], [0], [1], [1]]]) * np.ones((Ntype, Nconstraint, 1))
    can_short_tracker = np.ones((Ntype, Nconstraint, Ninit))
    can_short_newborn = np.array([[[1], [0], [0], [0]]]) * np.ones((Ntype, Nconstraint, 1))
    tau_info = np.ones((Ntype, Nconstraint, 1)) * dt
    Vhat_init = np.ones((Ntype, Nconstraint, 1)) * Vhat
    Vhat_vector = Vhat_init
    a_phi = (1 - phi ** 2)
    phi_sqr_a_phi = phi / np.sqrt(a_phi)
    a_phi_1 = 1 / a_phi
    sigma_Y_sq = sigma_Y ** 2

    for i in tqdm(range(1, Nc)):
    #for i in tqdm(range(1, Ninit)):
        # new cohort born (age 0), get wealth transfer, observe, invest
        tau_short = tau[:, -i:]  # shape(1, i)
        beta_cohort_type_short = beta_cohort_type[:, :, -i:]
        dZ_build_t = dZ_build[i - 1]
        dZ_SI_build_t = dZ_SI_build[i - 1]

        eta_st_eta_ss = eta_st_eta_ss * np.exp(
            (-0.5 * d_eta_st ** 2) * dt
            + d_eta_st * dZ_build_t
        )  # equation (11)

        X_parts = tax * np.exp(-tax * tau_short) * X * beta_cohort_type_short * eta_st_eta_ss * dt    # equation (18)
        X_t = np.sum(X_parts) / ( 1 - tax * dt)  # equation (18)  # dividing by (1-tax*dt) keeps sum(f_st*dt) at 1
        # eta_bar_t = np.sum(eta_bar_parts)

        eta_st_eta_ss = np.append(eta_st_eta_ss, eta_st_eta_ss_init, axis=2)
        X = np.append(X, eta_st_eta_ss_init * X_t, axis=2)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort
        # todo: eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
        #  eta_bar_t is the denominator; it creates issues if too close to 0
        #  so I rescale eta_bar to keep it away from 0, without changing f_st

        f_w_ist = X_parts / X_t / dt
        f_w_ist = np.append(f_w_ist, tax * eta_st_eta_ss_init * alpha_i, axis=2)

        beta_t = np.sum(f_w_ist * beta_i) * dt
        f_c_ist = f_w_ist * beta_i / beta_t

        # update beliefs
        if i < Ninit:
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'P')
        else:
            V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')
            dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'N')  # from eq(5)
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'P')  # from eq(8)
            dDelta_s_t = invest_tracker * dDelta_s_t_P + (1 - invest_tracker) * dDelta_s_t_N


        # add a new cohort to Vhat_vector and tau_info
        Vhat_vector = np.append(Vhat_vector, Vhat_init, axis=2)
        tau_info = np.append(tau_info, np.zeros((Ntype, Nconstraint, 1)), axis=2) + dt

        if i < Npre:
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(Delta_s_t, np.zeros((Ntype, Nconstraint, 1)), axis=2)  # newborns begin with 0 bias when there are not enough earlier observations
        else:
            init_bias = np.average(dZ_build[int(i - Npre): i]) / dt
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(
                Delta_s_t, init_bias * np.zeros((Ntype, Nconstraint, 1)), axis=2
            )  # newborns begin with Npre earlier observations of the dividend process

        # find the market clearing theta, given beliefs and consumption shares
        if i < Ninit:  # Ninit: initial rounds where the short-sale constraint is relaxed
            d_eta_st = (
                Delta_s_t  # relax the short-sale constraint in the beginning
            )

        else:
            invest_tracker = np.append(invest_tracker, invest_newborn, axis=2)  # indicator of current type, =1 for a cohort if type == P
            can_short_tracker = np.append(can_short_tracker, can_short_newborn, axis=2)

            possible_cons_share = f_c_ist * dt * invest_tracker
            possible_delta_st = Delta_s_t * invest_tracker

            lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound
            theta_t = bisection_partial_constraint(
                    solve_theta_partial_constraint, lowest_bound, 50, can_short_tracker, possible_delta_st, possible_cons_share,
                    sigma_Y
                )
            a = Delta_s_t + theta_t
            invest = 1 - (a < 0) * (can_short_tracker < 1)  # not invest if a<0 and can not short
            invest[:, 1] = 0  # exclusion type
            switch_P_to_N = invest_tracker * (1 - invest) * (can_short_tracker < 1) # switch to nonparti if type R&E & not investing this period
            switch_N_to_P = np.maximum(invest - invest_tracker, 0)  # switch to parti if not investing before & investing this period
            switch_N_to_P[:, :3] = 0  # only applicable to the E type
            switch = switch_N_to_P + switch_P_to_N
            # invest_tracker = invest * invest_tracker
            invest_tracker = invest_tracker + switch_N_to_P - switch_P_to_N
            d_eta_st = a * invest_tracker - theta_t

            # tau_info and V_hat has to change for the agents who switched to N
            Vhat_vector = np.append(V_st_P, Vhat * np.ones((Ntype, Nconstraint, 1)), axis=2) * switch_P_to_N + \
                          np.append(V_st_N, Vhat * np.ones((Ntype, Nconstraint, 1)), axis=2) * switch_N_to_P + \
                          Vhat_vector * (1 - switch)  # reset initial variance
            tau_info = dt * switch + tau_info * (1 - switch)  # reset clock

            # print(theta_t)

    return (
        Delta_s_t,
        eta_st_eta_ss,
        X,
        d_eta_st,
        invest_tracker,
        tau_info,
        Vhat_vector,
        can_short_tracker,
    )