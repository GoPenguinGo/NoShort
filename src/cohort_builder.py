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
    beta0: float,
    alpha_i: np.ndarray,
    rho_cohort_type: np.ndarray,
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
]:
    """builds up a sufficiently large set of cohorts in the economy, view each cohort as one agent with a constantly shrinking size

    Args:
        dZ_build (np.ndarray): shocks to the output for each period, shape (Nc-1, )
        dZ_SI_build(np.ndarray): shocks to the signal for each period, shape (Nc-1, )
        Nc (int): number of periods  = number of cohorts in the economy
        dt (float): unit of time
        tau (np.ndarray): time since birth for each cohort
        Ntype (int): number of types for time preference
        beta_i (np.ndarray): consumption wealth ratio of each type
        beta0 (float): consumption wealth ratio of new born cohort
        alpha_i (np.ndarray): density of each type
        rho_cohort_type (np.ndarray): alpha_i * exp(-(rho_i + nu)*(t-s))
        Vhat (float): initial variance of beliefs
        sigma_Y (float): sd of aggregate output growth
        tax (float): marginal rate of wealth tax
        phi (float): correlation between the signal and the output growth rate
        Npre (int): pre-trading periods
        Ninit (int): set-up periods when we treat the market as complete and do not search for theta
        mode_trade (str): {'complete', 'w_constraint', 'partial_constraint_old'}
        mode_learn (str): {'reentry', 'disappointment'}


    Returns:
        Delta_s_t (np.ndarray): estimation bias, shape(Ntype, Nc, )
        eta_st_eta_ss(np.ndarray): shape(Ntype, Nc, )
        X(np.ndarray):W_s * Xi_s, shape(Ntype, Nc, )
        d_eta_s t (np.ndarray): max(delta_st, -theta_t), shape(Ntype, Nc, )
        invest_tracker (np.ndarray): track if a cohort is still in the stock market, shape(Ntype, Nc, )
        tau_info (np.ndarray): t-t', time since the last time a cohort switch, shape(Ntype, Nc, )
        Vhat_vector (np.ndarray): variance at t', shape(Ntype, Nc, )
    """

    # initialize the matrices
    Delta_s_t = np.zeros((Ntype, 1))
    d_eta_st = np.zeros((Ntype, 1))
    X = np.ones((1, 1))
    eta_st_eta_ss_init = np.ones((Ntype, 1))
    eta_st_eta_ss = eta_st_eta_ss_init
    invest_tracker = np.ones((Ntype, Ninit), dtype=np.int8) if mode_trade != 'complete' else np.ones((Ntype, Nc), dtype=np.int8)
    tau_info = np.ones((Ntype, 1)) * dt
    Vhat_init = np.ones((Ntype, 1)) * Vhat
    Vhat_vector = Vhat_init
    a_phi = (1 - phi ** 2)
    phi_sqr_a_phi = phi / np.sqrt(a_phi)
    a_phi_1 = 1 / a_phi
    sigma_Y_sq = sigma_Y ** 2

    for i in tqdm(range(1, Nc)):
        # new cohort born (age 0), get wealth transfer, observe, invest
        rho_cohort_type_short = rho_cohort_type[:, -i:]
        dZ_build_t = dZ_build[i - 1]
        dZ_SI_build_t = dZ_SI_build[i - 1]

        eta_st_eta_ss = eta_st_eta_ss * np.exp(
            (-0.5 * d_eta_st ** 2) * dt
            + d_eta_st * dZ_build_t
        )  # equation (15)

        # from equation (20) and the description below it
        # X_t = W_t * xi_t, is the sum of tax * X_s * eta_st_eta_ss * rho_cohort_type_short * dt, s<t;
        # X is the collection of all X_s, s<t.
        X_parts = tax * X * eta_st_eta_ss * rho_cohort_type_short * dt
        X_t = np.sum(X_parts) / (1 - tax * beta0 * dt)  # dividing by (1-tax*dt) keeps sum(f_st*dt) at 1

        eta_st_eta_ss = np.append(eta_st_eta_ss, eta_st_eta_ss_init, axis=1)
        X = np.append(X, np.ones((1, 1)) * X_t, axis=1)
        X = X / X_t  # rescale, does not change the relative magnitude of each cohort
        # eta_bar_t goes to 0 too quickly if (1) mode != 'comp', and (2) initial window very small
        #  eta_bar_t is the denominator; it creates issues if too close to 0
        #  so we rescale eta_bar to keep it away from 0, without changing f_st

        f_c_ist = X_parts / X_t / dt
        f_c_ist = np.append(f_c_ist, tax * alpha_i * beta_i, axis=1)

        beta_t = 1 / np.sum(f_c_ist / beta_i * dt)
        f_w_ist = f_c_ist / beta_i * beta_t

        # update beliefs
        if mode_trade == 'complete':
            V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
            dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'P')
        elif mode_trade == 'w_constraint' or mode_trade == 'partial_constraint_rich' or mode_trade == 'partial_constraint_old':
            if i < Ninit:
                V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
                dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'P')
            else:
                V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')  # from eq(6)
                dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'N')  # from eq(9)
                V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
                dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'P')
                dDelta_s_t = invest_tracker * dDelta_s_t_P + (1 - invest_tracker) * dDelta_s_t_N

        else:
            print('mode_trade not found')
            exit()

        # add a new cohort to Vhat_vector and tau_info
        Vhat_vector = np.append(Vhat_vector, Vhat_init, axis=1)
        if mode_trade == 'complete':  # where tau_info is the same with age; no switch between N and P for complete market
            tau_info = tau[:, -i - 1:]
        else:  # where tau_info is t-t', where t' is either the last time a cohort switch, or the birth time
            tau_info = np.append(tau_info, np.zeros((Ntype, 1)), axis=1) + dt

        if i < Npre:
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(Delta_s_t, np.zeros((Ntype, 1)), axis=1)  # newborns begin with 0 bias when there are not enough observations
        else:
            init_bias = np.average(dZ_build[int(i - Npre):i]) / dt
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(
                Delta_s_t, init_bias * np.zeros((Ntype, 1)), axis=1
            )  # newborns begin with Npre observations of the dividend process

        # find the market clearing theta, given beliefs and consumption shares
        if i < Ninit or mode_trade == 'complete':  # Ninit: initial rounds where the short-sale constraint is relaxed
            d_eta_st = (
                Delta_s_t  # relax the short-sale constraint in the beginning
            )

        elif mode_trade == 'w_constraint':
            invest_tracker = np.append(invest_tracker, np.ones((Ntype, 1)), axis=1)  # indicator of current status, =1 for a cohort if type == P

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
                # tau_info and V_hat has to change for the agents who switch to N
                Vhat_vector = np.append(V_st_P, Vhat * np.ones((Ntype, 1)), axis=1) * switch_P_to_N + Vhat_vector * (1 - switch_P_to_N)  # reset prior variance V'
                tau_info = dt * switch_P_to_N + tau_info * (1 - switch_P_to_N)  # reset clock t'

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

                # tau_info and V_hat has to change for the agents who switch (either P to N or vice versa)
                Vhat_vector = np.append(V_st_P, Vhat * np.ones((Ntype, 1)), axis=1) * switch_P_to_N + \
                              np.append(V_st_N, Vhat * np.ones((Ntype, 1)), axis=1) * switch_N_to_P + \
                              Vhat_vector * (1 - switch)  # reset V'
                tau_info = dt * switch + tau_info * (1 - switch)  # reset t'

            else:
                print('mode_learn not found')
                break

        else:
            print('mode_trade not found')
            break

    return (
        Delta_s_t,
        eta_st_eta_ss,
        X,
        d_eta_st,
        invest_tracker,
        tau_info,
        Vhat_vector,
    )


def build_cohorts_mix_type(
    dZ_build: np.ndarray,
    dZ_SI_build: np.ndarray,
    Nc: int,
    dt: float,
    Ntype: int,
    Nconstraint: int,
    beta_i: np.ndarray,
    beta0: float,
    alpha_i: np.ndarray,
    rho_cohort_type: np.ndarray,
    Vhat: float,
    sigma_Y: float,
    tax: float,
    phi: float,
    Npre: int,
    Ninit: int,
    cohort_type_size: np.ndarray,
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
        unconstrained (designated participants);
        excluded (designated nonparticipants);
        disappointment (leave the stock market for good upon binding constraint);
        & reentry.

    Args:
        dZ_build (np.ndarray): shocks to the output for each period, shape (Nc-1, )
        dZ_SI_build(np.ndarray): shocks to the signal for each period, shape (Nc-1, )
        Nc (int): number of periods  = number of cohorts in the economy
        dt (float): unit of time
        tau (np.ndarray): time since birth for each cohort
        Ntype (int): number of types for time preference
        Nconstraint (int): number of types for trading (mix)
        beta_i (np.ndarray): consumption wealth ratio of each type
        beta0 (float): consumption wealth ratio of the new born cohort
        alpha_i (np.ndarray): density of each type
        rho_cohort_type (np.ndarray): alpha_i * exp(-(rho_i + nu)*(t-s))
        Vhat (float): initial variance of beliefs
        sigma_Y (float): sd of aggregate output growth
        tax (float): marginal rate of wealth tax
        phi (float): correlation between the signal and the output growth rate
        Npre (int): pre-trading periods
        Ninit (int): set-up periods when we treat the market as complete and do not search for theta

    Returns:
        Delta_s_t (np.ndarray): estimation bias, shape(Ntype, Nconstraint, Nc, )
        eta_st_eta_ss(np.ndarray): shape(Ntype, Nconstraint, Nc, )
        X(np.ndarray):W_s * Xi_s, shape(Ntype, Nconstraint, Nc, )
        d_eta_st (np.ndarray): max(delta_st, -theta_t), shape(Ntype, Nconstraint, Nc, )
        invest_tracker (np.ndarray): track if a cohort is still in the stock market, shape(Ntype, Nconstraint, Nc, )
        tau_info (np.ndarray): t-t', time since the last time a cohort switch, shape(Ntype, Nconstraint, Nc, )
        Vhat_vector (np.ndarray): variance at t', shape(Ntype, Nconstraint, Nc, )
        can_short_tracker (np.ndarray): track if a cohort can short, shape(Ntype, Nconstraint, Nc, )
    """

    Delta_s_t = np.zeros((Ntype, Nconstraint, 1), dtype=np.float32)
    d_eta_st = np.zeros((Ntype, Nconstraint, 1), dtype=np.float32)
    X = np.ones((1, 1, 1))
    eta_st_eta_ss_init = np.ones((Ntype, Nconstraint, 1))
    eta_st_eta_ss = eta_st_eta_ss_init
    invest_tracker = np.ones((Ntype, Nconstraint, Ninit), dtype=np.int8)
    invest_newborn = np.array([[[1], [0], [1], [1]]]) * np.ones((Ntype, Nconstraint, 1), dtype=np.int8)
    can_short_tracker = np.ones((Ntype, Nconstraint, Ninit), dtype=np.int8)
    can_short_newborn = np.array([[[1], [0], [0], [0]]]) * np.ones((Ntype, Nconstraint, 1), dtype=np.int8)
    tau_info = np.ones((Ntype, Nconstraint, 1)) * dt
    Vhat_init = np.ones((Ntype, Nconstraint, 1)) * Vhat
    Vhat_vector = Vhat_init
    a_phi = (1 - phi ** 2)
    phi_sqr_a_phi = phi / np.sqrt(a_phi)
    a_phi_1 = 1 / a_phi
    sigma_Y_sq = sigma_Y ** 2

    phi_mat = 0
    popu_parti_t = 1

    for i in tqdm(range(1, Nc)):
        # new cohort born (age 0), get wealth transfer, observe, invest
        rho_cohort_type_short = rho_cohort_type[:, :, -i:]
        dZ_build_t = dZ_build[i - 1]
        dZ_SI_build_t = dZ_SI_build[i - 1]

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
        f_c_ist = np.append(f_c_ist, tax * alpha_i * beta_i, axis=2)

        # beta_t = 1 / np.sum(f_c_ist / beta_i * dt)
        # f_w_ist = f_c_ist / beta_i * beta_t

        # update beliefs
        # if i < Ninit:
        #     V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
        #     dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'P')
        # else:
        #     V_st_N = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'N')  # from eq(6)
        #     dDelta_s_t_N = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_N, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'N')  # from eq(9)
        #     V_st_P = post_var(sigma_Y_sq, Vhat_vector, tau_info, a_phi, 'P')
        #     dDelta_s_t_P = dDelta_st_calculator(sigma_Y_sq, a_phi_1, phi_sqr_a_phi, dt, V_st_P, Delta_s_t, dZ_build_t, dZ_SI_build_t, 'P')
        #     dDelta_s_t = invest_tracker * dDelta_s_t_P + (1 - invest_tracker) * dDelta_s_t_N

        if i < Ninit:
            V_st = post_var(sigma_Y_sq, Vhat)[:, -i:]
            # dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, phi,  dt, V_st_P, Delta_s_t, dZ_build_t, 'P')
            dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, phi_mat, dt, V_st, Delta_s_t, dZ_build_t)  # from eq(9)
        else:
            V_st = post_var(sigma_Y_sq, Vhat)[:, -i:]  # from eq(6)
            phi_mat = (1 - popu_parti_t) * (1 - invest_tracker)
            dDelta_s_t = dDelta_st_calculator(sigma_Y_sq, phi_mat, dt, V_st, Delta_s_t, dZ_build_t)  # from eq(9)

        # add a new cohort to Vhat_vector and tau_info
        # Vhat_vector = np.append(Vhat_vector, Vhat_init, axis=2)
        # tau_info = np.append(tau_info, np.zeros((Ntype, Nconstraint, 1)), axis=2) + dt

        if i < Npre:
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(Delta_s_t, np.zeros((Ntype, Nconstraint, 1)), axis=2)  # newborns begin with 0 bias when there are not enough observations
        else:
            init_bias = np.average(dZ_build[int(i - Npre): i]) / dt
            Delta_s_t += dDelta_s_t
            Delta_s_t = np.append(
                Delta_s_t, init_bias * np.zeros((Ntype, Nconstraint, 1)), axis=2
            )  # newborns begin with Npre observations of the dividend process

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

            lowest_bound = -np.max(possible_delta_st[np.nonzero(possible_delta_st)])  # absolute lower bound where no agent holds the stock
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
            invest_tracker = invest_tracker + switch_N_to_P - switch_P_to_N
            d_eta_st = a * invest_tracker - theta_t

            popu_parti_t = np.sum(cohort_type_size[:, :, -i:] * invest_tracker)

            # tau_info and V_hat has to change for the agents who switch
            # Vhat_vector = np.append(V_st_P, Vhat * np.ones((Ntype, Nconstraint, 1)), axis=2) * switch_P_to_N + \
            #               np.append(V_st_N, Vhat * np.ones((Ntype, Nconstraint, 1)), axis=2) * switch_N_to_P + \
            #               Vhat_vector * (1 - switch)  # reset V'
            # tau_info = dt * switch + tau_info * (1 - switch)  # reset t'

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