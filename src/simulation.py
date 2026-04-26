import numpy as np
from typing import Tuple
from src.cohort_builder import build_cohorts_SI, build_cohorts_mix_type
from src.cohort_simulator import simulate_cohorts_SI, simulate_cohorts_mean_vola, simulate_cohorts_mix_type, \
    simulate_mean_vola_mix_type
from src.param import cutoffs_age


def simulate_SI(
        mode_trade: str,
        mode_learn: str,
        Nc: int,
        Nt: int,
        dt: float,
        nu: float,
        Vhat: float,
        mu_Y: float,
        sigma_Y: float,
        tax: float,
        beta0: float,
        phi: float,
        Npre: int,
        Ninit: int,
        T_hat: int,
        entry_bound: float,
        exit_bound: float,
        dZ_build: np.ndarray,
        dZ: np.ndarray,
        dZ_SI_build: np.ndarray,
        dZ_SI: np.ndarray,
        tau: np.ndarray,
        cutoffs_age: np.ndarray,
        Ntype: int,
        rho_i: np.ndarray,
        alpha_i: np.ndarray,
        beta_i: np.ndarray,
        rho_cohort_type: np.ndarray,
        cohort_type_size: np.ndarray,
        need_f: str,
        need_Delta: str,
        need_pi: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Bundles the cohort_builder function and the cohort_simulator function together

    Args:
        mode_trade (str): {'complete', 'w_constraint', 'partial_constraint_old'}
        mode_learn (str): {'reentry', 'disappointment'}
        Nc (int): number of cohorts in the economy
        Nt (int): number of periods in the simulation
        dt (float): per unit of time
        nu (float): rate of birth and death
        Vhat (float): initial variance
        mu_Y (float): as in eq(1), drift of aggregate output growth
        sigma_Y (float): as in eq(1), diffusion of aggregate output growth
        tax (float): marginal rate of wealth tax
        beta0 (float): average consumption wealth ratio of a new born cohort
        phi (float): correlation between the signal and the output growth rate
        Npre (float): pre-trading number of observations
        Ninit (int): set-up periods when we treat the market as complete and do not search for theta
        T_hat (float): pre-trading years
        dZ_build (np.ndarray): shocks to the output for cohort_builder function, shape (Nt, )
        dZ (np.ndarray): shocks to the output for cohort_simulator function, shape (Nt, )
        dZ_SI_build (np.ndarray):shocks to the signal for cohort_builder function, shape (Nt, )
        dZ_SI(np.ndarray): shocks to the signal for cohort_simulator function, shape (Nt, )
        tau (np.ndarray): time since birth t-s or age, shape(Nt)
        cutoffs_age (np.ndarray): cutoff ages for the age quartiles
        Ntype (int): number of types for time preference
        rho_i (np.ndarray): time preference for each type
        alpha_i (np.ndarray): density of each type
        beta_i (np.ndarray): consumption wealth ratio of each type
        rho_cohort_type (np.ndarray): alpha_i * exp(-(rho_i + nu)*(t-s))
        cohort_type_size (np.ndarray): cohort_size * alpha_i
        -*-*-*- avoid saving results unnecessary for particular analysis -*-*-*-
        need_f (str): {'True', 'False'}
        need_Delta (str): {'True', 'False'}
        need_pi (str): {'True', 'False'}

    Returns:
        r (np.ndarray): interest rate, shape(Nt, )
        theta (np.ndarray): market price of risk, shape(Nt, )
        f_c (np.ndarray): consumption share, shape(Nt, Ntype, Nc)
        Delta (np.ndarray): standardized estimation error, shape(Nt, Ntype, Nc)
        pi (np.ndarray): portfolio, shape(Nt, Ntype, Nc)
        parti (np.ndarray): participation rate, shape(Nt)
        Phi_bar_parti (np.ndarray): consumption share of participants, shape(Nt)
        Phi_tilde_parti (np.ndarray): wealth share of participants, shape(Nt)
        Delta_bar_parti (np.ndarray): consumption weighted average estimation error of participants, shape(Nt)
        Delta_tilde_parti (np.ndarray): wealth weighted average estimation error of participants, shape(Nt)
        dR (np.ndarray): realized stock returns, shape(Nt)
        mu_S (np.ndarray): expected equity risk premia, shape(Nt)
        sigma_S (np.ndarray): conditional volatility of stock returns, shape(Nt)
        beta (np.ndarray): aggregate consumption wealth ratio, shape(Nt)
        invest_mat (np.ndarray): whether a cohort is in the stock market, shape(Nt, Nc)
        parti_age_group: participation rate in age groups, shape(Nt, 4)
        parti_wealth_group: participation rate in wealth groups, shape(Nt, 4)
    """

    biasvec = dZ_build[-Npre:]  # dZt used in the build_cohorts function

    (
        Delta_s_t,
        eta_st_eta_ss,
        X,
        d_eta_st,
        invest_tracker,
        tau_info,
        Vhat_vector,
    ) = build_cohorts_SI(
        dZ_build,
        dZ_SI_build,
        Nc,
        dt,
        tau,
        Ntype,
        beta_i,
        beta0,
        alpha_i,
        rho_cohort_type,
        Vhat,
        sigma_Y,
        tax,
        phi,
        Npre,
        Ninit,
        entry_bound,
        exit_bound,
        mode_trade,
        mode_learn,
    )

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
    ) = simulate_cohorts_SI(
        biasvec,
        dZ,
        dZ_SI,
        Nt,
        Nc,
        tau,
        dt,
        Ntype,
        rho_i,
        alpha_i,
        beta_i,
        rho_cohort_type,
        beta0,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        tax,
        phi,
        T_hat,
        Npre,
        entry_bound,
        exit_bound,
        mode_trade,
        mode_learn,
        cohort_type_size,
        cutoffs_age,
        Delta_s_t,
        eta_st_eta_ss,
        X,
        d_eta_st,
        invest_tracker,
        tau_info,
        Vhat_vector,
        need_f,
        need_Delta,
        need_pi
    )

    return (
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
    )


def simulate_SI_mean_vola(
        mode_trade: str,
        mode_learn: str,
        Nc: int,
        Nt: int,
        dt: float,
        nu: float,
        Vhat: float,
        mu_Y: float,
        sigma_Y: float,
        tax: float,
        beta0: float,
        phi: float,
        Npre: int,
        Ninit: int,
        T_hat: int,
        entry_bound: float,
        exit_bound: float,
        dZ_build: np.ndarray,
        dZ: np.ndarray,
        dZ_SI_build: np.ndarray,
        dZ_SI: np.ndarray,
        tau: np.ndarray,
        Ntype: int,
        rho_i: np.ndarray,
        alpha_i: np.ndarray,
        beta_i: np.ndarray,
        rho_cohort_type: np.ndarray,
        cohort_type_size: np.ndarray,
        need_invest_matrix: str
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Bundles the cohort_builder_mean_vola function and the cohort_simulator_mean_vola function together

    Args:
        mode_trade (str): {'complete', 'w_constraint', 'partial_constraint_old'}
        mode_learn (str): {'reentry', 'disappointment'}
        Nc (int): number of cohorts in the economy
        Nt (int): number of periods in the simulation
        dt (float): per unit of time
        nu (float): rate of birth and death
        Vhat (float): initial variance
        mu_Y (float): as in eq(1), drift of aggregate output growth
        sigma_Y (float): as in eq(1), diffusion of aggregate output growth
        tax (float): marginal rate of wealth tax
        beta0 (float): average consumption wealth ratio of a new born cohort
        phi (float): correlation between the signal and the output growth rate
        Npre (float): pre-trading number of observations
        Ninit (int): set-up periods when we treat the market as complete and do not search for theta
        T_hat (float): pre-trading years
        dZ_build (np.ndarray): shocks to the output for cohort_builder function, shape (Nt, )
        dZ (np.ndarray): shocks to the output for cohort_simulator function, shape (Nt, )
        dZ_SI_build (np.ndarray):shocks to the signal for cohort_builder function, shape (Nt, )
        dZ_SI(np.ndarray): shocks to the signal for cohort_simulator function, shape (Nt, )
        tau (np.ndarray): time since birth t-s or age, shape(Nt)
        cutoffs_age (np.ndarray): cutoff ages for the age quartiles
        Ntype (int): number of types for time preference
        rho_i (np.ndarray): time preference for each type
        alpha_i (np.ndarray): density of each type
        beta_i (np.ndarray): consumption wealth ratio of each type
        rho_cohort_type (np.ndarray): alpha_i * exp(-(rho_i + nu)*(t-s))
        cohort_type_size (np.ndarray): cohort_size * alpha_i

    Returns:
        -*-*-*- time-series mean and standard deviation of the quantities -*-*-*-
        dR_matrix (np.ndarray): shape(2)
        theta_matrix (np.ndarray): shape(2)
        r_matrix (np.ndarray): shape(2)
        mu_S_matrix (np.ndarray): shape(2)
        sigma_S_matrix (np.ndarray): shape(2)
        beta_matrix (np.ndarray): shape(2)
        -*-*-*- time-series mean and standard deviation of the state variables determining the quantities -*-*-*-
        theta_save_matrix (np.ndarray): shape(2, 2)
        sigma_S_save_matrix (np.ndarray): shape(4, 2)
        parti_age_group_matrix (np.ndarray): shape(4), mean only
        parti_wealth_group_matrix (np.ndarray): shape(4), mean only
        cov_save_matrix (np.ndarray): shape(6, 2)
        parti (np.ndarray): shape(4), mean only
        cov_parti_matrix (np.ndarray): shape(4), mean only

    """

    biasvec = dZ_build[-Npre:]  # dZt used in the build_cohorts function

    (
        Delta_s_t,
        eta_st_eta_ss,
        X,
        d_eta_st,
        invest_tracker_build,
        tau_info_build,
        Vhat_vector,
    ) = build_cohorts_SI(
        dZ_build,
        dZ_SI_build,
        Nc,
        dt,
        tau,
        Ntype,
        beta_i,
        beta0,
        alpha_i,
        rho_cohort_type,
        Vhat,
        sigma_Y,
        tax,
        phi,
        Npre,
        Ninit,
        entry_bound,
        exit_bound,
        mode_trade,
        mode_learn,
    )

    (
        theta_ave,
        r_ave,
        mu_S_ave,
        sigma_S_ave,
        parti_age_ave,
        Delta_age_ave,
        reentry_time,
        exit_time,
        entry_cumu,
        entry_ave,
        exit_ave,
        cov_matrix,
        parti,
        regression_table1_b,
        regression_table2_b,
        # table_1c_ave
    ) = simulate_cohorts_mean_vola(
        biasvec,
        dZ,
        dZ_SI,
        Nt,
        tau,
        dt,
        Ntype,
        rho_i,
        alpha_i,
        beta_i,
        rho_cohort_type,
        beta0,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        tax,
        phi,
        T_hat,
        Npre,
        entry_bound,
        exit_bound,
        mode_trade,
        mode_learn,
        cohort_type_size,
        Delta_s_t,
        eta_st_eta_ss,
        X,
        d_eta_st,
        invest_tracker_build,
        tau_info_build,
        Vhat_vector,
        need_invest_matrix
    )

    return (
        theta_ave,
        r_ave,
        mu_S_ave,
        sigma_S_ave,
        parti_age_ave,
        Delta_age_ave,
        reentry_time,
        exit_time,
        entry_cumu,
        entry_ave,
        exit_ave,
        cov_matrix,
        parti,
        regression_table1_b,
        regression_table2_b,
        # table_1c_ave
    )


def simulate_mix_types(
        Nc: int,
        Nt: int,
        dt: float,
        nu: float,
        Vhat: float,
        mu_Y: float,
        sigma_Y: float,
        tax: float,
        beta0: float,
        phi: float,
        Npre: int,
        Ninit: int,
        T_hat: int,
        entry_bound: float,
        exit_bound: float,
        dZ_build: np.ndarray,
        dZ: np.ndarray,
        dZ_SI_build: np.ndarray,
        dZ_SI: np.ndarray,
        cutoffs_age: np.ndarray,
        Ntype: int,
        Nconstraint: int,
        rho_i: np.ndarray,
        alpha_i: np.ndarray,
        beta_i: np.ndarray,
        rho_cohort_type: np.ndarray,
        cohort_type_size: np.ndarray,
        need_f: str,
        need_Delta: str,
        need_pi: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Bundles the build_cohorts_mix_type function and the simulate_cohorts_mix_type function together
        a mixture of 4 different types of agents in each cohort:
        unconstrained (designated participants);
        excluded (designated nonparticipants);
        disappointment (leave the stock market for good upon binding constraint);
        & reentry.
    Args:
        Nc (int): number of cohorts in the economy
        Nt (int): number of periods in the simulation
        dt (float): per unit of time
        nu (float): rate of birth and death
        Vhat (float): initial variance
        mu_Y (float): as in eq(1), drift of aggregate output growth
        sigma_Y (float): as in eq(1), diffusion of aggregate output growth
        tax (float): marginal rate of wealth tax
        beta0 (float): average consumption wealth ratio of a new born cohort
        phi (float): correlation between the signal and the output growth rate
        Npre (float): pre-trading number of observations
        Ninit (int): set-up periods when we treat the market as complete and do not search for theta
        T_hat (float): pre-trading years
        dZ_build (np.ndarray): shocks to the output for cohort_builder function, shape (Nt, )
        dZ (np.ndarray): shocks to the output for cohort_simulator function, shape (Nt, )
        dZ_SI_build (np.ndarray):shocks to the signal for cohort_builder function, shape (Nt, )
        dZ_SI(np.ndarray): shocks to the signal for cohort_simulator function, shape (Nt, )
        tau (np.ndarray): time since birth t-s or age, shape(Nt)
        cutoffs_age (np.ndarray): cutoff ages for the age quartiles
        Ntype (int): number of types for time preference
        rho_i (np.ndarray): time preference for each type
        alpha_i (np.ndarray): density of each type
        beta_i (np.ndarray): consumption wealth ratio of each type
        rho_cohort_type (np.ndarray): alpha_i * exp(-(rho_i + nu)*(t-s))
        cohort_type_size (np.ndarray): cohort_size * alpha_i
        -*-*-*- avoid saving results unnecessary for particular analysis -*-*-*-
        need_f (str): {'True', 'False'}
        need_Delta (str): {'True', 'False'}
        need_pi (str): {'True', 'False'}

    Returns:
        r (np.ndarray): interest rate, shape(Nt, )
        theta (np.ndarray): market price of risk, shape(Nt, )
        f_c (np.ndarray): consumption share, shape(Nt, Ntype, Nc)
        Delta (np.ndarray): standardized estimation error, shape(Nt, Ntype, Nc)
        pi (np.ndarray): portfolio, shape(Nt, Ntype, Nc)
        parti (np.ndarray): participation rate, shape(Nt)
        Phi_bar_parti (np.ndarray): consumption share of participants, shape(Nt)
        Phi_tilde_parti (np.ndarray): wealth share of participants, shape(Nt)
        Delta_bar_parti (np.ndarray): consumption weighted average estimation error of participants, shape(Nt)
        Delta_tilde_parti (np.ndarray): wealth weighted average estimation error of participants, shape(Nt)
        dR (np.ndarray): realized stock returns, shape(Nt)
        mu_S (np.ndarray): expected equity risk premia, shape(Nt)
        sigma_S (np.ndarray): conditional volatility of stock returns, shape(Nt)
        beta (np.ndarray): aggregate consumption wealth ratio, shape(Nt)
        invest_mat (np.ndarray): whether a cohort is in the stock market, shape(Nt, Nc)
        parti_age_group: participation rate in age groups, shape(Nt, 4)
        parti_wealth_group: participation rate in wealth groups, shape(Nt, 4)
    """


    biasvec = dZ_build[-Npre:]  # dZt used in the build_cohorts function

    (
        Delta_s_t,
        eta_st_eta_ss,
        X,
        d_eta_st,
        invest_tracker,
        information_tracker,
        tau_info,
        Vhat_vector,
        can_short_tracker,
    ) = build_cohorts_mix_type(
        dZ_build,
        dZ_SI_build,
        Nc,
        dt,
        Ntype,
        Nconstraint,
        beta_i,
        beta0,
        alpha_i,
        rho_cohort_type,
        Vhat,
        sigma_Y,
        tax,
        phi,
        Npre,
        Ninit,
        entry_bound,
        exit_bound
    )

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
    ) = simulate_cohorts_mix_type(
        biasvec,
        dZ,
        dZ_SI,
        Nt,
        Nc,
        dt,
        Ntype,
        Nconstraint,
        rho_i,
        alpha_i,
        beta_i,
        rho_cohort_type,
        beta0,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        tax,
        phi,
        T_hat,
        Npre,
        entry_bound,
        exit_bound,
        cohort_type_size,
        cutoffs_age,
        Delta_s_t,
        eta_st_eta_ss,
        X,
        d_eta_st,
        invest_tracker,
        information_tracker,
        can_short_tracker,
        tau_info,
        Vhat_vector,
        need_f,
        need_Delta,
        need_pi,
    )

    return (
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
    )


def simulate_mix_mean_vola(
        Nc: int,
        Nt: int,
        dt: float,
        nu: float,
        Vhat: float,
        mu_Y: float,
        sigma_Y: float,
        tax: float,
        beta0: float,
        phi: float,
        Npre: int,
        Ninit: int,
        T_hat: int,
        dZ_build: np.ndarray,
        dZ: np.ndarray,
        dZ_SI_build: np.ndarray,
        dZ_SI: np.ndarray,
        Ntype: int,
        Nconstraint: int,
        rho_i: np.ndarray,
        alpha_i: np.ndarray,
        beta_i: np.ndarray,
        rho_cohort_type: np.ndarray,
        cohort_type_size: np.ndarray,
        need_invest_matrix: str
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Bundles the build_mean_vola_mix_type function and the simulate_mean_vola_mix_type function together
        a mixture of 4 different types of agents in each cohort:
        unconstrained (designated participants);
        excluded (designated nonparticipants);
        disappointment (leave the stock market for good upon binding constraint);
        & reentry.

    Args:
        Nc (int): number of cohorts in the economy
        Nt (int): number of periods in the simulation
        dt (float): per unit of time
        nu (float): rate of birth and death
        Vhat (float): initial variance
        mu_Y (float): as in eq(1), drift of aggregate output growth
        sigma_Y (float): as in eq(1), diffusion of aggregate output growth
        tax (float): marginal rate of wealth tax
        beta0 (float): average consumption wealth ratio of a new born cohort
        phi (float): correlation between the signal and the output growth rate
        Npre (float): pre-trading number of observations
        Ninit (int): set-up periods when we treat the market as complete and do not search for theta
        T_hat (float): pre-trading years
        dZ_build (np.ndarray): shocks to the output for cohort_builder function, shape (Nt, )
        dZ (np.ndarray): shocks to the output for cohort_simulator function, shape (Nt, )
        dZ_SI_build (np.ndarray):shocks to the signal for cohort_builder function, shape (Nt, )
        dZ_SI(np.ndarray): shocks to the signal for cohort_simulator function, shape (Nt, )
        tau (np.ndarray): time since birth t-s or age, shape(Nt)
        cutoffs_age (np.ndarray): cutoff ages for the age quartiles
        Ntype (int): number of types for time preference
        rho_i (np.ndarray): time preference for each type
        alpha_i (np.ndarray): density of each type
        beta_i (np.ndarray): consumption wealth ratio of each type
        rho_cohort_type (np.ndarray): alpha_i * exp(-(rho_i + nu)*(t-s))
        cohort_type_size (np.ndarray): cohort_size * alpha_i

    Returns:
        -*-*-*- time-series mean and standard deviation of the quantities -*-*-*-
        dR_matrix (np.ndarray): shape(2)
        theta_matrix (np.ndarray): shape(2)
        r_matrix (np.ndarray): shape(2)
        mu_S_matrix (np.ndarray): shape(2)
        sigma_S_matrix (np.ndarray): shape(2)
        beta_matrix (np.ndarray): shape(2)
        -*-*-*- time-series mean and standard deviation of the state variables determining the quantities -*-*-*-
        theta_save_matrix (np.ndarray): shape(2, 2)
        sigma_S_save_matrix (np.ndarray): shape(4, 2)
        parti_age_group_matrix (np.ndarray): shape(4), mean only
        parti_wealth_group_matrix (np.ndarray): shape(4), mean only
        cov_save_matrix (np.ndarray): shape(6, 2)
        parti (np.ndarray): shape(4), mean only
        cov_parti_matrix (np.ndarray): shape(4), mean only

    """

    biasvec = dZ_build[-Npre:]  # dZt used in the build_cohorts function

    (
        Delta_s_t,
        eta_st_eta_ss,
        X,
        d_eta_st,
        invest_tracker_build,
        tau_info_build,
        Vhat_vector,
        can_short_tracker_build,
    ) = build_cohorts_mix_type(
        dZ_build,
        dZ_SI_build,
        Nc,
        dt,
        Ntype,
        Nconstraint,
        beta_i,
        beta0,
        alpha_i,
        rho_cohort_type,
        Vhat,
        sigma_Y,
        tax,
        phi,
        Npre,
        Ninit,
    )

    (
        theta_ave,
        r_ave,
        mu_S_ave,
        sigma_S_ave,
        reentry_time,
        exit_time,
        entry_cumu,
        entry_ave,
        exit_ave,
        cov_matrix,
        parti,
        regression_table1_b,
        regression_table2_b
    ) = simulate_mean_vola_mix_type(
        biasvec,
        dZ,
        dZ_SI,
        Nt,
        dt,
        Ntype,
        Nconstraint,
        rho_i,
        alpha_i,
        beta_i,
        rho_cohort_type,
        beta0,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        tax,
        phi,
        T_hat,
        Npre,
        cohort_type_size,
        Delta_s_t,
        eta_st_eta_ss,
        X,
        d_eta_st,
        invest_tracker_build,
        can_short_tracker_build,
        tau_info_build,
        Vhat_vector,
        need_invest_matrix
    )

    return (
        theta_ave,
        r_ave,
        mu_S_ave,
        sigma_S_ave,
        reentry_time,
        exit_time,
        entry_cumu,
        entry_ave,
        exit_ave,
        cov_matrix,
        parti,
        regression_table1_b,
        regression_table2_b
    )
