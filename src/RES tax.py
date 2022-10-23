import time
import numpy as np
import matplotlib.pyplot as plt


# it is correct now; but much slower than matlab
#################  Define functions  ####################################################################
def PostVar(sigY, Vbar, tau):
    if type(tau) == np.ndarray:
        V = sigY**2 * Vbar / (sigY**2 * np.ones(len(tau)) + Vbar * tau)
    else:
        V = sigY**2 * Vbar / (sigY**2 + Vbar * tau)
    return V


def BuildUpCohortsMAIN(dZt, Nt, dt, rho, nu, Vbar, mu_Y, sigma_Y, bet, That):
    # builds up a sufficiently large set of cohorts
    Npre = int(That / dt)  # Number of pre-trading observations
    Zt = np.insert(np.cumsum(dZt), 0, 0)
    yg = (mu_Y - 0.5 * sigma_Y**2) * dt * np.ones(int(Nt - 1)) + sigma_Y * dZt
    Yt = np.insert(np.exp(np.cumsum(yg)), 0, 1)
    Xt = np.ones(Nt) * nu * bet
    Deltabar = np.zeros(Nt)
    Delta_s_t = np.zeros(1)
    IntVec = 1 * nu * bet
    tau = dt * np.ones(1)
    reduction = np.exp(-nu * dt)
    for i in range(1, Nt):
        Part = IntVec * np.exp(
            -(rho + 0.5 * Delta_s_t * Delta_s_t) * dt + Delta_s_t * dZt[i - 1]
        )
        if i == 1:
            Xt[i] = Part
            Deltabar[i] = Part * Delta_s_t
        else:
            Xt[i] = np.sum(Part)
            Deltabar[i] = np.sum(Part * Delta_s_t) / Xt[i]
        IntVec = reduction * Part
        IntVec = np.append(IntVec, bet * (1 - reduction) * Xt[i])
        f = IntVec / Xt[i]
        dDelta_s_t = (PostVar(sigma_Y, Vbar, tau) / sigma_Y**2) * (
            -Delta_s_t * dt + np.ones(len(Delta_s_t)) * dZt[i - 1]
        )
        if i < Npre:
            Delta_s_t = Delta_s_t + dDelta_s_t
            Delta_s_t = np.append(Delta_s_t, 0)
        else:
            DELbias = np.sum(dZt[int(i - Npre) : i]) / That
            Delta_s_t = Delta_s_t + dDelta_s_t
            Delta_s_t = np.append(Delta_s_t, DELbias)
        tau = tau + dt
        tau = np.append(tau, 0)
    return Deltabar, IntVec, Xt, Delta_s_t, Yt, Zt, f, tau


def SimCohortsMAIN(
    biasvec,
    dZt,
    Nt,
    tau,
    IntVec,
    Delta_s_t,
    dt,
    rho,
    nu,
    Vbar,
    mu_Y,
    sigma_Y,
    sigma_S,
    bet,
    That,
    Npre,
):
    # Initializing variables
    Xt2 = np.ones(Nt)
    Deltabar2 = np.ones(Nt)
    Et = np.ones(Nt)
    Vt = np.ones(Nt)

    part1 = np.zeros(Nt)
    dR = np.zeros(Nt)
    # counter = 0
    reduction = np.exp(-nu * dt)
    BIGDELTA = np.zeros((Nt, Nt))
    BIGF = np.zeros((Nt, Nt))

    RevNt = np.flip(range(Nt))
    fhat = reduction**RevNt * nu * dt
    fhat = fhat / np.sum(fhat)

    # Expected returns
    mu_S = np.zeros(Nt)  # expected return under true measure
    mu_S_t = np.zeros(Nt)  # expected return under agent-measure
    muhat_S_t = np.zeros(Nt)

    muC_s_t = np.zeros(Nt)  # drift log connp.sumption
    sigmaC_s_t = np.zeros(Nt)  # diffusion log connp.sumption

    # Aggregate quantities
    r_t = np.zeros(Nt)
    theta_t = np.zeros(Nt)
    fst = np.zeros(Nt)  # individual connp.sumption-wealth ratio
    for i in range(Nt):
        Part = IntVec * np.exp(-(0.5 * Delta_s_t**2) * dt + Delta_s_t * dZt[i])
        if i < Nt - 1:
            if i == 0:
                part1[i] = np.sum(biasvec) / That
            else:
                part1[i] = part1[i - 1] + (
                    PostVar(sigma_Y, Vbar, (i * dt)) / sigma_Y**2
                ) * (-part1[i - 1] * dt + dZt[i - 1])
        if i == 0:
            dR[i] = 0
        else:
            dR[i] = (
                mu_S[i - 1] - r_t[i - 1] + rho + mu_Y - sigma_Y**2 + nu * (1 - bet)
            ) * dt + sigma_S * dZt[i]

        Xt2[i] = np.sum(Part)
        Deltabar2[i] = np.sum(Part * Delta_s_t) / Xt2[i]
        f = Part / Xt2[i]

        BIGDELTA[i, :] = Delta_s_t

        mu_S[i] = sigma_S * sigma_Y - (sigma_S - sigma_Y) * Deltabar2[i]
        mu_S_t[i] = (
            sigma_S * sigma_Y
            + sigma_S * (-Deltabar2[i] + part1[i])
            + sigma_Y * Deltabar2[i]
        )
        muhat_S_t[i] = mu_S[i] + sigma_S * np.sum(fhat * Delta_s_t)

        r_t[i] = rho + mu_Y - sigma_Y**2 + nu * (1 - bet) + sigma_Y * Deltabar2[i]
        theta_t[i] = sigma_Y - Deltabar2[i]

        Et[i] = np.sum(
            f * (Vbar / (1 + (Vbar / sigma_Y**2) * dt * RevNt)) * (1 / sigma_Y)
        )
        Vt[i] = (np.sum(f * Delta_s_t**2) - Deltabar2[i] ** 2) * sigma_Y

        fst[i] = f[Nt - i - 1]
        BIGF[i, :] = f

        muC_s_t[i] = (
            mu_Y + nu * (1 - bet) + (sigma_Y - Deltabar2[i]) * (part1[i] - Deltabar2[i])
        )
        sigmaC_s_t[i] = sigma_Y + part1[i] - Deltabar2[i]

        # Updating:
        dDelta_s_t = (PostVar(sigma_Y, Vbar, tau) / sigma_Y**2) * (
            -Delta_s_t * dt + np.ones(len(Delta_s_t)) * dZt[i]
        )
        if i < Npre:
            DELbias = (np.sum(biasvec[i:]) + np.sum(dZt[:i])) / That
        else:
            DELbias = np.sum(dZt[i - Npre : i]) / That

        Delta_s_t = Delta_s_t[1:] + dDelta_s_t[1:]
        Delta_s_t = np.append(Delta_s_t, DELbias)
        IntVec = reduction * Part[1:]
        IntVec = np.append(IntVec, bet * (1 - reduction) * Xt2[i])

    Port = (part1 - Deltabar2) / sigma_S + (sigma_Y / sigma_S) * (
        1 - bet * np.flip(fhat) / fst
    )

    return (
        Xt2,
        Deltabar2,
        part1,
        mu_S,
        mu_S_t,
        muhat_S_t,
        r_t,
        theta_t,
        Port,
        muC_s_t,
        sigmaC_s_t,
        BIGF,
        BIGDELTA,
        Et,
        Vt,
        dR,
    )


#############################################################################################################

# Parameters
rho = 0.001  # Time discount factor
nu = 0.02  # Death rate
mu_Y = 0.02  # Growth rate of output
sigma_Y = 0.033  # Standard deviation of output
sigma_S = (
    sigma_Y  # In equilibrium the stock price diffusion is the same as output diffusion
)
w = 0.92  # Fraction of total output paid out as endowment

# Some pre-calculations
D = rho**2 + 4 * (rho * nu + nu**2) * (1 - w)
bet = (rho + 2 * nu - D**0.5) / (2 * nu)
rlog = rho + mu_Y - sigma_Y**2

# Setting prior variance
dt = 1 / 12  # time incremental
Tcohort = 500  # time horizon to keep track of cohorts
Nt = int(Tcohort / dt)

# test on the old code about the kink
Mpaths = 300
Nc = Nt
T_hats = dt * np.arange(1, 13, 1)
T_hat_dimension = len(T_hats)
# nus = [0.01, 0.02, 0.03]
nus = [0.02]
nu_dimension = len(nus)

# Generate matrix to store the results
dZ_matrix = np.zeros((Mpaths, Nt))
dZ_build_matrix = np.zeros((Mpaths, Nc - 1))
r_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))
theta_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))
pi_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))
f_parti_matrix = np.zeros((T_hat_dimension, nu_dimension, Mpaths))

for l in range(Mpaths):
    dZ_build = dt**0.5 * np.random.randn(int(Nc - 1))  # dZt for the build function
    dZ = dt**0.5 * np.random.randn(Nt)  # dZt for the simulate function
    dZ_matrix[l, :] = dZ
    dZ_build_matrix[l, :] = dZ_build
    for k, That in enumerate(T_hats):
        Vbar = (sigma_Y**2) / That  # prior variance
        Npre = int(That / dt)
        for m, nu in enumerate(nus):
            # this part is repetitive when there is only one value of nu
            D = rho**2 + 4 * (rho * nu + nu**2) * (1 - w)
            bet = (rho + 2 * nu - D**0.5) / (2 * nu)
            rlog = rho + mu_Y - sigma_Y**2
            # create age quartiles for analysis
            (Deltabar, IntVec, Xt, Delta_s_t, Yt, Zt, f, tau) = BuildUpCohortsMAIN(
                dZ_build, Nt, dt, rho, nu, Vbar, mu_Y, sigma_Y, bet, That
            )
            dZforbias = np.diff(Zt)
            biasvec = dZforbias[-Npre:]
            dZt = dt**0.5 * np.random.normal(0, 1, Nt)
            Zt = np.cumsum(dZt)
            (
                Xt2,
                Deltabar2,
                Part1,
                mu_S,
                mu_S_t,
                muhat_S_t,
                r_t,
                theta_t,
                Port,
                muC_s_t,
                sigmaC_s_t,
                BIGf,
                BIGDELTA,
                Et,
                Vt,
                dR,
            ) = SimCohortsMAIN(
                biasvec,
                dZt,
                Nt,
                tau,
                IntVec,
                Delta_s_t,
                dt,
                rho,
                nu,
                Vbar,
                mu_Y,
                sigma_Y,
                sigma_S,
                bet,
                That,
                Npre,
            )
            r_matrix[k, m, l] = np.mean(r_t)
            theta_matrix[k, m, l] = np.mean(theta_t)
        # covariance:
    print(l)

x = T_hats
y1 = np.mean(r_matrix, axis=2)
y2 = np.mean(theta_matrix, axis=2)

xlabels = ["interest rate", "market price of risk"]
ys = [y1, y2]

for i in range(len(ys)):
    y_nu = ys[i]
    for j in range(nu_dimension):
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        y = y_nu[:, j]
        ax.plot(x, y)
        ax.set_xlabel("initial window")
        ax.set_ylabel("mean " + xlabels[i])
        # plt.savefig('initial window and ' + xlabels[i] + '_' + mode + '_' + zoom_in + str(nu) + '.png', dpi=500,
        # format="png")
        # plt.savefig('initial window and ' + xlabels[i] + '_' + mode + '.png', dpi=500, format="png")
        plt.show()
        # plt.close()
