
#############################################################################################################

# TODO: @chingyulin: make cohort a class


time_s = time.time()
for i in tqdm(range(MC)):
    dZt = np.sqrt(dt) * np.random.randn(int(Nt - 1))
    (
        Deltabar,
        IntVec,
        Xt,
        Delta_s_t,
        Yt,
        Zt,
        f,
        tau,
        MaxDeltaTheta,
        DeltabarCondi,
        fCondi,

    ) = build_cohorts(
        dZt=dZt,
        Nt=Nt,
        dt=dt,
        rho=rho,
        Vhat=Vhat,
        mu_Y=mu_Y,
        sigma_Y=sigma_Y,
        beta=beta,
        T_hat=T_hat,
        nu=nu,
    )
    fMAT[i, :] = f
if time.time() - time_s > time_tolerance:
    print(f"It takes more than {time_tolerance}s to build up the cohorts")

# Initializing some variables
Mpaths = 100
Tsample = int(T_cohort / 100)
Nsamples = 100
stepcorr = int(Tsample / dt)
corrZport = np.zeros((Mpaths, Nsamples))
corrZMUs_t = np.zeros((Mpaths, Nsamples))
corrMU_sMUs_t = np.zeros((Mpaths, Nsamples))
corrMuSmuHat = np.zeros((Mpaths, 1))
fMAT = np.zeros((Mpaths, Nt))
mC = np.zeros((Mpaths, Nt))
sC = np.zeros((Mpaths, Nt))
DeltaHatMAT = np.zeros((Mpaths, Nt))
rMAT = np.zeros((Mpaths, Nt))
thetaMAT = np.zeros((Mpaths, Nt))
portMAT = np.zeros((Mpaths, Nt))
Zmat = np.zeros((Mpaths, Nt))

# Expected returns
muSMAT = np.zeros((Mpaths, Nt))  # Expected returns under the true measure
muSsMat = np.zeros(
    (Mpaths, Nt)
)  # Expected returns under the measure of the agent we track
muShatMAT = np.zeros(
    (Mpaths, Nt)
)  # Simple average of expected returns, or consensus belief
EtMAT = np.zeros((Mpaths, Nt))
VtMAT = np.zeros((Mpaths, Nt))
RxMAT = np.zeros((Mpaths, Nt))
muCst = np.zeros((Mpaths, Nsamples))
logmuCst = np.zeros((Mpaths, Nsamples))
sigCst = np.zeros((Mpaths, Nsamples))
stdCst = np.zeros((Mpaths, Nsamples))

# The main loop builds up the economy with a large number of cohorts, and simulates the stationary economy forward
for k in range(Mpaths):
    s = time.time()
    if k % 10 == 0:
        dZt = dt**0.5 * np.random.randn(int(Nt - 1))
        (
            Deltabar,
            IntVec,
            Xt,
            Delta_s_t,
            Yt,
            Zt,
            f,
            tau,
            MaxDeltaTheta,
            DeltabarCondi,
            fCondi,
        ) = build_cohorts(dZt, Nt, dt, rho, nu, Vbar, mu_Y, sigma_Y, beta, T_hat)

    dZforbias = np.diff(Zt)
    biasvec = dZforbias[-Npre:]
    dZt = dt**0.5 * np.random.randn(Nt)
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

    ) = simulate_cohorts(  # TODO: missing fCondi
        biasvec,
        dZt,
        Nt,
        tau,
        IntVec,
        Delta_s_t,
        dt,
        rho,
        nu,
        Vhat,
        mu_Y,
        sigma_Y,
        sigma_S,
        beta,
        T_hat,
        Npre,
        DeltabarCondi,
        fCondi,
    )

    RxMAT[k, :] = np.transpose(dR)
    EtMAT[k, :] = np.transpose(Et)
    VtMAT[k, :] = np.transpose(Et)
    DeltaHatMAT[k, :] = np.transpose(Deltabar2)
    rMAT[k, :] = np.transpose(r_t)
    thetaMAT[k, :] = np.transpose(theta_t)
    Zmat[k, :] = np.transpose(Zt)

    portMAT[k, :] = np.transpose(Port)

    muSMAT[k, :] = np.transpose(mu_S + rlog - r_t)
    muSsMat[k, :] = np.transpose(mu_S_t + rlog - r_t)
    muShatMAT[k, :] = np.transpose(muhat_S_t + rlog - r_t)

    mu_S = mu_S + rlog - r_t
    muhat_S_t = muhat_S_t + rlog - r_t
    mu_S_t = mu_S_t + rlog - r_t
    mC[k, :] = np.transpose(muC_s_t)
    sC[k, :] = np.transpose(sigmaC_s_t)
    corrMuSmuHat[k] = np.corrcoef(muhat_S_t, mu_S)[0, 1]
    fMAT[k, :] = np.mean(BIGf, axis=0)

    for l in range(Nsamples):
        a = int(l * stepcorr)
        b = int((l + 1) * stepcorr)
        corrZMUs_t[k, l] = np.corrcoef(Zt[a:b], mu_S_t[a:b])[0, 1]
        corrZport[k, l] = np.corrcoef(Zt[a:b], Port[a:b])[0, 1]
        corrMU_sMUs_t[k, l] = np.corrcoef(mu_S[a:b], mu_S_t[a:b])[0, 1]
        muCst[k, l] = np.mean(muC_s_t[a:b])
        logmuCst[k, l] = np.mean(muC_s_t[a:b]) - 0.5 * sum((sigmaC_s_t[a:b]) ** 2)
        sigCst[k, l] = np.mean(sigmaC_s_t[a:b])
        stdCst[k, l] = np.mean(abs(sigmaC_s_t[a:b]))
    print(time.time() - s)

MaxAge = 100
MaxAgeN = int(MaxAge / Tsample)
tperiod = range(Tsample, 100 + Tsample, Tsample)
meanZport = np.mean(corrZport, axis=0)
meanZmus_t = np.mean(corrZMUs_t, axis=0)

# Compute the mean values from the simulation
meanMus = np.mean(corrMU_sMUs_t, axis=0)
meanMuCst = np.mean(muCst, axis=0)
meanSCst = np.mean(sigCst, axis=0)
meanStdCst = np.mean(stdCst, axis=0)
meanLogMuCst = np.mean(logmuCst, axis=0)

# Figures
# Figure 1 in the paper
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4, 10))
ax1.plot(tperiod, meanMus[:MaxAgeN])
ax2.plot(tperiod, meanZmus_t[:MaxAgeN])

# Figure 2 in the paper
