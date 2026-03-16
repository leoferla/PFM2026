"""
Schwartz (1997) — Modelo 3: tres factores S + δ + r

Modelo base (Modelo 2 = Gibson-Schwartz = Gabillon 4.2):
    dS/S = mu dt + sigma1 dZ1
    dδ   = kappa*(alpha - δ) dt + sigma2 dZ2
    dZ1 dZ2 = rho_12 dt

Extension a Modelo 3:
    dS/S = (r_t - δ_t) dt + sigma1 dZ1   <- r_t y δ_t ESTOCASTICO
    dδ   = kappa*(alpha - δ) dt + sigma2 dZ2
    dr   = a*(b - r) dt + sigma3 dZ3      <- tasa de interes 

Correlaciones:
    dZ1 dZ2 = rho_12 dt    - spot vs CY
    dZ1 dZ3 = rho_13 dt    - spot vs tasa
    dZ2 dZ3 = rho_23 dt    - CY vs tasa
"""

import numpy as np
import matplotlib.pyplot as plt

def correlated_normals_3d(rng, n, rho_12, rho_13, rho_23):
    """
    Cholesky 3x3 general.
    Descomposicion:
        Z1 = U1
        Z2 = rho_12*U1 + L22*U2
        Z3 = rho_13*U1 + L32*U2 + L33*U3

    con:
        L22 = sqrt(1 - rho_12^2)
        L32 = (rho_23 - rho_13*rho_12) / L22
        L33 = sqrt(1 - rho_13^2 - L32^2)
    """

    U1 = rng.standard_normal(n)
    U2 = rng.standard_normal(n)
    U3 = rng.standard_normal(n)

    Z1 = U1

    L22 = np.sqrt(1.0 - rho_12**2)
    Z2  = rho_12 * U1 + L22 * U2

    L32 = (rho_23 - rho_13 * rho_12) / L22
    L33 = np.sqrt(max(0.0, 1.0 - rho_13**2 - L32**2))
    Z3  = rho_13 * U1 + L32 * U2 + L33 * U3

    return Z1, Z2, Z3

def simulate_schwartz_model3(S0=66.0, delta0=0.08, r0=0.03,
                              sigma1=0.30, sigma2=0.15, sigma3=0.015,
                              kappa=2.0, alpha=0.06,
                              a=0.5, b=0.03,
                              rho_12=-0.3, rho_13=0.15, rho_23=0.10,
                              T=2.0, n_steps=2*252, n_paths=50, seed=None):
    """
    Schwartz (1997) Modelo 3 — tres factores S + δ + r.
    """
    rng = np.random.default_rng(seed)
    dt    = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    S = np.zeros((n_paths, n_steps + 1))
    d = np.zeros((n_paths, n_steps + 1))
    r = np.zeros((n_paths, n_steps + 1))

    S[:, 0] = S0
    d[:, 0] = delta0
    r[:, 0] = r0

    # Coeficientes OU para δ
    exp_kdt  = np.exp(-kappa * dt)
    ou_std_d = sigma2 * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))

    # Coeficientes OU para r 
    exp_adt  = np.exp(-a * dt)
    ou_std_r = sigma3 * np.sqrt((1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a))

    for i in range(n_steps):
        Z1, Z2, Z3 = correlated_normals_3d(rng, n_paths,
                                            rho_12, rho_13, rho_23)

        # δ(t+dt) — OU
        d[:, i+1] = alpha + (d[:, i] - alpha) * exp_kdt + ou_std_d * Z2

        # r(t+dt) — OU
        r[:, i+1] = b + (r[:, i] - b) * exp_adt + ou_std_r * Z3

        # S(t+dt) — GBM
        drift = (r[:, i] - d[:, i] - 0.5 * sigma1**2) * dt
        S[:, i+1] = S[:, i] * np.exp(drift + sigma1 * np.sqrt(dt) * Z1)

    return times, S, d, r


def future_approx(S_t, delta_t, r_t, tau):
    """
    Futuro aproximado:
        F_approx = S_t * exp((r_t - δ_t) * tau)
    """
    return S_t * np.exp((r_t - delta_t) * tau)

def future_exact(S, delta, r_val, T,
                  sigma1, sigma2, sigma3,
                  kappa, alpha_hat, a, m_star,
                  rho_12, rho_13, rho_23):
    """
    Futuro EXACTO — Schwartz (1997), ecuaciones (26)-(28).

    ln F(S, δ, r, T) = ln S - δ * B_kappa + r * B_a + C(T)

    donde:
        B_kappa = (1 - exp(-kappa*T)) / kappa          ec. (27)
        B_a     = (1 - exp(-a*T)) / a                  ec. (27)
        C(T)    = 6 sumandos 

    """
    # Funciones B ec. (27) 

    B_kappa = (1.0 - np.exp(-kappa * T)) / kappa
    B_a     = (1.0 - np.exp(-a * T)) / a

    # Abreviaturas para C(T) 

    e_kT   = np.exp(-kappa * T)          # exp(-κT)
    e_2kT  = np.exp(-2.0 * kappa * T)    # exp(-2κT)
    e_aT   = np.exp(-a * T)              # exp(-aT)
    e_2aT  = np.exp(-2.0 * a * T)        # exp(-2aT)
    e_kaT  = np.exp(-(kappa + a) * T)    # exp(-(κ+a)T)

    # C(T) — ecuacion (28), sumando por sumando
    #
    # Sumando 1:
    c1 = (kappa * alpha_hat + sigma1 * sigma2 * rho_12) \
         * ((1.0 - e_kT) - kappa * T) / kappa**2

    # Sumando 2:
    c2 = -sigma2**2 \
         * (4.0*(1.0 - e_kT) - (1.0 - e_2kT) - 2.0*kappa*T) \
         / (4.0 * kappa**3)

    # Sumando 3:
    c3 = -(a * m_star + sigma1 * sigma3 * rho_13) \
         * ((1.0 - e_aT) - a * T) / a**2

    # Sumando 4:
    c4 = -sigma3**2 \
         * (4.0*(1.0 - e_aT) - (1.0 - e_2aT) - 2.0*a*T) \
         / (4.0 * a**3)

    # Sumando 5+6:  σ2*σ3*ρ2 * (dos fracciones)
    frac_A = ((1.0 - e_kT) + (1.0 - e_aT) - (1.0 - e_kaT)) \
             / (kappa * a * (kappa + a))

    frac_B = (kappa**2 * (1.0 - e_aT) + a**2 * (1.0 - e_kT)
              - kappa * a**2 * T - a * kappa**2 * T) \
             / (kappa**2 * a**2 * (kappa + a))

    c56 = sigma2 * sigma3 * rho_23 * (frac_A + frac_B)



    C_T = c1 + c2 + c3 + c4 + c56

    ln_F = np.log(S) - delta * B_kappa + r_val * B_a + C_T

    return np.exp(ln_F)

def main():


    S0      = 66.0
    delta0  = 0.08
    r0      = 0.03
    sigma1  = 0.30
    sigma2  = 0.15
    sigma3  = 0.015
    kappa   = 2.0
    alpha   = 0.06
    a       = 0.5
    b       = 0.03
    rho_12  = -0.3
    rho_13  = 0.15
    rho_23  = 0.10

    T       = 2.0
    n_steps = 2 * 252
    n_paths = 5000

    # 1) Simulacion

    times, S, d, r = simulate_schwartz_model3(
        S0=S0, delta0=delta0, r0=r0,
        sigma1=sigma1, sigma2=sigma2, sigma3=sigma3,
        kappa=kappa, alpha=alpha, a=a, b=b,
        rho_12=rho_12, rho_13=rho_13, rho_23=rho_23,
        T=T, n_steps=n_steps, n_paths=n_paths, seed=None)

    # 2) Trayectorias

    n_show = 20

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(n_show):
        axes[0].plot(times, S[i], linewidth=0.8, alpha=0.6)
    axes[0].set_xlabel("t (años)")
    axes[0].set_ylabel("S(t)")
    axes[0].set_title("Spot S(t)")
    axes[0].grid(True)

    for i in range(n_show):
        axes[1].plot(times, d[i] * 100, linewidth=0.8, alpha=0.5,
                     color="darkorange")
    axes[1].axhline(alpha * 100, color="black", linestyle="--",
                    linewidth=1.5, label=f"alpha = {alpha:.0%}")
    axes[1].set_xlabel("t (años)")
    axes[1].set_ylabel("δ(t)  (%)")
    axes[1].set_title("Convenience yield δ(t) — OU")
    axes[1].legend()
    axes[1].grid(True)

    for i in range(n_show):
        axes[2].plot(times, r[i] * 100, linewidth=0.8, alpha=0.5,
                     color="steelblue")
    axes[2].axhline(b * 100, color="black", linestyle="--",
                    linewidth=1.5, label=f"b = {b:.0%}")
    axes[2].set_xlabel("t (años)")
    axes[2].set_ylabel("r(t)  (%)")
    axes[2].set_title("Tasa de interés r(t) — OU")
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle("Schwartz (1997) Modelo 3: tres factores S + δ + r", y=1.01)
    plt.tight_layout()
    plt.show()

    # 3) Futuro vs Subyacente 

    t_obs = 1.0
    tau   = 1.0
    idx   = int(np.argmin(np.abs(times - t_obs)))

    S_obs = S[:, idx]
    d_obs = d[:, idx]
    r_obs = r[:, idx]

    F_approx_obs = future_approx(S_obs, d_obs, r_obs, tau=tau)
    F_exact_obs  = future_exact(S_obs, d_obs, r_obs, T=tau,
                                 sigma1=sigma1, sigma2=sigma2, sigma3=sigma3,
                                 kappa=kappa, alpha_hat=alpha, a=a, m_star=b,
                                 rho_12=rho_12, rho_13=rho_13, rho_23=rho_23)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(S_obs, F_approx_obs, s=5, alpha=0.2, color="gray",
               label="F approx = S·exp((r-δ)·τ)")
    ax.scatter(S_obs, F_exact_obs, s=5, alpha=0.2, color="crimson",
               label="F exacto (ec. 26-28)")
    ax.set_xlabel(f"Spot S(t={t_obs})")
    ax.set_ylabel(f"Futuro F(t={t_obs}, τ={tau})")
    ax.set_title("Futuro vs Spot: aproximado vs exacto")
    ax.legend(markerscale=3)
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Diferencia media entre ambos
    diff_pct = ((F_exact_obs - F_approx_obs) / F_approx_obs * 100)
    print(f"\nDiferencia F_exacto vs F_approx:")
    print(f"  Media = {diff_pct.mean():+.4f}%")
    print(f"  Std   = {diff_pct.std():.4f}%")

if __name__ == "__main__":
    main()
