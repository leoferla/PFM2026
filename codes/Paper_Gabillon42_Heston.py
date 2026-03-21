"""
Ecuación (9) de Gabillon / Gibson-Schwartz — extendido con Heston (1993)

Modelo original (ec. 9):
    dS/S = mu dt + sigma1 dZ1
    dδ   = k(alpha - δ) dt + sigma2 dZ2
    dZ1 dZ2 = rho dt

Extension con volatilidad estocástica (Heston 1993, ec. 6.4a-b):
    dS/S = mu dt + sqrt(v_t) dZ1         <- sqrt(v_t) estocastica
    dδ   = k(alpha - δ) dt + sigma2 dZ2
    dv_t = kappa_v*(m - v_t) dt + xi*sqrt(v_t) dZ3   <- varianza del spot

Correlaciones:
    dZ1 dZ2 = rho dt       <- spot vs CY
    dZ1 dZ3 = rho_sv dt    <- spot vs vol
    dZ2 dZ3 = 0            <- CY independiente de la varianza del spot (para simplificar)
"""
import numpy as np
import matplotlib.pyplot as plt

def correlated_normals(rng, n, rho):
    """Normales estándar correlacionadas con rho (estocastico2324.pdf pág. 21)"""
    U1 = rng.standard_normal(n)
    U2 = rng.standard_normal(n)
    Z1 = U1
    Z2 = rho * U1 + np.sqrt(max(0.0, 1.0 - rho**2)) * U2
    return Z1, Z2

def correlated_normals_3d(rng, n, rho_12, rho_13, rho_23=0):
    """
    Estructura de correlaciones:
        dZ1 dZ2 = rho_12 dt   (spot vs CY )
        dZ1 dZ3 = rho_13 dt   (spot vs volatilidad)
        dZ2 dZ3 = 0           (CY y volatilidad son independientes)

    Descomposicion de Cholesky 3x3:
        Z1 = U1
        Z2 = rho_12*U1 + sqrt(1 - rho_12^2)*U2
        Z3 = rho_13*U1 + c32*U2 + c33*U3

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

def simulate_eq9(S0=66.0, delta0=0.08, mu=0.05, sigma1=0.30,
                  kappa=2.0, alpha=0.06, sigma2=0.15,
                  rho=-0.3, T=2.0, n_steps=2*252, n_paths=50, seed=None):
    """
    Modelo original de Gibson-Schwartz (Gabillond 4.2).
    """
    rng = np.random.default_rng(seed)
    dt    = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    S = np.zeros((n_paths, n_steps + 1))
    d = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    d[:, 0] = delta0

    exp_kdt = np.exp(-kappa * dt)
    ou_std  = sigma2 * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
    c1      = (mu - 0.5 * sigma1**2) * dt
    c2      = sigma1 * np.sqrt(dt)

    for i in range(n_steps):
        Z1, Z2 = correlated_normals(rng, n_paths, rho)
        d[:, i+1] = alpha + (d[:, i] - alpha) * exp_kdt + ou_std * Z2
        S[:, i+1] = S[:, i] * np.exp(c1 + c2 * Z1)

    return times, S, d

def simulate_eq9_heston(S0=66.0, delta0=0.08, v0=0.09,
                         mu=0.05, sigma2=0.15, kappa=2.0, alpha=0.06,
                         kappa_v=2.0, m=0.09, xi=0.40,
                         rho=-0.3, rho_sv=-0.50, rho_cv=0,
                         T=2.0, n_steps=2*252, n_paths=50, seed=None):
    """
    Extension de simulate_eq9 con volatilidad estocástica tipo Heston.

    Añade el proceso CIR para la varianza v_t del spot:
        dv_t = kappa_v*(m - v_t)*dt + xi*sqrt(v_t)*dZ3

    sigma1 del modelo original se reemplaza por sqrt(v_t).

    Parametros nuevos respecto al original:
        v0      : varianza inicial del spot (sigma_S_0 = sqrt(v0))
        kappa_v : velocidad de reversion de v_t
        m       : nivel de largo plazo de v_t  (sigma_S_LP = sqrt(m))
        xi      : volatilidad de la varianza

    Condicion de Feller: 2*kappa_v*m > xi^2  (para que v_t > 0)
    """
    rng = np.random.default_rng(seed)

    dt    = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    S = np.zeros((n_paths, n_steps + 1))
    d = np.zeros((n_paths, n_steps + 1))
    v = np.zeros((n_paths, n_steps + 1))   # varianza del spot

    S[:, 0] = S0
    d[:, 0] = delta0
    v[:, 0] = v0

    # Coeficientes del OU
    exp_kdt = np.exp(-kappa * dt)
    ou_std  = sigma2 * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))

    for i in range(n_steps):
        # 1) Tres ruidos correlacionados
        #    Z1: spot,  Z2: CY,  Z3: varianza (Heston)
        Z1, Z2, Z3 = correlated_normals_3d(rng, n_paths, rho, rho_sv, rho_cv)

        # 2) δ(t) — OU
        d[:, i+1] = alpha + (d[:, i] - alpha) * exp_kdt + ou_std * Z2

        # 3) Varianza del spot — CIR de Heston (pag. 5 bonosunfactor2324.pdf))
        #    dv_t = kappa_v*(m - v_t)*dt + xi*sqrt(v_t)*sqrt(dt)*Z3
        v_now = np.maximum(v[:, i], 0.0)   # Evitar v<0
        v[:, i+1] = np.maximum(
            v_now + kappa_v * (m - v_now) * dt + xi * np.sqrt(v_now) * np.sqrt(dt) * Z3,
            0.0)

        # 4) S(t) — GBM con vol sqrt(v_t) en lugar de sigma1 fija
        S[:, i+1] = S[:, i] * np.exp(
            (mu - 0.5 * v_now) * dt + np.sqrt(v_now) * np.sqrt(dt) * Z1)

    return times, S, d, v
                             
def main():

    # Parámetros del modelo original (ec. 9)

    S0      = 66.0
    delta0  = 0.08
    mu      = 0.05
    sigma1  = 0.30    # vol del spot en modelo original (sigma1 fija)
    kappa   = 2.0
    alpha   = 0.06
    sigma2  = 0.15
    rho     = -0.3    # corr(spot, CY)
    r       = 0.03

    T       = 2.0
    n_steps = 2 * 252
    n_paths = 5000

    # Parámetros nuevos — Heston

    v0      = sigma1**2   # varianza inicial = sigma1^2 del modelo original
    m       = sigma1**2   # nivel LP de la varianza = mismo que sigma1^2
    kappa_v = 2.0         # velocidad reversion de v_t
    xi      = 0.40        # vol-of-vol
    rho_sv  = -0.50
    rho_cv  = 0

    # Condicion de Feller - Si 2*kappa_v*m >= xi*2 y r(0) > 0 entonces r(t) > 0.
    feller = 2.0 * kappa_v * m
    print(f"Condicion de Feller: 2*kappa_v*m = {feller:.3f}  >  xi^2 = {xi**2:.3f}  "
          f"{'OK' if feller > xi**2 else 'VIOLADA — v_t puede llegar a 0'}")

    # 1) Simulacion — modelo original (eq. 9)

    times_orig, S_orig, d_orig = simulate_eq9(
        S0=S0, delta0=delta0, mu=mu, sigma1=sigma1,
        kappa=kappa, alpha=alpha, sigma2=sigma2,
        rho=rho, T=T, n_steps=n_steps, n_paths=n_paths, seed=None)

    # 2) Simulacion — extension Heston

    times, S_paths, d_paths, v_paths = simulate_eq9_heston(
        S0=S0, delta0=delta0, v0=v0,
        mu=mu, sigma2=sigma2, kappa=kappa, alpha=alpha,
        kappa_v=kappa_v, m=m, xi=xi,
        rho=rho, rho_sv=rho_sv, rho_cv=rho_cv,
        T=T, n_steps=n_steps, n_paths=n_paths, seed=None)

    # 3) Trayectorias: comparacion directa original vs Heston

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i in range(min(20, n_paths)):
        axes[0].plot(times_orig, S_orig[i], linewidth=0.8, alpha=0.7)
    axes[0].set_xlabel("Tiempo t (años)")
    axes[0].set_ylabel("Spot S(t)")
    axes[0].set_title("Ecuación (9) original\nsigma1 = constante")
    axes[0].grid(True)

    for i in range(min(20, n_paths)):
        axes[1].plot(times, S_paths[i], linewidth=0.8, alpha=0.7)
    axes[1].set_xlabel("Tiempo t (años)")
    axes[1].set_ylabel("Spot S(t)")
    axes[1].set_title("Ecuación (9) + Heston\nsigma_S(t) = sqrt(v_t) estocástica")
    axes[1].grid(True)

    plt.suptitle("Gibson-Schwartz: original vs + Heston", y=1.01)
    plt.tight_layout()
    plt.show()

    # 4) Trayectorias de v_t (nuevo factor de Heston)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Varianza v_t
    for i in range(min(20, n_paths)):
        axes[0].plot(times, v_paths[i], linewidth=0.8, alpha=0.5,
                     color="darkorange")
    axes[0].axhline(m, color="black", linestyle="--", linewidth=1.5,
                    label=f"Nivel LP m = {m:.4f}  (sigma_LP={np.sqrt(m):.0%})")
    axes[0].axhline(v0, color="steelblue", linestyle=":",
                    linewidth=1.2, label=f"v0 = {v0:.4f}  (sigma0={np.sqrt(v0):.0%})")
    axes[0].set_xlabel("Tiempo t (años)")
    axes[0].set_ylabel("v_t = varianza del spot")
    axes[0].set_title("Varianza v_t — proceso CIR (Heston)")
    axes[0].legend()
    axes[0].grid(True)

    # Volatilidad estocastica sqrt(v_t)
    for i in range(min(20, n_paths)):
        axes[1].plot(times, np.sqrt(v_paths[i]) * 100, linewidth=0.8,
                     alpha=0.5, color="seagreen")
    axes[1].axhline(np.sqrt(m) * 100, color="black", linestyle="--",
                    linewidth=1.5, label=f"LP = {np.sqrt(m):.0%}")
    axes[1].axhline(sigma1 * 100, color="steelblue", linestyle=":",
                    linewidth=1.2, label=f"sigma1 original = {sigma1:.0%}")
    axes[1].set_xlabel("Tiempo t (años)")
    axes[1].set_ylabel("sqrt(v_t) = sigma_S(t)  (%)")
    axes[1].set_title("Volatilidad del spot sqrt(v_t)")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle("Heston: v_t (varianza del spot)", y=1.01)
    plt.tight_layout()
    plt.show()

    # 5) Las tres trayectorias juntas S(t), delta(t), sqrt(v_t) 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i in range(min(20, n_paths)):
        axes[0].plot(times, S_paths[i], linewidth=0.8, alpha=0.6)
    axes[0].set_xlabel("t (años)")
    axes[0].set_ylabel("S(t)")
    axes[0].set_title("Spot S(t)")
    axes[0].grid(True)

    for i in range(min(20, n_paths)):
        axes[1].plot(times, d_paths[i] * 100, linewidth=0.8, alpha=0.5,
                     color="darkorange")
    axes[1].axhline(alpha * 100, color="black", linestyle="--",
                    linewidth=1.5, label=f"alpha = {alpha:.0%}")
    axes[1].set_xlabel("t (años)")
    axes[1].set_ylabel("delta(t)  (%)")
    axes[1].set_title("Convenience yield δ(t) — OU")
    axes[1].legend()
    axes[1].grid(True)

    for i in range(min(20, n_paths)):
        axes[2].plot(times, np.sqrt(v_paths[i]) * 100, linewidth=0.8,
                     alpha=0.5, color="seagreen")
    axes[2].axhline(np.sqrt(m) * 100, color="black", linestyle="--",
                    linewidth=1.5, label=f"LP = {np.sqrt(m):.0%}")
    axes[2].set_xlabel("t (años)")
    axes[2].set_ylabel("sqrt(v_t)  (%)")
    axes[2].set_title("Volatilidad del spot sqrt(v_t) — Heston CIR")
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle("Ecuación (9) + Heston: tres factores S + δ + v_t", y=1.01)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
