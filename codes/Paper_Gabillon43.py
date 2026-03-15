"""
Gabillon (1991) — Sección 4.3
Modelo de dos factores: Precio Spot S(t) y Precio de Largo Plazo L(t)

Ecuación 11 del modelo:
    dS = mu_S * S dt + sigma_S * S dZ1      [GBM]
    dL = mu_L * L dt + sigma_L * L dZ2      [GBM]
    dZ1 dZ2 = rho dt

Precio del futuro — Ecuación 26:

    F(S, L, tau) = A(tau) * S^B(tau) * L^(1 - B(tau))

    B(tau) = exp(-beta * tau)
    v      = sigma_S^2 + sigma_L^2 - 2*rho*sigma_S*sigma_L
    A(tau) = exp( v / (4*beta) * (exp(-beta*tau) - exp(-2*beta*tau)) )

Propiedades de la fórmula:
    tau -> 0   =>  F -> S     (el futuro converge al spot en el corto plazo)
    tau -> inf =>  F -> L     (el futuro converge al largo plazo)

    El peso del spot es B(tau) = exp(-beta*tau), decreciente con tau.

Diferencia clave con la sección 4.2:
    En 4.2, el segundo factor es el convenience yield delta(t).
    En 4.3, el segundo factor es L(t).
"""

import numpy as np
import matplotlib.pyplot as plt


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def correlated_normals(rng, n, rho):
    """Normales estándar correlacionadas con coeficiente rho.
    Descomposición de Cholesky 2x2:
        Z1 = U1
        Z2 = rho*U1 + sqrt(1-rho^2)*U2
    """
    U1 = rng.standard_normal(n)
    U2 = rng.standard_normal(n)
    Z1 = U1
    Z2 = rho * U1 + np.sqrt(max(0.0, 1.0 - rho**2)) * U2
    return Z1, Z2

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def simulate_gabillon43(S0, L0, mu_S, mu_L, sigma_S, sigma_L, rho,
                        T, n_steps, n_paths, seed=None):
    """
    Simula trayectorias de S(t) y L(t).

    Ambos procesos son GBM:
        S(t+dt) = S(t) * exp((mu_S - 0.5*sigma_S^2)*dt + sigma_S*sqrt(dt)*Z1)
        L(t+dt) = L(t) * exp((mu_L - 0.5*sigma_L^2)*dt + sigma_L*sqrt(dt)*Z2)

    Inputs:
        S0, L0          : precios iniciales del spot y largo plazo
        mu_S, mu_L      : drifts
        sigma_S, sigma_L: volatilidades
        rho             : correlación entre dZ1 y dZ2
        T               : horizonte (años)
        n_steps         : pasos de tiempo
        n_paths         : trayectorias Monte Carlo
        seed            : semilla aleatoria

    Outputs:
        times : vector de tiempos (n_steps+1,)
        S     : trayectorias del spot      (n_paths, n_steps+1)
        L     : trayectorias del LP price  (n_paths, n_steps+1)
    """
    rng = np.random.default_rng(seed)

    dt    = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    S = np.zeros((n_paths, n_steps + 1))
    L = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    L[:, 0] = L0

    # Coeficientes constantes
    cS1 = (mu_S - 0.5 * sigma_S**2) * dt
    cS2 = sigma_S * np.sqrt(dt)
    cL1 = (mu_L - 0.5 * sigma_L**2) * dt
    cL2 = sigma_L * np.sqrt(dt)

    for i in range(n_steps):
        Z1, Z2 = correlated_normals(rng, n_paths, rho)
        S[:, i + 1] = S[:, i] * np.exp(cS1 + cS2 * Z1)
        L[:, i + 1] = L[:, i] * np.exp(cL1 + cL2 * Z2)

    return times, S, L

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def futures_price_gabillon43(S, L, tau, beta, sigma_S, sigma_L, rho):
    """
    Precio del futuro — Gabillon (1991) sección 4.3.

        F(S, L, tau) = A(tau) * S^B(tau) * L^(1 - B(tau))

        B(tau) = exp(-beta * tau)
        v      = sigma_S^2 + sigma_L^2 - 2*rho*sigma_S*sigma_L
        A(tau) = exp( v/(4*beta) * (exp(-beta*tau) - exp(-2*beta*tau)) )

    Inputs:
        S, L            : spot y largo plazo (escalares o arrays)
        tau             : tiempo a vencimiento en años
        beta            : velocidad de convergencia de S hacia L
        sigma_S, sigma_L: volatilidades
        rho             : correlación

    Output:
        F : precio del futuro
    """
    B = np.exp(-beta * tau)
    v = sigma_S**2 + sigma_L**2 - 2.0 * rho * sigma_S * sigma_L
    A = np.exp(v / (4.0 * beta) * (np.exp(-beta * tau) - np.exp(-2.0 * beta * tau)))
    return A * (S ** B) * (L ** (1.0 - B))

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def main():

    # Parámetros

    S0      = 66.0    # precio spot inicial
    L0      = 60.0    # precio largo plazo inicial

    mu_S    = 0.05    # drift del spot 
    mu_L    = 0.02    # drift del largo plazo
    sigma_S = 0.30    # volatilidad del spot (alta — vencimientos próximos)
    sigma_L = 0.10    # volatilidad del largo plazo (baja — vencimientos lejanos)
    rho     = 0.70    # correlación positiva
    beta    = 1.50    # velocidad de convergencia de S hacia L

    T       = 2.0
    n_steps = 2 * 252
    n_paths = 5000

    # 1) Simulación Monte Carlo

    times, S_paths, L_paths = simulate_gabillon43(
        S0=S0, L0=L0, mu_S=mu_S, mu_L=mu_L,
        sigma_S=sigma_S, sigma_L=sigma_L, rho=rho,
        T=T, n_steps=n_steps, n_paths=n_paths, seed=42)

    # 2) Trayectorias S(t) y L(t)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for i in range(min(20, n_paths)):
        axes[0].plot(times, S_paths[i], linewidth=0.8, alpha=0.7)
    axes[0].set_xlabel("Tiempo t (años)")
    axes[0].set_ylabel("Spot S(t)")
    axes[0].set_title("Gabillon 4.3: trayectorias S(t)")
    axes[0].grid(True)

    for i in range(min(20, n_paths)):
        axes[1].plot(times, L_paths[i], linewidth=0.8, alpha=0.7, color="darkorange")
    axes[1].set_xlabel("Tiempo t (años)")
    axes[1].set_ylabel("Largo plazo L(t)")
    axes[1].set_title("Gabillon 4.3: trayectorias L(t)")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    # 3) Curva de futuros teórica en t=0

    taus = np.linspace(0.01, 5.0, 200)

    F_curve = futures_price_gabillon43(S0, L0, taus, beta, sigma_S, sigma_L, rho)

    plt.figure(figsize=(9, 5))
    plt.plot(taus, F_curve, linewidth=2, label=f"F(S0, L0, τ)  β={beta}")
    plt.axhline(S0, color="steelblue",  linestyle="--", linewidth=1, label=f"Spot S0 = {S0}")
    plt.axhline(L0, color="darkorange", linestyle="--", linewidth=1, label=f"Largo plazo L0 = {L0}")
    plt.xlabel("Tiempo a vencimiento τ (años)")
    plt.ylabel("Futuro F(0, τ)")
    plt.title("Gabillon 4.3: curva de futuros")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4) Efecto de beta — velocidad de convergencia de S hacia L

    betas = [0.3, 0.8, 1.5, 3.0, 6.0]

    plt.figure(figsize=(9, 5))
    for b in betas:
        F_b = futures_price_gabillon43(S0, L0, taus, b, sigma_S, sigma_L, rho)
        plt.plot(taus, F_b, linewidth=2, label=f"β = {b}")
    plt.axhline(S0, color="steelblue",  linestyle=":", linewidth=1, label=f"S0 = {S0}")
    plt.axhline(L0, color="darkorange", linestyle=":", linewidth=1, label=f"L0 = {L0}")
    plt.xlabel("τ (años)")
    plt.ylabel("F(0, τ)")
    plt.title("Efecto de β sobre la curva de futuros (Gabillon 4.3)")
    plt.legend()
    plt.grid(True)
    plt.show()


    # 5) Contango vs Backwardation según S0 vs L0

    scenarios = [
        {"S0": 80.0, "L0": 60.0, "label": "Backwardation  (S0 > L0)", "color": "firebrick"},
        {"S0": 60.0, "L0": 60.0, "label": "Flat  (S0 = L0)",           "color": "gray"},
        {"S0": 50.0, "L0": 60.0, "label": "Contango  (S0 < L0)",       "color": "steelblue"},
    ]

    plt.figure(figsize=(9, 5))
    for sc in scenarios:
        F_sc = futures_price_gabillon43(sc["S0"], sc["L0"], taus, beta, sigma_S, sigma_L, rho)
        plt.plot(taus, F_sc, linewidth=2, label=sc["label"], color=sc["color"])
    plt.xlabel("τ (años)")
    plt.ylabel("F(0, τ)")
    plt.title(f"Gabillon 4.3: backwardation / contango según S0 vs L0  (β={beta})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 6) Futuro vs Spot y vs Largo Plazo en t_obs (nube de puntos)

    t_obs = 1.0
    tau   = 1.0

    idx   = int(np.argmin(np.abs(times - t_obs)))
    S_obs = S_paths[:, idx]
    L_obs = L_paths[:, idx]
    F_obs = futures_price_gabillon43(S_obs, L_obs, tau, beta, sigma_S, sigma_L, rho)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(S_obs, F_obs, s=6, alpha=0.3, color="steelblue")
    axes[0].set_xlabel(f"Spot S(t={t_obs})")
    axes[0].set_ylabel(f"Futuro F(t={t_obs}, τ={tau})")
    axes[0].set_title("F vs S")
    axes[0].grid(True)

    axes[1].scatter(L_obs, F_obs, s=6, alpha=0.3, color="darkorange")
    axes[1].set_xlabel(f"Largo plazo L(t={t_obs})")
    axes[1].set_ylabel(f"Futuro F(t={t_obs}, τ={tau})")
    axes[1].set_title("F vs L")
    axes[1].grid(True)

    plt.suptitle(f"Gabillon 4.3: dispersión del futuro en t={t_obs}, τ={tau}", y=1.01)
    plt.tight_layout()
    plt.show()

    # 7) Comparación directa 4.2 vs 4.3
    #    Gabillon 4.2: F = S * exp((r - delta) * tau)  [delta constante]
    #    Gabillon 4.3: F = A(tau) * S^B(tau) * L^(1-B(tau))

    r     = 0.03
    delta = 0.08   # convenience yield constante

    F_42 = S0 * np.exp((r - delta) * taus)
    F_43 = futures_price_gabillon43(S0, L0, taus, beta, sigma_S, sigma_L, rho)

    plt.figure(figsize=(9, 5))
    plt.plot(taus, F_42, linewidth=2, linestyle="--",
             label=f"Gabillon 4.2: S·exp((r−δ)τ)  [δ={delta}]")
    plt.plot(taus, F_43, linewidth=2,
             label=f"Gabillon 4.3: A(τ)·S^B(τ)·L^(1−B(τ))  [β={beta}]")
    plt.axhline(L0, color="darkorange", linestyle=":", linewidth=1, label=f"L0 = {L0}")
    plt.xlabel("τ (años)")
    plt.ylabel("F(0, τ)")
    plt.title("Gabillon 4.2 vs 4.3: comparación de curvas de futuros")
    plt.legend()
    plt.grid(True)
    plt.show()


# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if __name__ == "__main__": main()

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
