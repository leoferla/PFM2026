"""
Ecuación (9) - Simulación numérica (Monte Carlo) de:
    dS/S = mu dt + sigma1 dZ1
    dδ   = kappa (alpha - δ) dt + sigma2 dZ2
    corr(dZ1, dZ2) = rho dt

Y construcción de un FUTURO aproximado usando el estado (S_t, δ_t):
    F_approx(S, tau) = S_t * exp((r - δ_t) * tau)

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


def simulate_eq9(S0=66.0, delta0=0.08, mu=0.05, sigma1=0.30, kappa=2.0, alpha=0.06, sigma2=0.15,
    rho=-0.3, T=2.0, n_steps=2*252, n_paths=50, seed=None):
    """
    Simula trayectorias de S(t) y δ(t).

    - S(t): GBM (lognormal).
    - δ(t): Uhlenbeck & Ornstein (OU) (1930) (mean-reverting).
    - Z1 y Z2: ruidos normales correlacionados (rho).
    """
    rng = np.random.default_rng(seed)

    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    S = np.zeros((n_paths, n_steps + 1))
    d = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0
    d[:, 0] = delta0

    # Coeficientes del OU
    exp_kdt = np.exp(-kappa * dt)
    ou_std  = sigma2 * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
    
    # Coeficiente GBM
    c1 = (mu - 0.5 * sigma1**2) * dt
    c2 = sigma1 * np.sqrt(dt)

    for i in range(n_steps):
        # 1) Ruido aleatorio correlacionado
        Z1, Z2 = correlated_normals(rng, n_paths, rho)

        # 2) Actualizacion δ(t) (OU)
        d[:, i+1] = alpha + (d[:, i] - alpha) * exp_kdt + ou_std * Z2

        # 3) Actualizacion S(t) (GBM)
        S[:, i + 1] = S[:, i] * np.exp (c1 + c2 * Z1)

    return times, S, d


def future_approx(S_t, delta_t, r, tau):
    """
    Futuro aproximado a partir del estado (S_t, δ_t):
        F_approx = S_t * exp((r - δ_t) * tau)
    """
    return S_t * np.exp((r - delta_t) * tau)


def main():
    
    # Parámetros

    S0 = 66.0
    delta0 = 0.08

    mu = 0.05
    sigma1 = 0.30

    kappa = 2.0
    alpha = 0.06
    sigma2 = 0.15

    rho = -0.3

    r = 0.03

    T = 2.0
    n_steps = 2 * 252
    n_paths = 5000

    # 1) Simulación Monte Carlo
    
    times, S_paths, d_paths = simulate_eq9(S0=S0, delta0=delta0, mu=mu, sigma1=sigma1,
        kappa=kappa, alpha=alpha, sigma2=sigma2, rho=rho, T=T, n_steps=n_steps, n_paths=n_paths,
        seed=None)

    # 2) Trayectorias S(t)

    plt.figure(figsize=(9, 5))
    for i in range(min(20, n_paths)):
        plt.plot(times, S_paths[i], linewidth=1)
    plt.xlabel("Tiempo t (años)")
    plt.ylabel("Spot S(t)")
    plt.title("Ecuación (9): trayectorias simuladas de S(t)")
    plt.grid(True)
    plt.show()

    # 3) Trayectorias δ(t)

    plt.figure(figsize=(9, 5))
    for i in range(min(20, n_paths)):
        plt.plot(times, d_paths[i], linewidth=1)
    plt.xlabel("Tiempo t (años)")
    plt.ylabel("Convenience yield δ(t)")
    plt.title("Ecuación (9): trayectorias simuladas de δ(t) (OU mean-reverting)")
    plt.grid(True)
    plt.show()

    # 4) Se elige un tiempo de observación t_obs
    #    y un vencimiento tau para el futuro

    t_obs = 1.0    # en años, cuando haces la "foto"
    tau = 1.0      # futuro con 1 año de vencimiento desde t_obs
    
    # T = t_obs + tau

    idx = int(np.argmin(np.abs(times - t_obs)))
    S_obs = S_paths[:, idx]
    d_obs = d_paths[:, idx]

    # 5) FUTURO aproximado en t_obs

    F_obs = future_approx(S_obs, d_obs, r=r, tau=tau) # F(S,t)

    # 6) Futuro vs Subyacente 

    plt.figure(figsize=(9, 5))
    plt.scatter(S_obs, F_obs, s=40, alpha=0.7)
    plt.xlabel(f"Spot S(t_obs={times[idx]:.2f})")
    plt.ylabel(f"F_approx(t_obs, tau={tau})")
    plt.title("Futuro aproximado vs Spot (δ estocástico) - F vs S")
    plt.grid(True)
    plt.show()

    # Resumen numérico
    print("Resumen en t_obs:")
    print(f"t_obs usado: {times[idx]:.4f} años")
    print(f"S(t_obs): media={S_obs.mean():.4f} | std={S_obs.std():.4f}")
    print(f"δ(t_obs): media={d_obs.mean():.4f} | std={d_obs.std():.4f}")
    print(f"F_approx(t_obs): media={F_obs.mean():.4f} | std={F_obs.std():.4f}")


if __name__ == "__main__":
    main()
