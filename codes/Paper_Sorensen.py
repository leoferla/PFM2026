"""
Sørensen, C. (2002), "Modeling seasonality in agricultural commodity futures"

El log-precio se descompone en TRES piezas (ec. 1 del paper):

    ln S(t) = s(t) + x(t) + z(t)

    s(t) : ESTACIONALIDAD determinista (ec. 2):
           s(t) = sum_{k=1}^{K} [ gamma_k*cos(2*pi*k*t) + gamma_k^* *sin(2*pi*k*t) ]

    x(t) : GBM (ec. 3):
           dx = (mu - 0.5*sigma^2) dt + sigma dW1

    z(t) : OU (ec. 4).
           dz = -kappa*z dt + v dW2     
"""
import numpy as np
import matplotlib.pyplot as plt


def correlated_normals(rng, n, rho):
    """Normales estandar correlacionadas con rho."""
    U1 = rng.standard_normal(n)
    U2 = rng.standard_normal(n)
    Z1 = U1
    Z2 = rho * U1 + np.sqrt(max(0.0, 1.0 - rho**2)) * U2
    return Z1, Z2


def s_estacional(t, gammas, K):
    """
    Estacionalidad (ec. 2). gammas se reorganiza en una matriz (K, 2):
    la fila k-1 tiene [gamma_k, gamma_k*] = [coef. coseno, coef. seno].
    """
    G = gammas.reshape(K, 2)
    s = np.zeros_like(np.asarray(t, dtype=float))
    for k in range(1, K + 1):
        s += G[k-1, 0] * np.cos(2.0 * np.pi * k * t) \
           + G[k-1, 1] * np.sin(2.0 * np.pi * k * t)
    return s


def simulate_sorensen(S0=11.7, mu=0.02, sigma=0.20, kappa=0.95, v=0.25, rho=-0.3,
                      z0=0.0, gammas=None, K=2, T=2.0, n_steps=2*252, n_paths=50,
                      seed=None):
    """
    Simula los dos factores del paper y reconstruye el spot:
    - x(t): GBM.
    - z(t): OU.
    - S(t) = exp(s(t) + x(t) + z(t)).

    En t=0:  ln S0 = s(0) + x0 + z0  ->  x0 = ln S0 - s(0) - z0
    """
    rng = np.random.default_rng(seed)
    if gammas is None:
        gammas = np.zeros(2 * K)

    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    x = np.zeros((n_paths, n_steps + 1))    # GBM
    z = np.zeros((n_paths, n_steps + 1))    # OU
    x[:, 0] = np.log(S0) - s_estacional(0.0, gammas, K) - z0
    z[:, 0] = z0

    # Coeficientes del OU de z 
    exp_kdt = np.exp(-kappa * dt)
    ou_std = v * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))

    # Coeficientes del GBM de x
    c1 = (mu - 0.5 * sigma**2) * dt
    c2 = sigma * np.sqrt(dt)

    for i in range(n_steps):
        Z1, Z2 = correlated_normals(rng, n_paths, rho)
        x[:, i+1] = x[:, i] + c1 + c2 * Z1          
        z[:, i+1] = z[:, i] * exp_kdt + ou_std * Z2  

    s_t = s_estacional(times, gammas, K)
    S = np.exp(s_t[np.newaxis, :] + x + z)           # precio = s + x + z
    return times, S, x, z


def future_sorensen(S0, tau, z0, kappa, gammas, K, alpha, sigma, v, rho, lambda_z):
    """
    Futuro de Sørensen (ec. 7 del paper):

        ln F(tau) = s(tau) + A(tau) + x + z*exp(-kappa*tau)
        A(tau) = alpha*tau - (lambda_z - rho*sigma*v)/kappa * (1 - exp(-kappa*tau))
                           + v^2/(4*kappa) * (1 - exp(-2*kappa*tau)).
    """
    s0 = np.sum(gammas.reshape(K, 2)[:, 0])
    x = np.log(S0) - s0 - z0
    A = alpha * tau \
        - (lambda_z - rho * sigma * v) / kappa * (1.0 - np.exp(-kappa * tau)) \
        + v**2 / (4.0 * kappa) * (1.0 - np.exp(-2.0 * kappa * tau))
    ln_F = s_estacional(tau, gammas, K) + A + x + z0 * np.exp(-kappa * tau)
    return np.exp(ln_F)


def main():

    # Parametros
    S0 = 11.7
    mu = 0.02          
    sigma = 0.20       # volatilidad de x 
    kappa = 0.95       # velocidad de reversión de z 
    v = 0.25           # volatilidad de z 
    rho = -0.3
    z0 = 0.08

    alpha = 0.0
    lambda_z = 0.0

    # Estacionalidad (K=2)
    K = 2
    gammas = np.array([-0.001, 0.011,    # k=1: [cos, sin]  
                        0.000, -0.007])  # k=2: [cos, sin] 

    T = 2.0
    n_steps = 2 * 252
    n_paths = 5000

    # 1) Simulación
    times, S_paths, x_paths, z_paths = simulate_sorensen(
        S0=S0, mu=mu, sigma=sigma, kappa=kappa, v=v, rho=rho, z0=z0,
        gammas=gammas, K=K, T=T, n_steps=n_steps, n_paths=n_paths, seed=1)

    # 2) Trayectorias S(t)
    plt.figure(figsize=(9, 5))
    for i in range(min(20, n_paths)):
        plt.plot(times, S_paths[i], linewidth=1)
    plt.xlabel("Tiempo t (años)")
    plt.ylabel("Spot S(t)")
    plt.title("Sørensen: S(t) = exp( s(t) + x(t) + z(t) )")
    plt.grid(True)
    plt.show()

    # 3) Los dos factores por separado
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for i in range(min(20, n_paths)):
        ax[0].plot(times, x_paths[i], linewidth=1)
        ax[1].plot(times, z_paths[i], linewidth=1)
    ax[0].set_title("x(t): GBM")
    ax[0].set_xlabel("t (años)"); ax[0].grid(True)
    ax[1].set_title("z(t): OU")
    ax[1].set_xlabel("t (años)"); ax[1].legend(); ax[1].grid(True)
    plt.tight_layout()
    plt.show()

    # 4) Curva de futuros (ec. 7): K=1 vs K=2
    #    Cada armonico son 2 coeficientes (cos, sin)
    taus = np.linspace(0.01, 2.0, 300)

    plt.figure(figsize=(10, 6))
    F_base = future_sorensen(S0, taus, z0, kappa, np.zeros(2), 1, alpha, sigma, v, rho, lambda_z)
    plt.plot(taus, F_base, color="gray", ls="--", lw=1.5, label="Sin estacionalidad")
    for Kx, color in [(1, "steelblue"), (2, "firebrick")]:
        g = gammas[:2 * Kx]                           
        F = future_sorensen(S0, taus, z0, kappa, g, Kx, alpha, sigma, v, rho, lambda_z)
        plt.plot(taus, F, color=color, lw=2, label="K=%d" % Kx)
    plt.axhline(S0, color="black", linestyle=":", lw=1.2, label="Spot S0 = %.2f" % S0)
    plt.xlabel("Vencimiento tau (años)")
    plt.ylabel("F(0, tau)")
    plt.title("Sørensen: curva de futuros para K=1 y K=2\n"
              "ln F = s(tau) + A(tau) + x + z*exp(-kappa*tau)")
    plt.legend(fontsize=9)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
