"""
1) Simular el spot S_t como GBM (ecuación 1)
2) Calcular el precio del futuro (ecuación 7):
      F(S, tau) = S * exp((r + Cc) * tau)
"""
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1) Spot model (Ecuación 1)
# -----------------------------
def simulate_spot_gbm(S0, T, n_steps, n_paths, mu, sigma, seed=0):
    """Simula el spot con un movimiento browniano geométrico (GBM):
        dS = mu*S dt + sigma*S dW
        
    Solución exacta GBM:
        S_{t+dt} = S_t * exp((mu - 0.5*sigma^2)dt + sigma*sqrt(dt)*Z)

    Inputs:
        S0      : spot inicial
        T       : horizonte total (años)
        n_steps : número de pasos temporales
        n_paths : número de trayectorias
        mu      : drift (mundo real; para pricing riesgo-neutral sería diferente)
        sigma   : volatilidad (constante)
        seed    : semilla aleatoria

    Output:
        times   : vector de tiempos (0..T)
        S       : matriz (n_paths x (n_steps+1)) con trayectorias del spot"""
    np.random.seed(seed)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0

    for i in range(n_steps):
        Z = np.random.normal(size=n_paths)  # incrementos brownianos ~ N(0,1)
        S[:, i + 1] = S[:, i] * np.exp(
            (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    return times, S

# -----------------------------------------
# 2) Futures price (Ecuación 7)
# -----------------------------------------
def future_price_gabillon_41(S, tau, r, Cc):
    """Precio del futuro según Gabillon F(S, tau) = S * exp((r + Cc) * tau)
    Inputs:
        S   : spot 
        tau : time-to-maturity (años)
        r   : tipo de interés (constante)
        Cc  : coste marginal de carry/almacenamiento

    Output:
        F   : precio del futuro
    """
    return S * np.exp((r + Cc) * tau)

# -----------------------------
# 3) Principal 
# -----------------------------
def main():
    # Parámetros
    S0 = 80.0
    mu = 0.05      # drift del spot
    sigma = 0.35   # volatilidad spot
    r = 0.03       # tipo de interés
    Cc = 0.02      # coste de carry

    # 1) Spot
    T = 1.0            # 1 año
    n_steps = 252      # dias con el mercado operativo (aprox)
    n_paths = 50       # nº trayectorias

    times, S_paths = simulate_spot_gbm(
        S0=S0, T=T, n_steps=n_steps, n_paths=n_paths, mu=mu, sigma=sigma, seed=42)
    
    # Trayectorias del spot
    plt.figure(figsize=(9, 5))
    for i in range(min(20, n_paths)):  # dibujamos 20
        plt.plot(times, S_paths[i, :], linewidth=1)
    plt.xlabel("Tiempo t (años)")
    plt.ylabel("Spot S(t)")
    plt.title("Trayectorias simuladas del spot (GBM) - Ecuación (1)")
    plt.grid(True)
    plt.show()

    # 2) Curva de futuros en t=0
    taus = np.array([0.25, 0.50, 1.00, 2.00, 3.00])  # tiempo vencimiento en años
    F0 = future_price_gabillon_41(S0, taus, r, Cc)

    print("Curva de futuros en t=0 (Gabillon 4.1, ecuación 7):")
    print(f"S0={S0:.4f}, r={r:.4f}, Cc={Cc:.4f}")
    
    for tau, f in zip(taus, F0):
        print(f"  tau={tau:>4} años  ->  F(0,tau)={f:.6f}")

    # Curva de futuros en t=0
    plt.figure(figsize=(9, 5))
    plt.plot(taus, F0, marker="o", linewidth=2)
    plt.xlabel("Tiempo a vencimiento τ (años)")
    plt.ylabel("Futuro F(0,τ)")
    plt.title("Curva de futuros - Ecuación (7)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__": main()