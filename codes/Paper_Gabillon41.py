"""
1) Simular el spot S_t como GBM (ecuación 1)
2) Calcular el precio del futuro (ecuación 7):
      F(S, tau) = S * exp((r + Cc) * tau)
3) Opción europea sobre SPOT (4.1) por Monte Carlo:
      Call payoff: max(S_T - K, 0)
      Put  payoff: max(K - S_T, 0)
      Precio aprox: V0 = exp(-r T) * mean(payoff)
4) Calcular el precio del futuro con convenience yield (ecuación 8):
      F(S, tau) = S * exp((r + Cc - Cy) * tau

"""

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import math
from scipy.stats import norm

import numpy as np
import matplotlib.pyplot as plt

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class bscall:

    """Opción call, solución analítica.
    """

    def __init__ (self, r, sigma, E, T):

        self.r = r
        self.sigma = sigma
        self.E = E
        self.T = T
        self.c = r + 0.5 * sigma*sigma

    def __call__ (self, s, t):

        tau = self.T - t
        d1 = (math.log (s / self.E) + self.c * tau) / (self.sigma * math.sqrt (tau))
        d2 = d1 - self.sigma * math.sqrt (tau)

        Nd1 = norm.cdf (d1)
        Nd2 = norm.cdf (d2)

        return s * Nd1 - self.E * math.exp (-self.r * tau) * Nd2

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

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
        T       : tiempo total (años)
        n_steps : número de pasos temporales
        n_paths : número de trayectorias
        mu      : drift
        sigma   : volatilidad
        seed    : semilla aleatoria

    Output:
        times   : vector de tiempos (0..T)
        S       : matriz con trayectorias del spot"""

    np.random.seed(seed)
    dt = T / n_steps
    times = np.linspace(0.0, T, n_steps + 1)

    S = np.zeros((n_paths, n_steps + 1))
    S[:, 0] = S0

    c1 = (mu - 0.5 * sigma**2) * dt         # Simplemente para ahorrar tiempo.
    c2 = sigma * np.sqrt(dt)                # De esta forma, no operas con valores constantes.

    for i in range(n_steps):
        Z = np.random.normal(size=n_paths)  # incrementos brownianos ~ N(0,1)
        #-- S[:, i + 1] = S[:, i] * np.exp(
        #--     (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        S[:, i + 1] = S[:, i] * np.exp (c1 + c2 * Z)

    return times, S

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# -----------------------------------------
# 2) Futures price (Ecuación 7 y 8)
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

def future_price_with_convenience_yield(S, tau, r, Cc, Cy):
    
    """Precio del futuro según Gabillon con convenience yield F(S, tau) = S * exp((r + Cc - Cy) * tau)
    Inputs:
        S   : spot 
        tau : time-to-maturity (años)
        r   : tipo de interés (constante)
        Cc  : coste marginal de carry/almacenamiento
        Cy  : beneficio de tener el físico

    Output:
        F   : precio del futuro
    """
    
    return S * np.exp((r + Cc - Cy) * tau)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# -----------------------------------------
# 3) Opción europea sobre SPOT (Monte Carlo)
# -----------------------------------------
def payoff_call(S, K):
    return np.maximum(S - K, 0.0)

def payoff_put(S, K):
    return np.maximum(K - S, 0.0)

def option_price_mc_from_paths(S_paths, K, r, T, option_type="call"):
    """
    Precio MC usando S_T de las trayectorias ya simuladas:
        V0 ≈ exp(-rT) * mean(payoff(S_T)) (estocasticos2324.pdf pág.48 o metodos2324.pdf pág.5)
    """
    S_T = S_paths[:, -1]

    if option_type == "call":
        payoff = payoff_call(S_T, K) #Funcion de pago
    elif option_type == "put":
        payoff = payoff_put(S_T, K) #Funcion de pago
    else:
        raise ValueError("option_type debe ser 'call' o 'put'")

    price = np.exp(-r * T) * np.mean(payoff) #Precio de la opcion en tiempo t = 0
    return price, payoff

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

# -----------------------------
# 4) Principal 
# -----------------------------

def main():

    # Parámetros

    S0 = 66.0
    mu = 0.05          # drift del spot
    sigma = 0.15       # volatilidad spot
    r = 0.03           # tipo de interés
    Cc = 0.02          # coste de carry
    Cy = 0.08          # convenience yield (constante)

    # 1) Spot

    T = 1.0            # 1 año
    n_steps = 252      # dias con el mercado operativo (aprox)
    #-- n_paths = 5000     # nº trayectorias
    n_paths = 10000    # nº trayectorias

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

    # 2) Curva de futuros - el precio acordado hoy para intercambiar la commodity en tau

    taus = np.array([0.25, 0.50, 1.00, 2.00, 3.00])  # tiempo vencimiento en años
    #-- F0 = future_price_gabillon_41(S0, taus, r, Cc)
    F0_noCy = future_price_gabillon_41(S0, taus, r, Cc)
    F0_Cy   = future_price_with_convenience_yield(S0, taus, r, Cc, Cy)

    #-- print("Curva de futuros en t=0 (Gabillon 4.1, ecuación 7):")
    #-- print(f"S0={S0:.4f}, r={r:.4f}, Cc={Cc:.4f}")
    
    #for tau, f in zip(taus, F0):
       #print(f"  tau={tau:>4} años  ->  F(0,tau)={f:.6f}")

    # Curva de futuros con precio fijado hoy y distintos vencimientos

    plt.figure(figsize=(9, 5))
    #-- plt.plot(taus, F0, marker="o", linewidth=2)
    plt.plot(taus, F0_noCy, marker="o", linewidth=2, label="Sin convenience yield (ec. 7)")
    plt.plot(taus, F0_Cy,   marker="o", linewidth=2, label=f"Con convenience yield (ec. 8), Cy={Cy:.2%}")

    plt.xlabel("Tiempo a vencimiento τ (años)")
    plt.ylabel("Futuro F(0,τ)")
    plt.title("Curva de futuros - Ecuación (7) y (8)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 3) Efecto del Cy
    
    Cy_list = [0.00, 0.03, 0.05, 0.08, 0.10]  

    plt.figure(figsize=(9, 5))
    for Cy_i in Cy_list:
        F0_i = future_price_with_convenience_yield(S0, taus, r, Cc, Cy_i)
        plt.plot(taus, F0_i, marker="o", linewidth=2, label=f"Cy={Cy_i:.0%}")

    plt.xlabel("Tiempo a vencimiento τ (años)")
    plt.ylabel("Futuro F(0,τ)")
    plt.title("Efecto del convenience yield en la curva de futuros (t=0)")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 4) Curva de futuros respecto al spot (sin Cy))
    
    S_grid = np.linspace(S0*0.5, S0*1.5, 200)   # rango de spot para la gráfica

    plt.figure(figsize=(9, 5))
    for tau in taus:
        F_vs_S = future_price_gabillon_41(S_grid, tau, r, Cc)
        plt.plot(S_grid, F_vs_S, linewidth=2, label=f"τ = {tau} años")

    plt.xlabel("Spot S")
    plt.ylabel("Futuro F(S, τ)")
    plt.title("Precio del futuro respecto al spot (Gabillon 4.1) para varios τ")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 5) Curva de futuros respecto al spot (con Cy))
    
    plt.figure(figsize=(9, 5))
    for tau in taus:
        F_vs_S_Cy = future_price_with_convenience_yield(S_grid, tau, r, Cc, Cy)
        plt.plot(S_grid, F_vs_S_Cy, linewidth=2, label=f"τ = {tau} años")

    plt.xlabel("Spot S")
    plt.ylabel("Futuro F(S, τ)")
    plt.title("Precio del futuro respecto al spot con convenience yield para varios τ")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # 6) OPCIONES sobre SPOT (mercados2324.pdf)
    
    K = 70.0  # strike
    # A) Payoff a vencimiento (Opcion europea))
    S_pay = np.linspace(S0 * 0.3, S0 * 1.8, 400)
    call_pay = payoff_call(S_pay, K)
    put_pay  = payoff_put(S_pay, K)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(S_pay, call_pay, linewidth=2, label=f"max(S-K,0), K={K}")
    plt.xlabel("Spot a vencimiento S(T)")
    plt.ylabel("Payoff Call")
    plt.title("Función de pago - Call europea")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(S_pay, put_pay, linewidth=2, label=f"max(K-S,0), K={K}")
    plt.xlabel("Spot a vencimiento S(T)")
    plt.ylabel("Payoff Put")
    plt.title("Función de pago - Put europea")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # B) Precio por Monte Carlo usando las trayectorias simuladas
    # call_price, call_payoff = option_price_mc_from_paths(S_paths, K, r, T, option_type="call")
    # put_price,  put_payoff  = option_price_mc_from_paths(S_paths, K, r, T, option_type="put")

    # print("\n=== Opciones europeas sobre SPOT (Monte Carlo) ===")
    # print(f"Parámetros: S0={S0}, K={K}, T={T}, r={r}, mu={mu}, sigma={sigma}")
    # print(f"Call MC price: {call_price:.6f}")
    # print(f"Put  MC price: {put_price:.6f}")

    # C) Precio de la call vs S0
    #     Simulamos solo S_T directamente para varios S0.
    #-- S0_grid = np.linspace(S0 * 0.6, S0 * 1.4, 25)

    #################################################################################
    print ()
    print (' Monte Carlo: ')
    print ('        S0: ', S0)
    print ('         r: ', r)
    print ('        mu: ', mu)
    print ('     sigma: ', sigma)
    print ('         K: ', K)
    print ('         T: ', T)
    print ()
    #################################################################################

    S0_grid = np.linspace(S0 * 0.3, S0 * 1.8, 25)
    prices = []
    for S0_i in S0_grid:
        # simulamos S_T
        Z = np.random.normal(size=n_paths)
        S_T = S0_i * np.exp((mu - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)
        payoff = np.maximum(S_T - K, 0.0)
        prices.append(np.exp(-r*T) * np.mean(payoff))

    #-- plt.figure(figsize=(9, 5))
    #-- plt.plot(S0_grid, prices, marker="o", linewidth=2)
    #-- plt.xlabel("Spot")
    #-- plt.ylabel("Precio Call (MC)")
    #-- plt.title("Precio de la Call (MC) en función de S0")
    #-- plt.grid(True)

    mu = 0.05          # drift del spot
    sigma = 0.15       # volatilidad spot
    r = 0.03           # tipo de interés
    r = mu             # tipo de interés
    T = 1.0            # vencimiento

    #################################################################################
    print ()
    print (' Solución analítica: ')
    print ('        S0: ', S0)
    print ('         r: ', r)
    print ('     sigma: ', sigma)
    print ('         K: ', K)
    print ('         T: ', T)
    print ()
    #################################################################################

    analytic_call = bscall (r, sigma, K, T)
    y = []
    for S0_i in S0_grid:
        y.append (analytic_call (S0_i, 0.0))

    plt.figure(figsize=(9, 5))
    plt.plot(S_pay, call_pay, linewidth=2, label=f"max(S-K,0), K={K}")
    plt.plot(S0_grid, y, lw=2, label=r"Analytic solution")
    plt.plot(S0_grid, prices, marker="o", linewidth=2, label=f"Monte Carlo ({n_paths} paths)")
    plt.xlabel("Spot")
    plt.ylabel("Precio Call (MC)")
    plt.title("Precio de la Call (MC) en función de S0")
    plt.grid(True)
    plt.legend ()

    plt.show()
    

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if __name__ == "__main__": main()

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
