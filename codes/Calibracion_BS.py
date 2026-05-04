"""
Calibración de Black-Scholes
Método de Newton y Levenberg-Marquardt

"""

import numpy as np
from scipy.stats import norm
import openpyxl

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def bs_call(S, K, r, sigma, T):
    """
    Fórmula de Black-Scholes para una call europea.
    """
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Función objetivo: J(a) = sum (V_mercado - V_modelo)^2
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def func_J(a, S, V, K, r, T):
    """
    a: vector de parámetros a calibrar.
    Para 1 parámetro: a = [sigma]
    Para 2 parámetros: a = [sigma, r]  (r se sobreescribe)
    """
    sigma = a[0]
    if len(a) >= 2: r = a[1]
    if len(a) >= 3: K = a[2]
    if len(a) >= 4: T = a[3]

    j = 0.0
    for i in range(len(V)):
        z = V[i] - bs_call(S[i], K, r, sigma, T)
        j += z*z
    return j

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# G(a) = gradiente de J por diferencias finitas centradas
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def grad_J(a, S, V, K, r, T):
    """
    G_j = (J(a+h) - J(a-h)) / (2h)
    """
    n = len(a)
    g = np.zeros(n)

    for j in range(n):
        h = 1e-5 * max(abs(a[j]), 1.0)
        ap = a.copy(); ap[j] += h # sumo h
        am = a.copy(); am[j] -= h # resto h
        g[j] = (func_J(ap, S, V, K, r, T) - func_J(am, S, V, K, r, T)) / (2.0*h)
        
        # Derivada respecto de σ (primer parámetro, j=0)
        # Derivada respecto de r (segundo parámetro, j=1) y así sucesivamente

    return g

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Newton para G(a) = 0
# Mismo esquema que Euler_Implicito.py:
#   G = ...
#   JG[:,j] = (G(a+h) - G(a)) / h
#   Z = solve(JG, G)
#   a = a - Z
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def newton(S, V, K, r, T, a0, niter, eps):
    
    """
    - a0:    aproximación inicial de los parámetros.
    - niter: número máximo de iteraciones.
    - eps:   máximo error relativo permitido.
    """

    a = a0.copy()
    h = 1e-5
    n = len(a) # número de parámetros (1, 2, 3 o 4)

    print()
    print(' Newton, k: ', 0, '  a: ', a)

    for k in range(1, niter):

        # Gradiente
        gk = grad_J(a, S, V, K, r, T)

        # Jacobiano de G por diferencias finitas
        JG = np.zeros((n, n))
        for j in range(n):
            hj = h * max(abs(a[j]), 1.0) # paso adaptado al parámetro j
            ap = a.copy(); ap[j] += hj # sumo hj
            am = a.copy(); am[j] -= hj # resto hj
            gp = grad_J(ap, S, V, K, r, T) # gradiente en a+hj
            gm = grad_J(am, S, V, K, r, T) # gradiente en a-hj
            JG[:,j] = (gp - gm) / (2.0 * hj)

        # Resolver JG * Z = G
        Z = np.linalg.solve(JG, gk)

        # Actualizar
        a_new = a - Z
        err = np.linalg.norm(a_new - a) / np.linalg.norm(a_new)

        print(' Newton, k: ', k, '  err: ', err, '  a: ', a_new)

        if err < eps: break
        a = a_new.copy()

    return a_new

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Levenberg-Marquardt
# Basado en marquardt.py:
#   pk = -solve(hk + gamma*I, gk)
#   a = a + pk
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def marquardt(S, V, K, r, T, a0, niter, eps):
    
    """
    - a0:    aproximación inicial de los parámetros.
    - niter: número máximo de iteraciones.
    - eps:   máximo error relativo permitido.
    """
    
    gamma = 0.20
    a = a0.copy()
    n = len(a)
    h = 1e-5

    print()
    print(' Marquardt, k: ', 0, '  a: ', a)

    for k in range(1, niter):

        # Gradiente
        gk = grad_J(a, S, V, K, r, T)

        # Jacobiano de G
        JG = np.zeros((n, n))
        for j in range(n):
            hj = h * max(abs(a[j]), 1.0) # paso adaptado al parámetro j
            ap = a.copy(); ap[j] += hj # sumo hj
            am = a.copy(); am[j] -= hj # resto hj
            gp = grad_J(ap, S, V, K, r, T) # gradiente en a+hj
            gm = grad_J(am, S, V, K, r, T) # gradiente en a-hj
            JG[:,j] = (gp - gm) / (2.0 * hj)

        # Resolver (JG + gamma*I) * pk = -gk
        hk = JG + gamma * np.identity(n)
        pk = -np.linalg.solve(hk, gk)

        a_new = a + pk
        err = np.linalg.norm(a_new - a) / np.linalg.norm(a_new)

        print(' Marquardt, k: ', k, '  err: ', err, '  a: ', a_new)

        if err < eps: break
        a = a_new.copy()

    return a_new

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Cargar datos del Excel
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def cargar_spots(filepath, n_datos=10):

    wb = openpyxl.load_workbook(filepath, data_only=True)
    ws = wb['Spots']

    spots = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[0] is not None and row[1] is not None:
            spots.append(row[1])

    spots = spots[::-1]  # orden cronológico

    if n_datos is not None:
        spots = spots[:n_datos]

    return np.array(spots)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if __name__ == '__main__':

    # -----------------------------------------------------------------
    # Datos del Excel
    # -----------------------------------------------------------------

    filepath = r'C:\Users\Leonor\OneDrive - Universidad Politécnica de Madrid\Documentos\0. ETSIAE\TFM\TFM_Assets.xlsx'
    S = cargar_spots(filepath, n_datos=10)

    print()
    print(' Datos de spot:')
    for i in range(len(S)):
        print('   S[%d] = %.4f' % (i, S[i]))

    # -----------------------------------------------------------------
    # Parámetros "verdaderos" (para generar datos)
    # -----------------------------------------------------------------

    K = 85.0
    r = 0.03
    sigma = 0.25
    T = 1.0

    # Datos: V = BS(S, K, r, sigma, T)
    V = np.array([bs_call(S[i], K, r, sigma, T) for i in range(len(S))])

    print()
    print(' Parámetros verdaderos: K=%.2f, r=%.4f, sigma=%.4f, T=%.4f' % (K, r, sigma, T))
    print()
    for i in range(len(S)):
        print('   S[%d] = %.4f   V[%d] = %.6f' % (i, S[i], i, V[i]))

    niter = 50
    eps = 1e-10

    # -----------------------------------------------------------------
    # Caso 1: calibrar sigma (1 parámetro)
    # -----------------------------------------------------------------

    print()
    print(' =========================================')
    print(' Caso 1: calibrar sigma (K, r, T fijados)')
    print(' =========================================')

    a0 = np.array([0.10])

    print()
    print(' Newton:')
    a_opt = newton(S, V, K, r, T, a0, niter, eps)
    print('   sigma* = ', a_opt[0], '  (verdadero: ', sigma, ')')

    print()
    print(' Marquardt:')
    a_opt = marquardt(S, V, K, r, T, a0, niter, eps)
    print('   sigma* = ', a_opt[0], '  (verdadero: ', sigma, ')')

    # -----------------------------------------------------------------
    # Caso 2: calibrar sigma y r (2 parámetros)
    # -----------------------------------------------------------------

    print()
    print(' =========================================')
    print(' Caso 2: calibrar sigma, r (K, T fijados)')
    print(' =========================================')

    a0 = np.array([0.10, 0.01])

    print()
    print(' Newton:')
    a_opt = newton(S, V, K, r, T, a0, niter, eps)
    print('   sigma* = ', a_opt[0], '  (verdadero: ', sigma, ')')
    print('       r* = ', a_opt[1], '  (verdadero: ', r, ')')
    
    print()
    print(' Marquardt:')
    a_opt = marquardt(S, V, K, r, T, a0, niter, eps)
    print('   sigma* = ', a_opt[0], '  (verdadero: ', sigma, ')')
    print('       r* = ', a_opt[1], '  (verdadero: ', r, ')')


    # -----------------------------------------------------------------
    # Caso 3: calibrar sigma, r, K (3 parámetros)
    # -----------------------------------------------------------------

    print()
    print(' =========================================')
    print(' Caso 3: calibrar sigma, r, K (T fijado)')
    print(' =========================================')

    a0 = np.array([0.15, 0.01, 80.0])

    print()
    print(' Newton:')
    a_opt = newton(S, V, K, r, T, a0, niter, eps)
    print('   sigma* = ', a_opt[0], '  (verdadero: ', sigma, ')')
    print('       r* = ', a_opt[1], '  (verdadero: ', r, ')')
    print('       K* = ', a_opt[2], '  (verdadero: ', K, ')')
    
    print()
    print(' Marquardt:')
    a_opt = marquardt(S, V, K, r, T, a0, niter, eps)
    print('   sigma* = ', a_opt[0], '  (verdadero: ', sigma, ')')
    print('       r* = ', a_opt[1], '  (verdadero: ', r, ')')
    print('       K* = ', a_opt[2], '  (verdadero: ', K, ')')

    # -----------------------------------------------------------------
    # Caso 4: calibrar sigma, r, K, T (4 parámetros)
    # -----------------------------------------------------------------

    print()
    print(' =========================================')
    print(' Caso 4: calibrar sigma, r, K, T')
    print(' =========================================')

    a0 = np.array([0.15, 0.01, 80.0, 0.5])
    
    print()
    print(' Newton:')
    a_opt = newton(S, V, K, r, T, a0, niter, eps)
    print('   sigma* = ', a_opt[0], '  (verdadero: ', sigma, ')')
    print('       r* = ', a_opt[1], '  (verdadero: ', r, ')')
    print('       K* = ', a_opt[2], '  (verdadero: ', K, ')')
    print('       T* = ', a_opt[3], '  (verdadero: ', T, ')')

    print()
    print(' Marquardt:')
    a_opt = marquardt(S, V, K, r, T, a0, niter, eps)
    print('   sigma* = ', a_opt[0], '  (verdadero: ', sigma, ')')
    print('       r* = ', a_opt[1], '  (verdadero: ', r, ')')
    print('       K* = ', a_opt[2], '  (verdadero: ', K, ')')
    print('       T* = ', a_opt[3], '  (verdadero: ', T, ')')

    print()
