"""
Calibración de Black-Scholes.
Métodos de Newton y Levenberg-Marquardt.
"""

import numpy as np
from scipy.stats import norm
import openpyxl
import copy
import pylab as plt

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def bs_call (S, K, r, sigma, T):

    """
    Fórmula de Black-Scholes para una call europea.
    """

    d1 = (np.log (S/K) + (r + 0.5*sigma**2) * T) / (sigma * np.sqrt (T))
    d2 = d1 - sigma * np.sqrt (T)

    return S * norm.cdf (d1) - K * np.exp (-r*T) * norm.cdf (d2)

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

class obfunc:

    """
    Función objetivo, vector gradiente y matriz hessiana.
    """

    def __init__ (self, S, V):

        self.S = S [:]
        self.V = V [:]
        self.coeff = [0.12, 0.01, 0.001, 0.02]

    # ---------------------------------------------------------------

    def __call__ (self, a):

        """
        Función objetivo.
        """

        [sigma, r, K, T] = a [:]
    
        j = 0.0
        for i in range (len (V)):
    
            #####################################################################
            #print (' V: ', self.V[i], '       call: ', bs_call (self.S[i], K, r, sigma, T))
            #####################################################################
    
            z = self.V[i] - bs_call (self.S[i], K, r, sigma, T)
            j += z*z

        return j

    # ---------------------------------------------------------------

    def grad (self, a):

        """
        Vector gradiente.
        """

        n = len (a)
        g = np.zeros (n)
    
        for i in range (n):
            h = self.coeff [i] * a [i]
            am = copy.copy (a)
            ap = copy.copy (a)
            am [i] = a [i] - h
            ap [i] = a [i] + h
            g [i] = (self (ap) - self (am)) / (2.0 * h)
    
            #######################################################################
            #print ('\n         h: ', h)
            #print ('        am: ', am, '      J (am): ', self (am))
            #print ('        ap: ', ap, '      J (ap): ', self (ap), '        g: ', g[i])
            #######################################################################
    
        return g

    # ---------------------------------------------------------------

    def hess (self, a):

        """
        Matriz hessiana.
        """

        n = len (a)
        mat = np.zeros ((n, n))
    
        for i in range (n):
            h = self.coeff [i] * a [i]
            am = copy.copy (a)
            ap = copy.copy (a)
            am [i] = a [i] - h
            ap [i] = a [i] + h
    
            mat [i, i] = (self (am) - 2.0 * self (a) + self (ap)) / (h*h)

            for j in range (i):
                k = self.coeff [j] * a [j]
                ane = copy.copy (am)
                asw = copy.copy (am)
                ase = copy.copy (ap)
                anw = copy.copy (ap)
                asw [j] = am [j] - k
                anw [j] = am [j] + k
                ase [j] = ap [j] - k
                ane [j] = ap [j] + k

                ###################################################################################
                #print ()
                #print ('     a: ', a)
                #print ('    am: ', am)
                #print ('    ap: ', ap)
                #print ('   asw: ', asw)
                #print ('   anw: ', anw)
                #print ('   ase: ', ase)
                #print ('   ane: ', ane)
                ###################################################################################
    
                matij = (self (asw) - self (anw) - self (ase) + self (ane)) / (4.0 * h * k)
                mat [i, j] = matij
                mat [j, i] = matij
    
        return mat

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def newton (f, a0, niter, eps1, eps2):
    
    """
    Método de Newton para    gradJ (a) = 0.
    - a0:    aproximación inicial de los parámetros.
    - niter: número máximo de iteraciones.
    - eps:   máximo error relativo permitido.
    """

    a = a0.copy()
    #h = 1e-2
    #-- h = 1e-5
    #-- h = 1e-6
    n = len(a)        # número de parámetros (1, 2, 3 o 4)

    print ()
    #print (' Newton, k: ', 0, '  a: ', a)
    print (' %3d                      %16.8f  ' % (0, f (a)))

    for k in range (1, niter):

        # Gradiente y Hessiana.

        fk = f (a)
        gk = f.grad (a)
        hk = f.hess (a)

        #################################################################################
        #print ()
        #print ('    a: ', a)
        #print ('   gk: ', gk)
        #print ('   hk: ')
        #print (hk)
        #################################################################################

        # Resolución del sistema y actualización de la solución.

        delta = np.linalg.solve (hk, gk)
        a_new = a - delta

        err1 = np.linalg.norm (delta) / np.linalg.norm (a_new)
        err2 = abs (fk / f(a_new))

        print (' %3d     %12.6e     %12.6e     %16.8f  ' % (k, err1, err2, f (a_new)))

        if (err1 < eps1 or err2 < eps2): break
        a = a_new.copy ()

    return a_new

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def cargar_spots (filepath, n_datos=10):

    """
    Carga de datos del Excel.
    """

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

if (__name__ == '__main__'):

    # -----------------------------------------------------------------
    # Datos del Excel

    #-- filepath = r'C:\Users\Leonor\OneDrive - Universidad Politécnica de Madrid\Documentos\0. ETSIAE\TFM\TFM_Assets.xlsx'       # Leonor.
    filepath = r'../data/TFM_Assets.xlsx'                                                                                     # Íñigo.

    S = cargar_spots (filepath, n_datos=10)

    print ()
    print (' Datos de spot:')
    for i in range (len(S)):
        print ('   S[%d] = %.4f' % (i, S[i]))

    # -----------------------------------------------------------------
    # Datos sintéticos.

    S = np.linspace (75.0, 100.0, 26)
    S = np.linspace (50.0, 100.0, 51)

    # -----------------------------------------------------------------
    # Parámetros "verdaderos" (para generar datos)

    K = 85.0
    r = 0.03
    sigma = 0.25
    T = 1.0

    # Datos: V = BS (S, K, r, sigma, T)

    V = np.array ([bs_call(S[i], K, r, sigma, T) for i in range(len(S))])

    print ()
    print (' Parámetros verdaderos: K=%.2f, r=%.4f, sigma=%.4f, T=%.4f' % (K, r, sigma, T))
    print ()
    for i in range (len(S)):
        print ('   S[%d] = %.4f   V[%d] = %.6f' % (i, S[i], i, V[i]))

    # -----------------------------------------------------------------
    # Función objetivo.

    f = obfunc (S, V)

    # -----------------------------------------------------------------
    # Caso 4: calibración de sigma, r, K, T (4 parámetros)
    # -----------------------------------------------------------------

    print ()
    print (' =========================================')
    print (' Caso 4: calibración de sigma, r, K, T    ')
    print (' =========================================')

    niter = 50
    eps1 = 1.0e-8
    eps2 = 1.0e-3

    a0 = np.array ([0.15, 0.01, 80.0, 0.5])

    #-- a0 = np.array ([1.08*sigma, 0.92*r, 1.05*K, 0.95*T])
    #-- a0 = np.array ([1.01*sigma, 0.98*r, 1.02*K, 0.98*T])
    
    print ()
    print (' Newton:')
    a_opt = newton (f, a0, niter, eps1, eps2)

    print ()
    print ('                     Real            Aproximado')
    print ('    sigma:     %12.8f        %12.8f' % (sigma, a_opt [0]))
    print ('        r:     %12.8f        %12.8f' % (r, a_opt [1]))
    print ('        K:     %12.8f        %12.8f' % (K, a_opt [2]))
    print ('        T:     %12.8f        %12.8f' % (T, a_opt [3]))
    print ()

    # ----------------------------------------------------------------
    # Plot.

    sigma, r, K, T = a_opt [:]
    w = np.array ([bs_call (Si, K, r, sigma, T) for Si in S])

    plt.figure ()
    plt.plot (S, V, 'ro', label='Data')
    plt.plot (S, w, 'g--', lw=2, label='Calibrated curve')
    plt.grid (True)
    plt.legend ()
    plt.title (r'Black-Scholes calibration')

    plt.show ()

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

