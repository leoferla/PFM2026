# encoding: utf-8

"""
Gabillon: Modelo de dos factores
Método de diferencias finitas.
"""

import numpy as np
import pylab as plt
import time

import graf

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
# Condiciones de contorno.

def bound00 (x, y, t):

    k0 = 3.85
    return k0 * x

def bound02 (x, y, t):

    k0 = -0.15
    return k0 * x

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def dirichnorth (a, b, bc, x, y, t):

    # Frontera superior, Dirichlet.

    n = len (x) - 2
    m = len (y) - 2

    j = m+1
    for i in range (n+2):
        k = j * (n+2) + i
        so = k - (n+2)
        a [k,:] = 0.0
        a [k,k] = 1.0
        b [k] = bc (x[i],y[j],t)

    return [a,b]

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def dirichsouth (a, b, bc, x, y, t):

    # Frontera inferior, Dirichlet.

    n = len (x) - 2
    m = len (y) - 2

    j = 0
    for i in range (n+2):
        k = i
        a [k,:] = 0.0
        a[k,k] = 1.0
        b[k] = bound00 (x[i],y[j],t)

    return [a,b]

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def dirichwest (a, b, bc, x, y, t):

    # Frontera izquierda, Dirichlet.

    n = len (x) - 2
    m = len (y) - 2

    i = 0
    for j in range (m+2):
        k = j * (n+2) + i
        a [k,k] = 1.0
        b [k] = bc (x[i],y[j],t)

    return [a,b]

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

def diricheast (a, b, bc, x, y, t):

    # Frontera derecha, Dirichlet.

    n = len (x) - 2
    m = len (y) - 2

    i = n+1
    for j in range (m+2):
        k = j * (n+2) + i
        a[k,k] = 1.0
        b[k] = bound (x[i],y[j],t)
    
    return [a,b]

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
def neumannnorth (a, b, bc, x, y, t):

    # Frontera superior, Neumann.

    n = len (x) - 2
    m = len (y) - 2

    j = m+1
    for i in range (n+2):
        k = j * (n+2) + i
        so = k - (n+2)

        a [k,k] = 1.0 / dy
        a [k,so] = -1.0 / dy
        b [k] = bc

    return [a,b]

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
def neumannsouth (a, b, bc, x, y, t):

    # Frontera inferior, Neumann.

    n = len (x) - 2
    m = len (y) - 2

    j = 0
    for i in range (n+2):
        k = i
        no = k + (n+2)

        a[k,k] = 1.0 / dy
        a[k,no] = -1.0 / dy
        b[k] = bc

    return [a,b]

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
def neumannwest (a, b, bc, x, y, t):

    # Frontera izquierda, Neumann.

    n = len (x) - 2
    m = len (y) - 2

    i = 0
    for j in range (m+2):
        k = j * (n+2) + i
        ea = k + 1

        a [k,:] = 0.0
        a[k,k] = 1.0 / dx
        a[k,ea] = -1.0 / dx
        b[k] = bc

    return [a,b]

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
    
def neumanneast (a, b, bc, x, y, t):

    # Frontera derecha, Neumann.

    n = len (x) - 2
    m = len (y) - 2

    i = n+1
    for j in range (m+2):
        k = j * (n+2) + i
        we = k - 1

        a [k,:] = 0.0
        a [k,k] = 1.0 / dx
        a [k,we] = -1.0 / dx
        b [k] = bc

    return [a,b]

# -------------------------------------------------------------------
# Condición inicial.

def init (x,y):

    return x

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

if (__name__ == '__main__'):

    # Coeficientes.

    r = 0.03
    sigma1 = 0.45
    sigma2 = 0.40
    rho = 0.15

    kappa = 1.0
    lamb = 0.4
    alpha = 0.5

    # Mallado.
    
    tfin = 2.0
    nt = 40
    n = 31
    m = 31
    theta = 0.5
    
    print ()
    
    tfin = float (tfin)
    xini = -1.0
    xfin =  1.0
    yini = -1.0
    yfin =  1.0
    
    nnod = (n + 2) * (m + 2)
    
    x = np.linspace (xini, xfin, n+2)
    y = np.linspace (yini, yfin, m+2)
    dx = x [1] - x [0]
    dy = y [1] - y [0]
    dt = tfin / nt
    
    xx = np.zeros (nnod, 'f')
    yy = np.zeros (nnod, 'f')
    for j in range (m+2):
        for i in range (n+2):
            k = j * (n+2) + i                   # Aquí pasamos de la numeración (i,j) a la numeración global.
            xx [k] = x [i]
            yy [k] = y [j]
    
    # -------------------------------------------------------------------
    # Condición inicial.
    
    nf = 1
    plt.figure (nf)
    nfc = 41
    plt.figure (nfc)
    
    uold = np.zeros (nnod, 'f')
    for j in range (m+2):
        for i in range (n+2):
            k = j * (n+2) + i
            uold [k] = init (x[i], y[j])
    plt.figure (nf)
    graf.graf02 (xx, yy, uold, nf)
    vax = [xini, xfin, yini, yfin, min(uold), max(uold)]
    
    # -------------------------------------------------------------------
    # Matriz del sistema.

    c11 = 0.5 * sigma1**2
    c12 = rho * sigma1 * sigma2 
    c22 = 0.5 * sigma2**2

    dxi = 1.0 / (2.0 * dx)
    dyi = 1.0 / (2.0 * dy)
    dxyi = 1.0 / (4.0 * dx * dy)
    dx2i = 1.0 / (dx**2)
    dy2i = 1.0 / (dy**2)
    
    a = np.zeros ((nnod,nnod), 'f')
    
    for j in range (1,m+1):
        for i in range (1,n+1):
            k = j * (n+2) + i
            no = k + (n+2)
            so = k - (n+2)
            ea = k + 1
            we = k - 1
            nw = no - 1
            ne = no + 1
            sw = so - 1
            se = so + 1
    
            a [k,k] = 1./dt + c11 * theta * x[i]**2 * 2.0 * dx2i + c22 * theta * 2.0 * dy2i
            a [k,we] = -c11 * theta * x[i]**2 * dx2i + (r - y[j]) * x[i] * theta * dxi
            a [k,ea] = -c11 * theta * x[i]**2 * dx2i - (r - y[j]) * x[i] * theta * dxi
            a [k,no] = -c22 * theta * dy2i - (kappa * (alpha - y[j]) - lamb * sigma2) * theta * dyi
            a [k,so] = -c22 * theta * dy2i + (kappa * (alpha - y[j]) - lamb * sigma2) * theta * dyi
            a [k,nw] = c12 * theta * x[i] * dxyi
            a [k,ne] = -c12 * theta * x[i] * dxyi
            a [k,sw] = -c12 * theta * x[i] * dxyi
            a [k,se] = c12 * theta * x[i] * dxyi
 
    # -------------------------------------------------------------------
    # Bucle temporal.
    
    th1 = 1.0 - theta
    plt.ion ()
    
    for it in range (nt):
        t = (it+1) * dt
        print ('    Instante de tiempo: %8.4f' % (t), end='')
    
        # Vector segundo miembro.
    
        b = np.zeros (nnod, 'f')
        for j in range (1,m+1):
            for i in range (1,n+1):
                k = j * (n+2) + i
                no = k + (n+2)
                so = k - (n+2)
                ea = k + 1
                we = k - 1
                nw = no - 1
                ne = no + 1
                sw = so - 1
                se = so + 1
    
                b [k] = uold [k] / dt  \
                      + c11 * th1 * x[i]**2 * (uold[we] - 2.0 * uold[k] + uold[ea]) * dx2i \
                      + c12 * th1 * x[i] * (uold[ne] - uold[se] - uold[nw] + uold[sw]) * dxyi \
                      + c22 * th1 * (uold[so] - 2.0 * uold[k] + uold[no]) * dy2i \
                      + (r - y[j]) * x[i] * th1 * (uold[ea] - uold[we]) * dxi \
                      + (kappa * (alpha - y[j]) - lamb * sigma2) * th1 * (uold[no] - uold[so]) * dxi
    
        # Condiciones de contorno.
    
        [a,b] = neumanneast  (a, b, 0.0, x, y, t)
        [a,b] = neumannwest  (a, b, 0.0, x, y, t)
        [a,b] = dirichsouth (a, b, bound00, x, y, t)
        [a,b] = dirichnorth (a, b, bound02, x, y, t)
    
        # -------------------------------------------------------------------
        # Resolución del sistema de ecuaciones.
    
        u = np.linalg.solve (a,b)
        plt.figure (nf)
        plt.cla ()
        plt.figure (nfc)
        plt.cla ()
        graf.graf02 (xx, yy, u, nf, vax)
        graf.graf01 (xx, yy, u, '  ', nfc)
        plt.draw ()
        #-- plt.                  label axis
        time.sleep (0.02)
    
        # Actualización.
    
        uold = u[:]
    
        print ()
    
    # -------------------------------------------------------------------
    # Representación gráfica.
    
    plt.ioff ()
    plt.show ()

# -------------------------------------------------------------------

