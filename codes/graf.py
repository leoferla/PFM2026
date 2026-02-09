# encoding: utf-8

import numpy as np
import pylab as plt
import time

from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# -----------------------------------------------------------------------------------------

def graf01 (x, y, u, titulo=' ', nf=1):

    """
    Representación gráfica de una superficie sobre un dominio rectangular mediante curvas de nivel.

    Entradas:
    - xx:     lista de las abscisas de los nodos.
    - yy:     lista de las ordenadas de los nodos.
    - u:      lista de los valores a representar.
    - nf:     número de figura.
    - titulo: título.
    """

    # Representación gráfica.

    xmin = min (x)
    xmax = max (x)
    ymin = min (y)
    ymax = max (y)

    xi = plt.linspace (xmin, xmax, 101)
    yi = plt.linspace (ymin, ymax, 101)
    zi = griddata ((x,y), u, (xi[None,:], yi[:,None]), method='cubic')

    plt.figure (nf)
    plt.contour (xi,yi,zi,15,colors='k')
    plt.contourf (xi,yi,zi,15)
    #-- plt.colorbar ()
    #-- plt.scatter (xx,yy, marker='o', c='b', s=5)
    plt.axis ('equal')
    plt.axis ([xmin-0.05, xmax+.05, ymin-0.05, ymax+0.05])
    plt.axis ('off')
    plt.title (titulo)

# -----------------------------------------------------------------------------------------

def graf02 (x, y, u, nfig=1, vax=[]):

    fig = plt.figure (nfig)
    ax = fig.add_subplot (111, projection='3d')
    ax.plot_trisurf (x, y, u, cmap=cm.copper, antialiased=True, linewidth=0.0, shade=False)
    ax.plot ([1., 1.], [1., 1.], [0., 10.], 'k.')

# --------------------------------------------------------------------

if (__name__ == '__main__'):

    xx = [0., 0.5, 1., 0., 0.5, 1., 0., 0.5, 1.]    # Lista de las abscisas.
    yy = [0., 0., 0., 0.5, 0.5, 0.5, 1., 1., 1.]    # Lista de las ordenadas.
    u = [0., 0., 0., 0., 2., 1., 0., 3., 2.]        # Función a representar, discretizada en los nodos.

    deltat = 0.1

    plt.figure (1)
    #-- plt.figure (2)
    plt.ion ()

    for it in range (10):
        print ('    Instante de tiempo: %5.2f' % (it * deltat))

        u [4] += 0.1
        u [5] -= 0.1

        plt.figure (1)
        plt.cla ()
        #-- plt.figure (2)
        #-- plt.cla ()

        graf01 (xx, yy, u, 'Titulo', 1)
        #-- graf02 (xx, yy, u, 2)
        plt.draw ()
        time.sleep (0.02)

    plt.ioff ()
    graf01 (xx, yy, u, 'Titulo', 1)
    #-- graf02 (xx, yy, u, 2)
    plt.show ()

# --------------------------------------------------------------------

