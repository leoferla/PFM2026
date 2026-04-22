import numpy as np
import pylab as plt

# -----------------------------------------------------------------------------------------------------

def fdate (fecha):
    f = fecha.split ('/')
    dd = f [0]
    mm = f [1]
    aa = f [2]

    return '%4s.%2s.%2s' % (dd, mm, aa)

# -----------------------------------------------------------------------------------------------------

def read01 ():

    fic = open ('spots.csv')
    regs = fic.readlines ()

    date = []
    asset01 = []
    asset02 = []

    for r in regs:
        a = r.split ('$')
        date.append (fdate (a[0]))
        asset01.append (float (a[1]))
        asset02.append (float (a[2]))

    return [date, asset01, asset02]

# -----------------------------------------------------------------------------------------------------

def read02 ():

    fic = open ('futures.csv')
    regs = fic.readlines ()

    d1a1maturity = []
    d1a2maturity = []
    d2a1maturity  = []
    d2a2maturity  = []
    d3a1maturity  = []
    d3a2maturity  = []
    d4a1maturity  = []
    d4a2maturity  = []
    d5a1maturity  = []
    d5a2maturity  = []
    d6a1maturity  = []
    d6a2maturity  = []
    """
    d7a1maturity  = []
    d7a2maturity  = []
    """

    d1a1value = []
    d1a2value = []
    d2a1value = []
    d2a2value = []
    d3a1value = []
    d3a2value = []
    d4a1value = []
    d4a2value = []
    d5a1value = []
    d5a2value = []
    d6a1value = []
    d6a2value = []
    """
    d7a1value = []
    d7a2value = []
    """

    for r in regs:
        a = r.split ('$')

        d1a1maturity.append (fdate (a [0]))
        d1a1value.append (float (a [1]))
        try:
            d1a2maturity.append (fdate (a [2]))
            d1a2value.append (float (a [3]))
        except:
            pass

        d2a1maturity.append (fdate (a [4]))
        d2a1value.append (float (a [5]))
        try:
            d2a2maturity.append (fdate (a [6]))
            d2a2value.append (float (a [7]))
        except:
            pass

        d3a1maturity.append (fdate (a [8]))
        d3a1value.append (float (a [9]))
        try:
            d3a2maturity.append (fdate (a [10]))
            d3a2value.append (float (a [11]))
        except:
            pass

        d4a1maturity.append (fdate (a [12]))
        d4a1value.append (float (a [13]))
        try:
            d4a2maturity.append (fdate (a [14]))
            d4a2value.append (float (a [15]))
        except:
            pass

        d5a1maturity.append (fdate (a [16]))
        d5a1value.append (float (a [17]))
        try:
            d5a2maturity.append (fdate (a [18]))
            d5a2value.append (float (a [19]))
        except:
            pass

        d6a1maturity.append (fdate (a [20]))
        d6a1value.append (float (a [21]))
        try:
            d6a2maturity.append (fdate (a [22]))
            d6a2value.append (float (a [23]))
        except:
            pass

        """
        d7a1maturity.append (fdate (a [24]))
        d7a1value.append (float (a [25]))
        try:
            d7a2maturity.append (fdate (a [26]))
            d7a2value.append (float (a [27]))
        except:
            pass
        """

    datosd1 = [d1a1maturity, d1a1value, d1a2maturity, d1a2value]
    datosd2 = [d2a1maturity, d2a1value, d2a2maturity, d2a2value]
    datosd3 = [d3a1maturity, d3a1value, d3a2maturity, d3a2value]
    datosd4 = [d4a1maturity, d4a1value, d4a2maturity, d4a2value]
    datosd5 = [d5a1maturity, d5a1value, d5a2maturity, d5a2value]
    datosd6 = [d6a1maturity, d6a1value, d6a2maturity, d6a2value]
    #-- datosd7 = [d7a1maturity, d7a1value, d7a2maturity, d7a2value]

    return [datosd1, datosd2, datosd3, datosd4, datosd5, datosd6]

# -----------------------------------------------------------------------------------------------------

def graph01 (date, asset01, asset02):

    fig, ax = plt.subplots (2,1)
    ax [0].plot (asset01, 'r-', lw=2, label='Asset 1')
    ax [0].grid (True)
    ax [0].legend ()

    ax [1].plot (asset02, 'g-', lw=2, label='Asset 2')
    ax [1].grid (True)
    ax [1].legend ()

# -----------------------------------------------------------------------------------------------------

def graph02 (currentdates, datafutures):

    for k, data in enumerate (datafutures):
        mat1 = data [0]
        asset01 = data [1]
        mat2 = data [2]
        asset02 = data [3]

        fig, ax = plt.subplots (2,1)
        ax [0].plot (asset01, 'r-', lw=2, label='Asset 1')
        ax [0].grid (True)
        ax [0].legend ()
    
        ax [1].plot (asset02, 'g-', lw=2, label='Asset 2')
        ax [1].grid (True)
        ax [1].legend ()
        fig.suptitle ('Current date: ' + currentdates [k])

# -----------------------------------------------------------------------------------------------------

if (__name__ == '__main__'):

    [date, asset01, asset02] = read01 ()
    graph01 (date, asset01, asset02)

    # --------------------------------------------------------

    currentdates = ['2026.02.27', '2026.02.13', '2026.01.30', '2026.01.15', '2025.12.31', '2025.12.15']        # Última: '2025.11.27'
    datafutures = read02 ()
    graph02 (currentdates, datafutures)

    # --------------------------------------------------------

    plt.show ()

# -----------------------------------------------------------------------------------------------------
