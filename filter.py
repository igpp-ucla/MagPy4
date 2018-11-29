# filter.py - Short functions that deal with filtering data, used in FilterDialog.py
# and perhaps useful elsewhere.

import numpy

from math import pi, pow, sin, cos, cosh, acos, acosh, isnan, sqrt
from numpy import ndarray
from numpy import sin as SIN

# nf -> number frequency
# fc -> frequency cutoff
# dt -> resolution
# later make method to produce X
# def genX(nf, dx=0):

def XList(nf, DX=0, START=0):
    """
    """
    n = int((nf + 1) / 2)
    if nf % 2:    # odd
        start = 0
        dx = DX
    else:         # even
        start = START
        dx = 0
    X = [x + dx for x in range(start, n, 1)]
    return X

def genXArray(nf, DX=0, START=0):
    """
    """
    n = int((nf + 1.0) / 2.0)
    x = numpy.arange(START, n + START, 1) + DX
    return x

def genXList(nf, DX=0, START=0):
    """
    """
    n = int((nf + 1.0) / 2.0)
    X = [x + DX for x in range(START, n + START, 1)]
    return X

# Routine to generate Chebyshev window parameters when one of the three
# parameters nf (number of points, filter length), dp (filter ripple),
# or df (normalized transition width of filter) is unspecified.

def chebyshevParameters(nfx, dpx, dfx):
    """
    """
    nf = nfx
    dp = dpx
    df = dfx
    if df < 1.0 and df > 0.4:
        df = 0.4
    if nf == 0:
        # dp and df are specified; determine nf
        c0 = cos(pi * df)
        c1 = acosh((1.0 + dp) / dp)
        x = 1.0 + c1 / acosh(1.0 / c0)
        # Increment by 1 to give nf, which meets or exceeds specs on dp and df.
        nfx = x + 1
    elif df == 0.0:
        # nf and dp are specified; determine df
        xn = nf - 1
        c1 = acosh((1.0 + dp) / dp)
        c2 = cosh(c1 / xn)
        dfx = acos(1.0 / c2) / pi
    else:
        # nf and df are specified; determine dp
        xn = nf - 1
        c0 = cos(pi * df)
        c1 = xn * acosh(1.0 / c0)
        dpx = 1.0 / (cosh(c1) - 1.0)
    return nfx, dpx, dfx

# Filter list (g) generators

def lowpassResponse(nf, fc, dt):
    """
    """
    start = 1 if nf % 2 else 0
    dx = 0 if nf % 2 else 0.5
    X = genXArray(nf, DX=dx, START=start)
    f = 2.0 * pi * fc * dt
    G = SIN(f * X) / pi / X
    if nf % 2:
        g = [2.0 * fc * dt]
        g.extend(G)
        n = G.size
        gn = numpy.empty(n)
        gn[0] = 2.0 * fc * dt
        gn[1:] = G[:-1]
        return g[:-1]
    return G

def bandpassResponse(nf, lowf, highf, dt):
    """
    """
    f1 = pi * (highf - lowf) * dt
    f2 = pi * (highf + lowf) * dt
    start = 1 if nf % 2 else 0
    dx = 0 if nf % 2 else 0.5
    X = genXList(nf, DX=dx, START=start)
    G = [sin(f1 * x) / pi / x * 2 * cos(f2 * x) for x in X]
    if nf % 2:
        g = [2.0 * (highf - lowf) * dt]
        g.extend(G)
        return g[:-1]
    return G

# Coefficients list (w) generators

def rectangularWindow(nf):
    """
    """
    n = int((nf + 1) / 2)
    window = [1 for i in range(n)]
    return window

def triangularWindow(nf):
    """
    """
    n = int((nf + 1) / 2)
    X = XList(nf, DX=0.5, START=1)
    window = [1.0 - x / n for x in X]
    return window

def hammingWindow(nf, alpha, beta):
    """
    """
    fn = nf - 1
    X = XList(nf, DX=0.5, START=1)
    window = [alpha + beta * cos((2.0 * pi * x) / fn) for x in X]
    return window

def ino(x):
    """
    """
    e = 1
    de = 1
    x *= 0.5
    for i in range(1, 26, 1):
        de *= x / i
        sde = de * de
        e += sde
        if (e * 1.0e-8 - sde > 0.0):
            break
    return e

def kaiserWindow(nf, beta):
    """
    """
    bes = ino(beta)
    xind = (nf - 1) * (nf - 1)
    dx = 0.0 if nf % 2 else 0.5
    X = XList(nf, DX=dx, START=0)
    window = [ino(beta * sqrt(1 - (4.0 * x * x / xind))) / bes for x in X]
    if isnan(window[-1]):
        window[-1] = 0
    return window

def chebyshevWindow(nf, dp, df):
    """
    """
    if 0.4 < df < 1.0:
        df = 0.4
    xo = (3 - cos(pi * 2 * df)) / (1.0 + cos(pi * 2 * df))
    alpha = (xo + 1) / 2
    beta = (xo - 1) / 2
    c2 = (nf - 1) / 2
    f = [i / nf for i in range(nf)]
    X = [alpha * cos(pi * 2 * f[i]) + beta for i in range(nf)]
    P = [dp * coshOrcos(c2, x) for x in X]
    Pi = [0] * nf
    Pr = [0] * nf
    if nf % 2:
        Pi = [0] * nf
        Pr = P
    else:
        half = int(nf / 2)
        Pi[0:half] = [-P[i] * sin(pi * f[i]) for i in range(half)]
        Pr[0:half] = [+P[i] * cos(pi * f[i]) for i in range(half)]
        Pi[half:nf] = [+P[i] * sin(pi * f[i]) for i in range(half, nf)]
        Pr[half:nf] = [-P[i] * cos(pi * f[i]) for i in range(half, nf)]
    twn = pi * 2 / nf
    n = int((nf + 1) / 2)
    W = [0] * n
    for i in range(n):
        slist = [Pr[j] * cos(twn * i * j) + Pi[j] * sin(twn * i * j) for j in range(nf)]
        W[i] = sum(slist)
    w0 = W[0]
    window = [w / w0 for w in W]
    return window

def GByB(G, B, hnf):
    """G list, B{x,y,z,t} component list
    """
    nG = len(G)
    nB = len(B)
    SUM = [] * nB
    for i in range(nB):
        SUM[i][0:hnf] = [G[hnf - j - 1] * B[j] for j in range(0, hnf, 1)]
        SUM[i][hnf:-1] = [G[j - hnf + 1] * B[j] for j in range(hnf, nG, 1)]
    BFiltered = [sum(SUM[i]) for i in range(nB)]
    return BFiltered

def coshOrcos(c2, x):
    """ cosh() or cos()
    """
    X = cosh(c2 * acosh(x)) if abs(x) - 1 > 0 else cos(c2 * acos(x))
    return X
