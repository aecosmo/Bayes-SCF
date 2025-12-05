import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy
from scipy.signal import windows
from scipy.fftpack import fft, ifft, dct

# Cosmological parameters: (Planck 2018)
H0 = 67.66 # km / (Mpc s)
Om0 = 0.30966
Ode0 = 0.6888
# Ode0 = 1-Om0
c = 2.99792458e8 # speed of light

def inv_efunc(x): # Function used to calculate H(z), the Hubble parameter.
    # yy = 1/np.sqrt(Ogamma0*np.power((1+x),4) + Om0*np.power((1+x),3)+ Ok0*np.power((1+x),2.) + Ode0) # use Ode0
    yy = 1./np.sqrt(Om0*np.power((1+x),3)+ (1.-Om0)) # use Ode0
    return yy

def rrprime(zz, method):
    if (method == 'astropy'):
        from astropy.cosmology import FlatLambdaCDM # other choices: FLRW, wCDM, FlatLambdaCDM, and ohers
        #from astropy.cosmology import LambdaCDM # other choices: FLRW, wCDM, FlatLambdaCDM, and ohers

        cosmo = FlatLambdaCDM(H0, Om0)
        #cosmo = LambdaCDM(H0, Om0, Ode0)

        dc = cosmo.comoving_distance(zz) #https://docs.astropy.org/en/stable/api/astropy.cosmology.FlatLambdaCDM.html
        rprime = cosmo.hubble_distance*cosmo.inv_efunc(zz)*(1.+zz)**2/1420.

        r = dc.value
        rprime = rprime.value  # converting from astropy quantity type to np.float64 type

        print(f"Redshift = {zz} ")
        print(f'Comoving distance (from astropy): {dc}')
        print(f"rprime (from astropy) = {rprime} Mpc/MHz")

    if (method == 'analytical'):
        from scipy.integrate import quad
        cbyH0 = c*1e-3/(H0) # c*1e-3 speed of light in km/s, Hubble distance: cbyH0 in Mpc
        r = (cbyH0)*quad(inv_efunc,0,zz)[0]
        rprime = cbyH0*inv_efunc(zz)*(1.+zz)**2/1420.

        print(f"Redshift = {zz} ")
        print(f"r = {r} Mpc")
        print(f"rprime = {rprime} Mpc/MHz")
        print("Compare with the values obtained using astropy!")

    return r, rprime


def pk_intp(kv, karr, pkarr): # kv, k-array , pk-array
    y = np.interp(np.log(kv), np.log(karr), np.log(pkarr))
    return np.exp(y) # np.interp(kv, karr, pkarr)


def covtocl_fast(a, b):
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    n = len(a)
    out = np.empty(n, dtype=np.complex128)
    for k in range(n):
        out[k] = np.vdot(b[k:], a[:n-k]) # +  np.vdot(a[k:], b[:n-k])   
        
    return out


def bin_array(array, nb, k=None, log=False):

    array = np.asarray(array)
    
    # ---- Case 1: No k -> simple index binning ----
    if k is None:
        n = len(array)
        bins = np.linspace(0, n, nb+1, dtype=int)
        out   = np.zeros(nb)
        count = np.zeros(nb, dtype=int)
        
        for i in range(nb):
            chunk = array[bins[i]:bins[i+1]]
            out[i]   = np.mean(chunk) if len(chunk) > 0 else np.nan
            count[i] = len(chunk)
        
        return out, None, count

    # ---- Case 2: k-binning ----
    k = np.asarray(k)

    if len(array) != len(k):
        raise ValueError(f"bin_array: array length {len(array)} != k length {len(k)}")

    # set bin edges
    if log:
        kmin  = np.min(k[k > 0])
        edges = np.logspace(np.log10(kmin), np.log10(k.max()), nb+1)
    else:
        edges = np.linspace(k.min(), k.max(), nb+1)

    ybin  = np.zeros(nb)
    kbin  = np.zeros(nb)
    count = np.zeros(nb, dtype=int)

    for i in range(nb):
        if i < nb-1:
            mask = (k >= edges[i]) & (k < edges[i+1])
        else:
            mask = (k >= edges[i]) & (k <= edges[i+1])

        count[i] = np.sum(mask)

        if count[i] > 0:
            ybin[i] = np.mean(array[mask])
            kbin[i] = np.mean(k[mask])
        else:
            ybin[i] = np.nan
            kbin[i] = 0.5*(edges[i]+edges[i+1]) if not log else np.sqrt(edges[i]*edges[i+1])

    return ybin, kbin, count

"""
def flagdata(nc, mode='MWA', percent = 20):
    index = np.ones(nc)
    if (mode=='MWA'):
        flag1 = np.array([0, 1, 2, 3, 16, 28,29,30,31], dtype='int64')
        flag = np.array([], dtype='int64')
        for ii in range(24):
            flag = np.append(flag, flag1+32*ii)
        index[flag] = 0
    elif (mode=='RANDOM'):
        num = int(nc*percent/100)
        num = nc if (num > nc) else num
        flag = np.random.randint(0, nc, num)
        index[flag] = 0
    else:
        index = np.ones(nc)

    return index
"""

def flagdata(nc, mode='MWA', percent=20, seed=None):
    """
    Generate flag mask for spectral channels.

    Parameters
    ----------
    nc : int
        Number of channels.
    mode : str
        'MWA', 'RANDOM', or anything else meaning NOFLAG.
    percent : float
        Percentage of channels to randomly flag (only used if mode='RANDOM').
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    index : array
        Binary mask of shape (nc,), where 0 = flagged, 1 = unflagged.
    """

    index = np.ones(nc, dtype=float)

    # apply seed only when random mode is used
    rng = np.random.default_rng(seed) if seed is not None else np.random

    if mode == 'MWA':
        flag1 = np.array([0, 1, 2, 3, 16, 28, 29, 30, 31], dtype=int)
        flag = np.concatenate([flag1 + 32 * ii for ii in range(24)])
        flag = flag[flag < nc]  # safety guard if nc < full pattern
        index[flag] = 0

    elif mode == 'RANDOM':
        num = int(nc * percent / 100)
        num = min(num, nc)
        flag = rng.choice(nc, size=num, replace=False)
        index[flag] = 0
        
    elif mode == 'MWA+RANDOM':
        
        flag1 = np.array([0, 1, 2, 3, 16, 28, 29, 30, 31], dtype=int)
        flag = np.concatenate([flag1 + 32 * ii for ii in range(24)])
        flag = flag[flag < nc]  # safety guard if nc < full pattern
        index[flag] = 0
        
        num = int(nc * percent / 100)
        num = min(num, nc)
        flag = rng.choice(nc, size=num, replace=False)
        index[flag] = 0    

    return index


def draw_field_from_power(P_dft, seed=None):
    rng = np.random.default_rng(seed)

    N = len(P_dft)
    fk = np.zeros(N, dtype=complex)

    # k=0 mode
    fk[0] = 0 * rng.standard_normal() # zero-mean field

    # Nyquist mode (only if N even)
    if N % 2 == 0:
        fk[N//2] = np.sqrt(P_dft[N//2]/2) * rng.standard_normal()

    # Fill positive + mirrored negative modes
    for i in range(1, N//2):
        a, b = rng.standard_normal(), rng.standard_normal()
        amp = np.sqrt(P_dft[i] / 2)
        fk[i]  = amp * (a + 1j * b)
        fk[-i] = np.conjugate(fk[i])

    return np.fft.ifft(fk).real



# plot
import matplotlib.pyplot as plt
import numpy as np

