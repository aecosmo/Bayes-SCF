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

def get_model_pk(kp, kpara, KK, PK):
    nn = len(kpara)
    kv = np.zeros(nn)
    PKm = np.zeros(nn)
    for jj in range(nn):
        kv[jj] = np.sqrt(kp**2 + kpara[jj]**2)
        
        PKm[jj] =  pk_intp(kv[jj], KK, PK)
        # PKm[jj] =  np.interp(np.log(kv[jj]), np.log(KK), np.log(PK))

    return kv, PKm
    
def get_model_pk_2d(kper, kpara, KK, PK):
    nc = len(kpara)
    nell = len(kper)
    
    PKm = np.zeros((nell, nc))
    
    for ii in range(nell):
        for jj in range(nc):
            kv = np.sqrt(kper[ii]**2 + kpara[jj]**2)
            PKm[ii, jj] =  pk_intp(kv, KK, PK)
    return PKm
    
import numpy as np
from numpy.fft import ifft

def get_vis(PKm, r, rp, dnuc):
    """
    Generate Hermitian-symmetric Fourier coefficients consistent with PKm,
    so that inverse FFT gives a real brightness-temperature field whose
    variance follows the desired power-law slope.

    Parameters
    ----------
    PKm : array_like
        1D power spectrum evaluated on the FFT modes (same length as N).
    r : float
        Normalization/geometry factor (you are currently using r**2).
    rp, dnuc : float
        Currently unused, but can enter into the overall normalization
        if you want full physical units.
    """
    PKm = np.asarray(PKm)
    N   = PKm.size

    fact = r**2

    randX = np.zeros(N, dtype=np.complex128)

    # ---- DC mode (k = 0), purely real ----
    randX[0] = np.random.normal() * np.sqrt(PKm[0] / (2.0 * fact))

    # ---- Nyquist mode (only if N is even), purely real ----
    if N % 2 == 0:
        randX[N//2] = np.random.normal() * np.sqrt(PKm[N//2] / (2.0 * fact))
        kmax = N//2
    else:
        kmax = (N - 1) // 2 + 1  # last positive k index

    # ---- positive-frequency complex modes and their conjugates ----
    for k in range(1, kmax):
        re = np.random.normal()
        im = np.random.normal()
        z  = re + 1j * im
        amp = np.sqrt(PKm[k] / (2.0 * fact))
        randX[k]  = amp * z
        randX[-k] = np.conjugate(randX[k])

    # ---- inverse FFT -> real-space brightness temperature along ν ----
    VX = ifft(randX)  # should be real up to numerical noise

    # If you want exactly real:
    VX = VX.real

    return VX


import numpy as np
from numpy.fft import ifft

def get_vis_unit(PKm, r, rp, dnuc):
    """
    Generate Hermitian-symmetric Fourier coefficients consistent with PKm,
    so that inverse FFT gives a real brightness-temperature field whose
    variance follows the desired power-law slope.

    Parameters
    ----------
    PKm : array_like
        1D power spectrum evaluated on the FFT modes (same length as N).
    r : float
        Normalization/geometry factor (you are currently using r**2).
    rp, dnuc : float
        Currently unused, but can enter into the overall normalization
        if you want full physical units.
    """
    PKm = np.asarray(PKm)
    PKm = np.ones_like(PKm)
    N   = PKm.size

    fact = r**2

    randX = np.zeros(N, dtype=np.complex128)

    # ---- DC mode (k = 0), purely real ----
    randX[0] = np.random.normal() * np.sqrt(PKm[0] / (2.0 * fact))

    # ---- Nyquist mode (only if N is even), purely real ----
    if N % 2 == 0:
        randX[N//2] = np.random.normal() * np.sqrt(PKm[N//2] / (2.0 * fact))
        kmax = N//2
    else:
        kmax = (N - 1) // 2 + 1  # last positive k index

    # ---- positive-frequency complex modes and their conjugates ----
    for k in range(1, kmax):
        re = np.random.normal()
        im = np.random.normal()
        z  = re + 1j * im
        amp = np.sqrt(PKm[k] / (2.0 * fact))
        randX[k]  = amp * z
        randX[-k] = np.conjugate(randX[k])

    # ---- inverse FFT -> real-space brightness temperature along ν ----
    VX = ifft(randX)  # should be real up to numerical noise

    # If you want exactly real:
    VX = VX.real

    return VX


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


def spherical_bin_log(kper, kpar, pk2d, Nbin=20, kmin=None, kmax=None):
    """
    Log-spaced spherical binning of a 2D power spectrum P(kper, kpar).

    Parameters
    ----------
    kper : 1D array
        Perpendicular wavenumbers.
    kpar : 1D array
        Parallel wavenumbers.
    pk2d : 2D array (len(kpar), len(kper))
        Power spectrum on the (kpar, kper) grid.
    Nbin : int
        Number of spherical bins.
    kmin, kmax : float (optional)
        Range for spherical k. If None, computed from data.

    Returns
    -------
    k_bin_center : array
        Logarithmic spherical k bin centers.
    pk_1d : array
        Spherically averaged 1D P(k).
    counts : array
        Number of contributing 2D cells in each bin.
    """

    # Meshgrid of all k-space points
    KP, KPAR = np.meshgrid(kper, kpar)
    K = np.sqrt(KP**2 + KPAR**2)

    # Flatten arrays
    Kflat = K.ravel()
    PKflat = pk2d.ravel()

    # Define log-spaced bins
    if kmin is None: kmin = Kflat[Kflat > 0].min()
    if kmax is None: kmax = Kflat.max()

    edges = np.geomspace(kmin, kmax, Nbin + 1)
    k_bin_center = np.sqrt(edges[:-1] * edges[1:])  # geometric mean

    # Digitize
    inds = np.digitize(Kflat, edges) - 1

    # Storage
    pk_1d = np.zeros(Nbin)
    counts = np.zeros(Nbin, dtype=int)

    # Loop over bins
    for i in range(Nbin):
        mask = inds == i
        if np.any(mask):
            pk_1d[i] = np.mean(PKflat[mask])
            counts[i] = np.sum(mask)
        else:
            pk_1d[i] = np.nan  # empty bin

    return k_bin_center, pk_1d, counts
