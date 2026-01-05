import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy
from scipy.signal import windows
from scipy.fftpack import fft, ifft, dct, idct
from scipy import integrate

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


def flagdata(nc, mode='PERIODIC', percent=20, seed=None):

    index = np.ones(nc, dtype=float)

    # apply seed only when random mode is used
    rng = np.random.default_rng(seed) if seed is not None else np.random

    if mode == 'PERIODIC':
        flag1 = np.array([0, 1, 2, 3, 16, 28, 29, 30, 31], dtype=int)
        flag = np.concatenate([flag1 + 32 * ii for ii in range(24)])
        flag = flag[flag < nc]  # safety guard if nc < full pattern
        index[flag] = 0

    elif mode == 'RANDOM':
        num = int(nc * percent / 100)
        num = min(num, nc)
        flag = rng.choice(nc, size=num, replace=False)
        index[flag] = 0
        
    elif mode == 'PERIODIC+RANDOM':
        
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



import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern

def generate_gp_realizations_skl(Npoints, amplitude, length_scale, Nreal=1, kernel_type="RBF", sigma=1.0):
    
    x = np.arange(Npoints)[:, None]

    # FIXED length scale (no optimization!)
    kernel = amplitude * RBF(length_scale=length_scale) # Matern(length_scale=1.0, nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)

    # # Fit GP to zero data to define structure (mean=0)
    # gp.fit(x, np.zeros(Npoints))

    # Sample from prior covariance
    samples = gp.sample_y(x, n_samples=Nreal).T
    
    return samples

import numpy as np
import george
from george import kernels

def generate_gp_realizations(Npoints, amplitude, length_scale, Nreal=1, kernel_type="RBF", sigma=1.0):
    x = np.arange(Npoints)[:, None]

    # 1. Define Kernel with UNIT variance (1.0)
    # This keeps the matrix values nice and small (~1.0)
    metric = length_scale**2
    
    if kernel_type == "RBF":
        # Note: Using 1.0 here, not 'amplitude'
        kernel = 1.0 * kernels.ExpSquaredKernel(metric=metric)
    elif kernel_type == "Matern32":
        kernel = 1.0 * kernels.Matern32Kernel(metric=metric)
    else:
        kernel = 1.0 * kernels.ExpSquaredKernel(metric=metric)

    # 2. Setup GP
    gp = george.GP(kernel)

    # 3. Precompute
    # Now yerr=1e-10 is effective because the signal is 1.0
    gp.compute(x, yerr=1e-8) 

    # 4. Sample Unit Variance
    # shape: (Nreal, Npoints)
    unit_samples = gp.sample(x, size=Nreal)
    
    # 5. Apply Amplitude Here
    # If 'amplitude' is the Variance (K_0), multiply by sqrt(amplitude)
    # If 'amplitude' is the Standard Deviation, multiply by amplitude.
    # Based on your variable name, I assume you want the values to scale by 'amplitude'.
    # Note: If your kernel definition was k = A * exp(...), then A is variance.
    # So we multiply samples by sqrt(A).
    
    # Assuming 'amplitude' input is the VARIANCE factor (standard in GP kernels):
    scaled_samples = unit_samples * np.sqrt(amplitude)
    
    # If 'amplitude' input is actually STD DEV (e.g. 1e12 Kelvin), use:
    # scaled_samples = unit_samples * amplitude

    return scaled_samples # Transpose to match (Npoints, Nreal) if needed


def smooth_vcg(aa, NW):
    # bb = np.ones(NN)/NN
    NN = 2*NW+1
    win = np.hanning(NN)
    #win = np.kaiser(NN, 14) # kaiser is similar to dpss 
    
    aas = np.convolve(aa, win, mode='valid')  
    
    V = aa.copy()
    V[V!=0.] = 1.
    Vs = np.convolve(V, win, mode='valid') 
#     print(V.shape, Vs.shape)
    aas = aas/Vs
    bb = aa[NW:-NW]
    aas[bb==0.] = 0.
    return aas

def smooth_gpr_controlled_incorrect(aa, NN): # simple, works, but mathematically incorrect # it works because of the jitter included in the GP implementation.

    x = np.arange(len(aa))[:,None]
    m = aa != 0
    y = aa[m]

    # FIXED length scale
    kernel = 1.0 * RBF(length_scale=NN) # Matern(length_scale=1.0, nu=1.5)
    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)

    gp.fit(x[m], y)
    y_pred = gp.predict(x)

    # keep original flags
    y_pred[~m] = 0
    return y_pred


import numpy as np
import george
from george import kernels
import scipy.optimize as op

def smooth_gpr_controlled(aa, NN):
    x = np.arange(len(aa))[:, None]
    m = aa != 0
    y_raw = aa[m]

    # --- Normalize Data ---
    y_mean = np.mean(y_raw)
    y_std = np.std(y_raw)
    y_norm = (y_raw - y_mean) / y_std

    # --- Define Kernels ---
    # Initial guess: 80% smooth, 20% fast
    # Since data is normalized (variance ~ 1.0), these should sum to ~1.0
    initial_amp_smooth = 0.8
    initial_amp_fast = 0.2
    
    k_smooth_base = kernels.ExpSquaredKernel(metric=NN**2)
    k_fast_base   = kernels.Matern32Kernel(metric=1.0**2)
    
    # George automatically converts these floats to ConstantKernels (amplitudes)
    kernel = initial_amp_smooth * k_smooth_base + initial_amp_fast * k_fast_base

    # --- Setup GP ---
    gp = george.GP(kernel, mean=0.0, fit_mean=False)
    
    # Compute factorization on NORMALIZED data
    # yerr=1e-5 is relative to a signal of size ~1.0 
    gp.compute(x[m], yerr=1e-5)

    # --- Optimization Loop ---
    # Freeze everything that isn't a "log_constant" (amplitude)
    for name in gp.get_parameter_names():
        if "metric" in name:
            gp.freeze_parameter(name)
            
    # -- WRAPPERS FOR SCIPY --
    
    # Objective Function (Negative Log Likelihood)
    def nll(p):
        gp.set_parameter_vector(p)
        # Use y_norm here
        ll = gp.log_likelihood(y_norm, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    # Gradient Function (Negative Gradient)
    def grad_nll(p):
        gp.set_parameter_vector(p)
        # Use y_norm here
        return -gp.grad_log_likelihood(y_norm, quiet=True)

    # Run the optimizer
    p0 = gp.get_parameter_vector()
    
    # Pass wrapper functions
    results = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B")
    
    # Update GP with best parameters
    gp.set_parameter_vector(results.x)
    
    # Print learned amplitudes 
    amps = np.exp(results.x)
    print(f"Learned Ratios -> Smooth: {amps[0]:.3f}, Fast: {amps[1]:.3f}")

    # --- Decomposition & Prediction ---
    
    # Calculate weights based on NORMALIZED data
    # y_norm needs to be shape (N,), apply_inverse handles the rest
    weights = gp.solver.apply_inverse(y_norm[:, None])[:, 0]
    
    # Extract the Smooth Kernel component
    k_smooth_fitted = gp.kernel.k1
    
    # Project weights using ONLY the smooth kernel
    # Pass 2D arrays: x (target) and x[m] (source)
    K_star_smooth = k_smooth_fitted.get_value(x, x[m])
    
    # prediction in "normalized units"
    y_pred_norm = K_star_smooth.dot(weights)

    # --- De-normalize ---
    # Scale back to original units
    y_pred = y_pred_norm * y_std + y_mean
    
    # Restore missing channels to 0 
    y_pred[~m] = 0 
    
    return y_pred



def bin_power_spectrum(n1, nend, k_vals, pk_recovered, P_theory, NB):

    # Extract relevant region
    # k_vals = k_abs[n1:nend]
    Nrea = pk_recovered.shape[0]

    # Linear binning
    bins = np.linspace(k_vals.min(), k_vals.max(), NB + 1)
    k_centers = 0.5 * (bins[:-1] + bins[1:])

    # Storage
    binned_rec = np.full((Nrea, NB), np.nan)

    # ---- Bin each realization ----
    for i in range(Nrea):
        pk_slice = pk_recovered[i, n1:nend]
        for j in range(NB):
            mask = (k_vals >= bins[j]) & (k_vals < bins[j+1])
            if np.any(mask):
                binned_rec[i, j] = np.mean(pk_slice[mask])

    # Ensemble stats
    p_rec_mean = np.nanmean(binned_rec, axis=0)
    p_rec_err  = np.nanstd(binned_rec, axis=0) / np.sqrt(Nrea)

    # ---- Bin theory ----
    binned_th = np.full(NB, np.nan)
    th_slice = P_theory[n1:nend]

    for j in range(NB):
        mask = (k_vals >= bins[j]) & (k_vals < bins[j+1])
        if np.any(mask):
            binned_th[j] = np.mean(th_slice[mask])

    return k_centers, p_rec_mean, p_rec_err, binned_th, bins

def process_scf(fields_tota, flag, SCF, NN_gp=96, NN_hann=50):
    """
    Process fields according to the chosen SCF scheme: 'GP', 'Hann', or 'None'.

    Parameters
    ----------
    fields_tota : ndarray
        Array of input realizations with shape (n_realizations, n_samples).
    flag : ndarray
        Flag array (same length as fields_tota realizations).
    SCF : str
        One of {'GP', 'Hann', 'None'}.
    pfunc : module/object
        Object containing smoothing and covtocl_fast functions.
    NN_gp : int
        Smoothing window/scale for GP.
    NN_hann : int
        Smoothing window/scale for Hann.

    Returns
    -------
    fields_orig : ndarray
        Original (unsmoothed - smoothed) fields after SCF processing.
    fields_flag : ndarray
        fields_orig multiplied by the (possibly trimmed) flag.
    ml : ndarray
        Separation-count array from covtocl_fast.
    """

    if SCF == 'GP':
        NN = NN_gp
        fields_smth = np.array([
            smooth_gpr_controlled(realization, NN)
            for realization in fields_tota
        ])
        fields_orig = fields_tota - fields_smth
        fields_flag = fields_orig * flag
        ml = covtocl_fast(flag, flag)

    elif SCF == 'Hann':
        NN = NN_hann
        fields_smth = np.array([
            smooth_vcg(realization, NN)
            for realization in fields_tota
        ])
        # Trim edges
        fields_orig = fields_tota[:, NN:-NN] - fields_smth
        trimmed_flag = flag[NN:-NN]
        fields_flag = fields_orig * trimmed_flag
        ml = covtocl_fast(trimmed_flag, trimmed_flag)

    elif SCF == 'None':
        fields_orig = fields_tota
        fields_flag = fields_orig * flag
        ml = covtocl_fast(flag, flag)

    else:
        raise ValueError("SCF must be one of {'GP', 'Hann', 'None'}")

    return fields_orig, fields_flag, ml

def pk_fft(fields_flag, L, N):
    pk_recovered = []
    
    for field in fields_flag:
        # field = field # - np.mean(field)
        fk = np.fft.fft(field)
        # pk_fft = (L / N**2) * (fk * fk.conjugate()).real
        pk_fft = (1 / L) * (fk * fk.conjugate()).real
        pk_recovered.append(pk_fft)
        
    pk_recovered = np.array(pk_recovered)
    pk_recovered_fft = pk_recovered
    return pk_recovered_fft

# returns A for a cos transform 
def calc_A(Na, Nb):
    A=np.outer(np.arange(Na),np.arange(Nb))
    A=np.cos(np.pi*A/(Nb-1.))
    A[:,0]=0.5*A[:,0]
    A[:,Nb-1]=0.5*A[:,Nb-1]
    return A
    
def pk_dct(fields_flag, dL, ml, M, r, w):
    pk_recovered = []
    cl_recovered = []
    for field in fields_flag:
        # field = field - np.mean(field)*0
        cl_full = covtocl_fast(field, field)/ml
        # cl_full -= np.mean(cl_full)
    
        # Use only M-terms of the correlation:
        cl = cl_full[:M]             # shape (M,)
    
        # --- DCT estimator (discrete Wiener–Khinchin, cosine basis) ---
        pk_dct = idct(cl.real * w, type=1)/dL # dL = N * (L / N**2) 
        
        # A = calc_A(M, M)    
        # X = np.linalg.inv(A)
        # pk_dct = (M-1) * X@(w*cl.real)/dL
        
        pk_recovered.append(pk_dct)
        cl_recovered.append(cl)
    
    pk_recovered_dct = np.array(pk_recovered)   # shape (Nrea, M)
    
    Omi = 1/(4*np.pi*r**2)
    cl_recovered = np.array(cl_recovered)*Omi   # shape (Nrea, M)

    return pk_recovered_dct, cl_recovered

def save_data(fname_prefix, inp_signal, flag, SCF,
             kb, mean, err, binned_th):

    fname = f"{fname_prefix}_inp_signal-{inp_signal}_flag-{flag}_SCF-{SCF}.npz"

    np.savez(
        fname,
        kb=kb,
        mean=mean,
        err=err,
        binned_th=binned_th,
        inp_signal=inp_signal,
        flag=flag,
        SCF=SCF
    )

    # print("Saved:", fname)
    return fname

"""
def pk_intp(kv, karr, pkarr): # kv, k-array , pk-array
    y = np.interp(np.log(kv), np.log(karr), np.log(pkarr))
    return np.exp(y) # np.interp(kv, karr, pkarr)


def pk2cl(x, rp, dn, kp, dnuc, karr, pkarr): # x: kpara-array, dn: ndnu, kp:kper, dnuc:dnuc.
    kv = np.sqrt(kp**2 + x**2)
    y = np.cos(x*rp*dn) * pk_intp(kv, karr, pkarr) * np.power(np.sinc(x*rp*dnuc/(2.*np.pi)), 2.) 
    # less than 1% change at dnu=0 when sinc is omitted.
    return y


# Predicted Signal in a grid
def clHI(r, rp, kperp, dnu, karr, pkarr): # cl for 21 cm signal.
    nc, dnuc = len(dnu), dnu[1]
    cl21 = np.zeros(nc)
    x = np.linspace(0, 100*kperp, 10000)
    for jj in range(nc):
        cl21[jj] = integrate.simpson(pk2cl(x, rp, dnu[jj], kperp, dnuc, karr, pkarr), x)/(np.pi*r**2)
            
    return cl21

"""