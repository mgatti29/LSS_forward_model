"""
High-redshift lognormal extension of a lightcone for CMB lensing (compact CoSHiREx).

After CoSHiREx (Fusari, Jeffrey & Joachimi, in prep.; github.com/LorenzoFusari/CoSHiREx): beyond the last
simulated shell the matter field is continued with lognormal shells (top-hat windows uniform in scale factor up
to z = 1100) and the CMB lensing convergence is the Born sum over simulated + generated shells (glass
multi-plane weights, source plane at z = 1100).

Two modes (ladder_kappa):
  l_exact = 0 (default, fast)  Limber shell spectra (seconds).  Limber cross-spectra of non-overlapping shells
      vanish, so the generated shells are independent lognormal fields, drawn in parallel.  The kappa spectrum is
      exact to Limber accuracy; dropped: the ell < ~30 correlation between the simulated and generated parts
      (expected C_kk +0.4% at ell = 20, +5% at ell = 2, always < 1/10 of cosmic variance).
  l_exact = X  the multipoles ell <= X are drawn as in the reference code: exact line-of-sight CAMB auto- and
      cross-spectra and Gaussian conditioning on the shells less than delta_chi before (simulated shells
      Gaussianised as g = log(delta + shift)); ell > X independent with Limber spectra.  l_exact = 2 nside is
      the reference method at all ell (tens of minutes of CAMB at lmax 4000).

Recipe as the reference: nonlinear total-matter shell spectra with the HEALPix pixel window imprinted, shift 1,
ridge 1e-2 trace/K, real m = 0, delta = shift expm1(g - var/2).
"""

import copy
import multiprocessing
import os

import camb
import ducc0
import healpy as hp
import numpy as np
from cosmology import Cosmology
import glass.fields
import glass.lensing
import glass.shells


def triangle_index(i, j):
    """Position of the spectrum (i, j) in the glass list (0,0), (1,1), (1,0), (2,2), (2,1), (2,0), ..."""
    if j > i:
        i, j = j, i
    return i*(i + 1)//2 + (i - j)


def tophat_window(z_min, z_max, dz=0.001, max_points=400):
    """Top-hat RadialWindow sampled every dz, with at most max_points samples (CAMB cannot handle ~1e6)."""
    n = int(np.clip(np.ceil((z_max - z_min)/dz) + 1, 3, max_points))
    za = np.linspace(z_min, z_max, n)
    return glass.shells.RadialWindow(za, np.ones(n), 0.5*(z_min + z_max))


def ladder_windows(z_start, z_end=1100., num=100):
    """Top-hat windows uniform in scale factor between z_start and z_end (num shells)."""
    z = (1/np.linspace(1/(1 + z_end), 1/(1 + z_start), num + 1) - 1)[::-1]
    return [tophat_window(z[i], z[i + 1]) for i in range(num)]


def born_weights(windows, pars, z_source=1100.):
    """Born weights q_i of the shells (increasing z): kappa = sum_i q_i delta_i, identical to feeding the shells
    to glass MultiPlaneConvergence (which is linear in delta)."""
    n = len(windows)
    convergence = glass.lensing.MultiPlaneConvergence(Cosmology.from_camb(pars))
    for i, w in enumerate(windows):
        convergence.add_window(np.eye(n)[i], w)
    convergence.add_plane(np.zeros(n), z_source)
    return np.asarray(convergence.kappa, dtype=float)


def shell_cls(pars, windows, lmax, nonlinear=True, total_matter=True, kmax=20.):
    """
    Exact (line-of-sight) CAMB auto- and cross-spectra of the matter density in the shells (glass triangle
    order).  Total matter: the CAMB density spectra are rescaled by sqrt(R_i R_j), R = P_tot / P_cdm at
    k = (ell + 1/2) / chi_i.  The cost grows steeply with lmax.
    """
    p = copy.deepcopy(pars)
    p.Want_CMB = False
    p.WantCls = True                                    # (segfault otherwise)
    p.WantTransfer = True
    p.min_l = 1
    # the transfer kmax (default 0.9) must cover k = ell / chi of the nearest shells, or high-ell power is lost;
    # set_for_lmax(lens_potential_accuracy) switches NonLinear, so it is set after
    p.set_for_lmax(lmax, lens_potential_accuracy=1)
    p.Transfer.kmax = max(p.Transfer.kmax, kmax)
    p.NonLinear = camb.model.NonLinear_both if nonlinear else camb.model.NonLinear_none
    p.SourceTerms.limber_windows = False
    p.SourceTerms.counts_density = True
    for term in ('counts_redshift', 'counts_lensing', 'counts_velocity', 'counts_radial', 'counts_timedelay',
                 'counts_ISW', 'counts_potential', 'counts_evolve'):
        setattr(p.SourceTerms, term, False)
    p.SourceWindows = [camb.sources.SplinedSourceWindow(z=w.za, W=w.wa) for w in windows]
    results = camb.get_results(p)
    raw = results.get_source_cls_dict(lmax=lmax, raw_cl=True)
    n = len(windows)
    cls = [np.asarray(raw[f'W{i}xW{j}'], dtype=float) for i in range(1, n + 1) for j in range(i, 0, -1)]
    if total_matter:
        ell, h = np.arange(lmax + 1) + 0.5, p.H0/100.
        kw = dict(nonlinear=nonlinear, hubble_units=True, k_hunit=True, kmax=50., zmax=max(w.zeff for w in windows) + 1)
        p_tot = camb.get_matter_power_interpolator(copy.deepcopy(pars), var1='delta_tot', var2='delta_tot', **kw)
        p_cdm = camb.get_matter_power_interpolator(copy.deepcopy(pars), var1='delta_cdm', var2='delta_cdm', **kw)
        ratio = [p_tot.P(w.zeff, k)/p_cdm.P(w.zeff, k) for w in windows
                 for k in [np.clip(ell/(results.comoving_radial_distance(w.zeff)*h), 1e-4, 50.)]]
        for i in range(n):
            for j in range(i + 1):
                cls[triangle_index(i, j)] *= np.sqrt(ratio[i]*ratio[j])
    return cls


def limber_shell_cls(pars, windows, lmax, nonlinear=True, kmax=10., n_nodes=16):
    """
    Limber auto-spectra of the total matter density in non-overlapping top-hat shells (one per window).
    Per-shell spectra are not exact at ell < chi / shell width, but their weighted sum, the CMB lensing
    spectrum, is the Limber one (broad kernel).
    """
    p = copy.deepcopy(pars)
    p.NonLinear = camb.model.NonLinear_both if nonlinear else camb.model.NonLinear_none
    background = camb.get_background(p)
    z_lo, z_hi = min(float(np.min(w.za)) for w in windows), max(float(np.max(w.za)) for w in windows)
    pk = camb.get_matter_power_interpolator(p, zs=np.expm1(np.linspace(np.log1p(z_lo), np.log1p(z_hi), 150)),
                                            nonlinear=nonlinear, var1='delta_tot', var2='delta_tot',
                                            hubble_units=False, k_hunit=False, kmax=kmax, extrap_kmax=1e3)
    x, wx = np.polynomial.legendre.leggauss(n_nodes)
    ell = np.arange(lmax + 1) + 0.5
    cls = []
    for w in windows:
        z0, z1 = float(np.min(w.za)), float(np.max(w.za))
        z = 0.5*(z1 - z0)*x + 0.5*(z1 + z0)
        chi = background.comoving_radial_distance(z)
        # int dchi W(chi)^2 P / chi^2 with W(chi) = H / (c dz)  ->  int dz (H / c) / dz^2 P / chi^2
        weight = 0.5*wx*background.h_of_z(z)/(z1 - z0)/chi**2
        cl = sum(wj*pk.P(zj, ell/cj) for wj, zj, cj in zip(weight, z, chi))
        cl[:2] = 0.
        cls.append(cl)
    return cls


def gaussian_spectra(cls, windows, nside, shift=1.):
    """Gaussian spectra of the shifted lognormal fields, with the pixel window of nside imprinted."""
    cls = glass.fields.discretized_cls(cls, nside=nside)
    fields = glass.fields.lognormal_fields(windows, shift=lambda z: shift)
    return glass.fields.solve_gaussian_spectra(fields, cls)


def gaussian_alm(cl, lmax, rng):
    """Gaussian alm with spectrum cl (no monopole / dipole, real m = 0)."""
    ell, m = hp.Alm.getlm(lmax)
    cl = np.clip(np.asarray(cl, dtype=float), 0., None)
    cl[:2] = 0.
    scale = np.sqrt(cl[ell]/np.where(m == 0, 1., 2.))
    alm = scale*(rng.normal(size=len(ell)) + 1j*rng.normal(size=len(ell)))
    alm[m == 0] = alm[m == 0].real
    return alm


def gaussianise(delta, shift=1.):
    """g = log(delta + shift) - <g> of a simulated shell (raises if delta <= -shift somewhere)."""
    x = np.asarray(delta, dtype=float) + shift
    if not np.all(np.isfinite(x) & (x > 0)):
        raise ValueError(f'{np.count_nonzero(~(x > 0))} pixels with delta <= -shift: cannot Gaussianise this shell')
    g = np.log(x)
    return g - g.mean()


def conditional_alm(previous_alm, gls, previous_index, target_index, lmax, rng, ridge=1e-2):
    """
    Draw the alm (lmax) of a Gaussian shell given the alm of the K previous shells (multivariate normal
    conditioning, vectorised over ell); gls: Gaussian spectra in glass triangle order.
    """
    K = len(previous_index)
    sigma = np.zeros((lmax + 1, K, K))
    for a in range(K):
        for b in range(a, K):
            sigma[:, a, b] = sigma[:, b, a] = gls[triangle_index(previous_index[a], previous_index[b])][:lmax + 1]
    cross = np.array([gls[triangle_index(target_index, i)][:lmax + 1] for i in previous_index]).T   # (lmax+1, K)
    auto = np.asarray(gls[triangle_index(target_index, target_index)][:lmax + 1], dtype=float)
    weights, variance = np.zeros((lmax + 1, K)), np.zeros(lmax + 1)
    eps = ridge*np.trace(sigma[2:], axis1=1, axis2=2)/K
    weights[2:] = np.linalg.solve(sigma[2:] + eps[:, None, None]*np.eye(K), cross[2:, :, None])[..., 0]
    variance[2:] = np.clip(auto[2:] - np.sum(cross[2:]*weights[2:], axis=1), 0., None)
    ell, m = hp.Alm.getlm(lmax)
    mean = np.einsum('ak,ka->a', weights[ell], previous_alm)
    noise = rng.normal(size=len(ell)) + 1j*rng.normal(size=len(ell))
    scale = np.where(m == 0, np.sqrt(variance[ell]), np.sqrt(variance[ell]/2))
    alm = mean + scale*np.where(m == 0, noise.real, noise)
    alm[m == 0] = alm[m == 0].real
    return alm


def _gaussian_auto(args):
    cl, window, nside, shift = args
    g = np.clip(gaussian_spectra([cl], [window], nside, shift)[0], 0., None)
    g[:2] = 0.
    return g


def _kappa_part(args):
    """Weighted Born sum of a subset of independent lognormal shells (worker)."""
    gls, weights, seeds, low, l_exact, nside, shift, nthreads = args
    lmax = 2*nside
    geometry = ducc0.healpix.Healpix_Base(nside, 'RING').sht_info()
    low_index = hp.Alm.getidx(lmax, *hp.Alm.getlm(l_exact)) if l_exact else None
    kappa = np.zeros(12*nside**2)
    for cl, q, seed, low_alm in zip(gls, weights, seeds, low):
        alm = gaussian_alm(cl, lmax, np.random.default_rng(seed))
        if low_alm is not None:
            alm[low_index] = low_alm
        g = ducc0.sht.synthesis(alm=alm[None], lmax=lmax, spin=0, nthreads=nthreads, **geometry)[0]
        g -= glass.fields.cltovar(cl)/2
        np.exp(g, out=g)
        g -= 1.
        kappa += (q*shift)*g
    return kappa


def ladder_kappa(pars, z_start, nside, z_end=1100., num=100, shift=1., nonlinear=True, seed=None, l_exact=0,
                 last_shells=None, last_windows=None, delta_chi=400., z_source=1100., nproc=None):
    """
    CMB lensing convergence of lognormal shells from z_start to z_end (Born, source plane at z_source); add it
    to the Born convergence of the simulated shells below z_start.

    Parameters
    ----------
    pars : camb.CAMBparams
        Cosmology of the simulation.
    z_start : float
        Upper edge of the last simulated shell.
    nside : int
        Output NSIDE (lmax = 2 nside).
    z_end, num :
        num shells uniform in scale factor between z_start and z_end.
    shift : float
        Lognormal shift.
    nonlinear : bool
        Nonlinear (HMcode as set in pars) or linear shell spectra.
    seed : int or None
        Same seed and settings -> same map (independent of nproc).
    l_exact : int
        0 (default, fast): independent shells with Limber spectra.  X > 0: multipoles ell <= X conditioned on
        the previous shells (within delta_chi, including the simulated last_shells) with exact CAMB spectra.
    last_shells, last_windows :
        Simulated overdensity maps (nside) and windows of the last simulated shells (l_exact > 0 only).
    delta_chi : float
        Conditioning window [Mpc/h] (l_exact > 0).
    nproc : int or None
        Synthesis processes (default: cores / 32), each with cores / nproc SHT threads.

    Returns
    -------
    kappa : array (12 nside^2,)
    """
    lmax = 2*nside
    ladder = ladder_windows(z_start, z_end, num)
    conditioning = ([tophat_window(float(np.min(w.za)), float(np.max(w.za))) for w in last_windows]
                    if l_exact else [])
    windows = conditioning + ladder
    n_cond = len(conditioning)
    cls = limber_shell_cls(pars, windows, lmax, nonlinear)
    if l_exact:
        exact = shell_cls(pars, windows, l_exact, nonlinear)
        for i in range(len(windows)):
            cls[i][:l_exact + 1] = exact[triangle_index(i, i)]

    # spectrum solves: light, one process each; synthesis: ~1.5 GB per process, bound by the per-shell numpy
    # work (on a 256-thread node 8 processes x 32 threads is fastest)
    context = multiprocessing.get_context('fork')
    cores = len(os.sched_getaffinity(0))
    with context.Pool(min(len(windows), max(1, cores//8))) as pool:
        gls = pool.map(_gaussian_auto, [(cl, w, nside, shift) for cl, w in zip(cls, windows)])
    seeds = np.random.SeedSequence(seed).spawn(num + 1)
    low = [None]*num
    if l_exact:
        low = _conditional_low(pars, windows, n_cond, gls, exact, last_shells, l_exact, nside, shift,
                               delta_chi, np.random.default_rng(seeds[-1]))
    weights = born_weights(ladder, pars, z_source)
    nproc = nproc or max(1, cores//32)
    tasks = [(gls[n_cond + k::nproc], weights[k::nproc], seeds[k:num:nproc], low[k::nproc], l_exact, nside,
              shift, max(1, cores//nproc)) for k in range(nproc)]
    with context.Pool(nproc) as pool:
        return np.sum(pool.map(_kappa_part, tasks), axis=0)


def _conditional_low(pars, windows, n_cond, gls, exact, last_shells, l_exact, nside, shift, delta_chi, rng):
    """alm (ell <= l_exact) of the generated shells, drawn sequentially conditioned on the previous shells."""
    background = camb.get_background(pars)
    chi = np.array([background.comoving_radial_distance(w.zeff) for w in windows])*pars.H0/100.
    n = len(windows)
    low_gls = [np.zeros(l_exact + 1) for _ in range(n*(n + 1)//2)]
    for i in range(n):
        low_gls[triangle_index(i, i)] = gls[i][:l_exact + 1]
        for j in range(i):
            if chi[i] - chi[j] < delta_chi:           # Gaussian cross-spectra only where conditioning uses them
                pair = [exact[triangle_index(k, l)] for k, l in ((j, j), (i, i), (i, j))]
                low_gls[triangle_index(i, j)] = gaussian_spectra(pair, [windows[j], windows[i]], nside, shift)[2]
    alms = {i: hp.map2alm(gaussianise(d, shift), lmax=l_exact, iter=0) for i, d in enumerate(last_shells)}
    out = []
    for target in range(n_cond, n):
        previous = [i for i in sorted(alms) if chi[target] - chi[i] < delta_chi] or [target - 1]
        alms[target] = conditional_alm(np.array([alms[i] for i in previous]), low_gls, previous, target, l_exact, rng)
        out.append(alms[target])
        for i in [i for i in alms if chi[target] - chi[i] >= delta_chi]:
            del alms[i]
    return out


def kappa_cmb(shells, pars, z_source=1100.):
    """Born CMB lensing convergence from (delta, window) pairs in increasing redshift (glass MultiPlaneConvergence)."""
    convergence = glass.lensing.MultiPlaneConvergence(Cosmology.from_camb(pars))
    last = None
    for delta, window in shells:
        convergence.add_window(np.asarray(delta, dtype=float), window)
        last = delta
    convergence.add_plane(np.zeros(len(last)), z_source)
    return np.asarray(convergence.kappa, dtype=float)
