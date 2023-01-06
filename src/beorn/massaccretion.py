"""
Mass Accretion Model, collapsed fraction, etc..
"""
from .cosmo import D, hubble
from scipy.interpolate import splrep, splev, interp1d
from scipy.integrate import cumtrapz, trapz, quad, odeint
import numpy as np
from .constants import *


def mass_accretion_EPS(zz, mm,param):
    """
    Assuming EPS formula
    (see Eq. 6 in 1409.5228)

    mm : the initial mass bin at z = zstar
    """
    Dgrowth = []
    for i in range(len(zz)):
        Dgrowth.append(D(1/(zz[i]+1), param)) #growth factor
    Dgrowth = np.array(Dgrowth)
    import dmcosmo as dm
    par = dm.par()
    par.code.z = [0]  # [10.19,12.19,15.39,17.91,5.9,9,8.27]
    par.PS.q = 0.85  ###filter is tophat p=0.3 by default.
    par.cosmo.Ob, par.cosmo.Ol, par.cosmo.Om, par.cosmo.h = param.cosmo.Ob, param.cosmo.Ol, param.cosmo.Om, param.cosmo.h
    par.cosmo.ps = param.cosmo.ps
    HMF = dm.HMF(par)
    HMF.generate_HMF(par)
    var_tck = splrep(HMF.tab_M, HMF.sigma2)

    # free parameter
    fM = 0.6
    fracM = np.full(len(mm), fM)
    frac = interp1d(mm, fracM, axis=0, fill_value='extrapolate')

    Dg_tck = splrep(np.flip(zz), np.flip(Dgrowth))
    D_growth = lambda z: splev(z, Dg_tck)
    dDdz = lambda z: splev(z, Dg_tck, der=1)

    Maccr = np.zeros((len(zz), len(mm)))
    source = lambda M, z: (2 / np.pi) ** 0.5 * M / (splev(frac(M) * M, var_tck, ext=1) - splev(M, var_tck, ext=1)) ** 0.5 * 1.686 / D_growth(z) ** 2 * dDdz(z)
    Maccr[:, :] = odeint(source, mm, zz)
    Maccr = np.nan_to_num(Maccr, nan=0)

    Raccr = Maccr / mm[None, :]
    dMaccrdz = np.gradient(Maccr, zz, axis=0, edge_order=1)
    dMaccrdt = - dMaccrdz * (1 + zz)[:, None] * hubble(zz, param)[:, None] * sec_per_year / km_per_Mpc

    # remove NaN
    Raccr[np.isnan(Raccr)] = 0.0
    dMaccrdz[np.isnan(dMaccrdz)] = 0.0
    dMaccrdt[np.isnan(dMaccrdt)] = 0.0

    return Raccr * mm, dMaccrdt





