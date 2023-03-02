"""
External Parameters
"""
import pkg_resources

class Bunch(object):
    """
    translates dic['name'] into dic.name 
    """

    def __init__(self, data):
        self.__dict__.update(data)


def source_par():
    par = {
        "alpha_MAR" : 0.79,              # coefficient for exponential MAR
        "MAR" : 'EXP',                   # MAR model. Can be EXP or EPS.
        "type": 'SED',                   # source type. Can be 'Galaxies' or 'Miniqsos' or SED

        "E_min_sed_xray": 500,           # minimum energy of normalization of xrays in eV
        "E_max_sed_xray": 2000,          # minimum energy of normalization of xrays in eV

        "E_min_xray": 500,
        "E_max_xray": 2000,             # min and max energy that define what we call xrays.

        "alS_xray": 2.5,                 ##PL sed Xray part N ~ nu**-alS [nbr of photons/s/Hz]
        "cX":  3.4e40,                   # Xray normalization [(erg/s) * (yr/Msun)] (astro-ph/0607234 eq22)

        "N_al"    : 9690,                # nbr of lyal photons per baryons in stars
        "alS_lyal": 1.001,               ## PL for lyal

        "M_min" : 1e5,                   # Minimum mass of star forming halo. Mdark in HM
        'f_st': 0.05,
        'Mp': 1e11,
        'g1': 0.49,
        'g2': -0.61,
        'Mt': 1e7,
        'g3': 4,
        'g4': -1,

        'Nion'  : 2665,
        "f0_esc": 0.15,                   # photon escape fraction f_esc = f0_esc * (M/Mp)^pl_esc
        "Mp_esc": 1e10,
        "pl_esc": 0.0,

    }

    return Bunch(par)

def solver_par():
    par = {
        "z_ini" : 25,            ## Starting redshift
        "z_end" : 6,             ## Only for MAR. Redshift where to stop the solver
        "Nz": 500,               ##only used in simpler_faster
    }
    return Bunch(par)

def sim_par(): ## used when computing and painting profiles on a grid
    par = {
        "M_i_min" : 1e-2,
        "M_i_max" : 1e9,
        "binn" : 12,                # to define the initial halo mass at z_ini = solver.z
        "model_name": 'SED',        # Give a name to your sim, will be used to name all the files created.
        "Ncell" : 128,              # nbr of pixels of the final grid.
        "Lbox" : 100,               # Box lenght, in [Mpc/h]
        "mpi4py": 'no',             # run in parallel or not.
        "n_jobs": 1,                # number of cpus used.
        "halo_catalogs": None,      # path to the directory containing all the halo catalogs or a list of numpy arrays or a list of filenames.
        "halo_catalog_type": 'pickle', # type of halo catalogs will define how they can be read if not numpy array. 
        "store_grids": True,        # whether or not to store the grids. If not, will just store the power spectra.
        "dens_fields": None,         # path and name of the gridded density field. Used in run.py to compute dTb
        "dens_field_type": 'pkdgrav',  # Can be either 21cmFAST of pkdgrav. It adapts the format and normalization of the density field...
        "Nh_part_min":50,           # Minimum number of particles in halo to trust
        "cores" : 2,                # number of cores used in parallelisation
        "kmin": 3e-2,
        "kmax": 4,
        "kbin": 400,                ## either a path to a text files containing kbins edges values or an int (nbr of bins to measure PS)
        "thresh_xHII" :0.1,         ## mean(Tk_neutral) and mean(Tspin) are computed over cells that are below this fraction of ionized hydrogen.
        "thresh_pixel" : None,      ## when spreading the excess ionisation fraction, we treat all the connected regions with less that "thresh_pixel" as a single connected region(to speed up)
        "approx"  : True,           ## when spreading the excess ionisation fraction and running distance_tranform_edt, whether or not to do the subgrid approx.
        "data_dir": './',           ## Directory where data is saved. Nothing is saved if None is passed.
        "random_seed": 12345,
    }
    return Bunch(par)


def cosmo_par():
    par = {
    'Om' : 0.31,
    'Ob' : 0.045,
    'Ol' : 0.69,
    'rho_c' : 2.775e11,
    'h' : 0.68,
    's8': 0.83,
    'ps': "PCDM_Planck.dat",      ### This is the path to the input Linear Power Spectrum
    'corr_fct' : "corr_fct.dat",  ### This is the path where the corresponding correlation function will be stored. You can change it to anything.
    'HI_frac'  : 1-0.08,       # fraction of Helium. Only used when running H_He_Final. 1-fraction is Helium then.  0.2453 of total mass is in He according to BBN, so in terms of number density it is  1/(1+4*(1-f_He_bymass)/f_He_bymass)  ~0.075.
    "clumping" : 1,         # to rescale the background density. set to 1 to get the normal 2h profile term.
    "profile"  : 0,          # 0 for constant background density, 1 for 2h term profile
    "z_decoupl": 135,      # redshift at which the gas decouples from CMB and starts cooling adiabatically according to Tcmb0*(1+z)**2/(1+zdecoupl)
    }
    return Bunch(par)



def excursion_set_par():
    par = {
        ### HMF parameters that we use to normalise the collapsed fraction.
        "filter": 'tophat',           # tophat, sharpk or smoothk
        "c": 1,                       # scale to halo mass relation (1 for tophat, 2.5 for sharp-k, 3 for smooth-k)
        "q" : 0.8,                  # q for f(nu) [0.707,1,1] for [ST,smoothk or sharpk,PS] (q = 0.8 with tophat fits better the high redshift z>6 HMF)
        "p" : 0.3,                    # p for f(nu) [0.3,0.3,0] for [ST,smoothk or sharpk,PS]
        "delta_c" : 1.686,            # critical density
        "A" : 0.322,                  # A = 0.322 except 0.5 for PS Spherical collapse (to double check)
        "R_max": 40,                 # Mpc/h. The scale at which we start the excursion set.
        "n_rec": 3,                # mean number of recombination per baryon.
        "stepping":1,                # When doing the exc set, we smooth the field starting from large scale down to the pixel size. This parameters controls how fast you decrease the smoothing scale, in pixel units.
     }
    return Bunch(par)


def par():
    par = Bunch({
        "source": source_par(),
        "solver": solver_par(),
        "cosmo" : cosmo_par(),
        "sim" : sim_par(),
        "exc_set" : excursion_set_par(),
        })
    return par
